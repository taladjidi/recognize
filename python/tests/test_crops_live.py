"""Crop backfill during busy periods, covers first, and prompt SIGTERM."""
import os
import sys
import threading
import time
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(__file__))
from test_crops import FACE, _det, _marked_jpeg, svc  # noqa: F401  (svc is a fixture)


def test_backfill_does_covers_first(svc, tmp_path):  # noqa: F811
    db = svc.pipeline.db
    photo = str(tmp_path / "p.jpg")
    _marked_jpeg(photo, 1)
    small = _det(db, 10, "alice", {**FACE, "width": 0.05, "height": 0.05})
    big = _det(db, 10, "alice")
    _det(db, 10, "alice", {**FACE, "x": 0.1})
    db._execute("UPDATE oc_recognize_face_detections SET cluster_id = 1 WHERE id IN (?, ?)", (small, big))
    paths = {10: {"path": photo, "mimetype": "image/jpeg", "storage": 1, "owner": "alice"}}
    with patch.object(db, "resolve_file_paths", side_effect=lambda ids: {i: paths[i] for i in ids if i in paths}), \
            patch("service.CROP_BUSY_BATCH", 1):
        svc._crop_maintenance(busy=True)
    assert os.listdir(tmp_path / "crops" / "alice") == [f"{big}.jpg"]  # the cluster's largest, not id order


def test_busy_slice_runs_after_a_classifier_pass(svc):  # noqa: F811
    seen = []
    with patch.object(svc, "_run_classifier_passes", return_value=True), \
            patch.object(svc, "_crop_maintenance", side_effect=lambda busy=False: seen.append(busy)), \
            patch.object(svc.pipeline.db, "check_maintenance_mode", return_value=False), \
            patch.object(svc.pipeline.db, "set_app_value"), \
            patch.object(svc, "_refresh_enabled"), patch.object(svc, "_consume_deleted"), \
            patch("service.Service._sleep", side_effect=lambda s: setattr(svc, "_running", False)):
        svc._running = True
        svc._main_loop()
    assert seen == [True]


def _sleeper(pipe):
    time.sleep(60)


def test_sigterm_interrupts_a_running_worker_and_keeps_pending_rows(svc):  # noqa: F811
    threading.Timer(0.5, svc._handle_signal, args=(15, None)).start()
    t0 = time.monotonic()
    assert svc.pipeline._run_in_subprocess(_sleeper, (), timeout=600) == {}
    assert time.monotonic() - t0 < 5
    svc._fetch_and_resolve = MagicMock(return_value=([{"id": 1, "file_id": 10, "added_at": 1, "path": "x"}], [], []))
    svc.pipeline.db.pending_count = MagicMock(return_value=1)
    svc.pipeline.db.delete_pending_items = MagicMock()
    svc.pipeline._submit_result = MagicMock()
    assert svc._run_classifier_passes() is False  # stopping: no worker run, nothing submitted or deleted
    svc.pipeline.db.delete_pending_items.assert_not_called()
    svc.pipeline._submit_result.assert_not_called()


def test_idle_sleep_wakes_on_sigterm(tmp_path):
    """An idle 30 s back-off must not hold the stop: systemd's stop timeout is 30 s."""
    import threading, time as _t
    from service import Service
    svc = Service.__new__(Service)
    svc._running = True
    svc.pipeline = type("P", (), {"stopping": threading.Event()})()
    threading.Timer(0.3, svc.pipeline.stopping.set).start()
    t0 = _t.monotonic()
    svc._sleep(30)
    assert _t.monotonic() - t0 < 2
