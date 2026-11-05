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
            patch("service.CROP_BATCH", 1):
        svc._crop_maintenance()
    assert os.listdir(tmp_path / "crops" / "alice") == [f"{big}.jpg"]  # the cluster's largest, not id order


def test_crop_thread_progresses_during_a_pass_and_stops_promptly(svc, tmp_path):  # noqa: F811
    db = svc.pipeline.db
    photo = str(tmp_path / "p.jpg")
    _marked_jpeg(photo, 1)
    ids = [_det(db, 10, "alice", {**FACE, "x": 0.1 * i}) for i in range(3)]
    db._commit()
    paths = {10: {"path": photo, "mimetype": "image/jpeg", "storage": 1, "owner": "alice"}}
    root = tmp_path / "crops" / "alice"
    t = threading.Thread(target=svc._crop_loop)
    with patch("db.DB.resolve_file_paths", side_effect=lambda self_, ids_: {i: paths[i] for i in ids_ if i in paths},
               autospec=True), patch("service.CROP_THREAD_PAUSE", 0.05):
        t.start()
        t0 = time.monotonic()
        time.sleep(0.2)  # stands in for a long classifier pass on the main thread
        while time.monotonic() - t0 < 10 and not (root.is_dir() and len(os.listdir(root)) == 3):
            time.sleep(0.05)
        assert sorted(os.listdir(root)) == sorted(f"{i}.jpg" for i in ids)
        t1 = time.monotonic()
        svc.pipeline.stopping.set()
        t.join(5)
    assert not t.is_alive() and time.monotonic() - t1 < 2


def test_crop_thread_survives_exceptions(svc):  # noqa: F811
    with patch.object(svc, "_crop_maintenance", side_effect=RuntimeError("boom")), \
            patch("service.CROP_THREAD_PAUSE", 0.01):
        t = threading.Thread(target=svc._crop_loop)
        t.start()
        time.sleep(0.2)
        assert t.is_alive()
        svc.pipeline.stopping.set()
        t.join(2)
    assert not t.is_alive()


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


class _FakeStop:
    """Event stand-in: runs `n` loop turns, records every wait."""
    def __init__(self, n):
        self.n, self.waits = n, []

    def is_set(self):
        return len(self.waits) >= self.n

    def wait(self, t=None):
        self.waits.append(t)

    def set(self):
        self.n = 0


def test_crop_loop_ends_its_transaction_every_tick(svc):  # noqa: F811
    fake = MagicMock()
    with patch("db.DB", return_value=fake), patch.object(svc, "_crop_maintenance"), \
            patch.object(svc.pipeline, "stopping", _FakeStop(3)):
        svc._crop_loop()
    assert fake.rollback.call_count >= 3  # MariaDB REPEATABLE READ: no tick may pin a snapshot


def test_crop_loop_backs_off_after_an_error_and_while_idle(svc):  # noqa: F811
    import service
    with patch("db.DB", return_value=MagicMock()), patch.object(svc, "_crop_maintenance", side_effect=RuntimeError), \
            patch.object(svc.pipeline, "stopping", _FakeStop(2)):
        svc._crop_loop()
        assert svc.pipeline.stopping.waits[0] == service.POLL_MAX_INTERVAL
    with patch("db.DB", side_effect=AssertionError("pinged while idle")), \
            patch.object(svc.pipeline, "_enabled", set()), patch.object(svc.pipeline, "stopping", _FakeStop(2)):
        svc._crop_loop()
        assert svc.pipeline.stopping.waits[0] == service.POLL_MAX_INTERVAL
    svc._crop_rest_until = time.monotonic() + 1000
    with patch("db.DB", side_effect=AssertionError("pinged while resting")), patch.object(svc.pipeline, "stopping", _FakeStop(2)):
        svc._crop_loop()
        assert svc.pipeline.stopping.waits[0] > service.CROP_THREAD_PAUSE


def test_signal_handler_does_not_take_the_event_lock(svc):  # noqa: F811
    import signal
    old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        svc._setup_signals()
        stop = svc.pipeline.stopping
        with stop._cond:  # the signal landed while the main thread was inside Event.wait()
            t = threading.Thread(target=svc._handle_signal, args=(15, None), daemon=True)
            t.start()
            t.join(1)
            assert not t.is_alive(), "handler blocked on the Event lock"
        assert stop.wait(3)  # a watcher sets it once the lock is free
    finally:
        for s, h in old.items():
            signal.signal(s, h)
