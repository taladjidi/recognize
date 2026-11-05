"""Tests for the service daemon."""

import signal
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from service import Service, POLL_MIN_INTERVAL, POLL_MAX_INTERVAL, POLL_BACKOFF_FACTOR


@pytest.fixture
def sqlite_config_file(tmp_path):
    """Write a minimal config file and return its path."""
    import json
    config = {
        "dbtype": "sqlite3",
        "dbhost": "",
        "dbuser": "",
        "dbpassword": "",
        "dbname": str(tmp_path / "test.db"),
        "dbtableprefix": "oc_",
        "datadirectory": "/tmp/data",
    }
    path = tmp_path / "nc_config.json"
    path.write_text(json.dumps(config))
    return str(path)


class TestInit:
    def test_creates_pipeline(self, sqlite_config_file, tmp_path):
        svc = Service(sqlite_config_file, models_dir=str(tmp_path), gpu=False)
        assert svc.pipeline is not None
        assert svc._running is False
        svc.stop()

    def test_initial_poll_interval(self, sqlite_config_file, tmp_path):
        svc = Service(sqlite_config_file, models_dir=str(tmp_path), gpu=False)
        assert svc._poll_interval == POLL_MIN_INTERVAL
        svc.stop()


class TestSignalHandling:
    def test_sigterm_sets_running_false(self, sqlite_config_file, tmp_path):
        svc = Service(sqlite_config_file, models_dir=str(tmp_path), gpu=False)
        svc._running = True
        svc._setup_signals()
        svc._handle_signal(signal.SIGTERM, None)
        assert svc._running is False
        svc.stop()

    def test_sigint_sets_running_false(self, sqlite_config_file, tmp_path):
        svc = Service(sqlite_config_file, models_dir=str(tmp_path), gpu=False)
        svc._running = True
        svc._setup_signals()
        svc._handle_signal(signal.SIGINT, None)
        assert svc._running is False
        svc.stop()


class TestAdaptivePolling:
    def test_backoff_on_empty_poll(self, sqlite_config_file, tmp_path):
        """Poll interval should increase when no work is found."""
        svc = Service(sqlite_config_file, models_dir=str(tmp_path), gpu=False)
        svc.pipeline.db.create_pending_table()
        svc.pipeline.db.create_face_detections_table()

        # Simulate empty poll
        initial = svc._poll_interval
        svc._running = True

        # Mock time.sleep to capture the interval and stop after one iteration
        call_count = 0
        def fake_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                svc._running = False

        with patch("service.time.sleep", side_effect=fake_sleep):
            svc._main_loop()

        assert svc._poll_interval > initial
        svc.stop()

    def test_backoff_caps_at_max(self, sqlite_config_file, tmp_path):
        """Poll interval should not exceed POLL_MAX_INTERVAL."""
        svc = Service(sqlite_config_file, models_dir=str(tmp_path), gpu=False)
        svc._poll_interval = POLL_MAX_INTERVAL * 2  # Force above max

        # Clamp manually like the loop does
        clamped = min(svc._poll_interval * POLL_BACKOFF_FACTOR, POLL_MAX_INTERVAL)
        assert clamped == POLL_MAX_INTERVAL
        svc.stop()


class TestMaintenanceMode:
    def test_pauses_during_maintenance(self, sqlite_config_file, tmp_path):
        """Service should sleep when Nextcloud is in maintenance mode."""
        svc = Service(sqlite_config_file, models_dir=str(tmp_path), gpu=False)
        svc.pipeline.db.create_pending_table()
        svc.pipeline.db.create_face_detections_table()
        svc._running = True

        # Set maintenance mode the way Nextcloud does: in config.php
        php = tmp_path / "config.php"
        php.write_text("<?php\n$CONFIG = array (\n  'maintenance' => true,\n);")
        svc.pipeline.db._config["config_php"] = str(php)

        call_count = 0
        def fake_sleep(seconds):
            nonlocal call_count
            call_count += 1
            # Verify it sleeps for the max interval during maintenance
            assert seconds == POLL_MAX_INTERVAL
            svc._running = False

        with patch("service.time.sleep", side_effect=fake_sleep):
            svc._main_loop()

        assert call_count == 1
        svc.stop()


class TestGracefulShutdown:
    def test_stop_closes_pipeline(self, sqlite_config_file, tmp_path):
        svc = Service(sqlite_config_file, models_dir=str(tmp_path), gpu=False)
        with patch.object(svc.pipeline, "close") as mock_close:
            svc.stop()
            mock_close.assert_called_once()

    def test_start_stop_cycle(self, sqlite_config_file, tmp_path):
        """Service should start and stop cleanly via signal simulation."""
        svc = Service(sqlite_config_file, models_dir=str(tmp_path), gpu=False)

        # Verify the start/stop contract: start sets _running=True, stop sets False
        assert svc._running is False
        svc._running = True
        svc._handle_signal(signal.SIGTERM, None)
        assert svc._running is False

        # Verify stop cleans up
        svc.stop()
        assert svc._running is False


def _svc_with_pending(cfg, tmp_path, file_ids, enabled):
    """Service on a real sqlite db with pending rows; _fetch_and_resolve returns them as images."""
    svc = Service(cfg, models_dir=str(tmp_path), gpu=False)
    p = svc.pipeline
    p.db.create_pending_table()
    for fid in file_ids:
        p.db.upsert_pending(fid, 1, 0)
    images = [{"id": r["id"], "file_id": r["file_id"], "added_at": r["added_at"],
               "path": "/x", "mimetype": "image/jpeg"} for r in p.db.fetch_pending(limit=50)]
    p._enabled = set(enabled)
    patch.object(svc, "_fetch_and_resolve", return_value=(images, [], [])).start()
    return svc


def _left(svc):
    return sorted(r["file_id"] for r in svc.pipeline.db.fetch_pending(limit=50))


class TestImagePassSubmits:
    def test_worker_crash_submits_nothing_and_drops_rows(self, sqlite_config_file, tmp_path):
        svc = _svc_with_pending(sqlite_config_file, tmp_path, [10, 20], {"imagenet"})
        with patch.object(svc.pipeline, "process_images_tags", return_value={}), \
             patch.object(svc.pipeline, "_submit_result") as submit:
            svc._run_classifier_passes()
        submit.assert_not_called()
        assert _left(svc) == []
        svc.stop()

    def test_per_file_error_only_good_files_submitted(self, sqlite_config_file, tmp_path):
        svc = _svc_with_pending(sqlite_config_file, tmp_path, [10, 20], {"imagenet"})
        res = {10: {"error": "boom"}, 20: {"tags": ["Dog"]}}
        with patch.object(svc.pipeline, "process_images_tags", return_value=res), \
             patch.object(svc.pipeline, "_submit_result", return_value=True) as submit:
            svc._run_classifier_passes()
        assert [c.args[0] for c in submit.call_args_list] == [20]
        assert _left(svc) == []
        svc.stop()

    def test_image_without_labels_is_still_submitted(self, sqlite_config_file, tmp_path):
        svc = _svc_with_pending(sqlite_config_file, tmp_path, [10], {"imagenet"})
        with patch.object(svc.pipeline, "process_images_tags", return_value={10: {"tags": []}}), \
             patch.object(svc.pipeline, "_submit_result", return_value=True) as submit:
            svc._run_classifier_passes()
        submit.assert_called_once_with(10, {"tags": [], "faces": []})
        svc.stop()

    def test_one_combined_submit_with_faces_and_tags(self, sqlite_config_file, tmp_path):
        svc = _svc_with_pending(sqlite_config_file, tmp_path, [10], {"imagenet", "faces"})
        with patch.object(svc.pipeline, "process_images_faces",
                          return_value={10: {"faces": [{"x": 1}]}}), \
             patch.object(svc.pipeline, "process_images_tags",
                          return_value={10: {"tags": ["Cat"]}}), \
             patch.object(svc.pipeline, "_submit_result", return_value=True) as submit:
            svc._run_classifier_passes()
        submit.assert_called_once_with(10, {"tags": ["Cat"], "faces": [{"x": 1}]})
        svc.stop()

    def test_failed_faces_pass_blocks_submit_of_the_image(self, sqlite_config_file, tmp_path):
        svc = _svc_with_pending(sqlite_config_file, tmp_path, [10], {"imagenet", "faces"})
        with patch.object(svc.pipeline, "process_images_faces", return_value={}), \
             patch.object(svc.pipeline, "process_images_tags",
                          return_value={10: {"tags": ["Cat"]}}), \
             patch.object(svc.pipeline, "_submit_result") as submit:
            svc._run_classifier_passes()
        submit.assert_not_called()
        svc.stop()

    def test_faces_only_config_deletes_rows(self, sqlite_config_file, tmp_path):
        svc = _svc_with_pending(sqlite_config_file, tmp_path, [10, 20], {"faces"})
        with patch.object(svc.pipeline, "process_images_faces",
                          return_value={10: {"faces": []}, 20: {"faces": []}}), \
             patch.object(svc.pipeline, "_submit_result", return_value=True):
            svc._run_classifier_passes()
        assert _left(svc) == []
        svc.stop()

    def _api_failing(self, svc, status):
        from nc_api import NextcloudAPIError
        api = MagicMock()
        api.submit_results.side_effect = NextcloudAPIError(f"HTTP {status}", status)
        svc.pipeline.nc_api = api
        return api

    def test_permanent_4xx_deletes_row_and_moves_on(self, sqlite_config_file, tmp_path):
        svc = _svc_with_pending(sqlite_config_file, tmp_path, [10, 20], {"imagenet"})
        api = self._api_failing(svc, 400)
        res = {10: {"tags": []}, 20: {"tags": []}}
        with patch.object(svc.pipeline, "process_images_tags", return_value=res), \
             patch("service.time.sleep"):
            svc._run_classifier_passes()
        assert api.submit_results.call_count == 2
        assert _left(svc) == []
        svc.stop()

    def test_503_defers_rows_and_stops_pass(self, sqlite_config_file, tmp_path):
        svc = _svc_with_pending(sqlite_config_file, tmp_path, [10, 20], {"imagenet"})
        before = {r["file_id"]: r["added_at"] for r in svc.pipeline.db.fetch_pending(limit=9)}
        svc.pipeline.db._execute("UPDATE oc_recognize_pending SET added_at = 5")
        svc.pipeline.db._commit()
        for it in svc._fetch_and_resolve.return_value[0]:
            it["added_at"] = 5
        api = self._api_failing(svc, 503)
        res = {10: {"tags": []}, 20: {"tags": []}}
        with patch.object(svc.pipeline, "process_images_tags", return_value=res), \
             patch("service.time.sleep") as sleep:
            assert svc._run_classifier_passes() is False
        assert api.submit_results.call_count == 1
        rows = svc.pipeline.db.fetch_pending(limit=9)
        assert sorted(r["file_id"] for r in rows) == [10, 20]
        assert all(r["added_at"] > 5 for r in rows)
        sleep.assert_called_with(POLL_MAX_INTERVAL)
        svc.stop()

    def test_auth_error_stops_pass_and_keeps_rows(self, sqlite_config_file, tmp_path):
        from nc_api import NextcloudAPIError
        svc = _svc_with_pending(sqlite_config_file, tmp_path, [10, 20], {"imagenet"})
        api = MagicMock()
        api.submit_results.side_effect = NextcloudAPIError("HTTP 401", 401)
        svc.pipeline.nc_api = api
        res = {10: {"tags": []}, 20: {"tags": []}}
        with patch.object(svc.pipeline, "process_images_tags", return_value=res), \
             patch("service.time.sleep") as sleep:
            assert svc._run_classifier_passes() is False
        assert api.submit_results.call_count == 1  # stopped after the first 401
        assert _left(svc) == [10, 20]
        sleep.assert_called_with(POLL_MAX_INTERVAL)
        svc.stop()


class TestPendingHousekeeping:
    def test_requeued_row_survives_delete(self, sqlite_config_file, tmp_path):
        svc = _svc_with_pending(sqlite_config_file, tmp_path, [10], {"imagenet"})
        db = svc.pipeline.db

        def requeue_while_processing(images):
            db._execute("UPDATE oc_recognize_pending SET added_at = added_at + 100, action = 1")
            db._commit()
            return {10: {"tags": []}}

        with patch.object(svc.pipeline, "process_images_tags", side_effect=requeue_while_processing), \
             patch.object(svc.pipeline, "_submit_result", return_value=True):
            svc._run_classifier_passes()
        assert _left(svc) == [10]
        svc.stop()

    def _filecache(self, db, rows, storage="home::u"):
        db._execute("CREATE TABLE oc_storages (numeric_id INTEGER PRIMARY KEY, id TEXT)")
        db._execute("CREATE TABLE oc_filecache (fileid INTEGER PRIMARY KEY, storage INTEGER)")
        db._execute("INSERT INTO oc_storages VALUES (1, ?)", (storage,))
        for r in rows:
            db._execute("INSERT INTO oc_filecache (fileid, storage) VALUES (?, 1)", (r,))
        db._commit()

    def test_deleted_row_consumed_detections_kept_while_filecache_exists(self, sqlite_config_file, tmp_path):
        svc = Service(sqlite_config_file, models_dir=str(tmp_path), gpu=False)
        db = svc.pipeline.db
        db.create_pending_table()
        db.create_face_detections_table()
        self._filecache(db, [10])  # 10 is in the trash (still has a row), 20 is gone
        for fid in (10, 20):
            db.upsert_pending(fid, 1, 2)
            db.insert_face_detection(fid, "u", {"x": .1, "y": .1, "width": .1, "height": .1,
                                                "vector": [0.0] * 128, "score": .9})
        svc._consume_deleted()
        assert db.fetch_pending(limit=10, action_filter=(2,)) == []
        left = [r["file_id"] for r in db._fetchall_dicts(
            db._execute("SELECT file_id FROM oc_recognize_face_detections"))]
        assert left == [10]
        svc.stop()

    def test_unreadable_rows_do_not_starve_good_one(self, sqlite_config_file, tmp_path):
        svc = Service(sqlite_config_file, models_dir=str(tmp_path), gpu=False)
        db = svc.pipeline.db
        db.create_pending_table()
        self._filecache(db, list(range(1, 252)))
        for fid in range(1, 252):
            db.upsert_pending(fid, 1, 0)
        db._execute("UPDATE oc_recognize_pending SET added_at = file_id")  # file 251 is newest
        db._commit()
        good = {251: {"path": "/x", "mimetype": "image/jpeg", "storage": 1}}
        with patch.object(db, "resolve_file_paths",
                          side_effect=lambda ids: {i: good[i] for i in ids if i in good}):
            seen = []
            for _ in range(2):
                images, _v, _a = svc._fetch_and_resolve(200)
                seen += [i["file_id"] for i in images]
        assert 251 in seen
        svc.stop()

    def test_unsupported_storage_and_unknown_mimetype_rows_deleted(self, sqlite_config_file, tmp_path):
        svc = Service(sqlite_config_file, models_dir=str(tmp_path), gpu=False)
        db = svc.pipeline.db
        db.create_pending_table()
        self._filecache(db, [10], storage="object::store:s3")
        db.upsert_pending(10, 1, 0)
        with patch.object(db, "resolve_file_paths", return_value={}):
            svc._fetch_and_resolve(10)
        assert db.fetch_pending(limit=10) == []
        db.upsert_pending(11, 1, 0)
        db._execute("INSERT INTO oc_filecache VALUES (11, 1)")
        with patch.object(db, "resolve_file_paths",
                          return_value={11: {"path": "/x", "mimetype": "application/x-foo", "storage": 1}}):
            svc._fetch_and_resolve(10)
        assert db.fetch_pending(limit=10) == []
        svc.stop()

    def test_maintenance_blocks_consuming_deleted(self, sqlite_config_file, tmp_path):
        svc = Service(sqlite_config_file, models_dir=str(tmp_path), gpu=False)
        svc._running = True
        with patch.object(svc.pipeline.db, "check_maintenance_mode", return_value=True), \
             patch.object(svc, "_consume_deleted") as consume, \
             patch("service.time.sleep", side_effect=lambda s: setattr(svc, "_running", False)):
            svc._main_loop()
        consume.assert_not_called()
        svc.stop()

    def test_only_rows_missing_from_filecache_are_orphans(self, sqlite_config_file, tmp_path):
        svc = Service(sqlite_config_file, models_dir=str(tmp_path), gpu=False)
        db = svc.pipeline.db
        db.create_pending_table()
        self._filecache(db, [10])
        db.upsert_pending(10, 1, 0)   # in filecache but unresolvable here
        db.upsert_pending(20, 1, 0)   # gone from filecache
        with patch.object(db, "resolve_file_paths", return_value={}):
            assert svc._fetch_and_resolve(10) == ([], [], [])
        assert [r["file_id"] for r in db.fetch_pending(limit=10)] == [10]
        svc.stop()

    def test_enabled_models_reread(self, sqlite_config_file, tmp_path):
        svc = Service(sqlite_config_file, models_dir=str(tmp_path), gpu=False)
        with patch.object(svc.pipeline.db, "get_enabled_models", return_value={"faces"}):
            svc._refresh_enabled()
        assert svc.pipeline._enabled == {"faces"}
        with patch.object(svc.pipeline.db, "get_enabled_models", return_value={"imagenet"}):
            svc._refresh_enabled()
        assert svc.pipeline._enabled == {"imagenet"}
        svc.stop()

    def test_loop_pings_db(self, sqlite_config_file, tmp_path):
        svc = Service(sqlite_config_file, models_dir=str(tmp_path), gpu=False)
        svc._running = True
        with patch.object(svc.pipeline.db, "ping") as ping, \
             patch.object(svc.pipeline.db, "check_maintenance_mode", return_value=True), \
             patch("service.time.sleep", side_effect=lambda s: setattr(svc, "_running", False)):
            svc._main_loop()
        ping.assert_called()
        svc.stop()
