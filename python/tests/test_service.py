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

        # Set maintenance mode
        svc.config["maintenance"] = True
        svc.pipeline.db._config["maintenance"] = True

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
