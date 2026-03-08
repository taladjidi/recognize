"""Tests for the database layer using SQLite in-memory."""

import json
import time

import pytest

from db import DB


@pytest.fixture
def sqlite_config():
    return {
        "dbtype": "sqlite3",
        "dbhost": "",
        "dbuser": "",
        "dbpassword": "",
        "dbname": ":memory:",
        "dbtableprefix": "oc_",
        "datadirectory": "/tmp/data",
    }


@pytest.fixture
def db(sqlite_config):
    """Create a DB with pending + face_detections tables."""
    d = DB(sqlite_config)
    d.create_pending_table()
    d.create_face_detections_table()
    yield d
    d.close()


class TestConnection:
    def test_connect_sqlite(self, sqlite_config):
        d = DB(sqlite_config)
        d.ping()
        d.close()

    def test_invalid_prefix_rejected(self, sqlite_config):
        sqlite_config["dbtableprefix"] = "oc_; DROP TABLE--"
        with pytest.raises(ValueError, match="Invalid table prefix"):
            DB(sqlite_config)

    def test_empty_prefix_allowed(self, sqlite_config):
        sqlite_config["dbtableprefix"] = ""
        d = DB(sqlite_config)
        d.create_pending_table()
        d.close()


class TestPending:
    def test_upsert_and_fetch(self, db):
        db.upsert_pending(file_id=100, storage_id=1, action=0)
        db.upsert_pending(file_id=200, storage_id=1, action=0)

        rows = db.fetch_pending(limit=10)
        assert len(rows) == 2
        assert rows[0]["file_id"] == 100
        assert rows[1]["file_id"] == 200

    def test_upsert_updates_existing(self, db):
        db.upsert_pending(file_id=100, storage_id=1, action=0)
        time.sleep(0.01)  # Ensure different timestamp
        db.upsert_pending(file_id=100, storage_id=1, action=1)

        rows = db.fetch_pending(limit=10)
        assert len(rows) == 1
        assert rows[0]["action"] == 1  # Updated to action=1

    def test_delete_pending(self, db):
        db.upsert_pending(file_id=100, storage_id=1, action=0)
        db.upsert_pending(file_id=200, storage_id=1, action=0)

        rows = db.fetch_pending(limit=10)
        db.delete_pending([rows[0]["id"]])

        remaining = db.fetch_pending(limit=10)
        assert len(remaining) == 1
        assert remaining[0]["file_id"] == 200

    def test_delete_empty_list(self, db):
        db.delete_pending([])  # Should not raise

    def test_fetch_respects_action_filter(self, db):
        db.upsert_pending(file_id=100, storage_id=1, action=0)
        db.upsert_pending(file_id=200, storage_id=1, action=2)  # deletion

        creates = db.fetch_pending(limit=10, action_filter=(0,))
        assert len(creates) == 1
        assert creates[0]["file_id"] == 100

        deletes = db.fetch_pending(limit=10, action_filter=(2,))
        assert len(deletes) == 1
        assert deletes[0]["file_id"] == 200

    def test_fetch_orders_by_added_at(self, db):
        # Insert in reverse order with explicit timestamps
        db._execute(
            f"INSERT INTO {db._table('recognize_pending')} "
            f"(file_id, storage_id, action, added_at) VALUES (?, ?, ?, ?)",
            (300, 1, 0, 1000)
        )
        db._execute(
            f"INSERT INTO {db._table('recognize_pending')} "
            f"(file_id, storage_id, action, added_at) VALUES (?, ?, ?, ?)",
            (100, 1, 0, 500)
        )
        db._commit()

        rows = db.fetch_pending(limit=10)
        assert rows[0]["file_id"] == 100  # Earlier timestamp first
        assert rows[1]["file_id"] == 300

    def test_fetch_respects_limit(self, db):
        for i in range(10):
            db.upsert_pending(file_id=i, storage_id=1, action=0)

        rows = db.fetch_pending(limit=3)
        assert len(rows) == 3

    def test_pending_count(self, db):
        assert db.pending_count() == 0
        db.upsert_pending(file_id=1, storage_id=1, action=0)
        db.upsert_pending(file_id=2, storage_id=1, action=1)
        db.upsert_pending(file_id=3, storage_id=1, action=2)  # deletion
        assert db.pending_count() == 2  # Only action 0 and 1


class TestFaceDetections:
    def test_insert_face(self, db):
        face = {
            "x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4,
            "vector": [0.01] * 512,
        }
        result = db.insert_face_detection(file_id=42, user_id="alice", face=face)
        assert result is not None

    def test_get_face_detections_for_user(self, db):
        face = {
            "x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4,
            "vector": [0.5, 0.6, 0.7],
        }
        db.insert_face_detection(file_id=42, user_id="alice", face=face)
        db.insert_face_detection(file_id=43, user_id="alice", face=face)
        db.insert_face_detection(file_id=44, user_id="bob", face=face)

        alice_faces = db.get_face_detections_for_user("alice")
        assert len(alice_faces) == 2
        assert alice_faces[0]["face_vector"] == [0.5, 0.6, 0.7]

        bob_faces = db.get_face_detections_for_user("bob")
        assert len(bob_faces) == 1

    def test_delete_face_detections(self, db):
        face = {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4, "vector": [1.0]}
        db.insert_face_detection(file_id=42, user_id="alice", face=face)
        db.insert_face_detection(file_id=42, user_id="bob", face=face)

        db.delete_face_detections(file_id=42)

        assert len(db.get_face_detections_for_user("alice")) == 0
        assert len(db.get_face_detections_for_user("bob")) == 0

    def test_update_face_clusters(self, db):
        face = {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4, "vector": [1.0]}
        db.insert_face_detection(file_id=1, user_id="alice", face=face)
        db.insert_face_detection(file_id=2, user_id="alice", face=face)

        faces = db.get_face_detections_for_user("alice")
        ids = [f["id"] for f in faces]

        db.update_face_clusters(ids, [5, 5])

        updated = db.get_face_detections_for_user("alice")
        assert all(f["cluster_id"] == 5 for f in updated)

    def test_update_clusters_noise_label(self, db):
        """HDBSCAN uses -1 for noise points. These should become NULL."""
        face = {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4, "vector": [1.0]}
        db.insert_face_detection(file_id=1, user_id="alice", face=face)

        faces = db.get_face_detections_for_user("alice")
        db.update_face_clusters([faces[0]["id"]], [-1])

        updated = db.get_face_detections_for_user("alice")
        assert updated[0]["cluster_id"] is None


class TestMimeClassification:
    def test_image_mimes(self):
        assert DB.classify_mimetype("image/jpeg") == "image"
        assert DB.classify_mimetype("image/heic") == "image"

    def test_video_mimes(self):
        assert DB.classify_mimetype("video/mp4") == "video"
        assert DB.classify_mimetype("image/gif") == "video"  # GIFs are treated as video

    def test_audio_mimes(self):
        assert DB.classify_mimetype("audio/mpeg") == "audio"
        assert DB.classify_mimetype("audio/flac") == "audio"

    def test_unknown_mime(self):
        assert DB.classify_mimetype("application/pdf") is None
        assert DB.classify_mimetype("text/plain") is None


class TestStoragePath:
    def test_home_storage(self):
        path = DB._resolve_storage_path("home::alice", "files/photo.jpg", "/data")
        assert path == "/data/alice/files/photo.jpg"

    def test_local_storage(self):
        path = DB._resolve_storage_path("local::/mnt/shared/", "docs/file.pdf", "/data")
        assert path == "/mnt/shared/docs/file.pdf"

    def test_unsupported_storage(self):
        path = DB._resolve_storage_path("object::store:amazon::bucket", "file.jpg", "/data")
        assert path is None


class TestMaintenanceMode:
    def test_not_in_maintenance(self, db):
        assert db.check_maintenance_mode() is False

    def test_in_maintenance(self, sqlite_config):
        sqlite_config["maintenance"] = True
        d = DB(sqlite_config)
        assert d.check_maintenance_mode() is True
        d.close()
