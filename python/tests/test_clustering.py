"""Tests for semi-supervised face clustering."""

import numpy as np
import pytest

from clustering import cluster_user_faces, cluster_all_users
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
    d = DB(sqlite_config)
    d.create_face_detections_table()
    d.create_face_clusters_table()
    yield d
    d.close()


def _make_embedding(cluster_center, noise_scale=0.02):
    """Generate a random embedding near a cluster center."""
    vec = cluster_center + np.random.randn(512).astype(np.float32) * noise_scale
    vec /= np.linalg.norm(vec)
    return vec


def _insert_faces(db, user_id, embeddings, file_id_start=1000):
    """Insert face detections with given embeddings, return face IDs."""
    ids = []
    for i, emb in enumerate(embeddings):
        face = {
            "x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4,
            "vector": emb.tolist(),
        }
        fid = db.insert_face_detection(
            file_id=file_id_start + i, user_id=user_id, face=face,
        )
        ids.append(fid)
    return ids


class TestClusterUserFaces:
    def test_empty_user(self, db):
        """User with no faces should return zeros."""
        stats = cluster_user_faces(db, "nobody")
        assert stats["n_faces"] == 0
        assert stats["n_clusters"] == 0

    def test_no_fresh_detections(self, db):
        """If all detections are already clustered, skip clustering."""
        np.random.seed(42)
        center = np.random.randn(512).astype(np.float32)
        center /= np.linalg.norm(center)
        embeddings = [_make_embedding(center, noise_scale=0.02) for _ in range(10)]
        _insert_faces(db, "alice", embeddings)

        # First clustering: assigns to clusters
        stats1 = cluster_user_faces(db, "alice")
        assert stats1["n_assigned"] > 0

        # Second clustering: no fresh detections, should skip
        stats2 = cluster_user_faces(db, "alice")
        assert stats2["n_assigned"] == 0

    def test_creates_cluster_rows(self, db):
        """Clustering should create rows in recognize_face_clusters."""
        np.random.seed(42)
        center = np.random.randn(512).astype(np.float32)
        center /= np.linalg.norm(center)
        embeddings = [_make_embedding(center, noise_scale=0.02) for _ in range(10)]
        _insert_faces(db, "alice", embeddings)

        cluster_user_faces(db, "alice")

        clusters = db.get_face_clusters_for_user("alice")
        assert len(clusters) >= 1

    def test_cluster_ids_reference_cluster_table(self, db):
        """Detection cluster_ids should reference actual face_clusters rows."""
        np.random.seed(42)
        center = np.random.randn(512).astype(np.float32)
        center /= np.linalg.norm(center)
        embeddings = [_make_embedding(center, noise_scale=0.02) for _ in range(10)]
        _insert_faces(db, "alice", embeddings)

        cluster_user_faces(db, "alice")

        faces = db.get_face_detections_for_user("alice")
        clusters = db.get_face_clusters_for_user("alice")
        cluster_ids = {c["id"] for c in clusters}

        for f in faces:
            if f["cluster_id"] is not None:
                assert f["cluster_id"] in cluster_ids, (
                    f"Detection {f['id']} has cluster_id={f['cluster_id']} "
                    f"not in clusters table: {cluster_ids}"
                )

    def test_two_clusters(self, db):
        """Two well-separated groups should form two clusters."""
        np.random.seed(42)
        center_a = np.random.randn(512).astype(np.float32)
        center_a /= np.linalg.norm(center_a)
        center_b = -center_a

        embs_a = [_make_embedding(center_a, noise_scale=0.02) for _ in range(10)]
        embs_b = [_make_embedding(center_b, noise_scale=0.02) for _ in range(10)]
        _insert_faces(db, "alice", embs_a, file_id_start=1000)
        _insert_faces(db, "alice", embs_b, file_id_start=2000)

        stats = cluster_user_faces(db, "alice")
        assert stats["n_clusters"] >= 2

    def test_preserves_existing_clusters(self, db):
        """New detections near an existing cluster should join it, not create a new one."""
        np.random.seed(42)
        center = np.random.randn(512).astype(np.float32)
        center /= np.linalg.norm(center)

        # First batch: 10 faces → creates a cluster
        embs1 = [_make_embedding(center, noise_scale=0.02) for _ in range(10)]
        _insert_faces(db, "alice", embs1, file_id_start=1000)
        stats1 = cluster_user_faces(db, "alice")
        n_clusters_1 = stats1["n_clusters"]

        # Second batch: 5 more faces near same center
        embs2 = [_make_embedding(center, noise_scale=0.02) for _ in range(5)]
        _insert_faces(db, "alice", embs2, file_id_start=2000)
        stats2 = cluster_user_faces(db, "alice")

        # Should not create additional clusters for the same person
        assert stats2["n_clusters"] == n_clusters_1
        assert stats2["n_assigned"] > 0

    def test_respects_threshold(self, db):
        """Detections with threshold set should not be assigned if too far."""
        np.random.seed(42)
        center = np.random.randn(512).astype(np.float32)
        center /= np.linalg.norm(center)

        # Create faces and cluster them
        embs = [_make_embedding(center, noise_scale=0.02) for _ in range(10)]
        ids = _insert_faces(db, "alice", embs, file_id_start=1000)
        cluster_user_faces(db, "alice")

        # Now simulate user removing a face: set threshold very low so it
        # should be blocked from re-clustering
        t = db._table("recognize_face_detections")
        db._execute(
            f"UPDATE {t} SET cluster_id = NULL, threshold = 0.001 WHERE id = {db._ph()}",
            (ids[0],),
        )
        db._commit()

        # Re-cluster — the face with threshold should NOT be re-assigned
        # (its distance to any centroid will be > 0.001)
        stats = cluster_user_faces(db, "alice")
        face = [f for f in db.get_face_detections_for_user("alice") if f["id"] == ids[0]]
        assert len(face) == 1
        assert face[0]["cluster_id"] is None, "Face with tiny threshold should stay unassigned"


class TestClusterAllUsers:
    def test_multiple_users(self, db):
        """Cluster faces for multiple users independently."""
        np.random.seed(42)
        center_a = np.random.randn(512).astype(np.float32)
        center_a /= np.linalg.norm(center_a)
        center_b = np.random.randn(512).astype(np.float32)
        center_b /= np.linalg.norm(center_b)

        _insert_faces(db, "alice", [_make_embedding(center_a) for _ in range(10)], 1000)
        _insert_faces(db, "bob", [_make_embedding(center_b) for _ in range(10)], 2000)

        results = cluster_all_users(db, user_ids=["alice", "bob"])

        assert "alice" in results
        assert "bob" in results
        assert results["alice"]["n_faces"] > 0
        assert results["bob"]["n_faces"] > 0

    def test_error_handling(self, db):
        """Errors for one user shouldn't affect others."""
        np.random.seed(42)
        center = np.random.randn(512).astype(np.float32)
        center /= np.linalg.norm(center)
        _insert_faces(db, "alice", [_make_embedding(center) for _ in range(10)])

        results = cluster_all_users(db, user_ids=["alice", "nobody"])

        assert "alice" in results
        assert results["alice"]["n_faces"] > 0
        assert "nobody" in results
        assert results["nobody"]["n_faces"] == 0
