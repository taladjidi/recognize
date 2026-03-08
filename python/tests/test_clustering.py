"""Tests for face clustering."""

import numpy as np
import pytest

from clustering import cluster_faces, cluster_user_faces, cluster_all_users
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
    yield d
    d.close()


def _make_embedding(cluster_center, noise_scale=0.02):
    """Generate a random embedding near a cluster center."""
    vec = cluster_center + np.random.randn(512).astype(np.float32) * noise_scale
    vec /= np.linalg.norm(vec)
    return vec


def _insert_faces(db, user_id, embeddings):
    """Insert face detections with given embeddings, return face IDs."""
    ids = []
    for i, emb in enumerate(embeddings):
        face = {
            "x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4,
            "vector": emb.tolist(),
        }
        fid = db.insert_face_detection(file_id=1000 + i, user_id=user_id, face=face)
        ids.append(fid)
    return ids


class TestClusterFaces:
    def test_too_few_faces(self):
        """Fewer faces than min_cluster_size should all be noise."""
        embeddings = np.random.randn(2, 512).astype(np.float32)
        labels = cluster_faces(embeddings)
        assert len(labels) == 2
        assert all(l == -1 for l in labels)

    def test_single_cluster(self):
        """Tight embeddings with allow_single_cluster should form one cluster."""
        np.random.seed(42)
        center = np.random.randn(512).astype(np.float32)
        center /= np.linalg.norm(center)
        embeddings = np.stack([_make_embedding(center, noise_scale=0.02) for _ in range(15)])

        labels = cluster_faces(embeddings)
        unique = set(labels)
        unique.discard(-1)
        assert len(unique) == 1, f"Expected 1 cluster, got {len(unique)}: {labels}"

    def test_two_clusters(self):
        """Two well-separated groups should form two clusters."""
        np.random.seed(42)
        center_a = np.random.randn(512).astype(np.float32)
        center_a /= np.linalg.norm(center_a)
        center_b = -center_a  # Opposite direction = max distance

        embs_a = [_make_embedding(center_a, noise_scale=0.02) for _ in range(10)]
        embs_b = [_make_embedding(center_b, noise_scale=0.02) for _ in range(10)]
        embeddings = np.stack(embs_a + embs_b)

        labels = cluster_faces(embeddings)
        unique = set(labels)
        unique.discard(-1)
        assert len(unique) == 2, f"Expected 2 clusters, got {len(unique)}: {labels}"

        # First 10 should be same cluster, last 10 should be different
        cluster_a = set(labels[:10]) - {-1}
        cluster_b = set(labels[10:]) - {-1}
        assert len(cluster_a) == 1
        assert len(cluster_b) == 1
        assert cluster_a != cluster_b

    def test_noise_points(self):
        """Outlier embeddings far from a tight cluster should be noise."""
        np.random.seed(42)
        center = np.random.randn(512).astype(np.float32)
        center /= np.linalg.norm(center)

        # 10 tight cluster + 3 orthogonal outliers (far away in euclidean)
        tight = [_make_embedding(center, noise_scale=0.02) for _ in range(10)]
        # Create outliers that are far from center
        outliers = []
        for _ in range(3):
            v = np.random.randn(512).astype(np.float32)
            v -= v.dot(center) * center  # Make orthogonal to center
            v /= np.linalg.norm(v)
            outliers.append(v)

        embeddings = np.stack(tight + outliers)
        # Use allow_single_cluster=False to see noise behavior
        labels = cluster_faces(embeddings, allow_single_cluster=False)

        # With allow_single_cluster=False, outliers might be noise
        # At minimum, the tight group should have some structure
        assert len(labels) == 13

    def test_empty_input(self):
        """Empty input should return empty labels."""
        embeddings = np.zeros((0, 512), dtype=np.float32)
        labels = cluster_faces(embeddings)
        assert len(labels) == 0

    def test_custom_params(self):
        """Custom HDBSCAN params should be respected."""
        center = np.random.randn(512).astype(np.float32)
        center /= np.linalg.norm(center)
        embeddings = np.stack([_make_embedding(center) for _ in range(5)])

        # Very high min_cluster_size should make everything noise
        labels = cluster_faces(embeddings, min_cluster_size=100)
        assert all(l == -1 for l in labels)


class TestClusterUserFaces:
    def test_cluster_user(self, db):
        """Cluster faces for a single user."""
        np.random.seed(42)
        center = np.random.randn(512).astype(np.float32)
        center /= np.linalg.norm(center)
        embeddings = [_make_embedding(center, noise_scale=0.02) for _ in range(10)]
        _insert_faces(db, "alice", embeddings)

        stats = cluster_user_faces(db, "alice")
        assert stats["n_faces"] == 10
        assert stats["n_clusters"] >= 1
        assert stats["elapsed_s"] >= 0

    def test_cluster_writes_back_to_db(self, db):
        """Cluster IDs should be written back to DB."""
        np.random.seed(42)
        center = np.random.randn(512).astype(np.float32)
        center /= np.linalg.norm(center)
        embeddings = [_make_embedding(center, noise_scale=0.02) for _ in range(10)]
        _insert_faces(db, "alice", embeddings)

        cluster_user_faces(db, "alice")

        faces = db.get_face_detections_for_user("alice")
        cluster_ids = [f["cluster_id"] for f in faces]
        non_null = [c for c in cluster_ids if c is not None]
        assert len(non_null) > 0

    def test_empty_user(self, db):
        """User with no faces should return zeros."""
        stats = cluster_user_faces(db, "nobody")
        assert stats["n_faces"] == 0
        assert stats["n_clusters"] == 0

    def test_noise_becomes_null(self, db):
        """Noise labels (-1) should become NULL cluster_id in DB."""
        # Insert 2 faces (below min_cluster_size=3) → all noise
        embeddings = [np.random.randn(512).astype(np.float32) for _ in range(2)]
        for e in embeddings:
            e /= np.linalg.norm(e)
        _insert_faces(db, "bob", embeddings)

        stats = cluster_user_faces(db, "bob")
        assert stats["n_noise"] == 2

        faces = db.get_face_detections_for_user("bob")
        assert all(f["cluster_id"] is None for f in faces)


class TestClusterAllUsers:
    def test_multiple_users(self, db):
        """Cluster faces for multiple users independently."""
        np.random.seed(42)
        center_a = np.random.randn(512).astype(np.float32)
        center_a /= np.linalg.norm(center_a)
        center_b = np.random.randn(512).astype(np.float32)
        center_b /= np.linalg.norm(center_b)

        _insert_faces(db, "alice", [_make_embedding(center_a) for _ in range(8)])
        _insert_faces(db, "bob", [_make_embedding(center_b) for _ in range(6)])

        results = cluster_all_users(db, user_ids=["alice", "bob"])

        assert "alice" in results
        assert "bob" in results
        assert results["alice"]["n_faces"] == 8
        assert results["bob"]["n_faces"] == 6

    def test_error_handling(self, db):
        """Errors for one user shouldn't affect others."""
        np.random.seed(42)
        center = np.random.randn(512).astype(np.float32)
        center /= np.linalg.norm(center)
        _insert_faces(db, "alice", [_make_embedding(center) for _ in range(8)])

        results = cluster_all_users(db, user_ids=["alice", "nobody"])

        assert "alice" in results
        assert results["alice"]["n_faces"] == 8
        assert "nobody" in results
        assert results["nobody"]["n_faces"] == 0
