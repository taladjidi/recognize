"""Tests for DB.get_unclustered_signature (used by cluster_all_users to skip unchanged users)."""

import pytest

from db import DB


@pytest.fixture
def db():
    d = DB({
        "dbtype": "sqlite3",
        "dbhost": "",
        "dbuser": "",
        "dbpassword": "",
        "dbname": ":memory:",
        "dbtableprefix": "oc_",
        "datadirectory": "/tmp/data",
    })
    d.create_face_detections_table()
    d.create_face_clusters_table()
    yield d
    d.close()


def test_signature_tracks_unclustered_rows(db):
    assert db.get_unclustered_signature("alice") == (0, 0, 0)
    ids = []
    for i in range(3):
        face = {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4, "vector": [0.1] * 512}
        ids.append(db.insert_face_detection(file_id=100 + i, user_id="alice", face=face))
    assert db.get_unclustered_signature("alice") == (3, max(ids), sum(ids))
    cid = db.create_face_cluster("alice")
    db.assign_detection_to_cluster(ids[0], cid)
    db.commit()
    assert db.get_unclustered_signature("alice") == (2, max(ids), sum(ids) - ids[0])
