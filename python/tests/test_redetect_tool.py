"""tools/redetect_rotated.py: re-detect EXIF-rotated photos in place, keeping names and manual edits."""

import importlib.util
import os
import sys

import pytest
from PIL import Image

from db import DB

_spec = importlib.util.spec_from_file_location(
    "redetect_rotated", os.path.join(os.path.dirname(__file__), "..", "..", "tools", "redetect_rotated.py"))
rd = importlib.util.module_from_spec(_spec)
sys.modules["redetect_rotated"] = rd
_spec.loader.exec_module(rd)

OLD_VEC = [1.0, 0.0, 0.0, 0.0]
NEW_VEC = [0.0, 1.0, 0.0, 0.0]


def _box(x, y, w=0.2, h=0.2, vec=NEW_VEC):
    return {"x": x, "y": y, "width": w, "height": h, "vector": list(vec), "score": 0.9}


@pytest.fixture
def env(tmp_path):
    data = tmp_path / "data"
    db = DB({"dbtype": "sqlite3", "dbhost": "", "dbuser": "", "dbpassword": "", "dbname": str(tmp_path / "t.db"),
             "dbtableprefix": "oc_", "datadirectory": str(data)})
    db.create_face_detections_table()
    db.create_face_clusters_table()
    db._execute("CREATE TABLE oc_storages (numeric_id INTEGER PRIMARY KEY, id TEXT)")
    db._execute("CREATE TABLE oc_mimetypes (id INTEGER PRIMARY KEY, mimetype TEXT)")
    db._execute("CREATE TABLE oc_filecache (fileid INTEGER PRIMARY KEY, storage INTEGER, path TEXT, "
                "size INTEGER, mimetype INTEGER)")
    db._execute("INSERT INTO oc_storages VALUES (1, 'home::alice')")
    db._execute("INSERT INTO oc_mimetypes VALUES (1, 'image/jpeg'), (2, 'video/mp4')")
    db._commit()
    crop_root = tmp_path / "crops"
    crop_root.mkdir()
    yield db, data, str(crop_root)
    db.close()


def _photo(env, file_id, name, orientation, mime=1):
    db, data, _ = env
    p = data / "alice" / "files" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    exif[0x0112] = orientation
    Image.new("RGB", (40, 30), (90, 90, 90)).save(p, "JPEG", exif=exif)
    db._execute("INSERT INTO oc_filecache VALUES (?, 1, ?, ?, ?)", (file_id, f"files/{name}", p.stat().st_size, mime))
    db._commit()
    return str(p)


def _old(db, file_id, box, cluster_id=None, threshold=0.0):
    db.insert_face_detection(file_id, "alice", {**box, "vector": OLD_VEC})
    did = db._execute("SELECT MAX(id) FROM oc_recognize_face_detections").fetchone()[0]
    db._execute("UPDATE oc_recognize_face_detections SET cluster_id = ?, threshold = ? WHERE id = ?",
                (cluster_id, threshold, did))
    db._commit()
    return did


def _rows(db):
    cur = db._execute("SELECT id, file_id, user_id, x, y, width, height, face_vector, cluster_id, threshold "
                      "FROM oc_recognize_face_detections ORDER BY id")
    return {r["id"]: dict(r) for r in cur.fetchall()}


class Detector:
    def __init__(self, by_path):
        self.by_path, self.calls = by_path, []

    def __call__(self, path):
        self.calls.append(path)
        faces = [dict(f) for f in self.by_path.get(path, [])]
        return faces, [b"crop-%d" % i for i in range(len(faces))]


def _snapshot(env):
    db, _, root = env
    files = {os.path.join(d, f): open(os.path.join(d, f), "rb").read() for d, _, fs in os.walk(root) for f in fs}
    return _rows(db), files


def _scene(env):
    """One rotated photo: a named face, a removed face (threshold), a sideways false positive."""
    db, _, _ = env
    db._execute("INSERT INTO oc_recognize_face_clusters (id, title, user_id) VALUES (7, 'Tom', 'alice')")
    db._commit()
    p = _photo(env, 10, "a.jpg", 6)
    tom = _old(db, 10, _box(0.10, 0.10), cluster_id=7)
    removed = _old(db, 10, _box(0.50, 0.50), threshold=0.6)
    junk = _old(db, 10, _box(0.80, 0.05, 0.1, 0.1))  # unclustered, no threshold, no fresh match
    det = Detector({p: [_box(0.12, 0.11), _box(0.52, 0.49), _box(0.30, 0.70)]})
    return p, tom, removed, junk, det


def test_matched_rows_keep_id_cluster_threshold_and_name(env):
    db, _, root = env
    p, tom, removed, junk, det = _scene(env)
    s = rd.run(db, det, root)
    rows = _rows(db)
    assert rows[tom]["cluster_id"] == 7 and rows[tom]["x"] == pytest.approx(0.12)
    assert rows[tom]["face_vector"] == rd.json.dumps(NEW_VEC)
    assert rows[removed]["threshold"] == 0.6 and rows[removed]["cluster_id"] is None
    assert rows[removed]["y"] == pytest.approx(0.49)
    assert junk not in rows
    (new_id,) = set(rows) - {tom, removed}
    assert rows[new_id]["cluster_id"] is None and rows[new_id]["user_id"] == "alice"
    assert rows[new_id]["x"] == pytest.approx(0.30)
    assert db._execute("SELECT title FROM oc_recognize_face_clusters WHERE id = 7").fetchone()[0] == "Tom"
    for did in (tom, removed, new_id):  # every changed detection gets a fresh crop
        assert os.path.getsize(os.path.join(root, "alice", f"{did}.jpg")) > 0
    assert (s["files"], s["matched"], s["inserted"], s["deleted"]) == (1, 2, 1, 1)


def test_unmatched_clustered_or_thresholded_detections_are_never_deleted(env):
    db, _, root = env
    p = _photo(env, 10, "a.jpg", 8)
    named = _old(db, 10, _box(0.1, 0.1), cluster_id=3)
    removed = _old(db, 10, _box(0.5, 0.5), threshold=0.4)
    zero = _old(db, 10, _box(0.7, 0.7, 0.1, 0.1), cluster_id=0)
    s = rd.run(db, Detector({p: []}), root)  # nothing found upright
    rows = _rows(db)
    assert named in rows and rows[named]["cluster_id"] == 3 and rows[named]["face_vector"] == rd.json.dumps(OLD_VEC)
    assert removed in rows and zero not in rows
    assert (s["deleted"], s["kept"]) == (1, 2)


def test_rerun_changes_nothing(env):
    db, _, root = env
    det = _scene(env)[-1]
    rd.run(db, det, root)
    before = _snapshot(env)
    s = rd.run(db, det, root)
    assert _snapshot(env) == before
    assert (s["matched"], s["inserted"], s["deleted"], s["unchanged"]) == (0, 0, 0, 3)


def test_dry_run_writes_nothing_and_reports_the_same(env):
    db, _, root = env
    p, tom, removed, junk, det = _scene(env)
    before = _snapshot(env)
    dry = rd.run(db, det, root, dry_run=True)
    assert _snapshot(env) == before
    real = rd.run(db, det, root)
    assert dry == real


def test_only_rotated_images_with_faces_in_path_order(env):
    db, _, root = env
    upright = _photo(env, 1, "c.jpg", 1)
    rot_b = _photo(env, 2, "b.jpg", 3)
    rot_a = _photo(env, 3, "a.jpg", 6)
    video = _photo(env, 4, "v.jpg", 6, mime=2)
    _photo(env, 5, "noface.jpg", 6)
    for fid in (1, 2, 3, 4):
        _old(db, fid, _box(0.1, 0.1))
    det = Detector({})
    rd.run(db, det, root)
    assert det.calls == [rot_a, rot_b]
    assert upright and video


def test_limit_counts_changed_files_so_a_rerun_resumes(env):
    db, _, root = env
    paths = [_photo(env, i, f"{i}.jpg", 6) for i in (1, 2, 3)]
    for i in (1, 2, 3):
        _old(db, i, _box(0.1, 0.1))
    det = Detector({p: [_box(0.1, 0.1)] for p in paths})
    assert rd.run(db, det, root, limit=1)["matched"] == 1
    assert rd.run(db, det, root, limit=1)["matched"] == 1  # the first file is unchanged now: goes on
    s = rd.run(db, det, root)
    assert (s["matched"], s["unchanged"]) == (1, 2)


def test_detector_failure_leaves_the_file_alone(env):
    db, _, root = env
    p = _photo(env, 10, "a.jpg", 6)
    junk = _old(db, 10, _box(0.1, 0.1))

    def boom(path):
        raise OSError("unreadable")

    s = rd.run(db, boom, root)
    assert junk in _rows(db) and s["failed"] == 1


def test_greedy_match_prefers_best_iou_and_respects_minimum():
    old = [_box(0.0, 0.0), _box(0.5, 0.5)]
    new = [_box(0.51, 0.5), _box(0.01, 0.0), _box(0.9, 0.9, 0.05, 0.05)]
    assert sorted(rd.match(old, new)) == [(0, 1), (1, 0)]
    assert rd.iou(_box(0, 0), _box(0.5, 0.5)) == 0
