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
    # 0.6 - |OLD_VEC - NEW_VEC| < 0: floor, still a removed face (threshold > 0)
    assert rows[removed]["threshold"] == pytest.approx(1e-6) and rows[removed]["cluster_id"] is None
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


def test_clustered_row_needs_a_mutual_best_match(env):
    """Group photo: a named sideways box covers two faces. Its best face is someone else's
    better match, so greedy alone would hand it the second face (another person). It is
    left alone, and the second face (inside the named box) is not inserted either: the file
    is listed for review."""
    db, _, root = env
    p = _photo(env, 10, "a.jpg", 6)
    named = _old(db, 10, _box(0.10, 0.10, 0.40, 0.40), cluster_id=5)
    other = _old(db, 10, _box(0.11, 0.10, 0.25, 0.25))
    fresh = [_box(0.10, 0.10, 0.25, 0.25), _box(0.27, 0.27, 0.23, 0.23)]
    nb = _rows(db)[named]
    assert rd.iou(nb, fresh[0]) > rd.iou(nb, fresh[1]) >= rd.MIN_IOU  # greedy would pair named with fresh[1]
    s = rd.run(db, Detector({p: fresh}), root)
    rows = _rows(db)
    assert rows[named] == nb
    assert rows[other]["x"] == pytest.approx(0.10)
    assert set(rows) == {named, other}
    assert (s["matched"], s["inserted"], s["kept"], s["needs_review"]) == (1, 0, 1, [p])


def test_threshold_shrinks_by_the_vector_move(env):
    """PHP's threshold is the OLD vector's distance to the cluster; the new vector may sit
    closer. t - |v_old - v_new| keeps a removed face out (triangle inequality)."""
    db, _, root = env
    p = _photo(env, 10, "a.jpg", 6)
    removed = _old(db, 10, _box(0.5, 0.5), threshold=2.0)
    rd.run(db, Detector({p: [_box(0.5, 0.5)]}), root)
    assert _rows(db)[removed]["threshold"] == pytest.approx(2.0 - 2 ** 0.5)


@pytest.mark.parametrize("kind", ["named", "removed"])
def test_no_fresh_duplicate_of_a_kept_named_or_removed_face(env, kind):
    db, _, root = env
    p = _photo(env, 10, "a.jpg", 6)
    kept = _old(db, 10, _box(0.1, 0.1, 0.6, 0.6), **({"cluster_id": 4} if kind == "named" else {"threshold": 0.5}))
    inside, apart = _box(0.3, 0.3, 0.1, 0.1), _box(0.8, 0.8, 0.1, 0.1)
    assert 0 < rd.iou(_rows(db)[kept], inside) < rd.MIN_IOU  # unmatched, but the same face
    s = rd.run(db, Detector({p: [inside, apart]}), root)
    rows = _rows(db)
    assert len(rows) == 2 and kept in rows
    assert [r["x"] for i, r in rows.items() if i != kept] == [pytest.approx(0.8)]
    assert (s["inserted"], s["needs_review"]) == (1, [p])


def test_row_changed_during_detection_skips_the_file(env):
    """The user renames / moves a face while the file is being detected: nothing is written."""
    db, _, root = env
    p, tom, removed, junk, det = _scene(env)
    before = _rows(db)
    inner = det.__call__

    def detect(path):
        db._execute("UPDATE oc_recognize_face_detections SET cluster_id = 8 WHERE id = ?", (tom,))
        db._commit()
        return inner(path)

    s = rd.run(db, detect, root)
    before[tom]["cluster_id"] = 8
    assert _rows(db) == before
    assert (s["raced"], s["matched"], s["inserted"], s["deleted"]) == (1, 0, 0, 0)


def test_deleted_counts_rows_actually_deleted(env):
    db, _, root = env
    p, tom, removed, junk, det = _scene(env)
    inner = det.__call__

    def detect(path):  # the daemon clusters the false positive meanwhile: the guard keeps it
        db._execute("UPDATE oc_recognize_face_detections SET cluster_id = 9 WHERE id = ?", (junk,))
        db._commit()
        return inner(path)

    s = rd.run(db, detect, root)
    assert junk in _rows(db) and s["deleted"] == 0


def test_crash_after_commit_never_leaves_the_old_crop(env, monkeypatch):
    """Power cut between the DB commit and the new crop: the old (sideways-box) crop must be
    gone, so the backfill rebuilds it from the updated row."""
    import crops
    db, _, root = env
    p, tom, removed, junk, det = _scene(env)
    os.makedirs(os.path.join(root, "alice"))
    old_crop = os.path.join(root, "alice", f"{tom}.jpg")
    open(old_crop, "wb").write(b"old sideways crop")

    def cut(*a, **k):
        raise KeyboardInterrupt("power cut")

    monkeypatch.setattr(crops, "write_crop", cut)
    with pytest.raises(KeyboardInterrupt):
        rd.run(db, det, root)
    assert _rows(db)[tom]["x"] == pytest.approx(0.12)
    assert not os.path.exists(old_crop)


def test_dry_run_ends_the_read_snapshot_per_file(env, monkeypatch):
    db, _, root = env
    paths = [_photo(env, i, f"{i}.jpg", 6) for i in (1, 2, 3)]
    for i in (1, 2, 3):
        _old(db, i, _box(0.1, 0.1))
    calls = []
    real = db.commit
    monkeypatch.setattr(db, "commit", lambda: calls.append(1) or real())
    rd.run(db, Detector({p: [_box(0.1, 0.1)] for p in paths}), root, dry_run=True)
    assert len(calls) >= 3


def test_clustered_tie_is_not_a_match():
    named = {**_box(0.5, 0.5, 0.25, 0.25), "cluster_id": 3}
    left, right = _box(0.375, 0.5, 0.25, 0.25), _box(0.625, 0.5, 0.25, 0.25)  # exactly equal IoU
    assert rd.iou(named, left) == rd.iou(named, right) >= rd.MIN_IOU
    assert rd.match([named], [left, right]) == []
    assert len(rd.match([{**named, "cluster_id": None}], [left, right])) == 1


@pytest.fixture
def queue(env):
    db, _, _ = env
    db.create_pending_table()
    db._execute("CREATE TABLE oc_appconfig (appid TEXT, configkey TEXT, configvalue TEXT, "
                "PRIMARY KEY (appid, configkey))")
    db._execute("INSERT INTO oc_storages VALUES (2, 'local::/mnt/x/')")
    db._commit()
    return env


def _pending(db):
    return {r[0]: (r[1], r[2]) for r in db._execute(
        "SELECT file_id, storage_id, action FROM oc_recognize_pending").fetchall()}


def _clear_pending(db):
    db._execute("DELETE FROM oc_recognize_pending")
    db._commit()


def test_requeue_faceless_rotated_owner_images_only(queue):
    db, _, _ = queue
    _photo(queue, 1, "b.jpg", 6)          # rotated, no faces: requeued
    _photo(queue, 2, "a.jpg", 3)          # rotated, no faces: requeued
    _photo(queue, 3, "c.jpg", 1)          # upright: scanned, not requeued
    _photo(queue, 4, "d.jpg", 8)          # rotated, has a detection
    _old(db, 4, _box(0.1, 0.1))
    _photo(queue, 5, "e.jpg", 6, mime=2)  # not an image mimetype
    _photo(queue, 6, "f.jpg", 5)          # already pending (a delete): untouched
    db.upsert_pending(6, 1, 2)
    _photo(queue, 7, "g.jpg", 6)          # in the trash, not the owner's files/
    db._execute("UPDATE oc_filecache SET path = 'files_trashbin/files/g.jpg' WHERE fileid = 7")
    _photo(queue, 8, "h.jpg", 6)          # not a user storage
    db._execute("UPDATE oc_filecache SET storage = 2 WHERE fileid = 8")
    db._commit()
    s = rd.requeue_faceless(db)
    assert _pending(db) == {1: (1, 0), 2: (1, 0), 6: (1, 2)}  # 0 = PendingService::ACTION_CREATED
    assert (s["scanned"], s["rotated"], s["requeued"]) == (3, 2, 2)


def test_requeue_faceless_limit_resumes_and_a_finished_pass_requeues_nothing(queue):
    db, _, _ = queue
    for i, name in ((1, "c.jpg"), (2, "a.jpg"), (3, "b.jpg")):
        _photo(queue, i, name, 6)
    assert rd.requeue_faceless(db, limit=1)["requeued"] == 1
    assert set(_pending(db)) == {2}  # path order: a.jpg first
    _clear_pending(db)  # the daemon ran it and found no face: still faceless
    assert rd.requeue_faceless(db)["requeued"] == 2  # resumes after a.jpg
    assert set(_pending(db)) == {1, 3}
    _clear_pending(db)
    _photo(queue, 4, "z.jpg", 6)  # uploaded later: the fixed daemon already did it
    assert rd.requeue_faceless(db)["requeued"] == 0  # pass complete
    assert _pending(db) == {}


def test_requeue_never_overwrites_a_pending_row(queue, monkeypatch):
    """PHP queues a delete between the scan and the requeue: it must stay a delete."""
    db, _, _ = queue
    _photo(queue, 1, "a.jpg", 6)
    rows = db.faceless_image_files()
    db.upsert_pending(1, 1, 2)
    monkeypatch.setattr(db, "faceless_image_files", lambda: rows)
    rd.requeue_faceless(db)
    assert _pending(db) == {1: (1, 2)}


def test_requeue_faceless_dry_run_writes_nothing(queue):
    db, _, _ = queue
    _photo(queue, 1, "a.jpg", 6)
    _photo(queue, 2, "b.jpg", 6)
    assert rd.requeue_faceless(db, dry_run=True, limit=1)["requeued"] == 1
    assert _pending(db) == {}
    assert db._execute("SELECT COUNT(*) FROM oc_appconfig").fetchone()[0] == 0
    assert rd.requeue_faceless(db, dry_run=True)["requeued"] == 2


def test_removed_row_needs_a_mutual_best_match_too(env):
    """Same group photo, but the big sideways box is a face the user REMOVED from a person
    (threshold > 0, no cluster): it must not take a neighbour's face either (Opus re-review)."""
    db, _, root = env
    p = _photo(env, 10, "a.jpg", 6)
    removed = _old(db, 10, _box(0.10, 0.10, 0.40, 0.40), threshold=1.5)
    other = _old(db, 10, _box(0.11, 0.10, 0.25, 0.25))
    fresh = [_box(0.10, 0.10, 0.25, 0.25), _box(0.27, 0.27, 0.23, 0.23)]
    before = _rows(db)[removed]
    s = rd.run(db, Detector({p: fresh}), root)
    assert _rows(db)[removed] == before
    assert p in s["needs_review"]


def test_update_guard_tolerates_threshold_round_trip(env):
    """MariaDB may return a DOUBLE with fewer digits than it stores; the race guard must not
    treat that as a concurrent edit (it would skip every removed face forever)."""
    db, _, _ = env
    rid = _old(db, 10, _box(0.5, 0.5), threshold=0.123456789012345)
    row = dict(_rows(db)[rid], id=rid)
    row["threshold"] = 0.123456789012  # what a less precise round trip could give back
    face = {**_box(0.4, 0.4), "vector": [0.0] * 4}
    assert db.update_face_detection_box(row, face) == 1
