"""Face crops: geometry, EXIF orientation, atomic writes, path safety, backfill/cleanup."""

import errno
import io
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image, ImageOps

import crops
from crops import CROP_SIZE, crop_path, make_crop, open_raw, write_crop

W, H = 400, 300
FACE = {"x": 150 / W, "y": 100 / H, "width": 100 / W, "height": 100 / H}  # centre (200, 150)


def _decode(data):
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB")).astype(int)


def _centroid(arr, rgb):
    mask = (np.abs(arr - np.array(rgb)).sum(axis=2)) < 120
    ys, xs = np.nonzero(mask)
    assert len(xs) > 0, f"marker {rgb} not found"
    return xs.mean(), ys.mean()


def _marked_jpeg(path, orientation):
    """Raw W x H grey photo: blue dot at the face centre, red marker up-left of it."""
    a = np.full((H, W, 3), 128, np.uint8)
    a[140:160, 190:210] = (0, 0, 255)
    a[80:110, 130:160] = (255, 0, 0)
    img = Image.fromarray(a)
    exif = Image.Exif()
    exif[0x0112] = orientation
    img.save(path, "JPEG", quality=95, exif=exif)


@pytest.mark.parametrize("orientation,expected", [
    (1, (-1, -1)), (3, (1, 1)), (6, (1, -1)), (8, (-1, 1)),
])
def test_crop_is_turned_upright_like_the_photo(tmp_path, orientation, expected):
    path = str(tmp_path / f"o{orientation}.jpg")
    _marked_jpeg(path, orientation)
    img, orient = open_raw(path)
    assert orient == orientation and img.size == (W, H)  # raw pixels, as detection sees them
    crop = _decode(make_crop(img, FACE, orient))
    assert crop.shape == (CROP_SIZE, CROP_SIZE, 3)
    rx, ry = _centroid(crop, (255, 0, 0))
    got = (int(np.sign(rx - CROP_SIZE / 2)), int(np.sign(ry - CROP_SIZE / 2)))
    # Independent reference: where Pillow's exif_transpose puts the marker in the full photo.
    full = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB")).astype(int)
    fx, fy = _centroid(full, (255, 0, 0))
    bx, by = _centroid(full, (0, 0, 255))
    assert got == (int(np.sign(fx - bx)), int(np.sign(fy - by))) == expected
    bcx, bcy = _centroid(crop, (0, 0, 255))
    assert abs(bcx - CROP_SIZE / 2) < 8 and abs(bcy - CROP_SIZE / 2) < 8  # face stays centred


def _region_diff(img, box, region):
    ref = np.asarray(img.resize((CROP_SIZE, CROP_SIZE), Image.Resampling.LANCZOS, box=region)).astype(int)
    return np.abs(_decode(make_crop(img, box)) - ref).mean()


def _gradient():
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    return Image.fromarray(np.stack([xs * 255 // W, ys * 255 // H, (xs + ys) % 256], -1).astype(np.uint8))


def test_box_near_edge_is_shifted_inside_and_stays_square():
    img = _gradient()
    box = {"x": 0.95, "y": 0.9, "width": 0.05, "height": 0.1}  # 20 x 30 px at the corner
    # side 1.6 * 30 = 48, pushed back inside: (352, 252)-(400, 300)
    assert _region_diff(img, box, (352, 252, 400, 300)) < 3


def test_box_larger_than_image_is_clamped_to_the_short_side():
    img = _gradient()
    # side min(1.6 * 400, 400, 300) = 300, centred on x then shifted: (50, 0)-(350, 300)
    assert _region_diff(img, {"x": 0, "y": 0, "width": 1, "height": 1}, (50, 0, 350, 300)) < 3


def test_tiny_box_still_gives_a_valid_crop():
    data = make_crop(_gradient(), {"x": 0.5, "y": 0.5, "width": 0.0001, "height": 0.0001})
    assert _decode(data).shape == (CROP_SIZE, CROP_SIZE, 3)


def test_draft_decode_keeps_geometry(tmp_path):
    """The backfill's reduced-scale JPEG decode yields the same crop as a full decode."""
    path = str(tmp_path / "big.jpg")
    Image.fromarray(np.kron(np.asarray(_gradient()), np.ones((8, 8, 1))).astype(np.uint8)).save(path, quality=95)
    full, _ = open_raw(path)
    small, _ = open_raw(path, [FACE])
    assert small.size[0] < full.size[0]
    assert np.abs(_decode(make_crop(full, FACE)) - _decode(make_crop(small, FACE))).mean() < 4


# ── Writes ─────────────────────────────────────────────────────────────

def test_atomic_write_modes_and_no_temp_left(tmp_path):
    root = tmp_path / "crops"
    root.mkdir()
    assert write_crop(str(root), "alice", 7, b"one")
    assert write_crop(str(root), "alice", 7, b"two")
    p = root / "alice" / "7.jpg"
    assert p.read_bytes() == b"two"
    assert os.stat(p).st_mode & 0o777 == 0o640
    assert os.stat(root / "alice").st_mode & 0o777 == 0o750
    assert os.listdir(root / "alice") == ["7.jpg"]


def test_failed_replace_keeps_old_crop_and_cleans_temp(tmp_path):
    root = tmp_path / "crops"
    root.mkdir()
    write_crop(str(root), "alice", 7, b"old")
    with patch("crops.os.replace", side_effect=OSError(errno.ENOSPC, "No space left")):
        assert write_crop(str(root), "alice", 7, b"new") is False
        assert write_crop(str(root), "alice", 8, b"new") is False  # logged once, no raise
    assert (root / "alice" / "7.jpg").read_bytes() == b"old"
    assert os.listdir(root / "alice") == ["7.jpg"]


def test_missing_root_is_not_created(tmp_path):
    assert write_crop(str(tmp_path / "nope"), "alice", 1, b"x") is False
    assert not (tmp_path / "nope").exists()


@pytest.mark.parametrize("uid", ["", ".", "..", "../x", "a/b", "/abs", "a..b", "a\0b", None, 5])
def test_hostile_user_ids_rejected(tmp_path, uid):
    root = tmp_path / "crops"
    root.mkdir()
    assert crop_path(str(root), uid, 1) is None
    assert write_crop(str(root), uid, 1, b"x") is False
    assert os.listdir(root) == [] and sorted(os.listdir(tmp_path)) == ["crops"]


@pytest.mark.parametrize("det_id", ["1", "1/../x", -1, 0, True, 1.0, None])
def test_bad_detection_ids_rejected(tmp_path, det_id):
    assert crop_path(str(tmp_path), "alice", det_id) is None
    assert write_crop(str(tmp_path), "alice", det_id, b"x") is False
    assert os.listdir(tmp_path) == []


# ── Service: write at submit, idle backfill + cleanup ──────────────────

@pytest.fixture
def svc(tmp_path):
    import json
    from service import Service
    cfg = tmp_path / "nc_config.json"
    cfg.write_text(json.dumps({
        "dbtype": "sqlite3", "dbhost": "", "dbuser": "", "dbpassword": "",
        "dbname": str(tmp_path / "t.db"), "dbtableprefix": "oc_", "datadirectory": "/tmp/data",
        "face_crop_root": str(tmp_path / "crops"),
    }))
    s = Service(str(cfg), models_dir=str(tmp_path), gpu=False)
    (tmp_path / "crops").mkdir()
    s.pipeline.db.create_face_detections_table()
    s.pipeline._enabled = {"faces"}
    yield s
    s.stop()


def _det(db, file_id, user, box=FACE):
    db.insert_face_detection(file_id, user, {**box, "vector": [0.0] * 4})
    return db._execute("SELECT MAX(id) FROM oc_recognize_face_detections").fetchone()[0]


def test_submit_writes_crops_under_returned_ids(svc, tmp_path):
    api = MagicMock()
    api.submit_results.return_value = [5, 6]
    svc.pipeline.nc_api = api
    item = {"id": 1, "file_id": 10, "added_at": 1, "owner": "alice"}
    res = {"faces": [{"x": 0}, {"x": 1}], "crops": [b"a", b"b"], "tags": []}
    with patch.object(svc.pipeline.db, "delete_pending_items"):
        assert svc._submit_items([item], {10: res})
    assert api.submit_results.call_args.kwargs["faces"] == res["faces"]  # crops never sent
    assert "crops" not in api.submit_results.call_args.kwargs
    assert sorted(os.listdir(tmp_path / "crops" / "alice")) == ["5.jpg", "6.jpg"]
    api.submit_results.return_value = []  # existing detections kept -> backfill's job
    with patch.object(svc.pipeline.db, "delete_pending_items"):
        assert svc._submit_items([{**item, "owner": "bob"}], {10: res})
    assert not (tmp_path / "crops" / "bob").exists()


def test_backfill_creates_missing_only_and_cleanup_removes_orphans_only(svc, tmp_path):
    db = svc.pipeline.db
    root = tmp_path / "crops"
    photo = str(tmp_path / "p.jpg")
    _marked_jpeg(photo, 1)
    a1, a2 = _det(db, 10, "alice"), _det(db, 10, "alice", {**FACE, "x": 0.1})
    nobody = _det(db, 10, "")
    gone_file = _det(db, 11, "alice")  # its photo cannot be resolved
    (root / "alice").mkdir()
    (root / "alice" / f"{a1}.jpg").write_bytes(b"keep")          # existing crop left alone
    (root / "alice" / "999.jpg").write_bytes(b"orphan")
    for name in ("notes.txt", ".x.tmp", "12.jpg.bak", "abc.jpg"):
        (root / "alice" / name).write_bytes(b"not ours")
    (root / "bob").mkdir()
    (root / "bob" / f"{a1}.jpg").write_bytes(b"alice's id, bob's dir")
    outside = tmp_path / "123.jpg"
    outside.write_bytes(b"outside")
    paths = {10: {"path": photo, "mimetype": "image/jpeg", "storage": 1, "owner": "alice"}}
    with patch.object(db, "resolve_file_paths", side_effect=lambda ids: {i: paths[i] for i in ids if i in paths}):
        for _ in range(4):
            svc._crop_maintenance()
    alice = set(os.listdir(root / "alice"))
    assert alice == {f"{a1}.jpg", f"{a2}.jpg", "notes.txt", ".x.tmp", "12.jpg.bak", "abc.jpg"}
    assert (root / "alice" / f"{a1}.jpg").read_bytes() == b"keep"
    assert _decode((root / "alice" / f"{a2}.jpg").read_bytes()).shape == (CROP_SIZE, CROP_SIZE, 3)
    assert os.listdir(root / "bob") == []
    assert not (root / "" / f"{nobody}.jpg").exists() and f"{gone_file}.jpg" not in alice
    assert outside.read_bytes() == b"outside"
    assert svc._crop_rest_until > 0  # full pass + round done -> rests


def test_maintenance_is_bounded_and_never_raises(svc, tmp_path):
    with patch.object(svc.pipeline.db, "detections_after", side_effect=RuntimeError("db gone")):
        svc._crop_maintenance()  # logged, not raised
    calls = []
    with patch.object(svc.pipeline.db, "detections_after", side_effect=lambda a, n: calls.append(n) or []):
        svc._crop_maintenance()
    assert calls == [svc_mod().CROP_BATCH]


def svc_mod():
    import service
    return service


def test_owner_from_storage_id_matches_php():
    from db import DB
    assert DB.owner_from_storage_id("home::alice") == "alice"
    assert DB.owner_from_storage_id("object::user:alice") == "alice"
    assert DB.owner_from_storage_id("local::/mnt/x/") == ""
    assert DB.owner_from_storage_id(None) == ""


def test_crop_size_of_a_face(tmp_path):
    """Size check on a photo-like image: well under a preview (~90 KB)."""
    rng = np.random.default_rng(0)
    img = Image.fromarray((np.asarray(_gradient()) + rng.integers(0, 20, (H, W, 3))).clip(0, 255).astype(np.uint8))
    assert 2_000 < len(make_crop(img, FACE)) < 40_000
    assert crops.QUALITY == 82


def test_face_classifier_crops_align_with_faces_on_raw_pixels(tmp_path):
    import sys
    sys.modules.setdefault("onnxruntime", MagicMock())
    from classifiers.faces import FaceClassifier
    path = str(tmp_path / "o6.jpg")
    _marked_jpeg(path, 6)
    clf = FaceClassifier.__new__(FaceClassifier)  # no model: infer_one is stubbed
    seen = []
    clf.infer_one = lambda bgr: seen.append(bgr.shape) or [dict(FACE), dict(FACE)]
    faces, out = clf.classify_with_crops(path)
    assert seen == [(H, W, 3)]  # detection sees raw (unrotated) pixels
    assert len(out) == len(faces) == 2 and all(_decode(c).shape == (CROP_SIZE, CROP_SIZE, 3) for c in out)
