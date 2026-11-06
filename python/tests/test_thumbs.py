"""thumbs.py and the service's thumbnail pass: orientation, ICC, sizes, video, HEIC, stale, orphans, fail markers."""

import os
import shutil
import subprocess
import threading
from types import SimpleNamespace

import pytest
from PIL import Image, ImageCms

import thumbs
from db import DB
from service import Service

MTIME = 1_700_000_000


def _thumb(root, fid, size):
    return os.path.join(root, thumbs.thumb_name(fid, size))


def _make(tmp_path, src, mime="image/jpeg", fid=7, mtime=MTIME):
    root = tmp_path / "thumbs"
    root.mkdir(exist_ok=True)
    status = thumbs.make((str(root), fid, str(src), mime, mtime))
    return str(root), status


class TestMake:
    def test_exif_orientation_6_is_upright(self, tmp_path):
        src = tmp_path / "r.jpg"
        im = Image.new("RGB", (400, 200), (0, 0, 255))
        im.paste((255, 0, 0), (0, 0, 100, 100))  # raw top-left red
        exif = Image.Exif()
        exif[0x0112] = 6  # display = raw turned 90 degrees clockwise
        im.save(src, "JPEG", exif=exif, quality=95)
        root, status = _make(tmp_path, src)
        assert status == ("made", "")
        with Image.open(_thumb(root, 7, 1024)) as t:
            assert t.size == (200, 400)
            r, g, b = t.getpixel((190, 10))  # raw top-left lands top-right
            assert r > 200 and b < 60
            assert t.getexif().get(0x0112, 1) == 1
        with Image.open(_thumb(root, 7, 256)) as t:
            assert t.size == (128, 256)

    def test_icc_profile_kept(self, tmp_path):
        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        src = tmp_path / "p.jpg"
        Image.new("RGB", (64, 48), (10, 200, 30)).save(src, "JPEG", icc_profile=icc)
        root, _ = _make(tmp_path, src)
        for s in thumbs.SIZES:
            with Image.open(_thumb(root, 7, s)) as t:
                assert t.info.get("icc_profile") == icc

    def test_never_upscales_and_keeps_ratio(self, tmp_path):
        src = tmp_path / "s.png"
        Image.new("RGB", (300, 100), (5, 5, 5)).save(src)
        root, _ = _make(tmp_path, src, "image/png")
        assert Image.open(_thumb(root, 7, 1024)).size == (300, 100)
        assert Image.open(_thumb(root, 7, 256)).size == (256, 85)

    def test_large_jpeg_fits_box_and_decodes_reduced(self, tmp_path, monkeypatch):
        src = tmp_path / "big.jpg"
        Image.new("RGB", (4096, 3072), (100, 100, 100)).save(src, "JPEG")
        seen = []
        real = thumbs.ImageOps.exif_transpose
        monkeypatch.setattr(thumbs.ImageOps, "exif_transpose", lambda im: seen.append(im.size) or real(im))
        root, _ = _make(tmp_path, src)
        assert seen == [(1024, 768)]  # im.draft: DCT-scaled 1/4, not a 12 Mpx decode
        assert Image.open(_thumb(root, 7, 1024)).size == (1024, 768)
        assert Image.open(_thumb(root, 7, 256)).size == (256, 192)

    def test_files_carry_source_mtime(self, tmp_path):
        src = tmp_path / "m.jpg"
        Image.new("RGB", (10, 10)).save(src)
        root, _ = _make(tmp_path, src, mtime=MTIME)
        assert thumbs.state(root, 7, MTIME) == "fresh"
        assert thumbs.state(root, 7, MTIME + 1) == "todo"
        assert int(os.stat(_thumb(root, 7, 256)).st_mtime) == MTIME

    def test_fail_marker(self, tmp_path):
        src = tmp_path / "bad.jpg"
        src.write_bytes(b"not an image")
        root, (status, _) = _make(tmp_path, src)
        assert status == "failed"
        marker = thumbs.fail_path(root, 7)
        assert os.path.getsize(marker) == 0 and int(os.stat(marker).st_mtime) == MTIME
        assert thumbs.state(root, 7, MTIME) == "failed"
        assert thumbs.state(root, 7, MTIME + 1) == "todo"  # changed file: retried
        Image.new("RGB", (10, 10)).save(src, "JPEG")
        _, (status, _) = _make(tmp_path, src, mtime=MTIME + 1)
        assert status == "made" and not os.path.exists(marker)

    def test_file_that_breaks_drops_its_old_thumbs(self, tmp_path):
        src = tmp_path / "x.jpg"
        Image.new("RGB", (10, 10)).save(src, "JPEG")
        root, _ = _make(tmp_path, src)
        src.write_bytes(b"garbage")
        _make(tmp_path, src, mtime=MTIME + 5)
        assert not any(os.path.exists(_thumb(root, 7, s)) for s in thumbs.SIZES)

    def test_unreadable_source_gets_no_marker(self, tmp_path):
        root, (status, _) = _make(tmp_path, tmp_path / "gone.jpg")
        assert status == "missing" and not os.path.exists(thumbs.fail_path(root, 7))

    @pytest.mark.skipif(not os.access(thumbs.FFMPEG, os.X_OK), reason="no /usr/bin/ffmpeg")
    @pytest.mark.parametrize("duration", ["2", "0.3"])  # 0.3 s: shorter than the seek, first frame instead
    def test_video_frame(self, tmp_path, duration):
        src = tmp_path / "clip.mp4"
        subprocess.run([thumbs.FFMPEG, "-v", "error", "-f", "lavfi", "-i", f"testsrc=size=160x90:rate=10:duration={duration}",
                        "-pix_fmt", "yuv420p", str(src)], check=True)
        root, status = _make(tmp_path, src, "video/mp4")
        assert status == ("made", "")
        assert Image.open(_thumb(root, 7, 1024)).size == (160, 90)
        assert Image.open(_thumb(root, 7, 256)).size == (160, 90)

    def test_heic(self, tmp_path):
        pillow_heif = pytest.importorskip("pillow_heif")
        src = tmp_path / "p.heic"
        try:
            pillow_heif.from_pillow(Image.new("RGB", (64, 48), (200, 20, 20))).save(str(src))  # not via the opener crops registers
        except (KeyError, OSError, ValueError) as e:
            pytest.skip(f"pillow_heif cannot encode: {e}")
        root, status = _make(tmp_path, src, "image/heic")
        assert status == ("made", "")
        with Image.open(_thumb(root, 7, 1024)) as t:
            assert t.format == "JPEG" and t.size == (64, 48)


# ── The service pass (sqlite) ──────────────────────────────────────────

@pytest.fixture
def env(tmp_path):
    data = tmp_path / "data"
    cfg = {"dbtype": "sqlite3", "dbhost": "", "dbuser": "", "dbpassword": "", "dbname": str(tmp_path / "t.db"),
           "dbtableprefix": "oc_", "datadirectory": str(data), "config_php": str(tmp_path / "none.php")}
    db = DB(cfg)
    db._execute("CREATE TABLE oc_storages (numeric_id INTEGER PRIMARY KEY, id TEXT)")
    db._execute("CREATE TABLE oc_mimetypes (id INTEGER PRIMARY KEY, mimetype TEXT)")
    db._execute("CREATE TABLE oc_filecache (fileid INTEGER PRIMARY KEY, storage INTEGER, path TEXT, "
                "size INTEGER, mimetype INTEGER, mtime INTEGER)")
    db._execute("INSERT INTO oc_storages VALUES (1, 'home::alice'), (2, 'home::bob'), (3, 'object::user:carol'), "
                "(4, 'local::/srv')")
    db._execute("INSERT INTO oc_mimetypes VALUES (1, 'image/jpeg'), (2, 'text/plain'), (3, 'image')")
    db._commit()
    root = tmp_path / "thumbs"
    root.mkdir()
    svc = Service.__new__(Service)  # no pipeline or models: only what _thumb_pass reads
    svc.pipeline = SimpleNamespace(stopping=threading.Event())
    svc.thumb_root, svc.thumb_workers, svc._thumb_root_warned = str(root), 1, False
    yield svc, db, data
    db.close()


def _file(env, fid, storage, path, mime=1, mtime=MTIME, content=None):
    _, db, data = env
    user = {1: "alice", 2: "bob"}.get(storage, "x")
    p = data / user / path
    p.parent.mkdir(parents=True, exist_ok=True)
    if content is None:
        Image.new("RGB", (40, 30), (fid % 256, 0, 0)).save(p, "JPEG")
    else:
        p.write_bytes(content)
    db._execute("INSERT OR REPLACE INTO oc_filecache VALUES (?, ?, ?, ?, ?, ?)",
                (fid, storage, path, p.stat().st_size, mime, mtime))
    db._commit()


def _ino(root, fid):
    return os.stat(_thumb(root, fid, 256)).st_ino


class TestPass:
    def test_candidates_filter(self, env):
        svc, db, _ = env
        _file(env, 1, 1, "files/a.jpg")
        _file(env, 2, 2, "files/b.jpg")                # another user: still a candidate
        _file(env, 3, 1, "files_trashbin/files/t.jpg")  # trash
        _file(env, 4, 1, "files/n.txt", mime=2)
        _file(env, 5, 1, "files/empty.jpg", content=b"")
        _file(env, 6, 3, "files/s3.jpg")               # object storage: candidate, not on disk
        _file(env, 7, 4, "files/ext.jpg")              # external local storage: not a home
        _file(env, 8, 1, "files/dir", mime=3)          # the bare mimepart row
        rows = db.thumb_candidates(0, 100)
        assert [r["fileid"] for r in rows] == [1, 2, 6]
        assert rows[2]["path"] is None and rows[0]["path"].endswith("alice/files/a.jpg")
        assert rows[0]["mimetype"] == "image/jpeg" and rows[0]["mtime"] == MTIME
        assert [r["fileid"] for r in db.thumb_candidates(1, 1)] == [2]

    def test_pass_makes_skips_fresh_redoes_stale_and_removes_orphans(self, env):
        svc, db, _ = env
        root = svc.thumb_root
        _file(env, 101, 1, "files/a.jpg")
        _file(env, 102, 1, "files/bad.jpg", content=b"garbage")
        _file(env, 205, 2, "files/b.jpg")
        os.makedirs(os.path.join(root, "05"))
        for name in ("5-256.jpg", "5-1024.jpg", "5.fail", ".x.tmp", "notes.txt"):
            open(os.path.join(root, "05", name), "w").close()
        assert svc._thumb_pass(db)
        assert thumbs.state(root, 101, MTIME) == thumbs.state(root, 205, MTIME) == "fresh"
        assert thumbs.state(root, 102, MTIME) == "failed"
        assert sorted(os.listdir(os.path.join(root, "05"))) == ["205-1024.jpg", "205-256.jpg", "notes.txt"]

        ino, marker = _ino(root, 101), os.stat(thumbs.fail_path(root, 102)).st_ino
        assert svc._thumb_pass(db)
        assert _ino(root, 101) == ino and os.stat(thumbs.fail_path(root, 102)).st_ino == marker

        db._execute("UPDATE oc_filecache SET mtime = ? WHERE fileid = 101", (MTIME + 60,))
        db._execute("DELETE FROM oc_filecache WHERE fileid = 205")
        db._commit()
        assert svc._thumb_pass(db)
        assert _ino(root, 101) != ino and thumbs.state(root, 101, MTIME + 60) == "fresh"
        assert not os.path.exists(_thumb(root, 205, 256))

    def test_cut_pass_deletes_no_orphans(self, env):
        svc, db, _ = env
        _file(env, 101, 1, "files/a.jpg")
        os.makedirs(os.path.join(svc.thumb_root, "05"))
        orphan = os.path.join(svc.thumb_root, "05", "5-256.jpg")
        open(orphan, "w").close()
        svc.pipeline.stopping.set()
        assert not svc._thumb_pass(db)
        assert os.path.exists(orphan)

    def test_missing_root_is_a_no_op(self, env):
        svc, db, _ = env
        shutil.rmtree(svc.thumb_root)
        assert not svc._thumb_pass(db) and svc._thumb_root_warned
