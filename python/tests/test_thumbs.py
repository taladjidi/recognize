"""thumbs.py and the service's thumbnail pass: orientation, ICC, sizes, video, HEIC, stale, orphans, fail markers."""

import errno
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
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
        real = Image.Image.thumbnail
        monkeypatch.setattr(Image.Image, "thumbnail", lambda im, *a, **k: seen.append(im.size) or real(im, *a, **k))
        root, _ = _make(tmp_path, src)
        assert seen[0] == (1024, 768)  # im.draft: DCT-scaled 1/4, not a 12 Mpx decode
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


    @pytest.mark.parametrize("arr, mean", [
        (np.full((60, 60), 30000, np.uint16), 116),             # I;16: 30000/65535 of white
        (np.full((60, 60), 30000, np.int32), 116),              # I holding 16-bit data
        (np.linspace(100000, 200000, 3600).reshape(60, 60).astype(np.int32), 127),  # I: stretched
        (np.full((60, 60), 0.5, np.float32), 127),              # F in 0-1
        (np.linspace(-50, 50, 3600).reshape(60, 60).astype(np.float32), 127),  # F out of 0-1: stretched
    ], ids=["I16", "I-16bit", "I32", "F01", "Fwide"])
    def test_deep_grey_is_scaled_to_8_bit(self, tmp_path, arr, mean):
        src = tmp_path / "g.tif"
        Image.fromarray(arr).save(src)
        root, status = _make(tmp_path, src, "image/tiff")
        assert status == ("made", "")
        with Image.open(_thumb(root, 7, 1024)) as t:
            assert t.mode == "L" and abs(np.asarray(t).mean() - mean) < 3

    def test_cmyk_profile_is_dropped(self, tmp_path):
        src = tmp_path / "c.jpg"
        cmyk_icc = b"\0" * 16 + b"CMYK" + b"\0" * 108
        Image.new("CMYK", (64, 48), (0, 255, 255, 0)).save(src, "JPEG", icc_profile=cmyk_icc)
        assert Image.open(src).info.get("icc_profile") == cmyk_icc
        root, status = _make(tmp_path, src)
        assert status == ("made", "")
        with Image.open(_thumb(root, 7, 1024)) as t:
            assert t.mode == "RGB" and not t.info.get("icc_profile")
            r, g, b = t.getpixel((5, 5))
            assert r > 200 and g < 60 and b < 60

    def test_transparency_is_white(self, tmp_path):
        src = tmp_path / "a.png"
        Image.new("RGBA", (64, 48), (0, 0, 0, 0)).save(src)
        root, _ = _make(tmp_path, src, "image/png")
        with Image.open(_thumb(root, 7, 1024)) as t:
            assert t.getpixel((5, 5)) == (255, 255, 255)

    def test_non_jpeg_is_reduced_before_the_turn(self, tmp_path, monkeypatch):
        src = tmp_path / "big.png"
        im = Image.new("RGB", (4096, 3072), (0, 0, 255))
        im.paste((255, 0, 0), (0, 0, 1024, 1024))  # raw top-left red
        exif = Image.Exif()
        exif[0x0112] = 6
        im.save(src, exif=exif)
        seen = []
        real = Image.Image.transpose
        monkeypatch.setattr(Image.Image, "transpose", lambda im, m: seen.append(im.size) or real(im, m))
        root, status = _make(tmp_path, src, "image/png")
        assert status == ("made", "") and seen == [(1024, 768)]  # no full-size turned copy
        with Image.open(_thumb(root, 7, 1024)) as t:
            assert t.size == (768, 1024)
            r, g, b = t.getpixel((760, 10))  # raw top-left lands top-right
            assert r > 200 and b < 60

    def test_pixel_ceiling_spares_jpeg(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", None)  # as worker_init leaves it
        monkeypatch.setattr(thumbs, "MAX_PIXELS", 1000)
        for name in ("big.jpg", "big.png"):
            Image.new("RGB", (40, 30)).save(tmp_path / name)
        assert _make(tmp_path, tmp_path / "big.jpg", fid=1)[1] == ("made", "")
        root, (status, detail) = _make(tmp_path, tmp_path / "big.png", "image/png", fid=2)
        assert status == "failed" and "over 1000 pixels" in detail

    @pytest.mark.parametrize("exc", [
        subprocess.TimeoutExpired("ffmpeg", 20), MemoryError(), OSError(errno.EIO, "I/O error"),
        OSError(errno.EMFILE, "Too many open files"), OSError(errno.ENOMEM, "no memory"),
    ], ids=["timeout", "memory", "EIO", "EMFILE", "ENOMEM"])
    def test_transient_failure_gets_no_marker(self, tmp_path, monkeypatch, exc):
        src = tmp_path / "ok.jpg"
        Image.new("RGB", (10, 10)).save(src)
        root, _ = _make(tmp_path, src)

        def boom(*a):
            raise exc
        monkeypatch.setattr(thumbs, "_decode", boom)
        _, (status, _) = _make(tmp_path, src, mtime=MTIME + 1)
        assert status == "retry" and not os.path.exists(thumbs.fail_path(root, 7))
        assert os.path.exists(_thumb(root, 7, 256))  # the old thumbs stay until it decodes

    def test_truncated_file_gets_a_marker(self, tmp_path):
        src = tmp_path / "t.jpg"
        Image.effect_noise((600, 400), 64).convert("RGB").save(src, "JPEG")
        src.write_bytes(src.read_bytes()[:3000])
        root, (status, _) = _make(tmp_path, src)
        assert status == "failed" and os.path.exists(thumbs.fail_path(root, 7))

    @pytest.mark.parametrize("code, want", [(errno.ENOSPC, "nospace"), (errno.EDQUOT, "nospace"),
                                            (errno.EACCES, "error")])
    def test_write_failure(self, tmp_path, monkeypatch, code, want):
        src = tmp_path / "ok.jpg"
        Image.new("RGB", (10, 10)).save(src)

        def full(*a):
            raise OSError(code, os.strerror(code))
        monkeypatch.setattr(thumbs, "atomic_write", full)
        assert _make(tmp_path, src)[1][0] == want

    def test_worker_init(self):
        code = ("import ctypes, os, platform, thumbs; thumbs.worker_init()\n"
                "nr = {'x86_64': 252, 'aarch64': 31}[platform.machine()]  # ioprio_get\n"
                "print(os.nice(0), ctypes.CDLL(None).syscall(nr, 1, 0) >> 13, thumbs.Image.MAX_IMAGE_PIXELS)")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                             cwd=os.path.dirname(thumbs.__file__))
        assert out.stdout.split() == ["19", "3", "None"]  # nice 19, IOPRIO_CLASS_IDLE


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
                "size INTEGER, mimetype INTEGER, mtime INTEGER, mimepart INTEGER)")
    db._execute("INSERT INTO oc_storages VALUES (1, 'home::alice'), (2, 'home::bob'), (3, 'object::user:carol'), "
                "(4, 'local::/srv')")
    db._execute("INSERT INTO oc_mimetypes VALUES (1, 'image/jpeg'), (2, 'text/plain'), (3, 'image'), "
                "(4, 'video/mp4'), (5, 'video'), (6, 'text')")
    db._commit()
    root = tmp_path / "thumbs"
    root.mkdir()
    svc = Service.__new__(Service)  # no pipeline or models: only what _thumb_pass reads
    svc.pipeline = SimpleNamespace(stopping=threading.Event())
    svc.thumb_root, svc.thumb_workers, svc._thumb_root_warned = str(root), 1, False
    svc._thumb_enospc_logged, svc._thumb_suspects = False, set()
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
    db._execute("INSERT OR REPLACE INTO oc_filecache VALUES (?, ?, ?, ?, ?, ?, ?)",
                (fid, storage, path, p.stat().st_size, mime, mtime, MIMEPART[mime]))
    db._commit()


MIMEPART = {1: 3, 2: 6, 3: 3, 4: 5, 5: 5, 6: 6}  # as Nextcloud fills filecache.mimepart
CRASH, SLOW, FULL = 13, 14, 15


def _fake_make(job):
    """thumbs.make stand-in, pickled by name into the spawned worker: the fileid picks the behaviour."""
    if job[1] == CRASH:
        os.kill(os.getpid(), signal.SIGKILL)
    if job[1] == SLOW:
        time.sleep(60)
    if job[1] == FULL:
        return "nospace", f"{job[1]}: [Errno 28] No space left on device"
    return thumbs.make(job)


def _orphan(svc, fid=5):
    """Thumbs on disk for a fileid that is not in the DB."""
    for s in thumbs.SIZES:
        p = _thumb(svc.thumb_root, fid, s)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").close()
    return _thumb(svc.thumb_root, fid, 256)


def _run_pass(svc, db, timeout=60):
    out = []
    t = threading.Thread(target=lambda: out.append(svc._thumb_pass(db)), daemon=True)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), "the pass hangs"
    return out[0]


OLD_QUERY = (  # thumb_candidates before the single-query rewrite: the reference for its filter
    "SELECT fc.fileid, fc.path, m.mimetype, fc.mtime, fc.size, s.id AS storage_id_str FROM oc_filecache fc "
    "JOIN oc_storages s ON fc.storage = s.numeric_id JOIN oc_mimetypes m ON fc.mimetype = m.id "
    "WHERE fc.path LIKE 'files/%' AND fc.size > 0 "
    "AND (s.id LIKE 'home::%' OR s.id LIKE 'object::user:%') "
    "AND (m.mimetype LIKE 'image/%' OR m.mimetype LIKE 'video/%') ORDER BY fc.fileid")


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
        _file(env, 9, 2, "files/v.mp4", mime=4)
        rows = list(db.thumb_candidates(chunk=2))  # several chunks
        assert [r["fileid"] for r in rows] == [1, 2, 6, 9]
        assert rows[2]["path"] is None and rows[0]["path"].endswith("alice/files/a.jpg")
        assert rows[0]["mimetype"] == "image/jpeg" and rows[0]["mtime"] == MTIME
        assert rows[3]["mimetype"] == "video/mp4"

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

    def test_candidates_match_the_old_query(self, env):
        svc, db, _ = env
        for fid, (storage, path, mime, content) in enumerate([
                (1, "files/a.jpg", 1, None), (2, "files/b.jpg", 1, None), (1, "files_trashbin/files/t.jpg", 1, None),
                (1, "files/n.txt", 2, None), (1, "files/e.jpg", 1, b""), (3, "files/s3.jpg", 1, None),
                (4, "files/ext.jpg", 1, None), (1, "files/dir", 3, None), (2, "files/v.mp4", 4, None),
                (2, "files/vpart", 5, None), (1, "appdata_x/p.jpg", 1, None), (3, "files/v.mp4", 4, b"")], 1):
            _file(env, fid, storage, path, mime=mime, content=content)
        old = [{"fileid": r["fileid"], "mimetype": r["mimetype"], "mtime": r["mtime"], "size": r["size"],
                "path": db._resolve_storage_path(r["storage_id_str"], r["path"], db._config["datadirectory"])}
               for r in db._fetchall_dicts(db._execute(OLD_QUERY))]
        assert [r["fileid"] for r in old] == [1, 2, 6, 9]
        assert list(db.thumb_candidates(chunk=3)) == old

    def test_maintenance_mode_makes_and_deletes_nothing(self, env, tmp_path):
        svc, db, _ = env
        (tmp_path / "none.php").write_text("<?php $CONFIG = array (\n  'maintenance' => true,\n);")
        _file(env, 101, 1, "files/a.jpg")
        orphan = _orphan(svc)
        assert not svc._thumb_pass(db)
        assert os.path.exists(orphan) and thumbs.state(svc.thumb_root, 101, MTIME) == "todo"

    def test_maintenance_turned_on_mid_pass_deletes_nothing(self, env, monkeypatch):
        svc, db, _ = env
        _file(env, 101, 1, "files/a.jpg")
        orphan = _orphan(svc)
        calls = []
        monkeypatch.setattr(db, "check_maintenance_mode", lambda max_age=30: calls.append(max_age) or len(calls) > 1)
        assert not svc._thumb_pass(db)
        assert calls == [30, 0]  # the second read skips the cache
        assert os.path.exists(orphan) and thumbs.state(svc.thumb_root, 101, MTIME) == "fresh"

    def test_stop_while_reading_starts_no_pool(self, env, monkeypatch):
        svc, db, _ = env
        for fid in (101, 102, 103):
            _file(env, fid, 1, f"files/{fid}.jpg")
        orphan = _orphan(svc)
        real = thumbs.state
        monkeypatch.setattr(thumbs, "state", lambda *a: svc.pipeline.stopping.set() or real(*a))

        def no_pool(*a, **k):
            raise AssertionError("pool started after stop")
        monkeypatch.setattr("service.ProcessPoolExecutor", no_pool)
        assert not svc._thumb_pass(db)
        assert os.path.exists(orphan)

    def test_stop_while_decoding_ends_the_pass(self, env, monkeypatch):
        svc, db, _ = env
        _file(env, SLOW, 1, "files/slow.jpg")
        orphan = _orphan(svc)
        monkeypatch.setattr(thumbs, "make", _fake_make)
        threading.Timer(3, svc.pipeline.stopping.set).start()
        t0 = time.monotonic()
        assert not _run_pass(svc, db, timeout=20)
        assert time.monotonic() - t0 < 15 and os.path.exists(orphan)

    def test_worker_crash_aborts_the_pass_then_marks_the_culprit(self, env, monkeypatch, caplog):
        svc, db, _ = env
        for fid in (11, 12, CRASH, 16, 17):
            _file(env, fid, 1, f"files/{fid}.jpg")
        orphan = _orphan(svc)
        monkeypatch.setattr(thumbs, "make", _fake_make)
        assert not _run_pass(svc, db)
        assert os.path.exists(orphan) and CRASH in svc._thumb_suspects
        assert "decoder process died" in caplog.text
        assert _run_pass(svc, db)  # a new pool; the crasher alone dies again and gets a marker
        assert thumbs.state(svc.thumb_root, CRASH, MTIME) == "failed"
        assert all(thumbs.state(svc.thumb_root, f, MTIME) == "fresh" for f in (11, 12, 16, 17))
        assert not os.path.exists(orphan) and not svc._thumb_suspects

    def test_disk_full_aborts_the_pass_and_logs_once(self, env, monkeypatch, caplog):
        svc, db, _ = env
        _file(env, FULL, 1, "files/full.jpg")
        orphan = _orphan(svc)
        monkeypatch.setattr(thumbs, "make", _fake_make)
        assert not _run_pass(svc, db) and not _run_pass(svc, db)
        assert os.path.exists(orphan)
        assert caplog.text.count("No space left for thumbnails") == 1

    def test_empty_candidates_delete_nothing(self, env, caplog):
        svc, db, _ = env
        orphans = [_orphan(svc, fid) for fid in (5, 6)]
        caplog.set_level(logging.WARNING)
        assert svc._thumb_pass(db)
        assert all(os.path.exists(p) for p in orphans)
        assert "4 of 4 thumbs and markers have no candidate file (0 candidates)" in caplog.text

    def test_sweep_deleting_more_than_half_is_refused(self, env):
        svc, db, _ = env
        for fid in (101, 102, 103):
            _file(env, fid, 1, f"files/{fid}.jpg")
        assert svc._thumb_pass(db)
        db._execute("DELETE FROM oc_filecache WHERE fileid IN (102, 103)")
        db._commit()
        assert svc._thumb_pass(db)
        assert thumbs.state(svc.thumb_root, 102, MTIME) == "fresh"  # 4 of 6 would go: kept
        db._execute("DELETE FROM oc_filecache WHERE fileid = 103")
        _file(env, 102, 1, "files/102.jpg")
        assert svc._thumb_pass(db)
        assert not os.path.exists(_thumb(svc.thumb_root, 103, 256))  # 2 of 6: deleted

    def test_missing_root_is_a_no_op(self, env):
        svc, db, _ = env
        shutil.rmtree(svc.thumb_root)
        assert not svc._thumb_pass(db) and svc._thumb_root_warned
