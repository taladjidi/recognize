"""Photos grid thumbnails on disk: <thumb_root>/<fileid % 100:02d>/<fileid>-<size>.jpg.

Served by nginx (/recognize-thumbs/, secure_link) with URLs minted by api_server.py, so the
grid never goes through PHP. Derived data, rebuilt by the service's _thumb_loop: a thumb's
mtime is its source's filecache mtime, so stale == mtime differs. A source that fails to
decode gets a 0-byte <fileid>.fail with the same mtime and is not retried until it changes.
Nothing outside thumb_root is written, and only names this module makes are deleted.
"""

import os
import re
import subprocess
from io import BytesIO

from PIL import Image, ImageOps

import crops  # noqa: F401  registers the HEIC/HEIF opener (upright pixels, orientation 1)
from crops import atomic_write

DEFAULT_THUMB_ROOT = "/var/lib/nextcloud/recognize-thumbs"
SIZES = (256, 1024)  # ascending: the largest is decoded, the others derived from it
QUALITY = 80  # JPEG: ~5 ms to encode; WebP saved only 10-25% for 10-50x the time (2026-11-06)
FFMPEG = "/usr/bin/ffmpeg"
VIDEO_SEEK = "1"  # seconds: past fade-ins; clips shorter than this fall back to the first frame
VIDEO_TIMEOUT = 20
NAME_RE = re.compile(r"^([0-9]+)(?:-(?:%s)\.jpg|\.fail)$" % "|".join(map(str, SIZES)))
TMP_RE = re.compile(r"^\..*\.tmp$")  # crops.atomic_write temp names


def thumb_name(fileid, size):
    """Path relative to the root (and to the /recognize-thumbs/ URL prefix)."""
    return f"{fileid % 100:02d}/{fileid}-{size}.jpg"


def fail_path(root, fileid):
    return os.path.join(root, f"{fileid % 100:02d}", f"{fileid}.fail")


def _mtime_is(path, mtime, nonempty):
    try:
        st = os.stat(path)
    except OSError:
        return False
    return int(st.st_mtime) == mtime and (st.st_size > 0 or not nonempty)


def is_fresh(root, fileid, mtime):
    """Every size exists, is non-empty, and has the source's mtime."""
    return all(_mtime_is(os.path.join(root, thumb_name(fileid, s)), mtime, True) for s in SIZES)


def state(root, fileid, mtime):
    """'fresh', 'failed' (a marker for this mtime) or 'todo'."""
    if is_fresh(root, fileid, mtime):
        return "fresh"
    return "failed" if _mtime_is(fail_path(root, fileid), mtime, False) else "todo"


def _video_frame(src):
    """One RGB frame about VIDEO_SEEK s in (ffmpeg auto-rotates), else the first frame."""
    for seek in (["-ss", VIDEO_SEEK], []):
        out = subprocess.run(
            [FFMPEG, "-nostdin", "-v", "error", *seek, "-i", src, "-an", "-sn", "-dn",
             "-frames:v", "1", "-f", "image2pipe", "-c:v", "ppm", "-"],
            capture_output=True, timeout=VIDEO_TIMEOUT, check=False)
        if out.stdout:  # a seek past the end exits 0 with no frame
            return Image.open(BytesIO(out.stdout))
    raise ValueError(f"ffmpeg gave no frame: {out.stderr[-200:]!r}")


def _decode(src, mimetype, box):
    """(upright image fitted within box x box, never upscaled; ICC profile bytes or None)."""
    if mimetype.startswith("video/"):
        im, icc = _video_frame(src), None
    else:
        with Image.open(src) as raw:
            if raw.format == "JPEG":
                # Smallest DCT scale that still covers the fitted size: decodes 2-8x less.
                w, h = raw.size
                f = min(1.0, box / max(w, h))
                raw.draft("RGB", (max(1, int(w * f)), max(1, int(h * f))))
            icc = raw.info.get("icc_profile")
            im = ImageOps.exif_transpose(raw)  # loads; a copy even when already upright
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        im = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im.convert("RGBA"))
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    # Keep the profile only when it describes the output pixels (Display P3 stays P3; CMYK is dropped).
    if icc and icc[16:20] != (b"GRAY" if im.mode == "L" else b"RGB "):
        icc = None
    im.thumbnail((box, box), Image.Resampling.LANCZOS)
    return im, icc


def make(job):
    """Pool worker: write every size of one file, or its fail marker.

    job = (root, fileid, src path, mimetype, filecache mtime). Returns (status, detail),
    status 'made', 'failed' (marker written), 'missing' (unreadable now: retried next pass,
    no marker) or 'error' (writing failed). Never raises."""
    root, fileid, src, mimetype, mtime = job
    if not os.access(src, os.R_OK):
        return "missing", src
    try:
        try:
            im, icc = _decode(src, mimetype, SIZES[-1])
        except Exception as e:  # anything the decoders throw: corrupt, unsupported, ffmpeg timeout
            for s in SIZES:  # a changed file that no longer decodes must not keep its old thumbs
                try:
                    os.unlink(os.path.join(root, thumb_name(fileid, s)))
                except FileNotFoundError:
                    pass
            atomic_write(fail_path(root, fileid), b"", mtime)
            return "failed", f"{fileid}: {e}"
        for s in reversed(SIZES):
            if s != SIZES[-1]:
                im = im.copy()
                im.thumbnail((s, s), Image.Resampling.LANCZOS)
            buf = BytesIO()
            im.save(buf, "JPEG", quality=QUALITY, icc_profile=icc)
            atomic_write(os.path.join(root, thumb_name(fileid, s)), buf.getvalue(), mtime)
        try:
            os.unlink(fail_path(root, fileid))
        except FileNotFoundError:
            pass
        return "made", ""
    except Exception as e:
        return "error", f"{fileid}: {e}"


def remove_orphans(root, keep):
    """Delete thumbs and markers whose fileid is not in keep, and temp-file debris.

    Call with no writer running (debris is then always a killed write). Returns the count."""
    removed = 0
    with os.scandir(root) as shards:
        for d in shards:
            if not (len(d.name) == 2 and d.name.isdigit() and d.is_dir(follow_symlinks=False)):
                continue
            with os.scandir(d.path) as it:
                for e in it:
                    m = NAME_RE.match(e.name)
                    if (m and int(m.group(1)) not in keep or TMP_RE.match(e.name)) \
                            and e.is_file(follow_symlinks=False):
                        try:
                            os.unlink(e.path)
                            removed += 1
                        except FileNotFoundError:
                            pass
    return removed
