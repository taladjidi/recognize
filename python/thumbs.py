"""Photos grid thumbnails on disk: <thumb_root>/<fileid % 100:02d>/<fileid>-<size>.jpg.

Served by nginx (/recognize-thumbs/, secure_link) with URLs minted by api_server.py, so the
grid never goes through PHP. Derived data, rebuilt by the service's _thumb_loop: a thumb's
mtime is its source's filecache mtime, so stale == mtime differs. A source that fails to
decode gets a 0-byte <fileid>.fail with the same mtime and is not retried until it changes.
Nothing outside thumb_root is written, and only names this module makes are deleted.
"""

import ctypes
import errno
import logging
import os
import platform
import re
import signal
import subprocess
from io import BytesIO

from PIL import Image

import crops  # noqa: F401  registers the HEIC/HEIF opener (upright pixels, orientation 1)
from crops import _ORIENT, atomic_write

log = logging.getLogger(__name__)

DEFAULT_THUMB_ROOT = "/var/lib/nextcloud/recognize-thumbs"
SIZES = (256, 1024)  # ascending: the largest is decoded, the others derived from it
QUALITY = 80  # JPEG: ~5 ms to encode; WebP saved only 10-25% for 10-50x the time (2026-11-06)
FFMPEG = "/usr/bin/ffmpeg"
VIDEO_SEEK = "1"  # seconds: past fade-ins; clips shorter than this fall back to the first frame
VIDEO_TIMEOUT = 20
NAME_RE = re.compile(r"^([0-9]+)(?:-(?:%s)\.jpg|\.fail)$" % "|".join(map(str, SIZES)))
TMP_RE = re.compile(r"^\..*\.tmp$")  # crops.atomic_write temp names
MAX_PIXELS = 150_000_000  # a full decode of anything but a baseline JPEG (~450 MB RGB at this size)
JOB_TIMEOUT = 600  # seconds per file: SIGALRM ends a hung decode, the pass then marks it
# Failures that say nothing about the file: no marker, retried next pass.
TRANSIENT_ERRNOS = {errno.EIO, errno.EMFILE, errno.ENFILE, errno.ENOMEM, errno.EAGAIN}
IOPRIO_SET = {"x86_64": 251, "aarch64": 30}  # syscall numbers
_last_refusal = None


def _die_with_parent():
    ctypes.CDLL(None).prctl(1, 9)  # PR_SET_PDEATHSIG, SIGKILL


def worker_init(ffmpeg=FFMPEG):
    """Pool initializer: nice 19 and the idle I/O class, so decoding yields to classification and
    Nextcloud; Pillow's own pixel limit off, _decode applies MAX_PIXELS."""
    global FFMPEG
    FFMPEG = ffmpeg
    os.nice(19)
    Image.MAX_IMAGE_PIXELS = None
    try:
        with open("/proc/self/oom_score_adj", "w") as f:  # the kernel kills a decoder, not the daemon
            f.write("1000")
    except OSError:
        pass
    nr = IOPRIO_SET.get(platform.machine())
    if nr:  # best effort: ioprio_set(IOPRIO_WHO_PROCESS, this process, IOPRIO_CLASS_IDLE << 13)
        ctypes.CDLL(None, use_errno=True).syscall(nr, 1, 0, 3 << 13)


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
            capture_output=True, timeout=VIDEO_TIMEOUT, check=False,
            preexec_fn=_die_with_parent)  # a killed worker must not leave a 20 s ffmpeg behind
        if out.stdout:  # a seek past the end exits 0 with no frame
            return Image.open(BytesIO(out.stdout))
    raise ValueError(f"ffmpeg gave no frame: {out.stderr[-200:]!r}")


def _decode(src, mimetype, box):
    """(upright RGB or L image fitted within box x box, never upscaled; ICC profile bytes or None)."""
    if mimetype.startswith("video/"):
        im, icc = _video_frame(src), None
        im.thumbnail((box, box), Image.Resampling.LANCZOS)
    else:
        with Image.open(src) as im:
            w, h = im.size
            if w * h > MAX_PIXELS and (im.format != "JPEG" or im.info.get("progressive")):
                raise ValueError(f"{w}x{h} is over {MAX_PIXELS} pixels (only a baseline JPEG is decoded reduced)")
            if im.format == "JPEG":
                # Smallest DCT scale that still covers the fitted size: decodes 2-8x less.
                f = min(1.0, box / max(w, h))
                im.draft("RGB", (max(1, int(w * f)), max(1, int(h * f))))
            icc = im.info.get("icc_profile")
            orientation = im.getexif().get(0x0112, 1)  # read before the pixels: TIFF keeps it in the file
            if im.mode in ("P", "PA"):
                im = im.convert("RGBA")  # a palette is resized by nearest neighbour
            elif im.mode == "1":
                im = im.convert("L")
            elif im.mode.startswith("I;16"):
                im = im.convert("I")
            k = None
            if im.mode in ("I", "F"):  # 16/32-bit or float grey: map to 0-255, from the full image's range
                lo, hi = im.getextrema()
                if im.mode == "F" and lo >= 0 and hi <= 1:
                    lo, hi = 0, 1
                elif lo >= 0 and hi <= 65535:
                    lo, hi = 0, 65535  # 16-bit data keeps its exposure; anything else is stretched
                k = 255 / max(hi - lo, 1e-9)
            # Reduce before the transpose: a square box fits either way, and no full-size copy is made.
            im.thumbnail((box, box), Image.Resampling.LANCZOS)
            im.load()  # thumbnail() does not load an image that already fits, and the file closes here
        if k is not None:
            im = im.point(lambda v: v * k - lo * k).convert("L")
        if orientation in _ORIENT:
            im = im.transpose(_ORIENT[orientation])
    if im.mode in ("RGBA", "LA"):
        im = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im.convert("RGBA"))
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    # Keep the profile only when it describes the output pixels (Display P3 stays P3; CMYK is dropped).
    if icc and icc[16:20] != (b"GRAY" if im.mode == "L" else b"RGB "):
        icc = None
    return im, icc


def _transient(e):
    return isinstance(e, (subprocess.TimeoutExpired, MemoryError)) or \
        isinstance(e, OSError) and e.errno in TRANSIENT_ERRNOS


def mark_failed(root, fileid, mtime):
    """Write the fail marker; a changed file that no longer decodes must not keep its old thumbs."""
    for s in SIZES:
        try:
            os.unlink(os.path.join(root, thumb_name(fileid, s)))
        except FileNotFoundError:
            pass
    atomic_write(fail_path(root, fileid), b"", mtime)


def make(job):
    """Pool worker: write every size of one file, or its fail marker.

    job = (root, fileid, src path, mimetype, filecache mtime). Returns (status, detail),
    status 'made', 'failed' (marker written), 'missing' (unreadable now) or 'retry' (a timeout,
    no memory, an I/O error: both retried next pass, no marker), 'nospace' (thumb disk full) or
    'error' (writing failed). Never raises."""
    root, fileid, src, mimetype, mtime = job
    signal.alarm(JOB_TIMEOUT)  # SIGALRM's default action kills a hung decoder: the pass sees BrokenProcessPool
    try:
        if not os.access(src, os.R_OK):
            return "missing", src
        try:
            im, icc = _decode(src, mimetype, SIZES[-1])
        except Exception as e:  # anything the decoders throw: corrupt, truncated, unsupported
            if _transient(e):
                return "retry", f"{fileid}: {e!r}"
            mark_failed(root, fileid, mtime)
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
    except OSError as e:
        return "nospace" if e.errno in (errno.ENOSPC, errno.EDQUOT) else "error", f"{fileid}: {e}"
    except Exception as e:
        return "error", f"{fileid}: {e}"
    finally:
        signal.alarm(0)


def remove_orphans(root, keep):
    """Delete thumbs and markers whose fileid is not in keep, and temp-file debris.

    Deletes no thumb or marker when more than half of them would go: that is an empty or wrong
    filecache read, not deletions (logged; remove them by hand if they really were deleted).
    Call with no writer running (debris is then always a killed write). Returns the count."""
    global _last_refusal
    orphans, debris, total = [], [], 0
    with os.scandir(root) as shards:
        for d in shards:
            if not (len(d.name) == 2 and d.name.isdigit() and d.is_dir(follow_symlinks=False)):
                continue
            with os.scandir(d.path) as it:
                for e in it:
                    m = NAME_RE.match(e.name)
                    if not (m or TMP_RE.match(e.name)) or not e.is_file(follow_symlinks=False):
                        continue
                    if not m:
                        debris.append(e.path)
                        continue
                    total += 1
                    if int(m.group(1)) not in keep:
                        orphans.append(e.path)
    if 2 * len(orphans) > total:
        if _last_refusal != (len(orphans), total):  # once per state, not every pass
            log.warning("Thumbnails: %d of %d thumbs and markers have no candidate file (%d candidates); "
                        "not deleting them", len(orphans), total, len(keep))
            _last_refusal = (len(orphans), total)
        orphans = []
    else:
        _last_refusal = None
    removed = 0
    for path in orphans + debris:
        try:
            os.unlink(path)
            removed += 1
        except FileNotFoundError:
            pass
    return removed
