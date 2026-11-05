"""Face crop JPEGs on disk: <crop_root>/<user_id>/<detection_id>.jpg.

Crops are derived data (rebuilt by the idle backfill), so a failed write is
logged and skipped, never fatal. Nothing outside crop_root is ever written or
deleted, and only <int>.jpg files inside a user dir are ever deleted.
"""

import errno
import io
import logging
import os
import re
import stat
import tempfile

from PIL import Image

log = logging.getLogger(__name__)

DEFAULT_CROP_ROOT = "/var/lib/nextcloud/recognize-faces"
CROP_SIZE = 256
EXPAND = 2.0
QUALITY = 82
CROP_NAME_RE = re.compile(r"^([0-9]+)\.jpg$")

# EXIF orientation -> the transpose that shows the image upright (as ImageOps.exif_transpose).
# Applied to the crop: detection and the stored box are in raw pixel coordinates.
_ORIENT = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}

_enospc_logged = False


def crop_side(img_w, img_h, box):
    """Side in pixels of the square crop for a relative box {x, y, width, height}."""
    return max(1.0, min(EXPAND * max(box["width"] * img_w, box["height"] * img_h), img_w, img_h))


def make_crop(img, box, orientation=1):
    """JPEG bytes of the square around box (EXPAND x its larger side, kept inside img).

    img is the raw (not EXIF-rotated) RGB image the box was detected on.
    """
    w, h = img.size
    side = crop_side(w, h, box)
    cx = (box["x"] + box["width"] / 2) * w
    cy = (box["y"] + box["height"] / 2) * h
    # Clamp by shifting the square inside the image, so it stays square (no distortion).
    left = min(max(cx - side / 2, 0.0), w - side)
    top = min(max(cy - side / 2, 0.0), h - side)
    out = img.resize((CROP_SIZE, CROP_SIZE), Image.Resampling.LANCZOS,
                     box=(left, top, left + side, top + side))
    if orientation in _ORIENT:
        out = out.transpose(_ORIENT[orientation])
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=QUALITY)
    return buf.getvalue()


def open_raw(path, boxes=()):
    """(raw RGB image, EXIF orientation). With boxes, JPEGs decode at the smallest
    DCT scale that still gives every crop >= CROP_SIZE pixels (much faster)."""
    with Image.open(path) as im:
        orientation = im.getexif().get(0x0112, 1)
        if boxes and im.format == "JPEG":
            w, h = im.size
            scale = min(1.0, min(CROP_SIZE / crop_side(w, h, b) for b in boxes))
            im.draft("RGB", (int(w * scale) + 1, int(h * scale) + 1))
        return im.convert("RGB"), orientation


def safe_user_dir(root, user_id):
    """<root>/<user_id>, or None when user_id could escape root or is empty."""
    if not isinstance(user_id, str) or not user_id or user_id in (".", "..") \
            or "/" in user_id or "\0" in user_id or ".." in user_id:
        return None
    return os.path.join(root, user_id)


def crop_path(root, user_id, detection_id):
    d = safe_user_dir(root, user_id)
    if d is None or type(detection_id) is not int or detection_id <= 0:
        return None
    return os.path.join(d, f"{detection_id}.jpg")


def write_crop(root, user_id, detection_id, data, overwrite=False):
    """Atomically write one crop; False (logged) on any failure or an existing crop (unless overwrite)."""
    global _enospc_logged
    path = crop_path(root, user_id, detection_id)
    if path is None:
        log.warning("Refusing face crop path for user %r, detection %r", user_id, detection_id)
        return False
    d = os.path.dirname(path)
    tmp = None
    try:
        if not os.path.isdir(d):
            # Root itself must exist (deploy.sh, apache:nginx 2750): never created here. mkdir
            # inherits its setgid group, so files land group nginx. No chmod unless the umask
            # took bits: apache is not in group nginx, so its chmod would clear S_ISGID.
            os.mkdir(d, 0o750)
            mode = os.stat(d).st_mode
            if mode & 0o777 != 0o750:
                os.chmod(d, 0o750 | (mode & stat.S_ISGID))
                if mode & stat.S_ISGID and not os.stat(d).st_mode & stat.S_ISGID:
                    log.warning("umask took bits of %s and fixing them cleared setgid: nginx may get 403", d)
        if not overwrite and has_crop(root, user_id, detection_id):
            return False
        # shortcut: no fsync; a crop torn by a power cut is 0 bytes and the backfill redoes it.
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".", suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            os.fchmod(f.fileno(), 0o640)
            f.write(data)
        os.replace(tmp, path)
        _enospc_logged = False
        return True
    except OSError as e:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        if e.errno in (errno.ENOSPC, errno.EDQUOT):
            if not _enospc_logged:
                log.error("No space left for face crops in %s; skipping crops until a write succeeds", root)
                _enospc_logged = True
        else:
            log.warning("Cannot write face crop %s: %s", path, e)
        return False


def has_crop(root, user_id, detection_id):
    path = crop_path(root, user_id, detection_id)
    try:
        return path is not None and os.stat(path).st_size > 0
    except OSError:
        return False


def crop_ids(user_dir):
    """{detection_id: filename} of the <int>.jpg regular files in user_dir."""
    out = {}
    with os.scandir(user_dir) as it:
        for e in it:
            m = CROP_NAME_RE.match(e.name)
            if m and e.is_file(follow_symlinks=False):
                out[int(m.group(1))] = e.name
    return out
