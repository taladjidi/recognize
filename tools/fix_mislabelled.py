#!/usr/bin/env python3
"""Give mislabelled photos real content-matching files, keeping their ids (faces, names).

Nextcloud picks the preview provider from the extension, so a Photoshop PSD named .JPG or
a JPEG named .HEIC never gets a preview (found 2026-11-05: 66 PSD-as-JPG, 7 JPEG-as-HEIC
in the Faces clusters). Input: the "failed:" lines of tools/warm_previews.py.

  - PSD named .jpg/.jpeg: the original is first COPYed server-side to
    /Archives/mislabelled-originals/<path>.psd (+ a .nomedia marker, so Recognize and
    Photos ignore the backups), its size checked, then the file is flattened to RGB, saved
    as JPEG q95 (ICC profile kept) and PUT over the same path: same file id.
  - JPEG named .heic/.heif: MOVEd to the same name with .jpg (same file id, content
    unchanged); never overwrites an existing target.
Anything else is reported and left alone. Dry run unless --apply.

  tools/fix_mislabelled.py ~/Downloads/warm-2.log [--apply] [--user tangui]
WebDAV with the app password in ~/.config/nc-bench/<user> (as bench_faces.py).
"""
import argparse
import base64
import io
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image

BACKUP_ROOT = "Archives/mislabelled-originals"


class Dav:
    def __init__(self, base, user, pw):
        self.root = f"{base}/remote.php/dav/files/{urllib.parse.quote(user)}/"
        self.auth = "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()

    def url(self, path):
        return self.root + urllib.parse.quote(path.strip("/"))

    def req(self, method, path, data=None, headers=None, ok=(200, 201, 204, 206, 207)):
        r = urllib.request.Request(self.url(path), data=data, method=method,
                                   headers={"Authorization": self.auth, **(headers or {})})
        try:
            with urllib.request.urlopen(r, timeout=300) as resp:
                return resp.status, resp.read(), dict(resp.headers)
        except urllib.error.HTTPError as e:
            if e.code in ok:
                return e.code, e.read(), dict(e.headers)
            raise

    def exists(self, path):
        try:
            self.req("PROPFIND", path, headers={"Depth": "0"})
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            raise

    def size(self, path):
        _, body, _ = self.req("PROPFIND", path, headers={"Depth": "0"},
                              data=b'<?xml version="1.0"?><d:propfind xmlns:d="DAV:"><d:prop><d:getcontentlength/></d:prop></d:propfind>')
        m = re.search(rb"<d:getcontentlength>(\d+)</d:getcontentlength>", body)
        return int(m.group(1)) if m else -1

    def mkcol_p(self, path):
        cur = ""
        for part in path.strip("/").split("/"):
            cur = f"{cur}/{part}" if cur else part
            if not self.exists(cur):
                self.req("MKCOL", cur)


def true_format(dav, path):
    _, head, _ = dav.req("GET", path, headers={"Range": "bytes=0-262143"})
    try:
        return Image.open(io.BytesIO(head)).format
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("warm_log")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--user", default="tangui")
    ap.add_argument("--base", default="https://aladjidi.xyz")
    a = ap.parse_args()
    pw = open(os.path.expanduser(f"~/.config/nc-bench/{a.user}")).read().strip()
    dav = Dav(a.base, a.user, pw)
    paths = sorted({m.group(1) for line in open(a.warm_log)
                    if (m := re.search(r"failed: .*\((.+)\)\s*$", line))})
    counts = {"psd->jpeg": 0, "jpeg-heic->jpg": 0, "left alone": 0, "errors": 0}
    marker_done = set()
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        fmt = true_format(dav, p)
        try:
            if fmt == "PSD" and ext in (".jpg", ".jpeg"):
                backup = f"{BACKUP_ROOT}/{os.path.splitext(p)[0]}.psd"
                print(f"psd->jpeg  {p}\n           backup {backup}")
                if a.apply:
                    bdir = os.path.dirname(backup)
                    dav.mkcol_p(bdir)
                    if BACKUP_ROOT not in marker_done:
                        if not dav.exists(f"{BACKUP_ROOT}/.nomedia"):
                            dav.req("PUT", f"{BACKUP_ROOT}/.nomedia", data=b"")
                        marker_done.add(BACKUP_ROOT)
                    if not dav.exists(backup):
                        dav.req("COPY", p, headers={"Destination": dav.url(backup), "Overwrite": "F"})
                    if dav.size(backup) != dav.size(p):
                        raise RuntimeError("backup size differs from the original; not converting")
                    _, data, _ = dav.req("GET", p)
                    im = Image.open(io.BytesIO(data))
                    icc = im.info.get("icc_profile")
                    out = io.BytesIO()
                    im.convert("RGB").save(out, "JPEG", quality=95, **({"icc_profile": icc} if icc else {}))
                    dav.req("PUT", p, data=out.getvalue(), headers={"Content-Type": "image/jpeg"})
                    if true_format(dav, p) != "JPEG":
                        raise RuntimeError("uploaded file does not read back as JPEG")
                counts["psd->jpeg"] += 1
            elif fmt == "JPEG" and ext in (".heic", ".heif"):
                target = os.path.splitext(p)[0] + ".jpg"
                print(f"rename     {p}\n        -> {target}")
                if a.apply:
                    if dav.exists(target):
                        raise RuntimeError(f"target exists: {target}")
                    dav.req("MOVE", p, headers={"Destination": dav.url(target), "Overwrite": "F"})
                counts["jpeg-heic->jpg"] += 1
            else:
                print(f"left alone ({fmt or 'unreadable'}{ext}): {p}")
                counts["left alone"] += 1
        except Exception as e:  # one bad file must not stop the rest; nothing is overwritten before the checks
            print(f"ERROR {p}: {e}")
            counts["errors"] += 1
    print(("applied: " if a.apply else "dry run: ") + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return 1 if counts["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
