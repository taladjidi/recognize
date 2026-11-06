#!/usr/bin/env python3
"""Drop preview index rows whose file is gone, so Nextcloud regenerates them. No PHP.

WHY: oc_previews can list a preview whose file is missing. Nextcloud then serves the row
instead of generating, the read fails while the response is being sent, and the client
gets an HTML error page as HTTP 200 / image/jpeg (an empty Photos tile), or a 404 when it
was the "max" preview. Found 2026-11-06 on files previewed on 19 and 27 Sept: before
34.0.5, the loser of two concurrent generations of one preview deleted the shared file
(LocalPreviewStorage::deleteUnreferencedPreview is a no-op now).

Reads oc_previews with one query per 5000 rows, builds each path the way
LocalPreviewStorage::constructPath does, and stats it (threads: the HDD queues seeks).
--apply deletes the missing rows, or every preview (row + file) of a file whose max
preview is gone (the other sizes derive from it). A row whose file is back at apply time
is kept. Dry run unless --apply.

  sudo -u apache /usr/share/nextcloud/apps/recognize/python/.venv/bin/python \\
      tools/repair_previews.py [--apply] [--fileid N ...]
"""
import argparse
import hashlib
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

PAGE = 5000
EXT = {"image/png": "png", "image/gif": "gif", "image/jpeg": "jpg", "image/webp": "webp"}  # Preview::getExtension


def preview_name(r):
    """Preview::getName()."""
    name = (f"{r['version_id']}-" if r["version_id"] > -1 else "") + f"{r['width']}-{r['height']}"
    name += "-crop" if r["cropped"] else ""
    name += "-max" if r["max"] else ""
    return f"{name}.{EXT.get(r['mimetype'], 'png')}"


def preview_path(root, file_id, name):
    """LocalPreviewStorage::constructPath(): <root>/a/b/c/d/e/f/g/<fileid>/<name>, a..g = md5(fileid)[:7]."""
    return os.path.join(root, *hashlib.md5(str(file_id).encode()).hexdigest()[:7], str(file_id), name)


def _self_check():
    # Values from the 2026-11-06 diagnostics (fileid 827446, its log line and directory listing).
    row = {"version_id": -1, "width": 1024, "height": 768, "cropped": 0, "max": 0, "mimetype": "image/jpeg"}
    assert preview_name(row) == "1024-768.jpg"
    assert preview_name({**row, "height": 1024, "cropped": 1}) == "1024-1024-crop.jpg"
    assert preview_name({**row, "width": 2048, "height": 1536, "max": 1}) == "2048-1536-max.jpg"
    assert preview_name({**row, "version_id": 7}) == "7-1024-768.jpg"
    assert preview_path("/r", 827446, "x.jpg") == "/r/9/1/9/0/a/9/9/827446/x.jpg"


def find_preview_root(datadir, appdata):
    if appdata:
        return os.path.join(datadir, appdata, "preview")
    found = [d for d in sorted(os.listdir(datadir))
             if d.startswith("appdata_") and os.path.isdir(os.path.join(datadir, d, "preview"))]
    if len(found) != 1:
        sys.exit(f"found {found or 'no'} appdata_*/preview in {datadir}: pass --appdata appdata_<instanceid>")
    return os.path.join(datadir, found[0], "preview")


def main() -> int:
    _self_check()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--fileid", type=int, nargs="*", default=[], help="only these files (verbose)")
    ap.add_argument("--appdata", help="appdata_<instanceid> (auto-detected when there is one)")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--config", default="/usr/share/nextcloud/apps/recognize/python/nc_config.json")
    a = ap.parse_args()

    import config as nc_config
    from db import DB
    cfg = nc_config.load(a.config)
    db = DB(cfg)
    root = find_preview_root(cfg["datadirectory"], a.appdata)
    t_prev, t_mime = db._table("previews"), db._table("mimetypes")
    ph = db._ph()
    only = f" AND p.file_id IN ({db._ph(len(a.fileid))})" if a.fileid else ""
    sql = (f"SELECT p.id, p.file_id, p.width, p.height, p.cropped, p.max, p.version_id, p.mtime, m.mimetype "
           f"FROM {t_prev} p JOIN {t_mime} m ON m.id = p.mimetype_id "
           f"WHERE p.id > {ph}{only} ORDER BY p.id LIMIT {PAGE}")
    cols = ["id", "file_id", "width", "height", "cropped", "max", "version_id", "mtime", "mimetype"]

    t0 = time.monotonic()
    rows_total, files, bad = 0, set(), defaultdict(list)
    last = 0
    with ThreadPoolExecutor(a.threads) as ex:
        while True:
            page = [dict(zip(cols, r)) for r in db._execute(sql, (last, *a.fileid)).fetchall()]
            db.rollback()  # end the read snapshot: rows written meanwhile stay visible
            if not page:
                break
            last = page[-1]["id"]
            paths = [preview_path(root, r["file_id"], preview_name(r)) for r in page]
            for r, ok in zip(page, ex.map(os.path.isfile, paths)):
                files.add(r["file_id"])
                if not ok:
                    bad[r["file_id"]].append(r)
            rows_total += len(page)
            print(f"\r  checked {rows_total} rows, {sum(map(len, bad.values()))} missing "
                  f"({time.monotonic() - t0:.0f} s)", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    by_month = defaultdict(int)
    for rs in bad.values():
        for r in rs:
            by_month[time.strftime("%Y-%m", time.localtime(r["mtime"]))] += 1
    if a.fileid:
        for fid in a.fileid:
            names = [preview_name(r) for r in bad.get(fid, [])]
            print(f"file {fid}: {len(names)} missing{' (' + ', '.join(names) + ')' if names else ''}")

    deleted = 0
    max_gone = [fid for fid, rs in bad.items() if any(r["max"] for r in rs)]
    if a.apply and bad:
        for fid, rs in bad.items():
            if fid in max_gone:  # every size derives from max: reset the file
                cur = db._execute(f"SELECT p.id, p.file_id, p.width, p.height, p.cropped, p.max, p.version_id, "
                                  f"p.mtime, m.mimetype FROM {t_prev} p JOIN {t_mime} m ON m.id = p.mimetype_id "
                                  f"WHERE p.file_id = {ph}", (fid,))
                targets = [dict(zip(cols, r)) for r in cur.fetchall()]
                for r in targets:
                    try:
                        os.unlink(preview_path(root, fid, preview_name(r)))
                    except FileNotFoundError:
                        pass
            else:
                targets = [r for r in rs if not os.path.isfile(preview_path(root, fid, preview_name(r)))]
            ids = [r["id"] for r in targets]
            if ids:
                db._execute(f"DELETE FROM {t_prev} WHERE id IN ({db._ph(len(ids))})", tuple(ids))
                db._commit()
                deleted += len(ids)

    print("missing rows by month the row was written:")
    for m in sorted(by_month):
        print(f"  {m}  {by_month[m]}")
    n_bad = sum(map(len, bad.values()))
    print(f"{'applied' if a.apply else 'dry run'}: {rows_total} rows in {len(files)} files checked in "
          f"{time.monotonic() - t0:.0f} s; {n_bad} missing in {len(bad)} files ({len(max_gone)} with the max "
          f"preview gone); " + (f"deleted {deleted} rows" if a.apply else
          f"would delete the {n_bad} missing rows, and every row of the {len(max_gone)} max-gone files"))
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
