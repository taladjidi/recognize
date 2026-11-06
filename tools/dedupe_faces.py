#!/usr/bin/env python3
"""Remove near-duplicate face detections (same file, same user, boxes with IoU >= 0.9). No PHP.

WHY: before 2026-11-05 a rename or move re-queued the file and the daemon stored a second
set of faces next to the first (InternalController::insertFaces keeps existing ones now).
Same rule as upstream's RemoveDuplicateFaceDetections repair step (53ba8280, fff92960).

Per (file, user), detections are taken best first and each one overlapping an already kept
one (IoU >= 0.9) is a duplicate of it. Best first is:
  1. in a titled cluster (the user named the person),
  2. in any cluster (cluster_id > 0),
  3. carrying a user removal (threshold > 0, so a removed face cannot rejoin; not in upstream),
  then the lowest id.
A duplicate set holding detections of two different titled clusters (the user named the same
face as two people) is left alone and listed for manual review. A detection that is the last
face of a titled cluster is never deleted (listed instead; the ranking above makes this a
safety net only).

--apply deletes in one transaction per page of files: every set is re-read and re-planned
under a row lock first (the daemon and the user may change rows meanwhile). Any DB error, or
a DELETE removing other than the rows just read, rolls the page back and aborts. Pages
already committed stay deleted; a re-run continues (it is idempotent). Refused while Nextcloud
is in maintenance mode. Dry run (the same report, nothing written) unless --apply.

Not done here, on purpose:
- crops: the daemon's crop thread deletes <uid>/<id>.jpg once its detection is gone
  (service.py _cleanup_crops_one_user, one user dir per idle tick);
- clusters left empty stay: the clusterer never targets a cluster without samples, the API
  and the Photos listing only show clusters with faces, and file deletions leave empty
  clusters the same way. A titled cluster is never emptied (see above).

Output: one line per duplicate (deleted, or would be), the manual-review and protected sets,
a per-user table and a summary line. Redirect stdout to keep the audit log.

  sudo -u apache /usr/share/nextcloud/apps/recognize/python/.venv/bin/python \\
      tools/dedupe_faces.py [--apply] [--config .../nc_config.json] > /tmp/dedupe_faces.log
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

MIN_IOU = 0.9
PAGE = 2000  # files per scan page, and per transaction


def iou(a, b):
    ix = max(0.0, min(a["x"] + a["width"], b["x"] + b["width"]) - max(a["x"], b["x"]))
    iy = max(0.0, min(a["y"] + a["height"], b["y"] + b["height"]) - max(a["y"], b["y"]))
    inter = ix * iy
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / union if union > 0 else 0.0


def _clustered(r):
    return (r["cluster_id"] or 0) > 0


def _titled(r):
    return _clustered(r) and bool(r["title"])


def _rank(r):
    return (not _titled(r), not _clustered(r), not (r["threshold"] or 0) > 0, r["id"])


def plan(rows):
    """One user's detections in one file -> (deletes [(row, kept row)], conflicts [[rows]])."""
    sets = []
    for r in sorted(rows, key=_rank):
        s = next((s for s in sets if iou(s[0], r) >= MIN_IOU), None)
        if s is None:
            sets.append([r])
        else:
            s.append(r)
    deletes, conflicts = [], []
    for s in sets:
        if len({r["cluster_id"] for r in s if _titled(r)}) > 1:
            conflicts.append(s)
        else:
            deletes += [(r, s[0]) for r in s[1:]]
    return deletes, conflicts


def _select(db, where, params, lock=False):
    t, tc = db._table("recognize_face_detections"), db._table("recognize_face_clusters")
    # pgsql cannot lock the nullable side of an outer join; MariaDB has no FOR UPDATE OF.
    suffix = {"mysql": " FOR UPDATE", "pgsql": " FOR UPDATE OF d"}.get(db.dialect, "") if lock else ""
    cur = db._execute(
        f"SELECT d.id, d.file_id, d.user_id, d.x, d.y, d.width, d.height, d.cluster_id, d.threshold, "
        f"COALESCE(c.title, '') AS title FROM {t} d LEFT JOIN {tc} c ON c.id = d.cluster_id "
        f"WHERE {where} ORDER BY d.file_id, d.id{suffix}", params)
    return db._fetchall_dicts(cur)


def _groups(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["file_id"], r["user_id"])].append(r)
    return g


def _pages(db):
    """Detections of PAGE files at a time, by file_id (a file never spans two pages)."""
    t, ph = db._table("recognize_face_detections"), db._ph()
    last = -1
    while True:
        ids = [r[0] for r in db._execute(
            f"SELECT DISTINCT file_id FROM {t} WHERE file_id > {ph} ORDER BY file_id LIMIT {PAGE}",
            (last,)).fetchall()]
        if not ids:
            return
        rows = _select(db, f"d.file_id > {ph} AND d.file_id <= {ph}", (last, ids[-1]))
        db.rollback()  # end the read snapshot
        last = ids[-1]
        yield rows


def _settle(db, key, apply, out):
    """Re-read one (file, user) set (under lock with apply), re-plan, guard, delete (with apply). Caller commits."""
    t, ph = db._table("recognize_face_detections"), db._ph()
    deletes, conflicts = plan(_select(db, f"d.file_id = {ph} AND d.user_id = {ph}", key, lock=apply))
    out["conflicts"] += conflicts
    ids = {r["id"] for r, _ in deletes}
    keep = []
    for r, k in deletes:
        # shortcut: COUNT is not locked (pgsql refuses FOR UPDATE with aggregates); the kept row is.
        if _titled(r) and not db._execute(
                f"SELECT COUNT(*) FROM {t} WHERE cluster_id = {ph} AND id NOT IN ({db._ph(len(ids))})",
                (r["cluster_id"], *ids)).fetchone()[0]:
            out["protected"].append(r)
        else:
            keep.append((r, k))
    if apply and keep:
        n = db._execute(f"DELETE FROM {t} WHERE id IN ({db._ph(len(keep))})",
                        tuple(r["id"] for r, _ in keep)).rowcount
        if n != len(keep):
            raise RuntimeError(f"file {key[0]} user {key[1]}: deleted {n} rows, expected {len(keep)}")
    out["deleted"] += keep


def run(db, apply=False):
    """Returns {deleted: [(row, kept row)], conflicts: [[rows]], protected: [rows]}."""
    out = {"deleted": [], "conflicts": [], "protected": []}
    for rows in _pages(db):
        keys = [k for k, g in _groups(rows).items() if any(plan(g))]
        if not keys:
            continue
        try:
            if apply and db.dialect == "sqlite3":
                db._execute("BEGIN IMMEDIATE")  # the stdlib driver would not lock before the DELETE
            for k in keys:
                _settle(db, k, apply, out)
            if apply:
                db.commit()
            else:
                db.rollback()
        except BaseException:
            db.rollback()
            raise
    return out


def _paths(db, file_ids):
    try:
        found = {}
        ids = sorted(file_ids)
        for i in range(0, len(ids), 500):
            found.update(db.resolve_file_paths(ids[i:i + 500]))
        db.rollback()
        return {f: v["path"] for f, v in found.items()}
    except Exception as e:  # report only: a path is a convenience
        db.rollback()
        print(f"cannot resolve paths: {e}", file=sys.stderr)
        return {}


def report(db, out, apply):
    verb = "deleted" if apply else "would delete"
    for r, k in out["deleted"]:
        print(f"{verb} id={r['id']} file_id={r['file_id']} user={r['user_id']} cluster_id={r['cluster_id']} "
              f"title={r['title']!r} (kept id={k['id']} cluster_id={k['cluster_id']} title={k['title']!r})")
    paths = _paths(db, {s[0]["file_id"] for s in out["conflicts"]} | {r["file_id"] for r in out["protected"]})
    for s in out["conflicts"]:
        fid = s[0]["file_id"]
        print(f"manual review (one face, two named clusters), untouched: file_id={fid} "
              f"path={paths.get(fid, '?')} user={s[0]['user_id']} "
              + ", ".join(f"id={r['id']} cluster_id={r['cluster_id']} title={r['title']!r}" for r in s))
    for r in out["protected"]:
        print(f"kept, last face of titled cluster {r['cluster_id']} {r['title']!r}: id={r['id']} "
              f"file_id={r['file_id']} path={paths.get(r['file_id'], '?')} user={r['user_id']}")
    per = defaultdict(lambda: [set(), 0, 0])
    for r, _ in out["deleted"]:
        u = per[r["user_id"]]
        u[0].add(r["file_id"])
        u[1] += 1
        u[2] += _titled(r)
    print("user  files_with_duplicates  duplicate_rows  in_named_clusters")
    for user in sorted(per):
        f, n, named = per[user]
        print(f"{user}  {len(f)}  {n}  {named}")
    print(f"{'applied' if apply else 'dry run'}: {verb} {len(out['deleted'])} duplicate detections in "
          f"{len({(r['file_id'], r['user_id']) for r, _ in out['deleted']})} files of {len(per)} users; "
          f"{len(out['conflicts'])} sets for manual review; {len(out['protected'])} protected last faces")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--config", default="/usr/share/nextcloud/apps/recognize/python/nc_config.json")
    a = ap.parse_args()

    import config as nc_config
    from db import DB
    db = DB(nc_config.load(a.config))
    try:
        if a.apply and db.check_maintenance_mode(max_age=0):
            print("Nextcloud is in maintenance mode: refusing --apply", file=sys.stderr)
            return 1
        report(db, run(db, a.apply), a.apply)
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
