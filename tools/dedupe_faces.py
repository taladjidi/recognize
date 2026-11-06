#!/usr/bin/env python3
"""Remove near-duplicate face detections (same file, same user, boxes with IoU >= 0.9). No PHP.

WHY: before 2026-11-05 a rename or move re-queued the file and the daemon stored a second
set of faces next to the first (InternalController::insertFaces keeps existing ones now).
Same rule as upstream's RemoveDuplicateFaceDetections repair step (53ba8280, fff92960).

Per (file, user), detections are taken best first and each one overlapping an already kept
one (IoU >= 0.9; identical zero-area boxes count, a NULL box column never matches) is a
duplicate of it. Best first is:
  1. in a titled cluster (the user named the person),
  2. in any cluster (cluster_id > 0; ids start at 1, 0 is never written),
  3. carrying a user removal (threshold > 0, so a removed face cannot rejoin; not in upstream),
  4. rejected (cluster_id -1, listed in Unassigned) over never clustered (NULL), as upstream,
  then the lowest id.
Left alone and listed for manual review: a set holding two different titled clusters (the user
named the same face as two people), or a user removal (threshold > 0, no cluster) next to a
clustered duplicate (deleting either undoes a user decision). A detection that is the last
face of a titled cluster is never deleted (listed instead; a safety net only).

--apply (needs --backup) settles one (file, user) set per transaction: the set is re-read and
re-planned under a row lock (the daemon and the user may change rows meanwhile), the full rows
to delete (box, face_vector, threshold, cluster_id, cluster title) are appended to the backup
as JSON lines and fsynced, then deleted, their lines printed (flushed), and committed. A MySQL
deadlock or lock wait timeout retries the set once. Any other DB error, or a DELETE removing
other than the rows just read, rolls that set back and aborts; sets already committed are in
the backup and the log, and a re-run continues (it is idempotent). A line printed just before
a failed commit (the traceback follows it) was not deleted. Refused while Nextcloud is in
maintenance mode. Dry run (the same lines, nothing written) unless --apply.

--restore BACKUP re-inserts the backed-up rows (same ids) that are absent, in one transaction;
ids present again are skipped and listed. A row whose cluster was deleted since comes back -1.

Not done here, on purpose:
- crops: the daemon's crop thread deletes <uid>/<id>.jpg once its detection is gone
  (service.py _cleanup_crops_one_user, one user dir per idle tick); after a --restore, run
  tools/backfill_crops.py (or restart the daemon) to make the restored crops again;
- clusters left empty stay: the clusterer never targets a cluster without samples, the API
  and the Photos listing only show clusters with faces, and file deletions leave empty
  clusters the same way. A titled cluster is never emptied (see above).

Output: one line per duplicate (deleted, or would be), the manual-review and protected sets,
a per-user table and a summary line. Redirect stdout to keep the audit log.

Recommended procedure, with the daemon stopped so nothing else writes faces meanwhile (the tool
also works with the daemon running: each set's rows stay locked for one short transaction).
The backup dir is apache's, group aladjidi, so the user reads it without sudo:

  T=/home/aladjidi/Downloads/recognize/tools/dedupe_faces.py
  PY=/usr/share/nextcloud/apps/recognize/python/.venv/bin/python
  D=/home/aladjidi/Downloads/dedupe-faces
  sudo install -d -o apache -g aladjidi -m 0750 $D
  sudo -u apache $PY $T > ~/Downloads/dedupe_faces.dry.log                           # dry run
  sudo systemctl stop recognize
  sudo -u apache $PY $T --apply --backup $D/backup.jsonl > ~/Downloads/dedupe_faces.apply.log
  sudo systemctl start recognize
  less $D/backup.jsonl                                                   # one JSON row per line
  sudo -u apache $PY $T --restore $D/backup.jsonl > ~/Downloads/dedupe_faces.restore.log  # undo
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

MIN_IOU = 0.9
PAGE = 2000  # files per scan page (one transaction per (file, user) set)
RETRY_ERRNOS = (1213, 1205)  # MySQL deadlock, lock wait timeout (mysql.connector .errno)
_BOX = ("x", "y", "width", "height")


def iou(a, b):
    if any(r[k] is None for r in (a, b) for k in _BOX):  # production columns allow NULL
        return 0.0
    ix = max(0.0, min(a["x"] + a["width"], b["x"] + b["width"]) - max(a["x"], b["x"]))
    iy = max(0.0, min(a["y"] + a["height"], b["y"] + b["height"]) - max(a["y"], b["y"]))
    inter = ix * iy
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    if union > 0:
        return inter / union
    return 1.0 if all(a[k] == b[k] for k in _BOX) else 0.0  # identical zero-area boxes, as upstream


def _clustered(r):
    return (r["cluster_id"] or 0) > 0


def _titled(r):
    return _clustered(r) and bool(r["title"])


def _removed(r):
    return (r["threshold"] or 0) > 0


def _rank(r):
    return (not _titled(r), not _clustered(r), not _removed(r), r["cluster_id"] != -1, r["id"])


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
        if (len({r["cluster_id"] for r in s if _titled(r)}) > 1
                or (any(_removed(r) and not _clustered(r) for r in s) and any(map(_clustered, s)))):
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


def _line(verb, r, k):
    return (f"{verb} id={r['id']} file_id={r['file_id']} user={r['user_id']} cluster_id={r['cluster_id']} "
            f"title={r['title']!r} (kept id={k['id']} cluster_id={k['cluster_id']} title={k['title']!r})")


def _settle(db, key, backup):
    """Re-read one (file, user) set (under lock when applying, i.e. with a backup), re-plan, guard;
    with a backup: back up, delete, print. Caller commits. Returns the set's {deleted, conflicts, protected}."""
    t, tc, ph = db._table("recognize_face_detections"), db._table("recognize_face_clusters"), db._ph()
    deletes, conflicts = plan(_select(db, f"d.file_id = {ph} AND d.user_id = {ph}", key, lock=bool(backup)))
    res = {"deleted": [], "conflicts": conflicts, "protected": []}
    ids = {r["id"] for r, _ in deletes}
    for r, k in deletes:
        # shortcut: COUNT is not locked (pgsql refuses FOR UPDATE with aggregates); the kept row is.
        if _titled(r) and not db._execute(
                f"SELECT COUNT(*) FROM {t} WHERE cluster_id = {ph} AND id NOT IN ({db._ph(len(ids))})",
                (r["cluster_id"], *ids)).fetchone()[0]:
            res["protected"].append(r)
        else:
            res["deleted"].append((r, k))
    keep = res["deleted"]
    if backup and keep:
        del_ids = tuple(r["id"] for r, _ in keep)
        kept_of = {r["id"]: k["id"] for r, k in keep}
        full = db._fetchall_dicts(db._execute(
            f"SELECT d.*, c.title AS cluster_title FROM {t} d LEFT JOIN {tc} c ON c.id = d.cluster_id "
            f"WHERE d.id IN ({db._ph(len(del_ids))}) ORDER BY d.id", del_ids))
        if len(full) != len(del_ids):
            raise RuntimeError(f"file {key[0]} user {key[1]}: read {len(full)} rows to back up, "
                               f"expected {len(del_ids)}")
        for row in full:
            title = row.pop("cluster_title")
            backup.write(json.dumps({"row": row, "cluster_title": title, "kept_id": kept_of[row["id"]]}) + "\n")
        backup.flush()
        os.fsync(backup.fileno())
        n = db._execute(f"DELETE FROM {t} WHERE id IN ({db._ph(len(del_ids))})", del_ids).rowcount
        if n != len(del_ids):
            raise RuntimeError(f"file {key[0]} user {key[1]}: deleted {n} rows, expected {len(del_ids)}")
    for r, k in keep:
        print(_line("deleted" if backup else "would delete", r, k), flush=True)
    return res


def run(db, backup=None):
    """Dry run, or apply with backup (an open file). Returns {deleted: [(row, kept row)],
    conflicts: [[rows]], protected: [rows]} of the settled sets."""
    out = {"deleted": [], "conflicts": [], "protected": []}
    for rows in _pages(db):
        for key, g in _groups(rows).items():
            if not any(plan(g)):
                continue
            for attempt in (1, 2):
                try:
                    if backup and db.dialect == "sqlite3":
                        db._execute("BEGIN IMMEDIATE")  # the stdlib driver would not lock before the DELETE
                    res = _settle(db, key, backup)
                    if backup:
                        db.commit()
                    else:
                        db.rollback()
                    break
                except BaseException as e:
                    db.rollback()
                    if attempt == 2 or getattr(e, "errno", None) not in RETRY_ERRNOS:
                        raise
                    print(f"file {key[0]} user {key[1]}: {e}, retrying once", file=sys.stderr, flush=True)
            for k in out:
                out[k] += res[k]
    return out


def restore(db, path):
    """Re-insert backed-up rows whose id is absent, in one transaction.
    Returns (restored ids, skipped ids, ids made unassigned because their cluster is gone)."""
    t, tc, ph = db._table("recognize_face_detections"), db._table("recognize_face_clusters"), db._ph()
    restored, skipped, unassigned = [], [], []
    # The last line of an id wins: a retried set backs up the row it then deleted, and that
    # may differ from an earlier attempt's copy. A line cut off by a crash or a full disk is
    # skipped: a DELETE only ever follows a complete, fsynced line.
    rows = {}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)["row"]
            except (ValueError, KeyError, TypeError):
                print(f"warning: {path}:{n}: unreadable line skipped", file=sys.stderr)
                continue
            rows[row["id"]] = row
    try:
        for row in rows.values():
            if not all(re.fullmatch(r"[a-z_]+", c) for c in row):
                raise ValueError(f"bad column name in {path}: {sorted(row)}")
            if db._execute(f"SELECT 1 FROM {t} WHERE id = {ph}", (row["id"],)).fetchone():
                skipped.append(row["id"])
                continue
            if (row.get("cluster_id") or 0) > 0 and not db._execute(
                    f"SELECT 1 FROM {tc} WHERE id = {ph}", (row["cluster_id"],)).fetchone():
                row["cluster_id"] = -1  # as a cluster deletion does to its detections
                unassigned.append(row["id"])
            db._execute(f"INSERT INTO {t} ({', '.join(row)}) VALUES ({db._ph(len(row))})",
                        tuple(row.values()))
            restored.append(row["id"])
        db.commit()
    except BaseException:
        db.rollback()
        raise
    return restored, skipped, unassigned


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
    """Everything but the deletion lines, which _settle printed as it went."""
    verb = "deleted" if apply else "would delete"
    paths = _paths(db, {s[0]["file_id"] for s in out["conflicts"]} | {r["file_id"] for r in out["protected"]})
    for s in out["conflicts"]:
        fid = s[0]["file_id"]
        print(f"manual review (two named clusters, or a user removal next to a clustered duplicate), "
              f"untouched: file_id={fid} path={paths.get(fid, '?')} user={s[0]['user_id']} "
              + ", ".join(f"id={r['id']} cluster_id={r['cluster_id']} title={r['title']!r} "
                          f"threshold={r['threshold']}" for r in s))
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


def _open_backup(path):
    f = open(path, "a", encoding="utf-8")
    if f.tell() and open(path, "rb").read()[-1:] != b"\n":
        f.write("\n")  # end a line cut off by a crash, so the next one stays parseable
        f.flush()
        os.fsync(f.fileno())
    d = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY)
    try:
        os.fsync(d)  # a new file's directory entry survives a power cut
    finally:
        os.close(d)
    return f


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--restore", metavar="BACKUP", help="re-insert the rows of a --backup file")
    ap.add_argument("--backup", metavar="PATH", help="required with --apply: appends every deleted row (JSON lines)")
    ap.add_argument("--config", default="/usr/share/nextcloud/apps/recognize/python/nc_config.json")
    a = ap.parse_args()
    if a.apply and not a.backup:
        ap.error("--apply needs --backup PATH")

    import config as nc_config
    from db import DB
    db = DB(nc_config.load(a.config))
    try:
        if (a.apply or a.restore) and db.check_maintenance_mode(max_age=0):
            print("Nextcloud is in maintenance mode: refusing to write", file=sys.stderr)
            return 1
        if a.restore:
            restored, skipped, unassigned = restore(db, a.restore)
            for i in restored:
                print(f"restored id={i}" + (" as unassigned (its cluster was deleted)" if i in unassigned else ""))
            for i in skipped:
                print(f"skipped id={i}: present again")
            print(f"restore: {len(restored)} rows re-inserted, {len(skipped)} skipped as present, "
                  f"{len(unassigned)} unassigned because their cluster was deleted")
        elif a.apply:
            with _open_backup(a.backup) as backup:
                report(db, run(db, backup), True)
        else:
            report(db, run(db), False)
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
