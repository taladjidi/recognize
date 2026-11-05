"""Re-detect faces in EXIF-rotated photos, keeping cluster membership and names.

Before the EXIF fix the daemon detected faces on raw pixels, so in photos with an
EXIF orientation other than 1 faces were found sideways: misplaced boxes, poor
embeddings. This re-runs detection on the upright image and, per file:

- matches fresh faces to the existing detections by IoU in raw coordinates
  (greedy best IoU >= 0.3; a clustered row also needs a mutual unique best match)
  and UPDATEs each match in place (box + vector only: id, cluster_id, threshold
  and user_id stay, so names and manual edits survive);
- a removed face's threshold (PHP: old vector's distance to the cluster) shrinks by
  how far the vector moved, so the new vector cannot rejoin;
- the update applies only while cluster_id and threshold still hold the values read
  (a user edit during detection rolls the file back, counted as raced);
- inserts unmatched fresh faces unclustered, for the daemon to cluster, except one
  overlapping a kept named or removed row (likely the same face): the file is listed
  as "needs review" instead;
- deletes an unmatched old detection only if it has no cluster and no threshold
  (a likely sideways false positive); every other unmatched one is kept;
- rewrites the crop of every changed detection (the old crop is deleted before the
  commit, so a crash leaves no crop, which the backfill rebuilds, never a stale one).

Idempotent without any state: a detection whose box and vector already match the
fresh upright result (IoU >= 0.9, cosine >= 0.99) is left alone, so a re-run, or a
file detected by the fixed daemon, changes nothing. --limit N stops after N files
that changed, so repeated --limit runs work through the backlog (each re-examines
the already-fixed files, which costs a detection each but no writes).
--dry-run reports the same summary and writes nothing.

--requeue-faceless (instead, no model): rotated photos where the old code found no
face have no detection, so the pass above never sees them. This queues the owners'
rotated image files without detections (and not already pending) into
recognize_pending for the daemon, in path order, header reads only. --limit N stops
after N requeued; an appconfig cursor (recognize/redetect.facelessCursor) resumes,
and is set to 'done' after a full pass so faceless photos are queued only once.

  sudo -u apache /usr/share/nextcloud/apps/recognize/python/.venv/bin/python \\
      tools/redetect_rotated.py /usr/share/nextcloud/apps/recognize/python/nc_config.json \\
      [--dry-run] [--limit N] [--requeue-faceless]
"""
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

ACTION_CREATED = 0  # PendingService::ACTION_CREATED, what StorageCrawlJob queues existing files with
CURSOR_KEY, DONE = "redetect.facelessCursor", "done"
MIN_IOU = 0.3
SAME_IOU, SAME_COS = 0.9, 0.99
REPORT_EVERY = 200  # files


def iou(a, b):
    ix = max(0.0, min(a["x"] + a["width"], b["x"] + b["width"]) - max(a["x"], b["x"]))
    iy = max(0.0, min(a["y"] + a["height"], b["y"] + b["height"]) - max(a["y"], b["y"]))
    inter = ix * iy
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / union if union > 0 else 0.0


def _unique_best(scores):
    """Index of the single highest score, None on a tie or no scores."""
    if not scores:
        return None
    top = max(scores)
    return scores.index(top) if scores.count(top) == 1 else None


def match(old, new):
    """[(old index, new index)], greedy by best IoU, each used once, IoU >= MIN_IOU.

    A clustered (named) old row also needs a mutual unique best match: in a group photo
    greedy alone can hand it a neighbour's face. Refused, it stays as is and the face is
    inserted unclustered.
    """
    m = [[iou(o, n) for n in new] for o in old]
    pairs = sorted(((m[i][j], i, j) for i in range(len(old)) for j in range(len(new))), reverse=True)
    used_o, used_n, out = set(), set(), []
    for v, i, j in pairs:
        if v < MIN_IOU:
            break
        if (old[i].get("cluster_id") or 0) > 0 and not (
                _unique_best(m[i]) == j and _unique_best([r[j] for r in m]) == i):
            continue
        if i not in used_o and j not in used_n:
            used_o.add(i)
            used_n.add(j)
            out.append((i, j))
    return out


def _same(row, face):
    a, b = np.asarray(row["face_vector"], float), np.asarray(face["vector"], float)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return iou(row, face) >= SAME_IOU and den > 0 and float(a @ b) / den >= SAME_COS


def _unassigned(row):
    return (row["cluster_id"] is None or row["cluster_id"] <= 0) and not (row["threshold"] or 0) > 0


def candidates(db):
    """(file_id, path) of images with detections and an EXIF rotation, in path order (HDD)."""
    from crops import _ORIENT
    ids = db.face_detection_file_ids()
    infos = {}
    for i in range(0, len(ids), 500):
        infos.update(db.resolve_file_paths(ids[i:i + 500]))
    for fid, info in sorted(infos.items(), key=lambda kv: kv[1]["path"]):
        if not (info.get("mimetype") or "").startswith("image/"):
            continue
        try:
            with Image.open(info["path"]) as im:  # header only, no decode
                orientation = im.getexif().get(0x0112, 1)
        except Exception:
            continue
        if orientation in _ORIENT:
            yield fid, info["path"]


def run(db, detect, crop_root, dry_run=False, limit=None):
    """detect(path) -> (faces with raw boxes, crops). Returns the summary counts."""
    import crops as crops_mod
    s = dict(files=0, changed=0, matched=0, inserted=0, deleted=0, unchanged=0, kept=0, failed=0, raced=0,
             needs_review=[])
    for fid, path in candidates(db):
        if limit is not None and s["changed"] >= limit:
            break
        db.commit()  # end the read snapshot: fresh rows, and no read view held for hours (dry run too)
        old = db.get_face_detections_for_file(fid)
        if not old:
            continue
        s["files"] += 1
        if s["files"] % REPORT_EVERY == 0:
            print(_line(s), flush=True)
        try:
            faces, crops = detect(path)
        except Exception as e:
            print(f"failed {path}: {e}", file=sys.stderr, flush=True)
            s["failed"] += 1
            continue
        pairs = match(old, faces)
        updates = [(old[i], j) for i, j in pairs if not _same(old[i], faces[j])]
        unmatched = [old[i] for i in sorted(set(range(len(old))) - {i for i, _ in pairs})]
        deletes = [r for r in unmatched if _unassigned(r)]
        kept = [r for r in unmatched if not _unassigned(r)]
        fresh = sorted(set(range(len(faces))) - {j for _, j in pairs})
        # A fresh face touching a kept named/removed row is most likely that face: inserting it
        # would duplicate it in the person's cluster, or let a removed face rejoin. A human decides.
        inserts = [j for j in fresh if not any(iou(faces[j], r) > 0 for r in kept)]
        deleted = len(deletes)
        if not dry_run:
            for row, _ in updates:  # before the commit: a crop never shows another box than its row
                try:
                    os.unlink(crops_mod.crop_path(crop_root, row["user_id"], row["id"]) or "")
                except OSError:
                    pass
            if not all([db.update_face_detection_box(row, faces[j], _threshold(row, faces[j])) == 1
                        for row, j in updates]):
                db.rollback()  # a user edit since the read: leave the file for the next run
                s["raced"] += 1
                continue
            db.commit()
            written = [(row["user_id"], row["id"], crops[j]) for row, j in updates]
            user_id = old[0]["user_id"]
            for j in inserts:
                new_id = db.insert_face_detection(fid, user_id, faces[j])  # commits; None on pgsql
                if new_id:
                    written.append((user_id, new_id, crops[j]))
            # the deleted rows' crops go with the daemon's idle cleanup
            deleted = sum(db.delete_detection_if_unassigned(r["id"]) for r in deletes)
            db.commit()
            for uid, did, data in written:
                if data:
                    crops_mod.write_crop(crop_root, uid, did, data, overwrite=True)
        s["matched"] += len(updates)
        s["unchanged"] += len(pairs) - len(updates)
        s["inserted"] += len(inserts)
        s["deleted"] += deleted
        s["kept"] += len(kept)
        if len(inserts) < len(fresh):
            s["needs_review"].append(path)
        if updates or inserts or deleted:
            s["changed"] += 1
    return s


def _threshold(row, face):
    """A removed face's threshold is PHP's distance of the OLD vector to its cluster; shrink it
    by how far the vector moved so the new one stays out (triangle inequality)."""
    t = row["threshold"] or 0
    if t <= 0:
        return None
    moved = float(np.linalg.norm(np.asarray(row["face_vector"], float) - np.asarray(face["vector"], float)))
    return max(t - moved, 1e-6)


def requeue_faceless(db, dry_run=False, limit=None):
    """Queue owners' rotated image files that have no detection for the (fixed) daemon.

    Resumes from the appconfig cursor (the last requeued path); DONE after a full pass,
    so a photo that is faceless even upright is requeued once, never again.
    """
    from crops import _ORIENT
    t0 = time.monotonic()
    s = dict(scanned=0, rotated=0, requeued=0, remaining=0)
    cursor = db.get_app_value(CURSOR_KEY)
    if cursor == DONE:
        print(f"faceless requeue already complete (delete appconfig recognize/{CURSOR_KEY} to redo)")
        return s
    files = sorted((r["path"], r["fileid"], r["storage"]) for r in db.faceless_image_files()
                   if r["path"] and (cursor is None or r["path"] > cursor))
    for n, (path, fid, storage) in enumerate(files):
        if limit is not None and s["requeued"] >= limit:
            s["remaining"] = len(files) - n
            break
        s["scanned"] += 1
        try:
            with Image.open(path) as im:  # header only, no decode
                orientation = im.getexif().get(0x0112, 1)
        except Exception:
            continue
        if orientation not in _ORIENT:
            continue
        s["rotated"] += 1
        if dry_run:
            s["requeued"] += 1
        else:
            # never overwrites a row PHP queued since the scan (e.g. a delete)
            s["requeued"] += db.insert_pending_if_absent(fid, storage, ACTION_CREATED)
            db.set_app_value(CURSOR_KEY, path)
    else:
        if not dry_run:
            db.set_app_value(CURSOR_KEY, DONE)
    dt = max(time.monotonic() - t0, 1e-9)
    rate = s["scanned"] / dt
    eta = f", ~{s['remaining'] / rate:.0f} s left to scan at this rate" if s["remaining"] and rate else ""
    print(("dry run: " if dry_run else "") + _line(s) + f"  {dt:.1f} s, {rate:.0f} files/s{eta}", flush=True)
    return s


def _line(s):
    return " ".join(f"{k}={len(v) if isinstance(v, list) else v}" for k, v in s.items())


def main() -> int:
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    faceless = "--requeue-faceless" in args
    args = [a for a in args if a not in ("--dry-run", "--requeue-faceless")]
    limit = None
    if "--limit" in args:
        i = args.index("--limit")
        try:
            limit = int(args[i + 1])
            del args[i:i + 2]
        except (IndexError, ValueError):
            print(__doc__)
            return 2
    if len(args) != 1 or (limit is not None and limit < 1):
        print(__doc__)
        return 2
    from service import Service
    from classifiers.faces import FaceClassifier
    svc = Service(args[0], gpu=False)
    if faceless:
        requeue_faceless(svc.pipeline.db, dry_run, limit)
        return 0
    if not dry_run and not os.path.isdir(svc.crop_root):
        print(f"crop root {svc.crop_root} missing: run deploy.sh first")
        return 1
    clf = FaceClassifier(svc.models_dir, gpu=True)
    clf.warm_up()
    s = run(svc.pipeline.db, clf.classify_with_crops, svc.crop_root, dry_run, limit)
    for p in s["needs_review"]:
        print(f"needs review (a new face overlaps a named or removed one, not inserted): {p}")
    print(("dry run: " if dry_run else "") + _line(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
