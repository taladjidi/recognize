"""One-shot face crop backfill, run beside the daemon.

The daemon only backfills a small slice per classifier pass, so after the
read-path deploy ~19k existing detections would take most of a day to get
crops. Crop making is CPU work (reduced JPEG decode) while classification is
GPU-bound, so this makes covers first (each cluster's cover detection), then
everything, one task per source file (open once, crop every pending face) on a
fork pool, files ordered by filesystem path for HDD locality. Workers only
touch files, never the DB. --covers-only stops after the covers. --workers N
(default 2) sets the pool size. A stats line (files/s, crops/s, MB/s, p50/p95
of read, decode+crop, write per file) shows whether the run is disk-bound.
--force regenerates crops that already exist (covers first), e.g. after a
crop geometry or cover rule change. Writes are atomic (temp file + rename), so racing the daemon is harmless.

  sudo -u apache /usr/share/nextcloud/apps/recognize/python/.venv/bin/python \
      tools/backfill_crops.py [--force] [--covers-only] [--workers N] /usr/share/nextcloud/apps/recognize/python/nc_config.json
"""
import os
import sys
import io
import multiprocessing
import signal
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

TASK_TIMEOUT = 600  # s a file may take; a task lost to a killed worker is skipped after this
SLICE = 5000  # detections per DB query; files inside a slice are sorted by path
REPORT_EVERY = 200  # files


def crop_file(task):
    """Worker: (path, rows, crop_root, force) -> (crops written, bytes, t_read, t_decode, t_write)."""
    from crops import make_crop, open_raw, write_crop
    path, rows, crop_root, force = task
    t0 = time.monotonic()
    try:
        with open(path, "rb") as f:
            data = f.read()
        t1 = time.monotonic()
        img, orientation = open_raw(io.BytesIO(data), rows)
        crops = [(r, make_crop(img, r, orientation)) for r in rows]
        t2 = time.monotonic()
        n = sum(bool(write_crop(crop_root, r["user_id"], r["id"], c, overwrite=force)) for r, c in crops)
    except Exception as e:
        print(f"failed {path}: {e}", file=sys.stderr, flush=True)
        return 0, 0, 0.0, 0.0, 0.0
    return n, len(data), t1 - t0, t2 - t1, time.monotonic() - t2


def _init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # parent handles Ctrl-C and terminates the pool


def plan_slices(db, crop_root, force, covers_only):
    """Yield lists of (path, pending rows) per DB slice, each sorted by path; covers phase first."""
    from crops import has_crop
    seen = set()  # force mode: a cover redone in phase 1 is not redone in phase 2
    for covers in (True, False):
        if not covers and covers_only:
            return
        cursor = 0
        while True:
            rows = db.detections_after(cursor, SLICE, covers_only=covers)
            if not rows:
                break
            cursor = rows[-1]["id"]
            by_file = {}
            for r in rows:
                if r["id"] in seen or not (force or not has_crop(crop_root, r["user_id"], r["id"])):
                    continue
                by_file.setdefault(r["file_id"], []).append(r)
                if covers:
                    seen.add(r["id"])
            paths = db.resolve_file_paths(list(by_file)) if by_file else {}
            yield sorted(((paths[fid]["path"], rs) for fid, rs in by_file.items() if fid in paths), key=lambda t: t[0])


def _pct(xs, q):
    return sorted(xs)[min(len(xs) - 1, int(len(xs) * q))] if xs else 0.0


def stats_line(t0, files, crops, nbytes, cols):
    dt = max(time.monotonic() - t0, 1e-9)
    split = "  ".join(f"{name} {_pct(c, .5) * 1000:.0f}/{_pct(c, .95) * 1000:.0f}ms"
                      for name, c in zip(("read", "decode", "write"), cols))
    return (f"{dt:6.0f}s  files={files} crops={crops}  {files / dt:.1f} files/s  {crops / dt:.1f} crops/s  "
            f"{nbytes / 1e6 / dt:.1f} MB/s  p50/p95: {split}")


def run(db, crop_root, force=False, covers_only=False, workers=2):
    t0 = time.monotonic()
    files = crops = nbytes = 0
    cols = ([], [], [])
    pool = multiprocessing.get_context("fork").Pool(workers, _init_worker)
    try:
        for batch in plan_slices(db, crop_root, force, covers_only):
            jobs = [(p, pool.apply_async(crop_file, ((p, rs, crop_root, force),))) for p, rs in batch]
            for p, job in jobs:
                try:
                    n, size, *times = job.get(TASK_TIMEOUT)
                except multiprocessing.TimeoutError:  # worker killed (OOM): the pool never delivers this result
                    print(f"lost task, skipped: {p}", flush=True)
                    continue
                files += 1
                crops += n
                nbytes += size
                for c, t in zip(cols, times):
                    c.append(t)
                if files % REPORT_EVERY == 0:
                    print(stats_line(t0, files, crops, nbytes, cols), flush=True)
        pool.close()
    except KeyboardInterrupt:
        pool.terminate()  # finished crops are atomic renames, so they stay valid
        print("interrupted", flush=True)
        return 130
    finally:
        pool.terminate()
        pool.join()
    print(stats_line(t0, files, crops, nbytes, cols), flush=True)
    print("backfill pass complete")
    return 0


def main() -> int:
    args = sys.argv[1:]
    force = "--force" in args
    covers_only = "--covers-only" in args
    workers = 2
    if "--workers" in args:
        i = args.index("--workers")
        try:
            workers = int(args[i + 1])
            del args[i:i + 2]
        except (IndexError, ValueError):
            print(__doc__)
            return 2
    args = [a for a in args if a not in ("--force", "--covers-only")]
    if len(args) != 1 or workers < 1:
        print(__doc__)
        return 2
    from service import Service
    svc = Service(args[0], gpu=False)
    if not os.path.isdir(svc.crop_root):
        print(f"crop root {svc.crop_root} missing: run deploy.sh first")
        return 1
    return run(svc.pipeline.db, svc.crop_root, force, covers_only, workers)


if __name__ == "__main__":
    sys.exit(main())
