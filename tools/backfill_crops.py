"""One-shot face crop backfill, run beside the daemon.

The daemon only backfills a small slice per classifier pass, so after the
read-path deploy ~19k existing detections would take most of a day to get
crops. Crop making is CPU work (reduced JPEG decode) while classification is
GPU-bound, so this runs the daemon's own backfill (Service._backfill_crops:
cluster covers first, then everything) in large slices until a pass is done.
--force regenerates crops that already exist (covers first), e.g. after a
crop geometry or cover rule change. Writes are atomic (temp file + rename), so racing the daemon is harmless.

  sudo -u apache /usr/share/nextcloud/apps/recognize/python/.venv/bin/python \
      tools/backfill_crops.py [--force] /usr/share/nextcloud/apps/recognize/python/nc_config.json
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

from service import Service  # noqa: E402


def main() -> int:
    args = sys.argv[1:]
    force = "--force" in args
    args = [a for a in args if a != "--force"]
    if len(args) != 1:
        print(__doc__)
        return 2
    svc = Service(args[0], gpu=False)
    svc._crop_force = force
    if not os.path.isdir(svc.crop_root):
        print(f"crop root {svc.crop_root} missing: run deploy.sh first")
        return 1
    t0 = time.monotonic()
    last = -1
    while True:
        done = svc._backfill_crops(time.monotonic() + 3600, 500)
        created = svc._crop_created
        if created != last or done:
            phase = "all detections" if svc._crop_covers_done else "cluster covers"
            print(f"{time.monotonic() - t0:7.0f}s  {phase:15s} created={created} cursor={svc._crop_cursor}", flush=True)
            last = created
        if done:
            print("backfill pass complete")
            return 0


if __name__ == "__main__":
    sys.exit(main())
