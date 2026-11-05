#!/usr/bin/env python3
"""Generate, then verify, the previews the Faces cluster grid shows.

The grid loads one Nextcloud preview per photo (FileComponent.vue: 1024 px on desktop,
256 on mobile), generated on first view. This requests exactly those URLs, for every
photo of every face cluster (via the same /recognize-api the page uses), so they exist
before anyone opens a cluster. Read-only HTTP (GET); the server stores the previews.

A request slower than --cold-ms was most likely generated just now. Run it twice: the
second run should report ~0 cold and is the check that everything is generated.

  tools/warm_previews.py [--sizes 1024,256] [--concurrency 4] [--cold-ms 150] [--min 1]

Auth as tools/bench_faces.py: app password in ~/.config/nc-bench/<user>.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_faces as b  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default=str(b.THUMB_SIZE), help="comma list, e.g. 1024,256")
    ap.add_argument("--concurrency", type=int, default=4, help="parallel requests (server CPU + HDD)")
    ap.add_argument("--cold-ms", type=float, default=150.0)
    ap.add_argument("--min", type=int, default=1, help="min cluster size (1: every cluster, search can open small ones)")
    ap.add_argument("--max-photos", type=int, default=0, help="only the first N photos (smoke test)")
    ap.add_argument("--base", default="https://aladjidi.xyz")
    ap.add_argument("--user", default="tangui")
    ap.add_argument("--url-prefix", default="/index.php")
    a = ap.parse_args()
    pw = open(os.path.expanduser(f"~/.config/nc-bench/{a.user}")).read().strip()
    cl = b.Client(a.base, a.user, pw)
    sizes = [int(s) for s in a.sizes.split(",") if s]

    tok, _ = b.py_token(cl)
    s, d, _ = b.py_get(cl, tok, "/clusters", {"min": a.min})
    if s != 200:
        sys.exit(f"clusters failed: HTTP {s}")
    clusters = b.json.loads(d)
    files = {}
    t0 = time.monotonic()
    for i, c in enumerate(clusters):
        if time.monotonic() - t0 > 200:  # tokens live 300 s
            tok, _ = b.py_token(cl)
            t0 = time.monotonic()
        fs, r = b.py_cluster_files(cl, tok, c["id"])
        for f in fs:
            files[f["fileid"]] = f
    if a.max_photos:
        files = dict(list(files.items())[:a.max_photos])
    etag = lambda f: (f["etag"] or "").replace(chr(34), "").replace("&quot;", "")  # as FileComponent / bench
    urls = [(f["fileid"], size, f"{a.url_prefix}/apps/photos/api/v1/preview/{f['fileid']}?etag={etag(f)}&x={size}&y={size}")
            for size in sizes for f in files.values()]
    print(f"{len(clusters)} clusters, {len(files)} distinct photos, {len(urls)} previews to check "
          f"(sizes {sizes}, concurrency {a.concurrency})", flush=True)

    done = cold = 0
    failed = []
    lat = []
    t_start = time.monotonic()
    with ThreadPoolExecutor(a.concurrency) as ex:
        futs = {ex.submit(cl.req, "GET", u): (fid, size) for fid, size, u in urls}
        for fut in as_completed(futs):
            status, body, sec = fut.result()
            done += 1
            lat.append(sec * 1000)
            if status != 200:
                failed.append((futs[fut], status))
            elif sec * 1000 > a.cold_ms:
                cold += 1
            if done % 500 == 0 or done == len(urls):
                el = time.monotonic() - t_start
                rate = done / el if el else 0
                eta = (len(urls) - done) / rate if rate else 0
                print(f"  {done}/{len(urls)}  cold={cold} failed={len(failed)}  {rate:.1f}/s  "
                      f"p50={b.pct(lat, 50):.0f} ms p95={b.pct(lat, 95):.0f} ms  eta {eta / 60:.0f} min", flush=True)
    print(f"done: {done} previews, {cold} generated now (>{a.cold_ms:.0f} ms), {len(failed)} failed")
    for (fid, size), status in failed[:20]:
        print(f"  failed: fileid {fid} size {size} HTTP {status} ({files[fid]['path']})")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
