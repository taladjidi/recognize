#!/usr/bin/env python3
"""Grid thumbnails: PHP preview vs the nginx-served thumb (python/thumbs.py), on the same files.

Takes N file ids from the largest face clusters (python API), asks /v1/thumbs for their
signed URLs, then GETs both with the UI's concurrency and prints p50/p95/max and bytes.
Read-only, stdlib only; auth as tools/bench_faces.py (app password in ~/.config/nc-bench/<user>).

  tools/bench_thumbs.py --n 120 --runs 3
"""
import argparse, json, os, sys

from bench_faces import Client, fetch_all, path_of, py_get, py_token


def file_ids(cl, tok, n):
    s, d, _ = py_get(cl, tok, "/clusters", {"min": 1})
    if s != 200:
        sys.exit(f"clusters failed: HTTP {s}")
    ids = []
    for c in sorted(json.loads(d), key=lambda c: -c["count"]):
        s, d, _ = py_get(cl, tok, f"/clusters/{c['id']}", {"limit": n})
        ids += [f["fileid"] for f in json.loads(d)["files"] if f["fileid"] not in ids] if s == 200 else []
        if len(ids) >= n:
            return ids[:n]
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=120, help="file ids to fetch (at most 500: one /v1/thumbs call)")
    ap.add_argument("--size", type=int, default=1024, choices=(256, 1024))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--base", default="https://aladjidi.xyz")
    ap.add_argument("--user", default="tangui")
    ap.add_argument("--url-prefix", default="/index.php", help="prefix of /apps/photos URLs (generateUrl)")
    ap.add_argument("--token-file", default=None)
    a = ap.parse_args()
    tf = os.path.expanduser(a.token_file or f"~/.config/nc-bench/{a.user}")
    try:
        pw = open(tf).read().strip()
    except OSError:
        sys.exit(f"no app password at {tf} (see tools/README.md, Benchmarking)")
    cl = Client(a.base, a.user, pw)
    tok, _ = py_token(cl)
    ids = file_ids(cl, tok, min(a.n, 500))
    s, d, t = py_get(cl, tok, "/thumbs", {"ids": ",".join(map(str, ids))})
    if s != 200:
        sys.exit(f"thumbs failed: HTTP {s}")
    urls = json.loads(d)
    have = [i for i in ids if str(i) in urls]  # compare on the same files only
    print(f"{len(ids)} files, {len(have)} with thumbs; /v1/thumbs {t * 1000:.0f} ms, {len(d)} bytes")
    if not have:
        sys.exit("no thumbs yet: is the service's thumbnail pass running?")
    php = [f"{a.url_prefix}/apps/photos/api/v1/preview/{i}?x={a.size}&y={a.size}" for i in have]
    nginx = [path_of(urls[str(i)][str(a.size)]) for i in have]
    print(f"{'run':>4} {'source':6} {'p50 ms':>8} {'p95 ms':>8} {'max ms':>8} {'KB':>8} {'wall s':>7}  status")
    for run in range(1, a.runs + 1):
        for name, batch in (("php", php), ("nginx", nginx)):
            r = fetch_all(cl, batch)
            print(f"{run:>4} {name:6} {r['p50_ms']:8.1f} {r['p95_ms']:8.1f} {r['max_ms']:8.1f} "
                  f"{r['bytes'] / 1024:8.0f} {r['wall_s']:7.2f}  {r['status']}")


if __name__ == "__main__":
    main()
