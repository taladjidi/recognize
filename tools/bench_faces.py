#!/usr/bin/env python3
"""Headless benchmark of the Nextcloud Faces UI (read-only: PROPFIND and GET).

Replays the requests the deployed Photos fork (~/Downloads/photos, branch
port/face-fixes-stable34) makes, so UI speed can be compared before/after a
change with no browser. Stdlib only.

  tools/bench_faces.py --scenario all --runs 3
  tools/bench_faces.py --scenario overview --cover-size 256
  tools/bench_faces.py --scenario overview --compare bench-results/<old>.json

Limits: http.client is HTTP/1.1 (one connection per worker, 10 workers); the
browser multiplexes over one HTTP/2 connection. Compare runs of this tool with
each other, not with browser timings.
"""
import argparse, base64, html, http.client, json, os, ssl, statistics as st
import sys, threading, time, urllib.parse, xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

# getDavProperties() default set: node_modules/@nextcloud/files dav chunk, defaultDavProperties.
DEFAULT_PROPS = ["d:getcontentlength", "d:getcontenttype", "d:getetag", "d:getlastmodified",
                 "d:creationdate", "d:displayname", "d:quota-available-bytes", "d:resourcetype",
                 "nc:has-preview", "nc:is-encrypted", "nc:mount-type", "oc:comments-unread",
                 "oc:favorite", "oc:fileid", "oc:owner-display-name", "oc:owner-id",
                 "oc:permissions", "oc:size"]
# FetchFacesMixin.ts:21-40
LIST_PROPS = ["<nc:face-preview-image/>", "<nc:nbItems/>"]
CONTENT_PROPS = ["<nc:face-detections/>", "<nc:realpath/>"]
COUNT_PROPS = ["<nc:nbItems/>"]
COVER_SIZE = 1024   # FaceCover.vue:~103 coverUrl x=1024&y=1024
THUMB_SIZE = 1024   # FileComponent.vue:146 srcLarge (desktop; mobile is 256, 64 is the blur-up placeholder)
CONCURRENCY = 10    # PreviewQueue.ts:16 MAX_CONCURRENT
GRID_FIRST = 60


def propfind_body(extra):  # DavRequest.ts getPropFind
    props = "\n".join("<%s />" % p for p in DEFAULT_PROPS)
    return ('<?xml version="1.0"?>\n<d:propfind xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns" '
            'xmlns:nc="http://nextcloud.org/ns" xmlns:ocs="http://open-collaboration-services.org/ns">'
            '<d:prop>%s\n%s</d:prop></d:propfind>' % (props, "\n".join(extra)))


class Client:
    def __init__(self, base, user, pw, apikey=None):
        u = urllib.parse.urlparse(base)
        self.host, self.port, self.tls = u.hostname, u.port, u.scheme == "https"
        self.auth = "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()
        self.apikey = apikey  # DavClient.ts:12 sends X-Recognize-Api-Key on every request
        self.local = threading.local()

    def _conn(self):
        c = getattr(self.local, "c", None)
        if c is None:
            c = (http.client.HTTPSConnection(self.host, self.port, timeout=120, context=ssl.create_default_context())
                 if self.tls else http.client.HTTPConnection(self.host, self.port, timeout=120))
            self.local.c = c
        return c

    def req(self, method, path, body=None, headers=None):
        """Returns (status, bytes, seconds). Only PROPFIND and GET are allowed."""
        assert method in ("PROPFIND", "GET")
        h = {"Authorization": self.auth, "OCS-APIRequest": "true", **(headers or {})}
        if self.apikey:
            h["X-Recognize-Api-Key"] = self.apikey
        t = time.perf_counter()
        for attempt in (0, 1):
            try:
                c = self._conn()
                c.request(method, path, body=body, headers=h)
                r = c.getresponse()
                data = r.read()
                return r.status, data, time.perf_counter() - t
            except (http.client.HTTPException, OSError):
                self.local.c = None
                if attempt:
                    return 0, b"", time.perf_counter() - t

    def propfind(self, path, extra, depth):
        return self.req("PROPFIND", "/remote.php/dav" + path, propfind_body(extra).encode(),
                        {"Depth": str(depth), "Content-Type": "application/xml; charset=utf-8"})


def parse_multistatus(data):
    ns = {"d": "DAV:", "nc": "http://nextcloud.org/ns", "oc": "http://owncloud.org/ns"}
    out = []
    for r in ET.fromstring(data).findall("d:response", ns):
        def g(q):
            e = r.find(".//" + q, ns)
            return e.text if e is not None else None
        out.append({"href": g("d:href"), "nbItems": g("nc:nbItems"), "cover": g("nc:face-preview-image"),
                    "fileid": g("oc:fileid"), "etag": g("d:getetag")})
    return out


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else 0


def fetch_all(cl, urls):
    """GETs with the UI's concurrency; returns a summary dict."""
    t0 = time.perf_counter()
    with ThreadPoolExecutor(CONCURRENCY) as ex:
        res = list(ex.map(lambda u: cl.req("GET", u), urls))
    wall = time.perf_counter() - t0
    lat = [r[2] * 1000 for r in res]
    codes = {}
    for r in res:
        codes[r[0]] = codes.get(r[0], 0) + 1
    return {"n": len(urls), "p50_ms": pct(lat, 50), "p95_ms": pct(lat, 95), "max_ms": max(lat, default=0),
            "bytes": sum(len(r[1]) for r in res), "wall_s": wall, "status": codes}


def list_clusters(cl, uid):
    # FetchFacesMixin.ts:~85 list and :~226 unassigned count, in parallel as the UI does
    t0 = time.perf_counter()
    with ThreadPoolExecutor(2) as ex:
        f1 = ex.submit(cl.propfind, f"/recognize/{uid}/faces/", LIST_PROPS, 1)
        f2 = ex.submit(cl.propfind, f"/recognize/{uid}/unassigned-faces", COUNT_PROPS, 0)
        (s, d, tl), (s2, _, tu) = f1.result(), f2.result()
    wall = time.perf_counter() - t0
    if s >= 300:
        sys.exit(f"list PROPFIND failed: HTTP {s}")
    rows = parse_multistatus(d)[1:]  # first response is the collection itself
    clusters = [{"name": urllib.parse.unquote(r["href"].rstrip("/").split("/")[-1]),
                 "n": int(r["nbItems"] or 0), "cover": r["cover"]} for r in rows]
    clusters.sort(key=lambda c: -c["n"])  # FacesView.vue:82 orderedFaces
    return clusters, {"list_ms": tl * 1000, "list_bytes": len(d), "unassigned_ms": tu * 1000,
                      "unassigned_status": s2, "parallel_wall_ms": wall * 1000, "clusters": len(clusters)}


def cover_urls(clusters, prefix, size):
    out = []
    for c in clusters:
        try:
            fid = json.loads(html.unescape(c["cover"] or "null"))["fileid"]
        except (TypeError, KeyError, ValueError):
            continue
        out.append(f"{prefix}/apps/photos/api/v1/preview/{fid}?x={size}&y={size}")  # FaceCover.vue:~103
    return out


def overview(cl, a):
    clusters, m = list_clusters(cl, a.user)
    urls = cover_urls(clusters, a.url_prefix, a.cover_size)
    if a.max_covers:
        urls = urls[:a.max_covers]
    m["covers"] = fetch_all(cl, urls)
    return m


def cluster(cl, a):
    clusters, _ = list_clusters(cl, a.user)
    out = {}
    for label, c in (("largest", clusters[0]), ("median", clusters[len(clusters) // 2])):
        path = f"/recognize/{a.user}/faces/{urllib.parse.quote(c['name'])}"  # FetchFacesMixin.ts:~120
        s, d, t = cl.propfind(path, CONTENT_PROPS, 1)
        r = {"nbItems": c["n"], "propfind_ms": t * 1000, "bytes": len(d), "http": s}
        if s < 300:
            files = [f for f in parse_multistatus(d)[1:] if f["fileid"]][:GRID_FIRST]
            # FileComponent.vue:283 getItemURL; decodedEtag strips the quotes
            urls = [f"{a.url_prefix}/apps/photos/api/v1/preview/{f['fileid']}?etag="
                    f"{(f['etag'] or '').replace(chr(34), '').replace('&quot;', '')}"
                    f"&x={a.thumb_size}&y={a.thumb_size}" for f in files]
            r["thumbs"] = fetch_all(cl, urls)
        out[label] = r
    return out


def flat(d, p=""):
    for k, v in d.items():
        if isinstance(v, dict) and k != "status":
            yield from flat(v, p + k + ".")
        elif isinstance(v, (int, float)):
            yield p + k, v


def show(name, runs):
    keys = [k for k, _ in flat(runs[0])]
    rest = runs[1:] or runs  # run 1 is reported apart, median over the rest
    med = {k: st.median(dict(flat(r)).get(k, 0) for r in rest) for k in keys}
    print(f"\n== {name} ==  (run1 = cold-ish; median = runs 2..N)")
    print(f"{'metric':32}" + "".join(f"{'run' + str(i + 1):>12}" for i in range(len(runs))) + f"{'median':>12}")
    for k in keys:
        print(f"{k:32}" + "".join(f"{dict(flat(r)).get(k, 0):12.1f}" for r in runs) + f"{med[k]:12.1f}")
    for i, r in enumerate(runs):
        for lbl, sub in [("covers", r.get("covers")), ("largest", r.get("largest", {}).get("thumbs")),
                         ("median", r.get("median", {}).get("thumbs"))]:
            if sub:
                print(f"  run{i + 1} {lbl} HTTP: " + " ".join(f"{k}={v}" for k, v in sub["status"].items()))
    return med


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=["overview", "cluster", "all"], default="all")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--cover-size", type=int, default=COVER_SIZE)
    ap.add_argument("--thumb-size", type=int, default=THUMB_SIZE)
    ap.add_argument("--max-covers", type=int, default=0)
    ap.add_argument("--compare", help="older result JSON to diff against")
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
    kf = os.path.expanduser(f"~/.config/nc-bench/recognize-api-key")
    try:
        key = open(kf).read().strip()
    except OSError:
        key = None  # without it recognize DAV answers 403 under Basic auth; see tools/README.md
    cl = Client(a.base, a.user, pw, key)
    old = json.load(open(a.compare)) if a.compare else {}
    for name, fn in (("overview", overview), ("cluster", cluster)):
        if a.scenario not in (name, "all"):
            continue
        runs = [fn(cl, a) for _ in range(a.runs)]
        med = show(name, runs)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        os.makedirs("bench-results", exist_ok=True)
        path = f"bench-results/{ts}-{name}.json"
        json.dump({"scenario": name, "args": vars(a), "runs": runs, "median": med}, open(path, "w"), indent=1)
        print("wrote", path)
        if old.get("scenario") == name:
            print(f"\n-- delta vs {a.compare} (median) --")
            for k, v in med.items():
                o = old["median"].get(k)
                if o:
                    print(f"{k:32}{o:12.1f} -> {v:12.1f}  {100 * (v - o) / o:+7.1f}%")


if __name__ == "__main__":
    main()
