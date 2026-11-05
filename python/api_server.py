#!/usr/bin/env python3
"""Read API for the Photos Faces views (notes/faces-read-path.md, piece B).

Stdlib HTTP server on 127.0.0.1, reached through nginx at /recognize-api/.
Auth is a short-lived bearer token minted by lib/Service/FacesToken.php;
every query is scoped to the token's uid and to files in that user's home
storage under files/ (not trash). Covers and crops are nginx secure_link URLs.

    python api_server.py /path/to/nc_config.json
"""

import argparse
import base64
import hashlib
import hmac
import json
import logging
import os
import queue
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlsplit

import config as nc_config
from crops import DEFAULT_CROP_ROOT
from db import COVER_CANDIDATES, DB, cover_order

log = logging.getLogger("recognize.api")

DEFAULT_PORT = 8291
TOKEN_MAX_TTL = 300  # lib/Service/FacesToken.php TTL
CLOCK_SKEW = 60
CROP_PREFIX = "/recognize-faces"
CROP_TTL = 6 * 3600
MAX_Q = 100
MAX_LIMIT = 1000
POOL_IDLE = 8
DB_SLOTS = 8  # concurrent DB connections; more requests wait DB_WAIT s, then 503
DB_WAIT = 2.0
MAX_HANDLERS = 32  # concurrent connections; more wait HANDLER_WAIT s in the listen backlog, then are dropped
HANDLER_WAIT = 5.0
USER_CACHE_TTL = 60
_B64URL = re.compile(r"[A-Za-z0-9_-]+")
_CLUSTER_PATH = re.compile(r"/v1/clusters/(\d{1,18})")


# ── Tokens and signed links ─────────────────────────────────────────────

def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    if not _B64URL.fullmatch(s):
        raise ValueError("not base64url")
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def verify_token(header: str | None, secret: str, now: float | None = None) -> str | None:
    """The uid of a valid `Bearer <b64url(payload)>.<b64url(hmac)>` header, else None."""
    if not secret or not header or not header.startswith("Bearer "):
        return None
    try:
        p64, s64 = header[len("Bearer "):].strip().split(".")
        payload = _b64url_decode(p64)
        sig = _b64url_decode(s64)
        if not hmac.compare_digest(sig, hmac.new(secret.encode(), payload, hashlib.sha256).digest()):
            return None
        claims = json.loads(payload)
        v, uid, exp = claims["v"], claims["uid"], claims["exp"]
    except (ValueError, KeyError, TypeError):
        return None
    now = time.time() if now is None else now
    if v != 1 or not isinstance(uid, str) or not uid or type(exp) is not int:
        return None
    if not now < exp <= now + TOKEN_MAX_TTL + CLOCK_SKEW:
        return None
    return uid


def crop_url(uid: str, detection_id: int, secret: str, now: float | None = None) -> str:
    """nginx secure_link URL: md5 = b64url(md5("$secure_link_expires$uri <secret>")), $uri decoded."""
    now = time.time() if now is None else now
    # Next hourly boundary + 6 h: every link minted within the same hour is identical, so browsers cache it.
    expires = (int(now) // 3600 + 1) * 3600 + CROP_TTL
    name = f"{int(detection_id)}.jpg"
    digest = hashlib.md5(f"{expires}{CROP_PREFIX}/{uid}/{name} {secret}".encode()).digest()
    return f"{CROP_PREFIX}/{quote(uid, safe='')}/{name}?md5={b64url(digest)}&expires={expires}"


# ── Queries (one per endpoint) ──────────────────────────────────────────

def _owned(db: DB, extra_join: str = "") -> str:
    """Join + filter: the detection is uid's and its file is in uid's home storage under files/."""
    p = db._ph()
    return (
        f"JOIN {db._table('filecache')} fc ON fc.fileid = d.file_id "
        f"JOIN {db._table('storages')} s ON s.numeric_id = fc.storage {extra_join}"
        f"WHERE d.user_id = {p} AND fc.path LIKE {p} AND s.id IN ({p}, {p}) "
    )


def _owned_params(uid: str) -> tuple:
    # Same storage-id forms as InternalController::ownerFromStorageId.
    return (uid, "files/%", f"home::{uid}", f"object::user:{uid}")


def escape_like(q: str) -> str:
    return q.replace("!", "!!").replace("%", "!%").replace("_", "!_")




def _crop_or_none(uid, did, secret, crop_root):
    """Signed URL, or None when the crop file is missing or empty (nginx would 404).

    crop_root None skips the check.
    """
    if crop_root is not None:
        try:
            if os.stat(os.path.join(crop_root, uid, f"{did}.jpg")).st_size <= 0:
                return None
        except OSError:
            return None
    return crop_url(uid, did, secret)


def list_clusters(db: DB, uid: str, crop_secret: str, min_count: int = 1, title: str | None = None,
                  crop_root: str | None = None) -> list[dict]:
    """[{id, title, count, cover}] by count desc; cover is the best-ranked box (db.cover_order) that has a crop, else None."""
    p = db._ph()
    title_sql = f"AND LOWER(c.title) LIKE {p} ESCAPE '!' " if title is not None else ""
    sql = (
        f"SELECT t.cluster_id AS cid, c.title, t.cnt, t.id AS did FROM ("
        f"  SELECT d.id, d.cluster_id,"
        f"    COUNT(*) OVER (PARTITION BY d.cluster_id) AS cnt,"
        f"    ROW_NUMBER() OVER (PARTITION BY d.cluster_id ORDER BY {cover_order('d.')}) AS rn"
        f"  FROM {db._table('recognize_face_detections')} d {_owned(db)} AND d.cluster_id > 0"
        f") t JOIN {db._table('recognize_face_clusters')} c ON c.id = t.cluster_id "
        f"WHERE t.rn <= {COVER_CANDIDATES} AND t.cnt >= {p} AND c.user_id = {p} {title_sql}"
        f"ORDER BY t.cnt DESC, t.cluster_id, t.rn"
    )
    params = _owned_params(uid) + (min_count, uid)
    if title is not None:
        params += (f"%{escape_like(title.lower())}%",)
    rows = db._fetchall_dicts(db._execute(sql, params))
    out = {}
    for r in rows:
        c = out.setdefault(r["cid"], {"id": r["cid"], "title": r["title"], "count": r["cnt"], "cover": None})
        if c["cover"] is None:
            c["cover"] = _crop_or_none(uid, r["did"], crop_secret, crop_root)
    return list(out.values())


def get_cluster(db: DB, uid: str, crop_secret: str, cluster_id: int, offset: int, limit: int,
                crop_root: str | None = None) -> dict | None:
    """{id, title, count, files} for one of uid's clusters, newest first; None if not uid's."""
    p = db._ph()
    det, clusters = db._table("recognize_face_detections"), db._table("recognize_face_clusters")
    mime_join = f"JOIN {db._table('mimetypes')} m ON m.id = fc.mimetype "
    sql = (
        f"SELECT c.title, COUNT(*) OVER () AS cnt, d.id AS did, d.x, d.y, d.width, d.height,"
        f"  fc.fileid, fc.path, fc.name, m.mimetype, fc.etag, fc.mtime, fc.size "
        f"FROM {det} d JOIN {clusters} c ON c.id = d.cluster_id "
        f"{_owned(db, mime_join)} AND c.id = {p} AND c.user_id = {p} "
        f"ORDER BY fc.mtime DESC, d.id DESC LIMIT {int(limit)} OFFSET {int(offset)}"
    )
    rows = db._fetchall_dicts(db._execute(sql, _owned_params(uid) + (cluster_id, uid)))
    if rows:
        title, count = rows[0]["title"], rows[0]["cnt"]
    else:
        # Empty page (offset past the end, or not uid's): tell the two apart.
        cur = db._execute(
            f"SELECT c.title, (SELECT COUNT(*) FROM {det} d {_owned(db)} AND d.cluster_id = c.id) AS cnt "
            f"FROM {clusters} c WHERE c.id = {p} AND c.user_id = {p}",
            _owned_params(uid) + (cluster_id, uid),
        )
        head = db._fetchall_dicts(cur)
        if not head:
            return None
        title, count = head[0]["title"], head[0]["cnt"]
    return {
        "id": cluster_id, "title": title, "count": count,
        "files": [{
            "fileid": r["fileid"], "path": r["path"][len("files/"):], "name": r["name"],
            "mime": r["mimetype"], "etag": r["etag"], "mtime": r["mtime"], "size": r["size"],
            "detection": {
                "id": r["did"], "x": r["x"], "y": r["y"], "width": r["width"], "height": r["height"],
                "crop": _crop_or_none(uid, r["did"], crop_secret, crop_root),
            },
        } for r in rows],
    }


# ── HTTP ────────────────────────────────────────────────────────────────

def _int(qs: dict, name: str, default: int, lo: int, hi: int) -> int:
    raw = qs.get(name, [None])[-1]
    if raw is None:
        return default
    value = int(raw)
    if not lo <= value <= hi:
        raise ValueError(name)
    return value


class Busy(Exception):
    pass


class Api:
    def __init__(self, config: dict):
        self.config = config
        self.token_secret = config.get("api_token_secret", "")
        self.crop_secret = config.get("crop_link_secret", "")
        self.crop_root = config.get("face_crop_root") or DEFAULT_CROP_ROOT
        self._idle = queue.SimpleQueue()
        self._db_slots = threading.BoundedSemaphore(DB_SLOTS)
        self._users = {}  # uid -> (ok, monotonic expiry); only validly signed uids get here

    def _with_db(self, fn):
        if not self._db_slots.acquire(timeout=DB_WAIT):
            raise Busy
        try:
            return self._with_conn(fn)
        finally:
            self._db_slots.release()

    def _with_conn(self, fn):
        try:
            db = self._idle.get_nowait()
        except queue.Empty:
            db = DB(self.config)
        try:
            try:
                return fn(db)
            except Exception:
                # A dropped connection (MySQL restart, wait_timeout) fails the first query: reconnect once.
                log.warning("query failed, reconnecting", exc_info=True)
                db.close()
                db = DB(self.config)
                return fn(db)
        finally:
            try:
                db.commit()  # end the read snapshot so the next request sees new rows
            except Exception:
                db.close()
            else:
                if self._idle.qsize() < POOL_IDLE:
                    self._idle.put(db)
                else:
                    db.close()

    def user_ok(self, uid: str) -> bool:
        """uid exists in oc_users and is not disabled; cached USER_CACHE_TTL s."""
        now = time.monotonic()
        hit = self._users.get(uid)
        if hit and hit[1] > now:
            return hit[0]

        def check(db):
            p = db._ph()
            if not db._execute(f"SELECT 1 FROM {db._table('users')} WHERE uid = {p}", (uid,)).fetchall():
                return False
            return not db._execute(
                f"SELECT 1 FROM {db._table('preferences')} WHERE userid = {p} AND appid = 'core' "
                f"AND configkey = 'enabled' AND configvalue = 'false'", (uid,)).fetchall()

        ok = self._with_db(check)
        self._users[uid] = (ok, now + USER_CACHE_TTL)
        return ok

    def handle(self, path: str, qs: dict, auth: str | None) -> tuple[int, object]:
        if path == "/v1/health":
            return 200, {"ok": True}  # static: unauthenticated, so it must not cost a DB connection
        uid = verify_token(auth, self.token_secret)
        if uid is None:
            return 401, {}
        try:
            if not self.user_ok(uid):
                return 401, {}
            if path == "/v1/clusters":
                min_count = _int(qs, "min", 1, 1, 10**9)
                return 200, self._with_db(lambda db: list_clusters(db, uid, self.crop_secret, min_count, crop_root=self.crop_root))
            m = _CLUSTER_PATH.fullmatch(path)
            if m:
                offset = _int(qs, "offset", 0, 0, 10**9)
                limit = _int(qs, "limit", MAX_LIMIT, 1, MAX_LIMIT)
                body = self._with_db(
                    lambda db: get_cluster(db, uid, self.crop_secret, int(m.group(1)), offset, limit,
                                          crop_root=self.crop_root))
                return (200, body) if body is not None else (404, {"error": "not found"})
            if path == "/v1/search":
                q = qs.get("q", [""])[-1]
                if not q or len(q) > MAX_Q:
                    raise ValueError("q")
                return 200, self._with_db(lambda db: list_clusters(db, uid, self.crop_secret, 1, q, crop_root=self.crop_root))
        except ValueError:
            return 400, {"error": "bad parameter"}
        except Busy:
            return 503, {"error": "busy"}
        return 404, {"error": "not found"}


def _json_default(o):
    if isinstance(o, (bytes, bytearray)):
        return o.decode()
    return float(o)  # Decimal from MySQL aggregates


class Handler(BaseHTTPRequestHandler):
    server_version = "recognize-api"
    timeout = 30  # socket timeout: an idle or slow client frees its handler slot

    def do_GET(self):
        url = urlsplit(self.path)
        try:
            status, body = self.server.api.handle(
                url.path, parse_qs(url.query), self.headers.get("Authorization"))
        except Exception:
            log.exception("request failed: %s", url.path)
            status, body = 500, {"error": "internal"}
        data = json.dumps(body, default=_json_default, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "private, no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        # Paths only: the query string carries search terms (person names); the token is a header.
        log.info("%s", _QUERY.sub("", fmt % args))


_QUERY = re.compile(r"\?[^\s\"']*")


class Server(ThreadingHTTPServer):
    """At most MAX_HANDLERS handler threads; past that, connections wait in the listen backlog."""
    daemon_threads = True
    request_queue_size = 64

    def __init__(self, *args, **kwargs):
        self._slots = threading.BoundedSemaphore(MAX_HANDLERS)
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        if not self._slots.acquire(timeout=HANDLER_WAIT):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


def make_server(config: dict, host: str = "127.0.0.1", port: int | None = None) -> ThreadingHTTPServer:
    if port is None:
        port = int(config.get("api_port") or DEFAULT_PORT)
    server = Server((host, port), Handler)
    server.api = Api(config)
    return server


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Recognize Faces read API")
    parser.add_argument("config", nargs="?", default=None, help="path to nc_config.json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    config = nc_config.load(args.config)
    if not config.get("api_token_secret") or not config.get("crop_link_secret"):
        log.error("api_token_secret / crop_link_secret missing from nc_config.json: run setup_config.py")
        return 1
    server = make_server(config)
    log.info("listening on %s:%d", *server.server_address[:2])
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
