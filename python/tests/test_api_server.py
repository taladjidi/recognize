"""Tests for the Faces read API (api_server.py): tokens, signed crop links, ownership, limits."""

import base64
import hashlib
import hmac
import http.client
import json
import logging
import os
import re
import shutil
import socket
import stat
import subprocess
import threading
import time
import urllib.error
import urllib.request
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pytest

import api_server
from api_server import crop_url, get_cluster, list_clusters, make_server, verify_token
from db import DB

SECRET = "token-secret"
CROP = "crop-secret"
NOW = 1_700_003_000
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def mint(uid="alice", exp=NOW + 300, secret=SECRET, payload=None) -> str:
    raw = payload if payload is not None else json.dumps({"v": 1, "uid": uid, "exp": exp}).encode()
    return f"Bearer {b64(raw)}.{b64(hmac.new(secret.encode(), raw, hashlib.sha256).digest())}"


# ── Tokens ─────────────────────────────────────────────────────────────

class TestToken:
    def test_good(self):
        assert verify_token(mint(), SECRET, now=NOW) == "alice"

    def test_expired(self):
        assert verify_token(mint(exp=NOW), SECRET, now=NOW) is None
        assert verify_token(mint(exp=NOW - 1), SECRET, now=NOW) is None

    def test_too_far_in_future(self):
        assert verify_token(mint(exp=NOW + 360), SECRET, now=NOW) == "alice"  # 300 s + 60 s skew
        assert verify_token(mint(exp=NOW + 361), SECRET, now=NOW) is None

    def test_wrong_secret(self):
        assert verify_token(mint(secret="other"), SECRET, now=NOW) is None
        assert verify_token(mint(), "", now=NOW) is None

    def test_tampered_payload(self):
        good = mint()
        _, sig = good.split(".")
        forged = b64(json.dumps({"v": 1, "uid": "bob", "exp": NOW + 300}).encode())
        assert verify_token(f"Bearer {forged}.{sig}", SECRET, now=NOW) is None

    def test_tampered_signature(self):
        good = mint()
        flipped = good[:-2] + ("A" if good[-2] != "A" else "B") + good[-1]
        assert verify_token(flipped, SECRET, now=NOW) is None

    @pytest.mark.parametrize("header", [
        None, "", "alice", "Basic abc", "Bearer ", "Bearer a.b.c", "Bearer !!.!!",
        "Bearer " + b64(b"x") + "." + b64(b"y"),
    ])
    def test_garbage(self, header):
        assert verify_token(header, SECRET, now=NOW) is None

    @pytest.mark.parametrize("claims", [
        {"v": 2, "uid": "alice", "exp": NOW + 10},
        {"v": 1, "uid": "", "exp": NOW + 10},
        {"v": 1, "uid": 5, "exp": NOW + 10},
        {"v": 1, "uid": "alice", "exp": str(NOW + 10)},
        {"v": 1, "uid": "alice", "exp": float(NOW + 10)},
        {"v": 1, "uid": "alice"},
        ["alice"],
    ])
    def test_bad_claims_even_when_signed(self, claims):
        assert verify_token(mint(payload=json.dumps(claims).encode()), SECRET, now=NOW) is None

    @pytest.mark.skipif(shutil.which("php") is None, reason="php not installed")
    def test_php_minted_token_verifies(self):
        """lib/Service/FacesToken.php and verify_token agree on the format."""
        def php(secret, uid, exp):
            out = subprocess.run(["php", os.path.join(REPO, "tools", "check_token.php"), secret, uid, str(exp)],
                                 capture_output=True, text=True, check=True)
            return "Bearer " + out.stdout.strip()

        token = php("s3cr3t", "élise/o'k", NOW + 300)
        assert verify_token(token, "s3cr3t", now=NOW) == "élise/o'k"
        assert verify_token(token, "s3cr3t", now=NOW + 300) is None
        assert verify_token(token, "wrong", now=NOW) is None
        payload, sig = token[len("Bearer "):].split(".")
        other = php("s3cr3t", "mallory", NOW + 300).split(".")[0]
        assert verify_token(f"{other}.{sig}", "s3cr3t", now=NOW) is None


# ── Signed crop links ──────────────────────────────────────────────────

class TestCropUrl:
    # printf '%s' "1700028000/recognize-faces/u/1.jpg secret" | openssl md5 -binary | openssl base64 | tr +/ -_ | tr -d =
    KNOWN = "j6lCq5MEau8IVzQNOjmMjA"

    def test_known_vector(self):
        url = crop_url("u", 1, "secret", now=NOW)
        assert url == f"/recognize-faces/u/1.jpg?md5={self.KNOWN}&expires=1700028000"

    def test_expires_on_hour_boundary_plus_6h(self):
        for now in (NOW - 199, NOW, NOW + 3399):  # the same clock hour
            assert parse_qs(urlsplit(crop_url("u", 1, "s", now=now)).query)["expires"] == ["1700028000"]
        assert parse_qs(urlsplit(crop_url("u", 1, "s", now=NOW + 3400)).query)["expires"] == ["1700031600"]

    @pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl not installed")
    def test_matches_openssl_on_decoded_uri(self):
        """nginx signs the decoded $uri: a uid with a space is quoted in the URL, not in the hash."""
        url = crop_url("a b", 42, "secret", now=NOW)
        parts = urlsplit(url)
        assert parts.path == "/recognize-faces/a%20b/42.jpg"
        out = subprocess.run(["openssl", "md5", "-binary"], input=b"1700028000/recognize-faces/a b/42.jpg secret",
                             capture_output=True, check=True).stdout
        want = base64.b64encode(out).decode().replace("+", "-").replace("/", "_").rstrip("=")
        assert parse_qs(parts.query)["md5"] == [want]


# ── Ownership, search, limits (sqlite; MySQL too when RECOGNIZE_TEST_MYSQL is set) ──
# RECOGNIZE_TEST_MYSQL='{"dbhost": "localhost:/path/mysql.sock", "dbuser": "root", "dbpassword": "",
#   "dbname": "nc"}' runs the same tests on a throwaway MySQL database (its oc_ tables are dropped).

@pytest.fixture(params=["sqlite3", "mysql"])
def config(request, tmp_path):
    base = {"dbtableprefix": "oc_", "datadirectory": str(tmp_path),
            "api_token_secret": SECRET, "crop_link_secret": CROP}
    if request.param == "mysql":
        env = os.environ.get("RECOGNIZE_TEST_MYSQL")
        if not env:
            pytest.skip("RECOGNIZE_TEST_MYSQL not set")
        return base | {"dbtype": "mysql"} | json.loads(env)
    return base | {"dbtype": "sqlite3", "dbhost": "", "dbuser": "", "dbpassword": "",
                   "dbname": str(tmp_path / "nc.db")}


@pytest.fixture
def db(config):
    d = DB(config)
    text = "TEXT" if d.dialect == "sqlite3" else "VARCHAR(4000) COLLATE utf8mb4_bin"
    for table in ("storages", "mimetypes", "filecache", "recognize_face_detections", "recognize_face_clusters",
                  "users", "preferences"):
        d._execute(f"DROP TABLE IF EXISTS oc_{table}")
    d.create_face_detections_table()
    d.create_face_clusters_table()
    for sql in (
        f"CREATE TABLE oc_storages (numeric_id INTEGER PRIMARY KEY, id {text})",
        f"CREATE TABLE oc_mimetypes (id INTEGER PRIMARY KEY, mimetype {text})",
        f"CREATE TABLE oc_filecache (fileid INTEGER PRIMARY KEY, storage INTEGER, path {text}, name {text},"
        f" mimetype INTEGER, etag {text}, mtime INTEGER, size BIGINT)",
        f"CREATE TABLE oc_users (uid {text}, displayname {text})",
        f"CREATE TABLE oc_preferences (userid {text}, appid {text}, configkey {text}, configvalue {text})",
    ):
        d._execute(sql)
    rows = {
        "oc_storages": [(1, "home::alice"), (2, "home::bob"), (3, "object::user:carol")],
        "oc_mimetypes": [(1, "image/jpeg")],
        "oc_users": [("alice", "Alice"), ("bob", "Bob"), ("carol", "Carol"), ("dave", "Dave")],
        # dave is disabled; alice's other prefs (and another app's "enabled") do not disable her
        "oc_preferences": [("dave", "core", "enabled", "false"), ("alice", "core", "enabled", "true"),
                           ("alice", "files", "enabled", "false"), ("bob", "core", "lang", "false")],
        "oc_filecache": [
            (10, 1, "files/Photos/a.jpg", "a.jpg", 1, "e10", 100, 1000),
            (11, 1, "files/b.jpg", "b.jpg", 1, "e11", 200, 1100),
            (12, 1, "files_trashbin/files/t.jpg.d1", "t.jpg.d1", 1, "e12", 300, 1200),  # alice's trash
            (20, 2, "files/bob.jpg", "bob.jpg", 1, "e20", 400, 2000),  # bob's file
            (30, 3, "files/c.jpg", "c.jpg", 1, "e30", 500, 3000),  # carol on S3 primary
        ],
        "oc_recognize_face_clusters": [(1, "Alice Mum", "alice"), (2, "Bob Dad", "bob"), (3, "100%_real", "alice"),
                                       (4, "Carol", "carol")],
        # (id, file_id, user_id, x, y, height, width, cluster_id)
        "oc_recognize_face_detections": [
            (1, 10, "alice", 0.1, 0.1, 0.2, 0.2, 1),
            (2, 11, "alice", 0.1, 0.1, 0.5, 0.5, 1),  # largest: alice's cover
            (3, 12, "alice", 0.1, 0.1, 0.9, 0.9, 1),  # in trash: never counted or shown
            (4, 20, "alice", 0.1, 0.1, 0.9, 0.9, 1),  # alice's uid on bob's storage: not alice's file
            (5, 20, "bob", 0.1, 0.1, 0.3, 0.3, 2),
            (6, 10, "alice", 0.5, 0.5, 0.1, 0.1, 3),
            (7, 30, "carol", 0.1, 0.1, 0.4, 0.4, 4),
        ],
    }
    for table, values in rows.items():
        for v in values:
            if table == "oc_recognize_face_detections":
                d._execute(f"INSERT INTO {table} (id, file_id, user_id, x, y, height, width, cluster_id) "
                           f"VALUES ({d._ph(8)})", v)
            else:
                d._execute(f"INSERT INTO {table} VALUES ({d._ph(len(v))})", v)
    d.commit()
    yield d
    d.close()


def ids(clusters):
    return [c["id"] for c in clusters]


class TestQueries:
    def test_clusters_only_own_and_counts_exclude_trash_and_foreign_storage(self, db):
        out = list_clusters(db, "alice", CROP)
        assert [(c["id"], c["title"], c["count"]) for c in out] == [(1, "Alice Mum", 2), (3, "100%_real", 1)]
        assert out[0]["cover"].startswith("/recognize-faces/alice/2.jpg?")

    def test_bob_never_sees_alice(self, db):
        assert ids(list_clusters(db, "bob", CROP)) == [2]
        assert get_cluster(db, "bob", CROP, 1, 0, 100) is None
        assert get_cluster(db, "alice", CROP, 2, 0, 100) is None

    def test_object_storage_home(self, db):
        assert ids(list_clusters(db, "carol", CROP)) == [4]

    def test_min(self, db):
        assert ids(list_clusters(db, "alice", CROP, min_count=2)) == [1]

    def test_cluster_files(self, db):
        out = get_cluster(db, "alice", CROP, 1, 0, 100)
        assert (out["id"], out["title"], out["count"]) == (1, "Alice Mum", 2)
        assert [f["fileid"] for f in out["files"]] == [11, 10]  # newest first
        f = out["files"][1]
        assert f["path"] == "Photos/a.jpg" and f["name"] == "a.jpg" and f["mime"] == "image/jpeg"
        assert (f["etag"], f["mtime"], f["size"]) == ("e10", 100, 1000)
        assert f["detection"]["id"] == 1 and f["detection"]["crop"].startswith("/recognize-faces/alice/1.jpg?")

    def test_pagination_and_past_the_end(self, db):
        page = get_cluster(db, "alice", CROP, 1, 1, 1)
        assert page["count"] == 2 and [f["fileid"] for f in page["files"]] == [10]
        empty = get_cluster(db, "alice", CROP, 1, 5, 10)
        assert (empty["title"], empty["count"], empty["files"]) == ("Alice Mum", 2, [])

    @pytest.mark.parametrize("q,want", [("mum", [1]), ("MUM", [1]), ("%", [3]), ("_", [3]), ("0%_", [3]),
                                        ("a", [1, 3]), ("!", []), ("Dad", [])])
    def test_search_escapes_wildcards_and_is_scoped(self, db, q, want):
        assert ids(list_clusters(db, "alice", CROP, 1, q)) == want


# ── HTTP ───────────────────────────────────────────────────────────────

@pytest.fixture
def server(config, db):
    srv = make_server(config, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def get(base, path, uid="alice", token=None):
    headers = {}
    if uid is not None:
        headers["Authorization"] = token or mint(uid, exp=int(time.time()) + 250)
    try:
        with urllib.request.urlopen(urllib.request.Request(base + path, headers=headers)) as r:
            return r.status, r.headers, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, e.headers, json.load(e)


class TestHttp:
    def test_health_needs_no_token(self, server):
        assert get(server, "/v1/health", uid=None)[::2] == (200, {"ok": True})

    def test_no_or_bad_token_is_401_without_detail(self, server):
        assert get(server, "/v1/clusters", uid=None)[::2] == (401, {})
        assert get(server, "/v1/clusters", token=mint(exp=1))[::2] == (401, {})

    def test_clusters(self, server):
        status, headers, body = get(server, "/v1/clusters?min=2")
        assert status == 200 and ids(body) == [1]
        assert headers["Cache-Control"] == "private, no-store"
        assert headers["Content-Type"] == "application/json"

    def test_cluster_and_foreign_404(self, server):
        status, _, body = get(server, "/v1/clusters/1?offset=0&limit=1")
        assert status == 200 and body["count"] == 2 and len(body["files"]) == 1
        assert get(server, "/v1/clusters/2")[0] == 404
        assert get(server, "/v1/clusters/2", uid="bob")[0] == 200
        assert get(server, "/v1/clusters/999")[0] == 404

    def test_search(self, server):
        assert ids(get(server, "/v1/search?q=%25")[2]) == [3]
        assert get(server, "/v1/search?q=Dad")[2] == []

    @pytest.mark.parametrize("path", [
        "/v1/clusters/1?limit=1001", "/v1/clusters/1?limit=0", "/v1/clusters/1?offset=-1",
        "/v1/clusters/1?limit=x", "/v1/clusters?min=0", "/v1/search?q=" + "a" * 101, "/v1/search?q=",
    ])
    def test_limits_are_400(self, server, path):
        assert get(server, path)[0] == 400

    def test_search_at_limit_ok(self, server):
        assert get(server, "/v1/search?q=" + "a" * 100)[0] == 200

    def test_unknown_path_404(self, server):
        assert get(server, "/v1/nope")[0] == 404


class TestUsers:
    def test_unknown_or_disabled_user_is_401(self, server):
        assert get(server, "/v1/clusters", uid="bob")[0] == 200
        assert get(server, "/v1/clusters", uid="mallory")[::2] == (401, {})  # validly signed, no such user
        assert get(server, "/v1/clusters", uid="dave")[::2] == (401, {})  # disabled

    def test_user_check_is_cached_60s(self, config, db):
        api = api_server.Api(config)
        clock = [1000.0]
        with patch("api_server.time.monotonic", lambda: clock[0]):
            assert api.user_ok("alice")
            db._execute("INSERT INTO oc_preferences VALUES ('alice', 'core', 'enabled', 'false')")
            db._execute("DELETE FROM oc_preferences WHERE userid = 'alice' AND configvalue = 'true'")
            db.commit()
            clock[0] += 59
            assert api.user_ok("alice")  # cached
            clock[0] += 2
            assert not api.user_ok("alice")  # re-read after 60 s: now disabled


class TestLoad:
    def test_health_never_touches_the_db(self, config):
        with patch("api_server.DB", side_effect=AssertionError("health opened a DB connection")):
            assert api_server.Api(config).handle("/v1/health", {}, None) == (200, {"ok": True})

    def test_503_when_db_slots_are_taken(self, config, db, monkeypatch):
        monkeypatch.setattr(api_server, "DB_WAIT", 0.05)
        api = api_server.Api(config)
        token = mint(exp=int(time.time()) + 250)
        assert api.handle("/v1/clusters", {}, token)[0] == 200
        for _ in range(api_server.DB_SLOTS):
            assert api._db_slots.acquire(blocking=False)
        assert not api._db_slots.acquire(blocking=False)  # bounded at DB_SLOTS
        assert api.handle("/v1/clusters", {}, token)[0] == 503
        assert api.handle("/v1/health", {}, None)[0] == 200
        api._db_slots.release()
        assert api.handle("/v1/clusters", {}, token)[0] == 200

    def test_connection_flood_is_capped(self, config, db, monkeypatch):
        """Idle connections hold handler slots; past the cap new ones are dropped, not threaded."""
        monkeypatch.setattr(api_server, "MAX_HANDLERS", 2)
        monkeypatch.setattr(api_server, "HANDLER_WAIT", 0.1)
        srv = make_server(config, port=0)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        idle = [socket.create_connection(srv.server_address[:2]) for _ in range(2)]
        try:
            time.sleep(0.2)  # both accepted and parked in a handler
            with pytest.raises((urllib.error.URLError, ConnectionError, http.client.HTTPException)):
                get(base, "/v1/health", uid=None)
            assert threading.active_count() < 10
        finally:
            for s in idle:
                s.close()
        time.sleep(0.2)
        assert get(base, "/v1/health", uid=None)[0] == 200
        srv.shutdown()
        srv.server_close()

    def test_handler_timeout(self):
        assert api_server.Handler.timeout == 30


def test_log_has_path_not_query(server, caplog):
    with caplog.at_level(logging.INFO, logger="recognize.api"):
        get(server, "/v1/search?q=Secret%20Name")
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "/v1/search" in text and "Secret" not in text and "q=" not in text


def test_php_ttl_matches_verifier():
    src = open(os.path.join(REPO, "lib", "Service", "FacesToken.php")).read()
    assert re.search(r"const TTL = (\d+);", src).group(1) == str(api_server.TOKEN_MAX_TTL) == "300"


def test_pool_reconnects_after_dropped_connection(config, db):
    api = api_server.Api(config)
    assert ids(api._with_db(lambda d: list_clusters(d, "alice", CROP))) == [1, 3]
    pooled = api._idle.get_nowait()
    pooled._conn.close()  # as a MySQL restart would leave it
    api._idle.put(pooled)
    assert ids(api._with_db(lambda d: list_clusters(d, "alice", CROP))) == [1, 3]
    assert api._idle.qsize() == 1


# ── Generated deployment files ─────────────────────────────────────────

def test_generated_nginx_and_unit():
    import setup_config
    conf = setup_config.generate_nginx_conf(
        {"crop_link_secret": "S3", "face_crop_root": "/srv/crops/", "api_port": 9000})
    assert 'secure_link_md5 "$secure_link_expires$uri S3";' in conf
    assert "alias /srv/crops/;" in conf and "proxy_pass http://127.0.0.1:9000/;" in conf
    assert "location ^~ /recognize-faces/" in conf and "__" not in conf
    unit = setup_config.generate_service_file({}, "/nc", "/venv/bin/python", setup_config.API_SERVICE_TEMPLATE)
    assert "ExecStart=/venv/bin/python api_server.py " in unit and "Restart=on-failure" in unit
    assert "NoNewPrivileges=true" in unit and "__" not in unit


def test_crop_location_keeps_security_headers_and_python_dir_denied():
    import setup_config
    conf = setup_config.generate_nginx_conf({"crop_link_secret": "S3"})
    crop = conf[conf.index("location ^~ /recognize-faces/"):]
    crop = crop[:crop.index("\n}")]
    # add_header in a location drops every server-level one: these are copied from nextcloud.conf.
    for header in ('add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;',
                   "add_header Referrer-Policy no-referrer always;",
                   'add_header X-Robots-Tag "noindex, nofollow";',
                   "add_header X-Content-Type-Options nosniff;",
                   'add_header Cache-Control "private, max-age=21600";'):
        assert header in crop
    assert "location ^~ /apps/recognize/python/ { deny all; }" in conf


def test_secret_goes_to_php_on_stdin_never_in_argv(monkeypatch):
    import setup_config
    calls = []
    monkeypatch.setattr(setup_config.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)))
    for root in (False, True):
        monkeypatch.setattr(setup_config.os, "getuid", lambda: 0 if root else 1000)
        assert setup_config.store_secret_in_nextcloud("/nc", "TOPSECRET-1", "api_token_secret")
    for (cmd, kw), root in zip(calls, (False, True)):
        assert not any("TOPSECRET" in a for a in cmd)
        assert kw["input"] == "TOPSECRET-1" and kw["check"] is True
        php = cmd[cmd.index("php"):]
        assert php[1] == "-r" and php[3:] == ["api_token_secret"]
        assert "require '/nc/lib/base.php';" in php[2] and "stream_get_contents(STDIN)" in php[2]
        assert "setValueString('recognize', $argv[1], $value, sensitive: true)" in php[2]
        assert (cmd[0] == "sudo") == (root and setup_config._detect_web_user() is not None)
    monkeypatch.undo()  # real subprocess.run for the lint
    if shutil.which("php"):
        lint = subprocess.run(["php", "-l"], input="<?php " + calls[0][0][-2], capture_output=True, text=True)
        assert lint.returncode == 0, lint.stdout + lint.stderr


def test_setup_keeps_custom_crop_root_and_tightens_existing_secret_files(tmp_path, monkeypatch):
    import setup_config
    out = {k: str(tmp_path / n) for k, n in (("OUTPUT_CONFIG", "nc_config.json"), ("OUTPUT_SERVICE", "r.service"),
                                            ("OUTPUT_API_SERVICE", "a.service"), ("OUTPUT_NGINX", "n.conf"))}
    for k, v in out.items():
        monkeypatch.setattr(setup_config, k, v)
    with open(out["OUTPUT_CONFIG"], "w") as f:
        json.dump({"face_crop_root": "/srv/custom", "api_token_secret": "keep-a", "crop_link_secret": "keep-c"}, f)
    with open(out["OUTPUT_NGINX"], "w") as f:
        f.write("old")
    os.chmod(out["OUTPUT_CONFIG"], 0o644)
    os.chmod(out["OUTPUT_NGINX"], 0o644)
    monkeypatch.setattr(setup_config, "read_nextcloud_config", lambda root: {
        "dbtype": "mysql", "dbhost": "h", "dbname": "n", "dbuser": "u", "dbpassword": "p", "datadirectory": "/d"})
    monkeypatch.setattr(setup_config, "store_secret_in_nextcloud", lambda *a, **kw: True)
    monkeypatch.setattr(setup_config.sys, "argv", ["setup_config.py", "--nextcloud-root", "/nc"])
    setup_config.main()
    cfg = json.load(open(out["OUTPUT_CONFIG"]))
    assert cfg["face_crop_root"] == "/srv/custom"
    assert (cfg["api_token_secret"], cfg["crop_link_secret"]) == ("keep-a", "keep-c")
    assert stat.S_IMODE(os.stat(out["OUTPUT_CONFIG"]).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(out["OUTPUT_NGINX"]).st_mode) == 0o600
    assert "alias /srv/custom/;" in open(out["OUTPUT_NGINX"]).read()
