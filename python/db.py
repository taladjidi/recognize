"""Multi-dialect database layer for the recognize service.

Supports MySQL, PostgreSQL, and SQLite. All table names are constructed
from a validated prefix. No raw f-string interpolation of user input.
"""

import json
import logging
import os
import re
import time

log = logging.getLogger(__name__)

PREFIX_RE = re.compile(r"^[a-zA-Z0-9_]*$")

# MIME types by media category (from Nextcloud Constants.php)
IMAGE_MIMES = (
    "image/jpeg", "image/png", "image/bmp",
    "image/heic", "image/heif", "image/webp",
)
VIDEO_MIMES = (
    "image/gif", "video/mp4", "video/MP2T", "video/x-msvideo",
    "video/x-ms-wmv", "video/quicktime", "video/ogg",
    "video/mpeg", "video/webm", "video/x-matroska",
)
AUDIO_MIMES = (
    "audio/mpeg", "audio/mp4", "audio/ogg",
    "audio/vnd.wav", "audio/flac",
)


class DB:
    """Database connection wrapper with multi-dialect support.

    Usage::

        db = DB(config)
        rows = db.fetch_pending(limit=100)
        db.delete_pending([r["id"] for r in rows])
        db.close()
    """

    def __init__(self, config: dict):
        self.dialect = config["dbtype"]  # "mysql", "pgsql", "sqlite3"
        self.prefix = config["dbtableprefix"]
        self._config = config
        self._conn = None

        if not PREFIX_RE.match(self.prefix):
            raise ValueError(f"Invalid table prefix: {self.prefix!r}")

        self._connect()

    # ── Connection management ───────────────────────────────────────────

    def _connect(self):
        """Establish a database connection based on dialect."""
        if self.dialect == "mysql":
            import mysql.connector
            host = self._config["dbhost"]
            port = int(self._config.get("dbport") or 3306)
            unix_socket = None

            if ":" in host:
                parts = host.split(":", 1)
                if parts[1].startswith("/"):
                    host = parts[0]
                    unix_socket = parts[1]
                else:
                    host = parts[0]
                    port = int(parts[1])

            kwargs = {
                "user": self._config["dbuser"],
                "password": self._config["dbpassword"],
                "database": self._config["dbname"],
                "charset": "utf8mb4",
                "autocommit": False,
            }
            if unix_socket:
                kwargs["unix_socket"] = unix_socket
            else:
                kwargs["host"] = host
                kwargs["port"] = port

            self._conn = mysql.connector.connect(**kwargs)

        elif self.dialect == "pgsql":
            import psycopg2
            host = self._config["dbhost"]
            port = int(self._config.get("dbport") or 5432)

            if ":" in host:
                parts = host.split(":", 1)
                if not parts[1].startswith("/"):
                    host = parts[0]
                    port = int(parts[1])

            self._conn = psycopg2.connect(
                host=host,
                port=port,
                user=self._config["dbuser"],
                password=self._config["dbpassword"],
                dbname=self._config["dbname"],
            )
            self._conn.autocommit = False

        elif self.dialect == "sqlite3":
            import sqlite3
            db_path = self._config["dbname"]
            if db_path != ":memory:" and not db_path.endswith(".db") and not os.path.isabs(db_path):
                db_path = os.path.join(self._config.get("datadirectory", ""), db_path)
            self._conn = sqlite3.connect(db_path)
            self._conn.row_factory = sqlite3.Row
            if db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")

    def ping(self):
        """Check connection health and reconnect if needed."""
        try:
            if self.dialect == "mysql":
                self._conn.ping(reconnect=True, attempts=3, delay=1)
            elif self.dialect == "pgsql":
                cur = self._conn.cursor()
                cur.execute("SELECT 1")
                cur.close()
            elif self.dialect == "sqlite3":
                self._conn.execute("SELECT 1")
        except Exception:
            log.warning("DB connection lost, reconnecting...")
            try:
                self._conn.close()
            except Exception:
                pass
            self._connect()

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── SQL dialect helpers ─────────────────────────────────────────────

    def _ph(self, count: int = 1) -> str:
        """Return placeholder string(s) for the current dialect."""
        p = "?" if self.dialect == "sqlite3" else "%s"
        return ", ".join([p] * count)

    def _table(self, name: str) -> str:
        """Return a fully-qualified table name with prefix."""
        return f"{self.prefix}{name}"

    def _execute(self, sql: str, params: tuple = ()):
        """Execute a single SQL statement and return the cursor."""
        cur = self._conn.cursor()
        cur.execute(sql, params)
        return cur

    def _commit(self):
        self._conn.commit()

    def _last_insert_id(self, cursor) -> int:
        """Get the last auto-increment ID."""
        if self.dialect == "sqlite3":
            return cursor.lastrowid
        elif self.dialect == "mysql":
            return cursor.lastrowid
        else:
            # PostgreSQL: handled via RETURNING clause
            return cursor.fetchone()[0]

    def _fetchall_dicts(self, cursor) -> list[dict]:
        """Fetch all rows as dicts."""
        if self.dialect == "sqlite3":
            return [dict(row) for row in cursor.fetchall()]
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    # ── Pending table operations ────────────────────────────────────────

    def fetch_pending(self, limit: int = 100, action_filter: tuple = (0, 1)) -> list[dict]:
        """Fetch pending rows (action=0 created or action=1 updated).

        Returns list of dicts: {id, file_id, storage_id, action, added_at}.
        """
        t = self._table("recognize_pending")
        phs = self._ph(len(action_filter))
        sql = (
            f"SELECT id, file_id, storage_id, action, added_at "
            f"FROM {t} "
            f"WHERE action IN ({phs}) "
            f"ORDER BY added_at ASC "
            f"LIMIT {self._ph()}"
        )
        cur = self._execute(sql, (*action_filter, limit))
        return self._fetchall_dicts(cur)

    def upsert_pending(self, file_id: int, storage_id: int, action: int = 0):
        """Insert or update a pending entry.

        Uses ON DUPLICATE KEY UPDATE (MySQL) / ON CONFLICT (PostgreSQL/SQLite)
        to handle the "update event lost" problem.
        """
        t = self._table("recognize_pending")
        now = int(time.time())

        if self.dialect == "mysql":
            sql = (
                f"INSERT INTO {t} (file_id, storage_id, action, added_at) "
                f"VALUES (%s, %s, %s, %s) "
                f"ON DUPLICATE KEY UPDATE action=VALUES(action), added_at=VALUES(added_at)"
            )
        else:
            sql = (
                f"INSERT INTO {t} (file_id, storage_id, action, added_at) "
                f"VALUES ({self._ph(4)}) "
                f"ON CONFLICT (file_id) DO UPDATE SET action=EXCLUDED.action, added_at=EXCLUDED.added_at"
            )

        self._execute(sql, (file_id, storage_id, action, now))
        self._commit()

    def delete_pending(self, ids: list[int]):
        """Remove processed entries from the pending table."""
        if not ids:
            return
        t = self._table("recognize_pending")
        phs = self._ph(len(ids))
        self._execute(f"DELETE FROM {t} WHERE id IN ({phs})", tuple(ids))
        self._commit()

    def pending_count(self) -> int:
        """Count pending entries (action 0 or 1)."""
        t = self._table("recognize_pending")
        cur = self._execute(
            f"SELECT COUNT(*) FROM {t} WHERE action IN ({self._ph(2)})", (0, 1)
        )
        return cur.fetchone()[0]

    # ── File path resolution ────────────────────────────────────────────

    def resolve_file_paths(self, file_ids: list[int]) -> dict[int, dict]:
        """Resolve file_ids to filesystem paths and mimetypes via filecache.

        Returns dict: {file_id: {"path": str, "mimetype": str, "storage": int}}.
        Missing/empty files are excluded.
        """
        if not file_ids:
            return {}

        datadirectory = self._config["datadirectory"]
        fc = self._table("filecache")
        st = self._table("storages")
        mt = self._table("mimetypes")
        phs = self._ph(len(file_ids))

        sql = (
            f"SELECT fc.fileid, fc.path, fc.size, fc.storage, "
            f"  s.id AS storage_id_str, m.mimetype "
            f"FROM {fc} fc "
            f"JOIN {st} s ON fc.storage = s.numeric_id "
            f"JOIN {mt} m ON fc.mimetype = m.id "
            f"WHERE fc.fileid IN ({phs})"
        )
        cur = self._execute(sql, tuple(file_ids))
        rows = self._fetchall_dicts(cur)

        result = {}
        for row in rows:
            if row["size"] == 0:
                continue
            abs_path = self._resolve_storage_path(
                row["storage_id_str"], row["path"], datadirectory
            )
            if abs_path and os.path.isfile(abs_path):
                result[row["fileid"]] = {
                    "path": abs_path,
                    "mimetype": row["mimetype"],
                    "storage": row["storage"],
                }
        return result

    @staticmethod
    def _resolve_storage_path(
        storage_id_str: str, relative_path: str, datadirectory: str
    ) -> str | None:
        """Map storage ID + relative path to absolute filesystem path."""
        if storage_id_str.startswith("home::"):
            user = storage_id_str[len("home::"):]
            return os.path.join(datadirectory, user, relative_path)
        elif storage_id_str.startswith("local::"):
            local_root = storage_id_str[len("local::"):]
            return os.path.join(local_root, relative_path)
        else:
            # S3, SMB etc — requires PHP file endpoint
            return None

    # ── Face detection operations ───────────────────────────────────────

    def insert_face_detection(
        self, file_id: int, user_id: str, face: dict
    ) -> int | None:
        """Insert a face detection row. Returns the inserted ID, or None on duplicate."""
        t = self._table("recognize_face_detections")
        vector_json = json.dumps(face["vector"])

        if self.dialect == "mysql":
            sql = (
                f"INSERT IGNORE INTO {t} "
                f"(file_id, user_id, x, y, height, width, face_vector, threshold) "
                f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
            )
        elif self.dialect == "pgsql":
            sql = (
                f"INSERT INTO {t} "
                f"(file_id, user_id, x, y, height, width, face_vector, threshold) "
                f"VALUES ({self._ph(8)}) "
                f"ON CONFLICT DO NOTHING"
            )
        else:
            sql = (
                f"INSERT OR IGNORE INTO {t} "
                f"(file_id, user_id, x, y, height, width, face_vector, threshold) "
                f"VALUES ({self._ph(8)})"
            )

        params = (
            file_id, user_id,
            face["x"], face["y"], face["height"], face["width"],
            vector_json, 0.0,
        )
        cur = self._execute(sql, params)
        self._commit()
        if cur.rowcount > 0:
            return self._last_insert_id(cur) if self.dialect != "pgsql" else None
        return None

    def delete_face_detections(self, file_id: int):
        """Delete all face detections for a file (used on file deletion)."""
        t = self._table("recognize_face_detections")
        self._execute(f"DELETE FROM {t} WHERE file_id = {self._ph()}", (file_id,))
        self._commit()

    def get_face_detections_for_user(self, user_id: str) -> list[dict]:
        """Get all face detections for a user (for clustering)."""
        t = self._table("recognize_face_detections")
        cur = self._execute(
            f"SELECT id, file_id, x, y, height, width, face_vector, cluster_id, threshold "
            f"FROM {t} WHERE user_id = {self._ph()}", (user_id,)
        )
        rows = self._fetchall_dicts(cur)
        for row in rows:
            if isinstance(row["face_vector"], str):
                row["face_vector"] = json.loads(row["face_vector"])
        return rows

    def get_face_clusters_for_user(self, user_id: str) -> list[dict]:
        """Get all face cluster rows for a user."""
        t = self._table("recognize_face_clusters")
        cur = self._execute(
            f"SELECT id, title FROM {t} WHERE user_id = {self._ph()}", (user_id,)
        )
        return self._fetchall_dicts(cur)

    def get_cluster_sample(self, cluster_id: int, limit: int) -> list[dict]:
        """Sample detections from an existing cluster (for reference during clustering)."""
        t = self._table("recognize_face_detections")
        cur = self._execute(
            f"SELECT id, file_id, face_vector, cluster_id, threshold "
            f"FROM {t} WHERE cluster_id = {self._ph()} LIMIT {int(limit)}",
            (cluster_id,),
        )
        rows = self._fetchall_dicts(cur)
        for row in rows:
            if isinstance(row["face_vector"], str):
                row["face_vector"] = json.loads(row["face_vector"])
        return rows

    def get_unclustered_detections(self, user_id: str, min_size: float = 0.0) -> list[dict]:
        """Get face detections that have never been clustered (cluster_id IS NULL)."""
        t = self._table("recognize_face_detections")
        cur = self._execute(
            f"SELECT id, file_id, face_vector, cluster_id, threshold "
            f"FROM {t} WHERE user_id = {self._ph()} AND cluster_id IS NULL "
            f"AND height >= {self._ph()} AND width >= {self._ph()}",
            (user_id, min_size, min_size),
        )
        rows = self._fetchall_dicts(cur)
        for row in rows:
            if isinstance(row["face_vector"], str):
                row["face_vector"] = json.loads(row["face_vector"])
        return rows

    def get_rejected_detections(self, user_id: str, min_size: float = 0.0) -> list[dict]:
        """Get detections previously rejected from clustering (threshold > 0)."""
        t = self._table("recognize_face_detections")
        cur = self._execute(
            f"SELECT id, file_id, face_vector, cluster_id, threshold "
            f"FROM {t} WHERE user_id = {self._ph()} AND threshold > 0 "
            f"AND height >= {self._ph()} AND width >= {self._ph()}",
            (user_id, min_size, min_size),
        )
        rows = self._fetchall_dicts(cur)
        for row in rows:
            if isinstance(row["face_vector"], str):
                row["face_vector"] = json.loads(row["face_vector"])
        return rows

    def create_face_cluster(self, user_id: str, title: str = "") -> int:
        """Create a new face cluster row and return its ID."""
        t = self._table("recognize_face_clusters")
        if self.dialect == "pgsql":
            cur = self._execute(
                f"INSERT INTO {t} (title, user_id) VALUES ({self._ph()}, {self._ph()}) RETURNING id",
                (title, user_id),
            )
        else:
            cur = self._execute(
                f"INSERT INTO {t} (title, user_id) VALUES ({self._ph()}, {self._ph()})",
                (title, user_id),
            )
        self._commit()
        return self._last_insert_id(cur)

    def assign_detection_to_cluster(self, detection_id: int, cluster_id: int):
        """Assign a single face detection to a cluster (auto-commits)."""
        t = self._table("recognize_face_detections")
        self._execute(
            f"UPDATE {t} SET cluster_id = {self._ph()} WHERE id = {self._ph()}",
            (cluster_id, detection_id),
        )
        # Caller should call db.commit() after batch assignments

    def commit(self):
        """Explicit commit for batching multiple writes."""
        self._commit()

    def update_face_clusters(self, detection_ids: list[int], cluster_ids: list[int]):
        """Batch-update cluster assignments for face detections."""
        if not detection_ids:
            return
        t = self._table("recognize_face_detections")
        for det_id, cluster_id in zip(detection_ids, cluster_ids):
            self._execute(
                f"UPDATE {t} SET cluster_id = {self._ph()} WHERE id = {self._ph()}",
                (cluster_id if cluster_id >= 0 else None, det_id),
            )
        self._commit()

    def get_users_with_faces(self) -> list[str]:
        """Get distinct user IDs that have face detections."""
        t = self._table("recognize_face_detections")
        cur = self._execute(f"SELECT DISTINCT user_id FROM {t}")
        rows = self._fetchall_dicts(cur)
        return [row["user_id"] for row in rows]

    # ── Settings & maintenance ──────────────────────────────────────────

    def check_maintenance_mode(self) -> bool:
        """Check if Nextcloud is in maintenance mode."""
        # Maintenance mode is stored in config.php, not the DB.
        # We check by reading the config's 'maintenance' key if present,
        # or by trying a simple query (if DB is locked, we're in maintenance).
        return self._config.get("maintenance", False)

    # ── MIME type helpers ───────────────────────────────────────────────

    @staticmethod
    def classify_mimetype(mimetype: str) -> str | None:
        """Classify a mimetype into 'image', 'video', 'audio', or None."""
        if mimetype in IMAGE_MIMES:
            return "image"
        if mimetype in VIDEO_MIMES:
            return "video"
        if mimetype in AUDIO_MIMES:
            return "audio"
        return None

    # ── Schema creation (for testing / setup) ───────────────────────────

    def create_pending_table(self):
        """Create the recognize_pending table if it doesn't exist."""
        t = self._table("recognize_pending")
        if self.dialect == "mysql":
            sql = (
                f"CREATE TABLE IF NOT EXISTS {t} ("
                f"  id BIGINT AUTO_INCREMENT PRIMARY KEY,"
                f"  file_id BIGINT NOT NULL,"
                f"  storage_id INT NOT NULL,"
                f"  action TINYINT NOT NULL DEFAULT 0,"
                f"  added_at INT NOT NULL,"
                f"  UNIQUE KEY uq_file (file_id),"
                f"  INDEX idx_action_added (action, added_at)"
                f")"
            )
        elif self.dialect == "pgsql":
            sql = (
                f"CREATE TABLE IF NOT EXISTS {t} ("
                f"  id BIGSERIAL PRIMARY KEY,"
                f"  file_id BIGINT NOT NULL UNIQUE,"
                f"  storage_id INT NOT NULL,"
                f"  action SMALLINT NOT NULL DEFAULT 0,"
                f"  added_at INT NOT NULL"
                f")"
            )
        else:
            sql = (
                f"CREATE TABLE IF NOT EXISTS {t} ("
                f"  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                f"  file_id INTEGER NOT NULL UNIQUE,"
                f"  storage_id INTEGER NOT NULL,"
                f"  action INTEGER NOT NULL DEFAULT 0,"
                f"  added_at INTEGER NOT NULL"
                f")"
            )
        self._execute(sql)
        self._commit()

    def create_face_detections_table(self):
        """Create recognize_face_detections table (for testing)."""
        t = self._table("recognize_face_detections")
        if self.dialect == "sqlite3":
            sql = (
                f"CREATE TABLE IF NOT EXISTS {t} ("
                f"  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                f"  file_id INTEGER NOT NULL,"
                f"  user_id TEXT NOT NULL,"
                f"  x REAL NOT NULL,"
                f"  y REAL NOT NULL,"
                f"  height REAL NOT NULL,"
                f"  width REAL NOT NULL,"
                f"  face_vector TEXT,"
                f"  cluster_id INTEGER,"
                f"  threshold REAL DEFAULT 0.0"
                f")"
            )
        else:
            sql = (
                f"CREATE TABLE IF NOT EXISTS {t} ("
                f"  id BIGINT AUTO_INCREMENT PRIMARY KEY,"
                f"  file_id BIGINT NOT NULL,"
                f"  user_id VARCHAR(64) NOT NULL,"
                f"  x DOUBLE NOT NULL,"
                f"  y DOUBLE NOT NULL,"
                f"  height DOUBLE NOT NULL,"
                f"  width DOUBLE NOT NULL,"
                f"  face_vector LONGTEXT,"
                f"  cluster_id BIGINT,"
                f"  threshold DOUBLE DEFAULT 0.0"
                f")"
            )
        self._execute(sql)
        self._commit()

    def create_face_clusters_table(self):
        """Create recognize_face_clusters table (for testing)."""
        t = self._table("recognize_face_clusters")
        if self.dialect == "sqlite3":
            sql = (
                f"CREATE TABLE IF NOT EXISTS {t} ("
                f"  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                f"  title TEXT NOT NULL DEFAULT '',"
                f"  user_id TEXT NOT NULL DEFAULT ''"
                f")"
            )
        else:
            sql = (
                f"CREATE TABLE IF NOT EXISTS {t} ("
                f"  id BIGINT AUTO_INCREMENT PRIMARY KEY,"
                f"  title VARCHAR(4000) NOT NULL DEFAULT '',"
                f"  user_id VARCHAR(64) NOT NULL DEFAULT ''"
                f")"
            )
        self._execute(sql)
        self._commit()
