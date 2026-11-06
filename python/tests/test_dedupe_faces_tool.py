"""tools/dedupe_faces.py: remove near-duplicate face detections, keeping names and user edits."""

import importlib.util
import json
import os
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from db import DB
from service import Service

_spec = importlib.util.spec_from_file_location(
    "dedupe_faces", os.path.join(os.path.dirname(__file__), "..", "..", "tools", "dedupe_faces.py"))
dd = importlib.util.module_from_spec(_spec)
sys.modules["dedupe_faces"] = dd
_spec.loader.exec_module(dd)

W = 0.2


def _dx(r):
    """x shift of a W x W box that gives IoU r with the unshifted one."""
    return W * (1 - r) / (1 + r)


@pytest.fixture
def cfg(tmp_path):
    return {"dbtype": "sqlite3", "dbhost": "", "dbuser": "", "dbpassword": "", "dbname": str(tmp_path / "t.db"),
            "dbtableprefix": "oc_", "datadirectory": str(tmp_path / "data"),
            "config_php": str(tmp_path / "config.php")}


@pytest.fixture
def db(cfg):
    d = DB(cfg)
    d.create_face_detections_table()
    d.create_face_clusters_table()
    yield d
    d.close()


@pytest.fixture
def bk(tmp_path):
    with open(tmp_path / "backup.jsonl", "a", encoding="utf-8") as f:
        yield f


def _backup(tmp_path):
    with open(tmp_path / "backup.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _cluster(db, title="", user="alice"):
    return db.create_face_cluster(user, title)


def _det(db, file_id, x=0.1, user="alice", cluster_id=None, threshold=0.0, y=0.1, w=W, h=W):
    cur = db._execute("INSERT INTO oc_recognize_face_detections (file_id, user_id, x, y, width, height, "
                      "face_vector, cluster_id, threshold) VALUES (?, ?, ?, ?, ?, ?, '[1.0]', ?, ?)",
                      (file_id, user, x, y, w, h, cluster_id, threshold))
    db._commit()
    return cur.lastrowid


def _dump(db):
    return [tuple(r) for r in db._execute("SELECT * FROM oc_recognize_face_detections ORDER BY id").fetchall()]


def _ids(db):
    return {r[0] for r in _dump(db)}


def test_iou_boundary_089_kept_091_removed(db, bk):
    assert dd.iou({"x": 0, "y": 0, "width": W, "height": W},
                  {"x": _dx(0.91), "y": 0, "width": W, "height": W}) == pytest.approx(0.91)
    a = _det(db, 1)
    b = _det(db, 1, x=0.1 + _dx(0.91))
    c = _det(db, 2)
    d = _det(db, 2, x=0.1 + _dx(0.89))
    out = dd.run(db, bk)
    assert _ids(db) == {a, c, d}
    assert [(r["id"], k["id"]) for r, k in out["deleted"]] == [(b, a)]


def test_iou_exactly_09_is_a_duplicate(db, bk):
    assert dd.iou({"x": 0, "y": 0, "width": 10, "height": 1}, {"x": 0, "y": 0, "width": 9, "height": 1}) == 0.9
    a = _det(db, 1, x=0, y=0, w=10, h=1)
    _det(db, 1, x=0, y=0, w=9, h=1)
    dd.run(db, bk)
    assert _ids(db) == {a}


def test_zero_area_boxes_duplicates_only_when_identical(db, bk):
    a = _det(db, 1, w=0, h=0)
    _det(db, 1, w=0, h=0)  # identical: a duplicate, as upstream
    c = _det(db, 2, w=0, h=0)
    d = _det(db, 2, x=0.1 + 1e-6, w=0, h=0)
    e = _det(db, 3, w=0.2, h=0)  # zero-area lines overlapping on x
    f = _det(db, 3, x=0.15, w=0.2, h=0)
    dd.run(db, bk)
    assert _ids(db) == {a, c, d, e, f}


def test_null_box_column_never_matches(cfg, tmp_path):
    # Production x, y, width, height allow NULL (Version002002000Date20220614094721); the test schema does not.
    d = DB(cfg)
    d.create_face_clusters_table()
    d._execute("CREATE TABLE oc_recognize_face_detections (id INTEGER PRIMARY KEY AUTOINCREMENT, file_id INTEGER, "
               "user_id TEXT, x REAL, y REAL, height REAL, width REAL, face_vector TEXT NOT NULL, "
               "cluster_id INTEGER, threshold REAL DEFAULT 0.0)")
    a = _det(d, 1)
    _det(d, 1)
    nulls = {_det(d, 2), _det(d, 2, w=None), _det(d, 2, x=None), _det(d, 2, h=None)}
    with open(tmp_path / "b.jsonl", "a") as f:
        out = dd.run(d, f)
    assert _ids(d) == {a} | nulls and len(out["deleted"]) == 1
    d.close()


def test_keep_priority_titled_then_clustered_then_removed_then_oldest(db, bk):
    named, plain = _cluster(db, "Alice"), _cluster(db)
    f1 = [_det(db, 1), _det(db, 1, cluster_id=plain), _det(db, 1, cluster_id=named)]
    f2 = [_det(db, 2), _det(db, 2, cluster_id=plain)]
    f3 = [_det(db, 3, cluster_id=-1), _det(db, 3)]
    f4 = [_det(db, 4), _det(db, 4, threshold=0.5)]  # a user removal keeps its threshold
    dd.run(db, bk)
    assert _ids(db) == {f1[2], f2[1], f3[0], f4[1]}


def test_rejected_minus_one_beats_null_at_a_higher_id(db, bk):
    _det(db, 1)
    rejected = _det(db, 1, cluster_id=-1)  # listed in Unassigned: upstream keeps it
    dd.run(db, bk)
    assert _ids(db) == {rejected}


def test_user_removal_next_to_a_clustered_duplicate_goes_to_review(db, bk):
    alice = _cluster(db, "Alice")
    other = _det(db, 2, cluster_id=alice)  # Alice has another face
    removed = _det(db, 1, threshold=0.7)  # the user said: not Alice
    dup = _det(db, 1, x=0.101, cluster_id=alice)
    before = _dump(db)
    out = dd.run(db, bk)
    assert _dump(db) == before and not out["deleted"]
    assert [{r["id"] for r in s} for s in out["conflicts"]] == [{removed, dup}] and other in _ids(db)


def test_two_different_titled_clusters_untouched_and_reported(db, bk):
    alice, bob = _cluster(db, "Alice"), _cluster(db, "Bob")
    ids = {_det(db, 1, cluster_id=alice), _det(db, 1, cluster_id=bob), _det(db, 1)}
    _det(db, 2, cluster_id=alice)  # another face, so neither cluster would empty
    _det(db, 2, x=0.6, cluster_id=bob)
    before = _dump(db)
    out = dd.run(db, bk)
    assert _dump(db) == before
    assert [{r["id"] for r in s} for s in out["conflicts"]] == [ids]
    assert {r["title"] for r in out["conflicts"][0]} == {"Alice", "Bob", ""}


def test_last_face_of_a_titled_cluster_never_deleted(db, bk, monkeypatch):
    alice, bob = _cluster(db, "Alice"), _cluster(db, "Bob")
    only = _det(db, 1, cluster_id=alice)  # Alice's only face
    other = _det(db, 1)
    b1 = _det(db, 2, cluster_id=bob)
    b2 = _det(db, 2, cluster_id=bob)  # same face twice in Bob: one may go
    real = dd.plan
    # A planner bug that keeps the wrong row: the guard must still hold.
    monkeypatch.setattr(dd, "plan", lambda rows: ([(r, k) for k, r in real(rows)[0]], []))
    out = dd.run(db, bk)
    assert _ids(db) == {only, other, b2}
    assert [r["id"] for r in out["protected"]] == [only]
    assert b1 not in _ids(db)


def test_cross_user_and_cross_file_never_matched(db, bk):
    before = {_det(db, 1), _det(db, 1, user="bob"), _det(db, 2)}
    out = dd.run(db, bk)
    assert _ids(db) == before and not out["deleted"]


def test_other_users_row_at_the_same_box_survives_a_settled_set(db, bk):
    a = _det(db, 1)
    _det(db, 1, x=0.101)
    bob = _det(db, 1, x=0.1005, user="bob")  # newest: a re-read by file only would delete it
    dd.run(db, bk)
    assert _ids(db) == {a, bob}


def test_dry_run_writes_nothing_and_reports_the_same(db, monkeypatch, capsys):
    a = _det(db, 1)
    b = _det(db, 1, x=0.101)
    before = _dump(db)
    real, sqls = db._execute, []
    monkeypatch.setattr(db, "_execute", lambda sql, params=(): sqls.append(sql) or real(sql, params))
    out = dd.run(db)
    monkeypatch.undo()
    assert sqls and all(s.lstrip().startswith("SELECT") for s in sqls)  # not even a rolled-back DELETE
    assert _dump(db) == before
    assert [(r["id"], k["id"]) for r, k in out["deleted"]] == [(b, a)]
    assert f"would delete id={b} " in capsys.readouterr().out
    other = DB(db._config)  # no open write transaction left behind
    other._execute("DELETE FROM oc_recognize_face_detections WHERE id = ?", (a,))
    other.rollback()
    other.close()


@pytest.mark.parametrize("apply", [True, False])
def test_sqlite_set_is_write_locked_only_when_applying(db, tmp_path, monkeypatch, apply):
    _det(db, 1)
    _det(db, 1, x=0.101)
    other = sqlite3.connect(db._config["dbname"], timeout=0)
    real, seen = dd._select, []

    def select_then_write_elsewhere(d, where, params, lock=False):
        rows = real(d, where, params, lock)
        if "d.user_id" in where:  # the re-read in _settle
            try:
                other.execute("UPDATE oc_recognize_face_detections SET threshold = 0 WHERE id = -1")
                other.rollback()
                seen.append("written")
            except sqlite3.OperationalError as e:
                seen.append(str(e))
        return rows

    monkeypatch.setattr(dd, "_select", select_then_write_elsewhere)
    with open(tmp_path / "b.jsonl", "a") as f:
        dd.run(db, f if apply else None)
    other.close()
    assert seen == (["database is locked"] if apply else ["written"])


def test_re_read_locks_rows_for_update_only_when_applying(tmp_path):
    for dialect, suffix in (("mysql", " FOR UPDATE"), ("pgsql", " FOR UPDATE OF d")):
        sqls = []
        fake = SimpleNamespace(dialect=dialect, _table=lambda n: n, _ph=lambda n=1: ", ".join(["%s"] * n),
                               _execute=lambda sql, params=(): sqls.append(sql), _fetchall_dicts=lambda c: [])
        with open(tmp_path / "b.jsonl", "a") as f:
            dd._settle(fake, (1, "alice"), f)
        dd._settle(fake, (1, "alice"), None)
        assert sqls[0].endswith(suffix) and "FOR UPDATE" not in sqls[1]


def test_crops_left_to_the_daemon_which_removes_the_deleted_ones(db, bk, tmp_path):
    root = tmp_path / "crops"
    (root / "alice").mkdir(parents=True)
    a = _det(db, 1)
    b = _det(db, 1, x=0.101)
    c = _det(db, 2)  # newest id: the daemon only cleans ids up to the max detection id
    for i in (a, b, c):
        (root / "alice" / f"{i}.jpg").write_bytes(b"jpg")
    dd.run(db, bk)
    assert sorted(os.listdir(root / "alice")) == sorted(f"{i}.jpg" for i in (a, b, c))  # the tool leaves them
    Service._cleanup_crops_one_user(SimpleNamespace(crop_root=str(root), _crop_users=[]), db)
    assert sorted(os.listdir(root / "alice")) == sorted(f"{i}.jpg" for i in (a, c))


def test_second_run_is_a_no_op(db, bk):
    named = _cluster(db, "Alice")
    _det(db, 1)
    _det(db, 1, x=0.101, cluster_id=named)
    _det(db, 1, x=0.102)
    assert len(dd.run(db, bk)["deleted"]) == 2
    after = _dump(db)
    out = dd.run(db, bk)
    assert _dump(db) == after and not out["deleted"] and not out["conflicts"]


def test_rows_changed_since_the_scan_are_re_planned(db, bk, monkeypatch):
    alice, plain = _cluster(db, "Alice"), _cluster(db)
    ids = {_det(db, 1, cluster_id=alice), _det(db, 1, cluster_id=plain)}
    real = dd._groups

    def groups_then_user_names_it(rows):
        g = real(rows)
        db._execute("UPDATE oc_recognize_face_clusters SET title = 'Bob' WHERE id = ?", (plain,))
        db._commit()
        return g

    monkeypatch.setattr(dd, "_groups", groups_then_user_names_it)
    out = dd.run(db, bk)
    assert _ids(db) == ids and not out["deleted"] and len(out["conflicts"]) == 1


def _failing_deletes(db, monkeypatch, fail):
    """Patch db._execute: fail(n) is called on the n-th DELETE and may raise or return a cursor stand-in."""
    real, n = db._execute, [0]

    def ex(sql, params=()):
        if sql.startswith("DELETE"):
            n[0] += 1
            fake = fail(n[0])
            if fake is not None:
                real(sql, params)
                return fake
        return real(sql, params)

    monkeypatch.setattr(db, "_execute", ex)
    return n


def test_db_error_rolls_back_that_set_aborts_and_keeps_earlier_sets_logged(db, bk, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dd, "PAGE", 1)  # and across pages
    a1 = _det(db, 1)
    b1 = _det(db, 1, x=0.101)
    a2 = _det(db, 2)
    b2 = _det(db, 2, x=0.101)

    def fail(n):
        if n == 2:
            raise RuntimeError("disk I/O error")

    n = _failing_deletes(db, monkeypatch, fail)
    with pytest.raises(RuntimeError):
        dd.run(db, bk)
    monkeypatch.undo()
    assert n[0] == 2 and _ids(db) == {a1, a2, b2}  # not retried, set 2 untouched
    assert f"deleted id={b1} " in capsys.readouterr().out  # printed before the abort
    assert b1 in [e["row"]["id"] for e in _backup(tmp_path)]


def test_delete_rowcount_mismatch_aborts_and_leaves_the_set(db, bk, monkeypatch):
    a1 = _det(db, 1)
    _det(db, 1, x=0.101)
    a2 = _det(db, 2)
    b2 = _det(db, 2, x=0.101)
    _failing_deletes(db, monkeypatch, lambda n: SimpleNamespace(rowcount=0) if n == 2 else None)
    with pytest.raises(RuntimeError, match="deleted 0 rows, expected 1"):
        dd.run(db, bk)
    monkeypatch.undo()
    assert _ids(db) == {a1, a2, b2}


class _MySQLError(Exception):
    def __init__(self, errno):
        super().__init__(f"{errno} (40001): Deadlock found")
        self.errno = errno


@pytest.mark.parametrize("errno", [1213, 1205])
def test_deadlock_retries_the_set_once(db, bk, tmp_path, monkeypatch, capsys, errno):
    a1 = _det(db, 1)
    b1 = _det(db, 1, x=0.101)
    a2 = _det(db, 2)
    _det(db, 2, x=0.101)

    def fail(n):
        if n == 1:
            raise _MySQLError(errno)

    _failing_deletes(db, monkeypatch, fail)
    out = dd.run(db, bk)
    monkeypatch.undo()
    assert _ids(db) == {a1, a2} and len(out["deleted"]) == 2
    assert capsys.readouterr().out.count(f"deleted id={b1} ") == 1


def test_deadlock_twice_aborts_with_earlier_sets_committed(db, bk, monkeypatch):
    a1 = _det(db, 1)
    _det(db, 1, x=0.101)
    a2 = _det(db, 2)
    b2 = _det(db, 2, x=0.101)

    def fail(n):
        if n >= 2:
            raise _MySQLError(1213)

    n = _failing_deletes(db, monkeypatch, fail)
    with pytest.raises(_MySQLError):
        dd.run(db, bk)
    monkeypatch.undo()
    assert n[0] == 3 and _ids(db) == {a1, a2, b2}


def test_backup_holds_the_full_rows_and_the_cluster_title(db, bk, tmp_path):
    named, plain = _cluster(db, "Alice"), _cluster(db)
    _det(db, 2, cluster_id=named)  # Alice keeps a face elsewhere
    kept = _det(db, 1, cluster_id=named)
    dup = _det(db, 1, x=0.101, cluster_id=plain, threshold=0.3)
    dd.run(db, bk)
    assert _backup(tmp_path) == [{
        "row": {"id": dup, "file_id": 1, "user_id": "alice", "x": 0.101, "y": 0.1, "height": W, "width": W,
                "face_vector": "[1.0]", "cluster_id": plain, "threshold": 0.3},
        "cluster_title": "", "kept_id": kept}]


def test_restore_round_trip(db, cfg, tmp_path, monkeypatch, capsys):
    named, plain = _cluster(db, "Alice"), _cluster(db)
    _det(db, 1, cluster_id=named)
    gone = _det(db, 1, x=0.101, cluster_id=plain)
    _det(db, 1, x=0.102)
    _det(db, 2, threshold=0.4)
    _det(db, 2, x=0.101)
    before = _dump(db)
    (tmp_path / "nc.json").write_text(json.dumps(cfg))
    bkp = str(tmp_path / "out" / "backup.jsonl")
    os.mkdir(tmp_path / "out")
    monkeypatch.setattr(sys, "argv", ["dedupe_faces.py", "--apply", "--backup", bkp, "--config",
                                      str(tmp_path / "nc.json")])
    assert dd.main() == 0
    assert len(_dump(db)) == 2  # 3 deleted
    monkeypatch.setattr(sys, "argv", ["dedupe_faces.py", "--restore", bkp, "--config", str(tmp_path / "nc.json")])
    assert dd.main() == 0
    assert _dump(db) == before
    assert "restore: 3 rows re-inserted, 0 skipped" in capsys.readouterr().out
    assert dd.main() == 0  # every id present again: skipped, nothing changed
    assert _dump(db) == before
    assert "restore: 0 rows re-inserted, 3 skipped as present" in capsys.readouterr().out
    # A cluster deleted since: its detection comes back unassigned, not pointing at nothing.
    db._execute("DELETE FROM oc_recognize_face_detections WHERE id = ?", (gone,))
    db._execute("DELETE FROM oc_recognize_face_clusters WHERE id = ?", (plain,))
    db._commit()
    assert dd.main() == 0
    assert db._execute("SELECT cluster_id FROM oc_recognize_face_detections WHERE id = ?",
                       (gone,)).fetchone()[0] == -1


def test_restore_retried_set_backed_up_twice(db, bk, tmp_path, monkeypatch):
    _det(db, 1)
    _det(db, 1, x=0.101)
    before = _dump(db)

    def fail(n):
        if n == 1:
            raise _MySQLError(1213)  # after the backup write: the retry writes the row again

    _failing_deletes(db, monkeypatch, fail)
    dd.run(db, bk)
    monkeypatch.undo()
    assert len(_backup(tmp_path)) == 2
    assert dd.restore(db, str(tmp_path / "backup.jsonl"))[:2] == ([2], [])
    assert _dump(db) == before


def test_apply_needs_a_backup(cfg, tmp_path, monkeypatch):
    (tmp_path / "nc.json").write_text(json.dumps(cfg))
    monkeypatch.setattr(sys, "argv", ["dedupe_faces.py", "--apply", "--config", str(tmp_path / "nc.json")])
    with pytest.raises(SystemExit) as e:
        dd.main()
    assert e.value.code == 2


def test_apply_refused_in_maintenance_mode(cfg, db, tmp_path, monkeypatch, capsys):
    a = _det(db, 1)
    _det(db, 1, x=0.101)
    before = _dump(db)
    (tmp_path / "config.php").write_text("<?php\n$CONFIG = array (\n  'maintenance' => true,\n);\n")
    (tmp_path / "nc.json").write_text(json.dumps(cfg))
    monkeypatch.setattr(sys, "argv", ["dedupe_faces.py", "--apply", "--backup", str(tmp_path / "b.jsonl"),
                                      "--config", str(tmp_path / "nc.json")])
    assert dd.main() == 1
    assert _dump(db) == before
    (tmp_path / "config.php").write_text("<?php\n$CONFIG = array (\n  'maintenance' => false,\n);\n")
    assert dd.main() == 0
    assert a in _ids(db) and len(_ids(db)) == 1
    assert "applied: deleted 1 duplicate detections in 1 files of 1 users" in capsys.readouterr().out


def test_restore_takes_the_last_backup_of_an_id(db, bk, tmp_path):
    # A retry re-backs up the row it then deletes; an earlier attempt's copy may be stale.
    _det(db, 1)
    _det(db, 1, x=0.101)
    before = _dump(db)
    dd.run(db, bk)
    bk.flush()
    (line,) = _backup(tmp_path)
    stale = json.loads(json.dumps(line))
    stale["row"]["face_vector"] = "[9.9]"
    stale["row"]["cluster_id"] = -1
    path = tmp_path / "backup.jsonl"
    path.write_text(json.dumps(stale) + "\n" + json.dumps(line) + "\n", encoding="utf-8")
    assert dd.restore(db, str(path))[:2] == ([2], [])
    assert _dump(db) == before


def test_a_cut_off_backup_line_is_skipped_and_ended(db, tmp_path, capsys):
    _det(db, 1)
    _det(db, 1, x=0.101)
    before = _dump(db)
    path = tmp_path / "backup.jsonl"
    path.write_text('{"row": {"id": 99, "file_', encoding="utf-8")  # a crash mid-write
    with dd._open_backup(str(path)) as f:  # must end the cut-off line before appending
        dd.run(db, f)
    assert dd.restore(db, str(path))[:2] == ([2], [])
    assert _dump(db) == before
    assert "unreadable line skipped" in capsys.readouterr().err
