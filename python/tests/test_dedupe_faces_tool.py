"""tools/dedupe_faces.py: remove near-duplicate face detections, keeping names and user edits."""

import importlib.util
import json
import os
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


def _cluster(db, title="", user="alice"):
    return db.create_face_cluster(user, title)


def _det(db, file_id, x=0.1, user="alice", cluster_id=None, threshold=0.0):
    cur = db._execute("INSERT INTO oc_recognize_face_detections (file_id, user_id, x, y, width, height, "
                      "face_vector, cluster_id, threshold) VALUES (?, ?, ?, 0.1, ?, ?, '[1.0]', ?, ?)",
                      (file_id, user, x, W, W, cluster_id, threshold))
    db._commit()
    return cur.lastrowid


def _dump(db):
    return [tuple(r) for r in db._execute("SELECT * FROM oc_recognize_face_detections ORDER BY id").fetchall()]


def _ids(db):
    return {r[0] for r in _dump(db)}


def test_iou_boundary_089_kept_091_removed(db):
    assert dd.iou({"x": 0, "y": 0, "width": W, "height": W},
                  {"x": _dx(0.91), "y": 0, "width": W, "height": W}) == pytest.approx(0.91)
    a = _det(db, 1)
    b = _det(db, 1, x=0.1 + _dx(0.91))
    c = _det(db, 2)
    d = _det(db, 2, x=0.1 + _dx(0.89))
    out = dd.run(db, apply=True)
    assert _ids(db) == {a, c, d}
    assert [(r["id"], k["id"]) for r, k in out["deleted"]] == [(b, a)]


def test_keep_priority_titled_then_clustered_then_removed_then_oldest(db):
    named, plain = _cluster(db, "Alice"), _cluster(db)
    f1 = [_det(db, 1), _det(db, 1, cluster_id=plain), _det(db, 1, cluster_id=named)]
    f2 = [_det(db, 2), _det(db, 2, cluster_id=plain)]
    f3 = [_det(db, 3, cluster_id=-1), _det(db, 3)]
    f4 = [_det(db, 4), _det(db, 4, threshold=0.5)]  # a user removal keeps its threshold
    dd.run(db, apply=True)
    assert _ids(db) == {f1[2], f2[1], f3[0], f4[1]}


def test_two_different_titled_clusters_untouched_and_reported(db):
    alice, bob = _cluster(db, "Alice"), _cluster(db, "Bob")
    ids = {_det(db, 1, cluster_id=alice), _det(db, 1, cluster_id=bob), _det(db, 1)}
    _det(db, 2, cluster_id=alice)  # another face, so neither cluster would empty
    _det(db, 2, x=0.6, cluster_id=bob)
    before = _dump(db)
    out = dd.run(db, apply=True)
    assert _dump(db) == before
    assert [{r["id"] for r in s} for s in out["conflicts"]] == [ids]
    assert {r["title"] for r in out["conflicts"][0]} == {"Alice", "Bob", ""}


def test_last_face_of_a_titled_cluster_never_deleted(db, monkeypatch):
    alice, bob = _cluster(db, "Alice"), _cluster(db, "Bob")
    only = _det(db, 1, cluster_id=alice)  # Alice's only face
    other = _det(db, 1)
    b1 = _det(db, 2, cluster_id=bob)
    b2 = _det(db, 2, cluster_id=bob)  # same face twice in Bob: one may go
    real = dd.plan
    # A planner bug that keeps the wrong row: the guard must still hold.
    monkeypatch.setattr(dd, "plan", lambda rows: ([(r, k) for k, r in real(rows)[0]], []))
    out = dd.run(db, apply=True)
    assert _ids(db) == {only, other, b2}
    assert [r["id"] for r in out["protected"]] == [only]
    assert b1 not in _ids(db)


def test_cross_user_and_cross_file_never_matched(db):
    before = {_det(db, 1), _det(db, 1, user="bob"), _det(db, 2)}
    out = dd.run(db, apply=True)
    assert _ids(db) == before and not out["deleted"]


def test_dry_run_writes_nothing_and_reports_the_same(db, monkeypatch):
    a = _det(db, 1)
    b = _det(db, 1, x=0.101)
    before = _dump(db)
    real, sqls = db._execute, []
    monkeypatch.setattr(db, "_execute", lambda sql, params=(): sqls.append(sql) or real(sql, params))
    out = dd.run(db, apply=False)
    monkeypatch.undo()
    assert sqls and all(s.lstrip().startswith("SELECT") for s in sqls)  # not even a rolled-back DELETE
    assert _dump(db) == before
    assert [(r["id"], k["id"]) for r, k in out["deleted"]] == [(b, a)]
    other = DB(db._config)  # no open write transaction left behind
    other._execute("DELETE FROM oc_recognize_face_detections WHERE id = ?", (a,))
    other.rollback()
    other.close()


def test_crops_left_to_the_daemon_which_removes_the_deleted_ones(db, tmp_path):
    root = tmp_path / "crops"
    (root / "alice").mkdir(parents=True)
    a = _det(db, 1)
    b = _det(db, 1, x=0.101)
    c = _det(db, 2)  # newest id: the daemon only cleans ids up to the max detection id
    for i in (a, b, c):
        (root / "alice" / f"{i}.jpg").write_bytes(b"jpg")
    dd.run(db, apply=True)
    assert sorted(os.listdir(root / "alice")) == sorted(f"{i}.jpg" for i in (a, b, c))  # the tool leaves them
    Service._cleanup_crops_one_user(SimpleNamespace(crop_root=str(root), _crop_users=[]), db)
    assert sorted(os.listdir(root / "alice")) == sorted(f"{i}.jpg" for i in (a, c))


def test_second_run_is_a_no_op(db):
    named = _cluster(db, "Alice")
    _det(db, 1)
    _det(db, 1, x=0.101, cluster_id=named)
    _det(db, 1, x=0.102)
    assert len(dd.run(db, apply=True)["deleted"]) == 2
    after = _dump(db)
    out = dd.run(db, apply=True)
    assert _dump(db) == after and not out["deleted"] and not out["conflicts"]


def test_rows_changed_since_the_scan_are_re_planned(db, monkeypatch):
    alice, plain = _cluster(db, "Alice"), _cluster(db)
    ids = {_det(db, 1, cluster_id=alice), _det(db, 1, cluster_id=plain)}
    real = dd._groups

    def groups_then_user_names_it(rows):
        g = real(rows)
        db._execute("UPDATE oc_recognize_face_clusters SET title = 'Bob' WHERE id = ?", (plain,))
        db._commit()
        return g

    monkeypatch.setattr(dd, "_groups", groups_then_user_names_it)
    out = dd.run(db, apply=True)
    assert _ids(db) == ids and not out["deleted"] and len(out["conflicts"]) == 1


def test_db_error_rolls_back_the_whole_page_and_aborts(db, monkeypatch):
    a1 = _det(db, 1)
    _det(db, 1, x=0.101)
    _det(db, 2)
    _det(db, 2, x=0.101)
    before = _dump(db)
    real, deletes = db._execute, []

    def failing(sql, params=()):
        if sql.startswith("DELETE"):
            deletes.append(params)
            if len(deletes) == 2:
                raise RuntimeError("disk I/O error")
        return real(sql, params)

    monkeypatch.setattr(db, "_execute", failing)
    with pytest.raises(RuntimeError):
        dd.run(db, apply=True)
    monkeypatch.undo()
    assert len(deletes) == 2 and _dump(db) == before and a1 in _ids(db)


def test_apply_refused_in_maintenance_mode(cfg, db, tmp_path, monkeypatch, capsys):
    a = _det(db, 1)
    _det(db, 1, x=0.101)
    before = _dump(db)
    (tmp_path / "config.php").write_text("<?php\n$CONFIG = array (\n  'maintenance' => true,\n);\n")
    (tmp_path / "nc.json").write_text(json.dumps(cfg))
    monkeypatch.setattr(sys, "argv", ["dedupe_faces.py", "--apply", "--config", str(tmp_path / "nc.json")])
    assert dd.main() == 1
    assert _dump(db) == before
    (tmp_path / "config.php").write_text("<?php\n$CONFIG = array (\n  'maintenance' => false,\n);\n")
    assert dd.main() == 0
    assert a in _ids(db) and len(_ids(db)) == 1
    assert "applied: deleted 1 duplicate detections in 1 files of 1 users" in capsys.readouterr().out
