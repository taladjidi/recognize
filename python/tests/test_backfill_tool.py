"""tools/backfill_crops.py: path-ordered file tasks, covers-only, per-file worker, Ctrl-C."""

import importlib.util
import os
import sys
from unittest.mock import patch

from PIL import Image

from crops import crop_path

_spec = importlib.util.spec_from_file_location(
    "backfill_crops", os.path.join(os.path.dirname(__file__), "..", "..", "tools", "backfill_crops.py"))
bf = importlib.util.module_from_spec(_spec)
sys.modules["backfill_crops"] = bf
_spec.loader.exec_module(bf)


def _box(i, fid, x=0.4):
    return {"id": i, "file_id": fid, "user_id": "u", "x": x, "y": 0.4, "width": 0.2, "height": 0.2}


class FakeDb:
    def __init__(self, rows, paths, cover_ids):
        self.rows, self.paths, self.cover_ids = rows, paths, cover_ids
        self.calls = []

    def detections_after(self, after, limit, covers_only=False):
        self.calls.append(covers_only)
        pool = [r for r in self.rows if r["id"] > after and (not covers_only or r["id"] in self.cover_ids)]
        return pool[:limit]

    def resolve_file_paths(self, ids):
        return {i: {"path": self.paths[i]} for i in ids if i in self.paths}


def _setup(tmp_path, n=3):
    paths = {}
    for fid in range(1, n + 1):
        # names reversed vs ids, so path order differs from id order
        p = tmp_path / "src" / f"{n - fid}.jpg"
        p.parent.mkdir(exist_ok=True)
        Image.new("RGB", (400, 300), (200, 10, 10)).save(p)
        paths[fid] = str(p)
    rows = [_box(10 * f + k, f, 0.2 + 0.3 * k) for f in range(1, n + 1) for k in range(2)]
    return rows, paths


def test_slices_sorted_by_path_and_skip_existing(tmp_path):
    rows, paths = _setup(tmp_path)
    root = str(tmp_path / "crops")
    os.makedirs(os.path.join(root, "u"))
    open(crop_path(root, "u", 10), "wb").write(b"x" * 10)  # one of file 1's two already exists
    db = FakeDb(rows, paths, set())
    batches = list(bf.plan_slices(db, root, False, False))
    flat = [b for batch in batches for b in batch]
    assert [p for p, _ in batches[-1]] == sorted(p for p, _ in batches[-1])
    assert sum(len(rs) for _, rs in flat) == 5
    flat = [b for batch in bf.plan_slices(db, root, True, False) for b in batch]  # force: nothing filtered
    assert sum(len(rs) for _, rs in flat) == 6


def test_covers_only_stops_after_phase_one(tmp_path):
    rows, paths = _setup(tmp_path)
    db = FakeDb(rows, paths, {10, 20})
    root = str(tmp_path / "crops")
    os.makedirs(root)
    ids = [r["id"] for b in bf.plan_slices(db, root, False, True) for _, rs in b for r in rs]
    assert sorted(ids) == [10, 20]
    assert set(db.calls) == {True}
    ids = [r["id"] for b in bf.plan_slices(db, root, True, False) for _, rs in b for r in rs]
    assert sorted(ids) == [10, 11, 20, 21, 30, 31]  # covers phase, then the rest, no repeats under --force


def test_worker_crops_all_faces_of_a_file(tmp_path):
    rows, paths = _setup(tmp_path, 1)
    root = str(tmp_path / "crops")
    os.makedirs(os.path.join(root, "u"))
    n, size, *times = bf.crop_file((paths[1], rows, root, False))
    assert n == 2 and size > 0 and len(times) == 3
    assert all(os.path.getsize(crop_path(root, "u", r["id"])) > 0 for r in rows)
    assert bf.crop_file((paths[1], rows, root, False))[0] == 0  # no overwrite without force
    assert bf.crop_file((paths[1], rows, root, True))[0] == 2


def test_run_end_to_end_and_ctrl_c(tmp_path, capsys):
    rows, paths = _setup(tmp_path)
    root = str(tmp_path / "crops")
    os.makedirs(os.path.join(root, "u"))
    assert bf.run(FakeDb(rows, paths, set()), root, workers=2) == 0
    assert "files/s" in capsys.readouterr().out
    assert len(os.listdir(os.path.join(root, "u"))) == 6
    with patch.object(bf, "plan_slices", side_effect=KeyboardInterrupt):
        assert bf.run(FakeDb(rows, paths, set()), root, force=True, workers=2) == 130


def test_slices_survive_two_files_with_the_same_path(tmp_path):
    rows, paths = _setup(tmp_path, 2)
    paths[2] = paths[1]
    assert list(bf.plan_slices(FakeDb(rows, paths, set()), str(tmp_path / "c"), True, False))


def _killer_for(real):
    def crop_file(task):
        if task[0].endswith("/0.jpg"):
            os._exit(9)  # like the OOM killer
        return real(task)
    crop_file.__qualname__, crop_file.__module__ = "crop_file", "backfill_crops"  # picklable by reference
    return crop_file


def test_run_skips_a_task_lost_to_a_killed_worker(tmp_path, capsys):
    rows, paths = _setup(tmp_path)
    root = str(tmp_path / "crops")
    os.makedirs(os.path.join(root, "u"))
    with patch.object(bf, "crop_file", _killer_for(bf.crop_file)), patch.object(bf, "TASK_TIMEOUT", 3):
        assert bf.run(FakeDb(rows, paths, set()), root, workers=2) == 0
    assert len(os.listdir(os.path.join(root, "u"))) == 4
