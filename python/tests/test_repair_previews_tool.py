"""tools/repair_previews.py: drop preview rows whose file is gone, keep the rest."""

import importlib.util
import json
import os
import sys

import pytest

from db import DB

_spec = importlib.util.spec_from_file_location(
    "repair_previews", os.path.join(os.path.dirname(__file__), "..", "..", "tools", "repair_previews.py"))
rp = importlib.util.module_from_spec(_spec)
sys.modules["repair_previews"] = rp
_spec.loader.exec_module(rp)

COLS = "id, file_id, width, height, cropped, max, version_id, mtime, mimetype_id"


@pytest.fixture
def env(tmp_path):
    data = tmp_path / "data"
    root = data / "appdata_oc123" / "preview"
    root.mkdir(parents=True)
    cfg = {"dbtype": "sqlite3", "dbhost": "", "dbuser": "", "dbpassword": "", "dbname": str(tmp_path / "t.db"),
           "dbtableprefix": "oc_", "datadirectory": str(data)}
    (tmp_path / "nc_config.json").write_text(json.dumps(cfg))
    db = DB(cfg)
    db._execute("CREATE TABLE oc_mimetypes (id INTEGER PRIMARY KEY, mimetype TEXT)")
    db._execute(f"CREATE TABLE oc_previews ({COLS.replace(',', ' INTEGER,')} INTEGER)")
    db._execute("INSERT INTO oc_mimetypes VALUES (12, 'image/jpeg')")
    db._commit()

    def add(pid, fid, w, h, crop=0, mx=0, on_disk=True):
        db._execute(f"INSERT INTO oc_previews ({COLS}) VALUES (?, ?, ?, ?, ?, ?, -1, 1790538124, 12)",
                    (pid, fid, w, h, crop, mx))
        db._commit()
        path = rp.preview_path(str(root), fid, rp.preview_name(
            {"version_id": -1, "width": w, "height": h, "cropped": crop, "max": mx, "mimetype": "image/jpeg"}))
        if on_disk:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "wb").write(b"jpg")
        return path

    def run(*args):
        sys.argv = ["repair_previews.py", "--config", str(tmp_path / "nc_config.json"), *args]
        return rp.main()

    def ids():
        return sorted(r[0] for r in db._execute("SELECT id FROM oc_previews").fetchall())

    yield add, run, ids
    db.close()


def test_dry_run_changes_nothing(env, capsys):
    add, run, ids = env
    add(1, 827446, 2048, 1536, mx=1)
    add(2, 827446, 1024, 768, on_disk=False)
    assert run() == 0
    assert ids() == [1, 2]
    assert "1 missing in 1 files" in capsys.readouterr().out


def test_apply_deletes_only_missing_rows(env, capsys):
    # The 2026-11-06 case: max and crops on disk, the plain 64/256/1024 rows without files.
    add, run, ids = env
    keep = [add(1, 827446, 2048, 1536, mx=1), add(2, 827446, 1024, 1024, crop=1), add(3, 827446, 85, 64)]
    add(4, 827446, 64, 48, on_disk=False)
    add(5, 827446, 1024, 768, on_disk=False)
    add(6, 9, 256, 192)  # another file, intact
    assert run("--apply") == 0
    assert ids() == [1, 2, 3, 6]
    assert all(os.path.isfile(p) for p in keep)
    assert "2026-09  2" in capsys.readouterr().out


def test_apply_resets_a_file_whose_max_is_gone(env):
    add, run, ids = env
    add(1, 5, 2048, 1536, mx=1, on_disk=False)
    derived = add(2, 5, 1024, 768)
    add(3, 6, 1024, 768)
    assert run("--apply") == 0
    assert ids() == [3]
    assert not os.path.exists(derived)  # its file goes too: regenerated from the new max


def test_fileid_filter(env, capsys):
    add, run, ids = env
    add(1, 5, 64, 48, on_disk=False)
    add(2, 6, 64, 48, on_disk=False)
    assert run("--apply", "--fileid", "6") == 0
    assert ids() == [1]
    assert "file 6: 1 missing (64-48.jpg)" in capsys.readouterr().out
