"""Tests for config module."""

import json
import os
import tempfile

import pytest

from config import ConfigError, load, validate, get_models_dir, get_nvidia_lib_path


def _write_config(tmp_path, data):
    """Helper: write a config dict to a temp JSON file, return path."""
    path = os.path.join(tmp_path, "nc_config.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


@pytest.fixture
def valid_config():
    return {
        "dbtype": "mysql",
        "dbhost": "localhost",
        "dbport": 3306,
        "dbuser": "nextcloud",
        "dbpassword": "secret",
        "dbname": "nextcloud",
        "dbtableprefix": "oc_",
        "datadirectory": "/srv/nextcloud/data",
        "nextcloud_url": "http://localhost:8080",
        "internal_secret": "abc123",
    }


class TestValidate:
    def test_valid_mysql(self, valid_config):
        validate(valid_config)  # should not raise

    def test_valid_pgsql(self, valid_config):
        valid_config["dbtype"] = "pgsql"
        validate(valid_config)

    def test_valid_sqlite3(self, valid_config):
        valid_config["dbtype"] = "sqlite3"
        # sqlite doesn't require host/user/password
        valid_config["dbhost"] = ""
        valid_config["dbuser"] = ""
        valid_config["dbpassword"] = ""
        validate(valid_config)

    def test_missing_required_key(self, valid_config):
        del valid_config["dbname"]
        with pytest.raises(ConfigError, match="Missing required.*dbname"):
            validate(valid_config)

    def test_invalid_dbtype(self, valid_config):
        valid_config["dbtype"] = "oracle"
        with pytest.raises(ConfigError, match="Unsupported dbtype"):
            validate(valid_config)

    def test_invalid_prefix_sql_injection(self, valid_config):
        valid_config["dbtableprefix"] = "oc_; DROP TABLE users; --"
        with pytest.raises(ConfigError, match="Invalid dbtableprefix"):
            validate(valid_config)

    def test_empty_prefix_is_valid(self, valid_config):
        valid_config["dbtableprefix"] = ""
        validate(valid_config)  # Nextcloud allows empty prefix

    def test_relative_datadirectory(self, valid_config):
        valid_config["datadirectory"] = "data/"
        with pytest.raises(ConfigError, match="absolute path"):
            validate(valid_config)

    def test_mysql_requires_dbhost(self, valid_config):
        valid_config["dbhost"] = ""
        with pytest.raises(ConfigError, match="dbhost.*required"):
            validate(valid_config)


class TestLoad:
    def test_load_valid_file(self, tmp_path, valid_config):
        path = _write_config(str(tmp_path), valid_config)
        config = load(path)
        assert config["dbtype"] == "mysql"
        assert config["dbname"] == "nextcloud"

    def test_file_not_found(self):
        with pytest.raises(ConfigError, match="not found"):
            load("/nonexistent/path/nc_config.json")

    def test_invalid_json(self, tmp_path):
        path = os.path.join(str(tmp_path), "nc_config.json")
        with open(path, "w") as f:
            f.write("{invalid json")
        with pytest.raises(ConfigError, match="Invalid JSON"):
            load(path)

    def test_permission_denied(self, tmp_path):
        path = _write_config(str(tmp_path), {"dbtype": "mysql"})
        os.chmod(path, 0o000)
        try:
            with pytest.raises(ConfigError, match="permission denied"):
                load(path)
        finally:
            os.chmod(path, 0o644)  # restore for cleanup


class TestGetModelsDir:
    def test_default_path(self, valid_config):
        result = get_models_dir(valid_config)
        assert result.endswith("models")

    def test_custom_path(self, valid_config):
        valid_config["models_dir"] = "/opt/models"
        assert get_models_dir(valid_config) == "/opt/models"


class TestGetNvidiaLibPath:
    def test_returns_string(self):
        result = get_nvidia_lib_path()
        assert isinstance(result, str)
        # If nvidia packages are installed, should contain paths
        # If not, empty string is valid
