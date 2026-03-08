"""Load and validate Nextcloud configuration for the recognize service."""

import json
import os
import re
import sys

REQUIRED_KEYS = [
    "dbtype",
    "dbhost",
    "dbuser",
    "dbpassword",
    "dbname",
    "dbtableprefix",
    "datadirectory",
]

VALID_DBTYPES = {"mysql", "pgsql", "sqlite3"}
PREFIX_RE = re.compile(r"^[a-zA-Z0-9_]*$")


class ConfigError(Exception):
    """Raised when configuration is invalid or missing."""


def load(path: str | None = None) -> dict:
    """Load nc_config.json and validate required fields.

    Args:
        path: Explicit path to nc_config.json.
              Defaults to <script_dir>/nc_config.json.

    Returns:
        Validated configuration dict.

    Raises:
        ConfigError: If file is missing, unreadable, or has invalid content.
    """
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nc_config.json")

    if not os.path.isfile(path):
        raise ConfigError(f"Config file not found: {path}")

    try:
        with open(path) as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in {path}: {e}") from e
    except PermissionError:
        raise ConfigError(f"Cannot read {path}: permission denied")

    validate(config)
    return config


def validate(config: dict) -> None:
    """Validate configuration dict.

    Raises:
        ConfigError: If required keys are missing or values are invalid.
    """
    missing = [k for k in REQUIRED_KEYS if k not in config]
    if missing:
        raise ConfigError(f"Missing required config keys: {', '.join(missing)}")

    dbtype = config["dbtype"]
    if dbtype not in VALID_DBTYPES:
        raise ConfigError(
            f"Unsupported dbtype '{dbtype}'. Must be one of: {', '.join(sorted(VALID_DBTYPES))}"
        )

    prefix = config["dbtableprefix"]
    if not PREFIX_RE.match(prefix):
        raise ConfigError(
            f"Invalid dbtableprefix '{prefix}': must match [a-zA-Z0-9_]"
        )

    if dbtype != "sqlite3":
        for key in ("dbhost", "dbuser", "dbpassword"):
            if not config.get(key):
                raise ConfigError(f"'{key}' is required for dbtype '{dbtype}'")

    datadir = config["datadirectory"]
    if not os.path.isabs(datadir):
        raise ConfigError(f"datadirectory must be an absolute path, got: {datadir}")


def get_models_dir(config: dict) -> str:
    """Return the path to the models directory.

    Uses 'models_dir' from config if set, otherwise defaults to
    <script_dir>/../models/.
    """
    if "models_dir" in config:
        return config["models_dir"]
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


def get_nextcloud_url(config: dict) -> str:
    """Return the Nextcloud base URL for internal API calls."""
    url = config.get("nextcloud_url", "")
    if not url:
        raise ConfigError("'nextcloud_url' is required in config")
    return url.rstrip("/")


def get_internal_secret(config: dict) -> str:
    """Return the shared secret for authenticating to the PHP internal API."""
    secret = config.get("internal_secret", "")
    if not secret:
        raise ConfigError("'internal_secret' is required in config")
    return secret


def get_nvidia_lib_path() -> str:
    """Build LD_LIBRARY_PATH entries for NVIDIA pip packages.

    ONNX Runtime GPU needs CUDA 12 libs from the nvidia-* pip packages.
    Returns a colon-separated path string.
    """
    site_packages = None
    for p in sys.path:
        candidate = os.path.join(p, "nvidia")
        if os.path.isdir(candidate):
            site_packages = candidate
            break

    if site_packages is None:
        return ""

    lib_dirs = []
    for root, dirs, _files in os.walk(site_packages):
        if os.path.basename(root) == "lib" and "nvidia" in root:
            lib_dirs.append(root)

    return ":".join(lib_dirs)
