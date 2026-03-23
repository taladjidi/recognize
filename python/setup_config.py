#!/usr/bin/env python3
"""Generate nc_config.json and systemd service unit for the Recognize daemon.

Usage (run with appropriate permissions to read config.php):
    sudo python setup_config.py [--nextcloud-root /path/to/nextcloud]

Reads Nextcloud's config.php via PHP, extracts DB credentials, generates
a shared internal secret, and writes:
  1. nc_config.json — configuration for the Python service
  2. recognize.service — ready-to-install systemd unit file

The internal secret is stored in both nc_config.json and Nextcloud's
oc_appconfig table so the Python daemon and PHP controller can authenticate.
"""

import grp
import json
import os
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Python dependencies for the recognize service
BASE_DEPS = [
    'onnxruntime-gpu>=1.24',
    'numpy>=2.0',
    'pillow>=10.0',
    'opencv-python-headless>=4.8',
    'insightface>=0.7',
    'scikit-learn>=1.4',
    'requests>=2.31',
    'soundfile>=0.12',
    'scipy>=1.12',
]

DB_DEPS = [
    'mysql-connector-python>=8.0',
    # 'psycopg2-binary>=2.9',  # uncomment for PostgreSQL
]

NVIDIA_DEPS = [
    'nvidia-cuda-runtime-cu12',
    'nvidia-cublas-cu12',
    'nvidia-cudnn-cu12',
    'nvidia-cufft-cu12',
    'nvidia-curand-cu12',
    'nvidia-cusolver-cu12',
    'nvidia-cusparse-cu12',
]
NEXTCLOUD_ROOT = os.environ.get("NEXTCLOUD_ROOT", "/usr/share/nextcloud")
OUTPUT_CONFIG = os.path.join(SCRIPT_DIR, "nc_config.json")
SERVICE_TEMPLATE = os.path.join(SCRIPT_DIR, "recognize.service.template")
OUTPUT_SERVICE = os.path.join(SCRIPT_DIR, "recognize.service")

REQUIRED_KEYS = ["dbtype", "dbhost", "dbname", "dbuser", "dbpassword", "datadirectory"]


def read_nextcloud_config(nc_root):
    """Use PHP to parse config.php and return as dict."""
    config_php = os.path.join(nc_root, "config", "config.php")
    if not os.path.isfile(config_php):
        print(f"ERROR: {config_php} not found. Set NEXTCLOUD_ROOT env var.", file=sys.stderr)
        sys.exit(1)

    php_code = f"require('{config_php}'); echo json_encode($CONFIG);"
    try:
        result = subprocess.run(
            ["php", "-r", php_code],
            capture_output=True, text=True, timeout=10,
        )
    except FileNotFoundError:
        print("ERROR: php not found in PATH", file=sys.stderr)
        sys.exit(1)

    if result.returncode != 0:
        print(f"ERROR: php failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse PHP output: {e}", file=sys.stderr)
        sys.exit(1)


def generate_secret():
    """Generate a 64-char hex secret."""
    return os.urandom(32).hex()


def store_secret_in_nextcloud(nc_root, secret):
    """Store the internal secret in Nextcloud's appconfig via occ."""
    occ = os.path.join(nc_root, "occ")
    try:
        subprocess.run(
            ["php", occ, "config:app:set", "recognize", "internal_secret",
             "--value", secret],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"WARNING: Could not store secret via occ: {e}", file=sys.stderr)
        return False


def has_nvidia_gpu():
    """Check if an NVIDIA GPU is available."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0 and result.stdout.strip() != ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def setup_venv(venv_dir, source_python=None, force=False, no_gpu=False):
    """Create a Python venv and install all dependencies.

    Auto-detects NVIDIA GPU and installs CUDA pip packages when available.
    """
    python_bin = source_python or sys.executable

    if force and os.path.isdir(venv_dir):
        print(f"  Removing existing venv: {venv_dir}")
        shutil.rmtree(venv_dir)

    if os.path.isfile(os.path.join(venv_dir, "bin", "python")):
        print(f"  Venv already exists at {venv_dir}")
        print(f"  To force recreation: pass --force-venv or delete {venv_dir}")
        return os.path.join(venv_dir, "bin", "python")

    print(f"  Creating venv with --copies (self-contained)...")
    subprocess.run(
        [python_bin, "-m", "venv", "--copies", venv_dir],
        check=True,
    )

    pip_bin = os.path.join(venv_dir, "bin", "pip")
    deps = BASE_DEPS + DB_DEPS

    gpu = not no_gpu and has_nvidia_gpu()
    if gpu:
        print(f"  NVIDIA GPU detected — installing CUDA dependencies")
        deps += NVIDIA_DEPS
    else:
        print(f"  No NVIDIA GPU detected — CPU-only mode")

    print(f"  Installing {len(deps)} packages...")
    subprocess.run(
        [pip_bin, "install", "--no-cache-dir"] + deps,
        check=True,
    )

    return os.path.join(venv_dir, "bin", "python")


def get_nvidia_lib_path():
    """Build LD_LIBRARY_PATH for NVIDIA CUDA 12 libs from pip packages."""
    python_bin = sys.executable
    site_dir = os.path.join(
        os.path.dirname(python_bin), "..", "lib",
        f"python{sys.version_info[0]}.{sys.version_info[1]}",
        "site-packages", "nvidia",
    )
    site_dir = os.path.normpath(site_dir)

    if not os.path.isdir(site_dir):
        return ""

    lib_dirs = []
    for entry in os.listdir(site_dir):
        lib_path = os.path.join(site_dir, entry, "lib")
        if os.path.isdir(lib_path):
            lib_dirs.append(lib_path)

    return ":".join(lib_dirs)


def generate_service_file(config, nc_root, python_bin):
    """Generate systemd service unit from template."""
    if not os.path.isfile(SERVICE_TEMPLATE):
        print(f"WARNING: Template not found: {SERVICE_TEMPLATE}", file=sys.stderr)
        return None

    with open(SERVICE_TEMPLATE) as f:
        template = f.read()

    # Determine user/group (web server user)
    user = "apache"
    group = "apache"
    for candidate in ["apache", "www-data", "nginx", "http"]:
        try:
            grp.getgrnam(candidate)
            user = candidate
            group = candidate
            break
        except KeyError:
            continue

    nvidia_lib_path = get_nvidia_lib_path()
    models_dir = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "models"))
    data_dir = config.get("datadirectory", "/var/lib/nextcloud/data")

    replacements = {
        "__USER__": user,
        "__GROUP__": group,
        "__PYTHON_DIR__": SCRIPT_DIR,
        "__PYTHON_BIN__": python_bin,
        "__CONFIG_PATH__": OUTPUT_CONFIG,
        "__MODELS_DIR__": models_dir,
        "__FFMPEG_BIN__": "/usr/bin/ffmpeg",
        "__NVIDIA_LIB_PATH__": nvidia_lib_path,
        "__CUDA_DEVICES__": "0",
        "__DATA_DIR__": data_dir,
    }

    service = template
    for placeholder, value in replacements.items():
        service = service.replace(placeholder, value)

    return service


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate Recognize service configuration")
    parser.add_argument("--nextcloud-root", default=NEXTCLOUD_ROOT,
                        help=f"Nextcloud installation root (default: {NEXTCLOUD_ROOT})")
    parser.add_argument("--python-bin", default=sys.executable,
                        help=f"Python binary path (default: {sys.executable})")
    parser.add_argument("--setup-venv", metavar="VENV_DIR",
                        help="Create/update Python venv at VENV_DIR with all dependencies")
    parser.add_argument("--source-python", default=None,
                        help="Python binary to create the venv from (default: current interpreter)")
    parser.add_argument("--force-venv", action="store_true",
                        help="Delete and recreate the venv even if it exists")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Skip GPU detection, install CPU-only dependencies")
    args = parser.parse_args()

    # Optional: set up venv before anything else
    if args.setup_venv:
        print("=== Setting up Python venv ===")
        venv_python = setup_venv(
            args.setup_venv,
            source_python=args.source_python,
            force=args.force_venv,
            no_gpu=args.no_gpu,
        )
        print(f"  Venv Python: {venv_python}")
        # Use the venv python as the service python
        args.python_bin = venv_python
        print("Done\n")

    nc_root = args.nextcloud_root

    # 1. Read Nextcloud config
    print(f"Reading config from {nc_root}/config/config.php ...")
    nc_php_config = read_nextcloud_config(nc_root)

    missing = [k for k in REQUIRED_KEYS if k not in nc_php_config]
    if missing:
        print(f"ERROR: Missing config keys: {missing}", file=sys.stderr)
        sys.exit(1)

    # 2. Generate shared secret
    secret = generate_secret()

    # 3. Build nc_config.json
    nc_config = {
        "dbtype": nc_php_config["dbtype"],
        "dbhost": nc_php_config["dbhost"],
        "dbport": nc_php_config.get("dbport", ""),
        "dbname": nc_php_config["dbname"],
        "dbuser": nc_php_config["dbuser"],
        "dbpassword": nc_php_config["dbpassword"],
        "dbtableprefix": nc_php_config.get("dbtableprefix", "oc_"),
        "datadirectory": nc_php_config["datadirectory"],
        "nextcloud_url": nc_php_config.get("overwrite.cli.url", f"http://localhost/nextcloud"),
        "internal_secret": secret,
    }

    # 4. Write nc_config.json (0640 — readable by service user, not world)
    fd = os.open(OUTPUT_CONFIG, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o640)
    with os.fdopen(fd, "w") as f:
        json.dump(nc_config, f, indent=2)
        f.write("\n")

    print(f"  Config written to {OUTPUT_CONFIG}")
    print(f"  DB: {nc_config['dbtype']}://{nc_config['dbuser']}@{nc_config['dbhost']}/{nc_config['dbname']}")
    print(f"  Data dir: {nc_config['datadirectory']}")
    print(f"  NC URL: {nc_config['nextcloud_url']}")

    # 5. Store secret in Nextcloud appconfig (requires root or web-server user)
    print("\nStoring internal secret in Nextcloud appconfig...")
    print("  (requires root — runs: php occ config:app:set)")
    if store_secret_in_nextcloud(nc_root, secret):
        print("  Secret stored successfully")
    else:
        print("  WARNING: Could not store secret automatically.")
        print("  Run manually as root:")
        print(f"    sudo -u apache php {nc_root}/occ config:app:set recognize internal_secret --value {secret}")

    # 6. Generate systemd service file
    print("\nGenerating systemd service file...")
    service_content = generate_service_file(nc_config, nc_root, args.python_bin)
    if service_content:
        with open(OUTPUT_SERVICE, "w") as f:
            f.write(service_content)
        print(f"  Service file written to {OUTPUT_SERVICE}")

        nvidia_lib = get_nvidia_lib_path()
        print(f"\n  LD_LIBRARY_PATH = {nvidia_lib or '(none — CPU mode)'}")

        print(f"\n  To install (requires root):")
        print(f"    sudo cp {OUTPUT_SERVICE} /etc/systemd/system/recognize.service")
        print(f"    sudo systemctl daemon-reload")
        print(f"    sudo systemctl enable --now recognize")
    else:
        print("  Skipped (template not found)")

    print("\nSetup complete.")
    print("NOTE: If not running as root, the following steps require elevated privileges:")
    print("  - Writing config to app directory (step 4)")
    print("  - Storing secret via occ (step 5)")
    print("  - Installing systemd service (step 6)")


if __name__ == "__main__":
    main()
