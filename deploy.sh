#!/bin/bash
# Deploy Recognize Python ONNX service + simplified PHP FileListener.
# Run as root: sudo bash deploy.sh
set -euo pipefail

APP_DIR="/usr/share/nextcloud/apps/recognize"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
OCC="php /usr/share/nextcloud/occ"
VENV_DIR="$APP_DIR/python/.venv"

# Source Python used to create the venv (needs Python 3.13 with working pip)
SOURCE_PYTHON="/home/aladjidi/mambaforge/envs/313/bin/python"

echo "=== Step 1: Stop running processes ==="
systemctl stop recognize.service 2>/dev/null || true
pkill -f 'python.*service\.py' || true
pkill -f 'php /usr/share/nextcloud/occ recognize:classify' || true
sleep 1
echo "Done"

echo "=== Step 2: Create/update Python venv ==="
mkdir -p "$APP_DIR/python/classifiers" "$APP_DIR/python/data"

if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo "  Creating venv with --copies (self-contained, no symlinks to home dir)..."
    "$SOURCE_PYTHON" -m venv --copies "$VENV_DIR"
    echo "  Installing Python dependencies..."
    "$VENV_DIR/bin/pip" install --no-cache-dir \
        'onnxruntime-gpu>=1.24' \
        'numpy>=2.0' \
        'pillow>=10.0' \
        'opencv-python-headless>=4.8' \
        'insightface>=0.7' \
        'scikit-learn>=1.4' \
        'requests>=2.31' \
        'soundfile>=0.12' \
        'scipy>=1.12' \
        'mysql-connector-python>=8.0'
else
    echo "  Venv already exists at $VENV_DIR"
    echo "  To force recreation: rm -rf $VENV_DIR && re-run deploy.sh"
fi

PYTHON_BIN="$VENV_DIR/bin/python"
echo "  Python binary: $PYTHON_BIN"
echo "Done"

echo "=== Step 3: Deploy Python service files ==="
# Core service files
for f in service.py pipeline.py config.py db.py nc_api.py clustering.py \
         rules_engine.py setup_config.py \
         recognize.service.template; do
    [ -f "$SRC_DIR/python/$f" ] && cp -v "$SRC_DIR/python/$f" "$APP_DIR/python/"
done

# Classifier modules
for f in __init__.py base.py imagenet.py faces.py landmarks.py movinet.py musicnn.py; do
    [ -f "$SRC_DIR/python/classifiers/$f" ] && cp -v "$SRC_DIR/python/classifiers/$f" "$APP_DIR/python/classifiers/"
done

# Data files
for f in yamnet_class_map.csv; do
    [ -f "$SRC_DIR/python/data/$f" ] && cp -v "$SRC_DIR/python/data/$f" "$APP_DIR/python/data/"
done
echo "Done"

echo "=== Step 4: Deploy PHP + appinfo files ==="
rsync -av --delete "$SRC_DIR/lib/" "$APP_DIR/lib/"
rsync -av --delete "$SRC_DIR/appinfo/" "$APP_DIR/appinfo/"
echo "Done"

echo "=== Step 5: Fix ownership ==="
chown -R apache:apache "$APP_DIR/python/" "$APP_DIR/lib/" "$APP_DIR/appinfo/"
echo "Done"

echo "=== Step 6: Run DB migration (creates recognize_pending table) ==="
sudo -u apache $OCC maintenance:repair --include-expensive || true
echo "Done"

echo "=== Step 7: Clear old background jobs ==="
sudo -u apache $OCC background-job:delete 75207 2>/dev/null || true
# Remove all old ProcessFsActionsJob entries
sudo -u apache $OCC background-job:list --class 'OCA\Recognize\BackgroundJobs\ProcessFsActionsJob' 2>/dev/null | \
    grep -oP '^\s*\K\d+' | while read id; do
        echo "  Removing old job $id"
        sudo -u apache $OCC background-job:delete "$id" 2>/dev/null || true
    done || true
echo "Done"

echo "=== Step 8: Generate config + systemd unit ==="
"$PYTHON_BIN" "$APP_DIR/python/setup_config.py" --nextcloud-root /usr/share/nextcloud --python-bin "$PYTHON_BIN"
# Fix ownership of generated files
chown apache:apache "$APP_DIR/python/nc_config.json" 2>/dev/null || true
chown apache:apache "$APP_DIR/python/recognize.service" 2>/dev/null || true
echo "Done"

echo "=== Step 9: Install + start systemd service ==="
cp -v "$APP_DIR/python/recognize.service" /etc/systemd/system/recognize.service
systemctl daemon-reload
systemctl enable --now recognize
echo "Done"

echo ""
echo "=== Deployment complete ==="
echo "Check status:  systemctl status recognize"
echo "View logs:     journalctl -u recognize -f"
