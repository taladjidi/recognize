#!/bin/bash
set -euo pipefail

APP_DIR="/usr/share/nextcloud/apps/recognize"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
OCC="php /usr/share/nextcloud/occ"

echo "=== Stopping recognize:classify ==="
pkill -f 'php /usr/share/nextcloud/occ recognize:classify' || true
sleep 1
pkill -9 -f 'classifier_.*\.py' || true
echo "Done"

echo "=== Deploying Python files ==="
cp -v "$SRC_DIR"/python/gpu_setup.py "$APP_DIR/python/"
cp -v "$SRC_DIR"/python/base_classifier.py "$APP_DIR/python/"
cp -v "$SRC_DIR"/python/classifier_movinet.py "$APP_DIR/python/"
cp -v "$SRC_DIR"/python/classifier_musicnn.py "$APP_DIR/python/"
cp -v "$SRC_DIR"/python/classifier_faces.py "$APP_DIR/python/"
cp -v "$SRC_DIR"/python/classifier_imagenet.py "$APP_DIR/python/"
cp -v "$SRC_DIR"/python/classifier_landmarks.py "$APP_DIR/python/"
cp -v "$SRC_DIR"/python/classifier_geo.py "$APP_DIR/python/"
echo "Done"

echo "=== Deploying PHP files ==="
cp -v "$SRC_DIR"/lib/Service/FaceClusterAnalyzer.php "$APP_DIR/lib/Service/"
echo "Done"

echo "=== Setting MoViNet batch size to 200 ==="
sudo -u apache $OCC config:app:set recognize movinet.batchSize --value=200
echo "Done"

echo "=== Resetting all face detections ==="
sudo -u apache $OCC recognize:reset-face-clusters
mysql nextcloud -e "DELETE FROM oc_recognize_face_detections;"
echo "Done"

echo "=== Restarting classify ==="
echo "Run in tmux: sudo -u apache $OCC recognize:classify"
echo "After faces are re-scanned, run: sudo -u apache $OCC recognize:cluster-faces"
