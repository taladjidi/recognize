#!/bin/bash
# Download the CI test fixtures used by ClassifierTest.php.
# Run from the repo root: bash python/tests/download_fixtures.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RES_DIR="$REPO_ROOT/tests/res"

mkdir -p "$RES_DIR"
cd "$RES_DIR"

if [ -f alpine.JPG ] && [ -f Rock_Rejam.mp3 ] && [ -f jumpingjack.gif ]; then
    echo "Test fixtures already present in $RES_DIR"
    exit 0
fi

echo "Downloading test fixtures..."
wget -q https://github.com/nextcloud/recognize/releases/download/v3.4.0/test-files.zip
unzip -o test-files.zip
rm test-files.zip
echo "Done. Fixtures in $RES_DIR"
