"""Pytest configuration and shared fixtures.

Auto-discovers test media from tests/res/ (the same fixtures used by CI).
Download them with:
    cd tests/res && wget https://github.com/nextcloud/recognize/releases/download/v3.4.0/test-files.zip && unzip test-files.zip
"""

import os
import subprocess
import json
import sys

import pytest

# Paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PYTHON_DIR = os.path.join(REPO_ROOT, "python")
SRC_DIR = os.path.join(REPO_ROOT, "src")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
RES_DIR = os.path.join(REPO_ROOT, "tests", "res")


# ── Test media fixtures ──────────────────────────────────────────────


@pytest.fixture(scope="session")
def alpine_jpg():
    """Alpine mountain scene — expected: 'Alpine' tag."""
    path = os.path.join(RES_DIR, "alpine.JPG")
    if not os.path.isfile(path):
        pytest.skip("Test fixture missing: alpine.JPG (run download_fixtures.sh)")
    return path


@pytest.fixture(scope="session")
def eiffeltower_jpg():
    """Eiffel Tower photo — expected: landmark 'Eiffel Tower'."""
    path = os.path.join(RES_DIR, "eiffeltower.jpg")
    if not os.path.isfile(path):
        pytest.skip("Test fixture missing: eiffeltower.jpg")
    return path


@pytest.fixture(scope="session")
def food_jpg():
    """Food photo (casey-lee unsplash) — expected: food-related tags."""
    path = os.path.join(RES_DIR, "casey-lee-awj7sRviVXo-unsplash.jpg")
    if not os.path.isfile(path):
        pytest.skip("Test fixture missing: casey-lee-awj7sRviVXo-unsplash.jpg")
    return path


@pytest.fixture(scope="session")
def geotagged_jpg():
    """Image with GPS EXIF data."""
    path = os.path.join(RES_DIR, "geotagged.jpg")
    if not os.path.isfile(path):
        pytest.skip("Test fixture missing: geotagged.jpg")
    return path


@pytest.fixture(scope="session")
def geotagged_germany_jpg():
    """Image with GPS EXIF data (Germany)."""
    path = os.path.join(RES_DIR, "geotagged_germany.jpg")
    if not os.path.isfile(path):
        pytest.skip("Test fixture missing: geotagged_germany.jpg")
    return path


@pytest.fixture(scope="session")
def rock_mp3():
    """Rock/electronic music — expected: 'electronic' tag."""
    path = os.path.join(RES_DIR, "Rock_Rejam.mp3")
    if not os.path.isfile(path):
        pytest.skip("Test fixture missing: Rock_Rejam.mp3")
    return path


@pytest.fixture(scope="session")
def jumpingjack_gif():
    """Jumping jacks video — expected: 'jumping jacks' tag."""
    path = os.path.join(RES_DIR, "jumpingjack.gif")
    if not os.path.isfile(path):
        pytest.skip("Test fixture missing: jumpingjack.gif")
    return path


@pytest.fixture(scope="session")
def face_images():
    """Face dataset: dict of {person_name: [image_paths]}."""
    faceid_dir = os.path.join(RES_DIR, "FaceID-550")
    if not os.path.isdir(faceid_dir):
        pytest.skip("Test fixture missing: FaceID-550/")
    result = {}
    for person in sorted(os.listdir(faceid_dir)):
        person_dir = os.path.join(faceid_dir, person)
        if os.path.isdir(person_dir):
            images = sorted(
                [
                    os.path.join(person_dir, f)
                    for f in os.listdir(person_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
            )
            if images:
                result[person] = images
    return result


# ── Model availability fixtures ──────────────────────────────────────


@pytest.fixture(scope="session")
def has_imagenet_model():
    """Check if an EfficientNet SavedModel is available (native or TFJS-converted)."""
    native = os.path.join(MODELS_DIR, "efficientnetv2_native_saved", "saved_model.pb")
    tfjs = os.path.join(MODELS_DIR, "efficientnetv2_saved", "saved_model.pb")
    lite = os.path.join(MODELS_DIR, "efficientnet_lite4_saved", "saved_model.pb")
    if not (os.path.isfile(native) or os.path.isfile(tfjs) or os.path.isfile(lite)):
        pytest.skip(
            "No imagenet model found (run convert_models.py or download_native_v2xl.py)"
        )
    return True


@pytest.fixture(scope="session")
def has_landmark_models():
    """Check if all 6 landmark SavedModels are available."""
    regions = ["africa", "asia", "europe", "north_america", "south_america", "oceania"]
    for region in regions:
        path = os.path.join(MODELS_DIR, f"landmarks_{region}_saved", "saved_model.pb")
        if not os.path.isfile(path):
            pytest.skip(
                f"Model not converted: landmarks_{region}_saved (run convert_models.py)"
            )
    return True


@pytest.fixture(scope="session")
def has_movinet_model():
    """Check if MoViNet SavedModel is available (already in native format)."""
    path = os.path.join(MODELS_DIR, "movinet-a3", "saved_model.pb")
    if not os.path.isfile(path):
        pytest.skip("Model missing: movinet-a3")
    return True


@pytest.fixture(scope="session")
def has_musicnn_model():
    """Check if MusicNN SavedModel is available."""
    path = os.path.join(MODELS_DIR, "musicnn_saved", "saved_model.pb")
    if not os.path.isfile(path):
        pytest.skip("Model not converted: musicnn_saved (run convert_models.py)")
    return True


# ── Classifier runner helper ─────────────────────────────────────────


@pytest.fixture(scope="session")
def run_classifier():
    """Fixture that returns the run_python_classifier helper function."""
    return run_python_classifier


def run_python_classifier(classifier_name, input_path, timeout=120):
    """Run a Python classifier script and return parsed JSON output.

    Args:
        classifier_name: e.g. 'imagenet', 'landmarks', 'faces'
        input_path: path to a single test file
        timeout: max seconds to wait

    Returns:
        Parsed JSON result (list or dict), or None on failure.
    """
    script = os.path.join(PYTHON_DIR, f"classifier_{classifier_name}.py")
    python = sys.executable

    proc = subprocess.run(
        [python, script, input_path],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=PYTHON_DIR,
    )

    stderr_lines = proc.stderr.strip()
    if stderr_lines:
        # Print diagnostics for debugging
        for line in stderr_lines.split("\n")[-10:]:
            print(f"  [{classifier_name}] {line}", file=sys.stderr)

    if proc.returncode != 0:
        print(f"  [{classifier_name}] exit code: {proc.returncode}", file=sys.stderr)
        return None

    # Parse the first valid JSON line from stdout
    for line in proc.stdout.strip().split("\n"):
        line = line.strip()
        if line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None
