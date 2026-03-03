"""Parity tests: compare Node.js and Python classifier outputs.

Runs both classifiers on the same test files and compares results.
Allows small float deltas for numerical differences.

Usage:
    python -m pytest tests/test_parity.py -v
    python -m pytest tests/test_parity.py -v -k test_imagenet

Auto-discovers test fixtures from tests/res/:
    alpine.JPG, eiffeltower.jpg  →  imagenet, landmarks
    geotagged.jpg                →  geo
    FaceID-550/*/                →  faces
    Rock_Rejam.mp3               →  musicnn
    jumpingjack.gif              →  movinet

Environment variables (all optional):
    NODE_BINARY: path to node binary (default: node)
    PYTHON_BINARY: path to python binary (default: python3)
    RECOGNIZE_PUREJS: set to "true" to use WASM backend for JS classifiers
    TEST_IMAGE: override auto-discovered image path
    TEST_AUDIO: override auto-discovered audio path
    TEST_VIDEO: override auto-discovered video path
"""

import json
import os
import subprocess
import sys

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.join(PYTHON_DIR, "..")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
RES_DIR = os.path.join(PROJECT_ROOT, "tests", "res")

NODE_BINARY = os.environ.get("NODE_BINARY", "node")
PYTHON_BINARY = os.environ.get("PYTHON_BINARY", sys.executable)


def _find_fixture(env_var, *candidates):
    """Return env var override or first existing candidate from tests/res/."""
    override = os.environ.get(env_var, "")
    if override:
        return override
    for name in candidates:
        path = os.path.join(RES_DIR, name)
        if os.path.exists(path):
            return path
    return ""


TEST_IMAGE = _find_fixture(
    "TEST_IMAGE", "alpine.JPG", "casey-lee-awj7sRviVXo-unsplash.jpg"
)
TEST_LANDMARK_IMAGE = _find_fixture(
    "TEST_LANDMARK_IMAGE", "eiffeltower.jpg", "alpine.JPG"
)
TEST_GEO_IMAGE = _find_fixture("TEST_GEO_IMAGE", "geotagged.jpg", "geotagged2.jpg")
TEST_FACE_IMAGE = _find_fixture(
    "TEST_FACE_IMAGE",
    "FaceID-550/Nguyen_Ngoc_Nghia/Nguyen_Ngoc_Nghia_1.jpg",
    "FaceID-550/Sao_Mai/Sao_Mai_1.jpg",
)
TEST_AUDIO = _find_fixture("TEST_AUDIO", "Rock_Rejam.mp3")
TEST_VIDEO = _find_fixture("TEST_VIDEO", "jumpingjack.gif")


def _js_env():
    """Build env dict for JS subprocesses, passing through RECOGNIZE_PUREJS."""
    env = os.environ.copy()
    if os.environ.get("RECOGNIZE_PUREJS"):
        env["RECOGNIZE_PUREJS"] = os.environ["RECOGNIZE_PUREJS"]
    return env


def run_js_classifier(script, input_path):
    """Run a JS classifier and return parsed JSON output."""
    proc = subprocess.run(
        [NODE_BINARY, script, input_path],
        capture_output=True,
        text=True,
        env=_js_env(),
        timeout=180,
    )
    if proc.returncode != 0:
        print(f"JS STDERR: {proc.stderr[:500]}", file=sys.stderr)
        pytest.fail(f"JS classifier exited {proc.returncode}: {proc.stderr[:200]}")
    return _parse_last_json(proc.stdout)


def run_py_classifier(script, input_path):
    """Run a Python classifier and return parsed JSON output."""
    proc = subprocess.run(
        [PYTHON_BINARY, script, input_path],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        print(f"PY STDERR: {proc.stderr[:500]}", file=sys.stderr)
        pytest.fail(f"Python classifier exited {proc.returncode}: {proc.stderr[:200]}")
    return _parse_last_json(proc.stdout)


def _parse_last_json(stdout):
    """Parse the last valid JSON line from stdout (classifiers output JSON array last)."""
    lines = stdout.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def compare_labels(node_result, python_result, min_overlap=0.5):
    """Compare label lists, allowing order differences."""
    if node_result is None or python_result is None:
        pytest.skip("One or both classifiers returned no result")

    node_set = set(node_result) if isinstance(node_result, list) else set()
    python_set = set(python_result) if isinstance(python_result, list) else set()

    common = node_set & python_set
    node_only = node_set - python_set
    python_only = python_set - node_set

    print(f"Common labels: {common}")
    if node_only:
        print(f"Node.js only: {node_only}")
    if python_only:
        print(f"Python only: {python_only}")

    if len(node_set) > 0:
        overlap_ratio = len(common) / len(node_set)
        assert overlap_ratio >= min_overlap, (
            f"Less than {min_overlap:.0%} label overlap: {overlap_ratio:.0%}"
        )


@pytest.mark.skipif(not TEST_IMAGE, reason="No test image found in tests/res/")
class TestImagenetParity:
    def test_imagenet(self):
        node_result = run_js_classifier(
            os.path.join(SRC_DIR, "classifier_imagenet.js"), TEST_IMAGE
        )
        python_result = run_py_classifier(
            os.path.join(PYTHON_DIR, "classifier_imagenet.py"), TEST_IMAGE
        )
        print(f"Node.js: {node_result}")
        print(f"Python:  {python_result}")
        compare_labels(node_result, python_result)


@pytest.mark.skipif(not TEST_LANDMARK_IMAGE, reason="No landmark test image found")
class TestLandmarksParity:
    def test_landmarks(self):
        node_result = run_js_classifier(
            os.path.join(SRC_DIR, "classifier_landmarks.js"), TEST_LANDMARK_IMAGE
        )
        python_result = run_py_classifier(
            os.path.join(PYTHON_DIR, "classifier_landmarks.py"), TEST_LANDMARK_IMAGE
        )
        print(f"Node.js: {node_result}")
        print(f"Python:  {python_result}")
        # Landmarks may not match if the image isn't a well-known landmark
        if node_result and python_result:
            compare_labels(node_result, python_result)
        else:
            print(
                "One or both returned no landmarks (expected for non-landmark images)"
            )


@pytest.mark.skipif(
    not TEST_FACE_IMAGE, reason="No face test image found in tests/res/FaceID-550/"
)
class TestFacesParity:
    def test_faces(self):
        """Faces use different models (face-api 128-dim vs InsightFace 512-dim),
        so we only check that both detect faces and produce valid structure."""
        node_result = run_js_classifier(
            os.path.join(SRC_DIR, "classifier_faces.js"), TEST_FACE_IMAGE
        )
        python_result = run_py_classifier(
            os.path.join(PYTHON_DIR, "classifier_faces.py"), TEST_FACE_IMAGE
        )
        print(f"Node.js faces: {len(node_result) if node_result else 0}")
        print(f"Python faces:  {len(python_result) if python_result else 0}")

        if node_result and python_result:
            assert abs(len(node_result) - len(python_result)) <= 1, (
                f"Face count mismatch: node={len(node_result)}, python={len(python_result)}"
            )

            if python_result:
                face = python_result[0]
                assert "vector" in face
                assert len(face["vector"]) == 512  # InsightFace = 512-dim
                assert "x" in face and "y" in face
                assert "width" in face and "height" in face
                assert "score" in face
                assert "angle" in face


@pytest.mark.skipif(not TEST_AUDIO, reason="No test audio found in tests/res/")
class TestMusicnnParity:
    def test_musicnn(self):
        node_result = run_js_classifier(
            os.path.join(SRC_DIR, "classifier_musicnn.js"), TEST_AUDIO
        )
        python_result = run_py_classifier(
            os.path.join(PYTHON_DIR, "classifier_musicnn.py"), TEST_AUDIO
        )
        print(f"Node.js: {node_result}")
        print(f"Python:  {python_result}")
        compare_labels(node_result, python_result)


@pytest.mark.skipif(not TEST_VIDEO, reason="No test video found in tests/res/")
class TestMovinetParity:
    def test_movinet(self):
        node_result = run_js_classifier(
            os.path.join(SRC_DIR, "classifier_movinet.js"), TEST_VIDEO
        )
        python_result = run_py_classifier(
            os.path.join(PYTHON_DIR, "classifier_movinet.py"), TEST_VIDEO
        )
        print(f"Node.js: {node_result}")
        print(f"Python:  {python_result}")
        compare_labels(node_result, python_result)


@pytest.mark.skipif(not TEST_GEO_IMAGE, reason="No geotagged test image found")
class TestGeoParity:
    def test_geo(self):
        node_result = run_js_classifier(
            os.path.join(SRC_DIR, "classifier_geo.js"), TEST_GEO_IMAGE
        )
        python_result = run_py_classifier(
            os.path.join(PYTHON_DIR, "classifier_geo.py"), TEST_GEO_IMAGE
        )
        print(f"Node.js: {node_result}")
        print(f"Python:  {python_result}")
        # Geo should match exactly (same GPS coords → same country)
        if node_result and python_result:
            assert set(node_result) == set(python_result), (
                f"Geo mismatch: node={node_result}, python={python_result}"
            )
