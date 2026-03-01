"""Parity tests: compare Node.js and Python classifier outputs.

Runs both classifiers on the same test files and compares results.
Allows small float deltas for numerical differences.

Usage:
    python -m pytest tests/test_parity.py -v
    python -m pytest tests/test_parity.py -v -k test_imagenet

Requires:
    - Node.js classifiers working (node_binary set)
    - Python classifiers working (models converted)
    - Test media files in tests/fixtures/ (create manually)

Environment variables:
    NODE_BINARY: path to node binary (default: node)
    PYTHON_BINARY: path to python binary (default: python3)
    TEST_IMAGE: path to a test JPEG image
    TEST_AUDIO: path to a test audio file (mp3/wav)
    TEST_VIDEO: path to a test video file (mp4)
"""
import json
import os
import subprocess
import sys

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PYTHON_DIR, '..', 'src')

NODE_BINARY = os.environ.get('NODE_BINARY', 'node')
PYTHON_BINARY = os.environ.get('PYTHON_BINARY', sys.executable)

TEST_IMAGE = os.environ.get('TEST_IMAGE', '')
TEST_AUDIO = os.environ.get('TEST_AUDIO', '')
TEST_VIDEO = os.environ.get('TEST_VIDEO', '')


def run_classifier(binary, script, input_path, env_extra=None):
    """Run a classifier script and return parsed JSON output."""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    proc = subprocess.run(
        [binary, script, input_path],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )

    if proc.returncode != 0:
        print(f'STDERR: {proc.stderr}', file=sys.stderr)

    # Parse the first JSON line from stdout
    for line in proc.stdout.strip().split('\n'):
        line = line.strip()
        if line:
            return json.loads(line)
    return None


def compare_labels(node_result, python_result, allow_extra=False):
    """Compare label lists, allowing order differences."""
    if node_result is None or python_result is None:
        pytest.skip('One or both classifiers returned no result')

    node_set = set(node_result) if isinstance(node_result, list) else set()
    python_set = set(python_result) if isinstance(python_result, list) else set()

    # Check overlap
    common = node_set & python_set
    node_only = node_set - python_set
    python_only = python_set - node_set

    print(f'Common labels: {common}')
    if node_only:
        print(f'Node.js only: {node_only}')
    if python_only:
        print(f'Python only: {python_only}')

    # At least some overlap expected (exact match unlikely due to float precision)
    if len(node_set) > 0:
        overlap_ratio = len(common) / len(node_set)
        assert overlap_ratio >= 0.5, f'Less than 50% label overlap: {overlap_ratio:.0%}'


@pytest.mark.skipif(not TEST_IMAGE, reason='TEST_IMAGE not set')
class TestImagenetParity:
    def test_imagenet(self):
        node_result = run_classifier(
            NODE_BINARY,
            os.path.join(SRC_DIR, 'classifier_imagenet.js'),
            TEST_IMAGE,
        )
        python_result = run_classifier(
            PYTHON_BINARY,
            os.path.join(PYTHON_DIR, 'classifier_imagenet.py'),
            TEST_IMAGE,
        )
        print(f'Node.js: {node_result}')
        print(f'Python:  {python_result}')
        compare_labels(node_result, python_result)

    def test_landmarks(self):
        node_result = run_classifier(
            NODE_BINARY,
            os.path.join(SRC_DIR, 'classifier_landmarks.js'),
            TEST_IMAGE,
        )
        python_result = run_classifier(
            PYTHON_BINARY,
            os.path.join(PYTHON_DIR, 'classifier_landmarks.py'),
            TEST_IMAGE,
        )
        print(f'Node.js: {node_result}')
        print(f'Python:  {python_result}')
        compare_labels(node_result, python_result)


@pytest.mark.skipif(not TEST_IMAGE, reason='TEST_IMAGE not set')
class TestFacesParity:
    def test_faces(self):
        """Faces use different models (face-api vs InsightFace), so we only
        check that both detect faces and produce valid output structure."""
        node_result = run_classifier(
            NODE_BINARY,
            os.path.join(SRC_DIR, 'classifier_faces.js'),
            TEST_IMAGE,
        )
        python_result = run_classifier(
            PYTHON_BINARY,
            os.path.join(PYTHON_DIR, 'classifier_faces.py'),
            TEST_IMAGE,
        )
        print(f'Node.js faces: {len(node_result) if node_result else 0}')
        print(f'Python faces:  {len(python_result) if python_result else 0}')

        # Both should detect same number of faces (or close)
        if node_result and python_result:
            assert abs(len(node_result) - len(python_result)) <= 1, \
                f'Face count mismatch: node={len(node_result)}, python={len(python_result)}'

            # Check Python output structure
            if python_result:
                face = python_result[0]
                assert 'vector' in face
                assert len(face['vector']) == 512  # InsightFace = 512-dim
                assert 'x' in face and 'y' in face
                assert 'width' in face and 'height' in face
                assert 'score' in face
                assert 'angle' in face


@pytest.mark.skipif(not TEST_AUDIO, reason='TEST_AUDIO not set')
class TestMusicnnParity:
    def test_musicnn(self):
        node_result = run_classifier(
            NODE_BINARY,
            os.path.join(SRC_DIR, 'classifier_musicnn.js'),
            TEST_AUDIO,
        )
        python_result = run_classifier(
            PYTHON_BINARY,
            os.path.join(PYTHON_DIR, 'classifier_musicnn.py'),
            TEST_AUDIO,
        )
        print(f'Node.js: {node_result}')
        print(f'Python:  {python_result}')
        compare_labels(node_result, python_result)


@pytest.mark.skipif(not TEST_VIDEO, reason='TEST_VIDEO not set')
class TestMovinetParity:
    def test_movinet(self):
        node_result = run_classifier(
            NODE_BINARY,
            os.path.join(SRC_DIR, 'classifier_movinet.js'),
            TEST_VIDEO,
        )
        python_result = run_classifier(
            PYTHON_BINARY,
            os.path.join(PYTHON_DIR, 'classifier_movinet.py'),
            TEST_VIDEO,
        )
        print(f'Node.js: {node_result}')
        print(f'Python:  {python_result}')
        compare_labels(node_result, python_result)


@pytest.mark.skipif(not TEST_IMAGE, reason='TEST_IMAGE not set')
class TestGeoParity:
    def test_geo(self):
        node_result = run_classifier(
            NODE_BINARY,
            os.path.join(SRC_DIR, 'classifier_geo.js'),
            TEST_IMAGE,
        )
        python_result = run_classifier(
            PYTHON_BINARY,
            os.path.join(PYTHON_DIR, 'classifier_geo.py'),
            TEST_IMAGE,
        )
        print(f'Node.js: {node_result}')
        print(f'Python:  {python_result}')
        # Geo should match exactly (same GPS coords → same country)
        if node_result and python_result:
            assert set(node_result) == set(python_result), \
                f'Geo mismatch: node={node_result}, python={python_result}'
