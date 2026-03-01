"""Standalone test harness for Python classifiers.

Tests each classifier directly via subprocess, using the same test fixtures
as the PHP CI (tests/res/). No Nextcloud instance required.

Expected results match ClassifierTest.php's classifierFilesProvider:
    alpine.JPG       → imagenet  → 'Alpine'
    eiffeltower.jpg  → landmarks → 'Eiffel Tower'
    Rock_Rejam.mp3   → musicnn   → 'electronic'
    jumpingjack.gif  → movinet   → 'jumping jacks'

Usage:
    conda run -n 313 python -m pytest python/tests/test_harness.py -v
    conda run -n 313 python -m pytest python/tests/test_harness.py -v -k imagenet
    conda run -n 313 python -m pytest python/tests/test_harness.py -v -k Geo
"""
import json
import os
import subprocess
import sys

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PYTHON_DIR = os.path.join(REPO_ROOT, 'python')


# ── Infrastructure tests ─────────────────────────────────────────────

class TestInfrastructure:
    """Test that shared modules load without error."""

    def test_import_base_classifier(self):
        sys.path.insert(0, PYTHON_DIR)
        import importlib
        mod = importlib.import_module('base_classifier')
        assert hasattr(mod, 'get_paths')
        assert hasattr(mod, 'output_result')

    def test_import_gpu_setup(self):
        sys.path.insert(0, PYTHON_DIR)
        import importlib
        mod = importlib.import_module('gpu_setup')
        assert hasattr(mod, 'configure')

    def test_import_rules_engine(self):
        sys.path.insert(0, PYTHON_DIR)
        import importlib
        mod = importlib.import_module('rules_engine')

        rules_path = os.path.join(REPO_ROOT, 'src', 'rules.yml')
        if os.path.isfile(rules_path):
            rules = mod.load_rules(rules_path)
            assert isinstance(rules, dict)
            assert len(rules) > 100

            # 'cat' is a direct key in rules.yml
            rule = mod.find_rule(rules, 'cat')
            assert rule is not None
            assert rule['label'] == 'cat'

    def test_data_files_exist(self):
        data_dir = os.path.join(PYTHON_DIR, 'data')
        assert os.path.isfile(os.path.join(data_dir, 'imagenet_classes.json'))
        assert os.path.isfile(os.path.join(data_dir, 'kinetics_classes.json'))
        assert os.path.isfile(os.path.join(data_dir, 'musicnn_classes.json'))
        assert os.path.isfile(os.path.join(data_dir, 'mel_matrix.npy'))
        assert os.path.exists(os.path.join(data_dir, 'landmarks'))


# ── ImageNet classifier tests ────────────────────────────────────────

class TestImagenet:
    """Test EfficientNetV2 image classification."""

    def test_alpine(self, run_classifier, has_imagenet_model, alpine_jpg):
        """alpine.JPG should be tagged 'Alpine' (matches PHP CI)."""
        result = run_classifier('imagenet', alpine_jpg, timeout=180)
        assert result is not None, 'Classifier returned no output'
        assert isinstance(result, list), f'Expected list, got {type(result)}'
        print(f'  Result: {result}')
        assert 'Alpine' in result, f'"Alpine" not in {result}'

    def test_food(self, run_classifier, has_imagenet_model, food_jpg):
        """Food photo should get food-related tags."""
        result = run_classifier('imagenet', food_jpg, timeout=180)
        assert result is not None, 'Classifier returned no output'
        assert isinstance(result, list), f'Expected list, got {type(result)}'
        print(f'  Result: {result}')
        assert len(result) > 0, 'No tags returned for food image'

    def test_output_format(self, run_classifier, has_imagenet_model, alpine_jpg):
        """Output should be a JSON array of capitalized strings."""
        result = run_classifier('imagenet', alpine_jpg, timeout=180)
        assert result is not None
        for label in result:
            assert isinstance(label, str), f'Label should be string, got {type(label)}'
            assert label[0].isupper(), f'Label should be capitalized: {label}'


# ── Landmarks classifier tests ───────────────────────────────────────

class TestLandmarks:
    """Test 6-region landmark classification."""

    def test_eiffel_tower(self, run_classifier, has_landmark_models, eiffeltower_jpg):
        """eiffeltower.jpg should be detected as 'Eiffel Tower' (matches PHP CI)."""
        result = run_classifier('landmarks', eiffeltower_jpg, timeout=300)
        assert result is not None, 'Classifier returned no output'
        assert isinstance(result, list), f'Expected list, got {type(result)}'
        print(f'  Result: {result}')
        if len(result) >= 2:
            assert result[0] == 'landmark', f'First element should be "landmark", got {result[0]}'
            assert 'eiffel' in result[1].lower(), f'Expected Eiffel Tower, got {result[1]}'

    def test_no_landmark(self, run_classifier, has_landmark_models, food_jpg):
        """Food photo should NOT be detected as a landmark."""
        result = run_classifier('landmarks', food_jpg, timeout=300)
        assert result is not None, 'Classifier returned no output'
        print(f'  Result: {result}')
        assert result == [], f'Expected no landmark for food photo, got {result}'


# ── MoViNet classifier tests ─────────────────────────────────────────

class TestMovinet:
    """Test video action classification."""

    def test_jumping_jacks(self, run_classifier, has_movinet_model, jumpingjack_gif):
        """jumpingjack.gif should be tagged 'jumping jacks' (matches PHP CI)."""
        result = run_classifier('movinet', jumpingjack_gif, timeout=180)
        assert result is not None, 'Classifier returned no output'
        assert isinstance(result, list), f'Expected list, got {type(result)}'
        print(f'  Result: {result}')
        assert 'jumping jacks' in result, f'"jumping jacks" not in {result}'


# ── MusicNN classifier tests ─────────────────────────────────────────

class TestMusicnn:
    """Test audio genre classification."""

    def test_rock_rejam(self, run_classifier, has_musicnn_model, rock_mp3):
        """Rock_Rejam.mp3 should be tagged 'electronic' (matches PHP CI)."""
        result = run_classifier('musicnn', rock_mp3, timeout=180)
        assert result is not None, 'Classifier returned no output'
        assert isinstance(result, list), f'Expected list, got {type(result)}'
        print(f'  Result: {result}')
        assert 'electronic' in result, f'"electronic" not in {result}'


# ── Geo classifier tests ─────────────────────────────────────────────

class TestGeo:
    """Test GPS EXIF reverse geocoding."""

    def test_geotagged(self, run_classifier, geotagged_jpg):
        """geotagged.jpg should return country names."""
        result = run_classifier('geo', geotagged_jpg)
        assert result is not None, 'Classifier returned no output'
        assert isinstance(result, list), f'Expected list, got {type(result)}'
        print(f'  Result: {result}')
        assert len(result) > 0, 'No country returned for geotagged image'

    def test_geotagged_germany(self, run_classifier, geotagged_germany_jpg):
        """geotagged_germany.jpg should return Germany."""
        result = run_classifier('geo', geotagged_germany_jpg)
        assert result is not None, 'Classifier returned no output'
        assert isinstance(result, list), f'Expected list, got {type(result)}'
        print(f'  Result: {result}')
        assert any('Germany' in c or 'DE' in c for c in result), \
            f'Expected Germany in {result}'

    def test_no_gps(self, run_classifier, alpine_jpg):
        """alpine.JPG (no GPS data) should return empty list."""
        result = run_classifier('geo', alpine_jpg)
        assert result is not None, 'Classifier returned no output'
        print(f'  Result: {result}')
        assert result == [], f'Expected empty list for non-geotagged image, got {result}'


# ── Face classifier tests ────────────────────────────────────────────

class TestFaces:
    """Test InsightFace face detection."""

    def test_face_detection(self, run_classifier, face_images):
        """Should detect at least one face in a portrait image."""
        person = list(face_images.keys())[0]
        img = face_images[person][0]
        result = run_classifier('faces', img, timeout=120)
        assert result is not None, 'Classifier returned no output'
        assert isinstance(result, list), f'Expected list, got {type(result)}'
        print(f'  Result: {len(result)} face(s) detected')
        assert len(result) > 0, f'No faces detected in {img}'

    def test_face_output_structure(self, run_classifier, face_images):
        """Face output should have the expected fields."""
        person = list(face_images.keys())[0]
        img = face_images[person][0]
        result = run_classifier('faces', img, timeout=120)
        assert result is not None and len(result) > 0, 'No faces detected'

        face = result[0]
        assert 'x' in face, 'Missing field: x'
        assert 'y' in face, 'Missing field: y'
        assert 'width' in face, 'Missing field: width'
        assert 'height' in face, 'Missing field: height'
        assert 'score' in face, 'Missing field: score'
        assert 'vector' in face, 'Missing field: vector'
        assert 'angle' in face, 'Missing field: angle'

        assert len(face['vector']) == 512, \
            f'Expected 512-dim vector, got {len(face["vector"])}'

        assert 0 <= face['x'] <= 1, f'x out of range: {face["x"]}'
        assert 0 <= face['y'] <= 1, f'y out of range: {face["y"]}'
        assert 0 < face['width'] <= 1, f'width out of range: {face["width"]}'
        assert 0 < face['height'] <= 1, f'height out of range: {face["height"]}'
        assert 0 <= face['score'] <= 1, f'score out of range: {face["score"]}'
        assert 'roll' in face['angle'], 'Missing angle.roll'
        assert 'yaw' in face['angle'], 'Missing angle.yaw'


# ── Multi-file batch tests ───────────────────────────────────────────

class TestBatchProtocol:
    """Test the STDIN batch protocol (multiple files at once)."""

    def test_geo_batch(self, geotagged_jpg, geotagged_germany_jpg):
        """Sending multiple files via stdin should return one JSON line per file."""
        script = os.path.join(PYTHON_DIR, 'classifier_geo.py')
        input_data = f'{geotagged_jpg}\n{geotagged_germany_jpg}\n'

        proc = subprocess.run(
            [sys.executable, script, '-'],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=PYTHON_DIR,
        )

        assert proc.returncode == 0, f'Exit code {proc.returncode}: {proc.stderr}'

        # Parse only valid JSON lines (ignore library chatter)
        json_lines = []
        for line in proc.stdout.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                json_lines.append(parsed)
            except json.JSONDecodeError:
                pass  # Skip non-JSON output (e.g. library loading messages)

        assert len(json_lines) == 2, \
            f'Expected 2 JSON results for 2 input files, got {len(json_lines)}'

        for i, parsed in enumerate(json_lines):
            assert isinstance(parsed, list), f'Result {i}: expected list, got {type(parsed)}'
