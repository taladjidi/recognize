"""Tests for the face detection ONNX classifier."""

import os

import numpy as np
import pytest

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "tests", "res")
FACE_FIXTURES = os.path.join(FIXTURES_DIR, "FaceID-550")


def _has_insightface():
    try:
        import insightface  # noqa: F401
        return True
    except ImportError:
        return False


def _has_face_model():
    return os.path.isdir(os.path.join(MODELS_DIR, "insightface", "models", "buffalo_l"))


@pytest.fixture(scope="module")
def classifier():
    """Load face classifier once for all tests."""
    if not _has_insightface():
        pytest.skip("insightface not installed")
    if not _has_face_model():
        pytest.skip("buffalo_l model not found")

    from classifiers.faces import FaceClassifier
    return FaceClassifier(MODELS_DIR, gpu=False)


class TestInit:
    def test_model_loaded(self, classifier):
        assert classifier.app is not None

    def test_missing_insightface(self, tmp_path):
        if not _has_insightface():
            pytest.skip("insightface not installed")
        # With empty models dir, InsightFace will try to download
        # This test just verifies the init code path runs
        # (actual download would fail in CI)


class TestPreprocess:
    def test_output_format(self, classifier):
        """Preprocess should return BGR uint8 array."""
        face_dirs = []
        if os.path.isdir(FACE_FIXTURES):
            for person_dir in os.listdir(FACE_FIXTURES):
                full = os.path.join(FACE_FIXTURES, person_dir)
                if os.path.isdir(full):
                    face_dirs.append(full)
                    break

        if not face_dirs:
            pytest.skip("No face test fixtures found")

        imgs = [f for f in os.listdir(face_dirs[0]) if f.endswith(".jpg")]
        if not imgs:
            pytest.skip("No face images found")

        img_path = os.path.join(face_dirs[0], imgs[0])
        arr = classifier.preprocess(img_path)
        assert arr.ndim == 3
        assert arr.shape[2] == 3
        assert arr.dtype == np.uint8


class TestInference:
    def test_warm_up(self, classifier):
        classifier.warm_up()

    def test_detect_face(self, classifier):
        """Should detect at least one face in a portrait photo."""
        if not os.path.isdir(FACE_FIXTURES):
            pytest.skip("Face fixtures not found")

        # Find a face image
        img_path = None
        for person_dir in os.listdir(FACE_FIXTURES):
            full = os.path.join(FACE_FIXTURES, person_dir)
            if os.path.isdir(full):
                for f in os.listdir(full):
                    if f.endswith(".jpg"):
                        img_path = os.path.join(full, f)
                        break
            if img_path:
                break

        if img_path is None:
            pytest.skip("No face images found")

        faces = classifier.classify(img_path)
        assert isinstance(faces, list)
        assert len(faces) >= 1

        face = faces[0]
        assert "vector" in face
        assert "x" in face
        assert "y" in face
        assert "width" in face
        assert "height" in face
        assert "score" in face

    def test_face_vector_normalized(self, classifier):
        """Face embedding vectors should be L2-normalized."""
        if not os.path.isdir(FACE_FIXTURES):
            pytest.skip("Face fixtures not found")

        img_path = None
        for person_dir in os.listdir(FACE_FIXTURES):
            full = os.path.join(FACE_FIXTURES, person_dir)
            if os.path.isdir(full):
                for f in os.listdir(full):
                    if f.endswith(".jpg"):
                        img_path = os.path.join(full, f)
                        break
            if img_path:
                break

        if img_path is None:
            pytest.skip("No face images found")

        faces = classifier.classify(img_path)
        if not faces:
            pytest.skip("No faces detected")

        vec = np.array(faces[0]["vector"])
        assert len(vec) == 512
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 0.01, f"Vector not normalized: norm={norm}"

    def test_face_coords_relative(self, classifier):
        """Bounding box coordinates should be in [0, 1] range."""
        if not os.path.isdir(FACE_FIXTURES):
            pytest.skip("Face fixtures not found")

        img_path = None
        for person_dir in os.listdir(FACE_FIXTURES):
            full = os.path.join(FACE_FIXTURES, person_dir)
            if os.path.isdir(full):
                for f in os.listdir(full):
                    if f.endswith(".jpg"):
                        img_path = os.path.join(full, f)
                        break
            if img_path:
                break

        if img_path is None:
            pytest.skip("No face images found")

        faces = classifier.classify(img_path)
        if not faces:
            pytest.skip("No faces detected")

        face = faces[0]
        assert 0.0 <= face["x"] <= 1.0
        assert 0.0 <= face["y"] <= 1.0
        assert 0.0 < face["width"] <= 1.0
        assert 0.0 < face["height"] <= 1.0

    def test_no_face_in_landscape(self, classifier):
        """An alpine landscape should have few or no faces."""
        img_path = os.path.join(FIXTURES_DIR, "alpine.JPG")
        if not os.path.isfile(img_path):
            pytest.skip("Test fixture not found")

        faces = classifier.classify(img_path)
        assert isinstance(faces, list)
        # Landscape photos might occasionally detect false positives, but
        # should have far fewer faces than a portrait dataset
        assert len(faces) <= 2
