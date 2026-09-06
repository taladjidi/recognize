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


def _require_new_skimage():
    """Skip only if scikit-image predates from_estimate.

    Keyed on the library, not on whether the patch applied: if the patch is
    removed these tests must FAIL, not silently skip.
    """
    from skimage import transform as trans

    if not hasattr(trans.SimilarityTransform, "from_estimate"):
        pytest.skip("scikit-image < 0.26: legacy estimate() is still current")


class TestFaceAlignPatch:
    """The insightface face_align compatibility patch.

    insightface 1.0.1 calls scikit-image's SimilarityTransform.estimate(),
    deprecated in 0.26 and removed in 2.2. _patch_insightface_face_align()
    rewrites that one function to use from_estimate(). These tests pin the
    two properties that matter: the output must not drift (embeddings and
    every stored face cluster depend on the alignment matrix), and the
    deprecated call must be gone.
    """

    @staticmethod
    def _landmarks(n=100, seed=0):
        rng = np.random.default_rng(seed)
        base = np.array(
            [[38.3, 51.7], [73.5, 51.5], [56.0, 71.7], [41.5, 92.4], [70.7, 92.2]],
            dtype=np.float32,
        )
        return [(base + rng.normal(0, 12, (5, 2))).astype(np.float32) for _ in range(n)]

    def test_patch_output_is_identical_to_legacy(self):
        """Patched estimate_norm must match the original bit for bit."""
        if not _has_insightface():
            pytest.skip("insightface not installed")
        import warnings

        from insightface.utils import face_align

        from classifiers.faces import _patch_insightface_face_align

        legacy = face_align.estimate_norm
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                before = [legacy(lmk, size)
                          for lmk in self._landmarks() for size in (112, 128, 224)]

            _require_new_skimage()
            _patch_insightface_face_align()
            patched = face_align.estimate_norm
            assert patched is not legacy, "compatibility patch did not apply"

            after = [patched(lmk, size)
                     for lmk in self._landmarks() for size in (112, 128, 224)]
        finally:
            face_align.estimate_norm = legacy

        assert len(before) == len(after)
        for a, b in zip(before, after):
            assert np.array_equal(a, b), "alignment matrix drifted"

    def test_patch_emits_no_deprecation_warning(self):
        if not _has_insightface():
            pytest.skip("insightface not installed")
        import warnings

        from insightface.utils import face_align

        from classifiers.faces import _patch_insightface_face_align

        legacy = face_align.estimate_norm
        try:
            _require_new_skimage()
            _patch_insightface_face_align()
            assert face_align.estimate_norm is not legacy, "compatibility patch did not apply"
            with warnings.catch_warnings():
                warnings.simplefilter("error", FutureWarning)
                face_align.estimate_norm(self._landmarks(1)[0], 112)
        finally:
            face_align.estimate_norm = legacy

    def test_degenerate_landmarks_fall_back_to_identity(self):
        """Legacy code discarded estimate()'s result, leaving identity params."""
        if not _has_insightface():
            pytest.skip("insightface not installed")
        from insightface.utils import face_align

        from classifiers.faces import _patch_insightface_face_align

        legacy = face_align.estimate_norm
        try:
            _require_new_skimage()
            _patch_insightface_face_align()
            assert face_align.estimate_norm is not legacy, "compatibility patch did not apply"
            M = face_align.estimate_norm(np.zeros((5, 2), dtype=np.float32), 112)
        finally:
            face_align.estimate_norm = legacy

        assert np.array_equal(M, np.eye(3)[0:2, :])
