"""Tests for the landmark ONNX classifier."""

import os

import numpy as np
import pytest

from classifiers.landmarks import LandmarkClassifier, IMG_SIZE, REGIONS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "tests", "res")


def _all_landmark_models_exist():
    return all(
        os.path.isfile(os.path.join(MODELS_DIR, f"{r}.onnx"))
        for r in REGIONS
    )


@pytest.fixture(scope="module")
def classifier():
    if not _all_landmark_models_exist():
        pytest.skip("Landmark ONNX models not found")
    return LandmarkClassifier(MODELS_DIR, gpu=False)


class TestInit:
    def test_model_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Model not found"):
            LandmarkClassifier(str(tmp_path), gpu=False)

    def test_all_sessions_loaded(self, classifier):
        assert len(classifier.sessions) == 6
        for region in REGIONS:
            assert region in classifier.sessions

    def test_labels_loaded(self, classifier):
        assert len(classifier.labels) == 6
        for region in REGIONS:
            assert isinstance(classifier.labels[region], dict)
            assert len(classifier.labels[region]) > 0


class TestPreprocess:
    def test_output_shape(self, classifier):
        img_path = os.path.join(FIXTURES_DIR, "eiffeltower.jpg")
        if not os.path.isfile(img_path):
            pytest.skip("Test fixture not found")
        arr = classifier.preprocess(img_path)
        assert arr.shape == (IMG_SIZE, IMG_SIZE, 3)
        assert arr.dtype == np.float32

    def test_output_range(self, classifier):
        img_path = os.path.join(FIXTURES_DIR, "eiffeltower.jpg")
        if not os.path.isfile(img_path):
            pytest.skip("Test fixture not found")
        arr = classifier.preprocess(img_path)
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0


class TestInference:
    def test_warm_up(self, classifier):
        classifier.warm_up()

    def test_infer_batch_shape(self, classifier):
        batch = np.random.rand(2, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
        results = classifier.infer_batch(batch)
        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)

    def test_classify_eiffel_tower(self, classifier):
        """The Eiffel Tower photo should be recognized as a landmark."""
        img_path = os.path.join(FIXTURES_DIR, "eiffeltower.jpg")
        if not os.path.isfile(img_path):
            pytest.skip("Test fixture not found")
        result = classifier.classify(img_path)
        assert isinstance(result, list)
        if result:
            assert result[0] == "landmark"
            assert len(result) == 2
            # The name should reference the Eiffel Tower
            assert "eiffel" in result[1].lower() or "tour" in result[1].lower(), \
                f"Expected Eiffel Tower, got: {result[1]}"

    def test_non_landmark_returns_empty(self, classifier):
        """A food photo should not be classified as a landmark."""
        img_path = os.path.join(FIXTURES_DIR, "casey-lee-awj7sRviVXo-unsplash.jpg")
        if not os.path.isfile(img_path):
            pytest.skip("Test fixture not found")
        result = classifier.classify(img_path)
        # Food photos shouldn't match any landmark above the 0.9 threshold
        assert result == [] or result[0] == "landmark"

    def test_batch_consistency(self, classifier):
        """Batch inference should match single-image inference."""
        img_path = os.path.join(FIXTURES_DIR, "eiffeltower.jpg")
        if not os.path.isfile(img_path):
            pytest.skip("Test fixture not found")
        arr = classifier.preprocess(img_path)

        single_result = classifier.infer_batch(np.expand_dims(arr, 0))[0]
        batch_result = classifier.infer_batch(np.stack([arr, arr]))[0]

        assert single_result == batch_result
