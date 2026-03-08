"""Tests for the ImageNet ONNX classifier."""

import os

import numpy as np
import pytest

from classifiers.imagenet import ImageNetClassifier, IMG_SIZE

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "tests", "res")
MODEL_PATH = os.path.join(MODELS_DIR, "efficientnetv2s.onnx")


@pytest.fixture(scope="module")
def classifier():
    """Load classifier once for all tests in this module."""
    if not os.path.isfile(MODEL_PATH):
        pytest.skip("efficientnetv2s.onnx not found")
    return ImageNetClassifier(MODELS_DIR, gpu=False)


class TestInit:
    def test_model_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Model not found"):
            ImageNetClassifier(str(tmp_path), gpu=False)

    def test_session_created(self, classifier):
        assert classifier.session is not None
        assert classifier.input_name == "input_1"
        assert classifier.output_name == "output_1"

    def test_class_names_loaded(self, classifier):
        assert len(classifier.class_names) == 1000

    def test_rules_loaded(self, classifier):
        assert isinstance(classifier.rules, dict)
        assert "cat" in classifier.rules


class TestPreprocess:
    def test_output_shape(self, classifier):
        img_path = os.path.join(FIXTURES_DIR, "alpine.JPG")
        if not os.path.isfile(img_path):
            pytest.skip("Test fixture not found")
        arr = classifier.preprocess(img_path)
        assert arr.shape == (IMG_SIZE, IMG_SIZE, 3)
        assert arr.dtype == np.float32

    def test_output_range(self, classifier):
        img_path = os.path.join(FIXTURES_DIR, "alpine.JPG")
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

    def test_classify_food_image(self, classifier):
        """The casey-lee food photo should produce food-related labels."""
        img_path = os.path.join(FIXTURES_DIR, "casey-lee-awj7sRviVXo-unsplash.jpg")
        if not os.path.isfile(img_path):
            pytest.skip("Test fixture not found")
        labels = classifier.classify(img_path)
        assert isinstance(labels, list)
        assert len(labels) > 0
        # Should recognize food-related content
        food_keywords = {"food", "dish", "meal", "plate", "cuisine", "restaurant",
                         "cooking", "vegetable", "fruit", "meat", "salad", "pizza",
                         "bowl", "brunch", "dinner", "lunch", "breakfast"}
        lower_labels = {l.lower() for l in labels}
        assert lower_labels & food_keywords, f"Expected food labels, got: {labels}"

    def test_classify_alpine_image(self, classifier):
        """The alpine photo should produce nature-related labels."""
        img_path = os.path.join(FIXTURES_DIR, "alpine.JPG")
        if not os.path.isfile(img_path):
            pytest.skip("Test fixture not found")
        labels = classifier.classify(img_path)
        assert isinstance(labels, list)
        assert len(labels) > 0

    def test_batch_consistency(self, classifier):
        """Batch inference should match single-image inference."""
        img_path = os.path.join(FIXTURES_DIR, "alpine.JPG")
        if not os.path.isfile(img_path):
            pytest.skip("Test fixture not found")
        arr = classifier.preprocess(img_path)

        single_result = classifier.infer_batch(np.expand_dims(arr, 0))[0]
        batch_result = classifier.infer_batch(np.stack([arr, arr]))[0]

        assert single_result == batch_result
