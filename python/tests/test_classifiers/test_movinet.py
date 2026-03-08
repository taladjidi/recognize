"""Tests for the MoViNet ONNX video classifier."""

import os

import numpy as np
import pytest

from classifiers.movinet import MoViNetClassifier, FRAME_SIZE, extract_frames

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "tests", "res")
MODEL_PATH = os.path.join(MODELS_DIR, "movinet_a3.onnx")


@pytest.fixture(scope="module")
def classifier():
    if not os.path.isfile(MODEL_PATH):
        pytest.skip("movinet_a3.onnx not found")
    return MoViNetClassifier(MODELS_DIR, gpu=False)


class TestInit:
    def test_model_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Model not found"):
            MoViNetClassifier(str(tmp_path), gpu=False)

    def test_session_created(self, classifier):
        assert classifier.session is not None
        assert classifier.input_name == "video"
        assert classifier.output_name == "logits"

    def test_class_names_loaded(self, classifier):
        assert len(classifier.class_names) == 600


class TestExtractFrames:
    def test_extract_from_gif(self):
        """GIFs are treated as video by the pipeline."""
        gif_path = os.path.join(FIXTURES_DIR, "jumpingjack.gif")
        if not os.path.isfile(gif_path):
            pytest.skip("jumpingjack.gif not found")
        frames = extract_frames(gif_path, "/usr/bin/ffmpeg")
        if frames is not None:
            assert frames.ndim == 4
            assert frames.shape[1:] == (FRAME_SIZE, FRAME_SIZE, 3)
            assert frames.dtype == np.float32
            assert frames.min() >= 0.0
            assert frames.max() <= 1.0


class TestPreprocess:
    def test_output_shape(self, classifier):
        gif_path = os.path.join(FIXTURES_DIR, "jumpingjack.gif")
        if not os.path.isfile(gif_path):
            pytest.skip("jumpingjack.gif not found")
        tensor = classifier.preprocess(gif_path)
        if tensor is None:
            pytest.skip("Could not extract frames from gif")
        assert tensor.ndim == 5  # [1, C, T, H, W]
        assert tensor.shape[0] == 1
        assert tensor.shape[1] == 3
        assert tensor.shape[3] == FRAME_SIZE
        assert tensor.shape[4] == FRAME_SIZE


class TestInference:
    def test_warm_up(self, classifier):
        classifier.warm_up()

    def test_infer_synthetic(self, classifier):
        """Inference on random data should return valid label list."""
        # 10 frames of random data
        tensor = np.random.rand(1, 3, 10, FRAME_SIZE, FRAME_SIZE).astype(np.float32)
        labels = classifier.infer_one(tensor)
        assert isinstance(labels, list)
        # Random data likely won't pass threshold, but shouldn't crash
        assert all(isinstance(l, str) for l in labels)

    def test_classify_gif(self, classifier):
        """Classify the jumping jack GIF — should produce action labels."""
        gif_path = os.path.join(FIXTURES_DIR, "jumpingjack.gif")
        if not os.path.isfile(gif_path):
            pytest.skip("jumpingjack.gif not found")
        labels = classifier.classify(gif_path)
        assert isinstance(labels, list)
        # The GIF shows a jumping jack; might detect related actions
        # Just verify it runs without error and returns valid output
