"""Tests for the audio ONNX classifier (YAMNet / MusicNN)."""

import os

import numpy as np
import pytest

from classifiers.musicnn import AudioClassifier, transcode_audio, compute_mel_spectrogram

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "tests", "res")
YAMNET_PATH = os.path.join(MODELS_DIR, "yamnet.onnx")
MUSICNN_PATH = os.path.join(MODELS_DIR, "musicnn.onnx")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def _has_audio_model():
    return os.path.isfile(YAMNET_PATH) or os.path.isfile(MUSICNN_PATH)


@pytest.fixture(scope="module")
def classifier():
    if not _has_audio_model():
        pytest.skip("No audio ONNX model found")
    return AudioClassifier(MODELS_DIR, gpu=False)


class TestInit:
    def test_model_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No audio model found"):
            AudioClassifier(str(tmp_path), gpu=False)

    def test_session_created(self, classifier):
        assert classifier.session is not None

    def test_yamnet_or_musicnn(self, classifier):
        """Should have loaded one of the two models."""
        assert isinstance(classifier.use_yamnet, bool)
        if classifier.use_yamnet:
            assert classifier.input_name == "waveform"
        else:
            assert classifier.input_name == "input_1"


class TestTranscode:
    def test_transcode_mp3(self):
        mp3_path = os.path.join(FIXTURES_DIR, "Rock_Rejam.mp3")
        if not os.path.isfile(mp3_path):
            pytest.skip("Test audio fixture not found")
        audio = transcode_audio(mp3_path, sample_rate=16000)
        assert audio.dtype == np.float32
        assert audio.ndim == 1
        assert len(audio) > 0

    def test_transcode_different_rates(self):
        mp3_path = os.path.join(FIXTURES_DIR, "Rock_Rejam.mp3")
        if not os.path.isfile(mp3_path):
            pytest.skip("Test audio fixture not found")

        audio_16k = transcode_audio(mp3_path, sample_rate=16000)
        audio_8k = transcode_audio(mp3_path, sample_rate=8000)

        # 16kHz should have roughly double the samples of 8kHz
        ratio = len(audio_16k) / len(audio_8k)
        assert 1.8 < ratio < 2.2


class TestMelSpectrogram:
    def test_compute_mel(self):
        mel_matrix_path = os.path.join(DATA_DIR, "mel_matrix.npy")
        if not os.path.isfile(mel_matrix_path):
            pytest.skip("mel_matrix.npy not found")

        mel_matrix = np.load(mel_matrix_path)
        # Simulate 2 seconds of audio at 8kHz
        audio = np.random.randn(16000).astype(np.float32)
        mel = compute_mel_spectrogram(audio, mel_matrix)

        assert mel is not None
        assert mel.ndim == 3  # [frames, mel_bins, 1]
        assert mel.shape[1] == mel_matrix.shape[1]  # mel bins
        assert mel.shape[2] == 1

    def test_short_audio_returns_none(self):
        mel_matrix_path = os.path.join(DATA_DIR, "mel_matrix.npy")
        if not os.path.isfile(mel_matrix_path):
            pytest.skip("mel_matrix.npy not found")

        mel_matrix = np.load(mel_matrix_path)
        # Very short audio (less than one STFT frame)
        audio = np.zeros(100, dtype=np.float32)
        mel = compute_mel_spectrogram(audio, mel_matrix)
        assert mel is None


class TestInference:
    def test_warm_up(self, classifier):
        classifier.warm_up()

    def test_classify_rock_song(self, classifier):
        """Classify the rock song test fixture."""
        mp3_path = os.path.join(FIXTURES_DIR, "Rock_Rejam.mp3")
        if not os.path.isfile(mp3_path):
            pytest.skip("Test audio fixture not found")
        labels = classifier.classify(mp3_path)
        assert isinstance(labels, list)
        # Should produce some genre labels for a rock song
        if labels:
            assert all(isinstance(l, str) for l in labels)

    def test_yamnet_synthetic_waveform(self, classifier):
        """YAMNet should handle a synthetic waveform."""
        if not classifier.use_yamnet:
            pytest.skip("MusicNN loaded, not YAMNet")
        # 1 second of sine wave at 440Hz
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        waveform = 0.5 * np.sin(2 * np.pi * 440 * t)
        labels = classifier.infer_one(waveform)
        assert isinstance(labels, list)

    def test_musicnn_synthetic_batches(self, classifier):
        """MusicNN should handle synthetic mel spectrogram batches."""
        if classifier.use_yamnet:
            pytest.skip("YAMNet loaded, not MusicNN")
        batches = np.random.randn(2, 188, 96, 1).astype(np.float32)
        labels = classifier.infer_one(batches)
        assert isinstance(labels, list)
