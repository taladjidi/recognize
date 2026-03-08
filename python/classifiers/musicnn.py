"""Audio classifier using YAMNet (primary) and MusicNN (fallback), both ONNX.

YAMNet: 521 AudioSet classes via sigmoid, averaged across frames.
MusicNN: 15 MSD genre classes via softmax over mel spectrogram batches.

ONNX models:
  yamnet.onnx:   Input waveform [samples], Output output_0 [frames, 521]
  musicnn.onnx:  Input input_1 [N, 188, 96, 1], Output dense_1 [N, 15]
"""

import io
import json
import logging
import os
import subprocess

import numpy as np
import soundfile as sf

from classifiers.base import create_session, softmax
import rules_engine

log = logging.getLogger(__name__)

YAMNET_TOP_K = 30
MUSICNN_TOP_K = 6
MAX_DURATION = 120  # seconds
MUSICNN_BATCH_FRAMES = 188

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SCRIPT_DIR, "..", "data")
_SRC_DIR = os.path.join(_SCRIPT_DIR, "..", "..", "src")


def _load_yamnet_class_names(data_dir=None):
    """Load YAMNet class names from the AudioSet ontology CSV.

    The CSV is embedded in the ONNX model's assets, but we also keep
    a standalone copy. Falls back to the class_map asset path if available.
    """
    d = data_dir or _DATA_DIR
    csv_path = os.path.join(d, "yamnet_class_map.csv")
    if os.path.isfile(csv_path):
        import csv
        class_names = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_names.append(row["display_name"])
        return class_names
    return None


def _load_musicnn_classes(data_dir=None):
    """Load MusicNN class names."""
    d = data_dir or _DATA_DIR
    with open(os.path.join(d, "musicnn_classes.json")) as f:
        return json.load(f)


def _load_mel_matrix(data_dir=None):
    """Load mel filterbank matrix for MusicNN preprocessing."""
    d = data_dir or _DATA_DIR
    return np.load(os.path.join(d, "mel_matrix.npy"))


def transcode_audio(audio_path, sample_rate=16000, ffmpeg_binary="/usr/bin/ffmpeg"):
    """Transcode audio to mono float32 WAV at given sample rate.

    Args:
        audio_path: Path to audio file.
        sample_rate: Target sample rate (16000 for YAMNet, 8000 for MusicNN).
        ffmpeg_binary: Path to ffmpeg binary.

    Returns:
        numpy array of float32 samples.
    """
    cores = os.environ.get("RECOGNIZE_CORES", "0")
    cores_arg = ["-threads", cores] if cores and cores != "0" else []

    cmd = [
        ffmpeg_binary, "-i", audio_path,
        "-f", "wav", "-ac", "1", "-ar", str(sample_rate),
        "-acodec", "pcm_s16le", "-t", str(MAX_DURATION),
        *cores_arg, "-",
    ]

    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {proc.stderr.decode()[:200]}")

    audio_data, _ = sf.read(io.BytesIO(proc.stdout), dtype="float32")
    return audio_data


def compute_mel_spectrogram(audio_data, mel_matrix):
    """Compute mel spectrogram for MusicNN (numpy-only, no TF).

    Matches the JS MusicNN implementation: STFT -> magnitude -> mel filterbank -> log.
    """
    # STFT parameters matching the original
    frame_length = 512
    frame_step = 256
    n_fft = 512

    # Pad audio to ensure full frames
    n_samples = len(audio_data)
    n_frames = 1 + (n_samples - frame_length) // frame_step
    if n_frames <= 0:
        return None

    # Create frames
    indices = np.arange(frame_length)[None, :] + np.arange(n_frames)[:, None] * frame_step
    # Clip indices to avoid out-of-bounds
    indices = np.clip(indices, 0, n_samples - 1)
    frames = audio_data[indices]

    # Apply Hann window
    window = np.hanning(frame_length).astype(np.float32)
    windowed = frames * window

    # FFT
    spectrum = np.fft.rfft(windowed, n=n_fft)
    magnitude = np.abs(spectrum).astype(np.float32)

    # Mel filterbank
    mel_spec = magnitude @ mel_matrix

    # Log scale
    mel_spec = np.log(np.clip(mel_spec, 1e-6, None))

    # Add channel dimension [frames, mel_bins, 1]
    return mel_spec[:, :, np.newaxis]


class AudioClassifier:
    """YAMNet / MusicNN audio classifier using ONNX Runtime."""

    def __init__(self, models_dir, gpu=True, data_dir=None, src_dir=None,
                 ffmpeg_binary="/usr/bin/ffmpeg"):
        """Initialize the audio classifier.

        Tries YAMNet first (yamnet.onnx), falls back to MusicNN (musicnn.onnx).

        Args:
            models_dir: Directory containing ONNX model files.
            gpu: Whether to use GPU providers.
            data_dir: Override for class data directory.
            src_dir: Override for rules YAML directory.
            ffmpeg_binary: Path to ffmpeg binary.
        """
        self.ffmpeg_binary = ffmpeg_binary
        s = src_dir or _SRC_DIR

        yamnet_path = os.path.join(models_dir, "yamnet.onnx")
        musicnn_path = os.path.join(models_dir, "musicnn.onnx")

        if os.path.isfile(yamnet_path):
            self.use_yamnet = True
            self.session = create_session(yamnet_path, gpu=gpu)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.class_names = _load_yamnet_class_names(data_dir)
            self.rules = rules_engine.load_rules(os.path.join(s, "yamnet_rules.yml"))
            log.info("Audio classifier initialized (YAMNet)")
        elif os.path.isfile(musicnn_path):
            self.use_yamnet = False
            self.session = create_session(musicnn_path, gpu=gpu)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.musicnn_classes = _load_musicnn_classes(data_dir)
            self.mel_matrix = _load_mel_matrix(data_dir)
            self.rules = rules_engine.load_rules(os.path.join(s, "musicnn_rules.yml"))
            log.info("Audio classifier initialized (MusicNN fallback)")
        else:
            raise FileNotFoundError(
                f"No audio model found. Expected yamnet.onnx or musicnn.onnx in {models_dir}"
            )

    def preprocess(self, audio_path):
        """Transcode audio file to waveform or mel spectrogram.

        Returns:
            For YAMNet: numpy array [samples] float32 (16kHz mono).
            For MusicNN: numpy array [N, 188, 96, 1] float32 (mel spec batches).
            None if audio is too short.
        """
        if self.use_yamnet:
            return transcode_audio(audio_path, sample_rate=16000,
                                   ffmpeg_binary=self.ffmpeg_binary)
        else:
            audio_data = transcode_audio(audio_path, sample_rate=8000,
                                         ffmpeg_binary=self.ffmpeg_binary)
            mel_spec = compute_mel_spectrogram(audio_data, self.mel_matrix)
            if mel_spec is None:
                return None

            total_frames = mel_spec.shape[0]
            num_batches = total_frames // MUSICNN_BATCH_FRAMES
            if num_batches == 0:
                return None

            # Skip initial silence (same logic as original)
            offset = min(MUSICNN_BATCH_FRAMES * 30, total_frames - num_batches * MUSICNN_BATCH_FRAMES)
            mel_spec = mel_spec[offset:]
            num_batches = mel_spec.shape[0] // MUSICNN_BATCH_FRAMES

            if num_batches == 0:
                return None

            batches = np.stack([
                mel_spec[i * MUSICNN_BATCH_FRAMES:(i + 1) * MUSICNN_BATCH_FRAMES]
                for i in range(num_batches)
            ])
            return batches

    def infer_one(self, preprocessed):
        """Run inference on preprocessed audio data.

        Args:
            preprocessed: Output from preprocess().

        Returns:
            List of label strings.
        """
        if self.use_yamnet:
            return self._infer_yamnet(preprocessed)
        else:
            return self._infer_musicnn(preprocessed)

    def _infer_yamnet(self, waveform):
        """Run YAMNet inference on a waveform."""
        outputs = self.session.run(None, {self.input_name: waveform})
        scores = outputs[0]  # [frames, 521]
        avg_scores = np.mean(scores, axis=0)

        if self.class_names is None:
            # Fallback: return raw indices
            indices = np.argsort(avg_scores)[::-1][:YAMNET_TOP_K]
            return [str(idx) for idx in indices if avg_scores[idx] >= 0.01]

        indices = np.argsort(avg_scores)[::-1][:YAMNET_TOP_K]
        results = [
            {"className": self.class_names[idx], "probability": float(avg_scores[idx])}
            for idx in indices
        ]
        return rules_engine.apply_rules(results, self.rules, uppercase=False)

    def _infer_musicnn(self, batches):
        """Run MusicNN inference on mel spectrogram batches."""
        logits = self.session.run(
            [self.output_name], {self.input_name: batches}
        )[0]

        # Average softmax across batches
        probs_per_batch = softmax(logits, axis=-1)
        avg_probs = np.mean(probs_per_batch, axis=0).flatten()

        indices = np.argsort(avg_probs)[::-1][:MUSICNN_TOP_K]
        results = [
            {"className": self.musicnn_classes[idx], "probability": float(avg_probs[idx])}
            for idx in indices
        ]
        return rules_engine.apply_rules(results, self.rules, uppercase=False)

    def classify(self, audio_path):
        """Convenience: preprocess + infer a single audio file.

        Returns:
            List of label strings, or [] if audio too short.
        """
        preprocessed = self.preprocess(audio_path)
        if preprocessed is None:
            return []
        return self.infer_one(preprocessed)

    def warm_up(self):
        """Run a dummy inference to warm up the session."""
        if self.use_yamnet:
            dummy = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            self.session.run(None, {self.input_name: dummy})
        else:
            dummy = np.zeros((1, MUSICNN_BATCH_FRAMES, 96, 1), dtype=np.float32)
            self.session.run([self.output_name], {self.input_name: dummy})
        log.info("Audio classifier warm-up complete")
