"""Video action classifier using MoViNet-A3 (ONNX).

Preprocessing: FFmpeg -> 256x256 @2fps raw RGB -> normalize [0,1] -> [1, 3, T, 256, 256] (NCTHW)
Postprocessing: softmax -> top-6 -> threshold 0.85

ONNX model: movinet_a3.onnx (PyTorch export, NCTHW format)
  Input:  video   [batch, 3, frames, 256, 256] float32
  Output: logits  [batch, 600] float32
"""

import json
import logging
import os
import subprocess

import numpy as np

from classifiers.base import create_session, softmax

log = logging.getLogger(__name__)

FRAME_SIZE = 256
TOP_K = 6
THRESHOLD = 0.85

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SCRIPT_DIR, "..", "data")


def _load_class_names(data_dir=None):
    """Load Kinetics-600 class names."""
    d = data_dir or _DATA_DIR
    path = os.path.join(d, "kinetics_classes.json")
    with open(path) as f:
        return json.load(f)


def extract_frames(video_path, ffmpeg_binary="/usr/bin/ffmpeg"):
    """Extract frames from video using FFmpeg at 2fps, 256x256 raw RGB.

    Args:
        video_path: Path to video file.
        ffmpeg_binary: Path to ffmpeg binary.

    Returns:
        numpy array [N, H, W, 3] float32 normalized to [0,1], or None on error.
    """
    cores = os.environ.get("RECOGNIZE_CORES", "0")
    cores_arg = ["-threads", cores] if cores and cores != "0" else []

    cmd = [
        ffmpeg_binary, "-t", "30", "-i", video_path,
        "-vf", f"fps=2,scale={FRAME_SIZE}:{FRAME_SIZE}",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        *cores_arg, "-",
    ]

    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    if proc.returncode != 0:
        log.warning("FFmpeg error: %s", proc.stderr.decode()[:200])
        return None

    data = proc.stdout
    frame_bytes = FRAME_SIZE * FRAME_SIZE * 3
    if len(data) < frame_bytes:
        return None

    n_frames = len(data) // frame_bytes
    frames = np.frombuffer(data[:n_frames * frame_bytes], dtype=np.uint8)
    frames = frames.reshape(n_frames, FRAME_SIZE, FRAME_SIZE, 3).astype(np.float32) / 255.0
    return frames


class MoViNetClassifier:
    """MoViNet-A3 video action classifier using ONNX Runtime."""

    def __init__(self, models_dir, gpu=True, data_dir=None,
                 ffmpeg_binary="/usr/bin/ffmpeg"):
        """Initialize the classifier.

        Args:
            models_dir: Directory containing movinet_a3.onnx.
            gpu: Whether to use GPU providers.
            data_dir: Override for class names directory.
            ffmpeg_binary: Path to ffmpeg binary.
        """
        model_path = os.path.join(models_dir, "movinet_a3.onnx")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.session = create_session(model_path, gpu=gpu)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.class_names = _load_class_names(data_dir)
        self.ffmpeg_binary = ffmpeg_binary
        log.info("MoViNet classifier initialized")

    def preprocess(self, video_path):
        """Extract and preprocess video frames.

        Returns:
            numpy array [1, 3, T, 256, 256] float32 (NCTHW), or None on error.
        """
        frames = extract_frames(video_path, self.ffmpeg_binary)
        if frames is None:
            return None
        # Convert from [T, H, W, C] to [1, C, T, H, W] (NCTHW for PyTorch ONNX)
        tensor = np.transpose(frames, (3, 0, 1, 2))  # [C, T, H, W]
        return np.expand_dims(tensor, axis=0)  # [1, C, T, H, W]

    def infer_one(self, video_tensor):
        """Run inference on a preprocessed video tensor.

        Args:
            video_tensor: numpy array [1, 3, T, 256, 256] float32.

        Returns:
            List of action label strings above threshold.
        """
        logits = self.session.run(
            [self.output_name], {self.input_name: video_tensor}
        )[0]
        probs = softmax(logits, axis=-1).flatten()

        indices = np.argsort(probs)[::-1][:TOP_K]
        labels = []
        seen = set()
        for idx in indices:
            if probs[idx] >= THRESHOLD:
                label = self.class_names[idx]
                if label not in seen:
                    seen.add(label)
                    labels.append(label)
        return labels

    def classify(self, video_path):
        """Convenience: preprocess + infer a single video.

        Returns:
            List of action label strings, or [] on error.
        """
        tensor = self.preprocess(video_path)
        if tensor is None:
            return []
        return self.infer_one(tensor)

    def warm_up(self):
        """Run a dummy inference to warm up the session."""
        # Minimal: 1 frame
        dummy = np.zeros((1, 3, 1, FRAME_SIZE, FRAME_SIZE), dtype=np.float32)
        self.session.run([self.output_name], {self.input_name: dummy})
        log.info("MoViNet warm-up complete")
