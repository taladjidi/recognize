"""Video action classifier using MoViNet-A3.

Replaces src/classifier_movinet.js.
Model is already in SavedModel format — no conversion needed.

Preprocessing: FFmpeg → 176x176 @2fps raw RGB → normalize [0,1] → [1, N, 176, 176, 3]
Postprocessing: softmax → top-6 → threshold 0.85

Reference: src/movinet/MovinetModel.js lines 49-96
"""

import json
import os
import subprocess
import sys

import gpu_setup

tf = gpu_setup.configure()
import numpy as np

import base_classifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

FRAME_SIZE = 176
TOP_K = 6
THRESHOLD = 0.85

# Load class names
with open(os.path.join(DATA_DIR, "kinetics_classes.json")) as f:
    KINETICS_CLASSES = json.load(f)


def extract_frames(video_path, ffmpeg_binary):
    """Extract frames from video using FFmpeg at 2fps, 176x176 raw RGB.

    Outputs raw RGB24 pixels directly instead of MJPEG, avoiding the
    JPEG encode (FFmpeg) + JPEG decode (PIL) overhead.
    """
    cores_arg = []
    cores = os.environ.get("RECOGNIZE_CORES", "0")
    if cores and cores != "0":
        cores_arg = ["-threads", cores]

    cmd = [
        ffmpeg_binary,
        "-t",
        "30",
        "-i",
        video_path,
        "-vf",
        f"fps=2,scale={FRAME_SIZE}:{FRAME_SIZE}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        *cores_arg,
        "-",
    ]

    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        print(f"FFmpeg error: {proc.stderr.decode()[:200]}", file=sys.stderr)
        return None

    data = proc.stdout
    frame_bytes = FRAME_SIZE * FRAME_SIZE * 3
    if len(data) < frame_bytes:
        return None

    # Reshape raw bytes directly into [N, H, W, 3] float32 normalized to [0,1]
    n_frames = len(data) // frame_bytes
    frames = np.frombuffer(data[: n_frames * frame_bytes], dtype=np.uint8)
    frames = (
        frames.reshape(n_frames, FRAME_SIZE, FRAME_SIZE, 3).astype(np.float32) / 255.0
    )
    return frames


def main():
    model_path = os.path.join(MODELS_DIR, "movinet-a3")
    if not os.path.isdir(model_path):
        print(f"ERROR: Model not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    print("Loading MoViNet model...", file=sys.stderr)
    loaded = tf.saved_model.load(model_path)
    model = loaded.signatures["serving_default"]
    print("Model loaded", file=sys.stderr)

    ffmpeg_binary = base_classifier.get_ffmpeg_binary()
    paths = base_classifier.get_paths()

    for path in paths:
        try:
            frames = extract_frames(path, ffmpeg_binary)
            if frames is None or len(frames) == 0:
                base_classifier.output_error()
                continue

            # Add batch dim: [1, N, 176, 176, 3]
            frame_batch = np.expand_dims(frames, axis=0)

            output = model(image=tf.constant(frame_batch))
            logits = output["classifier_head"]
            probs = tf.nn.softmax(logits).numpy().flatten()

            # Get top-K above threshold
            indices = np.argsort(probs)[::-1][:TOP_K]
            labels = []
            seen = set()
            for idx in indices:
                if probs[idx] >= THRESHOLD:
                    label = KINETICS_CLASSES[idx]
                    if label not in seen:
                        seen.add(label)
                        labels.append(label)

            base_classifier.output_result(labels)

        except Exception as e:
            print(f"Error processing {path}: {e}", file=sys.stderr)
            base_classifier.output_error()


if __name__ == "__main__":
    main()
