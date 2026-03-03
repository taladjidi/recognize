"""Video action classifier using MoViNet-A3 (Stream variant).

Replaces src/classifier_movinet.js.
Model is already in SavedModel format — no conversion needed.

Preprocessing: FFmpeg → 256x256 @2fps raw RGB → normalize [0,1] → frame-by-frame [1, 1, 256, 256, 3]
Postprocessing: softmax → top-6 → threshold 0.85

The stream variant uses (2+1)D convolutions (conv2d) instead of Conv3D,
avoiding the GPU segfault with TF 2.20. Falls back to base model on CPU
if stream model is not found.

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

FRAME_SIZE = 256
TOP_K = 6
THRESHOLD = 0.85

# Load class names
with open(os.path.join(DATA_DIR, "kinetics_classes.json")) as f:
    KINETICS_CLASSES = json.load(f)


def extract_frames(video_path, ffmpeg_binary):
    """Extract frames from video using FFmpeg at 2fps, 256x256 raw RGB.

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


def _load_stream_model(model_path):
    """Load MoViNet-A3 Stream model. Returns (init_states_fn, call_fn)."""
    loaded = tf.saved_model.load(model_path)
    init_fn = loaded.signatures["init_states"]
    call_fn = loaded.signatures["call"]
    return init_fn, call_fn


def _load_base_model(model_path):
    """Load MoViNet-A3 Base model (CPU fallback). Returns signature fn."""
    with tf.device("/CPU:0"):
        loaded = tf.saved_model.load(model_path)
        return loaded.signatures["serving_default"]


def _infer_stream(init_fn, call_fn, frames):
    """Run streaming inference: init states, feed frames one-by-one, return final logits."""
    # Initialize states for input shape [1, 1, H, W, 3]
    states = init_fn(input_shape=tf.constant([1, 1, FRAME_SIZE, FRAME_SIZE, 3]))

    # Feed frames one at a time
    logits = None
    for i in range(len(frames)):
        frame = tf.constant(frames[i : i + 1][np.newaxis])  # [1, 1, H, W, 3]
        inputs = {**states, "image": frame}
        output = call_fn(**inputs)
        # Separate logits from updated states
        logits = output["logits"]
        states = {k: v for k, v in output.items() if k != "logits"}

    return logits


def _infer_base(model, frames):
    """Run base model inference: all frames at once (CPU only)."""
    frame_batch = np.expand_dims(frames, axis=0)  # [1, N, H, W, 3]
    with tf.device("/CPU:0"):
        output = model(image=tf.constant(frame_batch))
        return output["classifier_head"]


def main():
    stream_path = os.path.join(MODELS_DIR, "movinet-a3-stream")
    base_path = os.path.join(MODELS_DIR, "movinet-a3")
    use_stream = os.path.isdir(stream_path)

    if use_stream:
        print("Loading MoViNet-A3 Stream model...", file=sys.stderr)
        init_fn, call_fn = _load_stream_model(stream_path)
        print("Model loaded (stream)", file=sys.stderr)
    elif os.path.isdir(base_path):
        print(
            "Stream model not found, falling back to base model (CPU)...",
            file=sys.stderr,
        )
        base_model = _load_base_model(base_path)
        print("Model loaded (base, CPU)", file=sys.stderr)
    else:
        print(
            f"ERROR: No MoViNet model found at {stream_path} or {base_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    ffmpeg_binary = base_classifier.get_ffmpeg_binary()
    paths = base_classifier.get_paths()

    for path in paths:
        try:
            frames = extract_frames(path, ffmpeg_binary)
            if frames is None or len(frames) == 0:
                base_classifier.output_error()
                continue

            if use_stream:
                logits = _infer_stream(init_fn, call_fn, frames)
            else:
                logits = _infer_base(base_model, frames)

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
