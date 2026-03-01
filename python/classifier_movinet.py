"""Video action classifier using MoViNet-A3.

Replaces src/classifier_movinet.js.
Model is already in SavedModel format — no conversion needed.

Preprocessing: FFmpeg → 176x176 @2fps → normalize [0,1] → [1, N, 176, 176, 3]
Postprocessing: softmax → top-6 → threshold 0.85

Reference: src/movinet/MovinetModel.js lines 49-96
"""
import json
import os
import subprocess
import sys
import tempfile

import gpu_setup
tf = gpu_setup.configure()
import numpy as np
from PIL import Image
import io

import base_classifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

FRAME_SIZE = 176
TOP_K = 6
THRESHOLD = 0.85

# Load class names
with open(os.path.join(DATA_DIR, 'kinetics_classes.json')) as f:
    KINETICS_CLASSES = json.load(f)


def get_ffmpeg_binary():
    """Get ffmpeg binary path from env or system."""
    ffmpeg = os.environ.get('FFMPEG_BINARY', '')
    if ffmpeg and os.path.isfile(ffmpeg):
        return ffmpeg
    # Try system ffmpeg
    import shutil
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        return ffmpeg
    raise RuntimeError('ffmpeg not found. Set FFMPEG_BINARY env var.')


def extract_frames(video_path, ffmpeg_binary):
    """Extract frames from video using FFmpeg at 2fps, 176x176."""
    cores_arg = []
    cores = os.environ.get('RECOGNIZE_CORES', '0')
    if cores and cores != '0':
        cores_arg = ['-threads', cores]

    cmd = [
        ffmpeg_binary,
        '-t', '30',
        '-i', video_path,
        '-s', f'{FRAME_SIZE}x{FRAME_SIZE}',
        '-vf', 'fps=2',
        '-c:v', 'mjpeg',
        '-f', 'image2pipe',
        *cores_arg,
        '-',
    ]

    print('Starting transcoding', file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True)
    print('Finished transcoding', file=sys.stderr)

    # Split MJPEG stream into individual JPEG frames
    data = proc.stdout
    frames = []
    # JPEG files start with FF D8 and end with FF D9
    start = 0
    while True:
        idx = data.find(b'\xff\xd8', start)
        if idx == -1:
            break
        end = data.find(b'\xff\xd9', idx + 2)
        if end == -1:
            break
        end += 2  # include the FF D9 marker
        frames.append(data[idx:end])
        start = end

    return frames


def decode_frames(frame_buffers):
    """Decode JPEG buffers to numpy arrays."""
    tensors = []
    for i, buf in enumerate(frame_buffers):
        img = Image.open(io.BytesIO(buf))
        img = img.convert('RGB')
        arr = np.array(img, dtype=np.float32)
        tensors.append(arr)
        print(f'decoded {i+1}/{len(frame_buffers)} images', file=sys.stderr)
    return tensors


def get_top_k(values, k):
    """Get top-k classes and probabilities."""
    indices = np.argsort(values)[::-1][:k]
    results = []
    for idx in indices:
        results.append({
            'className': KINETICS_CLASSES[idx],
            'probability': float(values[idx]),
        })
    return results


def main():
    model_path = os.path.join(MODELS_DIR, 'movinet-a3')
    if not os.path.isdir(model_path):
        print(f'ERROR: Model not found at {model_path}', file=sys.stderr)
        sys.exit(1)

    print('Loading MoViNet model...', file=sys.stderr)
    model = tf.saved_model.load(model_path)
    print('Model loaded', file=sys.stderr)

    ffmpeg_binary = get_ffmpeg_binary()
    paths = base_classifier.get_paths()

    for path in paths:
        try:
            frame_buffers = extract_frames(path, ffmpeg_binary)
            if not frame_buffers:
                base_classifier.output_error()
                continue

            frame_arrays = decode_frames(frame_buffers)

            # Stack frames: [N, 176, 176, 3], normalize to [0, 1]
            frame_tensor = np.stack(frame_arrays, axis=0) / 255.0
            # Add batch dim: [1, N, 176, 176, 3]
            frame_batch = np.expand_dims(frame_tensor, axis=0).astype(np.float32)

            # Run inference
            input_tensor = tf.constant(frame_batch)
            logits = model(input_tensor)
            if isinstance(logits, dict):
                # Some SavedModel formats return dicts
                logits = list(logits.values())[0]
            probs = tf.nn.softmax(logits).numpy().flatten()

            # Get top-K
            results = get_top_k(probs, TOP_K)

            # Filter by threshold
            labels = []
            for result in results:
                print(repr(result), file=sys.stderr)
                if result['probability'] >= THRESHOLD:
                    labels.append(result['className'])

            # Deduplicate
            seen = set()
            unique_labels = []
            for label in labels:
                if label not in seen:
                    seen.add(label)
                    unique_labels.append(label)

            base_classifier.output_result(unique_labels)

        except Exception as e:
            print(f'Error processing {path}: {e}', file=sys.stderr)
            base_classifier.output_error()


if __name__ == '__main__':
    main()
