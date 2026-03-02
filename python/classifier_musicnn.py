"""Music genre classifier using MusicNN.

Replaces src/classifier_musicnn.js.

Preprocessing: FFmpeg → 8kHz mono WAV → STFT(512,256) → mel matrix multiply → log → 188-frame batches
Postprocessing: softmax per batch → average → top-6 → musicnn_rules.yml filtering

Reference: src/musicnn/MusicnnModel.js lines 46-85
"""
import json
import math
import os
import subprocess
import sys

import gpu_setup
tf = gpu_setup.configure()
import numpy as np
import soundfile as sf
import io

import base_classifier
import rules_engine

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
SRC_DIR = os.path.join(SCRIPT_DIR, '..', 'src')

TOP_K = 6
BATCH_FRAMES = 188
MAX_DURATION = 120  # seconds

# Load class names
with open(os.path.join(DATA_DIR, 'musicnn_classes.json')) as f:
    MSD_CLASSES = json.load(f)

# Load mel matrix (exact copy from JS)
MEL_MATRIX = np.load(os.path.join(DATA_DIR, 'mel_matrix.npy'))

# Load rules
rules = rules_engine.load_rules(os.path.join(SRC_DIR, 'musicnn_rules.yml'))


def get_ffmpeg_binary():
    """Get ffmpeg binary path from env or system."""
    ffmpeg = os.environ.get('FFMPEG_BINARY', '')
    if ffmpeg and os.path.isfile(ffmpeg):
        return ffmpeg
    import shutil
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        return ffmpeg
    raise RuntimeError('ffmpeg not found. Set FFMPEG_BINARY env var.')


def transcode_audio(song_path, ffmpeg_binary):
    """Transcode audio to 8kHz mono 16-bit PCM WAV using FFmpeg."""
    cores_arg = []
    cores = os.environ.get('RECOGNIZE_CORES', '0')
    if cores and cores != '0':
        cores_arg = ['-threads', cores]

    cmd = [
        ffmpeg_binary,
        '-i', song_path,
        '-f', 'wav',
        '-ac', '1',
        '-ar', '8000',
        '-acodec', 'pcm_s16le',
        '-t', str(MAX_DURATION),
        *cores_arg,
        '-',
    ]

    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f'FFmpeg error: {proc.stderr.decode()[:200]}')

    # Decode WAV from stdout
    audio_data, sample_rate = sf.read(io.BytesIO(proc.stdout), dtype='float32')
    return audio_data


def compute_mel_spectrogram(audio_data):
    """Compute mel spectrogram matching the JS implementation.

    JS: tf.signal.stft(song, 512, 256, 512) → abs → matMul(melMatrix) → log(clip(1e-6, MAX))
    """
    song = tf.constant(audio_data, dtype=tf.float32)

    # STFT: frame_length=512, frame_step=256, fft_length=512
    stft = tf.signal.stft(song, frame_length=512, frame_step=256, fft_length=512)
    spectrogram = tf.abs(stft)

    # Mel matrix multiply
    mel_matrix = tf.constant(MEL_MATRIX, dtype=tf.float32)
    mel_spec = tf.matmul(spectrogram, mel_matrix)

    # Log with clipping
    mel_spec = tf.math.log(tf.clip_by_value(mel_spec, 1e-6, np.finfo(np.float64).max))

    # Add trailing dimension: [frames, 96, 1]
    mel_spec = tf.expand_dims(mel_spec, -1)

    return mel_spec


def main():
    model_path = os.path.join(MODELS_DIR, 'musicnn_saved')
    if not os.path.isdir(model_path):
        print(f'ERROR: Model not found at {model_path}. Run convert_models.py first.', file=sys.stderr)
        sys.exit(1)

    print('Loading MusicNN model...', file=sys.stderr)
    loaded = tf.saved_model.load(model_path)
    model = loaded.signatures['serving_default']
    input_key = list(model.structured_input_signature[1].keys())[0]
    output_key = list(model.structured_outputs.keys())[0]
    print('Model loaded', file=sys.stderr)

    ffmpeg_binary = get_ffmpeg_binary()
    paths = base_classifier.get_paths()

    for path in paths:
        try:
            audio_data = transcode_audio(path, ffmpeg_binary)
            mel_spec = compute_mel_spectrogram(audio_data)

            total_frames = mel_spec.shape[0]

            # Align to BATCH_FRAMES: slice from offset so remainder is divisible
            # JS: slice from min(188*30, total - floor(total/188)*188)
            num_batches = total_frames // BATCH_FRAMES
            if num_batches == 0:
                print(f'Audio too short for classification: {path}', file=sys.stderr)
                base_classifier.output_error()
                continue

            offset = min(BATCH_FRAMES * 30, total_frames - num_batches * BATCH_FRAMES)
            mel_spec = mel_spec[offset:]
            num_batches = mel_spec.shape[0] // BATCH_FRAMES

            # Split into batches and stack: [num_batches, 188, 96, 1]
            batches = tf.stack([mel_spec[i * BATCH_FRAMES:(i + 1) * BATCH_FRAMES] for i in range(num_batches)])

            # Run inference via serving_default signature
            output = model(**{input_key: batches})
            logits = output[output_key]

            # Softmax per batch, then average
            logit_batches = tf.split(logits, num_batches, axis=0)
            prob_batches = [tf.nn.softmax(lb) for lb in logit_batches]
            probabilities = tf.reduce_mean(tf.stack(prob_batches), axis=0).numpy().flatten()

            # Get top-K
            indices = np.argsort(probabilities)[::-1][:TOP_K]
            results = []
            for idx in indices:
                results.append({
                    'className': MSD_CLASSES[idx],
                    'probability': float(probabilities[idx]),
                })

            # Apply rules (no uppercase for musicnn)
            labels = rules_engine.apply_rules(results, rules, uppercase=False)

            base_classifier.output_result(labels)

        except Exception as e:
            print(f'Error processing {path}: {e}', file=sys.stderr)
            base_classifier.output_error()


if __name__ == '__main__':
    main()
