"""Music genre classifier using YAMNet.

Replaces src/classifier_musicnn.js (filename kept for PHP dispatch).

Preprocessing: FFmpeg → 16kHz mono float32 waveform
Inference: YAMNet handles all audio feature extraction internally
Postprocessing: average sigmoid scores across frames → top-K → yamnet_rules.yml filtering

Falls back to MusicNN if YAMNet model is not found.

Reference: src/musicnn/MusicnnModel.js lines 46-85
"""

import csv
import io
import json
import os
import subprocess
import sys

import gpu_setup

tf = gpu_setup.configure()
import numpy as np
import soundfile as sf

import base_classifier
import rules_engine

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
SRC_DIR = os.path.join(SCRIPT_DIR, "..", "src")

TOP_K = 6  # for MusicNN fallback (15 classes)
YAMNET_TOP_K = 30  # YAMNet has 521 classes; need more to capture genre-specific scores
MAX_DURATION = 120  # seconds


def _load_yamnet_class_names(model):
    """Load class names from YAMNet model assets CSV."""
    csv_path = model.class_map_path().numpy().decode()
    class_names = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_names.append(row["display_name"])
    return class_names


def transcode_audio_16khz(song_path, ffmpeg_binary):
    """Transcode audio to 16kHz mono float32 WAV using FFmpeg (for YAMNet)."""
    cores_arg = []
    cores = os.environ.get("RECOGNIZE_CORES", "0")
    if cores and cores != "0":
        cores_arg = ["-threads", cores]

    cmd = [
        ffmpeg_binary,
        "-i",
        song_path,
        "-f",
        "wav",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        "-t",
        str(MAX_DURATION),
        *cores_arg,
        "-",
    ]

    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {proc.stderr.decode()[:200]}")

    audio_data, sample_rate = sf.read(io.BytesIO(proc.stdout), dtype="float32")
    return audio_data


# ── MusicNN fallback (kept for backward compat if YAMNet not downloaded) ──


BATCH_FRAMES = 188


def _load_musicnn_classes():
    path = os.path.join(DATA_DIR, "musicnn_classes.json")
    with open(path) as f:
        return json.load(f)


def _load_mel_matrix():
    return np.load(os.path.join(DATA_DIR, "mel_matrix.npy"))


def transcode_audio_8khz(song_path, ffmpeg_binary):
    """Transcode audio to 8kHz mono 16-bit PCM WAV (for MusicNN fallback)."""
    cores_arg = []
    cores = os.environ.get("RECOGNIZE_CORES", "0")
    if cores and cores != "0":
        cores_arg = ["-threads", cores]

    cmd = [
        ffmpeg_binary,
        "-i",
        song_path,
        "-f",
        "wav",
        "-ac",
        "1",
        "-ar",
        "8000",
        "-acodec",
        "pcm_s16le",
        "-t",
        str(MAX_DURATION),
        *cores_arg,
        "-",
    ]

    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {proc.stderr.decode()[:200]}")

    audio_data, sample_rate = sf.read(io.BytesIO(proc.stdout), dtype="float32")
    return audio_data


def compute_mel_spectrogram(audio_data, mel_matrix):
    """Compute mel spectrogram matching the JS MusicNN implementation."""
    song = tf.constant(audio_data, dtype=tf.float32)
    stft = tf.signal.stft(song, frame_length=512, frame_step=256, fft_length=512)
    spectrogram = tf.abs(stft)
    mel_mat = tf.constant(mel_matrix, dtype=tf.float32)
    mel_spec = tf.matmul(spectrogram, mel_mat)
    mel_spec = tf.math.log(tf.clip_by_value(mel_spec, 1e-6, np.finfo(np.float64).max))
    mel_spec = tf.expand_dims(mel_spec, -1)
    return mel_spec


def _run_musicnn(model, input_key, output_key, audio_data, mel_matrix, msd_classes, rules_data):
    """Run MusicNN inference pipeline (fallback path)."""
    mel_spec = compute_mel_spectrogram(audio_data, mel_matrix)
    total_frames = mel_spec.shape[0]
    num_batches = total_frames // BATCH_FRAMES
    if num_batches == 0:
        return None  # too short

    offset = min(BATCH_FRAMES * 30, total_frames - num_batches * BATCH_FRAMES)
    mel_spec = mel_spec[offset:]
    num_batches = mel_spec.shape[0] // BATCH_FRAMES

    batches = tf.stack(
        [mel_spec[i * BATCH_FRAMES : (i + 1) * BATCH_FRAMES] for i in range(num_batches)]
    )

    output = model(**{input_key: batches})
    logits = output[output_key]

    logit_batches = tf.split(logits, num_batches, axis=0)
    prob_batches = [tf.nn.softmax(lb) for lb in logit_batches]
    probabilities = tf.reduce_mean(tf.stack(prob_batches), axis=0).numpy().flatten()

    indices = np.argsort(probabilities)[::-1][:TOP_K]
    results = [
        {"className": msd_classes[idx], "probability": float(probabilities[idx])}
        for idx in indices
    ]
    return rules_engine.apply_rules(results, rules_data, uppercase=False)


def main():
    yamnet_path = os.path.join(MODELS_DIR, "yamnet_saved")
    musicnn_path = os.path.join(MODELS_DIR, "musicnn_saved")
    use_yamnet = os.path.isdir(yamnet_path)

    if use_yamnet:
        print("Loading YAMNet model...", file=sys.stderr)
        model = tf.saved_model.load(yamnet_path)
        class_names = _load_yamnet_class_names(model)
        rules_data = rules_engine.load_rules(os.path.join(SRC_DIR, "yamnet_rules.yml"))
        print("Model loaded (YAMNet)", file=sys.stderr)
    elif os.path.isdir(musicnn_path):
        print(
            "YAMNet not found, falling back to MusicNN...", file=sys.stderr
        )
        loaded = tf.saved_model.load(musicnn_path)
        musicnn_model = loaded.signatures["serving_default"]
        input_key = list(musicnn_model.structured_input_signature[1].keys())[0]
        output_key = list(musicnn_model.structured_outputs.keys())[0]
        msd_classes = _load_musicnn_classes()
        mel_matrix = _load_mel_matrix()
        rules_data = rules_engine.load_rules(os.path.join(SRC_DIR, "musicnn_rules.yml"))
        print("Model loaded (MusicNN fallback)", file=sys.stderr)
    else:
        print(
            "ERROR: No audio model found. Download YAMNet or run convert_models.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    ffmpeg_binary = base_classifier.get_ffmpeg_binary()
    paths = base_classifier.get_paths()

    for path in paths:
        try:
            if use_yamnet:
                # YAMNet pipeline: 16kHz mono → model handles preprocessing
                audio_data = transcode_audio_16khz(path, ffmpeg_binary)
                waveform = tf.constant(audio_data, dtype=tf.float32)
                scores, embeddings, spectrogram = model(waveform)

                # Average sigmoid scores across frames
                avg_scores = tf.reduce_mean(scores, axis=0).numpy()

                # Get top-K (use higher K for YAMNet's 521 classes)
                indices = np.argsort(avg_scores)[::-1][:YAMNET_TOP_K]
                results = [
                    {
                        "className": class_names[idx],
                        "probability": float(avg_scores[idx]),
                    }
                    for idx in indices
                ]

                labels = rules_engine.apply_rules(results, rules_data, uppercase=False)
            else:
                # MusicNN fallback pipeline: 8kHz mono → STFT → mel → batched inference
                audio_data = transcode_audio_8khz(path, ffmpeg_binary)
                labels = _run_musicnn(
                    musicnn_model, input_key, output_key,
                    audio_data, mel_matrix, msd_classes, rules_data,
                )
                if labels is None:
                    print(f"Audio too short for classification: {path}", file=sys.stderr)
                    base_classifier.output_error()
                    continue

            base_classifier.output_result(labels)

        except Exception as e:
            print(f"Error processing {path}: {e}", file=sys.stderr)
            base_classifier.output_error()


if __name__ == "__main__":
    main()
