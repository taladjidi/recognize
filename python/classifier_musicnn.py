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


def init_model():
    """Initialize audio model for multiprocess pipeline."""
    yamnet_path = os.path.join(MODELS_DIR, "yamnet_saved")
    musicnn_path = os.path.join(MODELS_DIR, "musicnn_saved")
    use_yamnet = os.path.isdir(yamnet_path)

    state = {"use_yamnet": use_yamnet}

    if use_yamnet:
        print("Loading YAMNet model...", file=sys.stderr)
        model = tf.saved_model.load(yamnet_path)
        state["model"] = model
        state["class_names"] = _load_yamnet_class_names(model)
        state["rules"] = rules_engine.load_rules(os.path.join(SRC_DIR, "yamnet_rules.yml"))
        print("Model loaded (YAMNet)", file=sys.stderr)
    elif os.path.isdir(musicnn_path):
        print("YAMNet not found, falling back to MusicNN...", file=sys.stderr)
        loaded = tf.saved_model.load(musicnn_path)
        state["musicnn_model"] = loaded.signatures["serving_default"]
        state["input_key"] = list(state["musicnn_model"].structured_input_signature[1].keys())[0]
        state["output_key"] = list(state["musicnn_model"].structured_outputs.keys())[0]
        state["msd_classes"] = _load_musicnn_classes()
        state["mel_matrix"] = _load_mel_matrix()
        state["rules"] = rules_engine.load_rules(os.path.join(SRC_DIR, "musicnn_rules.yml"))
        print("Model loaded (MusicNN fallback)", file=sys.stderr)
    else:
        raise RuntimeError("No audio model found. Download YAMNet or run convert_models.py.")

    return state


def infer_one(model_state, audio_data):
    """Run inference on preprocessed audio waveform. Returns list of label strings."""
    if model_state["use_yamnet"]:
        model = model_state["model"]
        class_names = model_state["class_names"]
        rules_data = model_state["rules"]

        waveform = tf.constant(audio_data, dtype=tf.float32)
        scores, embeddings, spectrogram = model(waveform)
        avg_scores = tf.reduce_mean(scores, axis=0).numpy()

        indices = np.argsort(avg_scores)[::-1][:YAMNET_TOP_K]
        results = [
            {"className": class_names[idx], "probability": float(avg_scores[idx])}
            for idx in indices
        ]
        return rules_engine.apply_rules(results, rules_data, uppercase=False)
    else:
        labels = _run_musicnn(
            model_state["musicnn_model"], model_state["input_key"],
            model_state["output_key"], audio_data, model_state["mel_matrix"],
            model_state["msd_classes"], model_state["rules"],
        )
        return labels if labels is not None else []


def create_pipeline(paths_iterable):
    """Load model once, yield (path, labels) for each input path.

    labels is a list of genre/tag strings or [] on error.
    """
    yamnet_path = os.path.join(MODELS_DIR, "yamnet_saved")
    musicnn_path = os.path.join(MODELS_DIR, "musicnn_saved")
    use_yamnet = os.path.isdir(yamnet_path)

    model = musicnn_model = input_key = output_key = None
    msd_classes = mel_matrix = None
    class_names = rules_data = None

    if use_yamnet:
        print("Loading YAMNet model...", file=sys.stderr)
        model = tf.saved_model.load(yamnet_path)
        class_names = _load_yamnet_class_names(model)
        rules_data = rules_engine.load_rules(os.path.join(SRC_DIR, "yamnet_rules.yml"))
        print("Model loaded (YAMNet)", file=sys.stderr)
    elif os.path.isdir(musicnn_path):
        print("YAMNet not found, falling back to MusicNN...", file=sys.stderr)
        loaded = tf.saved_model.load(musicnn_path)
        musicnn_model = loaded.signatures["serving_default"]
        input_key = list(musicnn_model.structured_input_signature[1].keys())[0]
        output_key = list(musicnn_model.structured_outputs.keys())[0]
        msd_classes = _load_musicnn_classes()
        mel_matrix = _load_mel_matrix()
        rules_data = rules_engine.load_rules(os.path.join(SRC_DIR, "musicnn_rules.yml"))
        print("Model loaded (MusicNN fallback)", file=sys.stderr)
    else:
        print("ERROR: No audio model found. Download YAMNet or run convert_models.py.", file=sys.stderr)
        sys.exit(1)

    ffmpeg_binary = base_classifier.get_ffmpeg_binary()

    def transcode_fn(p):
        if use_yamnet:
            return transcode_audio_16khz(p, ffmpeg_binary)
        return transcode_audio_8khz(p, ffmpeg_binary)

    for path, audio_data in base_classifier.prefetch_map(paths_iterable, transcode_fn, prefetch=4):
        try:
            if audio_data is None:
                yield path, []
                continue

            if use_yamnet:
                waveform = tf.constant(audio_data, dtype=tf.float32)
                scores, embeddings, spectrogram = model(waveform)

                avg_scores = tf.reduce_mean(scores, axis=0).numpy()

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
                labels = _run_musicnn(
                    musicnn_model, input_key, output_key,
                    audio_data, mel_matrix, msd_classes, rules_data,
                )
                if labels is None:
                    print(f"Audio too short for classification: {path}", file=sys.stderr)
                    yield path, []
                    continue

            yield path, labels

        except Exception as e:
            print(f"Error processing {path}: {e}", file=sys.stderr)
            yield path, []


def main():
    for path, labels in create_pipeline(base_classifier.iter_paths()):
        base_classifier.output_result(labels)


if __name__ == "__main__":
    main()
