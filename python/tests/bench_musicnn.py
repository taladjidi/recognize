"""Benchmark for classifier_musicnn.py — MusicNN audio genre classification.

Measures: model load, FFmpeg transcode, STFT/mel spectrogram computation,
batched inference, softmax averaging + rules postprocessing.

Usage:
    RECOGNIZE_GPU=true python python/tests/bench_musicnn.py
"""

import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(TESTS_DIR)
PROJECT_ROOT = os.path.dirname(PYTHON_DIR)
sys.path.insert(0, PYTHON_DIR)

from benchmark_utils import (
    TimingContext,
    BenchmarkResult,
    BenchmarkReport,
    gpu_snapshot,
    print_err,
)

os.environ.setdefault("RECOGNIZE_GPU", "true")

import gpu_setup

tf = gpu_setup.configure()
import numpy as np

import base_classifier
from classifier_musicnn import (
    MODELS_DIR,
    TOP_K,
    BATCH_FRAMES,
    MSD_CLASSES,
    rules,
    transcode_audio,
    compute_mel_spectrogram,
)
import rules_engine

RES_DIR = os.path.join(PROJECT_ROOT, "tests", "res")


def find_test_audio(repeat=5):
    """Find test audio files, repeat to simulate batch."""
    base = []
    for f in ["Rock_Rejam.mp3"]:
        p = os.path.join(RES_DIR, f)
        if os.path.isfile(p):
            base.append(p)
    if not base:
        print_err("No test audio found in tests/res/")
        sys.exit(1)
    return (base * repeat)[:repeat]


def run_benchmark():
    report = BenchmarkReport("MusicNN (audio genre classification)")

    model_path = os.path.join(MODELS_DIR, "musicnn_saved")
    if not os.path.isdir(model_path):
        print_err(f"Model not found at {model_path}")
        sys.exit(1)

    # Model load
    gpu_before = gpu_snapshot()
    with TimingContext("model_load") as t_load:
        loaded = tf.saved_model.load(model_path)
        model = loaded.signatures["serving_default"]
        input_key = list(model.structured_input_signature[1].keys())[0]
        output_key = list(model.structured_outputs.keys())[0]
    gpu_after = gpu_snapshot()

    print_err(f"Model load: {t_load.elapsed:.3f}s")
    print_err(
        f"GPU memory: {gpu_before['memory_used_mb']:.0f} → {gpu_after['memory_used_mb']:.0f} MB"
    )

    ffmpeg_binary = base_classifier.get_ffmpeg_binary()
    paths = find_test_audio()
    print_err(f"Benchmarking {len(paths)} audio files...")

    for i, path in enumerate(paths):
        is_warmup = i == 0

        # FFmpeg transcode
        with TimingContext("ffmpeg") as t_ffmpeg:
            audio_data = transcode_audio(path, ffmpeg_binary)

        # Mel spectrogram
        with TimingContext("mel_spectrogram") as t_mel:
            mel_spec = compute_mel_spectrogram(audio_data)

        total_frames = mel_spec.shape[0]
        num_batches = total_frames // BATCH_FRAMES
        if num_batches == 0:
            print_err(f"  Audio too short: {path}")
            continue

        # Batch construction
        with TimingContext("preprocess") as t_pre:
            offset = min(BATCH_FRAMES * 30, total_frames - num_batches * BATCH_FRAMES)
            mel_spec = mel_spec[offset:]
            num_batches = mel_spec.shape[0] // BATCH_FRAMES
            batches = tf.stack(
                [
                    mel_spec[j * BATCH_FRAMES : (j + 1) * BATCH_FRAMES]
                    for j in range(num_batches)
                ]
            )

        # Inference (already batched)
        with TimingContext("inference") as t_inf:
            output = model(**{input_key: batches})
            logits = output[output_key]

        # Postprocess
        with TimingContext("postprocess") as t_post:
            logit_batches = tf.split(logits, num_batches, axis=0)
            prob_batches = [tf.nn.softmax(lb) for lb in logit_batches]
            probabilities = (
                tf.reduce_mean(tf.stack(prob_batches), axis=0).numpy().flatten()
            )

            indices = np.argsort(probabilities)[::-1][:TOP_K]
            results = [
                {
                    "className": MSD_CLASSES[idx],
                    "probability": float(probabilities[idx]),
                }
                for idx in indices
            ]
            labels = rules_engine.apply_rules(results, rules, uppercase=False)

        gpu = gpu_snapshot()

        total = (
            t_ffmpeg.elapsed
            + t_mel.elapsed
            + t_pre.elapsed
            + t_inf.elapsed
            + t_post.elapsed
        )
        result = BenchmarkResult(
            classifier="musicnn",
            file_path=os.path.basename(path),
            model_load_s=t_load.elapsed if i == 0 else 0.0,
            preprocess_s=t_pre.elapsed,
            inference_s=t_inf.elapsed,
            postprocess_s=t_post.elapsed,
            total_s=total,
            gpu_mem_mb=gpu["memory_used_mb"],
            gpu_util_pct=gpu["utilization_pct"],
            is_warmup=is_warmup,
            extra={
                "ffmpeg_s": t_ffmpeg.elapsed,
                "mel_spectrogram_s": t_mel.elapsed,
                "n_batches": num_batches,
                "total_frames": int(total_frames),
            },
        )
        report.add(result)

        if is_warmup:
            print_err(
                f"  Warm-up: ffmpeg={t_ffmpeg.elapsed * 1000:.0f}ms mel={t_mel.elapsed * 1000:.0f}ms "
                f"inf={t_inf.elapsed * 1000:.0f}ms ({num_batches} batches) → {labels}"
            )
        elif i == 1:
            print_err(
                f"  Steady:  ffmpeg={t_ffmpeg.elapsed * 1000:.0f}ms mel={t_mel.elapsed * 1000:.0f}ms "
                f"inf={t_inf.elapsed * 1000:.0f}ms"
            )

    return report


if __name__ == "__main__":
    report = run_benchmark()
    print_err(report.summary_table())
    print(report.to_json())
