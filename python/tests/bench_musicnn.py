"""Benchmark for classifier_musicnn.py — audio genre classification (YAMNet or MusicNN).

Measures: model load, FFmpeg transcode, inference, postprocessing.

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
import rules_engine
from classifier_musicnn import (
    MODELS_DIR,
    YAMNET_TOP_K,
    SRC_DIR,
    transcode_audio_16khz,
    _load_yamnet_class_names,
)

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
    report = BenchmarkReport("YAMNet (audio genre classification)")

    yamnet_path = os.path.join(MODELS_DIR, "yamnet_saved")
    if not os.path.isdir(yamnet_path):
        print_err(f"YAMNet model not found at {yamnet_path}")
        sys.exit(1)

    # Model load
    gpu_before = gpu_snapshot()
    with TimingContext("model_load") as t_load:
        model = tf.saved_model.load(yamnet_path)
        class_names = _load_yamnet_class_names(model)
        rules_data = rules_engine.load_rules(os.path.join(SRC_DIR, "yamnet_rules.yml"))
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
            audio_data = transcode_audio_16khz(path, ffmpeg_binary)

        # Inference (YAMNet handles preprocessing internally)
        with TimingContext("inference") as t_inf:
            waveform = tf.constant(audio_data, dtype=tf.float32)
            scores, embeddings, spectrogram = model(waveform)

        # Postprocess
        with TimingContext("postprocess") as t_post:
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

        gpu = gpu_snapshot()

        total = t_ffmpeg.elapsed + t_inf.elapsed + t_post.elapsed
        result = BenchmarkResult(
            classifier="yamnet",
            file_path=os.path.basename(path),
            model_load_s=t_load.elapsed if i == 0 else 0.0,
            preprocess_s=0.0,  # YAMNet does preprocessing internally
            inference_s=t_inf.elapsed,
            postprocess_s=t_post.elapsed,
            total_s=total,
            gpu_mem_mb=gpu["memory_used_mb"],
            gpu_util_pct=gpu["utilization_pct"],
            is_warmup=is_warmup,
            extra={
                "ffmpeg_s": t_ffmpeg.elapsed,
                "n_frames": int(scores.shape[0]),
            },
        )
        report.add(result)

        if is_warmup:
            print_err(
                f"  Warm-up: ffmpeg={t_ffmpeg.elapsed * 1000:.0f}ms "
                f"inf={t_inf.elapsed * 1000:.0f}ms ({scores.shape[0]} frames) → {labels}"
            )
        elif i == 1:
            print_err(
                f"  Steady:  ffmpeg={t_ffmpeg.elapsed * 1000:.0f}ms "
                f"inf={t_inf.elapsed * 1000:.0f}ms"
            )

    return report


if __name__ == "__main__":
    report = run_benchmark()
    print_err(report.summary_table())
    print(report.to_json())
