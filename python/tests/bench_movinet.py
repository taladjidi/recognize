"""Benchmark for classifier_movinet.py — MoViNet video action classification.

Measures: model load, FFmpeg transcode + frame extraction, inference, postprocessing.

Usage:
    RECOGNIZE_GPU=true python python/tests/bench_movinet.py
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
from classifier_movinet import (
    MODELS_DIR,
    TOP_K,
    THRESHOLD,
    KINETICS_CLASSES,
    extract_frames,
)

RES_DIR = os.path.join(PROJECT_ROOT, "tests", "res")


def find_test_videos(repeat=5):
    """Find test video files, repeat to simulate batch."""
    base = []
    for f in ["jumpingjack.gif"]:
        p = os.path.join(RES_DIR, f)
        if os.path.isfile(p):
            base.append(p)
    if not base:
        print_err("No test videos found in tests/res/")
        sys.exit(1)
    return (base * repeat)[:repeat]


def run_benchmark():
    report = BenchmarkReport("MoViNet (video action classification)")

    model_path = os.path.join(MODELS_DIR, "movinet-a3")
    if not os.path.isdir(model_path):
        print_err(f"Model not found at {model_path}")
        sys.exit(1)

    # Model load
    gpu_before = gpu_snapshot()
    with TimingContext("model_load") as t_load:
        loaded = tf.saved_model.load(model_path)
        model = loaded.signatures["serving_default"]
    gpu_after = gpu_snapshot()

    print_err(f"Model load: {t_load.elapsed:.3f}s")
    print_err(
        f"GPU memory: {gpu_before['memory_used_mb']:.0f} → {gpu_after['memory_used_mb']:.0f} MB"
    )

    ffmpeg_binary = base_classifier.get_ffmpeg_binary()
    paths = find_test_videos()
    print_err(f"Benchmarking {len(paths)} videos...")

    for i, path in enumerate(paths):
        is_warmup = i == 0

        # FFmpeg transcode + raw frame extraction (combined in new API)
        with TimingContext("extract") as t_extract:
            frames = extract_frames(path, ffmpeg_binary)

        if frames is None or len(frames) == 0:
            print_err(f"  No frames extracted from {path}")
            continue

        n_frames = len(frames)

        # Numpy stacking (frames already normalized)
        with TimingContext("preprocess") as t_pre:
            frame_batch = np.expand_dims(frames, axis=0)

        # Inference
        with TimingContext("inference") as t_inf:
            output = model(image=tf.constant(frame_batch))
            logits = output["classifier_head"]
            probs = tf.nn.softmax(logits).numpy().flatten()

        # Postprocess
        with TimingContext("postprocess") as t_post:
            indices = np.argsort(probs)[::-1][:TOP_K]
            labels = list(
                dict.fromkeys(
                    KINETICS_CLASSES[idx] for idx in indices if probs[idx] >= THRESHOLD
                )
            )

        gpu = gpu_snapshot()

        total = t_extract.elapsed + t_pre.elapsed + t_inf.elapsed + t_post.elapsed
        result = BenchmarkResult(
            classifier="movinet",
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
                "extract_s": t_extract.elapsed,
                "n_frames": n_frames,
            },
        )
        report.add(result)

        if is_warmup:
            print_err(
                f"  Warm-up: extract={t_extract.elapsed * 1000:.0f}ms "
                f"inf={t_inf.elapsed * 1000:.0f}ms ({n_frames} frames) → {labels}"
            )
        elif i == 1:
            print_err(
                f"  Steady:  extract={t_extract.elapsed * 1000:.0f}ms "
                f"inf={t_inf.elapsed * 1000:.0f}ms"
            )

    return report


if __name__ == "__main__":
    report = run_benchmark()
    print_err(report.summary_table())
    print(report.to_json())
