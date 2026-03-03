"""Benchmark for classifier_landmarks.py — 6 regional landmark models.

Measures: model load (x6 v1 Session), preprocessing (PIL resize 321x321),
batched per-region inference timing, top-k postprocessing.

Usage:
    RECOGNIZE_GPU=true python python/tests/bench_landmarks.py
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

from classifier_landmarks import (
    REGIONS,
    TOP_K,
    THRESHOLD,
    BATCH_SIZE,
    load_labels,
    preprocess_image,
    get_top_k,
    _load_models_v1,
)

RES_DIR = os.path.join(PROJECT_ROOT, "tests", "res")
IMAGE_FILES = ["alpine.JPG", "eiffeltower.jpg", "casey-lee-awj7sRviVXo-unsplash.jpg"]


def find_test_images(repeat=7):
    base = [
        os.path.join(RES_DIR, f)
        for f in IMAGE_FILES
        if os.path.isfile(os.path.join(RES_DIR, f))
    ]
    if not base:
        print_err("No test images found in tests/res/")
        sys.exit(1)
    return (base * (repeat + 1))[: len(base) * repeat]


def run_benchmark():
    report = BenchmarkReport("Landmarks (6 regional models, v1 Session + batched)")

    all_labels = load_labels()

    # Load all 6 models using v1 Session API
    gpu_before = gpu_snapshot()
    with TimingContext("model_load_all") as t_load:
        sessions, input_names, output_names = _load_models_v1()
    gpu_after = gpu_snapshot()

    print_err(f"Loaded {len(sessions)} models in {t_load.elapsed:.3f}s")
    print_err(
        f"GPU memory: {gpu_before['memory_used_mb']:.0f} → {gpu_after['memory_used_mb']:.0f} MB"
    )

    paths = find_test_images()
    print_err(f"Benchmarking {len(paths)} images in batches of {BATCH_SIZE}...")

    batch_idx = 0
    for batch_start in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[batch_start : batch_start + BATCH_SIZE]
        is_warmup = batch_idx == 0
        n = len(batch_paths)

        # Preprocess batch
        with TimingContext("preprocess") as t_pre:
            arrays = [preprocess_image(p) for p in batch_paths]
            batch_tensor = np.stack(arrays, axis=0)

        # Batched per-region inference (6 calls for entire batch)
        region_times = {}
        image_results = [[] for _ in range(n)]
        with TimingContext("inference_total") as t_inf:
            for region in REGIONS:
                with TimingContext(f"inference_{region}") as t_region:
                    all_values = sessions[region].run(
                        output_names[region],
                        feed_dict={input_names[region]: batch_tensor},
                    )
                region_times[region] = t_region.elapsed

                for vi in range(n):
                    values = all_values[vi].flatten()
                    results = get_top_k(values, TOP_K, all_labels[region])
                    for r in results:
                        if r["probability"] >= THRESHOLD:
                            image_results[vi].append(r)

        # Postprocess
        with TimingContext("postprocess") as t_post:
            for vi in range(n):
                image_results[vi].sort(key=lambda x: x["probability"], reverse=True)

        gpu = gpu_snapshot()

        result = BenchmarkResult(
            classifier="landmarks",
            file_path=f"batch_{batch_idx} ({n} files)",
            model_load_s=t_load.elapsed if batch_idx == 0 else 0.0,
            preprocess_s=t_pre.elapsed / n,
            inference_s=t_inf.elapsed / n,
            postprocess_s=t_post.elapsed / n,
            total_s=(t_pre.elapsed + t_inf.elapsed + t_post.elapsed) / n,
            gpu_mem_mb=gpu["memory_used_mb"],
            gpu_util_pct=gpu["utilization_pct"],
            is_warmup=is_warmup,
            extra=region_times,
        )
        report.add(result)

        if is_warmup:
            region_summary = " ".join(
                f"{r.split('_')[-1][:3]}={t * 1000:.0f}ms"
                for r, t in region_times.items()
            )
            print_err(
                f"  Warm-up batch ({n}): pre={t_pre.elapsed * 1000:.0f}ms inf={t_inf.elapsed * 1000:.0f}ms [{region_summary}]"
            )
        else:
            print_err(
                f"  Batch {batch_idx} ({n}): pre={t_pre.elapsed * 1000:.0f}ms inf={t_inf.elapsed * 1000:.0f}ms "
                f"({t_inf.elapsed / n * 1000:.0f}ms/file)"
            )

        batch_idx += 1

    return report


if __name__ == "__main__":
    report = run_benchmark()
    print_err(report.summary_table())
    print(report.to_json())
