"""Benchmark: ImageNet (EfficientNetV2-S) ONNX classifier.

Measures model load, warm-up, single-file, and batched inference latency.
Run: python tests/bench_imagenet.py [--no-gpu]
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmark_utils import TimingContext, BenchmarkResult, BenchmarkReport, gpu_snapshot

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
TEST_RES = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "res")
TEST_IMAGES = [
    os.path.join(TEST_RES, "alpine.JPG"),
    os.path.join(TEST_RES, "casey-lee-awj7sRviVXo-unsplash.jpg"),
    os.path.join(TEST_RES, "eiffeltower.jpg"),
]


def bench_imagenet(n_repeats=10, gpu=True):
    from classifiers.imagenet import ImageNetClassifier

    report = BenchmarkReport("ImageNet (EfficientNetV2-S ONNX)")

    # Model load
    with TimingContext("model_load") as t_load:
        clf = ImageNetClassifier(MODELS_DIR, gpu=gpu)
    gpu_after_load = gpu_snapshot()

    # Warm-up
    with TimingContext("warmup") as t_warmup:
        clf.warm_up()

    report.add(BenchmarkResult(
        classifier="imagenet",
        file_path="(model load)",
        model_load_s=t_load.elapsed,
        inference_s=t_warmup.elapsed,
        is_warmup=True,
        gpu_mem_mb=gpu_after_load["memory_used_mb"],
    ))

    # Single-file inference
    for i in range(n_repeats):
        img_path = TEST_IMAGES[i % len(TEST_IMAGES)]

        with TimingContext("preprocess") as t_pre:
            arr = clf.preprocess(img_path)

        with TimingContext("inference") as t_inf:
            batch = np.expand_dims(arr, 0)
            labels_list = clf.infer_batch(batch)

        labels = labels_list[0]
        gpu_snap = gpu_snapshot()
        report.add(BenchmarkResult(
            classifier="imagenet",
            file_path=os.path.basename(img_path),
            preprocess_s=t_pre.elapsed,
            inference_s=t_inf.elapsed,
            total_s=t_pre.elapsed + t_inf.elapsed,
            gpu_mem_mb=gpu_snap["memory_used_mb"],
            gpu_util_pct=gpu_snap["utilization_pct"],
        ))

    # Batched inference
    batch_size = len(TEST_IMAGES)
    arrays = [clf.preprocess(p) for p in TEST_IMAGES]
    batch = np.stack(arrays)

    with TimingContext("batch_inference") as t_batch:
        clf.infer_batch(batch)

    report.add(BenchmarkResult(
        classifier="imagenet",
        file_path=f"(batch of {batch_size})",
        inference_s=t_batch.elapsed,
        total_s=t_batch.elapsed,
        extra={"batch_size": batch_size,
               "per_image_s": t_batch.elapsed / batch_size},
    ))

    return report


def main():
    gpu = "--no-gpu" not in sys.argv
    report = bench_imagenet(gpu=gpu)
    print(report.summary_table())
    print(report.to_json())


if __name__ == "__main__":
    main()
