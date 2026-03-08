"""Benchmark: MoViNet-A3 (ONNX) video action classifier.

Measures model load, warm-up, FFmpeg extraction, inference.
Run: python tests/bench_movinet.py [--no-gpu]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmark_utils import TimingContext, BenchmarkResult, BenchmarkReport, gpu_snapshot

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
TEST_RES = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "res")
TEST_VIDEO = os.path.join(TEST_RES, "jumpingjack.gif")  # Short animated GIF works as video


def bench_movinet(n_repeats=3, gpu=True):
    from classifiers.movinet import MoViNetClassifier

    report = BenchmarkReport("MoViNet-A3 (ONNX)")

    # Model load
    with TimingContext("model_load") as t_load:
        clf = MoViNetClassifier(MODELS_DIR, gpu=gpu)
    gpu_after_load = gpu_snapshot()

    # Warm-up
    with TimingContext("warmup") as t_warmup:
        clf.warm_up()

    report.add(BenchmarkResult(
        classifier="movinet",
        file_path="(model load)",
        model_load_s=t_load.elapsed,
        inference_s=t_warmup.elapsed,
        is_warmup=True,
        gpu_mem_mb=gpu_after_load["memory_used_mb"],
    ))

    if not os.path.isfile(TEST_VIDEO):
        print(f"Test video not found: {TEST_VIDEO}", file=sys.stderr)
        return report

    for i in range(n_repeats):
        with TimingContext("total") as t_total:
            with TimingContext("classify") as t_cls:
                labels = clf.classify(TEST_VIDEO)

        gpu_snap = gpu_snapshot()
        report.add(BenchmarkResult(
            classifier="movinet",
            file_path=os.path.basename(TEST_VIDEO),
            inference_s=t_cls.elapsed,
            total_s=t_total.elapsed,
            gpu_mem_mb=gpu_snap["memory_used_mb"],
            extra={"labels": str(labels)},
        ))

    return report


def main():
    gpu = "--no-gpu" not in sys.argv
    report = bench_movinet(gpu=gpu)
    print(report.summary_table())
    print(report.to_json())


if __name__ == "__main__":
    main()
