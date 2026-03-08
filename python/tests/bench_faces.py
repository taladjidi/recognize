"""Benchmark: InsightFace (buffalo_l) face detection + embedding.

Measures model load, warm-up, per-image detection, and embedding extraction.
Run: python tests/bench_faces.py [--no-gpu]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmark_utils import TimingContext, BenchmarkResult, BenchmarkReport, gpu_snapshot

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
TEST_RES = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "res")

FACE_DIR = os.path.join(TEST_RES, "FaceID-550")
FACE_IMAGES = []
for person in sorted(os.listdir(FACE_DIR)) if os.path.isdir(FACE_DIR) else []:
    person_dir = os.path.join(FACE_DIR, person)
    if os.path.isdir(person_dir):
        for f in sorted(os.listdir(person_dir))[:3]:
            FACE_IMAGES.append(os.path.join(person_dir, f))

# Also include non-face image to measure zero-detection case
NON_FACE_IMAGE = os.path.join(TEST_RES, "alpine.JPG")


def bench_faces(n_repeats=5, gpu=True):
    from classifiers.faces import FaceClassifier

    report = BenchmarkReport("Faces (InsightFace buffalo_l ONNX)")

    # Model load
    with TimingContext("model_load") as t_load:
        clf = FaceClassifier(MODELS_DIR, gpu=gpu)
    gpu_after_load = gpu_snapshot()

    # Warm-up
    with TimingContext("warmup") as t_warmup:
        clf.warm_up()

    report.add(BenchmarkResult(
        classifier="faces",
        file_path="(model load)",
        model_load_s=t_load.elapsed,
        inference_s=t_warmup.elapsed,
        is_warmup=True,
        gpu_mem_mb=gpu_after_load["memory_used_mb"],
    ))

    # Face images
    images = FACE_IMAGES[:n_repeats] if FACE_IMAGES else []
    for img_path in images:
        with TimingContext("preprocess") as t_pre:
            img_bgr = clf.preprocess(img_path)

        with TimingContext("inference") as t_inf:
            faces = clf.infer_one(img_bgr)

        gpu_snap = gpu_snapshot()
        report.add(BenchmarkResult(
            classifier="faces",
            file_path=os.path.basename(img_path),
            preprocess_s=t_pre.elapsed,
            inference_s=t_inf.elapsed,
            total_s=t_pre.elapsed + t_inf.elapsed,
            gpu_mem_mb=gpu_snap["memory_used_mb"],
            extra={"n_faces": len(faces)},
        ))

    # Non-face image
    if os.path.isfile(NON_FACE_IMAGE):
        with TimingContext("preprocess") as t_pre:
            img_bgr = clf.preprocess(NON_FACE_IMAGE)
        with TimingContext("inference") as t_inf:
            faces = clf.infer_one(img_bgr)
        report.add(BenchmarkResult(
            classifier="faces",
            file_path="alpine.JPG (no faces)",
            preprocess_s=t_pre.elapsed,
            inference_s=t_inf.elapsed,
            total_s=t_pre.elapsed + t_inf.elapsed,
            extra={"n_faces": len(faces)},
        ))

    return report


def main():
    gpu = "--no-gpu" not in sys.argv
    report = bench_faces(gpu=gpu)
    print(report.summary_table())
    print(report.to_json())


if __name__ == "__main__":
    main()
