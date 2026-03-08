"""End-to-end pipeline benchmark.

Simulates the full service flow: pending rows → resolve paths → classify →
submit results. Uses a real SQLite DB and real models but mock NC API.

Run: python tests/bench_e2e.py [--no-gpu] [--n-files N]
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmark_utils import TimingContext, gpu_snapshot

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
TEST_RES = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "res")


def build_test_files():
    """Collect test files from tests/res/ directory."""
    files = []
    for name in os.listdir(TEST_RES):
        path = os.path.join(TEST_RES, name)
        if not os.path.isfile(path):
            continue
        ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
        if ext in ("jpg", "jpeg", "png", "webp", "bmp"):
            files.append({"path": path, "mimetype": "image/jpeg"})
        elif ext in ("gif",):
            files.append({"path": path, "mimetype": "image/gif"})
        elif ext in ("mp3", "ogg", "flac", "wav"):
            files.append({"path": path, "mimetype": "audio/mpeg"})
        elif ext in ("mp4", "webm", "mkv", "avi"):
            files.append({"path": path, "mimetype": "video/mp4"})

    # Add face images
    face_dir = os.path.join(TEST_RES, "FaceID-550")
    if os.path.isdir(face_dir):
        for person in sorted(os.listdir(face_dir)):
            person_dir = os.path.join(face_dir, person)
            if not os.path.isdir(person_dir):
                continue
            for f in sorted(os.listdir(person_dir))[:2]:
                files.append({
                    "path": os.path.join(person_dir, f),
                    "mimetype": "image/jpeg",
                })

    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--n-files", type=int, default=0,
                        help="Limit number of files (0=all)")
    args = parser.parse_args()

    from pipeline import Pipeline
    from db import DB

    test_files = build_test_files()
    if args.n_files > 0:
        test_files = test_files[:args.n_files]

    print(f"Test files: {len(test_files)}", file=sys.stderr)
    for f in test_files:
        print(f"  {os.path.basename(f['path'])} ({f['mimetype']})", file=sys.stderr)

    # Build a minimal config for SQLite (in-memory)
    import tempfile
    tmp = tempfile.mkdtemp()
    config = {
        "dbtype": "sqlite3",
        "dbhost": "",
        "dbuser": "",
        "dbpassword": "",
        "dbname": os.path.join(tmp, "bench.db"),
        "dbtableprefix": "oc_",
        "datadirectory": "/tmp/data",
    }

    # Init pipeline
    with TimingContext("pipeline_init") as t_init:
        pipeline = Pipeline(config, MODELS_DIR, gpu=not args.no_gpu)
        pipeline.enable_classifiers()

    print(f"Pipeline init: {t_init.elapsed:.2f}s", file=sys.stderr)
    gpu_init = gpu_snapshot()

    # Build pending-like rows (simulate what service._main_loop produces)
    rows = []
    for i, f in enumerate(test_files):
        rows.append({
            "id": i + 1,
            "file_id": 1000 + i,
            "path": f["path"],
            "mimetype": f["mimetype"],
        })

    # Process
    with TimingContext("process_batch") as t_batch:
        processed = pipeline.process_batch(rows)

    gpu_after = gpu_snapshot()

    # Results
    results = {
        "n_files": len(test_files),
        "n_processed": processed,
        "pipeline_init_s": round(t_init.elapsed, 3),
        "process_batch_s": round(t_batch.elapsed, 3),
        "throughput_files_per_s": round(processed / t_batch.elapsed, 2) if t_batch.elapsed > 0 else 0,
        "avg_per_file_ms": round(t_batch.elapsed / max(processed, 1) * 1000, 1),
        "gpu_mem_after_init_mb": gpu_init["memory_used_mb"],
        "gpu_mem_after_batch_mb": gpu_after["memory_used_mb"],
    }

    # By media type
    from db import DB as _DB
    type_counts = {"image": 0, "video": 0, "audio": 0, "other": 0}
    for f in test_files:
        mt = _DB.classify_mimetype(f["mimetype"])
        type_counts[mt] = type_counts.get(mt, 0) + 1
    results["by_media_type"] = type_counts

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  Pipeline Benchmark Results", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    print(f"  Files:      {results['n_files']}", file=sys.stderr)
    print(f"  Processed:  {results['n_processed']}", file=sys.stderr)
    print(f"  Total time: {results['process_batch_s']:.2f}s", file=sys.stderr)
    print(f"  Throughput: {results['throughput_files_per_s']:.1f} files/s", file=sys.stderr)
    print(f"  Avg/file:   {results['avg_per_file_ms']:.0f}ms", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    print(json.dumps(results, indent=2))

    pipeline.close()


if __name__ == "__main__":
    main()
