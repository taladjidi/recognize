"""End-to-end subprocess benchmark mimicking PHP's dispatch protocol.

Launches each classifier as a subprocess (same as PHP does), sends file paths
via STDIN, and measures:
- Time to first JSON output (startup + model load + first file)
- Per-file throughput (time between consecutive JSON lines)
- Total wall time for a batch

Usage:
    RECOGNIZE_GPU=true python python/tests/bench_e2e.py
"""

import json
import os
import subprocess
import sys
import time

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(TESTS_DIR)
PROJECT_ROOT = os.path.dirname(PYTHON_DIR)

PYTHON_BIN = sys.executable
RES_DIR = os.path.join(PROJECT_ROOT, "tests", "res")

# Classifiers and their test files
CLASSIFIERS = {
    "imagenet": {
        "script": os.path.join(PYTHON_DIR, "classifier_imagenet.py"),
        "files": [
            "alpine.JPG",
            "eiffeltower.jpg",
            "casey-lee-awj7sRviVXo-unsplash.jpg",
        ],
        "repeat": 7,  # ~21 files
    },
    "landmarks": {
        "script": os.path.join(PYTHON_DIR, "classifier_landmarks.py"),
        "files": ["alpine.JPG", "eiffeltower.jpg"],
        "repeat": 5,  # ~10 files
    },
    "faces": {
        "script": os.path.join(PYTHON_DIR, "classifier_faces.py"),
        "files": [],  # populated from FaceID-550
        "repeat": 1,
    },
    "movinet": {
        "script": os.path.join(PYTHON_DIR, "classifier_movinet.py"),
        "files": ["jumpingjack.gif"],
        "repeat": 3,
    },
    "musicnn": {
        "script": os.path.join(PYTHON_DIR, "classifier_musicnn.py"),
        "files": ["Rock_Rejam.mp3"],
        "repeat": 3,
    },
}


def collect_face_images():
    """Collect face images from FaceID-550."""
    images = []
    face_dir = os.path.join(RES_DIR, "FaceID-550")
    if os.path.isdir(face_dir):
        for person in sorted(os.listdir(face_dir)):
            person_dir = os.path.join(face_dir, person)
            if os.path.isdir(person_dir):
                for f in sorted(os.listdir(person_dir))[:10]:  # limit per person
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        images.append(os.path.join(person_dir, f))
    return images


def build_file_list(config):
    """Build full paths from config, repeating as needed."""
    if config["files"]:
        base = [
            os.path.join(RES_DIR, f)
            for f in config["files"]
            if os.path.isfile(os.path.join(RES_DIR, f))
        ]
    else:
        base = collect_face_images()
    return (base * config["repeat"])[: len(base) * config["repeat"]]


def benchmark_classifier(name, config):
    """Run classifier via subprocess with stdin protocol, measure timings."""
    paths = build_file_list(config)
    if not paths:
        print(f"  [{name}] No test files found, skipping", file=sys.stderr)
        return None

    script = config["script"]
    if not os.path.isfile(script):
        print(f"  [{name}] Script not found: {script}", file=sys.stderr)
        return None

    stdin_data = "\n".join(paths)

    env = os.environ.copy()
    env.setdefault("RECOGNIZE_GPU", "true")

    print(f"  [{name}] Sending {len(paths)} files via stdin...", file=sys.stderr)

    start = time.perf_counter()
    proc = subprocess.Popen(
        [PYTHON_BIN, script, "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=PROJECT_ROOT,
    )

    # Send all paths at once, then close stdin
    proc.stdin.write(stdin_data)
    proc.stdin.close()

    # Read JSON lines as they arrive
    line_times = []
    results = []
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        t = time.perf_counter() - start
        line_times.append(t)
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            pass

    proc.wait(timeout=600)
    total_elapsed = time.perf_counter() - start

    stderr_output = proc.stderr.read()

    if proc.returncode != 0:
        print(f"  [{name}] FAILED (exit {proc.returncode})", file=sys.stderr)
        if stderr_output:
            for line in stderr_output.strip().split("\n")[-5:]:
                print(f"    {line}", file=sys.stderr)
        return None

    # Compute metrics
    time_to_first = line_times[0] if line_times else total_elapsed
    per_file_times = []
    for j in range(1, len(line_times)):
        per_file_times.append(line_times[j] - line_times[j - 1])

    avg_per_file = sum(per_file_times) / len(per_file_times) if per_file_times else 0
    # Exclude first file from throughput (includes model load)
    steady_per_file = per_file_times[1:] if len(per_file_times) > 1 else per_file_times
    avg_steady = sum(steady_per_file) / len(steady_per_file) if steady_per_file else 0

    result = {
        "classifier": name,
        "n_files": len(paths),
        "n_results": len(results),
        "total_s": round(total_elapsed, 3),
        "time_to_first_s": round(time_to_first, 3),
        "avg_per_file_ms": round(avg_per_file * 1000, 1),
        "avg_steady_ms": round(avg_steady * 1000, 1),
        "throughput_files_per_s": round(1.0 / avg_steady, 2) if avg_steady > 0 else 0,
    }

    print(
        f"  [{name}] Done: {total_elapsed:.1f}s total, first={time_to_first:.1f}s, "
        f"steady={avg_steady * 1000:.0f}ms/file",
        file=sys.stderr,
    )

    return result


def main():
    combined = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": "end-to-end subprocess (mimics PHP dispatch)",
        "classifiers": {},
    }

    total_start = time.perf_counter()

    for name, config in CLASSIFIERS.items():
        print(f"\n{'=' * 50}", file=sys.stderr)
        print(f"  E2E: {name}", file=sys.stderr)
        print(f"{'=' * 50}", file=sys.stderr)

        result = benchmark_classifier(name, config)
        if result:
            combined["classifiers"][name] = result

    total_elapsed = time.perf_counter() - total_start

    # Summary
    print(f"\n{'=' * 50}", file=sys.stderr)
    print(f"  E2E SUMMARY ({total_elapsed:.1f}s total)", file=sys.stderr)
    print(f"{'=' * 50}", file=sys.stderr)
    print(
        f"  {'Classifier':12s} {'First':>8s} {'Steady':>10s} {'Through':>10s}",
        file=sys.stderr,
    )
    print(
        f"  {'':12s} {'(s)':>8s} {'(ms/file)':>10s} {'(files/s)':>10s}", file=sys.stderr
    )
    for name, data in combined["classifiers"].items():
        print(
            f"  {name:12s} {data['time_to_first_s']:8.1f} {data['avg_steady_ms']:10.0f} "
            f"{data['throughput_files_per_s']:10.1f}",
            file=sys.stderr,
        )

    combined["total_wall_time_s"] = round(total_elapsed, 2)
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
