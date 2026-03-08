"""Run all classifier benchmarks in subprocess isolation.

Each classifier gets its own process to avoid GPU memory conflicts.
Aggregates results into a combined JSON report.

Run: python tests/bench_all.py [--no-gpu] [--json]
"""

import json
import os
import subprocess
import sys
import time

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

BENCHMARKS = [
    ("imagenet", "bench_imagenet.py"),
    ("faces", "bench_faces.py"),
    ("landmarks", "bench_landmarks.py"),
    ("movinet", "bench_movinet.py"),
    ("musicnn", "bench_musicnn.py"),
]


def run_one(name, script, gpu=True):
    """Run a single benchmark in a subprocess."""
    cmd = [PYTHON, os.path.join(BENCH_DIR, script)]
    if not gpu:
        cmd.append("--no-gpu")

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  Running: {name}", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"  FAILED ({proc.returncode}): {proc.stderr[:500]}", file=sys.stderr)
        return {"error": proc.stderr[:500], "elapsed_s": elapsed}

    # Output contains summary table (text) then JSON
    lines = proc.stdout.strip().split("\n")
    # Find the JSON blob (starts with '{')
    json_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("{"):
            json_start = i
            break

    if json_start is not None:
        # Print the text portion to stderr
        for line in lines[:json_start]:
            print(line, file=sys.stderr)
        # Parse the JSON
        json_text = "\n".join(lines[json_start:])
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass

    # Fallback: print all output
    print(proc.stdout, file=sys.stderr)
    return {"raw_output": proc.stdout[:1000], "elapsed_s": elapsed}


def main():
    gpu = "--no-gpu" not in sys.argv
    json_output = "--json" in sys.argv

    results = {}
    total_t0 = time.perf_counter()

    for name, script in BENCHMARKS:
        results[name] = run_one(name, script, gpu=gpu)

    total_elapsed = time.perf_counter() - total_t0

    combined = {
        "total_elapsed_s": round(total_elapsed, 2),
        "gpu": gpu,
        "benchmarks": results,
    }

    if json_output:
        print(json.dumps(combined, indent=2))
    else:
        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"  All benchmarks complete in {total_elapsed:.1f}s", file=sys.stderr)
        print(f"{'=' * 60}", file=sys.stderr)
        print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
