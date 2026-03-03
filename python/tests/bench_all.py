"""Run all classifier benchmarks and produce a combined report.

Runs each benchmark as a subprocess to avoid GPU memory conflicts between
TF-based and ONNX-based classifiers sharing the same process.

Usage:
    RECOGNIZE_GPU=true python python/tests/bench_all.py
    RECOGNIZE_GPU=true python python/tests/bench_all.py 2>summary.txt >baseline.json
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

BENCHMARKS = [
    ('imagenet', 'bench_imagenet.py'),
    ('landmarks', 'bench_landmarks.py'),
    ('faces', 'bench_faces.py'),
    ('movinet', 'bench_movinet.py'),
    ('musicnn', 'bench_musicnn.py'),
]


def run_single_benchmark(name, script):
    """Run a benchmark script as subprocess, return parsed JSON result."""
    script_path = os.path.join(TESTS_DIR, script)
    env = os.environ.copy()
    env.setdefault('RECOGNIZE_GPU', 'true')

    print(f'\n{"=" * 60}', file=sys.stderr)
    print(f'  Running {name} benchmark...', file=sys.stderr)
    print(f'{"=" * 60}', file=sys.stderr)

    start = time.perf_counter()
    proc = subprocess.run(
        [PYTHON_BIN, script_path],
        capture_output=True, text=True, env=env,
        cwd=PROJECT_ROOT, timeout=600,
    )
    elapsed = time.perf_counter() - start

    # Print stderr (human-readable summary) to our stderr
    if proc.stderr:
        for line in proc.stderr.strip().split('\n'):
            print(f'  [{name}] {line}', file=sys.stderr)

    if proc.returncode != 0:
        print(f'  [{name}] FAILED (exit code {proc.returncode})', file=sys.stderr)
        return None

    # Parse JSON from stdout
    try:
        result = json.loads(proc.stdout)
        result['wall_time_s'] = round(elapsed, 2)
        return result
    except json.JSONDecodeError:
        print(f'  [{name}] Failed to parse JSON output', file=sys.stderr)
        return None


def main():
    combined = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'gpu_env': os.environ.get('RECOGNIZE_GPU', 'not set'),
        'benchmarks': {},
    }

    total_start = time.perf_counter()

    for name, script in BENCHMARKS:
        result = run_single_benchmark(name, script)
        if result:
            combined['benchmarks'][name] = result

    total_elapsed = time.perf_counter() - total_start

    # Summary
    print(f'\n{"=" * 60}', file=sys.stderr)
    print(f'  ALL BENCHMARKS COMPLETE ({total_elapsed:.1f}s total)', file=sys.stderr)
    print(f'{"=" * 60}', file=sys.stderr)

    for name, data in combined['benchmarks'].items():
        classifiers = data.get('classifiers', {})
        for cls_name, cls_data in classifiers.items():
            avg_total = cls_data.get('avg_total_ms', 0)
            n = cls_data.get('n_files', 0)
            throughput = 1000.0 / avg_total if avg_total > 0 else 0
            print(f'  {cls_name:12s}: {avg_total:7.1f} ms/file  ({throughput:.1f} files/s, n={n})', file=sys.stderr)

    combined['total_wall_time_s'] = round(total_elapsed, 2)

    # JSON to stdout
    print(json.dumps(combined, indent=2))


if __name__ == '__main__':
    main()
