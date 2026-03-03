"""Reusable benchmark utilities for measuring classifier performance.

Provides:
- TimingContext: context manager that records wall time for a named phase
- BenchmarkResult: dataclass holding all timing data for one file/batch
- BenchmarkReport: aggregates results and prints summary tables
- gpu_snapshot(): captures GPU memory/utilization via nvidia-smi
"""
import dataclasses
import json
import subprocess
import sys
import time
from typing import Optional


class TimingContext:
    """Context manager that records wall-clock time.

    Usage:
        with TimingContext("preprocess") as t:
            do_work()
        print(t.elapsed)  # seconds as float
    """

    def __init__(self, phase: str):
        self.phase = phase
        self.elapsed = 0.0
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start
        return False


@dataclasses.dataclass
class BenchmarkResult:
    classifier: str
    file_path: str
    model_load_s: float = 0.0
    preprocess_s: float = 0.0
    inference_s: float = 0.0
    postprocess_s: float = 0.0
    total_s: float = 0.0
    gpu_mem_mb: float = 0.0
    gpu_util_pct: float = 0.0
    is_warmup: bool = False
    extra: dict = dataclasses.field(default_factory=dict)


class BenchmarkReport:
    """Collects BenchmarkResults and prints summary tables."""

    def __init__(self, name: str = ""):
        self.name = name
        self.results: list[BenchmarkResult] = []

    def add(self, result: BenchmarkResult):
        self.results.append(result)

    def _steady_state(self) -> list[BenchmarkResult]:
        return [r for r in self.results if not r.is_warmup]

    def _by_classifier(self) -> dict[str, list[BenchmarkResult]]:
        groups: dict[str, list[BenchmarkResult]] = {}
        for r in self._steady_state():
            groups.setdefault(r.classifier, []).append(r)
        return groups

    def summary_table(self) -> str:
        lines = []
        lines.append(f"\n{'=' * 80}")
        if self.name:
            lines.append(f"  {self.name}")
            lines.append(f"{'=' * 80}")

        for classifier, results in self._by_classifier().items():
            n = len(results)
            if n == 0:
                continue

            avg_pre = sum(r.preprocess_s for r in results) / n
            avg_inf = sum(r.inference_s for r in results) / n
            avg_post = sum(r.postprocess_s for r in results) / n
            avg_total = sum(r.total_s for r in results) / n

            # Model load is typically measured once
            model_load = next((r.model_load_s for r in self.results
                               if r.classifier == classifier and r.model_load_s > 0), 0.0)

            # Warmup inference
            warmup = next((r for r in self.results
                           if r.classifier == classifier and r.is_warmup), None)

            lines.append(f"\n  {classifier} ({n} files, steady-state)")
            lines.append(f"  {'-' * 60}")
            if model_load > 0:
                lines.append(f"  Model load:       {model_load:8.3f} s")
            if warmup:
                lines.append(f"  Warm-up inference: {warmup.inference_s:8.3f} s")
            lines.append(f"  Avg preprocess:   {avg_pre * 1000:8.1f} ms")
            lines.append(f"  Avg inference:    {avg_inf * 1000:8.1f} ms")
            lines.append(f"  Avg postprocess:  {avg_post * 1000:8.1f} ms")
            lines.append(f"  Avg total/file:   {avg_total * 1000:8.1f} ms")
            lines.append(f"  Throughput:       {1.0 / avg_total if avg_total > 0 else 0:8.1f} files/s")

            # GPU stats
            gpu_mems = [r.gpu_mem_mb for r in results if r.gpu_mem_mb > 0]
            if gpu_mems:
                lines.append(f"  GPU memory:       {max(gpu_mems):8.0f} MB (max)")

            # Extra fields
            extra_keys = set()
            for r in results:
                extra_keys.update(r.extra.keys())
            for key in sorted(extra_keys):
                vals = [r.extra[key] for r in results if key in r.extra
                        and isinstance(r.extra[key], (int, float))]
                if vals:
                    avg_val = sum(vals) / len(vals)
                    lines.append(f"  Avg {key}: {avg_val * 1000:8.1f} ms")

        lines.append(f"\n{'=' * 80}")
        return '\n'.join(lines)

    def to_dict(self) -> dict:
        output = {"name": self.name, "classifiers": {}}
        for classifier, results in self._by_classifier().items():
            n = len(results)
            if n == 0:
                continue
            model_load = next((r.model_load_s for r in self.results
                               if r.classifier == classifier and r.model_load_s > 0), 0.0)
            warmup = next((r for r in self.results
                           if r.classifier == classifier and r.is_warmup), None)
            output["classifiers"][classifier] = {
                "n_files": n,
                "model_load_s": round(model_load, 4),
                "warmup_inference_s": round(warmup.inference_s, 4) if warmup else None,
                "avg_preprocess_ms": round(sum(r.preprocess_s for r in results) / n * 1000, 2),
                "avg_inference_ms": round(sum(r.inference_s for r in results) / n * 1000, 2),
                "avg_postprocess_ms": round(sum(r.postprocess_s for r in results) / n * 1000, 2),
                "avg_total_ms": round(sum(r.total_s for r in results) / n * 1000, 2),
                "gpu_mem_max_mb": round(max((r.gpu_mem_mb for r in results if r.gpu_mem_mb > 0), default=0), 0),
            }
            # Extra averages
            extra_keys = set()
            for r in results:
                extra_keys.update(r.extra.keys())
            extras = {}
            for key in sorted(extra_keys):
                vals = [r.extra[key] for r in results if key in r.extra
                        and isinstance(r.extra[key], (int, float))]
                if vals:
                    extras[f"avg_{key}_ms"] = round(sum(vals) / len(vals) * 1000, 2)
            if extras:
                output["classifiers"][classifier]["extra"] = extras
        return output

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def gpu_snapshot() -> dict:
    """Query nvidia-smi for current GPU memory and utilization."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
             '--format=csv,noheader,nounits'],
            text=True, timeout=5, stderr=subprocess.DEVNULL,
        )
        parts = out.strip().split(',')
        return {
            'memory_used_mb': float(parts[0].strip()),
            'utilization_pct': float(parts[1].strip()),
        }
    except Exception:
        return {'memory_used_mb': 0.0, 'utilization_pct': 0.0}


def print_err(*args, **kwargs):
    """Print to stderr."""
    print(*args, file=sys.stderr, **kwargs)
