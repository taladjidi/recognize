"""Benchmark for classifier_movinet.py — MoViNet video action classification.

Measures: model load, FFmpeg transcode, frame decode (PIL), numpy stacking,
inference, top-k postprocessing.

Usage:
    RECOGNIZE_GPU=true python python/tests/bench_movinet.py
"""
import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(TESTS_DIR)
PROJECT_ROOT = os.path.dirname(PYTHON_DIR)
sys.path.insert(0, PYTHON_DIR)

from benchmark_utils import TimingContext, BenchmarkResult, BenchmarkReport, gpu_snapshot, print_err

os.environ.setdefault('RECOGNIZE_GPU', 'true')

import gpu_setup
tf = gpu_setup.configure()
import numpy as np

from classifier_movinet import (
    MODELS_DIR, FRAME_SIZE, TOP_K, THRESHOLD, KINETICS_CLASSES,
    get_ffmpeg_binary, extract_frames, decode_frames, get_top_k,
)

RES_DIR = os.path.join(PROJECT_ROOT, 'tests', 'res')


def find_test_videos(repeat=5):
    """Find test video files, repeat to simulate batch."""
    base = []
    for f in ['jumpingjack.gif']:
        p = os.path.join(RES_DIR, f)
        if os.path.isfile(p):
            base.append(p)
    if not base:
        print_err('No test videos found in tests/res/')
        sys.exit(1)
    return (base * repeat)[:repeat]


def run_benchmark():
    report = BenchmarkReport('MoViNet (video action classification)')

    model_path = os.path.join(MODELS_DIR, 'movinet-a3')
    if not os.path.isdir(model_path):
        print_err(f'Model not found at {model_path}')
        sys.exit(1)

    # Model load
    gpu_before = gpu_snapshot()
    with TimingContext('model_load') as t_load:
        loaded = tf.saved_model.load(model_path)
        model = loaded.signatures['serving_default']
    gpu_after = gpu_snapshot()

    print_err(f'Model load: {t_load.elapsed:.3f}s')
    print_err(f'GPU memory: {gpu_before["memory_used_mb"]:.0f} → {gpu_after["memory_used_mb"]:.0f} MB')

    ffmpeg_binary = get_ffmpeg_binary()
    paths = find_test_videos()
    print_err(f'Benchmarking {len(paths)} videos...')

    for i, path in enumerate(paths):
        is_warmup = (i == 0)

        # FFmpeg transcode
        with TimingContext('ffmpeg') as t_ffmpeg:
            frame_buffers = extract_frames(path, ffmpeg_binary)

        if not frame_buffers:
            print_err(f'  No frames extracted from {path}')
            continue

        # Frame decode
        with TimingContext('frame_decode') as t_decode:
            frame_arrays = decode_frames(frame_buffers)

        # Numpy stacking + normalization
        with TimingContext('preprocess') as t_pre:
            frame_tensor = np.stack(frame_arrays, axis=0) / 255.0
            frame_batch = np.expand_dims(frame_tensor, axis=0).astype(np.float32)

        # Inference
        with TimingContext('inference') as t_inf:
            input_tensor = tf.constant(frame_batch)
            output = model(image=input_tensor)
            logits = output['classifier_head']
            probs = tf.nn.softmax(logits).numpy().flatten()

        # Postprocess
        with TimingContext('postprocess') as t_post:
            results = get_top_k(probs, TOP_K)
            labels = [r['className'] for r in results if r['probability'] >= THRESHOLD]
            labels = list(dict.fromkeys(labels))  # deduplicate preserving order

        gpu = gpu_snapshot()

        total = t_ffmpeg.elapsed + t_decode.elapsed + t_pre.elapsed + t_inf.elapsed + t_post.elapsed
        result = BenchmarkResult(
            classifier='movinet',
            file_path=os.path.basename(path),
            model_load_s=t_load.elapsed if i == 0 else 0.0,
            preprocess_s=t_pre.elapsed,
            inference_s=t_inf.elapsed,
            postprocess_s=t_post.elapsed,
            total_s=total,
            gpu_mem_mb=gpu['memory_used_mb'],
            gpu_util_pct=gpu['utilization_pct'],
            is_warmup=is_warmup,
            extra={
                'ffmpeg_s': t_ffmpeg.elapsed,
                'frame_decode_s': t_decode.elapsed,
                'n_frames': len(frame_buffers),
            },
        )
        report.add(result)

        if is_warmup:
            print_err(f'  Warm-up: ffmpeg={t_ffmpeg.elapsed*1000:.0f}ms decode={t_decode.elapsed*1000:.0f}ms '
                      f'inf={t_inf.elapsed*1000:.0f}ms ({len(frame_buffers)} frames) → {labels}')
        elif i == 1:
            print_err(f'  Steady:  ffmpeg={t_ffmpeg.elapsed*1000:.0f}ms decode={t_decode.elapsed*1000:.0f}ms '
                      f'inf={t_inf.elapsed*1000:.0f}ms')

    return report


if __name__ == '__main__':
    report = run_benchmark()
    print_err(report.summary_table())
    print(report.to_json())
