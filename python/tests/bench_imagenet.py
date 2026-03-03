"""Benchmark for classifier_imagenet.py — EfficientNet image classification.

Measures: model load, preprocessing (PIL resize), inference (sess.run),
postprocessing (top-k + rules engine).

Usage:
    RECOGNIZE_GPU=true python python/tests/bench_imagenet.py
"""
import os
import sys

# Ensure python/ is on the path for imports
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(TESTS_DIR)
PROJECT_ROOT = os.path.dirname(PYTHON_DIR)
sys.path.insert(0, PYTHON_DIR)

from benchmark_utils import TimingContext, BenchmarkResult, BenchmarkReport, gpu_snapshot, print_err

# Configure GPU before importing TF
os.environ.setdefault('RECOGNIZE_GPU', 'true')

import gpu_setup
tf = gpu_setup.configure()
import numpy as np

# Import classifier functions
from classifier_imagenet import (
    select_model, preprocess_image, get_top_k, _load_model_v1,
    rules, IMAGENET_CLASSES, TOP_K, BATCH_SIZE,
)
import rules_engine

RES_DIR = os.path.join(PROJECT_ROOT, 'tests', 'res')
IMAGE_FILES = ['alpine.JPG', 'eiffeltower.jpg', 'casey-lee-awj7sRviVXo-unsplash.jpg']


def find_test_images(repeat=7):
    """Build list of test images, repeating fixtures to simulate a realistic batch."""
    base = [os.path.join(RES_DIR, f) for f in IMAGE_FILES if os.path.isfile(os.path.join(RES_DIR, f))]
    if not base:
        print_err('No test images found in tests/res/')
        sys.exit(1)
    # Repeat to get ~20 images
    paths = (base * (repeat + 1))[:len(base) * repeat]
    return paths


def run_benchmark():
    report = BenchmarkReport('ImageNet (EfficientNet)')

    # Model selection
    model_path, img_size, input_min, model_name = select_model()
    print_err(f'Model: {model_name} ({img_size}x{img_size})')

    # Benchmark: model load
    gpu_before = gpu_snapshot()
    with TimingContext('model_load') as t_load:
        sess, input_name, output_name = _load_model_v1(model_path)
        with sess.graph.as_default():
            output_tensor = sess.graph.get_tensor_by_name(output_name)
            softmax_tensor = tf.nn.softmax(output_tensor)
    gpu_after = gpu_snapshot()
    print_err(f'Model load: {t_load.elapsed:.3f}s')
    print_err(f'GPU memory: {gpu_before["memory_used_mb"]:.0f} → {gpu_after["memory_used_mb"]:.0f} MB')

    paths = find_test_images()
    print_err(f'Benchmarking {len(paths)} images...')

    # Benchmark batched inference (matches production behavior)
    batch_idx = 0
    for batch_start in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[batch_start:batch_start + BATCH_SIZE]
        is_warmup = (batch_idx == 0)

        # Preprocess batch
        with TimingContext('preprocess') as t_pre:
            arrays = [preprocess_image(p, img_size, input_min) for p in batch_paths]
            batch_tensor = np.stack(arrays, axis=0)

        # Batched inference
        with TimingContext('inference') as t_inf:
            all_probs = sess.run(softmax_tensor, feed_dict={input_name: batch_tensor})

        # Postprocess each result
        with TimingContext('postprocess') as t_post:
            for j in range(len(batch_paths)):
                probs = all_probs[j].flatten()
                results = get_top_k(probs, TOP_K)
                labels = rules_engine.apply_rules(results, rules, uppercase=True)

        gpu = gpu_snapshot()
        n = len(batch_paths)

        result = BenchmarkResult(
            classifier='imagenet',
            file_path=f'batch_{batch_idx} ({n} files)',
            model_load_s=t_load.elapsed if batch_idx == 0 else 0.0,
            preprocess_s=t_pre.elapsed / n,
            inference_s=t_inf.elapsed / n,
            postprocess_s=t_post.elapsed / n,
            total_s=(t_pre.elapsed + t_inf.elapsed + t_post.elapsed) / n,
            gpu_mem_mb=gpu['memory_used_mb'],
            gpu_util_pct=gpu['utilization_pct'],
            is_warmup=is_warmup,
            extra={'batch_size': n},
        )
        report.add(result)

        if is_warmup:
            print_err(f'  Warm-up batch ({n}): pre={t_pre.elapsed*1000:.0f}ms inf={t_inf.elapsed*1000:.0f}ms total={t_pre.elapsed+t_inf.elapsed:.1f}s')
        else:
            print_err(f'  Batch {batch_idx} ({n}): pre={t_pre.elapsed*1000:.0f}ms inf={t_inf.elapsed*1000:.0f}ms '
                      f'({t_inf.elapsed/n*1000:.0f}ms/file)')

        batch_idx += 1

    return report


if __name__ == '__main__':
    report = run_benchmark()
    print_err(report.summary_table())
    print(report.to_json())
