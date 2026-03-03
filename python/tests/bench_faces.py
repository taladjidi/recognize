"""Benchmark for classifier_faces.py — InsightFace face detection/embedding.

Measures: ONNX Runtime init, PIL→BGR preprocessing, app.get() inference,
bbox/embedding extraction.

Usage:
    RECOGNIZE_GPU=true python python/tests/bench_faces.py
"""
import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(TESTS_DIR)
PROJECT_ROOT = os.path.dirname(PYTHON_DIR)
sys.path.insert(0, PYTHON_DIR)

from benchmark_utils import TimingContext, BenchmarkResult, BenchmarkReport, gpu_snapshot, print_err

os.environ.setdefault('RECOGNIZE_GPU', 'true')

# Import base_classifier early for matplotlib fix
import base_classifier  # noqa: F401

import numpy as np
from PIL import Image

# Import faces classifier setup (triggers ONNX/CUDA lib preloading)
from classifier_faces import _build_providers, gpu_requested, MODELS_DIR
from insightface.app import FaceAnalysis

RES_DIR = os.path.join(PROJECT_ROOT, 'tests', 'res')


def find_test_images():
    """Collect face test images from FaceID-550 + regular test images."""
    images = []
    face_dir = os.path.join(RES_DIR, 'FaceID-550')
    if os.path.isdir(face_dir):
        for person in sorted(os.listdir(face_dir)):
            person_dir = os.path.join(face_dir, person)
            if os.path.isdir(person_dir):
                for f in sorted(os.listdir(person_dir)):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        images.append(os.path.join(person_dir, f))

    # Add regular test images
    for f in ['alpine.JPG', 'eiffeltower.jpg']:
        p = os.path.join(RES_DIR, f)
        if os.path.isfile(p):
            images.append(p)

    return images


def run_benchmark():
    report = BenchmarkReport('Faces (InsightFace/ONNX Runtime)')

    providers = _build_providers()
    provider_names = [p[0] if isinstance(p, tuple) else p for p in providers]
    print_err(f'Providers: {provider_names}')

    insightface_root = os.path.join(MODELS_DIR, 'insightface')
    os.makedirs(insightface_root, exist_ok=True)

    # Model load
    gpu_before = gpu_snapshot()
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    with TimingContext('model_load') as t_load:
        try:
            app = FaceAnalysis(
                name='buffalo_l',
                allowed_modules=['detection', 'recognition', 'landmark_3d_68'],
                providers=providers,
                root=insightface_root,
            )
            app.prepare(ctx_id=0 if gpu_requested else -1, det_size=(640, 640))
        finally:
            sys.stdout = real_stdout
    gpu_after = gpu_snapshot()

    print_err(f'Model load: {t_load.elapsed:.3f}s')
    print_err(f'GPU memory: {gpu_before["memory_used_mb"]:.0f} → {gpu_after["memory_used_mb"]:.0f} MB')

    paths = find_test_images()
    print_err(f'Benchmarking {len(paths)} images...')

    for i, path in enumerate(paths):
        is_warmup = (i == 0)

        # Preprocess
        with TimingContext('preprocess') as t_pre:
            img = Image.open(path).convert('RGB')
            img_array = np.array(img)
            img_bgr = img_array[:, :, ::-1]
            height, width = img_bgr.shape[:2]

        # Inference (detection + recognition + landmark)
        with TimingContext('inference') as t_inf:
            faces = app.get(img_bgr)

        # Postprocess (bbox conversion, embedding extraction)
        with TimingContext('postprocess') as t_post:
            vectors = []
            for face in faces:
                bbox = face.bbox
                x1, y1, x2, y2 = bbox
                rel_x = float(x1) / width
                rel_y = float(y1) / height
                rel_width = float(x2 - x1) / width
                rel_height = float(y2 - y1) / height
                score = float(face.det_score)
                embedding = face.embedding.tolist()
                angle = {}
                if hasattr(face, 'pose') and face.pose is not None:
                    pose = face.pose
                    angle = {
                        'roll': float(pose[2]) if len(pose) > 2 else 0.0,
                        'yaw': float(pose[1]) if len(pose) > 1 else 0.0,
                    }
                vectors.append({
                    'angle': angle, 'vector': embedding,
                    'x': rel_x, 'y': rel_y,
                    'height': rel_height, 'width': rel_width,
                    'score': score,
                })

        gpu = gpu_snapshot()

        result = BenchmarkResult(
            classifier='faces',
            file_path=os.path.basename(path),
            model_load_s=t_load.elapsed if i == 0 else 0.0,
            preprocess_s=t_pre.elapsed,
            inference_s=t_inf.elapsed,
            postprocess_s=t_post.elapsed,
            total_s=t_pre.elapsed + t_inf.elapsed + t_post.elapsed,
            gpu_mem_mb=gpu['memory_used_mb'],
            gpu_util_pct=gpu['utilization_pct'],
            is_warmup=is_warmup,
            extra={'n_faces': len(faces)},
        )
        report.add(result)

        if is_warmup:
            print_err(f'  Warm-up: pre={t_pre.elapsed*1000:.1f}ms inf={t_inf.elapsed*1000:.1f}ms post={t_post.elapsed*1000:.1f}ms faces={len(faces)}')
        elif i == 1:
            print_err(f'  Steady:  pre={t_pre.elapsed*1000:.1f}ms inf={t_inf.elapsed*1000:.1f}ms post={t_post.elapsed*1000:.1f}ms faces={len(faces)}')

    return report


if __name__ == '__main__':
    report = run_benchmark()
    print_err(report.summary_table())
    print(report.to_json())
