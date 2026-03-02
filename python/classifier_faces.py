"""Face detection classifier using InsightFace.

Replaces src/classifier_faces.js (which used @vladmandic/face-api).

Key differences from JS version:
- InsightFace uses ONNX Runtime (supports CUDA 12+) instead of TensorFlow
- Produces 512-dim face embeddings (JS version was 128-dim)
- Bounding boxes are in absolute pixels → converted to relative coordinates

Output per image: JSON array of face objects with fields:
  angle: {roll, yaw}
  vector: [512 floats]
  x, y, height, width: relative coordinates (0-1)
  score: detection confidence

Reference: src/classifier_faces.js, lib/Classifiers/Images/ClusteringFaceClassifier.php
"""
import json
import os
import sys

import numpy as np
from PIL import Image

import base_classifier

# Configure ONNX Runtime GPU usage based on RECOGNIZE_GPU env var
gpu_requested = os.environ.get('RECOGNIZE_GPU', 'false').lower() == 'true'
if not gpu_requested:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
else:
    # ONNX Runtime needs CUDA libs (cuBLAS, cuDNN, etc.) but can't find them
    # when they're installed as pip nvidia-* packages. Pre-load them with
    # ctypes RTLD_GLOBAL so the dynamic linker can resolve them.
    import ctypes
    nvidia_base = os.path.join(
        os.path.dirname(os.path.abspath(sys.executable)),
        '..', 'lib', 'python' + '.'.join(str(v) for v in sys.version_info[:2]),
        'site-packages', 'nvidia',
    )
    nvidia_base = os.path.normpath(nvidia_base)
    if os.path.isdir(nvidia_base):
        for entry in os.listdir(nvidia_base):
            lib_path = os.path.join(nvidia_base, entry, 'lib')
            if os.path.isdir(lib_path):
                for f in os.listdir(lib_path):
                    if f.endswith('.so') or '.so.' in f:
                        try:
                            ctypes.CDLL(os.path.join(lib_path, f), mode=ctypes.RTLD_GLOBAL)
                        except OSError:
                            pass

import onnxruntime as ort
from insightface.model_zoo.model_zoo import PickableInferenceSession

# Inject optimized SessionOptions into all InsightFace ONNX sessions
_sess_options = ort.SessionOptions()
_sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
_sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
_sess_options.enable_mem_pattern = True
_sess_options.enable_mem_reuse = True

_original_pickable_init = PickableInferenceSession.__init__

def _patched_pickable_init(self, model_path, **kwargs):
    kwargs.setdefault('sess_options', _sess_options)
    _original_pickable_init(self, model_path, **kwargs)

PickableInferenceSession.__init__ = _patched_pickable_init

from insightface.app import FaceAnalysis


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')


def _trt_available():
    """Check if TensorRT is actually usable (not just listed by ORT)."""
    if 'TensorrtExecutionProvider' not in ort.get_available_providers():
        return False
    try:
        import tensorrt  # noqa: F401
        return True
    except ImportError:
        return False


def _build_providers():
    """Build provider list with optimal settings for the available hardware."""
    if not gpu_requested:
        return ['CPUExecutionProvider']

    insightface_root = os.path.join(MODELS_DIR, 'insightface')
    providers = []

    # TensorRT: fastest provider for conv-heavy models like InsightFace.
    # First run builds engines (slow), cached engines reused thereafter.
    if _trt_available():
        trt_cache_dir = os.path.join(insightface_root, 'trt_cache')
        os.makedirs(trt_cache_dir, exist_ok=True)
        providers.append(('TensorrtExecutionProvider', {
            'device_id': '0',
            'trt_max_workspace_size': str(2 * 1024 * 1024 * 1024),
            'trt_fp16_enable': '1',
            'trt_engine_cache_enable': '1',
            'trt_engine_cache_path': trt_cache_dir,
            'trt_timing_cache_enable': '1',
            'trt_timing_cache_path': trt_cache_dir,
            'trt_builder_optimization_level': '3',
        }))
        print('  TensorRT available — using as primary provider', file=sys.stderr)

    # CUDA EP: optimized convolution search + TF32 on Ampere+ GPUs
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.append(('CUDAExecutionProvider', {
            'device_id': '0',
            'arena_extend_strategy': 'kSameAsRequested',
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'cudnn_conv_use_max_workspace': '1',
            'do_copy_in_default_stream': '1',
            'use_tf32': '1',
        }))

    providers.append('CPUExecutionProvider')
    return providers


def main():
    print('Loading InsightFace model (buffalo_l)...', file=sys.stderr)

    providers = _build_providers()
    provider_names = [p[0] if isinstance(p, tuple) else p for p in providers]
    print(f'  Providers: {provider_names}', file=sys.stderr)

    insightface_root = os.path.join(MODELS_DIR, 'insightface')
    os.makedirs(insightface_root, exist_ok=True)

    # Redirect stdout→stderr during InsightFace init to keep stdout clean
    # for JSON-only protocol (InsightFace prints debug messages to stdout)
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        # Only load detection + recognition + 3D landmarks (for pose).
        # Skip genderage and 2d106det which we don't need.
        app = FaceAnalysis(
            name='buffalo_l',
            allowed_modules=['detection', 'recognition', 'landmark_3d_68'],
            providers=providers,
            root=insightface_root,
        )
        app.prepare(ctx_id=0 if gpu_requested else -1, det_size=(640, 640))
    finally:
        sys.stdout = real_stdout

    loaded_models = [f'{task}({os.path.basename(m.model_file)})' for task, m in app.models.items()]
    print(f'  Loaded models: {", ".join(loaded_models)}', file=sys.stderr)
    print('InsightFace model loaded', file=sys.stderr)

    paths = base_classifier.get_paths()
    print(f'Processing {len(paths)} images', file=sys.stderr)

    for i, path in enumerate(paths):
        try:
            print(f'Processing image {i+1}/{len(paths)}: {os.path.basename(path)}', file=sys.stderr)
            img = Image.open(path).convert('RGB')
            img_array = np.array(img)
            # InsightFace expects BGR
            img_bgr = img_array[:, :, ::-1]

            height, width = img_bgr.shape[:2]
            print(f'  Image size: {width}x{height}, detecting faces...', file=sys.stderr)

            faces = app.get(img_bgr)
            print(f'  Found {len(faces)} faces', file=sys.stderr)

            vectors = []
            for face in faces:
                # Bounding box: [x1, y1, x2, y2] in absolute pixels
                bbox = face.bbox
                x1, y1, x2, y2 = bbox

                # Convert to relative coordinates (matching JS relativeBox format)
                rel_x = float(x1) / width
                rel_y = float(y1) / height
                rel_width = float(x2 - x1) / width
                rel_height = float(y2 - y1) / height

                # Detection score
                score = float(face.det_score)

                # Face embedding: 512-dim vector
                embedding = face.embedding.tolist()

                # Pose angles: InsightFace returns [pitch, yaw, roll] in face.pose
                # JS face-api returns {angle: {roll, yaw, pitch}}
                # PHP checks MAX_FACE_YAW=50 and MAX_FACE_ROLL=30
                angle = {}
                if hasattr(face, 'pose') and face.pose is not None:
                    pose = face.pose
                    angle = {
                        'roll': float(pose[2]) if len(pose) > 2 else 0.0,
                        'yaw': float(pose[1]) if len(pose) > 1 else 0.0,
                    }

                vectors.append({
                    'angle': angle,
                    'vector': embedding,
                    'x': rel_x,
                    'y': rel_y,
                    'height': rel_height,
                    'width': rel_width,
                    'score': score,
                })

            base_classifier.output_result(vectors)
            print(f'  Output {len(vectors)} face vectors', file=sys.stderr)

        except Exception as e:
            print(f'Error processing {path}: {e}', file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            base_classifier.output_error()


if __name__ == '__main__':
    main()
