"""Common ONNX Runtime session management and utilities.

Provides GPU/CPU provider selection, session creation, and shared
numeric helpers used by all classifiers.
"""

import logging
import os

import numpy as np
import onnxruntime as ort

log = logging.getLogger(__name__)



def build_providers(gpu=True):
    """Build ONNX Runtime execution provider list.

    Args:
        gpu: If True, try CUDA/TensorRT providers before CPU.

    Returns:
        List of provider tuples/strings for InferenceSession.
    """
    if not gpu:
        return ["CPUExecutionProvider"]

    providers = []
    available = ort.get_available_providers()

    if "TensorrtExecutionProvider" in available:
        try:
            import tensorrt  # noqa: F401
            providers.append((
                "TensorrtExecutionProvider",
                {
                    "device_id": "0",
                    "trt_max_workspace_size": str(2 * 1024 * 1024 * 1024),
                    "trt_fp16_enable": "1",
                    "trt_engine_cache_enable": "1",
                    "trt_builder_optimization_level": "3",
                },
            ))
            log.info("TensorRT provider enabled")
        except ImportError:
            pass

    if "CUDAExecutionProvider" in available:
        providers.append((
            "CUDAExecutionProvider",
            {
                "device_id": "0",
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "cudnn_conv_use_max_workspace": "1",
                "do_copy_in_default_stream": "1",
                "use_tf32": "1",
            },
        ))
        log.info("CUDA provider enabled")

    providers.append("CPUExecutionProvider")
    return providers


def create_session(model_path, gpu=True, sess_options=None):
    """Create an ONNX Runtime InferenceSession.

    Args:
        model_path: Path to .onnx model file.
        gpu: If True, attempt GPU providers.
        sess_options: Optional SessionOptions. If None, creates optimized defaults.

    Returns:
        ort.InferenceSession
    """
    if sess_options is None:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.enable_mem_pattern = True
        sess_options.enable_mem_reuse = True

    # Enable ONNX RT profiling if ORT_PROFILE env var is set
    if os.environ.get("ORT_PROFILE"):
        sess_options.enable_profiling = True
        log.info("ORT profiling enabled for %s", os.path.basename(model_path))

    providers = build_providers(gpu=gpu)
    session = ort.InferenceSession(model_path, sess_options, providers=providers)
    active = session.get_providers()
    log.info("Session created for %s, providers: %s", os.path.basename(model_path), active)
    return session


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def get_top_k(values, k, class_names):
    """Get top-k classes and probabilities.

    Args:
        values: 1D array of probabilities/scores.
        k: Number of top results.
        class_names: Dict or list mapping index to class name.

    Returns:
        List of dicts with 'className' and 'probability' keys.
    """
    indices = np.argsort(values)[::-1][:k]
    results = []
    for idx in indices:
        key = str(idx) if isinstance(class_names, dict) else idx
        name = class_names[key] if isinstance(class_names, dict) else class_names[idx]
        results.append({
            "className": name,
            "probability": float(values[idx]),
        })
    return results
