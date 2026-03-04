"""Image classifier using EfficientNet.

Replaces src/classifier_imagenet.js.

Model selection:
- GPU mode: EfficientNetV2-S (384x384, normalize to [0,1]) — default for GPU ≥4GB
- GPU mode: EfficientNetV2-XL (512x512, normalize to [-1,1]) — via env override
- CPU mode: EfficientNet-Lite4 (380x380, normalize to [0,1])

Preprocessing: PIL resize → normalize → [N, H, W, 3] (batched)
Postprocessing: softmax → top-7 → rules.yml filtering → category aggregation → uppercase

Reference: src/efficientnet/EfficientnetModel.js lines 47-78, src/classifier_imagenet.js
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import gpu_setup

tf = gpu_setup.configure()
import numpy as np
from PIL import Image

import base_classifier
import rules_engine

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
SRC_DIR = os.path.join(SCRIPT_DIR, "..", "src")

TOP_K = 7

# Load class names
with open(os.path.join(DATA_DIR, "imagenet_classes.json")) as f:
    IMAGENET_CLASSES = json.load(f)

# Load rules
rules = rules_engine.load_rules(os.path.join(SRC_DIR, "rules.yml"))


GPU_VRAM_THRESHOLD_MB = 4 * 1024  # Need >4 GB VRAM for V2-S/V2-XL


def _get_gpu_memory_mb():
    """Get total GPU memory in MB, or 0 if unavailable."""
    try:
        import subprocess

        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
        return float(out.strip().split("\n")[0])
    except Exception:
        return 0


def select_model():
    """Select EfficientNet model variant.

    Selection priority:
    1. RECOGNIZE_IMAGENET_MODEL env var (set via Recognize admin settings):
       'efficientnetv2s' → V2-S (384x384, [0,255])
       'efficientnetv2' → V2-XL (prefer native over TFJS-converted)
       'efficientnet_lite4' → always Lite4
    2. GPU memory heuristic: V2-S for GPU ≥4GB VRAM (was V2-XL)
    3. Fallback: whatever model is available

    Returns (model_path, img_size, input_min, model_label).
    """
    v2s_path = os.path.join(MODELS_DIR, "efficientnetv2s_saved")
    v2_native_path = os.path.join(MODELS_DIR, "efficientnetv2_native_saved")
    v2_tfjs_path = os.path.join(MODELS_DIR, "efficientnetv2_saved")
    lite_path = os.path.join(MODELS_DIR, "efficientnet_lite4_saved")

    has_v2s = os.path.isdir(v2s_path)
    has_v2_native = os.path.isdir(v2_native_path)
    has_v2_tfjs = os.path.isdir(v2_tfjs_path)
    has_v2 = has_v2_native or has_v2_tfjs
    v2_path = v2_native_path if has_v2_native else v2_tfjs_path
    v2_label = "EfficientNetV2-XL" + (" (native)" if has_v2_native else " (TFJS)")
    has_lite = os.path.isdir(lite_path)

    if not has_v2s and not has_v2 and not has_lite:
        print(
            "ERROR: No imagenet model found. Run convert_models.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check for explicit user override via env var (set by PHP from app config)
    override = os.environ.get("RECOGNIZE_IMAGENET_MODEL", "auto").lower()
    if override == "efficientnetv2s" and has_v2s:
        print("Model override: EfficientNetV2-S (from settings)", file=sys.stderr)
        return v2s_path, 384, 0, "EfficientNetV2-S"
    elif override == "efficientnetv2" and has_v2:
        print(f"Model override: {v2_label} (from settings)", file=sys.stderr)
        return v2_path, 512, -1, v2_label
    elif override == "efficientnet_lite4" and has_lite:
        print("Model override: EfficientNet-Lite4 (from settings)", file=sys.stderr)
        return lite_path, 380, 0, "EfficientNet-Lite4"

    # Auto-select based on GPU memory
    has_gpu = bool(tf.config.list_physical_devices("GPU"))
    if has_gpu:
        vram = _get_gpu_memory_mb()
        if vram >= GPU_VRAM_THRESHOLD_MB:
            # V2-S is default for GPU (10x smaller than V2-XL, only 3% accuracy drop)
            if has_v2s:
                print(
                    f"Auto-selected EfficientNetV2-S ({vram:.0f} MB VRAM >= {GPU_VRAM_THRESHOLD_MB} MB threshold)",
                    file=sys.stderr,
                )
                return v2s_path, 384, 0, "EfficientNetV2-S"
            elif has_v2:
                print(
                    f"Auto-selected {v2_label} ({vram:.0f} MB VRAM >= {GPU_VRAM_THRESHOLD_MB} MB threshold)",
                    file=sys.stderr,
                )
                return v2_path, 512, -1, v2_label
        if has_lite:
            print(
                f"Auto-selected EfficientNet-Lite4 ({vram:.0f} MB VRAM < {GPU_VRAM_THRESHOLD_MB} MB threshold)",
                file=sys.stderr,
            )
            return lite_path, 380, 0, "EfficientNet-Lite4"

    # Fallback: GPU without both models, or CPU
    if has_gpu and has_v2s:
        return v2s_path, 384, 0, "EfficientNetV2-S"
    elif has_gpu and has_v2:
        return v2_path, 512, -1, v2_label
    elif has_lite:
        return lite_path, 380, 0, "EfficientNet-Lite4"
    elif has_v2s:
        return v2s_path, 384, 0, "EfficientNetV2-S"
    else:
        return v2_path, 512, -1, v2_label


BATCH_SIZE = 16  # Images per inference call; tune based on GPU memory


def preprocess_image(img_path, img_size, input_min):
    """Load, resize, and normalize an image.

    Matches EfficientnetModel.js:
    - Normalize from [0, 255] to [inputMin, 1] using:
      normalized = image * ((1 - inputMin) / 255.0) + inputMin
    - Resize to img_size x img_size bilinear
    Returns [H, W, 3] array (no batch dim) for stacking into batches.
    """
    img = Image.open(img_path).convert("RGB")

    # Resize with bilinear interpolation
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size), Image.BILINEAR)

    arr = np.array(img, dtype=np.float32)

    # Normalize: V2-XL to [-1, 1], V2-S and Lite4 to [0, 1]
    normalization_constant = (1.0 - input_min) / 255.0
    arr = arr * normalization_constant + input_min

    return arr


def get_top_k(values, k):
    """Get top-k classes and probabilities."""
    indices = np.argsort(values)[::-1][:k]
    results = []
    for idx in indices:
        results.append(
            {
                "className": IMAGENET_CLASSES[str(idx)],
                "probability": float(values[idx]),
            }
        )
    return results


def _load_model(model_path):
    """Load a SavedModel using the v2 API.

    Returns the serving_default signature function.
    """
    loaded = tf.saved_model.load(model_path)
    return loaded.signatures["serving_default"]


def _preprocess_one(args):
    """Preprocess a single image; returns (index, array) or (index, None) on error."""
    idx, path, img_size, input_min = args
    try:
        return idx, preprocess_image(path, img_size, input_min)
    except Exception:
        return idx, None


def main():
    model_path, img_size, input_min, model_name = select_model()

    print(f"Loading {model_name} model...", file=sys.stderr)
    infer_fn = _load_model(model_path)
    # Identify the input key from the signature
    input_key = list(infer_fn.structured_input_signature[1].keys())[0]
    print("Model loaded", file=sys.stderr)

    paths = base_classifier.get_paths()

    # Process in batches with parallel preprocessing
    for batch_start in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[batch_start : batch_start + BATCH_SIZE]

        # Parallel preprocessing (PIL releases GIL during I/O and resize)
        preprocessed = [None] * len(batch_paths)
        args = [(i, p, img_size, input_min) for i, p in enumerate(batch_paths)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for idx, arr in pool.map(_preprocess_one, args):
                preprocessed[idx] = arr

        # Split into valid (for batched inference) and failed
        valid_indices = [i for i, arr in enumerate(preprocessed) if arr is not None]
        failed_indices = [i for i, arr in enumerate(preprocessed) if arr is None]

        if valid_indices:
            # Stack valid images into [N, H, W, 3] and run single inference
            batch_tensor = np.stack([preprocessed[i] for i in valid_indices], axis=0)
            output = infer_fn(**{input_key: tf.constant(batch_tensor)})
            logits = list(output.values())[0].numpy()
            all_probs = tf.nn.softmax(logits).numpy()

        # Emit results in input order (PHP counts JSON lines to match files)
        prob_idx = 0
        for i in range(len(batch_paths)):
            if i in failed_indices:
                print(f"Error processing {batch_paths[i]}", file=sys.stderr)
                base_classifier.output_error()
            else:
                probs = all_probs[prob_idx].flatten()
                prob_idx += 1
                results = get_top_k(probs, TOP_K)
                labels = rules_engine.apply_rules(results, rules, uppercase=True)
                base_classifier.output_result(labels)


if __name__ == "__main__":
    main()
