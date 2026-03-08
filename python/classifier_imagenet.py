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

# Extensions that tf.io.decode_image can handle natively (C++ decoders, no GIL)
_TF_DECODABLE_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.bmp', '.gif'})


def preprocess_image(img_path, img_size, input_min):
    """Load, resize, and normalize an image with PIL.

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


def init_model():
    """Initialize model for multiprocess pipeline. Returns model state dict.

    The returned dict is NOT picklable (contains TF functions) — it must
    stay in the process that calls this function (the GPU process).
    """
    model_path, img_size, input_min, model_label = select_model()
    print(f"Loading {model_label} model...", file=sys.stderr)
    infer_fn = _load_model(model_path)
    input_key = list(infer_fn.structured_input_signature[1].keys())[0]
    print("Model loaded", file=sys.stderr)
    return {
        "infer_fn": infer_fn,
        "input_key": input_key,
        "img_size": img_size,
        "input_min": input_min,
        "model_label": model_label,
        "batch_size": BATCH_SIZE,
    }


def infer_batch(model_state, batch_array):
    """Run inference on a preprocessed numpy array [N, H, W, 3].

    Returns list of N label lists (e.g. [["Cat", "Animal"], ["Dog"], ...]).
    """
    output = model_state["infer_fn"](
        **{model_state["input_key"]: tf.constant(batch_array)}
    )
    logits = list(output.values())[0].numpy()
    all_probs = tf.nn.softmax(logits).numpy()
    results = []
    for i in range(all_probs.shape[0]):
        probs = all_probs[i].flatten()
        top_k_results = get_top_k(probs, TOP_K)
        labels = rules_engine.apply_rules(top_k_results, rules, uppercase=True)
        results.append(labels)
    return results


def _infer_and_label(infer_fn, input_key, img_batch):
    """Run inference on a batch tensor and return list of label lists."""
    output = infer_fn(**{input_key: img_batch})
    logits = list(output.values())[0].numpy()
    all_probs = tf.nn.softmax(logits).numpy()
    results = []
    for j in range(all_probs.shape[0]):
        probs = all_probs[j].flatten()
        top_k = get_top_k(probs, TOP_K)
        labels = rules_engine.apply_rules(top_k, rules, uppercase=True)
        results.append(labels)
    return results


def _pipeline_tfdata(paths, infer_fn, input_key, img_size, input_min):
    """tf.data-based pipeline for maximum GPU utilization.

    Uses TF native image decoding (C++, no GIL) and resizing with automatic
    parallel prefetching. Files that fail TF decode (corrupt or unsupported)
    are retried with PIL as a fallback.
    """
    normalization_constant = (1.0 - input_min) / 255.0

    # Split paths by format: tf-native vs PIL-only
    tf_indices = []
    pil_indices = []
    for i, p in enumerate(paths):
        ext = os.path.splitext(p)[1].lower()
        if ext in _TF_DECODABLE_EXTS:
            tf_indices.append(i)
        else:
            pil_indices.append(i)

    results = [None] * len(paths)

    # Fast path: tf.data with native C++ decode + parallel map + auto prefetch
    # Uses tf.py_function for per-file error handling: returns a success flag
    # so corrupt files are marked (not silently dropped) and retried with PIL.
    if tf_indices:
        tf_paths = [paths[i] for i in tf_indices]

        def _safe_decode(path_bytes):
            """Decode image with TF ops, return (img, success).

            Runs inside tf.py_function so try/except works (eager mode).
            """
            path_str = path_bytes.numpy().decode("utf-8")
            try:
                raw = tf.io.read_file(path_str)
                img = tf.io.decode_image(raw, channels=3, expand_animations=False)
                img = tf.image.resize(img, [img_size, img_size])
                img = tf.cast(img, tf.float32) * normalization_constant + input_min
                return img, True
            except Exception as e:
                print(f"TF decode error for {path_str}: {e}", file=sys.stderr)
                return tf.zeros([img_size, img_size, 3], dtype=tf.float32), False

        def tf_preprocess(orig_idx, path_tensor):
            img, ok = tf.py_function(
                _safe_decode, [path_tensor], [tf.float32, tf.bool],
            )
            img.set_shape([img_size, img_size, 3])
            return orig_idx, img, ok

        ds = tf.data.Dataset.from_tensor_slices(
            (tf.constant(tf_indices, dtype=tf.int32), tf_paths)
        )
        ds = ds.map(tf_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        for idx_batch, img_batch, ok_batch in ds:
            orig_indices = idx_batch.numpy().tolist()
            ok_flags = ok_batch.numpy().tolist()

            # Only infer on successfully decoded images
            valid_mask = [i for i, ok in enumerate(ok_flags) if ok]
            if valid_mask:
                valid_imgs = tf.gather(img_batch, valid_mask)
                labels_list = _infer_and_label(infer_fn, input_key, valid_imgs)
                for vi, batch_i in enumerate(valid_mask):
                    results[orig_indices[batch_i]] = labels_list[vi]

            # Mark failed TF decodes for PIL retry
            for batch_i, ok in enumerate(ok_flags):
                if not ok:
                    pil_indices.append(orig_indices[batch_i])

    # PIL path: HEIC/WebP + any files that failed TF decode
    if pil_indices:
        pil_items = [(i, paths[i]) for i in pil_indices]
        for start in range(0, len(pil_items), BATCH_SIZE):
            chunk = pil_items[start:start + BATCH_SIZE]
            valid = []
            for orig_idx, p in chunk:
                try:
                    arr = preprocess_image(p, img_size, input_min)
                    valid.append((orig_idx, arr))
                except Exception as e:
                    print(f"Error processing {p}: {e}", file=sys.stderr)
                    results[orig_idx] = []

            if valid:
                batch_tensor = np.stack([arr for _, arr in valid], axis=0)
                labels_list = _infer_and_label(
                    infer_fn, input_key, tf.constant(batch_tensor)
                )
                for vi, (orig_idx, _) in enumerate(valid):
                    results[orig_idx] = labels_list[vi]

    for i, p in enumerate(paths):
        yield p, results[i] if results[i] is not None else []


def _pipeline_streaming(path_source, infer_fn, input_key, img_size, input_min):
    """Queue-based streaming pipeline for stdin input.

    Uses batched_producer_consumer with 3 batches prefetched to keep
    the GPU continuously fed while preprocessing runs in parallel threads.
    """
    def preprocess_one(path):
        return preprocess_image(path, img_size, input_min)

    for batch_paths, preprocessed in base_classifier.batched_producer_consumer(
        path_source, preprocess_one, BATCH_SIZE,
        prefetch_batches=3, workers=4,
    ):
        valid_indices = [i for i, arr in enumerate(preprocessed) if arr is not None]

        all_probs = None
        if valid_indices:
            batch_tensor = np.stack([preprocessed[i] for i in valid_indices], axis=0)
            output = infer_fn(**{input_key: tf.constant(batch_tensor)})
            logits = list(output.values())[0].numpy()
            all_probs = tf.nn.softmax(logits).numpy()

        prob_idx = 0
        for i in range(len(batch_paths)):
            if preprocessed[i] is None:
                print(f"Error processing {batch_paths[i]}", file=sys.stderr)
                yield batch_paths[i], []
            else:
                probs = all_probs[prob_idx].flatten()
                prob_idx += 1
                top_k = get_top_k(probs, TOP_K)
                labels = rules_engine.apply_rules(top_k, rules, uppercase=True)
                yield batch_paths[i], labels


def create_pipeline(paths_iterable):
    """Load model once, yield (path, labels) for each input path.

    Labels is a list of strings (e.g. ["Cat", "Animal"]) or [] on error.

    Automatically selects the best pipeline strategy:
    - List input (db_worker): tf.data with native C++ image decode, parallel
      prefetch, and automatic batching for maximum GPU utilization.
    - Generator input (stdin streaming): queue-based producer-consumer with
      3 batches prefetched in parallel threads.
    """
    model_path, img_size, input_min, model_name = select_model()

    print(f"Loading {model_name} model...", file=sys.stderr)
    infer_fn = _load_model(model_path)
    input_key = list(infer_fn.structured_input_signature[1].keys())[0]
    print("Model loaded", file=sys.stderr)

    if isinstance(paths_iterable, list):
        yield from _pipeline_tfdata(
            paths_iterable, infer_fn, input_key, img_size, input_min
        )
    else:
        yield from _pipeline_streaming(
            paths_iterable, infer_fn, input_key, img_size, input_min
        )


def main():
    for path, labels in create_pipeline(base_classifier.iter_paths()):
        base_classifier.output_result(labels)


if __name__ == "__main__":
    main()
