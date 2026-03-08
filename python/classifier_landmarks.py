"""Landmark classifier using 6 regional EfficientNet models.

Replaces src/classifier_landmarks.js.

Preprocessing: PIL resize 321x321 → normalize [0,1] → [N, 321, 321, 3] (batched)
Runs all 6 regional models per batch, takes highest confidence above 0.9.
Output: ["landmark", "Name"] or []

Reference: src/classifier_landmarks.js
"""

import json
import os
import sys

import gpu_setup

tf = gpu_setup.configure()
import numpy as np
from PIL import Image

import base_classifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

IMG_SIZE = 321
TOP_K = 7
THRESHOLD = 0.9
BATCH_SIZE = 16

# Extensions that tf.io.decode_image can handle natively (C++ decoders, no GIL)
_TF_DECODABLE_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.bmp', '.gif'})

REGIONS = [
    "landmarks_africa",
    "landmarks_asia",
    "landmarks_europe",
    "landmarks_north_america",
    "landmarks_south_america",
    "landmarks_oceania",
]


def load_labels():
    """Load landmark labels for all regions."""
    labels = {}
    for region in REGIONS:
        # landmarks_africa → africa.json, landmarks_north_america → north_america.json
        filename = region.removeprefix("landmarks_") + ".json"
        filepath = os.path.join(DATA_DIR, "landmarks", filename)
        with open(filepath) as f:
            data = json.load(f)
        labels[region] = data["name"]
    return labels


def preprocess_image(img_path):
    """Load, resize, and normalize an image to [0, 1].

    Returns [H, W, 3] array (no batch dim) for stacking into batches.
    """
    img = Image.open(img_path).convert("RGB")

    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    return np.array(img, dtype=np.float32) / 255.0


def get_top_k(values, k, label_dict):
    """Get top-k classes and probabilities (no softmax — raw logits).

    label_dict has string keys ("0", "1", ...) from the JSON label files.
    """
    indices = np.argsort(values)[::-1][:k]
    results = []
    for idx in indices:
        key = str(int(idx))
        if key in label_dict:
            results.append(
                {
                    "className": label_dict[key],
                    "probability": float(values[idx]),
                }
            )
    return results


def _load_models_v1():
    """Load all 6 regional models using v1 Session API for faster inference.

    Returns (sessions, input_names, output_names) dicts keyed by region.
    """
    sessions = {}
    input_names = {}
    output_names = {}

    for region in REGIONS:
        model_path = os.path.join(MODELS_DIR, f"{region}_saved")
        if not os.path.isdir(model_path):
            print(
                f"ERROR: Model not found at {model_path}. Run convert_models.py first.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"Loading {region} model...", file=sys.stderr)
        sess = tf.compat.v1.Session(
            graph=tf.compat.v1.Graph(),
            config=tf.compat.v1.ConfigProto(allow_soft_placement=True),
        )
        meta = tf.compat.v1.saved_model.loader.load(
            sess, [tf.compat.v1.saved_model.tag_constants.SERVING], model_path
        )
        sig = meta.signature_def["serving_default"]
        input_names[region] = list(sig.inputs.values())[0].name
        output_names[region] = list(sig.outputs.values())[0].name
        sessions[region] = sess

    return sessions, input_names, output_names


def _run_regions(sessions, input_names, output_names, all_labels, batch_tensor, n):
    """Run all 6 regional models on a batch, return per-image label lists."""
    image_results = [[] for _ in range(n)]
    for region in REGIONS:
        all_values = sessions[region].run(
            output_names[region],
            feed_dict={input_names[region]: batch_tensor},
        )
        for i in range(n):
            values = all_values[i].flatten()
            results = get_top_k(values, TOP_K, all_labels[region])
            for r in results:
                if r["probability"] >= THRESHOLD:
                    image_results[i].append(r)

    final = []
    for i in range(n):
        all_r = image_results[i]
        all_r.sort(key=lambda x: x["probability"], reverse=True)
        if all_r:
            final.append(["landmark", all_r[0]["className"]])
        else:
            final.append([])
    return final


def init_model():
    """Initialize all 6 landmark models for multiprocess pipeline."""
    all_labels = load_labels()
    print("Loading landmark models (v1 Session)...", file=sys.stderr)
    sessions, input_names, output_names = _load_models_v1()
    print("All landmark models loaded", file=sys.stderr)
    return {
        "sessions": sessions,
        "input_names": input_names,
        "output_names": output_names,
        "all_labels": all_labels,
        "batch_size": BATCH_SIZE,
    }


def infer_batch(model_state, batch_array):
    """Run all 6 regional models on a preprocessed batch [N, 321, 321, 3].

    Returns list of N result lists (e.g. [["landmark", "Eiffel Tower"], [], ...]).
    """
    return _run_regions(
        model_state["sessions"], model_state["input_names"],
        model_state["output_names"], model_state["all_labels"],
        batch_array, batch_array.shape[0],
    )


def _pipeline_tfdata(paths, sessions, input_names, output_names, all_labels):
    """tf.data-based pipeline for maximum GPU utilization."""
    # Split paths by format
    tf_indices = []
    pil_indices = []
    for i, p in enumerate(paths):
        ext = os.path.splitext(p)[1].lower()
        if ext in _TF_DECODABLE_EXTS:
            tf_indices.append(i)
        else:
            pil_indices.append(i)

    results = [None] * len(paths)

    if tf_indices:
        tf_paths = [paths[i] for i in tf_indices]

        def _safe_decode(path_bytes):
            path_str = path_bytes.numpy().decode("utf-8")
            try:
                raw = tf.io.read_file(path_str)
                img = tf.io.decode_image(raw, channels=3, expand_animations=False)
                img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
                img = tf.cast(img, tf.float32) / 255.0
                return img, True
            except Exception as e:
                print(f"TF decode error for {path_str}: {e}", file=sys.stderr)
                return tf.zeros([IMG_SIZE, IMG_SIZE, 3], dtype=tf.float32), False

        def tf_preprocess(orig_idx, path_tensor):
            img, ok = tf.py_function(
                _safe_decode, [path_tensor], [tf.float32, tf.bool],
            )
            img.set_shape([IMG_SIZE, IMG_SIZE, 3])
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

            valid_mask = [i for i, ok in enumerate(ok_flags) if ok]
            if valid_mask:
                valid_imgs = tf.gather(img_batch, valid_mask).numpy()
                labels_list = _run_regions(
                    sessions, input_names, output_names, all_labels,
                    valid_imgs, len(valid_mask),
                )
                for vi, batch_i in enumerate(valid_mask):
                    results[orig_indices[batch_i]] = labels_list[vi]

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
                    arr = preprocess_image(p)
                    valid.append((orig_idx, arr))
                except Exception as e:
                    print(f"Error processing {p}: {e}", file=sys.stderr)
                    results[orig_idx] = []

            if valid:
                batch_np = np.stack([arr for _, arr in valid], axis=0)
                labels_list = _run_regions(
                    sessions, input_names, output_names, all_labels,
                    batch_np, len(valid),
                )
                for vi, (orig_idx, _) in enumerate(valid):
                    results[orig_idx] = labels_list[vi]

    for i, p in enumerate(paths):
        yield p, results[i] if results[i] is not None else []


def _pipeline_streaming(path_source, sessions, input_names, output_names, all_labels):
    """Queue-based streaming pipeline for stdin input."""
    for batch_paths, preprocessed in base_classifier.batched_producer_consumer(
        path_source, preprocess_image, BATCH_SIZE,
        prefetch_batches=3, workers=4,
    ):
        valid_indices = [i for i, arr in enumerate(preprocessed) if arr is not None]

        labels_list = None
        if valid_indices:
            batch_tensor = np.stack([preprocessed[i] for i in valid_indices], axis=0)
            labels_list = _run_regions(
                sessions, input_names, output_names, all_labels,
                batch_tensor, len(valid_indices),
            )

        label_idx = 0
        for i in range(len(batch_paths)):
            if preprocessed[i] is None:
                print(f"Error processing {batch_paths[i]}", file=sys.stderr)
                yield batch_paths[i], []
            else:
                yield batch_paths[i], labels_list[label_idx]
                label_idx += 1


def create_pipeline(paths_iterable):
    """Load models once, yield (path, result) for each input path.

    result is ["landmark", "Name"] or [].

    Automatically selects the best pipeline strategy:
    - List input (db_worker): tf.data with native C++ image decode.
    - Generator input (stdin streaming): queue-based producer-consumer
      with 3 batches prefetched in parallel threads.
    """
    all_labels = load_labels()

    print("Loading landmark models (v1 Session)...", file=sys.stderr)
    sessions, input_names, output_names = _load_models_v1()
    print("All landmark models loaded", file=sys.stderr)

    if isinstance(paths_iterable, list):
        yield from _pipeline_tfdata(
            paths_iterable, sessions, input_names, output_names, all_labels
        )
    else:
        yield from _pipeline_streaming(
            paths_iterable, sessions, input_names, output_names, all_labels
        )


def main():
    for path, result in create_pipeline(base_classifier.iter_paths()):
        base_classifier.output_result(result)


if __name__ == "__main__":
    main()
