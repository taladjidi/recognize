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
from concurrent.futures import ThreadPoolExecutor

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


def _preprocess_one(args):
    """Preprocess a single image; returns (index, array) or (index, None) on error."""
    idx, path = args
    try:
        return idx, preprocess_image(path)
    except Exception:
        return idx, None


def main():
    all_labels = load_labels()

    print("Loading landmark models (v1 Session)...", file=sys.stderr)
    sessions, input_names, output_names = _load_models_v1()
    print("All landmark models loaded", file=sys.stderr)

    paths = base_classifier.get_paths()

    # Process in batches
    for batch_start in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[batch_start : batch_start + BATCH_SIZE]

        # Parallel preprocessing
        preprocessed = [None] * len(batch_paths)
        args = [(i, p) for i, p in enumerate(batch_paths)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for idx, arr in pool.map(_preprocess_one, args):
                preprocessed[idx] = arr

        valid_indices = [i for i, arr in enumerate(preprocessed) if arr is not None]
        failed_indices = set(i for i, arr in enumerate(preprocessed) if arr is None)

        # Per-image results: list of lists of {className, probability} dicts
        image_results = [[] for _ in range(len(batch_paths))]

        if valid_indices:
            # Stack valid images: [N, 321, 321, 3]
            batch_tensor = np.stack([preprocessed[i] for i in valid_indices], axis=0)

            # Run each of the 6 regional models on the full batch (6 calls, not 6*N)
            for region in REGIONS:
                all_values = sessions[region].run(
                    output_names[region],
                    feed_dict={input_names[region]: batch_tensor},
                )
                # all_values shape: [N, num_classes]
                for vi, valid_i in enumerate(valid_indices):
                    values = all_values[vi].flatten()
                    results = get_top_k(values, TOP_K, all_labels[region])
                    for r in results:
                        if r["probability"] >= THRESHOLD:
                            image_results[valid_i].append(r)

        # Emit results in input order
        for i in range(len(batch_paths)):
            if i in failed_indices:
                print(f"Error processing {batch_paths[i]}", file=sys.stderr)
                base_classifier.output_error()
            else:
                all_results = image_results[i]
                all_results.sort(key=lambda x: x["probability"], reverse=True)
                if all_results:
                    base_classifier.output_result(
                        ["landmark", all_results[0]["className"]]
                    )
                else:
                    base_classifier.output_result([])


if __name__ == "__main__":
    main()
