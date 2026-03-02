"""Landmark classifier using 6 regional EfficientNet models.

Replaces src/classifier_landmarks.js.

Preprocessing: PIL resize 321x321 → normalize [0,1] → [1, 321, 321, 3]
Runs all 6 regional models per image, takes highest confidence above 0.9.
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
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

IMG_SIZE = 321
INPUT_MIN = 0
TOP_K = 7
THRESHOLD = 0.9

REGIONS = [
    'landmarks_africa',
    'landmarks_asia',
    'landmarks_europe',
    'landmarks_north_america',
    'landmarks_south_america',
    'landmarks_oceania',
]

REGION_LABEL_FILES = {
    'landmarks_africa': 'africa.json',
    'landmarks_asia': 'asia.json',
    'landmarks_europe': 'europe.json',
    'landmarks_north_america': 'north_america.json',
    'landmarks_south_america': 'south_america.json',
    'landmarks_oceania': 'oceania.json',
}


def load_labels():
    """Load landmark labels for all regions."""
    labels = {}
    for region, filename in REGION_LABEL_FILES.items():
        filepath = os.path.join(DATA_DIR, 'landmarks', filename)
        with open(filepath) as f:
            data = json.load(f)
        labels[region] = data['name']
    return labels


def preprocess_image(img_path):
    """Load, resize, and normalize an image.

    Normalize from [0, 255] to [0, 1]: pixel * (1/255)
    """
    img = Image.open(img_path).convert('RGB')

    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    arr = np.array(img, dtype=np.float32)

    # Normalize to [0, 1]
    normalization_constant = (1.0 - INPUT_MIN) / 255.0
    arr = arr * normalization_constant + INPUT_MIN

    return np.expand_dims(arr, axis=0)


def get_top_k(values, k, label_dict):
    """Get top-k classes and probabilities (no softmax — raw logits).

    label_dict has string keys ("0", "1", ...) from the JSON label files.
    """
    indices = np.argsort(values)[::-1][:k]
    results = []
    for idx in indices:
        key = str(int(idx))
        if key in label_dict:
            results.append({
                'className': label_dict[key],
                'probability': float(values[idx]),
            })
    return results


def main():
    # Load all labels
    all_labels = load_labels()

    # Load all 6 regional models via serving_default signature
    models = {}
    input_keys = {}
    output_keys = {}
    for region in REGIONS:
        model_path = os.path.join(MODELS_DIR, f'{region}_saved')
        if not os.path.isdir(model_path):
            print(f'ERROR: Model not found at {model_path}. Run convert_models.py first.', file=sys.stderr)
            sys.exit(1)
        print(f'Loading {region} model...', file=sys.stderr)
        loaded = tf.saved_model.load(model_path)
        sig = loaded.signatures['serving_default']
        models[region] = sig
        input_keys[region] = list(sig.structured_input_signature[1].keys())[0]
        output_keys[region] = list(sig.structured_outputs.keys())[0]

    print('All landmark models loaded', file=sys.stderr)

    paths = base_classifier.get_paths()

    for path in paths:
        try:
            input_tensor = preprocess_image(path)
            input_tf = tf.constant(input_tensor)

            # Collect results from all 6 models
            all_results = []
            for region in REGIONS:
                model = models[region]
                output = model(**{input_keys[region]: input_tf})
                values = output[output_keys[region]].numpy().flatten()

                # No softmax — raw logits, topK=7
                results = get_top_k(values, TOP_K, all_labels[region])
                for r in results:
                    if r['probability'] >= THRESHOLD:
                        all_results.append(r)

            print(path, file=sys.stderr)
            # Sort by probability descending, take best
            all_results.sort(key=lambda x: x['probability'], reverse=True)
            print(repr(all_results), file=sys.stderr)

            if all_results:
                base_classifier.output_result(['landmark', all_results[0]['className']])
            else:
                base_classifier.output_result([])

        except Exception as e:
            print(f'Error processing {path}: {e}', file=sys.stderr)
            base_classifier.output_error()


if __name__ == '__main__':
    main()
