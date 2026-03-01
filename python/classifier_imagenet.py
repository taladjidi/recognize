"""Image classifier using EfficientNetV2.

Replaces src/classifier_imagenet.js.

Preprocessing: PIL resize 512x512 bilinear → normalize to [-1,1] → [1, 512, 512, 3]
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
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
SRC_DIR = os.path.join(SCRIPT_DIR, '..', 'src')

IMG_SIZE = 512
INPUT_MIN = -1
TOP_K = 7

# Load class names
with open(os.path.join(DATA_DIR, 'imagenet_classes.json')) as f:
    IMAGENET_CLASSES = json.load(f)

# Load rules
rules = rules_engine.load_rules(os.path.join(SRC_DIR, 'rules.yml'))


def preprocess_image(img_path):
    """Load, resize, and normalize an image.

    Matches EfficientnetModel.js:
    - Normalize from [0, 255] to [inputMin, 1] using:
      normalized = image * ((1 - inputMin) / 255.0) + inputMin
    - Resize to 512x512 bilinear with align_corners=True
    """
    img = Image.open(img_path).convert('RGB')

    # Resize with bilinear interpolation
    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    arr = np.array(img, dtype=np.float32)

    # Normalize to [-1, 1]: pixel * (1 - (-1))/255 + (-1) = pixel * 2/255 - 1
    normalization_constant = (1.0 - INPUT_MIN) / 255.0
    arr = arr * normalization_constant + INPUT_MIN

    # Add batch dimension: [1, 512, 512, 3]
    return np.expand_dims(arr, axis=0)


def get_top_k(values, k):
    """Get top-k classes and probabilities."""
    indices = np.argsort(values)[::-1][:k]
    results = []
    for idx in indices:
        results.append({
            'className': IMAGENET_CLASSES[str(idx)],
            'probability': float(values[idx]),
        })
    return results


def main():
    model_path = os.path.join(MODELS_DIR, 'efficientnetv2_saved')
    if not os.path.isdir(model_path):
        print(f'ERROR: Model not found at {model_path}. Run convert_models.py first.', file=sys.stderr)
        sys.exit(1)

    print('Loading EfficientNetV2 model...', file=sys.stderr)
    model = tf.saved_model.load(model_path)
    print('Model loaded', file=sys.stderr)

    paths = base_classifier.get_paths()

    for path in paths:
        try:
            input_tensor = preprocess_image(path)
            input_tf = tf.constant(input_tensor)

            # Run inference
            logits = model(input_tf)
            if isinstance(logits, dict):
                logits = list(logits.values())[0]

            # Apply softmax
            probs = tf.nn.softmax(logits).numpy().flatten()

            # Get top-K
            results = get_top_k(probs, TOP_K)

            # Apply rules (with uppercase)
            labels = rules_engine.apply_rules(results, rules, uppercase=True)

            base_classifier.output_result(labels)

        except Exception as e:
            print(f'Error processing {path}: {e}', file=sys.stderr)
            base_classifier.output_error()


if __name__ == '__main__':
    main()
