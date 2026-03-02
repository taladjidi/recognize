"""Image classifier using EfficientNet.

Replaces src/classifier_imagenet.js.

Model selection (matches JS behavior):
- GPU mode: EfficientNetV2-XL (512x512, normalize to [-1,1])
- CPU mode: EfficientNet-Lite4 (380x380, normalize to [0,1])

Preprocessing: PIL resize → normalize → [1, H, W, 3]
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

TOP_K = 7

# Load class names
with open(os.path.join(DATA_DIR, 'imagenet_classes.json')) as f:
    IMAGENET_CLASSES = json.load(f)

# Load rules
rules = rules_engine.load_rules(os.path.join(SRC_DIR, 'rules.yml'))


def select_model():
    """Select model based on GPU availability, matching JS behavior.

    JS: efficientnetv2 (512, min=-1) for GPU, efficientnet_lite4 (380, min=0) for PUREJS.
    """
    has_gpu = bool(tf.config.list_physical_devices('GPU'))
    v2_path = os.path.join(MODELS_DIR, 'efficientnetv2_saved')
    lite_path = os.path.join(MODELS_DIR, 'efficientnet_lite4_saved')

    if has_gpu and os.path.isdir(v2_path):
        return v2_path, 512, -1, 'EfficientNetV2-XL'
    elif os.path.isdir(lite_path):
        return lite_path, 380, 0, 'EfficientNet-Lite4'
    elif os.path.isdir(v2_path):
        return v2_path, 512, -1, 'EfficientNetV2-XL'
    else:
        print('ERROR: No imagenet model found. Run convert_models.py first.', file=sys.stderr)
        sys.exit(1)


def preprocess_image(img_path, img_size, input_min):
    """Load, resize, and normalize an image.

    Matches EfficientnetModel.js:
    - Normalize from [0, 255] to [inputMin, 1] using:
      normalized = image * ((1 - inputMin) / 255.0) + inputMin
    - Resize to img_size x img_size bilinear
    """
    img = Image.open(img_path).convert('RGB')

    # Resize with bilinear interpolation
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size), Image.BILINEAR)

    arr = np.array(img, dtype=np.float32)

    # Normalize: V2 to [-1, 1], Lite4 to [0, 1]
    normalization_constant = (1.0 - input_min) / 255.0
    arr = arr * normalization_constant + input_min

    # Add batch dimension: [1, H, W, 3]
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


def _load_model_v1(model_path):
    """Load a SavedModel using v1 Session API.

    More memory-efficient for frozen graphs (weights as constants in the
    graph) since it avoids parsing the entire protobuf into Python objects.
    Returns (session, input_tensor_name, output_tensor_name).
    """
    sess = tf.compat.v1.Session(
        graph=tf.compat.v1.Graph(),
        config=tf.compat.v1.ConfigProto(allow_soft_placement=True),
    )
    meta_graph_def = tf.compat.v1.saved_model.loader.load(
        sess,
        [tf.compat.v1.saved_model.tag_constants.SERVING],
        model_path,
    )
    sig = meta_graph_def.signature_def['serving_default']
    input_name = list(sig.inputs.values())[0].name
    output_name = list(sig.outputs.values())[0].name
    return sess, input_name, output_name


def main():
    model_path, img_size, input_min, model_name = select_model()

    print(f'Loading {model_name} model...', file=sys.stderr)
    sess, input_name, output_name = _load_model_v1(model_path)
    # Build a softmax op in the session graph
    with sess.graph.as_default():
        output_tensor = sess.graph.get_tensor_by_name(output_name)
        softmax_tensor = tf.nn.softmax(output_tensor)
    print('Model loaded', file=sys.stderr)

    paths = base_classifier.get_paths()

    for path in paths:
        try:
            input_tensor = preprocess_image(path, img_size, input_min)

            # Run inference via v1 Session
            probs = sess.run(softmax_tensor, feed_dict={input_name: input_tensor})
            probs = probs.flatten()

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
