"""ImageNet classifier using EfficientNetV2-S (ONNX).

Preprocessing: PIL resize 384x384 bilinear -> normalize [0,1] -> [N, H, W, 3]
Postprocessing: softmax -> top-7 -> rules.yml filtering -> category aggregation -> uppercase

ONNX model: efficientnetv2s.onnx
  Input:  input_1  [N, 384, 384, 3] float32
  Output: output_1 [N, 1000] float32 (logits)
"""

import json
import logging
import os

import numpy as np
from PIL import Image

from classifiers.base import create_session, softmax, get_top_k
import rules_engine

log = logging.getLogger(__name__)

IMG_SIZE = 384
TOP_K = 7
BATCH_SIZE = 16

# Paths resolved relative to this file
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SCRIPT_DIR, "..", "data")
_SRC_DIR = os.path.join(_SCRIPT_DIR, "..", "..", "src")


def _load_class_names(data_dir=None):
    """Load ImageNet class names from JSON."""
    d = data_dir or _DATA_DIR
    path = os.path.join(d, "imagenet_classes.json")
    with open(path) as f:
        return json.load(f)


def _load_rules(src_dir=None):
    """Load classification rules from YAML."""
    d = src_dir or _SRC_DIR
    return rules_engine.load_rules(os.path.join(d, "rules.yml"))


class ImageNetClassifier:
    """EfficientNetV2-S ImageNet classifier using ONNX Runtime."""

    def __init__(self, models_dir, gpu=True, data_dir=None, src_dir=None):
        """Initialize the classifier.

        Args:
            models_dir: Directory containing efficientnetv2s.onnx.
            gpu: Whether to use GPU providers.
            data_dir: Override for class names directory.
            src_dir: Override for rules YAML directory.
        """
        model_path = os.path.join(models_dir, "efficientnetv2s.onnx")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.session = create_session(model_path, gpu=gpu)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.class_names = _load_class_names(data_dir)
        self.rules = _load_rules(src_dir)
        log.info("ImageNet classifier initialized")

    def preprocess(self, img_path):
        """Load, resize, and normalize an image.

        Args:
            img_path: Path to image file.

        Returns:
            numpy array [H, W, 3] float32 normalized to [0, 1].
        """
        img = Image.open(img_path).convert("RGB")
        if img.size != (IMG_SIZE, IMG_SIZE):
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0

    def infer_batch(self, batch):
        """Run inference on a preprocessed batch.

        Args:
            batch: numpy array [N, H, W, 3] float32.

        Returns:
            List of N label lists (e.g. [["Cat", "Animal"], ["Dog"], ...]).
        """
        logits = self.session.run(
            [self.output_name], {self.input_name: batch}
        )[0]
        probs = softmax(logits, axis=-1)

        results = []
        for i in range(probs.shape[0]):
            top_k = get_top_k(probs[i].flatten(), TOP_K, self.class_names)
            labels = rules_engine.apply_rules(top_k, self.rules, uppercase=True)
            results.append(labels)
        return results

    def classify(self, img_path):
        """Convenience: preprocess + infer a single image.

        Returns:
            List of label strings.
        """
        arr = self.preprocess(img_path)
        batch = np.expand_dims(arr, axis=0)
        return self.infer_batch(batch)[0]

    def warm_up(self):
        """Run a dummy inference to warm up the session."""
        dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        self.session.run([self.output_name], {self.input_name: dummy})
        log.info("ImageNet warm-up complete")
