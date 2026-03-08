"""Landmark classifier using 6 regional EfficientNet models (ONNX).

Runs all 6 regional models per image, takes highest confidence above threshold.
Output: ["landmark", "Name"] or []

ONNX models: landmarks_{africa,asia,europe,north_america,south_america,oceania}.onnx
  Input:  images_0  [N, 321, 321, 3] float32
  Output: module_apply_default/transpose_1:0  [N, num_classes] float32
"""

import json
import logging
import os

import numpy as np
from PIL import Image

from classifiers.base import create_session

log = logging.getLogger(__name__)

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


def _load_labels(src_dir):
    """Load landmark labels for all regions from src/landmarks/*.json."""
    labels = {}
    landmarks_dir = os.path.join(src_dir, "landmarks")
    for region in REGIONS:
        filename = region.removeprefix("landmarks_") + ".json"
        filepath = os.path.join(landmarks_dir, filename)
        with open(filepath) as f:
            data = json.load(f)
        labels[region] = data["name"]
    return labels


class LandmarkClassifier:
    """6-region landmark classifier using ONNX Runtime."""

    def __init__(self, models_dir, gpu=True, src_dir=None):
        """Initialize all 6 regional models.

        Args:
            models_dir: Directory containing landmarks_*.onnx files.
            gpu: Whether to use GPU providers.
            src_dir: Override for label data directory (contains landmarks/ subdir).
        """
        s = src_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
        self.labels = _load_labels(s)
        self.sessions = {}
        self.input_names = {}
        self.output_names = {}

        for region in REGIONS:
            model_path = os.path.join(models_dir, f"{region}.onnx")
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")

            session = create_session(model_path, gpu=gpu)
            self.sessions[region] = session
            self.input_names[region] = session.get_inputs()[0].name
            self.output_names[region] = session.get_outputs()[0].name
            log.info("Loaded %s", region)

        log.info("Landmark classifier initialized (%d regions)", len(REGIONS))

    def preprocess(self, img_path):
        """Load, resize, and normalize an image.

        Returns:
            numpy array [H, W, 3] float32 normalized to [0, 1].
        """
        img = Image.open(img_path).convert("RGB")
        if img.size != (IMG_SIZE, IMG_SIZE):
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0

    def infer_batch(self, batch):
        """Run all 6 regional models on a preprocessed batch.

        Args:
            batch: numpy array [N, 321, 321, 3] float32.

        Returns:
            List of N result lists. Each is ["landmark", "Name"] or [].
        """
        n = batch.shape[0]
        image_results = [[] for _ in range(n)]

        for region in REGIONS:
            output = self.sessions[region].run(
                [self.output_names[region]],
                {self.input_names[region]: batch},
            )[0]

            for i in range(n):
                values = output[i].flatten()
                indices = np.argsort(values)[::-1][:TOP_K]
                label_dict = self.labels[region]
                for idx in indices:
                    key = str(int(idx))
                    if key in label_dict and values[idx] >= THRESHOLD:
                        image_results[i].append({
                            "className": label_dict[key],
                            "probability": float(values[idx]),
                        })

        final = []
        for i in range(n):
            candidates = image_results[i]
            candidates.sort(key=lambda x: x["probability"], reverse=True)
            if candidates:
                final.append(["landmark", candidates[0]["className"]])
            else:
                final.append([])
        return final

    def classify(self, img_path):
        """Convenience: preprocess + infer a single image.

        Returns:
            ["landmark", "Name"] or [].
        """
        arr = self.preprocess(img_path)
        batch = np.expand_dims(arr, axis=0)
        return self.infer_batch(batch)[0]

    def warm_up(self):
        """Run a dummy inference to warm up all sessions."""
        dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        for region in REGIONS:
            self.sessions[region].run(
                [self.output_names[region]],
                {self.input_names[region]: dummy},
            )
        log.info("Landmark warm-up complete")
