"""ONNX-based classifiers for the recognize app.

Each classifier follows a common interface:
  - __init__(models_dir, gpu=True, ...)
  - preprocess(path) -> preprocessed data
  - infer_batch/infer_one(data) -> results
  - classify(path) -> results (convenience)
  - warm_up() -> None

Classifiers are NOT eagerly imported here to avoid loading heavy dependencies
(insightface, onnxruntime) at import time. Import them directly:

    from classifiers.imagenet import ImageNetClassifier
    from classifiers.faces import FaceClassifier
    from classifiers.landmarks import LandmarkClassifier
    from classifiers.movinet import MoViNetClassifier
    from classifiers.musicnn import AudioClassifier
"""

__all__ = [
    "ImageNetClassifier",
    "FaceClassifier",
    "LandmarkClassifier",
    "MoViNetClassifier",
    "AudioClassifier",
]
