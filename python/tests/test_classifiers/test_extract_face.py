"""_extract_face drops detections that would serialise to invalid JSON or an empty box."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

for _mod in ("onnxruntime", "PIL", "PIL.Image"):  # _extract_face is pure numpy
    try:
        __import__(_mod)
    except ImportError:
        sys.modules[_mod] = MagicMock()

from classifiers.faces import _extract_face


def _face(bbox=(10, 10, 50, 50), emb=None):
    emb = np.ones(4, dtype=np.float32) if emb is None else emb
    return SimpleNamespace(bbox=np.array(bbox, dtype=float), embedding=emb,
                           det_score=0.9, pose=None)


def test_good_face_kept():
    assert _extract_face(_face(), 100, 100)["width"] == 0.4


def test_zero_norm_embedding_skipped():
    assert _extract_face(_face(emb=np.zeros(4, dtype=np.float32)), 100, 100) is None


def test_non_finite_embedding_skipped():
    assert _extract_face(_face(emb=np.array([np.nan, 1, 1, 1], dtype=np.float32)), 100, 100) is None


def test_box_outside_image_skipped():
    assert _extract_face(_face(bbox=(120, 10, 150, 50)), 100, 100) is None  # x2 clamps below x1
