"""Face detection and recognition using InsightFace (ONNX).

Uses InsightFace's buffalo_l model pack (det + recognition + landmark_3d_68).
Produces 512-dim normalized face embeddings with bounding boxes in relative coords.

Output per image: list of face dicts with keys:
  angle: {roll, yaw}
  vector: [512 floats] (L2-normalized)
  x, y, height, width: relative coordinates (0-1)
  score: detection confidence
"""

import logging
import os
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

log = logging.getLogger(__name__)


def _patch_insightface_session_options():
    """Inject optimized SessionOptions into InsightFace's ONNX sessions."""
    from insightface.model_zoo.model_zoo import PickableInferenceSession

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.enable_mem_pattern = True
    opts.enable_mem_reuse = True

    original_init = PickableInferenceSession.__init__

    def patched_init(self, model_path, **kwargs):
        kwargs.setdefault("sess_options", opts)
        original_init(self, model_path, **kwargs)

    PickableInferenceSession.__init__ = patched_init


def _build_providers(gpu=True):
    """Build provider list for InsightFace."""
    if not gpu:
        return ["CPUExecutionProvider"]

    from classifiers.base import build_providers
    return build_providers(gpu=True)


def _extract_face(face, width, height):
    """Extract face data dict from an InsightFace detection result."""
    x1, y1, x2, y2 = face.bbox

    angle = {}
    if hasattr(face, "pose") and face.pose is not None:
        pose = face.pose
        angle = {
            "roll": float(pose[2]) if len(pose) > 2 else 0.0,
            "yaw": float(pose[1]) if len(pose) > 1 else 0.0,
        }

    embedding = face.embedding / np.linalg.norm(face.embedding)

    return {
        "angle": angle,
        "vector": embedding.tolist(),
        "x": float(x1) / width,
        "y": float(y1) / height,
        "height": float(y2 - y1) / height,
        "width": float(x2 - x1) / width,
        "score": float(face.det_score),
    }


class FaceClassifier:
    """InsightFace face detection + recognition classifier."""

    def __init__(self, models_dir, gpu=True):
        """Initialize InsightFace.

        Args:
            models_dir: Directory containing insightface/ subdirectory.
            gpu: Whether to use GPU providers.
        """
        _patch_insightface_session_options()
        from insightface.app import FaceAnalysis

        insightface_root = os.path.join(models_dir, "insightface")
        os.makedirs(insightface_root, exist_ok=True)

        providers = _build_providers(gpu=gpu)
        provider_names = [p[0] if isinstance(p, tuple) else p for p in providers]
        log.info("Face classifier providers: %s", provider_names)

        # InsightFace prints to stdout; redirect to stderr
        real_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            self.app = FaceAnalysis(
                name="buffalo_l",
                allowed_modules=["detection", "recognition", "landmark_3d_68"],
                providers=providers,
                root=insightface_root,
            )
            # det_size controls face detection input resolution. 640x640 needs ~4GB
            # VRAM for conv buffers alone; 320x320 needs ~1GB. Recognition model
            # (w600k_r50) always runs at 112x112 regardless.
            # Detect faces down to ~40px at 320x320 (sufficient for most photos).
            det = (320, 320) if gpu else (640, 640)  # CPU has no VRAM limit
            self.app.prepare(ctx_id=0 if gpu else -1, det_size=det)
        finally:
            sys.stdout = real_stdout

        log.info("Face classifier initialized")

    def preprocess(self, img_path):
        """Load image as BGR numpy array (InsightFace format).

        Args:
            img_path: Path to image file.

        Returns:
            numpy array [H, W, 3] uint8 BGR.
        """
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)
        return arr[:, :, ::-1]  # RGB -> BGR

    def infer_one(self, img_bgr):
        """Run face detection on a single BGR image.

        Args:
            img_bgr: numpy array [H, W, 3] uint8 BGR.

        Returns:
            List of face dicts.
        """
        height, width = img_bgr.shape[:2]
        faces = self.app.get(img_bgr)
        return [_extract_face(face, width, height) for face in faces]

    def classify(self, img_path):
        """Convenience: preprocess + detect faces in a single image.

        Returns:
            List of face dicts.
        """
        img_bgr = self.preprocess(img_path)
        return self.infer_one(img_bgr)

    def warm_up(self):
        """Run a dummy inference to warm up the session."""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.app.get(dummy)
        log.info("Face classifier warm-up complete")
