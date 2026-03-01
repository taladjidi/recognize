"""Face detection classifier using InsightFace.

Replaces src/classifier_faces.js (which used @vladmandic/face-api).

Key differences from JS version:
- InsightFace uses ONNX Runtime (supports CUDA 12+) instead of TensorFlow
- Produces 512-dim face embeddings (JS version was 128-dim)
- Bounding boxes are in absolute pixels → converted to relative coordinates

Output per image: JSON array of face objects with fields:
  angle: {roll, yaw}
  vector: [512 floats]
  x, y, height, width: relative coordinates (0-1)
  score: detection confidence

Reference: src/classifier_faces.js, lib/Classifiers/Images/ClusteringFaceClassifier.php
"""
import json
import os
import sys

import numpy as np
from PIL import Image

import base_classifier

# Configure ONNX Runtime GPU usage based on RECOGNIZE_GPU env var
gpu_requested = os.environ.get('RECOGNIZE_GPU', 'false').lower() == 'true'
if not gpu_requested:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

import insightface
from insightface.app import FaceAnalysis


def main():
    print('Loading InsightFace model (buffalo_l)...', file=sys.stderr)

    # Use GPU providers if requested
    if gpu_requested:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    app = FaceAnalysis(name='buffalo_l', providers=providers)
    app.prepare(ctx_id=0 if gpu_requested else -1, det_size=(640, 640))
    print('InsightFace model loaded', file=sys.stderr)

    paths = base_classifier.get_paths()

    for path in paths:
        try:
            img = Image.open(path).convert('RGB')
            img_array = np.array(img)
            # InsightFace expects BGR
            img_bgr = img_array[:, :, ::-1]

            height, width = img_bgr.shape[:2]

            faces = app.get(img_bgr)

            vectors = []
            for face in faces:
                # Bounding box: [x1, y1, x2, y2] in absolute pixels
                bbox = face.bbox
                x1, y1, x2, y2 = bbox

                # Convert to relative coordinates (matching JS relativeBox format)
                rel_x = float(x1) / width
                rel_y = float(y1) / height
                rel_width = float(x2 - x1) / width
                rel_height = float(y2 - y1) / height

                # Detection score
                score = float(face.det_score)

                # Face embedding: 512-dim vector
                embedding = face.embedding.tolist()

                # Pose angles: InsightFace returns [pitch, yaw, roll] in face.pose
                # JS face-api returns {angle: {roll, yaw, pitch}}
                # PHP checks MAX_FACE_YAW=50 and MAX_FACE_ROLL=30
                angle = {}
                if hasattr(face, 'pose') and face.pose is not None:
                    pose = face.pose
                    angle = {
                        'roll': float(pose[2]) if len(pose) > 2 else 0.0,
                        'yaw': float(pose[1]) if len(pose) > 1 else 0.0,
                    }

                vectors.append({
                    'angle': angle,
                    'vector': embedding,
                    'x': rel_x,
                    'y': rel_y,
                    'height': rel_height,
                    'width': rel_width,
                    'score': score,
                })

            base_classifier.output_result(vectors)

        except Exception as e:
            print(f'Error processing {path}: {e}', file=sys.stderr)
            base_classifier.output_error()


if __name__ == '__main__':
    main()
