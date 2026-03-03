"""GPU detection test script.

Replaces src/test_gputensorflow.js.
Exits 0 if TensorFlow detects a GPU, exits 1 otherwise.
"""

import sys
import os

# Force GPU visibility for this test
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"GPU detected: {[g.name for g in gpus]}", file=sys.stderr)
    sys.exit(0)
else:
    print("No GPU detected by TensorFlow", file=sys.stderr)
    sys.exit(1)
