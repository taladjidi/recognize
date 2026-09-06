#!/usr/bin/env python3
"""Verify the recognize venv is usable BY THE SERVICE USER.

Exit 0 only if every check passes. Each check prints PASS/FAIL so a
partial failure is visible rather than masked by the first exception.
"""
import json
import os
import sys

FAILED = []

def check(name, fn):
    try:
        detail = fn()
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    except Exception as e:
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        FAILED.append(name)

def stdlib():
    import encodings, ssl, sqlite3, json as _j, ctypes  # noqa: F401
    return f"python {sys.version.split()[0]}"

def prefix_exists():
    # the exact failure mode we are fixing: sys.prefix chain must resolve
    if not os.path.isdir(sys.base_prefix):
        raise RuntimeError(f"base_prefix missing: {sys.base_prefix}")
    return f"base_prefix={sys.base_prefix}"

def numpy_():
    import numpy
    return numpy.__version__

def onnx():
    import onnxruntime
    return onnxruntime.__version__

def onnx_gpu():
    """Not just 'is it listed' - actually build a CUDA session and run it.

    ORT silently falls back to CPU when cuDNN/CUDA libs are unloadable, so
    provider availability alone proves nothing.
    """
    import numpy as np, onnxruntime
    provs = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" not in provs:
        raise RuntimeError(f"no CUDA provider; have {provs}")
    mdir = os.environ.get("MODELS_DIR", "/usr/share/nextcloud/apps/recognize/models")
    model = os.path.join(mdir, "efficientnetv2s.onnx")
    if not os.path.exists(model):
        raise RuntimeError(f"missing {model}")
    so = onnxruntime.SessionOptions()
    so.log_severity_level = 3
    sess = onnxruntime.InferenceSession(model, so, providers=["CUDAExecutionProvider"])
    used = sess.get_providers()
    if "CUDAExecutionProvider" not in used:
        raise RuntimeError(f"fell back to CPU; session uses {used}")
    inp = sess.get_inputs()[0]
    shape = [1 if (isinstance(d, str) or d is None) else d for d in inp.shape]
    sess.run(None, {inp.name: np.zeros(shape, dtype=np.float32)})
    return f"real GPU inference OK ({used[0]})"

def insight():
    import insightface
    return insightface.__version__

def sklearn_():
    import sklearn
    return sklearn.__version__

def cv2_():
    import cv2
    return cv2.__version__

def others():
    import PIL, soundfile, requests, scipy  # noqa: F401
    return None

def mysql():
    import mysql.connector  # noqa: F401
    return None

def db_connect():
    cfg_path = os.environ.get("NC_CONFIG", "/usr/share/nextcloud/apps/recognize/python/nc_config.json")
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    import mysql.connector
    conn = mysql.connector.connect(
        host=cfg["dbhost"], user=cfg["dbuser"],
        password=cfg["dbpassword"], database=cfg["dbname"],
    )
    cur = conn.cursor()
    cur.execute("SELECT 1")
    cur.fetchall()
    conn.close()
    return f"{cfg['dbtype']}://{cfg['dbuser']}@{cfg['dbhost']}/{cfg['dbname']}"

def models_load():
    import onnxruntime
    mdir = os.environ.get("MODELS_DIR", "/usr/share/nextcloud/apps/recognize/models")
    onnxs = sorted(f for f in os.listdir(mdir) if f.endswith(".onnx"))
    if not onnxs:
        raise RuntimeError(f"no .onnx models in {mdir}")
    loaded = []
    for f in onnxs:
        onnxruntime.InferenceSession(
            os.path.join(mdir, f), providers=["CPUExecutionProvider"]
        )
        loaded.append(f)
    return f"{len(loaded)} models load: {', '.join(loaded[:4])}{'...' if len(loaded) > 4 else ''}"

def service_imports():
    """Import EVERY classifier, not just faces.

    pipeline.py imports classifiers lazily inside functions, so a missing
    transitive dep (e.g. PyYAML via rules_engine, used by imagenet and
    musicnn) lets the service start clean and only crash when it first
    classifies a photo. Import them all up front so that surfaces here.
    """
    sys.path.insert(0, "/usr/share/nextcloud/apps/recognize/python")
    import config, db, nc_api, clustering, pipeline, rules_engine  # noqa: F401
    from classifiers.base import build_providers, create_session  # noqa: F401
    from classifiers.faces import FaceClassifier  # noqa: F401
    from classifiers.imagenet import ImageNetClassifier  # noqa: F401
    from classifiers.landmarks import LandmarkClassifier  # noqa: F401
    from classifiers.movinet import MoViNetClassifier  # noqa: F401
    from classifiers.musicnn import AudioClassifier  # noqa: F401
    return "all service modules + 6 classifiers import"


def insightface_load():
    """The real thing: load FaceAnalysis exactly as classifiers.faces does."""
    import os
    mdir = os.environ.get("MODELS_DIR", "/usr/share/nextcloud/apps/recognize/models")
    sys.path.insert(0, "/usr/share/nextcloud/apps/recognize/python")
    from classifiers.faces import FaceClassifier
    fc = FaceClassifier(models_dir=mdir, gpu=True)
    return f"FaceAnalysis loaded from {os.path.join(mdir, 'insightface')}"


def no_cpu_onnxruntime_shadow():
    """onnxruntime (CPU) and onnxruntime-gpu share the same namespace.

    insightface depends on plain `onnxruntime`, so pip will happily install
    it over onnxruntime-gpu and silently strip the CUDA provider. Catch the
    collision directly rather than inferring it from a missing provider.
    """
    import glob, os, onnxruntime
    sp = os.path.dirname(os.path.dirname(onnxruntime.__file__))
    cpu = glob.glob(os.path.join(sp, "onnxruntime-*.dist-info"))
    gpu = glob.glob(os.path.join(sp, "onnxruntime_gpu-*.dist-info"))
    if cpu and gpu:
        raise RuntimeError(
            f"BOTH installed: {os.path.basename(cpu[0])} shadows "
            f"{os.path.basename(gpu[0])} - reinstall onnxruntime-gpu alone")
    if not gpu:
        raise RuntimeError("onnxruntime-gpu not installed")
    return os.path.basename(gpu[0])

print(f"running as uid={os.getuid()} python={sys.executable}")
for n, f in [
    ("stdlib imports", stdlib), ("base_prefix resolves", prefix_exists),
    ("numpy", numpy_), ("onnxruntime", onnx),
    ("no CPU onnxruntime shadow", no_cpu_onnxruntime_shadow),
    ("onnxruntime CUDA", onnx_gpu),
    ("insightface", insight), ("scikit-learn", sklearn_), ("opencv", cv2_),
    ("PIL/soundfile/requests/scipy", others), ("mysql-connector", mysql),
    ("DB connects", db_connect), ("ONNX models load", models_load),
    ("service modules", service_imports),
    ("insightface FaceAnalysis loads", insightface_load),
]:
    check(n, f)

print()
if FAILED:
    print(f"RESULT: FAIL ({len(FAILED)}): {', '.join(FAILED)}")
    sys.exit(1)
print("RESULT: PASS - venv is usable by this user")
