"""Every ONNX classifier must ask onnxruntime for the same provider list.

Regression guard for "video worker runs on CPU": the video and audio
workers must go through the same CUDA setup as the image classifiers.
onnxruntime is faked (no GPU in CI); the live check is the journal line
"Session created for movinet_a3.onnx, providers: ['CUDAExecutionProvider', ...]".
"""

import sys
import types

import pytest


class _FakeSession:
    calls = []

    def __init__(self, path, opts=None, providers=None):
        _FakeSession.calls.append((path, providers))
        self._providers = [p[0] if isinstance(p, tuple) else p for p in providers]

    def get_providers(self):
        return self._providers

    def get_inputs(self):
        return [types.SimpleNamespace(name="in")]

    get_outputs = get_inputs


@pytest.fixture
def fake_ort(monkeypatch):
    _FakeSession.calls = []
    ort = types.SimpleNamespace(
        InferenceSession=_FakeSession,
        SessionOptions=lambda: types.SimpleNamespace(),
        GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL=1),
        ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL=1),
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    # rules need PyYAML, which the test venv lacks; rules are not under test
    monkeypatch.setitem(sys.modules, "rules_engine",
                        types.SimpleNamespace(load_rules=lambda p: []))
    monkeypatch.setitem(sys.modules, "soundfile", types.SimpleNamespace())
    for m in ("classifiers.base", "classifiers.imagenet", "classifiers.movinet",
              "classifiers.musicnn"):
        monkeypatch.delitem(sys.modules, m, raising=False)
    return _FakeSession


def test_video_and_audio_use_image_providers(fake_ort, tmp_path):
    for name in ("efficientnetv2s.onnx", "movinet_a3.onnx", "yamnet.onnx"):
        (tmp_path / name).write_bytes(b"")
    from classifiers.imagenet import ImageNetClassifier
    from classifiers.movinet import MoViNetClassifier
    from classifiers.musicnn import AudioClassifier

    ImageNetClassifier(str(tmp_path), gpu=True)
    MoViNetClassifier(str(tmp_path), gpu=True)
    AudioClassifier(str(tmp_path), gpu=True)

    provs = [p for _, p in fake_ort.calls]
    assert len(provs) == 3
    assert provs[0][0][0] == "CUDAExecutionProvider"
    assert provs[0] == provs[1] == provs[2]
