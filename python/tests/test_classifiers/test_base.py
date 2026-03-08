"""Tests for classifiers.base module (ONNX session utilities)."""

import numpy as np
import pytest

from classifiers.base import softmax, get_top_k, build_providers


class TestSoftmax:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        assert abs(result.sum() - 1.0) < 1e-6
        assert result[2] > result[1] > result[0]

    def test_batch(self):
        x = np.array([[1.0, 2.0], [3.0, 1.0]])
        result = softmax(x, axis=-1)
        assert result.shape == (2, 2)
        assert abs(result[0].sum() - 1.0) < 1e-6
        assert abs(result[1].sum() - 1.0) < 1e-6

    def test_numerical_stability(self):
        """Large values should not produce inf/nan."""
        x = np.array([1000.0, 1001.0, 1002.0])
        result = softmax(x)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert abs(result.sum() - 1.0) < 1e-6

    def test_negative_values(self):
        x = np.array([-1.0, -2.0, -3.0])
        result = softmax(x)
        assert abs(result.sum() - 1.0) < 1e-6


class TestGetTopK:
    def test_dict_class_names(self):
        values = np.array([0.1, 0.5, 0.3, 0.05, 0.05])
        names = {"0": "cat", "1": "dog", "2": "bird", "3": "fish", "4": "snake"}
        results = get_top_k(values, 3, names)
        assert len(results) == 3
        assert results[0]["className"] == "dog"
        assert results[0]["probability"] == pytest.approx(0.5)
        assert results[1]["className"] == "bird"

    def test_list_class_names(self):
        values = np.array([0.1, 0.8, 0.1])
        names = ["cat", "dog", "bird"]
        results = get_top_k(values, 2, names)
        assert len(results) == 2
        assert results[0]["className"] == "dog"

    def test_k_larger_than_array(self):
        values = np.array([0.5, 0.5])
        names = {"0": "a", "1": "b"}
        results = get_top_k(values, 5, names)
        assert len(results) == 2


class TestBuildProviders:
    def test_cpu_only(self):
        providers = build_providers(gpu=False)
        assert providers == ["CPUExecutionProvider"]

    def test_gpu_includes_cpu_fallback(self):
        providers = build_providers(gpu=True)
        assert "CPUExecutionProvider" in providers
