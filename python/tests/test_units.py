"""Unit tests for shared Python modules (no models or media fixtures required).

Tests base_classifier, rules_engine, and gpu_setup functions that can be
tested in isolation without TensorFlow, ONNX, or test media files.

Usage:
    python -m pytest python/tests/test_units.py -v
"""

import io
import json
import os
import sys

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(TESTS_DIR)
REPO_ROOT = os.path.dirname(PYTHON_DIR)
sys.path.insert(0, PYTHON_DIR)


# ── base_classifier tests ────────────────────────────────────────────


class TestGetPaths:
    """Test base_classifier.get_paths() stdin and argv modes."""

    def test_paths_from_argv(self, monkeypatch):
        import base_classifier

        monkeypatch.setattr(sys, "argv", ["script.py", "/a/b.jpg", "/c/d.png"])
        paths = base_classifier.get_paths()
        assert paths == ["/a/b.jpg", "/c/d.png"]

    def test_paths_from_stdin(self, monkeypatch):
        import base_classifier

        monkeypatch.setattr(sys, "argv", ["script.py", "-"])
        monkeypatch.setattr(sys, "stdin", io.StringIO("/a/b.jpg\n/c/d.png\n\n"))
        paths = base_classifier.get_paths()
        assert paths == ["/a/b.jpg", "/c/d.png"]

    def test_empty_lines_filtered(self, monkeypatch):
        import base_classifier

        monkeypatch.setattr(sys, "argv", ["script.py", "-"])
        monkeypatch.setattr(sys, "stdin", io.StringIO("\n\n/x.jpg\n\n"))
        paths = base_classifier.get_paths()
        assert paths == ["/x.jpg"]

    def test_no_args_exits(self, monkeypatch):
        import base_classifier

        monkeypatch.setattr(sys, "argv", ["script.py"])
        with pytest.raises(SystemExit):
            base_classifier.get_paths()


class TestOutputResult:
    """Test base_classifier JSON output protocol."""

    def test_output_result_json(self, capsys):
        import base_classifier

        base_classifier.output_result(["Alpine", "Mountain"])
        captured = capsys.readouterr()
        assert json.loads(captured.out.strip()) == ["Alpine", "Mountain"]

    def test_output_error_empty_list(self, capsys):
        import base_classifier

        base_classifier.output_error()
        captured = capsys.readouterr()
        assert json.loads(captured.out.strip()) == []

    def test_output_result_faces_format(self, capsys):
        import base_classifier

        face = {"x": 0.1, "y": 0.2, "vector": [0.0] * 512, "score": 0.95}
        base_classifier.output_result([face])
        captured = capsys.readouterr()
        result = json.loads(captured.out.strip())
        assert len(result) == 1
        assert result[0]["score"] == 0.95
        assert len(result[0]["vector"]) == 512


class TestGetFfmpegBinary:
    """Test base_classifier.get_ffmpeg_binary() env and PATH lookup."""

    def test_from_env_var(self, monkeypatch, tmp_path):
        import base_classifier

        fake = tmp_path / "ffmpeg"
        fake.write_text("#!/bin/sh\n")
        monkeypatch.setenv("FFMPEG_BINARY", str(fake))
        assert base_classifier.get_ffmpeg_binary() == str(fake)

    def test_from_system_path(self, monkeypatch):
        import base_classifier
        import shutil

        monkeypatch.delenv("FFMPEG_BINARY", raising=False)
        if shutil.which("ffmpeg"):
            result = base_classifier.get_ffmpeg_binary()
            assert "ffmpeg" in result
        else:
            with pytest.raises(RuntimeError, match="ffmpeg not found"):
                base_classifier.get_ffmpeg_binary()

    def test_missing_raises(self, monkeypatch):
        import base_classifier

        monkeypatch.setenv("FFMPEG_BINARY", "/nonexistent/ffmpeg")
        monkeypatch.setattr("shutil.which", lambda x: None)
        with pytest.raises(RuntimeError, match="ffmpeg not found"):
            base_classifier.get_ffmpeg_binary()


# ── rules_engine tests ───────────────────────────────────────────────


class TestRulesEngine:
    """Test rules YAML loading, alias resolution, and threshold filtering."""

    @pytest.fixture
    def sample_rules(self):
        return {
            "cat": {"label": "cat", "threshold": 0.1, "categories": ["animal"]},
            "dog": {"label": "dog", "threshold": 0.2, "categories": ["animal"]},
            "tabby": {"see": "cat"},
            "persian cat": {"see": "tabby"},
            "mountain": {"label": "mountain", "threshold": 0.5},
            "hidden": {"label": "hidden", "threshold": 0.99},
        }

    def test_load_rules_from_yaml(self, tmp_path):
        import rules_engine

        rules_file = tmp_path / "rules.yml"
        rules_file.write_text("cat:\n  label: cat\n  threshold: 0.1\n")
        rules = rules_engine.load_rules(str(rules_file))
        assert "cat" in rules
        assert rules["cat"]["label"] == "cat"

    def test_find_rule_direct(self, sample_rules):
        import rules_engine

        rule = rules_engine.find_rule(sample_rules, "cat")
        assert rule is not None
        assert rule["label"] == "cat"

    def test_find_rule_see_alias(self, sample_rules):
        import rules_engine

        rule = rules_engine.find_rule(sample_rules, "tabby")
        assert rule is not None
        assert rule["label"] == "cat"

    def test_find_rule_chained_alias(self, sample_rules):
        import rules_engine

        rule = rules_engine.find_rule(sample_rules, "persian cat")
        assert rule is not None
        assert rule["label"] == "cat"

    def test_find_rule_missing(self, sample_rules):
        import rules_engine

        rule = rules_engine.find_rule(sample_rules, "nonexistent")
        assert rule is None

    def test_apply_rules_threshold_filter(self, sample_rules):
        import rules_engine

        results = [
            {"className": "cat", "probability": 0.5},
            {"className": "hidden", "probability": 0.3},
        ]
        labels = rules_engine.apply_rules(results, sample_rules)
        assert "cat" in labels
        assert "hidden" not in labels

    def test_apply_rules_uppercase(self, sample_rules):
        import rules_engine

        results = [{"className": "cat", "probability": 0.5}]
        labels = rules_engine.apply_rules(results, sample_rules, uppercase=True)
        assert "Cat" in labels

    def test_apply_rules_no_uppercase(self, sample_rules):
        import rules_engine

        results = [{"className": "cat", "probability": 0.5}]
        labels = rules_engine.apply_rules(results, sample_rules, uppercase=False)
        assert "cat" in labels

    def test_apply_rules_categories(self, sample_rules):
        import rules_engine

        results = [{"className": "cat", "probability": 0.5}]
        labels = rules_engine.apply_rules(results, sample_rules)
        assert "animal" in labels

    def test_apply_rules_category_aggregation(self, sample_rules):
        """When multiple classes share a category, their probabilities aggregate."""
        import rules_engine

        results = [
            {"className": "cat", "probability": 0.15},
            {"className": "dog", "probability": 0.15},
        ]
        # sqrt(0.15^2 + 0.15^2) ≈ 0.212 > max threshold 0.2
        labels = rules_engine.apply_rules(results, sample_rules)
        assert "animal" in labels

    def test_apply_rules_deduplication(self, sample_rules):
        import rules_engine

        results = [
            {"className": "cat", "probability": 0.5},
            {"className": "tabby", "probability": 0.4},
        ]
        labels = rules_engine.apply_rules(results, sample_rules)
        assert labels.count("cat") == 1

    def test_apply_rules_comma_split(self, sample_rules):
        """Class names with commas use only the first part for rule lookup."""
        import rules_engine

        results = [{"className": "cat, domestic", "probability": 0.5}]
        labels = rules_engine.apply_rules(results, sample_rules)
        assert "cat" in labels

    def test_apply_rules_unknown_class(self, sample_rules):
        import rules_engine

        results = [{"className": "spaceship", "probability": 0.99}]
        labels = rules_engine.apply_rules(results, sample_rules)
        assert labels == []

    def test_apply_rules_empty_results(self, sample_rules):
        import rules_engine

        labels = rules_engine.apply_rules([], sample_rules)
        assert labels == []


class TestRulesEngineProduction:
    """Test with the actual rules.yml from the repository."""

    @pytest.fixture
    def real_rules(self):
        import rules_engine

        rules_path = os.path.join(REPO_ROOT, "src", "rules.yml")
        if not os.path.isfile(rules_path):
            pytest.skip("src/rules.yml not found")
        return rules_engine.load_rules(rules_path)

    def test_real_rules_has_many_entries(self, real_rules):
        assert len(real_rules) > 100

    def test_real_rules_cat_exists(self, real_rules):
        import rules_engine

        rule = rules_engine.find_rule(real_rules, "cat")
        assert rule is not None
        assert rule["label"] == "cat"

    def test_real_rules_alp_exists(self, real_rules):
        import rules_engine

        rule = rules_engine.find_rule(real_rules, "alp")
        assert rule is not None


# ── gpu_setup tests ──────────────────────────────────────────────────


class TestGpuSetup:
    """Test gpu_setup.configure() with mocked environment."""

    def test_configure_cpu_mode(self, monkeypatch):
        monkeypatch.setenv("RECOGNIZE_GPU", "false")
        monkeypatch.setenv("TF_CPP_MIN_LOG_LEVEL", "3")
        import gpu_setup

        tf = gpu_setup.configure()
        assert tf is not None
        # In CPU mode, CUDA should be hidden
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""

    def test_configure_returns_tensorflow(self, monkeypatch):
        monkeypatch.setenv("RECOGNIZE_GPU", "false")
        monkeypatch.setenv("TF_CPP_MIN_LOG_LEVEL", "3")
        import gpu_setup

        tf = gpu_setup.configure()
        assert hasattr(tf, "constant")
        assert hasattr(tf, "saved_model")


# ── classifier_geo unit tests ────────────────────────────────────────


class TestGeoConvertDms:
    """Test GPS coordinate conversion (no external dependencies)."""

    def test_convert_north_east(self):
        from unittest.mock import MagicMock

        sys.path.insert(0, PYTHON_DIR)
        from classifier_geo import convert_dms_to_dd

        # Mock exifread IfdTag values: 48°51'24" N, 2°21'7" E (near Eiffel Tower)
        lat_values = [
            MagicMock(num=48, den=1),
            MagicMock(num=51, den=1),
            MagicMock(num=24, den=1),
        ]
        lon_values = [
            MagicMock(num=2, den=1),
            MagicMock(num=21, den=1),
            MagicMock(num=7, den=1),
        ]

        lat = convert_dms_to_dd(lat_values, "N")
        lon = convert_dms_to_dd(lon_values, "E")

        assert 48.85 < lat < 48.86
        assert 2.35 < lon < 2.36

    def test_convert_south_west(self):
        from unittest.mock import MagicMock

        from classifier_geo import convert_dms_to_dd

        values = [
            MagicMock(num=33, den=1),
            MagicMock(num=51, den=1),
            MagicMock(num=54, den=1),
        ]
        dd = convert_dms_to_dd(values, "S")
        assert dd < 0
        assert abs(dd) > 33
