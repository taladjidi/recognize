"""Tests for the classification pipeline."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pipeline import Pipeline, MEDIA_CLASSIFIERS
from db import DB

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "res")


@pytest.fixture
def sqlite_config():
    return {
        "dbtype": "sqlite3",
        "dbhost": "",
        "dbuser": "",
        "dbpassword": "",
        "dbname": ":memory:",
        "dbtableprefix": "oc_",
        "datadirectory": "/tmp/data",
    }


class TestMediaClassifierMapping:
    def test_image_classifiers(self):
        assert "imagenet" in MEDIA_CLASSIFIERS["image"]
        assert "landmarks" in MEDIA_CLASSIFIERS["image"]
        assert "faces" in MEDIA_CLASSIFIERS["image"]

    def test_video_classifiers(self):
        assert "movinet" in MEDIA_CLASSIFIERS["video"]

    def test_audio_classifiers(self):
        assert "musicnn" in MEDIA_CLASSIFIERS["audio"]


class TestInit:
    def test_pipeline_creates_db(self, sqlite_config):
        pipeline = Pipeline(sqlite_config, MODELS_DIR, gpu=False)
        assert pipeline.db is not None
        pipeline.close()

    def test_enable_all_classifiers(self, sqlite_config):
        pipeline = Pipeline(sqlite_config, MODELS_DIR, gpu=False)
        pipeline.enable_classifiers()
        assert pipeline._enabled == {"imagenet", "landmarks", "faces", "movinet", "musicnn"}
        pipeline.close()

    def test_enable_subset(self, sqlite_config):
        pipeline = Pipeline(sqlite_config, MODELS_DIR, gpu=False)
        pipeline.enable_classifiers({"imagenet", "faces"})
        assert pipeline._enabled == {"imagenet", "faces"}
        pipeline.close()


class TestLazyInit:
    def test_classifier_not_loaded_until_needed(self, sqlite_config):
        pipeline = Pipeline(sqlite_config, MODELS_DIR, gpu=False)
        pipeline.enable_classifiers({"imagenet"})
        assert len(pipeline._classifiers) == 0
        pipeline.close()


class TestProcessImages:
    @pytest.fixture
    def pipeline(self, sqlite_config):
        p = Pipeline(sqlite_config, MODELS_DIR, gpu=False)
        p.enable_classifiers({"imagenet"})
        yield p
        p.close()

    def test_process_single_image(self, pipeline):
        img_path = os.path.join(FIXTURES_DIR, "alpine.JPG")
        if not os.path.isfile(img_path):
            pytest.skip("Test fixture not found")
        if not os.path.isfile(os.path.join(MODELS_DIR, "efficientnetv2s.onnx")):
            pytest.skip("ImageNet ONNX model not found")

        items = [{"file_id": 1, "path": img_path, "mimetype": "image/jpeg"}]
        results = pipeline.process_images(items)

        assert 1 in results
        assert "tags" in results[1]
        assert isinstance(results[1]["tags"], list)

    def test_process_multiple_images(self, pipeline):
        alpine = os.path.join(FIXTURES_DIR, "alpine.JPG")
        food = os.path.join(FIXTURES_DIR, "casey-lee-awj7sRviVXo-unsplash.jpg")
        if not os.path.isfile(alpine) or not os.path.isfile(food):
            pytest.skip("Test fixtures not found")
        if not os.path.isfile(os.path.join(MODELS_DIR, "efficientnetv2s.onnx")):
            pytest.skip("ImageNet ONNX model not found")

        items = [
            {"file_id": 1, "path": alpine, "mimetype": "image/jpeg"},
            {"file_id": 2, "path": food, "mimetype": "image/jpeg"},
        ]
        results = pipeline.process_images(items)

        assert 1 in results
        assert 2 in results
        assert len(results[1]["tags"]) > 0
        assert len(results[2]["tags"]) > 0

    def test_invalid_path_handled(self, pipeline):
        if not os.path.isfile(os.path.join(MODELS_DIR, "efficientnetv2s.onnx")):
            pytest.skip("ImageNet ONNX model not found")

        items = [{"file_id": 99, "path": "/nonexistent/image.jpg", "mimetype": "image/jpeg"}]
        results = pipeline.process_images(items)
        # Should not crash; file_id should still be in results (with empty tags)
        assert 99 in results


class TestProcessVideos:
    def test_process_gif(self, sqlite_config):
        gif_path = os.path.join(FIXTURES_DIR, "jumpingjack.gif")
        if not os.path.isfile(gif_path):
            pytest.skip("jumpingjack.gif not found")
        if not os.path.isfile(os.path.join(MODELS_DIR, "movinet_a3.onnx")):
            pytest.skip("MoViNet ONNX model not found")

        pipeline = Pipeline(sqlite_config, MODELS_DIR, gpu=False)
        pipeline.enable_classifiers({"movinet"})

        items = [{"file_id": 1, "path": gif_path, "mimetype": "image/gif"}]
        results = pipeline.process_videos(items)

        assert 1 in results
        assert "tags" in results[1]
        pipeline.close()

    def test_disabled_returns_empty(self, sqlite_config):
        pipeline = Pipeline(sqlite_config, MODELS_DIR, gpu=False)
        pipeline.enable_classifiers({"imagenet"})  # movinet not enabled

        items = [{"file_id": 1, "path": "/some/video.mp4", "mimetype": "video/mp4"}]
        results = pipeline.process_videos(items)

        assert results[1]["tags"] == []
        pipeline.close()


class TestProcessAudio:
    def test_process_mp3(self, sqlite_config):
        mp3_path = os.path.join(FIXTURES_DIR, "Rock_Rejam.mp3")
        if not os.path.isfile(mp3_path):
            pytest.skip("Rock_Rejam.mp3 not found")
        has_yamnet = os.path.isfile(os.path.join(MODELS_DIR, "yamnet.onnx"))
        has_musicnn = os.path.isfile(os.path.join(MODELS_DIR, "musicnn.onnx"))
        if not has_yamnet and not has_musicnn:
            pytest.skip("No audio ONNX model found")

        pipeline = Pipeline(sqlite_config, MODELS_DIR, gpu=False)
        pipeline.enable_classifiers({"musicnn"})

        items = [{"file_id": 1, "path": mp3_path, "mimetype": "audio/mpeg"}]
        results = pipeline.process_audio(items)

        assert 1 in results
        assert "tags" in results[1]
        pipeline.close()


class TestProcessBatch:
    def test_empty_batch(self, sqlite_config):
        pipeline = Pipeline(sqlite_config, MODELS_DIR, gpu=False)
        pipeline.enable_classifiers()
        assert pipeline.process_batch([]) == 0
        pipeline.close()

    def test_missing_file_skipped(self, sqlite_config):
        pipeline = Pipeline(sqlite_config, MODELS_DIR, gpu=False)
        pipeline.enable_classifiers()
        pipeline.db.create_pending_table()

        rows = [{
            "id": 1, "file_id": 42,
            "path": "/nonexistent/file.jpg",
            "mimetype": "image/jpeg",
        }]
        processed = pipeline.process_batch(rows)
        assert processed == 0
        pipeline.close()

    def test_unsupported_mime_skipped(self, sqlite_config):
        pipeline = Pipeline(sqlite_config, MODELS_DIR, gpu=False)
        pipeline.enable_classifiers()
        pipeline.db.create_pending_table()

        rows = [{
            "id": 1, "file_id": 42,
            "path": "/some/file.pdf",
            "mimetype": "application/pdf",
        }]
        processed = pipeline.process_batch(rows)
        assert processed == 0
        pipeline.close()


class TestSubmitResult:
    def test_submit_via_nc_api(self, sqlite_config):
        mock_api = MagicMock()
        pipeline = Pipeline(sqlite_config, MODELS_DIR, gpu=False, nc_api=mock_api)

        pipeline._submit_result(42, {"tags": ["Cat", "Animal"], "faces": []})

        mock_api.submit_results.assert_called_once_with(
            file_id=42, tags=["Cat", "Animal"], faces=None
        )
        pipeline.close()

    def test_submit_with_faces(self, sqlite_config):
        mock_api = MagicMock()
        pipeline = Pipeline(sqlite_config, MODELS_DIR, gpu=False, nc_api=mock_api)

        faces = [{"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4, "vector": [0.1] * 512}]
        pipeline._submit_result(42, {"tags": [], "faces": faces})

        mock_api.submit_results.assert_called_once_with(
            file_id=42, tags=None, faces=faces
        )
        pipeline.close()

    def test_submit_no_api_no_crash(self, sqlite_config):
        """Without nc_api, results are just logged (no crash)."""
        pipeline = Pipeline(sqlite_config, MODELS_DIR, gpu=False, nc_api=None)
        pipeline._submit_result(42, {"tags": ["Cat"], "faces": []})
        pipeline.close()

    def test_api_error_handled(self, sqlite_config):
        mock_api = MagicMock()
        mock_api.submit_results.side_effect = Exception("Connection refused")
        pipeline = Pipeline(sqlite_config, MODELS_DIR, gpu=False, nc_api=mock_api)

        # Should not raise
        pipeline._submit_result(42, {"tags": ["Cat"], "faces": []})
        pipeline.close()


class TestMimetypeRouting:
    def test_classify_mimetype(self):
        assert DB.classify_mimetype("image/jpeg") == "image"
        assert DB.classify_mimetype("image/gif") == "video"  # GIF → video
        assert DB.classify_mimetype("video/mp4") == "video"
        assert DB.classify_mimetype("audio/mpeg") == "audio"
        assert DB.classify_mimetype("application/pdf") is None
