"""Tests for the Nextcloud internal API client using mock HTTP."""

import json
from unittest.mock import patch, MagicMock

import pytest
import requests

from nc_api import NextcloudAPI, NextcloudAPIError


@pytest.fixture
def api():
    return NextcloudAPI("http://localhost:8080", "test-secret", timeout=5)


class TestInit:
    def test_headers_set(self, api):
        assert api._session.headers["X-Recognize-Secret"] == "test-secret"
        assert api._session.headers["Accept"] == "application/json"

    def test_trailing_slash_stripped(self):
        a = NextcloudAPI("http://localhost:8080/", "s")
        assert a.base_url == "http://localhost:8080"


class TestSubmitResults:
    def test_submit_tags(self, api):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok"}

        with patch.object(api._session, "request", return_value=mock_resp) as mock_req:
            result = api.submit_results(file_id=42, tags=["Cat", "Animal"])

        assert result is True
        call_args = mock_req.call_args
        assert call_args[0] == ("POST", "http://localhost:8080/apps/recognize/api/internal/results")
        body = call_args[1]["json"]
        assert body["file_id"] == 42
        assert body["tags"] == ["Cat", "Animal"]

    def test_submit_faces(self, api):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok"}

        faces = [{"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4, "vector": [0.1] * 512}]
        with patch.object(api._session, "request", return_value=mock_resp):
            result = api.submit_results(file_id=42, faces=faces)

        assert result is True

    def test_submit_empty(self, api):
        """Submitting with no tags or faces should still work (marks as processed)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok"}

        with patch.object(api._session, "request", return_value=mock_resp):
            result = api.submit_results(file_id=42)

        assert result is True

    def test_server_error_retries(self, api):
        error_resp = MagicMock()
        error_resp.status_code = 500
        error_resp.text = "Internal Server Error"

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"status": "ok"}

        with patch.object(api._session, "request", side_effect=[error_resp, ok_resp]) as mock:
            with patch("nc_api.time.sleep"):  # Don't actually sleep
                result = api.submit_results(file_id=42, tags=["Cat"])

        assert result is True
        assert mock.call_count == 2

    def test_server_error_exhausts_retries(self, api):
        error_resp = MagicMock()
        error_resp.status_code = 500
        error_resp.text = "Internal Server Error"

        with patch.object(api._session, "request", return_value=error_resp):
            with patch("nc_api.time.sleep"):
                with pytest.raises(NextcloudAPIError, match="Server error 500"):
                    api.submit_results(file_id=42, tags=["Cat"])

    def test_client_error_no_retry(self, api):
        error_resp = MagicMock()
        error_resp.status_code = 403
        error_resp.text = "Forbidden"

        with patch.object(api._session, "request", return_value=error_resp):
            with pytest.raises(NextcloudAPIError, match="HTTP 403"):
                api.submit_results(file_id=42, tags=["Cat"])

    def test_connection_error_retries(self, api):
        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"status": "ok"}

        with patch.object(
            api._session, "request",
            side_effect=[requests.ConnectionError("refused"), ok_resp]
        ):
            with patch("nc_api.time.sleep"):
                result = api.submit_results(file_id=42, tags=["Cat"])

        assert result is True


class TestGetFile:
    def test_get_file_success(self, api):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"\xff\xd8\xff\xe0"  # JPEG magic bytes
        mock_resp.headers = {"Content-Type": "image/jpeg"}

        with patch.object(api._session, "get", return_value=mock_resp):
            content, mimetype = api.get_file(file_id=42)

        assert content == b"\xff\xd8\xff\xe0"
        assert mimetype == "image/jpeg"

    def test_get_file_not_found(self, api):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "Not Found"

        with patch.object(api._session, "get", return_value=mock_resp):
            with pytest.raises(NextcloudAPIError, match="not found"):
                api.get_file(file_id=999)

    def test_get_file_server_error_retries(self, api):
        error_resp = MagicMock()
        error_resp.status_code = 502
        error_resp.text = "Bad Gateway"

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.content = b"data"
        ok_resp.headers = {"Content-Type": "image/png"}

        with patch.object(api._session, "get", side_effect=[error_resp, ok_resp]):
            with patch("nc_api.time.sleep"):
                content, mimetype = api.get_file(file_id=42)

        assert content == b"data"


class TestClose:
    def test_close_session(self, api):
        with patch.object(api._session, "close") as mock_close:
            api.close()
            mock_close.assert_called_once()
