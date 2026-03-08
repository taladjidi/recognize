"""HTTP client for the Nextcloud internal API endpoints.

Handles tag assignment (via PHP ORM for proper event dispatch) and
file streaming (for S3/encrypted storage).
"""

import logging
import time
from urllib.parse import urljoin

import requests

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds


class NextcloudAPIError(Exception):
    """Raised when the Nextcloud internal API returns an error."""
    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


class NextcloudAPI:
    """Client for the recognize app's internal PHP API.

    Authenticates via a shared secret in the X-Recognize-Secret header.

    Usage::

        api = NextcloudAPI("http://localhost:8080", "secret123")
        api.submit_results(file_id=42, tags=["Cat", "Animal"])
        content, mimetype = api.get_file(file_id=42)
    """

    def __init__(self, base_url: str, secret: str, timeout: int = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "X-Recognize-Secret": secret,
            "Accept": "application/json",
        })

    def submit_results(
        self,
        file_id: int,
        tags: list[str] | None = None,
        faces: list[dict] | None = None,
    ) -> bool:
        """Submit classification results for a file.

        Tags are written via PHP's ISystemTagManager (fires events, updates etags).
        Faces are written to recognize_face_detections.

        Args:
            file_id: Nextcloud file ID.
            tags: List of tag names (e.g., ["Cat", "Animal"]).
            faces: List of face dicts with keys: x, y, width, height, vector.

        Returns:
            True on success.

        Raises:
            NextcloudAPIError: On HTTP error or invalid response.
        """
        url = f"{self.base_url}/apps/recognize/api/internal/results"
        payload = {"file_id": file_id}
        if tags:
            payload["tags"] = tags
        if faces:
            payload["faces"] = faces

        response = self._request("POST", url, json=payload)
        return response.get("status") == "ok"

    def get_file(self, file_id: int) -> tuple[bytes, str]:
        """Download a file via the PHP storage abstraction.

        Used for files on S3/encrypted/SMB storage that Python can't
        read directly from disk.

        Args:
            file_id: Nextcloud file ID.

        Returns:
            Tuple of (file_content_bytes, mimetype_string).

        Raises:
            NextcloudAPIError: On HTTP error or file not found.
        """
        url = f"{self.base_url}/apps/recognize/api/internal/file/{file_id}"

        for attempt in range(MAX_RETRIES):
            try:
                resp = self._session.get(url, timeout=self.timeout, stream=True)
                if resp.status_code == 200:
                    content = resp.content
                    mimetype = resp.headers.get("Content-Type", "application/octet-stream")
                    return content, mimetype
                elif resp.status_code == 404:
                    raise NextcloudAPIError(f"File {file_id} not found", 404)
                elif resp.status_code >= 500:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_BACKOFF * (attempt + 1))
                        continue
                    raise NextcloudAPIError(
                        f"Server error {resp.status_code}: {resp.text[:200]}",
                        resp.status_code,
                    )
                else:
                    raise NextcloudAPIError(
                        f"Unexpected status {resp.status_code}: {resp.text[:200]}",
                        resp.status_code,
                    )
            except requests.ConnectionError as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF * (attempt + 1))
                    continue
                raise NextcloudAPIError(f"Connection failed: {e}") from e

        raise NextcloudAPIError("Max retries exceeded")

    def _request(self, method: str, url: str, **kwargs) -> dict:
        """Make an HTTP request with retry logic.

        Returns the parsed JSON response body.
        """
        for attempt in range(MAX_RETRIES):
            try:
                resp = self._session.request(
                    method, url, timeout=self.timeout, **kwargs
                )
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code >= 500:
                    log.warning(
                        "API server error %d on attempt %d: %s",
                        resp.status_code, attempt + 1, resp.text[:200],
                    )
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_BACKOFF * (attempt + 1))
                        continue
                    raise NextcloudAPIError(
                        f"Server error {resp.status_code} after {MAX_RETRIES} retries",
                        resp.status_code,
                    )
                else:
                    raise NextcloudAPIError(
                        f"HTTP {resp.status_code}: {resp.text[:200]}",
                        resp.status_code,
                    )
            except requests.ConnectionError as e:
                log.warning("Connection error on attempt %d: %s", attempt + 1, e)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF * (attempt + 1))
                    continue
                raise NextcloudAPIError(f"Connection failed after {MAX_RETRIES} retries") from e

        raise NextcloudAPIError("Max retries exceeded")

    def close(self):
        """Close the underlying HTTP session."""
        self._session.close()
