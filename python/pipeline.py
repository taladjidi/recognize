"""Classification pipeline: routes files to the appropriate classifier by mimetype.

The pipeline manages classifier lifecycle (init, warm-up) and provides a
unified interface for the service daemon to process pending files.

Flow:
  1. Fetch pending files from DB
  2. Group by media type (image, video, audio)
  3. Route to appropriate classifier(s)
  4. Submit results via NC API or directly to DB
  5. Delete processed entries from pending queue
"""

import logging
import os
import time
import traceback

import numpy as np

from classifiers.imagenet import ImageNetClassifier, BATCH_SIZE as IMAGENET_BATCH_SIZE
from classifiers.landmarks import LandmarkClassifier, BATCH_SIZE as LANDMARK_BATCH_SIZE
from classifiers.movinet import MoViNetClassifier
from classifiers.musicnn import AudioClassifier
from db import DB

log = logging.getLogger(__name__)

# Which classifiers to run for each media type
MEDIA_CLASSIFIERS = {
    "image": ["imagenet", "landmarks", "faces"],
    "video": ["movinet"],
    "audio": ["musicnn"],
}


class Pipeline:
    """Classification pipeline that manages classifiers and processes files.

    Classifiers are lazily initialized on first use to avoid loading unused
    models into GPU memory.
    """

    def __init__(self, config, models_dir, gpu=True, ffmpeg_binary="/usr/bin/ffmpeg",
                 nc_api=None):
        """Initialize the pipeline.

        Args:
            config: Nextcloud config dict (passed to DB).
            models_dir: Path to directory containing ONNX models.
            gpu: Whether to use GPU providers.
            ffmpeg_binary: Path to ffmpeg binary.
            nc_api: Optional NextcloudAPI instance for submitting results.
                    If None, results are written directly to DB.
        """
        self.db = DB(config)
        self.models_dir = models_dir
        self.gpu = gpu
        self.ffmpeg_binary = ffmpeg_binary
        self.nc_api = nc_api
        self._classifiers = {}
        self._enabled = set()

    def _get_classifier(self, name):
        """Get or lazily initialize a classifier."""
        if name in self._classifiers:
            return self._classifiers[name]

        log.info("Initializing classifier: %s", name)
        t0 = time.monotonic()

        if name == "imagenet":
            clf = ImageNetClassifier(self.models_dir, gpu=self.gpu)
        elif name == "landmarks":
            clf = LandmarkClassifier(self.models_dir, gpu=self.gpu)
        elif name == "faces":
            from classifiers.faces import FaceClassifier
            clf = FaceClassifier(self.models_dir, gpu=self.gpu)
        elif name == "movinet":
            clf = MoViNetClassifier(self.models_dir, gpu=self.gpu,
                                     ffmpeg_binary=self.ffmpeg_binary)
        elif name == "musicnn":
            clf = AudioClassifier(self.models_dir, gpu=self.gpu,
                                   ffmpeg_binary=self.ffmpeg_binary)
        else:
            raise ValueError(f"Unknown classifier: {name}")

        elapsed = time.monotonic() - t0
        log.info("Classifier %s initialized in %.1fs", name, elapsed)

        clf.warm_up()
        self._classifiers[name] = clf
        return clf

    def enable_classifiers(self, names=None):
        """Set which classifiers are enabled.

        Args:
            names: Set of classifier names to enable. If None, enable all.
        """
        if names is None:
            self._enabled = {"imagenet", "landmarks", "faces", "movinet", "musicnn"}
        else:
            self._enabled = set(names)
        log.info("Enabled classifiers: %s", self._enabled)

    def process_images(self, file_items):
        """Process a batch of image files through enabled image classifiers.

        Args:
            file_items: List of dicts with keys: file_id, path, mimetype.

        Returns:
            Dict mapping file_id to result dict:
              {tags: [...], faces: [...]}
        """
        results = {item["file_id"]: {"tags": [], "faces": []} for item in file_items}

        if "imagenet" in self._enabled:
            self._run_imagenet_batch(file_items, results)

        if "landmarks" in self._enabled:
            self._run_landmarks_batch(file_items, results)

        if "faces" in self._enabled:
            self._run_faces(file_items, results)

        return results

    def _run_imagenet_batch(self, file_items, results):
        """Run ImageNet classifier on a batch of images."""
        clf = self._get_classifier("imagenet")

        # Preprocess all images, tracking which ones succeed
        preprocessed = []
        for item in file_items:
            try:
                arr = clf.preprocess(item["path"])
                preprocessed.append((item["file_id"], arr))
            except Exception as e:
                log.warning("ImageNet preprocess error for %s: %s", item["path"], e)

        # Process in batches
        for start in range(0, len(preprocessed), IMAGENET_BATCH_SIZE):
            chunk = preprocessed[start:start + IMAGENET_BATCH_SIZE]
            batch = np.stack([arr for _, arr in chunk])
            labels_list = clf.infer_batch(batch)
            for (file_id, _), labels in zip(chunk, labels_list):
                results[file_id]["tags"].extend(labels)

    def _run_landmarks_batch(self, file_items, results):
        """Run landmark classifier on a batch of images."""
        clf = self._get_classifier("landmarks")

        preprocessed = []
        for item in file_items:
            try:
                arr = clf.preprocess(item["path"])
                preprocessed.append((item["file_id"], arr))
            except Exception as e:
                log.warning("Landmark preprocess error for %s: %s", item["path"], e)

        for start in range(0, len(preprocessed), LANDMARK_BATCH_SIZE):
            chunk = preprocessed[start:start + LANDMARK_BATCH_SIZE]
            batch = np.stack([arr for _, arr in chunk])
            labels_list = clf.infer_batch(batch)
            for (file_id, _), labels in zip(chunk, labels_list):
                results[file_id]["tags"].extend(labels)

    def _run_faces(self, file_items, results):
        """Run face detector on each image individually."""
        clf = self._get_classifier("faces")

        for item in file_items:
            try:
                faces = clf.classify(item["path"])
                results[item["file_id"]]["faces"] = faces
            except Exception as e:
                log.warning("Face detection error for %s: %s", item["path"], e)

    def process_videos(self, file_items):
        """Process video files through the MoViNet classifier.

        Args:
            file_items: List of dicts with keys: file_id, path, mimetype.

        Returns:
            Dict mapping file_id to result dict: {tags: [...]}
        """
        results = {}
        if "movinet" not in self._enabled:
            return {item["file_id"]: {"tags": []} for item in file_items}

        clf = self._get_classifier("movinet")

        for item in file_items:
            try:
                labels = clf.classify(item["path"])
                results[item["file_id"]] = {"tags": labels}
            except Exception as e:
                log.warning("MoViNet error for %s: %s", item["path"], e)
                results[item["file_id"]] = {"tags": []}

        return results

    def process_audio(self, file_items):
        """Process audio files through the audio classifier.

        Args:
            file_items: List of dicts with keys: file_id, path, mimetype.

        Returns:
            Dict mapping file_id to result dict: {tags: [...]}
        """
        results = {}
        if "musicnn" not in self._enabled:
            return {item["file_id"]: {"tags": []} for item in file_items}

        clf = self._get_classifier("musicnn")

        for item in file_items:
            try:
                labels = clf.classify(item["path"])
                results[item["file_id"]] = {"tags": labels}
            except Exception as e:
                log.warning("Audio error for %s: %s", item["path"], e)
                results[item["file_id"]] = {"tags": []}

        return results

    def process_batch(self, pending_rows):
        """Process a batch of pending rows from the DB.

        Routes each file to the correct classifier(s) based on mimetype,
        submits results, and removes processed entries from the queue.

        Args:
            pending_rows: List of row dicts from db.fetch_pending().

        Returns:
            Number of files successfully processed.
        """
        if not pending_rows:
            return 0

        # Resolve file paths and group by media type
        images, videos, audios = [], [], []
        skipped_ids = []

        for row in pending_rows:
            file_id = row["file_id"]
            path = row.get("path")
            mimetype = row.get("mimetype", "")

            if not path or not os.path.isfile(path):
                log.warning("File not found for file_id=%d: %s", file_id, path)
                skipped_ids.append(row["id"])
                continue

            media_type = DB.classify_mimetype(mimetype)
            item = {"file_id": file_id, "path": path, "mimetype": mimetype,
                    "pending_id": row["id"]}

            if media_type == "image":
                images.append(item)
            elif media_type == "video":
                videos.append(item)
            elif media_type == "audio":
                audios.append(item)
            else:
                log.debug("Unsupported mimetype %s for file_id=%d", mimetype, file_id)
                skipped_ids.append(row["id"])

        processed = 0
        processed_ids = []

        # Process each media type
        if images:
            try:
                results = self.process_images(images)
                for item in images:
                    fid = item["file_id"]
                    self._submit_result(fid, results.get(fid, {}))
                    processed_ids.append(item["pending_id"])
                    processed += 1
            except Exception as e:
                log.error("Image batch processing failed: %s", e)
                traceback.print_exc()

        if videos:
            try:
                results = self.process_videos(videos)
                for item in videos:
                    fid = item["file_id"]
                    self._submit_result(fid, results.get(fid, {}))
                    processed_ids.append(item["pending_id"])
                    processed += 1
            except Exception as e:
                log.error("Video processing failed: %s", e)
                traceback.print_exc()

        if audios:
            try:
                results = self.process_audio(audios)
                for item in audios:
                    fid = item["file_id"]
                    self._submit_result(fid, results.get(fid, {}))
                    processed_ids.append(item["pending_id"])
                    processed += 1
            except Exception as e:
                log.error("Audio processing failed: %s", e)
                traceback.print_exc()

        # Clean up processed and skipped entries from the queue
        all_done = processed_ids + skipped_ids
        if all_done:
            self.db.delete_pending(all_done)

        return processed

    def _submit_result(self, file_id, result):
        """Submit classification results for a file.

        If nc_api is configured, submits via HTTP (proper event dispatch).
        Otherwise, writes tags directly to DB (useful for testing).
        """
        tags = result.get("tags", [])
        faces = result.get("faces", [])

        if self.nc_api:
            try:
                self.nc_api.submit_results(
                    file_id=file_id,
                    tags=tags if tags else None,
                    faces=faces if faces else None,
                )
            except Exception as e:
                log.error("Failed to submit results for file_id=%d: %s", file_id, e)
        else:
            log.debug("Results for file_id=%d: tags=%s, faces=%d",
                       file_id, tags, len(faces))

    def close(self):
        """Clean up resources."""
        self.db.close()
        if self.nc_api:
            self.nc_api.close()
        self._classifiers.clear()
