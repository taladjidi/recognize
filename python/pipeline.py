"""Classification pipeline: routes files to the appropriate classifier by mimetype.

Each classifier group runs in a **subprocess** to guarantee GPU memory is fully
released between passes.  ONNX Runtime's CUDA allocator never frees memory back
to the device, so the only reliable way to reclaim VRAM is process termination.

Architecture:
  Main process  — orchestrates work, manages DB, submits results
  Worker process — loads model(s), processes files, returns results via pipe, exits

Model groups (ordered by VRAM requirement):
  1. faces     — insightface (det + rec + landmark) ~1-2 GB — runs ALONE
  2. tags      — imagenet + landmarks ~1.4 GB — coexist in VRAM
  3. movinet   — video classifier ~200 MB
  4. musicnn   — audio classifier ~100 MB
"""

import json
import logging
import multiprocessing as mp
import os
import time
import traceback

import numpy as np

from db import DB

log = logging.getLogger(__name__)

# Batch sizes for GPU inference
IMAGENET_BATCH_SIZE = 32
LANDMARK_BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Worker functions — each runs in a subprocess with its own CUDA context
# ---------------------------------------------------------------------------

def _worker_faces(models_dir, gpu, file_items_json, result_pipe):
    """Subprocess: load insightface, detect faces, send results, exit."""
    try:
        from classifiers.faces import FaceClassifier
        clf = FaceClassifier(models_dir, gpu=gpu)
        clf.warm_up()

        results = {}
        for item in json.loads(file_items_json):
            try:
                faces = clf.classify(item["path"])
                results[item["file_id"]] = {"faces": faces}
            except Exception as e:
                log.warning("Face detection error for %s: %s", item["path"], e)
                results[item["file_id"]] = {"faces": []}

        result_pipe.send(results)
    except Exception as e:
        log.error("Face worker failed: %s\n%s", e, traceback.format_exc())
        result_pipe.send({})
    finally:
        result_pipe.close()


def _worker_tags(models_dir, gpu, file_items_json, result_pipe):
    """Subprocess: load imagenet + landmarks, classify, send results, exit."""
    try:
        from classifiers.imagenet import ImageNetClassifier, BATCH_SIZE as IMG_BS
        from classifiers.landmarks import LandmarkClassifier, BATCH_SIZE as LM_BS

        file_items = json.loads(file_items_json)
        results = {item["file_id"]: {"tags": []} for item in file_items}

        # ImageNet
        clf = ImageNetClassifier(models_dir, gpu=gpu)
        clf.warm_up()
        preprocessed = []
        for item in file_items:
            try:
                arr = clf.preprocess(item["path"])
                preprocessed.append((item["file_id"], arr))
            except Exception as e:
                log.warning("ImageNet preprocess error for %s: %s", item["path"], e)

        for start in range(0, len(preprocessed), IMG_BS):
            chunk = preprocessed[start:start + IMG_BS]
            batch = np.stack([arr for _, arr in chunk])
            labels_list = clf.infer_batch(batch)
            for (file_id, _), labels in zip(chunk, labels_list):
                results[file_id]["tags"].extend(labels)

        # Landmarks
        clf2 = LandmarkClassifier(models_dir, gpu=gpu)
        clf2.warm_up()
        preprocessed = []
        for item in file_items:
            try:
                arr = clf2.preprocess(item["path"])
                preprocessed.append((item["file_id"], arr))
            except Exception as e:
                log.warning("Landmark preprocess error for %s: %s", item["path"], e)

        for start in range(0, len(preprocessed), LM_BS):
            chunk = preprocessed[start:start + LM_BS]
            batch = np.stack([arr for _, arr in chunk])
            labels_list = clf2.infer_batch(batch)
            for (file_id, _), labels in zip(chunk, labels_list):
                results[file_id]["tags"].extend(labels)

        result_pipe.send(results)
    except Exception as e:
        log.error("Tags worker failed: %s\n%s", e, traceback.format_exc())
        result_pipe.send({})
    finally:
        result_pipe.close()


def _worker_video(models_dir, gpu, ffmpeg_binary, file_items_json, result_pipe):
    """Subprocess: load movinet, classify videos, send results, exit."""
    try:
        from classifiers.movinet import MoViNetClassifier
        clf = MoViNetClassifier(models_dir, gpu=gpu, ffmpeg_binary=ffmpeg_binary)
        clf.warm_up()

        results = {}
        for item in json.loads(file_items_json):
            try:
                labels = clf.classify(item["path"])
                results[item["file_id"]] = {"tags": labels}
            except Exception as e:
                log.warning("MoViNet error for %s: %s", item["path"], e)
                results[item["file_id"]] = {"tags": []}

        result_pipe.send(results)
    except Exception as e:
        log.error("Video worker failed: %s\n%s", e, traceback.format_exc())
        result_pipe.send({})
    finally:
        result_pipe.close()


def _worker_audio(models_dir, gpu, ffmpeg_binary, file_items_json, result_pipe):
    """Subprocess: load musicnn, classify audio, send results, exit."""
    try:
        from classifiers.musicnn import AudioClassifier
        clf = AudioClassifier(models_dir, gpu=gpu, ffmpeg_binary=ffmpeg_binary)
        clf.warm_up()

        results = {}
        for item in json.loads(file_items_json):
            try:
                labels = clf.classify(item["path"])
                results[item["file_id"]] = {"tags": labels}
            except Exception as e:
                log.warning("Audio error for %s: %s", item["path"], e)
                results[item["file_id"]] = {"tags": []}

        result_pipe.send(results)
    except Exception as e:
        log.error("Audio worker failed: %s\n%s", e, traceback.format_exc())
        result_pipe.send({})
    finally:
        result_pipe.close()


# ---------------------------------------------------------------------------
# Pipeline — orchestrates subprocess workers
# ---------------------------------------------------------------------------

class Pipeline:
    """Classification pipeline using subprocess isolation for GPU memory.

    Each classifier group runs in a child process that exits after processing,
    guaranteeing full GPU memory release between passes.
    """

    def __init__(self, config, models_dir, gpu=True, ffmpeg_binary="/usr/bin/ffmpeg",
                 nc_api=None):
        self.config = config
        self.db = DB(config)
        self.models_dir = models_dir
        self.gpu = gpu
        self.ffmpeg_binary = ffmpeg_binary
        self.nc_api = nc_api
        self._enabled = set()

    def enable_classifiers(self, names=None):
        if names is None:
            self._enabled = {"imagenet", "landmarks", "faces", "movinet", "musicnn"}
        else:
            self._enabled = set(names)
        log.info("Enabled classifiers: %s", self._enabled)

    def _run_in_subprocess(self, target, args, timeout=600):
        """Run a worker function in a subprocess, return results via pipe.

        The subprocess gets its own CUDA context. When it exits, all GPU
        memory is freed by the OS — no leaks possible.

        Args:
            target: Worker function (must accept result_pipe as last arg).
            args: Args to pass before result_pipe.
            timeout: Max seconds to wait for the worker.

        Returns:
            Results dict from the worker, or empty dict on failure.
        """
        parent_conn, child_conn = mp.Pipe(duplex=False)
        proc = mp.Process(target=target, args=(*args, child_conn), daemon=True)
        proc.start()
        child_conn.close()  # Parent doesn't write to child's end

        try:
            if parent_conn.poll(timeout):
                results = parent_conn.recv()
            else:
                log.error("Worker %s timed out after %ds", target.__name__, timeout)
                proc.kill()
                results = {}
        except EOFError:
            log.error("Worker %s crashed (pipe closed)", target.__name__)
            results = {}
        finally:
            parent_conn.close()
            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()
                proc.join()

        return results

    def _serialize_items(self, file_items):
        """Serialize file items to JSON for passing to subprocess."""
        return json.dumps([{"file_id": i["file_id"], "path": i["path"]}
                          for i in file_items])

    def process_images_faces(self, file_items):
        """Run face detection in a subprocess."""
        if "faces" not in self._enabled or not file_items:
            return {}
        t0 = time.monotonic()
        results = self._run_in_subprocess(
            _worker_faces,
            (self.models_dir, self.gpu, self._serialize_items(file_items)),
        )
        log.info("Faces subprocess: %d items in %.1fs",
                 len(file_items), time.monotonic() - t0)
        return results

    def process_images_tags(self, file_items):
        """Run imagenet + landmarks in a subprocess."""
        if not file_items:
            return {}
        if "imagenet" not in self._enabled and "landmarks" not in self._enabled:
            return {}
        t0 = time.monotonic()
        results = self._run_in_subprocess(
            _worker_tags,
            (self.models_dir, self.gpu, self._serialize_items(file_items)),
        )
        log.info("Tags subprocess: %d items in %.1fs",
                 len(file_items), time.monotonic() - t0)
        return results

    def process_videos(self, file_items):
        """Run movinet in a subprocess."""
        if "movinet" not in self._enabled or not file_items:
            return {}
        t0 = time.monotonic()
        results = self._run_in_subprocess(
            _worker_video,
            (self.models_dir, self.gpu, self.ffmpeg_binary,
             self._serialize_items(file_items)),
        )
        log.info("Video subprocess: %d items in %.1fs",
                 len(file_items), time.monotonic() - t0)
        return results

    def process_audio(self, file_items):
        """Run musicnn in a subprocess."""
        if "musicnn" not in self._enabled or not file_items:
            return {}
        t0 = time.monotonic()
        results = self._run_in_subprocess(
            _worker_audio,
            (self.models_dir, self.gpu, self.ffmpeg_binary,
             self._serialize_items(file_items)),
        )
        log.info("Audio subprocess: %d items in %.1fs",
                 len(file_items), time.monotonic() - t0)
        return results

    def _submit_result(self, file_id, result):
        """Submit classification results for a file."""
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
