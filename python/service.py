"""Recognize service daemon.

Long-running process that polls the recognize_pending table, routes files
to classifiers, and submits results. Designed to run under systemd.

Features:
  - Adaptive polling: 100ms -> 30s exponential backoff when idle, instant reset
  - Graceful shutdown on SIGTERM/SIGINT
  - Periodic face clustering
  - Maintenance mode detection (pauses during Nextcloud maintenance)
  - Health logging
"""

import argparse
import logging
import multiprocessing as mp
import os
import signal
import sys
import time

log = logging.getLogger("recognize")

# Adaptive polling parameters
POLL_MIN_INTERVAL = 0.1      # 100ms when busy
POLL_MAX_INTERVAL = 30.0     # 30s when idle
POLL_BACKOFF_FACTOR = 1.5    # Multiply interval on empty poll
BATCH_SIZE = 200             # Files per classifier pass
MAX_PASS_FILES = 5000        # Max files per single classifier pass

# Clustering interval
CLUSTER_INTERVAL = 300       # Re-cluster every 5 minutes


class Service:
    """Main service daemon."""

    def __init__(self, config_path, models_dir=None, gpu=True,
                 ffmpeg_binary="/usr/bin/ffmpeg"):
        """Initialize the service.

        Args:
            config_path: Path to nc_config.json.
            models_dir: Path to ONNX models directory. Defaults to ../models/.
            gpu: Whether to use GPU providers.
            ffmpeg_binary: Path to ffmpeg binary.
        """
        from config import load, get_models_dir
        from pipeline import Pipeline
        from nc_api import NextcloudAPI

        self.config = load(config_path)
        self.models_dir = models_dir or get_models_dir(self.config)

        # Set up NC API for result submission
        nc_url = self.config.get("nextcloud_url", "")
        nc_secret = self.config.get("internal_secret", "")
        nc_api = None
        if nc_url and nc_secret:
            nc_api = NextcloudAPI(nc_url, nc_secret)
            log.info("NC API configured: %s", nc_url)

        self.pipeline = Pipeline(
            self.config, self.models_dir,
            gpu=gpu, ffmpeg_binary=ffmpeg_binary,
            nc_api=nc_api,
        )
        self.pipeline.enable_classifiers()

        self._running = False
        self._poll_interval = POLL_MIN_INTERVAL
        self._last_cluster_time = 0.0

    def start(self):
        """Start the service main loop."""
        self._running = True
        self._setup_signals()

        log.info("Service starting (models_dir=%s)", self.models_dir)

        try:
            self._main_loop()
        except KeyboardInterrupt:
            log.info("Interrupted")
        finally:
            self.stop()

    def stop(self):
        """Clean shutdown."""
        self._running = False
        log.info("Service stopping")
        self.pipeline.close()
        log.info("Service stopped")

    def _setup_signals(self):
        """Register signal handlers for graceful shutdown."""
        try:
            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGINT, self._handle_signal)
        except ValueError:
            # signal() only works in main thread; skip in tests/subthreads
            log.debug("Signal handlers not registered (not main thread)")

    def _handle_signal(self, signum, frame):
        """Signal handler: set running=False for graceful exit."""
        sig_name = signal.Signals(signum).name
        log.info("Received %s, shutting down gracefully...", sig_name)
        self._running = False

    def _main_loop(self):
        """Classifier-pass loop: load one model group, process all files, unload.

        Instead of loading all models and processing small batches, this loads
        one classifier at a time and processes all pending files for that type.
        This maximizes GPU utilization on limited VRAM (8GB).

        Pass order: faces → imagenet+landmarks → movinet → musicnn
        """
        while self._running:
            # Check maintenance mode
            if self.pipeline.db.check_maintenance_mode():
                log.info("Nextcloud in maintenance mode, waiting...")
                time.sleep(POLL_MAX_INTERVAL)
                continue

            had_work = self._run_classifier_passes()

            if not had_work:
                # Exponential backoff when idle
                self._poll_interval = min(
                    self._poll_interval * POLL_BACKOFF_FACTOR,
                    POLL_MAX_INTERVAL,
                )
            else:
                self._poll_interval = POLL_MIN_INTERVAL

            # Periodic face clustering
            now = time.monotonic()
            if now - self._last_cluster_time > CLUSTER_INTERVAL:
                self._run_clustering()
                self._last_cluster_time = now

            # Sleep with interruptibility
            if self._running:
                time.sleep(self._poll_interval)

    def _fetch_and_resolve(self, limit):
        """Fetch pending rows and resolve to file paths.

        Returns:
            Tuple of (images, videos, audios) — each a list of item dicts,
            or (None, None, None) on error.
        """
        try:
            rows = self.pipeline.db.fetch_pending(limit=limit)
        except Exception as e:
            log.error("Failed to fetch pending: %s", e)
            return None, None, None

        if not rows:
            return [], [], []

        file_ids = [r["file_id"] for r in rows]
        try:
            path_map = self.pipeline.db.resolve_file_paths(file_ids)
        except Exception as e:
            log.error("Failed to resolve paths: %s", e)
            return None, None, None

        images, videos, audios = [], [], []
        orphan_ids = []

        for r in rows:
            info = path_map.get(r["file_id"])
            if not info:
                orphan_ids.append(r["id"])
                continue

            import os
            path = info["path"]
            mimetype = info["mimetype"]
            if not path or not os.path.isfile(path):
                orphan_ids.append(r["id"])
                continue

            item = {"id": r["id"], "file_id": r["file_id"],
                    "path": path, "mimetype": mimetype}
            media_type = self.pipeline.db.classify_mimetype(mimetype)
            if media_type == "image":
                images.append(item)
            elif media_type == "video":
                videos.append(item)
            elif media_type == "audio":
                audios.append(item)
            else:
                orphan_ids.append(r["id"])

        if orphan_ids:
            log.info("Removing %d orphaned pending entries", len(orphan_ids))
            self.pipeline.db.delete_pending(orphan_ids)

        return images, videos, audios

    def _run_classifier_passes(self):
        """Run each classifier group as a separate pass over pending files.

        Returns True if any work was done.
        """
        images, videos, audios = self._fetch_and_resolve(limit=BATCH_SIZE)
        if images is None:
            time.sleep(POLL_MAX_INTERVAL)
            return False

        if not images and not videos and not audios:
            return False

        had_work = False
        pending_count = self.pipeline.db.pending_count()

        # === Pass 1: Faces (subprocess — gets full VRAM) ===
        if images and "faces" in self.pipeline._enabled:
            face_results = self.pipeline.process_images_faces(images)
            for item in images:
                fid = item["file_id"]
                result = face_results.get(fid, {})
                if result.get("faces"):
                    self.pipeline._submit_result(fid, {"tags": [], "faces": result["faces"]})
            had_work = True

        # === Pass 2: Tags (subprocess — imagenet + landmarks coexist) ===
        if images and ("imagenet" in self.pipeline._enabled or
                       "landmarks" in self.pipeline._enabled):
            tag_results = self.pipeline.process_images_tags(images)
            processed_ids = []
            for item in images:
                fid = item["file_id"]
                result = tag_results.get(fid, {})
                tags = result.get("tags", [])
                if tags:
                    self.pipeline._submit_result(fid, {"tags": tags, "faces": []})
                processed_ids.append(item["id"])

            if processed_ids:
                self.pipeline.db.delete_pending(processed_ids)
            log.info("Processed %d images (%d pending)", len(images), pending_count)
            had_work = True

        # === Pass 3: Video (subprocess) ===
        if videos:
            vid_results = self.pipeline.process_videos(videos)
            processed_ids = []
            for item in videos:
                fid = item["file_id"]
                self.pipeline._submit_result(fid, vid_results.get(fid, {}))
                processed_ids.append(item["id"])
            if processed_ids:
                self.pipeline.db.delete_pending(processed_ids)
            log.info("Processed %d videos", len(videos))
            had_work = True

        # === Pass 4: Audio (subprocess) ===
        if audios:
            aud_results = self.pipeline.process_audio(audios)
            processed_ids = []
            for item in audios:
                fid = item["file_id"]
                self.pipeline._submit_result(fid, aud_results.get(fid, {}))
                processed_ids.append(item["id"])
            if processed_ids:
                self.pipeline.db.delete_pending(processed_ids)
            log.info("Processed %d audio files", len(audios))
            had_work = True

        return had_work

    def _run_clustering(self):
        """Run face clustering for all users with recent changes."""
        try:
            from clustering import cluster_all_users

            # Get unique users from recent face detections
            user_ids = self.pipeline.db.get_users_with_faces()
            if not user_ids:
                return

            log.info("Running face clustering for %d users", len(user_ids))
            results = cluster_all_users(self.pipeline.db, user_ids=user_ids)

            for user_id, stats in results.items():
                if "error" in stats:
                    log.warning("Clustering failed for %s: %s", user_id, stats["error"])
                elif stats["n_faces"] > 0:
                    log.info(
                        "  %s: %d faces -> %d clusters, %d noise",
                        user_id, stats["n_faces"], stats["n_clusters"], stats["n_noise"],
                    )
        except Exception as e:
            log.error("Clustering failed: %s", e)


def main():
    parser = argparse.ArgumentParser(description="Recognize classification service")
    parser.add_argument("config", help="Path to nc_config.json")
    parser.add_argument("--models-dir", help="Path to ONNX models directory")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    parser.add_argument("--ffmpeg", default="/usr/bin/ffmpeg", help="Path to ffmpeg binary")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--profile", metavar="OUTPUT",
                        help="Enable cProfile, write stats to OUTPUT file")
    parser.add_argument("--ort-profile", action="store_true",
                        help="Enable ONNX Runtime profiling (chrome://tracing JSON)")

    args = parser.parse_args()

    # Use 'spawn' to get clean CUDA contexts in worker subprocesses.
    # 'fork' would inherit the parent's CUDA state and leak memory.
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # Already set

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Auto-configure LD_LIBRARY_PATH for NVIDIA CUDA 12 libs if not already set
    # (systemd sets this via the service template; this handles manual runs)
    if not args.no_gpu and not os.environ.get("LD_LIBRARY_PATH"):
        from config import get_nvidia_lib_path
        nvidia_path = get_nvidia_lib_path()
        if nvidia_path:
            os.environ["LD_LIBRARY_PATH"] = nvidia_path
            log.info("Auto-set LD_LIBRARY_PATH=%s", nvidia_path)

    if args.ort_profile:
        os.environ["ORT_PROFILE"] = "1"

    service = Service(
        config_path=args.config,
        models_dir=args.models_dir,
        gpu=not args.no_gpu,
        ffmpeg_binary=args.ffmpeg,
    )

    if args.profile:
        import cProfile
        profiler = cProfile.Profile()
        log.info("cProfile enabled, output: %s", args.profile)
        profiler.enable()
        try:
            service.start()
        finally:
            profiler.disable()
            profiler.dump_stats(args.profile)
            log.info("Profile written to %s (view with: snakeviz %s)",
                     args.profile, args.profile)
    else:
        service.start()


if __name__ == "__main__":
    main()
