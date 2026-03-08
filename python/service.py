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
import os
import signal
import sys
import time

log = logging.getLogger("recognize")

# Adaptive polling parameters
POLL_MIN_INTERVAL = 0.1      # 100ms when busy
POLL_MAX_INTERVAL = 30.0     # 30s when idle
POLL_BACKOFF_FACTOR = 1.5    # Multiply interval on empty poll
BATCH_SIZE = 50              # Files per poll cycle

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
        """Poll-process-sleep loop with adaptive backoff."""
        while self._running:
            # Check maintenance mode
            if self.pipeline.db.check_maintenance_mode():
                log.info("Nextcloud in maintenance mode, waiting...")
                time.sleep(POLL_MAX_INTERVAL)
                continue

            # Fetch pending files
            try:
                rows = self.pipeline.db.fetch_pending(limit=BATCH_SIZE)
            except Exception as e:
                log.error("Failed to fetch pending: %s", e)
                time.sleep(POLL_MAX_INTERVAL)
                continue

            if rows:
                # Reset backoff on activity
                self._poll_interval = POLL_MIN_INTERVAL

                # Resolve file paths from filecache
                file_ids = [r["file_id"] for r in rows]
                try:
                    path_map = self.pipeline.db.resolve_file_paths(file_ids)
                except Exception as e:
                    log.error("Failed to resolve paths: %s", e)
                    time.sleep(1.0)
                    continue

                # Merge pending rows with resolved paths into the format
                # process_batch expects: {id, file_id, path, mimetype}
                merged = []
                orphan_ids = []
                for r in rows:
                    info = path_map.get(r["file_id"])
                    if info:
                        merged.append({
                            "id": r["id"],
                            "file_id": r["file_id"],
                            "path": info["path"],
                            "mimetype": info["mimetype"],
                        })
                    else:
                        # File not in filecache (deleted?) — remove from queue
                        orphan_ids.append(r["id"])

                if orphan_ids:
                    log.info("Removing %d orphaned pending entries", len(orphan_ids))
                    self.pipeline.db.delete_pending(orphan_ids)

                # Process batch
                t0 = time.monotonic()
                processed = self.pipeline.process_batch(merged)
                elapsed = time.monotonic() - t0

                pending_count = self.pipeline.db.pending_count()
                log.info(
                    "Processed %d/%d files in %.1fs (%d pending)",
                    processed, len(merged), elapsed, pending_count,
                )
            else:
                # Exponential backoff when idle
                self._poll_interval = min(
                    self._poll_interval * POLL_BACKOFF_FACTOR,
                    POLL_MAX_INTERVAL,
                )

            # Periodic face clustering
            now = time.monotonic()
            if now - self._last_cluster_time > CLUSTER_INTERVAL:
                self._run_clustering()
                self._last_cluster_time = now

            # Sleep with interruptibility
            if self._running:
                time.sleep(self._poll_interval)

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
