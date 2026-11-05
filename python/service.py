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
        # Honour the admin page's per-model toggles. Previously every
        # classifier ran regardless, so the settings had no effect.
        self._refresh_enabled()
        self._warned_unsupported = set()

        self._running = False
        self._poll_interval = POLL_MIN_INTERVAL
        self._last_cluster_time = 0.0
        self._cluster_state = {}  # user_id -> unclustered signature of last run

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
            # Heartbeat: the admin page reads daemon.lastSeen to tell whether
            # the daemon is alive. Written before the maintenance-mode check so
            # a paused daemon still reports itself as running.
            self.pipeline.db.set_app_value("daemon.lastSeen", str(int(time.time())))

            try:
                self.pipeline.db.ping()  # reconnect if MySQL/Postgres dropped us
                self._refresh_enabled()
                self._consume_deleted()
            except Exception as e:
                log.error("Database unavailable: %s", e)
                time.sleep(POLL_MAX_INTERVAL)
                continue

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

    def _refresh_enabled(self):
        """Re-read the admin page's model toggles (cheap query) so changes apply live."""
        models = self.pipeline.db.get_enabled_models()
        wanted = None if models is None else set(models)
        if wanted is None:
            wanted = {"imagenet", "landmarks", "faces", "movinet", "musicnn"}
        if wanted != self.pipeline._enabled:
            self.pipeline.enable_classifiers(wanted)

    def _consume_deleted(self):
        """Consume ACTION_DELETED rows. Detections go only once the filecache row is
        gone: trashing keeps the fileid and a restore must not lose faces or names."""
        db = self.pipeline.db
        rows = db.fetch_pending(limit=BATCH_SIZE, action_filter=(2,))
        if not rows:
            return
        alive = db.existing_file_ids([r["file_id"] for r in rows])
        for r in rows:
            if r["file_id"] not in alive:
                db.delete_face_detections(r["file_id"])
        db.delete_pending_items(rows)

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
            unresolved = [f for f in file_ids if f not in path_map]
            alive = self.pipeline.db.existing_file_ids(unresolved)
        except Exception as e:
            log.error("Failed to resolve paths: %s", e)
            return None, None, None

        images, videos, audios = [], [], []
        orphans = []

        # shortcut: unresolvable-but-existing rows (S3/SMB, empty, odd mimetype) stay
        # pending and are refetched every loop; more than BATCH_SIZE of them would
        # starve the queue. Upgrade: attempt counter or skip-offset.
        for r in rows:
            info = path_map.get(r["file_id"])
            if not info:
                if r["file_id"] not in alive:
                    orphans.append(r)  # filecache row gone: nothing left to classify
                elif r["file_id"] not in self._warned_unsupported:
                    self._warned_unsupported.add(r["file_id"])
                    log.warning("file_id=%d cannot be read from local storage; leaving it pending",
                                r["file_id"])
                continue

            item = {"id": r["id"], "file_id": r["file_id"], "added_at": r["added_at"],
                    "path": info["path"], "mimetype": info["mimetype"]}
            media_type = self.pipeline.db.classify_mimetype(info["mimetype"])
            if media_type == "image":
                images.append(item)
            elif media_type == "video":
                videos.append(item)
            elif media_type == "audio":
                audios.append(item)

        if orphans:
            log.info("Removing %d orphaned pending entries", len(orphans))
            self.pipeline.db.delete_pending_items(orphans)

        return images, videos, audios

    @staticmethod
    def _good(items, *result_sets):
        """Merge per-pass results for items; None for a file any pass failed or never reached."""
        merged = {}
        for item in items:
            fid = item["file_id"]
            parts = [rs.get(fid) for rs in result_sets]
            if any(p is None or "error" in p for p in parts):
                merged[fid] = None
            else:
                merged[fid] = {k: v for p in parts for k, v in p.items()}
        return merged

    def _submit_items(self, items, results):
        """Submit results[file_id] (None = failed) and drop the pending rows that are finished.

        shortcut: a file that failed classification loses its pending row without being
        tagged, so the daily CatchUpJob retries it; a poison file is retried daily forever.
        Upgrade: attempt counter. A failed submit keeps the row. Returns False when the
        API rejected our credentials (stop the pass).
        """
        done = []
        for item in items:
            res = results.get(item["file_id"])
            if res is None or self.pipeline._submit_result(item["file_id"], res):
                done.append(item)
            elif self.pipeline.api_blocked:
                break
        self.pipeline.db.delete_pending_items(done)
        return not self.pipeline.api_blocked

    def _run_classifier_passes(self):
        """Run each classifier group as a separate pass over pending files.

        A file is submitted (= tagged processed) only if every applicable
        classifier produced a real result for it.

        Returns True if any work was done.
        """
        self.pipeline.api_blocked = False
        images, videos, audios = self._fetch_and_resolve(limit=BATCH_SIZE)
        if images is None:
            time.sleep(POLL_MAX_INTERVAL)
            return False

        if not images and not videos and not audios:
            return False

        enabled = self.pipeline._enabled
        pending_count = self.pipeline.db.pending_count()

        # Images: faces pass, then tags pass (each in its own subprocess), then ONE
        # combined result per image.
        if images:
            do_faces = "faces" in enabled
            do_tags = "imagenet" in enabled or "landmarks" in enabled
            face_res = self.pipeline.process_images_faces(images) if do_faces else {}
            tag_res = self.pipeline.process_images_tags(images) if do_tags else {}
            sets = ([face_res] if do_faces else []) + ([tag_res] if do_tags else [])
            results = self._good(images, *sets) if sets else {}
            for fid, r in results.items():
                if r is not None:
                    r.setdefault("tags", [])
                    r.setdefault("faces", [])
            if not self._submit_items(images, results):
                return self._back_off()
            log.info("Processed %d images (%d pending)", len(images), pending_count)

        # Video / audio: a type whose classifier is disabled yields {} -> not submitted.
        for items, run, label in ((videos, self.pipeline.process_videos, "videos"),
                                  (audios, self.pipeline.process_audio, "audio files")):
            if not items:
                continue
            if not self._submit_items(items, self._good(items, run(items))):
                return self._back_off()
            log.info("Processed %d %s", len(items), label)

        return True

    def _back_off(self):
        time.sleep(POLL_MAX_INTERVAL)
        return False

    def _run_clustering(self):
        """Run face clustering for all users with recent changes."""
        try:
            from clustering import cluster_all_users

            # Get unique users from recent face detections
            user_ids = self.pipeline.db.get_users_with_faces()
            if not user_ids:
                return

            log.info("Running face clustering for %d users", len(user_ids))
            results = cluster_all_users(
                self.pipeline.db, user_ids=user_ids, state=self._cluster_state,
            )

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

    if service.pipeline.nc_api is None:
        log.error("nextcloud_url / internal_secret missing in %s: results cannot be "
                  "submitted, refusing to start", args.config)
        service.stop()
        sys.exit(2)

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
