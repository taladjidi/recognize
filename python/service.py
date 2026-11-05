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

# Face crops: idle-time backfill + orphan cleanup
CROP_BATCH = 200             # detections examined per idle tick
CROP_TICK_SECONDS = 2.0      # wall-clock budget per idle tick, so new work waits at most this
CROP_BUSY_BATCH = 25         # detections examined after each classifier pass, so backfill moves under load
CROP_RESCAN_INTERVAL = 3600  # after a full backfill pass and cleanup round, rest this long
CROP_LOG_EVERY = 500         # INFO line every N crops created by the backfill


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

        from crops import DEFAULT_CROP_ROOT
        self.crop_root = self.config.get("face_crop_root") or DEFAULT_CROP_ROOT
        self._crop_cursor = 0          # backfill: last detection id examined
        self._crop_created = 0         # backfill: crops created this pass
        self._crop_users = []          # cleanup: user dirs left in this round
        self._crop_covers_done = False  # backfill phase 1: each cluster's largest box first
        self._crop_done = {"backfill": False, "cleanup": False}
        self._crop_rest_until = 0.0
        self._crop_root_warned = False

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
        self.pipeline.stopping.set()

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
                # Maintenance first: nothing may be deleted while Nextcloud is down.
                if self.pipeline.db.check_maintenance_mode():
                    log.info("Nextcloud in maintenance mode, waiting...")
                    self._sleep(POLL_MAX_INTERVAL)
                    continue
                self._refresh_enabled()
                self._consume_deleted()
            except Exception as e:
                log.error("Database unavailable: %s", e)
                self._sleep(POLL_MAX_INTERVAL)
                continue

            had_work = self._run_classifier_passes()

            if not had_work:
                # Exponential backoff when idle
                self._poll_interval = min(
                    self._poll_interval * POLL_BACKOFF_FACTOR,
                    POLL_MAX_INTERVAL,
                )
                self._crop_maintenance()
            else:
                self._poll_interval = POLL_MIN_INTERVAL
            if had_work and self._running:
                self._crop_maintenance(busy=True)

            # Periodic face clustering
            now = time.monotonic()
            if now - self._last_cluster_time > CLUSTER_INTERVAL:
                self._run_clustering()
                self._last_cluster_time = now

            # Sleep with interruptibility
            if self._running:
                self._sleep(self._poll_interval)

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
        alive = db.file_storages([r["file_id"] for r in rows])
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
            alive = self.pipeline.db.file_storages(unresolved)
        except Exception as e:
            log.error("Failed to resolve paths: %s", e)
            return None, None, None

        images, videos, audios = [], [], []
        drop, defer = [], []

        def warn_once(key, msg, *args):
            if key not in self._warned_unsupported:
                self._warned_unsupported.add(key)
                log.warning(msg, *args)

        for r in rows:
            info = path_map.get(r["file_id"])
            if not info:
                storage = alive.get(r["file_id"])
                if r["file_id"] not in alive:
                    drop.append(r)  # filecache row gone: nothing left to classify
                elif storage and not storage.startswith(("home::", "local::")):
                    warn_once(storage, "storage %s cannot be read locally; its files are not classified", storage)
                    drop.append(r)  # can never be read here
                else:
                    # shortcut: empty/missing-on-disk files are retried forever at the back
                    # of the queue (so they cannot starve it); upgrade: attempt counter.
                    defer.append(r)
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
            else:
                warn_once(info["mimetype"], "mimetype %s is not classified; dropping its pending rows",
                          info["mimetype"])
                drop.append(r)

        if drop:
            log.info("Removing %d unclassifiable pending entries", len(drop))
            self.pipeline.db.delete_pending_items(drop)
        if defer:
            self.pipeline.db.defer_pending_items(defer)

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
        Upgrade: attempt counter. Returns False when the pass must stop: wrong credentials
        (rows kept as is) or a transient failure (remaining rows deferred to the back).
        A permanent 4xx drops the row, like a failed classification.
        """
        done, stopped = [], None
        for i, item in enumerate(items):
            res = results.get(item["file_id"])
            if res is None:
                done.append(item)
            elif self.pipeline._submit_result(item["file_id"], res):
                done.append(item)
                if res.get("crops"):
                    self._write_crops(item, res["crops"], self.pipeline.detection_ids)
            elif self.pipeline.submit_failure == "permanent":
                done.append(item)
            else:
                stopped = self.pipeline.submit_failure
                if stopped != "blocked":
                    self.pipeline.db.defer_pending_items(items[i:])
                break
        self.pipeline.db.delete_pending_items(done)
        return stopped is None

    def _run_classifier_passes(self):
        """Run each classifier group as a separate pass over pending files.

        A file is submitted (= tagged processed) only if every applicable
        classifier produced a real result for it.

        Returns True if any work was done.
        """
        images, videos, audios = self._fetch_and_resolve(limit=BATCH_SIZE)
        if images is None:
            self._sleep(POLL_MAX_INTERVAL)
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
            if self.pipeline.stopping.is_set():
                return False  # SIGTERM mid-batch: leave the pending rows as they are
            sets = ([face_res] if do_faces else []) + ([tag_res] if do_tags else [])
            results = self._good(images, *sets) if sets else {}
            for fid, r in results.items():
                if r is not None:
                    r.setdefault("tags", [])
                    r.setdefault("faces", [])
            if not self._submit_items(images, results):
                return self._back_off()
            log.info("Processed %d images (%d pending)", len(images), pending_count)

        # Video / audio: a disabled classifier yields no result -> None -> the row is
        # dropped and the file stays untagged (catch-up retries once it is enabled).
        for items, run, label in ((videos, self.pipeline.process_videos, "videos"),
                                  (audios, self.pipeline.process_audio, "audio files")):
            if not items:
                continue
            res = run(items)
            if self.pipeline.stopping.is_set():
                return False
            if not self._submit_items(items, self._good(items, res)):
                return self._back_off()
            log.info("Processed %d %s", len(items), label)

        return True

    def _write_crops(self, item, crops, detection_ids):
        """Write the crops of a just-submitted image (PHP returned their detection ids)."""
        from crops import write_crop
        owner = item.get("owner", "")
        if not owner or len(detection_ids) != len(crops):
            return  # no owner, or PHP kept the old detections ([]): the backfill covers it
        for det_id, data in zip(detection_ids, crops):
            if data:
                write_crop(self.crop_root, owner, det_id, data)

    def _crop_maintenance(self, busy=False):
        """Idle tick: a bounded slice of crop backfill, then one user dir of orphan cleanup.

        busy: a small backfill slice after a classifier pass (no cleanup).

        Never raises: crops are derived data and must not stop classification.
        """
        now = time.monotonic()
        if now < self._crop_rest_until or "faces" not in self.pipeline._enabled:
            return
        if not os.path.isdir(self.crop_root):
            if not self._crop_root_warned:
                log.warning("Face crop root %s missing (run deploy.sh); no face crops", self.crop_root)
                self._crop_root_warned = True
            return
        try:
            deadline = now + CROP_TICK_SECONDS
            if not self._crop_done["backfill"]:
                self._crop_done["backfill"] = self._backfill_crops(deadline, CROP_BUSY_BATCH if busy else CROP_BATCH)
            if busy:
                return
            if not self._crop_done["cleanup"] and time.monotonic() < deadline:
                self._crop_done["cleanup"] = self._cleanup_crops_one_user()
            if all(self._crop_done.values()):
                self._crop_done = {"backfill": False, "cleanup": False}
                self._crop_rest_until = time.monotonic() + CROP_RESCAN_INTERVAL
        except Exception as e:
            log.error("Face crop maintenance failed: %s", e)

    def _backfill_crops(self, deadline, batch=CROP_BATCH):
        """Create missing crops for up to `batch` detections; True when a full pass ended.

        A pass first covers only each cluster's largest box, then everything.
        """
        from crops import has_crop, make_crop, open_raw, write_crop
        db = self.pipeline.db
        rows = db.detections_after(self._crop_cursor, batch, covers_only=not self._crop_covers_done)
        if not rows and not self._crop_covers_done:
            self._crop_covers_done, self._crop_cursor = True, 0
            return False
        if not rows:
            self._crop_covers_done = False
            if self._crop_created:
                log.info("Face crop backfill pass done: %d created", self._crop_created)
            self._crop_cursor, self._crop_created = 0, 0
            return True
        by_file = {}
        for r in rows:
            by_file.setdefault(r["file_id"], []).append(r)
        missing = {fid: [r for r in rs if not has_crop(self.crop_root, r["user_id"], r["id"])]
                   for fid, rs in by_file.items()}
        missing = {fid: rs for fid, rs in missing.items() if rs}
        paths = db.resolve_file_paths(list(missing)) if missing else {}
        for fid, rs in by_file.items():
            if time.monotonic() >= deadline:
                return False  # cursor stays before this file: resumed next tick
            info = paths.get(fid)
            if fid in missing and info:
                try:
                    img, orientation = open_raw(info["path"], rs)
                    for r in missing[fid]:
                        if write_crop(self.crop_root, r["user_id"], r["id"], make_crop(img, r, orientation)):
                            self._crop_created += 1
                            if self._crop_created % CROP_LOG_EVERY == 0:
                                log.info("Face crop backfill: %d created (at detection %d)",
                                         self._crop_created, r["id"])
                except Exception as e:
                    log.warning("Face crop backfill failed for file %d: %s", fid, e)
            self._crop_cursor = max(r["id"] for r in rs)
        return False

    def _cleanup_crops_one_user(self):
        """Delete orphan crops in one user dir; True when every user dir was visited this round."""
        from crops import crop_ids, safe_user_dir
        if not self._crop_users:
            with os.scandir(self.crop_root) as it:
                self._crop_users = sorted(e.name for e in it if e.is_dir(follow_symlinks=False))
            if not self._crop_users:
                return True
        user = self._crop_users.pop()
        user_dir = safe_user_dir(self.crop_root, user)
        if user_dir is not None:
            files = crop_ids(user_dir)
            alive = self.pipeline.db.existing_detection_ids(user, sorted(files)) if files else set()
            removed = 0
            for det_id, name in files.items():
                if det_id not in alive:
                    try:
                        os.unlink(os.path.join(user_dir, name))
                        removed += 1
                    except FileNotFoundError:
                        pass
            if removed:
                log.info("Face crops: removed %d orphans for %s", removed, user)
        return not self._crop_users

    def _sleep(self, seconds):
        """Wait, but wake at once on SIGTERM: POLL_MAX_INTERVAL equals systemd's stop timeout."""
        self.pipeline.stopping.wait(seconds)

    def _back_off(self):
        self._sleep(POLL_MAX_INTERVAL)
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
