"""Shared STDIN/STDOUT IPC protocol for all classifier scripts.

Protocol (matches the Node.js classifiers):
- When argv[1] is '-', read file paths from stdin (one per line)
- Otherwise, file paths are taken from argv[1:]
- For each file, output one JSON line to stdout
- On error for a file, output '[]' to stdout
- Diagnostics go to stderr
"""

import json
import os
import sys

# Set a stable writable matplotlib config dir before any transitive import
# (TensorFlow and InsightFace pull in matplotlib). Avoids permission errors
# when running as the apache user whose home dir is not writable.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-recognize")

# Register HEIC/HEIF support so PIL can open iPhone photos directly
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


def get_paths():
    """Read file paths from stdin (if arg is '-') or from argv."""
    if len(sys.argv) < 2:
        print(
            "Usage: python classifier_<model>.py <file1> [file2 ...] | python classifier_<model>.py -",
            file=sys.stderr,
        )
        sys.exit(1)

    if sys.argv[1] == "-":
        paths = sys.stdin.read().split("\n")
    else:
        paths = sys.argv[1:]

    return [p for p in paths if p.strip()]


def iter_paths():
    """Yield file paths one at a time from stdin or argv.

    Unlike get_paths(), this does not block until all stdin is consumed.
    Results can be emitted as each file completes, and the process can be
    killed between files without losing work.
    """
    if len(sys.argv) < 2:
        print(
            "Usage: python classifier_<model>.py <file1> [file2 ...] | python classifier_<model>.py -",
            file=sys.stderr,
        )
        sys.exit(1)

    if sys.argv[1] == "-":
        for line in sys.stdin:
            path = line.rstrip("\n")
            if path:
                yield path
    else:
        for path in sys.argv[1:]:
            if path.strip():
                yield path


def iter_batches(batch_size):
    """Yield lists of up to batch_size paths from iter_paths().

    For classifiers that benefit from batched GPU inference (imagenet, landmarks).
    """
    batch = []
    for path in iter_paths():
        batch.append(path)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def prefetch_map(iterable, fn, prefetch=4):
    """Apply fn to items in background threads, staying prefetch items ahead.

    Yields (item, result) tuples. While the main thread does GPU inference,
    background threads preprocess upcoming items (image loading, FFmpeg, etc.),
    keeping the GPU fed. If fn(item) raises, result is None.
    """
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    def safe_fn(item):
        try:
            return fn(item)
        except Exception as e:
            print(f"Prefetch error for {item}: {e}", file=sys.stderr)
            return None

    with ThreadPoolExecutor(max_workers=prefetch) as pool:
        buf = deque()
        it = iter(iterable)

        # Fill initial buffer
        for _ in range(prefetch):
            try:
                item = next(it)
            except StopIteration:
                break
            buf.append((item, pool.submit(safe_fn, item)))

        # Stream: yield oldest, submit next
        for item in it:
            old_item, future = buf.popleft()
            yield old_item, future.result()
            buf.append((item, pool.submit(safe_fn, item)))

        # Drain remaining
        while buf:
            old_item, future = buf.popleft()
            yield old_item, future.result()


def output_result(result):
    """Write one JSON line to stdout and flush."""
    print(json.dumps(result, ensure_ascii=False), flush=True)


def output_error():
    """Write empty result for a failed file."""
    output_result([])


def get_ffmpeg_binary():
    """Get ffmpeg binary path from FFMPEG_BINARY env var or system PATH."""
    import shutil

    ffmpeg = os.environ.get("FFMPEG_BINARY", "")
    if ffmpeg and os.path.isfile(ffmpeg):
        return ffmpeg
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    raise RuntimeError("ffmpeg not found. Set FFMPEG_BINARY env var.")
