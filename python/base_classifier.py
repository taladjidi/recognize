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
