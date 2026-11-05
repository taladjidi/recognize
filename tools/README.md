# tools

## Benchmarking the Faces UI

Headless, read-only (PROPFIND and GET only). Replays the requests the deployed
Photos fork makes; see the comments in `bench_faces.py` for the source lines.

One-time, create an app password (sudo; the file stays 0600 in your home):

    mkdir -p ~/.config/nc-bench && (umask 077; sudo -u apache php /usr/share/nextcloud/occ user:auth-tokens:add --name=headless-bench tangui | tail -1 > ~/.config/nc-bench/tangui)

Also needed: the recognize DAV plugin refuses Basic-auth requests without an
`X-Recognize-Api-Key` (the browser gets one from the page state). It is valid
24 h, so regenerate when the bench starts answering 403:

    (umask 077; sudo -u apache php tools/gen_bench_apikey.php > ~/.config/nc-bench/recognize-api-key)

Client side (results go to `bench-results/<UTC>-<scenario>.json`, gitignored):

    tools/bench_faces.py --scenario all --runs 3
    tools/bench_faces.py --scenario overview --max-covers 200          # quick
    tools/bench_faces.py --scenario overview --cover-size 256          # sensitivity check
    tools/bench_faces.py --scenario overview --compare bench-results/<old>.json

`--api python` replays the JSON path the Photos fork uses now (faces-token, then
`/recognize-api/v1/clusters` with Bearer auth, covers, paged cluster fetch); the
grid thumbnails are still Nextcloud previews. Results go to `*-python.json`; the
metric names match the DAV ones, so `--compare` works across modes. Adds `token_ms`
and `covers_null`. Same Basic-auth app password, no API key needed:

    tools/bench_faces.py --api python --scenario all --runs 2
    tools/bench_faces.py --api python --scenario overview --compare bench-results/<dav>.json

Run 1 is reported apart (cold-ish); the median is over runs 2..N. HTTP/1.1 with
10 workers, so compare tool runs with each other, not with browser timings.

Server side (which part is the database), run as the web user:

    sudo -u apache php tools/probe_faces_timing.php tangui [clusterId]
