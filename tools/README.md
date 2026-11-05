# tools

## Benchmarking the Faces UI

Headless, read-only (PROPFIND and GET only). Replays the requests the deployed
Photos fork makes; see the comments in `bench_faces.py` for the source lines.

One-time, create an app password (sudo; the file stays 0600 in your home):

    mkdir -p ~/.config/nc-bench && (umask 077; sudo -u apache php /usr/share/nextcloud/occ user:auth-tokens:add --name=headless-bench tangui | tail -1 > ~/.config/nc-bench/tangui)

Client side (results go to `bench-results/<UTC>-<scenario>.json`, gitignored):

    tools/bench_faces.py --scenario all --runs 3
    tools/bench_faces.py --scenario overview --max-covers 200          # quick
    tools/bench_faces.py --scenario overview --cover-size 256          # sensitivity check
    tools/bench_faces.py --scenario overview --compare bench-results/<old>.json

Run 1 is reported apart (cold-ish); the median is over runs 2..N. HTTP/1.1 with
10 workers, so compare tool runs with each other, not with browser timings.

Server side (which part is the database), run as the web user:

    sudo -u apache php tools/probe_faces_timing.php tangui [clusterId]
