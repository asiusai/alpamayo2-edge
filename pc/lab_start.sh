#!/bin/bash
# Start (or restart) the PC lab on one GPU in the background: ./lab_start.sh GPU [lab_server.py options]
set -euo pipefail
cd "$(dirname "$0")"; source ./env.sh
gpu=$1; shift
lab=lab/gpu$gpu; mkdir -p "$lab"
if [ -f "$lab/pid" ] && kill -0 "$(cat "$lab/pid")" 2>/dev/null; then kill "$(cat "$lab/pid")"; while kill -0 "$(cat "$lab/pid")" 2>/dev/null; do sleep 0.5; done; fi
rm -f "$lab/ready"
nohup "$PY" lab_server.py --gpu "$gpu" "$@" > "$lab/server.log" 2>&1 &
echo $! > "$lab/pid"
until [ -f "$lab/ready" ]; do kill -0 "$(cat "$lab/pid")" 2>/dev/null || { tail -20 "$lab/server.log"; exit 1; }; sleep 1; done
tail -1 "$lab/server.log"
