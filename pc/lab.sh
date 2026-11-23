#!/bin/bash
# Run a snippet in a running PC lab and print its output: ./lab.sh GPU snippet.py [timeout_s]
set -euo pipefail
gpu=$1; snippet=$2; limit=${3:-3600}
lab=${LAB:-$(cd "$(dirname "$0")" && pwd)/lab/gpu$gpu}
[ -f "$lab/ready" ] || { echo "lab on GPU $gpu is not ready: $lab" >&2; exit 1; }
name=$(date +%s%N)-$(basename "$snippet" .py)
cp "$snippet" "$lab/in/$name.tmp" && mv "$lab/in/$name.tmp" "$lab/in/$name.py"
for ((i=0; i<limit*20; i++)); do
  if [ -f "$lab/out/$name.txt" ]; then cat "$lab/out/$name.txt"; tail -1 "$lab/out/$name.txt" | grep -q '^== done' && exit 0 || exit 1; fi
  sleep 0.05
done
echo "timeout waiting for $name" >&2; exit 2
