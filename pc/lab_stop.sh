#!/bin/bash
# Stop the PC lab on one GPU: ./lab_stop.sh GPU
cd "$(dirname "$0")"; f=lab/gpu$1/pid
[ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null; rm -f "$f" lab/gpu$1/ready
