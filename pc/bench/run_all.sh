#!/bin/bash
# Throughput and latency suite; results in bench/*.json. SLOTS: a build_slots.py stream (realistic consecutive 100 ms slots).
cd "$(dirname "$0")/.."; source env.sh
SLOTS=data/slots/00000091--c1af5805a9--0.npz
rm -f bench/*.json
$PY tools/bench.py --gpu 0 --mode sweep --calls 48 --out bench/single-gpu0-sweep.json > bench/single-gpu0-sweep.log 2>&1
$PY tools/bench.py --gpu 0 --mode cached --calls 48 --out bench/single-gpu0-cached.json > bench/single-gpu0-cached.log 2>&1
for mode in rolling live; do $PY tools/bench.py --gpu 0 --mode $mode --slots $SLOTS --warm 40 --calls 120 --out bench/single-gpu0-$mode.json > bench/single-gpu0-$mode.log 2>&1; done
for mode in sweep live; do
  t=$(( $(date +%s) + 120 ))
  extra=""; [ $mode = live ] && extra="--slots $SLOTS --warm 40 --calls 120"
  for g in 0 1; do $PY tools/bench.py --gpu $g --mode $mode --calls 96 $extra --start-at $t --out bench/dual-gpu$g-$mode.json > bench/dual-gpu$g-$mode.log 2>&1 & done
  wait
done
for f in bench/*.json; do echo "$f $(cat $f)"; done
