#!/bin/bash
# Sixth student queue: as queue5, for a student of STEPS Euler steps (env STEPS, default 2): car loss and all evaluations at STEPS.
# Usage: [STEPS=2] nv_train_queue6.sh NAME MANIFEST EPOCHS [TRAIN ARGS...]. One process per GPU. Log: /mnt/nvme-500/alpamayo-train/dist-queue.log
A=/mnt/nvme-500/alpamayo-train; N=$A/nvdistill/samples; T=$A/turns/samples-120/heldout; R=$A/teacher/samples/heldout; S=$A/lanes/samples
P=$A/predictions/dist; L=$A/dist-queue.log
F=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$F")   # finetune/ and the alpamayo2-edge checkout
LANES=${LANE_TABLE:-/home/john/alpamayo-work/analysis/lane-table-c8cacb.npz}   # analysis/lane_table.py output for the drive-test routes
STEPS=${STEPS:-2}; name=$1; manifest=$2; epochs=$3; shift 3
log() { echo "$(date +%T) $*" >> $L; }
cd $REPO/train && source env.sh
log "queue6 start (steps $STEPS) $name $*"
rm -rf "${A:?}/runs/${name:?}"
ALPAMAYO_TARGET=gt $PY train.py --out $A/runs/$name --gpus 0,1 --samples-per-step 8 --n-noise 8 --rank 16 --epochs $epochs --lr 1.5e-4 \
  --car-step-weight 1 --car-steps $STEPS --car-grad-steps 2 --val-steps $STEPS --val $N/heldout --val-limit 64 --val-every 250 --save-every 250 --log-every 20 \
  --manifest $manifest "$@" > $A/runs-$name.log 2>&1
if [ ! -f $A/runs/$name/final/adapter.json ]; then log "$name FAILED"; exit 1; fi
log "$name trained"
W=$A/runs/$name/final
# The drive-test set first: it is the arbiter.
$PY sample.py $S --out $P/$name-lanes.npz --gpus 0,1 --steps $STEPS --lora $W > $P/$name-lanes.log 2>&1
python3 $REPO/analysis/lanes_score.py $LANES $S stock=$P/stock-lanes.npz dist-gt10=$P/dist-gt10-lanes.npz $name=$P/$name-lanes.npz >> $L 2>&1
log "$name lanes scored"
$PY sample.py $N/heldout --out $P/$name-nv.npz --gpus 0,1 --steps $STEPS --lora $W > $P/$name-nv.log 2>&1
(cd $F && python3 nv_distill_score.py $N/heldout stock-s10=$P/stock-nv.npz dist-gt10=$P/dist-gt10-nv.npz $name=$P/$name-nv.npz >> $L 2>&1)
$PY sample.py $T --kinds clean --out $P/$name-turns.npz --gpus 0 --steps $STEPS --lora $W > $P/$name-turns.log 2>&1 &
$PY sample.py $R --out $P/$name-rec.npz --gpus 1 --steps $STEPS --lora $W > $P/$name-rec.log 2>&1 &
wait
(cd $F && python3 score.py $name $P/$name-turns.npz $P/$name-rec.npz --json $P/$name.score.json >> $L 2>&1)
log "$name scored"
H=$A/samples/heldout
$PY sample.py $H --kinds recovery,clean --out $P/$name-shift.npz --gpus 0,1 --steps $STEPS --lora $W > $P/$name-shift.log 2>&1
(cd $F && python3 evaluate.py $H $P/$name-shift.npz >> $L 2>&1)
log "$name shift scored"
