#!/bin/bash
# Six-to-two camera distillation set: decode (CPU) and teacher (one process per GPU) over PhysicalAI chunks 0-8.
W=/home/john/alpamayo-work/nvdistill; S=$W/spool; O=/mnt/nvme-500/alpamayo-train/nvdistill/samples; L=$W/queue.log
D2=/home/john/alpamayo-work/physicalai2
F=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$F")   # finetune/ and the alpamayo2-edge checkout
N=$REPO/pc/vendor/nvidia
log() { echo "$(date +%T) $*" >> $L; }
(
  cd $N
  .venv/bin/python $F/nv_distill_decode.py /mnt/nvme-500/alpamayo-train/physicalai $S 0 --workers 8 > $W/decode-0.log 2>&1
  log "decoded chunk 0"
  for c in 1 2 3 4 5 6 7 8; do
    until grep -aq "^done $c vehicle_dimensions" $D2/download.log; do sleep 30; done
    final=(); [ $c = 8 ] && final=(--final)
    .venv/bin/python $F/nv_distill_decode.py $D2 $S $c --workers 8 "${final[@]}" > $W/decode-$c.log 2>&1
    log "decoded chunk $c"
  done
) &
until grep -q "queue done" /home/john/alpamayo-work/nvshift/queue.log 2>/dev/null; do sleep 20; done
log "teacher start"
cd $REPO/train && source env.sh && cd $F
$PY nv_distill_teacher.py 0 $S $O --shard 0/2 > $W/teacher-0.log 2>&1 &
$PY nv_distill_teacher.py 1 $S $O --shard 1/2 > $W/teacher-1.log 2>&1 &
wait
log "distill set done: $(ls $O/train/*.npz | wc -l) train, $(ls $O/heldout/*.npz | wc -l) heldout"
