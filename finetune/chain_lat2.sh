#!/bin/bash
# After dist-gt2: the lat10 recipe (cross-track loss) for 2 Euler steps, then the same plus Karel synthetic recovery samples.
A=/mnt/nvme-500/alpamayo-train; L=$A/dist-queue.log
F=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$F")   # finetune/ and the alpamayo2-edge checkout
until grep -q "dist-gt2 shift scored\|dist-gt2 FAILED" $L; do sleep 20; done
sleep 10
STEPS=2 $F/nv_train_queue6.sh lat2 $A/manifests/nv15.txt 1 --lat-weight 2
STEPS=2 $F/nv_train_queue6.sh lat2r $A/manifests/nv15rec.txt 1 --lat-weight 2
