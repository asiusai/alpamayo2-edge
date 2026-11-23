#!/bin/bash
# Navigation-text test on NVIDIA heldout clips: the VLM generates its reasoning with the given text, then the expert plans.
# Stock and lat10, 10 steps, zero noise. Turn clips get the matching instruction; straight clips get a turn instruction far ahead.
A=/mnt/nvme-500/alpamayo-train; H=$A/nvdistill/samples/heldout; O=$A/predictions/nav
F=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$F")   # finetune/ and the alpamayo2-edge checkout
cd $REPO/train && source env.sh
run() {  # NAME STEMS NAV [LORA]
  ALPAMAYO_NAV="$3" $PY sample.py $H --out $O/$1.npz --gpus 0,1 --steps 10 --generate --stems $O/$2.txt ${4:+--lora $4} > $O/$1.log 2>&1
}
for model in stock lat10; do
  lora=; [ $model = lat10 ] && lora=$A/runs/lat10/final
  run $model-left-nav left "Turn left in 30m" $lora
  run $model-right-nav right "Turn right in 30m" $lora
  run $model-left-cs left "Continue straight." $lora
  run $model-right-cs right "Continue straight." $lora
  run $model-straight-far straight "Turn right in 800m" $lora
  run $model-straight-cs straight "Continue straight." $lora
done
echo done > $O/done
