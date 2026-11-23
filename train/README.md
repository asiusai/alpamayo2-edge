# Fine-tuning the Alpamayo 2 Edge action expert

This document covers `train/`, `finetune/` and `analysis/`.
The students are LoRA adapters on the action expert. They make the two-camera car setup drive closer to the lane centre and turn fully.

Status: implemented and used. `lat10` ran on the car (Karel) on 2026-11-23. `lat2r` is the current candidate: it has the best offline result and is staged on the car. It is not road-tested.

## What a student is

| Item | Value |
| --- | --- |
| Trainable part | LoRA rank 16 (alpha 32) on `expert.expert.layers.N.{self_attn.q,k,v,o_proj, mlp.fc1,fc2}` (168 linears). |
| Frozen part | Vision, language model, reasoning and action projections. Reasoning tokens come from the frozen car recipe. |
| Data | NVIDIA PhysicalAI-AV clips, front wide and front tele cameras only (the car layout), navigation text "Continue straight.". |
| Target | The human path of the clip for clean samples (`ALPAMAYO_TARGET=gt`). Recovery samples keep their teacher target. |
| Loss 1 | Flow matching, 8 (t, noise) draws per sample. |
| Loss 2 | Car loss: MSE on the car's own N-step Euler result from zero noise (`--car-step-weight 1 --car-steps N --car-grad-steps 2`). |
| Loss 3 (optional) | Cross-track error in metres of that result (`--lat-weight 2`). |
| Optimizer | AdamW, lr 1.5e-4, 8 samples per GPU per step, 1 pass over the manifest, both RX 7900 XTX (about 55-75 minutes). |

Train and run with the same step count. A student trained with a 1-step car loss leans more at 10 steps.

| Name | Euler steps | Lateral loss | Manifest | Use |
| --- | --- | --- | --- | --- |
| `dist-gt10` | 10 | no | `nv15` | Pinned in the runtime. |
| `lat10` | 10 | yes | `nv15` | Pinned in the runtime. Road-tested on 2026-11-23. |
| `dist-gt2` | 2 | no | `nv15` | Offline only. |
| `lat2` | 2 | yes | `nv15` | Offline only. |
| `lat2r` | 2 | yes | `nv15rec` | **Current candidate.** Pinned in the runtime (`alpamayo` `d1edf7aa8`) and staged on the car. Not road-tested. |

`nv15` is the 6,350 PhysicalAI training samples drawn 1.5 times (9,525 lines; 6,011 clean and 3,514 recovery lines).
`nv15rec` adds 3,500 synthetic recovery samples from Karel drives (`finetune/samples.py --recovery`).

## Results

Zero initial noise. Each model runs with its own step count. Lateral values are in metres.

Drive-test score (`analysis/lanes_score.py`, 337 marked straight moments of Karel's 2026-11-22 routes, right positive):
plan offset from the lane centre at x = keep x car offset + bias. The car settles at bias / (1 - keep).

| Model | Keep 20/40/60 m | Bias 20/40/60 m | Settles 20/40/60 m | Scatter at 40 m (RMS) |
| --- | --- | --- | --- | --- |
| comma small model (reference) | 0.81/0.53/0.36 | -0.01/-0.01/-0.03 | -0.03/-0.03/-0.04 | 0.21 |
| Stock, 10 steps | 0.91/0.69/0.48 | -0.04/-0.13/-0.23 | -0.48/-0.44/-0.44 | 0.41 |
| `dist-gt10` | 0.89/0.60/0.44 | -0.03/-0.08/-0.10 | -0.26/-0.19/-0.18 | 0.50 |
| `lat10` | 0.90/0.68/0.49 | -0.02/-0.07/-0.10 | -0.26/-0.21/-0.20 | 0.36 |
| `dist-gt2` | 0.89/0.62/0.45 | -0.02/-0.06/-0.07 | -0.17/-0.15/-0.13 | 0.46 |
| `lat2` | 0.89/0.66/0.44 | -0.05/-0.12/-0.21 | -0.43/-0.36/-0.38 | 0.38 |
| `lat2r` (current candidate) | 0.90/0.69/0.48 | -0.01/0.00/+0.01 | -0.06/-0.02/+0.03 | 0.31 |

Heldout sets (`finetune/nv_distill_score.py` on PhysicalAI clips, `finetune/score.py` on Karel turns):

| Model | PhysicalAI turn ratio at 3 s | Lateral error at 3 s | Recovery: error kept at 20/40/60 m | Karel turn ratio at 3 s | Karel turn path error (median) |
| --- | --- | --- | --- | --- | --- |
| Six-camera teacher (stock) | 0.89 | 0.11 | 0.78/0.42/0.27 | - | - |
| Stock, two cameras, 10 steps | 0.83 | 0.18 | 0.84/0.54/0.38 | 0.75 | - |
| `dist-gt10` | 0.90 | 0.16 | 0.88/0.62/0.45 | 0.88 | 0.73 |
| `lat10` | 0.86 | 0.13 | 0.83/0.52/0.34 | 0.82 | 0.58 |
| `dist-gt2` | 0.93 | 0.18 | 0.90/0.73/0.60 | 0.94 | 0.78 |
| `lat2` | 0.89 | 0.14 | 0.83/0.53/0.36 | 0.89 | 0.68 |
| `lat2r` (current candidate) | 0.88 | 0.15 | 0.79/0.47/0.26 | 0.85 | 0.66 |

Road test of `lat10` (2026-11-23, about 85 km/h, 3.4 m lanes, no lane centring): the car settled 0.31 m left of the lane centre.
The drive-test score had predicted 0.20-0.26 m. Stock settled 0.41 m left. Planner lane centring removed the offset (0.01 m right).

Findings:
- No recipe makes the plan correct an offset more firmly than stock. Stock corrects about as gently as the human driver.
- The lateral loss lowers the scatter around the lane centre. The Karel recovery samples remove the left bias at 2 steps.
- Two cameras lean left more than six cameras on the same clips. Zero initial noise adds a left lean at speed.
- Fresh random noise per plan is not a fix: the turn ratio falls to about 0.5.

## Expected layout

The scripts keep the paths of the PC that built the students. Change them at the top of each script or with the listed variables.

| Path | Content | Change with |
| --- | --- | --- |
| `train/.venv` | PyTorch-ROCm 6.4 environment (`uv sync --frozen` with `pyproject.toml` and `uv.lock`). | - |
| `pc/vendor/nvidia` | NVIDIA alpamayo2-edge `02f55a7` checkout (clean). The trainer verifies the revision. | - |
| `pc/data/original-checkpoint` | Original BF16 checkpoint from NVIDIA. | `--checkpoint` |
| `/mnt/nvme-500/alpamayo-train` | Data root: `nvdistill/samples`, `samples`, `turns`, `teacher`, `lanes`, `frozen-cs`, `manifests`, `runs`, `predictions`, `exports`. | `ALPAMAYO_TRAIN_DATA` (trainer); `A=` in the shell scripts |
| Car footage root | `<route>--<segment>/{rlog.zst,fcamera.mp4,ecamera.mp4}` | `ALPAMAYO_FOOTAGE` or `finetune/paths.local.json` |
| openpilot fork checkout | `selfdrive/alpamayo` for `payload.py`, `samples.py`, `render.py` and `analysis/` (`PYTHONPATH`). | `PYTHONPATH` |
| `/home/john/alpamayo-chestnut-pc/live-karel.json` | Car camera geometry for `samples.py`. | Constant in `samples.py` |
| `/home/john/alpamayo-work/analysis/lane-table-c8cacb.npz` | `analysis/lane_table.py` output for the drive-test routes. | `LANE_TABLE` |

Weights, datasets, predictions and logs are not in this repository. NVIDIA's code and weights have their own licences.

Rule for these GPUs: one process per GPU. A second process on the same GPU caused a memory fault and a GPU reset.

## Commands

```sh
cd train && source env.sh          # PY, PYTHONPATH (trainer + NVIDIA source); A below is the data root
A=/mnt/nvme-500/alpamayo-train; F=../finetune
```

### 1. Build the data

```sh
# PhysicalAI-AV clips: download, decode (CPU), six-camera teacher plans (one process per GPU)
(cd ../pc/vendor/nvidia && .venv/bin/python $F/nv_download.py $A/physicalai 0 1 2 3 4 5 6 7 8)
(cd ../pc/vendor/nvidia && .venv/bin/python $F/nv_distill_decode.py $A/physicalai SPOOL 0 1 2 3 4 5 6 7 8 --workers 8)
$PY $F/nv_distill_teacher.py 0 SPOOL $A/nvdistill/samples --shard 0/2 &
$PY $F/nv_distill_teacher.py 1 SPOOL $A/nvdistill/samples --shard 1/2 & wait

# Reasoning tokens of the frozen car recipe (PC runtime, pc/env.sh environment, one process per GPU)
(source ../pc/env.sh && cd $F && ALPAMAYO_SHARD=0/2 $PY frozen.py 0 --nav "Continue straight." $A/nvdistill/samples/heldout $A/nvdistill/samples/train)

# Karel samples (openpilot environment): one segment -> payload -> clean and recovery samples
python3 $F/payload.py PAYLOAD SEGMENT_DIR 40 && python3 $F/samples.py PAYLOAD $A/samples/train --recovery 2
python3 $F/build_lanes.py $A/lanes          # drive-test set; build_turns.py, build_teacher.py: Karel heldout sets
```

`finetune/nv_distill_queue.sh` is the queue that ran the PhysicalAI steps. A manifest is a text file with one sample `.npz` path per line.

### 2. Train

```sh
ALPAMAYO_TARGET=gt $PY train.py --out $A/runs/lat10 --gpus 0,1 --samples-per-step 8 --n-noise 8 --rank 16 --epochs 1 --lr 1.5e-4 \
  --car-step-weight 1 --car-steps 10 --car-grad-steps 2 --lat-weight 2 --val-steps 10 \
  --val $A/nvdistill/samples/heldout --val-limit 64 --val-every 250 --save-every 250 --log-every 20 --manifest $A/manifests/nv15.txt
```

- `dist-gt10`: the same command without `--lat-weight 2`.
- 2-step students: `--car-steps 2 --val-steps 2`. `lat2r` uses `--manifest $A/manifests/nv15rec.txt`.
- `finetune/nv_train_queue6.sh NAME MANIFEST EPOCHS [TRAIN ARGS]` trains one student and runs every score below. `STEPS` sets the step count.
- `finetune/chain_lat2.sh` shows the two calls that made `lat2` and `lat2r`.

### 3. Sample and score

```sh
W=$A/runs/lat10/final; P=$A/predictions/dist
$PY sample.py $A/lanes/samples --out $P/lat10-lanes.npz --gpus 0,1 --steps 10 --lora $W
python3 ../analysis/lanes_score.py LANE_TABLE.npz $A/lanes/samples stock=$P/stock-lanes.npz lat10=$P/lat10-lanes.npz

$PY sample.py $A/nvdistill/samples/heldout --out $P/lat10-nv.npz --gpus 0,1 --steps 10 --lora $W
(cd $F && python3 nv_distill_score.py $A/nvdistill/samples/heldout stock-s10=$P/stock-nv.npz lat10=$P/lat10-nv.npz)

$PY sample.py $A/turns/samples-120/heldout --kinds clean --out $P/lat10-turns.npz --gpus 0 --steps 10 --lora $W
$PY sample.py $A/teacher/samples/heldout --out $P/lat10-rec.npz --gpus 1 --steps 10 --lora $W
(cd $F && python3 score.py lat10 $P/lat10-turns.npz $P/lat10-rec.npz --json $P/lat10.score.json)

$PY sample.py $A/samples/heldout --kinds recovery,clean --out $P/lat10-shift.npz --gpus 0,1 --steps 10 --lora $W
(cd $F && python3 evaluate.py $A/samples/heldout $P/lat10-shift.npz)
```

`sample.py` uses the car's sampler: zero initial noise, Euler steps, frozen reasoning tokens. Without `--lora` it scores the stock model.

### 4. Export for the car and the PC runtime

```sh
$F/export_local.sh lat10      # export.py (merged BF16 checkpoint) + prepare-fp8.py (car FP8 cache) + check: only expert layers differ
```

The script prints `fp8_index_sha256` and `fp8_weights_sha256`. The runtime accepts a fine-tune only when
`selected/assets/artifacts.json` lists these digests under `fp8_finetunes`. `finetune/export_and_pin.sh` adds the pin in an openpilot checkout.
Run a pinned fine-tune on the PC with `alpamayo_pc.load(gpu, diffusion_steps=2, zero_noise=True, fp8_cache=$A/exports/lat2r-fp8, cache_dir=...)`.
The PC snapshot accepts 1 to 3 Euler steps, so it cannot run the 10-step students at their step count (see `pc/README.md`).

## Files

| Folder | Files |
| --- | --- |
| `train/` | `train.py`, `sample.py`, `export.py`, package `alpamayo_train` (data, model, LoRA, one process per GPU). `check_parity.py`, `roundtrip_report.py`, `make_synthetic.py` are checks. |
| `finetune/` data | `nv_download.py`, `nv_distill_decode.py`, `nv_distill_teacher.py`, `payload.py`, `samples.py`, `render.py`, `frozen.py`, `build_*.py`, `comma_payload.py`, `classify.py`, `manifest.py`, `label_ambiguous.py`. |
| `finetune/` scores | `nv_distill_score.py`, `score.py`, `evaluate.py`, `evaluate_turns.py`, `evaluate_recovery.py`, `eval_runtime.py` (car recipe on the PC runtime). |
| `finetune/` tests | Camera sets (`nv_shift*.py`, `physicalai_*.py`, `demo_*.py`), initial noise (`noise_*.py`), ego history (`history_*.py`), navigation text (`nav_test.sh`, `nav_score.py`). |
| `finetune/` export | `export_local.sh`, `export_and_pin.sh`, `prepare-fp8.py`. |
| `analysis/` | Drive analysis on openpilot logs: `lane_table.py` and `lanes_score.py` (drive-test score), `straights*.py`, `centering_*.py` (lane centring replay and simulation), `history_bias.py`, `calib_by_route.py`, `lateral_modes.py`, `end_scan.py`, `events.py`, `lanebias.py`. |

The one-off queue scripts, logs, backups, datasets and weights of the experiments are not included.
