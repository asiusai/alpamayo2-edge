# Fine-tuning data tools and experiment history

The current recipe, commands and results are in [`train/README.md`](../train/README.md).
The text below is the experiment history up to 2026-11-22 (recovery data, turn data, the `t180-r16` and `t120-r16` adapters).
Its paths name the original work folders; the files are now in this folder and in `train/`.


Goal: teach the trajectory part (action expert) of Alpamayo 2 Edge to return to the lane after a lateral or heading error.
The reasoning stays frozen: training conditions on the reasoning tokens that the frozen car recipe generated for each sample.

## Pipeline

1. `payload.py OUT SEGMENT [MAX]`: one recorded segment to logged states plus only the camera frames each query moment needs.
   Moments are human-steered (lateral control off from -2.1 s to +6.5 s), at least 4 m/s, on a 1 s grid. Frames follow the car's selection rule.
2. `samples.py PAYLOAD OUT --recovery K`: per moment one clean sample and K recovery samples.
   - Inputs equal the live pipeline (`render.py` is bit-identical to `selected.camera.nv12_image` at zero shift; history from `history.ego_history`).
   - Recovery: the views come from a pose shifted `dy` (0.3-1.0 m) and/or turned `dpsi` (0.6-2.6 deg); past views and history drift consistently into that error.
   - Target: the human's driven path (vehicle model), with the lateral error decaying to zero over 3 s (cubic Hermite), in the shifted query frame.
   - Known artifact: the road-plane shift moves objects above the road only approximately. The car's hood stays fixed.
3. `frozen.py GPU DIR...`: the frozen car recipe (PC tinygrad runtime) writes `<sample>.frozen.npz` with `reasoning_token_ids`, `baseline_xyz`, `baseline_rot`.
4. `evaluate.py DIR [PREDICTIONS.npz]`: fraction of the starting error kept at 20/40/60 m (recovery), lateral error and turn ratio (clean).
5. `build_all.py N`: steps 1-2 for the top N human-driven segments (max 2 per route); about 1 route in 7 goes to `heldout`.

Data: `/mnt/nvme-500/alpamayo-train/samples/{test,train,heldout}` (not under `/mnt/nvme-500/tmp`, which is cleaned).
Environment: `source /home/john/alpamayo2-edge/pc/env.sh` and
`PYTHONPATH=$W:$W/openpilot:/home/john/alpamayo2-edge/pc/vendor/opendbc` with `W=/home/john/alpamayo-work/alpamayo-w4-integration-20261118`.

## Baseline (frozen model, test set: 120 samples, one night route)

| Error | kept at 20 m | 40 m | 60 m |
|---|---|---|---|
| shift | 1.18 | 0.74 | 1.91 |
| heading | 0.61 | 0.63 | 0.76 |
| both | 0.79 | 1.03 | 1.19 |

Clean: 0.19 m median lateral error over 3 s; turn heading ratio 0.60. This matches the road data (no recovery, under-turning).

## Turn data on native 180/60 views (2026-11-21)

Focus: turning and straight driving. Lane-holding recovery is not trained (the action-expert LoRA did not learn it, even on train data).
- `build_turns.py ROOT 180,60 [--also-120]`: every human-driven moment (1 s grid) of every segment with a turn, plus straight-only
  segments with at least 5 moments. Clean samples only. `samples.py --fov 180,60` renders the lenses' native coverage
  (geometry `output_fov`; wide equidistant 180 deg, narrow pinhole 60 deg). `--also-120` adds stock 120/30 views of heldout moments.
- Navigation text is always "Follow the road." (no route text, turn-signal or lane-change intents; removed from the car in `036af281d`).
- `evaluate_turns.py DIR [PREDICTIONS.npz]`: turn ratio at 3 s and 6.4 s, early turning on approaches, path error on turns and straights.
- Data: `/mnt/nvme-500/alpamayo-train/turns/{samples/{train,heldout},samples-120/heldout}`.

Zero-shot, one heldout segment (45 moments, stock model): 120/30 turn ratio 0.47 at 3 s; 180/60 0.34. The reasoning text stays sensible on 180/60.

## Results: t180-r16 (2026-11-21)

Expert LoRA r16 (q,k,v,o,fc1,fc2), lr 1.5e-4 cosine, 2 epochs, 5,858 clean 180/60 samples (route-choice turns excluded by
`label_ambiguous.py`), flow matching + 1-step car loss, both 7900 XTX (~55 min). Heldout: 569 clean moments, other routes.

| Heldout (clean), 1 Euler step | stock 180/60 | t180-r16 | stock 120/30 |
|---|---|---|---|
| Turn ratio at 3 s / 6.4 s | 0.42 / 0.29 | 0.93 / 0.78 | 0.62 / 0.41 |
| Path error, turns | 2.02 m | 0.76 m | 0.91 m |
| Path error, straights | 0.45 m | 0.37 m | 0.45 m |
| Straight lateral bias at 6.4 s (left +) | +1.34 m (97% left) | -0.21 m (37% left) | +1.19 m (92% left) |

The stock model's left bias on straights (Karel `rpyCalib` yaw -5.5 deg, wide extra -2 deg) is the drive's slow left fade.
Export: `exports/t180-r16` (BF16) and `exports/t180-r16-fp8` (car cache; only the 168 expert linears and their scales differ).

## Results: t120-r16 and Euler steps (2026-11-22)

Same recipe on the same moments rendered at 120/30 (6,080 clean samples). Heldout, clean:

| | stock 120/30 s1 | t120-r16 s1 | stock 120/30 s10 | t120-r16 s10 | t180-r16 s1 | t180-r16 s10 |
|---|---|---|---|---|---|---|
| Turn ratio 3 s / 6.4 s | 0.62 / 0.41 | 0.95 / 0.78 | 0.65 / 0.37 | 0.96 / 0.75 | 0.93 / 0.78 | 0.98 / 0.82 |
| Path error, turns | 0.93 m | 0.81 m | 0.75 m | 0.78 m | 0.76 m | 0.94 m |
| Path error, straights | 0.45 m | 0.34 m | 0.22 m | 0.20 m | 0.37 m | 0.27 m |

t120-r16 straight bias at 6.4 s: -0.10 m (45% left). Both adapters are exported (`exports/t120-r16[-fp8]`, `exports/t180-r16[-fp8]`).
More Euler steps tighten straights; the car keeps 1 step until the 2-3 step check and the car-time cost are known.

## Car runtime check and 2-step training (2026-11-22)

- `eval_runtime.py` (PC copy of the car runtime, FP8, own greedy reasoning, 1 step) with the pinned `t180-r16` FP8 cache and the
  base BF16 checkpoint: turn ratio 0.94 / 0.78, turn path 0.77 m, straights 0.39 m (stock in the same runtime: 0.42 / 0.29,
  2.04 m, 0.45 m). Reasoning identical to stock on 675/675 samples; call time unchanged.
- `t180-c2` (car loss through 2 Euler steps): at 2 steps equal to `t180-r16` at 2 steps (straights 0.25 vs 0.26 m, turns
  0.89 m both); at 1 step slightly worse. Not kept. `t180-r16` at 1 step is the install candidate; 2 steps (`diffusion_steps`)
  is the option for tighter straights at the cost of turn accuracy.
- `t180-r32-e3` (rank 32, 3 epochs): turns better (ratio 0.95 / 0.85, turn path 0.74 m) but straights worse (0.43 m, bias
  -0.36 m right). Not kept; `t180-r16` stays the balanced choice. The VLM LoRA path (`train.py --vlm-rank --teacher`,
  `sample.py --generate`) is smoke-tested (14 GB, 0.83 s/sample) but not trained: expert-only LoRA already closes the 180/60 gap.

## Steps, the left pull, gentle LR and comma data (2026-11-22)

- Stock straight-road bias at 6.4 s (120/30): +1.22 m at 1 step, +0.62 at 2, +0.00 at 10. The car's fixed initial noise with
  one Euler step causes much of the left pull; the fine-tunes learned to cancel it (t120-r16: -0.10 / -0.20 m at 1 / 2 steps).
  Steps 2-3 halve the plan-to-plan spread (t120-r16 IQR at 6.4 s 0.94 -> 0.51 m); 10 steps lean left again on fine-tunes.
- Best: `t120-r16` at 2 steps (turns 0.68 m, straights 0.20 m, lateral at 3 s 0.17 m; car runtime 0.69 / 0.21 / 0.18 m).
- `t120-g3` (lr 5e-5, 3-step car loss): worse (2 steps: turns 0.73 m, straights 0.32 m, bias +0.52 m). Not kept.
- Training never shares a GPU: a concurrent eval caused a memory fault and GPU reset (`gpu_queue.sh` runs jobs one per GPU).
- comma5k comma four (`mici`) data, same OS04C10 cameras and intrinsics as the Dragon: `comma_payload.py` builds payloads from
  the stripped rlog, `localizer.safetensors` (calibration, 100 Hz ECEF pose/velocity -> deviceMotion; target and ego history
  from the driven path) and `frame_info.safetensors`; car geometry from opendbc platform specs. `build_comma.py`: 481/482
  segments, 11,223 train and 2,060 heldout moments per view set (human driving, cruise off), up to highway speed.
- Installed on Karel (2026-11-22 11:19): openpilot `2c1a5e198`, `t120-r16`, 120/30 views, 2 Euler steps, 100 W.
