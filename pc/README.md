# Alpamayo 2 Edge on the PC RX 7900 XTX GPUs

This runtime runs the car's selected Alpamayo recipe (openpilot `alpamayo-w4` at `caa346c94`) on the two RX 7900 XTX GPUs (gfx1100) with KFD.
It has the same `rt.predict(sample)` contract as Chestnut. Use it for replays, input sweeps, and closed-loop simulation.
It does not use or open the Chestnut USB device.

Status: implemented and measured on 2026-11-21. It matches Chestnut on identical inputs below the input-noise level (see [Parity](#parity)).
The car code has moved on to `alpamayo` `d1edf7aa8`. The PC snapshot is not updated: see [Update to the current car code](#update-to-the-current-car-code-not-done).

## What runs

| Part | PC implementation |
| --- | --- |
| Code | `openpilot/selfdrive/alpamayo` from `caa346c94` plus the fork changes up to `2c1a5e198` (view FOV, fine-tune pins, power limit field), and the adapter `pc/alpamayo_pc/port.py`. |
| Fine-tunes | `selected/assets/artifacts.json` is the `d1edf7aa8` file: it pins `t120-r16`, `t180-r16`, `dist-gt10`, `lat10` and `lat2r`. A pinned fine-tune runs with its FP8 cache and the base BF16 checkpoint. |
| Weights | The car's pinned assets: compact BF16 checkpoint, FP8 `weights.bin` (GPU leaf-proof check), fixed prefill scales, car noise. |
| Recipe | Unchanged: W4 serial MLP, FP8 activation and weight rounding, BF16 elsewhere, split prefill, 32 draft heads, 1 to 3 Euler steps (default 1), greedy, 64 tokens. |
| Frozen HIP kernels | The 48 gfx1200 sources compile for gfx1100 with `gfx11_shim.h`. The shim gives exact FP8 E4M3 conversion and gfx12 WMMA through gfx11 WMMA. |
| FP8 tensor-core GEMMs | Operands convert exactly to BF16. Weights convert once at startup (4.6 GB). BF16 WMMA accumulates in FP32. |
| Vision pair, projector, prefill MLP | The general tinygrad path of the same recipe replaces the fused gfx12 objects. This is 2-6x faster on gfx1100. |
| Prefill attention | Native gfx11 flash attention (`kernels/prefill_attention.cpp`): same online-softmax arithmetic, 99.96% bit-equal, max 1 BF16 ulp, 2x faster. |
| Attention projections | Native gfx11 BF16 GEMM (`kernels/gemm_bf16.cpp`) for K=2048, N<=2048: bit-identical to the tinygrad GEMM, 15% faster in graphs. |
| GPU stability | No child processes while the GPU is open (see [GPU faults](#gpu-faults-fixed)); KFD queue evictions stay at 0 ms. |
| Multi-GPU | One process per GPU (data parallel). Model split does not help: one call fits on one GPU. |

FP8 has no hardware path on RDNA3. FP8 values still round exactly as on the car; only the multiply runs in BF16 or FP16, which is exact for FP8 inputs.
Results are not bit-identical to Chestnut. Summation order inside WMMA and in the replaced kernels differs.
`ALPAMAYO_PC_FAST=0` keeps every frozen kernel (shimmed) and FP8-operand GEMMs. It is 2.3x slower and only useful for checks.
`ALPAMAYO_PC_FLASH=0` and `ALPAMAYO_PC_NATIVE_GEMM=0` switch back to the shimmed attention object and the tinygrad GEMM.

## Setup (already done on this PC)

```sh
cd ~/alpamayo2-edge/pc
source env.sh      # PY = vendor/nvidia/.venv (NVIDIA 02f55a7; torch 2.8 CPU, av, scipy); PYTHONPATH = pc, vendor/tinygrad, vendor/opendbc
```

| Path | Content |
| --- | --- |
| `pc/openpilot` | `git archive caa346c94:openpilot selfdrive/alpamayo selfdrive/controls/lib/drive_helpers.py common cereal` of the fork, with the later changes named above, plus `cereal_imports/car.capnp`, three empty `__init__.py` files and the fork's `LICENSE`. Committed, without `common/hardware/comma/id_rsa` (upstream openpilot's public device setup key; the PC does not use it). |
| `pc/vendor/tinygrad` | tinygrad `89846dcfb383a98ee01cfec3bba440ae7dcf9b1e`, git-clean (the runtime verifies the pin). |
| `pc/vendor/nvidia` | NVIDIA alpamayo2-edge `02f55a7aeabc1a2b9f7a80445c08e40bddc89f5e` with its uv venv (copied from the relocated checkout; editable path updated). |
| `pc/vendor/opendbc` | opendbc `27fa1c5d8eed3238a0a78c52ff5abb6cdc807bd2` (only for cereal and payload tools). |
| `pc/assets` | `checkpoint` (compact, 120 MB), `fp8-cache` (`index.json`; `weights.bin` rebuilt with `finetune/prepare-fp8.py` from the original checkpoint, sha256 `dd6b716a...` as pinned; leaf proofs), `fixed`, `geometry.json`. |
| `pc/cache/gpuN` | tinygrad compile cache and persistent captured graphs, one per GPU. |
| `pc/objects` | gfx1100 HIP objects, keyed by source hash. |

Only `pc/openpilot`, the adapter, tools, lab snippets and bench results are in Git. `vendor`, `assets`, `data`, `cache`, `objects` and `parity` are local.

Disk use in home: 10.5 GB runtime (7.1 GB venv, 3.0 GB assets, sources, caches), 36 GB data (`pc/data`: 30 GB sim payloads, 5 GB original checkpoint, 0.7 GB Karel plans), 0.9 GB parity inputs.
Nothing under `/mnt/nvme-500/tmp` is needed: that folder is cleaned automatically.
Nothing is written to `/mnt/nvme-256`; the parity and sim tools only read footage and payloads.

## Commands

Start a persistent lab (model loads in 12 s; first calls compile and capture graphs for about 3 minutes per new shape set):

```sh
cd ~/alpamayo2-edge/pc
./lab_start.sh 0                 # GPU 0 -> lab/gpu0; ./lab_start.sh 1 for GPU 1. Optional: --warmup, --diffusion-steps 2, --zero-noise
./lab.sh 0 lab/snippets/run_samples.py     # 48 Karel drive inputs, stage medians
./lab_stop.sh 0
```

Snippets use the device lab contract (`rt`, `np`, `Tensor`, `dtypes`, `timeit`, `lab_import`, plus `GPU`).
Device lab scripts run unchanged through `tools/pc_sim.py`, which maps `/data/...` paths to PC copies:

```sh
cat > /tmp/sweep0.py <<'X'
exec(open('/home/john/alpamayo2-edge/pc/tools/pc_sim.py').read())
run_device(CLOSEDLOOP+'/inputsweep.py')     # or CLOSEDLOOP+'/batch.py' for closed-loop controller runs
X
./lab.sh 0 /tmp/sweep0.py 36000
```

Payloads: the 15 trimmed 34 s routes in `pc/data/closedloop/trims` (copied from the Karel-footage payloads). Results go to `pc/sim-out/gpu<N>/`.
Scripts come from `/mnt/nvme-500/tmp/alpamayo-closedloop-20261121` while it exists, else from the copy in `pc/data/closedloop`.
For two GPUs, give each lab a different route list (edit `ROUTES` or `routes` in a copy of the script). Both labs skip results already in their own output file.
`lab/snippets/sim_smoke.py` runs a 6 s closed-loop check.

Direct use from Python (one GPU per process):

```python
import alpamayo_pc
rt = alpamayo_pc.load(0)            # selected Runtime on GPU 0; same predict()/prime()/prefill_ahead()/reset()/close()
# A pinned student: alpamayo_pc.load(0, diffusion_steps=2, zero_noise=True, fp8_cache=EXPORTS/'lat2r-fp8', cache_dir=alpamayo_pc.PC/'cache/lat2r-fp8-gpu0')
p = rt.predict(sample)              # sample: image_frames (10,1,3,432,768) uint8, camera_indices, absolute_timestamps, ego_history_xyz/rot, nav_text
```

Batch tools:

```sh
$PY tools/run_samples.py --gpu 0 [--diffusion-steps N] [--zero-noise] [--fp8-cache DIR] [--rounds N] out.json SAMPLE.npz ...   # saved inputs -> predictions
$PY tools/bench.py --gpu 0 --mode sweep|rolling|cached|live [--slots data/slots/ROUTE.npz] [--diffusion-steps N] [--zero-noise]   # throughput and latency
$PY tools/build_slots.py ROUTE data/slots/ROUTE.npz 20         # render consecutive 100 ms slots of a recorded route (CPU)
$PY tools/compare_runs.py a.json b.json                          # ADE/FDE/heading/tokens between two runs
$PY tools/extract_karel_frames.py ... ; lab/snippets + tools/karel_parity.py   # rebuild Karel plan inputs from footage (see file headers)
```

## Speed

Measured 2026-11-21 with `bench/run_all.sh` on the `544f13e5d` snapshot (after the fault fix; `kfd_evicted_ms` 0 in every run, no GPU faults).
`caa346c94` changes only CPU-side code (frontend, planner, supervisor, camera warp); it gives bit-identical predictions on the 48 parity inputs.
Model call latency is wall time to the synchronized result. One process per GPU.

| Workload (`tools/bench.py` mode) | Inputs | p50 / p90 latency | Calls/s, GPU 0 | Calls/s, both GPUs |
| --- | --- | --- | --- | --- |
| `live`: car worker pipeline (prime current pair, early prefill parts between calls) | route `00000091` slots, 100 ms | 108.7 / 110.6 ms | 6.3 | 6.2 + 6.4 = 12.6 |
| `rolling`: closed-loop sim style, 2 new images per call, no pipelining | same slots | 163.5 / 164.6 ms | 6.0 | - |
| `sweep`: 10 new images per call (input sweeps, cold vision) | 48 Karel inputs | 321.7 / 382.4 ms | 3.0 | 3.0 + 3.2 = 6.2 |
| `cached`: repeated input (vision and early parts cached) | 1 Karel input | 82.7 / 83.6 ms | 12.1 | - |

Reference: Chestnut median model call 93 ms (live worker), 121 ms in the 48 sampled live Karel plans; 417 ms per cold call (10 new images) on the 48 parity inputs.
Live mode is 1.17x the Chestnut's 93 ms. Sweep calls are 1.3x faster than Chestnut cold calls per GPU, 2.6x with both GPUs.

Where the time goes (live mode, GPU timestamps on the clock, 113 ms per call):

| Stage | GPU ms | Notes |
| --- | --- | --- |
| Prefill, last part (1344 rows) GEMMs | 48 | BF16 WMMA at about 70-78 TFLOPS in graphs (tinygrad schedule; native kernel only for q/k/v/o). |
| Vision, current pair (prime) | 25-30 | Tiled BF16 GEMMs about 60 TFLOPS, frozen vision attention (shimmed) 3.4 ms. |
| Prefill attention, last part | 9.8 | Native flash kernel; was 28 ms shimmed. |
| Expert, 1 Euler step | 10-12 | 64-row GEMMs and about 1000 small kernels. |
| Norms, quantize, casts, decode, other | 15 | |

History of the live p50: 162 ms (shimmed attention) -> 134 ms (flash attention) -> 111 ms (prime overlapped with CPU preparation, realistic slots) -> 109 ms (native q/k/v/o GEMM).
For comparison, the plain shim port took about 730 ms per sweep call; the first BF16-operand version took 346 ms per sweep call.

## GPU faults (fixed)

Three amdgpu resets on 2026-11-21 (shader fault at address 0x2000) came from KFD queue evictions of the runtime process.
A forked child inherits the render-node file that holds the KFD VM. When the child closes it (exec, exit, close_fds), `amdgpu_flush` waits on the VM.
That wait enables the KFD eviction fence, so KFD evicts and restores every queue of the process through the gfx11 MES firmware.
The children were tinygrad's compile pool (`PARALLEL`, at every graph capture) and git/clang checks after the device opened.
Under load MES failed queue removals (`MES failed to response msg=3`, `Failed to evict queue`) and the GPU reset.
Evidence: `stats_<gpu>/evicted_ms` rose 313 ms during load and 944 ms in 32 calls; bpftrace showed `close -> amdgpu_flush -> kgd2kfd_schedule_evict_and_restore_process`.
Fix in `port.py`: `PARALLEL=0`, source checks and the warp-sampler compile run before the device opens, and a fork hook logs any later fork.
Result: `evicted_ms` stays 0 in load and runs (`bench.py` reports `kfd_evicted_ms_end`). Do not start subprocesses from a process that holds the GPU.

Not fixed: evictions under host memory pressure. On 2026-11-23 other jobs kept the PC at load 45-55 with 0.3 GB free memory and
constant page compaction (about 8,000 migrated pages per second). Every KFD process then collected evictions, including an idle PyTorch-ROCm import:
`evicted_ms` rose to 6-25 s within minutes in runtime processes that started no child process.
Two runs ended with the same MES failure and a GPU reset: GPU 0 at 14:31 (this snapshot, `tools/run_samples.py`) and GPU 1 at 15:07 (lab start).
A reset on one GPU also ends the runtime process on the other GPU (KFD sends the hardware exception to every process).
Probable cause: page migration invalidates the userptr buffers that tinygrad registers with KFD. This is not verified.
Check `/sys/class/kfd/kfd/proc/<pid>/stats_*/evicted_ms` early in a run. Do not run the runtime while the PC is short of free memory.

## Parity

All runs use the car noise (`8c29e314...`) and 1 Euler step. Distances are over the 64-point, 6.4 s trajectory.
The 48 inputs are rebuilt Karel drive inputs (`pc/parity/karel/samples/*.npz`). Chestnut ran the same bytes on Karel (`chestnut.json`).

| Comparison (same 48 inputs) | ADE median / p90 / max | FDE median / p90 | Max heading diff median / p90 | Same tokens |
| --- | --- | --- | --- | --- |
| PC (current default) vs Chestnut, identical bytes | 0.046 / 0.108 / 0.261 m | 0.14 / 0.33 m | 0.13 / 0.84 deg | 40/48 |
| PC vs PC with +-1 LSB pixel noise (input-noise level) | 0.079 / 0.194 / 1.19 m | 0.21 / 0.49 m | 0.15 / 1.09 deg | 37/48 |
| PC plain port (`ALPAMAYO_PC_FAST=0`) vs Chestnut | 0.056 / 0.106 / 2.19 m | 0.16 / 0.33 m | 0.15 / 0.70 deg | 37/48 |
| Chestnut vs its live Karel plans (HEVC footage vs live NV12) | 0.53 / 1.56 / 4.5 m | 1.43 / 4.35 m | 0.83 / 20.1 deg | 30/48 |

The PC matches Chestnut at about half the +-1 LSB input-noise level. All 8 token differences are wording near-ties with the same maneuver
("Remain stopped" vs "Remain stationary", "slowing lead vehicle" vs "lead vehicle ahead brakes").
The native GEMM and flash attention are bit-identical or 1-ulp to the kernels they replace; the default path gives the same trajectories as before them.
Remaining differences: WMMA summation order (gfx11 vs gfx12), and the general-path vision, projector and prefill-MLP GEMMs that replace fused gfx12 objects.

Driving accuracy on 24 turn samples (mean 3 s heading error to the driven path, measured before the samples were deleted): PC 9.24 deg.
The BF16 CPU references gave 9.28 (1 step), 9.62 (1 step, W4 MLP) and 9.88 deg (10 steps).
Repeated PC runs with the same call history are bit-identical, and GPU 0 and GPU 1 give the same bits. The draft history (previous plan text) can change outputs slightly with call order, on Chestnut too.

## Training (LoRA) on these GPUs

Implemented in [`train/`](../train/README.md) with PyTorch-ROCm: LoRA rank 16 on the action expert, one process per GPU.
The tinygrad runtime stays inference-only. It gives the frozen reasoning tokens for the training samples (`finetune/frozen.py`)
and checks an exported fine-tune in the car recipe (`finetune/eval_runtime.py`).

## Update to the current car code (not done)

Status: attempted on 2026-11-23 and withdrawn. `pc/openpilot` stays at the snapshot named in [What runs](#what-runs).

The car code (`alpamayo` `7abef47a2`, pins from `d1edf7aa8`) adds since this snapshot:
1 to 10 Euler steps, a dedicated expert layer (`expert_block`: merged Q/K/V GEMM with per-column weight scales, 24 launches per layer),
the expert FC2 scale from the signed FC1 maximum, two new frozen vision attention objects with direct K/V loads,
a vision FC1 GELU object without range scaling, `compute_profile`, planner lane centring, the navigation distance gate and the worker SIGUSR2 fix.

Facts from the attempt (new archive in `pc/openpilot`, adapter with BF16 copies of the merged expert Q/K/V weights):
- The three new frozen sources compile for gfx1100 with `gfx11_shim.h`. All source patches of `port.py` still apply.
- The runtime started, captured its graphs and gave 48 of 48 plans on the Karel parity inputs (1 step, pinned noise, GPU 1).
- The plans do not match this snapshot's stored run (`parity/karel/pc-gpu1-caa346c94.json`): same tokens on 46 of 48 inputs,
  but ADE median 0.62 m, p90 1.20 m, max 5.1 m; FDE median 2.0 m; heading difference median 4.1 degrees. No trajectory was bit-equal.
  The accepted level is the Chestnut parity above (ADE median 0.05 m). On the car the same code change is bit-identical.
- The difference is in the expert on gfx1100. The first suspect is the merged Q/K/V GEMM: `column_scale=True` never ran on the PC default path before. This is not confirmed.
- No valid timing exists. Host load made CPU stages 3 to 30 times slower (preprocess 414 ms instead of 13 ms).
  GPU-bound stages of the new code: vision 175 ms (this snapshot 166 ms), prefill 116 ms (104 ms), both under load. No 10-step run was made.
- The diagnosis stopped after the second GPU reset of the day (see [GPU faults](#gpu-faults-fixed)).

Needed to finish: a PC without memory pressure; compare the merged Q/K/V GEMM with three separate GEMMs on the same input; fix the adapter;
then repeat the 48-input comparison and the timing at 1 and 10 steps before and after.

## Open problems

- Live latency is 109 ms vs the Chestnut's 93 ms. The rest is GEMM efficiency (about 70-78 of 123 TFLOPS peak) and the vision pair on the clock.
  Next steps: a faster native GEMM for FC1/FC2 and the vision tiles (larger tiles without register spills), and native kernels for the vision attention and expert.
- Duplicate images break the early prefill: if two images of an early part are byte-identical, `Runtime.run_part` passes one feature tensor twice to its
  stitch graph and tinygrad raises `JitError: duplicate inputs to JIT`. This is car code, not the PC adapter. It can happen with repeated source frames.
- Closed-loop sim on the PC (`lab/snippets/sim_check.py`, 20 s, `Live` controller = deployed tracker; `Feedforward` in controllers.py steers inverted):
  route `0000005f`: mean |dy| 0.46 m, max 0.77 m; route `00000080`: mean 0.84 m, max 3.6 m, 5.7 s beyond 1 m. Recorded controller: max 0.09 and 0.67 m.
  Not yet compared with a Karel sim run of the same routes and code.
- The frozen vision attention, expert attention and serial-decode objects still run through `gfx11_shim.h`.
- First use of each new input shape set captures graphs for minutes per GPU (in-process compile, `PARALLEL=0`). Captured graphs persist in `pc/cache/gpuN/graphs`.
  The update attempt removed the stored graphs of GPU 1; the next run captures them again.
- `/mnt/nvme-500/tmp` is cleaned automatically; everything the runtime needs is under `pc/`. `tools/turn_accuracy.py` needs the deleted turn samples.
