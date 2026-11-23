# Alpamayo GPU kernel optimization on Karel, 2026-11-23

Status: finished 2026-11-23. The three bit-exact wins (section 7 A and B) and the opt-in `compute_profile` option (7 C) are in the
openpilot fork (`alpamayo` `8400cc92a`) and in `pc/openpilot` of this repository. Road validation is not done.

Device: Karel (Dragon + Chestnut RX 9060, USB+AMD tinygrad backend), 100 W power limit, stock weights, 10 diffusion steps.
Code under test: openpilot `alpamayo` `0b3ebf0d7`.
Paths: `K` in the commands is this folder (`kernels/`): `lab/` snippets, `kern/` candidate kernel sources, `gen_gemm.py`, `hipc.py` (PC compile for gfx1200).
The work folder also held `out/` (raw outputs), `src/` (worktree), `refreeze/` and compiled `.hsaco` objects; they are not in this repository.
`./lab.sh` is the device lab client. It has the same snippet contract as `pc/lab.sh` and `pc/lab_server.py`.

## 1. Measurement method

All times below are GPU-side times inside captured TinyJit graphs.

- `lab/stamplib.py` captures a stage with one extra 1-thread launch after every kernel (`kern/stamp.cpp`).
- The stamp reads the previous kernel's output, so the queue flushes before it. It stores the GPU real-time counter (100 MHz, checked against the host clock).
- The difference between two stamps is one kernel's time, including about 1 us of launch cost.
- Stamps serialize kernels. Stamped stage totals agree with the unstamped graph wall time within 2% (wall minus about 1.5 ms submit).
- tinygrad `PROFILE=1` timestamps do not work on this USB backend: the timestamp slots read back as zero, and the graph runs about 8 times slower.

### The GPU state changes kernel times by up to 1.7 times

This is the main reason why earlier isolated kernel timings did not predict full-model results.

| Observation (boot state, 100 W) | Value |
| --- | --- |
| Two expert steps, graph 1 to 5 after 0.3 s idle | 15.85, 11.44, 10.52, 9.34, 9.34 ms |
| Final prefill part, first graph after idle and then back-to-back | 56.9, 64.8, 65.6 ms |
| Sustained prefill | 100 W, 1,888 MHz average GFXCLK (range 500-3,500 MHz) |
| Sustained vision pair | 100 W, 2,114-2,137 MHz |
| Sustained expert steps | 89 W, 2,336 MHz |

- Short kernels after an idle gap run slowly: the clock ramps up with load over tens of milliseconds.
- Long GEMM-heavy graphs are power-limited: the 100 W limit lowers the clock after about 50 ms.
- Compare kernels only in the same state. The short loop reports "warm" (back-to-back) and "after idle" times, with alternating order.
- Stage tables below use a production-like duty cycle (`lab/cycle.py`): `[vision pair, final prefill, 10 expert steps]`, idle, `part 0`, idle, `[vision pair, part 1]`, idle, repeated every 300-350 ms.

## 2. Inventory

### GPU time per plan in the duty cycle (boot state)

| Stage | In cycle | Warm back-to-back | Kernel launches |
| --- | ---: | ---: | ---: |
| Vision pair, current images (critical path) | 31.8 ms | 31.0 ms | 553 |
| Final prefill part, 1,344 rows + 32 heads (critical path) | 55.8 ms | 64.5 ms | 500 |
| Expert, 10 steps (critical path) | 49.1 ms (4.91 per step) | 46.7 ms (4.67 per step) | 682 per step |
| Prefill part 0, 576 rows (ahead) | 28.8 ms | 24.4 ms | 335 |
| Vision pair, -0.1 s images (ahead) | 33.8 ms | 31.0 ms | 553 |
| Prefill part 1, 192 rows (ahead) | 11.7 ms | 11.7 ms | 335 |
| **Critical path** | **136.7 ms** (GPU span 140.6 ms) | | |
| **All GPU work per plan** | **211.0 ms** | | |

With 2 expert steps the expert stage falls to about 9.8 ms; vision (65.6 ms) and prefill (96.3 ms) are then 96% of the GPU time per plan.
The cycle numbers reproduced within 0.3% in two lab sessions.

### Final prefill part (1,344 rows; in cycle 55.8 ms; per-kernel values warm back-to-back, 64.5 ms)

| Kernel | Calls | us per call | ms | Share | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| `alpamayo_prefill_residual` (o_proj and fc2 + residual) | 56 | 355.7 | 19.92 | 30.9% | FP8 WMMA, 86 TFLOPS warm, 100 TFLOPS in cycle |
| `agent_packed_epilogue` (fc1 + ReLU^2 + FP8 pack) | 28 | 596.0 | 16.69 | 25.9% | 85 TFLOPS warm, 98 in cycle |
| `agent_prefill_cache_attention` | 28 | 448.6 | 12.56 | 19.5% | BF16 WMMA, about 35 TFLOPS |
| `alpamayo_fp8_gemm` (q, k, v) | 84 | 98.7 | 8.29 | 12.9% | 76 TFLOPS |
| `alpamayo_norm_fp8` (RMS norm to packed FP8) | 56 | 53.5 | 2.99 | 4.6% | memory pass |
| `E_1344_4_8_16_2_2` (K RoPE + K/V store into the cache) | 28 | 50.0 | 1.40 | 2.2% | memory pass |
| `E_672_4_4_2_16_2_4` (Q RoPE) | 28 | 49.6 | 1.39 | 2.2% | memory pass |
| `E_84_16_4_8_16_4` (attention output to packed FP8) | 28 | 19.2 | 0.54 | 0.8% | memory pass |
| 32 heads: `alpamayo_fp8_gemv`, norms, argmax | 164 | | 0.69 | 1.1% | |

Parts 0 and 1 have the same kernels per layer. Part 0 (24.4 ms warm): residual 38.6%, epilogue 29.1%, q/k/v 17.0%, attention 5.2%, norm 4.7%, RoPE and store 4.5%, pack 0.9%.
Part 1 (11.7 ms): residual 38.3%, epilogue 26.0%, q/k/v 21.2%, attention 7.5%, norm 3.0%, RoPE and store 3.0%, pack 0.8%.

### Vision pair (two images; 31.0 ms warm)

| Kernel | Calls | us per call | ms | Share | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| `fc1_k` (FC1 + GELU + maxima) | 25 | 297.2 | 7.43 | 23.9% | 48 TFLOPS: GELU epilogue and separate accumulation |
| `alpamayo_online` (attention, 25 FP8 + 2 BF16 layers) | 27 | 230.9 | 6.24 | 20.1% | BF16 WMMA, 21 TFLOPS |
| `fc2_k` (FC2 + residual) | 25 | 177.6 | 4.44 | 14.3% | 81 TFLOPS |
| `qkv_k` (merged Q/K/V) | 25 | 145.3 | 3.63 | 11.7% | 79 TFLOPS |
| `o_k` (out_proj + residual) | 25 | 64.7 | 1.62 | 5.2% | |
| `alpamayo_attention_values_3x4` (BF16 linears, layers 0-1) | 11 | 122.9 | 1.35 | 4.4% | |
| `alpamayo_norm_fast` (LayerNorm + row maxima) | 50 | 20.2 | 1.01 | 3.3% | |
| `alpamayo_attention_values` (BF16 FC1, layers 0-1) | 2 | 477.8 | 0.96 | 3.1% | 30 TFLOPS |
| `E_45_269_4_16_4_2` (GELU output to packed FP8) | 25 | 33.9 | 0.85 | 2.7% | |
| `E_45_36_8_16_4_2` (three FP8 quantize + pack per layer) | 75 | 9.1 | 0.69 | 2.2% | |
| `pj_k` (projector GEMMs) | 2 | 338.9 | 0.68 | 2.2% | |
| `E_16_15_18_2_16_4_3`, `E_15_96_32_4_3_3`, `E_30_1076_3_4_4_4` (BF16 layers, patch embedding) | 12 | | 1.21 | 3.9% | |
| Other (norms, reductions, 222 launches under 3 us) | 256 | | 0.9 | 3.0% | |

### Expert step (682 launches; 4.67 ms warm)

| Kernel | Calls | us per call | ms per step | Share |
| --- | ---: | ---: | ---: | ---: |
| `alpamayo_expert_attention` | 28 | 70.8 | 1.98 | 42.4% |
| `alpamayo_fp8_gemm_1_2_residual` (o_proj, fc2) | 56 | 12.7 | 0.71 | 15.2% |
| `alpamayo_fp8_gemm_2_2` (q/k/v, fc1) | 56 | 11.8 | 0.66 | 14.1% |
| `r_64_16_240` (maximum of ReLU^2(fc1)) | 28 | 11.7 | 0.33 | 7.0% |
| `r_2_4_32_2_2_2_4_4_3_2` (first action-projection linear) | 1 | 173.5 | 0.17 | 3.7% |
| `alpamayo_absmax` | 56 | 2.0 | 0.11 | 2.3% |
| `alpamayo_norm` | 60 | 1.6 | 0.10 | 2.1% |
| 12 other elementwise and reduction kernels | 397 | 0.7-2.9 | 0.61 | 13.2% |

### Reasoning fallbacks (only when the prefill draft is not accepted)

| Graph | GPU time warm | After 0.3 s idle | Largest kernels |
| --- | ---: | ---: | --- |
| 16-row verifier | 11.4 ms | 16.7 ms | `alpamayo_fp8_gemm_1_4` 55.7% (169 x 37.5 us), `r_16_16_576` 9.3%, `E_8_8_72_8_16_4` 8.0%, `E_1152_4_8_16_4` 6.3% |
| 32-row verifier | 11.9 ms | 17.8 ms | `alpamayo_fp8_gemm_1_4` 55.3%, `E_8_8_72_8_16_4` 7.7%, `r_32_16_576` 6.6% |
| Six-token serial block | 33.7 ms (5.6 per token) | 40.7 ms | W4 FC2 20.5%, W4 FC1 20.3%, `qkv_rope_store` 16.4%, scores 10.3%, `fp8_gemv` 9.8%, values 9.2% |

Raw tables: `out/fresh_inv.txt`, `out/inv_decode.txt`.

## 3. Short per-kernel test loop

One-time setup in a running lab (about 1 minute). Run from the folder of the device lab client with `LAB_HOST=karel`.
`K` is this folder.

```sh
# 1. Real stage arguments (three seeded predictions; also checks the reference digests)
./lab.sh $K/lab/prep.py $K/lab/stamplib.py $K/lab/kloop.py $K/kern/stamp.cpp $K/kern/stamp.hsaco
# 2. Record the real inputs of one frozen kind for all layers (15-25 s, once per kind)
./lab.sh $K/lab/rec_vision.py                      # KINDS defaults to vision_fc1_gelu_max; edit KINDS for others
# 3. Edit a kernel, compile on the PC (2 s), run the A/B (20 s for three variants)
python3 $K/hipc.py $K/kern/fc1_nosmall.cpp $K/kern/fc1_nosmall.hsaco
./lab.sh $K/lab/ab_fc1.py $K/kern/fc1_nosmall.cpp $K/kern/fc1_nosmall.hsaco     # set KIND and VARS in the snippet
```

- `kloop.record(kind, run)` runs a stage once outside a graph and keeps the real input buffers of every call of that frozen kind.
- `kloop.ab(calls, variants, kind=...)` captures one graph per variant: one launch per recorded call plus a stamp.
- It compares every output with the current object on the GPU, bit for bit, and prints GPU microseconds per call.
- It reports "warm" (back-to-back) and "after 0.3 s idle" times, alternates the order, and uses 6 rounds.
- A variant with other launch dimensions is `{'pair': (source, binary), 'grid': g, 'local': l}` (`lab/sweep.py`, `gen_gemm.py`).
- Generated tinygrad kernels take a `fxn_map` instead of an object pair.
- Prefill kinds record with `rt.prefill.language(...)` (`lab/ab_mlp.py`, 11 s).
- Example output:

```
variant                warm us/call vs current  after idle vs current  bit-exact
current                       251.2      +0.0%       265.7      +0.0%  reference
vattn_direct_nowpe            230.8      -8.1%       245.2      -7.7%  yes
```

Stage-level commands:

```sh
./lab.sh $K/lab/fresh_inv.py $K/lab/inv_all.py $K/lab/cycle.py   # stamped inventory of all stages + duty cycle (4 min)
./lab.sh $K/lab/inv_decode.py                                    # verifier and serial block inventory (3 min)
./lab.sh $K/lab/ab_expert.py $K/lab/expert_new.py                # expert_block variants, bit-exact actions + ms per step
```

Rules found for the loop:

- Call `gc.collect()` between runs (done in `kloop.ab`): the runtime disables automatic collection and graphs leak GPU memory (8 GB card).
- After A/B runs with swapped objects, the frozen vision pair capture fails in that lab ("Existing compiled vision operation differs"). Restart the lab before a full prediction.
- An isolated win must be checked in the stage graph and in the duty cycle. Under the 100 W limit a faster kernel slows its neighbours (section 5).

## 4. Attempts

"Isolated" = short loop on real inputs of all layers, warm. Percent values are against the current object in the same run.

| # | Kernel | Idea | Bit-exact | Before -> after | Verdict |
| --- | --- | --- | --- | --- | --- |
| 1 | `alpamayo_online`, kind `vision_attention_max` (25 FP8 layers) | Each wave reads K and transposed V straight from memory (gfx12 transposing load); no shared-memory staging, no workgroup barriers | Yes: 25 real layers (output and maxima), 9 synthetic calls with queries scaled by 6, 40, 0.01 | 251.2 -> 230.8 us (-8.1%, 10 rounds); in the vision graph 230.9 -> 218.1 us | **Win** |
| 2 | `alpamayo_online`, kind `vision_attention` (2 BF16 layers) | Same change | Yes: 2 real calls | 256.5 -> 237.2 us (-7.5%) | **Win** |
| 3 | `fc1_k`, kind `vision_fc1_gelu_max` | GELU without the exp2 range scaling (`1+e` is 1 with or without it) | Yes: 25 real layers; all 65,536 BF16 inputs on the GPU | 236.8 -> 231.9 us (-2.1%, 10 rounds); in the vision graph 297.2 -> 294.6 us | **Win (small)** |
| 4 | #1 + waves per execution unit 4, 6, 12, 16 (current 8) | Occupancy | Yes | -1.9%, -4.6%, -5.8%, +158% against -7.8% for 8 | No gain |
| 5 | #1 + plain `exp2` when no argument is below -126, rare path left to the compiler | Fewer softmax operations | **No**: 68-71 of 41.5 M words in 21 of 25 calls | -12.3% to -13.5% | Rejected, not adopted |
| 6 | #5 with both paths written out (the frozen object fuses `raw*c-next` for keys 0-15 only) | Exact version of #5 | Yes | -6.9% to -7.0% (worse than #1 alone) | No gain |
| 7 | #1 + per-row skip of the softmax rescale | Fewer multiplies | Yes | -8.7% against -8.3% (noise) | No gain |
| 8 | Staged kernel + #5 | Exp path alone | **No** | -1.9% | Rejected |
| 9 | `fc1_k` prefetch depth 2 | Template `PF` | Yes | +32% | Worse |
| 10 | `fc1_k` WMMA accumulation (`ACC=1`), prefetch 1, 2, 3 | Removes 144 adds per K step | **No**: 239 of 155 M words in 21 of 25 calls | -14% to -15% | Not adopted (numeric change) |
| 11 | `fc2_k` template sweep (10 variants: prefetch 1/2/4, tiles 3x4, 2x6, 3x8, 5x4, 6x4, 2 waves per group) | Tile and schedule | Yes, all | Best: 5x4 -3.5% warm, -0.4% after idle; 2 waves -3.2%; 24 accumulators +200% to +600% | No robust gain |
| 12 | `qkv_k` template sweep (10 variants) | Tile and schedule | Yes, all | -2.4% to +9.1%; 6x4 +125% | No gain |
| 13 | `o_k` template sweep (9 variants) | Tile and schedule | Yes, all | Prefetch 1: -5.6% (3.6 us, 0.09 ms per pair); others -2.6% to +6.2% | No robust gain; prefetch 1 not checked in the stage |
| 14 | `agent_packed_epilogue` (prefill FC1) K-loop unroll 1 and 4 (current 2) | Loop unrolling | Yes | +4.2%, -0.5% | No gain |
| 15 | GPU clock: soft minimum GFXCLK 1,800 and 2,300 MHz | Remove the ramp after idle | n/a | Expert graphs after idle unchanged (15.9, 11.5, 10.5, 9.35 ms) | No effect |
| 16 | GPU: SMU workload profile "compute" (boot state is none of the single profiles) | Faster clock ramp after idle; power limit unchanged (100 W measured) | n/a (no arithmetic change) | Duty cycle, GPU per plan 211.0 -> 203 ms: part 0 28.8 -> 23.9, ahead vision 33.8 -> 30.4, vision 31.8 -> 30.4, prefill 55.8 -> 56.5-56.8, expert 49.1 -> 49.6; critical span 140.6 -> 140.0 | **Win outside kernels**; opt-in patch, full pipeline unverified |
| 17 | Same, profile "fullscreen 3D" and "default" | | n/a | Fullscreen 3D: expert steps +32%, vision +21%; default: part 0 32.0, ahead vision 36.2 ms | Worse |
| 18 | tinygrad `PROFILE=1` timestamps | Per-kernel times | n/a | Slots read zero over USB; graph 8 times slower | Unusable; replaced by stamps |
| E1 | Expert `expert_block`: FC2 scale from the signed maximum of the FC1 output (`r_64_16_240` no longer computes ReLU^2 per element) | ReLU and the rounded BF16 square are monotonic | Yes: two-step actions on real arguments | 4.692 -> 4.499 ms per step (-4.1%), 682 launches | **Win**: full-prediction parity identical; 10-step production graph 47.44 -> 45.49 ms |
| E2 | E1 + staged maximum (15,360 x 16, 60 x 256, 60) | Shorter per-thread loops | Yes | 4.460 ms per step (-5.0%), 710 launches | Within noise of E1; not chosen |
| E3 | Staged maximum of ReLU^2 without the monotonic step | | Yes | 4.521 ms per step (-3.6%), 710 launches | Smaller than E1 |

### Stage and whole-prediction result of wins 1-3 together

Checkout copy `/data/alpamayo-bench/kopt-checkout` with the three swapped objects and the regenerated vision pair plan, fresh lab.

| Measurement | Current | Candidate |
| --- | ---: | ---: |
| Full-prediction parity, `run_sequence` n=3 | `30f6623d22a0d6e1`, `b1555edfe779218d`, `3a805827a3549e04` | identical |
| Vision pair, warm back-to-back | 31.03 ms | 30.88 ms (-0.5%) |
| Vision pair in the duty cycle (critical path) | 31.83 ms | 31.08 ms (-2.4%) |
| Ahead vision pair in the duty cycle | 33.82 ms | 33.16 ms (-2.0%) |
| Critical path GPU span in the duty cycle | 140.6 ms | 140.1 ms |
| All GPU work per plan in the duty cycle | 211.0 ms | 209.7 ms |
| 10-step trajectory graph (with attempt E1 added) | 47.44 ms | 45.49 ms (-4.1%) |
| `run_sequence` median (lab, random images, includes a 0.4 s upload) | 552-572 ms | 574 ms (no measurable change) |

- In the vision graph, attention saved 0.35 ms and FC1 0.07 ms per pair. The other kernels became 0.25 ms slower.
- Cause: warm back-to-back vision runs at the 100 W limit. The new attention has no barrier waits, so it draws more power and the clock falls.
- The gain is real where the limit does not bind (first graphs after idle): 0.7 ms per pair in the duty cycle.

## 5. Findings that change where to look

1. **The power limit and the clock ramp set most of the time.** At 100 W the prefill runs at 1,888 MHz of 3,500 MHz. GEMMs reach 76-86 TFLOPS warm and 98-100 TFLOPS in the duty cycle; these kernels are at the WMMA rate of the current power state.
2. **Under the limit only less work helps.** Removing waits (barriers, staging) raises power and lowers the clock. Removing instructions or rows lowers energy and time.
3. **After an idle gap the first graphs are slow** (expert steps 1.7 times). `AM_PERF_LEVEL=high` is disabled below 144 W; a soft minimum clock does not help; the SMU compute profile does.
4. **Isolated kernel timings mislead** unless the GPU state is the same. Stamps inside the stage graph in a duty cycle reproduce within 0.3%.
5. Prefill GEMMs are 70% of the prefill and have no kernel-level headroom at 100 W. Memory passes (norm, RoPE, store, pack) are 9.7% of the final part.

## 6. Remaining ideas, ranked by expected gain at 100 W

| Rank | Idea | Expected gain | Cost and risk |
| ---: | --- | --- | --- |
| 1 | Enable the SMU compute workload profile (`compute_profile` patch) and measure the real pipeline | About 8 ms less GPU time per plan (4%), mostly on work that runs ahead; more plans per second | One SMU message; needs a replay or bench run; the profile stays in the GPU until a full reset |
| 2 | Skip unused padding rows in the final prefill part (rows above prompt+31; 48 of 1,344 rows for the 2,025-token prompt) | About 2.0 ms of the critical path (3.6% of the final part); real rows keep their bits | New 1,296-row kernel shapes for attention and FC1, a second graph, fallback for long prompts |
| 3 | Apply Q RoPE inside the prefill attention when it loads the query tile | 75.6 us per layer over the three parts = 2.1 ms per plan (1.4 ms critical) | New attention objects with exact RoPE rounding; verify with the loop |
| 4 | Done in the worktree: expert FC2 scale from the signed maximum (attempt E1) | 0.19 ms per step: 1.9 ms per 10-step plan on the critical path, 0.4 ms at 2 steps | Verified: parity identical, 10-step graph 47.44 -> 45.49 ms; no bench replay yet |
| 5 | Expert: first action-projection linear `r_2_4_32_2_2_2_4_4_3_2` (173 us for a 64x512x60 product) | 0.17 ms per step: 1.7 ms per 10-step plan, 0.3 ms at 2 steps | Needs the exact reduction order of the generated kernel |
| 6 | Vision FC1 with WMMA accumulation (attempt 10) | 1.0 ms per pair, 2 ms per plan | Not bit-exact (1.5 words per million); needs trajectory evidence |
| 7 | Prefill attention writes packed FP8 directly (removes `E_84_16_4_8_16_4`) | 30.8 us per layer over the parts = 0.9 ms per plan | New attention objects; exact fixed-scale quantization |
| 8 | Vision BF16 layers 0-1 (3.5 ms per pair at 26-30 TFLOPS; BF16 peak 44) | Up to 1 ms per pair | Generated kernels; exact reduction order required |
| 9 | K RoPE and KV store inside a hand-written Q/K/V GEMM | 76 us per layer = 2.1 ms per plan | Largest change; RoPE pairs span two output tiles |
| 10 | Exact vision attention exp path (attempt 6) | None measured | Closed |

Not worth repeating: tile and prefetch sweeps of the vision GEMM template, waves per unit for attention, launch-count reduction.

## 7. Wins: files, objects and verification

Nothing was installed on the car during this work. The changes are committed in the fork as `8400cc92a`.

### A. Vision attention and FC1 GELU objects (bit-exact, verified)

| Kind | Old source hash | New source hash | New object bytes | Lab source |
| --- | --- | --- | ---: | --- |
| `vision_attention_max` | `6f701907...` | `33251bf8c1e5...` | 21,928 | `kern/vattn_direct_nowpe.cpp` |
| `vision_attention` | `557da7f2...` | `1d508bb1e070...` | 20,520 | `kern/vattn_plain_direct.cpp` |
| `vision_fc1_gelu_max` | `c85f36b4...` | `8861ae7cd37c...` | 46,616 | `kern/fc1_nosmall.cpp` |

Changed files under `openpilot/selfdrive/alpamayo/selected/`:

- `assets/gfx1200/`: the three new `.cpp` and `.hsaco` pairs; the three old pairs removed.
- `assets/manifest.json` (SHA-256 `6b8bc561...`).
- `vision_pair/gfx1200.json.gz` (regenerated plan, SHA-256 `28a0d728...`, 38 programs, all reused).
- `vision_pair/programs.py` (`SHA256`), `vision_pair/__init__.py` (`ORDERED_PROGRAMS = b6da5dc0...`).
- `README.md` (three sentences).

Verification:

1. Short loop on real inputs of every layer: outputs and maxima bit-identical (attempts 1-3). Synthetic scaled queries: bit-identical.
2. GELU: all 65,536 BF16 inputs give identical bits on the GPU (`kern/gelu_check.cpp`).
3. Plan regenerated in a fresh lab with `bench/lab/refreeze_plan.py` (`lab/refreeze_plan_kopt.py`); `refreeze_objects.py` alone leaves stale table keys.
4. Fresh lab on the checkout copy: `run_sequence` n=3 digests equal the reference; stage results in section 4.

Not verified: a bench replay or road test; cached graphs are recaptured after the change (new graph-cache key, about 5 minutes once).

### B. Expert FC2 scale from the signed maximum (bit-exact on the tested input)

- `selected/model.py`, `expert_block`: four changed lines (attempt E1). No kernel object changes.
- Verified: two-step trajectory actions bit-identical in stamped graphs; 4.692 -> 4.499 ms per step warm.
- Verified: `run_sequence` n=3 digests equal the reference, together with the three vision objects (instance patch in the lab, trajectory graph recaptured).
- Verified: unstamped 10-step trajectory graph 47.44 -> 45.49 ms (-1.94 ms, -4.1%), same lab, same minute.
- Not verified: the edited `model.py` file itself was not loaded on the device; the lab ran the same function from `lab/expert_new.py` (mode `signed`).
- Limit of the argument: a NaN in the FC1 output can give another scale. The current code has no NaN handling either.

### C. `compute_profile` option (opt-in, default off)

- `selected/runtime.py`: `Config.compute_profile`, one SMU message after GPU setup, manifest field. `runtime.py`: allowed inference field. `selected/README.md`: three sentences.
- The SMU message was tested from the lab (attempts 16-17). The code path in `runtime.py` was **not run**.
- Editing `selected/runtime.py` changes the graph-cache key even when the option is off.

## 8. Files

- `REPORT.md`: this report.
- `lab/stamplib.py`, `kern/stamp.cpp|.hsaco`: in-graph timestamps.
- `lab/kloop.py`, `lab/rec_vision.py`, `lab/ab_*.py`, `lab/sweep*.py`, `gen_gemm.py`: short loop and sweeps.
- `lab/prep.py`, `lab/inv_all.py`, `lab/fresh_inv.py`, `lab/inv_decode.py`, `lab/cycle.py`: arguments, inventory, duty cycle.
- `lab/clk1.py` to `lab/clk6.py`: clock, soft minimum and workload profile tests.
- `lab/expert_new.py`, `lab/ab_expert.py`: expert block variants.
- `lab/refreeze.py`, `lab/replan.py`, `lab/order2.py`, `refreeze/`: object swap, plan regeneration and its outputs.
- `kern/`: all candidate sources and objects (`vattn_*`, `fc1_*`, `fc2_*`, `qkv_*`, `o_*`, `mlp_*`, `gelu_check`).
- `out/`: raw outputs (`fresh_inv.txt`, `kopt_inv.txt`, `inv_decode.txt`, `kopt_prep.txt`, disassemblies).
- `src/`: git worktree with the changes of section 7 (not in this repository; committed in the fork).
