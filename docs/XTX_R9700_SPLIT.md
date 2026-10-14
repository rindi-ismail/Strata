# A mixed AMD pair: Radeon AI PRO R9700 (gfx1201) + RX 7900 XTX (gfx1100)

This fork (`rindi-ismail/Strata`, branch `xtx-r9700-tuning`) tunes Strata 0.1.44.1 for one machine with two different
AMD architectures in a layer split, and adds two small engine changes that the mix needs. Everything here was measured
on that one machine on 2026-10-14; the raw logs, scripts and the final config are in
[bench/results/2026-10-14-community-r9700-7900xtx-split/](../bench/results/2026-10-14-community-r9700-7900xtx-split/README.md).

## Summary

| | Stock 0.1.44.1, setup's split | This branch + config |
|---|---|---|
| Prompt read, fresh 32K tokens | 1,277 tok/s | 1,870 tok/s (+46%) |
| Prompt read, fresh 157K tokens | 1,846-1,866 tok/s | 3,006 tok/s (+62%) |
| Decode, 4 questions on a cached 157K prefix | 64.7-66.8 tok/s | 65.3-72.9 tok/s, mean 69.4 |
| Decode, the same 4 questions with no prefix | 79.5-82.1 tok/s | 82.3-83.9 tok/s |
| Decode ms per verify window, 157K / short | 32.3-32.9 / 27.0-27.5 | 30.5-33.8 (median 30.7) / 26.0-27.0 |

The decode figures in the left column were measured with change 1 already in (it does not touch decode) and with
`STRATA_TOPK_DECODE_REG=0`, i.e. the stock decode path at setup's split 26. The prompt gain comes from change 1 below; the decode gain from change 2 (+~5% at 157K) and the split/trim choice (+~3%).
Draft acceptance (tokens per window, 2.2-2.4) did not change in any arm.

## The machine

- GPUs: AMD Radeon AI PRO R9700 32 GB (gfx1201, RDNA4) and AMD Radeon RX 7900 XTX 24 GB (gfx1100, RDNA3). Both cards are
  compute-only; the Intel iGPU drives the desktop. The board bifurcates the two x16 slots to x8/x8: the engine's probe
  reads 27.9 GB/s host-to-device on the R9700 (PCIe 5.0 x8) and 13.3 GB/s on the XTX (PCIe 4.0 x8).
- HIP order: device 0 = XTX, device 1 = R9700. The config uses `"gpu": [1, 0]`, so stage 0 (CUDA0 in the log) is the
  R9700 and stage 1 (CUDA1) is the XTX, which also holds the head.
- CPU Intel Core Ultra 7 270K Plus, 125 GiB RAM, Ubuntu 26.04.1, kernel 7.0.0, ROCm 7.2.4, hipBLASLt 1.2.2 (100202).
- Model: Qwen3.8-Flash-Next GSQ-RCO IQ3_S native pack, `--max-context 262144`, `--kv int8 --kv-resident 32768`,
  `--spec 4 --spec-min-p 0.5` with the MTP draft layer, `--prefill auto:32768`, PLE on SSD (`--ple-io direct`).
  `--pipeline-windows 2` is the 0.1.44 default for two stages and stayed on (it measured fastest: 96-97 tok/s on a short
  4-prompt mix vs 88 with `--pipeline-windows 1` and 89 with `0`).
- Engine built locally with setup's HIP build for `gfx1100;gfx1201` (`-DSTRATA_PREFILL_MMQ=ON`).

## Change 1: one prompt-GEMM tuning table per card (`src/prefill/gemm.cu`)

**Problem.** `STRATA_HIPBLASLT_TUNING` names one table, and a table is scoped to one architecture family. On a gfx1201 +
gfx1100 split setup writes the gfx1201 table; the XTX rejects it at start
(`prefill gemm: tuning architecture mismatch: file=gfx1201 runtime=gfx1100; using hipBLASEx`) and runs its stage's
prompt GEMMs on untuned hipBLASEx, although the repo ships `tools/hip/gfx1100-hipblaslt-100202.txt` for the same
hipBLASLt version.

**Change.** The variable may list several tables, separated by `:` (`;` on Windows). Each card loads the first one whose
architecture and hipBLASLt version match it; if none matches, the errors of all are printed and the card falls back to
hipBLASEx as before. A single path behaves exactly as before. The tuning table format, the loader
(`hipblaslt_tuning.hpp`) and the per-solution checks are unchanged. The SYCL copy of the loader is not touched.

```json
"env": {
  "STRATA_HIPBLASLT_TUNING": "<repo>/tools/hip/gfx1201-hipblaslt-100202.txt:<repo>/tools/hip/gfx1100-hipblaslt-100202.txt"
}
```

The log then shows both: `hipBLASLt tuning enabled (26 rows, gfx1100, version 100202)` and `(24 rows, gfx1201, ...)`.

**Measured** (fresh restart per arm, a fresh 32K-token prompt with a nonce first so nothing is reused): 1,277 / 1,278 /
1,276 tok/s without, 1,874 / 1,873 / 1,868 with (three restarts each). A 156,996-token prompt: 1,846 and 1,866 tok/s
without, 3,006 with. Decode is unchanged (95.3 / 94.3 / 95.2 vs 94.4 / 95.8 on the short mix). The XTX's prompt
GEMMs now use the tuned hipBLASLt solutions, which round differently from hipBLASEx; this is the same table the repo
ships for 7900 XTX users, and answer quality was not measured separately.

## Change 2: the decode top-k on the register kernel (`src/kernels/cuda/qsa_select.cu`, HIP only)

**Problem.** A decode profile (`STRATA_VERIFY_PROFILE=1 STRATA_DECODE_TIMING=1 STRATA_SPLIT_TIMING=1`) at 1K, 64K and 157K
tokens of context showed the same tokens per window (about 2) at every length; the extra time at long context is per
window, and most of it was the QSA layers' `scores+topk`: 0.10 -> 1.31 -> 2.05 ms per window on the R9700 stage and
0.08 -> 0.98 -> 1.47 on the XTX stage (raw: `04-profile-stock-topk.log`).

On HIP, `qsa_block_topk` keeps the capacity rule when the caller passes no block count, and decode passes none (its
graphs are captured). With `--max-context 262144` the capacity is 65,538 blocks, more than `TK_T * TK_PER` = 33,792,
so every decode window fell to `qsa_block_topk_ref`: one 256-thread workgroup per query that reads every key from
memory on each of its four radix passes. A decode window has 2-6 queries, so the selection ran on 2-6 of the GPU's
compute units. The `TK_PER_MAX` = 66 register kernel exists on AMD for exactly this capacity (its comment says so) but
was only reached by the prompt path.

**Change.**
1. On HIP, a call without a block count may use the `TK_PER_MAX` register kernel when the capacity fits it.
   `STRATA_TOPK_DECODE_REG=0` restores the old dispatch. The kernel reads only the live `n_bid + 1` blocks and returns
   the same ids as the reference (ties to the lowest index, cells ascending). Short contexts (`n_kv <= width`) still
   return at once in both kernels.
2. Inside `block_topk_reg_kernel` (HIP build only), the per-pass digit search is a block suffix scan over the 256 bins
   (8 wave32 warps and an `atomicMax`) instead of thread 0 walking up to 255 bins serially. Same digit, same count above
   it. The CUDA build keeps the walk.

**Correctness.** `src/kernels/qsa_topk_parity.cpp` (the repo's parity test, built by hand against this branch's
`libstrata_kernels.a`) passes on both cards: `--selftest`, and decode-shaped calls (4 queries, no block count, capacity
262,144) at 4,096 / 32,768 / 98,304 / 160,965 / 262,144 tokens, with identical ids for continuous scores, 64 score
levels (many ties at the threshold) and all-equal scores. `qsa_select_bench` at 160,965 tokens and 4 queries agrees.

**Measured, kernel only** (the parity test's timing, 4 queries, reference -> this branch):

| Context | R9700 (gfx1201) | 7900 XTX (gfx1100) |
|---|---|---|
| 4,096 | 0.039 -> 0.021 ms | 0.037 -> 0.020 ms |
| 32,768 | 0.063 -> 0.031 ms | 0.052 -> 0.027 ms |
| 98,304 | 0.171 -> 0.059 ms | 0.126 -> 0.050 ms |
| 160,965 | 0.218 -> 0.092 ms | 0.162 -> 0.083 ms |
| 262,144 | 0.383 -> 0.148 ms | 0.277 -> 0.142 ms |

**Measured, end to end** (`scripts/blong.py`: one restart per arm, four fixed questions with no prefix, then a
156,996-token prefix read once and the same four questions on it from the cache, 300 tokens each, greedy; the engine's
`decode timing` line gives ms per window). Step 1 alone, two interleaved pairs: 157K 66.8 / 64.7 -> 69.8 / 68.6 tok/s
(32.9 / 32.3 -> 31.1 / 31.9 ms per window), short context unchanged (27.5 / 27.0 -> 27.5 / 27.5 ms). Step 2 on top: no
change visible end to end (31.5 / 32.4 ms at 157K); it halves the kernel in isolation but the kernel is a small part
of a window by then. Profiled after step 1: `scores+topk` at 157K 1.30 ms (R9700) and 1.11 ms (XTX) per window.

## The config: split 24 and per-stage dense weights

With the pipeline on, the R9700 stage was the slower one (profiled: ~15-19 ms per window vs ~12.5-14 for the XTX), so
layers were moved from it to the XTX rather than the other way. `STRATA_STAGE_TRIM=1` (each card loads only its own
layers' dense weights) makes room on the XTX for the extra layers' experts.

| Arm (157K bench, after change 2) | short ms/window | 157K ms/window | 157K tok/s |
|---|---|---|---|
| split 26 (setup's choice), three runs | 27.45 / 27.01 / 27.65 | 31.40 / 31.10 / 32.36 | 66.4 / 66.5 / 65.6 |
| split 25 | 27.05 | 30.94 | 67.1 |
| split 25 + trim, two runs | 26.82 / 26.54 | 31.15 / 30.88 | 67.7 / 66.5 |
| **split 24 + trim**, three runs | 26.66 / 26.96 / 25.96 | 30.73 / 33.77 / 30.48 | 70.0 / 65.3 / 72.9 |
| split 23 + trim | 26.01 | 30.48 | 71.9 |

Split 24 + trim was chosen (three runs; split 23 has one). With it the XTX runs layers 24-47 and caches 9,769 of their
12,288 profiled experts with 770 MiB of VRAM left, the same headroom as before; the R9700 holds all 12,288 of layers 0-23
with 5.5 GiB left. The second split-24 run's 157K figure (33.77 ms) has one slow answer (53 tok/s); the others are 30.5-30.7.

```json
"layer_split": "24",
"env": { "STRATA_STAGE_TRIM": "1", "STRATA_HIPBLASLT_TUNING": "...gfx1201...:...gfx1100...", "STRATA_HIP_WMMA": "1" }
```

The full config is [strata-iq3s-default.json](../bench/results/2026-10-14-community-r9700-7900xtx-split/strata-iq3s-default.json).

## Measured and rejected on this pair

Short 4-prompt decode mix, fresh restart per arm (`01-flags-sweep-4prompt.log`, `02-asym-sweep.log`). The default arm
measured 94.4-97.0 tok/s across restarts.

| Setting | Result |
|---|---|
| `--pipeline-windows 1` / `0` | 88 / 89 tok/s (default 2: 96-97) |
| `--layer-split auto` | 69.6 tok/s |
| `--spec 3` / `5` / `6` | 96.3 / 93.3 / 90.9 (default 4: 96.2) |
| `--spec-min-p 0.35` / `0.65` | 94.8 / 94.1 (default 0.5) |
| `STRATA_HC_Q8=1` (forced) | 84.9 (on a 88 base) |
| `STRATA_SPLIT_MTP_BATCH=1` | 93.8 decode, 32K prompt unchanged (1,278) |
| `STRATA_SH_STREAM=1` | 75.1 (much slower) |
| the gfx1151 (Strix Halo) decode switch set, process-wide | 81.2 (slower) |
| split 27 / 28 (more layers on the R9700), with or without trim | 87.8-93.0 |
| `--kv-resident 0` (all KV in VRAM), 157K bench | slower at both lengths (short 28.8-29.0, 157K 32.9-34.4 ms/window) |

Not tried: a lower `--vram-reserve-mib` (the engine warns that too little reserve can stall decode, #2065),
`--adapt-async`, and the opt-in `STRATA_SELECT_WMMA` scorer.

## What still costs time at long context

After both changes, a 157K window costs ~31 ms against ~26.5 ms for a short one. The remaining ~4 ms is the QSA
scoring kernel (warp scorer, ~0.08 ms per layer at 161K on the R9700, about 250 GB/s), the KV blocks streamed from
RAM (`--kv-resident 32768`: 92-95% of block reads hit VRAM at 157K), and the attention gather. Removing all of it would
bring 157K to short-context speed, about 80-85 tok/s on this bench; beyond that the short window itself (GDN layers,
expert matrices, the hand-off between the cards, the pipeline guessing only full-accept windows) is the limit.

## Limits of these measurements

- One machine, one model, one day. Decode varies about +-1.5 tok/s between restarts of the same config; prefill
  varies under 0.5%.
- A greedy-text hash per arm was recorded but differs between restarts of the same config, so it is no identity check;
  answer quality was not measured. Change 2's ids are checked bit for bit by the parity test; change 1 changes the
  XTX's prompt arithmetic (tuned hipBLASLt vs hipBLASEx); the trim and the split change which experts sit in VRAM.
- The CUDA and SYCL backends were not built. Change 1 is in shared CUDA/HIP code but only executes on HIP builds (the
  hipBLASLt path); change 2 is inside `#if defined(__HIPCC__)` blocks.

## Applying it to a stock checkout

```sh
git remote add fork https://github.com/rindi-ismail/Strata.git && git fetch fork
git cherry-pick <the two engine commits on fork/xtx-r9700-tuning>   # or: git diff 296b59d fork/xtx-r9700-tuning -- src | git apply
cmake --build build-hip --target strata -j 12
cp build-hip/strata engine/strata.new && mv engine/strata.new engine/strata   # a running engine keeps the old file open
```

Then set `layer_split`, `STRATA_STAGE_TRIM` and the two-table `STRATA_HIPBLASLT_TUNING` in the config and restart. If the
log says `tuning architecture mismatch` the XTX has lost its table (an engine without change 1 is running).
