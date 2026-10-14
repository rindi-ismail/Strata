# 2026-10-14: Radeon AI PRO R9700 (gfx1201) + RX 7900 XTX (gfx1100), layer split, IQ3_S at 262K

Strata 0.1.44.1 (`296b59d`) built from source for `gfx1100;gfx1201`, Intel Core Ultra 7 270K Plus, 125 GiB RAM,
Ubuntu 26.04.1, ROCm 7.2.4, Qwen3.8-Flash-Next GSQ-RCO IQ3_S native pack, `--max-context 262144`, int8 KV. The write-up
(what was changed, why, every number and its limits) is [docs/XTX_R9700_SPLIT.md](../../../docs/XTX_R9700_SPLIT.md).

## Files

| File | What it is |
|---|---|
| `strata-iq3s-default.json` | the final config (split 24, `STRATA_STAGE_TRIM=1`, two prompt tuning tables). Paths are this machine's. |
| `raw/01-flags-sweep-4prompt.log` | flag sweep, short 4-prompt decode mix, one restart per arm. Lines 1-6 were measured on a `--pipeline-windows 1` base by mistake (`base pw1` = 88); `pw0` there is `--pipeline-windows` removed, i.e. the default 2. From `def-b` on, the base is the default. |
| `raw/02-asym-sweep.log` | the mixed-pair sweep (`scripts/tune4.py`): decode on the short mix + a fresh 32K prompt per arm. `lt2` = both tuning tables (change 1). The text hash differs between restarts of the same config, so it is no identity check. |
| `raw/03-lt2-confirm.log` | two more restarts with both tables, then `scripts/speed.py` (short decode + a fresh ~157K prompt). |
| `raw/04-profile-stock-topk.log` | `scripts/prof.py` with `STRATA_VERIFY_PROFILE=1 STRATA_DECODE_TIMING=1 STRATA_SPLIT_TIMING=1` at 1K / 64K / 157K, stock top-k dispatch. Profiling turns off some fusions, so the absolute speeds are lower than unprofiled. |
| `raw/05-profile-topk-register.log` | the same with change 2, step 1 (decode top-k on the register kernel). |
| `raw/06-single-request-ab-noisy.log` | an unprofiled A/B with one request per context length: too noisy to use (the same 1K request gave 99 and 70 tok/s); kept to show why `blong.py` was written. |
| `raw/07-long-context-bench.log` | `scripts/blong.py`, every arm in run order. `topk-off*` = `STRATA_TOPK_DECODE_REG=0`; `topk-on*` = change 2 step 1; from `topk2` on, the binary has both steps; `cur*` = split 26 without trim on that binary. |
| `raw/08-engine-log-157k-prefill.log` | the engine's own lines for the 156,996-token prompt: two before change 1, one after. |
| `scripts/` | the four runners. They rewrite the config, restart the server with the local `ai restart` wrapper and read the engine log; adjust `REPO`, `CFG`, the model name and the port for another machine. |

## Method

- One fresh engine restart per arm; a warm-up request before measuring.
- Short decode mix: four prompts (Go function, B-tree explanation, a refactor, a JSON list), 300 tokens, greedy, two rounds.
- Prompt read: a fresh prompt with a random nonce first, so nothing is reused; tok/s = prompt tokens / time to first token.
- Long-context decode (`blong.py`): the same four questions with no prefix, then a 156,996-token prefix (repository
  source and docs) read once and the four questions on it, served from the prompt cache; client tok/s per answer, plus
  the engine's `decode timing` line (ms per verify window, tokens per window) for each answer.
- Kernel parity and timing: `src/kernels/qsa_topk_parity.cpp` and `src/kernels/qsa_select_bench.cpp` compiled by hand
  against the branch's `libstrata_kernels.a`, run on each card with `HIP_VISIBLE_DEVICES`:

```sh
clang++ -x hip -DSTRATA_USE_HIP=1 -D__HIP_PLATFORM_AMD__=1 --rocm-path=/opt/rocm -O3 -std=c++20 \
  --offload-arch=gfx1100 --offload-arch=gfx1201 -include include/strata/hip_compat/cuda_runtime.h \
  -Iinclude -Iinclude/strata/hip_compat -Ithird_party/ggml src/kernels/qsa_topk_parity.cpp \
  -x none build-hip/libstrata_kernels.a --hip-link -L/opt/rocm/lib -lamdhip64 -o qsa_topk_parity
HIP_VISIBLE_DEVICES=1 ./qsa_topk_parity --selftest
HIP_VISIBLE_DEVICES=1 ./qsa_topk_parity 160965 4 20 262144 0   # decode-shaped: 4 queries, no block count
```

## Not tested

CUDA and SYCL builds; other models and quants; answer quality (only the top-k ids are checked bit for bit); more than
one machine.
