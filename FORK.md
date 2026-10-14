# About this fork

`rindi-ismail/Strata` is a fork of [Niko1221/Strata](https://github.com/Niko1221/Strata). `main` follows upstream;
the work is on the branch **`xtx-r9700-tuning`**, based on upstream `296b59d` (engine 0.1.44.1).

It tunes Strata for one PC with a mixed AMD pair in a layer split, a Radeon AI PRO R9700 (gfx1201) and an RX 7900 XTX
(gfx1100), and carries two small engine changes:

1. `src/prefill/gemm.cu`: `STRATA_HIPBLASLT_TUNING` may list several tables (`a.txt:b.txt`); each card loads the one for
   its architecture. Before, the second architecture rejected the table and ran untuned (157K prompt read 1,866 -> 3,006 tok/s).
2. `src/kernels/cuda/qsa_select.cu` (HIP only): decode's QSA top-k uses the `TK_PER_MAX` register kernel at
   `--max-context 262144` instead of the one-workgroup reference kernel, and the register kernel's digit search is a
   parallel scan. Same ids (the repo's `qsa_topk_parity` passes on both cards); 157K decode about +5%.
   `STRATA_TOPK_DECODE_REG=0` turns it off.

Plus the measured config (split 24, `STRATA_STAGE_TRIM=1`). Everything, with the numbers and their limits:
[docs/XTX_R9700_SPLIT.md](docs/XTX_R9700_SPLIT.md); raw logs and scripts:
[bench/results/2026-10-14-community-r9700-7900xtx-split/](bench/results/2026-10-14-community-r9700-7900xtx-split/README.md).
