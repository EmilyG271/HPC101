# Lab 2 Bonus: RISC-V RVV + SpaceMiT IME

Date: 2026-08-04

## Environment

- DevPod: `h3250102096-bonus`
- ISA: RISC-V RV64GCV, SpaceMiT X60/K60 with `_ime`
- Compiler: GCC 14.2.0
- Build: `cmake -S . -B build-rv && cmake --build build-rv -j4`
- Test partition: `lab2rv`, 4 CPU cores

## Implementation

- RVV `f32m8` loads, multiplies, reductions, and max reductions accelerate router dot products and per-token absolute maxima.
- Tokens are grouped by selected expert so each IME operation processes up to four tokens together.
- Expert weights are repacked once in `preprocess` into `[output/4][K/8][4][8]` IME tiles.
- Gate and Up projections share each packed activation load and use two signed `vmadot` accumulators.
- Down projection uses signed `vmadot`; native signed input removes the earlier uint8 zero-point shift and row-sum compensation.
- IME K loops are unrolled four times to reduce branch overhead.
- Single-token cases execute routed experts in parallel and write each result to a unique Top-K slot.
- Batch cases use expert-level OpenMP parallelism with compact thread-local accumulation buffers.
- Non-RISC-V builds retain a portable reference fallback.

## Final validation

Command:

```bash
hpc submit -p lab2rv -c 4 bash -c \
  './build-rv/lab2 1 256 128 16 4 100; \
   ./build-rv/lab2 1 1024 512 16 4 100; \
   ./build-rv/lab2 128 256 128 16 4 100; \
   ./build-rv/lab2 1024 512 128 512 2 1'
```

Job: `36417`

| Scenario | Shape `N×D×H×E×K` | Baseline | Optimized | Speedup | Correctness |
|---|---:|---:|---:|---:|---|
| S1 | `1×256×128×16×4` | 0.197540 s | 0.016264 s | 12.1458× | Passed, RMSE `9.16e-08` |
| S2 | `1×1024×512×16×4` | 3.072500 s | 0.183989 s | 16.6994× | Passed, RMSE `7.20e-08` |
| S3 | `128×256×128×16×4` | 25.384500 s | 0.482546 s | 52.6054× | Passed, RMSE `1.18e-07` |
| S4 stress | `1024×512×128×512×2` | 3.328700 s | 0.180520 s | 18.4395× | Passed, RMSE `2.47e-04` |

S1 and S2 exceed the PDF's Bonus 60-point checkpoints (11× and 12×). S3 passes correctness but remains below its 62× checkpoint; profiling showed the grouped routed-expert Gate/Up IME path is the dominant remaining hotspot.
