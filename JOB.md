# i.sentinel1.preproc - assignable jobs

Updated 2026-11-13.

## Job A - OpenCL host integration (ready)
**Goal:** add `--opencl` to `src/s1calibrate.c`, using `src/s1calibrate.cl`, without changing results.
**Inputs:** `src/s1calibrate.c`, `src/s1calibrate.cl`, cached SAFE, `sentinel1_core.py`.
**Work:** create context/queue/program; upload LUT/index/azimuth/dn buffers; enqueue per block; read back linear+valid; fall back to OpenMP when no device is available.
**Acceptance:**
- C+OpenMP remains the default and still builds with `make -C src`.
- `--opencl` masks equal CPU masks exactly; linear max abs diff <= 1e-5.
- Benchmark on Intel Iris Xe and PoCL CPU; report wall time, CPU%, device.
- `engine` metadata reports `opencl` only when it actually ran.
**Non-goals:** do not change the science formulas; do not make OpenCL default before it wins.

## Job B - regression and packaging (ready)
**Goal:** make the C engine reproducible for another machine.
**Work:** build C via CMake; untrack the built binary; add a 512x512 C-vs-Python test; document GDAL/expat/OpenCL prerequisites.
**Acceptance:** fresh clone builds with documented commands; tests pass without a cached SAFE; no binary in git.

## Job C - structural speedup (design)
**Goal:** reduce the 665-scene runtime beyond the 4 s calibration.
**Work:** two-warp crop or AOI-window warp, scene-level parallel workers, temp-file lifecycle.
**Acceptance:** measured end-to-end on 5 scenes; memory and disk bounded; no change in masks.
**Non-goals:** GPU for its own sake; no SNAP.
