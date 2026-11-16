# i.sentinel1.preproc - LLM Handoff

Updated 2026-11-14. Read before editing.

## Entry points
- `i.sentinel1.preproc.py`: GRASS module; calibrates, warps, imports, writes dB/mask/provenance.
- `sentinel1_core.py`: Python numeric reference; used when C binary is absent.
- `src/s1calibrate.c`: C+OpenMP engine; parses XML with expat, reads/writes GDAL, GCPs preserved.
- `src/s1calibrate.cl`: OpenCL arithmetic kernel, run through `--opencl`.
- `tests/test_core.py`: 10 passing tests.
- `tests/test_engines.py`: 5 tests comparing the C, OpenCL and Python engines
  on a synthetic 512x512 scene, including the forced-profile failure paths.

## Build and run
```bash
gcc -O3 -march=native -fopenmp -o s1calibrate src/s1calibrate.c \
  $(pkg-config --cflags --libs gdal expat) -lm
./s1calibrate --measurement iw-vv.tiff \
  --calibration calibration-iw-vv.xml --noise noise-iw-vv.xml \
  --output cal.tif --mask mask.tif --lut gamma --threads 8
```

Python discovers `s1calibrate` via `S1CAL_BIN`, `PATH`, or repo root. Set
`S1CAL_DISABLE_C=1` to force the Python reference, or pass
`engine=c|opencl|python` to the module. A forced engine that is unavailable is
fatal; it never falls back in silence.

```bash
./src/s1calibrate --opencl-list          # devices, with the detected profile
./src/s1calibrate ... --opencl           # first GPU, profile detected
./src/s1calibrate ... --opencl --opencl-device 1 --profile cpu --opencl-timing
```

`S1CAL_KERNEL` overrides the path of `s1calibrate.cl`, which is otherwise read
from beside the binary, then `../etc`, then `../share`.

## Science invariants
- `noiseRangeLut` is power in DN2; `signal2 = max(dn2 - range*azimuth, 0)`.
- Calibration LUT is 26 x 645; noise is 26 x 648; resample noise grids.
- `gamma0 = signal2 / gammaLut2`; dB uses `4.342944819032518 * log(x)`.
- Uncompressed Float32 temp is intentional: DEFLATE was 28 s of the old 33 s.
- Output temp keeps WGS84 GCPs; `gdalwarp -order 1` does the ellipsoid correction.

## Performance
- C+OpenMP 4 s; Python 35.5 s; `gdalwarp` 4.3 s; imports ~1.6 s.
- 8 threads: ~480-580% CPU on uncompressed writes; further scaling is I/O-bound.

### OpenCL benchmark, 2026-11-14

144 M pixels (12 000 x 12 000 synthetic scene), mains power, `balanced` power
profile, i7-1165G7 with Iris Xe, 36 blocks at `--memory-mb 2048`:

| Engine | Wall | CPU | Arithmetic | Note |
|---|---|---|---|---|
| C+OpenMP, 8 threads | **0.80 s** | 723 % | --- | the default |
| C+OpenMP, 1 thread | 1.01 s | 165 % | --- | |
| OpenCL, Iris Xe (igpu) | 0.81 s | **181 %** | 0.153 s | kernel 0.059 s, 27.1 GB/s |
| OpenCL, PoCL CPU | 1.76 s | 272 % | 0.571 s | kernel 0.453 s, 3.5 GB/s |

**The stage is I/O bound, so OpenCL gives no wall-clock gain on one scene.**
The arithmetic is 0.15 s of a 0.8 s run; the rest is GDAL reading 290 MB and
writing 576 MB of float plus 144 MB of mask. One CPU core already keeps up.

`--opencl` is kept, and is not a dead end, because it costs **181 % CPU
against 723 %**. Stage 04 runs 4-8 concurrent scene workers, so the question
that matters is throughput with the cores freed for I/O and `gdalwarp`, not
latency on one scene. That benchmark has not been run yet; do not claim a
throughput gain until it has.

### Precision: the two engines are not identical, by construction

The kernel works in single precision, the C and Python engines in double. On
the 144 M-pixel scene:

- max relative difference over the signal range: **6e-8**;
- **one** mask pixel in 144 million differs;
- 0.044 % of pixels exceed 1e-5 relative, all of them at the noise floor.

The reason is cancellation, not a formula difference. Gamma0 is
`max(DN^2 - noise, 0) / gamma^2`, and where `DN^2` approaches the noise term
the subtraction cancels, so a last-bit difference in the noise term becomes a
large relative difference in a tiny result. The worst case above sits at
gamma0 = 1.4e-11, which is -108 dB and physically meaningless. With
`--subtract-noise 0` there is no cancellation and the engines agree to 3.7e-7
relative, which is the honest measure of the kernel's accuracy.

The same cancellation separates the C and Python engines, which are both
double: the Python reference resamples the noise LUT onto the calibration
pixel grid while the C engine interpolates on the native noise grid.

## Next
See `TODO.md` and `JOB.md`. The OpenCL path is implemented, tested and
benchmarked; the open question is concurrent-worker throughput, not the kernel.
Do not change the default engine away from C+OpenMP without a measurement on
mains power, logged with the power state.
