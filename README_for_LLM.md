# i.sentinel1.preproc - LLM Handoff

Updated 2026-11-13. Read before editing.

## Entry points
- `i.sentinel1.preproc.py`: GRASS module; calibrates, warps, imports, writes dB/mask/provenance.
- `sentinel1_core.py`: Python numeric reference; used when C binary is absent.
- `src/s1calibrate.c`: C+OpenMP engine; parses XML with expat, reads/writes GDAL, GCPs preserved.
- `src/s1calibrate.cl`: OpenCL arithmetic kernel; host integration still TODO.
- `tests/test_core.py`: 10 passing tests.

## Build and run
```bash
gcc -O3 -march=native -fopenmp -o s1calibrate src/s1calibrate.c \
  $(pkg-config --cflags --libs gdal expat) -lm
./s1calibrate --measurement iw-vv.tiff \
  --calibration calibration-iw-vv.xml --noise noise-iw-vv.xml \
  --output cal.tif --mask mask.tif --lut gamma --threads 8
```

Python discovers `s1calibrate` via `S1CAL_BIN`, `PATH`, or repo root. Set
`S1CAL_DISABLE_C=1` to force the Python reference.

## Science invariants
- `noiseRangeLut` is power in DN2; `signal2 = max(dn2 - range*azimuth, 0)`.
- Calibration LUT is 26 x 645; noise is 26 x 648; resample noise grids.
- `gamma0 = signal2 / gammaLut2`; dB uses `4.342944819032518 * log(x)`.
- Uncompressed Float32 temp is intentional: DEFLATE was 28 s of the old 33 s.
- Output temp keeps WGS84 GCPs; `gdalwarp -order 1` does the ellipsoid correction.

## Performance
- C+OpenMP 4 s; Python 35.5 s; `gdalwarp` 4.3 s; imports ~1.6 s.
- 8 threads: ~480-580% CPU on uncompressed writes; further scaling is I/O-bound.
- OpenCL kernel must be benchmarked against CPU; Iris Xe/PoCL is not automatically faster.

## Next
See `TODO.md` and `JOB.md`. Host OpenCL integration is the only missing code path; do not
claim it works until `--opencl` is implemented and compared to the CPU masks.
