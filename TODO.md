# i.sentinel1.preproc - TODO

Updated 2026-11-13.

## Done
- [x] Python reference core and 10 tests.
- [x] GRASS module, `item=` and `input_dir=` modes.
- [x] C+OpenMP engine `src/s1calibrate.c`, uncompressed temp output.
- [x] Numerical equivalence vs Python: max abs diff 4.8e-7, masks equal.
- [x] 4 s per full VV scene (vs 35.5 s Python).
- [x] Python frontend auto-discovers C binary (`S1CAL_BIN`, `S1CAL_DISABLE_C=1`).
- [x] OpenCL kernel source `src/s1calibrate.cl`.
- [x] CMake target for the C engine; compiled binary untracked and ignored.
- [x] `JOB.md` with acceptance criteria.

## Next
- [ ] Test the CMake engine target in the GRASS build tree.
- [ ] Host OpenCL integration in `src/s1calibrate.c` behind `--opencl`.
- [ ] Build kernel with `clBuildProgram`, choose device (`--opencl-device`).
- [ ] Upload LUT/index/azimuth/dn buffers; enqueue per block; download results.
- [ ] Benchmark C+OpenMP vs OpenCL on Intel Iris Xe and PoCL; keep faster default.
- [ ] Test `--opencl` vs CPU mask equality and 1e-5 linear tolerance.
- [ ] Wire `engine` metadata to report `c-openmp` or `opencl`.
- [ ] Add 512x512 regression test for C and OpenCL paths.
