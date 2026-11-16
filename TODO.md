# i.sentinel1.preproc - TODO

Updated 2026-11-14.

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
- [x] Host OpenCL integration behind `--opencl`, with `--opencl-device`,
      `--opencl-list`, `--profile auto|cpu|igpu|dgpu|egpu` and
      `--opencl-timing` (or `S1CAL_OPENCL_PROFILE=1`).
- [x] Device profiles: `cpu` host-allocated buffers, `igpu` mapped transfers
      with driver-placed buffers, `dgpu`/`egpu` explicit copies. A forced
      profile the device cannot honour is fatal, never a silent fallback.
- [x] Kernel built with `clBuildProgram`; the build log is printed on failure.
      Kernel located beside the binary, then `../etc`, `../share`, then
      `S1CAL_KERNEL`; not finding it is fatal.
- [x] LUTs uploaded once per scene, block buffers allocated once and reused;
      a block larger than `CL_DEVICE_MAX_MEM_ALLOC_SIZE` is refused with the
      advice to lower `--memory-mb`.
- [x] Fixed two kernel defects: `__global` qualifier missing on the LUT row
      pointers (would not compile), and `DN*DN` in float (now exact in uint).
- [x] `tests/test_engines.py`: 512x512 synthetic-scene regression for the C,
      OpenCL and Python engines, plus the forced-profile failure paths.
- [x] GRASS options `engine=auto|c|opencl|python`, `profile=`, `opencl_device=`;
      provenance records `engine`, `device` and `precision`.
- [x] Benchmarked on mains power: see README_for_LLM.md. **The default stays
      C+OpenMP**, because the stage is I/O bound and OpenCL gives no wall-clock
      gain; `--opencl` is kept because it costs 181 % CPU against 723 %, which
      is what parallel scene workers need.

- [x] Manual rewritten as Markdown (source of truth) with the HTML committed
      alongside; `ENGINES` section documents the three engines, the profiles
      and the single-precision caveat.
- [x] Addon builds through the grass-addons symlink
      (`src/imagery/i.sentinel1.preproc`) and installs into the dist tree and
      `~/.grass8/addons`: script, `sentinel1_core.py` via `ETCFILES`, HTML and
      man page. `s1calibrate` and its kernel go to the addon `bin/`.
- [x] Annotation XML read through GDAL's VSI layer (`VSIFOpenL`), not `fopen`,
      so a `/vsicurl/` calibration or noise path works. The Planetary Computer
      serves the annotations over HTTPS with no local copy, and `fopen` failed
      with ENOENT; only the measurement band had been routed through VSI. The
      reader also refuses a body under 1 KiB, so an error page or a redirect is
      reported instead of being handed to expat.
- [x] Measurement band read **windowed**: only the part of the swath the
      current region needs.
      The window comes from a second-order inverse fit of the scene's GCPs,
      checked against its own residual, with 512 px of margin and a fall back
      to the whole band on a bad fit. LUTs are applied at absolute source
      coordinates, so a windowed product is bit-identical to the same
      rectangle of a whole-band one (`tests/test_engines.py`, 2 new tests, one
      of which also checks the GCP shift). `--window X,Y,W,H` on the binary,
      automatic in the module, recorded in the provenance JSON.
- [x] `cache=` option: download whole bands to a directory instead, reusing a
      file already present at the expected size (HEAD, not GET).

### Why the window matters more than any engine choice

Measured on this link, 2026-11-16: **1.0 MiB/s aggregate**, identical at 1, 4
and 8 concurrent connections, so it is the link, not a per-connection cap, and
parallel scene workers buy nothing on the download. One VV band is 0.46 GiB.

| Route | Data for 665 scenes x 2 pol | Time at 1.0 MiB/s |
|---|---|---|
| Whole band over `/vsicurl/` | 612 GiB | 174 h (7.3 days) |
| Windowed to the Sharjah AOI (4.2 %) | 26 GiB | 7.3 h |

The first attempt at a whole-band read failed after 48 minutes: the signed URL
expired mid-read and Azure answered the remaining ranges with partial bodies
(`TIFFReadEncodedTile() failed`). A window is both small enough to finish
inside the token's life and 24 times less data.

## Next
- [ ] Re-benchmark with 4-8 concurrent scene workers, which is the case where
      the OpenCL path should win on throughput rather than on one scene.
- [ ] Test the CMake engine target in the GRASS build tree (not yet run);
      the GNU Make path through grass-addons is validated.
- [ ] Install into `/usr/local/grass86` (needs root; the user's password was
      not available in the session that built it).
- [ ] Try `cl_khr_fp64` for the noise subtraction on devices that have it, to
      see whether the noise-floor cancellation narrows; measure before keeping.
- [ ] Overlapped transfers (two queues, double-buffered blocks) for a `dgpu`
      or `egpu` profile. Pointless on the iGPU, where the transfer is a map.
- [ ] Intel `ocloc compile` output for SIMD width and spills, then decide
      whether a vectorised variant along the contiguous axis is worth it.
