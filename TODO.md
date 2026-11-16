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
- [x] **Fixed a defect that made the delivered mask useless.** The engines
      wrote 0 both for a DN of 0 (outside the swath) and for a pixel the noise
      subtraction drove to the floor, and `_warp` used `-srcnodata 0` for both
      rasters, so every invalid pixel became nodata: the first real end-to-end
      product had a VV mask of 25,701,142 cells all equal to 1, with not one 0,
      and 3.48 M VH pixels (13.5 %) silently null instead of at the noise
      floor. The dB floor of -40 was unreachable for the same reason. Now
      outside the swath is NaN / 255 and declared as the band nodata, while a
      noise-clipped pixel keeps a real 0 in all three engines and reaches the
      dB floor. Pinned by `test_outside_the_swath_is_nodata_and_noise_clipped_is_zero`.
- [x] Inverse GCP fit tries orders 2, 3, 4 and takes the lowest whose residual
      stays under a quarter of the margin. Order 2 left 98 px of residual on a
      real IW scene, which tripped the old fixed 64 px limit and fell back to
      the whole band.

### Why the window matters more than any engine choice

Measured on this link, 2026-11-16: **1.0 MiB/s aggregate**, identical at 1, 4
and 8 concurrent connections, so it is the link, not a per-connection cap, and
parallel scene workers buy nothing on the download. One VV band is 0.46 GiB
compressed.

Measured on the Sharjah AOI (45.9 x 57.4 km, EPSG:32640, 10 m) against the
scene `S1A_IW_GRDH_1SDV_20240415T021552_..._053438_067B9A`, whose band is
25724 x 15608:

| Route | Window | Per scene, VV+VH | 665 scenes |
|---|---|---|---|
| Whole band | --- | 11 m 35 s | 174 h |
| Windowed | 7561 x 6452 at (6131, 412) = **12.2 %** | **2 m 25 s** | **~21 h** |

The window is 12.2 % of the band, not the 4.2 % the area ratio suggests,
because the AOI is rotated with respect to the swath, so its bounding box in
source line and pixel space is much larger than its area, and 512 px of margin
is added on each side.

The first attempt at a whole-band read failed after 48 minutes: the signed URL
expired mid-read and Azure answered the remaining ranges with partial bodies
(`TIFFReadEncodedTile() failed`). A window is both small enough to finish
inside the token's life and eight times less data.

### Window validation on a real scene

A windowed run and a whole-band run of the **same code** over the same scene
and region, so the window is the only variable:

| | VV | VH |
|---|---|---|
| Mask value counts | identical (0: 74,696, 1: 25,701,142) | identical (0: 3,479,998, 1: 22,295,840) |
| Mask disagreements | 0 | 0 |
| Max abs dB difference | 5.2e-7 | 3.1e-5 |
| Pixels differing at all | 825 of 25,775,838 (0.003 %) | 2,207 (0.009 %) |
| Pixels differing > 0.01 dB | 0 | 0 |

The residual is `gdalwarp` arithmetic: the window moves the source grid origin,
so the resampling weights come out of slightly different floating-point
coordinates. It is physically irrelevant at 1e-5 dB.

The earlier comparison against the pre-fix product showed differences up to
-6 dB, always negative. That was the nodata fix, not the window: bilinear
resampling now includes noise-floor pixels the old run skipped as nodata, which
pulls neighbours down. Confirmed by the gradient -- mean -0.139 dB (VV) and
-0.453 dB (VH) next to a mask-0 pixel against -0.0002 and -0.001 dB away from
one, decaying monotonically as the exclusion radius widens.
- [x] **Orbit state vectors: source found and implemented.** ESA's auxiliary
      orbit files are served by the public `s1-orbits` bucket over anonymous
      HTTPS, no account or token. Copernicus Data Space OData searches openly
      but returns 401 on download; `qc.sentinel1.eo.esa.int` does not resolve
      from this machine. New `sentinel1_orbit.py` (shipped via ETCFILES) plus
      `orbit=none|auto|precise|restituted` and `orbit_dir=`. Verified on the
      test scene: 9,361 state vectors at 10 s spacing, radius 7,073 km and
      speed 7,597 m/s at the acquisition instant. 19 tests, 18 offline with
      the network replaced and 1 marked `network` that checks the bucket is
      still anonymous (`pytest.ini` deselects it by default).
      The vectors are **recorded, not used**: calibration is still against the
      ellipsoid and the provenance says so. One bug worth remembering, caught
      only by running it against the live bucket: the listing key was built as
      `AUX_AUX_POEORB`, which sorts before every real key, so S3 restarted at
      the 2014 beginning of the archive and the scene was reported as having no
      orbit. The offline test passed throughout because it asserted only the
      date in the key; it now asserts the whole key.

## Next
- [ ] Re-benchmark with 4-8 concurrent scene workers, which is the case where
      the OpenCL path should win on throughput rather than on one scene.
- [ ] Test the CMake engine target in the GRASS build tree (not yet run);
      the GNU Make path through grass-addons is validated.
- [ ] Install into `/usr/local/grass86` (needs root; the user's password was
      not available in the session that built it).
- [ ] Terrain correction on top of the state vectors, in four steps: interpolate
      the vectors to each azimuth line (10 s spacing at 7.6 km/s needs better
      than linear); zero-Doppler range-azimuth solution per DEM cell for the
      local incidence angle; layover and shadow (`r.horizon` gives the horizon
      angles); radiometric normalisation by the local illuminated area. Only
      the last changes pixel values, and it should not be claimed until it is
      validated against the `sentinel-1-rtc` collection.
- [ ] Try `cl_khr_fp64` for the noise subtraction on devices that have it, to
      see whether the noise-floor cancellation narrows; measure before keeping.
- [ ] Overlapped transfers (two queues, double-buffered blocks) for a `dgpu`
      or `egpu` profile. Pointless on the iGPU, where the transfer is a map.
- [ ] Intel `ocloc compile` output for SIMD width and spills, then decide
      whether a vectorised variant along the contiguous axis is worth it.
