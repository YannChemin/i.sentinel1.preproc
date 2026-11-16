## DESCRIPTION

*i.sentinel1.preproc* calibrates a Sentinel-1 GRD product to sigma0, beta0 or
gamma0, removes thermal noise, geocodes the result to the current GRASS region
and writes a validity mask.

The input is either a local Sentinel-1 GRD `.SAFE` directory or a Planetary
Computer `sentinel-1-grd` STAC item id. In item mode the module reads the
measurement asset and the per-polarisation calibration and noise XML assets.
The measurement is raw uint16 digital numbers with WGS84 ground control points
and no CRS.

The output is a Float32 linear-power map, an optional dB map and an optional
uint8 mask. The dB floor is -40 dB. Pixels at or below the thermal-noise floor
are written at the floor and marked invalid in the mask.

## NOTES

The calibration LUT is interpolated bilinearly in source line and pixel
coordinates and applied at the source pixel centre. The default
**calibration=gamma0** uses the product's gamma LUT, which assumes the
ellipsoid incidence angle. This is appropriate for the flat coastal AOIs that
motivated the module.

`noiseRangeLut` is a power in DN squared, not an amplitude. The module
subtracts it directly from the squared digital number. The azimuth LUT is a
dimensionless multiplier close to one, applied inside its swath window.

In item mode only the part of the swath the current region needs is read. An
IW GRD band is about 250 km square, while a city-scale region is a few tens of
kilometres across, so roughly 4 % of the band carries the product and the rest
would be read, calibrated and discarded by the warp. The scene has no
geotransform, only a grid of ground control points, so the window is found by
fitting a second-order polynomial from longitude and latitude to source pixel
and line over those points and evaluating it along the region's boundary, with
512 pixels of margin. The fit's own residual on the control points is checked;
a scene whose geometry it cannot describe falls back to the whole band with a
warning rather than returning a window in the wrong place. The window is
reported by **--verbose** and recorded in the provenance JSON, and the
calibration and noise LUTs are applied at absolute source coordinates, so a
windowed product is identical to the same rectangle of a whole-band one.

By default the band is then read over `/vsicurl/`, which for a window of this
size finishes well inside the signed URL's lifetime. **cache=** instead
downloads each whole band to a directory of your choosing, which is what
repeated runs over one scene, or an unreliable link, want; a file already
present at the expected size is reused.

The geometry is the product's own GCP polynomial, warped with
`gdalwarp -order 1` to the current region. The module does not use SNAP. It
does not perform DEM-based radiometric terrain correction in this version, so
it does not produce a layover or shadow mask.

The Planetary Computer GRD item does not contain the product annotation or
orbit state vectors. Its `schema-product-*` asset is the RFI annotation, not
the product annotation. DEM terrain correction therefore requires a local
`.SAFE` product or a separate orbit provider.

The module writes temporary full-scene Float32 rasters for the exact
calibration path. Use **-k** to keep them for inspection; otherwise they are
removed.

## ENGINES

Three engines compute the same calibration. **engine=auto**, the default,
prefers the compiled `s1calibrate` binary and falls back to the Python
reference when it is absent. **engine=c**, **engine=opencl** and
**engine=python** force one; a forced engine that is not available is a fatal
error rather than a silent fallback, so a run stays reproducible.

Engine | Precision | Role
--- | --- | ---
`c` | double | the default: C with OpenMP over the scene blocks
`opencl` | single | the arithmetic on an OpenCL device, leaving the cores for I/O
`python` | double | the numeric reference, about nine times slower

With **engine=opencl**, **profile** selects how data reaches the device:
`auto` detects it, `cpu` allocates buffers in host memory, `igpu` maps
transfers instead of copying them, and `dgpu` or `egpu` copy explicitly. A
profile the device cannot honour is refused. **opencl_device** takes an index
from `s1calibrate --opencl-list`, and `S1CAL_OPENCL_PROFILE=1` reports the
transfer and kernel time.

The OpenCL engine is not a faster engine for a single scene. The stage is
bound by GDAL reading the measurement band and writing the Float32 result, so
on a 144 million pixel scene both engines take about 0.8 s while the arithmetic
is 0.15 s of it. What the OpenCL path changes is CPU occupancy, about 180 %
against 720 %, which matters when several scenes are processed concurrently.

Because the kernel works in single precision, the two engines agree to about
6e-8 relative over the signal range but not exactly. Gamma0 is
`max(DN^2 - noise, 0) / gamma^2`, and at the noise floor that subtraction
cancels, so a last-bit difference in the noise term becomes a large relative
difference in a very small result. On a 144 million pixel scene one mask pixel
differed. The provenance JSON records `engine`, `device` and `precision`, so a
product can be traced to the engine that produced it.

## EXAMPLES

Calibrate one Planetary Computer item over the current region and write dB maps
and masks:

```sh
g.region raster=aoi res=10
i.sentinel1.preproc item=S1A_IW_GRDH_1SDV_20240427T021605_20240427T021630_053413_067E8F_7C1D \
    output=s1_20240427 calibration=gamma0 -d -m
```

Calibrate a local SAFE product:

```sh
i.sentinel1.preproc input_dir=/data/S1A_IW_GRDH_1SDV_20240427T021605_20240427T021630_053413_067E8F_7C1D.SAFE \
    output=s1_20240427 polarisations=vv,vh -d -m
```

Run the arithmetic on an OpenCL device and report the transfer and kernel time:

```sh
s1calibrate --opencl-list
S1CAL_OPENCL_PROFILE=1 i.sentinel1.preproc item=S1A_IW_GRDH_1SDV_20240427T021605_20240427T021630_053413_067E8F_7C1D \
    output=s1_20240427 engine=opencl opencl_device=0 -d -m
```

## OUTPUT

For **output=s1_20240427**, **polarisations=vv** and **calibration=gamma0**:

`s1_20240427_vv_gamma0_linear`
: linear gamma0 power

`s1_20240427_vv_gamma0_dB`
: dB gamma0, written with **-d**

`s1_20240427_vv_gamma0_mask`
: 1 valid, 0 invalid, written with **-m**

`s1_20240427_group`
: group containing all polarisation maps

A JSON provenance record is written under `cell_misc/` or under the directory
given with **metadata=**.

## REQUIREMENTS

GRASS GIS 8.4 or later, GDAL with GCP support, Python 3 with `numpy`,
`rasterio`, `pystac-client`, `planetary-computer` and `requests` for item mode.

The compiled engine needs GDAL, expat and OpenMP development files.
**engine=opencl** additionally needs an OpenCL implementation at build time and
a device at run time; without them the module still works through the C or
Python engines.

## SEE ALSO

*[i.group](i.group.md)*,
*[r.import](r.import.md)*,
*[r.in.gdal](r.in.gdal.md)*

## AUTHOR

Yann Chemin
