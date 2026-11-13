# i.sentinel1.preproc

A GRASS GIS addon that calibrates Sentinel-1 GRD to sigma0, beta0 or gamma0,
removes thermal noise, geocodes the result to the current region and writes a
validity mask. It reads either a local GRD `.SAFE` directory or a Planetary
Computer `sentinel-1-grd` STAC item.

The addon is written for the OlmoEarth Sharjah flood and Dubai RTA retraining
pipelines. It exists so stage 04 does not depend on SNAP or on an ad-hoc
calibration block.

## Status

Phase 1 implements the ellipsoid calibration and geocoding path:

- calibration and noise XML parsers;
- bilinear LUT interpolation in source line/pixel coordinates;
- thermal-noise subtraction, with `noiseRangeLut` treated as a power in DN²;
- exact calibration in source geometry, then `gdalwarp -order 1` with the
  product GCPs;
- linear, dB and mask outputs, plus a JSON provenance record.

DEM-based radiometric terrain correction is not implemented. It needs the
product annotation and orbit state vectors, which the Planetary Computer GRD
item does not contain. Use a local `.SAFE` product for that work when the
terrain path is added.

## Install

From a GRASS session:

```sh
g.extension extension=i.sentinel1.preproc \
    url=https://github.com/YannChemin/i.sentinel1.preproc
```

Or link the checkout into a GRASS addons tree and run `make` in this directory:

```sh
ln -s "$PWD" /path/to/grass-addons/src/imagery/i.sentinel1.preproc
make MODULE_TOPDIR=/path/to/grass
```

## Requirements

- GRASS GIS 8.4 or later
- GDAL with GCP support (`gdalwarp`)
- Python 3 with `numpy`, `rasterio`, `pystac-client`, `planetary-computer` and
  `requests`

The GRASS Python environment must see those packages. On this machine the
working invocation is:

```sh
grass /path/to/location/PERMANENT --exec /path/to/venv/bin/python \
    /path/to/i.sentinel1.preproc/i.sentinel1.preproc.py \
    item=... output=... -d -m
```

After installation, the same call is simply `i.sentinel1.preproc` inside the
GRASS session.

## Usage

```sh
g.region raster=aoi res=10
i.sentinel1.preproc item=S1A_IW_GRDH_1SDV_20240427T021605_20240427T021630_053413_067E8F_7C1D \
    output=s1_20240427 calibration=gamma0 -d -m
```

For a local product:

```sh
i.sentinel1.preproc input_dir=/data/S1A_IW_GRDH_1SDV_20240427T021605_20240427T021630_053413_067E8F_7C1D.SAFE \
    output=s1_20240427 polarisations=vv,vh -d -m
```

Output names for the example above:

- `s1_20240427_vv_gamma0_linear` — linear power
- `s1_20240427_vv_gamma0_dB` — dB, with `-d`
- `s1_20240427_vv_gamma0_mask` — 1 valid, 0 invalid, with `-m`
- `s1_20240427_group` — group with all maps

The module always writes the linear map; `-d` adds the dB map. The dB floor is
-40 dB. Pixels at or below the noise floor are written at the floor and marked
invalid.

## Why the exact path first

A spatially varying LUT cannot be applied by GDAL during warp. The two possible
implementations are:

1. calibrate in source geometry, then warp the calibrated Float32 raster;
2. warp the DN and the LUTs separately, then calibrate in the target grid.

Phase 1 uses the first path. It is exact and easier to validate, at the cost of
a full-scene temporary raster (about 1.6 GB per polarisation). A two-warp batch
path can be added after the measured difference between the two is below the
acceptance gate.

## Calibration and noise

For each source pixel:

```text
dn2         = dn²
noise_power = noiseRangeLut * noiseAzimuthLut
signal2     = max(dn2 - noise_power, 0)
gamma0      = signal2 / gammaLut²
```

`noiseRangeLut` is a power in DN². Squaring it would put the noise floor near
+12 dB and suppress most of a scene. The azimuth LUT is a dimensionless
multiplier near one.

The GCP polynomial is the product's own geometry. The first delivery does not
resample the DN before calibration. It does not claim a layover or shadow mask.

## Tests

```sh
python -m pytest -q tests/test_core.py
```

The unit tests use synthetic calibration and noise XML with known LUT values.
They cover parsing, interpolation, the noise floor, the dB floor and the GCP
count gate.

## Licence

Unlicense. See `LICENSE`.
