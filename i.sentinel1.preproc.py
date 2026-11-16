#!/usr/bin/env python3
# %Module
# % description: Calibrates Sentinel-1 GRD to sigma0, beta0 or gamma0, removes thermal noise and geocodes it to the current GRASS region.
# % keyword: imagery
# % keyword: SAR
# % keyword: Sentinel-1
# % keyword: calibration
# % keyword: radar
# % keyword: STAC
# % keyword: Planetary Computer
# %end

# %option
# % key: input_dir
# % type: string
# % required: no
# % multiple: no
# % description: Local Sentinel-1 GRD .SAFE directory
# % guisection: Input
# %end

# %option
# % key: item
# % type: string
# % required: no
# % multiple: no
# % description: Planetary Computer STAC item id from sentinel-1-grd
# % guisection: Input
# %end

# %option
# % key: stac
# % type: string
# % required: no
# % multiple: no
# % answer: https://planetarycomputer.microsoft.com/api/stac/v1
# % description: STAC endpoint used with item=
# % guisection: Input
# %end

# %option
# % key: polarisations
# % type: string
# % required: no
# % multiple: yes
# % answer: vv,vh
# % description: Polarisation subset of vv,vh,hh,hv
# % guisection: Input
# %end

# %option
# % key: output
# % type: string
# % required: yes
# % multiple: no
# % description: Prefix for output raster names
# % guisection: Output
# %end

# %option
# % key: calibration
# % type: string
# % required: no
# % multiple: no
# % options: sigma0,beta0,gamma0
# % answer: gamma0
# % description: Calibration LUT to apply
# % guisection: Calibration
# %end

# %option
# % key: noise
# % type: string
# % required: no
# % multiple: no
# % options: yes,no
# % answer: yes
# % description: Subtract thermal noise
# % guisection: Calibration
# %end

# %option
# % key: border
# % type: integer
# % required: no
# % multiple: no
# % answer: 0
# % description: Set this many region-edge pixels to mask 0
# % guisection: Calibration
# %end

# %option
# % key: memory
# % type: integer
# % required: no
# % multiple: no
# % answer: 2048
# % description: Approximate memory budget for block processing in MB
# % guisection: Performance
# %end

# %option
# % key: nprocs
# % type: integer
# % required: no
# % multiple: no
# % answer: 4
# % description: Threads passed to GDAL warp
# % guisection: Performance
# %end

# %option
# % key: engine
# % type: string
# % required: no
# % multiple: no
# % answer: auto
# % options: auto,c,opencl,python
# % description: Calibration engine; auto prefers the C+OpenMP binary and falls back to the Python reference
# % guisection: Performance
# %end

# %option
# % key: profile
# % type: string
# % required: no
# % multiple: no
# % answer: auto
# % options: auto,cpu,igpu,dgpu,egpu
# % description: OpenCL device profile; auto detects it from the device
# % guisection: Performance
# %end

# %option
# % key: opencl_device
# % type: integer
# % required: no
# % multiple: no
# % description: OpenCL device index as listed by s1calibrate --opencl-list
# % guisection: Performance
# %end

# %option
# % key: resolution
# % type: double
# % required: no
# % multiple: no
# % description: Expected region resolution in metres; must match the current region
# % guisection: Output
# %end

# %option
# % key: metadata
# % type: string
# % required: no
# % multiple: no
# % description: Directory for the provenance JSON
# % guisection: Output
# %end

# %option
# % key: cache
# % type: string
# % required: no
# % multiple: no
# % label: Directory for the downloaded measurement and annotation files
# % description: Default: a temporary directory removed at the end of the run
# % guisection: Performance
# %end

# %flag
# % key: d
# % description: Also write a dB map next to the linear map
# % guisection: Output
# %end

# %flag
# % key: m
# % description: Write a validity mask (1 valid, 0 invalid)
# % guisection: Output
# %end

# %flag
# % key: k
# % description: Keep temporary warped files
# % guisection: Performance
# %end

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.windows import Window

import grass.script as gs

try:
    # Running from the source directory, which Python puts on sys.path itself.
    import sentinel1_core as core
except ImportError:
    # Installed: the module lives in $GISBASE/etc/i.sentinel1.preproc/ or in the
    # same directory under GRASS_ADDON_BASE, which set_path() locates.
    gs.utils.set_path(modulename="i.sentinel1.preproc")
    import sentinel1_core as core

MODULE = "i.sentinel1.preproc"
VERSION = "0.1.0"
COLLECTION = "sentinel-1-grd"
CALIBRATION_LUTS = {"sigma0": "sigma", "beta0": "beta", "gamma0": "gamma"}
VALID_POLARISATIONS = ("vv", "vh", "hh", "hv")


def _fatal(message):
    gs.fatal(f"{MODULE}: {message}")


def _region():
    parsed = gs.parse_command("g.region", flags="pg")
    return {
        "n": float(parsed["n"]),
        "s": float(parsed["s"]),
        "e": float(parsed["e"]),
        "w": float(parsed["w"]),
        "nsres": float(parsed["nsres"]),
        "ewres": float(parsed["ewres"]),
        "rows": int(parsed["rows"]),
        "cols": int(parsed["cols"]),
        "projection": parsed.get("projection"),
        "zone": parsed.get("zone"),
    }


def _region_wkt():
    return gs.read_command("g.proj", flags="w").strip()


def _overwrite():
    try:
        return bool(gs.overwrite())
    except AttributeError:
        return False


def _source_time_from_manifest(path):
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    match = re.search(r'startTime="([^"]+)"', text)
    if match is None:
        match = re.search(r"<safe:startTime>([^<]+)</safe:startTime>", text)
    if match is None:
        return None
    try:
        return datetime.fromisoformat(match.group(1).replace("Z", "+00:00"))
    except ValueError:
        return None


def _item_source(item_id, stac_url):
    import planetary_computer
    import pystac_client

    catalog = pystac_client.Client.open(stac_url, modifier=planetary_computer.sign_inplace)
    search = catalog.search(collections=[COLLECTION], ids=[item_id], max_items=1)
    items = list(search.items())
    if not items:
        _fatal(f"no {COLLECTION} item with id {item_id}")
    item = items[0]
    acquisition = item.datetime
    if acquisition is None:
        start = item.properties.get("start_datetime")
        if start:
            acquisition = datetime.fromisoformat(start.replace("Z", "+00:00"))
    return item, acquisition


def _safe_file(root, patterns):
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def _safe_source(root, polarisation):
    measurement = _safe_file(
        root,
        [
            f"measurement/*-{polarisation}-*.tiff",
            f"measurement/*-{polarisation}-*.tif",
            f"measurement/*{polarisation}*.tiff",
            f"measurement/*{polarisation}*.tif",
        ],
    )
    calibration = _safe_file(
        root,
        [
            f"annotation/calibration/calibration-*-{polarisation}-*.xml",
            f"annotation/calibration/calibration*{polarisation}*.xml",
        ],
    )
    noise = _safe_file(
        root,
        [
            f"annotation/calibration/noise-*-{polarisation}-*.xml",
            f"annotation/calibration/noise*{polarisation}*.xml",
        ],
    )
    if measurement is None:
        return None
    if calibration is None:
        _fatal(f"no calibration XML for {polarisation} in {root}")
    if noise is None:
        _fatal(f"no noise XML for {polarisation} in {root}")
    return measurement, calibration, noise, _source_time_from_manifest(root / "manifest.safe")


#: Pixels of margin added around the computed source window. The inverse fit
#: is good to a few pixels on this grid, the warp needs neighbours for bilinear
#: resampling, and a GRD band is cheap at this scale, so the margin is generous.
WINDOW_PAD = 512

#: Polynomial orders tried for the inverse fit, lowest first.
WINDOW_FIT_ORDERS = (2, 3, 4)

#: The fit is accepted while its own residual on the control points stays
#: below this share of the margin, so the margin always dominates the fit
#: error. A real IW GRD scene needs third order: a second-order fit over a
#: 250 km swath left 98 source pixels of residual, which at 512 px of margin
#: is safe but not comfortable, and at a smaller margin would not be.
WINDOW_RESIDUAL_FRACTION_OF_PAD = 0.25


def _source_window(measurement_path, region, pad=WINDOW_PAD):
    """Source line and pixel window covering the current computational region.

    An IW GRD band is about 250 km square; a city-scale region is a few tens of
    kilometres across, so roughly 4 % of the band carries the product and the
    rest is read, calibrated and thrown away by the warp. Over a 1 MiB/s link
    that difference is the whole feasibility of a multi-year archive: 665
    scenes is 612 GiB whole-band against 26 GiB windowed.

    The scene has no geotransform, only a grid of ground control points, so the
    window is found by fitting a second-order polynomial from longitude and
    latitude to source pixel and line over those points and evaluating it at
    the region's boundary, trying increasing polynomial orders and taking the
    lowest one whose own residual on the control points stays well inside the
    margin. A scene whose geometry none of them describes falls back to the
    whole band with a warning, rather than silently returning a window in the
    wrong place.

    :param measurement_path: The measurement band, local or ``/vsicurl/``.
    :param region: The current region, as :func:`_region` returns it.
    :returns: ``(x, y, width, height)``, or ``None`` for the whole band.
    """
    from rasterio.warp import transform as warp_transform

    with rasterio.open(measurement_path) as src:
        gcps, gcp_crs = src.gcps
        src_width, src_height = src.width, src.height
    if len(gcps) < 10:
        gs.warning("too few GCPs to fit a source window; reading the whole band")
        return None

    lon = np.array([g.x for g in gcps], dtype=float)
    lat = np.array([g.y for g in gcps], dtype=float)
    col = np.array([g.col for g in gcps], dtype=float)
    row = np.array([g.row for g in gcps], dtype=float)

    # Centred and scaled, so the normal equations stay well conditioned.
    lon0, lat0 = lon.mean(), lat.mean()
    lon_s = max(float(np.ptp(lon)), 1e-6)
    lat_s = max(float(np.ptp(lat)), 1e-6)

    def design(a, b, order):
        """Polynomial terms up to ``order`` in the scaled coordinates."""
        u = (a - lon0) / lon_s
        v = (b - lat0) / lat_s
        return np.column_stack(
            [u**i * v**j for d in range(order + 1) for i, j in
             ((d - k, k) for k in range(d + 1))]
        )

    # The lowest order that describes the geometry is used: a higher order
    # follows the control points more closely but is freer to misbehave between
    # them, and the margin has to cover whichever error is left.
    limit = pad * WINDOW_RESIDUAL_FRACTION_OF_PAD
    chosen = None
    for order in WINDOW_FIT_ORDERS:
        matrix = design(lon, lat, order)
        if matrix.shape[0] < matrix.shape[1] + 3:
            break  # Not enough control points to constrain this order.
        col_fit, *_ = np.linalg.lstsq(matrix, col, rcond=None)
        row_fit, *_ = np.linalg.lstsq(matrix, row, rcond=None)
        residual = max(
            float(np.abs(matrix @ col_fit - col).max()),
            float(np.abs(matrix @ row_fit - row).max()),
        )
        gs.verbose(
            f"inverse GCP fit order {order}: residual {residual:.1f} source pixels"
        )
        if residual <= limit:
            chosen = (order, col_fit, row_fit, residual)
            break
    if chosen is None:
        gs.warning(
            f"no inverse GCP fit up to order {WINDOW_FIT_ORDERS[-1]} stayed "
            f"within {limit:.0f} px of the {pad} px margin; reading the "
            "whole band"
        )
        return None
    order, col_fit, row_fit, residual = chosen
    gs.verbose(
        f"inverse GCP fit: order {order}, residual {residual:.1f} source pixels"
    )

    # The region boundary is sampled, not just cornered: in the scene's own
    # geometry a projected rectangle is a curved quadrilateral, so its extreme
    # line and pixel can fall on an edge rather than at a corner.
    steps = np.linspace(0.0, 1.0, 21)
    xs = region["w"] + (region["e"] - region["w"]) * steps
    ys = region["s"] + (region["n"] - region["s"]) * steps
    edge_x = np.concatenate([xs, xs, np.full_like(ys, region["w"]),
                             np.full_like(ys, region["e"])])
    edge_y = np.concatenate([np.full_like(xs, region["s"]),
                             np.full_like(xs, region["n"]), ys, ys])
    region_crs = CRS.from_wkt(_region_wkt())
    lon_edge, lat_edge = warp_transform(
        region_crs, gcp_crs, edge_x.tolist(), edge_y.tolist()
    )
    probe = design(np.asarray(lon_edge), np.asarray(lat_edge), order)
    cols = probe @ col_fit
    rows = probe @ row_fit

    x0 = int(np.floor(cols.min())) - pad
    y0 = int(np.floor(rows.min())) - pad
    x1 = int(np.ceil(cols.max())) + pad
    y1 = int(np.ceil(rows.max())) + pad
    x0 = max(0, min(x0, src_width - 1))
    y0 = max(0, min(y0, src_height - 1))
    x1 = max(x0 + 1, min(x1, src_width))
    y1 = max(y0 + 1, min(y1, src_height))
    if (x1 - x0) >= src_width and (y1 - y0) >= src_height:
        return None
    share = 100.0 * (x1 - x0) * (y1 - y0) / (src_width * src_height)
    gs.verbose(
        f"source window {x1 - x0}+{x0} x {y1 - y0}+{y0} "
        f"of {src_width} x {src_height} ({share:.1f} % of the band)"
    )
    return x0, y0, x1 - x0, y1 - y0


def _stage_asset(href, destination, label):
    """Download one remote asset to a local file, with retries.

    The measurement band is not read over ``/vsicurl/``. A Sentinel-1 GRD band
    is about 1 GB of tiled GeoTIFF, and reading it tile by tile over HTTPS took
    48 minutes on this AOI, by which time the Planetary Computer's signed URL
    had expired: Azure then answered the remaining range requests with partial
    bodies and GDAL failed with ``TIFFReadEncodedTile() failed``. A signed URL
    is valid for about an hour, so the only safe pattern is to fetch the whole
    asset sequentially, well inside that window, and calibrate from disk.

    A file already present with the expected size is kept, so ``cache=`` makes
    a re-run cheap and a failed run resumable at scene granularity.

    :param href: The signed asset URL.
    :param destination: Local path to write.
    :param label: Short name used in the progress messages.
    :returns: The local path.
    """
    import requests

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # A cached asset is settled with a HEAD, so a re-run does not make the
    # server start sending a gigabyte that is already on disk.
    if destination.is_file():
        try:
            head = requests.head(href, timeout=60, allow_redirects=True)
            size = int(head.headers.get("Content-Length") or 0)
            if size and destination.stat().st_size == size:
                gs.verbose(f"{label}: cached, {size} bytes")
                return destination
        except Exception as exc:  # noqa: BLE001 - the download below retries
            gs.verbose(f"{label}: cache check failed ({exc}); downloading")

    attempts = 4
    for attempt in range(1, attempts + 1):
        try:
            with requests.get(href, stream=True, timeout=120) as response:
                response.raise_for_status()
                expected = int(response.headers.get("Content-Length") or 0)
                # Written beside the destination and renamed, so an interrupted
                # download can never be mistaken for a complete asset.
                partial = destination.with_name(destination.name + ".part")
                received = 0
                last_percent = -1
                with open(partial, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=4 << 20):
                        handle.write(chunk)
                        received += len(chunk)
                        if expected:
                            percent = int(100 * received / expected)
                            if percent != last_percent:
                                # Progress is reported to MetaMonitor through
                                # the GRASS percent channel.
                                # https://github.com/YannChemin/MetaMonitor
                                gs.percent(received, expected, 2)
                                last_percent = percent
                if expected and received != expected:
                    raise OSError(
                        f"{label}: got {received} bytes, expected {expected}"
                    )
                partial.replace(destination)
                gs.percent(1, 1, 1)
                gs.verbose(f"{label}: {received} bytes")
                return destination
        except Exception as exc:  # noqa: BLE001 - retried, then fatal
            if attempt == attempts:
                _fatal(f"{label}: download failed after {attempts} attempts: {exc}")
            delay = 2**attempt
            gs.warning(
                f"{label}: attempt {attempt} of {attempts} failed ({exc}); "
                f"retrying in {delay} s"
            )
            time.sleep(delay)
    return destination


def _vsicurl(path):
    text = str(path)
    if text.startswith(("http://", "https://")):
        return "/vsicurl/" + text
    return text


def _block_side(memory_mb, width, height):
    pixels = max(256 * 256, int(memory_mb * 1024 * 1024 / 64))
    side = int(math.sqrt(pixels))
    return max(256, min(side, 2048, width, height))


def _gcp_dataset(path, width, height, dtype, nodata, gcps, gcp_crs):
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": dtype,
        "crs": None,
        "transform": Affine.identity(),
        "nodata": nodata,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "deflate",
        "bigtiff": "YES",
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
        dataset = rasterio.open(path, "w", **profile)
    dataset.gcps = (gcps, gcp_crs)
    return dataset


def _find_calibrator():
    if os.environ.get("S1CAL_DISABLE_C") == "1":
        return None
    candidates = [
        os.environ.get("S1CAL_BIN"),
        shutil.which("s1calibrate"),
        str(Path(__file__).resolve().parent / "s1calibrate"),
        str(Path(__file__).resolve().parent / "bin" / "s1calibrate"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _run_calibrator(
    binary,
    source_path,
    calibration_path,
    noise_path,
    output_path,
    mask_path,
    lut_name,
    subtract_noise,
    memory_mb,
    threads,
    opencl=None,
    window=None,
):
    command = [
        binary,
        "--measurement",
        _vsicurl(source_path),
        "--calibration",
        _vsicurl(calibration_path),
        "--noise",
        _vsicurl(noise_path),
        "--output",
        str(output_path),
        "--mask",
        str(mask_path),
        "--lut",
        lut_name,
        "--subtract-noise",
        "1" if subtract_noise else "0",
        "--memory-mb",
        str(memory_mb),
        "--threads",
        str(threads),
    ]
    if window:
        command += ["--window", ",".join(str(int(v)) for v in window)]
    if opencl:
        command.append("--opencl")
        if opencl.get("device") is not None:
            command += ["--opencl-device", str(opencl["device"])]
        if opencl.get("profile") and opencl["profile"] != "auto":
            command += ["--profile", opencl["profile"]]
        if opencl.get("timing"):
            command.append("--opencl-timing")
    gs.debug(" ".join(command))
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        _fatal(
            "s1calibrate failed:\n"
            + (completed.stderr or completed.stdout or "no diagnostic output")
        )
    # The engine reports its device and profile on stderr. Surface it, because
    # which device ran the arithmetic belongs in the run log.
    for line in (completed.stderr or "").splitlines():
        if line.startswith("s1calibrate: OpenCL"):
            gs.verbose(line)
    return completed.stderr or ""


def _write_calibrated(
    source_path,
    calibration_path,
    noise_path,
    output_path,
    mask_path,
    lut_name,
    subtract_noise,
    memory_mb,
    opencl=None,
    engine="auto",
    window=None,
):
    binary = None if engine == "python" else _find_calibrator()
    if engine in ("c", "opencl") and binary is None:
        # A forced engine that is not available is an error: falling back
        # silently would make a run unreproducible and the timings a lie.
        _fatal(
            f"engine={engine} requested but the s1calibrate binary was not "
            "found; build it with make -C src, or set S1CAL_BIN"
        )
    if binary is not None:
        threads = max(1, min(int(os.cpu_count() or 1), 8))
        log = _run_calibrator(
            binary,
            source_path,
            calibration_path,
            noise_path,
            output_path,
            mask_path,
            lut_name,
            subtract_noise,
            memory_mb,
            threads,
            opencl,
            window,
        )
        with rasterio.open(output_path) as calibrated, rasterio.open(mask_path) as mask:
            gcps, gcp_crs = calibrated.gcps
            core.validate_gcp_count(len(gcps))
            width, height = calibrated.width, calibrated.height
        return {
            "gcp_count": len(gcps),
            "width": width,
            "height": height,
            "block_size": _block_side(memory_mb, width, height),
            "engine": "opencl" if opencl else "c-openmp",
            "threads": threads,
            # The device line is kept verbatim, so a product can be traced to
            # the device that produced it. The OpenCL kernel works in single
            # precision and the C engine in double: they agree to about 6e-8
            # relative over the signal range, and disagree only at the noise
            # floor, where the DN-squared minus noise subtraction cancels and
            # gamma0 is physically meaningless. See tests/test_engines.py.
            "device": next(
                (
                    line.split("OpenCL ", 1)[1]
                    for line in log.splitlines()
                    if line.startswith("s1calibrate: OpenCL ")
                    and "timing" not in line
                ),
                None,
            )
            if opencl
            else None,
            "precision": "float32" if opencl else "float64",
            "source_window": list(window) if window else None,
        }

    calibration = core.parse_calibration(calibration_path)
    if subtract_noise:
        noise = core.parse_noise(noise_path)
    else:
        noise = core.Noise(
            range_lines=np.array([0], dtype=np.int64),
            range_pixels=np.array([0], dtype=np.int64),
            range_lut=np.zeros((1, 1), dtype=np.float64),
            azimuth=(),
        )

    with rasterio.open(_vsicurl(source_path)) as source:
        gcps, gcp_crs = source.gcps
        core.validate_gcp_count(len(gcps))
        if gcp_crs is None:
            gcp_crs = CRS.from_epsg(4326)
        # The same window as the compiled engines, so the three stay comparable.
        # The LUTs are indexed by absolute source line and pixel, so the window
        # only shifts the coordinates handed to the block; the GCPs shift with
        # it, because they are expressed in source pixel coordinates.
        off_x, off_y = (int(window[0]), int(window[1])) if window else (0, 0)
        width = int(window[2]) if window else source.width
        height = int(window[3]) if window else source.height
        shifted = [
            type(g)(
                row=g.row - off_y,
                col=g.col - off_x,
                x=g.x,
                y=g.y,
                z=g.z,
                id=g.id,
                info=g.info,
            )
            for g in gcps
        ]
        side = _block_side(memory_mb, width, height)
        calibrated = _gcp_dataset(
            output_path, width, height, "float32", float("nan"), shifted, gcp_crs
        )
        mask = _gcp_dataset(
            mask_path, width, height, "uint8", 255, shifted, gcp_crs
        )
        try:
            for row in range(0, height, side):
                rows = min(side, height - row)
                for column in range(0, width, side):
                    columns = min(side, width - column)
                    source_window = Window(off_x + column, off_y + row, columns, rows)
                    target_window = Window(column, row, columns, rows)
                    dn = source.read(1, window=source_window)
                    line_coords = np.arange(
                        off_y + row, off_y + row + rows, dtype=np.float64
                    )[:, None]
                    pixel_coords = np.arange(
                        off_x + column, off_x + column + columns, dtype=np.float64
                    )[None, :]
                    linear, valid = core.calibrate_block(
                        dn,
                        line_coords,
                        pixel_coords,
                        calibration,
                        noise,
                        lut_name=lut_name,
                        subtract_noise=subtract_noise,
                    )
                    calibrated.write(linear, 1, window=target_window)
                    mask.write(valid, 1, window=target_window)
        finally:
            calibrated.close()
            mask.close()
    return {
        "gcp_count": len(gcps),
        "width": width,
        "height": height,
        "block_size": side,
        "engine": "python",
        "source_window": list(window) if window else None,
    }


def _warp(source_path, target_path, region, resample, nprocs, dtype, nodata):
    """Warp one calibrated raster onto the current region.

    ``nodata`` must be a value the calibration cannot produce. It used to be
    hard-coded to 0 for both rasters, which silently destroyed the validity
    mask: the engines write 0 for an invalid pixel, so every one of them became
    nodata and the delivered mask could only ever contain 1. The float raster
    now uses NaN and the mask 255.

    :param nodata: Source and destination nodata value.
    """
    predictor = "3" if dtype == "float32" else "1"
    command = [
        "gdalwarp",
        "-overwrite",
        "-s_srs",
        "EPSG:4326",
        "-t_srs",
        _region_wkt(),
        "-te",
        str(region["w"]),
        str(region["s"]),
        str(region["e"]),
        str(region["n"]),
        "-tr",
        str(region["ewres"]),
        str(region["nsres"]),
        "-r",
        resample,
        "-order",
        "1",
        "-srcnodata",
        str(nodata),
        "-dstnodata",
        str(nodata),
        "-multi",
        "-wo",
        f"NUM_THREADS={nprocs}",
        "-of",
        "GTiff",
        "-co",
        "COMPRESS=DEFLATE",
        "-co",
        "TILED=YES",
        "-co",
        "BIGTIFF=YES",
        "-co",
        f"PREDICTOR={predictor}",
        str(source_path),
        str(target_path),
    ]
    gs.debug(" ".join(command))
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        _fatal(
            "gdalwarp failed:\n"
            + (completed.stderr.strip() or completed.stdout.strip())
        )


def _import_raster(path, name):
    gs.run_command(
        "r.in.gdal",
        input=str(path),
        output=name,
        flags="o",
        overwrite=_overwrite(),
    )


def _stats(name):
    parsed = gs.parse_command("r.univar", map=name, flags="g")
    out = {}
    for key in ("min", "max", "mean", "stddev", "median", "n", "null_cells"):
        if key in parsed:
            try:
                out[key] = float(parsed[key])
            except ValueError:
                out[key] = parsed[key]
    return out


def _timestamp(name, acquisition):
    if acquisition is None:
        return
    gs.run_command("r.timestamp", map=name, date=acquisition.strftime("%d %b %Y %H:%M:%S"))


def _border_mask(name, border, region):
    if border <= 0:
        return
    rows = region["rows"]
    columns = region["cols"]
    expression = (
        f"{name} = if(row() <= {border} || row() > {rows} - {border} || "
        f"col() <= {border} || col() > {columns} - {border}, 0, {name})"
    )
    gs.run_command("r.mapcalc", expression=expression, overwrite=_overwrite())


def _metadata_path(metadata_option, first_map):
    if metadata_option:
        directory = Path(metadata_option).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{first_map}_metadata.json"
    gisenv = gs.gisenv()
    directory = (
        Path(gisenv["GISDBASE"])
        / gisenv["LOCATION_NAME"]
        / gisenv["MAPSET"]
        / "cell_misc"
        / first_map
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "description.json"


def _process_polarisation(
    polarisation,
    measurement,
    calibration_path,
    noise_path,
    acquisition,
    source_record,
    region,
    options,
    flags,
    tempdir,
):
    lut_name = CALIBRATION_LUTS[options["calibration"]]
    subtract_noise = options["noise"] == "yes"
    base = f"{options['output']}_{polarisation}_{options['calibration']}"
    linear_map = f"{base}_linear"
    db_map = f"{base}_dB"
    mask_map = f"{base}_mask"

    calibrated_path = tempdir / f"{polarisation}_calibrated.tif"
    mask_source = tempdir / f"{polarisation}_mask.tif"
    warped_path = tempdir / f"{polarisation}_calibrated_warped.tif"
    warped_mask = tempdir / f"{polarisation}_mask_warped.tif"

    # The window is computed from the scene's own GCPs against the current
    # region, so only the part of the swath the region needs is read.
    window = _source_window(_vsicurl(measurement), region)

    gs.message(f"Calibrating {polarisation} with {options['calibration']}")
    geometry = _write_calibrated(
        measurement,
        calibration_path,
        noise_path,
        calibrated_path,
        mask_source,
        lut_name,
        subtract_noise,
        int(options["memory"]),
        opencl=(
            {
                "device": (
                    int(options["opencl_device"])
                    if options.get("opencl_device")
                    else None
                ),
                "profile": options.get("profile") or "auto",
                "timing": os.environ.get("S1CAL_OPENCL_PROFILE") is not None,
            }
            if options.get("engine") == "opencl"
            else None
        ),
        engine=options.get("engine") or "auto",
        window=window,
    )

    _warp(
        calibrated_path,
        warped_path,
        region,
        "bilinear",
        int(options["nprocs"]),
        "float32",
        "nan",
    )
    _import_raster(warped_path, linear_map)

    maps = {"linear": linear_map}
    if flags["d"]:
        expression = (
            f"{db_map} = if({linear_map} > 0, "
            f"4.342944819032518 * log({linear_map}), {core.DB_FLOOR})"
        )
        gs.run_command("r.mapcalc", expression=expression, overwrite=_overwrite())
        maps["db"] = db_map

    if flags["m"]:
        _warp(
            mask_source,
            warped_mask,
            region,
            "near",
            int(options["nprocs"]),
            "uint8",
            "255",
        )
        _import_raster(warped_mask, mask_map)
        _border_mask(mask_map, int(options["border"]), region)
        maps["mask"] = mask_map

    for name in maps.values():
        _timestamp(name, acquisition)

    record = {
        "polarisation": polarisation,
        "source": source_record,
        "calibration": options["calibration"],
        "terrain": "ellipsoid",
        "noise_subtracted": subtract_noise,
        "border_trim": int(options["border"]),
        "geometry": geometry,
        "maps": maps,
        "stats": {key: _stats(name) for key, name in maps.items()},
    }
    gs.message(f"Wrote {linear_map}" + (f" and {db_map}" if flags["d"] else ""))
    return record


def main():
    raw_polarisations = options["polarisations"]
    if isinstance(raw_polarisations, str):
        raw_polarisations = raw_polarisations.split(",")
    options["polarisations"] = [
        str(value).strip().lower()
        for value in raw_polarisations
        if str(value).strip()
    ]
    for polarisation in options["polarisations"]:
        if polarisation not in VALID_POLARISATIONS:
            _fatal(f"unsupported polarisation {polarisation}")

    if bool(options["input_dir"]) == bool(options["item"]):
        _fatal("give exactly one of input_dir= or item=")
    if int(options["border"]) < 0:
        _fatal("border= must be zero or positive")
    if int(options["memory"]) < 64:
        _fatal("memory= must be at least 64 MB")
    if int(options["nprocs"]) < 1:
        _fatal("nprocs= must be at least 1")

    region = _region()
    if options["resolution"]:
        resolution = float(options["resolution"])
        if not (
            math.isclose(resolution, region["ewres"], rel_tol=1e-6, abs_tol=1e-6)
            and math.isclose(
                resolution, region["nsres"], rel_tol=1e-6, abs_tol=1e-6
            )
        ):
            _fatal(
                "resolution= must match the current region; set the region with "
                "g.region res=..."
            )

    tempdir = Path(tempfile.mkdtemp(prefix=f"{MODULE}."))
    keep = bool(flags["k"])
    records = []
    acquisition = None

    try:
        if options["item"]:
            item, acquisition = _item_source(options["item"], options["stac"])
            source_mode = {
                "mode": "item",
                "collection": COLLECTION,
                "item": item.id,
                "stac": options["stac"],
            }
            staging = (
                Path(options["cache"]).expanduser() / item.id
                if options["cache"]
                else tempdir
            )
            for polarisation in options["polarisations"]:
                measurement = item.assets.get(polarisation)
                calibration_asset = item.assets.get(f"schema-calibration-{polarisation}")
                noise_asset = item.assets.get(f"schema-noise-{polarisation}")
                if measurement is None:
                    gs.warning(f"item has no {polarisation} asset; skipping")
                    continue
                if calibration_asset is None or noise_asset is None:
                    _fatal(f"item lacks calibration/noise XML for {polarisation}")
                # Without cache= the band is read over /vsicurl/, restricted
                # to the window the region needs: a few per cent of the swath,
                # which finishes well inside the signed URL's lifetime. With
                # cache= the whole band is fetched to disk instead, which is
                # what repeated runs over one scene, or an unreliable link,
                # want. The annotations are read through GDAL's virtual file
                # layer either way; they are about a megabyte each.
                if options["cache"]:
                    gs.message(f"Fetching {polarisation} assets")
                    source_measurement = str(
                        _stage_asset(
                            measurement.href,
                            staging / f"{polarisation}.tiff",
                            f"{polarisation} measurement",
                        )
                    )
                    source_calibration = str(
                        _stage_asset(
                            calibration_asset.href,
                            staging / f"calibration-{polarisation}.xml",
                            f"{polarisation} calibration",
                        )
                    )
                    source_noise = str(
                        _stage_asset(
                            noise_asset.href,
                            staging / f"noise-{polarisation}.xml",
                            f"{polarisation} noise",
                        )
                    )
                else:
                    source_measurement = measurement.href
                    source_calibration = calibration_asset.href
                    source_noise = noise_asset.href
                source_record = dict(source_mode)
                source_record["measurement_asset"] = polarisation
                source_record["staged_locally"] = bool(options["cache"])
                source_record["cache"] = (
                    str(staging) if options["cache"] else None
                )
                records.append(
                    _process_polarisation(
                        polarisation,
                        source_measurement,
                        source_calibration,
                        source_noise,
                        acquisition,
                        source_record,
                        region,
                        options,
                        flags,
                        tempdir,
                    )
                )
        else:
            root = Path(options["input_dir"]).expanduser()
            if not root.is_dir():
                _fatal(f"input_dir does not exist: {root}")
            source_mode = {"mode": "input_dir", "path": str(root.resolve())}
            for polarisation in options["polarisations"]:
                source = _safe_source(root, polarisation)
                if source is None:
                    gs.warning(f"no measurement for {polarisation} in {root}; skipping")
                    continue
                measurement, calibration_path, noise_path, acquisition = source
                source_record = dict(source_mode)
                source_record["measurement_file"] = str(measurement)
                records.append(
                    _process_polarisation(
                        polarisation,
                        measurement,
                        calibration_path,
                        noise_path,
                        acquisition,
                        source_record,
                        region,
                        options,
                        flags,
                        tempdir,
                    )
                )
    finally:
        if keep:
            gs.message(f"Temporary files kept in {tempdir}")
        else:
            shutil.rmtree(tempdir, ignore_errors=True)

    if not records:
        _fatal("no polarisation was processed")

    all_maps = []
    for record in records:
        all_maps.extend(record["maps"].values())
    group = f"{options['output']}_group"
    gs.run_command("i.group", group=group, input=all_maps, overwrite=_overwrite())

    metadata = {
        "module": MODULE,
        "version": VERSION,
        "created": datetime.now().astimezone().isoformat(),
        "region": region,
        "group": group,
        "records": records,
    }
    path = _metadata_path(options["metadata"], records[0]["maps"]["linear"])
    path.write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    gs.message(f"Provenance: {path}")
    return 0


if __name__ == "__main__":
    options, flags = gs.parser()
    sys.exit(main())
