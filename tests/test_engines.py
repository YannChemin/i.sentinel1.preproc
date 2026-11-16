# MODULE:    tests/test_engines.py
# AUTHOR(S): Yann Chemin
# PURPOSE:   Compare the C+OpenMP, OpenCL and Python calibration engines.
# SPDX-License-Identifier: Unlicense
"""Regression tests over a synthetic 512 x 512 Sentinel-1 GRD scene.

The three engines must agree on the same input. They do not agree bit for bit
and are not expected to: the C and Python engines work in double precision and
the OpenCL kernel in single, so the comparison is a tolerance, stated here
rather than discovered later.

A synthetic scene is used rather than a real one because a real IW GRD
measurement band is about 800 MB and the arithmetic under test does not care:
it needs a uint16 raster with GCPs, a calibration LUT and a noise LUT whose
pixel grids differ, which is exactly what this builds. The LUT values span
four orders of magnitude and the DN field includes zeros and values the noise
subtraction drives to zero, so the mask logic and the division are both
exercised.

Run with::

    python3 -m pytest tests/test_engines.py -v
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import sentinel1_core as core

ADDON_DIR = Path(__file__).resolve().parent.parent

#: Block side of the synthetic scene. Large enough to cross a work-group
#: boundary on every device tested, small enough to run in a second.
SIDE = 512

#: Relative tolerance between the double-precision engines and the kernel.
#: Single precision carries about seven significant digits, and the
#: calibration is a ratio, so the error does not accumulate beyond that.
OPENCL_RTOL = 1e-5

#: Absolute tolerance, and the reason there has to be one.
#:
#: Gamma0 is ``max(DN^2 - noise, 0) / gamma^2``. At the noise floor the
#: subtraction cancels two nearly equal numbers, so a difference in the last
#: bits of the noise term becomes a large *relative* difference in a very small
#: result. That happens between any two implementations, not only between
#: precisions: the Python reference resamples the noise LUT onto the
#: calibration pixel grid, while the C engine interpolates on the native noise
#: grid, and the two differ in the last bits.
#:
#: The absolute size of the disagreement is what matters physically. Gamma0
#: over this scene runs from about 1e-3 to 10, so 1e-7 is four orders of
#: magnitude below anything the flood detection looks at, while still being
#: far too tight to hide a formula error: getting the noise term or the LUT
#: wrong moves values by percent, not by 1e-7.
ABS_TOL = 1e-7

CALIBRATION_XML = """<?xml version="1.0"?>
<calibration>
  <calibrationVectorList>
{vectors}
  </calibrationVectorList>
</calibration>
"""

CAL_VECTOR = """    <calibrationVector>
      <line>{line}</line>
      <pixel>{pixels}</pixel>
      <sigmaNought>{sigma}</sigmaNought>
      <betaNought>{beta}</betaNought>
      <gamma>{gamma}</gamma>
    </calibrationVector>"""

NOISE_XML = """<?xml version="1.0"?>
<noise>
  <noiseRangeVectorList>
{vectors}
  </noiseRangeVectorList>
  <noiseAzimuthVectorList>
    <noiseAzimuthVector>
      <swath>IW1</swath>
      <firstAzimuthLine>0</firstAzimuthLine>
      <lastAzimuthLine>{last_line}</lastAzimuthLine>
      <firstRangeSample>0</firstRangeSample>
      <lastRangeSample>{last_pixel}</lastRangeSample>
      <line>{az_lines}</line>
      <noiseAzimuthLut>{az_lut}</noiseAzimuthLut>
    </noiseAzimuthVector>
  </noiseAzimuthVectorList>
</noise>
"""

NOISE_VECTOR = """    <noiseRangeVector>
      <line>{line}</line>
      <pixel>{pixels}</pixel>
      <noiseRangeLut>{lut}</noiseRangeLut>
    </noiseRangeVector>"""


def _report_difference(label, a, b):
    """Print where two engines disagree most, so the number is never hidden.

    :returns: ``(max_abs, max_rel)`` over the pixels where ``b`` is non-zero.
    """
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    nonzero = b > 0.0
    rel = np.zeros_like(diff)
    rel[nonzero] = diff[nonzero] / b[nonzero]
    max_abs = float(diff.max())
    max_rel = float(rel.max())
    index = int(np.argmax(rel))
    print(
        f"{label}: max abs {max_abs:.3e}, max rel {max_rel:.3e} "
        f"at value {float(b.ravel()[index]):.3e} "
        f"({int(np.count_nonzero(diff > 0)):d} pixel(s) differ at all)"
    )
    return max_abs, max_rel


def _join(values, fmt="{:.6f}"):
    """Format a sequence the way the product XML does, space separated."""
    return " ".join(fmt.format(v) for v in values)


def _calibration_xml(lines, pixels):
    """Build a calibration XML whose gamma LUT varies in both directions."""
    vectors = []
    for i, line in enumerate(lines):
        # Values of the order a real gamma LUT has, rising across the swath.
        gamma = 300.0 + 120.0 * np.arange(len(pixels)) + 40.0 * i
        vectors.append(
            CAL_VECTOR.format(
                line=line,
                pixels=_join(pixels, "{:d}"),
                sigma=_join(gamma * 1.05),
                beta=_join(gamma * 0.5),
                gamma=_join(gamma),
            )
        )
    return CALIBRATION_XML.format(vectors="\n".join(vectors))


def _noise_xml(lines, pixels, last_line, last_pixel):
    """Build a noise XML on a different pixel grid from the calibration one."""
    vectors = []
    for i, line in enumerate(lines):
        # noiseRangeLut is power in DN squared, so these are large numbers.
        lut = 4.0e3 + 1.5e3 * np.arange(len(pixels)) + 500.0 * i
        vectors.append(
            NOISE_VECTOR.format(line=line, pixels=_join(pixels, "{:d}"), lut=_join(lut))
        )
    az_lines = [0, last_line // 2, last_line]
    return NOISE_XML.format(
        vectors="\n".join(vectors),
        last_line=last_line,
        last_pixel=last_pixel,
        az_lines=_join(az_lines, "{:d}"),
        az_lut=_join([1.0, 1.15, 0.95]),
    )


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    """Write a synthetic scene and return its paths and DN array.

    :returns: a dict with ``dir``, ``measurement``, ``calibration``,
        ``noise`` and ``dn``.
    """
    gdal = pytest.importorskip("osgeo.gdal", reason="GDAL Python bindings needed")
    from osgeo import gdal as gdal_mod
    from osgeo import osr

    gdal_mod.UseExceptions()
    work = tmp_path_factory.mktemp("s1scene")

    rng = np.random.default_rng(20261114)
    # A log-normal field spans the dynamic range of real backscatter; the
    # explicit zeros and the low tail make the mask logic do work.
    dn = (rng.lognormal(mean=5.6, sigma=0.9, size=(SIDE, SIDE))).astype(np.float64)
    dn[dn > 65535.0] = 65535.0
    dn = dn.astype(np.uint16)
    dn[::97, ::89] = 0
    dn[5:9, 5:9] = 30  # below the noise floor, so the mask must reject it

    measurement = work / "iw-vv.tiff"
    driver = gdal_mod.GetDriverByName("GTiff")
    dataset = driver.Create(str(measurement), SIDE, SIDE, 1, gdal_mod.GDT_UInt16)
    dataset.GetRasterBand(1).WriteArray(dn)

    # Real GRD measurement bands carry WGS84 GCPs and no geotransform; the
    # engine refuses a scene with fewer than six, so the test supplies nine.
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    gcps = []
    for r in (0, SIDE // 2, SIDE - 1):
        for c in (0, SIDE // 2, SIDE - 1):
            gcps.append(
                gdal_mod.GCP(55.3 + 0.001 * c, 25.4 - 0.001 * r, 0.0, c + 0.5, r + 0.5)
            )
    dataset.SetGCPs(gcps, srs.ExportToWkt())
    dataset.FlushCache()
    dataset = None

    cal_lines = [0, SIDE // 3, 2 * SIDE // 3, SIDE - 1]
    cal_pixels = list(range(0, SIDE, 64)) + [SIDE - 1]
    # Deliberately a different range grid from the calibration one: real
    # products differ (645 against 648 samples) and the engines must resample.
    noise_pixels = list(range(0, SIDE, 48)) + [SIDE - 1]

    calibration = work / "calibration-iw-vv.xml"
    noise = work / "noise-iw-vv.xml"
    calibration.write_text(_calibration_xml(cal_lines, cal_pixels))
    noise.write_text(_noise_xml(cal_lines, noise_pixels, SIDE - 1, SIDE - 1))

    return {
        "dir": work,
        "measurement": measurement,
        "calibration": calibration,
        "noise": noise,
        "dn": dn,
    }


def _binary():
    """Return the s1calibrate binary, or skip when it has not been built."""
    override = os.environ.get("S1CAL_BIN")
    if override and Path(override).exists():
        return Path(override)
    for candidate in (ADDON_DIR / "src" / "s1calibrate", ADDON_DIR / "s1calibrate"):
        if candidate.exists():
            return candidate
    found = shutil.which("s1calibrate")
    if found:
        return Path(found)
    pytest.skip("s1calibrate not built; run make -C src")


def _has_opencl(binary):
    """True when this binary was built with OpenCL and a device is present."""
    result = subprocess.run(
        [str(binary), "--opencl-list"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return False
    return "no OpenCL device" not in result.stdout and result.stdout.strip() != ""


def _run(binary, scene, stem, extra=()):
    """Run one engine and read back its calibrated raster and mask."""
    from osgeo import gdal as gdal_mod

    out = scene["dir"] / f"{stem}.tif"
    mask = scene["dir"] / f"{stem}_mask.tif"
    command = [
        str(binary),
        "--measurement",
        str(scene["measurement"]),
        "--calibration",
        str(scene["calibration"]),
        "--noise",
        str(scene["noise"]),
        "--output",
        str(out),
        "--mask",
        str(mask),
        "--lut",
        "gamma",
        "--threads",
        "4",
        "--memory-mb",
        "256",
        *extra,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, (
        f"{stem} failed: {result.returncode}\n{result.stdout}\n{result.stderr}"
    )
    linear = gdal_mod.Open(str(out)).ReadAsArray()
    valid = gdal_mod.Open(str(mask)).ReadAsArray()
    return linear, valid, result.stderr


def test_c_engine_matches_python_reference(scene):
    """The C engine must reproduce the Python reference in double precision."""
    binary = _binary()
    linear_c, valid_c, _ = _run(binary, scene, "cpu")

    calibration = core.parse_calibration(scene["calibration"].read_bytes())
    noise = core.parse_noise(scene["noise"].read_bytes())
    # interp_lut broadcasts a column of line coordinates against a row of
    # pixel coordinates, so both are one-dimensional, not a meshgrid.
    line_coords = np.arange(SIDE)
    pixel_coords = np.arange(SIDE)
    linear_py, valid_py = core.calibrate_block(
        scene["dn"], line_coords, pixel_coords, calibration, noise, "gamma", True
    )

    _report_difference("C vs Python", linear_c, linear_py)
    np.testing.assert_allclose(linear_c, linear_py, rtol=1e-6, atol=ABS_TOL)
    assert np.array_equal(valid_c, valid_py), (
        f"{int(np.count_nonzero(valid_c != valid_py))} mask pixel(s) differ"
    )


def test_opencl_engine_matches_c_engine(scene):
    """The OpenCL kernel must match the C engine to single-precision accuracy."""
    binary = _binary()
    if not _has_opencl(binary):
        pytest.skip("s1calibrate built without OpenCL, or no OpenCL device present")

    linear_c, valid_c, _ = _run(binary, scene, "cpu_for_ocl")
    linear_ocl, valid_ocl, log = _run(binary, scene, "ocl", extra=["--opencl"])

    assert "profile" in log, "the OpenCL engine must report the device profile"

    # Compare where the CPU engine found signal: a pixel the noise subtraction
    # drives to zero has no relative error to speak of.
    signal = linear_c > 0.0
    _report_difference("OpenCL vs C", linear_ocl, linear_c)
    np.testing.assert_allclose(
        linear_ocl[signal], linear_c[signal], rtol=OPENCL_RTOL, atol=ABS_TOL
    )

    # The mask may differ only where the signal sits within float rounding of
    # zero. Anything else is a logic difference, not a precision one.
    differing = np.flatnonzero((valid_c != valid_ocl).ravel())
    if differing.size:
        borderline = linear_c.ravel()[differing]
        assert np.all(borderline < OPENCL_RTOL * np.median(linear_c[signal])), (
            f"{differing.size} mask pixel(s) differ away from the zero crossing"
        )


def test_opencl_without_subtract_noise(scene):
    """With the noise term off, both engines must still agree."""
    binary = _binary()
    if not _has_opencl(binary):
        pytest.skip("s1calibrate built without OpenCL, or no OpenCL device present")

    linear_c, valid_c, _ = _run(
        binary, scene, "cpu_nonoise", extra=["--subtract-noise", "0"]
    )
    linear_ocl, valid_ocl, _ = _run(
        binary, scene, "ocl_nonoise", extra=["--opencl", "--subtract-noise", "0"]
    )
    signal = linear_c > 0.0
    # With no noise term there is no cancellation, so this comparison is the
    # strict one: it is purely float against double on the same formula.
    _report_difference("OpenCL vs C, no noise", linear_ocl, linear_c)
    np.testing.assert_allclose(
        linear_ocl[signal], linear_c[signal], rtol=OPENCL_RTOL, atol=0.0
    )
    # Without the noise term the mask is DN > 0 on both paths, so it is exact.
    assert np.array_equal(valid_c, valid_ocl)


def test_forced_profile_is_rejected_when_the_device_cannot_honour_it(scene):
    """A forced profile the device cannot provide must fail, not fall back."""
    binary = _binary()
    if not _has_opencl(binary):
        pytest.skip("s1calibrate built without OpenCL, or no OpenCL device present")

    listing = subprocess.run(
        [str(binary), "--opencl-list"], capture_output=True, text=True, check=False
    ).stdout
    gpu_index = None
    for row in listing.splitlines():
        if "profile igpu" in row or "profile dgpu" in row:
            gpu_index = row.split(":", 1)[0].strip()
            break
    if gpu_index is None:
        pytest.skip("no GPU device to force a CPU profile on")

    result = subprocess.run(
        [
            str(binary),
            "--measurement",
            str(scene["measurement"]),
            "--calibration",
            str(scene["calibration"]),
            "--noise",
            str(scene["noise"]),
            "--output",
            str(scene["dir"] / "forced.tif"),
            "--mask",
            str(scene["dir"] / "forced_mask.tif"),
            "--opencl",
            "--opencl-device",
            gpu_index,
            "--profile",
            "cpu",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "not a CPU device" in result.stderr


def test_profile_requires_opencl(scene):
    """--profile without --opencl is a mistake and must be reported as one."""
    binary = _binary()
    result = subprocess.run(
        [
            str(binary),
            "--measurement",
            str(scene["measurement"]),
            "--calibration",
            str(scene["calibration"]),
            "--noise",
            str(scene["noise"]),
            "--output",
            str(scene["dir"] / "noengine.tif"),
            "--mask",
            str(scene["dir"] / "noengine_mask.tif"),
            "--profile",
            "igpu",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "--profile applies to the OpenCL engine" in result.stderr
