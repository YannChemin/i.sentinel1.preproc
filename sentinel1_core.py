"""Numeric core for Sentinel-1 GRD calibration and thermal-noise removal.

This module has no GRASS dependency so it can be unit-tested with a normal
Python interpreter.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

DB_FLOOR = -40.0
POWER_FLOOR = 10.0 ** (DB_FLOOR / 10.0)


@dataclass(frozen=True)
class Calibration:
    lines: np.ndarray
    pixels: np.ndarray
    sigma: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray

    def lut(self, name: str) -> np.ndarray:
        if name not in {"sigma", "beta", "gamma"}:
            raise ValueError(f"unknown calibration LUT: {name}")
        return getattr(self, name)


@dataclass(frozen=True)
class AzimuthVector:
    swath: str
    first_line: int
    last_line: int
    first_pixel: int
    last_pixel: int
    lines: np.ndarray
    lut: np.ndarray


@dataclass(frozen=True)
class Noise:
    range_lines: np.ndarray
    range_pixels: np.ndarray
    range_lut: np.ndarray
    azimuth: tuple[AzimuthVector, ...]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if _local(child.tag) == name]


def _child(element: ET.Element, name: str) -> ET.Element:
    found = _children(element, name)
    if not found:
        raise ValueError(f"missing <{name}> in {_local(element.tag)}")
    return found[0]


def _child_text(element: ET.Element, name: str) -> str:
    text = _child(element, name).text
    if text is None or not text.strip():
        raise ValueError(f"empty <{name}> in {_local(element.tag)}")
    return text.strip()


def _ints(text: str) -> np.ndarray:
    return np.array([int(value) for value in text.split()], dtype=np.int64)


def _floats(text: str) -> np.ndarray:
    return np.array([float(value) for value in text.split()], dtype=np.float64)


def _read_source(source: str | Path | bytes) -> bytes:
    if isinstance(source, bytes):
        return source
    text = str(source)
    if text.startswith(("http://", "https://")):
        import requests

        response = requests.get(text, timeout=120)
        response.raise_for_status()
        return response.content
    return Path(text).read_bytes()


def _normalise_grid(
    lines: Iterable[int],
    pixels: Iterable[int],
    lut: Iterable[float],
    *,
    name: str,
    allow_zero: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lines_array = np.asarray(lines, dtype=np.int64).ravel()
    pixels_array = np.asarray(pixels, dtype=np.int64).ravel()
    lut_array = np.asarray(lut, dtype=np.float64).ravel()

    if lines_array.size == 0:
        raise ValueError(f"{name}: no line values")
    if pixels_array.size == 0:
        raise ValueError(f"{name}: no pixel values")
    if lut_array.size != pixels_array.size:
        raise ValueError(
            f"{name}: {lut_array.size} LUT values for {pixels_array.size} pixels"
        )
    if np.any(np.diff(lines_array) <= 0):
        raise ValueError(f"{name}: line values are not strictly increasing")
    if np.any(np.diff(pixels_array) <= 0):
        raise ValueError(f"{name}: pixel values are not strictly increasing")
    if not np.all(np.isfinite(lut_array)):
        raise ValueError(f"{name}: non-finite LUT value")
    if allow_zero:
        if np.any(lut_array < 0):
            raise ValueError(f"{name}: negative LUT value")
    elif np.any(lut_array <= 0):
        raise ValueError(f"{name}: non-positive LUT value")

    return lines_array, pixels_array, lut_array


def _resample_lut(
    pixels: np.ndarray, lut: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    if np.array_equal(pixels, reference):
        return lut
    return np.interp(reference, pixels, lut)


def _parse_calibration_root(root: ET.Element) -> Calibration:
    vectors = [e for e in root.iter() if _local(e.tag) == "calibrationVector"]
    if not vectors:
        raise ValueError("calibration XML has no calibrationVector")

    rows: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    reference_pixels: np.ndarray | None = None

    for vector in vectors:
        line = _ints(_child_text(vector, "line"))
        if line.size != 1:
            raise ValueError("calibrationVector must have exactly one line value")
        pixels = _ints(_child_text(vector, "pixel"))
        sigma = _floats(_child_text(vector, "sigmaNought"))
        beta = _floats(_child_text(vector, "betaNought"))
        gamma = _floats(_child_text(vector, "gamma"))

        _, checked_pixels, checked_sigma = _normalise_grid(
            line, pixels, sigma, name="sigmaNought", allow_zero=False
        )
        _, _, checked_beta = _normalise_grid(
            line, pixels, beta, name="betaNought", allow_zero=False
        )
        _, _, checked_gamma = _normalise_grid(
            line, pixels, gamma, name="gamma", allow_zero=False
        )
        if reference_pixels is None:
            reference_pixels = checked_pixels
        else:
            checked_sigma = _resample_lut(
                checked_pixels, checked_sigma, reference_pixels
            )
            checked_beta = _resample_lut(checked_pixels, checked_beta, reference_pixels)
            checked_gamma = _resample_lut(
                checked_pixels, checked_gamma, reference_pixels
            )

        line_value = int(line[0])
        if line_value in rows:
            raise ValueError("calibration XML has duplicate line values")
        rows[line_value] = (checked_pixels, checked_sigma, checked_beta, checked_gamma)

    lines = np.array(sorted(rows), dtype=np.int64)
    pixels = rows[int(lines[0])][0]
    sigma = np.vstack([rows[int(line)][1] for line in lines])
    beta = np.vstack([rows[int(line)][2] for line in lines])
    gamma = np.vstack([rows[int(line)][3] for line in lines])
    return Calibration(lines=lines, pixels=pixels, sigma=sigma, beta=beta, gamma=gamma)


#: Range-noise vector element, and its pre-IPF-2.90 name. Products acquired
#: before March 2018 call the element "noiseVector" and its LUT "noiseLut",
#: with the same children; they carry no azimuth vectors at all, which needs
#: no special case because the azimuth factor defaults to 1.
RANGE_VECTOR_TAGS = ("noiseRangeVector", "noiseVector")
RANGE_LUT_TAGS = ("noiseRangeLut", "noiseLut")


def _first_child_text(element: ET.Element, names: tuple[str, ...]) -> str:
    """Text of the first child matching any of several element names.

    :param names: Candidate local names, in order of preference.
    :raises ValueError: If the element has none of them.
    """
    for name in names:
        for child in element:
            if _local(child.tag) == name:
                return child.text or ""
    raise ValueError(f"element has none of {', '.join(names)}")


def _parse_noise_root(root: ET.Element) -> Noise:
    range_vectors = [
        e for e in root.iter() if _local(e.tag) in RANGE_VECTOR_TAGS
    ]
    if not range_vectors:
        raise ValueError(
            "noise XML has neither noiseRangeVector (IPF 2.90 and later) "
            "nor noiseVector (earlier products)"
        )

    rows: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    reference_pixels: np.ndarray | None = None

    for vector in range_vectors:
        line = _ints(_child_text(vector, "line"))
        if line.size != 1:
            raise ValueError(
                "a noise range vector must have exactly one line value"
            )
        pixels = _ints(_child_text(vector, "pixel"))
        lut = _floats(_first_child_text(vector, RANGE_LUT_TAGS))
        _, checked_pixels, checked_lut = _normalise_grid(
            line, pixels, lut, name="noise range LUT", allow_zero=True
        )
        if reference_pixels is None:
            reference_pixels = checked_pixels
        else:
            checked_lut = _resample_lut(checked_pixels, checked_lut, reference_pixels)
        line_value = int(line[0])
        if line_value in rows:
            raise ValueError("noise XML has duplicate line values")
        rows[line_value] = (checked_pixels, checked_lut)

    lines = np.array(sorted(rows), dtype=np.int64)
    pixels = rows[int(lines[0])][0]
    range_lut = np.vstack([rows[int(line)][1] for line in lines])

    azimuth: list[AzimuthVector] = []
    for vector in [e for e in root.iter() if _local(e.tag) == "noiseAzimuthVector"]:
        line_values = _ints(_child_text(vector, "line"))
        lut_values = _floats(_child_text(vector, "noiseAzimuthLut"))
        if line_values.size != lut_values.size:
            raise ValueError(
                "noiseAzimuthVector: line and noiseAzimuthLut lengths differ"
            )
        if np.any(np.diff(line_values) < 0):
            raise ValueError("noiseAzimuthVector: line values decrease")
        if not np.all(np.isfinite(lut_values)) or np.any(lut_values <= 0):
            raise ValueError("noiseAzimuthVector: invalid LUT value")
        azimuth.append(
            AzimuthVector(
                swath=_child_text(vector, "swath"),
                first_line=int(_child_text(vector, "firstAzimuthLine")),
                last_line=int(_child_text(vector, "lastAzimuthLine")),
                first_pixel=int(_child_text(vector, "firstRangeSample")),
                last_pixel=int(_child_text(vector, "lastRangeSample")),
                lines=line_values,
                lut=lut_values,
            )
        )

    return Noise(
        range_lines=lines,
        range_pixels=pixels,
        range_lut=range_lut,
        azimuth=tuple(azimuth),
    )


def parse_calibration(source: str | Path | bytes) -> Calibration:
    root = ET.fromstring(_read_source(source))
    return _parse_calibration_root(root)


def parse_noise(source: str | Path | bytes) -> Noise:
    root = ET.fromstring(_read_source(source))
    return _parse_noise_root(root)


def interp_lut(
    lut: np.ndarray,
    lines: np.ndarray,
    pixels: np.ndarray,
    line_coords: np.ndarray,
    pixel_coords: np.ndarray,
) -> np.ndarray:
    lut = np.asarray(lut, dtype=np.float64)
    lines = np.asarray(lines, dtype=np.float64).ravel()
    pixels = np.asarray(pixels, dtype=np.float64)
    line_coords = np.asarray(line_coords, dtype=np.float64).reshape(-1, 1)
    pixel_coords = np.asarray(pixel_coords, dtype=np.float64).reshape(1, -1)

    if lut.ndim == 1:
        lut = lut[None, :]
    if pixels.ndim == 1:
        if lut.shape[0] == 1:
            pixels = pixels[None, :]
        else:
            pixels = np.broadcast_to(pixels, lut.shape)
    if lut.shape != pixels.shape:
        raise ValueError("LUT and pixel grid shapes differ")
    if lut.shape[0] != lines.size:
        raise ValueError("LUT row count and line count differ")

    if lines.size == 1:
        row = np.interp(pixel_coords[0], pixels[0], lut[0])
        return np.broadcast_to(row, (line_coords.shape[0], pixel_coords.shape[1])).copy()

    lower = np.searchsorted(lines, line_coords, side="right") - 1
    lower = np.clip(lower, 0, lines.size - 2)
    upper = lower + 1
    fraction = (line_coords - lines[lower]) / (lines[upper] - lines[lower])
    fraction = np.clip(fraction, 0.0, 1.0)

    used = np.unique(np.concatenate([lower.ravel(), upper.ravel()]))
    interpolated_rows = {
        int(index): np.interp(pixel_coords[0], pixels[int(index)], lut[int(index)])
        for index in used
    }

    out = np.empty((line_coords.shape[0], pixel_coords.shape[1]), dtype=np.float64)
    for row in range(out.shape[0]):
        low = interpolated_rows[int(lower[row, 0])]
        high = interpolated_rows[int(upper[row, 0])]
        out[row] = low + (high - low) * fraction[row, 0]
    return out


def azimuth_lut(
    noise: Noise, line_coords: np.ndarray, pixel_coords: np.ndarray
) -> np.ndarray:
    line_coords = np.asarray(line_coords, dtype=np.float64).reshape(-1, 1)
    pixel_coords = np.asarray(pixel_coords, dtype=np.float64).reshape(1, -1)
    out = np.ones((line_coords.shape[0], pixel_coords.shape[1]), dtype=np.float64)

    for vector in noise.azimuth:
        rows = np.where(
            (line_coords[:, 0] >= vector.first_line)
            & (line_coords[:, 0] <= vector.last_line)
        )[0]
        columns = np.where(
            (pixel_coords[0] >= vector.first_pixel)
            & (pixel_coords[0] <= vector.last_pixel)
        )[0]
        if rows.size == 0 or columns.size == 0:
            continue
        values = np.interp(
            line_coords[rows, 0], vector.lines.astype(np.float64), vector.lut
        )
        out[np.ix_(rows, columns)] = values[:, None]

    return out


def calibrate_block(
    dn: np.ndarray,
    line_coords: np.ndarray,
    pixel_coords: np.ndarray,
    calibration: Calibration,
    noise: Noise,
    lut_name: str = "gamma",
    subtract_noise: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    lut = interp_lut(
        calibration.lut(lut_name),
        calibration.lines,
        calibration.pixels,
        line_coords,
        pixel_coords,
    )
    dn = np.asarray(dn, dtype=np.float64)
    if dn.shape != lut.shape:
        raise ValueError("DN block and LUT block shapes differ")

    dn_squared = dn * dn
    if subtract_noise:
        noise_range = interp_lut(
            noise.range_lut,
            noise.range_lines,
            noise.range_pixels,
            line_coords,
            pixel_coords,
        )
        noise_power = np.maximum(noise_range * azimuth_lut(noise, line_coords, pixel_coords), 0.0)
        signal_squared = np.maximum(dn_squared - noise_power, 0.0)
        valid = (dn > 0.0) & (signal_squared > 0.0)
    else:
        signal_squared = dn_squared
        valid = dn > 0.0

    linear = signal_squared / (lut * lut)
    linear = np.where(np.isfinite(linear), linear, 0.0)
    linear = np.maximum(linear, 0.0)
    valid &= np.isfinite(linear)
    # DN 0 is outside the swath, not a measurement of zero backscatter, so it
    # is no data. A pixel the noise subtraction drove to zero keeps a real 0
    # and reaches the dB floor; conflating the two made the validity mask
    # unable to report anything but 1, because the warp treated 0 as nodata.
    linear = np.where(dn > 0.0, linear, np.nan)

    return linear.astype(np.float32), valid.astype(np.uint8)


def to_db(linear: np.ndarray, valid: np.ndarray, floor_db: float = DB_FLOOR) -> np.ndarray:
    linear = np.asarray(linear, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    floor = 10.0 ** (floor_db / 10.0)
    out = np.full(linear.shape, floor, dtype=np.float64)
    positive = valid & (linear > 0.0)
    out[positive] = linear[positive]
    out = 10.0 * np.log10(out)
    return np.maximum(out, floor_db).astype(np.float32)


def validate_gcp_count(count: int, minimum: int = 6) -> None:
    if count < minimum:
        raise ValueError(
            f"source has {count} GCPs; at least {minimum} are needed for a "
            "first-order warp"
        )
