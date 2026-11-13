import numpy as np
import pytest

import sentinel1_core as core


CALIBRATION_XML = b"""<?xml version="1.0"?>
<calibration>
  <calibrationVectorList>
    <calibrationVector>
      <line>0</line>
      <pixel>0 10 20</pixel>
      <sigmaNought>100 200 400</sigmaNought>
      <betaNought>50 100 200</betaNought>
      <gamma>100 200 400</gamma>
    </calibrationVector>
    <calibrationVector>
      <line>30</line>
      <pixel>0 10 20</pixel>
      <sigmaNought>200 400 800</sigmaNought>
      <betaNought>100 200 400</betaNought>
      <gamma>200 400 800</gamma>
    </calibrationVector>
  </calibrationVectorList>
</calibration>
"""

NOISE_XML = b"""<?xml version="1.0"?>
<noise>
  <noiseRangeVectorList>
    <noiseRangeVector>
      <line>0</line>
      <pixel>0 10 20</pixel>
      <noiseRangeLut>0 100 400</noiseRangeLut>
    </noiseRangeVector>
    <noiseRangeVector>
      <line>30</line>
      <pixel>0 10 20</pixel>
      <noiseRangeLut>0 400 1600</noiseRangeLut>
    </noiseRangeVector>
  </noiseRangeVectorList>
  <noiseAzimuthVectorList>
    <noiseAzimuthVector>
      <swath>IW1</swath>
      <firstAzimuthLine>0</firstAzimuthLine>
      <lastAzimuthLine>30</lastAzimuthLine>
      <firstRangeSample>0</firstRangeSample>
      <lastRangeSample>20</lastRangeSample>
      <line>0 30</line>
      <noiseAzimuthLut>1.0 1.2</noiseAzimuthLut>
    </noiseAzimuthVector>
  </noiseAzimuthVectorList>
</noise>
"""


def test_parse_calibration_shape_and_values():
    calibration = core.parse_calibration(CALIBRATION_XML)
    assert calibration.lines.tolist() == [0, 30]
    assert calibration.pixels.tolist() == [0, 10, 20]
    assert calibration.sigma.shape == (2, 3)
    assert calibration.gamma[0].tolist() == [100.0, 200.0, 400.0]
    assert calibration.gamma[1].tolist() == [200.0, 400.0, 800.0]


def test_parse_noise_shape_and_values():
    noise = core.parse_noise(NOISE_XML)
    assert noise.range_lines.tolist() == [0, 30]
    assert noise.range_lut[0].tolist() == [0.0, 100.0, 400.0]
    assert len(noise.azimuth) == 1
    vector = noise.azimuth[0]
    assert vector.swath == "IW1"
    assert vector.first_line == 0
    assert vector.last_line == 30
    assert vector.first_pixel == 0
    assert vector.last_pixel == 20


def test_parse_noise_resamples_different_pixel_grids():
    noise_xml = NOISE_XML.replace(
        b"<line>30</line>\n      <pixel>0 10 20</pixel>\n      <noiseRangeLut>0 400 1600</noiseRangeLut>",
        b"<line>30</line>\n      <pixel>0 20</pixel>\n      <noiseRangeLut>0 1600</noiseRangeLut>",
    )
    noise = core.parse_noise(noise_xml)
    assert noise.range_pixels.tolist() == [0, 10, 20]
    assert noise.range_lut[1].tolist() == [0.0, 800.0, 1600.0]


def test_interp_lut_nodes_midpoint_and_clamp():
    calibration = core.parse_calibration(CALIBRATION_XML)
    lines = np.array([[0], [30]], dtype=float)
    pixels = np.array([[0, 5, 10, 15, 20, 25]], dtype=float)
    values = core.interp_lut(
        calibration.gamma,
        calibration.lines,
        calibration.pixels,
        lines,
        pixels,
    )
    assert values[0, 0] == pytest.approx(100.0)
    assert values[0, 1] == pytest.approx(150.0)
    assert values[0, 2] == pytest.approx(200.0)
    assert values[0, 5] == pytest.approx(400.0)
    assert values[1, 1] == pytest.approx(300.0)


def test_azimuth_interpolates_and_defaults_to_one():
    noise = core.parse_noise(NOISE_XML)
    lines = np.array([[15], [40]], dtype=float)
    pixels = np.array([[10, 30]], dtype=float)
    values = core.azimuth_lut(noise, lines, pixels)
    assert values[0, 0] == pytest.approx(1.1)
    assert values[0, 1] == pytest.approx(1.0)
    assert values[1, 0] == pytest.approx(1.0)


def test_calibrate_constant_dn_and_lut():
    calibration = core.parse_calibration(CALIBRATION_XML)
    noise = core.parse_noise(NOISE_XML)
    dn = np.full((2, 3), 200.0, dtype=np.uint16)
    lines = np.array([[0], [30]], dtype=float)
    pixels = np.array([[0, 10, 20]], dtype=float)
    linear, valid = core.calibrate_block(
        dn, lines, pixels, calibration, noise, lut_name="gamma"
    )
    assert valid.all()
    assert linear[0, 0] == pytest.approx(4.0)
    assert linear[1, 0] == pytest.approx(1.0)


def test_noise_clipping_sets_mask_zero_and_floor():
    calibration = core.parse_calibration(CALIBRATION_XML)
    noise = core.parse_noise(NOISE_XML)
    dn = np.array([[10, 100]], dtype=np.uint16)
    lines = np.array([[0]], dtype=float)
    pixels = np.array([[10, 0]], dtype=float)
    linear, valid = core.calibrate_block(
        dn, lines, pixels, calibration, noise, lut_name="gamma"
    )
    assert valid.tolist() == [[0, 1]]
    db = core.to_db(linear, valid)
    assert db[0, 0] == pytest.approx(core.DB_FLOOR)
    assert db[0, 1] == pytest.approx(20.0 * np.log10(100.0 / 100.0))


def test_to_db_floor_and_positive():
    linear = np.array([[0.0, 1.0, 0.01]], dtype=np.float32)
    valid = np.array([[0, 1, 1]], dtype=np.uint8)
    db = core.to_db(linear, valid)
    assert db[0, 0] == pytest.approx(core.DB_FLOOR)
    assert db[0, 1] == pytest.approx(0.0)
    assert db[0, 2] == pytest.approx(-20.0)


def test_calibration_rejects_non_increasing_lines():
    bad = CALIBRATION_XML.replace(b"<line>30</line>", b"<line>0</line>")
    with pytest.raises(ValueError, match="duplicate"):
        core.parse_calibration(bad)


def test_gcp_count_gate():
    core.validate_gcp_count(6)
    with pytest.raises(ValueError, match="at least"):
        core.validate_gcp_count(5)
