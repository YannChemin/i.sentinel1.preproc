# MODULE:    tests/test_noise_formats.py
# AUTHOR(S): Yann Chemin
# PURPOSE:   Test both Sentinel-1 noise annotation formats are read alike.
# SPDX-License-Identifier: Unlicense
"""Both Sentinel-1 noise annotation formats must parse to the same noise.

ESA renamed the range-noise elements with IPF 2.90, in March 2018:

====================  ====================
before                2.90 and later
====================  ====================
``noiseVector``       ``noiseRangeVector``
``noiseLut``          ``noiseRangeLut``
(none)                ``noiseAzimuthVector``
====================  ====================

The children are otherwise identical, so a reader that only knows the newer
names rejects every product acquired before March 2018 -- 62 of the 665 scenes
over the Sharjah archive, 9 %, and they are the earliest, so losing them
shortens any baseline at one end rather than thinning it evenly.

The element names and LUT values below are copied from a real 2017 annotation,
``S1A_IW_GRDH_1SDV_20170507T142433``, so the test fails if the convention is
mis-remembered rather than only if the code regresses.
"""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

import sentinel1_core as core

#: Real pixel positions and noise LUT values from the 2017 annotation.
PIXELS = [0, 40, 80, 120, 160, 200]
LUT = [
    1.682137e03,
    1.659369e03,
    1.637453e03,
    1.616060e03,
    1.595699e03,
    1.575819e03,
]


def _noise_xml(vector_tag, lut_tag, lines=(0, 100), azimuth=False):
    """Build a noise annotation using the given element names."""
    rows = ""
    for line in lines:
        rows += (
            f"<{vector_tag}>"
            f"<azimuthTime>2017-05-07T14:24:33.903269</azimuthTime>"
            f"<line>{line}</line>"
            f'<pixel count="{len(PIXELS)}">'
            + " ".join(str(p) for p in PIXELS)
            + "</pixel>"
            f'<{lut_tag} count="{len(LUT)}">'
            + " ".join(f"{v:.6e}" for v in LUT)
            + f"</{lut_tag}>"
            f"</{vector_tag}>"
        )
    az = ""
    if azimuth:
        az = (
            "<noiseAzimuthVectorList>"
            "<noiseAzimuthVector>"
            "<swath>IW</swath>"
            "<firstAzimuthLine>0</firstAzimuthLine>"
            "<lastAzimuthLine>100</lastAzimuthLine>"
            "<firstRangeSample>0</firstRangeSample>"
            "<lastRangeSample>200</lastRangeSample>"
            '<line count="2">0 100</line>'
            '<noiseAzimuthLut count="2">1.0 1.0</noiseAzimuthLut>'
            "</noiseAzimuthVector>"
            "</noiseAzimuthVectorList>"
        )
    listname = f"{vector_tag}List"
    return (
        '<?xml version="1.0" encoding="UTF-8"?><noise>'
        f'<{listname} count="{len(lines)}">{rows}</{listname}>'
        f"{az}</noise>"
    )


def _parse(xml):
    return core._parse_noise_root(ET.fromstring(xml))


def test_the_pre_2018_format_is_accepted():
    """A 2017 product must parse, not raise."""
    noise = _parse(_noise_xml("noiseVector", "noiseLut"))
    assert noise is not None


def test_both_formats_give_the_same_range_noise():
    """The rename must not change a single number."""
    old = _parse(_noise_xml("noiseVector", "noiseLut"))
    new = _parse(_noise_xml("noiseRangeVector", "noiseRangeLut"))
    assert np.array_equal(old.range_lut, new.range_lut)
    assert np.array_equal(old.range_lines, new.range_lines)
    assert np.array_equal(old.range_pixels, new.range_pixels)


def test_the_real_lut_values_survive_parsing():
    """Guards against a unit or scaling change slipping in.

    The noise LUT is power in DN squared, used as-is; anything that rescaled it
    would silently shift every calibrated pixel.
    """
    noise = _parse(_noise_xml("noiseVector", "noiseLut"))
    assert noise.range_lut[0][0] == pytest.approx(LUT[0])
    assert noise.range_lut[0][-1] == pytest.approx(LUT[-1])


def test_the_old_format_has_no_azimuth_vectors():
    """The azimuth factor must default to 1, not fail or come out zero.

    A zero azimuth factor would drive the noise subtraction to zero and make
    every pixel look like signal; a failure would reject the product outright.
    Both were real risks of the rename.
    """
    noise = _parse(_noise_xml("noiseVector", "noiseLut"))
    assert len(noise.azimuth) == 0


def test_a_noise_xml_with_neither_element_is_refused():
    xml = '<?xml version="1.0"?><noise><noiseVectorList count="0"/></noise>'
    with pytest.raises(ValueError, match="neither noiseRangeVector"):
        _parse(xml)


def test_the_error_names_both_element_names():
    """So the next reader is not sent looking for an element that never existed."""
    xml = '<?xml version="1.0"?><noise/>'
    with pytest.raises(ValueError) as excinfo:
        _parse(xml)
    message = str(excinfo.value)
    assert "noiseRangeVector" in message
    assert "noiseVector" in message
