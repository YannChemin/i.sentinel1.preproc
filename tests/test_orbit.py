# MODULE:    tests/test_orbit.py
# AUTHOR(S): Yann Chemin
# PURPOSE:   Test orbit file selection and state vector parsing offline.
# SPDX-License-Identifier: Unlicense
"""Tests for :mod:`sentinel1_orbit`, with the network replaced.

The listing and the download both go through an ``opener`` callable, so these
tests use a fake bucket built from real ESA file names and a hand-written
Earth Explorer file. Nothing here touches the network: a test that needs ASF to
be reachable is not a test of this code.

There is one network test, marked ``network``, that checks the public bucket is
still anonymous. Run it deliberately::

    python3 -m pytest tests/test_orbit.py -m network
"""

from datetime import datetime, timedelta, timezone

import pytest

import sentinel1_orbit as orbit

#: Real file names from the bucket, so the parsing is tested against the
#: actual ESA convention rather than an invented one.
POEORB = [
    "AUX_POEORB/S1A_OPER_AUX_POEORB_OPOD_20240501T070731_V20240410T225942_20240412T005942.EOF",
    "AUX_POEORB/S1A_OPER_AUX_POEORB_OPOD_20240504T070742_V20240413T225942_20240415T005942.EOF",
    "AUX_POEORB/S1A_OPER_AUX_POEORB_OPOD_20240505T070741_V20240414T225942_20240416T005942.EOF",
    "AUX_POEORB/S1B_OPER_AUX_POEORB_OPOD_20240505T070741_V20240414T225942_20240416T005942.EOF",
]

SENSING = datetime(2024, 4, 15, 2, 15, 52, tzinfo=timezone.utc)


def _listing(keys, truncated=False, token=""):
    """Build an S3 list-objects-v2 response."""
    items = "".join(f"<Contents><Key>{k}</Key></Contents>" for k in keys)
    extra = f"<NextContinuationToken>{token}</NextContinuationToken>" if token else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>"
        f"{items}{extra}</ListBucketResult>"
    ).encode()


def _osv(epoch, index):
    """One OSV record, on a plausible Sentinel-1 orbit."""
    return (
        "<OSV>"
        f"<UTC>UTC={epoch.strftime('%Y-%m-%dT%H:%M:%S.%f')}</UTC>"
        f"<Absolute_Orbit>+{53436 + index}</Absolute_Orbit>"
        "<X unit=\"m\">-2101967.080475</X>"
        "<Y unit=\"m\">6308205.120945</Y>"
        "<Z unit=\"m\">2413321.747429</Z>"
        "<VX unit=\"m/s\">734.400877</VX>"
        "<VY unit=\"m/s\">2924.316936</VY>"
        "<VZ unit=\"m/s\">-6973.604353</VZ>"
        "<Quality>NOMINAL</Quality>"
        "</OSV>"
    )


def _eof(count=4, step_s=10, descending=False):
    """An Earth Explorer orbit file with ``count`` state vectors."""
    base = datetime(2024, 4, 14, 22, 59, 42, tzinfo=timezone.utc)
    epochs = [base + timedelta(seconds=i * step_s) for i in range(count)]
    if descending:
        epochs = list(reversed(epochs))
    rows = "".join(_osv(e, i) for i, e in enumerate(epochs))
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="no" ?>'
        "<Earth_Explorer_File><Data_Block><List_of_OSVs>"
        f"{rows}"
        "</List_of_OSVs></Data_Block></Earth_Explorer_File>"
    ).encode()


def test_name_parsing_pulls_out_the_validity_window():
    info = orbit.parse_orbit_name(POEORB[2])
    assert info["mission"] == "S1A"
    assert info["product"] == "POEORB"
    assert info["start"] == datetime(2024, 4, 14, 22, 59, 42, tzinfo=timezone.utc)
    assert info["stop"] == datetime(2024, 4, 16, 0, 59, 42, tzinfo=timezone.utc)
    assert info["generated"] == datetime(2024, 5, 5, 7, 7, 41, tzinfo=timezone.utc)


def test_a_name_that_is_not_an_orbit_file_is_refused():
    with pytest.raises(orbit.OrbitError):
        orbit.parse_orbit_name("AUX_POEORB/README.txt")


def test_selection_takes_the_covering_file():
    info = orbit.select_orbit(POEORB, SENSING)
    assert info is not None
    # The 0504 file stops at 2024-04-15T00:59:42, before the acquisition; only
    # the 0505 one covers 02:15:52.
    assert info["generated"].day == 5
    assert "20240505" in info["name"]


def test_selection_prefers_the_later_generation_when_both_cover():
    reprocessed = (
        "AUX_POEORB/S1A_OPER_AUX_POEORB_OPOD_20260101T000000"
        "_V20240414T225942_20240416T005942.EOF"
    )
    info = orbit.select_orbit([*POEORB, reprocessed], SENSING)
    assert info["generated"].year == 2026, "a reprocessed orbit must win"


def test_selection_honours_the_validity_margin():
    # 2024-04-16T00:59:42 is the end of validity, so an acquisition one minute
    # before it is covered, but not with a five minute interpolation margin.
    late = datetime(2024, 4, 16, 0, 58, 42, tzinfo=timezone.utc)
    assert orbit.select_orbit(POEORB, late) is not None
    assert orbit.select_orbit(POEORB, late, margin_s=300) is None


def test_find_orbit_filters_by_mission():
    calls = []

    def opener(url):
        calls.append(url)
        return _listing(POEORB)

    info, key = orbit.find_orbit("S1B", SENSING, kind="precise", opener=opener)
    assert info["mission"] == "S1B"
    assert key.startswith("AUX_POEORB/S1B_")
    # The listing must start after the acquisition date, not at the archive's
    # beginning: a file is always generated after the data it describes.
    #
    # The whole key is asserted, not just the date. An earlier version wrote
    # "AUX_AUX_POEORB" here, which sorts before every real key, so the listing
    # restarted at 2014 and the scene was reported as having no orbit at all.
    # A test that checked only the timestamp passed throughout.
    from urllib.parse import unquote

    assert "start-after" in calls[0]
    start_after = unquote(calls[0].split("start-after=")[1])
    assert start_after == (
        "AUX_POEORB/S1B_OPER_AUX_POEORB_OPOD_20240415T000000"
    ), start_after


def test_find_orbit_follows_a_truncated_listing():
    pages = [
        _listing(POEORB[:1], truncated=True, token="PAGE2"),
        _listing(POEORB[2:3]),
    ]

    def opener(url):
        return pages.pop(0)

    info, _ = orbit.find_orbit("S1A", SENSING, kind="precise", opener=opener)
    assert "20240505" in info["name"]
    assert not pages, "the second page was not requested"


def test_auto_prefers_restituted_for_a_recent_acquisition():
    asked = []

    def opener(url):
        asked.append(url)
        return _listing([])

    recent = datetime.now(timezone.utc) - timedelta(days=2)
    with pytest.raises(orbit.OrbitError):
        orbit.find_orbit("S1A", recent, kind="auto", opener=opener)
    # Precise orbits do not exist yet at two days old, so the restituted
    # product must be tried first.
    assert "AUX_RESORB" in asked[0]
    assert any("AUX_POEORB" in url for url in asked)


def test_auto_prefers_precise_for_an_old_acquisition():
    asked = []

    def opener(url):
        asked.append(url)
        return _listing([])

    with pytest.raises(orbit.OrbitError):
        orbit.find_orbit("S1A", SENSING, kind="auto", opener=opener)
    assert "AUX_POEORB" in asked[0]


def test_a_naive_sensing_time_is_refused():
    with pytest.raises(orbit.OrbitError):
        orbit.find_orbit("S1A", datetime(2024, 4, 15, 2, 15, 52), kind="precise")


def test_unknown_kind_is_refused():
    with pytest.raises(orbit.OrbitError):
        orbit.find_orbit("S1A", SENSING, kind="approximate")


def test_fetch_writes_the_file_and_reuses_the_cache(tmp_path):
    payload = _eof()
    downloads = []

    def opener(url):
        if "list-type" in url:
            return _listing(POEORB)
        downloads.append(url)
        return payload

    path, info = orbit.fetch_orbit(
        "S1A", SENSING, tmp_path, kind="precise", opener=opener
    )
    assert path.is_file() and path.read_bytes() == payload
    assert path.name == info["name"]
    assert len(downloads) == 1
    # No ".part" file may be left behind.
    assert not list(tmp_path.glob("*.part"))

    orbit.fetch_orbit("S1A", SENSING, tmp_path, kind="precise", opener=opener)
    assert len(downloads) == 1, "a cached orbit file was downloaded again"


def test_fetch_reports_a_download_failure_as_an_orbit_error(tmp_path):
    def opener(url):
        if "list-type" in url:
            return _listing(POEORB)
        raise OSError("connection reset")

    with pytest.raises(orbit.OrbitError, match="cannot download"):
        orbit.fetch_orbit("S1A", SENSING, tmp_path, kind="precise", opener=opener)


def test_state_vectors_are_parsed():
    sv = orbit.parse_state_vectors(_eof(count=3))
    assert len(sv["epochs"]) == 3
    assert sv["x"][0] == pytest.approx(-2101967.080475)
    assert sv["vz"][0] == pytest.approx(-6973.604353)
    assert sv["absolute_orbit"][0] == 53436
    assert sv["epochs"][0] == datetime(
        2024, 4, 14, 22, 59, 42, tzinfo=timezone.utc
    )


def test_state_vectors_reject_a_file_with_no_records():
    empty = (
        b'<?xml version="1.0"?><Earth_Explorer_File><Data_Block>'
        b"<List_of_OSVs></List_of_OSVs></Data_Block></Earth_Explorer_File>"
    )
    with pytest.raises(orbit.OrbitError, match="no OSV records"):
        orbit.parse_state_vectors(empty)


def test_state_vectors_reject_epochs_out_of_order():
    with pytest.raises(orbit.OrbitError, match="strictly increasing"):
        orbit.parse_state_vectors(_eof(count=4, descending=True))


def test_state_vectors_reject_an_implausible_orbit():
    # A ground position rather than an orbit: the radius check must catch it.
    bad = _eof(count=2).replace(b"-2101967.080475", b"-2101.080475")
    with pytest.raises(orbit.OrbitError, match="not a Sentinel-1 orbit"):
        orbit.parse_state_vectors(bad)


@pytest.mark.network
def test_the_public_bucket_is_still_anonymous():
    """The one test that needs the network: ASF still serves without an account.

    If this fails, the orbit source has changed and the module needs another
    one; that is worth knowing explicitly rather than as a pipeline failure.
    """
    path_info, key = orbit.find_orbit("S1A", SENSING, kind="precise")
    assert key.startswith("AUX_POEORB/S1A_")
    assert path_info["product"] == "POEORB"


def test_key_prefix_is_not_doubled():
    """The product type already carries its own ``AUX_`` prefix."""
    assert orbit.orbit_key_prefix("S1A", "AUX_POEORB") == (
        "S1A_OPER_AUX_POEORB_OPOD_"
    )
    assert orbit.orbit_key_prefix("S1C", "AUX_RESORB") == (
        "S1C_OPER_AUX_RESORB_OPOD_"
    )
    # And it must match what the real file names actually look like.
    for key in POEORB:
        name = key.rsplit("/", 1)[-1]
        info = orbit.parse_orbit_name(name)
        prefix = orbit.orbit_key_prefix(info["mission"], f"AUX_{info['product']}")
        assert name.startswith(prefix), f"{name} does not start with {prefix}"
