# MODULE:    sentinel1_orbit.py
# AUTHOR(S): Yann Chemin
# PURPOSE:   Locate, fetch and parse Sentinel-1 orbit state vectors.
# SPDX-License-Identifier: Unlicense
"""Sentinel-1 orbit state vectors from the official ESA auxiliary files.

MetaMonitor: https://github.com/YannChemin/MetaMonitor

A Sentinel-1 GRD product served by the Planetary Computer carries ground
control points but no orbit state vectors: its ``schema-product-*`` asset is
the RFI annotation, not the product annotation. Without state vectors there is
no radar look vector, and without a look vector there is no local incidence
angle, no layover mask and no shadow mask, so terrain correction cannot be
done. This module supplies the missing vectors.

ESA publishes two auxiliary orbit products, both as Earth Explorer XML:

``AUX_POEORB``
    Precise orbit ephemerides, a few centimetres of accuracy, published about
    20 days after acquisition. The right choice for an archive.

``AUX_RESORB``
    Restituted orbits, published within hours, decimetre accuracy. The only
    choice for a scene younger than the precise latency.

Source
------

The files are read from the public ``s1-orbits`` bucket that Alaska Satellite
Facility operates, over anonymous HTTPS. No account, token or ``.netrc`` is
needed, which is what makes this usable from an unattended pipeline. Verified
on 2026-11-16: an object fetch returns HTTP 200 and the anonymous bucket
listing works, so the covering file can be found from the acquisition time
alone.

The alternative is the Copernicus Data Space OData catalogue. Its *search* is
open, and it was used to confirm the file naming, but a download returns HTTP
401 without an OAuth token, so it is not used here. ESA's own
``qc.sentinel1.eo.esa.int`` API did not resolve from this machine.

What this module does and does not do
-------------------------------------

It finds, downloads, caches and parses the state vectors, and reports which
file was used. It does **not** perform terrain correction: turning state
vectors into a local incidence angle, layover and shadow masks is separate
work, and `i.sentinel1.preproc` still calibrates against the ellipsoid.
"""

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

#: Anonymous HTTPS front of the ASF public bucket holding the ESA orbit files.
BUCKET_URL = "https://s1-orbits.s3.amazonaws.com"

#: Orbit kinds and the ESA product type each maps to.
PRODUCT_TYPES = {"precise": "AUX_POEORB", "restituted": "AUX_RESORB"}

#: Precise orbits appear about 20 days after acquisition. A scene younger than
#: this cannot have one, so ``auto`` goes straight to the restituted product
#: rather than paying for a listing that cannot succeed.
PRECISE_LATENCY_DAYS = 20

#: Pages of the bucket listing to walk before giving up. One page is 1000 keys,
#: which at roughly three files a day is most of a year of generation dates, so
#: one page almost always contains the covering file.
MAX_LIST_PAGES = 3

#: S3 list-objects-v2 XML namespace.
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

#: ``<mission>_OPER_AUX_<type>_OPOD_<generated>_V<start>_<stop>.EOF``
NAME_RE = re.compile(
    r"^(?P<mission>S1[A-D])_OPER_AUX_(?P<product>POEORB|RESORB)_\w+_"
    r"(?P<generated>\d{8}T\d{6})_V(?P<start>\d{8}T\d{6})_"
    r"(?P<stop>\d{8}T\d{6})\.EOF$"
)

_STAMP = "%Y%m%dT%H%M%S"


class OrbitError(Exception):
    """No usable orbit file could be found, fetched or parsed."""


def _utc(text):
    """Parse a compact ``YYYYmmddTHHMMSS`` stamp as an aware UTC datetime."""
    return datetime.strptime(text, _STAMP).replace(tzinfo=timezone.utc)


def parse_orbit_name(name):
    """Pull the mission, product type and validity window out of a file name.

    :param name: Bare file name, with or without a bucket prefix.
    :returns: A dict with ``mission``, ``product``, ``generated``, ``start``
        and ``stop``; the last three are aware UTC datetimes.
    :raises OrbitError: If the name is not an ESA orbit file name.
    """
    base = str(name).rsplit("/", 1)[-1]
    match = NAME_RE.match(base)
    if match is None:
        raise OrbitError(f"not a Sentinel-1 orbit file name: {base}")
    return {
        "name": base,
        "mission": match.group("mission"),
        "product": match.group("product"),
        "generated": _utc(match.group("generated")),
        "start": _utc(match.group("start")),
        "stop": _utc(match.group("stop")),
    }


def orbit_key_prefix(mission, product):
    """The leading part of an orbit file name, up to the generation stamp.

    :param mission: ``S1A`` ... ``S1D``.
    :param product: An ESA product type such as ``AUX_POEORB``, which already
        includes its own ``AUX_`` prefix.
    :returns: For example ``S1A_OPER_AUX_POEORB_OPOD_``.
    """
    return f"{mission}_OPER_{product}_OPOD_"


def covers(info, sensing, margin_s=0.0):
    """Say whether an orbit file's validity window contains an instant.

    :param info: A :func:`parse_orbit_name` result.
    :param sensing: Aware datetime of the acquisition.
    :param margin_s: Require this many seconds of validity on either side, so
        that interpolating around the acquisition does not run off the end of
        the state-vector list.
    :returns: ``True`` if the window covers the instant with the margin.
    """
    margin = timedelta(seconds=margin_s)
    return info["start"] + margin <= sensing <= info["stop"] - margin


def select_orbit(names, sensing, margin_s=0.0):
    """Choose the best orbit file for an acquisition.

    Where several files cover the instant, the most recently generated wins:
    ESA reprocesses, and the later generation is the better solution.

    :param names: Candidate file names.
    :param sensing: Aware datetime of the acquisition.
    :param margin_s: Validity margin, as in :func:`covers`.
    :returns: The chosen :func:`parse_orbit_name` result, or ``None``.
    """
    best = None
    for name in names:
        try:
            info = parse_orbit_name(name)
        except OrbitError:
            continue  # The bucket holds other things; ignore them.
        if not covers(info, sensing, margin_s):
            continue
        if best is None or info["generated"] > best["generated"]:
            best = info
    return best


def _list_keys(prefix, start_after, opener, pages=MAX_LIST_PAGES):
    """Walk an anonymous S3 listing and yield object keys.

    :param opener: Callable taking a URL and returning bytes, so the network
        can be replaced in tests.
    """
    token = None
    for _ in range(pages):
        url = f"{BUCKET_URL}/?list-type=2&prefix={prefix}"
        if token:
            url += f"&continuation-token={_quote(token)}"
        elif start_after:
            url += f"&start-after={_quote(start_after)}"
        root = ET.fromstring(opener(url))
        for key in root.findall(".//s3:Contents/s3:Key", S3_NS):
            if key.text:
                yield key.text
        truncated = root.findtext("s3:IsTruncated", default="false", namespaces=S3_NS)
        token = root.findtext(
            "s3:NextContinuationToken", default="", namespaces=S3_NS
        )
        if truncated.lower() != "true" or not token:
            return


def _quote(value):
    """Percent-encode a listing parameter."""
    from urllib.parse import quote

    return quote(value, safe="")


def find_orbit(mission, sensing, kind="auto", opener=None, margin_s=0.0):
    """Find the orbit file covering an acquisition.

    :param mission: ``S1A``, ``S1B``, ``S1C`` or ``S1D``.
    :param sensing: Aware datetime of the acquisition.
    :param kind: ``precise``, ``restituted``, or ``auto`` to take the precise
        product when the acquisition is old enough for one to exist and fall
        back to the restituted product otherwise.
    :param opener: Callable taking a URL and returning bytes; defaults to an
        anonymous HTTPS GET.
    :param margin_s: Validity margin, as in :func:`covers`.
    :returns: ``(info, key)`` where ``key`` is the object key in the bucket.
    :raises OrbitError: If nothing covering the acquisition is published.
    """
    if sensing.tzinfo is None:
        raise OrbitError("sensing time must be timezone aware")
    opener = opener or _http_get
    if kind == "auto":
        age = datetime.now(timezone.utc) - sensing
        order = (
            ["precise", "restituted"]
            if age >= timedelta(days=PRECISE_LATENCY_DAYS)
            else ["restituted", "precise"]
        )
    elif kind in PRODUCT_TYPES:
        order = [kind]
    else:
        raise OrbitError(f"unknown orbit kind {kind!r}")

    problems = []
    for candidate in order:
        product = PRODUCT_TYPES[candidate]
        prefix = f"{product}/"
        # Keys sort by generation time, and a file is always generated after
        # the data it describes, so the listing can start at the acquisition
        # date and skip the whole archive before it.
        # product already carries the "AUX_" prefix, and so does the file
        # name: "S1A_OPER_AUX_POEORB_OPOD_...". Writing "AUX_" again sorted
        # the key before every real one, so the listing restarted at the 2014
        # beginning of the archive and three pages never reached the scene.
        start_after = (
            f"{prefix}{orbit_key_prefix(mission, product)}"
            f"{sensing.strftime('%Y%m%d')}T000000"
        )
        keys = [
            key
            for key in _list_keys(prefix, start_after, opener)
            if key.rsplit("/", 1)[-1].startswith(mission)
        ]
        info = select_orbit(keys, sensing, margin_s)
        if info is not None:
            return info, f"{prefix}{info['name']}"
        problems.append(f"no {product} covering {sensing.isoformat()}")
    raise OrbitError("; ".join(problems))


def _http_get(url, timeout=120):
    """Anonymous HTTPS GET returning bytes.

    A generic client name is sent deliberately: an unattended pipeline must not
    put anyone's identity or contact details into an outgoing request.
    """
    import urllib.request

    request = urllib.request.Request(
        url, headers={"User-Agent": "i.sentinel1.preproc orbit fetcher"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_orbit(mission, sensing, cache_dir, kind="auto", opener=None,
                progress=None, margin_s=0.0):
    """Download the orbit file for an acquisition, reusing a cached copy.

    :param cache_dir: Directory for the downloaded files.
    :param progress: Optional callable ``(done, total)`` for progress
        reporting; a GRASS module passes ``gs.percent``.
    :returns: ``(path, info)``.
    :raises OrbitError: If nothing suitable is published or the download fails.
    """
    info, key = find_orbit(mission, sensing, kind, opener, margin_s)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / info["name"]
    if path.is_file() and path.stat().st_size > 0:
        return path, info

    opener = opener or _http_get
    try:
        payload = opener(f"{BUCKET_URL}/{key}")
    except Exception as exc:  # noqa: BLE001 - reported as an orbit failure
        raise OrbitError(f"cannot download {key}: {exc}") from exc
    if progress is not None:
        progress(len(payload), len(payload))
    # Written beside the target and renamed, so an interrupted download is
    # never mistaken for a complete orbit file.
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(payload)
    partial.replace(path)
    return path, info


def parse_state_vectors(path_or_bytes):
    """Read the orbit state vectors out of an Earth Explorer orbit file.

    Positions and velocities are Earth-fixed, in metres and metres per second.
    The list is dense: a precise file covering one day carries about 9 400
    vectors, one every 10 s.

    :param path_or_bytes: Path to the file, or its contents.
    :returns: A dict with ``epochs`` (aware UTC datetimes) and the lists
        ``x``, ``y``, ``z``, ``vx``, ``vy``, ``vz``, plus ``absolute_orbit``.
    :raises OrbitError: If the file carries no usable vectors.
    """
    if isinstance(path_or_bytes, (bytes, bytearray)):
        root = ET.fromstring(path_or_bytes)
    else:
        root = ET.parse(str(path_or_bytes)).getroot()
    rows = root.findall("./Data_Block/List_of_OSVs/OSV")
    if not rows:
        raise OrbitError("orbit file carries no OSV records")

    out = {
        "epochs": [],
        "x": [],
        "y": [],
        "z": [],
        "vx": [],
        "vy": [],
        "vz": [],
        "absolute_orbit": [],
    }
    for row in rows:
        utc = row.findtext("UTC")
        if not utc:
            raise OrbitError("an OSV record has no UTC field")
        # The field is written as "UTC=2024-04-14T22:59:42.000000".
        stamp = utc.split("=", 1)[1] if "=" in utc else utc
        out["epochs"].append(
            datetime.fromisoformat(stamp).replace(tzinfo=timezone.utc)
        )
        for field, key in (
            ("X", "x"), ("Y", "y"), ("Z", "z"),
            ("VX", "vx"), ("VY", "vy"), ("VZ", "vz"),
        ):
            value = row.findtext(field)
            if value is None:
                raise OrbitError(f"an OSV record has no {field} field")
            out[key].append(float(value))
        orbit = row.findtext("Absolute_Orbit")
        out["absolute_orbit"].append(int(orbit) if orbit else None)

    # A quality check that costs nothing and catches a truncated or corrupt
    # file: the epochs must increase, and the orbital radius must be sane.
    if any(b <= a for a, b in zip(out["epochs"][:-1], out["epochs"][1:])):
        raise OrbitError("orbit state vector epochs are not strictly increasing")
    radius = (out["x"][0] ** 2 + out["y"][0] ** 2 + out["z"][0] ** 2) ** 0.5
    if not 6.9e6 < radius < 7.3e6:
        raise OrbitError(
            f"first state vector is {radius / 1e3:.0f} km from the geocentre, "
            "which is not a Sentinel-1 orbit"
        )
    return out
