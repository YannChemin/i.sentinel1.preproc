<h2>DESCRIPTION</h2>

<em>i.sentinel1.preproc</em> calibrates a Sentinel-1 GRD product to
sigma0, beta0 or gamma0, removes thermal noise, geocodes the result to the
current GRASS region and writes a validity mask.

The input is either a local Sentinel-1 GRD <tt>.SAFE</tt> directory or a
Planetary Computer <tt>sentinel-1-grd</tt> STAC item id. In item mode the module
reads the measurement asset and the per-polarisation calibration and noise XML
assets. The measurement is raw uint16 digital numbers with WGS84 ground
control points and no CRS.

The output is a Float32 linear-power map, an optional dB map and an optional
uint8 mask. The dB floor is -40 dB. Pixels at or below the thermal-noise floor
are written at the floor and marked invalid in the mask.

<h2>NOTES</h2>

The calibration LUT is interpolated bilinearly in source line and pixel
coordinates and applied at the source pixel centre. The default
<tt>calibration=gamma0</tt> uses the product's gamma LUT, which assumes the
ellipsoid incidence angle. This is appropriate for the flat coastal AOIs that
motivated the module.

<tt>noiseRangeLut</tt> is a power in DN squared, not an amplitude. The module
subtracts it directly from the squared digital number. The azimuth LUT is a
dimensionless multiplier close to one, applied inside its swath window.

The geometry is the product's own GCP polynomial, warped with
<tt>gdalwarp -order 1</tt> to the current region. The module does not use SNAP.
It does not perform DEM-based radiometric terrain correction in this version,
so it does not produce a layover or shadow mask.

The Planetary Computer GRD item does not contain the product annotation or
orbit state vectors. Its <tt>schema-product-*</tt> asset is the RFI annotation,
not the product annotation. DEM terrain correction therefore requires a local
<tt>.SAFE</tt> product or a separate orbit provider.

The module writes temporary full-scene Float32 rasters for the exact
calibration path. Use <b>-k</b> to keep them for inspection; otherwise they are
removed.

<h2>EXAMPLES</h2>

Calibrate one Planetary Computer item over the current region and write dB
maps and masks:

<div class="code"><pre>
g.region raster=aoi res=10
i.sentinel1.preproc item=S1A_IW_GRDH_1SDV_20240427T021605_20240427T021630_053413_067E8F_7C1D \
    output=s1_20240427 calibration=gamma0 -d -m
</pre></div>

Calibrate a local SAFE product:

<div class="code"><pre>
i.sentinel1.preproc input_dir=/data/S1A_IW_GRDH_1SDV_20240427T021605_20240427T021630_053413_067E8F_7C1D.SAFE \
    output=s1_20240427 polarisations=vv,vh -d -m
</pre></div>

<h2>OUTPUT</h2>

For <tt>output=s1_20240427</tt>, <tt>polarisations=vv</tt> and
<tt>calibration=gamma0</tt>:

<dl>
<dt><tt>s1_20240427_vv_gamma0_linear</tt></dt>
<dd>linear gamma0 power</dd>
<dt><tt>s1_20240427_vv_gamma0_dB</tt></dt>
<dd>dB gamma0, written with <b>-d</b></dd>
<dt><tt>s1_20240427_vv_gamma0_mask</tt></dt>
<dd>1 valid, 0 invalid, written with <b>-m</b></dd>
<dt><tt>s1_20240427_group</tt></dt>
<dd>group containing all polarisation maps</dd>
</dl>

A JSON provenance record is written under <tt>cell_misc/</tt> or under the
directory given with <tt>metadata=</tt>.

<h2>REQUIREMENTS</h2>

GRASS GIS 8.4 or later, GDAL with GCP support, Python 3 with
<tt>numpy</tt>, <tt>rasterio</tt>, <tt>pystac-client</tt>,
<tt>planetary-computer</tt> and <tt>requests</tt> for item mode.

<h2>SEE ALSO</h2>

<em><a href="r.in.gdal.html">r.in.gdal</a></em>,
<em><a href="r.import.html">r.import</a></em>,
<em><a href="i.group.html">i.group</a></em>

<h2>AUTHOR</h2>

Yann Chemin
