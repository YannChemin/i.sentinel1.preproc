/* OpenCL arithmetic kernel for Sentinel-1 GRD calibration.
 * Host precomputes LUT interval indices/weights per row and per column,
 * plus the azimuth LUT for the block. This kernel does the per-pixel work.
 */
__kernel void s1calibrate(
    __global const ushort *dn,
    __global const float *azimuth,
    __global const float *cal_lut,
    __global const float *noise_lut,
    __global const int *cal_row0,
    __global const int *cal_row1,
    __global const float *cal_rowt,
    __global const int *noise_row0,
    __global const int *noise_row1,
    __global const float *noise_rowt,
    __global const int *cal_col0,
    __global const int *cal_col1,
    __global const float *cal_colt,
    __global const int *noise_col0,
    __global const int *noise_col1,
    __global const float *noise_colt,
    __global float *linear,
    __global uchar *valid,
    const int cols,
    const int rows,
    const int cal_npix,
    const int noise_npix,
    const int subtract_noise)
{
    const int c = get_global_id(0);
    const int r = get_global_id(1);
    if (c >= cols || r >= rows) {
        return;
    }
    const int p = r * cols + c;

    const int cr0 = cal_row0[r];
    const int cr1 = cal_row1[r];
    const float crt = cal_rowt[r];
    const int cc0 = cal_col0[c];
    const int cc1 = cal_col1[c];
    const float cct = cal_colt[c];
    const float *c0 = cal_lut + cr0 * cal_npix;
    const float *c1 = cal_lut + cr1 * cal_npix;
    const float cv =
        ((c0[cc0] + (c0[cc1] - c0[cc0]) * cct) * (1.0f - crt)) +
        ((c1[cc0] + (c1[cc1] - c1[cc0]) * cct) * crt);

    float noise_power = 0.0f;
    if (subtract_noise) {
        const int nr0 = noise_row0[r];
        const int nr1 = noise_row1[r];
        const float nrt = noise_rowt[r];
        const int nc0 = noise_col0[c];
        const int nc1 = noise_col1[c];
        const float nct = noise_colt[c];
        const float *n0 = noise_lut + nr0 * noise_npix;
        const float *n1 = noise_lut + nr1 * noise_npix;
        const float nv =
            ((n0[nc0] + (n0[nc1] - n0[nc0]) * nct) * (1.0f - nrt)) +
            ((n1[nc0] + (n1[nc1] - n1[nc0]) * nct) * nrt);
        noise_power = nv * azimuth[p];
    }

    const float value = (float)dn[p];
    float signal = value * value - noise_power;
    if (signal < 0.0f) {
        signal = 0.0f;
    }
    linear[p] = signal / (cv * cv);
    valid[p] = (value > 0.0f && signal > 0.0f) ? (uchar)1 : (uchar)0;
}
