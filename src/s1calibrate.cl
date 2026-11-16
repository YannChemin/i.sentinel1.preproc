/* OpenCL arithmetic kernel for Sentinel-1 GRD calibration.
 *
 * SPDX-License-Identifier: Unlicense
 *
 * The host precomputes the LUT interval indices and weights per row and per
 * column, plus the azimuth LUT for the block, so the kernel does per-pixel
 * work only: two bilinear LUT reads, the noise subtraction and the division.
 * That keeps it memory bound, which is what the device profiles in the host
 * are tuned for.
 *
 * The kernel works in single precision while the C and Python engines work in
 * double. The calibration is a ratio of numbers of order 1e2 to 1e5, so the
 * relative error stays near float epsilon, but the two engines are not
 * bit-identical and the mask can differ on a pixel whose signal sits within
 * float rounding of zero. The host reports both differences rather than
 * hiding them.
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
    __global const float *c0 = cal_lut + cr0 * cal_npix;
    __global const float *c1 = cal_lut + cr1 * cal_npix;
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
        __global const float *n0 = noise_lut + nr0 * noise_npix;
        __global const float *n1 = noise_lut + nr1 * noise_npix;
        const float nv =
            ((n0[nc0] + (n0[nc1] - n0[nc0]) * nct) * (1.0f - nrt)) +
            ((n1[nc0] + (n1[nc1] - n1[nc0]) * nct) * nrt);
        noise_power = nv * azimuth[p];
    }

    /* DN squared is exact as an integer: the largest value, 65535 squared, is
     * under 2^32, while a float carries only 24 bits of mantissa and would
     * round it by up to 256. The conversion to float still rounds, but the
     * squaring itself no longer contributes error. */
    const uint d = (uint)dn[p];
    const float dn2 = (float)(d * d);
    float signal = dn2 - noise_power;
    if (signal < 0.0f) {
        signal = 0.0f;
    }
    linear[p] = signal / (cv * cv);
    valid[p] = (d > 0u && signal > 0.0f) ? (uchar)1 : (uchar)0;
}
