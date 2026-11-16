#define _GNU_SOURCE

#include <errno.h>
#include <expat.h>
#include <getopt.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <time.h>
#include <unistd.h>

#include <cpl_conv.h>
#include <cpl_error.h>
#include <cpl_vsi.h>
#include <cpl_string.h>
#include <gdal.h>

#ifdef S1CAL_HAVE_OPENCL
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#endif

#define VERSION "0.1.0"

typedef struct {
    int *v;
    size_t n;
    size_t cap;
} IntVec;

typedef struct {
    double *v;
    size_t n;
    size_t cap;
} DblVec;

typedef struct {
    char *s;
    size_t n;
    size_t cap;
} Text;

typedef struct {
    IntVec line;
    IntVec pixel;
    DblVec sigma;
    DblVec beta;
    DblVec gamma;
} CalRow;

typedef struct {
    CalRow *v;
    size_t n;
    size_t cap;
} CalRows;

typedef struct {
    IntVec line;
    IntVec pixel;
    DblVec lut;
} RangeRow;

typedef struct {
    RangeRow *v;
    size_t n;
    size_t cap;
} RangeRows;

typedef struct {
    char swath[32];
    int first_line;
    int last_line;
    int first_pixel;
    int last_pixel;
    IntVec line;
    DblVec lut;
} AzVec;

typedef struct {
    AzVec *v;
    size_t n;
    size_t cap;
} AzRows;

typedef struct {
    int nlines;
    double *lines;
    int npix;
    double *pixels;
    double *sigma;
    double *beta;
    double *gamma;
} Cal;

typedef struct {
    int nlines;
    double *lines;
    int npix;
    double *pixels;
    double *lut;
    AzVec *az;
    int naz;
} Noise;

enum {
    VEC_NONE = 0,
    VEC_CAL = 1,
    VEC_RANGE = 2,
    VEC_AZ = 3
};

typedef struct {
    int vec;
    char tag[64];
    Text text;
    IntVec line;
    IntVec pixel;
    DblVec sigma;
    DblVec beta;
    DblVec gamma;
    DblVec lut;
    char swath[32];
    int first_line;
    int last_line;
    int first_pixel;
    int last_pixel;
    CalRows cal;
    RangeRows range;
    AzRows az;
} Parser;

static void die(const char *message)
{
    fprintf(stderr, "s1calibrate: %s\n", message);
    exit(EXIT_FAILURE);
}

static void *xmalloc(size_t size)
{
    void *p = malloc(size ? size : 1);
    if (p == NULL) {
        die("out of memory");
    }
    return p;
}

static void *xrealloc(void *p, size_t size)
{
    void *q = realloc(p, size ? size : 1);
    if (q == NULL) {
        die("out of memory");
    }
    return q;
}

static void int_push(IntVec *a, int value)
{
    if (a->n == a->cap) {
        a->cap = a->cap ? a->cap * 2 : 16;
        a->v = xrealloc(a->v, a->cap * sizeof(int));
    }
    a->v[a->n++] = value;
}

static void dbl_push(DblVec *a, double value)
{
    if (a->n == a->cap) {
        a->cap = a->cap ? a->cap * 2 : 16;
        a->v = xrealloc(a->v, a->cap * sizeof(double));
    }
    a->v[a->n++] = value;
}

static void text_reset(Text *t)
{
    t->n = 0;
}

static void text_append(Text *t, const char *s, int len)
{
    if (len <= 0) {
        return;
    }
    if (t->n + (size_t)len + 1 > t->cap) {
        size_t cap = t->cap ? t->cap : 64;
        while (cap < t->n + (size_t)len + 1) {
            cap *= 2;
        }
        t->s = xrealloc(t->s, cap);
        t->cap = cap;
    }
    memcpy(t->s + t->n, s, (size_t)len);
    t->n += (size_t)len;
    t->s[t->n] = '\0';
}

static void parse_ints(const Text *t, IntVec *out)
{
    const char *p = t->s ? t->s : "";
    while (*p) {
        while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) {
            p++;
        }
        if (!*p) {
            break;
        }
        errno = 0;
        char *end = NULL;
        long value = strtol(p, &end, 10);
        if (end == p || errno != 0) {
            die("invalid integer in XML");
        }
        int_push(out, (int)value);
        p = end;
    }
}

static void parse_doubles(const Text *t, DblVec *out)
{
    const char *p = t->s ? t->s : "";
    while (*p) {
        while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) {
            p++;
        }
        if (!*p) {
            break;
        }
        errno = 0;
        char *end = NULL;
        double value = strtod(p, &end);
        if (end == p || errno != 0) {
            die("invalid number in XML");
        }
        dbl_push(out, value);
        p = end;
    }
}

static void reset_vector(Parser *p)
{
    p->line.n = 0;
    p->pixel.n = 0;
    p->sigma.n = 0;
    p->beta.n = 0;
    p->gamma.n = 0;
    p->lut.n = 0;
    p->swath[0] = '\0';
    p->first_line = 0;
    p->last_line = 0;
    p->first_pixel = 0;
    p->last_pixel = 0;
}

static void copy_ints(const IntVec *src, IntVec *dst)
{
    dst->n = 0;
    for (size_t i = 0; i < src->n; i++) {
        int_push(dst, src->v[i]);
    }
}

static void copy_doubles(const DblVec *src, DblVec *dst)
{
    dst->n = 0;
    for (size_t i = 0; i < src->n; i++) {
        dbl_push(dst, src->v[i]);
    }
}

static void push_cal_row(Parser *p)
{
    if (p->line.n != 1) {
        die("calibrationVector must have one line value");
    }
    if (p->pixel.n == 0 || p->sigma.n != p->pixel.n || p->beta.n != p->pixel.n ||
        p->gamma.n != p->pixel.n) {
        die("calibrationVector arrays have inconsistent lengths");
    }
    if (p->cal.n == p->cal.cap) {
        p->cal.cap = p->cal.cap ? p->cal.cap * 2 : 16;
        p->cal.v = xrealloc(p->cal.v, p->cal.cap * sizeof(CalRow));
    }
    CalRow *row = &p->cal.v[p->cal.n++];
    memset(row, 0, sizeof(*row));
    copy_ints(&p->line, &row->line);
    copy_ints(&p->pixel, &row->pixel);
    copy_doubles(&p->sigma, &row->sigma);
    copy_doubles(&p->beta, &row->beta);
    copy_doubles(&p->gamma, &row->gamma);
    reset_vector(p);
}

static void push_range_row(Parser *p)
{
    if (p->line.n != 1) {
        die("noiseRangeVector must have one line value");
    }
    if (p->pixel.n == 0 || p->lut.n != p->pixel.n) {
        die("noiseRangeVector arrays have inconsistent lengths");
    }
    if (p->range.n == p->range.cap) {
        p->range.cap = p->range.cap ? p->range.cap * 2 : 16;
        p->range.v = xrealloc(p->range.v, p->range.cap * sizeof(RangeRow));
    }
    RangeRow *row = &p->range.v[p->range.n++];
    memset(row, 0, sizeof(*row));
    copy_ints(&p->line, &row->line);
    copy_ints(&p->pixel, &row->pixel);
    copy_doubles(&p->lut, &row->lut);
    reset_vector(p);
}

static void push_az_row(Parser *p)
{
    if (p->line.n == 0 || p->line.n != p->lut.n) {
        die("noiseAzimuthVector arrays have inconsistent lengths");
    }
    if (p->az.n == p->az.cap) {
        p->az.cap = p->az.cap ? p->az.cap * 2 : 8;
        p->az.v = xrealloc(p->az.v, p->az.cap * sizeof(AzVec));
    }
    AzVec *row = &p->az.v[p->az.n++];
    memset(row, 0, sizeof(*row));
    snprintf(row->swath, sizeof(row->swath), "%s", p->swath);
    row->first_line = p->first_line;
    row->last_line = p->last_line;
    row->first_pixel = p->first_pixel;
    row->last_pixel = p->last_pixel;
    copy_ints(&p->line, &row->line);
    copy_doubles(&p->lut, &row->lut);
    reset_vector(p);
}

static int all_equal(const IntVec *a, const IntVec *b)
{
    if (a->n != b->n) {
        return 0;
    }
    for (size_t i = 0; i < a->n; i++) {
        if (a->v[i] != b->v[i]) {
            return 0;
        }
    }
    return 1;
}

static void validate_increasing(const double *v, int n, const char *name)
{
    for (int i = 1; i < n; i++) {
        if (!(v[i] > v[i - 1])) {
            fprintf(stderr, "s1calibrate: %s is not strictly increasing\n", name);
            exit(EXIT_FAILURE);
        }
    }
}

static double interp1d(const double *x, const double *y, int n, double q)
{
    if (n <= 0) {
        return 0.0;
    }
    if (n == 1) {
        return y[0];
    }
    if (q <= x[0]) {
        return y[0];
    }
    if (q >= x[n - 1]) {
        return y[n - 1];
    }
    int lo = 0;
    int hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (x[mid] <= q) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    double t = (q - x[lo]) / (x[hi] - x[lo]);
    return y[lo] + (y[hi] - y[lo]) * t;
}

static double interp_ints(const IntVec *x, const DblVec *y, double q)
{
    double *xd = xmalloc(x->n * sizeof(double));
    for (size_t i = 0; i < x->n; i++) {
        xd[i] = (double)x->v[i];
    }
    double value = interp1d(xd, y->v, (int)x->n, q);
    free(xd);
    return value;
}

static void build_cal(const CalRows *rows, Cal *out)
{
    if (rows->n == 0) {
        die("calibration XML has no calibrationVector");
    }
    const IntVec *reference = &rows->v[0].pixel;
    out->nlines = (int)rows->n;
    out->lines = xmalloc((size_t)out->nlines * sizeof(double));
    out->npix = (int)reference->n;
    out->pixels = xmalloc((size_t)out->npix * sizeof(double));
    for (int i = 0; i < out->npix; i++) {
        out->pixels[i] = (double)reference->v[i];
    }
    size_t count = (size_t)out->nlines * (size_t)out->npix;
    out->sigma = xmalloc(count * sizeof(double));
    out->beta = xmalloc(count * sizeof(double));
    out->gamma = xmalloc(count * sizeof(double));

    for (int r = 0; r < out->nlines; r++) {
        const CalRow *row = &rows->v[r];
        if (row->line.n != 1) {
            die("calibrationVector must have one line value");
        }
        out->lines[r] = (double)row->line.v[0];
        const DblVec *source[3] = {&row->sigma, &row->beta, &row->gamma};
        double *target[3] = {out->sigma, out->beta, out->gamma};
        for (int k = 0; k < 3; k++) {
            if (source[k]->n != row->pixel.n) {
                die("calibrationVector arrays have inconsistent lengths");
            }
            for (int j = 0; j < out->npix; j++) {
                double value;
                if (all_equal(&row->pixel, reference)) {
                    value = source[k]->v[j];
                } else {
                    value = interp_ints(&row->pixel, source[k], out->pixels[j]);
                }
                if (!(value > 0.0) || !isfinite(value)) {
                    die("calibration LUT has a non-positive value");
                }
                target[k][(size_t)r * out->npix + j] = value;
            }
        }
    }
    validate_increasing(out->lines, out->nlines, "calibration line");
    validate_increasing(out->pixels, out->npix, "calibration pixel");
}

static void build_noise(const RangeRows *rows, const AzRows *az, Noise *out)
{
    if (rows->n == 0) {
        die("noise XML has no noiseRangeVector");
    }
    const IntVec *reference = &rows->v[0].pixel;
    out->nlines = (int)rows->n;
    out->lines = xmalloc((size_t)out->nlines * sizeof(double));
    out->npix = (int)reference->n;
    out->pixels = xmalloc((size_t)out->npix * sizeof(double));
    for (int i = 0; i < out->npix; i++) {
        out->pixels[i] = (double)reference->v[i];
    }
    size_t count = (size_t)out->nlines * (size_t)out->npix;
    out->lut = xmalloc(count * sizeof(double));

    for (int r = 0; r < out->nlines; r++) {
        const RangeRow *row = &rows->v[r];
        if (row->line.n != 1 || row->lut.n != row->pixel.n) {
            die("noiseRangeVector arrays have inconsistent lengths");
        }
        out->lines[r] = (double)row->line.v[0];
        for (int j = 0; j < out->npix; j++) {
            double value;
            if (all_equal(&row->pixel, reference)) {
                value = row->lut.v[j];
            } else {
                value = interp_ints(&row->pixel, &row->lut, out->pixels[j]);
            }
            if (value < 0.0 || !isfinite(value)) {
                die("noiseRangeLut has a negative or non-finite value");
            }
            out->lut[(size_t)r * out->npix + j] = value;
        }
    }
    validate_increasing(out->lines, out->nlines, "noise line");
    validate_increasing(out->pixels, out->npix, "noise pixel");

    out->naz = (int)az->n;
    out->az = xmalloc((size_t)out->naz * sizeof(AzVec));
    for (int i = 0; i < out->naz; i++) {
        out->az[i] = az->v[i];
    }
}

static void XMLCALL start_element(void *userData, const XML_Char *name, const XML_Char **atts)
{
    (void)atts;
    Parser *p = userData;
    snprintf(p->tag, sizeof(p->tag), "%s", name);
    text_reset(&p->text);

    if (strcmp(name, "calibrationVector") == 0) {
        p->vec = VEC_CAL;
        reset_vector(p);
    } else if (strcmp(name, "noiseRangeVector") == 0) {
        p->vec = VEC_RANGE;
        reset_vector(p);
    } else if (strcmp(name, "noiseAzimuthVector") == 0) {
        p->vec = VEC_AZ;
        reset_vector(p);
    }
}

static void XMLCALL end_element(void *userData, const XML_Char *name)
{
    Parser *p = userData;
    if (strcmp(name, "calibrationVector") == 0) {
        push_cal_row(p);
        p->vec = VEC_NONE;
        return;
    }
    if (strcmp(name, "noiseRangeVector") == 0) {
        push_range_row(p);
        p->vec = VEC_NONE;
        return;
    }
    if (strcmp(name, "noiseAzimuthVector") == 0) {
        push_az_row(p);
        p->vec = VEC_NONE;
        return;
    }
    if (strcmp(name, "line") == 0) {
        parse_ints(&p->text, &p->line);
    } else if (strcmp(name, "pixel") == 0) {
        parse_ints(&p->text, &p->pixel);
    } else if (strcmp(name, "sigmaNought") == 0) {
        parse_doubles(&p->text, &p->sigma);
    } else if (strcmp(name, "betaNought") == 0) {
        parse_doubles(&p->text, &p->beta);
    } else if (strcmp(name, "gamma") == 0) {
        parse_doubles(&p->text, &p->gamma);
    } else if (strcmp(name, "noiseRangeLut") == 0) {
        parse_doubles(&p->text, &p->lut);
    } else if (strcmp(name, "noiseAzimuthLut") == 0) {
        parse_doubles(&p->text, &p->lut);
    } else if (strcmp(name, "swath") == 0) {
        snprintf(p->swath, sizeof(p->swath), "%s", p->text.s ? p->text.s : "");
    } else if (strcmp(name, "firstAzimuthLine") == 0) {
        p->first_line = atoi(p->text.s ? p->text.s : "0");
    } else if (strcmp(name, "lastAzimuthLine") == 0) {
        p->last_line = atoi(p->text.s ? p->text.s : "0");
    } else if (strcmp(name, "firstRangeSample") == 0) {
        p->first_pixel = atoi(p->text.s ? p->text.s : "0");
    } else if (strcmp(name, "lastRangeSample") == 0) {
        p->last_pixel = atoi(p->text.s ? p->text.s : "0");
    }
}

static void XMLCALL character_data(void *userData, const XML_Char *s, int len)
{
    Parser *p = userData;
    text_append(&p->text, s, len);
}

/* The annotation XML is read through GDAL's virtual file layer, not fopen, so
 * that a /vsicurl/ path works. The measurement band already arrives that way,
 * and the Planetary Computer serves the calibration and noise annotations over
 * HTTPS with no local copy; fopen on such a path fails with ENOENT. */
static char *read_file(const char *path, size_t *length)
{
    VSILFILE *f = VSIFOpenL(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "s1calibrate: cannot open %s: %s\n", path,
                CPLGetLastErrorMsg()[0] ? CPLGetLastErrorMsg() : strerror(errno));
        exit(EXIT_FAILURE);
    }
    if (VSIFSeekL(f, 0, SEEK_END) != 0) {
        die("cannot seek XML file");
    }
    vsi_l_offset size = VSIFTellL(f);
    /* A signed-URL redirect or an error page would be small; an annotation is
     * hundreds of kilobytes. Refuse anything that cannot be an annotation
     * rather than handing expat an HTML body. */
    if (size < 1024 || size > (vsi_l_offset)1 << 28) {
        fprintf(stderr, "s1calibrate: %s is %llu bytes, not an annotation XML\n",
                path, (unsigned long long)size);
        exit(EXIT_FAILURE);
    }
    if (VSIFSeekL(f, 0, SEEK_SET) != 0) {
        die("cannot rewind XML file");
    }
    char *buffer = xmalloc((size_t)size + 1);
    size_t read = VSIFReadL(buffer, 1, (size_t)size, f);
    VSIFCloseL(f);
    if (read != (size_t)size) {
        die("cannot read XML file");
    }
    buffer[size] = '\0';
    *length = (size_t)size;
    return buffer;
}

static void parse_xml(const char *path, Parser *parser)
{
    size_t length = 0;
    char *buffer = read_file(path, &length);
    XML_Parser expat = XML_ParserCreate(NULL);
    if (expat == NULL) {
        die("cannot create XML parser");
    }
    XML_SetUserData(expat, parser);
    XML_SetElementHandler(expat, start_element, end_element);
    XML_SetCharacterDataHandler(expat, character_data);
    if (XML_Parse(expat, buffer, (int)length, XML_TRUE) != XML_STATUS_OK) {
        fprintf(
            stderr,
            "s1calibrate: XML parse error in %s at line %lu: %s\n",
            path,
            (unsigned long)XML_GetCurrentLineNumber(expat),
            XML_ErrorString(XML_GetErrorCode(expat))
        );
        exit(EXIT_FAILURE);
    }
    XML_ParserFree(expat);
    free(buffer);
}

static void load_calibration(const char *path, Cal *cal)
{
    Parser parser;
    memset(&parser, 0, sizeof(parser));
    parse_xml(path, &parser);
    build_cal(&parser.cal, cal);
}

static void load_noise(const char *path, Noise *noise)
{
    Parser parser;
    memset(&parser, 0, sizeof(parser));
    parse_xml(path, &parser);
    build_noise(&parser.range, &parser.az, noise);
}

static void find_interval(const double *x, int n, double q, int *i0, int *i1, double *t)
{
    if (n <= 1) {
        *i0 = 0;
        *i1 = 0;
        *t = 0.0;
        return;
    }
    if (q <= x[0]) {
        *i0 = 0;
        *i1 = 0;
        *t = 0.0;
        return;
    }
    if (q >= x[n - 1]) {
        *i0 = n - 1;
        *i1 = n - 1;
        *t = 0.0;
        return;
    }
    int lo = 0;
    int hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (x[mid] <= q) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    *i0 = lo;
    *i1 = hi;
    *t = (q - x[lo]) / (x[hi] - x[lo]);
}

static int block_side(int memory_mb, int width, int height)
{
    long pixels = (long)memory_mb * 1024L * 1024L / 64L;
    int side = (int)sqrt((double)pixels);
    if (side < 256) {
        side = 256;
    }
    if (side > 2048) {
        side = 2048;
    }
    if (side > width) {
        side = width;
    }
    if (side > height) {
        side = height;
    }
    return side;
}

static double *select_lut(const Cal *cal, const char *name)
{
    if (strcmp(name, "sigma") == 0) {
        return cal->sigma;
    }
    if (strcmp(name, "beta") == 0) {
        return cal->beta;
    }
    if (strcmp(name, "gamma") == 0) {
        return cal->gamma;
    }
    die("unknown calibration LUT");
    return NULL;
}

static void fill_azimuth(const Noise *noise, int row_start, int col_start, int cols, int rows, float *az)
{
    size_t count = (size_t)cols * (size_t)rows;
    for (size_t i = 0; i < count; i++) {
        az[i] = 1.0f;
    }
    for (int v = 0; v < noise->naz; v++) {
        const AzVec *vector = &noise->az[v];
        int c0 = vector->first_pixel - col_start;
        int c1 = vector->last_pixel - col_start;
        if (c0 < 0) {
            c0 = 0;
        }
        if (c1 >= cols) {
            c1 = cols - 1;
        }
        if (c0 > c1) {
            continue;
        }
        double* xd = xmalloc(vector->line.n * sizeof(double));
        for (size_t k = 0; k < vector->line.n; k++) {
            xd[k] = (double)vector->line.v[k];
        }
        for (int r = 0; r < rows; r++) {
            int line = row_start + r;
            if (line < vector->first_line || line > vector->last_line) {
                continue;
            }
            double value = interp1d(xd, vector->lut.v, (int)vector->line.n, (double)line);
            for (int c = c0; c <= c1; c++) {
                az[(size_t)r * cols + c] = (float)value;
            }
        }
        free(xd);
    }
}

/* ------------------------------------------------------------------------- */
/* OpenCL engine                                                             */
/*                                                                           */
/* The arithmetic is memory bound: two bilinear LUT reads, a multiply, a      */
/* subtract and a divide per pixel. So the host does not try to be clever     */
/* about the kernel; it is the data movement that decides whether a device    */
/* beats the C+OpenMP path, and that is what the device profile selects.      */
/*                                                                           */
/* Profiles, as required of every OpenCL path in this codebase:               */
/*   cpu   a CPU device through PoCL. Buffers are host-allocated, because a   */
/*         copy to a CPU device is a wasted pass over the only memory bus.    */
/*   igpu  an integrated GPU with unified memory. Transfers are mapped, not   */
/*         copied, but the buffers are left for the driver to place: on this  */
/*         hardware CL_MEM_ALLOC_HOST_PTR made Intel's kernels slower.        */
/*   dgpu  a discrete card on internal PCIe. Explicit copies.                 */
/*   egpu  a discrete card behind a slow link. Same as dgpu here, since the   */
/*         kernel is one pass per block and has nothing to overlap with; it   */
/*         is kept as a separate name so a forced choice is reportable.       */
/* ------------------------------------------------------------------------- */
#ifdef S1CAL_HAVE_OPENCL

#define S1CAL_CL_ARGS 23

typedef struct {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    char device_name[256];
    char platform_name[256];
    char profile[8];
    char reason[256];
    cl_device_type type;
    cl_bool unified;
    cl_ulong global_mem;
    cl_ulong max_alloc;
    cl_uint compute_units;
    size_t wg_multiple;

    int host_ptr;   /* allocate buffers in host memory (CPU devices only) */
    int mapped;     /* move data by map/unmap instead of read/write */
    int timing;     /* report per-kernel time and achieved bandwidth */

    size_t capacity;   /* pixels per block the buffers were sized for */
    int side;          /* rows/cols per block the index buffers were sized for */

    cl_mem dn, azimuth, cal_lut, noise_lut;
    cl_mem cal_row0, cal_row1, cal_rowt;
    cl_mem noise_row0, noise_row1, noise_rowt;
    cl_mem cal_col0, cal_col1, cal_colt;
    cl_mem noise_col0, noise_col1, noise_colt;
    cl_mem linear, valid;

    /* Host-side float staging, since the CPU engine works in double. */
    int *h_cal_row0, *h_cal_row1, *h_noise_row0, *h_noise_row1;
    float *h_cal_rowt, *h_noise_rowt;
    int *h_cal_col0, *h_cal_col1, *h_noise_col0, *h_noise_col1;
    float *h_cal_colt, *h_noise_colt;

    double t_upload, t_kernel, t_download;
    double bytes_up, bytes_down;
    long blocks;
} ClEngine;

static void cl_die(const char *message, cl_int status)
{
    fprintf(stderr, "s1calibrate: %s (OpenCL error %d)\n", message, (int)status);
    exit(EXIT_FAILURE);
}

static void cl_check(cl_int status, const char *message)
{
    if (status != CL_SUCCESS) {
        cl_die(message, status);
    }
}

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* Find the kernel source. Installed beside the binary, or pointed at by
 * S1CAL_KERNEL, or in the source tree during development. Failing to find it
 * is fatal: silently falling back to the CPU would hide that --opencl did
 * nothing. */
static char *cl_kernel_source(size_t *length)
{
    const char *override = getenv("S1CAL_KERNEL");
    if (override != NULL) {
        return read_file(override, length);
    }

    char exe[4096];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n > 0) {
        exe[n] = '\0';
        char *slash = strrchr(exe, '/');
        if (slash != NULL) {
            *slash = '\0';
            const char *suffixes[] = {
                "/s1calibrate.cl",
                "/../etc/s1calibrate.cl",
                "/../share/s1calibrate.cl",
                "/s1calibrate.cl",
            };
            for (size_t i = 0; i < sizeof(suffixes) / sizeof(suffixes[0]); i++) {
                char path[4160];
                snprintf(path, sizeof(path), "%s%s", exe, suffixes[i]);
                FILE *probe = fopen(path, "rb");
                if (probe != NULL) {
                    fclose(probe);
                    return read_file(path, length);
                }
            }
        }
    }
    die("cannot find s1calibrate.cl; set S1CAL_KERNEL to its path");
    return NULL;
}

static const char *cl_profile_of(cl_device_type type, cl_bool unified)
{
    if (type & CL_DEVICE_TYPE_CPU) {
        return "cpu";
    }
    if (unified) {
        return "igpu";
    }
    return "dgpu";
}

/* Enumerate every device of every platform in a stable order, so that
 * --opencl-device refers to the same device across runs. */
static int cl_enumerate(cl_platform_id *platforms, cl_device_id *devices, int max)
{
    cl_uint nplatforms = 0;
    if (clGetPlatformIDs(0, NULL, &nplatforms) != CL_SUCCESS || nplatforms == 0) {
        return 0;
    }
    cl_platform_id *ids = xmalloc(nplatforms * sizeof(cl_platform_id));
    cl_check(clGetPlatformIDs(nplatforms, ids, NULL), "cannot list OpenCL platforms");

    int count = 0;
    for (cl_uint p = 0; p < nplatforms && count < max; p++) {
        cl_uint ndevices = 0;
        if (clGetDeviceIDs(ids[p], CL_DEVICE_TYPE_ALL, 0, NULL, &ndevices) != CL_SUCCESS) {
            continue;
        }
        cl_device_id *devs = xmalloc(ndevices * sizeof(cl_device_id));
        if (clGetDeviceIDs(ids[p], CL_DEVICE_TYPE_ALL, ndevices, devs, NULL) == CL_SUCCESS) {
            for (cl_uint d = 0; d < ndevices && count < max; d++) {
                platforms[count] = ids[p];
                devices[count] = devs[d];
                count++;
            }
        }
        free(devs);
    }
    free(ids);
    return count;
}

static void cl_list_devices(void)
{
    cl_platform_id platforms[64];
    cl_device_id devices[64];
    int count = cl_enumerate(platforms, devices, 64);
    if (count == 0) {
        printf("no OpenCL device found\n");
        return;
    }
    for (int i = 0; i < count; i++) {
        char pname[256] = "";
        char dname[256] = "";
        cl_device_type type = 0;
        cl_bool unified = CL_FALSE;
        cl_ulong mem = 0;
        cl_uint units = 0;
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(pname), pname, NULL);
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(dname), dname, NULL);
        clGetDeviceInfo(devices[i], CL_DEVICE_TYPE, sizeof(type), &type, NULL);
        clGetDeviceInfo(
            devices[i], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(unified), &unified, NULL
        );
        clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, NULL);
        clGetDeviceInfo(
            devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(units), &units, NULL
        );
        printf(
            "%d: %s / %s [profile %s, %u unit(s), %.1f GiB]\n",
            i,
            pname,
            dname,
            cl_profile_of(type, unified),
            (unsigned)units,
            (double)mem / 1073741824.0
        );
    }
}

static void cl_engine_init(
    ClEngine *e,
    int device_index,
    const char *forced_profile,
    int timing,
    int side,
    size_t capacity,
    const float *cal_lut,
    int cal_nlines,
    int cal_npix,
    const float *noise_lut,
    int noise_nlines,
    int noise_npix
)
{
    memset(e, 0, sizeof(*e));
    e->timing = timing;
    e->side = side;
    e->capacity = capacity;

    cl_platform_id platforms[64];
    cl_device_id devices[64];
    int count = cl_enumerate(platforms, devices, 64);
    if (count == 0) {
        die("--opencl requested but no OpenCL device is available");
    }

    /* Without an explicit index, prefer a GPU over a CPU device: the CPU
     * device competes with the OpenMP engine for the same cores. */
    if (device_index < 0) {
        device_index = 0;
        for (int i = 0; i < count; i++) {
            cl_device_type type = 0;
            clGetDeviceInfo(devices[i], CL_DEVICE_TYPE, sizeof(type), &type, NULL);
            if (type & CL_DEVICE_TYPE_GPU) {
                device_index = i;
                break;
            }
        }
    }
    if (device_index >= count) {
        fprintf(
            stderr,
            "s1calibrate: --opencl-device %d out of range; %d device(s) present\n",
            device_index,
            count
        );
        exit(EXIT_FAILURE);
    }

    e->platform = platforms[device_index];
    e->device = devices[device_index];
    clGetPlatformInfo(
        e->platform, CL_PLATFORM_NAME, sizeof(e->platform_name), e->platform_name, NULL
    );
    clGetDeviceInfo(
        e->device, CL_DEVICE_NAME, sizeof(e->device_name), e->device_name, NULL
    );
    clGetDeviceInfo(e->device, CL_DEVICE_TYPE, sizeof(e->type), &e->type, NULL);
    clGetDeviceInfo(
        e->device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(e->unified), &e->unified, NULL
    );
    clGetDeviceInfo(
        e->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(e->global_mem), &e->global_mem, NULL
    );
    clGetDeviceInfo(
        e->device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(e->max_alloc), &e->max_alloc, NULL
    );
    clGetDeviceInfo(
        e->device,
        CL_DEVICE_MAX_COMPUTE_UNITS,
        sizeof(e->compute_units),
        &e->compute_units,
        NULL
    );

    const char *detected = cl_profile_of(e->type, e->unified);
    if (forced_profile == NULL || strcmp(forced_profile, "auto") == 0) {
        snprintf(e->profile, sizeof(e->profile), "%s", detected);
        snprintf(
            e->reason,
            sizeof(e->reason),
            "detected: type %s, unified memory %s",
            (e->type & CL_DEVICE_TYPE_CPU) ? "CPU" : "GPU",
            e->unified ? "yes" : "no"
        );
    } else {
        /* A forced profile that the device cannot honour is an error, not a
         * silent fallback: the point of forcing one is to compare it. */
        if (strcmp(forced_profile, "cpu") == 0 && !(e->type & CL_DEVICE_TYPE_CPU)) {
            die("profile=cpu forced on a device that is not a CPU device");
        }
        if (strcmp(forced_profile, "igpu") == 0 && !e->unified) {
            die("profile=igpu forced on a device without unified host memory");
        }
        if ((strcmp(forced_profile, "dgpu") == 0 || strcmp(forced_profile, "egpu") == 0)
            && (e->type & CL_DEVICE_TYPE_CPU)) {
            die("profile=dgpu/egpu forced on a CPU device");
        }
        snprintf(e->profile, sizeof(e->profile), "%s", forced_profile);
        snprintf(
            e->reason, sizeof(e->reason), "forced on the command line (detected %s)",
            detected
        );
    }

    /* Host-allocated buffers are for CPU devices only. On this laptop's Iris
     * Xe, CL_MEM_ALLOC_HOST_PTR made Intel's own kernels about 15 % slower, so
     * the igpu profile maps transfers but lets the driver place the buffers. */
    e->host_ptr = (strcmp(e->profile, "cpu") == 0);
    e->mapped = (strcmp(e->profile, "cpu") == 0 || strcmp(e->profile, "igpu") == 0);

    cl_int status = CL_SUCCESS;
    e->context = clCreateContext(NULL, 1, &e->device, NULL, NULL, &status);
    cl_check(status, "cannot create an OpenCL context");
    e->queue = clCreateCommandQueue(e->context, e->device, 0, &status);
    cl_check(status, "cannot create an OpenCL command queue");

    size_t source_length = 0;
    char *source = cl_kernel_source(&source_length);
    const char *sources[1] = {source};
    e->program = clCreateProgramWithSource(e->context, 1, sources, &source_length, &status);
    cl_check(status, "cannot create the OpenCL program");
    status = clBuildProgram(e->program, 1, &e->device, "-cl-std=CL1.2", NULL, NULL);
    if (status != CL_SUCCESS) {
        size_t log_length = 0;
        clGetProgramBuildInfo(
            e->program, e->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_length
        );
        char *log = xmalloc(log_length + 1);
        clGetProgramBuildInfo(
            e->program, e->device, CL_PROGRAM_BUILD_LOG, log_length, log, NULL
        );
        log[log_length] = '\0';
        fprintf(stderr, "s1calibrate: OpenCL build failed:\n%s\n", log);
        free(log);
        exit(EXIT_FAILURE);
    }
    free(source);
    e->kernel = clCreateKernel(e->program, "s1calibrate", &status);
    cl_check(status, "cannot create the s1calibrate kernel");
    clGetKernelWorkGroupInfo(
        e->kernel,
        e->device,
        CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        sizeof(e->wg_multiple),
        &e->wg_multiple,
        NULL
    );
    if (e->wg_multiple == 0) {
        e->wg_multiple = 1;
    }

    cl_mem_flags extra = e->host_ptr ? CL_MEM_ALLOC_HOST_PTR : 0;
    size_t pixels = capacity;

    /* The largest single buffer is the float output; refuse to run rather than
     * let the driver fail halfway through a scene. */
    size_t largest = pixels * sizeof(cl_float);
    if ((cl_ulong)largest > e->max_alloc) {
        fprintf(
            stderr,
            "s1calibrate: block of %zu pixel(s) needs a %.1f MiB buffer but the "
            "device allows %.1f MiB; lower --memory-mb\n",
            pixels,
            (double)largest / 1048576.0,
            (double)e->max_alloc / 1048576.0
        );
        exit(EXIT_FAILURE);
    }

#define S1CAL_BUF(field, flags, bytes)                                            \
    do {                                                                          \
        e->field = clCreateBuffer(                                                \
            e->context, (flags) | extra, (bytes), NULL, &status                   \
        );                                                                        \
        cl_check(status, "cannot allocate device buffer " #field);                 \
    } while (0)

    S1CAL_BUF(dn, CL_MEM_READ_ONLY, pixels * sizeof(cl_ushort));
    S1CAL_BUF(azimuth, CL_MEM_READ_ONLY, pixels * sizeof(cl_float));
    S1CAL_BUF(linear, CL_MEM_WRITE_ONLY, pixels * sizeof(cl_float));
    S1CAL_BUF(valid, CL_MEM_WRITE_ONLY, pixels * sizeof(cl_uchar));
    S1CAL_BUF(cal_row0, CL_MEM_READ_ONLY, (size_t)side * sizeof(cl_int));
    S1CAL_BUF(cal_row1, CL_MEM_READ_ONLY, (size_t)side * sizeof(cl_int));
    S1CAL_BUF(cal_rowt, CL_MEM_READ_ONLY, (size_t)side * sizeof(cl_float));
    S1CAL_BUF(noise_row0, CL_MEM_READ_ONLY, (size_t)side * sizeof(cl_int));
    S1CAL_BUF(noise_row1, CL_MEM_READ_ONLY, (size_t)side * sizeof(cl_int));
    S1CAL_BUF(noise_rowt, CL_MEM_READ_ONLY, (size_t)side * sizeof(cl_float));
    S1CAL_BUF(cal_col0, CL_MEM_READ_ONLY, (size_t)side * sizeof(cl_int));
    S1CAL_BUF(cal_col1, CL_MEM_READ_ONLY, (size_t)side * sizeof(cl_int));
    S1CAL_BUF(cal_colt, CL_MEM_READ_ONLY, (size_t)side * sizeof(cl_float));
    S1CAL_BUF(noise_col0, CL_MEM_READ_ONLY, (size_t)side * sizeof(cl_int));
    S1CAL_BUF(noise_col1, CL_MEM_READ_ONLY, (size_t)side * sizeof(cl_int));
    S1CAL_BUF(noise_colt, CL_MEM_READ_ONLY, (size_t)side * sizeof(cl_float));
#undef S1CAL_BUF

    /* The LUTs are a few hundred kilobytes and constant for the scene, so they
     * are uploaded once and stay resident. */
    size_t cal_bytes = (size_t)cal_nlines * (size_t)cal_npix * sizeof(cl_float);
    size_t noise_bytes = (size_t)noise_nlines * (size_t)noise_npix * sizeof(cl_float);
    e->cal_lut = clCreateBuffer(
        e->context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        cal_bytes,
        (void *)cal_lut,
        &status
    );
    cl_check(status, "cannot upload the calibration LUT");
    e->noise_lut = clCreateBuffer(
        e->context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        noise_bytes,
        (void *)noise_lut,
        &status
    );
    cl_check(status, "cannot upload the noise LUT");

    e->h_cal_row0 = xmalloc((size_t)side * sizeof(int));
    e->h_cal_row1 = xmalloc((size_t)side * sizeof(int));
    e->h_cal_rowt = xmalloc((size_t)side * sizeof(float));
    e->h_noise_row0 = xmalloc((size_t)side * sizeof(int));
    e->h_noise_row1 = xmalloc((size_t)side * sizeof(int));
    e->h_noise_rowt = xmalloc((size_t)side * sizeof(float));
    e->h_cal_col0 = xmalloc((size_t)side * sizeof(int));
    e->h_cal_col1 = xmalloc((size_t)side * sizeof(int));
    e->h_cal_colt = xmalloc((size_t)side * sizeof(float));
    e->h_noise_col0 = xmalloc((size_t)side * sizeof(int));
    e->h_noise_col1 = xmalloc((size_t)side * sizeof(int));
    e->h_noise_colt = xmalloc((size_t)side * sizeof(float));

    fprintf(
        stderr,
        "s1calibrate: OpenCL %s / %s, profile %s (%s), %u unit(s), "
        "work-group multiple %zu, buffers %s, transfers %s\n",
        e->platform_name,
        e->device_name,
        e->profile,
        e->reason,
        (unsigned)e->compute_units,
        e->wg_multiple,
        e->host_ptr ? "host-allocated" : "driver-placed",
        e->mapped ? "mapped" : "copied"
    );
}

/* Upload one host array, by map/unmap on unified memory or by copy otherwise. */
static void cl_put(ClEngine *e, cl_mem buffer, const void *host, size_t bytes)
{
    cl_int status = CL_SUCCESS;
    if (bytes == 0) {
        return;
    }
    if (e->mapped) {
        void *p = clEnqueueMapBuffer(
            e->queue, buffer, CL_TRUE, CL_MAP_WRITE, 0, bytes, 0, NULL, NULL, &status
        );
        cl_check(status, "cannot map a device buffer for writing");
        memcpy(p, host, bytes);
        cl_check(
            clEnqueueUnmapMemObject(e->queue, buffer, p, 0, NULL, NULL),
            "cannot unmap a device buffer"
        );
    } else {
        cl_check(
            clEnqueueWriteBuffer(e->queue, buffer, CL_TRUE, 0, bytes, host, 0, NULL, NULL),
            "cannot write a device buffer"
        );
    }
    e->bytes_up += (double)bytes;
}

static void cl_get(ClEngine *e, cl_mem buffer, void *host, size_t bytes)
{
    cl_int status = CL_SUCCESS;
    if (bytes == 0) {
        return;
    }
    if (e->mapped) {
        void *p = clEnqueueMapBuffer(
            e->queue, buffer, CL_TRUE, CL_MAP_READ, 0, bytes, 0, NULL, NULL, &status
        );
        cl_check(status, "cannot map a device buffer for reading");
        memcpy(host, p, bytes);
        cl_check(
            clEnqueueUnmapMemObject(e->queue, buffer, p, 0, NULL, NULL),
            "cannot unmap a device buffer"
        );
    } else {
        cl_check(
            clEnqueueReadBuffer(e->queue, buffer, CL_TRUE, 0, bytes, host, 0, NULL, NULL),
            "cannot read a device buffer"
        );
    }
    e->bytes_down += (double)bytes;
}

static void cl_engine_run(
    ClEngine *e,
    const uint16_t *dn,
    const float *azimuth,
    int cols,
    int rows,
    int cal_npix,
    int noise_npix,
    int subtract_noise,
    float *linear,
    uint8_t *valid
)
{
    size_t pixels = (size_t)cols * (size_t)rows;
    cl_int status = CL_SUCCESS;

    double t0 = now_seconds();
    cl_put(e, e->dn, dn, pixels * sizeof(cl_ushort));
    cl_put(e, e->azimuth, azimuth, pixels * sizeof(cl_float));
    cl_put(e, e->cal_row0, e->h_cal_row0, (size_t)rows * sizeof(cl_int));
    cl_put(e, e->cal_row1, e->h_cal_row1, (size_t)rows * sizeof(cl_int));
    cl_put(e, e->cal_rowt, e->h_cal_rowt, (size_t)rows * sizeof(cl_float));
    cl_put(e, e->noise_row0, e->h_noise_row0, (size_t)rows * sizeof(cl_int));
    cl_put(e, e->noise_row1, e->h_noise_row1, (size_t)rows * sizeof(cl_int));
    cl_put(e, e->noise_rowt, e->h_noise_rowt, (size_t)rows * sizeof(cl_float));
    cl_put(e, e->cal_col0, e->h_cal_col0, (size_t)cols * sizeof(cl_int));
    cl_put(e, e->cal_col1, e->h_cal_col1, (size_t)cols * sizeof(cl_int));
    cl_put(e, e->cal_colt, e->h_cal_colt, (size_t)cols * sizeof(cl_float));
    cl_put(e, e->noise_col0, e->h_noise_col0, (size_t)cols * sizeof(cl_int));
    cl_put(e, e->noise_col1, e->h_noise_col1, (size_t)cols * sizeof(cl_int));
    cl_put(e, e->noise_colt, e->h_noise_colt, (size_t)cols * sizeof(cl_float));
    double t1 = now_seconds();

    cl_int arg = 0;
    cl_mem buffers[18] = {
        e->dn,         e->azimuth,    e->cal_lut,    e->noise_lut,
        e->cal_row0,   e->cal_row1,   e->cal_rowt,   e->noise_row0,
        e->noise_row1, e->noise_rowt, e->cal_col0,   e->cal_col1,
        e->cal_colt,   e->noise_col0, e->noise_col1, e->noise_colt,
        e->linear,     e->valid
    };
    for (int i = 0; i < 18; i++) {
        status |= clSetKernelArg(e->kernel, arg++, sizeof(cl_mem), &buffers[i]);
    }
    cl_int k_cols = cols;
    cl_int k_rows = rows;
    cl_int k_cal_npix = cal_npix;
    cl_int k_noise_npix = noise_npix;
    cl_int k_subtract = subtract_noise;
    status |= clSetKernelArg(e->kernel, arg++, sizeof(cl_int), &k_cols);
    status |= clSetKernelArg(e->kernel, arg++, sizeof(cl_int), &k_rows);
    status |= clSetKernelArg(e->kernel, arg++, sizeof(cl_int), &k_cal_npix);
    status |= clSetKernelArg(e->kernel, arg++, sizeof(cl_int), &k_noise_npix);
    status |= clSetKernelArg(e->kernel, arg++, sizeof(cl_int), &k_subtract);
    cl_check(status, "cannot set the kernel arguments");

    /* Round the global size up to the work-group multiple along the
     * contiguous axis; the kernel bounds-checks, so the padding is free. */
    size_t local[2] = {e->wg_multiple, 1};
    while (local[0] > 1 && (size_t)cols < local[0]) {
        local[0] /= 2;
    }
    size_t global[2] = {
        ((((size_t)cols) + local[0] - 1) / local[0]) * local[0], (size_t)rows
    };
    cl_check(
        clEnqueueNDRangeKernel(
            e->queue, e->kernel, 2, NULL, global, local, 0, NULL, NULL
        ),
        "cannot enqueue the s1calibrate kernel"
    );
    /* Event timestamps are wrong on some drivers, so the kernel is timed with
     * clFinish and the wall clock. */
    cl_check(clFinish(e->queue), "the s1calibrate kernel failed");
    double t2 = now_seconds();

    cl_get(e, e->linear, linear, pixels * sizeof(cl_float));
    cl_get(e, e->valid, valid, pixels * sizeof(cl_uchar));
    double t3 = now_seconds();

    e->t_upload += t1 - t0;
    e->t_kernel += t2 - t1;
    e->t_download += t3 - t2;
    e->blocks++;
}

static void cl_engine_report(const ClEngine *e)
{
    if (!e->timing || e->blocks == 0) {
        return;
    }
    double total = e->t_upload + e->t_kernel + e->t_download;
    double moved = e->bytes_up + e->bytes_down;
    fprintf(
        stderr,
        "s1calibrate: OpenCL timing over %ld block(s): upload %.3f s, kernel "
        "%.3f s, download %.3f s, total %.3f s\n",
        e->blocks,
        e->t_upload,
        e->t_kernel,
        e->t_download,
        total
    );
    fprintf(
        stderr,
        "s1calibrate: moved %.1f MiB (%.1f up, %.1f down); %.2f GB/s end to end, "
        "%.2f GB/s during the kernel\n",
        moved / 1048576.0,
        e->bytes_up / 1048576.0,
        e->bytes_down / 1048576.0,
        total > 0.0 ? moved / total / 1e9 : 0.0,
        e->t_kernel > 0.0 ? moved / e->t_kernel / 1e9 : 0.0
    );
}

static void cl_engine_free(ClEngine *e)
{
    cl_mem all[18] = {
        e->dn,         e->azimuth,    e->cal_lut,    e->noise_lut,
        e->cal_row0,   e->cal_row1,   e->cal_rowt,   e->noise_row0,
        e->noise_row1, e->noise_rowt, e->cal_col0,   e->cal_col1,
        e->cal_colt,   e->noise_col0, e->noise_col1, e->noise_colt,
        e->linear,     e->valid
    };
    for (int i = 0; i < 18; i++) {
        if (all[i] != NULL) {
            clReleaseMemObject(all[i]);
        }
    }
    if (e->kernel != NULL) {
        clReleaseKernel(e->kernel);
    }
    if (e->program != NULL) {
        clReleaseProgram(e->program);
    }
    if (e->queue != NULL) {
        clReleaseCommandQueue(e->queue);
    }
    if (e->context != NULL) {
        clReleaseContext(e->context);
    }
    free(e->h_cal_row0);
    free(e->h_cal_row1);
    free(e->h_cal_rowt);
    free(e->h_noise_row0);
    free(e->h_noise_row1);
    free(e->h_noise_rowt);
    free(e->h_cal_col0);
    free(e->h_cal_col1);
    free(e->h_cal_colt);
    free(e->h_noise_col0);
    free(e->h_noise_col1);
    free(e->h_noise_colt);
}

#endif /* S1CAL_HAVE_OPENCL */

/* Move GCPs from source pixel coordinates into window coordinates. */
static void shift_gcps(GDAL_GCP *gcps, int count, int off_x, int off_y)
{
    for (int i = 0; i < count; i++) {
        gcps[i].dfGCPPixel -= (double)off_x;
        gcps[i].dfGCPLine -= (double)off_y;
    }
}

static int calibrate(
    const char *measurement_path,
    const char *calibration_path,
    const char *noise_path,
    const char *output_path,
    const char *mask_path,
    const char *lut_name,
    int subtract_noise,
    int memory_mb,
    int threads,
    int use_opencl,
    int opencl_device,
    const char *opencl_profile,
    int opencl_timing,
    int win_x,
    int win_y,
    int win_w,
    int win_h
)
{
#ifndef S1CAL_HAVE_OPENCL
    if (use_opencl) {
        die("this s1calibrate was built without OpenCL; rebuild with S1CAL_HAVE_OPENCL");
    }
    (void)opencl_device;
    (void)opencl_profile;
    (void)opencl_timing;
#endif
    Cal cal;
    memset(&cal, 0, sizeof(cal));
    load_calibration(calibration_path, &cal);

    Noise noise;
    memset(&noise, 0, sizeof(noise));
    if (subtract_noise) {
        load_noise(noise_path, &noise);
    } else {
        noise.nlines = 1;
        noise.npix = 1;
        noise.lines = xmalloc(sizeof(double));
        noise.pixels = xmalloc(sizeof(double));
        noise.lut = xmalloc(sizeof(double));
        noise.lines[0] = 0.0;
        noise.pixels[0] = 0.0;
        noise.lut[0] = 0.0;
        noise.naz = 0;
    }

    double *lut = select_lut(&cal, lut_name);

    GDALAllRegister();
    GDALDatasetH source = GDALOpen(measurement_path, GA_ReadOnly);
    if (source == NULL) {
        die("cannot open measurement raster");
    }
    GDALRasterBandH source_band = GDALGetRasterBand(source, 1);
    if (source_band == NULL) {
        die("measurement raster has no band 1");
    }
    if (GDALGetRasterDataType(source_band) != GDT_UInt16) {
        die("measurement raster is not uint16");
    }
    int src_width = GDALGetRasterXSize(source);
    int src_height = GDALGetRasterYSize(source);

    /* The window is the part of the swath the caller actually needs. An IW GRD
     * band is about 250 km square while a city-scale area of interest is a few
     * tens of kilometres, so reading the whole band moves roughly 25 times more
     * data than the product requires. Everything below indexes the calibration
     * and noise LUTs by absolute source line and pixel, so a window is only a
     * matter of adding the offset; the LUT values themselves do not move. */
    int off_x = win_x > 0 ? win_x : 0;
    int off_y = win_y > 0 ? win_y : 0;
    if (off_x >= src_width || off_y >= src_height) {
        die("window origin is outside the measurement raster");
    }
    int width = win_w > 0 ? win_w : src_width - off_x;
    int height = win_h > 0 ? win_h : src_height - off_y;
    if (off_x + width > src_width) {
        width = src_width - off_x;
    }
    if (off_y + height > src_height) {
        height = src_height - off_y;
    }
    if (width <= 0 || height <= 0) {
        die("window does not overlap the measurement raster");
    }
    if (width != src_width || height != src_height) {
        fprintf(
            stderr,
            "s1calibrate: window %d+%d x %d+%d of %d x %d (%.1f %% of the band)\n",
            width, off_x, height, off_y, src_width, src_height,
            100.0 * ((double)width * height) / ((double)src_width * src_height)
        );
    }

    int gcp_count = GDALGetGCPCount(source);
    if (gcp_count < 6) {
        die("measurement raster has fewer than six GCPs");
    }
    const GDAL_GCP *source_gcps = GDALGetGCPs(source);
    const char *gcp_projection = GDALGetGCPProjection(source);

    GDALDriverH driver = GDALGetDriverByName("GTiff");
    if (driver == NULL) {
        die("GTiff driver not available");
    }
    char **options = NULL;
    options = CSLSetNameValue(options, "TILED", "YES");
    options = CSLSetNameValue(options, "COMPRESS", "NONE");
    options = CSLSetNameValue(options, "BIGTIFF", "YES");
    GDALDatasetH output = GDALCreate(
        driver, output_path, width, height, 1, GDT_Float32, options
    );
    if (output == NULL) {
        die("cannot create calibrated raster");
    }
    /* The GCPs are expressed in source pixel coordinates, so a window shifts
     * them by its origin. They are kept whole rather than culled to the window:
     * gdalwarp accepts GCPs outside the raster, and dropping the distant ones
     * would change the polynomial that every earlier product was warped with. */
    GDAL_GCP *output_gcps = GDALDuplicateGCPs(gcp_count, source_gcps);
    shift_gcps(output_gcps, gcp_count, off_x, off_y);
    if (GDALSetGCPs(output, gcp_count, output_gcps, gcp_projection) != CE_None) {
        die("cannot set GCPs on calibrated raster");
    }

    char **mask_options = NULL;
    mask_options = CSLSetNameValue(mask_options, "TILED", "YES");
    mask_options = CSLSetNameValue(mask_options, "COMPRESS", "NONE");
    mask_options = CSLSetNameValue(mask_options, "BIGTIFF", "YES");
    GDALDatasetH mask = GDALCreate(
        driver, mask_path, width, height, 1, GDT_Byte, mask_options
    );
    if (mask == NULL) {
        die("cannot create mask raster");
    }
    GDAL_GCP *mask_gcps = GDALDuplicateGCPs(gcp_count, source_gcps);
    shift_gcps(mask_gcps, gcp_count, off_x, off_y);
    if (GDALSetGCPs(mask, gcp_count, mask_gcps, gcp_projection) != CE_None) {
        die("cannot set GCPs on mask raster");
    }
    GDALRasterBandH output_band = GDALGetRasterBand(output, 1);
    GDALRasterBandH mask_band = GDALGetRasterBand(mask, 1);
    /* NaN for the float band and 255 for the mask: neither can be produced by
     * the calibration, so nodata never collides with a measurement. 0 in the
     * mask means "invalid", which is information, not absence of data. */
    if (GDALSetRasterNoDataValue(output_band, (double)NAN) != CE_None) {
        die("cannot set nodata on calibrated raster");
    }
    if (GDALSetRasterNoDataValue(mask_band, 255.0) != CE_None) {
        die("cannot set nodata on mask raster");
    }

    int side = block_side(memory_mb, width, height);
    size_t block_pixels = (size_t)side * (size_t)side;
    uint16_t *dn = xmalloc(block_pixels * sizeof(uint16_t));
    float *azimuth = xmalloc(block_pixels * sizeof(float));
    float *linear = xmalloc(block_pixels * sizeof(float));
    uint8_t *valid = xmalloc(block_pixels);
    int *cal_pi0 = xmalloc((size_t)side * sizeof(int));
    int *cal_pi1 = xmalloc((size_t)side * sizeof(int));
    double *cal_pt = xmalloc((size_t)side * sizeof(double));
    int *noise_pi0 = xmalloc((size_t)side * sizeof(int));
    int *noise_pi1 = xmalloc((size_t)side * sizeof(int));
    double *noise_pt = xmalloc((size_t)side * sizeof(double));

    omp_set_num_threads(threads);

#ifdef S1CAL_HAVE_OPENCL
    ClEngine engine;
    float *cal_lut_f = NULL;
    float *noise_lut_f = NULL;
    if (use_opencl) {
        /* The kernel works in single precision, so the LUTs are converted
         * once per scene rather than per block. */
        size_t cal_count = (size_t)cal.nlines * (size_t)cal.npix;
        size_t noise_count = (size_t)noise.nlines * (size_t)noise.npix;
        cal_lut_f = xmalloc(cal_count * sizeof(float));
        noise_lut_f = xmalloc(noise_count * sizeof(float));
        for (size_t i = 0; i < cal_count; i++) {
            cal_lut_f[i] = (float)lut[i];
        }
        for (size_t i = 0; i < noise_count; i++) {
            noise_lut_f[i] = (float)noise.lut[i];
        }
        cl_engine_init(
            &engine,
            opencl_device,
            opencl_profile,
            opencl_timing,
            side,
            block_pixels,
            cal_lut_f,
            cal.nlines,
            cal.npix,
            noise_lut_f,
            noise.nlines,
            noise.npix
        );
    }
#endif

    for (int row = 0; row < height; row += side) {
        int rows = side < height - row ? side : height - row;
        for (int col = 0; col < width; col += side) {
            int cols = side < width - col ? side : width - col;
            if (GDALRasterIO(
                    source_band,
                    GF_Read,
                    off_x + col,
                    off_y + row,
                    cols,
                    rows,
                    dn,
                    cols,
                    rows,
                    GDT_UInt16,
                    0,
                    0
                ) != CE_None) {
                die("cannot read measurement block");
            }
            for (int c = 0; c < cols; c++) {
                find_interval(
                    cal.pixels,
                    cal.npix,
                    (double)(off_x + col + c),
                    &cal_pi0[c],
                    &cal_pi1[c],
                    &cal_pt[c]
                );
                find_interval(
                    noise.pixels,
                    noise.npix,
                    (double)(off_x + col + c),
                    &noise_pi0[c],
                    &noise_pi1[c],
                    &noise_pt[c]
                );
            }
            fill_azimuth(&noise, off_y + row, off_x + col, cols, rows, azimuth);

#ifdef S1CAL_HAVE_OPENCL
            if (use_opencl) {
                for (int c = 0; c < cols; c++) {
                    engine.h_cal_col0[c] = cal_pi0[c];
                    engine.h_cal_col1[c] = cal_pi1[c];
                    engine.h_cal_colt[c] = (float)cal_pt[c];
                    engine.h_noise_col0[c] = noise_pi0[c];
                    engine.h_noise_col1[c] = noise_pi1[c];
                    engine.h_noise_colt[c] = (float)noise_pt[c];
                }
                for (int r = 0; r < rows; r++) {
                    int ci0, ci1, ni0, ni1;
                    double ct, nt;
                    find_interval(
                        cal.lines, cal.nlines, (double)(off_y + row + r),
                        &ci0, &ci1, &ct
                    );
                    find_interval(
                        noise.lines, noise.nlines, (double)(off_y + row + r),
                        &ni0, &ni1, &nt
                    );
                    engine.h_cal_row0[r] = ci0;
                    engine.h_cal_row1[r] = ci1;
                    engine.h_cal_rowt[r] = (float)ct;
                    engine.h_noise_row0[r] = ni0;
                    engine.h_noise_row1[r] = ni1;
                    engine.h_noise_rowt[r] = (float)nt;
                }
                cl_engine_run(
                    &engine,
                    dn,
                    azimuth,
                    cols,
                    rows,
                    cal.npix,
                    noise.npix,
                    subtract_noise,
                    linear,
                    valid
                );
            }
#endif
            if (!use_opencl) {
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < rows; r++) {
                int line = off_y + row + r;
                int ci0, ci1;
                double ct;
                find_interval(cal.lines, cal.nlines, (double)line, &ci0, &ci1, &ct);
                int ni0, ni1;
                double nt;
                find_interval(noise.lines, noise.nlines, (double)line, &ni0, &ni1, &nt);
                const double *cal0 = lut + (size_t)ci0 * cal.npix;
                const double *cal1 = lut + (size_t)ci1 * cal.npix;
                const double *noise0 = noise.lut + (size_t)ni0 * noise.npix;
                const double *noise1 = noise.lut + (size_t)ni1 * noise.npix;
                size_t base = (size_t)r * (size_t)cols;
                for (int c = 0; c < cols; c++) {
                    int cpi0 = cal_pi0[c];
                    int cpi1 = cal_pi1[c];
                    double cpt = cal_pt[c];
                    double cal_value =
                        (cal0[cpi0] + (cal0[cpi1] - cal0[cpi0]) * cpt) * (1.0 - ct) +
                        (cal1[cpi0] + (cal1[cpi1] - cal1[cpi0]) * cpt) * ct;
                    int npi0 = noise_pi0[c];
                    int npi1 = noise_pi1[c];
                    double npt = noise_pt[c];
                    double noise_value =
                        (noise0[npi0] + (noise0[npi1] - noise0[npi0]) * npt) * (1.0 - nt) +
                        (noise1[npi0] + (noise1[npi1] - noise1[npi0]) * npt) * nt;
                    double value = (double)dn[base + (size_t)c];
                    double dn2 = value * value;
                    double noise_power =
                        noise_value * (double)azimuth[base + (size_t)c];
                    double signal = dn2 - noise_power;
                    if (signal < 0.0) {
                        signal = 0.0;
                    }
                    if (value <= 0.0) {
                        /* DN 0 is outside the swath, not a measurement of zero
                         * backscatter. It is written as NaN, the band's nodata,
                         * so that the warp can drop it while a pixel the noise
                         * subtraction drove to the floor stays a real 0 and
                         * reaches the dB floor instead of becoming nodata. */
                        linear[base + (size_t)c] = NAN;
                        valid[base + (size_t)c] = (uint8_t)0;
                    }
                    else {
                        linear[base + (size_t)c] =
                            (float)(signal / (cal_value * cal_value));
                        valid[base + (size_t)c] =
                            (signal > 0.0) ? (uint8_t)1 : (uint8_t)0;
                    }
                }
            }
            }

            if (GDALRasterIO(
                    output_band,
                    GF_Write,
                    col,
                    row,
                    cols,
                    rows,
                    linear,
                    cols,
                    rows,
                    GDT_Float32,
                    0,
                    0
                ) != CE_None) {
                die("cannot write calibrated block");
            }
            if (GDALRasterIO(
                    mask_band,
                    GF_Write,
                    col,
                    row,
                    cols,
                    rows,
                    valid,
                    cols,
                    rows,
                    GDT_Byte,
                    0,
                    0
                ) != CE_None) {
                die("cannot write mask block");
            }
        }
    }

#ifdef S1CAL_HAVE_OPENCL
    if (use_opencl) {
        cl_engine_report(&engine);
        cl_engine_free(&engine);
        free(cal_lut_f);
        free(noise_lut_f);
    }
#endif

    GDALClose(output);
    GDALClose(mask);
    GDALClose(source);
    CSLDestroy(options);
    CSLDestroy(mask_options);
    free(dn);
    free(azimuth);
    free(linear);
    free(valid);
    free(cal_pi0);
    free(cal_pi1);
    free(cal_pt);
    free(noise_pi0);
    free(noise_pi1);
    free(noise_pt);
    free(cal.lines);
    free(cal.pixels);
    free(cal.sigma);
    free(cal.beta);
    free(cal.gamma);
    free(noise.lines);
    free(noise.pixels);
    free(noise.lut);
    free(noise.az);
    return 0;
}

static void usage(void)
{
    fprintf(
        stderr,
        "Usage: s1calibrate --measurement FILE --calibration XML --noise XML "
        "--output FILE --mask FILE [options]\n"
        "  --lut gamma|sigma|beta        calibration LUT (default gamma)\n"
        "  --subtract-noise 0|1          thermal-noise removal (default 1)\n"
        "  --memory-mb N                 block memory budget (default 2048)\n"
        "  --threads N                   OpenMP threads (default all)\n"
        "  --opencl                      run the arithmetic on an OpenCL device\n"
        "  --opencl-device N             device index (default: first GPU)\n"
        "  --opencl-list                 list the OpenCL devices and exit\n"
        "  --profile auto|cpu|igpu|dgpu|egpu\n"
        "                                device profile; auto detects it\n"
        "  --opencl-timing               report transfer and kernel time\n"
        "  --window X,Y,W,H              calibrate only this source window;\n"
        "                                the default is the whole band\n"
        "  --version                     print version\n"
        "\n"
        "The OpenCL engine works in single precision and the CPU engine in\n"
        "double, so the two agree to about float epsilon rather than exactly.\n"
        "S1CAL_OPENCL_PROFILE=1 is equivalent to --opencl-timing, and\n"
        "S1CAL_KERNEL overrides the path of s1calibrate.cl.\n"
    );
}

int main(int argc, char **argv)
{
    const char *measurement = NULL;
    const char *calibration = NULL;
    const char *noise = NULL;
    const char *output = NULL;
    const char *mask = NULL;
    const char *lut = "gamma";
    int subtract_noise = 1;
    int memory_mb = 2048;
    int threads = omp_get_max_threads();
    int use_opencl = 0;
    int opencl_device = -1;
    const char *opencl_profile = "auto";
    int opencl_timing = getenv("S1CAL_OPENCL_PROFILE") != NULL;
    int win_x = 0, win_y = 0, win_w = 0, win_h = 0;

    static const struct option long_options[] = {
        {"measurement", required_argument, NULL, 'm'},
        {"calibration", required_argument, NULL, 'c'},
        {"noise", required_argument, NULL, 'n'},
        {"output", required_argument, NULL, 'o'},
        {"mask", required_argument, NULL, 'k'},
        {"lut", required_argument, NULL, 'l'},
        {"subtract-noise", required_argument, NULL, 's'},
        {"memory-mb", required_argument, NULL, 'M'},
        {"threads", required_argument, NULL, 't'},
        {"opencl", no_argument, NULL, 'O'},
        {"opencl-device", required_argument, NULL, 'D'},
        {"opencl-list", no_argument, NULL, 'L'},
        {"profile", required_argument, NULL, 'p'},
        {"opencl-timing", no_argument, NULL, 'T'},
        {"window", required_argument, NULL, 'W'},
        {"version", no_argument, NULL, 'v'},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0}
    };

    int option;
    while ((option = getopt_long(argc, argv, "", long_options, NULL)) != -1) {
        switch (option) {
            case 'm':
                measurement = optarg;
                break;
            case 'c':
                calibration = optarg;
                break;
            case 'n':
                noise = optarg;
                break;
            case 'o':
                output = optarg;
                break;
            case 'k':
                mask = optarg;
                break;
            case 'l':
                lut = optarg;
                break;
            case 's':
                subtract_noise = atoi(optarg);
                break;
            case 'M':
                memory_mb = atoi(optarg);
                break;
            case 't':
                threads = atoi(optarg);
                break;
            case 'W':
                if (sscanf(optarg, "%d,%d,%d,%d", &win_x, &win_y, &win_w, &win_h)
                    != 4) {
                    die("--window takes X,Y,W,H");
                }
                if (win_x < 0 || win_y < 0 || win_w <= 0 || win_h <= 0) {
                    die("--window needs a non-negative origin and a positive size");
                }
                break;
            case 'O':
                use_opencl = 1;
                break;
            case 'D':
                opencl_device = atoi(optarg);
                use_opencl = 1;
                break;
            case 'L':
#ifdef S1CAL_HAVE_OPENCL
                cl_list_devices();
                return EXIT_SUCCESS;
#else
                die("this s1calibrate was built without OpenCL");
#endif
            case 'p':
                opencl_profile = optarg;
                break;
            case 'T':
                opencl_timing = 1;
                break;
            case 'v':
                printf("%s\n", VERSION);
                return EXIT_SUCCESS;
            case 'h':
                usage();
                return EXIT_SUCCESS;
            default:
                usage();
                return EXIT_FAILURE;
        }
    }

    if (measurement == NULL || calibration == NULL || output == NULL || mask == NULL) {
        usage();
        return EXIT_FAILURE;
    }
    if (subtract_noise && noise == NULL) {
        die("--noise is required when --subtract-noise is 1");
    }
    if (memory_mb < 64) {
        die("--memory-mb must be at least 64");
    }
    if (threads < 1) {
        threads = 1;
    }
    if (strcmp(opencl_profile, "auto") != 0 && strcmp(opencl_profile, "cpu") != 0
        && strcmp(opencl_profile, "igpu") != 0 && strcmp(opencl_profile, "dgpu") != 0
        && strcmp(opencl_profile, "egpu") != 0) {
        die("--profile must be auto, cpu, igpu, dgpu or egpu");
    }
    if (!use_opencl && strcmp(opencl_profile, "auto") != 0) {
        die("--profile applies to the OpenCL engine; add --opencl");
    }

    return calibrate(
        measurement,
        calibration,
        noise,
        output,
        mask,
        lut,
        subtract_noise,
        memory_mb,
        threads,
        use_opencl,
        opencl_device,
        opencl_profile,
        opencl_timing,
        win_x,
        win_y,
        win_w,
        win_h
    );
}
