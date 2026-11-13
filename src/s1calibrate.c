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

#include <cpl_conv.h>
#include <cpl_error.h>
#include <cpl_string.h>
#include <gdal.h>

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

static char *read_file(const char *path, size_t *length)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "s1calibrate: cannot open %s: %s\n", path, strerror(errno));
        exit(EXIT_FAILURE);
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        die("cannot seek XML file");
    }
    long size = ftell(f);
    if (size < 0) {
        die("cannot size XML file");
    }
    rewind(f);
    char *buffer = xmalloc((size_t)size + 1);
    size_t read = fread(buffer, 1, (size_t)size, f);
    fclose(f);
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

static int calibrate(
    const char *measurement_path,
    const char *calibration_path,
    const char *noise_path,
    const char *output_path,
    const char *mask_path,
    const char *lut_name,
    int subtract_noise,
    int memory_mb,
    int threads
)
{
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
    int width = GDALGetRasterXSize(source);
    int height = GDALGetRasterYSize(source);
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
    GDAL_GCP *output_gcps = GDALDuplicateGCPs(gcp_count, source_gcps);
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
    if (GDALSetGCPs(mask, gcp_count, mask_gcps, gcp_projection) != CE_None) {
        die("cannot set GCPs on mask raster");
    }
    GDALRasterBandH output_band = GDALGetRasterBand(output, 1);
    GDALRasterBandH mask_band = GDALGetRasterBand(mask, 1);

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

    for (int row = 0; row < height; row += side) {
        int rows = side < height - row ? side : height - row;
        for (int col = 0; col < width; col += side) {
            int cols = side < width - col ? side : width - col;
            if (GDALRasterIO(
                    source_band,
                    GF_Read,
                    col,
                    row,
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
                    (double)(col + c),
                    &cal_pi0[c],
                    &cal_pi1[c],
                    &cal_pt[c]
                );
                find_interval(
                    noise.pixels,
                    noise.npix,
                    (double)(col + c),
                    &noise_pi0[c],
                    &noise_pi1[c],
                    &noise_pt[c]
                );
            }
            fill_azimuth(&noise, row, col, cols, rows, azimuth);

            #pragma omp parallel for schedule(static)
            for (int r = 0; r < rows; r++) {
                int line = row + r;
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
                    double calibrated = signal / (cal_value * cal_value);
                    linear[base + (size_t)c] = (float)calibrated;
                    valid[base + (size_t)c] =
                        (value > 0.0 && signal > 0.0) ? (uint8_t)1 : (uint8_t)0;
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
        "  --version                     print version\n"
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

    return calibrate(
        measurement,
        calibration,
        noise,
        output,
        mask,
        lut,
        subtract_noise,
        memory_mb,
        threads
    );
}
