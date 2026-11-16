# Build the Sentinel-1 calibration engine.
#
# SPDX-License-Identifier: Unlicense
#
# The OpenCL engine is optional and is built when the OpenCL headers and
# library are present. Without them the binary still builds, and --opencl then
# fails loudly rather than falling back to the CPU without saying so.

CC ?= cc
CFLAGS ?= -O3 -march=native -Wall -Wextra
CFLAGS += -fopenmp
GDAL_CFLAGS := $(shell pkg-config --cflags gdal expat)
GDAL_LIBS := $(shell pkg-config --libs gdal expat)

HAVE_OPENCL := $(shell pkg-config --exists OpenCL && echo yes)
ifeq ($(HAVE_OPENCL),yes)
CFLAGS += -DS1CAL_HAVE_OPENCL $(shell pkg-config --cflags OpenCL)
OPENCL_LIBS := $(shell pkg-config --libs OpenCL)
else
OPENCL_LIBS :=
endif

s1calibrate: s1calibrate.c
	$(CC) $(CFLAGS) $(GDAL_CFLAGS) -o $@ $< $(GDAL_LIBS) $(OPENCL_LIBS) -lm
ifeq ($(HAVE_OPENCL),yes)
	@echo "s1calibrate: OpenCL engine enabled"
else
	@echo "s1calibrate: OpenCL not found, CPU engine only"
endif

# Without OpenCL even when it is installed, for comparison.
cpu-only: s1calibrate.c
	$(CC) $(CFLAGS) $(GDAL_CFLAGS) -o s1calibrate $< $(GDAL_LIBS) -lm

# The kernel is read at run time from beside the binary, so a plain make
# leaves them together; the GRASS build installs both into the addon bin.
install-kernel: s1calibrate.cl
	@test -n "$(DESTDIR)" || { echo "set DESTDIR to the install directory"; exit 1; }
	install -m 0644 s1calibrate.cl $(DESTDIR)/s1calibrate.cl

clean:
	rm -f s1calibrate

.PHONY: clean cpu-only install-kernel
