CC ?= cc
CFLAGS ?= -O3 -march=native -Wall -Wextra
CFLAGS += -fopenmp
LDLIBS += $(shell pkg-config --libs gdal expat) -lm

s1calibrate: s1calibrate.c
	$(CC) $(CFLAGS) $(shell pkg-config --cflags gdal expat) -o $@ $< $(LDLIBS)

clean:
	rm -f s1calibrate

.PHONY: clean
