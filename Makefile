MODULE_TOPDIR = ../..

PGM = i.sentinel1.preproc

# sentinel1_core is the numeric reference the module imports; Script.make
# installs it into $(ETC)/$(PGM)/, which grass.script.utils.set_path finds.
ETCFILES = sentinel1_core

include $(MODULE_TOPDIR)/include/Make/Script.make
include $(MODULE_TOPDIR)/include/Make/Python.make

default: script
