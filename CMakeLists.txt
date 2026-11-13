cmake_minimum_required(VERSION 3.22)

project(i.sentinel1.preproc LANGUAGES C)

include(build_addon)

build_addon(
  ${PROJECT_NAME}
  SOURCES ${PROJECT_NAME}.py
  DOCFILES
    ${PROJECT_NAME}.md)

option(S1CAL_BUILD_ENGINE "Build the C+OpenMP calibration engine" ON)

if(S1CAL_BUILD_ENGINE)
  find_package(GDAL QUIET)
  find_package(EXPAT QUIET)
  find_package(OpenMP QUIET COMPONENTS C)

  if(GDAL_FOUND AND EXPAT_FOUND AND OpenMP_C_FOUND)
    add_executable(s1calibrate src/s1calibrate.c)
    target_link_libraries(s1calibrate PRIVATE
      GDAL::GDAL
      EXPAT::EXPAT
      OpenMP::OpenMP_C
      m)
    set_target_properties(s1calibrate PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${OUTDIR}/${GRASSAddon_BinDIR})
    install(TARGETS s1calibrate RUNTIME DESTINATION ${GRASSAddon_BinDIR})
    message(STATUS "s1calibrate: C+OpenMP engine enabled")
  else()
    message(WARNING "s1calibrate not built: need GDAL, expat and OpenMP development files")
  endif()

  # OpenCL is optional; the host integration is tracked in TODO.md. When it
  # lands, link OpenCL::OpenCL here and gate the kernel behind --opencl.
endif()
