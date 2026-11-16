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

    # The OpenCL engine is optional. Without it the binary still builds and
    # --opencl fails loudly, rather than falling back to the CPU in silence.
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
      target_compile_definitions(s1calibrate PRIVATE S1CAL_HAVE_OPENCL)
      target_link_libraries(s1calibrate PRIVATE OpenCL::OpenCL)
      # The kernel is read at run time, so it is installed beside the binary;
      # s1calibrate looks there, then in ../etc and ../share, then at
      # S1CAL_KERNEL.
      configure_file(src/s1calibrate.cl
        ${OUTDIR}/${GRASSAddon_BinDIR}/s1calibrate.cl COPYONLY)
      install(FILES src/s1calibrate.cl DESTINATION ${GRASSAddon_BinDIR})
      message(STATUS "s1calibrate: OpenCL engine enabled")
    else()
      message(STATUS "s1calibrate: OpenCL not found, CPU engine only")
    endif()
  else()
    message(WARNING "s1calibrate not built: need GDAL, expat and OpenMP development files")
  endif()
endif()
