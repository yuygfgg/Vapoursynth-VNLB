cmake_minimum_required(VERSION 3.24)

function(vnlb_read_project_version out_var)
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" _VNLB_PYPROJECT)
  string(REGEX MATCH "(^|\n)version[ \t]*=[ \t]*\"([0-9]+\\.[0-9]+\\.[0-9]+)\""
    _VNLB_PROJECT_VERSION_MATCH
    "${_VNLB_PYPROJECT}"
  )
  if(NOT _VNLB_PROJECT_VERSION_MATCH)
    message(FATAL_ERROR "Could not read numeric [project].version from pyproject.toml")
  endif()
  set(${out_var} "${CMAKE_MATCH_2}" PARENT_SCOPE)
endfunction()

vnlb_read_project_version(VNLB_PROJECT_VERSION)

project(
  VapourSynthVNLB
  VERSION "${VNLB_PROJECT_VERSION}"
  DESCRIPTION "Video Non-Local Bayes denoising plugin for VapourSynth"
  LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vnlb_version.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/vnlb_version.hpp"
  @ONLY
)

option(VNLB_BUILD_PLUGIN "Build the VapourSynth native plugin target" OFF)
option(VNLB_ENABLE_TESTS "Enable project tests" ON)
option(VNLB_ENABLE_LAPACK "Enable a LAPACK-compatible eigensolver backend" ON)
option(VNLB_ENABLE_ACCELERATE "Enable the Accelerate BLAS/LAPACK backend on Apple platforms" ON)
option(VNLB_ENABLE_WARNINGS "Enable project warning flags" ON)
option(VNLB_ENABLE_WARNINGS_AS_ERRORS "Treat project warnings as errors" OFF)

set(_VNLB_RELEASE_OPTIMIZATIONS_DEFAULT OFF)
string(TOUPPER "${CMAKE_BUILD_TYPE}" _VNLB_BUILD_TYPE_UPPER)
if(CMAKE_CONFIGURATION_TYPES OR _VNLB_BUILD_TYPE_UPPER STREQUAL "RELEASE")
  set(_VNLB_RELEASE_OPTIMIZATIONS_DEFAULT ON)
endif()

option(VNLB_USE_LTO "Enable link time optimization for Release builds" ${_VNLB_RELEASE_OPTIMIZATIONS_DEFAULT})
set(VNLB_LTO_MODE "full" CACHE STRING "LTO mode for Clang-like Release builds: full or thin")
set_property(CACHE VNLB_LTO_MODE PROPERTY STRINGS full thin)
option(VNLB_ENABLE_MARCH_NATIVE "Enable -march=native for Release builds" ${_VNLB_RELEASE_OPTIMIZATIONS_DEFAULT})

cmake_policy(SET CMP0069 NEW)
if(VNLB_USE_LTO)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(NOT VNLB_LTO_MODE MATCHES "^(full|thin)$")
      message(FATAL_ERROR "VNLB_LTO_MODE must be either full or thin")
    endif()
    set(CMAKE_CXX_COMPILE_OPTIONS_IPO "-flto=${VNLB_LTO_MODE}")
    set(CMAKE_CXX_LINK_OPTIONS_IPO "-flto=${VNLB_LTO_MODE}")
  endif()
else()
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE FALSE)
endif()
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_DEBUG FALSE)

include(VNLBCompilerOptions)

add_library(vnlb_project_options INTERFACE)
target_compile_features(vnlb_project_options INTERFACE cxx_std_23)
vnlb_apply_compiler_options(vnlb_project_options)
if(VNLB_ENABLE_MARCH_NATIVE AND NOT MSVC)
  target_compile_options(vnlb_project_options INTERFACE
    $<$<CONFIG:Release>:-march=native>
  )
endif()

set(VNLB_HIGHWAY_PROVIDER)
find_package(hwy CONFIG QUIET)
if(TARGET hwy::hwy)
  add_library(vnlb_highway INTERFACE)
  get_target_property(_VNLB_HIGHWAY_INCLUDE_DIRS
    hwy::hwy INTERFACE_INCLUDE_DIRECTORIES
  )
  if(_VNLB_HIGHWAY_INCLUDE_DIRS)
    target_include_directories(vnlb_highway SYSTEM INTERFACE
      ${_VNLB_HIGHWAY_INCLUDE_DIRS}
    )
  endif()

  get_target_property(_VNLB_HIGHWAY_COMPILE_DEFINITIONS
    hwy::hwy INTERFACE_COMPILE_DEFINITIONS
  )
  if(_VNLB_HIGHWAY_COMPILE_DEFINITIONS)
    target_compile_definitions(vnlb_highway INTERFACE
      ${_VNLB_HIGHWAY_COMPILE_DEFINITIONS}
    )
  endif()

  get_target_property(_VNLB_HIGHWAY_COMPILE_FEATURES
    hwy::hwy INTERFACE_COMPILE_FEATURES
  )
  if(_VNLB_HIGHWAY_COMPILE_FEATURES)
    target_compile_features(vnlb_highway INTERFACE
      ${_VNLB_HIGHWAY_COMPILE_FEATURES}
    )
  endif()

  set(VNLB_HIGHWAY_PROVIDER "hwy::hwy")
else()
  find_path(VNLB_HIGHWAY_INCLUDE_DIR
    NAMES hwy/highway.h
  )
  mark_as_advanced(VNLB_HIGHWAY_INCLUDE_DIR)
  if(NOT VNLB_HIGHWAY_INCLUDE_DIR)
    message(FATAL_ERROR
      "Highway is required. Install the hwy package or set "
      "VNLB_HIGHWAY_INCLUDE_DIR to a directory containing hwy/highway.h."
    )
  endif()

  add_library(vnlb_highway INTERFACE)
  target_include_directories(vnlb_highway SYSTEM INTERFACE
    "${VNLB_HIGHWAY_INCLUDE_DIR}"
  )
  set(VNLB_HIGHWAY_PROVIDER "${VNLB_HIGHWAY_INCLUDE_DIR}")
endif()

function(vnlb_collect_openblas_roots out_var)
  set(_roots)
  foreach(_var IN ITEMS OpenBLAS_ROOT OPENBLAS_ROOT)
    if(DEFINED ${_var})
      list(APPEND _roots "${${_var}}")
    endif()
    if(DEFINED ENV{${_var}})
      list(APPEND _roots "$ENV{${_var}}")
    endif()
  endforeach()

  if(APPLE)
    list(APPEND _roots
      "/opt/homebrew/opt/openblas"
      "/usr/local/opt/openblas"
    )
  endif()
  list(APPEND _roots
    "/opt/OpenBLAS"
    "/opt/openblas"
    "/opt/local"
    "/usr/local"
    "/usr"
  )
  list(REMOVE_DUPLICATES _roots)
  set(${out_var} ${_roots} PARENT_SCOPE)
endfunction()

function(vnlb_make_openblas_target_from_vars out_var)
  set(_includes)
  foreach(_candidate IN LISTS
      OpenBLAS_INCLUDE_DIRS
      OpenBLAS_INCLUDE_DIR
      OPENBLAS_INCLUDE_DIRS
      OPENBLAS_INCLUDE_DIR
  )
    if(_candidate)
      list(APPEND _includes "${_candidate}")
    endif()
  endforeach()

  set(_libraries)
  foreach(_candidate IN LISTS
      OpenBLAS_LIBRARIES
      OpenBLAS_LIBRARY
      OPENBLAS_LIBRARIES
      OPENBLAS_LIBRARY
  )
    if(_candidate)
      list(APPEND _libraries "${_candidate}")
    endif()
  endforeach()

  if(_includes AND _libraries)
    list(REMOVE_DUPLICATES _includes)
    list(REMOVE_DUPLICATES _libraries)
    if(NOT TARGET vnlb_openblas)
      add_library(vnlb_openblas INTERFACE)
      target_include_directories(vnlb_openblas SYSTEM INTERFACE ${_includes})
      target_link_libraries(vnlb_openblas INTERFACE ${_libraries})
    endif()
    set(${out_var} vnlb_openblas PARENT_SCOPE)
  else()
    set(${out_var} "" PARENT_SCOPE)
  endif()
endfunction()

function(vnlb_find_openblas out_var)
  set(${out_var} "" PARENT_SCOPE)
  vnlb_collect_openblas_roots(_openblas_roots)

  find_package(OpenBLAS CONFIG QUIET HINTS ${_openblas_roots})
  if(TARGET OpenBLAS::OpenBLAS)
    set(${out_var} OpenBLAS::OpenBLAS PARENT_SCOPE)
    return()
  elseif(TARGET OpenBLAS::openblas)
    set(${out_var} OpenBLAS::openblas PARENT_SCOPE)
    return()
  elseif(OpenBLAS_FOUND)
    vnlb_make_openblas_target_from_vars(_openblas_target)
    if(_openblas_target)
      set(${out_var} ${_openblas_target} PARENT_SCOPE)
      return()
    endif()
  endif()

  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(VNLB_OPENBLAS_PC QUIET openblas)
  endif()

  set(_openblas_include_hints
    ${_openblas_roots}
    ${VNLB_OPENBLAS_PC_INCLUDE_DIRS}
  )
  find_path(VNLB_OPENBLAS_CBLAS_INCLUDE_DIR
    NAMES cblas.h
    HINTS ${_openblas_include_hints}
    PATH_SUFFIXES include include/openblas
  )
  find_path(VNLB_OPENBLAS_LAPACKE_INCLUDE_DIR
    NAMES lapacke.h
    HINTS ${_openblas_include_hints}
    PATH_SUFFIXES include include/openblas
  )

  set(_openblas_library_hints
    ${_openblas_roots}
    ${VNLB_OPENBLAS_PC_LIBRARY_DIRS}
  )
  find_library(VNLB_OPENBLAS_LIBRARY
    NAMES openblas openblasp
    HINTS ${_openblas_library_hints}
    PATH_SUFFIXES lib lib64
  )
  find_library(VNLB_LAPACKE_LIBRARY
    NAMES lapacke
    HINTS ${_openblas_library_hints}
    PATH_SUFFIXES lib lib64
  )

  mark_as_advanced(
    VNLB_OPENBLAS_CBLAS_INCLUDE_DIR
    VNLB_OPENBLAS_LAPACKE_INCLUDE_DIR
    VNLB_OPENBLAS_LIBRARY
    VNLB_LAPACKE_LIBRARY
  )

  if(VNLB_OPENBLAS_CBLAS_INCLUDE_DIR AND
      VNLB_OPENBLAS_LAPACKE_INCLUDE_DIR AND
      VNLB_OPENBLAS_LIBRARY)
    set(_openblas_includes
      "${VNLB_OPENBLAS_CBLAS_INCLUDE_DIR}"
      "${VNLB_OPENBLAS_LAPACKE_INCLUDE_DIR}"
    )
    list(REMOVE_DUPLICATES _openblas_includes)
    set(_openblas_libraries)
    if(VNLB_LAPACKE_LIBRARY)
      list(APPEND _openblas_libraries "${VNLB_LAPACKE_LIBRARY}")
    endif()
    list(APPEND _openblas_libraries "${VNLB_OPENBLAS_LIBRARY}")
    if(NOT TARGET vnlb_openblas)
      add_library(vnlb_openblas INTERFACE)
      target_include_directories(vnlb_openblas SYSTEM INTERFACE
        ${_openblas_includes}
      )
      target_link_libraries(vnlb_openblas INTERFACE
        ${_openblas_libraries}
      )
    endif()
    set(${out_var} vnlb_openblas PARENT_SCOPE)
  endif()
endfunction()

set(VNLB_LINALG_BACKEND_SOURCE)
set(VNLB_LINALG_BACKEND "handwritten")
set(VNLB_LINALG_BACKEND_LIBRARIES)
set(VNLB_LINALG_USE_ACCELERATE OFF)
set(VNLB_LINALG_USE_OPENBLAS OFF)
if(VNLB_ENABLE_LAPACK)
  if(APPLE AND VNLB_ENABLE_ACCELERATE)
    find_library(VNLB_ACCELERATE_FRAMEWORK Accelerate)
    mark_as_advanced(VNLB_ACCELERATE_FRAMEWORK)
    if(VNLB_ACCELERATE_FRAMEWORK)
      set(VNLB_LINALG_BACKEND "Accelerate")
      set(VNLB_LINALG_USE_ACCELERATE ON)
      set(VNLB_LINALG_BACKEND_SOURCE
        src/linalg/backend_blas_lapack.cpp
      )
      list(APPEND VNLB_LINALG_BACKEND_LIBRARIES
        "${VNLB_ACCELERATE_FRAMEWORK}"
      )
    endif()
  endif()

  if(NOT VNLB_LINALG_BACKEND_SOURCE)
    vnlb_find_openblas(VNLB_OPENBLAS_TARGET)
    if(VNLB_OPENBLAS_TARGET)
      set(VNLB_LINALG_BACKEND "OpenBLAS")
      set(VNLB_LINALG_USE_OPENBLAS ON)
      set(VNLB_LINALG_BACKEND_SOURCE
        src/linalg/backend_blas_lapack.cpp
      )
      list(APPEND VNLB_LINALG_BACKEND_LIBRARIES
        ${VNLB_OPENBLAS_TARGET}
      )
    endif()
  endif()
endif()

add_library(vnlb_linalg STATIC
  src/linalg/linalg.cpp
  ${VNLB_LINALG_BACKEND_SOURCE}
)
target_include_directories(vnlb_linalg
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(vnlb_linalg
  PUBLIC
    vnlb_project_options
    vnlb_highway
)
target_compile_definitions(vnlb_linalg
  PUBLIC
    VNLB_LINALG_USE_ACCELERATE=$<BOOL:${VNLB_LINALG_USE_ACCELERATE}>
    VNLB_LINALG_USE_OPENBLAS=$<BOOL:${VNLB_LINALG_USE_OPENBLAS}>
)
if(VNLB_LINALG_BACKEND_LIBRARIES)
  target_link_libraries(vnlb_linalg
    PUBLIC
      ${VNLB_LINALG_BACKEND_LIBRARIES}
  )
endif()
set_target_properties(vnlb_linalg PROPERTIES
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)

add_library(vnlb_flow INTERFACE)
target_include_directories(vnlb_flow
  INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(vnlb_flow
  INTERFACE
    vnlb_project_options
)

add_library(vnlb_aggregate STATIC
  src/aggregate/aggregate.cpp
)
target_include_directories(vnlb_aggregate
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(vnlb_aggregate
  PUBLIC
    vnlb_project_options
)
set_target_properties(vnlb_aggregate PROPERTIES
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)

add_library(vnlb_core STATIC
  src/core/core.cpp
)
target_include_directories(vnlb_core
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(vnlb_core
  PUBLIC
    vnlb_project_options
    vnlb_aggregate
    vnlb_flow
    vnlb_linalg
  PRIVATE
    vnlb_highway
)
set_target_properties(vnlb_core PROPERTIES
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)

if(VNLB_BUILD_PLUGIN)
  find_package(VapourSynth REQUIRED)
  add_library(vnlb MODULE
    src/vs/plugin.cpp
  )
  target_link_libraries(vnlb
    PRIVATE
      vnlb_core
      VapourSynth::VapourSynth
  )
  target_include_directories(vnlb
    PRIVATE
      "${CMAKE_CURRENT_BINARY_DIR}/generated"
  )
  set_target_properties(vnlb PROPERTIES
    CXX_EXTENSIONS OFF
    PREFIX "lib"
    OUTPUT_NAME "vnlb"
  )
  if(APPLE)
    set_target_properties(vnlb PROPERTIES
      SUFFIX ".dylib"
    )
  endif()
endif()

if(VNLB_ENABLE_TESTS)
  include(CTest)
  if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(
      NAME reference-fixtures
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/validate_fixtures.py"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    add_executable(vnlb_linalg_tests
      tests/cpp/test_linalg.cpp
    )
    target_link_libraries(vnlb_linalg_tests
      PRIVATE
        vnlb_linalg
    )
    set_target_properties(vnlb_linalg_tests PROPERTIES
      CXX_EXTENSIONS OFF
    )
    add_test(NAME linalg COMMAND vnlb_linalg_tests)

    add_executable(vnlb_core_tests
      tests/cpp/test_core.cpp
    )
    target_link_libraries(vnlb_core_tests
      PRIVATE
        vnlb_core
    )
    set_target_properties(vnlb_core_tests PROPERTIES
      CXX_EXTENSIONS OFF
    )
    add_test(NAME core COMMAND vnlb_core_tests)

    if(VNLB_BUILD_PLUGIN)
      add_test(
        NAME reference-comparison
        COMMAND
          "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_reference.py"
          "$<TARGET_FILE:vnlb>"
          "${CMAKE_CURRENT_SOURCE_DIR}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      )

      add_test(
        NAME vs-integration-bestsource
        COMMAND
          "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/vs/integration_bestsource.py"
          "$<TARGET_FILE:vnlb>"
          "${CMAKE_CURRENT_SOURCE_DIR}"
          "${CMAKE_CURRENT_BINARY_DIR}/bestsource-cache"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      )
    endif()
  endif()
endif()

message(STATUS "VNLB configured")
message(STATUS "  VNLB_BUILD_PLUGIN: ${VNLB_BUILD_PLUGIN}")
message(STATUS "  VNLB_ENABLE_LAPACK: ${VNLB_ENABLE_LAPACK}")
message(STATUS "  VNLB_ENABLE_ACCELERATE: ${VNLB_ENABLE_ACCELERATE}")
message(STATUS "  VNLB_USE_LTO: ${VNLB_USE_LTO}")
message(STATUS "  VNLB_ENABLE_MARCH_NATIVE: ${VNLB_ENABLE_MARCH_NATIVE}")
message(STATUS "  VNLB_HIGHWAY_PROVIDER: ${VNLB_HIGHWAY_PROVIDER}")
message(STATUS "  VNLB_LINALG_BACKEND: ${VNLB_LINALG_BACKEND}")
message(STATUS "  VNLB_LINALG_USE_ACCELERATE: ${VNLB_LINALG_USE_ACCELERATE}")
message(STATUS "  VNLB_LINALG_USE_OPENBLAS: ${VNLB_LINALG_USE_OPENBLAS}")
