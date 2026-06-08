cmake_minimum_required(VERSION 3.24)

project(
  VapourSynthVNLB
  VERSION 0.1.0
  DESCRIPTION "Video Non-Local Bayes denoising plugin for VapourSynth"
  LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(VNLB_BUILD_PLUGIN "Build the VapourSynth native plugin target" OFF)
option(VNLB_ENABLE_TESTS "Enable project tests" ON)
option(VNLB_ENABLE_LAPACK "Enable a LAPACK-compatible eigensolver backend" ON)
option(VNLB_ENABLE_CBLAS "Enable optional CBLAS kernels after benchmarking" OFF)
option(VNLB_ENABLE_WARNINGS "Enable project warning flags" ON)
option(VNLB_ENABLE_WARNINGS_AS_ERRORS "Treat project warnings as errors" OFF)

set(_VNLB_RELEASE_OPTIMIZATIONS_DEFAULT OFF)
string(TOUPPER "${CMAKE_BUILD_TYPE}" _VNLB_BUILD_TYPE_UPPER)
if(CMAKE_CONFIGURATION_TYPES OR _VNLB_BUILD_TYPE_UPPER STREQUAL "RELEASE")
  set(_VNLB_RELEASE_OPTIMIZATIONS_DEFAULT ON)
endif()

option(VNLB_USE_LTO "Enable link time optimization for Release builds" ${_VNLB_RELEASE_OPTIMIZATIONS_DEFAULT})
option(VNLB_ENABLE_MARCH_NATIVE "Enable -march=native for Release builds" ${_VNLB_RELEASE_OPTIMIZATIONS_DEFAULT})

cmake_policy(SET CMP0069 NEW)
if(VNLB_USE_LTO)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
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

set(VNLB_LINALG_BACKEND_SOURCE)
set(VNLB_LINALG_USE_ACCELERATE OFF)
if(APPLE AND VNLB_ENABLE_LAPACK)
  set(VNLB_LINALG_USE_ACCELERATE ON)
  set(VNLB_LINALG_BACKEND_SOURCE
    src/linalg/backend_accelerate.cpp
  )
endif()

set(VNLB_LINALG_USE_AARCH64_NEON OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|AARCH64)$")
  set(VNLB_LINALG_USE_AARCH64_NEON ON)
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
)
target_compile_definitions(vnlb_linalg
  PUBLIC
    VNLB_LINALG_USE_AARCH64_NEON=$<BOOL:${VNLB_LINALG_USE_AARCH64_NEON}>
  PRIVATE
    VNLB_LINALG_USE_ACCELERATE=$<BOOL:${VNLB_LINALG_USE_ACCELERATE}>
)
if(APPLE AND VNLB_ENABLE_LAPACK)
  target_link_libraries(vnlb_linalg
    PUBLIC
      "-framework Accelerate"
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

set(VNLB_CORE_USE_AARCH64_NEON OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|AARCH64)$")
  set(VNLB_CORE_USE_AARCH64_NEON ON)
endif()

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
)
target_compile_definitions(vnlb_core
  PRIVATE
    VNLB_CORE_USE_AARCH64_NEON=$<BOOL:${VNLB_CORE_USE_AARCH64_NEON}>
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
message(STATUS "  VNLB_ENABLE_CBLAS: ${VNLB_ENABLE_CBLAS}")
message(STATUS "  VNLB_USE_LTO: ${VNLB_USE_LTO}")
message(STATUS "  VNLB_ENABLE_MARCH_NATIVE: ${VNLB_ENABLE_MARCH_NATIVE}")
message(STATUS "  VNLB_LINALG_USE_ACCELERATE: ${VNLB_LINALG_USE_ACCELERATE}")
message(STATUS "  VNLB_LINALG_USE_AARCH64_NEON: ${VNLB_LINALG_USE_AARCH64_NEON}")
message(STATUS "  VNLB_CORE_USE_AARCH64_NEON: ${VNLB_CORE_USE_AARCH64_NEON}")
