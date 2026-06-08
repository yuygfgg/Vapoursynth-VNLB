include(FindPackageHandleStandardArgs)

find_package(Python3 COMPONENTS Interpreter QUIET)

set(_VNLB_VS_INCLUDE_FROM_PYTHON "")
if(Python3_Interpreter_FOUND)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import vapoursynth; print(vapoursynth.get_include())"
    RESULT_VARIABLE _VNLB_VS_PYTHON_RESULT
    OUTPUT_VARIABLE _VNLB_VS_INCLUDE_FROM_PYTHON
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT _VNLB_VS_PYTHON_RESULT EQUAL 0)
    set(_VNLB_VS_INCLUDE_FROM_PYTHON "")
  endif()
endif()

find_path(
  VapourSynth_INCLUDE_DIR
  NAMES VapourSynth4.h VapourSynth.h
  HINTS "${_VNLB_VS_INCLUDE_FROM_PYTHON}"
)

find_package_handle_standard_args(
  VapourSynth
  REQUIRED_VARS VapourSynth_INCLUDE_DIR
)

if(VapourSynth_FOUND AND NOT TARGET VapourSynth::VapourSynth)
  add_library(VapourSynth::VapourSynth INTERFACE IMPORTED)
  target_include_directories(
    VapourSynth::VapourSynth
    INTERFACE "${VapourSynth_INCLUDE_DIR}"
  )
endif()

mark_as_advanced(VapourSynth_INCLUDE_DIR)
