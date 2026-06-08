function(vnlb_apply_compiler_options target)
  if(NOT VNLB_ENABLE_WARNINGS)
    return()
  endif()

  if(MSVC)
    target_compile_options(${target} INTERFACE /W4)
    if(VNLB_ENABLE_WARNINGS_AS_ERRORS)
      target_compile_options(${target} INTERFACE /WX)
    endif()
  else()
    target_compile_options(${target} INTERFACE
      -Wall
      -Wextra
      -Wpedantic
      -Wconversion
      -Wshadow
    )
    if(VNLB_ENABLE_WARNINGS_AS_ERRORS)
      target_compile_options(${target} INTERFACE -Werror)
    endif()
  endif()
endfunction()
