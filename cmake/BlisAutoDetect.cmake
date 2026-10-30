# Hardware auto-detection, mirroring configure's auto_detect():
# compile+run build/detect/config/config_detect.c together with the cpuid
# framework files; on RISC-V the detection is done at preprocessing time
# (cross-compile friendly). When cross-compiling and no detection is
# possible, the caller must have selected an explicit BLIS_CONFIG.

function(blis_autodetect out_var)
    set(_result "")
    message(STATUS "BLIS: auto-detecting sub-configuration...")

    # --- RISC-V: preprocess-time detection (works even when cross-compiling) ---
    execute_process(
        COMMAND "${CMAKE_C_COMPILER}" -E "${BLIS_SOURCE_DIR}/build/detect/riscv/bli_riscv_cpuid.h"
        RESULT_VARIABLE _rv
        OUTPUT_VARIABLE _out
        ERROR_QUIET
    )
    if(_rv EQUAL 0)
        string(REGEX MATCHALL "(^|\n)[^#\n][^\n]*" _lines "${_out}")
        string(REGEX REPLACE "[\n \t]" "" _lines "${_lines}")
        if(_lines MATCHES "^rv(32|64)i(v?)$")
            message(STATUS "BLIS: detected '${_lines}' (RISC-V compile-time detection)")
            set(${out_var} "${_lines}" PARENT_SCOPE)
            return()
        endif()
    endif()

    # --- Everything else: compile and run config_detect ---
    set(_fsrc "${BLIS_SOURCE_DIR}/frame/base")
    set(_srcs
        "${BLIS_SOURCE_DIR}/build/detect/config/config_detect.c"
        "${_fsrc}/bli_arch.c"
        "${_fsrc}/bli_cpuid.c"
        "${_fsrc}/bli_env.c"
    )

    # Include dirs: those containing the headers configure searches for.
    set(_incs "")
    foreach(
        h
        bli_system.h
        bli_type_defs.h
        bli_arch.h
        bli_cpuid.h
        bli_env.h
        bli_malloc.h
        bli_pthread.h
    )
        file(GLOB_RECURSE _found "${BLIS_SOURCE_DIR}/frame/*/${h}")
        list(GET _found 0 _f)
        get_filename_component(_d "${_f}" DIRECTORY)
        list(APPEND _incs "-I${_d}")
    endforeach()
    list(REMOVE_DUPLICATES _incs)

    # One -DBLIS_CONFIG_<C> per config macro referenced by bli_cpuid.c.
    file(STRINGS "${_fsrc}/bli_cpuid.c" _cfglines REGEX "BLIS_CONFIG_[A-Z0-9_]+")
    set(_defs "")
    foreach(l ${_cfglines})
        string(REGEX MATCHALL "BLIS_CONFIG_[A-Z0-9_]+" _m "${l}")
        list(APPEND _defs ${_m})
    endforeach()
    list(REMOVE_DUPLICATES _defs)
    list(TRANSFORM _defs PREPEND "-D")

    set(_bin_dir "${PROJECT_BINARY_DIR}/CMakeFiles/config-detect")
    file(MAKE_DIRECTORY "${_bin_dir}")
    set(_cflags -DBLIS_CONFIGURETIME_CPUID -std=c99 -D_GNU_SOURCE)
    if(BLIS_CC_VENDOR MATCHES "clang|aocc")
        list(APPEND _cflags -Wno-tautological-compare)
    endif()
    set(_link_libs "")
    if(NOT BLIS_IS_MSVC AND BLIS_ENABLE_SYSTEM)
        list(APPEND _link_libs -lpthread)
    endif()
    execute_process(
        COMMAND "${CMAKE_C_COMPILER}" ${_cflags} ${_incs} ${_defs} ${_srcs} ${_link_libs} -o "${_bin_dir}/config_detect"
        RESULT_VARIABLE _rv
        OUTPUT_QUIET
        ERROR_VARIABLE _cerr
    )
    if(NOT _rv EQUAL 0)
        file(REMOVE_RECURSE "${_bin_dir}")
        message(
            FATAL_ERROR
            "BLIS: failed to compile config_detect (${_cerr}). "
            "Select a configuration explicitly with -DBLIS_CONFIG=<name>."
        )
    endif()

    set(_run_rv 1)
    if(NOT CMAKE_CROSSCOMPILING)
        execute_process(
            COMMAND "${_bin_dir}/config_detect"
            RESULT_VARIABLE _run_rv
            OUTPUT_VARIABLE _result
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            TIMEOUT 30
        )
    endif()
    file(REMOVE_RECURSE "${_bin_dir}")
    if(NOT _run_rv EQUAL 0 OR _result STREQUAL "")
        message(
            FATAL_ERROR
            "BLIS: could not run config_detect (cross-compiling?). "
            "Select a configuration explicitly with -DBLIS_CONFIG=<name>."
        )
    endif()

    message(STATUS "BLIS: detected sub-configuration '${_result}'")
    set(${out_var} "${_result}" PARENT_SCOPE)
endfunction()
