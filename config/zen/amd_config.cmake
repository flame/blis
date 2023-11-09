##Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. ##

if(NOT WIN32)
    if(NOT (DEBUG_TYPE STREQUAL "off"))
        set(CDBGFLAGS -g)
    endif()

    if(DEBUG_TYPE STREQUAL "noopt")
        set(COPTFLAGS -O0)
    else() # off or opt
        set(COPTFLAGS -O2 -fomit-frame-pointer)
    endif()
endif()

# Flags specific to optimized kernels.
# NOTE: The -fomit-frame-pointer option is needed for some kernels because
# they make explicit use of the rbp register.
if(MSVC)
    set(COPTFLAGS /Oy)
    set(CKOPTFLAGS ${COPTFLAGS})
else()
    set(CKOPTFLAGS ${COPTFLAGS} -O3)
endif()

if(MSVC)
    set(CKVECFLAGS -mavx2 -mfma -mno-fma4 -mno-tbm -mno-xop -mno-lwp)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CKVECFLAGS -mavx2 -mfpmath=sse -mfma)
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(CKVECFLAGS -mavx2 -mfpmath=sse -mfma -mno-fma4 -mno-tbm -mno-xop -mno-lwp)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE clang_full_version_string)
    string(REGEX MATCH "^[^\n]*" CLANG_VERSION_STRING "${clang_full_version_string}")
    string(REGEX MATCHALL "(AOCC.LLVM)" CLANG_STRING "${CLANG_VERSION_STRING}")
    if("${CLANG_STRING}" MATCHES "(AOCC.LLVM)")
        list(APPEND CKVECFLAGS -mllvm -disable-licm-vrp)
    endif()
else()
    message(FATAL_ERROR "gcc or clang are required for this configuration.")
endif()

# Flags specific to reference kernels.
set(CROPTFLAGS ${CKOPTFLAGS})
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CRVECFLAGS ${CKVECFLAGS} -funsafe-math-optimizations -ffp-contract=fast)
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(CRVECFLAGS ${CKVECFLAGS} -funsafe-math-optimizations -ffp-contract=fast)
else()
    set(CRVECFLAGS ${CKVECFLAGS})
endif()
