##Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. ##

if(NOT WIN32)
    if(NOT (DEBUG_TYPE STREQUAL "off"))
        set(CDBGFLAGS -g)
    endif()

    if(DEBUG_TYPE STREQUAL "noopt")
        set(COPTFLAGS -O0)
    else() # off or opt
        set(COPTFLAGS -O3)
    endif()
endif()

# Flags specific to optimized kernels.
if(MSVC)
    set(CKOPTFLAGS ${COPTFLAGS})
else()
    set(CKOPTFLAGS ${COPTFLAGS} -O3)
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # Placeholder in case we want to add gcc-specific flags.
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "icc")
    # Placeholder in case we want to add icc-specific flags.
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # Placeholder in case we want to add clang-specific flags.
else()
    message(FATAL_ERROR "gcc, icc, or clang is required for this configuration.")
endif()

# Flags specific to reference kernels.
set(CROPTFLAGS ${CKOPTFLAGS})
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CRVECFLAGS ${CKVECFLAGS})
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CRVECFLAGS ${CKVECFLAGS})
else()
    set(CRVECFLAGS ${CKVECFLAGS})
endif()
