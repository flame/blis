##Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. ##

# Include file containing common flags for all AMD architectures
include(${CMAKE_SOURCE_DIR}/config/zen/amd_config.cmake)
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
# NOTE: The -fomit-frame-pointer option is needed for some kernels because
# they make explicit use of the rbp register.
if(MSVC)
    set(CKOPTFLAGS ${COPTFLAGS} /Oy)
else()
    set(CKOPTFLAGS ${COPTFLAGS} -fomit-frame-pointer)
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    list(APPEND CKVECFLAGS -march=znver1)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.0.0)
        list(APPEND CKOPTFLAGS -fno-tree-partial-pre -fno-tree-pre -fno-tree-loop-vectorize -fno-gcse)
    endif()
endif()

# Flags specific to reference kernels.
set(CROPTFLAGS ${CKOPTFLAGS})
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CRVECFLAGS ${CKVECFLAGS})
else()
    set(CRVECFLAGS ${CKVECFLAGS})
endif()
