##Copyright (C) 2023, Advanced Micro Devices, Inc ##

# For architecture independent files we still need to define
# the required flags.
if(MSVC)
    if(NOT ("${CMAKE_BUILD_TYPE}" MATCHES "Release"))
        set(CDBGFLAGS /Zo)
    endif()
    if("${CMAKE_BUILD_TYPE}" MATCHES "Debug")
        set(COPTFLAGS /Od)
    else() # Release or RelWithDebInfo
        set(COPTFLAGS /O2)
    endif()
else()
    if(NOT (DEBUG_TYPE STREQUAL "off"))
        set(CDBGFLAGS -g)
    endif()

    if(DEBUG_TYPE STREQUAL "noopt")
        set(COPTFLAGS -O0)
    else() # off or opt
        set(COPTFLAGS -O3)
    endif()
endif()
