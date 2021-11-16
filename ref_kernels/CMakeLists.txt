##Copyright (C) 2021, Advanced Micro Devices, Inc. All rights reserved.##

if(${TARGET_ARCH} STREQUAL amdzen)
add_subdirectory(${CMAKE_BINARY_DIR}/ref_kernels/generic ${CMAKE_BINARY_DIR}/ref_kernels/generic)
add_subdirectory(${CMAKE_BINARY_DIR}/ref_kernels/zen ${CMAKE_BINARY_DIR}/ref_kernels/zen)
add_subdirectory(${CMAKE_BINARY_DIR}/ref_kernels/zen2 ${CMAKE_BINARY_DIR}/ref_kernels/zen2)
add_subdirectory(${CMAKE_BINARY_DIR}/ref_kernels/zen3 ${CMAKE_BINARY_DIR}/ref_kernels/zen3)
add_subdirectory(${CMAKE_BINARY_DIR}/ref_kernels/zen4 ${CMAKE_BINARY_DIR}/ref_kernels/zen4)
else()
target_sources("${PROJECT_NAME}"
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/bli_cntx_ref.c
        )

set(SUBDIRECTORIES "1" "1f" "1m" "3" "ind")

#Add all subdirectories
foreach(VAR ${SUBDIRECTORIES})
    add_subdirectory(${VAR})
endforeach()
endif()
