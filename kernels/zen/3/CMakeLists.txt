##Copyright (C) 2020-2023, Advanced Micro Devices, Inc. All rights reserved.##

add_library(zen_3
     OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_small.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsm_small.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_dgemm_avx2_k1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_zgemm_avx2_k1.c
    )
target_compile_options(zen_3 PRIVATE /arch:AVX2)
if(BUILD_SHARED_LIBS)
    target_compile_definitions(zen_3 PUBLIC -DBLIS_IS_BUILDING_LIBRARY)
endif()
add_subdirectory(sup)
