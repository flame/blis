##Copyright (C) 2022-2023, Advanced Micro Devices, Inc. All rights reserved.##

add_library(skx_3
     OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_dgemm_skx_asm_16x14.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_sgemm_skx_asm_32x12_l2.c
    )
target_compile_options(skx_3 PRIVATE /arch:AVX2 /arch:AVX512)
if(BUILD_SHARED_LIBS)
    target_compile_definitions(skx_3 PUBLIC -DBLIS_IS_BUILDING_LIBRARY)
endif()
