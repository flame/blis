##Copyright (C) 2022-23, Advanced Micro Devices, Inc. All rights reserved.##

add_library(zen4_3
     OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmtrsm_l_zen_16x14.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmtrsm_u_zen_16x14.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmtrsm_l_zen4_8x24.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmtrsm_u_zen4_8x24.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_dgemm_zen4_asm_32x6.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_dgemm_zen4_asm_8x24.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsm_small_AVX512.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_zgemm_zen4_asm_12x4.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_zero_zmm.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_zgemm_zen4_asm_4x12.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_zgemmtrsm_l_4x12.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_zgemmtrsm_u_4x12.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_dgemm_avx512_k1.c
    )

target_compile_options(zen4_3 PRIVATE /arch:AVX2 /arch:AVX512)
if(BUILD_SHARED_LIBS)
    target_compile_definitions(zen4_3 PUBLIC -DBLIS_IS_BUILDING_LIBRARY)
endif()

add_subdirectory(sup)
