##Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.##

target_sources("${PROJECT_NAME}" 
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_dgemm_skx_asm_16x14.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_sgemm_skx_asm_32x12_l2.c
    )
