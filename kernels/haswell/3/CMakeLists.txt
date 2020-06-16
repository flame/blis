##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_haswell_asm_d6x8.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_haswell_asm_d8x6.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmtrsm_l_haswell_asm_d6x8.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmtrsm_u_haswell_asm_d6x8.c
    )

add_subdirectory(sup)
