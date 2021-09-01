##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources("${PROJECT_NAME}"
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_small.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsm_small.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_dgemm_ref_k1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_sqp_kernels.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_sqp.c
    )

add_subdirectory(sup)
