##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_small.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_syrk_small.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsm_small.c
    )

add_subdirectory(sup)
