##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}"
        PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmtrsm_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsm_ref.c
    )

add_subdirectory(bb)
