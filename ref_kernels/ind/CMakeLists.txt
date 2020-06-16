##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}"
        PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm1m_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm3m1_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm3mh_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm4m1_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm4mb_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm4mh_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmtrsm1m_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmtrsm3m1_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmtrsm4m1_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsm1m_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsm3m1_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsm4m1_ref.c
    )
