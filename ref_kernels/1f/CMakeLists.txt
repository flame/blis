##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}"
        PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_axpy2v_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_axpyf_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_dotaxpyv_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_dotxaxpyf_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_dotxf_ref.c
    )

