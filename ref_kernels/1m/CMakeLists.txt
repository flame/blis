##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}"
        PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_cxk_1er_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_cxk_3mis_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_cxk_4mi_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_cxk_bb_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_cxk_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_cxk_rih_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_unpackm_cxk_ref.c
    )
