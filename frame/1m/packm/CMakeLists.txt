##Copyright (C) 2020 - 2023, Advanced Micro Devices, Inc. All rights reserved.##

target_sources("${PROJECT_NAME}"
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_blk_var1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_blk_var1_md.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_check.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_cntl.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_cxk.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_cxk_1er.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_cxk_3mis.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_cxk_4mi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_cxk_rih.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_init.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_int.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_part.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_struc_cxk.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_struc_cxk_1er.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_struc_cxk_3mis.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_struc_cxk_4mi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_struc_cxk_md.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_struc_cxk_rih.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_thrinfo.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_unb_var1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_pack_full.c
    )

