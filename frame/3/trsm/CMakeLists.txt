##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsm.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsm_blk_var1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsm_blk_var2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsm_blk_var3.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsm_cntl.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsm_cntl.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsm_front.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsm_front.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsm_int.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsm_int.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsm_ll_ker_var2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsm_lu_ker_var2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsm_packab.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsm_rl_ker_var2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsm_ru_ker_var2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsm_var.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsm_xx_ker_var2.c
	)

