##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_unpackm.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_unpackm_blk_var1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_unpackm_blk_var1.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_unpackm_check.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_unpackm_check.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_unpackm_cntl.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_unpackm_cntl.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_unpackm_cxk.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_unpackm_cxk.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_unpackm_int.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_unpackm_int.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_unpackm_unb_var1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_unpackm_unb_var1.h
	)
