##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_hemv.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_hemv_unb_var1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_hemv_unb_var2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_hemv_unb_var3.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_hemv_unb_var4.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_hemv_unf_var1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_hemv_unf_var1a.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_hemv_unf_var3.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_hemv_unf_var3a.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_hemv_var.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_hemv_var_oapi.c
	)

