##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trmv.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trmv_unb_var1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trmv_unb_var2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trmv_unf_var1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trmv_unf_var2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trmv_var.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trmv_var_oapi.c
	)
