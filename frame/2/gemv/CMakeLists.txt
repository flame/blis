##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemv.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemv_unb_var1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemv_unb_var2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemv_unf_var1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemv_unf_var2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemv_var.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemv_var_oapi.c
	)

