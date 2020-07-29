##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmt.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmt_front.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmt_front.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmt_ker_var2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmt_sup_var1n2m.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmt_var.h
	)

