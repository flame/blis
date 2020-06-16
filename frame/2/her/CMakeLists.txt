##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_her.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_her_unb_var1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_her_unb_var2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_her_var.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_her_var_oapi.c
	)

