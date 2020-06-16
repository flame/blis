##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_her2.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_her2_unb_var1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_her2_unb_var2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_her2_unb_var3.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_her2_unb_var4.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_her2_unf_var1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_her2_unf_var4.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_her2_var.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_her2_var_oapi.c
	)

