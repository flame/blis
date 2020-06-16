##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trmm.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trmm_front.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trmm_front.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trmm_ll_ker_var2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trmm_lu_ker_var2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trmm_rl_ker_var2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trmm_ru_ker_var2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trmm_var.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trmm_xx_ker_var2.c
	)

