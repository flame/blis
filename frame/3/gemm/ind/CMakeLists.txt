##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_ind_opt.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm4mb_ker_var2.c
	)
