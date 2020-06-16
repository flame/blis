##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_blk_var1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_blk_var2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_blk_var3.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_cntl.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_cntl.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_front.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_front.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_int.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_int.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_ker_var1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_ker_var2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_ker_var2_md.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_md.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_md.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_md_c2r_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_md_c2r_ref.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_packab.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_var.h
	)

	add_subdirectory(ind)
