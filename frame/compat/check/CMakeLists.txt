##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}"
    PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/bla_gemm_check.h
${CMAKE_CURRENT_SOURCE_DIR}/bla_gemv_check.h
${CMAKE_CURRENT_SOURCE_DIR}/bla_ger_check.h
${CMAKE_CURRENT_SOURCE_DIR}/bla_hemm_check.h
${CMAKE_CURRENT_SOURCE_DIR}/bla_hemv_check.h
${CMAKE_CURRENT_SOURCE_DIR}/bla_her2_check.h
${CMAKE_CURRENT_SOURCE_DIR}/bla_her2k_check.h
${CMAKE_CURRENT_SOURCE_DIR}/bla_her_check.h
${CMAKE_CURRENT_SOURCE_DIR}/bla_herk_check.h
${CMAKE_CURRENT_SOURCE_DIR}/bla_symm_check.h
${CMAKE_CURRENT_SOURCE_DIR}/bla_symv_check.h
${CMAKE_CURRENT_SOURCE_DIR}/bla_syr2_check.h
${CMAKE_CURRENT_SOURCE_DIR}/bla_syr2k_check.h
${CMAKE_CURRENT_SOURCE_DIR}/bla_syr_check.h
${CMAKE_CURRENT_SOURCE_DIR}/bla_syrk_check.h
${CMAKE_CURRENT_SOURCE_DIR}/bla_trmm_check.h
${CMAKE_CURRENT_SOURCE_DIR}/bla_trmv_check.h
${CMAKE_CURRENT_SOURCE_DIR}/bla_trsm_check.h
${CMAKE_CURRENT_SOURCE_DIR}/bla_trsv_check.h
	)


	

