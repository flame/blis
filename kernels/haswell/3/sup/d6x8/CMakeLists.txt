##Copyright (C) 2020-2021, Advanced Micro Devices, Inc. All rights reserved.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_r_haswell_ref_dMx1.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_haswell_asm_dMx1.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_haswell_asm_dMx2.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_haswell_asm_dMx4.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_haswell_asm_dMx8.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_haswell_asm_dMx2.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_haswell_asm_dMx4.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_haswell_asm_dMx6.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_haswell_asm_dMx8.c
    )


