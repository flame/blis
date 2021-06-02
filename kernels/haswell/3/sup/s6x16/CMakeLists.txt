##Copyright (C) 2020-2021, Advanced Micro Devices, Inc. All rights reserved.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_r_haswell_ref_sMx1.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_haswell_asm_sMx1.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_haswell_asm_sMx12.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_haswell_asm_sMx16.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_haswell_asm_sMx2.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_haswell_asm_sMx4.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_haswell_asm_sMx8.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_haswell_asm_sMx12.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_haswell_asm_sMx16.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_haswell_asm_sMx2.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_haswell_asm_sMx4.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_haswell_asm_sMx6.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_haswell_asm_sMx8.c
 )


