##Copyright (C) 2020-2021, Advanced Micro Devices, Inc. All rights reserved.##

target_sources("${PROJECT_NAME}" 
     PRIVATE
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_haswell_asm_c3xk.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_haswell_asm_c8xk.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_haswell_asm_d6xk.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_haswell_asm_d8xk.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_haswell_asm_s16xk.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_haswell_asm_s6xk.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_haswell_asm_z3xk.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_haswell_asm_z4xk.c
 )
