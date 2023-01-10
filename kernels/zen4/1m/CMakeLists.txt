##Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.##

target_sources("${PROJECT_NAME}" 
     PRIVATE
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_zen4_asm_d16xk.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_zen4_asm_d24xk.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_zen4_asm_d32xk.c
 )
