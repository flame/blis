##Copyright (C) 2020-2021, Advanced Micro Devices, Inc. All rights reserved.##

add_library(haswell_1m
     OBJECT
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_haswell_asm_c3xk.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_haswell_asm_c8xk.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_haswell_asm_d6xk.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_haswell_asm_d8xk.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_haswell_asm_s16xk.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_haswell_asm_s6xk.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_haswell_asm_z3xk.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_haswell_asm_z4xk.c
 )

target_compile_options(haswell_1m PRIVATE /arch:AVX2)