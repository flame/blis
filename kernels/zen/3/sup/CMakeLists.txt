##Copyright (C) 2020-2023, Advanced Micro Devices, Inc. All rights reserved.##

add_library(zen_3_sup
     OBJECT
${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_zen_asm_s6x16.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_zen_asm_s6x16m.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_zen_asm_s6x16n.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_zen_asm_z3x4.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_zen_asm_z3x4m.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_zen_asm_z3x4n.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_zen_asm_c3x8.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_zen_asm_c3x8m.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_zen_asm_c3x8n.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_zen_asm_s6x16.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_zen_asm_s6x16m.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_zen_asm_s6x16n.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_zen_asm_z3x4.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_zen_asm_z3x4m.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_zen_asm_z3x4n.c
    )
target_compile_options(zen_3_sup PRIVATE /arch:AVX2)
if(BUILD_SHARED_LIBS)
    target_compile_definitions(zen_3_sup PUBLIC -DBLIS_IS_BUILDING_LIBRARY)
endif()
