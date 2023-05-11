##Copyright (C) 2020-2023, Advanced Micro Devices, Inc. All rights reserved.##

add_library(zen4_3supd24x8
     OBJECT
${CMAKE_CURRENT_SOURCE_DIR}/bli_dgemmsup_rv_zen4_asm_Mx1.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_dgemmsup_rv_zen4_asm_Mx2.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_dgemmsup_rv_zen4_asm_Mx3.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_dgemmsup_rv_zen4_asm_Mx4.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_dgemmsup_rv_zen4_asm_Mx5.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_dgemmsup_rv_zen4_asm_Mx6.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_dgemmsup_rv_zen4_asm_Mx7.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_dgemmsup_rv_zen4_asm_Mx8.c
    )

target_compile_options(zen4_3supd24x8 PRIVATE /arch:AVX2 /arch:AVX512)