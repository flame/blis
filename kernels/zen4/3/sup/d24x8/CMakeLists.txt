##Copyright (C) 2020-2023, Advanced Micro Devices, Inc. All rights reserved.##

target_sources("${PROJECT_NAME}" 
     PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/bli_dgemmsup_rv_zen4_asm_Mx1.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_dgemmsup_rv_zen4_asm_Mx2.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_dgemmsup_rv_zen4_asm_Mx3.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_dgemmsup_rv_zen4_asm_Mx4.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_dgemmsup_rv_zen4_asm_Mx5.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_dgemmsup_rv_zen4_asm_Mx6.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_dgemmsup_rv_zen4_asm_Mx7.c
${CMAKE_CURRENT_SOURCE_DIR}/bli_dgemmsup_rv_zen4_asm_Mx8.c
    )