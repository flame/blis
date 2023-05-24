##Copyright (C) 2020-2023, Advanced Micro Devices, Inc. All rights reserved.##

add_library(haswell_3supd6x8
     OBJECT
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

target_compile_options(haswell_3supd6x8 PRIVATE /arch:AVX2)
if(BUILD_SHARED_LIBS)
    target_compile_definitions(haswell_3supd6x8 PUBLIC -DBLIS_IS_BUILDING_LIBRARY)
endif()
