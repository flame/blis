##Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.##

add_library(haswell_3
     OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_haswell_asm_d6x8.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_haswell_asm_d8x6.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmtrsm_l_haswell_asm_d6x8.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmtrsm_u_haswell_asm_d6x8.c
    )

target_compile_options(haswell_3 PRIVATE /arch:AVX2)
if(BUILD_SHARED_LIBS)
    target_compile_definitions(haswell_3 PUBLIC -DBLIS_IS_BUILDING_LIBRARY)
endif()

add_subdirectory(sup)
