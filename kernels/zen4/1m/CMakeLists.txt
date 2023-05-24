##Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.##

add_library(zen4_1m
     OBJECT
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_zen4_asm_d8xk.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_zen4_asm_d16xk.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_zen4_asm_d24xk.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_zen4_asm_d32xk.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_zen4_asm_z12xk.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_packm_zen4_asm_z4xk.c
 )

target_compile_options(zen4_1m PRIVATE  /U__PRFCHW__ /arch:AVX2 /arch:AVX512)
if(BUILD_SHARED_LIBS)
    target_compile_definitions(zen4_1m PUBLIC -DBLIS_IS_BUILDING_LIBRARY)
endif()
