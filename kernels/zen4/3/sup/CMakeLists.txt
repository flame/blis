##Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.##

add_library(zen4_3sup
     OBJECT
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_zen_s6x64.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_zen_s6x64.h
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_zen_s6x64m.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_zen_s6x64n.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_zen_s6x64.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_zen_s6x64.h
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_zen_s6x64m.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_zen_s6x64n.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_dgemmsup_rv_zen4_asm_24x8m.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_cv_zen4_z12x4m.c
)
target_compile_options(zen4_3sup PRIVATE /arch:AVX2 /arch:AVX512)
if(BUILD_SHARED_LIBS)
    target_compile_definitions(zen4_3sup PUBLIC -DBLIS_IS_BUILDING_LIBRARY)
endif()

add_subdirectory(d24x8)
