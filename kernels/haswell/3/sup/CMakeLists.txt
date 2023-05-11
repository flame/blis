##Copyright (C) 2020-2023, Advanced Micro Devices, Inc. All rights reserved.##

add_library(haswell_3sup
     OBJECT
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_haswell_asm_d6x8m.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_haswell_asm_d6x8n.c
     #${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_haswell_asm_s6x16m.c
     #${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_haswell_asm_s6x16n.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_haswell_asm_d6x8m.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_haswell_asm_d6x8n.c
     #${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_haswell_asm_s6x16m.c
     #${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_haswell_asm_s6x16n.c
    )
target_compile_options(haswell_3sup PRIVATE /arch:AVX2)

add_subdirectory(d6x8)
#add_subdirectory(s6x16)