##Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.##

target_sources("${PROJECT_NAME}" 
     PRIVATE
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_zen_s6x64.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_zen_s6x64.h
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_zen_s6x64m.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rd_zen_s6x64n.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_zen_s6x64.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_zen_s6x64.h
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_zen_s6x64m.c
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmsup_rv_zen_s6x64n.c
    )
