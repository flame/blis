##Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.##

target_sources("${PROJECT_NAME}" 
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmtrsm_l_zen_16x14.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmtrsm_u_zen_16x14.c
    )
