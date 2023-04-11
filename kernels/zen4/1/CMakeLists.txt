##Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.##

target_sources("${PROJECT_NAME}" 
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_amaxv_zen_int_avx512.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_scalv_zen_int_avx512.c
    )
