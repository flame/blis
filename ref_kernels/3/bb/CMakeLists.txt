##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}"
        PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmbb_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmtrsmbb_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsmbb_ref.c
    )

