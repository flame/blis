##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_axpyf_zen_int_8.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_dotxf_zen_int_8.c
    )
