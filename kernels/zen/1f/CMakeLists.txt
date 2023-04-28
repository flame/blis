##Copyright (C) 2020-2022, Advanced Micro Devices, Inc. All rights reserved.##

add_library(zen_1f
     OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_axpyf_zen_int_8.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_dotxf_zen_int_8.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_axpyf_zen_int_5.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_axpyf_zen_int_4.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_axpyf_zen_int_6.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_axpy2v_zen_int.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_dotxaxpyf_zen_int_8.c           
    )
target_compile_options(zen_1f PRIVATE /arch:AVX2)