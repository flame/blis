##Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.##

add_library(zen_1
     OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_amaxv_zen_int.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_axpbyv_zen_int.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_axpbyv_zen_int10.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_axpyv_zen_int.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_axpyv_zen_int10.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_copyv_zen_int.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_dotv_zen_int.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_dotv_zen_int10.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_dotxv_zen_int.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_scalv_zen_int.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_scalv_zen_int10.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_setv_zen_int.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_swapv_zen_int8.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_norm2_zen_int.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_scal2v_zen_int.c
    )
target_compile_options(zen_1 PRIVATE /arch:AVX2)