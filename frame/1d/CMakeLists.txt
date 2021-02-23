##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved. ##

target_sources("${PROJECT_NAME}"
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1d_check.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1d_fpa.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1d_oapi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1d_oapi_ba.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1d_oapi_ex.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1d_tapi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1d_tapi_ba.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1d_tapi_ex.c
    )
