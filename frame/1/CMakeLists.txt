##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved. ##

target_sources("${PROJECT_NAME}"
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1v_check.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1v_fpa.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1v_oapi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1v_oapi_ba.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1v_oapi_ex.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1v_tapi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1v_tapi_ba.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1v_tapi_ex.c
)

