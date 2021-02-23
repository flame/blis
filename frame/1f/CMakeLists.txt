##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources("${PROJECT_NAME}"
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1f_check.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1f_fpa.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1f_oapi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1f_oapi_ba.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1f_oapi_ex.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1f_tapi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1f_tapi_ba.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1f_tapi_ex.c
    )
