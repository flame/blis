##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved. ##

target_sources("${PROJECT_NAME}"
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l0_check.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l0_fpa.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l0_oapi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l0_tapi.c
    )

add_subdirectory(copysc)
