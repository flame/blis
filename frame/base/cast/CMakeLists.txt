##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources("${PROJECT_NAME}"
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/bli_castm.c
    ${CMAKE_CURRENT_LIST_DIR}/bli_castnzm.c
    ${CMAKE_CURRENT_LIST_DIR}/bli_castv.c
    )


