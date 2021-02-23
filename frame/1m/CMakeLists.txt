##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources("${PROJECT_NAME}"
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1m_check.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1m_fpa.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1m_oapi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1m_oapi_ba.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1m_oapi_ex.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1m_tapi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1m_tapi_ba.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1m_tapi_ex.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1m_unb_var1.c
    )

set(SUBDIRECTORIES "packm" "unpackm")

#Add all subdirectories
foreach(VAR ${SUBDIRECTORIES})
    add_subdirectory(${VAR})
endforeach()
