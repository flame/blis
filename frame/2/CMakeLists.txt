##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources("${PROJECT_NAME}"
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l2_check.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l2_fpa.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l2_oapi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l2_oapi_ba.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l2_oapi_ex.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l2_tapi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l2_tapi_ba.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l2_tapi_ex.c
    )

set(SUBDIRECTORIES "gemv" "ger" "hemv" "her" "her2" "trmv" "trsv")

#Add all subdirectories
foreach(VAR ${SUBDIRECTORIES})
    add_subdirectory(${VAR})
endforeach()
