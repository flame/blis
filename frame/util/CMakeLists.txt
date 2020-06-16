##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}"
        PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_util.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_util_check.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_util_check.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_util_fpa.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_util_fpa.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_util_ft.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_util_oapi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_util_oapi.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_util_oapi_ba.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_util_oapi_ex.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_util_tapi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_util_tapi.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_util_tapi_ba.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_util_tapi_ex.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_util_unb_var1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_util_unb_var1.h
	)
