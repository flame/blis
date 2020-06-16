##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1f.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1f_check.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1f_check.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1f_fpa.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1f_fpa.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1f_ft.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1f_ft_ker.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1f_ker.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1f_ker_prot.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1f_oapi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1f_oapi.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1f_oapi_ba.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1f_oapi_ex.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1f_tapi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1f_tapi.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1f_tapi_ba.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1f_tapi_ex.c
	)
