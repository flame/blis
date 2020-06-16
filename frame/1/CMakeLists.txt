##Copyright (C) 2020, Advanced Micro Devices, Inc ##

target_sources("${PROJECT_NAME}" 
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1v.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1v_check.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1v_check.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1v_fpa.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1v_fpa.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1v_ft.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1v_ft_ker.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1v_ker.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1v_ker_prot.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1v_oapi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1v_oapi.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1v_oapi_ba.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1v_oapi_ex.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1v_tapi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1v_tapi.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1v_tapi_ba.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1v_tapi_ex.c
	)

