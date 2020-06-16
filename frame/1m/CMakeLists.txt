##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1m.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1m_check.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1m_check.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1m_fpa.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1m_fpa.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1m_ft.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1m_ft_ker.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1m_ker.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1m_ker_prot.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1m_oapi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1m_oapi.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1m_oapi_ba.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1m_oapi_ex.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1m_oft_var.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1m_tapi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1m_tapi.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1m_tapi_ba.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1m_tapi_ex.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1m_unb_var1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l1m_unb_var1.h
	)

set(SUBDIRECTORIES "packm" "unpackm")

#Add all subdirectories 
foreach(VAR ${SUBDIRECTORIES})
	add_subdirectory(${VAR})
endforeach()
