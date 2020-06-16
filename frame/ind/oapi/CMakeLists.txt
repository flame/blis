##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_3m4m1m_oapi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_ind_oapi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_ind_oapi.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_nat_oapi.c
 	)
