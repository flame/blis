##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_herk.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_herk_front.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_herk_front.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_herk_l_ker_var2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_herk_u_ker_var2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_herk_var.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_herk_x_ker_var2.c
	)

