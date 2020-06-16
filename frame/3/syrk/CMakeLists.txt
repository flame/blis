##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_syrk.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_syrk_front.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_syrk_front.h
	)

