##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trmm3.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trmm3_front.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trmm3_front.h
	)

