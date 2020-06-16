##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_her2k.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_her2k_front.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_her2k_front.h
	)

