##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}" 
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/bli_dlamch.c
    ${CMAKE_CURRENT_LIST_DIR}/bli_dlamch.h
    ${CMAKE_CURRENT_LIST_DIR}/bli_lsame.c
    ${CMAKE_CURRENT_LIST_DIR}/bli_lsame.h
    ${CMAKE_CURRENT_LIST_DIR}/bli_slamch.c
    ${CMAKE_CURRENT_LIST_DIR}/bli_slamch.h
	)

