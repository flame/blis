##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}"
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/bli_castm.c
    ${CMAKE_CURRENT_LIST_DIR}/bli_castm.h
    ${CMAKE_CURRENT_LIST_DIR}/bli_castnzm.c
    ${CMAKE_CURRENT_LIST_DIR}/bli_castnzm.h
    ${CMAKE_CURRENT_LIST_DIR}/bli_castv.c
    ${CMAKE_CURRENT_LIST_DIR}/bli_castv.h
	)


