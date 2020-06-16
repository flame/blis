##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_amaxv_zen_int.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_axpyv_zen_int.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_axpyv_zen_int10.c
	${CMAKE_CURRENT_SOURCE_DIR}/bli_copyv_zen_int.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_dotv_zen_int.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_dotv_zen_int10.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_dotxv_zen_int.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_scalv_zen_int.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_scalv_zen_int10.c
	${CMAKE_CURRENT_SOURCE_DIR}/bli_setv_zen_int.c
	${CMAKE_CURRENT_SOURCE_DIR}/bli_swapv_zen_int8.c
    )
