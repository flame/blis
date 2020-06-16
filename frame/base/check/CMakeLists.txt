##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE
     ${CMAKE_CURRENT_LIST_DIR}/bli_obj_check.c
     ${CMAKE_CURRENT_LIST_DIR}/bli_obj_check.h
     ${CMAKE_CURRENT_LIST_DIR}/bli_part_check.c
     ${CMAKE_CURRENT_LIST_DIR}/bli_part_check.h
	)

