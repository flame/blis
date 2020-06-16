##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE
     ${CMAKE_CURRENT_LIST_DIR}/bli_projm.c
     ${CMAKE_CURRENT_LIST_DIR}/bli_projm.h
     ${CMAKE_CURRENT_LIST_DIR}/bli_projv.c
     ${CMAKE_CURRENT_LIST_DIR}/bli_projv.h
	)


