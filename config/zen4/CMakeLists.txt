##Copyright (C) 2022, Advanced Micro Devices, Inc ##

target_sources("${PROJECT_NAME}"
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_cntx_init_zen4.c
	${CMAKE_CURRENT_SOURCE_DIR}/bli_family_zen4.h
   )
