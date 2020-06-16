##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}"
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_cblas.h
	)

#Add all subdirectories 
	add_subdirectory(f77_sub)
	add_subdirectory(src)


	

