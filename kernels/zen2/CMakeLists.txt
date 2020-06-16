##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_kernels_zen2.h
	)
	
add_subdirectory(1f)
add_subdirectory(2)


