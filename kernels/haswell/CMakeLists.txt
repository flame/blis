##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE 
	 ${CMAKE_CURRENT_SOURCE_DIR}/bli_kernels_haswell.h)

add_subdirectory(3)

