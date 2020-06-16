##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE 
	 ${CMAKE_CURRENT_SOURCE_DIR}/bli_kernels_zen.h
     )

set(SUBDIRECTORIES "1" "1f" "1m" "3")

#Add all subdirectories 
foreach(VAR ${SUBDIRECTORIES})
	add_subdirectory(${VAR})
endforeach()


