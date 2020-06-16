##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_ind.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_ind.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_ind.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_ind.h			   
	)

set(SUBDIRECTORIES "cntx" "oapi" "tapi" "ukernels")

#Add all subdirectories 
foreach(VAR ${SUBDIRECTORIES})
	add_subdirectory(${VAR})
endforeach()

