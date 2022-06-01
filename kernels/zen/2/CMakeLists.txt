##Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.##

target_sources("${PROJECT_NAME}"
     PRIVATE
	 ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemv_zen_ref.c
	 ${CMAKE_CURRENT_SOURCE_DIR}/bli_her2_zen_int_4.c
	 ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemv_zen_int_4.c
     )

# Select AMD specific sources for AMD configurations.
if(${TARGET_ARCH} STREQUAL zen OR 
${TARGET_ARCH} STREQUAL zen2 OR 
${TARGET_ARCH} STREQUAL zen3 OR 
${TARGET_ARCH} STREQUAL zen4 OR
${TARGET_ARCH} STREQUAL amdzen)
     target_sources("${PROJECT_NAME}"
     PRIVATE
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_her_zen_int_amd.c
     )
endif()