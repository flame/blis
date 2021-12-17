##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources("${PROJECT_NAME}"
     PRIVATE
	 ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemv_zen_ref.c
	 ${CMAKE_CURRENT_SOURCE_DIR}/bli_her2_zen_int_4.c
     )



