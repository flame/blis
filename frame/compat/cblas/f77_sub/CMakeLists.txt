##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}"
    PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/f77_amax_sub.c
${CMAKE_CURRENT_SOURCE_DIR}/f77_amax_sub.h
${CMAKE_CURRENT_SOURCE_DIR}/f77_asum_sub.c
${CMAKE_CURRENT_SOURCE_DIR}/f77_asum_sub.h
${CMAKE_CURRENT_SOURCE_DIR}/f77_dot_sub.c
${CMAKE_CURRENT_SOURCE_DIR}/f77_dot_sub.h
${CMAKE_CURRENT_SOURCE_DIR}/f77_nrm2_sub.c
${CMAKE_CURRENT_SOURCE_DIR}/f77_nrm2_sub.h
	)


	

