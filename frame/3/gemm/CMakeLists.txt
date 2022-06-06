##Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.##

target_sources("${PROJECT_NAME}"
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_blk_var1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_blk_var2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_blk_var3.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_cntl.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_int.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_ker_var1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_ker_var2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_ker_var2_md.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_md.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_md_c2r_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_packab.c
    )

# Select AMD specific sources for AMD configurations.
if(${TARGET_ARCH} STREQUAL zen OR 
${TARGET_ARCH} STREQUAL zen2 OR 
${TARGET_ARCH} STREQUAL zen3 OR
${TARGET_ARCH} STREQUAL zen4 OR
${TARGET_ARCH} STREQUAL amdzen)
     target_sources("${PROJECT_NAME}"
     PRIVATE
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_front_amd.c
     )
else()
     target_sources("${PROJECT_NAME}"
     PRIVATE
     ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemm_front.c
     )
endif()

    add_subdirectory(ind)
