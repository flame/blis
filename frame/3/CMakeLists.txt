##Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.##

target_sources("${PROJECT_NAME}"
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_blocksize.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_check.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_cntl.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_direct.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_oapi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_oapi_ba.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_oapi_ex.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_packm.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_prune.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup_packm_a.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup_packm_b.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup_packm_var.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup_var12.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup_var1n2m.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_tapi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_tapi_ba.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_tapi_ex.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_thrinfo.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_ukr_fpa.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_ukr_oapi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_ukr_tapi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_smart_threading.c
    )
# Select AMD specific sources for AMD configurations.
if(${TARGET_ARCH} STREQUAL zen OR 
   ${TARGET_ARCH} STREQUAL zen2 OR 
   ${TARGET_ARCH} STREQUAL zen3 OR
   ${TARGET_ARCH} STREQUAL zen4 OR 
   ${TARGET_ARCH} STREQUAL amdzen)
    target_sources("${PROJECT_NAME}"
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup_int_amd.c
    )
else()
    target_sources("${PROJECT_NAME}"
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup_int.c
    )
endif()

set(SUBDIRECTORIES "gemm" "hemm" "her2k" "herk" "symm" "syr2k" "syrk" "trmm" "trmm3" "trsm" "gemmt")

#Add all subdirectories
foreach(VAR ${SUBDIRECTORIES})
    add_subdirectory(${VAR})
endforeach()
