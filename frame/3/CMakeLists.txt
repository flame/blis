##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}" 
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_blocksize.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_blocksize.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_check.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_check.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_cntl.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_cntl.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_direct.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_direct.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_ft_ex.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_ft_ukr.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_oapi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_oapi.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_oapi_ba.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_oapi_ex.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_oft.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_oft_var.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_packm.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_packm.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_prune.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_prune.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup_ft_ker.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup_int.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup_int.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup_ker.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup_ker_prot.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup_oft.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup_packm_a.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup_packm_a.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup_packm_b.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup_packm_b.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup_packm_var.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup_packm_var.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup_ref.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup_var12.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup_var1n2m.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup_vars.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_tapi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_tapi.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_tapi_ba.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_tapi_ex.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_thrinfo.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_thrinfo.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_ukr.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_ukr_fpa.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_ukr_fpa.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_ukr_oapi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_ukr_oapi.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_ukr_prot.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_ukr_tapi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_ukr_tapi.h
	)

set(SUBDIRECTORIES "gemm" "hemm" "her2k" "herk" "symm" "syr2k" "syrk" "trmm" "trmm3" "trsm")

#Add all subdirectories 
foreach(VAR ${SUBDIRECTORIES})
	add_subdirectory(${VAR})
endforeach()
