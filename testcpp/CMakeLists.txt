##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/cpp)

add_executable(test_asum_blis test_asum.cc)
target_link_libraries(test_asum_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_axpy_blis test_axpy.cc)
target_link_libraries(test_axpy_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_copy_blis test_copy.cc)
target_link_libraries(test_copy_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_dot_blis test_dot.cc)
target_link_libraries(test_dot_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_dotc_blis test_dotc.cc)
target_link_libraries(test_dotc_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_gbmv_blis test_gbmv.cc)
target_link_libraries(test_gbmv_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_gemm_blis test_gemm.cc)
target_link_libraries(test_gemm_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_gemv_blis test_gemv.cc)
target_link_libraries(test_gemv_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_ger_blis test_ger.cc)
target_link_libraries(test_ger_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_gerc_blis test_gerc.cc)
target_link_libraries(test_gerc_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_geru_blis test_geru.cc)
target_link_libraries(test_geru_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_hemm_blis test_hemm.cc)
target_link_libraries(test_hemm_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_hemv_blis test_hemv.cc)
target_link_libraries(test_hemv_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_her2_blis test_her2.cc)
target_link_libraries(test_her2_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_her_blis test_her.cc)
target_link_libraries(test_her_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_herk_blis test_herk.cc)
target_link_libraries(test_herk_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_hpr2_blis test_hpr2.cc)
target_link_libraries(test_hpr2_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_hpr_blis test_hpr.cc)
target_link_libraries(test_hpr_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_nrm2_blis test_nrm2.cc)
target_link_libraries(test_nrm2_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_rot_blis test_rot.cc)
target_link_libraries(test_rot_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_rotg_blis test_rotg.cc)
target_link_libraries(test_rotg_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_rotm_blis test_rotm.cc)
target_link_libraries(test_rotm_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_rotmg_blis test_rotmg.cc)
target_link_libraries(test_rotmg_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_scal_blis test_scal.cc)
target_link_libraries(test_scal_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_sdsdot_blis test_sdsdot.cc)
target_link_libraries(test_sdsdot_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_spr2_blis test_spr2.cc)
target_link_libraries(test_spr2_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_spr_blis test_spr.cc)
target_link_libraries(test_spr_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_swap_blis test_swap.cc)
target_link_libraries(test_swap_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_symm_blis test_symm.cc)
target_link_libraries(test_symm_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_syr2_blis test_syr2.cc)
target_link_libraries(test_syr2_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_syr2k_blis test_syr2k.cc)
target_link_libraries(test_syr2k_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_syr_blis test_syr.cc)
target_link_libraries(test_syr_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_syrk_blis test_syrk.cc)
target_link_libraries(test_syrk_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_tbmv_blis test_tbmv.cc)
target_link_libraries(test_tbmv_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_tbsv_blis test_tbsv.cc)
target_link_libraries(test_tbsv_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_tpmv_blis test_tpmv.cc)
target_link_libraries(test_tpmv_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_tpsv_blis test_tpsv.cc)
target_link_libraries(test_tpsv_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_trmm_blis test_trmm.cc)
target_link_libraries(test_trmm_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_trsm_blis test_trsm.cc)
target_link_libraries(test_trsm_blis PRIVATE "${LIB_NAME}.lib" )

add_executable(test_trsv_blis test_trsv.cc)
target_link_libraries(test_trsv_blis PRIVATE "${LIB_NAME}.lib" )
