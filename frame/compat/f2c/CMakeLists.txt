##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources("${PROJECT_NAME}"
    PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/bla_cabs1.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_gbmv.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_hbmv.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_hpmv.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_hpr.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_hpr2.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_lsame.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_rot.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_rotg.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_rotm.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_rotmg.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_sbmv.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_spmv.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_spr.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_spr2.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_tbmv.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_tbsv.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_tpmv.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_tpsv.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_xerbla.c
    )

#Add all subdirectories
    add_subdirectory(util)



