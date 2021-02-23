##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

target_sources("${PROJECT_NAME}"
    PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/bla_c_abs.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_c_div.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_d_abs.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_d_cnjg.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_d_imag.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_d_sign.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_f__cabs.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_r_abs.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_r_cnjg.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_r_imag.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_r_sign.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_z_abs.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_z_div.c
    )



