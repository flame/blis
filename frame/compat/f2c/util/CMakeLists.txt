##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}"
    PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/bla_c_abs.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_c_abs.h
${CMAKE_CURRENT_SOURCE_DIR}/bla_c_div.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_c_div.h
${CMAKE_CURRENT_SOURCE_DIR}/bla_d_abs.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_d_abs.h
${CMAKE_CURRENT_SOURCE_DIR}/bla_d_cnjg.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_d_cnjg.h
${CMAKE_CURRENT_SOURCE_DIR}/bla_d_imag.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_d_imag.h
${CMAKE_CURRENT_SOURCE_DIR}/bla_d_sign.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_d_sign.h
${CMAKE_CURRENT_SOURCE_DIR}/bla_f__cabs.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_f__cabs.h
${CMAKE_CURRENT_SOURCE_DIR}/bla_r_abs.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_r_abs.h
${CMAKE_CURRENT_SOURCE_DIR}/bla_r_cnjg.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_r_cnjg.h
${CMAKE_CURRENT_SOURCE_DIR}/bla_r_imag.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_r_imag.h
${CMAKE_CURRENT_SOURCE_DIR}/bla_r_sign.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_r_sign.h
${CMAKE_CURRENT_SOURCE_DIR}/bla_z_abs.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_z_abs.h
${CMAKE_CURRENT_SOURCE_DIR}/bla_z_div.c
${CMAKE_CURRENT_SOURCE_DIR}/bla_z_div.h
	)

	

