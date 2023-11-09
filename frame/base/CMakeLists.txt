##Copyright (C) 2021 - 2023, Advanced Micro Devices, Inc. All rights reserved.##

target_sources("${PROJECT_NAME}"
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_apool.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_arch.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_array.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_blksz.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_check.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_clock.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_cntl.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_cntx.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_const.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_cpuid.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_env.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_error.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_func.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_getopt.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gks.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_info.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_init.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_machval.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_malloc.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_mbool.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_pba.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_memsys.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_obj.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_obj_scalar.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_pack.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_param_map.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_part.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_pool.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_prune.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_query.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_rntm.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_sba.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_setgetijm.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_setgetijv.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_setri.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_string.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_winsys.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_pack_compute_utils.c
    )

#Add all subdirectories
    add_subdirectory(cast)
    add_subdirectory(check)
    add_subdirectory(noopt)
    add_subdirectory(proj)

