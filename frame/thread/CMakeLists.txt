##Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.##

target_sources("${PROJECT_NAME}"
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_compute_decor_openmp.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_compute_decor_single.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_decor_openmp.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_decor_pthreads.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_decor_single.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup_decor_openmp.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup_decor_pthreads.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_l3_sup_decor_single.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_pthread.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_thrcomm.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_thrcomm_openmp.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_thrcomm_pthreads.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_thrcomm_single.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_thread.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_thrinfo.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_thrinfo_sup.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_pack_full_decor_openmp.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_pack_full_decor_single.c
    )
