##Copyright (C) 2020-2022, Advanced Micro Devices, Inc. All rights reserved.##

target_sources("${PROJECT_NAME}"
     PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_axpyf_zen_int_8.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_dotxf_zen_int_8.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_axpyf_zen_int_5.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_axpyf_zen_int_4.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_axpyf_zen_int_6.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_axpy2v_zen_int.c
<<<<<<< HEAD
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_dotxaxpyf_zen_int_8.c           
=======
>>>>>>> 8b5b2707... Optimized daxpy2v implementation
    )
