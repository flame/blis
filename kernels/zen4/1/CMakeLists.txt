##Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.##

add_library(zen4_1
     OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_amaxv_zen_int_avx512.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_scalv_zen_int_avx512.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_dotv_zen_int_avx512.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_axpyv_zen_int_avx512.c
    )

target_compile_options(zen4_1 PRIVATE /arch:AVX2 /arch:AVX512)
if(BUILD_SHARED_LIBS)
    target_compile_definitions(zen4_1 PUBLIC -DBLIS_IS_BUILDING_LIBRARY)
endif()
