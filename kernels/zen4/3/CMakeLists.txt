#[=[

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2024, Advanced Micro Devices, Inc. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

]=]

add_library(zen4_3
     OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmtrsm_l_zen_16x14.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmtrsm_u_zen_16x14.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmtrsm_l_zen4_8x24.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_gemmtrsm_u_zen4_8x24.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_dgemm_zen4_asm_32x6.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_dgemm_zen4_asm_8x24.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_trsm_small_AVX512.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_ztrsm_small_AVX512.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_zgemm_zen4_asm_12x4.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_zero_zmm.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_zgemm_zen4_asm_4x12.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_zgemmtrsm_l_4x12.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_zgemmtrsm_u_4x12.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_dgemm_avx512_k1.c
    )

target_compile_options(zen4_3 PRIVATE /arch:AVX2 /arch:AVX512)
if(BUILD_SHARED_LIBS)
    target_compile_definitions(zen4_3 PUBLIC -DBLIS_IS_BUILDING_LIBRARY)
endif()

add_subdirectory(sup)
