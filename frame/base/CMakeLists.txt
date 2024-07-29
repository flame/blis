#[=[

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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

