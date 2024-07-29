#[=[

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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

# FLAGS that are specific to the 'zen4' architecture are added here.
# FLAGS that are common for all the AMD architectures are present in
# config/zen/amd_config.mk.

# Include file containing common flags for all AMD architectures
include(${CMAKE_SOURCE_DIR}/config/zen/amd_config.cmake)
if(NOT WIN32)
    if(NOT (DEBUG_TYPE STREQUAL "off"))
        set(CDBGFLAGS -g)
    endif()

    if(DEBUG_TYPE STREQUAL "noopt")
        set(COPTFLAGS -O0)
    else() # off or opt
        set(COPTFLAGS -O3)
    endif()
endif()

# Flags specific to LPGEMM kernels.
set(CKLPOPTFLAGS "")

# Flags specific to optimized kernels.
# NOTE: The -fomit-frame-pointer option is needed for some kernels because
# they make explicit use of the rbp register.
if(MSVC)
    set(CKOPTFLAGS ${COPTFLAGS} /Oy)
else()
    set(CKOPTFLAGS ${COPTFLAGS} -fomit-frame-pointer)
endif()

if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    if(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0.0)
        # gcc 13.0 or later
        list(APPEND CKVECFLAGS -march=znver4)
        list(APPEND CRVECFLAGS -march=znver4)
        # Update CKLPOPTFLAGS for gcc to use O3 optimization without
        # -ftree-pre and -ftree-partial-pre flag. These flag results
        # in suboptimal code generation for instrinsic based kernels.
        # The -ftree-loop-vectorize results in inefficient code gen
        # for amd optimized l1 kernels based on instrinsics.
        list(APPEND CKLPOPTFLAGS -fno-tree-partial-pre -fno-tree-pre -fno-tree-loop-vectorize)
    elseif(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0.0)
        # gcc 11.0 or later
        list(APPEND CKVECFLAGS -march=znver3 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx512vnni -mavx512bf16 -mavx512vbmi)
        list(APPEND CRVECFLAGS -march=znver3)
        list(APPEND CKLPOPTFLAGS -fno-tree-partial-pre -fno-tree-pre -fno-tree-loop-vectorize)
    elseif(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 9.0.0)
        # gcc 9.0 or later
        list(APPEND CKVECFLAGS -march=znver2 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx512vnni -mavx512vbmi)
        list(APPEND CRVECFLAGS -march=znver2)
        list(APPEND CKLPOPTFLAGS -fno-tree-partial-pre -fno-tree-pre -fno-tree-loop-vectorize)
    elseif(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 8.0.0)
        # gcc 8.0 or later
        list(APPEND CKVECFLAGS -march=znver1 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx512vnni -mavx512vbmi)
        list(APPEND CRVECFLAGS -march=znver1)
    elseif(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 7.0.0)
        # gcc 7.0 or later
        list(APPEND CKVECFLAGS -march=znver1 -mavx512f -mavx512dq -mavx512bw -mavx512vl)
        list(APPEND CRVECFLAGS -march=znver1)
    else()
        # If gcc is older than 7.0.0 but at least 6.1.0, then we can use -march=znver1
        # as the fallback option.
        list(APPEND CKVECFLAGS -march=znver1 -mno-avx256-split-unaligned-store)
        list(APPEND CRVECFLAGS -march=znver1 -mno-avx256-split-unaligned-store)
    endif()
endif() # gcc

if("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
    # AOCC clang has various formats for the version line

    # AOCC.LLVM.2.0.0.B191.2019_07_19 clang version 8.0.0 (CLANG: Jenkins AOCC_2_0_0-Build#191) (based on LLVM AOCC.LLVM.2.0.0.B191.2019_07_19)
    # AOCC.LLVM.2.1.0.B1030.2019_11_12 clang version 9.0.0 (CLANG: Build#1030) (based on LLVM AOCC.LLVM.2.1.0.B1030.2019_11_12)
    # AMD clang version 10.0.0 (CLANG: AOCC_2.2.0-Build#93 2020_06_25) (based on LLVM Mirror.Version.10.0.0)
    # AMD clang version 11.0.0 (CLANG: AOCC_2.3.0-Build#85 2020_11_10) (based on LLVM Mirror.Version.11.0.0)
    # AMD clang version 12.0.0 (CLANG: AOCC_3.0.0-Build#2 2020_11_05) (based on LLVM Mirror.Version.12.0.0)
    # AMD clang version 14.0.0 (CLANG: AOCC_4.0.0-Build#98 2022_06_15) (based on LLVM Mirror.Version.14.0.0)
    # For our purpose we just want to know if it version 2x or 3x or 4x

    # But also set these in case we are using upstream LLVM clang
    execute_process(COMMAND ${CMAKE_C_COMPILER} --version OUTPUT_VARIABLE clang_full_version_string)
    string(REGEX MATCH "^[^\n]*" CLANG_VERSION_STRING "${clang_full_version_string}")
    string(REGEX MATCHALL "(AOCC_2|AOCC_3|AOCC_4|AOCC|LLVM|clang)" CLANG_STRING "${CLANG_VERSION_STRING}")
    string(REGEX REPLACE ".*clang version ([0-9]+\\.[0-9]+).*" "\\1" CLANG_VERSION "${CLANG_VERSION_STRING}")

    if(NOT WIN32)
      set(alignloops "-falign-loops=64")
    endif()
    if("${CLANG_STRING}" MATCHES "AOCC_4")
      # AOCC version 4x we will enable znver4
      list(APPEND CKVECFLAGS -march=znver4 ${alignloops})
      list(APPEND CRVECFLAGS -march=znver4)
    elseif("${CLANG_STRING}" MATCHES "AOCC_3")
      # AOCC version 3x we will enable znver3
      list(APPEND CKVECFLAGS -march=znver3 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx512vnni -mavx512bf16 -mavx512vbmi ${alignloops})
      list(APPEND CRVECFLAGS -march=znver3)
    elseif("${CLANG_STRING}" MATCHES "(AOCC_2|LLVM)")
      # AOCC version 2x we will enable znver2
      list(APPEND CKVECFLAGS -march=znver2 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx512vnni -mavx512vbmi)
      list(APPEND CRVECFLAGS -march=znver2)
    elseif(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 16.0.0)
      # LLVM clang 16.0 or later
      list(APPEND CKVECFLAGS -march=znver4 ${alignloops})
      list(APPEND CRVECFLAGS -march=znver4)
    elseif(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0.0)
      # LLVM clang 13.0 or later
      list(APPEND CKVECFLAGS -march=znver3 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx512vnni -mavx512bf16 -mavx512vbmi ${alignloops})
      list(APPEND CRVECFLAGS -march=znver3)
    elseif(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 9.0.0)
      # LLVM clang 9.0 or later
      list(APPEND CKVECFLAGS -march=znver2 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx512vnni -mavx512bf16 -mavx512vbmi ${alignloops})
      list(APPEND CRVECFLAGS -march=znver2)
    else()
      list(APPEND CKVECFLAGS -march=znver1 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mavx512vnni -mavx512vbmi ${alignloops})
      list(APPEND CRVECFLAGS -march=znver1)
    endif()
endif()

# Flags specific to reference kernels.
set(CROPTFLAGS ${CKOPTFLAGS})
set(CRVECFLAGS ${CKVECFLAGS})
