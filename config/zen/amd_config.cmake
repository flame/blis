#[=[

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

if(MSVC)
    set(CKVECFLAGS -mavx2 -mfma -mno-fma4 -mno-tbm -mno-xop -mno-lwp)

elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    if(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 15.0.0)
        # gcc 15.1.0 fails to compile SUP kernels if -ftree-slp-vectorize
        # is enabled, which is default in -O2 and higher
        list(APPEND CKOPTFLAGS -fno-tree-slp-vectorize)
    endif()
    set(CKVECFLAGS -mavx2 -mfpmath=sse -mfma)

elseif("${CMAKE_C_COMPILER_ID}" MATCHES "Clang")
    set(CKVECFLAGS -mavx2 -mfpmath=sse -mfma -mno-fma4 -mno-tbm -mno-xop -mno-lwp)
    execute_process(COMMAND ${CMAKE_C_COMPILER} --version OUTPUT_VARIABLE clang_full_version_string)
    string(REGEX MATCH "^[^\n]*" CLANG_VERSION_STRING "${clang_full_version_string}")
    string(REGEX MATCHALL "(AOCC.LLVM)" CLANG_STRING "${CLANG_VERSION_STRING}")
    if("${CLANG_STRING}" MATCHES "(AOCC.LLVM)")
        list(APPEND CKVECFLAGS -mllvm -disable-licm-vrp)
    endif()

else()
    message(FATAL_ERROR "gcc or clang are required for this configuration.")
endif()

# Flags specific to reference kernels.
set(CROPTFLAGS ${CKOPTFLAGS})
if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(CRVECFLAGS ${CKVECFLAGS} -funsafe-math-optimizations -ffp-contract=fast)
elseif("${CMAKE_C_COMPILER_ID}" MATCHES "Clang")
    set(CRVECFLAGS ${CKVECFLAGS} -funsafe-math-optimizations -ffp-contract=fast)
else()
    set(CRVECFLAGS ${CKVECFLAGS})
endif()
