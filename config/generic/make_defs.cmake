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

# Flags specific to optimized kernels.
if(MSVC)
    set(CKOPTFLAGS ${COPTFLAGS})
else()
    set(CKOPTFLAGS ${COPTFLAGS} -O3)
endif()

if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    # Placeholder in case we want to add gcc-specific flags.
elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "icc")
    # Placeholder in case we want to add icc-specific flags.
elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
    # Placeholder in case we want to add clang-specific flags.
else()
    message(FATAL_ERROR "gcc, icc, or clang is required for this configuration.")
endif()

# Flags specific to reference kernels.
set(CROPTFLAGS ${CKOPTFLAGS})
if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(CRVECFLAGS ${CKVECFLAGS})
elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
    set(CRVECFLAGS ${CKVECFLAGS})
else()
    set(CRVECFLAGS ${CKVECFLAGS})
endif()
