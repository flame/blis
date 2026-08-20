/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2025 - 2026, Advanced Micro Devices, Inc. All rights reserved.

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

*/

#include "immintrin.h"
#include "blis.h"

#define ARCH_SIMD_BITS  512
#define GEMV_ARCH_SUFFIX zen4_int
#define GEMV_BLK_SUFFIX_T(ch, MR, NR)  PASTEMAC5(ch, gemv_t_block_, MR, _, NR, _avx512)
#include "../../zen/2/bli_pp_common.h"
#include "bli_x86_asm_macros.h"

#ifdef BLIS_ENABLE_OPENMP
#include <omp.h>
    #define SHOULD_CALL_ST_s (size < 22000)
    #define SHOULD_CALL_ST_d (size < 11000)
    #define SHOULD_CALL_ST_c (size < 15000)
    #define SHOULD_CALL_ST_z (size < 8000 )
#endif

#include "../../zen/2/bli_gemv_t_impl.h"

GENERATE_KERNEL(float,    s, 48, 8)   // float:    MR=48, NR=8
GENERATE_KERNEL(double,   d, 32, 8)   // double:   MR=32, NR=8
GENERATE_KERNEL(scomplex, c, 40, 8)   // scomplex: MR=40, NR=8
GENERATE_KERNEL(dcomplex, z, 20, 8)   // dcomplex: MR=20, NR=8
