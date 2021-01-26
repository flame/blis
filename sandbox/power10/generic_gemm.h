/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

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

// Prototypes and template for the 5-loop gemm algorithm

#include "bli_sandbox.h"

#define GEMM_PASTEMAC_(ch)           bli_ ## ch ## gemm_
#define GEMM_PASTEMAC(ch)            GEMM_PASTEMAC_(ch)

#define GENERIC_GEMM_PROTO(ch, DTYPE_IN, DTYPE_OUT) \
void GEMM_PASTEMAC(ch) \
    ( \
        dim_t MR, dim_t NR, dim_t KC, dim_t NC, dim_t MC, \
        int m, int n, int k, \
        DTYPE_IN* restrict A, int rs_a, int cs_a, int A_align, \
        DTYPE_IN* restrict B, int rs_b, int cs_b, int B_align, \
        DTYPE_OUT* restrict C, int rs_c, int cs_c, \
        DTYPE_OUT* alpha, DTYPE_OUT* beta \
    )

GENERIC_GEMM_PROTO( sb, bfloat16,   float);
GENERIC_GEMM_PROTO( sh,  float16,   float);
GENERIC_GEMM_PROTO(i16,  int16_t, int32_t);
GENERIC_GEMM_PROTO( i8,   int8_t, int32_t);
GENERIC_GEMM_PROTO( i4,  nibbles, int32_t);

