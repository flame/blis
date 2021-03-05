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

// This file contains the API points for the low precision POWER10 GEMM kernels

#include "generic_gemm.h"
#include "gemm_api.h"

#define GEMM_FUNC(ch, DTYPE_IN, DTYPE_OUT, A_ALIGNMENT, B_ALIGNMENT, MR, NR, MC, KC, NC) \
\
void GEMM_FUNC_NAME(ch) \
    ( \
        trans_t transa, \
        trans_t transb, \
        dim_t   m, \
        dim_t   n, \
        dim_t   k, \
        DTYPE_OUT*  alpha, \
        DTYPE_IN*   a, inc_t rsa, inc_t csa, \
        DTYPE_IN*   b, inc_t rsb, inc_t csb, \
        DTYPE_OUT*  beta, \
        DTYPE_OUT*  c, inc_t rsc, inc_t csc \
    ) \
{ \
\
    if (transa != BLIS_NO_TRANSPOSE || transb != BLIS_NO_TRANSPOSE) { \
        printf("Transpose functionality not implemented yet.\n"); \
    } \
\
    GEMM_PASTEMAC(ch) \
    ( \
        MR, NR, MC, KC, NC, \
        m, n, k, \
        a, rsa, csa, A_ALIGNMENT, \
        b, rsb, csb, B_ALIGNMENT, \
        c, rsc, csc, \
        alpha, beta \
    ); \
} \

//          ch       dt_in   dt_out           MR   NR     MC     KC     NC
GEMM_FUNC(  sb,   bfloat16,   float,   0,  0,  8,  16,  1664,  1026,  4096);
GEMM_FUNC(  sh,    float16,   float,   0,  0,  8,  16,  1664,  1026,  4096);
GEMM_FUNC( i16,    int16_t, int32_t,   0,  0,  8,  16,  1664,  1026,  4096);
GEMM_FUNC(  i8,     int8_t, int32_t,   0,  0,  8,  16,  1664,  1026,  4096);
GEMM_FUNC(  i4,    nibbles, int32_t,   0,  0,  8,  16,  1664,  1026,  4096);
