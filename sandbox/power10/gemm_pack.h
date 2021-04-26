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

// Templates for packing routines prototypes

#include "bli_sandbox.h"

#define PACK_FUNC_NAME_(ch, mat) ch ## _pack ## mat
#define PACK_FUNC_NAME(ch, mat)  PACK_FUNC_NAME_(ch, mat)

#define PACK_MACRO_PROTO(ch, DTYPE_IN) \
\
void PACK_FUNC_NAME(ch, A) \
    (  \
        dim_t MR, \
        int m, int k, \
        DTYPE_IN* ap, int rs_a, int cs_a, \
        DTYPE_IN* apack \
    ); \
\
void PACK_FUNC_NAME(ch, B) \
    ( \
        dim_t NR, \
        int k, int n, \
        DTYPE_IN* bp, int rs_b, int cs_b, \
        DTYPE_IN* bpack \
    ); 

PACK_MACRO_PROTO(sb, bfloat16)
PACK_MACRO_PROTO(sh, float16)
PACK_MACRO_PROTO(i16, int16_t)
PACK_MACRO_PROTO(i8, int8_t)
PACK_MACRO_PROTO(i4, nibbles)
