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
    - Neither the name of copyright holder(s) nor the names
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

#ifndef BLIS_SANDBOX_H
#define BLIS_SANDBOX_H

#include "blis.h"
#include "gemm_api.h"

// NOTE: This header is the only header required to be present in the sandbox
// implementation directory.

// This header is used to create the typedefs needed for low precision

// int4 type 
typedef union
{
    uint8_t v;
    struct
    {
        uint8_t nib1:4;
        uint8_t nib2:4;
    } bits;
} nibbles;

// bfloat16 
typedef union
{
    uint16_t v;
    struct
    {
        uint16_t m:7;
        uint16_t e:8;
        uint16_t s:1;
    } bits;
} bfloat16;

// ieee float16 
typedef union
{
    uint16_t v;
    struct
    {
        uint16_t m:10;
        uint16_t e:5;
        uint16_t s:1;
    } bits;
} float16;

#define P10_PG_SIZE 4096

GEMM_UKR_PROT2( bfloat16,   float,  sb, gemm_power10_mma_8x16 )
GEMM_UKR_PROT2(  float16,   float,  sh, gemm_power10_mma_8x16 )
GEMM_UKR_PROT2(  int16_t, int32_t, i16, gemm_power10_mma_8x16 )
GEMM_UKR_PROT2(   int8_t, int32_t,  i8, gemm_power10_mma_8x16 )
GEMM_UKR_PROT2(  nibbles, int32_t,  i4, gemm_power10_mma_8x16 )

/* Creates a function that initializes a matrix of type ctype with random vals */
#define RandomMatrixMacro(ch, ctype, rand_func) \
    RM_PROT(ch, ctype) \
    { \
    for ( int i=0; i<m; i++ ) \
        for ( int j=0; j<n; j++ ) \
            *(ap + j*cs_a + i*rs_a) = \
                (ctype) rand_func(); \
    }

/* Creates a function that initializes a matrix of type ctype with random vals */
#define RandomMatrixBounded(ch, ctype, rand_func) \
    RM_B_PROT(ch, ctype) \
    { \
    for ( int i=0; i<m; i++ ) \
        for ( int j=0; j<n; j++ ) \
            *(ap + j*cs_a + i*rs_a) = \
                (ctype) rand_func() % (upper - lower + 1) + lower; \
    }

GEMM_FUNC_PROT(  float16,   float,  sh);
GEMM_FUNC_PROT( bfloat16,   float,  sb);
GEMM_FUNC_PROT(  int16_t, int32_t, i16);
GEMM_FUNC_PROT(   int8_t, int32_t,  i8);
GEMM_FUNC_PROT(  nibbles, int32_t,  i4);

#endif
