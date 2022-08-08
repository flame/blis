/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef BLIS_GEMM_INT16_NFRINGE
#define BLIS_GEMM_INT16_NFRINGE
#include "lpgemm_post_ops.h"

// 6x16 int8o16 kernel
void lpgemm_rowvar_u8s8s16o16_6x16
	(
       const dim_t    m0,
       const dim_t    k0,
       const uint8_t  *a,
       const dim_t    rs_a,
       const dim_t    cs_a,
       const dim_t    ps_a,
       const int8_t   *b,
       const dim_t    rs_b,
       const dim_t    cs_b,
       int16_t        *c,
       const dim_t    rs_c,
       const int16_t  alpha,
       const int16_t  beta,
       bool           is_last_k,
       dim_t          post_op_c_i,
       dim_t          post_op_c_j,
       lpgemm_post_op *post_ops_list
	);

// 6xlt16 int8o16 kernel
void lpgemm_rowvar_u8s8s16o16_6xlt16
	 (
	   const dim_t    m0,
	   const dim_t    k0,
	   const uint8_t  *a,
	   const dim_t    rs_a,
	   const dim_t    cs_a,
	   const dim_t    ps_a,
	   const int8_t   *b,
	   const dim_t    rs_b,
	   const dim_t    cs_b,
	   int16_t        *c,
	   const dim_t    rs_c,
	   const int16_t  alpha,
	   const int16_t  beta,
	   const dim_t    n0_rem,
	   bool           is_last_k,
	   dim_t          post_op_c_i,
	   dim_t          post_op_c_j,
	   lpgemm_post_op *post_ops_list
	 );

#endif // BLIS_GEMM_INT16_NFRINGE