/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

  Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

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
#include "xmmintrin.h"
#include "blis.h"

#ifdef BLIS_ADDON_LPGEMM

#include "lpgemm_kernel_macros_f32_avx2.h"

// When n=1 is load 16x1 from B and load MRx16 from A and perform dot product
//  to produce C output of MRX1. The vectorization is done in k loop and
//  the horizontal reduction done to produce one output from each
//  accumulator register
void lpgemv_n_one_kernel_f32_avx2_ker_ft
(
    const dim_t           m0,
    const dim_t           k,
    const float           *a,
    const dim_t           rs_a,
    const dim_t           cs_a,
    const AOCL_MEMORY_TAG mtag_a,
    const float           *b,
    const dim_t           rs_b,
    const dim_t           cs_b,
    const AOCL_MEMORY_TAG mtag_b,
    float                 *c,
    const dim_t           rs_c,
    const dim_t           cs_c,
    const float           alpha,
    const float           beta,
    const dim_t           MR,
    const dim_t           KC,
    lpgemm_post_op        *post_op_list,
    lpgemm_post_op_attr   *post_op_attr
)
{
//TODO: Created dummy function as place holder to get 
//rid of linking issues in other zen configurations.
//AVX2 varient wil be implemented in next commits.
//Code will take LPGEMM path for LPGEMV in AVX2 env.
}

#endif // BLIS_ADDON_LPGEMM
