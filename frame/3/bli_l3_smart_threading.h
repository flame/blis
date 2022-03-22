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

#ifdef AOCL_DYNAMIC

#ifndef BLIS_L3_SMART_THREADING_H
#define BLIS_L3_SMART_THREADING_H

// Smart threading encompasses the following multi-threading related
// optimizations:
//	1. Selection of optimal number of threads (BLIS_NUM_THREADS) based
//	on matrix dimensions.
// 	2. Factorization of threads along m and n dimensions (BLIS_IC_NT,
// 	BLIS_JC_NT) based on matrix dimensions and cache friendliness.
// 	3. Transformation of native to SUP path based on the per thread matrix
// 	dimensions after thread factorization, given that per thread dimensions
// 	are within SUP limits.
// 	4. Enabling packing of B alone in SUP path if native -> SUP path
// 	transformation happened and depending on per thread matrix dimensions.
// This function captures smart threading logic fine tuned for gemm SUP path.
// Optimal thread selection is not enabled now.
err_t bli_gemm_smart_threading_sup
     (
       num_t dt,
       siz_t elem_size,
       const bool is_rrr_rrc_rcr_crr,
       const dim_t m,
       const dim_t n,
       const dim_t k,
       const dim_t max_available_nt,
       cntx_t* cntx,
       rntm_t* rntm
     );

#endif //BLIS_L3_SMART_THREADING_H

#endif
