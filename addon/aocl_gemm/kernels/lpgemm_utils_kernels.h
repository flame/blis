/*

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

*/

#ifndef BLIS_LPGEMM_UTILS_KERN_H
#define BLIS_LPGEMM_UTILS_KERN_H

typedef void (*lpgemm_util_l1_op_f32_kernel_t)
     (
       const dim_t n,
       float*     x,
       const inc_t incx
     );

#define LPGEMM_UTIL_L1_OP_KERNEL(V_type,OP_type) \
void lpgemm_util_ ## OP_type ## _kernel \
     ( \
       const dim_t n, \
       V_type*     x, \
       const inc_t incx \
     ) \

// AVX512
LPGEMM_UTIL_L1_OP_KERNEL(float,f32_gelu_tanh_avx512);
LPGEMM_UTIL_L1_OP_KERNEL(float,f32_gelu_erf_avx512);
LPGEMM_UTIL_L1_OP_KERNEL(float,f32_softmax_avx512);

// AVX2
LPGEMM_UTIL_L1_OP_KERNEL(float,f32_gelu_tanh_avx2);
LPGEMM_UTIL_L1_OP_KERNEL(float,f32_gelu_erf_avx2);
LPGEMM_UTIL_L1_OP_KERNEL(float,f32_softmax_avx2);

#endif //BLIS_LPGEMM_UTILS_KERN_H
