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

#ifndef AOCL_GEMM_F32F32F32OF32_UTILS_H
#define AOCL_GEMM_F32F32F32OF32_UTILS_H

// Returns the size of buffer in bytes required for the reordered matrix.
BLIS_EXPORT_ADDON siz_t aocl_get_reorder_buf_size_f32f32f32of32
     (
       const char  mat_type,
       const dim_t k,
       const dim_t n
     );

// Performs reordering of input matrix. Reordering is the process of packing
// the entire matrix upfront, so that the benefits of packed matrix is obtained
// without incurring the packing costs during matmul computation.
BLIS_EXPORT_ADDON void aocl_reorder_f32f32f32of32
     (
       const char   mat_type,
       const float* input_buf_addr_b,
       float*       reorder_buf_addr_b,
       const dim_t  k,
       const dim_t  n,
       const dim_t  ldb
     );

#endif //AOCL_GEMM_F32F32F32OF32_UTILS_H
