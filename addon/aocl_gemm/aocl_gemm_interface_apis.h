/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef AOCL_GEMM_INTERFACE_H
#define AOCL_GEMM_INTERFACE_H

#include "aocl_gemm_post_ops.h"
#include "aocl_bf16_type.h"

// Returns the size of buffer in bytes required for the reordered matrix.
#define AOCL_GEMM_GET_REORDER_BUF_SIZE(LP_SFX) \
BLIS_EXPORT_ADDON siz_t aocl_get_reorder_buf_size_ ## LP_SFX \
     ( \
       const char  order, \
       const char  trans, \
       const char  mat_type, \
       const dim_t k, \
       const dim_t n \
     ) \

AOCL_GEMM_GET_REORDER_BUF_SIZE(f32f32f32of32);
AOCL_GEMM_GET_REORDER_BUF_SIZE(u8s8s32os32);
AOCL_GEMM_GET_REORDER_BUF_SIZE(u8s8s16os16);
AOCL_GEMM_GET_REORDER_BUF_SIZE(bf16bf16f32of32);
AOCL_GEMM_GET_REORDER_BUF_SIZE(s8s8s32os32);
AOCL_GEMM_GET_REORDER_BUF_SIZE(s8s8s16os16);

// Performs reordering of input matrix. Reordering is the process of packing
// the entire matrix upfront, so that the benefits of packed matrix is obtained
// without incurring the packing costs during matmul computation.
#define AOCL_GEMM_REORDER(B_type,LP_SFX) \
BLIS_EXPORT_ADDON void aocl_reorder_ ## LP_SFX \
     ( \
       const char    order, \
       const char    trans, \
       const char    mat_type, \
       const B_type* input_buf_addr, \
       B_type*       reorder_buf_addr, \
       const dim_t   k, \
       const dim_t   n, \
       const dim_t   ldb \
     ) \

AOCL_GEMM_REORDER(float,f32f32f32of32);
AOCL_GEMM_REORDER(int8_t,u8s8s32os32);
AOCL_GEMM_REORDER(int8_t,u8s8s16os16);
AOCL_GEMM_REORDER(bfloat16,bf16bf16f32of32);
AOCL_GEMM_REORDER(int8_t,s8s8s32os32);
AOCL_GEMM_REORDER(int8_t,s8s8s16os16);

// Only supports matrices in row major format. This api can perform gemm with
// both normal as well as reordered B matrix as opposesd to sgemm (only
// supports former). This api can be considered analogous to packed sgemm api.
#define AOCL_GEMM_MATMUL(A_type,B_type,C_type,Sum_type,LP_SFX) \
BLIS_EXPORT_ADDON void aocl_gemm_ ## LP_SFX \
     ( \
       const char     order, \
       const char     transa, \
       const char     transb, \
       const dim_t    m, \
       const dim_t    n, \
       const dim_t    k, \
       const Sum_type alpha, \
       const A_type*  a, \
       const dim_t    lda, \
       const char     mem_format_a, \
       const B_type*  b, \
       const dim_t    ldb, \
       const char     mem_format_b, \
       const Sum_type beta, \
       C_type*        c, \
       const dim_t    ldc, \
       aocl_post_op*  post_op_unparsed \
     ) \

AOCL_GEMM_MATMUL(float,float,float,float,f32f32f32of32);
AOCL_GEMM_MATMUL(uint8_t,int8_t,int32_t,int32_t,u8s8s32os32);
AOCL_GEMM_MATMUL(uint8_t,int8_t,int16_t,int16_t,u8s8s16os16);
AOCL_GEMM_MATMUL(bfloat16,bfloat16,float,float,bf16bf16f32of32);
AOCL_GEMM_MATMUL(uint8_t,int8_t,int8_t,int32_t,u8s8s32os8);
AOCL_GEMM_MATMUL(uint8_t,int8_t,int8_t,int16_t,u8s8s16os8);
AOCL_GEMM_MATMUL(uint8_t,int8_t,uint8_t,int16_t,u8s8s16ou8);
AOCL_GEMM_MATMUL(bfloat16,bfloat16,bfloat16,float,bf16bf16f32obf16);
AOCL_GEMM_MATMUL(int8_t,int8_t,int32_t,int32_t,s8s8s32os32);
AOCL_GEMM_MATMUL(int8_t,int8_t,int8_t,int32_t,s8s8s32os8);
AOCL_GEMM_MATMUL(int8_t,int8_t,int16_t,int16_t,s8s8s16os16);
AOCL_GEMM_MATMUL(int8_t,int8_t,int8_t,int16_t,s8s8s16os8);

#endif // AOCL_GEMM_INTERFACE_H
