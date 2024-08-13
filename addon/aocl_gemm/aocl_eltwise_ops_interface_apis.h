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

#ifndef AOCL_ELTWISE_OPS_INTERFACE_H
#define AOCL_ELTWISE_OPS_INTERFACE_H

#include "aocl_gemm_post_ops.h"
#include "aocl_bf16_type.h"

#define AOCL_UTIL_ELTWISE_OPS(A_type,B_type,LP_SFX) \
BLIS_EXPORT_ADDON void aocl_gemm_eltwise_ops_ ## LP_SFX \
     ( \
       const char     order, \
       const char     transa, \
       const char     transb, \
       const dim_t    m, \
       const dim_t    n, \
       const A_type*  a, \
       const dim_t    lda, \
       B_type*        b, \
       const dim_t    ldb, \
       aocl_post_op*  post_op_unparsed \
     ) \

AOCL_UTIL_ELTWISE_OPS(bfloat16,float,bf16of32);
AOCL_UTIL_ELTWISE_OPS(bfloat16,bfloat16,bf16obf16);
AOCL_UTIL_ELTWISE_OPS(float,float,f32of32);

#endif // AOCL_ELTWISE_OPS_INTERFACE_H
