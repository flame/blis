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

#ifndef BLIS_LPGEMM_ELTWISE_OPS_KERN_H
#define BLIS_LPGEMM_ELTWISE_OPS_KERN_H

#define LPGEMM_ELTWISE_OPS_KERNEL(A_type,B_type,LP_SFX) \
void lpgemm_eltwise_ops_kernel_ ## LP_SFX \
     ( \
       const dim_t         m0, \
       const dim_t         n0, \
       const A_type*       a, \
       const dim_t         rs_a, \
       const dim_t         cs_a, \
       B_type*             b, \
       const dim_t         rs_b, \
       const dim_t         cs_b, \
       lpgemm_post_op*     post_ops_list, \
       lpgemm_post_op_attr post_ops_attr \
     ) \

LPGEMM_ELTWISE_OPS_KERNEL(bfloat16,float,bf16of32_6x64);
LPGEMM_ELTWISE_OPS_KERNEL(float,float,f32of32_6x64);

#define LPGEMM_ELTWISE_OPS_M_FRINGE_KERNEL(A_type,B_type,LP_SFX) \
void lpgemm_eltwise_ops_kernel_ ## LP_SFX \
     ( \
       const dim_t         n0, \
       const A_type*       a, \
       const dim_t         rs_a, \
       const dim_t         cs_a, \
       B_type*             b, \
       const dim_t         rs_b, \
       const dim_t         cs_b, \
       lpgemm_post_op*     post_ops_list, \
       lpgemm_post_op_attr post_ops_attr \
     ) \

LPGEMM_ELTWISE_OPS_M_FRINGE_KERNEL(bfloat16,float,bf16of32_5x64);
LPGEMM_ELTWISE_OPS_M_FRINGE_KERNEL(bfloat16,float,bf16of32_4x64);
LPGEMM_ELTWISE_OPS_M_FRINGE_KERNEL(bfloat16,float,bf16of32_3x64);
LPGEMM_ELTWISE_OPS_M_FRINGE_KERNEL(bfloat16,float,bf16of32_2x64);
LPGEMM_ELTWISE_OPS_M_FRINGE_KERNEL(bfloat16,float,bf16of32_1x64);
LPGEMM_ELTWISE_OPS_M_FRINGE_KERNEL(float,float,f32of32_5x64);
LPGEMM_ELTWISE_OPS_M_FRINGE_KERNEL(float,float,f32of32_4x64);
LPGEMM_ELTWISE_OPS_M_FRINGE_KERNEL(float,float,f32of32_3x64);
LPGEMM_ELTWISE_OPS_M_FRINGE_KERNEL(float,float,f32of32_2x64);
LPGEMM_ELTWISE_OPS_M_FRINGE_KERNEL(float,float,f32of32_1x64);

#endif //BLIS_LPGEMM_ELTWISE_OPS_KERN_H
