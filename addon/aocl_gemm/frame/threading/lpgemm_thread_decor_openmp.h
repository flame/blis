/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef LPGEMM_THREAD_DECOR_OPENMP_H
#define LPGEMM_THREAD_DECOR_OPENMP_H

#include "lpgemm_types.h"
#include "lpgemm_post_ops.h"
#include "aocl_bf16_type.h"

#ifdef BLIS_ENABLE_OPENMP

#define GEN_LPGEMM_OPENMP_DECORATOR_FN(A_type,B_type,C_type,LPGEMM_SFX) \
void lpgemm_ ## LPGEMM_SFX ## _openmp_thread_decorator \
     ( \
       const dim_t           m, \
       const dim_t           n, \
       const dim_t           k, \
       const A_type*         a, \
       const dim_t           rs_a, \
       const dim_t           cs_a, \
       const AOCL_MEMORY_TAG mtag_a, \
       const B_type*         b, \
       const dim_t           rs_b, \
       const dim_t           cs_b, \
       const AOCL_MEMORY_TAG mtag_b, \
       C_type*               c, \
       const dim_t           rs_c, \
       const dim_t           cs_c, \
       const C_type          alpha, \
       const C_type          beta, \
       rntm_t*               rntm_g, \
       lpgemm_cntx_t*        lcntx, \
       lpgemm_post_op*       post_op_list, \
       AOCL_STORAGE_TYPE     c_downscale \
     ); \

GEN_LPGEMM_OPENMP_DECORATOR_FN(uint8_t,int8_t,int32_t,u8s8s32o32)
GEN_LPGEMM_OPENMP_DECORATOR_FN(bfloat16,bfloat16,float,bf16bf16f32of32)
GEN_LPGEMM_OPENMP_DECORATOR_FN(float,float,float,f32f32f32of32)
GEN_LPGEMM_OPENMP_DECORATOR_FN(int8_t,int8_t,int32_t,s8s8s32o32)


#define GEN_BATCH_LPGEMM_OPENMP_DECORATOR_FN(A_type,B_type,C_type,LPGEMM_SFX) \
void batch_lpgemm_ ## LPGEMM_SFX ## _openmp_thread_decorator \
     ( \
       const dim_t            batch_size, \
       const dim_t*           m, \
       const dim_t*           n, \
       const dim_t*           k, \
       const A_type**         a, \
       const dim_t*           rs_a, \
       const dim_t*           cs_a, \
       const AOCL_MEMORY_TAG* mtag_a, \
       const B_type**         b, \
       const dim_t*           rs_b, \
       const dim_t*           cs_b, \
       const AOCL_MEMORY_TAG* mtag_b, \
       C_type**               c, \
       const dim_t*           rs_c, \
       const dim_t*           cs_c, \
       const C_type*          alpha, \
       const C_type*          beta, \
       rntm_t*                rntm_g, \
       lpgemm_cntx_t*         lcntx, \
       lpgemm_post_op(*post_op_list)[AOCL_MAX_POST_OPS], \
       AOCL_STORAGE_TYPE      c_downscale \
     ); \

GEN_BATCH_LPGEMM_OPENMP_DECORATOR_FN(bfloat16,bfloat16,float,bf16bf16f32of32)
GEN_BATCH_LPGEMM_OPENMP_DECORATOR_FN(float,float,float,f32f32f32of32)
GEN_BATCH_LPGEMM_OPENMP_DECORATOR_FN(uint8_t,int8_t,int32_t,u8s8s32o32)
GEN_BATCH_LPGEMM_OPENMP_DECORATOR_FN(int8_t,int8_t,int32_t,s8s8s32o32)


#define GEN_BATCH_LPGEMM_OPENMP_DECORATOR_FN_MXP(A_type,B_type,C_type,LPGEMM_SFX) \
void batch_lpgemm_ ## LPGEMM_SFX ## _openmp_thread_decorator \
     ( \
       const dim_t            batch_size, \
       const dim_t*           m, \
       const dim_t*           n, \
       const dim_t*           k, \
       const A_type**         a, \
       const dim_t*           rs_a, \
       const dim_t*           cs_a, \
       const AOCL_MEMORY_TAG* mtag_a, \
       const B_type**         b, \
       const dim_t*           rs_b, \
       const dim_t*           cs_b, \
       AOCL_MEMORY_TAG*       mtag_b, \
       C_type**               c, \
       const dim_t*           rs_c, \
       const dim_t*           cs_c, \
       const C_type*          alpha, \
       const C_type*          beta, \
       rntm_t*                rntm_g, \
       lpgemm_cntx_t*         lcntx, \
       lpgemm_pre_op(*pre_op_list)[AOCL_MAX_PRE_OPS], \
       lpgemm_post_op(*post_op_list)[AOCL_MAX_POST_OPS], \
       AOCL_STORAGE_TYPE      c_downscale \
     ); \

GEN_BATCH_LPGEMM_OPENMP_DECORATOR_FN_MXP(bfloat16,int8_t,float,bf16s4f32of32)


#define GEN_LPGEMM_OPENMP_DECORATOR_FN1(A_type,B_type,C_type,LPGEMM_SFX) \
void lpgemm_ ## LPGEMM_SFX ## _openmp_thread_decorator \
     ( \
       const dim_t           m, \
       const dim_t           n, \
       const dim_t           k, \
       const A_type*         a, \
       const dim_t           rs_a, \
       const dim_t           cs_a, \
       const AOCL_MEMORY_TAG mtag_a, \
       const B_type*         b, \
       const dim_t           rs_b, \
       const dim_t           cs_b, \
      AOCL_MEMORY_TAG        mtag_b, \
       C_type*               c, \
       const dim_t           rs_c, \
       const dim_t           cs_c, \
       const C_type          alpha, \
       const C_type          beta, \
       rntm_t*               rntm_g, \
       lpgemm_cntx_t*        lcntx, \
       lpgemm_pre_op*        pre_op_list, \
       lpgemm_post_op*       post_op_list, \
       AOCL_STORAGE_TYPE     c_downscale \
     ); \

GEN_LPGEMM_OPENMP_DECORATOR_FN1(bfloat16, int8_t, float, bf16s4f32of32)

#define GEN_LPGEMM_OPENMP_DECORATOR_FN2(A_type,B_type,C_type,LPGEMM_SFX) \
void lpgemm_ ## LPGEMM_SFX ## _openmp_thread_decorator \
     ( \
       const dim_t           m, \
       const dim_t           n, \
       const dim_t           k, \
       const A_type*         a, \
       const dim_t           rs_a, \
       const dim_t           cs_a, \
       const AOCL_MEMORY_TAG mtag_a, \
       const B_type*         b, \
       const dim_t           rs_b, \
       const dim_t           cs_b, \
      AOCL_MEMORY_TAG        mtag_b, \
       float*               c, \
       const dim_t           rs_c, \
       const dim_t           cs_c, \
       const C_type          alpha, \
       const C_type          beta, \
       rntm_t*               rntm_g, \
       lpgemm_cntx_t*        lcntx, \
       lpgemm_group_post_op* grp_post_op_list, \
       lpgemm_post_op*       post_op_list, \
       AOCL_STORAGE_TYPE     c_downscale \
     ); \

GEN_LPGEMM_OPENMP_DECORATOR_FN2(int8_t, int8_t, int32_t, s8s8s32o32_sym_quant)

#define GEN_UTIL_ELTWISE_OPS_OPENMP_DECORATOR_FN(A_type,B_type,LPGEMM_SFX) \
void lpgemm_eltwise_ops_ ## LPGEMM_SFX ## _openmp_thread_decorator \
     ( \
       const dim_t                 m, \
       const dim_t                 n, \
       const A_type*               a, \
       const dim_t                 rs_a, \
       const dim_t                 cs_a, \
       B_type*                     b, \
       const dim_t                 rs_b, \
       const dim_t                 cs_b, \
       rntm_t*                     rntm_g, \
       lpgemm_eltwise_ops_cntx_t* lcntx, \
       lpgemm_post_op*             post_op_list, \
       AOCL_STORAGE_TYPE           c_downscale \
     ); \

GEN_UTIL_ELTWISE_OPS_OPENMP_DECORATOR_FN(bfloat16,float,bf16of32)
GEN_UTIL_ELTWISE_OPS_OPENMP_DECORATOR_FN(float,float,f32of32)

#else

#define GEN_LPGEMM_DECORATOR_FN(A_type,B_type,C_type,LPGEMM_SFX) \
void lpgemm_ ## LPGEMM_SFX ## _thread_decorator \
     ( \
       const dim_t           m, \
       const dim_t           n, \
       const dim_t           k, \
       const A_type*         a, \
       const dim_t           rs_a, \
       const dim_t           cs_a, \
       const AOCL_MEMORY_TAG mtag_a, \
       const B_type*         b, \
       const dim_t           rs_b, \
       const dim_t           cs_b, \
       const AOCL_MEMORY_TAG mtag_b, \
       C_type*               c, \
       const dim_t           rs_c, \
       const dim_t           cs_c, \
       const C_type          alpha, \
       const C_type          beta, \
       rntm_t*               rntm_g, \
       lpgemm_cntx_t*        lcntx, \
       lpgemm_post_op*       post_op_list, \
       AOCL_STORAGE_TYPE     c_downscale \
     ); \

GEN_LPGEMM_DECORATOR_FN(uint8_t,int8_t,int32_t,u8s8s32o32)
GEN_LPGEMM_DECORATOR_FN(bfloat16,bfloat16,float,bf16bf16f32of32)
GEN_LPGEMM_DECORATOR_FN(float,float,float,f32f32f32of32)
GEN_LPGEMM_DECORATOR_FN(int8_t,int8_t,int32_t,s8s8s32o32)


#define GEN_BATCH_LPGEMM_DECORATOR_FN(A_type,B_type,C_type,LPGEMM_SFX) \
void batch_lpgemm_ ## LPGEMM_SFX ## _thread_decorator \
     ( \
       const dim_t            bs, \
       const dim_t*           m, \
       const dim_t*           n, \
       const dim_t*           k, \
       const A_type**         a, \
       const dim_t*           rs_a, \
       const dim_t*           cs_a, \
       const AOCL_MEMORY_TAG* mtag_a, \
       const B_type**         b, \
       const dim_t*           rs_b, \
       const dim_t*           cs_b, \
       const AOCL_MEMORY_TAG* mtag_b, \
       C_type**               c, \
       const dim_t*           rs_c, \
       const dim_t*           cs_c, \
       const C_type*          alpha, \
       const C_type*          beta, \
       rntm_t*                rntm_g, \
       lpgemm_cntx_t*         lcntx, \
       lpgemm_post_op(*post_op_list)[AOCL_MAX_POST_OPS], \
       AOCL_STORAGE_TYPE     c_downscale \
     ); \

GEN_BATCH_LPGEMM_DECORATOR_FN(bfloat16,bfloat16,float,bf16bf16f32of32)
GEN_BATCH_LPGEMM_DECORATOR_FN(float,float,float,f32f32f32of32)
GEN_BATCH_LPGEMM_DECORATOR_FN(uint8_t,int8_t,int32_t,u8s8s32o32)
GEN_BATCH_LPGEMM_DECORATOR_FN(int8_t,int8_t,int32_t,s8s8s32o32)

#define GEN_BATCH_LPGEMM_DECORATOR_FN_MP(A_type,B_type,C_type,LPGEMM_SFX) \
void batch_lpgemm_ ## LPGEMM_SFX ## _thread_decorator \
     ( \
       const dim_t            bs, \
       const dim_t*           m, \
       const dim_t*           n, \
       const dim_t*           k, \
       const A_type**         a, \
       const dim_t*           rs_a, \
       const dim_t*           cs_a, \
       const AOCL_MEMORY_TAG* mtag_a, \
       const B_type**         b, \
       const dim_t*           rs_b, \
       const dim_t*           cs_b, \
       const AOCL_MEMORY_TAG* mtag_b, \
       C_type**               c, \
       const dim_t*           rs_c, \
       const dim_t*           cs_c, \
       const C_type*          alpha, \
       const C_type*          beta, \
       rntm_t*                rntm_g, \
       lpgemm_cntx_t*         lcntx, \
       lpgemm_pre_op(*pre_op_list)[AOCL_MAX_PRE_OPS], \
       lpgemm_post_op(*post_op_list)[AOCL_MAX_POST_OPS], \
       AOCL_STORAGE_TYPE     c_downscale \
     ); \

GEN_BATCH_LPGEMM_DECORATOR_FN_MP(bfloat16,int8_t,float,bf16s4f32of32)


#define GEN_LPGEMM_DECORATOR_FN1(A_type,B_type,C_type,LPGEMM_SFX) \
void lpgemm_ ## LPGEMM_SFX ## _thread_decorator \
     ( \
       const dim_t           m, \
       const dim_t           n, \
       const dim_t           k, \
       const A_type*         a, \
       const dim_t           rs_a, \
       const dim_t           cs_a, \
       const AOCL_MEMORY_TAG mtag_a, \
       const B_type*         b, \
       const dim_t           rs_b, \
       const dim_t           cs_b, \
       const AOCL_MEMORY_TAG mtag_b, \
       C_type*               c, \
       const dim_t           rs_c, \
       const dim_t           cs_c, \
       const C_type          alpha, \
       const C_type          beta, \
       rntm_t*               rntm_g, \
       lpgemm_cntx_t*        lcntx, \
       lpgemm_pre_op*        pre_op_list, \
       lpgemm_post_op*       post_op_list, \
       AOCL_STORAGE_TYPE     c_downscale \
     ); \

GEN_LPGEMM_DECORATOR_FN1(bfloat16, int8_t, float, bf16s4f32of32)

#define GEN_LPGEMM_OPENMP_DECORATOR_FN2(A_type,B_type,C_type,LPGEMM_SFX) \
void lpgemm_ ## LPGEMM_SFX ## _thread_decorator \
     ( \
       const dim_t           m, \
       const dim_t           n, \
       const dim_t           k, \
       const A_type*         a, \
       const dim_t           rs_a, \
       const dim_t           cs_a, \
       const AOCL_MEMORY_TAG mtag_a, \
       const B_type*         b, \
       const dim_t           rs_b, \
       const dim_t           cs_b, \
      AOCL_MEMORY_TAG        mtag_b, \
       float*               c, \
       const dim_t           rs_c, \
       const dim_t           cs_c, \
       const C_type          alpha, \
       const C_type          beta, \
       rntm_t*               rntm_g, \
       lpgemm_cntx_t*        lcntx, \
       lpgemm_group_post_op* grp_post_op_list, \
       lpgemm_post_op*       post_op_list, \
       AOCL_STORAGE_TYPE     c_downscale \
     ); \

GEN_LPGEMM_OPENMP_DECORATOR_FN2(int8_t, int8_t, int32_t, s8s8s32o32_sym_quant)

#define GEN_UTIL_ELTWISE_OPS_DECORATOR_FN(A_type,B_type,LPGEMM_SFX) \
void lpgemm_eltwise_ops_ ## LPGEMM_SFX ## _thread_decorator \
     ( \
       const dim_t                 m, \
       const dim_t                 n, \
       const A_type*               a, \
       const dim_t                 rs_a, \
       const dim_t                 cs_a, \
       B_type*                     b, \
       const dim_t                 rs_b, \
       const dim_t                 cs_b, \
       rntm_t*                     rntm_g, \
       lpgemm_eltwise_ops_cntx_t* lcntx, \
       lpgemm_post_op*             post_op_list, \
       AOCL_STORAGE_TYPE           c_downscale \
     ); \

GEN_UTIL_ELTWISE_OPS_DECORATOR_FN(bfloat16,float,bf16of32)
GEN_UTIL_ELTWISE_OPS_DECORATOR_FN(float,float,f32of32)

#endif

#endif //LPGEMM_THREAD_DECOR_OPENMP_H
