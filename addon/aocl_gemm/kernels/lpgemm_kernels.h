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

#ifndef BLIS_LPGEMM_KERN_H
#define BLIS_LPGEMM_KERN_H

#include "lpgemm_post_ops.h"
#include "aocl_bf16_type.h"

// Disable BF16 kernel in cases where compilers support other avx 512
// features except BF16 ISA.
#if ( defined( BLIS_GCC ) && ( ( __GNUC__ < 11 ) || \
	  ( ( __GNUC__ == 11 ) && ( __GNUC_MINOR__ < 2 ) ) ) && defined(BLIS_KERNELS_ZEN4) )
#define LPGEMM_BF16_JIT
#define BPREFETCH_JIT
//#define DUMP_JIT_CODE
#endif

typedef void (*lpgemm_m_fringe_f32_ker_ft)
    (
       const dim_t         k0,
       const float*        a,
       const dim_t         rs_a,
       const dim_t         cs_a,
       const float*        b,
       const dim_t         rs_b,
       const dim_t         cs_b,
       float*              c,
       const dim_t         rs_c,
       const float         alpha,
       const float         beta,
       lpgemm_post_op*     post_ops_list,
       lpgemm_post_op_attr post_ops_attr
    );

#define LPGEMM_MAIN_KERN(A_type,B_type,C_type,LP_SFX) \
void lpgemm_rowvar_ ## LP_SFX \
     ( \
       const dim_t         m0, \
       const dim_t         n0, \
       const dim_t         k0, \
       const A_type*       a, \
       const dim_t         rs_a, \
       const dim_t         cs_a, \
       const dim_t         ps_a, \
       const B_type*       b, \
       const dim_t         rs_b, \
       const dim_t         cs_b, \
       C_type*             c, \
       const dim_t         rs_c, \
       const dim_t         cs_c, \
       const C_type        alpha, \
       const C_type        beta, \
       lpgemm_post_op*     post_ops_list, \
       lpgemm_post_op_attr post_ops_attr \
     ) \

LPGEMM_MAIN_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_6x64);
LPGEMM_MAIN_KERN(bfloat16,bfloat16,float,bf16bf16f32of32_6x64);
LPGEMM_MAIN_KERN(float,float,float,f32f32f32of32_6x16m);
LPGEMM_MAIN_KERN(float,float,float,f32f32f32of32_avx512_6x64m);
LPGEMM_MAIN_KERN(int8_t,int8_t,int32_t,s8s8s32os32_6x64);


#define LPGEMM_MAIN_KERN1(A_type,B_type,C_type,LP_SFX) \
void lpgemm_rowvar_ ## LP_SFX \
     ( \
       const dim_t         m0, \
       const dim_t         n0, \
       const dim_t         k0, \
       const A_type*       a, \
       const dim_t         rs_a, \
       const dim_t         cs_a, \
       const dim_t         ps_a, \
       const B_type*       b, \
       const dim_t         rs_b, \
       const dim_t         cs_b, \
       C_type*             c, \
       const dim_t         rs_c, \
       const dim_t         cs_c, \
       const C_type        alpha, \
       const C_type        beta, \
       lpgemm_post_op*     post_ops_list, \
       lpgemm_post_op_attr post_ops_attr, \
       lpgemm_pre_op_attr  pre_ops_attr \
     ) \

LPGEMM_MAIN_KERN1(bfloat16,int8_t,float,bf16s4f32of32_6x64m);

#define LPGEMM_M_FRINGE_KERN(A_type,B_type,C_type,LP_SFX) \
void lpgemm_rowvar_ ## LP_SFX \
     ( \
       const dim_t         k0, \
       const A_type*       a, \
       const dim_t         rs_a, \
       const dim_t         cs_a, \
       const B_type*       b, \
       const dim_t         rs_b, \
       const dim_t         cs_b, \
       C_type*             c, \
       const dim_t         rs_c, \
       const C_type        alpha, \
       const C_type        beta, \
       lpgemm_post_op*     post_ops_list, \
       lpgemm_post_op_attr post_ops_attr \
     ) \

LPGEMM_M_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_5x64);
LPGEMM_M_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_4x64);
LPGEMM_M_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_3x64);
LPGEMM_M_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_2x64);
LPGEMM_M_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_1x64);

LPGEMM_M_FRINGE_KERN(bfloat16,bfloat16,float,bf16bf16f32of32_5x64);
LPGEMM_M_FRINGE_KERN(bfloat16,bfloat16,float,bf16bf16f32of32_4x64);
LPGEMM_M_FRINGE_KERN(bfloat16,bfloat16,float,bf16bf16f32of32_3x64);
LPGEMM_M_FRINGE_KERN(bfloat16,bfloat16,float,bf16bf16f32of32_2x64);
LPGEMM_M_FRINGE_KERN(bfloat16,bfloat16,float,bf16bf16f32of32_1x64);

LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_5x64);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_4x64);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_3x64);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_2x64);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_1x64);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_5x48);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_4x48);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_3x48);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_2x48);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_1x48);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_5x32);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_4x32);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_3x32);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_2x32);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_1x32);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_5x16);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_4x16);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_3x16);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_2x16);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_1x16);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_5x8);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_4x8);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_3x8);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_2x8);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_1x8);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_5x4);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_4x4);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_3x4);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_2x4);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_1x4);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_5x2);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_4x2);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_3x2);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_2x2);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_1x2);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_5x1);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_4x1);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_3x1);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_2x1);
LPGEMM_M_FRINGE_KERN(float,float,float,f32f32f32of32_1x1);

LPGEMM_M_FRINGE_KERN(int8_t,int8_t,int32_t,s8s8s32os32_5x64);
LPGEMM_M_FRINGE_KERN(int8_t,int8_t,int32_t,s8s8s32os32_4x64);
LPGEMM_M_FRINGE_KERN(int8_t,int8_t,int32_t,s8s8s32os32_3x64);
LPGEMM_M_FRINGE_KERN(int8_t,int8_t,int32_t,s8s8s32os32_2x64);
LPGEMM_M_FRINGE_KERN(int8_t,int8_t,int32_t,s8s8s32os32_1x64);


#define LPGEMM_M_FRINGE_KERN1(A_type,B_type,C_type,LP_SFX) \
void lpgemm_rowvar_ ## LP_SFX \
     ( \
       const dim_t         k0, \
       const A_type*       a, \
       const dim_t         rs_a, \
       const dim_t         cs_a, \
       const B_type*       b, \
       const dim_t         rs_b, \
       const dim_t         cs_b, \
       C_type*             c, \
       const dim_t         rs_c, \
       const C_type        alpha, \
       const C_type        beta, \
       lpgemm_post_op*     post_ops_list, \
       lpgemm_post_op_attr post_ops_attr, \
       lpgemm_pre_op_attr pre_ops_attr \
     ) \

LPGEMM_M_FRINGE_KERN1(bfloat16,int8_t,float,bf16s4f32of32_5x64);
LPGEMM_M_FRINGE_KERN1(bfloat16,int8_t,float,bf16s4f32of32_4x64);
LPGEMM_M_FRINGE_KERN1(bfloat16,int8_t,float,bf16s4f32of32_3x64);
LPGEMM_M_FRINGE_KERN1(bfloat16,int8_t,float,bf16s4f32of32_2x64);
LPGEMM_M_FRINGE_KERN1(bfloat16,int8_t,float,bf16s4f32of32_1x64);


#define LPGEMM_N_FRINGE_KERN(A_type,B_type,C_type,LP_SFX) \
void lpgemm_rowvar_ ## LP_SFX \
     ( \
       const dim_t         m0, \
       const dim_t         k0, \
       const A_type*       a, \
       const dim_t         rs_a, \
       const dim_t         cs_a, \
       const dim_t         ps_a, \
       const B_type*       b, \
       const dim_t         rs_b, \
       const dim_t         cs_b, \
       C_type*             c, \
       const dim_t         rs_c, \
       const C_type        alpha, \
       const C_type        beta, \
       lpgemm_post_op*     post_ops_list, \
       lpgemm_post_op_attr post_ops_attr \
     ) \

LPGEMM_N_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_6x16);
LPGEMM_N_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_12x16);
LPGEMM_N_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_6x32);
LPGEMM_N_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_9x32);
LPGEMM_N_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_6x48);

LPGEMM_N_FRINGE_KERN(bfloat16,bfloat16,float,bf16bf16f32of32_6x16);
LPGEMM_N_FRINGE_KERN(bfloat16,bfloat16,float,bf16bf16f32of32_6x32);
LPGEMM_N_FRINGE_KERN(bfloat16,bfloat16,float,bf16bf16f32of32_6x48);

LPGEMM_N_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_6x48m);
LPGEMM_N_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_6x32m);
LPGEMM_N_FRINGE_KERN(float,float,float,f32f32f32of32_6x8m);
LPGEMM_N_FRINGE_KERN(float,float,float,f32f32f32of32_6x4m);
LPGEMM_N_FRINGE_KERN(float,float,float,f32f32f32of32_6x2m);
LPGEMM_N_FRINGE_KERN(float,float,float,f32f32f32of32_6x1m);

LPGEMM_N_FRINGE_KERN(int8_t,int8_t,int32_t,s8s8s32os32_6x16);
LPGEMM_N_FRINGE_KERN(int8_t,int8_t,int32_t,s8s8s32os32_6x32);
LPGEMM_N_FRINGE_KERN(int8_t,int8_t,int32_t,s8s8s32os32_6x48);


#define LPGEMM_N_FRINGE_KERN1(A_type,B_type,C_type,LP_SFX) \
void lpgemm_rowvar_ ## LP_SFX \
     ( \
       const dim_t         m0, \
       const dim_t         k0, \
       const A_type*       a, \
       const dim_t         rs_a, \
       const dim_t         cs_a, \
       const dim_t         ps_a, \
       const B_type*       b, \
       const dim_t         rs_b, \
       const dim_t         cs_b, \
       C_type*             c, \
       const dim_t         rs_c, \
       const C_type        alpha, \
       const C_type        beta, \
       lpgemm_post_op*     post_ops_list, \
       lpgemm_post_op_attr post_ops_attr, \
       lpgemm_pre_op_attr  pre_ops_attr \
     ) \

LPGEMM_N_FRINGE_KERN1(bfloat16,int8_t,float,bf16s4f32of32_6x16m);
LPGEMM_N_FRINGE_KERN1(bfloat16,int8_t,float,bf16s4f32of32_6x32m);
LPGEMM_N_FRINGE_KERN1(bfloat16,int8_t,float,bf16s4f32of32_6x48m);



#define LPGEMM_N_LT_NR0_FRINGE_KERN(A_type,B_type,C_type,LP_SFX) \
void lpgemm_rowvar_ ## LP_SFX \
     ( \
       const dim_t         m0, \
       const dim_t         k0, \
       const A_type*       a, \
       const dim_t         rs_a, \
       const dim_t         cs_a, \
       const dim_t         ps_a, \
       const B_type*       b, \
       const dim_t         rs_b, \
       const dim_t         cs_b, \
       C_type*             c, \
       const dim_t         rs_c, \
       const C_type        alpha, \
       const C_type        beta, \
       const dim_t         n0_rem, \
       lpgemm_post_op*     post_ops_list, \
       lpgemm_post_op_attr post_ops_attr \
     ) \

LPGEMM_N_LT_NR0_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_6xlt16);
LPGEMM_N_LT_NR0_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_12xlt16);

LPGEMM_N_LT_NR0_FRINGE_KERN(bfloat16,bfloat16,float,bf16bf16f32of32_6xlt16);

LPGEMM_N_LT_NR0_FRINGE_KERN(int8_t,int8_t,int32_t,s8s8s32os32_6xlt16);

#define LPGEMM_N_LT_NR0_FRINGE_KERN1(A_type,B_type,C_type,LP_SFX) \
void lpgemm_rowvar_ ## LP_SFX \
     ( \
       const dim_t         m0, \
       const dim_t         k0, \
       const A_type*       a, \
       const dim_t         rs_a, \
       const dim_t         cs_a, \
       const dim_t         ps_a, \
       const B_type*       b, \
       const dim_t         rs_b, \
       const dim_t         cs_b, \
       C_type*             c, \
       const dim_t         rs_c, \
       const C_type        alpha, \
       const C_type        beta, \
       const dim_t         n0_rem, \
       lpgemm_post_op*     post_ops_list, \
       lpgemm_post_op_attr post_ops_attr, \
       lpgemm_pre_op_attr  pre_ops_attr \
     ) \

LPGEMM_N_LT_NR0_FRINGE_KERN1(bfloat16,int8_t,float,bf16s4f32of32_6xlt16m);


#define LPGEMM_MN_FRINGE_KERN(A_type,B_type,C_type,LP_SFX) \
void lpgemm_rowvar_ ## LP_SFX \
     ( \
       const dim_t         k0, \
       const A_type*       a, \
       const dim_t         rs_a, \
       const dim_t         cs_a, \
       const B_type*       b, \
       const dim_t         rs_b, \
       const dim_t         cs_b, \
       C_type*             c, \
       const dim_t         rs_c, \
       const C_type        alpha, \
       const C_type        beta, \
       lpgemm_post_op*     post_ops_list, \
       lpgemm_post_op_attr post_ops_attr \
     ) \

LPGEMM_MN_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_5x16);
LPGEMM_MN_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_4x16);
LPGEMM_MN_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_3x16);
LPGEMM_MN_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_2x16);
LPGEMM_MN_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_1x16);
LPGEMM_MN_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_5x32);
LPGEMM_MN_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_4x32);
LPGEMM_MN_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_3x32);
LPGEMM_MN_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_2x32);
LPGEMM_MN_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_1x32);
LPGEMM_MN_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_5x48);
LPGEMM_MN_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_4x48);
LPGEMM_MN_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_3x48);
LPGEMM_MN_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_2x48);
LPGEMM_MN_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_1x48);

LPGEMM_MN_FRINGE_KERN(bfloat16,bfloat16,float,bf16bf16f32of32_5x16);
LPGEMM_MN_FRINGE_KERN(bfloat16,bfloat16,float,bf16bf16f32of32_4x16);
LPGEMM_MN_FRINGE_KERN(bfloat16,bfloat16,float,bf16bf16f32of32_3x16);
LPGEMM_MN_FRINGE_KERN(bfloat16,bfloat16,float,bf16bf16f32of32_2x16);
LPGEMM_MN_FRINGE_KERN(bfloat16,bfloat16,float,bf16bf16f32of32_1x16);
LPGEMM_MN_FRINGE_KERN(bfloat16,bfloat16,float,bf16bf16f32of32_5x32);
LPGEMM_MN_FRINGE_KERN(bfloat16,bfloat16,float,bf16bf16f32of32_4x32);
LPGEMM_MN_FRINGE_KERN(bfloat16,bfloat16,float,bf16bf16f32of32_3x32);
LPGEMM_MN_FRINGE_KERN(bfloat16,bfloat16,float,bf16bf16f32of32_2x32);
LPGEMM_MN_FRINGE_KERN(bfloat16,bfloat16,float,bf16bf16f32of32_1x32);
LPGEMM_MN_FRINGE_KERN(bfloat16,bfloat16,float,bf16bf16f32of32_5x48);
LPGEMM_MN_FRINGE_KERN(bfloat16,bfloat16,float,bf16bf16f32of32_4x48);
LPGEMM_MN_FRINGE_KERN(bfloat16,bfloat16,float,bf16bf16f32of32_3x48);
LPGEMM_MN_FRINGE_KERN(bfloat16,bfloat16,float,bf16bf16f32of32_2x48);
LPGEMM_MN_FRINGE_KERN(bfloat16,bfloat16,float,bf16bf16f32of32_1x48);

LPGEMM_MN_FRINGE_KERN(int8_t,int8_t,int32_t,s8s8s32os32_5x16);
LPGEMM_MN_FRINGE_KERN(int8_t,int8_t,int32_t,s8s8s32os32_4x16);
LPGEMM_MN_FRINGE_KERN(int8_t,int8_t,int32_t,s8s8s32os32_3x16);
LPGEMM_MN_FRINGE_KERN(int8_t,int8_t,int32_t,s8s8s32os32_2x16);
LPGEMM_MN_FRINGE_KERN(int8_t,int8_t,int32_t,s8s8s32os32_1x16);
LPGEMM_MN_FRINGE_KERN(int8_t,int8_t,int32_t,s8s8s32os32_5x32);
LPGEMM_MN_FRINGE_KERN(int8_t,int8_t,int32_t,s8s8s32os32_4x32);
LPGEMM_MN_FRINGE_KERN(int8_t,int8_t,int32_t,s8s8s32os32_3x32);
LPGEMM_MN_FRINGE_KERN(int8_t,int8_t,int32_t,s8s8s32os32_2x32);
LPGEMM_MN_FRINGE_KERN(int8_t,int8_t,int32_t,s8s8s32os32_1x32);
LPGEMM_MN_FRINGE_KERN(int8_t,int8_t,int32_t,s8s8s32os32_5x48);
LPGEMM_MN_FRINGE_KERN(int8_t,int8_t,int32_t,s8s8s32os32_4x48);
LPGEMM_MN_FRINGE_KERN(int8_t,int8_t,int32_t,s8s8s32os32_3x48);
LPGEMM_MN_FRINGE_KERN(int8_t,int8_t,int32_t,s8s8s32os32_2x48);
LPGEMM_MN_FRINGE_KERN(int8_t,int8_t,int32_t,s8s8s32os32_1x48);

#define LPGEMM_MN_FRINGE_KERN1(A_type,B_type,C_type,LP_SFX) \
void lpgemm_rowvar_ ## LP_SFX \
     ( \
       const dim_t         k0, \
       const A_type*       a, \
       const dim_t         rs_a, \
       const dim_t         cs_a, \
       const B_type*       b, \
       const dim_t         rs_b, \
       const dim_t         cs_b, \
       C_type*             c, \
       const dim_t         rs_c, \
       const C_type        alpha, \
       const C_type        beta, \
       lpgemm_post_op*     post_ops_list, \
       lpgemm_post_op_attr post_ops_attr, \
       lpgemm_pre_op_attr  pre_ops_attr \
     ) \

LPGEMM_MN_FRINGE_KERN1(bfloat16,int8_t,float,bf16s4f32of32_5x16);
LPGEMM_MN_FRINGE_KERN1(bfloat16,int8_t,float,bf16s4f32of32_4x16);
LPGEMM_MN_FRINGE_KERN1(bfloat16,int8_t,float,bf16s4f32of32_3x16);
LPGEMM_MN_FRINGE_KERN1(bfloat16,int8_t,float,bf16s4f32of32_2x16);
LPGEMM_MN_FRINGE_KERN1(bfloat16,int8_t,float,bf16s4f32of32_1x16);
LPGEMM_MN_FRINGE_KERN1(bfloat16,int8_t,float,bf16s4f32of32_5x32);
LPGEMM_MN_FRINGE_KERN1(bfloat16,int8_t,float,bf16s4f32of32_4x32);
LPGEMM_MN_FRINGE_KERN1(bfloat16,int8_t,float,bf16s4f32of32_3x32);
LPGEMM_MN_FRINGE_KERN1(bfloat16,int8_t,float,bf16s4f32of32_2x32);
LPGEMM_MN_FRINGE_KERN1(bfloat16,int8_t,float,bf16s4f32of32_1x32);
LPGEMM_MN_FRINGE_KERN1(bfloat16,int8_t,float,bf16s4f32of32_5x48);
LPGEMM_MN_FRINGE_KERN1(bfloat16,int8_t,float,bf16s4f32of32_4x48);
LPGEMM_MN_FRINGE_KERN1(bfloat16,int8_t,float,bf16s4f32of32_3x48);
LPGEMM_MN_FRINGE_KERN1(bfloat16,int8_t,float,bf16s4f32of32_2x48);
LPGEMM_MN_FRINGE_KERN1(bfloat16,int8_t,float,bf16s4f32of32_1x48);

#define LPGEMM_MN_LT_NR0_FRINGE_KERN(A_type,B_type,C_type,LP_SFX) \
void lpgemm_rowvar_ ## LP_SFX \
     ( \
       const dim_t         k0, \
       const A_type*       a, \
       const dim_t         rs_a, \
       const dim_t         cs_a, \
       const B_type*       b, \
       const dim_t         rs_b, \
       const dim_t         cs_b, \
       C_type*             c, \
       const dim_t         rs_c, \
       const C_type        alpha, \
       const C_type        beta, \
       const dim_t         n0_rem, \
       lpgemm_post_op*     post_ops_list, \
       lpgemm_post_op_attr post_ops_attr \
     ) \

LPGEMM_MN_LT_NR0_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_5xlt16);
LPGEMM_MN_LT_NR0_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_4xlt16);
LPGEMM_MN_LT_NR0_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_3xlt16);
LPGEMM_MN_LT_NR0_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_2xlt16);
LPGEMM_MN_LT_NR0_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_1xlt16);

LPGEMM_MN_LT_NR0_FRINGE_KERN(bfloat16,bfloat16,float,bf16bf16f32of32_5xlt16);
LPGEMM_MN_LT_NR0_FRINGE_KERN(bfloat16,bfloat16,float,bf16bf16f32of32_4xlt16);
LPGEMM_MN_LT_NR0_FRINGE_KERN(bfloat16,bfloat16,float,bf16bf16f32of32_3xlt16);
LPGEMM_MN_LT_NR0_FRINGE_KERN(bfloat16,bfloat16,float,bf16bf16f32of32_2xlt16);
LPGEMM_MN_LT_NR0_FRINGE_KERN(bfloat16,bfloat16,float,bf16bf16f32of32_1xlt16);

LPGEMM_MN_LT_NR0_FRINGE_KERN(int8_t,int8_t,int32_t,s8s8s32os32_5xlt16);
LPGEMM_MN_LT_NR0_FRINGE_KERN(int8_t,int8_t,int32_t,s8s8s32os32_4xlt16);
LPGEMM_MN_LT_NR0_FRINGE_KERN(int8_t,int8_t,int32_t,s8s8s32os32_3xlt16);
LPGEMM_MN_LT_NR0_FRINGE_KERN(int8_t,int8_t,int32_t,s8s8s32os32_2xlt16);
LPGEMM_MN_LT_NR0_FRINGE_KERN(int8_t,int8_t,int32_t,s8s8s32os32_1xlt16);

#define LPGEMM_MN_LT_NR0_FRINGE_KERN1(A_type,B_type,C_type,LP_SFX) \
void lpgemm_rowvar_ ## LP_SFX \
     ( \
       const dim_t         k0, \
       const A_type*       a, \
       const dim_t         rs_a, \
       const dim_t         cs_a, \
       const B_type*       b, \
       const dim_t         rs_b, \
       const dim_t         cs_b, \
       C_type*             c, \
       const dim_t         rs_c, \
       const C_type        alpha, \
       const C_type        beta, \
       const dim_t         n0_rem, \
       lpgemm_post_op*     post_ops_list, \
       lpgemm_post_op_attr post_ops_attr, \
       lpgemm_pre_op_attr  pre_ops_attr \
     ) \

LPGEMM_MN_LT_NR0_FRINGE_KERN1(bfloat16,int8_t,float,bf16s4f32of32_5xlt16);
LPGEMM_MN_LT_NR0_FRINGE_KERN1(bfloat16,int8_t,float,bf16s4f32of32_4xlt16);
LPGEMM_MN_LT_NR0_FRINGE_KERN1(bfloat16,int8_t,float,bf16s4f32of32_3xlt16);
LPGEMM_MN_LT_NR0_FRINGE_KERN1(bfloat16,int8_t,float,bf16s4f32of32_2xlt16);
LPGEMM_MN_LT_NR0_FRINGE_KERN1(bfloat16,int8_t,float,bf16s4f32of32_1xlt16);

#define LPGEMV_M_EQ1_KERN(A_type,B_type,C_type,LP_SFX) \
void lpgemv_m_one_ ## LP_SFX \
( \
	const dim_t           n0, \
	const dim_t           k, \
	const A_type          *a, \
	const dim_t           rs_a, \
	const dim_t           cs_a, \
	const AOCL_MEMORY_TAG mtag_a, \
	const B_type          *b, \
	dim_t                 rs_b, \
	const dim_t           cs_b, \
	const AOCL_MEMORY_TAG mtag_b, \
	C_type                *c, \
	const dim_t           rs_c, \
	const dim_t           cs_c, \
	const C_type          alpha, \
	const C_type          beta, \
	dim_t                 NR, \
	const dim_t           KC, \
	const dim_t           n_sub_updated, \
	const dim_t           jc_cur_loop_rem, \
	lpgemm_post_op        *post_op, \
	lpgemm_post_op_attr   *post_op_attr \
  ) \

LPGEMV_M_EQ1_KERN(float, float, float,f32f32f32of32);
LPGEMV_M_EQ1_KERN(bfloat16,bfloat16,float,bf16bf16f32of32);
LPGEMV_M_EQ1_KERN(uint8_t,int8_t,int32_t,u8s8s32os32);
LPGEMV_M_EQ1_KERN(int8_t,int8_t,int32_t,s8s8s32os32);

#define LPGEMV_N_EQ1_KERN(A_type,B_type,C_type,LP_SFX) \
void lpgemv_n_one_ ## LP_SFX \
( \
	const dim_t           m0, \
	const dim_t           k, \
	const A_type          *a, \
	const dim_t           rs_a, \
	const dim_t           cs_a, \
	const AOCL_MEMORY_TAG mtag_a, \
	const B_type          *b, \
	const dim_t           rs_b, \
	const dim_t           cs_b, \
	const AOCL_MEMORY_TAG mtag_b, \
	C_type                *c, \
	const dim_t           rs_c, \
	const dim_t           cs_c, \
	const C_type          alpha, \
	const C_type          beta, \
	const dim_t           MR, \
	const dim_t           KC, \
	lpgemm_post_op        *post_op, \
	lpgemm_post_op_attr   *post_op_attr \
) \

LPGEMV_N_EQ1_KERN(float, float, float,f32f32f32of32);
LPGEMV_N_EQ1_KERN(bfloat16, bfloat16, float,bf16bf16f32of32);
LPGEMV_N_EQ1_KERN(uint8_t,int8_t,int32_t,u8s8s32os32);
LPGEMV_N_EQ1_KERN(int8_t,int8_t,int32_t,s8s8s32os32);

#endif //BLIS_LPGEMM_KERN_H
