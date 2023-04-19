/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022-2023, Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef LPGEMM_5LOOP_INTF_H
#define LPGEMM_5LOOP_INTF_H

#include "lpgemm_types.h"
#include "lpgemm_post_ops.h"
#include "aocl_bf16_type.h"

#define LPGEMM_5LOOP(A_type,B_type,C_type,LP_SFX) \
void lpgemm_rowvar_ ## LP_SFX \
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
       rntm_t*               rntm, \
       lpgemm_thrinfo_t*     thread, \
       lpgemm_cntx_t*        lcntx, \
       lpgemm_post_op*       post_op_list, \
       bool                  c_downscale \
     ) \

LPGEMM_5LOOP(uint8_t,int8_t,int32_t,u8s8s32o32);
LPGEMM_5LOOP(uint8_t,int8_t,int16_t,u8s8s16o16);
LPGEMM_5LOOP(float,float,float,f32f32f32of32);
LPGEMM_5LOOP(bfloat16,bfloat16,float,bf16bf16f32of32);
LPGEMM_5LOOP(int8_t,int8_t,int32_t,s8s8s32o32);
LPGEMM_5LOOP(int8_t,int8_t,int16_t,s8s8s16o16);
#endif // LPGEMM_5LOOP_INTF_H
