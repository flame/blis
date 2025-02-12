/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef BLIS_GEMM_BF16_PACKB
#define BLIS_GEMM_BF16_PACKB

#include "aocl_bf16_type.h"

BLIS_INLINE dim_t get_packb_bf16bf16f32of32_min_NR()
{
	// This is the minimum NR' required for use in bf16bf16f32 kernels. The idea
	// here is that since k needs to be a multiple of 2 (BF16 instr), NR'=16
	// results in total of 2 * NR' = 64 bytes to be loaded, which fits in 1 ZMM
	// register. Thus the smallest n fringe kernel dimension has n=16, and thus
	// any rounding for buffer sizes should be to 16.
	return 16;
}

typedef void (*pack_f32bf16)
  (
       bfloat16*,
       const float*,
       const dim_t,
       const dim_t,
       const dim_t,
       const dim_t,
       dim_t*,
       dim_t*
  );

typedef void (*pack_s4bf16)
  (
    bfloat16 *,
    const int8_t *,
    const dim_t,
    const dim_t,
    dim_t *,
    dim_t *,
    lpgemm_pre_op_attr
  );

typedef void (*pack_bf16)
     (
       bfloat16*,
       const bfloat16*,
       const dim_t,
       const dim_t,
       const dim_t,
       const dim_t,
       dim_t*,
       dim_t*
     );

typedef void (*unpack_bf16)
     (
       const bfloat16*,
       bfloat16*,
       const dim_t,
       const dim_t,
       const dim_t,
       const dim_t
     );

typedef void (*unpack_bf16f32)
     (
       const bfloat16*,
       float*,
       const dim_t,
       const dim_t,
       const dim_t,
       const dim_t
     );


typedef void (*pack_s4)
     (
       int8_t*,
       const int8_t*,
       const dim_t,
       const dim_t,
       const dim_t,
       const dim_t,
       dim_t*,
       dim_t*,
       lpgemm_pre_op*,
       AOCL_MATRIX_TYPE
     );

void packb_mxp_nr64_f32obf16
    (
      bfloat16 *pack_b_buffer_bf16bf16f32of32,
      const float *b,
      const dim_t rs_b,
      const dim_t cs_b,
      const dim_t NC,
      const dim_t KC,
      dim_t *rs_p,
      dim_t *cs_p
    );

void packb_nr64_bf16bf16f32of32
     (
       bfloat16*       pack_b_buffer_bf16bf16f32of32,
       const bfloat16* b,
       const dim_t     rs_b,
       const dim_t     cs_b,
       const dim_t     NC,
       const dim_t     KC,
       dim_t*          rs_p,
       dim_t*          cs_p
     );

void packb_nr64_bf16s4f32of32
     (
       int8_t*       pack_b_buffer,
       const int8_t* b,
       const dim_t   rs_b,
       const dim_t   cs_b,
       const dim_t   NC,
       const dim_t   KC,
       dim_t*        rs_p,
       dim_t*        cs_p,
      lpgemm_pre_op* pre_op,
      AOCL_MATRIX_TYPE mtag
     );

void packsclb_nr64_bf16s4f32of32
    (
      bfloat16* packb_bf16,
      const int8_t* b,
      const dim_t NC,
      const dim_t KC,
      dim_t *rs_p,
      dim_t *cs_p,
      lpgemm_pre_op_attr pre_ops_attr
    );

void packa_mr16_bf16bf16f32of32
     (
       bfloat16* pack_a_buffer,
       const bfloat16* a,
       const dim_t rs_a,
       const dim_t cs_a,
       const dim_t MC,
       const dim_t KC,
       dim_t* rs_p,
       dim_t* cs_p
     );

void unpackb_nr64_bf16bf16f32of32
     (
        const bfloat16* unpack_b_buffer_bf16bf16f32of32,
        bfloat16*       b,
        const dim_t     NC,
        const dim_t     KC,
        dim_t           rs_b,
        dim_t           cs_b
      );

void unpackb_nr64_bf16_f32
    (
      const bfloat16* b,
      float*       unpack_b_buffer,
      const dim_t	  NC,
      const dim_t     KC,
      dim_t           rs_b,
      dim_t           cs_b
    );

void cvt_bf16_f32(
    float*	      cvt_buffer,
    const bfloat16* a,
    const dim_t     rs_a,
    const dim_t     cs_a,
    const dim_t     MC,
    const dim_t     KC,
    const dim_t      rs_p,
    const dim_t      cs_p
  );

#endif //BLIS_GEMM_BF16_PACKB
