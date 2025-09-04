/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef BLIS_GEMM_INT8_PACKB_S8
#define BLIS_GEMM_INT8_PACKB_S8

BLIS_INLINE dim_t get_packb_s8s8s32o32_min_NR()
{
	// This is the minimum NR' required for use in u8s8s32 kernels. The idea
	// here is that since k needs to be a multiple of 4 (VNNI instr), NR'=16
	// results in total of 4 * NR' = 64 bytes to be loaded, which fits in 1 ZMM
	// register. Thus the smallest n fringe kernel dimension has n=16, and thus
	// any rounding for buffer sizes should be to 16.
	return 16;
}

typedef void (*packb_s32_s8)
     (
       int8_t*,
       int32_t*,
       const int8_t*,
       const dim_t,
       const dim_t,
       const dim_t,
       const dim_t,
       dim_t*,
       dim_t*
     );

void packb_nr64_s8s8s32os32
     (
       int8_t*       pack_b_buffer_s8s8s32o32,
       int32_t*      pack_b_column_sum,
       const int8_t* b,
       const dim_t   rs_b,
       const dim_t   cs_b,
       const dim_t   NC,
       const dim_t   KC,
       dim_t*        rs_p,
       dim_t*        cs_p
     );

#endif //BLIS_GEMM_INT8_PACKB_S8

