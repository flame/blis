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

#ifndef AOCL_LPGEMM_SIGMOID_AVX2_H
#define AOCL_LPGEMM_SIGMOID_AVX2_H

// Sigmoid(in_reg) = 1 / (1 + exp(-1 * in_reg)).
// in_reg is expected to contain float values.
#define SIGMOID_F32_AVX2_DEF(in_reg, al_in, r, r2, z, dn, ex_out) \
  al_in = _mm256_mul_ps ( in_reg, _mm256_set1_ps(-1) ); \
	EXPF_AVX2(al_in, r, r2, z, dn, ex_out); \
    ex_out = ( __m256i )_mm256_add_ps( ( __m256 )ex_out, _mm256_set1_ps( 1 ) ); \
	in_reg = _mm256_div_ps( _mm256_set1_ps ( 1 ), ( __m256 )ex_out ); \

// Sigmoid(in_reg) = 1 / (1 + exp(-1 * in_reg)).
// in_reg is expected to contain float values.
#define SIGMOID_F32_SSE_DEF(in_reg, al_in, r, r2, z, dn, ex_out) \
  al_in = _mm_mul_ps ( in_reg, _mm_set1_ps(-1) ); \
	EXPF_SSE(al_in, r, r2, z, dn, ex_out); \
    ex_out = ( __m128i )_mm_add_ps( ( __m128 )ex_out, _mm_set1_ps( 1 ) ); \
	in_reg = _mm_div_ps( _mm_set1_ps ( 1 ), ( __m128 )ex_out ); \

#endif // AOCL_LPGEMM_SIGMOID_AVX2_H