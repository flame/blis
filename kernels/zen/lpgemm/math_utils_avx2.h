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
#ifndef AOCL_LPGEMM_MATH_UTILS_AVX2_H
#define AOCL_LPGEMM_MATH_UTILS_AVX2_H

#define c1 0x1.0000014439a91p0
#define c2 0x1.62e43170e3344p-1
#define c3 0x1.ebf906bc4c115p-3
#define c4 0x1.c6ae2bb88c0c8p-5
#define c5 0x1.3d1079db4ef69p-7
#define c6 0x1.5f8905cb0cc4ep-10

#define TBL_LN2 0x1.71547652b82fep+0
#define EXPF_HUGE 0x1.8p+23
#define EXPF_MIN -88.7228393f
#define EXPF_MAX 88.7228393f
#define inf 1.0/0.0
#define sign -2147483648

//Trignometric EXP and TANH functions for AVX2

#define POLY_EVAL_6_AVX2(r, r2, z) \
    r2 = _mm256_mul_ps (r, r); \
    z = _mm256_fmadd_ps (r2, _mm256_fmadd_ps (r, _mm256_set1_ps(c4), _mm256_set1_ps(c3)), _mm256_fmadd_ps (r, _mm256_set1_ps(c2), _mm256_set1_ps(c1))); \
    r2 = _mm256_mul_ps (r2, r2); \
    r = _mm256_fmadd_ps (r2, _mm256_fmadd_ps (r, _mm256_set1_ps(c6), _mm256_set1_ps(c5)), z); \

#define EXPF_AVX2(x, r, r2, z, dn, q) \
    z = _mm256_mul_ps (x, _mm256_set1_ps(TBL_LN2));	\
	  dn = _mm256_add_ps (z , _mm256_set1_ps(EXPF_HUGE));  \
    r = _mm256_sub_ps (z , _mm256_sub_ps (dn , _mm256_set1_ps(EXPF_HUGE)));  \
\
    POLY_EVAL_6_AVX2 (r, r2, z); \
\
    q = _mm256_add_epi32((__m256i) (r), _mm256_sllv_epi32 ((__m256i)dn, _mm256_set1_epi32 (23)) ); \
    q =  (__m256i)_mm256_blendv_ps ((__m256)q, _mm256_set1_ps(inf), _mm256_cmp_ps (_mm256_set1_ps(88.0), x, 1)); \
    q =  (__m256i)_mm256_blendv_ps ((__m256)q, _mm256_set1_ps(0.0), _mm256_cmp_ps (x, _mm256_set1_ps(-88.0), 1));

#define TANHF_AVX2(x_tanh, r, r2, x, z, dn, q) \
    x = _mm256_mul_ps (_mm256_andnot_ps(_mm256_set1_ps(-0.0f), x_tanh), _mm256_set1_ps(-2) ); \
\
    EXPF_AVX2(x, r, r2, z, dn, q); \
\
    z =  _mm256_add_ps ((__m256)q, _mm256_set1_ps(-1)); \
    z = _mm256_div_ps (z, _mm256_add_ps (z, _mm256_set1_ps(2))); \
    z = _mm256_mul_ps (z, _mm256_set1_ps(-1)); \
    x_tanh = (_mm256_xor_ps (_mm256_and_ps (x_tanh, (__m256)(_mm256_set1_epi32(sign))), z)) ;

#endif // AOCL_LPGEMM_MATH_UTILS_AVX2_H
