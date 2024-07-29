/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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
#ifndef AOCL_LPGEMM_GELU_DEF_AVX512_H
#define AOCL_LPGEMM_GELU_DEF_AVX512_H

/* TANH GeLU (x) = 0.5* x * (1 + tanh ( 0.797884 * ( x + ( 0.044715 * x^3 ) ) ) )  */
#define GELU_TANH_F32_AVX512_DEF(reg, r, r2, x, z, dn, x_tanh, q) \
\
	r2 = _mm512_mul_ps (reg, reg); \
	r2 = _mm512_mul_ps (r2, reg); \
	x_tanh = _mm512_fmadd_ps (_mm512_set1_ps (0.044715), r2, reg); \
	x_tanh = _mm512_mul_ps (x_tanh, _mm512_set1_ps (0.797884)); \
\
	/*x_tanh = tanhf(x_tanh) */  \
	TANHF_AVX512(x_tanh, r, r2, x, z, dn, q); \
\
	x_tanh = _mm512_add_ps (x_tanh, _mm512_set1_ps (1)); \
	x_tanh = _mm512_mul_ps (x_tanh, reg); \
	reg = _mm512_mul_ps (x_tanh, _mm512_set1_ps (0.5));


/* ERF GeLU (x) = 0.5* x * (1 + erf (x * 0.707107 ))  */
#define GELU_ERF_F32_AVX512_DEF(reg, r, x, x_erf) \
\
  x_erf = _mm512_mul_ps (reg, _mm512_set1_ps (0.707107)); \
\
  /*x_erf = erf(x_erf) */  \
  ERF_AVX512(x_erf, r, x); \
\
  x_erf = _mm512_add_ps (x_erf, _mm512_set1_ps (1)); \
  x_erf = _mm512_mul_ps (x_erf, reg); \
  reg = _mm512_mul_ps (x_erf, _mm512_set1_ps (0.5));

#endif // AOCL_LPGEMM_GELU_DEF_AVX512_H
