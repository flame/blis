/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.

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

#include <immintrin.h>

#include "blis.h"
#include "lpgemm_mn_fringe_s16.h"

// 4x32 int8o16 kernel
void lpgemm_rowvar_u8s8s16o16_4x16
	(
		const dim_t k0,
		const uint8_t *a,
		const dim_t rs_a,
		const dim_t cs_a,
		const int8_t *b,
		const dim_t rs_b,
		const dim_t cs_b,
		int16_t *c,
		const dim_t rs_c,
		const int16_t alpha,
		const int16_t beta
	)
{
	dim_t NR = 16;

	// The division is done by considering the vpmaddubsw instruction
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	// B matrix storage.
	__m256i b0;

	// A matrix storage.
	__m256i a_int32_0;
	__m256i inter_vec;

	//  Registers to use for accumulating C.
	__m256i c_int16_0p0 = _mm256_setzero_si256();
	__m256i c_int16_1p0 = _mm256_setzero_si256();
	__m256i c_int16_2p0 = _mm256_setzero_si256();
	__m256i c_int16_3p0 = _mm256_setzero_si256();

	for (dim_t kr = 0; kr < k_full_pieces; kr += 1)
	{
		dim_t offset = kr * 2;

		b0 = _mm256_loadu_si256((__m256i const *)(b + (32 * kr) + (NR * 0)));

		// Broadcast a[0,kr:kr+2].
		a_int32_0 = _mm256_set1_epi16(*(uint16_t *)(a + (rs_a * 0) + (cs_a * offset)));

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_0p0 = _mm256_add_epi16(inter_vec, c_int16_0p0);

		// Broadcast a[1,kr:kr+2].
		a_int32_0 = _mm256_set1_epi16(*(uint16_t *)(a + (rs_a * 1) + (cs_a * offset)));

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_1p0 = _mm256_add_epi16(inter_vec, c_int16_1p0);

		// Broadcast a[2,kr:kr+2].
		a_int32_0 = _mm256_set1_epi16(*(uint16_t *)(a + (rs_a * 2) + (cs_a * offset)));

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[2,0-15] = a[2,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_2p0 = _mm256_add_epi16(inter_vec, c_int16_2p0);

		// Broadcast a[3,kr:kr+2].
		a_int32_0 = _mm256_set1_epi16(*(uint16_t *)(a + (rs_a * 3) + (cs_a * offset)));

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[3,0-31] = a[3,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_3p0 = _mm256_add_epi16(inter_vec, c_int16_3p0);
	}

	// Handle k remainder.
	if (k_partial_pieces > 0)
	{
		uint8_t a_element[4];

		b0 = _mm256_loadu_si256((__m256i const *)(b + (32 * k_full_pieces) + (NR * 0)));

		a_element[0] = *(a + (rs_a * 0) + (cs_a * (k_full_pieces * 2)));
		a_int32_0 = _mm256_set1_epi8(a_element[0]);

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_0p0 = _mm256_add_epi16(inter_vec, c_int16_0p0);

		a_element[1] = *(a + (rs_a * 1) + (cs_a * (k_full_pieces * 2)));
		a_int32_0 = _mm256_set1_epi8(a_element[1]);

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_1p0 = _mm256_add_epi16(inter_vec, c_int16_1p0);

		a_element[2] = *(a + (rs_a * 2) + (cs_a * (k_full_pieces * 2)));
		a_int32_0 = _mm256_set1_epi8(a_element[2]);

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_2p0 = _mm256_add_epi16(inter_vec, c_int16_2p0);

		a_element[3] = *(a + (rs_a * 3) + (cs_a * (k_full_pieces * 2)));
		a_int32_0 = _mm256_set1_epi8(a_element[3]);

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_3p0 = _mm256_add_epi16(inter_vec, c_int16_3p0);
	}

	// Load alpha and beta
	__m256i selector1 = _mm256_set1_epi16(alpha);
	__m256i selector2 = _mm256_set1_epi16(beta);

	// Scale by alpha
	c_int16_0p0 = _mm256_mullo_epi16(selector1, c_int16_0p0);

	c_int16_1p0 = _mm256_mullo_epi16(selector1, c_int16_1p0);

	c_int16_2p0 = _mm256_mullo_epi16(selector1, c_int16_2p0);

	c_int16_3p0 = _mm256_mullo_epi16(selector1, c_int16_3p0);

	// Scale C by beta.
	if (beta != 0)
	{
		// c[0,0-15]
		selector1 = _mm256_loadu_si256((__m256i const *)(c + (rs_c * 0) + (0 * 16)));
		selector1 = _mm256_mullo_epi16(selector2, selector1);
		c_int16_0p0 = _mm256_add_epi16(selector1, c_int16_0p0);

		// c[1,0-15]
		selector1 = _mm256_loadu_si256((__m256i const *)(c + (rs_c * 1) + (0 * 16)));
		selector1 = _mm256_mullo_epi16(selector2, selector1);
		c_int16_1p0 = _mm256_add_epi16(selector1, c_int16_1p0);

		// c[2,0-15]
		selector1 = _mm256_loadu_si256((__m256i const *)(c + (rs_c * 2) + (0 * 16)));
		selector1 = _mm256_mullo_epi16(selector2, selector1);
		c_int16_2p0 = _mm256_add_epi16(selector1, c_int16_2p0);

		// c[3,0-15]
		selector1 = _mm256_loadu_si256((__m256i const *)(c + (rs_c * 3) + (0 * 16)));
		selector1 = _mm256_mullo_epi16(selector2, selector1);
		c_int16_3p0 = _mm256_add_epi16(selector1, c_int16_3p0);
	}

	// Store the results.
	// c[0,0-15]
	_mm256_storeu_si256((__m256i *)(c + (rs_c * 0) + (0 * 16)), c_int16_0p0);

	// c[1,0-15]
	_mm256_storeu_si256((__m256i *)(c + (rs_c * 1) + (0 * 16)), c_int16_1p0);

	// c[2,0-15]
	_mm256_storeu_si256((__m256i *)(c + (rs_c * 2) + (0 * 16)), c_int16_2p0);

	// c[3,0-15]
	_mm256_storeu_si256((__m256i *)(c + (rs_c * 3) + (0 * 16)), c_int16_3p0);
}

// 4x16 int8o16 kernel
void lpgemm_rowvar_u8s8s16o16_4xlt16
	(
		const dim_t k0,
		const uint8_t *a,
		const dim_t rs_a,
		const dim_t cs_a,
		const int8_t *b,
		const dim_t rs_b,
		const dim_t cs_b,
		int16_t *c,
		const dim_t rs_c,
		const int16_t alpha,
		const int16_t beta,
		dim_t n0_rem
	)
{
	dim_t NR = 16;

	// The division is done by considering the vpmaddubsw instruction
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	// B matrix storage.
	__m256i b0;

	// A matrix storage.
	__m256i a_int32_0;
	__m256i inter_vec;

	int16_t buf0[16];
	int16_t buf1[16];
	int16_t buf2[16];
	int16_t buf3[16];

	//  Registers to use for accumulating C.
	__m256i c_int16_0p0 = _mm256_setzero_si256();

	__m256i c_int16_1p0 = _mm256_setzero_si256();

	__m256i c_int16_2p0 = _mm256_setzero_si256();

	__m256i c_int16_3p0 = _mm256_setzero_si256();

	for (dim_t kr = 0; kr < k_full_pieces; kr += 1)
	{
		dim_t offset = kr * 2;

		b0 = _mm256_loadu_si256((__m256i const *)(b + (32 * kr) + (NR * 0)));

		// Broadcast a[0,kr:kr+2].
		a_int32_0 = _mm256_set1_epi16(*(uint16_t *)(a + (rs_a * 0) + (cs_a * offset)));

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_0p0 = _mm256_add_epi16(inter_vec, c_int16_0p0);

		// Broadcast a[1,kr:kr+2].
		a_int32_0 = _mm256_set1_epi16(*(uint16_t *)(a + (rs_a * 1) + (cs_a * offset)));

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_1p0 = _mm256_add_epi16(inter_vec, c_int16_1p0);

		// Broadcast a[2,kr:kr+2].
		a_int32_0 = _mm256_set1_epi16(*(uint16_t *)(a + (rs_a * 2) + (cs_a * offset)));

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[2,0-15] = a[2,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_2p0 = _mm256_add_epi16(inter_vec, c_int16_2p0);

		// Broadcast a[3,kr:kr+2].
		a_int32_0 = _mm256_set1_epi16(*(uint16_t *)(a + (rs_a * 3) + (cs_a * offset)));

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[3,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_3p0 = _mm256_add_epi16(inter_vec, c_int16_3p0);
	}

	// Handle k remainder.
	if (k_partial_pieces > 0)
	{
		uint8_t a_element[4];

		b0 = _mm256_loadu_si256((__m256i const *)(b + (32 * k_full_pieces) + (NR * 0)));

		a_element[0] = *(a + (rs_a * 0) + (cs_a * (k_full_pieces * 2)));
		a_int32_0 = _mm256_set1_epi8(a_element[0]);

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_0p0 = _mm256_add_epi16(inter_vec, c_int16_0p0);

		a_element[1] = *(a + (rs_a * 1) + (cs_a * (k_full_pieces * 2)));
		a_int32_0 = _mm256_set1_epi8(a_element[1]);

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_1p0 = _mm256_add_epi16(inter_vec, c_int16_1p0);

		a_element[2] = *(a + (rs_a * 2) + (cs_a * (k_full_pieces * 2)));
		a_int32_0 = _mm256_set1_epi8(a_element[2]);

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[2,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_2p0 = _mm256_add_epi16(inter_vec, c_int16_2p0);

		a_element[3] = *(a + (rs_a * 3) + (cs_a * (k_full_pieces * 2)));
		a_int32_0 = _mm256_set1_epi8(a_element[3]);

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_3p0 = _mm256_add_epi16(inter_vec, c_int16_3p0);
	}

	// Load alpha and beta
	__m256i selector1 = _mm256_set1_epi16(alpha);
	__m256i selector2 = _mm256_set1_epi16(beta);

	// Scale by alpha
	c_int16_0p0 = _mm256_mullo_epi16(selector1, c_int16_0p0);

	c_int16_1p0 = _mm256_mullo_epi16(selector1, c_int16_1p0);

	c_int16_2p0 = _mm256_mullo_epi16(selector1, c_int16_2p0);

	c_int16_3p0 = _mm256_mullo_epi16(selector1, c_int16_3p0);

	// Scale C by beta.
	if (beta != 0)
	{
		memcpy(buf0, (c + (rs_c * 0)), (n0_rem * sizeof(int16_t)));
		memcpy(buf1, (c + (rs_c * 1)), (n0_rem * sizeof(int16_t)));
		memcpy(buf2, (c + (rs_c * 2)), (n0_rem * sizeof(int16_t)));
		memcpy(buf3, (c + (rs_c * 3)), (n0_rem * sizeof(int16_t)));

		// c[0,0-15]
		selector1 = _mm256_loadu_si256((__m256i const *)buf0);
		selector1 = _mm256_mullo_epi16(selector2, selector1);
		c_int16_0p0 = _mm256_add_epi16(selector1, c_int16_0p0);

		// c[1,0-15]
		selector1 = _mm256_loadu_si256((__m256i const *)buf1);
		selector1 = _mm256_mullo_epi16(selector2, selector1);
		c_int16_1p0 = _mm256_add_epi16(selector1, c_int16_1p0);

		// c[2,0-15]
		selector1 = _mm256_loadu_si256((__m256i const *)buf2);
		selector1 = _mm256_mullo_epi16(selector2, selector1);
		c_int16_2p0 = _mm256_add_epi16(selector1, c_int16_2p0);

		// c[3,0-15]
		selector1 = _mm256_loadu_si256((__m256i const *)buf3);
		selector1 = _mm256_mullo_epi16(selector2, selector1);
		c_int16_3p0 = _mm256_add_epi16(selector1, c_int16_3p0);
	}

	// c[0,0-15]
	_mm256_storeu_si256((__m256i_u *)buf0, c_int16_0p0);

	// c[1,0-15]
	_mm256_storeu_si256((__m256i_u *)buf1, c_int16_1p0);

	// c[2,0-15]
	_mm256_storeu_si256((__m256i_u *)buf2, c_int16_2p0);

	// c[3,0-15]
	_mm256_storeu_si256((__m256i_u *)buf3, c_int16_3p0);

	memcpy(c + (rs_c * 0) + (0 * 16), buf0, (n0_rem * sizeof(int16_t)));

	// c[1,0-15]
	memcpy(c + (rs_c * +1) + (0 * 16), buf1, (n0_rem * sizeof(int16_t)));

	// c[2,0-15]
	memcpy(c + (rs_c * +2) + (0 * 16), buf2, (n0_rem * sizeof(int16_t)));

	// c[3,0-15]
	memcpy(c + (rs_c * +3) + (0 * 16), buf3, (n0_rem * sizeof(int16_t)));
}

// 2x16 int8o16 kernel
void lpgemm_rowvar_u8s8s16o16_2x16
	(
		const dim_t k0,
		const uint8_t *a,
		const dim_t rs_a,
		const dim_t cs_a,
		const int8_t *b,
		const dim_t rs_b,
		const dim_t cs_b,
		int16_t *c,
		const dim_t rs_c,
		const int16_t alpha,
		const int16_t beta
	)
{
	dim_t NR = 16;

	// The division is done by considering the vpmaddubsw instruction
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	// B matrix storage.
	__m256i b0;

	// A matrix storage.
	__m256i a_int32_0;
	__m256i inter_vec;

	//  Registers to use for accumulating C.
	__m256i c_int16_0p0 = _mm256_setzero_si256();

	__m256i c_int16_1p0 = _mm256_setzero_si256();

	for (dim_t kr = 0; kr < k_full_pieces; kr += 1)
	{
		dim_t offset = kr * 2;

		b0 = _mm256_loadu_si256((__m256i const *)(b + (32 * kr) + (NR * 0)));

		// Broadcast a[0,kr:kr+2].
		a_int32_0 = _mm256_set1_epi16(*(uint16_t *)(a + (rs_a * 0) + (cs_a * offset)));

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_0p0 = _mm256_add_epi16(inter_vec, c_int16_0p0);

		// Broadcast a[1,kr:kr+2].
		a_int32_0 = _mm256_set1_epi16(*(uint16_t *)(a + (rs_a * 1) + (cs_a * offset)));

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[1,0-15] = a[1,kr:kr+4]*b[kr:kr+2,0-31]
		c_int16_1p0 = _mm256_add_epi16(inter_vec, c_int16_1p0);
	}
	// Handle k remainder.
	if (k_partial_pieces > 0)
	{
		uint8_t a_element[2];

		b0 = _mm256_loadu_si256((__m256i const *)(b + (32 * k_full_pieces) + (NR * 0)));

		a_element[0] = *(a + (rs_a * 0) + (cs_a * (k_full_pieces * 2)));
		a_int32_0 = _mm256_set1_epi8(a_element[0]);

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_0p0 = _mm256_add_epi16(inter_vec, c_int16_0p0);

		a_element[1] = *(a + (rs_a * 1) + (cs_a * (k_full_pieces * 2)));
		a_int32_0 = _mm256_set1_epi8(a_element[1]);

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_1p0 = _mm256_add_epi16(inter_vec, c_int16_1p0);
	}

	// Load alpha and beta
	__m256i selector1 = _mm256_set1_epi16(alpha);
	__m256i selector2 = _mm256_set1_epi16(beta);

	// Scale by alpha
	c_int16_0p0 = _mm256_mullo_epi16(selector1, c_int16_0p0);

	c_int16_1p0 = _mm256_mullo_epi16(selector1, c_int16_1p0);

	// Scale C by beta.
	if (beta != 0)
	{
		// c[0,0-15]
		selector1 = _mm256_loadu_si256((__m256i const *)(c + (rs_c * 0) + (0 * 16)));
		selector1 = _mm256_mullo_epi16(selector2, selector1);
		c_int16_0p0 = _mm256_add_epi16(selector1, c_int16_0p0);

		// c[1,0-15]
		selector1 = _mm256_loadu_si256((__m256i const *)(c + (rs_c * 1) + (0 * 16)));
		selector1 = _mm256_mullo_epi16(selector2, selector1);
		c_int16_1p0 = _mm256_add_epi16(selector1, c_int16_1p0);
	}

	// Store the results.
	// c[0,0-15]
	_mm256_storeu_si256((__m256i *)(c + (rs_c * 0) + (0 * 16)), c_int16_0p0);

	// c[1,0-15]
	_mm256_storeu_si256((__m256i *)(c + (rs_c * 1) + (0 * 16)), c_int16_1p0);
}

// 2xlt16 int8o16 kernel
void lpgemm_rowvar_u8s8s16o16_2xlt16
	(
		const dim_t k0,
		const uint8_t *a,
		const dim_t rs_a,
		const dim_t cs_a,
		const int8_t *b,
		const dim_t rs_b,
		const dim_t cs_b,
		int16_t *c,
		const dim_t rs_c,
		const int16_t alpha,
		const int16_t beta,
		dim_t n0_rem
	)
{
	dim_t NR = 16;

	// The division is done by considering the vpmaddubsw instruction
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	// B matrix storage.
	__m256i b0;

	// A matrix storage.
	__m256i a_int32_0;
	__m256i inter_vec;

	int16_t buf0[16];
	int16_t buf1[16];

	//  Registers to use for accumulating C.
	__m256i c_int16_0p0 = _mm256_setzero_si256();

	__m256i c_int16_1p0 = _mm256_setzero_si256();

	for (dim_t kr = 0; kr < k_full_pieces; kr += 1)
	{
		dim_t offset = kr * 2;

		b0 = _mm256_loadu_si256((__m256i const *)(b + (32 * kr) + (NR * 0)));

		// Broadcast a[0,kr:kr+2].
		a_int32_0 = _mm256_set1_epi16(*(uint16_t *)(a + (rs_a * 0) + (cs_a * offset)));

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_0p0 = _mm256_add_epi16(inter_vec, c_int16_0p0);

		// Broadcast a[1,kr:kr+2].
		a_int32_0 = _mm256_set1_epi16(*(uint16_t *)(a + (rs_a * 1) + (cs_a * offset)));

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 4.
		// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_1p0 = _mm256_add_epi16(inter_vec, c_int16_1p0);
	}
	// Handle k remainder.
	if (k_partial_pieces > 0)
	{
		uint8_t a_element[4];

		b0 = _mm256_loadu_si256((__m256i const *)(b + (32 * k_full_pieces) + (NR * 0)));

		a_element[0] = *(a + (rs_a * 0) + (cs_a * (k_full_pieces * 2)));
		a_int32_0 = _mm256_set1_epi8(a_element[0]);

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_0p0 = _mm256_add_epi16(inter_vec, c_int16_0p0);

		a_element[1] = *(a + (rs_a * 1) + (cs_a * (k_full_pieces * 2)));
		a_int32_0 = _mm256_set1_epi8(a_element[1]);

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_1p0 = _mm256_add_epi16(inter_vec, c_int16_1p0);
	}

	// Load alpha and beta
	__m256i selector1 = _mm256_set1_epi16(alpha);
	__m256i selector2 = _mm256_set1_epi16(beta);

	// Scale by alpha
	c_int16_0p0 = _mm256_mullo_epi16(selector1, c_int16_0p0);

	c_int16_1p0 = _mm256_mullo_epi16(selector1, c_int16_1p0);

	// Scale C by beta.
	if (beta != 0)
	{
		memcpy(buf0, (c + (rs_c * 0)), (n0_rem * sizeof(int16_t)));
		memcpy(buf1, (c + (rs_c * 1)), (n0_rem * sizeof(int16_t)));

		// c[0,0-15]
		selector1 = _mm256_loadu_si256((__m256i const *)buf0);
		selector1 = _mm256_mullo_epi16(selector2, selector1);
		c_int16_0p0 = _mm256_add_epi16(selector1, c_int16_0p0);

		// c[1,0-15]
		selector1 = _mm256_loadu_si256((__m256i const *)buf1);
		selector1 = _mm256_mullo_epi16(selector2, selector1);
		c_int16_1p0 = _mm256_add_epi16(selector1, c_int16_1p0);
	}

	// c[0,0-15]
	_mm256_storeu_si256((__m256i_u *)buf0, c_int16_0p0);

	// c[1,0-15]
	_mm256_storeu_si256((__m256i_u *)buf1, c_int16_1p0);

	// c[0,0-15]
	memcpy(c + (rs_c * 0) + (0 * 16), buf0, (n0_rem * sizeof(int16_t)));

	// c[1,0-15]
	memcpy(c + (rs_c * +1) + (0 * 16), buf1, (n0_rem * sizeof(int16_t)));
}

// 1x16 int8o16 kernel
void lpgemm_rowvar_u8s8s16o16_1x16
	(
		const dim_t k0,
		const uint8_t *a,
		const dim_t rs_a,
		const dim_t cs_a,
		const int8_t *b,
		const dim_t rs_b,
		const dim_t cs_b,
		int16_t *c,
		const dim_t rs_c,
		const int16_t alpha,
		const int16_t beta
	)
{
	int NR = 16;

	// The division is done by considering the vpmaddubsw instruction
	int k_full_pieces = k0 / 2;
	int k_partial_pieces = k0 % 2;

	// B matrix storage.
	__m256i b0;

	// A matrix storage.
	__m256i a_int32_0;
	__m256i inter_vec;

	//  Registers to use for accumulating C.
	__m256i c_int16_0p0 = _mm256_setzero_si256();

	for (int kr = 0; kr < k_full_pieces; kr += 1)
	{
		int offset = kr * 2;

		// Broadcast a[0,kr:kr+2].
		a_int32_0 = _mm256_set1_epi16(*(uint16_t *)(a + (rs_a * 0) + (cs_a * offset)));

		b0 = _mm256_loadu_si256((__m256i const *)(b + (32 * kr) + (NR * 0)));

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_0p0 = _mm256_add_epi16(inter_vec, c_int16_0p0);
	}
	// Handle k remainder.
	if (k_partial_pieces > 0)
	{
		uint8_t a_element[1];

		b0 = _mm256_loadu_si256((__m256i const *)(b + (64 * k_full_pieces) + (NR * 0)));

		a_element[0] = *(a + (rs_a * 0) + (cs_a * (k_full_pieces * 2)));
		a_int32_0 = _mm256_set1_epi8(a_element[0]);

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_0p0 = _mm256_add_epi16(inter_vec, c_int16_0p0);
	}

	// Load alpha and beta
	__m256i selector1 = _mm256_set1_epi16(alpha);
	__m256i selector2 = _mm256_set1_epi16(beta);

	// Scale by alpha
	c_int16_0p0 = _mm256_mullo_epi16(selector1, c_int16_0p0);

	// Scale C by beta.
	if (beta != 0)
	{
		// c[0,0-15]
		selector1 = _mm256_loadu_si256((__m256i const *)(c + (rs_c * 0) + (0 * 16)));
		selector1 = _mm256_mullo_epi16(selector2, selector1);
		c_int16_0p0 = _mm256_add_epi16(selector1, c_int16_0p0);
	}

	// Store the results.
	// c[0,0-15]
	_mm256_storeu_si256((__m256i *)(c + (rs_c * 0) + (0 * 16)), c_int16_0p0);
}

// 1xlt16 int8o16 kernel
void lpgemm_rowvar_u8s8s16o16_1xlt16
	(
		const int k0,
		const uint8_t *a,
		const int rs_a,
		const int cs_a,
		const int8_t *b,
		const int rs_b,
		const int cs_b,
		int16_t *c,
		const int rs_c,
		const int16_t alpha,
		const int16_t beta,
		dim_t n0_rem
	)
{
	int NR = 16;

	// The division is done by considering the vpmaddubsw instruction
	int k_full_pieces = k0 / 2;
	int k_partial_pieces = k0 % 2;

	// B matrix storage.
	__m256i b0;

	// A matrix storage.
	__m256i a_int32_0;
	__m256i inter_vec;

	int16_t buf0[16];

	//  Registers to use for accumulating C.
	__m256i c_int16_0p0 = _mm256_setzero_si256();

	for (int kr = 0; kr < k_full_pieces; kr += 1)
	{
		int offset = kr * 2;

		b0 = _mm256_loadu_si256((__m256i const *)(b + (32 * kr) + (NR * 0)));

		// Broadcast a[0,kr:kr+2].
		a_int32_0 = _mm256_set1_epi16(*(uint16_t *)(a + (rs_a * 0) + (cs_a * offset)));

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_0p0 = _mm256_add_epi16(inter_vec, c_int16_0p0);
	}
	// Handle k remainder.
	if (k_partial_pieces > 0)
	{
		uint8_t a_element[4];

		b0 = _mm256_loadu_si256((__m256i const *)(b + (32 * k_full_pieces) + (NR * 0)));

		a_element[0] = *(a + (rs_a * 0) + (cs_a * (k_full_pieces * 2)));
		a_int32_0 = _mm256_set1_epi8(a_element[0]);

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_0p0 = _mm256_add_epi16(inter_vec, c_int16_0p0);
	}

	// Load alpha and beta
	__m256i selector1 = _mm256_set1_epi16(alpha);
	__m256i selector2 = _mm256_set1_epi16(beta);

	// Scale by alpha
	c_int16_0p0 = _mm256_mullo_epi16(selector1, c_int16_0p0);

	// Scale C by beta.
	if (beta != 0)
	{
		memcpy(buf0, (c + (rs_c * 0)), (n0_rem * sizeof(int16_t)));

		// c[0,0-15]
		selector1 = _mm256_loadu_si256((__m256i const *)buf0);
		selector1 = _mm256_mullo_epi16(selector2, selector1);
		c_int16_0p0 = _mm256_add_epi16(selector1, c_int16_0p0);
	}

	// c[0,0-15]
	_mm256_storeu_si256((__m256i_u *)buf0, c_int16_0p0);

	memcpy(c + (rs_c * 0) + (0 * 16), buf0, (n0_rem * sizeof(int16_t)));
}
