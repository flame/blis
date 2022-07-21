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
#include "lpgemm_6x32rowmajor.h"
#include "lpgemm_m_fringe_s16.h"
#include "lpgemm_n_fringe_s16.h"

// 6x32 int8o16 kernel
void lpgemm_rowvar_u8s8s16o16_6x32
	(
		const dim_t m0,
		const dim_t n0,
		const dim_t k0,
		const uint8_t *a,
		const dim_t rs_a,
		const dim_t cs_a,
		const dim_t ps_a,
		const int8_t *b,
		const dim_t rs_b,
		const dim_t cs_b,
		int16_t *c,
		const dim_t rs_c,
		const dim_t cs_c,
		const int16_t alpha,
		const int16_t beta
	)
{
	dim_t MR = 6;
	dim_t NR = 32;

	dim_t m_full_pieces = m0 / MR;
	dim_t m_full_pieces_loop_limit = m_full_pieces * MR;
	dim_t m_partial_pieces = m0 % MR;

	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	// When n fringe cases are encountered
	if (n0 < NR)
	{
		// Split into multiple smaller fringe kernels, so as to maximize
		// vectorization after packing. Any n0 < NR(32) can be expressed
		// as n0 = 16 + n`.
		dim_t n0_rem = n0 % 16;
		dim_t n0_16 = n0 / 16;
		dim_t k0_updated = k0;

		// Making multiple of 2 to suit k in vpmaddubsw
      	k0_updated += (k0_updated & 0x1);

		if (n0_16 == 1)
		{
			lpgemm_rowvar_u8s8s16o16_6x16(
				m0, k0_updated,
				a, rs_a, cs_a, ps_a,
				b, ((rs_b / 2) * 1), cs_b,
				c, rs_c,
				alpha, beta);

			b = b + (16 * k0_updated);
			c = c + 16;
		}

		if (n0_rem > 0)
		{
			lpgemm_rowvar_u8s8s16o16_6xlt16(
				m0, k0_updated,
				a, rs_a, cs_a, ps_a,
				b, ((rs_b / 2) * 1), cs_b,
				c, rs_c,
				alpha, beta, n0_rem);
		}

		// If fringe cases are encountered, return early
		return;
	}

	// B matrix storage.
	__m256i b0, b1;

	// A matrix storage.
	__m256i a_int32_0, a_int32_1;

	// Intermediate vectors
	__m256i inter_vec[4];

	for (dim_t ir = 0; ir < m_full_pieces_loop_limit; ir += MR)
	{
		// Registers to use for accumulating C.
		__m256i c_int16_0p0 = _mm256_setzero_si256();
		__m256i c_int16_0p1 = _mm256_setzero_si256();

		__m256i c_int16_1p0 = _mm256_setzero_si256();
		__m256i c_int16_1p1 = _mm256_setzero_si256();

		__m256i c_int16_2p0 = _mm256_setzero_si256();
		__m256i c_int16_2p1 = _mm256_setzero_si256();

		__m256i c_int16_3p0 = _mm256_setzero_si256();
		__m256i c_int16_3p1 = _mm256_setzero_si256();

		__m256i c_int16_4p0 = _mm256_setzero_si256();
		__m256i c_int16_4p1 = _mm256_setzero_si256();

		__m256i c_int16_5p0 = _mm256_setzero_si256();
		__m256i c_int16_5p1 = _mm256_setzero_si256();

		for (dim_t kr = 0; kr < k_full_pieces; kr += 1)
		{
			dim_t offset = kr * 2;

			// Broadcast a[0,kr:kr+2].
			a_int32_0 = _mm256_set1_epi16(*(uint16_t *)(a + (rs_a * 0) + (cs_a * offset)));

			b0 = _mm256_loadu_si256((__m256i const *)(b + (64 * kr) + (NR * 0)));
			b1 = _mm256_loadu_si256((__m256i const *)(b + (64 * kr) + (NR * 1)));

			// Broadcast a[1,kr:kr+2].
			a_int32_1 = _mm256_set1_epi16(*(uint16_t *)(a + (rs_a * 1) + (cs_a * offset)));

			// Seperate register for intermediate op
			inter_vec[0] = _mm256_maddubs_epi16(a_int32_0, b0);
			inter_vec[1] = _mm256_maddubs_epi16(a_int32_0, b1);

			// Perform column direction mat-mul with k = 2.
			// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
			c_int16_0p0 = _mm256_add_epi16(inter_vec[0], c_int16_0p0);
			c_int16_0p1 = _mm256_add_epi16(inter_vec[1], c_int16_0p1);

			// Broadcast a[2,kr:kr+2].
			a_int32_0 = _mm256_set1_epi16(*(uint16_t *)(a + (rs_a * 2) + (cs_a * offset)));

			// Seperate register for intermediate op
			inter_vec[2] = _mm256_maddubs_epi16(a_int32_1, b0);
			inter_vec[3] = _mm256_maddubs_epi16(a_int32_1, b1);

			// Perform column direction mat-mul with k = 2.
			// c[1,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
			c_int16_1p0 = _mm256_add_epi16(inter_vec[2], c_int16_1p0);
			c_int16_1p1 = _mm256_add_epi16(inter_vec[3], c_int16_1p1);

			// Broadcast a[3,kr:kr+2].
			a_int32_1 = _mm256_set1_epi16(*(uint16_t *)(a + (rs_a * 3) + (cs_a * offset)));

			// Seperate register for intermediate op
			inter_vec[0] = _mm256_maddubs_epi16(a_int32_0, b0);
			inter_vec[1] = _mm256_maddubs_epi16(a_int32_0, b1);

			// Perform column direction mat-mul with k = 2.
			// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
			c_int16_2p0 = _mm256_add_epi16(inter_vec[0], c_int16_2p0);
			c_int16_2p1 = _mm256_add_epi16(inter_vec[1], c_int16_2p1);

			// Broadcast a[4,kr:kr+2].
			a_int32_0 = _mm256_set1_epi16(*(uint16_t *)(a + (rs_a * 4) + (cs_a * offset)));

			// Seperate register for intermediate op
			inter_vec[2] = _mm256_maddubs_epi16(a_int32_1, b0);
			inter_vec[3] = _mm256_maddubs_epi16(a_int32_1, b1);

			// Perform column direction mat-mul with k = 2.
			// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
			c_int16_3p0 = _mm256_add_epi16(inter_vec[2], c_int16_3p0);
			c_int16_3p1 = _mm256_add_epi16(inter_vec[3], c_int16_3p1);

			// Broadcast a[5,kr:kr+2].
			a_int32_1 = _mm256_set1_epi16(*(uint16_t *)(a + (rs_a * 5) + (cs_a * offset)));

			// Seperate register for intermediate op
			inter_vec[0] = _mm256_maddubs_epi16(a_int32_0, b0);
			inter_vec[1] = _mm256_maddubs_epi16(a_int32_0, b1);

			// Perform column direction mat-mul with k = 2.
			// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+4,0-31]
			c_int16_4p0 = _mm256_add_epi16(inter_vec[0], c_int16_4p0);
			c_int16_4p1 = _mm256_add_epi16(inter_vec[1], c_int16_4p1);

			// Seperate register for intermediate op
			inter_vec[2] = _mm256_maddubs_epi16(a_int32_1, b0);
			inter_vec[3] = _mm256_maddubs_epi16(a_int32_1, b1);

			// Perform column direction mat-mul with k = 2.
			// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+4,0-31]
			c_int16_5p0 = _mm256_add_epi16(inter_vec[2], c_int16_5p0);
			c_int16_5p1 = _mm256_add_epi16(inter_vec[3], c_int16_5p1);
		}

		// Handle k remainder.
		if (k_partial_pieces > 0)
		{
			uint8_t a_element[6];

			b0 = _mm256_loadu_si256((__m256i const *)(b + (64 * k_full_pieces) + (NR * 0)));
			b1 = _mm256_loadu_si256((__m256i const *)(b + (64 * k_full_pieces) + (NR * 1)));

			a_element[0] = *(a + (rs_a * 0) + (cs_a * (k_full_pieces * 2)));
			a_int32_0 = _mm256_set1_epi8(a_element[0]);

			// Seperate register for intermediate op
			inter_vec[0] = _mm256_maddubs_epi16(a_int32_0, b0);
			inter_vec[1] = _mm256_maddubs_epi16(a_int32_0, b1);

			// Perform column direction mat-mul with k = 2.
			// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
			c_int16_0p0 = _mm256_add_epi16(inter_vec[0], c_int16_0p0);
			c_int16_0p1 = _mm256_add_epi16(inter_vec[1], c_int16_0p1);

			a_element[1] = *(a + (rs_a * 1) + (cs_a * (k_full_pieces * 2)));
			a_int32_1 = _mm256_set1_epi8(a_element[1]);

			// Seperate register for intermediate op
			inter_vec[2] = _mm256_maddubs_epi16(a_int32_1, b0);
			inter_vec[3] = _mm256_maddubs_epi16(a_int32_1, b1);

			// Perform column direction mat-mul with k = 2.
			// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+4,0-31]
			c_int16_1p0 = _mm256_add_epi16(inter_vec[2], c_int16_1p0);
			c_int16_1p1 = _mm256_add_epi16(inter_vec[3], c_int16_1p1);

			a_element[2] = *(a + (rs_a * 2) + (cs_a * (k_full_pieces * 2)));
			a_int32_0 = _mm256_set1_epi8(a_element[2]);

			// Seperate register for intermediate op
			inter_vec[0] = _mm256_maddubs_epi16(a_int32_0, b0);
			inter_vec[1] = _mm256_maddubs_epi16(a_int32_0, b1);

			// Perform column direction mat-mul with k = 2.
			// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
			c_int16_2p0 = _mm256_add_epi16(inter_vec[0], c_int16_2p0);
			c_int16_2p1 = _mm256_add_epi16(inter_vec[1], c_int16_2p1);

			a_element[3] = *(a + (rs_a * 3) + (cs_a * (k_full_pieces * 2)));
			a_int32_1 = _mm256_set1_epi8(a_element[3]);

			// Seperate register for intermediate op
			inter_vec[2] = _mm256_maddubs_epi16(a_int32_1, b0);
			inter_vec[3] = _mm256_maddubs_epi16(a_int32_1, b1);

			// Perform column direction mat-mul with k = 2.
			// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
			c_int16_3p0 = _mm256_add_epi16(inter_vec[2], c_int16_3p0);
			c_int16_3p1 = _mm256_add_epi16(inter_vec[3], c_int16_3p1);

			a_element[4] = *(a + (rs_a * 4) + (cs_a * (k_full_pieces * 2)));
			a_int32_0 = _mm256_set1_epi8(a_element[4]);

			// Seperate register for intermediate op
			inter_vec[0] = _mm256_maddubs_epi16(a_int32_0, b0);
			inter_vec[1] = _mm256_maddubs_epi16(a_int32_0, b1);

			// Perform column direction mat-mul with k = 2.
			// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
			c_int16_4p0 = _mm256_add_epi16(inter_vec[0], c_int16_4p0);
			c_int16_4p1 = _mm256_add_epi16(inter_vec[1], c_int16_4p1);

			a_element[5] = *(a + (rs_a * 5) + (cs_a * (k_full_pieces * 2)));
			a_int32_1 = _mm256_set1_epi8(a_element[5]);

			// Seperate register for intermediate op
			inter_vec[2] = _mm256_maddubs_epi16(a_int32_1, b0);
			inter_vec[3] = _mm256_maddubs_epi16(a_int32_1, b1);

			// Perform column direction mat-mul with k = 2.
			// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
			c_int16_5p0 = _mm256_add_epi16(inter_vec[2], c_int16_5p0);
			c_int16_5p1 = _mm256_add_epi16(inter_vec[3], c_int16_5p1);
		}

		// Load alpha and beta
		__m256i selector1 = _mm256_set1_epi16(alpha);
		__m256i selector2 = _mm256_set1_epi16(beta);

		// Scale by alpha
		c_int16_0p0 = _mm256_mullo_epi16(selector1, c_int16_0p0);
		c_int16_0p1 = _mm256_mullo_epi16(selector1, c_int16_0p1);

		c_int16_1p0 = _mm256_mullo_epi16(selector1, c_int16_1p0);
		c_int16_1p1 = _mm256_mullo_epi16(selector1, c_int16_1p1);

		c_int16_2p0 = _mm256_mullo_epi16(selector1, c_int16_2p0);
		c_int16_2p1 = _mm256_mullo_epi16(selector1, c_int16_2p1);

		c_int16_3p0 = _mm256_mullo_epi16(selector1, c_int16_3p0);
		c_int16_3p1 = _mm256_mullo_epi16(selector1, c_int16_3p1);

		c_int16_4p0 = _mm256_mullo_epi16(selector1, c_int16_4p0);
		c_int16_4p1 = _mm256_mullo_epi16(selector1, c_int16_4p1);

		c_int16_5p0 = _mm256_mullo_epi16(selector1, c_int16_5p0);
		c_int16_5p1 = _mm256_mullo_epi16(selector1, c_int16_5p1);

		// Scale C by beta.
		if (beta != 0)
		{
			// c[0,0-15]
			selector1 = _mm256_loadu_si256((__m256i const *)(c + (rs_c * (ir + 0)) + (0 * 16)));
			selector1 = _mm256_mullo_epi16(selector2, selector1);
			c_int16_0p0 = _mm256_add_epi16(selector1, c_int16_0p0);

			// c[0, 16-31]
			selector1 = _mm256_loadu_si256((__m256i const *)(c + (rs_c * (ir + 0)) + (1 * 16)));
			selector1 = _mm256_mullo_epi16(selector2, selector1);
			c_int16_0p1 = _mm256_add_epi16(selector1, c_int16_0p1);

			// c[1,0-15]
			selector1 = _mm256_loadu_si256((__m256i const *)(c + (rs_c * (ir + 1)) + (0 * 16)));
			selector1 = _mm256_mullo_epi16(selector2, selector1);
			c_int16_1p0 = _mm256_add_epi16(selector1, c_int16_1p0);

			// c[1,16-31]
			selector1 = _mm256_loadu_si256((__m256i const *)(c + (rs_c * (ir + 1)) + (1 * 16)));
			selector1 = _mm256_mullo_epi16(selector2, selector1);
			c_int16_1p1 = _mm256_add_epi16(selector1, c_int16_1p1);

			// c[2,0-15]
			selector1 = _mm256_loadu_si256((__m256i const *)(c + (rs_c * (ir + 2)) + (0 * 16)));
			selector1 = _mm256_mullo_epi16(selector2, selector1);
			c_int16_2p0 = _mm256_add_epi16(selector1, c_int16_2p0);

			// c[2,16-31]
			selector1 = _mm256_loadu_si256((__m256i const *)(c + (rs_c * (ir + 2)) + (1 * 16)));
			selector1 = _mm256_mullo_epi16(selector2, selector1);
			c_int16_2p1 = _mm256_add_epi16(selector1, c_int16_2p1);

			// c[3,0-15]
			selector1 = _mm256_loadu_si256((__m256i const *)(c + (rs_c * (ir + 3)) + (0 * 16)));
			selector1 = _mm256_mullo_epi16(selector2, selector1);
			c_int16_3p0 = _mm256_add_epi16(selector1, c_int16_3p0);

			// c[3,16-31]
			selector1 = _mm256_loadu_si256((__m256i const *)(c + (rs_c * (ir + 3)) + (1 * 16)));
			selector1 = _mm256_mullo_epi16(selector2, selector1);
			c_int16_3p1 = _mm256_add_epi16(selector1, c_int16_3p1);

			// c[4,0-15]
			selector1 = _mm256_loadu_si256((__m256i const *)(c + (rs_c * (ir + 4)) + (0 * 16)));
			selector1 = _mm256_mullo_epi16(selector2, selector1);
			c_int16_4p0 = _mm256_add_epi16(selector1, c_int16_4p0);

			// c[4,16-31]
			selector1 = _mm256_loadu_si256((__m256i const *)(c + (rs_c * (ir + 4)) + (1 * 16)));
			selector1 = _mm256_mullo_epi16(selector2, selector1);
			c_int16_4p1 = _mm256_add_epi16(selector1, c_int16_4p1);

			// c[5,0-15]
			selector1 = _mm256_loadu_si256((__m256i const *)(c + (rs_c * (ir + 5)) + (0 * 16)));
			selector1 = _mm256_mullo_epi16(selector2, selector1);
			c_int16_5p0 = _mm256_add_epi16(selector1, c_int16_5p0);

			// c[5,16-31]
			selector1 = _mm256_loadu_si256((__m256i const *)(c + (rs_c * (ir + 5)) + (1 * 16)));
			selector1 = _mm256_mullo_epi16(selector2, selector1);
			c_int16_5p1 = _mm256_add_epi16(selector1, c_int16_5p1);
		}

		// Store the results.
		// c[0,0-15]
		_mm256_storeu_si256((__m256i *)(c + (rs_c * (ir + 0)) + (0 * 16)), c_int16_0p0);

		// c[0, 16-31]
		_mm256_storeu_si256((__m256i *)(c + (rs_c * (ir + 0)) + (1 * 16)), c_int16_0p1);

		// c[1,0-15]
		_mm256_storeu_si256((__m256i *)(c + (rs_c * (ir + 1)) + (0 * 16)), c_int16_1p0);

		// c[1,16-31]
		_mm256_storeu_si256((__m256i *)(c + (rs_c * (ir + 1)) + (1 * 16)), c_int16_1p1);

		// c[2,0-15]
		_mm256_storeu_si256((__m256i *)(c + (rs_c * (ir + 2)) + (0 * 16)), c_int16_2p0);

		// c[2,16-31]
		_mm256_storeu_si256((__m256i *)(c + (rs_c * (ir + 2)) + (1 * 16)), c_int16_2p1);

		// c[3,0-15]
		_mm256_storeu_si256((__m256i *)(c + (rs_c * (ir + 3)) + (0 * 16)), c_int16_3p0);

		// c[3,16-31]
		_mm256_storeu_si256((__m256i *)(c + (rs_c * (ir + 3)) + (1 * 16)), c_int16_3p1);

		// c[4,0-15]
		_mm256_storeu_si256((__m256i *)(c + (rs_c * (ir + 4)) + (0 * 16)), c_int16_4p0);

		// c[4,16-31]
		_mm256_storeu_si256((__m256i *)(c + (rs_c * (ir + 4)) + (1 * 16)), c_int16_4p1);

		// c[5,0-15]
		_mm256_storeu_si256((__m256i *)(c + (rs_c * (ir + 5)) + (0 * 16)), c_int16_5p0);

		// c[5,16-31]
		_mm256_storeu_si256((__m256i *)(c + (rs_c * (ir + 5)) + (1 * 16)), c_int16_5p1);

		a = a + (MR * ps_a);
	}

	if (m_partial_pieces > 0)
	{
		// Split into multiple smaller fringe kernels, so as to maximize
		// vectorization after packing. Any m0 < MR(6) can be expressed
		// as a combination of numbers from the set {4, 2, 1}.
		dim_t m_partial4 = m_partial_pieces / 4;
		m_partial_pieces = m_partial_pieces % 4;

		dim_t m_partial2 = m_partial_pieces / 2;
		dim_t m_partial = m_partial_pieces % 2;

		if (m_partial4 == 1)
		{
			lpgemm_rowvar_u8s8s16o16_4x32(
				k0,
				a, rs_a, cs_a,
				b, rs_b, cs_b,
				(c + (rs_c * m_full_pieces_loop_limit)), rs_c,
				alpha, beta);

			// a pointer increment
			a = a + (4 * ps_a);
			m_full_pieces_loop_limit += 4;
		}

		if (m_partial2 == 1)
		{
			lpgemm_rowvar_u8s8s16o16_2x32(
				k0,
				a, rs_a, cs_a,
				b, rs_b, cs_b,
				(c + (rs_c * m_full_pieces_loop_limit)), rs_c,
				alpha, beta);

			// a pointer increment
			a = a + (2 * ps_a);
			m_full_pieces_loop_limit += 2;
		}

		if (m_partial == 1)
		{
			lpgemm_rowvar_u8s8s16o16_1x32(
				k0,
				a, rs_a, cs_a,
				b, rs_b, cs_b,
				(c + (rs_c * m_full_pieces_loop_limit)), rs_c,
				alpha, beta);
		}
	}
}