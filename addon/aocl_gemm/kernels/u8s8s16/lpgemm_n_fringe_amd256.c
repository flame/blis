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
#include "lpgemm_kernels.h"
#include "lpgemm_s16_kern_macros.h"

// 6x16 int8o16 kernel
LPGEMM_N_FRINGE_KERN(uint8_t,int8_t,int16_t,u8s8s16o16_6x16)
{
	dim_t MR = 6;
	dim_t NR = 16;

	static void *post_ops_labels[] =
		{
			&&POST_OPS_6x16_DISABLE,
			&&POST_OPS_BIAS_6x16,
			&&POST_OPS_RELU_6x16,
			&&POST_OPS_RELU_SCALE_6x16,
			&&POST_OPS_DOWNSCALE_6x16
		};

	dim_t m_full_pieces = m0 / MR;
	dim_t m_full_pieces_loop_limit = m_full_pieces * MR;
	dim_t m_partial_pieces = m0 % MR;

	// The division is done by considering the vpmaddubsw instruction
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	// B matrix storage.
	__m256i b0;

	// A matrix storage.
	__m256i a_int32_0;
	__m256i inter_vec;

	for (dim_t ir = 0; ir < m_full_pieces_loop_limit; ir += MR)
	{
		//  Registers to use for accumulating C.
		__m256i c_int16_0p0 = _mm256_setzero_si256();

		__m256i c_int16_1p0 = _mm256_setzero_si256();

		__m256i c_int16_2p0 = _mm256_setzero_si256();

		__m256i c_int16_3p0 = _mm256_setzero_si256();

		__m256i c_int16_4p0 = _mm256_setzero_si256();

		__m256i c_int16_5p0 = _mm256_setzero_si256();

		for (dim_t kr = 0; kr < k_full_pieces; kr += 1)
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

			// Broadcast a[1,kr:kr+2].
			a_int32_0 = _mm256_set1_epi16(*(uint16_t *)(a + (rs_a * 1) + (cs_a * offset)));

			// Seperate register for intermediate op
			inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

			// Perform column direction mat-mul with k = 2.
			// c[1,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
			c_int16_1p0 = _mm256_add_epi16(inter_vec, c_int16_1p0);

			// Broadcast a[2,kr:kr+2].
			a_int32_0 = _mm256_set1_epi16(*(uint16_t *)(a + (rs_a * 2) + (cs_a * offset)));

			// Seperate register for intermediate op
			inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

			// Perform column direction mat-mul with k = 2.
			// c[2,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
			c_int16_2p0 = _mm256_add_epi16(inter_vec, c_int16_2p0);

			// Broadcast a[3,kr:kr+2].
			a_int32_0 = _mm256_set1_epi16(*(uint16_t *)(a + (rs_a * 3) + (cs_a * offset)));

			// Seperate register for intermediate op
			inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

			// Perform column direction mat-mul with k = 2.
			// c[3,0-15] = a[3,kr:kr+2]*b[kr:kr+2,0-31]
			c_int16_3p0 = _mm256_add_epi16(inter_vec, c_int16_3p0);

			// Broadcast a[4,kr:kr+2].
			a_int32_0 = _mm256_set1_epi16(*(uint16_t *)(a + (rs_a * 4) + (cs_a * offset)));

			// Seperate register for intermediate op
			inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

			// Perform column direction mat-mul with k = 2.
			// c[4,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
			c_int16_4p0 = _mm256_add_epi16(inter_vec, c_int16_4p0);

			// Broadcast a[5,kr:kr+2].
			a_int32_0 = _mm256_set1_epi16(*(uint16_t *)(a + (rs_a * 5) + (cs_a * offset)));

			// Seperate register for intermediate op
			inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

			// Perform column direction mat-mul with k = 2.
			// c[5,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
			c_int16_5p0 = _mm256_add_epi16(inter_vec, c_int16_5p0);
		}

		// Handle k remainder.
		if (k_partial_pieces > 0)
		{
			uint8_t a_kfringe;

			b0 = _mm256_loadu_si256((__m256i const *)(b + (32 * k_full_pieces) + (NR * 0)));

			a_kfringe = *(a + (rs_a * 0) + (cs_a * (k_full_pieces * 2)));
			a_int32_0 = _mm256_set1_epi8(a_kfringe);

			// Seperate register for intermediate op
			inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

			// Perform column direction mat-mul with k = 2.
			// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
			c_int16_0p0 = _mm256_add_epi16(inter_vec, c_int16_0p0);

			a_kfringe = *(a + (rs_a * 1) + (cs_a * (k_full_pieces * 2)));
			a_int32_0 = _mm256_set1_epi8(a_kfringe);

			// Seperate register for intermediate op
			inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

			// Perform column direction mat-mul with k = 2.
			// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-31]
			c_int16_1p0 = _mm256_add_epi16(inter_vec, c_int16_1p0);

			a_kfringe = *(a + (rs_a * 2) + (cs_a * (k_full_pieces * 2)));
			a_int32_0 = _mm256_set1_epi8(a_kfringe);

			// Seperate register for intermediate op
			inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

			// Perform column direction mat-mul with k = 2.
			// c[2,0-15] = a[2,kr:kr+2]*b[kr:kr+2,0-31]
			c_int16_2p0 = _mm256_add_epi16(inter_vec, c_int16_2p0);

			a_kfringe = *(a + (rs_a * 3) + (cs_a * (k_full_pieces * 2)));
			a_int32_0 = _mm256_set1_epi8(a_kfringe);

			// Seperate register for intermediate op
			inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

			// Perform column direction mat-mul with k = 2.
			// c[3,0-15] = a[3,kr:kr+2]*b[kr:kr+2,0-31]
			c_int16_3p0 = _mm256_add_epi16(inter_vec, c_int16_3p0);

			a_kfringe = *(a + (rs_a * 4) + (cs_a * (k_full_pieces * 2)));
			a_int32_0 = _mm256_set1_epi8(a_kfringe);

			// Seperate register for intermediate op
			inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

			// Perform column direction mat-mul with k = 2.
			// c[4,0-15] = a[4,kr:kr+2]*b[kr:kr+2,0-31]
			c_int16_4p0 = _mm256_add_epi16(inter_vec, c_int16_4p0);

			a_kfringe = *(a + (rs_a * 5) + (cs_a * (k_full_pieces * 2)));
			a_int32_0 = _mm256_set1_epi8(a_kfringe);

			// Seperate register for intermediate op
			inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

			// Perform column direction mat-mul with k = 2.
			// c[5,0-15] = a[5,kr:kr+2]*b[kr:kr+2,0-31]
			c_int16_5p0 = _mm256_add_epi16(inter_vec, c_int16_5p0);
		}

		// Load alpha and beta
		__m256i selector1 = _mm256_set1_epi16(alpha);
		__m256i selector2 = _mm256_set1_epi16(beta);

		// Scale by alpha
		c_int16_0p0 = _mm256_mullo_epi16(selector1, c_int16_0p0);

		c_int16_1p0 = _mm256_mullo_epi16(selector1, c_int16_1p0);

		c_int16_2p0 = _mm256_mullo_epi16(selector1, c_int16_2p0);

		c_int16_3p0 = _mm256_mullo_epi16(selector1, c_int16_3p0);

		c_int16_4p0 = _mm256_mullo_epi16(selector1, c_int16_4p0);

		c_int16_5p0 = _mm256_mullo_epi16(selector1, c_int16_5p0);

		// Scale C by beta.
		if (beta != 0)
		{
			// c[0,0-15]
			selector1 = _mm256_loadu_si256((__m256i const *)(c + (rs_c * (ir + 0)) + (0 * 16)));
			selector1 = _mm256_mullo_epi16(selector2, selector1);
			c_int16_0p0 = _mm256_add_epi16(selector1, c_int16_0p0);

			// c[1,0-15]
			selector1 = _mm256_loadu_si256((__m256i const *)(c + (rs_c * (ir + 1)) + (0 * 16)));
			selector1 = _mm256_mullo_epi16(selector2, selector1);
			c_int16_1p0 = _mm256_add_epi16(selector1, c_int16_1p0);

			// c[2,0-15]
			selector1 = _mm256_loadu_si256((__m256i const *)(c + (rs_c * (ir + 2)) + (0 * 16)));
			selector1 = _mm256_mullo_epi16(selector2, selector1);
			c_int16_2p0 = _mm256_add_epi16(selector1, c_int16_2p0);

			// c[3,0-15]
			selector1 = _mm256_loadu_si256((__m256i const *)(c + (rs_c * (ir + 3)) + (0 * 16)));
			selector1 = _mm256_mullo_epi16(selector2, selector1);
			c_int16_3p0 = _mm256_add_epi16(selector1, c_int16_3p0);

			// c[4,0-15]
			selector1 = _mm256_loadu_si256((__m256i const *)(c + (rs_c * (ir + 4)) + (0 * 16)));
			selector1 = _mm256_mullo_epi16(selector2, selector1);
			c_int16_4p0 = _mm256_add_epi16(selector1, c_int16_4p0);

			// c[5,0-15]
			selector1 = _mm256_loadu_si256((__m256i const *)(c + (rs_c * (ir + 5)) + (0 * 16)));
			selector1 = _mm256_mullo_epi16(selector2, selector1);
			c_int16_5p0 = _mm256_add_epi16(selector1, c_int16_5p0);
		}

		// Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_6x16:
		{
			selector1 =
				_mm256_loadu_si256( (__m256i const *)((int16_t *)post_ops_list_temp->op_args1 +
								post_op_c_j + ( 0 * 16 )) );
			
			// c[0,0-15]
			c_int16_0p0 = _mm256_add_epi16( selector1, c_int16_0p0 );

			// c[1,0-15]
			c_int16_1p0 = _mm256_add_epi16( selector1, c_int16_1p0 );

			// c[2,0-15]
			c_int16_2p0 = _mm256_add_epi16( selector1, c_int16_2p0 );

			// c[3,0-15]
			c_int16_3p0 = _mm256_add_epi16( selector1, c_int16_3p0 );

			// c[4,0-15]
			c_int16_4p0 = _mm256_add_epi16( selector1, c_int16_4p0 );

			// c[5,0-15]
			c_int16_5p0 = _mm256_add_epi16( selector1, c_int16_5p0 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_6x16:
		{
			selector1 = _mm256_setzero_si256 ();

			// c[0,0-15]
			c_int16_0p0 = _mm256_max_epi16( selector1, c_int16_0p0 );

			// c[1,0-15]
			c_int16_1p0 = _mm256_max_epi16( selector1, c_int16_1p0 );

			// c[2,0-15]
			c_int16_2p0 = _mm256_max_epi16( selector1, c_int16_2p0 );

			// c[3,0-15]
			c_int16_3p0 = _mm256_max_epi16( selector1, c_int16_3p0 );

			// c[4,0-15]
			c_int16_4p0 = _mm256_max_epi16( selector1, c_int16_4p0 );

			// c[5,0-15]
			c_int16_5p0 = _mm256_max_epi16( selector1, c_int16_5p0 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_6x16:
		{
			selector2 =
				_mm256_set1_epi16( *( ( int16_t* )post_ops_list_temp->op_args2 ) );

			// c[0,0-15]
			RELU_SCALE_OP_S16_AVX2(c_int16_0p0)

			// c[1,0-15]
			RELU_SCALE_OP_S16_AVX2(c_int16_1p0)

			// c[2,0-15]
			RELU_SCALE_OP_S16_AVX2(c_int16_2p0)

			// c[3,0-15]
			RELU_SCALE_OP_S16_AVX2(c_int16_3p0)

			// c[4,0-15]
			RELU_SCALE_OP_S16_AVX2(c_int16_4p0)

			// c[5,0-15]
			RELU_SCALE_OP_S16_AVX2(c_int16_5p0)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_DOWNSCALE_6x16:
		{
			__m128i temp[2];
			__m256i temp_32[2];
			__m256 temp_float[2];
			__m256 scale_1, scale_2;
			__m256 res_1, res_2;
			__m256i store_reg;

			/* Load the scale vector values into the register*/
			scale_1 =
				_mm256_loadu_ps(
				(float *)post_ops_list_temp->scale_factor +
				post_op_c_j + (0 * 8));
			scale_2 =
				_mm256_loadu_ps(
				(float *)post_ops_list_temp->scale_factor +
				post_op_c_j + (1 * 8));

			BLI_MM256_S16_DOWNSCALE2(c_int16_0p0, c_int16_1p0, 0, 1);

			BLI_MM256_S16_DOWNSCALE2(c_int16_2p0, c_int16_3p0, 2, 3);

			BLI_MM256_S16_DOWNSCALE2(c_int16_4p0, c_int16_5p0, 4, 5);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_6x16_DISABLE:
		;

		// Store the results.
		// c[0,0-15]
		_mm256_storeu_si256( (__m256i *)(c + ( rs_c *  ( ir + 0 ) ) + ( 0 * 16 ) ), c_int16_0p0 );

		// c[1,0-15]
		_mm256_storeu_si256( (__m256i *)(c + ( rs_c * ( ir + 1 ) ) + ( 0 * 16 ) ), c_int16_1p0 );

		// c[2,0-15]
		_mm256_storeu_si256( (__m256i *)(c + ( rs_c * ( ir + 2 ) ) + ( 0 * 16 ) ), c_int16_2p0 );

		// c[3,0-15]
		_mm256_storeu_si256( (__m256i *)(c + ( rs_c * ( ir + 3 ) ) + ( 0 * 16 ) ), c_int16_3p0 );

		// c[4,0-15]
		_mm256_storeu_si256( (__m256i *)(c + ( rs_c * ( ir + 4 ) ) + ( 0 * 16 ) ), c_int16_4p0 );

		// c[5,0-15]
		_mm256_storeu_si256( (__m256i *)(c + ( rs_c * ( ir + 5 ) ) + ( 0 * 16 ) ), c_int16_5p0 );
		
		a = a + ( MR * ps_a );
		post_op_c_i += MR;
	}

	if (m_partial_pieces > 0)
	{
		dim_t m_partial4 = m_partial_pieces / 4;
		m_partial_pieces = m_partial_pieces % 4;

		dim_t m_partial2 = m_partial_pieces / 2;
		dim_t m_partial = m_partial_pieces % 2;

		if (m_partial4 == 1)
		{
			lpgemm_rowvar_u8s8s16o16_4x16(
				k0,
				a, rs_a, cs_a,
				b, rs_b, cs_b,
				(c + (rs_c * m_full_pieces_loop_limit)), rs_c,
				alpha, beta,
				is_last_k,
				post_op_c_i, post_op_c_j,
				post_ops_list, rs_c_downscale);

			// a pointer increment
			a = a + (4 * ps_a);
			m_full_pieces_loop_limit += 4;
			post_op_c_i += 4;
		}

		if (m_partial2 == 1)
		{
			lpgemm_rowvar_u8s8s16o16_2x16(
				k0,
				a, rs_a, cs_a,
				b, rs_b, cs_b,
				(c + (rs_c * m_full_pieces_loop_limit)), rs_c,
				alpha, beta,
				is_last_k,
				post_op_c_i, post_op_c_j,
				post_ops_list, rs_c_downscale);

			// a pointer increment
			a = a + (2 * ps_a);
			m_full_pieces_loop_limit += 2;
			post_op_c_i += 2;
		}

		if (m_partial == 1)
		{
			lpgemm_rowvar_u8s8s16o16_1x16(
				k0,
				a, rs_a, cs_a,
				b, rs_b, cs_b,
				(c + (rs_c * m_full_pieces_loop_limit)), rs_c,
				alpha, beta,
				is_last_k,
				post_op_c_i, post_op_c_j,
				post_ops_list, rs_c_downscale);
			post_op_c_i += 1;
		}
	}
}

// 6xlt16 int8o16 kernel
LPGEMM_N_LT_NR0_FRINGE_KERN(uint8_t,int8_t,int16_t,u8s8s16o16_6xlt16)
{
	dim_t MR = 6;

	static void *post_ops_labels[] =
		{
			&&POST_OPS_6xlt16_DISABLE,
			&&POST_OPS_BIAS_6xlt16,
			&&POST_OPS_RELU_6xlt16,
			&&POST_OPS_RELU_SCALE_6xlt16,
			&&POST_OPS_DOWNSCALE_6xlt16
		};

	dim_t m_full_pieces = m0 / MR;
	dim_t m_full_pieces_loop_limit = m_full_pieces * MR;
	dim_t m_partial_pieces = m0 % MR;

	// The division is done by considering the vpmaddubsw instruction
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int16_t buf0[16];
	int16_t buf1[16];
	int16_t buf2[16];
	int16_t buf3[16];
	int16_t buf4[16];
	int16_t buf5[16];

	// B matrix storage.
	__m256i b0;

	// A matrix storage.
	__m256i a_int32_0;
	__m256i inter_vec;

	for (dim_t ir = 0; ir < m_full_pieces_loop_limit; ir += MR)
	{
		//  Registers to use for accumulating C.
		__m256i c_int16_0p0 = _mm256_setzero_si256();

		__m256i c_int16_1p0 = _mm256_setzero_si256();

		__m256i c_int16_2p0 = _mm256_setzero_si256();

		__m256i c_int16_3p0 = _mm256_setzero_si256();

		__m256i c_int16_4p0 = _mm256_setzero_si256();

		__m256i c_int16_5p0 = _mm256_setzero_si256();

		for (dim_t kr = 0; kr < k_full_pieces; kr += 1)
		{
			dim_t offset = kr * 2;

			b0 = _mm256_loadu_si256((__m256i const *)(b + (32 * kr) + (cs_b * 0)));

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

			// Broadcast a[4,kr:kr+2].
			a_int32_0 = _mm256_set1_epi16(*(uint16_t *)(a + (rs_a * 4) + (cs_a * offset)));

			// Seperate register for intermediate op
			inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

			// Perform column direction mat-mul with k = 2.
			// c[4,0-15] = a[4,kr:kr+2]*b[kr:kr+4,0-31]
			c_int16_4p0 = _mm256_add_epi16(inter_vec, c_int16_4p0);

			// Broadcast a[5,kr:kr+4].
			a_int32_0 = _mm256_set1_epi16(*(uint16_t *)(a + (rs_a * 5) + (cs_a * offset)));

			// Seperate register for intermediate op
			inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

			// Perform column direction mat-mul with k = 2.
			// c[5,0-15] = a[5,kr:kr+2]*b[kr:kr+2,0-31]
			c_int16_5p0 = _mm256_add_epi16(inter_vec, c_int16_5p0);
		}

		// Handle k remainder.
		if (k_partial_pieces > 0)
		{
			uint8_t a_kfringe;

			b0 = _mm256_loadu_si256((__m256i const *)(b + (32 * k_full_pieces) + (cs_b * 0)));

			a_kfringe = *(a + (rs_a * 0) + (cs_a * (k_full_pieces * 2)));
			a_int32_0 = _mm256_set1_epi8(a_kfringe);

			inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

			// Perform column direction mat-mul with k = 2.
			// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
			c_int16_0p0 = _mm256_add_epi16(inter_vec, c_int16_0p0);

			a_kfringe = *(a + (rs_a * 1) + (cs_a * (k_full_pieces * 2)));
			a_int32_0 = _mm256_set1_epi8(a_kfringe);

			// Seperate register for intermediate op
			inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

			// Perform column direction mat-mul with k = 2.
			// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-31]
			c_int16_1p0 = _mm256_add_epi16(inter_vec, c_int16_1p0);

			a_kfringe = *(a + (rs_a * 2) + (cs_a * (k_full_pieces * 2)));
			a_int32_0 = _mm256_set1_epi8(a_kfringe);

			// Seperate register for intermediate op
			inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

			// Perform column direction mat-mul with k = 2.
			// c[2,0-15] = a[2,kr:kr+2]*b[kr:kr+2,0-31]
			c_int16_2p0 = _mm256_add_epi16(inter_vec, c_int16_2p0);

			a_kfringe = *(a + (rs_a * 3) + (cs_a * (k_full_pieces * 2)));
			a_int32_0 = _mm256_set1_epi8(a_kfringe);

			// Seperate register for intermediate op
			inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

			// Perform column direction mat-mul with k = 2.
			// c[3,0-15] = a[3,kr:kr+2]*b[kr:kr+2,0-31]
			c_int16_3p0 = _mm256_add_epi16(inter_vec, c_int16_3p0);

			a_kfringe = *(a + (rs_a * 4) + (cs_a * (k_full_pieces * 2)));
			a_int32_0 = _mm256_set1_epi8(a_kfringe);

			// Seperate register for intermediate op
			inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

			// Perform column direction mat-mul with k = 2.
			// c[4,0-15] = a[4,kr:kr+2]*b[kr:kr+2,0-31]
			c_int16_4p0 = _mm256_add_epi16(inter_vec, c_int16_4p0);

			a_kfringe = *(a + (rs_a * 5) + (cs_a * (k_full_pieces * 2)));
			a_int32_0 = _mm256_set1_epi8(a_kfringe);

			// Seperate register for intermediate op
			inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

			// Perform column direction mat-mul with k = 2.
			// c[5,0-15] = a[5,kr:kr+2]*b[kr:kr+2,0-31]
			c_int16_5p0 = _mm256_add_epi16(inter_vec, c_int16_5p0);
		}

		// Load alpha and beta
		__m256i selector1 = _mm256_set1_epi16(alpha);
		__m256i selector2 = _mm256_set1_epi16(beta);

		// Scale by alpha
		c_int16_0p0 = _mm256_mullo_epi16(selector1, c_int16_0p0);

		c_int16_1p0 = _mm256_mullo_epi16(selector1, c_int16_1p0);

		c_int16_2p0 = _mm256_mullo_epi16(selector1, c_int16_2p0);

		c_int16_3p0 = _mm256_mullo_epi16(selector1, c_int16_3p0);

		c_int16_4p0 = _mm256_mullo_epi16(selector1, c_int16_4p0);

		c_int16_5p0 = _mm256_mullo_epi16(selector1, c_int16_5p0);

		// Scale C by beta.
		if (beta != 0)
		{
			memcpy(buf0, (c + (rs_c * (ir + 0))), (n0_rem * sizeof(int16_t)));
			memcpy(buf1, (c + (rs_c * (ir + 1))), (n0_rem * sizeof(int16_t)));
			memcpy(buf2, (c + (rs_c * (ir + 2))), (n0_rem * sizeof(int16_t)));
			memcpy(buf3, (c + (rs_c * (ir + 3))), (n0_rem * sizeof(int16_t)));
			memcpy(buf4, (c + (rs_c * (ir + 4))), (n0_rem * sizeof(int16_t)));
			memcpy(buf5, (c + (rs_c * (ir + 5))), (n0_rem * sizeof(int16_t)));

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

			// c[4,0-15]
			selector1 = _mm256_loadu_si256((__m256i const *)buf4);
			selector1 = _mm256_mullo_epi16(selector2, selector1);
			c_int16_4p0 = _mm256_add_epi16(selector1, c_int16_4p0);

			// c[5,0-15]
			selector1 = _mm256_loadu_si256((__m256i const *)buf5);
			selector1 = _mm256_mullo_epi16(selector2, selector1);
			c_int16_5p0 = _mm256_add_epi16(selector1, c_int16_5p0);
		}

		// Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_6xlt16:
		{
			memcpy( buf0, ( ( int16_t* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 0 * 16 ) ),
							( n0_rem * sizeof( int16_t ) ) );

			selector1 =
				_mm256_loadu_si256( ( __m256i const* )buf0 );
			
			// c[0,0-15]
			c_int16_0p0 = _mm256_add_epi16( selector1, c_int16_0p0 );

			// c[1,0-15]
			c_int16_1p0 = _mm256_add_epi16( selector1, c_int16_1p0 );

			// c[2,0-15]
			c_int16_2p0 = _mm256_add_epi16( selector1, c_int16_2p0 );

			// c[3,0-15]
			c_int16_3p0 = _mm256_add_epi16( selector1, c_int16_3p0 );

			// c[4,0-15]
			c_int16_4p0 = _mm256_add_epi16( selector1, c_int16_4p0 );

			// c[5,0-15]
			c_int16_5p0 = _mm256_add_epi16( selector1, c_int16_5p0 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_6xlt16:
		{
			selector1 = _mm256_setzero_si256 ();

			// c[0,0-15]
			c_int16_0p0 = _mm256_max_epi16( selector1, c_int16_0p0 );

			// c[1,0-15]
			c_int16_1p0 = _mm256_max_epi16( selector1, c_int16_1p0 );

			// c[2,0-15]
			c_int16_2p0 = _mm256_max_epi16( selector1, c_int16_2p0 );

			// c[3,0-15]
			c_int16_3p0 = _mm256_max_epi16( selector1, c_int16_3p0 );

			// c[4,0-15]
			c_int16_4p0 = _mm256_max_epi16( selector1, c_int16_4p0 );

			// c[5,0-15]
			c_int16_5p0 = _mm256_max_epi16( selector1, c_int16_5p0 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_6xlt16:
		{
			selector2 =
				_mm256_set1_epi16( *( ( int16_t* )post_ops_list_temp->op_args2 ) );

			// c[0,0-15]
			RELU_SCALE_OP_S16_AVX2(c_int16_0p0)

			// c[1,0-15]
			RELU_SCALE_OP_S16_AVX2(c_int16_1p0)

			// c[2,0-15]
			RELU_SCALE_OP_S16_AVX2(c_int16_2p0)

			// c[3,0-15]
			RELU_SCALE_OP_S16_AVX2(c_int16_3p0)

			// c[4,0-15]
			RELU_SCALE_OP_S16_AVX2(c_int16_4p0)

			// c[5,0-15]
			RELU_SCALE_OP_S16_AVX2(c_int16_5p0)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_DOWNSCALE_6xlt16:
		{
			__m128i temp[2];
			__m256i temp_32[2];
			__m256 temp_float[2];
			__m256 scale_1, scale_2;
			__m256 res_1, res_2;
			__m256i store_reg;

			float float_buf[16];
			int8_t store_buf[16];

			memcpy( float_buf, ( ( float* )post_ops_list_temp->scale_factor +
					post_op_c_j ), ( n0_rem * sizeof( float ) ) );

			// Load the scale vector values into the register
			scale_1 = _mm256_loadu_ps(float_buf + (0 * 8));
			scale_2 = _mm256_loadu_ps(float_buf + (1 * 8));

			BLI_MM256_S16_DOWNSCALE2_LT16(c_int16_0p0, c_int16_1p0, 0, 1)

			BLI_MM256_S16_DOWNSCALE2_LT16(c_int16_2p0, c_int16_3p0, 2, 3)

			BLI_MM256_S16_DOWNSCALE2_LT16(c_int16_4p0, c_int16_5p0, 4, 5)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_6xlt16_DISABLE:
		;

		// Store the results.
		// c[0,0-15]
		_mm256_storeu_si256((__m256i *)buf0, c_int16_0p0);

		// c[1,0-15]
		_mm256_storeu_si256((__m256i *)buf1, c_int16_1p0);

		// c[2,0-15]
		_mm256_storeu_si256((__m256i *)buf2, c_int16_2p0);

		// c[3,0-15]
		_mm256_storeu_si256((__m256i *)buf3, c_int16_3p0);

		// c[4,0-15]
		_mm256_storeu_si256((__m256i *)buf4, c_int16_4p0);

		// c[5,0-15]
		_mm256_storeu_si256((__m256i *)buf5, c_int16_5p0);

		memcpy(c + (rs_c * (ir + 0)) + (0 * 16), buf0, (n0_rem * sizeof(int16_t)));

		// c[1,0-15]
		memcpy(c + (rs_c * (ir + 1)) + (0 * 16), buf1, (n0_rem * sizeof(int16_t)));

		// c[2,0-15]
		memcpy(c + (rs_c * (ir + 2)) + (0 * 16), buf2, (n0_rem * sizeof(int16_t)));

		// c[3,0-15]
		memcpy(c + (rs_c * (ir + 3)) + (0 * 16), buf3, (n0_rem * sizeof(int16_t)));

		// c[4,0-15]
		memcpy(c + (rs_c * (ir + 4)) + (0 * 16), buf4, (n0_rem * sizeof(int16_t)));

		// c[5,0-15]
		memcpy(c + (rs_c * (ir + 5)) + (0 * 16), buf5, (n0_rem * sizeof(int16_t)));

		a = a + (MR * ps_a);
		post_op_c_i += MR;
	}

	if (m_partial_pieces > 0)
	{
		dim_t m_partial4 = m_partial_pieces / 4;
		m_partial_pieces = m_partial_pieces % 4;

		dim_t m_partial2 = m_partial_pieces / 2;
		dim_t m_partial = m_partial_pieces % 2;

		if (m_partial4 == 1)
		{
			lpgemm_rowvar_u8s8s16o16_4xlt16(
				k0,
				a, rs_a, cs_a,
				b, rs_b, cs_b,
				(c + (rs_c * m_full_pieces_loop_limit)), rs_c,
				alpha, beta, n0_rem,
				is_last_k,
				post_op_c_i, post_op_c_j,
				post_ops_list, rs_c_downscale);

			// a pointer increment
			a = a + (4 * ps_a);
			m_full_pieces_loop_limit += 4;
			post_op_c_i += 4;
		}

		if (m_partial2 == 1)
		{
			lpgemm_rowvar_u8s8s16o16_2xlt16(
				k0,
				a, rs_a, cs_a,
				b, rs_b, cs_b,
				(c + (rs_c * m_full_pieces_loop_limit)), rs_c,
				alpha, beta, n0_rem,
				is_last_k,
				post_op_c_i, post_op_c_j,
				post_ops_list, rs_c_downscale);

			// a pointer increment
			a = a + (2 * ps_a);
			m_full_pieces_loop_limit += 2;
			post_op_c_i += 2;
		}

		if (m_partial == 1)
		{
			lpgemm_rowvar_u8s8s16o16_1xlt16(
				k0,
				a, rs_a, cs_a,
				b, rs_b, cs_b,
				(c + (rs_c * m_full_pieces_loop_limit)), rs_c,
				alpha, beta, n0_rem,
				is_last_k,
				post_op_c_i, post_op_c_j,
				post_ops_list, rs_c_downscale);
			post_op_c_i += 1;
		}
	}
}
