/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022-23, Advanced Micro Devices, Inc. All rights reserved.

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

#ifdef BLIS_ADDON_LPGEMM

#include "lpgemm_s16_kern_macros.h"

// 4x32 int8o16 kernel
LPGEMM_MN_FRINGE_KERN(uint8_t,int8_t,int16_t,u8s8s16o16_4x16)
{
	dim_t NR = 16;

	static void *post_ops_labels[] =
		{
			&&POST_OPS_4x16_DISABLE,
			&&POST_OPS_BIAS_4x16,
			&&POST_OPS_RELU_4x16,
			&&POST_OPS_RELU_SCALE_4x16,
			&&POST_OPS_GELU_TANH_4x16,
			&&POST_OPS_GELU_ERF_4x16,
			&&POST_OPS_CLIP_4x16,
			&&POST_OPS_DOWNSCALE_4x16
		};

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
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_2p0 = _mm256_add_epi16(inter_vec, c_int16_2p0);

		a_kfringe = *(a + (rs_a * 3) + (cs_a * (k_full_pieces * 2)));
		a_int32_0 = _mm256_set1_epi8(a_kfringe);

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_3p0 = _mm256_add_epi16(inter_vec, c_int16_3p0);
	}

	// Load alpha and beta
	__m256i selector1 = _mm256_set1_epi16(alpha);
	__m256i selector2 = _mm256_set1_epi16(beta);

	if ( alpha != 1 )
	{
		// Scale by alpha
		c_int16_0p0 = _mm256_mullo_epi16(selector1, c_int16_0p0);

		c_int16_1p0 = _mm256_mullo_epi16(selector1, c_int16_1p0);

		c_int16_2p0 = _mm256_mullo_epi16(selector1, c_int16_2p0);

		c_int16_3p0 = _mm256_mullo_epi16(selector1, c_int16_3p0);
	}

	// Scale C by beta.
	if (beta != 0)
	{
		// For the downscaled api (C-s8), the output C matrix values
		// needs to be upscaled to s16 to be used for beta scale.
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
			 ( post_ops_attr.is_first_k == TRUE ) )
		{
			// c[0,0-15]
			S8_S16_BETA_OP(c_int16_0p0,0,0,0,selector1,selector2)

			// c[1,0-15]
			S8_S16_BETA_OP(c_int16_1p0,0,1,0,selector1,selector2)

			// c[2,0-15]
			S8_S16_BETA_OP(c_int16_2p0,0,2,0,selector1,selector2)

			// c[3,0-15]
			S8_S16_BETA_OP(c_int16_3p0,0,3,0,selector1,selector2)
		}
		else
		{
			// c[0,0-15]
			S16_S16_BETA_OP(c_int16_0p0,0,0,0,selector1,selector2)

			// c[1,0-15]
			S16_S16_BETA_OP(c_int16_1p0,0,1,0,selector1,selector2)

			// c[2,0-15]
			S16_S16_BETA_OP(c_int16_2p0,0,2,0,selector1,selector2)

			// c[3,0-15]
			S16_S16_BETA_OP(c_int16_3p0,0,3,0,selector1,selector2)
		}
	}

	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_4x16:
	{
		selector1 =
			_mm256_loadu_si256( (__m256i const *)((int16_t *)post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 0 * 16 )) );

		// c[0,0-15]
		c_int16_0p0 = _mm256_add_epi16( selector1, c_int16_0p0 );

		// c[1,0-15]
		c_int16_1p0 = _mm256_add_epi16( selector1, c_int16_1p0 );

		// c[2,0-15]
		c_int16_2p0 = _mm256_add_epi16( selector1, c_int16_2p0 );

		// c[3,0-15]
		c_int16_3p0 = _mm256_add_epi16( selector1, c_int16_3p0 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_4x16:
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

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_4x16:
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

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_TANH_4x16:
	{
		__m256 dn, z, x, r2, r, y1, y2, x_tanh;
		__m256i q;

		// c[0,0-15]
		GELU_TANH_S16_AVX2(c_int16_0p0, y1, y2, r, r2, x, z, dn, x_tanh, q)

		// c[1,0-15]
		GELU_TANH_S16_AVX2(c_int16_1p0, y1, y2, r, r2, x, z, dn, x_tanh, q)

		// c[2,0-15]
		GELU_TANH_S16_AVX2(c_int16_2p0, y1, y2, r, r2, x, z, dn, x_tanh, q)

		// c[3,0-15]
		GELU_TANH_S16_AVX2(c_int16_3p0, y1, y2, r, r2, x, z, dn, x_tanh, q)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_ERF_4x16:
	{
		__m256 x, r, y1, y2, x_erf;

		// c[0,0-15]
		GELU_ERF_S16_AVX2(c_int16_0p0, y1, y2, r, x, x_erf)

		// c[1,0-15]
		GELU_ERF_S16_AVX2(c_int16_1p0, y1, y2, r, x, x_erf)

		// c[2,0-15]
		GELU_ERF_S16_AVX2(c_int16_2p0, y1, y2, r, x, x_erf)

		// c[3,0-15]
		GELU_ERF_S16_AVX2(c_int16_3p0, y1, y2, r, x, x_erf)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_CLIP_4x16:
	{
		__m256i min = _mm256_set1_epi16( *( int16_t* )post_ops_list_temp->op_args2 );
		__m256i max = _mm256_set1_epi16( *( int16_t* )post_ops_list_temp->op_args3 );

		// c[0,0-15]
		CLIP_S16_AVX2(c_int16_0p0, min, max)

		// c[1,0-15]
		CLIP_S16_AVX2(c_int16_1p0, min, max)

		// c[2,0-15]
		CLIP_S16_AVX2(c_int16_2p0, min, max)

		// c[3,0-15]
		CLIP_S16_AVX2(c_int16_3p0, min, max)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}

POST_OPS_DOWNSCALE_4x16:
	{
		__m128i temp[2];
		__m256i temp_32[2];
		__m256 temp_float[2];
		__m256 scale_1, scale_2;
		__m256 res_1, res_2;

		/* Load the scale vector values into the register*/
		scale_1 =
			_mm256_loadu_ps(
			(float *)post_ops_list_temp->scale_factor +
			post_ops_attr.post_op_c_j + (0 * 8));
		scale_2 =
			_mm256_loadu_ps(
			(float *)post_ops_list_temp->scale_factor +
			post_ops_attr.post_op_c_j + (1 * 8));

		// Scale first 16 columns of the 4 rows.
		CVT_MULRND_CVT16(c_int16_0p0, scale_1, scale_2)
		CVT_MULRND_CVT16(c_int16_1p0, scale_1, scale_2)
		CVT_MULRND_CVT16(c_int16_2p0, scale_1, scale_2)
		CVT_MULRND_CVT16(c_int16_3p0, scale_1, scale_2)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_4x16_DISABLE:
	;

	// Case where the output C matrix is s8 (downscaled) and this is the
	// final write for a given block within C.
	if ( ( post_ops_attr.buf_downscale != NULL ) &&
		 ( post_ops_attr.is_last_k == TRUE ) )
	{
		// Store the results in downscaled type (int8 instead of int32).
		__m128i temp[2];

		// c[0-1,0-15]
		CVT_STORE_S16_S8_2ROW(c_int16_0p0, c_int16_1p0, 0, 1, 0);

		// c[2-3,0-15]
		CVT_STORE_S16_S8_2ROW(c_int16_2p0, c_int16_3p0, 2, 3, 0);
	}
	// Case where the output C matrix is s16 or is the temp buffer used to
	// store intermediate s16 accumulated values for downscaled (C-s8) api.
	else
	{
		// Store the results.
		// c[0,0-15]
		_mm256_storeu_si256( (__m256i *)(c + ( rs_c * 0 ) + ( 0 * 16 ) ), c_int16_0p0 );

		// c[1,0-15]
		_mm256_storeu_si256( (__m256i *)(c + ( rs_c * 1 ) + ( 0 * 16 ) ), c_int16_1p0 );

		// c[2,0-15]
		_mm256_storeu_si256( (__m256i *)(c + ( rs_c * 2 ) + ( 0 * 16 ) ), c_int16_2p0 );

		// c[3,0-15]
		_mm256_storeu_si256( (__m256i *)(c + ( rs_c * 3 ) + ( 0 * 16 ) ), c_int16_3p0 );
	}
}

// 4x16 int8o16 kernel
LPGEMM_MN_LT_NR0_FRINGE_KERN(uint8_t,int8_t,int16_t,u8s8s16o16_4xlt16)
{
	dim_t NR = 16;

	static void *post_ops_labels[] =
		{
			&&POST_OPS_4xlt16_DISABLE,
			&&POST_OPS_BIAS_4xlt16,
			&&POST_OPS_RELU_4xlt16,
			&&POST_OPS_RELU_SCALE_4xlt16,
			&&POST_OPS_GELU_TANH_4xlt16,
			&&POST_OPS_GELU_ERF_4xlt16,
			&&POST_OPS_CLIP_4xlt16,
			&&POST_OPS_DOWNSCALE_4xlt16
		};

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
		// c[2,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_2p0 = _mm256_add_epi16(inter_vec, c_int16_2p0);

		a_kfringe = *(a + (rs_a * 3) + (cs_a * (k_full_pieces * 2)));
		a_int32_0 = _mm256_set1_epi8(a_kfringe);

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_3p0 = _mm256_add_epi16(inter_vec, c_int16_3p0);
	}

	// Load alpha and beta
	__m256i selector1 = _mm256_set1_epi16(alpha);
	__m256i selector2 = _mm256_set1_epi16(beta);

	if ( alpha != 1 )
	{
		// Scale by alpha
		c_int16_0p0 = _mm256_mullo_epi16(selector1, c_int16_0p0);

		c_int16_1p0 = _mm256_mullo_epi16(selector1, c_int16_1p0);

		c_int16_2p0 = _mm256_mullo_epi16(selector1, c_int16_2p0);

		c_int16_3p0 = _mm256_mullo_epi16(selector1, c_int16_3p0);
	}

	// Scale C by beta.
	if (beta != 0)
	{
		// For the downscaled api (C-s8), the output C matrix values
		// needs to be upscaled to s16 to be used for beta scale.
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
			 ( post_ops_attr.is_first_k == TRUE ) )
		{
			dim_t n0_rem_dscale_bytes = n0_rem * sizeof( int8_t );

			S8_S16_BETA_NLT16_MEMCP_UTIL(buf0, 0, n0_rem_dscale_bytes);
			S8_S16_BETA_NLT16_MEMCP_UTIL(buf1, 1, n0_rem_dscale_bytes);
			S8_S16_BETA_NLT16_MEMCP_UTIL(buf2, 2, n0_rem_dscale_bytes);
			S8_S16_BETA_NLT16_MEMCP_UTIL(buf3, 3, n0_rem_dscale_bytes);

			// c[0,0-15]
			S8_S16_BETA_OP_NLT16(c_int16_0p0,buf0,selector1,selector2)

			// c[1,0-15]
			S8_S16_BETA_OP_NLT16(c_int16_1p0,buf1,selector1,selector2)

			// c[2,0-15]
			S8_S16_BETA_OP_NLT16(c_int16_2p0,buf2,selector1,selector2)

			// c[3,0-15]
			S8_S16_BETA_OP_NLT16(c_int16_3p0,buf3,selector1,selector2)
		}
		else
		{
			dim_t n0_rem_bytes = n0_rem * sizeof( int16_t );

			memcpy( buf0, ( c + ( rs_c * 0 ) ), n0_rem_bytes );
			memcpy( buf1, ( c + ( rs_c * 1 ) ), n0_rem_bytes );
			memcpy( buf2, ( c + ( rs_c * 2 ) ), n0_rem_bytes );
			memcpy( buf3, ( c + ( rs_c * 3 ) ), n0_rem_bytes );

			// c[0,0-15]
			S16_S16_BETA_OP_NLT16(c_int16_0p0,buf0,selector1,selector2)

			// c[1,0-15]
			S16_S16_BETA_OP_NLT16(c_int16_1p0,buf1,selector1,selector2)

			// c[2,0-15]
			S16_S16_BETA_OP_NLT16(c_int16_2p0,buf2,selector1,selector2)

			// c[3,0-15]
			S16_S16_BETA_OP_NLT16(c_int16_3p0,buf3,selector1,selector2)
		}
	}

	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_4xlt16:
	{
		memcpy( buf0, ( ( int16_t* )post_ops_list_temp->op_args1
			+ post_ops_attr.post_op_c_j + ( 0 * 16 ) ), ( n0_rem * sizeof( int16_t ) ) );

		selector1 =
			_mm256_loadu_si256( (__m256i const *) buf0 );

		// c[0,0-15]
		c_int16_0p0 = _mm256_add_epi16( selector1, c_int16_0p0 );

		// c[1,0-15]
		c_int16_1p0 = _mm256_add_epi16( selector1, c_int16_1p0 );

		// c[2,0-15]
		c_int16_2p0 = _mm256_add_epi16( selector1, c_int16_2p0 );

		// c[3,0-15]
		c_int16_3p0 = _mm256_add_epi16( selector1, c_int16_3p0 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_4xlt16:
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

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_4xlt16:
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

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_TANH_4xlt16:
	{
		__m256 dn, z, x, r2, r, y1, y2, x_tanh;
		__m256i q;

		// c[0,0-15]
		GELU_TANH_S16_AVX2(c_int16_0p0, y1, y2, r, r2, x, z, dn, x_tanh, q)

		// c[1,0-15]
		GELU_TANH_S16_AVX2(c_int16_1p0, y1, y2, r, r2, x, z, dn, x_tanh, q)

		// c[2,0-15]
		GELU_TANH_S16_AVX2(c_int16_2p0, y1, y2, r, r2, x, z, dn, x_tanh, q)

		// c[3,0-15]
		GELU_TANH_S16_AVX2(c_int16_3p0, y1, y2, r, r2, x, z, dn, x_tanh, q)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_ERF_4xlt16:
	{
		__m256 x, r, y1, y2, x_erf;

		// c[0,0-15]
		GELU_ERF_S16_AVX2(c_int16_0p0, y1, y2, r, x, x_erf)

		// c[1,0-15]
		GELU_ERF_S16_AVX2(c_int16_1p0, y1, y2, r, x, x_erf)

		// c[2,0-15]
		GELU_ERF_S16_AVX2(c_int16_2p0, y1, y2, r, x, x_erf)

		// c[3,0-15]
		GELU_ERF_S16_AVX2(c_int16_3p0, y1, y2, r, x, x_erf)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_CLIP_4xlt16:
	{
		__m256i min = _mm256_set1_epi16( *( int16_t* )post_ops_list_temp->op_args2 );
		__m256i max = _mm256_set1_epi16( *( int16_t* )post_ops_list_temp->op_args3 );

		// c[0,0-15]
		CLIP_S16_AVX2(c_int16_0p0, min, max)

		// c[1,0-15]
		CLIP_S16_AVX2(c_int16_1p0, min, max)

		// c[2,0-15]
		CLIP_S16_AVX2(c_int16_2p0, min, max)

		// c[3,0-15]
		CLIP_S16_AVX2(c_int16_3p0, min, max)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}

POST_OPS_DOWNSCALE_4xlt16:
	{
		__m128i temp[2];
		__m256i temp_32[2];
		__m256 temp_float[2];
		__m256 scale_1, scale_2;
		__m256 res_1, res_2;

		float float_buf[16];

		memcpy( float_buf, ( ( float* )post_ops_list_temp->scale_factor +
				post_ops_attr.post_op_c_j ), ( n0_rem * sizeof( float ) ) );

		// Load the scale vector values into the register
		scale_1 = _mm256_loadu_ps(float_buf + (0 * 8));
		scale_2 = _mm256_loadu_ps(float_buf + (1 * 8));

		// Scale first 16 columns of the 6 rows.
		CVT_MULRND_CVT16(c_int16_0p0, scale_1, scale_2)
		CVT_MULRND_CVT16(c_int16_1p0, scale_1, scale_2)
		CVT_MULRND_CVT16(c_int16_2p0, scale_1, scale_2)
		CVT_MULRND_CVT16(c_int16_3p0, scale_1, scale_2)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_4xlt16_DISABLE:
	;

	// Case where the output C matrix is s8 (downscaled) and this is the
	// final write for a given block within C.
	if ( ( post_ops_attr.buf_downscale != NULL ) &&
		 ( post_ops_attr.is_last_k == TRUE ) )
	{
		// Store the results in downscaled type (int8 instead of int32).
		__m128i temp[2];

		// c[0-1,0-15]
		CVT_STORE_S16_S8_2ROW_NLT16(c_int16_0p0, c_int16_1p0, buf0, buf1);

		// c[2-3,0-15]
		CVT_STORE_S16_S8_2ROW_NLT16(c_int16_2p0, c_int16_3p0, buf2, buf3);

		dim_t n0_rem_dscale_bytes = n0_rem * sizeof( int8_t );

		CVT_STORE_S16_S8_NLT16_MEMCP_UTIL(buf0, 0, n0_rem_dscale_bytes);
		CVT_STORE_S16_S8_NLT16_MEMCP_UTIL(buf1, 1, n0_rem_dscale_bytes);
		CVT_STORE_S16_S8_NLT16_MEMCP_UTIL(buf2, 2, n0_rem_dscale_bytes);
		CVT_STORE_S16_S8_NLT16_MEMCP_UTIL(buf3, 3, n0_rem_dscale_bytes);
	}
	// Case where the output C matrix is s16 or is the temp buffer used to
	// store intermediate s16 accumulated values for downscaled (C-s8) api.
	else
	{
		// Store the results.
		// c[0,0-15]
		_mm256_storeu_si256( ( __m256i* )buf0, c_int16_0p0 );

		// c[1,0-15]
		_mm256_storeu_si256( ( __m256i* )buf1, c_int16_1p0 );

		// c[2,0-15]
		_mm256_storeu_si256( ( __m256i* )buf2, c_int16_2p0 );

		// c[3,0-15]
		_mm256_storeu_si256( ( __m256i* )buf3, c_int16_3p0 );

		dim_t n0_rem_bytes = n0_rem * sizeof( int16_t );

		memcpy( c + ( rs_c * 0 ) + ( 0 * 16 ), buf0, n0_rem_bytes );

		// c[1,0-15]
		memcpy( c + ( rs_c * 1 ) + ( 0 * 16 ), buf1, n0_rem_bytes );

		// c[2,0-15]
		memcpy( c + ( rs_c * 2 ) + ( 0 * 16 ), buf2, n0_rem_bytes );

		// c[3,0-15]
		memcpy( c + ( rs_c * 3 ) + ( 0 * 16 ), buf3, n0_rem_bytes );
	}
}

// 2x16 int8o16 kernel
LPGEMM_MN_FRINGE_KERN(uint8_t,int8_t,int16_t,u8s8s16o16_2x16)
{
	dim_t NR = 16;

	static void *post_ops_labels[] =
		{
			&&POST_OPS_2x16_DISABLE,
			&&POST_OPS_BIAS_2x16,
			&&POST_OPS_RELU_2x16,
			&&POST_OPS_RELU_SCALE_2x16,
			&&POST_OPS_GELU_TANH_2x16,
			&&POST_OPS_GELU_ERF_2x16,
			&&POST_OPS_CLIP_2x16,	
			&&POST_OPS_DOWNSCALE_2x16
		};

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
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_1p0 = _mm256_add_epi16(inter_vec, c_int16_1p0);
	}

	// Load alpha and beta
	__m256i selector1 = _mm256_set1_epi16(alpha);
	__m256i selector2 = _mm256_set1_epi16(beta);

	if ( alpha != 1 )
	{
		// Scale by alpha
		c_int16_0p0 = _mm256_mullo_epi16(selector1, c_int16_0p0);

		c_int16_1p0 = _mm256_mullo_epi16(selector1, c_int16_1p0);
	}

	// Scale C by beta.
	if (beta != 0)
	{
		// For the downscaled api (C-s8), the output C matrix values
		// needs to be upscaled to s16 to be used for beta scale.
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
			 ( post_ops_attr.is_first_k == TRUE ) )
		{
			// c[0,0-15]
			S8_S16_BETA_OP(c_int16_0p0,0,0,0,selector1,selector2)

			// c[1,0-15]
			S8_S16_BETA_OP(c_int16_1p0,0,1,0,selector1,selector2)
		}
		else
		{
			// c[0,0-15]
			S16_S16_BETA_OP(c_int16_0p0,0,0,0,selector1,selector2)

			// c[1,0-15]
			S16_S16_BETA_OP(c_int16_1p0,0,1,0,selector1,selector2)
		}
	}

	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_2x16:
	{
		selector1 =
			_mm256_loadu_si256( (__m256i const *)((int16_t *)post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 0 * 16 )) );

		// c[0,0-15]
		c_int16_0p0 = _mm256_add_epi16( selector1, c_int16_0p0 );

		// c[1,0-15]
		c_int16_1p0 = _mm256_add_epi16( selector1, c_int16_1p0 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_2x16:
	{
		selector1 = _mm256_setzero_si256 ();

		// c[0,0-15]
		c_int16_0p0 = _mm256_max_epi16( selector1, c_int16_0p0 );

		// c[1,0-15]
		c_int16_1p0 = _mm256_max_epi16( selector1, c_int16_1p0 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_2x16:
	{
		selector2 =
			_mm256_set1_epi16( *( ( int16_t* )post_ops_list_temp->op_args2 ) );

		// c[0,0-15]
		RELU_SCALE_OP_S16_AVX2(c_int16_0p0)

		// c[1,0-15]
		RELU_SCALE_OP_S16_AVX2(c_int16_1p0)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_TANH_2x16:
	{
		__m256 dn, z, x, r2, r, y1, y2, x_tanh;
		__m256i q;

		// c[0,0-15]
		GELU_TANH_S16_AVX2(c_int16_0p0, y1, y2, r, r2, x, z, dn, x_tanh, q)

		// c[1,0-15]
		GELU_TANH_S16_AVX2(c_int16_1p0, y1, y2, r, r2, x, z, dn, x_tanh, q)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_ERF_2x16:
	{
		__m256 x, r, y1, y2, x_erf;

		// c[0,0-15]
		GELU_ERF_S16_AVX2(c_int16_0p0, y1, y2, r, x, x_erf)

		// c[1,0-15]
		GELU_ERF_S16_AVX2(c_int16_1p0, y1, y2, r, x, x_erf)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_CLIP_2x16:
	{
		__m256i min = _mm256_set1_epi16( *( int16_t* )post_ops_list_temp->op_args2 );
		__m256i max = _mm256_set1_epi16( *( int16_t* )post_ops_list_temp->op_args3 );

		// c[0,0-15]
		CLIP_S16_AVX2(c_int16_0p0, min, max)

		// c[1,0-15]
		CLIP_S16_AVX2(c_int16_1p0, min, max)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}

POST_OPS_DOWNSCALE_2x16:
	{
		__m128i temp[2];
		__m256i temp_32[2];
		__m256 temp_float[2];
		__m256 scale_1, scale_2;
		__m256 res_1, res_2;

		/* Load the scale vector values into the register*/
		scale_1 =
			_mm256_loadu_ps(
			(float *)post_ops_list_temp->scale_factor +
			post_ops_attr.post_op_c_j + (0 * 8));
		scale_2 =
			_mm256_loadu_ps(
			(float *)post_ops_list_temp->scale_factor +
			post_ops_attr.post_op_c_j + (1 * 8));

		// Scale first 16 columns of the 2 rows.
		CVT_MULRND_CVT16(c_int16_0p0, scale_1, scale_2)
		CVT_MULRND_CVT16(c_int16_1p0, scale_1, scale_2)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_2x16_DISABLE:
	;

	// Case where the output C matrix is s8 (downscaled) and this is the
	// final write for a given block within C.
	if ( ( post_ops_attr.buf_downscale != NULL ) &&
		 ( post_ops_attr.is_last_k == TRUE ) )
	{
		// Store the results in downscaled type (int8 instead of int32).
		__m128i temp[2];

		// c[0-1,0-15]
		CVT_STORE_S16_S8_2ROW(c_int16_0p0, c_int16_1p0, 0, 1, 0);
	}
	// Case where the output C matrix is s16 or is the temp buffer used to
	// store intermediate s16 accumulated values for downscaled (C-s8) api.
	else
	{
		// Store the results.
		// c[0,0-15]
		_mm256_storeu_si256( (__m256i *)(c + ( rs_c * 0 ) + ( 0 * 16 ) ), c_int16_0p0 );

		// c[1,0-15]
		_mm256_storeu_si256( (__m256i *)(c + ( rs_c * 1 ) + ( 0 * 16 ) ), c_int16_1p0 );
	}
}

// 2xlt16 int8o16 kernel
LPGEMM_MN_LT_NR0_FRINGE_KERN(uint8_t,int8_t,int16_t,u8s8s16o16_2xlt16)
{
	dim_t NR = 16;

	static void *post_ops_labels[] =
		{
			&&POST_OPS_2xlt16_DISABLE,
			&&POST_OPS_BIAS_2xlt16,
			&&POST_OPS_RELU_2xlt16,
			&&POST_OPS_RELU_SCALE_2xlt16,
			&&POST_OPS_GELU_TANH_2xlt16,
			&&POST_OPS_GELU_ERF_2xlt16,
			&&POST_OPS_CLIP_2xlt16,
			&&POST_OPS_DOWNSCALE_2xlt16
		};

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
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_1p0 = _mm256_add_epi16(inter_vec, c_int16_1p0);
	}

	// Load alpha and beta
	__m256i selector1 = _mm256_set1_epi16(alpha);
	__m256i selector2 = _mm256_set1_epi16(beta);

	if ( alpha != 1 )
	{
		// Scale by alpha
		c_int16_0p0 = _mm256_mullo_epi16(selector1, c_int16_0p0);

		c_int16_1p0 = _mm256_mullo_epi16(selector1, c_int16_1p0);
	}

	// Scale C by beta.
	if (beta != 0)
	{
		// For the downscaled api (C-s8), the output C matrix values
		// needs to be upscaled to s16 to be used for beta scale.
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
			 ( post_ops_attr.is_first_k == TRUE ) )
		{
			dim_t n0_rem_dscale_bytes = n0_rem * sizeof( int8_t );

			S8_S16_BETA_NLT16_MEMCP_UTIL(buf0, 0, n0_rem_dscale_bytes);
			S8_S16_BETA_NLT16_MEMCP_UTIL(buf1, 1, n0_rem_dscale_bytes);

			// c[0,0-15]
			S8_S16_BETA_OP_NLT16(c_int16_0p0,buf0,selector1,selector2)

			// c[1,0-15]
			S8_S16_BETA_OP_NLT16(c_int16_1p0,buf1,selector1,selector2)
		}
		else
		{
			dim_t n0_rem_bytes = n0_rem * sizeof( int16_t );

			memcpy( buf0, ( c + ( rs_c * 0 ) ), n0_rem_bytes );
			memcpy( buf1, ( c + ( rs_c * 1 ) ), n0_rem_bytes );

			// c[0,0-15]
			S16_S16_BETA_OP_NLT16(c_int16_0p0,buf0,selector1,selector2)

			// c[1,0-15]
			S16_S16_BETA_OP_NLT16(c_int16_1p0,buf1,selector1,selector2)
		}
	}

	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_2xlt16:
	{
		memcpy( buf0, ( ( int16_t* )post_ops_list_temp->op_args1 +
			post_ops_attr.post_op_c_j + ( 0 * 16 ) ), ( n0_rem * sizeof( int16_t ) ) );

		selector1 =
			_mm256_loadu_si256( (__m256i const *) buf0);

		// c[0,0-15]
		c_int16_0p0 = _mm256_add_epi16( selector1, c_int16_0p0 );

		// c[1,0-15]
		c_int16_1p0 = _mm256_add_epi16( selector1, c_int16_1p0 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_2xlt16:
	{
		selector1 = _mm256_setzero_si256 ();

		// c[0,0-15]
		c_int16_0p0 = _mm256_max_epi16( selector1, c_int16_0p0 );

		// c[1,0-15]
		c_int16_1p0 = _mm256_max_epi16( selector1, c_int16_1p0 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_2xlt16:
	{
		selector2 =
			_mm256_set1_epi16( *( ( int16_t* )post_ops_list_temp->op_args2 ) );

		// c[0,0-15]
		RELU_SCALE_OP_S16_AVX2(c_int16_0p0)

		// c[1,0-15]
		RELU_SCALE_OP_S16_AVX2(c_int16_1p0)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_TANH_2xlt16:
	{
		__m256 dn, z, x, r2, r, y1, y2, x_tanh;
		__m256i q;

		// c[0,0-15]
		GELU_TANH_S16_AVX2(c_int16_0p0, y1, y2, r, r2, x, z, dn, x_tanh, q)

		// c[1,0-15]
		GELU_TANH_S16_AVX2(c_int16_1p0, y1, y2, r, r2, x, z, dn, x_tanh, q)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_ERF_2xlt16:
	{
		__m256 x, r, y1, y2, x_erf;

		// c[0,0-15]
		GELU_ERF_S16_AVX2(c_int16_0p0, y1, y2, r, x, x_erf)

		// c[1,0-15]
		GELU_ERF_S16_AVX2(c_int16_1p0, y1, y2, r, x, x_erf)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_CLIP_2xlt16:
	{
		__m256i min = _mm256_set1_epi16( *( int16_t* )post_ops_list_temp->op_args2 );
		__m256i max = _mm256_set1_epi16( *( int16_t* )post_ops_list_temp->op_args3 );

		// c[0,0-15]
		CLIP_S16_AVX2(c_int16_0p0, min, max)

		// c[1,0-15]
		CLIP_S16_AVX2(c_int16_1p0, min, max)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}

POST_OPS_DOWNSCALE_2xlt16:
	{
		__m128i temp[2];
		__m256i temp_32[2];
		__m256 temp_float[2];
		__m256 scale_1, scale_2;
		__m256 res_1, res_2;

		float float_buf[16];

		memcpy( float_buf, ( ( float* )post_ops_list_temp->scale_factor +
				post_ops_attr.post_op_c_j ), ( n0_rem * sizeof( float ) ) );

		// Load the scale vector values into the register
		scale_1 = _mm256_loadu_ps(float_buf + (0 * 8));
		scale_2 = _mm256_loadu_ps(float_buf + (1 * 8));

		// Scale first 16 columns of the 6 rows.
		CVT_MULRND_CVT16(c_int16_0p0, scale_1, scale_2)
		CVT_MULRND_CVT16(c_int16_1p0, scale_1, scale_2)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_2xlt16_DISABLE:
	;

	// Case where the output C matrix is s8 (downscaled) and this is the
	// final write for a given block within C.
	if ( ( post_ops_attr.buf_downscale != NULL ) &&
		 ( post_ops_attr.is_last_k == TRUE ) )
	{
		// Store the results in downscaled type (int8 instead of int32).
		__m128i temp[2];

		// c[0-1,0-15]
		CVT_STORE_S16_S8_2ROW_NLT16(c_int16_0p0, c_int16_1p0, buf0, buf1);

		dim_t n0_rem_dscale_bytes = n0_rem * sizeof( int8_t );

		CVT_STORE_S16_S8_NLT16_MEMCP_UTIL(buf0, 0, n0_rem_dscale_bytes);
		CVT_STORE_S16_S8_NLT16_MEMCP_UTIL(buf1, 1, n0_rem_dscale_bytes);
	}
	// Case where the output C matrix is s16 or is the temp buffer used to
	// store intermediate s16 accumulated values for downscaled (C-s8) api.
	else
	{
		// Store the results.
		// c[0,0-15]
		_mm256_storeu_si256( ( __m256i* )buf0, c_int16_0p0 );

		// c[1,0-15]
		_mm256_storeu_si256( ( __m256i* )buf1, c_int16_1p0 );

		dim_t n0_rem_bytes = n0_rem * sizeof( int16_t );

		memcpy( c + ( rs_c * 0 ) + ( 0 * 16 ), buf0, n0_rem_bytes );

		// c[1,0-15]
		memcpy( c + ( rs_c * 1 ) + ( 0 * 16 ), buf1, n0_rem_bytes );
	}
}

// 1x16 int8o16 kernel
LPGEMM_MN_FRINGE_KERN(uint8_t,int8_t,int16_t,u8s8s16o16_1x16)
{
	int NR = 16;

	static void *post_ops_labels[] =
		{
			&&POST_OPS_1x16_DISABLE,
			&&POST_OPS_BIAS_1x16,
			&&POST_OPS_RELU_1x16,
			&&POST_OPS_RELU_SCALE_1x16,
			&&POST_OPS_GELU_TANH_1x16,
			&&POST_OPS_GELU_ERF_1x16,
			&&POST_OPS_CLIP_1x16,
			&&POST_OPS_DOWNSCALE_1x16
		};

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
		uint8_t a_kfringe;

		b0 = _mm256_loadu_si256((__m256i const *)(b + (32 * k_full_pieces) + (NR * 0)));

		a_kfringe = *(a + (rs_a * 0) + (cs_a * (k_full_pieces * 2)));
		a_int32_0 = _mm256_set1_epi8(a_kfringe);

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_0p0 = _mm256_add_epi16(inter_vec, c_int16_0p0);
	}

	// Load alpha and beta
	__m256i selector1 = _mm256_set1_epi16(alpha);
	__m256i selector2 = _mm256_set1_epi16(beta);

	if ( alpha != 1 )
	{
		// Scale by alpha
		c_int16_0p0 = _mm256_mullo_epi16(selector1, c_int16_0p0);
	}

	// Scale C by beta.
	if (beta != 0)
	{
		// For the downscaled api (C-s8), the output C matrix values
		// needs to be upscaled to s16 to be used for beta scale.
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
			 ( post_ops_attr.is_first_k == TRUE ) )
		{
			// c[0,0-15]
			S8_S16_BETA_OP(c_int16_0p0,0,0,0,selector1,selector2)
		}
		else
		{
			// c[0,0-15]
			S16_S16_BETA_OP(c_int16_0p0,0,0,0,selector1,selector2)
		}
	}

	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_1x16:
	{
		selector1 =
			_mm256_loadu_si256( (__m256i const *)((int16_t *)post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 0 * 16 )) );

		// c[0,0-15]
		c_int16_0p0 = _mm256_add_epi16( selector1, c_int16_0p0 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_1x16:
	{
		selector1 = _mm256_setzero_si256 ();

		// c[0,0-15]
		c_int16_0p0 = _mm256_max_epi16( selector1, c_int16_0p0 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_1x16:
	{
		selector2 =
			_mm256_set1_epi16( *( ( int16_t* )post_ops_list_temp->op_args2 ) );

		// c[0,0-15]
		RELU_SCALE_OP_S16_AVX2(c_int16_0p0)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_TANH_1x16:
	{
		__m256 dn, z, x, r2, r, y1, y2, x_tanh;
		__m256i q;

		// c[0,0-15]
		GELU_TANH_S16_AVX2(c_int16_0p0, y1, y2, r, r2, x, z, dn, x_tanh, q)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_ERF_1x16:
	{
		__m256 x, r, y1, y2, x_erf;

		// c[0,0-15]
		GELU_ERF_S16_AVX2(c_int16_0p0, y1, y2, r, x, x_erf)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_CLIP_1x16:
	{
		__m256i min = _mm256_set1_epi16( *( int16_t* )post_ops_list_temp->op_args2 );
		__m256i max = _mm256_set1_epi16( *( int16_t* )post_ops_list_temp->op_args3 );

		// c[0,0-15]
		CLIP_S16_AVX2(c_int16_0p0, min, max)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}

POST_OPS_DOWNSCALE_1x16:
	{
		__m128i temp[2];
		__m256i temp_32[2];
		__m256 temp_float[2];
		__m256 scale_1, scale_2;
		__m256 res_1, res_2;

		/* Load the scale vector values into the register*/
		scale_1 =
			_mm256_loadu_ps(
			(float *)post_ops_list_temp->scale_factor +
			post_ops_attr.post_op_c_j + (0 * 8));
		scale_2 =
			_mm256_loadu_ps(
			(float *)post_ops_list_temp->scale_factor +
			post_ops_attr.post_op_c_j + (1 * 8));

		// Scale first 16 columns of the 2 rows.
		CVT_MULRND_CVT16(c_int16_0p0, scale_1, scale_2)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_1x16_DISABLE:
	;

	// Case where the output C matrix is s8 (downscaled) and this is the
	// final write for a given block within C.
	if ( ( post_ops_attr.buf_downscale != NULL ) &&
		 ( post_ops_attr.is_last_k == TRUE ) )
	{
		// Store the results in downscaled type (int8 instead of int32).
		__m128i temp[2];
		__m256i zero_reg = _mm256_setzero_si256();

		// c[0-1,0-15]
		CVT_STORE_S16_S8_1ROW(c_int16_0p0, zero_reg, 0, 0);
	}
	// Case where the output C matrix is s16 or is the temp buffer used to
	// store intermediate s16 accumulated values for downscaled (C-s8) api.
	else
	{
		// Store the results.
		// c[0,0-15]
		_mm256_storeu_si256( (__m256i *)(c + ( rs_c * 0 ) + ( 0 * 16 ) ), c_int16_0p0 );
	}
}

// 1xlt16 int8o16 kernel
LPGEMM_MN_LT_NR0_FRINGE_KERN(uint8_t,int8_t,int16_t,u8s8s16o16_1xlt16)
{
	int NR = 16;

	static void *post_ops_labels[] =
		{
			&&POST_OPS_1xlt16_DISABLE,
			&&POST_OPS_BIAS_1xlt16,
			&&POST_OPS_RELU_1xlt16,
			&&POST_OPS_RELU_SCALE_1xlt16,
			&&POST_OPS_GELU_TANH_1xlt16,
			&&POST_OPS_GELU_ERF_1xlt16,
			&&POST_OPS_CLIP_1xlt16,
			&&POST_OPS_DOWNSCALE_1xlt16
		};

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
		uint8_t a_kfringe;

		b0 = _mm256_loadu_si256((__m256i const *)(b + (32 * k_full_pieces) + (NR * 0)));

		a_kfringe = *(a + (rs_a * 0) + (cs_a * (k_full_pieces * 2)));
		a_int32_0 = _mm256_set1_epi8(a_kfringe);

		// Seperate register for intermediate op
		inter_vec = _mm256_maddubs_epi16(a_int32_0, b0);

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_int16_0p0 = _mm256_add_epi16(inter_vec, c_int16_0p0);
	}

	// Load alpha and beta
	__m256i selector1 = _mm256_set1_epi16(alpha);
	__m256i selector2 = _mm256_set1_epi16(beta);

	if ( alpha != 1 )
	{
		// Scale by alpha
		c_int16_0p0 = _mm256_mullo_epi16(selector1, c_int16_0p0);
	}

	// Scale C by beta.
	if (beta != 0)
	{
		// For the downscaled api (C-s8), the output C matrix values
		// needs to be upscaled to s16 to be used for beta scale.
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
			 ( post_ops_attr.is_first_k == TRUE ) )
		{
			dim_t n0_rem_dscale_bytes = n0_rem * sizeof( int8_t );

			S8_S16_BETA_NLT16_MEMCP_UTIL(buf0, 0, n0_rem_dscale_bytes);

			// c[0,0-15]
			S8_S16_BETA_OP_NLT16(c_int16_0p0,buf0,selector1,selector2)
		}
		else
		{
			dim_t n0_rem_bytes = n0_rem * sizeof( int16_t );

			memcpy( buf0, ( c + ( rs_c * 0 ) ), n0_rem_bytes );

			// c[0,0-15]
			S16_S16_BETA_OP_NLT16(c_int16_0p0,buf0,selector1,selector2)
		}
	}

	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_1xlt16:
	{
		memcpy( buf0, ( ( int16_t* )post_ops_list_temp->op_args1
			+ post_ops_attr.post_op_c_j + ( 0 * 16 ) ), ( n0_rem * sizeof( int16_t ) ) );

		selector1 =
			_mm256_loadu_si256( (__m256i const *)buf0 );

		// c[0,0-15]
		c_int16_0p0 = _mm256_add_epi16( selector1, c_int16_0p0 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_1xlt16:
	{
		selector1 = _mm256_setzero_si256 ();

		// c[0,0-15]
		c_int16_0p0 = _mm256_max_epi16( selector1, c_int16_0p0 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_1xlt16:
	{
		selector2 =
			_mm256_set1_epi16( *( ( int16_t* )post_ops_list_temp->op_args2 ) );

		// c[0,0-15]
		RELU_SCALE_OP_S16_AVX2(c_int16_0p0)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_TANH_1xlt16:
	{
		__m256 dn, z, x, r2, r, y1, y2, x_tanh;
		__m256i q;

		// c[0,0-15]
		GELU_TANH_S16_AVX2(c_int16_0p0, y1, y2, r, r2, x, z, dn, x_tanh, q)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_ERF_1xlt16:
	{
		__m256 x, r, y1, y2, x_erf;

		// c[0,0-15]
		GELU_ERF_S16_AVX2(c_int16_0p0, y1, y2, r, x, x_erf)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_CLIP_1xlt16:
	{
		__m256i min = _mm256_set1_epi16( *( int16_t* )post_ops_list_temp->op_args2 );
		__m256i max = _mm256_set1_epi16( *( int16_t* )post_ops_list_temp->op_args3 );

		// c[0,0-15]
		CLIP_S16_AVX2(c_int16_0p0, min, max)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}

POST_OPS_DOWNSCALE_1xlt16:
	{
		__m128i temp[2];
		__m256i temp_32[2];
		__m256 temp_float[2];
		__m256 scale_1, scale_2;
		__m256 res_1, res_2;

		float float_buf[16];

		memcpy( float_buf, ( ( float* )post_ops_list_temp->scale_factor +
				post_ops_attr.post_op_c_j ), ( n0_rem * sizeof( float ) ) );

		// Load the scale vector values into the register
		scale_1 = _mm256_loadu_ps(float_buf + (0 * 8));
		scale_2 = _mm256_loadu_ps(float_buf + (1 * 8));

		// Scale first 16 columns of the 2 rows.
		CVT_MULRND_CVT16(c_int16_0p0, scale_1, scale_2)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_1xlt16_DISABLE:
	;

	// Case where the output C matrix is s8 (downscaled) and this is the
	// final write for a given block within C.
	if ( ( post_ops_attr.buf_downscale != NULL ) &&
		 ( post_ops_attr.is_last_k == TRUE ) )
	{
		// Store the results in downscaled type (int8 instead of int32).
		__m128i temp[2];
		__m256i zero_reg = _mm256_setzero_si256();

		// c[0-1,0-15]
		CVT_STORE_S16_S8_1ROW_NLT16(c_int16_0p0, zero_reg, buf0);

		dim_t n0_rem_dscale_bytes = n0_rem * sizeof( int8_t );

		CVT_STORE_S16_S8_NLT16_MEMCP_UTIL(buf0, 0, n0_rem_dscale_bytes);
	}
	// Case where the output C matrix is s16 or is the temp buffer used to
	// store intermediate s16 accumulated values for downscaled (C-s8) api.
	else
	{
		// Store the results.
		// c[0,0-15]
		_mm256_storeu_si256( ( __m256i* )buf0, c_int16_0p0 );

		dim_t n0_rem_bytes = n0_rem * sizeof( int16_t );

		memcpy( c + ( rs_c * 0 ) + ( 0 * 16 ), buf0, n0_rem_bytes );
	}
}
#endif
