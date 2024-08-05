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

#include <immintrin.h>
#include <string.h>
#include "blis.h"

#ifdef BLIS_ADDON_LPGEMM

#include "lpgemm_f32_kern_macros.h"

#ifndef LPGEMM_BF16_JIT
// 5xlt16 bf16 fringe kernel
LPGEMM_MN_LT_NR0_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_5xlt16)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_5xLT16_DISABLE,
						  &&POST_OPS_BIAS_5xLT16,
						  &&POST_OPS_RELU_5xLT16,
						  &&POST_OPS_RELU_SCALE_5xLT16,
						  &&POST_OPS_GELU_TANH_5xLT16,
						  &&POST_OPS_GELU_ERF_5xLT16,
						  &&POST_OPS_CLIP_5xLT16,
						  &&POST_OPS_DOWNSCALE_5xLT16,
						  &&POST_OPS_MATRIX_ADD_5xLT16,
						  &&POST_OPS_SWISH_5xLT16,
						  &&POST_OPS_MATRIX_MUL_5xLT16,
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int16_t a_kfringe_buf = 0;

    // B matrix storage bfloat type
	__m512bh b0;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();

	__m512 c_float_2p0 = _mm512_setzero_ps();

	__m512 c_float_3p0 = _mm512_setzero_ps();

	__m512 c_float_4p0 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );

		// Broadcast a[1,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 1 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );

		// Broadcast a[2,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 2 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-15] = a[2,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );

		// Broadcast a[3,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 3 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[3,0-15] = a[3,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );

		// Broadcast a[4,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 4 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[4,0-15] = a[4,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );
	}
	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		a_kfringe_buf = *( a + (rs_a * 0) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );

		// Broadcast a[1,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 1) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );

		// Broadcast a[2,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 2) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-15] = a[2,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );

		// Broadcast a[3,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 3) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[3,0-15] = a[3,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );

		// Broadcast a[4,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 4) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[4,0-15] = a[4,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );
	}

	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	if ( alpha != 1 )
	{
		// Scale by alpha
		c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );

		c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );

		c_float_2p0 = _mm512_mul_ps( selector1, c_float_2p0 );

		c_float_3p0 = _mm512_mul_ps( selector1, c_float_3p0 );

		c_float_4p0 = _mm512_mul_ps( selector1, c_float_4p0 );
	}

	// Scale C by beta.
	if ( beta != 0 )
	{
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
				 ( post_ops_attr.is_first_k == TRUE ) )
		{
			__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

			// c[0,0-15]
			BF16_F32_BETA_OP_NLT16F_MASK( load_mask, c_float_0p0, 0, 0, \
							selector1, selector2 );

			// c[1,0-15]
			BF16_F32_BETA_OP_NLT16F_MASK( load_mask, c_float_1p0, 1, 0, \
							selector1, selector2 );

			// c[2,0-15]
			BF16_F32_BETA_OP_NLT16F_MASK( load_mask, c_float_2p0, 2, 0, \
							selector1, selector2 );

			// c[3,0-15]
			BF16_F32_BETA_OP_NLT16F_MASK( load_mask, c_float_3p0, 3, 0, \
							selector1, selector2 );

			// c[4,0-15]
			BF16_F32_BETA_OP_NLT16F_MASK( load_mask, c_float_4p0, 4, 0, \
							selector1, selector2 );
		}
		else
		{
			__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

			// c[0,0-15]
			F32_F32_BETA_OP_NLT16F_MASK(c, load_mask, c_float_0p0, 0, 0, 0, \
							selector1, selector2);

			// c[1,0-15]
			F32_F32_BETA_OP_NLT16F_MASK(c, load_mask, c_float_1p0, 0, 1, 0, \
							selector1, selector2);

			// c[2,0-15]
			F32_F32_BETA_OP_NLT16F_MASK(c, load_mask, c_float_2p0, 0, 2, 0, \
							selector1, selector2);

			// c[3,0-15]
			F32_F32_BETA_OP_NLT16F_MASK(c, load_mask, c_float_3p0, 0, 3, 0, \
							selector1, selector2);

			// c[4,0-15]
			F32_F32_BETA_OP_NLT16F_MASK(c, load_mask, c_float_4p0, 0, 4, 0, \
							selector1, selector2);
		}
	}
	// Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_5xLT16:
		{
			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
				 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );
				if ( post_ops_attr.c_stor_type == BF16 )
				{
					BF16_F32_BIAS_LOAD(selector1, bias_mask, 0);
				}
				else
				{
					selector1 =
						_mm512_maskz_loadu_ps
						(
						  bias_mask,
						  ( float* )post_ops_list_temp->op_args1 +
						  post_ops_attr.post_op_c_j
						);
				}

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

				// c[2,0-15]
				c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );

				// c[3,0-15]
				c_float_3p0 = _mm512_add_ps( selector1, c_float_3p0 );

				// c[4,0-15]
				c_float_4p0 = _mm512_add_ps( selector1, c_float_4p0 );
			}
			else
			{
				__m512 selector3;
				__m512 selector4;
				__m512 selector5;
				if ( post_ops_attr.c_stor_type == BF16 )
				{
					__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
					BF16_F32_BIAS_BCAST(selector1, bias_mask, 0);
					BF16_F32_BIAS_BCAST(selector2, bias_mask, 1);
					BF16_F32_BIAS_BCAST(selector3, bias_mask, 2);
					BF16_F32_BIAS_BCAST(selector4, bias_mask, 3);
					BF16_F32_BIAS_BCAST(selector5, bias_mask, 4);
				}
				else
				{
					selector1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 0 ) );
					selector2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 1 ) );
					selector3 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 2 ) );
					selector4 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 3 ) );
					selector5 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 4 ) );
				}

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );

				// c[2,0-15]
				c_float_2p0 = _mm512_add_ps( selector3, c_float_2p0 );

				// c[3,0-15]
				c_float_3p0 = _mm512_add_ps( selector4, c_float_3p0 );

				// c[4,0-15]
				c_float_4p0 = _mm512_add_ps( selector5, c_float_4p0 );
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_5xLT16:
		{
			selector1 = _mm512_setzero_ps();

			// c[0,0-15]
			c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

			// c[1,0-15]
			c_float_1p0 = _mm512_max_ps( selector1, c_float_1p0 );

			// c[2,0-15]
			c_float_2p0 = _mm512_max_ps( selector1, c_float_2p0 );

			// c[3,0-15]
			c_float_3p0 = _mm512_max_ps( selector1, c_float_3p0 );

			// c[4,0-15]
			c_float_4p0 = _mm512_max_ps( selector1, c_float_4p0 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_5xLT16:
		{
			selector1 = _mm512_setzero_ps();
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

			__mmask16 relu_cmp_mask;

			// c[0, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_0p0)

			// c[1, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_1p0)

			// c[2, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_2p0)

			// c[3, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_3p0)

			// c[4, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_4p0)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_TANH_5xLT16:
		{
			__m512 dn, z, x, r2, r, x_tanh;
			__m512i q;

			// c[0, 0-15]
			GELU_TANH_F32_AVX512(c_float_0p0, r, r2, x, z, dn, x_tanh, q)

			// c[1, 0-15]
			GELU_TANH_F32_AVX512(c_float_1p0, r, r2, x, z, dn, x_tanh, q)

			// c[2, 0-15]
			GELU_TANH_F32_AVX512(c_float_2p0, r, r2, x, z, dn, x_tanh, q)

			// c[3, 0-15]
			GELU_TANH_F32_AVX512(c_float_3p0, r, r2, x, z, dn, x_tanh, q)

			// c[4, 0-15]
			GELU_TANH_F32_AVX512(c_float_4p0, r, r2, x, z, dn, x_tanh, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_ERF_5xLT16:
		{
			__m512 x, r, x_erf;

			// c[0, 0-15]
			GELU_ERF_F32_AVX512(c_float_0p0, r, x, x_erf)

			// c[1, 0-15]
			GELU_ERF_F32_AVX512(c_float_1p0, r, x, x_erf)

			// c[2, 0-15]
			GELU_ERF_F32_AVX512(c_float_2p0, r, x, x_erf)

			// c[3, 0-15]
			GELU_ERF_F32_AVX512(c_float_3p0, r, x, x_erf)

			// c[4, 0-15]
			GELU_ERF_F32_AVX512(c_float_4p0, r, x, x_erf)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_CLIP_5xLT16:
		{
			__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
			__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

			// c[0, 0-15]
			CLIP_F32_AVX512(c_float_0p0, min, max)

			// c[1, 0-15]
			CLIP_F32_AVX512(c_float_1p0, min, max)

			// c[2, 0-15]
			CLIP_F32_AVX512(c_float_2p0, min, max)

			// c[3, 0-15]
			CLIP_F32_AVX512(c_float_3p0, min, max)

			// c[4, 0-15]
			CLIP_F32_AVX512(c_float_4p0, min, max)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}

POST_OPS_DOWNSCALE_5xLT16:
	{
		__m512 selector3 = _mm512_setzero_ps();
		__m512 selector4 = _mm512_setzero_ps();
		__m512 selector5 = _mm512_setzero_ps();

		__m512 zero_point0 = _mm512_setzero_ps();
		__m512 zero_point1 = _mm512_setzero_ps();
		__m512 zero_point2 = _mm512_setzero_ps();
		__m512 zero_point3 = _mm512_setzero_ps();
		__m512 zero_point4 = _mm512_setzero_ps();

		__mmask16 zp_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

		// Need to account for row vs column major swaps. For scalars
		// scale and zero point, no implications.
		// Even though different registers are used for scalar in column
		// and row major downscale path, all those registers will contain
		// the same value.
		// Also the same value is loaded to different registers so that
		// branching can be reduced and same code/register can be used
		// irrespective of whether scalar or vector op.
		if ( post_ops_list_temp->scale_factor_len == 1 )
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector4 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector5 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
		}

		// bf16 zero point value (scalar or vector).
		if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
		{
			zero_point0 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point1 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point2 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point3 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point4 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
		}

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 = _mm512_maskz_loadu_ps( zp_mask,
						  ( float* )post_ops_list_temp->scale_factor +
						  post_ops_attr.post_op_c_j + ( 0 * 16 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector1,zero_point0);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector1,zero_point0);

			// c[3, 0-15]
			SCL_MULRND_F32(c_float_3p0,selector1,zero_point0);

			// c[4, 0-15]
			SCL_MULRND_F32(c_float_4p0,selector1,zero_point0);
		}
		else
		{
			// If original output was columns major, then by the time
			// kernel sees it, the matrix would be accessed as if it were
			// transposed. Due to this the scale as well as zp array will
			// be accessed by the ic index, and each scale/zp element
			// corresponds to an entire row of the transposed output array,
			// instead of an entire column.
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 1 ) );
				selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 2 ) );
				selector4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 3 ) );
				selector5 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 4 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 0 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 1 ) ) );
				zero_point2 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 2 ) ) );
				zero_point3 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 3 ) ) );
				zero_point4 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 4 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector2,zero_point1);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector3,zero_point2);

			// c[3, 0-15]
			SCL_MULRND_F32(c_float_3p0,selector4,zero_point3);

			// c[4, 0-15]
			SCL_MULRND_F32(c_float_4p0,selector5,zero_point4);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_ADD_5xLT16:
	{
		__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			BF16_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,0);

			// c[1:0-15]
			BF16_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,1);

			// c[2:0-15]
			BF16_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,2);

			// c[3:0-15]
			BF16_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,3);

			// c[4:0-15]
			BF16_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,4);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			F32_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,0);

			// c[1:0-15]
			F32_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,1);

			// c[2:0-15]
			F32_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,2);

			// c[3:0-15]
			F32_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,3);

			// c[4:0-15]
			F32_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,4);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_MUL_5xLT16:
	{
		__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,0);

			// c[1:0-15]
			BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,1);

			// c[2:0-15]
			BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,2);

			// c[3:0-15]
			BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,3);

			// c[4:0-15]
			BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,4);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			F32_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,0);

			// c[1:0-15]
			F32_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,1);

			// c[2:0-15]
			F32_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,2);

			// c[3:0-15]
			F32_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,3);

			// c[4:0-15]
			F32_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,4);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}	
POST_OPS_SWISH_5xLT16:
	{
		selector1 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__m512 al_in, r, r2, z, dn;
		__m512i ex_out;

		// c[0, 0-15]
		SWISH_F32_AVX512_DEF(c_float_0p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[1, 0-15]
		SWISH_F32_AVX512_DEF(c_float_1p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[2, 0-15]
		SWISH_F32_AVX512_DEF(c_float_2p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[3, 0-15]
		SWISH_F32_AVX512_DEF(c_float_3p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[4, 0-15]
		SWISH_F32_AVX512_DEF(c_float_4p0, selector1, al_in, r, r2, z, dn, ex_out);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_5xLT16_DISABLE:
	;

	if ( ( post_ops_attr.buf_downscale != NULL ) && ( post_ops_attr.is_last_k == TRUE ) )
		{
			__mmask16 mask_all1 = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

			// Store the results in downscaled type (int8 instead of int32).
			// c[0,0-15]
			CVT_STORE_F32_BF16_MASK(c_float_0p0,0,0);

			// c[1,0-15]
			CVT_STORE_F32_BF16_MASK(c_float_1p0,1,0);

			// c[2,0-15]
			CVT_STORE_F32_BF16_MASK(c_float_2p0,2,0);

			// c[3,0-15]
			CVT_STORE_F32_BF16_MASK(c_float_3p0,3,0);

			// c[4,0-15]
			CVT_STORE_F32_BF16_MASK(c_float_4p0,4,0);
		}

	else
	{
		__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

		// Store the results.
		// c[0,0-15]
		_mm512_mask_storeu_ps( c + ( rs_c * 0 ), load_mask, c_float_0p0 );

		// c[1,0-15]
		_mm512_mask_storeu_ps( c + ( rs_c * 1 ), load_mask, c_float_1p0 );

		// c[2,0-15]
		_mm512_mask_storeu_ps( c + ( rs_c * 2 ), load_mask, c_float_2p0 );

		// c[3,0-15]
		_mm512_mask_storeu_ps( c + ( rs_c * 3 ), load_mask, c_float_3p0 );

		// c[4,0-15]
		_mm512_mask_storeu_ps( c + ( rs_c * 4 ), load_mask, c_float_4p0 );
	}
}

// 4xlt16 bf16 fringe kernel
LPGEMM_MN_LT_NR0_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_4xlt16)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_4xLT16_DISABLE,
						  &&POST_OPS_BIAS_4xLT16,
						  &&POST_OPS_RELU_4xLT16,
						  &&POST_OPS_RELU_SCALE_4xLT16,
						  &&POST_OPS_GELU_TANH_4xLT16,
						  &&POST_OPS_GELU_ERF_4xLT16,
						  &&POST_OPS_CLIP_4xLT16,
						  &&POST_OPS_DOWNSCALE_4xLT16,
						  &&POST_OPS_MATRIX_ADD_4xLT16,
						  &&POST_OPS_SWISH_4xLT16,
						  &&POST_OPS_MATRIX_MUL_4xLT16
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int16_t a_kfringe_buf = 0;

    // B matrix storage bfloat type
	__m512bh b0;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();

	__m512 c_float_2p0 = _mm512_setzero_ps();

	__m512 c_float_3p0 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );

		// Broadcast a[1,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 1 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );

		// Broadcast a[2,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 2 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-15] = a[2,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );

		// Broadcast a[3,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 3 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[3,0-15] = a[3,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );
	}
	// Handle k remainder.

	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		a_kfringe_buf = *( a + (rs_a * 0) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );

		// Broadcast a[1,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 1) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );

		// Broadcast a[2,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 2) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-15] = a[2,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );

		// Broadcast a[3,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 3) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[3,0-15] = a[3,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );
	}

	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	if ( alpha != 1 )
	{
		// Scale by alpha
		c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );

		c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );

		c_float_2p0 = _mm512_mul_ps( selector1, c_float_2p0 );

		c_float_3p0 = _mm512_mul_ps( selector1, c_float_3p0 );
	}

	// Scale C by beta.
	if ( beta != 0 )
	{
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
				 ( post_ops_attr.is_first_k == TRUE ) )
		{
			__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

			// c[0,0-15]
			BF16_F32_BETA_OP_NLT16F_MASK( load_mask, c_float_0p0, 0, 0, \
							selector1, selector2 );

			// c[1,0-15]
			BF16_F32_BETA_OP_NLT16F_MASK( load_mask, c_float_1p0, 1, 0, \
							selector1, selector2 );

			// c[2,0-15]
			BF16_F32_BETA_OP_NLT16F_MASK( load_mask, c_float_2p0, 2, 0, \
							selector1, selector2 );

			// c[3,0-15]
			BF16_F32_BETA_OP_NLT16F_MASK( load_mask, c_float_3p0, 3, 0, \
							selector1, selector2 );
		}
		else
		{
			__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

			// c[0,0-15]
			F32_F32_BETA_OP_NLT16F_MASK(c, load_mask, c_float_0p0, 0, 0, 0, \
							selector1, selector2);

			// c[1,0-15]
			F32_F32_BETA_OP_NLT16F_MASK(c, load_mask, c_float_1p0, 0, 1, 0, \
							selector1, selector2);

			// c[2,0-15]
			F32_F32_BETA_OP_NLT16F_MASK(c, load_mask, c_float_2p0, 0, 2, 0, \
							selector1, selector2);

			// c[3,0-15]
			F32_F32_BETA_OP_NLT16F_MASK(c, load_mask, c_float_3p0, 0, 3, 0, \
							selector1, selector2);
		}
	}
	// Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_4xLT16:
		{
			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
				 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );
				if ( post_ops_attr.c_stor_type == BF16 )
				{
					BF16_F32_BIAS_LOAD(selector1, bias_mask, 0);
				}
				else
				{
					selector1 =
						_mm512_maskz_loadu_ps
						(
						  bias_mask,
						  ( float* )post_ops_list_temp->op_args1 +
						  post_ops_attr.post_op_c_j
						);
				}

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

				// c[2,0-15]
				c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );

				// c[3,0-15]
				c_float_3p0 = _mm512_add_ps( selector1, c_float_3p0 );
			}
			else
			{
				__m512 selector3;
				__m512 selector4;
				if ( post_ops_attr.c_stor_type == BF16 )
				{
					__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
					BF16_F32_BIAS_BCAST(selector1, bias_mask, 0);
					BF16_F32_BIAS_BCAST(selector2, bias_mask, 1);
					BF16_F32_BIAS_BCAST(selector3, bias_mask, 2);
					BF16_F32_BIAS_BCAST(selector4, bias_mask, 3);
				}
				else
				{
					selector1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 0 ) );
					selector2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 1 ) );
					selector3 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 2 ) );
					selector4 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 3 ) );
				}

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );

				// c[2,0-15]
				c_float_2p0 = _mm512_add_ps( selector3, c_float_2p0 );

				// c[3,0-15]
				c_float_3p0 = _mm512_add_ps( selector4, c_float_3p0 );
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_4xLT16:
		{
			selector1 = _mm512_setzero_ps();

			// c[0,0-15]
			c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

			// c[1,0-15]
			c_float_1p0 = _mm512_max_ps( selector1, c_float_1p0 );

			// c[2,0-15]
			c_float_2p0 = _mm512_max_ps( selector1, c_float_2p0 );

			// c[3,0-15]
			c_float_3p0 = _mm512_max_ps( selector1, c_float_3p0 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_4xLT16:
		{
			selector1 = _mm512_setzero_ps();
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

			__mmask16 relu_cmp_mask;

			// c[0, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_0p0)

			// c[1, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_1p0)

			// c[2, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_2p0)

			// c[3, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_3p0)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_TANH_4xLT16:
		{
			__m512 dn, z, x, r2, r, x_tanh;
			__m512i q;

			// c[0, 0-15]
			GELU_TANH_F32_AVX512(c_float_0p0, r, r2, x, z, dn, x_tanh, q)

			// c[1, 0-15]
			GELU_TANH_F32_AVX512(c_float_1p0, r, r2, x, z, dn, x_tanh, q)

			// c[2, 0-15]
			GELU_TANH_F32_AVX512(c_float_2p0, r, r2, x, z, dn, x_tanh, q)

			// c[3, 0-15]
			GELU_TANH_F32_AVX512(c_float_3p0, r, r2, x, z, dn, x_tanh, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_ERF_4xLT16:
		{
			__m512 x, r, x_erf;

			// c[0, 0-15]
			GELU_ERF_F32_AVX512(c_float_0p0, r, x, x_erf)

			// c[1, 0-15]
			GELU_ERF_F32_AVX512(c_float_1p0, r, x, x_erf)

			// c[2, 0-15]
			GELU_ERF_F32_AVX512(c_float_2p0, r, x, x_erf)

			// c[3, 0-15]
			GELU_ERF_F32_AVX512(c_float_3p0, r, x, x_erf)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_CLIP_4xLT16:
		{
			__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
			__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

			// c[0, 0-15]
			CLIP_F32_AVX512(c_float_0p0, min, max)

			// c[1, 0-15]
			CLIP_F32_AVX512(c_float_1p0, min, max)

			// c[2, 0-15]
			CLIP_F32_AVX512(c_float_2p0, min, max)

			// c[3, 0-15]
			CLIP_F32_AVX512(c_float_3p0, min, max)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_DOWNSCALE_4xLT16:
	{
		__m512 selector3 = _mm512_setzero_ps();
		__m512 selector4 = _mm512_setzero_ps();

		__m512 zero_point0 = _mm512_setzero_ps();
		__m512 zero_point1 = _mm512_setzero_ps();
		__m512 zero_point2 = _mm512_setzero_ps();
		__m512 zero_point3 = _mm512_setzero_ps();

		__mmask16 zp_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

		// Need to account for row vs column major swaps. For scalars
		// scale and zero point, no implications.
		// Even though different registers are used for scalar in column
		// and row major downscale path, all those registers will contain
		// the same value.
		// Also the same value is loaded to different registers so that
		// branching can be reduced and same code/register can be used
		// irrespective of whether scalar or vector op.
		if ( post_ops_list_temp->scale_factor_len == 1 )
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector4 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
		}

		// bf16 zero point value (scalar or vector).
		if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
		{
			zero_point0 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point1 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point2 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point3 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
		}

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 = _mm512_maskz_loadu_ps( zp_mask,
						  ( float* )post_ops_list_temp->scale_factor +
						  post_ops_attr.post_op_c_j + ( 0 * 16 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector1,zero_point0);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector1,zero_point0);

			// c[3, 0-15]
			SCL_MULRND_F32(c_float_3p0,selector1,zero_point0);
		}
		else
		{
			// If original output was columns major, then by the time
			// kernel sees it, the matrix would be accessed as if it were
			// transposed. Due to this the scale as well as zp array will
			// be accessed by the ic index, and each scale/zp element
			// corresponds to an entire row of the transposed output array,
			// instead of an entire column.
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 1 ) );
				selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 2 ) );
				selector4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 3 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 0 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 1 ) ) );
				zero_point2 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 2 ) ) );
				zero_point3 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 3 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector2,zero_point1);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector3,zero_point2);

			// c[3, 0-15]
			SCL_MULRND_F32(c_float_3p0,selector4,zero_point3);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_ADD_4xLT16:
	{
		__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			BF16_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,0);

			// c[1:0-15]
			BF16_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,1);

			// c[2:0-15]
			BF16_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,2);

			// c[3:0-15]
			BF16_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,3);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			F32_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,0);

			// c[1:0-15]
			F32_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,1);

			// c[2:0-15]
			F32_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,2);

			// c[3:0-15]
			F32_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,3);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_MUL_4xLT16:
	{
		__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,0);

			// c[1:0-15]
			BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,1);

			// c[2:0-15]
			BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,2);

			// c[3:0-15]
			BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,3);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			F32_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,0);

			// c[1:0-15]
			F32_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,1);

			// c[2:0-15]
			F32_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,2);

			// c[3:0-15]
			F32_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,3);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}	
POST_OPS_SWISH_4xLT16:
	{
		selector1 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__m512 al_in, r, r2, z, dn;
		__m512i ex_out;

		// c[0, 0-15]
		SWISH_F32_AVX512_DEF(c_float_0p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[1, 0-15]
		SWISH_F32_AVX512_DEF(c_float_1p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[2, 0-15]
		SWISH_F32_AVX512_DEF(c_float_2p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[3, 0-15]
		SWISH_F32_AVX512_DEF(c_float_3p0, selector1, al_in, r, r2, z, dn, ex_out);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_4xLT16_DISABLE:
	;

	if ( ( post_ops_attr.buf_downscale != NULL ) && ( post_ops_attr.is_last_k == TRUE ) )
	{
		__mmask16 mask_all1 = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

		// Store the results in downscaled type (int8 instead of int32).
		// c[0,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_0p0,0,0);

		// c[1,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_1p0,1,0);

		// c[2,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_2p0,2,0);

		// c[3,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_3p0,3,0);
	}
	else
	{
		__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

		// Store the results.
		// c[0,0-15]
		_mm512_mask_storeu_ps( c + ( rs_c * 0 ), load_mask, c_float_0p0 );

		// c[1,0-15]
		_mm512_mask_storeu_ps( c + ( rs_c * 1 ), load_mask, c_float_1p0 );

		// c[2,0-15]
		_mm512_mask_storeu_ps( c + ( rs_c * 2 ), load_mask, c_float_2p0 );

		// c[3,0-15]
		_mm512_mask_storeu_ps( c + ( rs_c * 3 ), load_mask, c_float_3p0 );
	}

}

// 3xlt16 bf16 fringe kernel
LPGEMM_MN_LT_NR0_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_3xlt16)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_3xLT16_DISABLE,
						  &&POST_OPS_BIAS_3xLT16,
						  &&POST_OPS_RELU_3xLT16,
						  &&POST_OPS_RELU_SCALE_3xLT16,
						  &&POST_OPS_GELU_TANH_3xLT16,
						  &&POST_OPS_GELU_ERF_3xLT16,
						  &&POST_OPS_CLIP_3xLT16,
						  &&POST_OPS_DOWNSCALE_3xLT16,
						  &&POST_OPS_MATRIX_ADD_3xLT16,
						  &&POST_OPS_SWISH_3xLT16,
						  &&POST_OPS_MATRIX_MUL_3xLT16
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int16_t a_kfringe_buf = 0;

    // B matrix storage bfloat type
	__m512bh b0;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();

	__m512 c_float_2p0 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );

		// Broadcast a[1,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 1 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );

		// Broadcast a[2,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 2 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-15] = a[2,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
	}
	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		a_kfringe_buf = *( a + (rs_a * 0) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );

		// Broadcast a[1,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 1) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );

		// Broadcast a[2,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 2) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-15] = a[2,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
	}

	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	if ( alpha != 1 )
	{
		// Scale by alpha
		c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );

		c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );

		c_float_2p0 = _mm512_mul_ps( selector1, c_float_2p0 );
	}

	// Scale C by beta.
	if ( beta != 0 )
	{
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
				 ( post_ops_attr.is_first_k == TRUE ) )
		{
			__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

			// c[0,0-15]
			BF16_F32_BETA_OP_NLT16F_MASK( load_mask, c_float_0p0, 0, 0, \
							selector1, selector2 );

			// c[1,0-15]
			BF16_F32_BETA_OP_NLT16F_MASK( load_mask, c_float_1p0, 1, 0, \
							selector1, selector2 );

			// c[2,0-15]
			BF16_F32_BETA_OP_NLT16F_MASK( load_mask, c_float_2p0, 2, 0, \
							selector1, selector2 );
		}
		else
		{
			__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

			// c[0,0-15]
			F32_F32_BETA_OP_NLT16F_MASK(c, load_mask, c_float_0p0, 0, 0, 0, \
							selector1, selector2);

			// c[1,0-15]
			F32_F32_BETA_OP_NLT16F_MASK(c, load_mask, c_float_1p0, 0, 1, 0, \
							selector1, selector2);

			// c[2,0-15]
			F32_F32_BETA_OP_NLT16F_MASK(c, load_mask, c_float_2p0, 0, 2, 0, \
							selector1, selector2);
		}
	}
	// Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_3xLT16:
		{
			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
				 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );
				if ( post_ops_attr.c_stor_type == BF16 )
				{
					BF16_F32_BIAS_LOAD(selector1, bias_mask, 0);
				}
				else
				{
					selector1 =
						_mm512_maskz_loadu_ps
						(
						  bias_mask,
						  ( float* )post_ops_list_temp->op_args1 +
						  post_ops_attr.post_op_c_j
						);
				}

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

				// c[2,0-15]
				c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );
			}
			else
			{
				__m512 selector3;
				if ( post_ops_attr.c_stor_type == BF16 )
				{
					__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
					BF16_F32_BIAS_BCAST(selector1, bias_mask, 0);
					BF16_F32_BIAS_BCAST(selector2, bias_mask, 1);
					BF16_F32_BIAS_BCAST(selector3, bias_mask, 2);
				}
				else
				{
					selector1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 0 ) );
					selector2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 1 ) );
					selector3 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 2 ) );
				}

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );

				// c[2,0-15]
				c_float_2p0 = _mm512_add_ps( selector3, c_float_2p0 );
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_3xLT16:
		{
			selector1 = _mm512_setzero_ps();

			// c[0,0-15]
			c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

			// c[1,0-15]
			c_float_1p0 = _mm512_max_ps( selector1, c_float_1p0 );

			// c[2,0-15]
			c_float_2p0 = _mm512_max_ps( selector1, c_float_2p0 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_3xLT16:
		{
			selector1 = _mm512_setzero_ps();
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

			__mmask16 relu_cmp_mask;

			// c[0, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_0p0)

			// c[1, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_1p0)

			// c[2, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_2p0)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_TANH_3xLT16:
		{
			__m512 dn, z, x, r2, r, x_tanh;
			__m512i q;

			// c[0, 0-15]
			GELU_TANH_F32_AVX512(c_float_0p0, r, r2, x, z, dn, x_tanh, q)

			// c[1, 0-15]
			GELU_TANH_F32_AVX512(c_float_1p0, r, r2, x, z, dn, x_tanh, q)

			// c[2, 0-15]
			GELU_TANH_F32_AVX512(c_float_2p0, r, r2, x, z, dn, x_tanh, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_ERF_3xLT16:
		{
			__m512 x, r, x_erf;

			// c[0, 0-15]
			GELU_ERF_F32_AVX512(c_float_0p0, r, x, x_erf)

			// c[1, 0-15]
			GELU_ERF_F32_AVX512(c_float_1p0, r, x, x_erf)

			// c[2, 0-15]
			GELU_ERF_F32_AVX512(c_float_2p0, r, x, x_erf)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_CLIP_3xLT16:
		{
			__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
			__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

			// c[0, 0-15]
			CLIP_F32_AVX512(c_float_0p0, min, max)

			// c[1, 0-15]
			CLIP_F32_AVX512(c_float_1p0, min, max)

			// c[2, 0-15]
			CLIP_F32_AVX512(c_float_2p0, min, max)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}

POST_OPS_DOWNSCALE_3xLT16:
	{
		__m512 selector3 = _mm512_setzero_ps();

		__m512 zero_point0 = _mm512_setzero_ps();
		__m512 zero_point1 = _mm512_setzero_ps();
		__m512 zero_point2 = _mm512_setzero_ps();

		__mmask16 zp_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

		// Need to account for row vs column major swaps. For scalars
		// scale and zero point, no implications.
		// Even though different registers are used for scalar in column
		// and row major downscale path, all those registers will contain
		// the same value.
		// Also the same value is loaded to different registers so that
		// branching can be reduced and same code/register can be used
		// irrespective of whether scalar or vector op.
		if ( post_ops_list_temp->scale_factor_len == 1 )
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
		}

		// bf16 zero point value (scalar or vector).
		if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
		{
			zero_point0 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point1 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point2 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
		}

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 = _mm512_maskz_loadu_ps( zp_mask,
						  ( float* )post_ops_list_temp->scale_factor +
						  post_ops_attr.post_op_c_j + ( 0 * 16 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector1,zero_point0);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector1,zero_point0);
		}
		else
		{
			// If original output was columns major, then by the time
			// kernel sees it, the matrix would be accessed as if it were
			// transposed. Due to this the scale as well as zp array will
			// be accessed by the ic index, and each scale/zp element
			// corresponds to an entire row of the transposed output array,
			// instead of an entire column.
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 1 ) );
				selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 2 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 0 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 1 ) ) );
				zero_point2 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 2 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector2,zero_point1);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector3,zero_point2);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_ADD_3xLT16:
	{
		__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			BF16_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,0);

			// c[1:0-15]
			BF16_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,1);

			// c[2:0-15]
			BF16_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,2);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			F32_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,0);

			// c[1:0-15]
			F32_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,1);

			// c[2:0-15]
			F32_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,2);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_MUL_3xLT16:
	{
		__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,0);

			// c[1:0-15]
			BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,1);

			// c[2:0-15]
			BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,2);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			F32_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,0);

			// c[1:0-15]
			F32_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,1);

			// c[2:0-15]
			F32_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,2);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_SWISH_3xLT16:
	{
		selector1 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__m512 al_in, r, r2, z, dn;
		__m512i ex_out;

		// c[0, 0-15]
		SWISH_F32_AVX512_DEF(c_float_0p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[1, 0-15]
		SWISH_F32_AVX512_DEF(c_float_1p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[2, 0-15]
		SWISH_F32_AVX512_DEF(c_float_2p0, selector1, al_in, r, r2, z, dn, ex_out);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_3xLT16_DISABLE:
	;

	if ( ( post_ops_attr.buf_downscale != NULL ) && ( post_ops_attr.is_last_k == TRUE ) )
	{
		__mmask16 mask_all1 = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

		// Store the results in downscaled type (int8 instead of int32).
		// c[0,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_0p0,0,0);

		// c[1,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_1p0,1,0);

		// c[2,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_2p0,2,0);
	}

	else
	{
		__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

		// Store the results.
		// c[0,0-15]
		_mm512_mask_storeu_ps( c + ( rs_c * 0 ), load_mask, c_float_0p0 );

		// c[1,0-15]
		_mm512_mask_storeu_ps( c + ( rs_c * 1 ), load_mask, c_float_1p0 );

		// c[2,0-15]
		_mm512_mask_storeu_ps( c + ( rs_c * 2 ), load_mask, c_float_2p0 );
	}

}

// 2xlt16 bf16 fringe kernel
LPGEMM_MN_LT_NR0_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_2xlt16)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_2xLT16_DISABLE,
						  &&POST_OPS_BIAS_2xLT16,
						  &&POST_OPS_RELU_2xLT16,
						  &&POST_OPS_RELU_SCALE_2xLT16,
						  &&POST_OPS_GELU_TANH_2xLT16,
						  &&POST_OPS_GELU_ERF_2xLT16,
						  &&POST_OPS_CLIP_2xLT16,
						  &&POST_OPS_DOWNSCALE_2xLT16,
						  &&POST_OPS_MATRIX_ADD_2xLT16,
						  &&POST_OPS_SWISH_2xLT16,
						  &&POST_OPS_MATRIX_MUL_2xLT16
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int16_t a_kfringe_buf = 0;

    // B matrix storage bfloat type
	__m512bh b0;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );

		// Broadcast a[1,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 1 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
	}
	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		a_kfringe_buf = *( a + (rs_a * 0) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );

		// Broadcast a[1,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 1) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
	}

	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	if ( alpha != 1 )
	{
		// Scale by alpha
		c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );

		c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );
	}

	// Scale C by beta.
	if ( beta != 0 )
	{
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
				 ( post_ops_attr.is_first_k == TRUE ) )
		{
			__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

			// c[0,0-15]
			BF16_F32_BETA_OP_NLT16F_MASK( load_mask, c_float_0p0, 0, 0, \
							selector1, selector2 );

			// c[1,0-15]
			BF16_F32_BETA_OP_NLT16F_MASK( load_mask, c_float_1p0, 1, 0, \
							selector1, selector2 );
		}
		else
		{
			__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

			// c[0,0-15]
			F32_F32_BETA_OP_NLT16F_MASK(c, load_mask, c_float_0p0, 0, 0, 0, \
							selector1, selector2);

			// c[1,0-15]
			F32_F32_BETA_OP_NLT16F_MASK(c, load_mask, c_float_1p0, 0, 1, 0, \
							selector1, selector2);
		}
	}
	// Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_2xLT16:
		{
			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
				 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );
				if ( post_ops_attr.c_stor_type == BF16 )
				{
					BF16_F32_BIAS_LOAD(selector1, bias_mask, 0);
				}
				else
				{
					selector1 =
						_mm512_maskz_loadu_ps
						(
						  bias_mask,
						  ( float* )post_ops_list_temp->op_args1 +
						  post_ops_attr.post_op_c_j
						);
				}

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );
			}
			else
			{
				if ( post_ops_attr.c_stor_type == BF16 )
				{
					__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
					BF16_F32_BIAS_BCAST(selector1, bias_mask, 0);
					BF16_F32_BIAS_BCAST(selector2, bias_mask, 1);
				}
				else
				{
					selector1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 0 ) );
					selector2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 1 ) );
				}

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_2xLT16:
		{
			selector1 = _mm512_setzero_ps();

			// c[0,0-15]
			c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

			// c[1,0-15]
			c_float_1p0 = _mm512_max_ps( selector1, c_float_1p0 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_2xLT16:
		{
			selector1 = _mm512_setzero_ps();
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

			__mmask16 relu_cmp_mask;

			// c[0, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_0p0)

			// c[1, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_1p0)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_TANH_2xLT16:
		{
			__m512 dn, z, x, r2, r, x_tanh;
			__m512i q;

			// c[0, 0-15]
			GELU_TANH_F32_AVX512(c_float_0p0, r, r2, x, z, dn, x_tanh, q)

			// c[1, 0-15]
			GELU_TANH_F32_AVX512(c_float_1p0, r, r2, x, z, dn, x_tanh, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_ERF_2xLT16:
		{
			__m512 x, r, x_erf;

			// c[0, 0-15]
			GELU_ERF_F32_AVX512(c_float_0p0, r, x, x_erf)

			// c[1, 0-15]
			GELU_ERF_F32_AVX512(c_float_1p0, r, x, x_erf)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_CLIP_2xLT16:
		{
			__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
			__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

			// c[0, 0-15]
			CLIP_F32_AVX512(c_float_0p0, min, max)

			// c[1, 0-15]
			CLIP_F32_AVX512(c_float_1p0, min, max)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}

POST_OPS_DOWNSCALE_2xLT16:
	{
		__m512 zero_point0 = _mm512_setzero_ps();
		__m512 zero_point1 = _mm512_setzero_ps();

		__mmask16 zp_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

		// Need to account for row vs column major swaps. For scalars
		// scale and zero point, no implications.
		// Even though different registers are used for scalar in column
		// and row major downscale path, all those registers will contain
		// the same value.
		// Also the same value is loaded to different registers so that
		// branching can be reduced and same code/register can be used
		// irrespective of whether scalar or vector op.
		if ( post_ops_list_temp->scale_factor_len == 1 )
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
		}

		// bf16 zero point value (scalar or vector).
		if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
		{
			zero_point0 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point1 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
		}

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 = _mm512_maskz_loadu_ps( zp_mask,
						  ( float* )post_ops_list_temp->scale_factor +
						  post_ops_attr.post_op_c_j + ( 0 * 16 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector1,zero_point0);
		}
		else
		{
			// If original output was columns major, then by the time
			// kernel sees it, the matrix would be accessed as if it were
			// transposed. Due to this the scale as well as zp array will
			// be accessed by the ic index, and each scale/zp element
			// corresponds to an entire row of the transposed output array,
			// instead of an entire column.
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 1 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 0 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 1 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector2,zero_point1);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_ADD_2xLT16:
	{
		__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			BF16_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,0);

			// c[1:0-15]
			BF16_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,1);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			F32_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,0);

			// c[1:0-15]
			F32_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,1);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_MUL_2xLT16:
	{
		__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,0);

			// c[1:0-15]
			BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,1);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			F32_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,0);

			// c[1:0-15]
			F32_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,1);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}	
POST_OPS_SWISH_2xLT16:
	{
		selector1 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__m512 al_in, r, r2, z, dn;
		__m512i ex_out;

		// c[0, 0-15]
		SWISH_F32_AVX512_DEF(c_float_0p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[1, 0-15]
		SWISH_F32_AVX512_DEF(c_float_1p0, selector1, al_in, r, r2, z, dn, ex_out);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_2xLT16_DISABLE:
	;

	if ( ( post_ops_attr.buf_downscale != NULL ) && ( post_ops_attr.is_last_k == TRUE ) )
	{
		__mmask16 mask_all1 = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

		// Store the results in downscaled type (int8 instead of int32).
		// c[0,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_0p0,0,0);

		// c[1,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_1p0,1,0);
	}

	else
	{
		__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

		// Store the results.
		// c[0,0-15]
		_mm512_mask_storeu_ps( c + ( rs_c * 0 ), load_mask, c_float_0p0 );

		// c[1,0-15]
		_mm512_mask_storeu_ps( c + ( rs_c * 1 ), load_mask, c_float_1p0 );
	}

}

// 1xlt16 bf16 fringe kernel
LPGEMM_MN_LT_NR0_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_1xlt16)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_1xLT16_DISABLE,
						  &&POST_OPS_BIAS_1xLT16,
						  &&POST_OPS_RELU_1xLT16,
						  &&POST_OPS_RELU_SCALE_1xLT16,
						  &&POST_OPS_GELU_TANH_1xLT16,
						  &&POST_OPS_GELU_ERF_1xLT16,
						  &&POST_OPS_CLIP_1xLT16,
						  &&POST_OPS_DOWNSCALE_1xLT16,
						  &&POST_OPS_MATRIX_ADD_1xLT16,
						  &&POST_OPS_SWISH_1xLT16,
						  &&POST_OPS_MATRIX_MUL_1xLT16
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int16_t a_kfringe_buf = 0;

    // B matrix storage bfloat type
	__m512bh b0;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
	}
	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		a_kfringe_buf = *( a + (rs_a * 0) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
	}

	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	if ( alpha != 1 )
	{
		// Scale by alpha
		c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );
	}

	// Scale C by beta.
	if ( beta != 0 )
	{
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
				 ( post_ops_attr.is_first_k == TRUE ) )
		{
			__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

			// c[0,0-15]
			BF16_F32_BETA_OP_NLT16F_MASK( load_mask, c_float_0p0, 0, 0, \
							selector1, selector2 );
		}
		else
		{
			__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

			// c[0,0-15]
			F32_F32_BETA_OP_NLT16F_MASK(c, load_mask, c_float_0p0, 0, 0, 0, \
							selector1, selector2);
		}
	}
	// Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_1xLT16:
		{
			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
				 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );
				if ( post_ops_attr.c_stor_type == BF16 )
				{
					BF16_F32_BIAS_LOAD(selector1, bias_mask, 0);
				}
				else
				{
					selector1 =
						_mm512_maskz_loadu_ps
						(
						  bias_mask,
						  ( float* )post_ops_list_temp->op_args1 +
						  post_ops_attr.post_op_c_j
						);
				}

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );
			}
			else
			{
				if ( post_ops_attr.c_stor_type == BF16 )
				{
					__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
					BF16_F32_BIAS_BCAST(selector1, bias_mask, 0);
				}
				else
				{
					selector1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
									+ post_ops_attr.post_op_c_i + 0 ) );
				}

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_1xLT16:
		{
			selector1 = _mm512_setzero_ps();

			// c[0,0-15]
			c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_1xLT16:
		{
			selector1 = _mm512_setzero_ps();
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

			__mmask16 relu_cmp_mask;

			// c[0, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_0p0)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_TANH_1xLT16:
		{
			__m512 dn, z, x, r2, r, x_tanh;
			__m512i q;

			// c[0, 0-15]
			GELU_TANH_F32_AVX512(c_float_0p0, r, r2, x, z, dn, x_tanh, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_ERF_1xLT16:
		{
			__m512 x, r, x_erf;

			// c[0, 0-15]
			GELU_ERF_F32_AVX512(c_float_0p0, r, x, x_erf)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_CLIP_1xLT16:
		{
			__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
			__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

			// c[0, 0-15]
			CLIP_F32_AVX512(c_float_0p0, min, max)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}

POST_OPS_DOWNSCALE_1xLT16:
	{
		__m512 zero_point0 = _mm512_setzero_ps();

		__mmask16 zp_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

		// Need to account for row vs column major swaps. For scalars
		// scale and zero point, no implications.
		// Even though different registers are used for scalar in column
		// and row major downscale path, all those registers will contain
		// the same value.
		// Also the same value is loaded to different registers so that
		// branching can be reduced and same code/register can be used
		// irrespective of whether scalar or vector op.
		if ( post_ops_list_temp->scale_factor_len == 1 )
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
		}

		// bf16 zero point value (scalar or vector).
		if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
		{
			zero_point0 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
		}

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 = _mm512_maskz_loadu_ps( zp_mask,
						  ( float* )post_ops_list_temp->scale_factor +
						  post_ops_attr.post_op_c_j + ( 0 * 16 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);
		}
		else
		{
			// If original output was columns major, then by the time
			// kernel sees it, the matrix would be accessed as if it were
			// transposed. Due to this the scale as well as zp array will
			// be accessed by the ic index, and each scale/zp element
			// corresponds to an entire row of the transposed output array,
			// instead of an entire column.
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 0 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 0 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_ADD_1xLT16:
	{
		__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			BF16_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,0);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			F32_F32_MATRIX_ADD_1COL_PAR(load_mask,selector1,0);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_MUL_1xLT16:
	{
		__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,0);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			F32_F32_MATRIX_MUL_1COL_PAR(load_mask,selector1,0);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_SWISH_1xLT16:
	{
		selector1 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__m512 al_in, r, r2, z, dn;
		__m512i ex_out;

		// c[0, 0-15]
		SWISH_F32_AVX512_DEF(c_float_0p0, selector1, al_in, r, r2, z, dn, ex_out);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_1xLT16_DISABLE:
	;

	if ( ( post_ops_attr.buf_downscale != NULL ) && ( post_ops_attr.is_last_k == TRUE ) )
		{
			__mmask16 mask_all1 = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

			// Store the results in downscaled type (int8 instead of int32).
			// c[0,0-15]
			CVT_STORE_F32_BF16_MASK(c_float_0p0,0,0);
		}

	else
	{
		__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

		// Store the results.
		// c[0,0-15]
		_mm512_mask_storeu_ps( c + ( rs_c * 0 ), load_mask, c_float_0p0 );
	}

}

// 5x16 bf16 kernel
LPGEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_5x16)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_5x16_DISABLE,
						  &&POST_OPS_BIAS_5x16,
						  &&POST_OPS_RELU_5x16,
						  &&POST_OPS_RELU_SCALE_5x16,
						  &&POST_OPS_GELU_TANH_5x16,
						  &&POST_OPS_GELU_ERF_5x16,
						  &&POST_OPS_CLIP_5x16,
						  &&POST_OPS_DOWNSCALE_5x16,
						  &&POST_OPS_MATRIX_ADD_5x16,
						  &&POST_OPS_SWISH_5x16,
						  &&POST_OPS_MATRIX_MUL_5x16
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int16_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();

	__m512 c_float_2p0 = _mm512_setzero_ps();

	__m512 c_float_3p0 = _mm512_setzero_ps();

	__m512 c_float_4p0 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );

		// Broadcast a[1,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 1 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );

		// Broadcast a[2,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 2 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-15] = a[2,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );

		// Broadcast a[3,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 3 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[3,0-15] = a[3,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );

		// Broadcast a[4,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 4 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[4,0-15] = a[4,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );
	}
	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		a_kfringe_buf = *( a + (rs_a * 0) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );

		// Broadcast a[1,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 1) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );

		// Broadcast a[2,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 2) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-15] = a[2,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );

		// Broadcast a[3,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 3) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[3,0-15] = a[3,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );

		// Broadcast a[4,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 4) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[4,0-15] = a[4,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );
	}

	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	if ( alpha != 1 )
	{
		// Scale by alpha
		c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );

		c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );

		c_float_2p0 = _mm512_mul_ps( selector1, c_float_2p0 );

		c_float_3p0 = _mm512_mul_ps( selector1, c_float_3p0 );

		c_float_4p0 = _mm512_mul_ps( selector1, c_float_4p0 );
	}

	// Scale C by beta.
	if ( beta != 0 )
	{
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
				 ( post_ops_attr.is_first_k == TRUE ) )
		{

			// c[0,0-15]
			BF16_F32_BETA_OP( c_float_0p0, 0, 0, 0, \
							selector1, selector2 );

			// c[1,0-15]
			BF16_F32_BETA_OP( c_float_1p0, 0, 1, 0, \
							selector1, selector2 );

			// c[2,0-15]
			BF16_F32_BETA_OP( c_float_2p0, 0, 2, 0, \
							selector1, selector2 );

			// c[3,0-15]
			BF16_F32_BETA_OP( c_float_3p0, 0, 3, 0, \
							selector1, selector2 );

			// c[4,0-15]
			BF16_F32_BETA_OP( c_float_4p0, 0, 4, 0, \
							selector1, selector2 );
		}
		else
		{
			// c[0,0-15]
			F32_F32_BETA_OP(c_float_0p0, 0, 0, 0, \
							selector1, selector2);

			// c[1,0-15]
			F32_F32_BETA_OP(c_float_1p0, 0, 1, 0, \
							selector1, selector2);

			// c[2,0-15]
			F32_F32_BETA_OP(c_float_2p0, 0, 2, 0, \
							selector1, selector2);

			// c[3,0-15]
			F32_F32_BETA_OP(c_float_3p0, 0, 3, 0, \
							selector1, selector2);

			// c[4,0-15]
			F32_F32_BETA_OP(c_float_4p0, 0, 4, 0, \
							selector1, selector2);
		}
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_5x16:
	{
		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_LOAD(selector1, bias_mask, 0);
			}
			else
			{
				selector1 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j );
			}

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

			// c[2,0-15]
			c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );

			// c[3,0-15]
			c_float_3p0 = _mm512_add_ps( selector1, c_float_3p0 );

			// c[4,0-15]
			c_float_4p0 = _mm512_add_ps( selector1, c_float_4p0 );
		}
		else
		{
			__m512 selector3;
			__m512 selector4;
			__m512 selector5;
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_BCAST(selector1, bias_mask, 0);
				BF16_F32_BIAS_BCAST(selector2, bias_mask, 1);
				BF16_F32_BIAS_BCAST(selector3, bias_mask, 2);
				BF16_F32_BIAS_BCAST(selector4, bias_mask, 3);
				BF16_F32_BIAS_BCAST(selector5, bias_mask, 4);
			}
			else
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 1 ) );
				selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 2 ) );
				selector4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 3 ) );
				selector5 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 4 ) );
			}

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );

			// c[2,0-15]
			c_float_2p0 = _mm512_add_ps( selector3, c_float_2p0 );

			// c[3,0-15]
			c_float_3p0 = _mm512_add_ps( selector4, c_float_3p0 );

			// c[4,0-15]
			c_float_4p0 = _mm512_add_ps( selector5, c_float_4p0 );
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_5x16:
	{
		selector1 = _mm512_setzero_ps();

		// c[0,0-15]
		c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

		// c[1,0-15]
		c_float_1p0 = _mm512_max_ps( selector1, c_float_1p0 );

		// c[2,0-15]
		c_float_2p0 = _mm512_max_ps( selector1, c_float_2p0 );

		// c[3,0-15]
		c_float_3p0 = _mm512_max_ps( selector1, c_float_3p0 );

		// c[4,0-15]
		c_float_4p0 = _mm512_max_ps( selector1, c_float_4p0 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_5x16:
	{
		selector1 = _mm512_setzero_ps();
		selector2 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__mmask16 relu_cmp_mask;

		// c[0, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_0p0)

		// c[1, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_1p0)

		// c[2, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_2p0)

		// c[3, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_3p0)

		// c[4, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_4p0)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_TANH_5x16:
	{
		__m512 dn, z, x, r2, r, x_tanh;
		__m512i q;

		// c[0, 0-15]
		GELU_TANH_F32_AVX512(c_float_0p0, r, r2, x, z, dn, x_tanh, q)

		// c[1, 0-15]
		GELU_TANH_F32_AVX512(c_float_1p0, r, r2, x, z, dn, x_tanh, q)

		// c[2, 0-15]
		GELU_TANH_F32_AVX512(c_float_2p0, r, r2, x, z, dn, x_tanh, q)

		// c[3, 0-15]
		GELU_TANH_F32_AVX512(c_float_3p0, r, r2, x, z, dn, x_tanh, q)

		// c[4, 0-15]
		GELU_TANH_F32_AVX512(c_float_4p0, r, r2, x, z, dn, x_tanh, q)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_ERF_5x16:
	{
		__m512 x, r, x_erf;

		// c[0, 0-15]
		GELU_ERF_F32_AVX512(c_float_0p0, r, x, x_erf)

		// c[1, 0-15]
		GELU_ERF_F32_AVX512(c_float_1p0, r, x, x_erf)

		// c[2, 0-15]
		GELU_ERF_F32_AVX512(c_float_2p0, r, x, x_erf)

		// c[3, 0-15]
		GELU_ERF_F32_AVX512(c_float_3p0, r, x, x_erf)

		// c[4, 0-15]
		GELU_ERF_F32_AVX512(c_float_4p0, r, x, x_erf)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_CLIP_5x16:
	{
		__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
		__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

		// c[0, 0-15]
		CLIP_F32_AVX512(c_float_0p0, min, max)

		// c[1, 0-15]
		CLIP_F32_AVX512(c_float_1p0, min, max)

		// c[2, 0-15]
		CLIP_F32_AVX512(c_float_2p0, min, max)

		// c[3, 0-15]
		CLIP_F32_AVX512(c_float_3p0, min, max)

		// c[4, 0-15]
		CLIP_F32_AVX512(c_float_4p0, min, max)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_DOWNSCALE_5x16:
	{
		__m512 selector3 = _mm512_setzero_ps();
		__m512 selector4 = _mm512_setzero_ps();
		__m512 selector5 = _mm512_setzero_ps();

		__m512 zero_point0 = _mm512_setzero_ps();
		__m512 zero_point1 = _mm512_setzero_ps();
		__m512 zero_point2 = _mm512_setzero_ps();
		__m512 zero_point3 = _mm512_setzero_ps();
		__m512 zero_point4 = _mm512_setzero_ps();

		__mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );

		// Need to account for row vs column major swaps. For scalars
		// scale and zero point, no implications.
		// Even though different registers are used for scalar in column
		// and row major downscale path, all those registers will contain
		// the same value.
		// Also the same value is loaded to different registers so that
		// branching can be reduced and same code/register can be used
		// irrespective of whether scalar or vector op.
		if ( post_ops_list_temp->scale_factor_len == 1 )
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector4 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector5 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
		}

		// bf16 zero point value (scalar or vector).
		if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
		{
			zero_point0 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point1 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point2 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point3 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point4 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
		}

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 = _mm512_maskz_loadu_ps( zp_mask,
						  ( float* )post_ops_list_temp->scale_factor +
						  post_ops_attr.post_op_c_j + ( 0 * 16 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector1,zero_point0);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector1,zero_point0);

			// c[3, 0-15]
			SCL_MULRND_F32(c_float_3p0,selector1,zero_point0);

			// c[4, 0-15]
			SCL_MULRND_F32(c_float_4p0,selector1,zero_point0);
		}
		else
		{
			// If original output was columns major, then by the time
			// kernel sees it, the matrix would be accessed as if it were
			// transposed. Due to this the scale as well as zp array will
			// be accessed by the ic index, and each scale/zp element
			// corresponds to an entire row of the transposed output array,
			// instead of an entire column.
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 1 ) );
				selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 2 ) );
				selector4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 3 ) );
				selector5 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 4 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 0 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 1 ) ) );
				zero_point2 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 2 ) ) );
				zero_point3 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 3 ) ) );
				zero_point4 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 4 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector2,zero_point1);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector3,zero_point2);

			// c[3, 0-15]
			SCL_MULRND_F32(c_float_3p0,selector4,zero_point3);

			// c[4, 0-15]
			SCL_MULRND_F32(c_float_4p0,selector5,zero_point4);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_ADD_5x16:
	{
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			BF16_F32_MATRIX_ADD_1COL(selector1,0);

			// c[1:0-15]
			BF16_F32_MATRIX_ADD_1COL(selector1,1);

			// c[2:0-15]
			BF16_F32_MATRIX_ADD_1COL(selector1,2);

			// c[3:0-15]
			BF16_F32_MATRIX_ADD_1COL(selector1,3);

			// c[4:0-15]
			BF16_F32_MATRIX_ADD_1COL(selector1,4);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			F32_F32_MATRIX_ADD_1COL(selector1,0);

			// c[1:0-15]
			F32_F32_MATRIX_ADD_1COL(selector1,1);

			// c[2:0-15]
			F32_F32_MATRIX_ADD_1COL(selector1,2);

			// c[3:0-15]
			F32_F32_MATRIX_ADD_1COL(selector1,3);

			// c[4:0-15]
			F32_F32_MATRIX_ADD_1COL(selector1,4);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_MUL_5x16:
	{
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			BF16_F32_MATRIX_MUL_1COL(selector1,0);

			// c[1:0-15]
			BF16_F32_MATRIX_MUL_1COL(selector1,1);

			// c[2:0-15]
			BF16_F32_MATRIX_MUL_1COL(selector1,2);

			// c[3:0-15]
			BF16_F32_MATRIX_MUL_1COL(selector1,3);

			// c[4:0-15]
			BF16_F32_MATRIX_MUL_1COL(selector1,4);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			F32_F32_MATRIX_MUL_1COL(selector1,0);

			// c[1:0-15]
			F32_F32_MATRIX_MUL_1COL(selector1,1);

			// c[2:0-15]
			F32_F32_MATRIX_MUL_1COL(selector1,2);

			// c[3:0-15]
			F32_F32_MATRIX_MUL_1COL(selector1,3);

			// c[4:0-15]
			F32_F32_MATRIX_MUL_1COL(selector1,4);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_SWISH_5x16:
	{
		selector1 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__m512 al_in, r, r2, z, dn;
		__m512i ex_out;

		// c[0, 0-15]
		SWISH_F32_AVX512_DEF(c_float_0p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[1, 0-15]
		SWISH_F32_AVX512_DEF(c_float_1p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[2, 0-15]
		SWISH_F32_AVX512_DEF(c_float_2p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[3, 0-15]
		SWISH_F32_AVX512_DEF(c_float_3p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[4, 0-15]
		SWISH_F32_AVX512_DEF(c_float_4p0, selector1, al_in, r, r2, z, dn, ex_out);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_5x16_DISABLE:
	;

	if ( ( post_ops_attr.buf_downscale != NULL ) && ( post_ops_attr.is_last_k == TRUE ) )
	{
		// Generate a mask16 of all 1's.
		__m512i selector_a = _mm512_setzero_epi32();
		__m512i selector_b = _mm512_set1_epi32( 10 );
		__mmask16 mask_all1 = _mm512_cmplt_epi32_mask( selector_a, selector_b );

		// Store the results in downscaled type (int8 instead of int32).
		// c[0,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_0p0,0,0);

		// c[1,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_1p0,1,0);

		// c[2,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_2p0,2,0);

		// c[3,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_3p0,3,0);

		// c[4,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_4p0,4,0);
	}

	else
	{
		// Store the results.
		// c[0,0-15]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 0*16 ), c_float_0p0 );

		// c[1,0-15]
		_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 0*16 ), c_float_1p0 );

		// c[2,0-15]
		_mm512_storeu_ps( c + ( rs_c * 2 ) + ( 0*16 ), c_float_2p0 );

		// c[3,0-15]
		_mm512_storeu_ps( c + ( rs_c * 3 ) + ( 0*16 ), c_float_3p0 );

		// c[4,0-15]
		_mm512_storeu_ps( c + ( rs_c * 4 ) + ( 0*16 ), c_float_4p0 );
	}
}

// 4x16 bf16 kernel
LPGEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_4x16)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_4x16_DISABLE,
						  &&POST_OPS_BIAS_4x16,
						  &&POST_OPS_RELU_4x16,
						  &&POST_OPS_RELU_SCALE_4x16,
						  &&POST_OPS_GELU_TANH_4x16,
						  &&POST_OPS_GELU_ERF_4x16,
						  &&POST_OPS_CLIP_4x16,
						  &&POST_OPS_DOWNSCALE_4x16,
						  &&POST_OPS_MATRIX_ADD_4x16,
						  &&POST_OPS_SWISH_4x16,
						  &&POST_OPS_MATRIX_MUL_4x16
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int16_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();

	__m512 c_float_2p0 = _mm512_setzero_ps();

	__m512 c_float_3p0 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );

		// Broadcast a[1,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 1 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );

		// Broadcast a[2,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 2 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-15] = a[2,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );

		// Broadcast a[3,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 3 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[3,0-15] = a[3,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );
	}
	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		a_kfringe_buf = *( a + (rs_a * 0) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );

		// Broadcast a[1,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 1) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );

		// Broadcast a[2,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 2) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-15] = a[2,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );

		// Broadcast a[3,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 3) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[3,0-15] = a[3,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );
	}

	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	if ( alpha != 1 )
	{
		// Scale by alpha
		c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );

		c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );

		c_float_2p0 = _mm512_mul_ps( selector1, c_float_2p0 );

		c_float_3p0 = _mm512_mul_ps( selector1, c_float_3p0 );
	}

	// Scale C by beta.
	if ( beta != 0 )
	{
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
				 ( post_ops_attr.is_first_k == TRUE ) )
		{

			// c[0,0-15]
			BF16_F32_BETA_OP( c_float_0p0, 0, 0, 0, \
							selector1, selector2 );

			// c[1,0-15]
			BF16_F32_BETA_OP( c_float_1p0, 0, 1, 0, \
							selector1, selector2 );

			// c[2,0-15]
			BF16_F32_BETA_OP( c_float_2p0, 0, 2, 0, \
							selector1, selector2 );

			// c[3,0-15]
			BF16_F32_BETA_OP( c_float_3p0, 0, 3, 0, \
							selector1, selector2 );
		}
		else
		{
			// c[0,0-15]
			F32_F32_BETA_OP(c_float_0p0, 0, 0, 0, \
							selector1, selector2);

			// c[1,0-15]
			F32_F32_BETA_OP(c_float_1p0, 0, 1, 0, \
							selector1, selector2);

			// c[2,0-15]
			F32_F32_BETA_OP(c_float_2p0, 0, 2, 0, \
							selector1, selector2);

			// c[3,0-15]
			F32_F32_BETA_OP(c_float_3p0, 0, 3, 0, \
							selector1, selector2);
		}
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_4x16:
	{
		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_LOAD(selector1, bias_mask, 0);
			}
			else
			{
				selector1 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j );
			}

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

			// c[2,0-15]
			c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );

			// c[3,0-15]
			c_float_3p0 = _mm512_add_ps( selector1, c_float_3p0 );
		}
		else
		{
			__m512 selector3;
			__m512 selector4;
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_BCAST(selector1, bias_mask, 0);
				BF16_F32_BIAS_BCAST(selector2, bias_mask, 1);
				BF16_F32_BIAS_BCAST(selector3, bias_mask, 2);
				BF16_F32_BIAS_BCAST(selector4, bias_mask, 3);
			}
			else
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 1 ) );
				selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 2 ) );
				selector4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 3 ) );
			}

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );

			// c[2,0-15]
			c_float_2p0 = _mm512_add_ps( selector3, c_float_2p0 );

			// c[3,0-15]
			c_float_3p0 = _mm512_add_ps( selector4, c_float_3p0 );
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_4x16:
	{
		selector1 = _mm512_setzero_ps();

		// c[0,0-15]
		c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

		// c[1,0-15]
		c_float_1p0 = _mm512_max_ps( selector1, c_float_1p0 );

		// c[2,0-15]
		c_float_2p0 = _mm512_max_ps( selector1, c_float_2p0 );

		// c[3,0-15]
		c_float_3p0 = _mm512_max_ps( selector1, c_float_3p0 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_4x16:
	{
		selector1 = _mm512_setzero_ps();
		selector2 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__mmask16 relu_cmp_mask;

		// c[0, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_0p0)

		// c[1, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_1p0)

		// c[2, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_2p0)

		// c[3, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_3p0)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_TANH_4x16:
	{
		__m512 dn, z, x, r2, r, x_tanh;
		__m512i q;

		// c[0, 0-15]
		GELU_TANH_F32_AVX512(c_float_0p0, r, r2, x, z, dn, x_tanh, q)

		// c[1, 0-15]
		GELU_TANH_F32_AVX512(c_float_1p0, r, r2, x, z, dn, x_tanh, q)

		// c[2, 0-15]
		GELU_TANH_F32_AVX512(c_float_2p0, r, r2, x, z, dn, x_tanh, q)

		// c[3, 0-15]
		GELU_TANH_F32_AVX512(c_float_3p0, r, r2, x, z, dn, x_tanh, q)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_ERF_4x16:
	{
		__m512 x, r, x_erf;

		// c[0, 0-15]
		GELU_ERF_F32_AVX512(c_float_0p0, r, x, x_erf)

		// c[1, 0-15]
		GELU_ERF_F32_AVX512(c_float_1p0, r, x, x_erf)

		// c[2, 0-15]
		GELU_ERF_F32_AVX512(c_float_2p0, r, x, x_erf)

		// c[3, 0-15]
		GELU_ERF_F32_AVX512(c_float_3p0, r, x, x_erf)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_CLIP_4x16:
	{
		__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
		__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

		// c[0, 0-15]
		CLIP_F32_AVX512(c_float_0p0, min, max)

		// c[1, 0-15]
		CLIP_F32_AVX512(c_float_1p0, min, max)

		// c[2, 0-15]
		CLIP_F32_AVX512(c_float_2p0, min, max)

		// c[3, 0-15]
		CLIP_F32_AVX512(c_float_3p0, min, max)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_DOWNSCALE_4x16:
	{
		__m512 selector3 = _mm512_setzero_ps();
		__m512 selector4 = _mm512_setzero_ps();

		__m512 zero_point0 = _mm512_setzero_ps();
		__m512 zero_point1 = _mm512_setzero_ps();
		__m512 zero_point2 = _mm512_setzero_ps();
		__m512 zero_point3 = _mm512_setzero_ps();

		__mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );

		// Need to account for row vs column major swaps. For scalars
		// scale and zero point, no implications.
		// Even though different registers are used for scalar in column
		// and row major downscale path, all those registers will contain
		// the same value.
		// Also the same value is loaded to different registers so that
		// branching can be reduced and same code/register can be used
		// irrespective of whether scalar or vector op.
		if ( post_ops_list_temp->scale_factor_len == 1 )
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector4 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
		}

		// bf16 zero point value (scalar or vector).
		if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
		{
			zero_point0 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point1 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point2 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point3 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
		}

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 = _mm512_maskz_loadu_ps( zp_mask,
						  ( float* )post_ops_list_temp->scale_factor +
						  post_ops_attr.post_op_c_j + ( 0 * 16 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector1,zero_point0);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector1,zero_point0);

			// c[3, 0-15]
			SCL_MULRND_F32(c_float_3p0,selector1,zero_point0);
		}
		else
		{
			// If original output was columns major, then by the time
			// kernel sees it, the matrix would be accessed as if it were
			// transposed. Due to this the scale as well as zp array will
			// be accessed by the ic index, and each scale/zp element
			// corresponds to an entire row of the transposed output array,
			// instead of an entire column.
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 1 ) );
				selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 2 ) );
				selector4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 3 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 0 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 1 ) ) );
				zero_point2 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 2 ) ) );
				zero_point3 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 3 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector2,zero_point1);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector3,zero_point2);

			// c[3, 0-15]
			SCL_MULRND_F32(c_float_3p0,selector4,zero_point3);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_ADD_4x16:
	{
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			BF16_F32_MATRIX_ADD_1COL(selector1,0);

			// c[1:0-15]
			BF16_F32_MATRIX_ADD_1COL(selector1,1);

			// c[2:0-15]
			BF16_F32_MATRIX_ADD_1COL(selector1,2);

			// c[3:0-15]
			BF16_F32_MATRIX_ADD_1COL(selector1,3);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			F32_F32_MATRIX_ADD_1COL(selector1,0);

			// c[1:0-15]
			F32_F32_MATRIX_ADD_1COL(selector1,1);

			// c[2:0-15]
			F32_F32_MATRIX_ADD_1COL(selector1,2);

			// c[3:0-15]
			F32_F32_MATRIX_ADD_1COL(selector1,3);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_MUL_4x16:
	{
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			BF16_F32_MATRIX_MUL_1COL(selector1,0);

			// c[1:0-15]
			BF16_F32_MATRIX_MUL_1COL(selector1,1);

			// c[2:0-15]
			BF16_F32_MATRIX_MUL_1COL(selector1,2);

			// c[3:0-15]
			BF16_F32_MATRIX_MUL_1COL(selector1,3);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			F32_F32_MATRIX_MUL_1COL(selector1,0);

			// c[1:0-15]
			F32_F32_MATRIX_MUL_1COL(selector1,1);

			// c[2:0-15]
			F32_F32_MATRIX_MUL_1COL(selector1,2);

			// c[3:0-15]
			F32_F32_MATRIX_MUL_1COL(selector1,3);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_SWISH_4x16:
	{
		selector1 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__m512 al_in, r, r2, z, dn;
		__m512i ex_out;

		// c[0, 0-15]
		SWISH_F32_AVX512_DEF(c_float_0p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[1, 0-15]
		SWISH_F32_AVX512_DEF(c_float_1p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[2, 0-15]
		SWISH_F32_AVX512_DEF(c_float_2p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[3, 0-15]
		SWISH_F32_AVX512_DEF(c_float_3p0, selector1, al_in, r, r2, z, dn, ex_out);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_4x16_DISABLE:
	;

	if ( ( post_ops_attr.buf_downscale != NULL ) && ( post_ops_attr.is_last_k == TRUE ) )
	{
		// Generate a mask16 of all 1's.
		__m512i selector_a = _mm512_setzero_epi32();
		__m512i selector_b = _mm512_set1_epi32( 10 );
		__mmask16 mask_all1 = _mm512_cmplt_epi32_mask( selector_a, selector_b );

		// Store the results in downscaled type (int8 instead of int32).
		// c[0,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_0p0,0,0);

		// c[1,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_1p0,1,0);

		// c[2,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_2p0,2,0);

		// c[3,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_3p0,3,0);
	}

	else
	{
		// Store the results.
		// c[0,0-15]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 0*16 ), c_float_0p0 );

		// c[1,0-15]
		_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 0*16 ), c_float_1p0 );

		// c[2,0-15]
		_mm512_storeu_ps( c + ( rs_c * 2 ) + ( 0*16 ), c_float_2p0 );

		// c[3,0-15]
		_mm512_storeu_ps( c + ( rs_c * 3 ) + ( 0*16 ), c_float_3p0 );
	}
}

// 3x16 bf16 kernel
LPGEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_3x16)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_3x16_DISABLE,
						  &&POST_OPS_BIAS_3x16,
						  &&POST_OPS_RELU_3x16,
						  &&POST_OPS_RELU_SCALE_3x16,
						  &&POST_OPS_GELU_TANH_3x16,
						  &&POST_OPS_GELU_ERF_3x16,
						  &&POST_OPS_CLIP_3x16,
						  &&POST_OPS_DOWNSCALE_3x16,
						  &&POST_OPS_MATRIX_ADD_3x16,
						  &&POST_OPS_SWISH_3x16,
						  &&POST_OPS_MATRIX_MUL_3x16
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int16_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();

	__m512 c_float_2p0 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );

		// Broadcast a[1,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 1 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );

		// Broadcast a[2,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 2 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-15] = a[2,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
	}
	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		a_kfringe_buf = *( a + (rs_a * 0) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );

		// Broadcast a[1,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 1) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );

		// Broadcast a[2,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 2) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-15] = a[2,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
	}

	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	if ( alpha != 1 )
	{
		// Scale by alpha
		c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );

		c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );

		c_float_2p0 = _mm512_mul_ps( selector1, c_float_2p0 );
	}

	// Scale C by beta.
	if ( beta != 0 )
	{
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
				 ( post_ops_attr.is_first_k == TRUE ) )
		{

			// c[0,0-15]
			BF16_F32_BETA_OP( c_float_0p0, 0, 0, 0, \
							selector1, selector2 );

			// c[1,0-15]
			BF16_F32_BETA_OP( c_float_1p0, 0, 1, 0, \
							selector1, selector2 );

			// c[2,0-15]
			BF16_F32_BETA_OP( c_float_2p0, 0, 2, 0, \
							selector1, selector2 );
		}
		else
		{
			// c[0,0-15]
			F32_F32_BETA_OP(c_float_0p0, 0, 0, 0, \
							selector1, selector2);

			// c[1,0-15]
			F32_F32_BETA_OP(c_float_1p0, 0, 1, 0, \
							selector1, selector2);

			// c[2,0-15]
			F32_F32_BETA_OP(c_float_2p0, 0, 2, 0, \
							selector1, selector2);
		}

	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_3x16:
	{
		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_LOAD(selector1, bias_mask, 0);
			}
			else
			{
				selector1 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j );
			}

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

			// c[2,0-15]
			c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );
		}
		else
		{
			__m512 selector3;
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_BCAST(selector1, bias_mask, 0);
				BF16_F32_BIAS_BCAST(selector2, bias_mask, 1);
				BF16_F32_BIAS_BCAST(selector3, bias_mask, 2);
			}
			else
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 1 ) );
				selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 2 ) );
			}

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );

			// c[2,0-15]
			c_float_2p0 = _mm512_add_ps( selector3, c_float_2p0 );
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_3x16:
	{
		selector1 = _mm512_setzero_ps();

		// c[0,0-15]
		c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

		// c[1,0-15]
		c_float_1p0 = _mm512_max_ps( selector1, c_float_1p0 );

		// c[2,0-15]
		c_float_2p0 = _mm512_max_ps( selector1, c_float_2p0 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_3x16:
	{
		selector1 = _mm512_setzero_ps();
		selector2 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__mmask16 relu_cmp_mask;

		// c[0, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_0p0)

		// c[1, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_1p0)

		// c[2, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_2p0)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_TANH_3x16:
	{
		__m512 dn, z, x, r2, r, x_tanh;
		__m512i q;

		// c[0, 0-15]
		GELU_TANH_F32_AVX512(c_float_0p0, r, r2, x, z, dn, x_tanh, q)

		// c[1, 0-15]
		GELU_TANH_F32_AVX512(c_float_1p0, r, r2, x, z, dn, x_tanh, q)

		// c[2, 0-15]
		GELU_TANH_F32_AVX512(c_float_2p0, r, r2, x, z, dn, x_tanh, q)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_ERF_3x16:
	{
		__m512 x, r, x_erf;

		// c[0, 0-15]
		GELU_ERF_F32_AVX512(c_float_0p0, r, x, x_erf)

		// c[1, 0-15]
		GELU_ERF_F32_AVX512(c_float_1p0, r, x, x_erf)

		// c[2, 0-15]
		GELU_ERF_F32_AVX512(c_float_2p0, r, x, x_erf)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_CLIP_3x16:
	{
		__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
		__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

		// c[0, 0-15]
		CLIP_F32_AVX512(c_float_0p0, min, max)

		// c[1, 0-15]
		CLIP_F32_AVX512(c_float_1p0, min, max)

		// c[2, 0-15]
		CLIP_F32_AVX512(c_float_2p0, min, max)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_DOWNSCALE_3x16:
	{
		__m512 selector3 = _mm512_setzero_ps();

		__m512 zero_point0 = _mm512_setzero_ps();
		__m512 zero_point1 = _mm512_setzero_ps();
		__m512 zero_point2 = _mm512_setzero_ps();

		__mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );

		// Need to account for row vs column major swaps. For scalars
		// scale and zero point, no implications.
		// Even though different registers are used for scalar in column
		// and row major downscale path, all those registers will contain
		// the same value.
		// Also the same value is loaded to different registers so that
		// branching can be reduced and same code/register can be used
		// irrespective of whether scalar or vector op.
		if ( post_ops_list_temp->scale_factor_len == 1 )
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
		}

		// bf16 zero point value (scalar or vector).
		if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
		{
			zero_point0 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point1 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point2 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
		}

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 = _mm512_maskz_loadu_ps( zp_mask,
						  ( float* )post_ops_list_temp->scale_factor +
						  post_ops_attr.post_op_c_j + ( 0 * 16 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector1,zero_point0);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector1,zero_point0);
		}
		else
		{
			// If original output was columns major, then by the time
			// kernel sees it, the matrix would be accessed as if it were
			// transposed. Due to this the scale as well as zp array will
			// be accessed by the ic index, and each scale/zp element
			// corresponds to an entire row of the transposed output array,
			// instead of an entire column.
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 1 ) );
				selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 2 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 0 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 1 ) ) );
				zero_point2 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 2 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector2,zero_point1);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector3,zero_point2);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_ADD_3x16:
	{
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			BF16_F32_MATRIX_ADD_1COL(selector1,0);

			// c[1:0-15]
			BF16_F32_MATRIX_ADD_1COL(selector1,1);

			// c[2:0-15]
			BF16_F32_MATRIX_ADD_1COL(selector1,2);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			F32_F32_MATRIX_ADD_1COL(selector1,0);

			// c[1:0-15]
			F32_F32_MATRIX_ADD_1COL(selector1,1);

			// c[2:0-15]
			F32_F32_MATRIX_ADD_1COL(selector1,2);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_MUL_3x16:
	{
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			BF16_F32_MATRIX_MUL_1COL(selector1,0);

			// c[1:0-15]
			BF16_F32_MATRIX_MUL_1COL(selector1,1);

			// c[2:0-15]
			BF16_F32_MATRIX_MUL_1COL(selector1,2);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			F32_F32_MATRIX_MUL_1COL(selector1,0);

			// c[1:0-15]
			F32_F32_MATRIX_MUL_1COL(selector1,1);

			// c[2:0-15]
			F32_F32_MATRIX_MUL_1COL(selector1,2);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}	
POST_OPS_SWISH_3x16:
	{
		selector1 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__m512 al_in, r, r2, z, dn;
		__m512i ex_out;

		// c[0, 0-15]
		SWISH_F32_AVX512_DEF(c_float_0p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[1, 0-15]
		SWISH_F32_AVX512_DEF(c_float_1p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[2, 0-15]
		SWISH_F32_AVX512_DEF(c_float_2p0, selector1, al_in, r, r2, z, dn, ex_out);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_3x16_DISABLE:
	;

	if ( ( post_ops_attr.buf_downscale != NULL ) && ( post_ops_attr.is_last_k == TRUE ) )
	{
		// Generate a mask16 of all 1's.
		__m512i selector_a = _mm512_setzero_epi32();
		__m512i selector_b = _mm512_set1_epi32( 10 );
		__mmask16 mask_all1 = _mm512_cmplt_epi32_mask( selector_a, selector_b );

		// Store the results in downscaled type (int8 instead of int32).
		// c[0,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_0p0,0,0);

		// c[1,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_1p0,1,0);

		// c[2,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_2p0,2,0);
	}

	else
	{
		// Store the results.
		// c[0,0-15]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 0*16 ), c_float_0p0 );

		// c[1,0-15]
		_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 0*16 ), c_float_1p0 );

		// c[2,0-15]
		_mm512_storeu_ps( c + ( rs_c * 2 ) + ( 0*16 ), c_float_2p0 );
	}
}

// 2x16 bf16 kernel
LPGEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_2x16)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_2x16_DISABLE,
						  &&POST_OPS_BIAS_2x16,
						  &&POST_OPS_RELU_2x16,
						  &&POST_OPS_RELU_SCALE_2x16,
						  &&POST_OPS_GELU_TANH_2x16,
						  &&POST_OPS_GELU_ERF_2x16,
						  &&POST_OPS_CLIP_2x16,
						  &&POST_OPS_DOWNSCALE_2x16,
						  &&POST_OPS_MATRIX_ADD_2x16,
						  &&POST_OPS_SWISH_2x16,
						  &&POST_OPS_MATRIX_MUL_2x16
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int16_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );

		// Broadcast a[1,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 1 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
	}
	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		a_kfringe_buf = *( a + (rs_a * 0) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );

		// Broadcast a[1,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 1) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
	}

	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	if ( alpha != 1 )
	{
		// Scale by alpha
		c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );

		c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );
	}

	// Scale C by beta.
	if ( beta != 0 )
	{
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
				 ( post_ops_attr.is_first_k == TRUE ) )
		{

			// c[0,0-15]
			BF16_F32_BETA_OP( c_float_0p0, 0, 0, 0, \
							selector1, selector2 );

			// c[1,0-15]
			BF16_F32_BETA_OP( c_float_1p0, 0, 1, 0, \
							selector1, selector2 );
		}
		else
		{
			// c[0,0-15]
			F32_F32_BETA_OP(c_float_0p0, 0, 0, 0, \
							selector1, selector2);

			// c[1,0-15]
			F32_F32_BETA_OP(c_float_1p0, 0, 1, 0, \
							selector1, selector2);
		}

	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_2x16:
	{
		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_LOAD(selector1, bias_mask, 0);
			}
			else
			{
				selector1 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j );
			}

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );
		}
		else
		{
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_BCAST(selector1, bias_mask, 0);
				BF16_F32_BIAS_BCAST(selector2, bias_mask, 1);
			}
			else
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 1 ) );
			}

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_2x16:
	{
		selector1 = _mm512_setzero_ps();

		// c[0,0-15]
		c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

		// c[1,0-15]
		c_float_1p0 = _mm512_max_ps( selector1, c_float_1p0 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_2x16:
	{
		selector1 = _mm512_setzero_ps();
		selector2 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__mmask16 relu_cmp_mask;

		// c[0, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_0p0)

		// c[1, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_1p0)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_TANH_2x16:
	{
		__m512 dn, z, x, r2, r, x_tanh;
		__m512i q;

		// c[0, 0-15]
		GELU_TANH_F32_AVX512(c_float_0p0, r, r2, x, z, dn, x_tanh, q)

		// c[1, 0-15]
		GELU_TANH_F32_AVX512(c_float_1p0, r, r2, x, z, dn, x_tanh, q)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_ERF_2x16:
	{
		__m512 x, r, x_erf;

		// c[0, 0-15]
		GELU_ERF_F32_AVX512(c_float_0p0, r, x, x_erf)

		// c[1, 0-15]
		GELU_ERF_F32_AVX512(c_float_1p0, r, x, x_erf)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_CLIP_2x16:
	{
		__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
		__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

		// c[0, 0-15]
		CLIP_F32_AVX512(c_float_0p0, min, max)

		// c[1, 0-15]
		CLIP_F32_AVX512(c_float_1p0, min, max)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}

POST_OPS_DOWNSCALE_2x16:
	{
		__m512 zero_point0 = _mm512_setzero_ps();
		__m512 zero_point1 = _mm512_setzero_ps();

		__mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );

		// Need to account for row vs column major swaps. For scalars
		// scale and zero point, no implications.
		// Even though different registers are used for scalar in column
		// and row major downscale path, all those registers will contain
		// the same value.
		// Also the same value is loaded to different registers so that
		// branching can be reduced and same code/register can be used
		// irrespective of whether scalar or vector op.
		if ( post_ops_list_temp->scale_factor_len == 1 )
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
		}

		// bf16 zero point value (scalar or vector).
		if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
		{
			zero_point0 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point1 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
		}

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 = _mm512_maskz_loadu_ps( zp_mask,
						  ( float* )post_ops_list_temp->scale_factor +
						  post_ops_attr.post_op_c_j + ( 0 * 16 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector1,zero_point0);
		}
		else
		{
			// If original output was columns major, then by the time
			// kernel sees it, the matrix would be accessed as if it were
			// transposed. Due to this the scale as well as zp array will
			// be accessed by the ic index, and each scale/zp element
			// corresponds to an entire row of the transposed output array,
			// instead of an entire column.
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 1 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 0 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 1 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector2,zero_point1);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_ADD_2x16:
	{
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			BF16_F32_MATRIX_ADD_1COL(selector1,0);

			// c[1:0-15]
			BF16_F32_MATRIX_ADD_1COL(selector1,1);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			F32_F32_MATRIX_ADD_1COL(selector1,0);

			// c[1:0-15]
			F32_F32_MATRIX_ADD_1COL(selector1,1);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_MUL_2x16:
	{
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			BF16_F32_MATRIX_MUL_1COL(selector1,0);

			// c[1:0-15]
			BF16_F32_MATRIX_MUL_1COL(selector1,1);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			F32_F32_MATRIX_MUL_1COL(selector1,0);

			// c[1:0-15]
			F32_F32_MATRIX_MUL_1COL(selector1,1);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_SWISH_2x16:
	{
		selector1 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__m512 al_in, r, r2, z, dn;
		__m512i ex_out;

		// c[0, 0-15]
		SWISH_F32_AVX512_DEF(c_float_0p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[1, 0-15]
		SWISH_F32_AVX512_DEF(c_float_1p0, selector1, al_in, r, r2, z, dn, ex_out);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_2x16_DISABLE:
	;

	if ( ( post_ops_attr.buf_downscale != NULL ) && ( post_ops_attr.is_last_k == TRUE ) )
	{
		// Generate a mask16 of all 1's.
		__m512i selector_a = _mm512_setzero_epi32();
		__m512i selector_b = _mm512_set1_epi32( 10 );
		__mmask16 mask_all1 = _mm512_cmplt_epi32_mask( selector_a, selector_b );

		// Store the results in downscaled type (int8 instead of int32).
		// c[0,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_0p0,0,0);

		// c[1,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_1p0,1,0);
	}

	else
	{
		// Store the results.
		// c[0,0-15]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 0*16 ), c_float_0p0 );

		// c[1,0-15]
		_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 0*16 ), c_float_1p0 );
	}
}

// 1x16 bf16 kernel
LPGEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_1x16)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_1x16_DISABLE,
						  &&POST_OPS_BIAS_1x16,
						  &&POST_OPS_RELU_1x16,
						  &&POST_OPS_RELU_SCALE_1x16,
						  &&POST_OPS_GELU_TANH_1x16,
						  &&POST_OPS_GELU_ERF_1x16,
						  &&POST_OPS_CLIP_1x16,
						  &&POST_OPS_DOWNSCALE_1x16,
						  &&POST_OPS_MATRIX_ADD_1x16,
						  &&POST_OPS_SWISH_1x16,
						  &&POST_OPS_MATRIX_MUL_1x16
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int16_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
	}
	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		a_kfringe_buf = *( a + (rs_a * 0) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
	}

	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	if ( alpha != 1 )
	{
		// Scale by alpha
		c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );
	}

	// Scale C by beta.
	if ( beta != 0 )
	{
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
				 ( post_ops_attr.is_first_k == TRUE ) )
		{

			// c[0,0-15]
			BF16_F32_BETA_OP( c_float_0p0, 0, 0, 0, \
							selector1, selector2 );
		}
		else
		{
			// c[0,0-15]
			F32_F32_BETA_OP(c_float_0p0, 0, 0, 0, \
							selector1, selector2);
		}

	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_1x16:
	{
		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_LOAD(selector1, bias_mask, 0);
			}
			else
			{
				selector1 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j );
			}

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );
		}
		else
		{
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_BCAST(selector1, bias_mask, 0);
			}
			else
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 0 ) );
			}

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_1x16:
	{
		selector1 = _mm512_setzero_ps();

		// c[0,0-15]
		c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_1x16:
	{
		selector1 = _mm512_setzero_ps();
		selector2 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__mmask16 relu_cmp_mask;

		// c[0, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_0p0)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_TANH_1x16:
	{
		__m512 dn, z, x, r2, r, x_tanh;
		__m512i q;

		// c[0, 0-15]
		GELU_TANH_F32_AVX512(c_float_0p0, r, r2, x, z, dn, x_tanh, q)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_ERF_1x16:
	{
		__m512 x, r, x_erf;

		// c[0, 0-15]
		GELU_ERF_F32_AVX512(c_float_0p0, r, x, x_erf)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_CLIP_1x16:
	{
		__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
		__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

		// c[0, 0-15]
		CLIP_F32_AVX512(c_float_0p0, min, max)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_DOWNSCALE_1x16:
	{
		__m512 zero_point0 = _mm512_setzero_ps();

		__mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );

		// Need to account for row vs column major swaps. For scalars
		// scale and zero point, no implications.
		// Even though different registers are used for scalar in column
		// and row major downscale path, all those registers will contain
		// the same value.
		// Also the same value is loaded to different registers so that
		// branching can be reduced and same code/register can be used
		// irrespective of whether scalar or vector op.
		if ( post_ops_list_temp->scale_factor_len == 1 )
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
		}

		// bf16 zero point value (scalar or vector).
		if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
		{
			zero_point0 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
		}

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 = _mm512_maskz_loadu_ps( zp_mask,
						  ( float* )post_ops_list_temp->scale_factor +
						  post_ops_attr.post_op_c_j + ( 0 * 16 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);
		}
		else
		{
			// If original output was columns major, then by the time
			// kernel sees it, the matrix would be accessed as if it were
			// transposed. Due to this the scale as well as zp array will
			// be accessed by the ic index, and each scale/zp element
			// corresponds to an entire row of the transposed output array,
			// instead of an entire column.
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 0 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 0 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_ADD_1x16:
	{
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			BF16_F32_MATRIX_ADD_1COL(selector1,0);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			F32_F32_MATRIX_ADD_1COL(selector1,0);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_MUL_1x16:
	{
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			BF16_F32_MATRIX_MUL_1COL(selector1,0);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15]
			F32_F32_MATRIX_MUL_1COL(selector1,0);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_SWISH_1x16:
	{
		selector1 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__m512 al_in, r, r2, z, dn;
		__m512i ex_out;

		// c[0, 0-15]
		SWISH_F32_AVX512_DEF(c_float_0p0, selector1, al_in, r, r2, z, dn, ex_out);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_1x16_DISABLE:
	;

	if ( ( post_ops_attr.buf_downscale != NULL ) && ( post_ops_attr.is_last_k == TRUE ) )
	{
		// Generate a mask16 of all 1's.
		__m512i selector_a = _mm512_setzero_epi32();
		__m512i selector_b = _mm512_set1_epi32( 10 );
		__mmask16 mask_all1 = _mm512_cmplt_epi32_mask( selector_a, selector_b );

		// Store the results in downscaled type (int8 instead of int32).
		// c[0,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_0p0,0,0);
	}
	else
	{
		// Store the results.
		// c[0,0-15]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 0*16 ), c_float_0p0 );
	}
}

// 5x32 bf16 kernel
LPGEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_5x32)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_5x32_DISABLE,
						  &&POST_OPS_BIAS_5x32,
						  &&POST_OPS_RELU_5x32,
						  &&POST_OPS_RELU_SCALE_5x32,
						  &&POST_OPS_GELU_TANH_5x32,
						  &&POST_OPS_GELU_ERF_5x32,
						  &&POST_OPS_CLIP_5x32,
						  &&POST_OPS_DOWNSCALE_5x32,
						  &&POST_OPS_MATRIX_ADD_5x32,
						  &&POST_OPS_SWISH_5x32,
						  &&POST_OPS_MATRIX_MUL_5x32
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int16_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;
	__m512bh b1;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();
	__m512 c_float_0p1 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();
	__m512 c_float_1p1 = _mm512_setzero_ps();

	__m512 c_float_2p0 = _mm512_setzero_ps();
	__m512 c_float_2p1 = _mm512_setzero_ps();

	__m512 c_float_3p0 = _mm512_setzero_ps();
	__m512 c_float_3p1 = _mm512_setzero_ps();

	__m512 c_float_4p0 = _mm512_setzero_ps();
	__m512 c_float_4p1 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 1 ) );

		// Broadcast a[0,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );

		// Broadcast a[1,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 1 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-31] = a[1,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );

		// Broadcast a[2,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 2 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-31] = a[2,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
		c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );

		// Broadcast a[3,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 3 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[3,0-31] = a[3,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );
		c_float_3p1 = _mm512_dpbf16_ps( c_float_3p1, a_bf16_0, b1 );

		// Broadcast a[4,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 4 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[4,0-31] = a[4,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );
		c_float_4p1 = _mm512_dpbf16_ps( c_float_4p1, a_bf16_0, b1 );
	}
	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 1 ) );

		// Broadcast a[0,kr:kr+2].
		a_kfringe_buf = *( a + (rs_a * 0) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );

		// Broadcast a[1,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 1) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-31] = a[1,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );

		// Broadcast a[2,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 2) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-31] = a[2,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
		c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );

		// Broadcast a[3,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 3) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[3,0-31] = a[3,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );
		c_float_3p1 = _mm512_dpbf16_ps( c_float_3p1, a_bf16_0, b1 );

		// Broadcast a[4,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 4) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[4,0-31] = a[4,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );
		c_float_4p1 = _mm512_dpbf16_ps( c_float_4p1, a_bf16_0, b1 );
	}

	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );\

	if ( alpha != 1 )
	{
		// Scale by alpha
		c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );
		c_float_0p1 = _mm512_mul_ps( selector1, c_float_0p1 );

		c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );
		c_float_1p1 = _mm512_mul_ps( selector1, c_float_1p1 );

		c_float_2p0 = _mm512_mul_ps( selector1, c_float_2p0 );
		c_float_2p1 = _mm512_mul_ps( selector1, c_float_2p1 );

		c_float_3p0 = _mm512_mul_ps( selector1, c_float_3p0 );
		c_float_3p1 = _mm512_mul_ps( selector1, c_float_3p1 );

		c_float_4p0 = _mm512_mul_ps( selector1, c_float_4p0 );
		c_float_4p1 = _mm512_mul_ps( selector1, c_float_4p1 );
	}

	// Scale C by beta.
	if ( beta != 0 )
	{
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
				 ( post_ops_attr.is_first_k == TRUE ) )
		{

			// c[0,0-15]
			BF16_F32_BETA_OP( c_float_0p0, 0, 0, 0, selector1, selector2 );

			// c[0, 16-31]
			BF16_F32_BETA_OP( c_float_0p1, 0, 0, 1, selector1, selector2 );

			// c[1,0-15]
			BF16_F32_BETA_OP( c_float_1p0, 0, 1, 0, selector1, selector2 );

			// c[1, 16-31]
			BF16_F32_BETA_OP( c_float_1p1, 0, 1, 1, selector1, selector2 );

			// c[2,0-15]
			BF16_F32_BETA_OP( c_float_2p0, 0, 2, 0, selector1, selector2 );

			// c[2, 16-31]
			BF16_F32_BETA_OP( c_float_2p1, 0, 2, 1, selector1, selector2 );

			// c[3,0-15]
			BF16_F32_BETA_OP( c_float_3p0, 0, 3, 0, selector1, selector2 );

			// c[3, 16-31]
			BF16_F32_BETA_OP( c_float_3p1, 0, 3, 1, selector1, selector2 );

			// c[4,0-15]
			BF16_F32_BETA_OP( c_float_4p0, 0, 4, 0, selector1, selector2 );

			// c[4, 16-31]
			BF16_F32_BETA_OP( c_float_4p1, 0, 4, 1, selector1, selector2 );
		}
		else
		{
			// c[0,0-15]
			F32_F32_BETA_OP( c_float_0p0, 0, 0, 0, selector1, selector2 );

			// c[0, 16-31]
			F32_F32_BETA_OP( c_float_0p1, 0, 0, 1, selector1, selector2 );

			// c[1,0-15]
			F32_F32_BETA_OP( c_float_1p0, 0, 1, 0, selector1, selector2 );

			// c[1, 16-31]
			F32_F32_BETA_OP( c_float_1p1, 0, 1, 1, selector1, selector2 );

			// c[2,0-15]
			F32_F32_BETA_OP( c_float_2p0, 0, 2, 0, selector1, selector2 );

			// c[2, 16-31]
			F32_F32_BETA_OP( c_float_2p1, 0, 2, 1, selector1, selector2 );

			// c[3,0-15]
			F32_F32_BETA_OP( c_float_3p0, 0, 3, 0, selector1, selector2 );

			// c[3, 16-31]
			F32_F32_BETA_OP( c_float_3p1, 0, 3, 1, selector1, selector2 );

			// c[4,0-15]
			F32_F32_BETA_OP( c_float_4p0, 0, 4, 0, selector1, selector2 );

			// c[4, 16-31]
			F32_F32_BETA_OP( c_float_4p1, 0, 4, 1, selector1, selector2 );
		}
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_5x32:
	{
		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_LOAD(selector1, bias_mask, 0);
				BF16_F32_BIAS_LOAD(selector2, bias_mask, 1);
			}
			else
			{
				selector1 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				selector2 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
			}

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_add_ps( selector2, c_float_0p1 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

			// c[1, 16-31]
			c_float_1p1 = _mm512_add_ps( selector2, c_float_1p1 );

			// c[2,0-15]
			c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );

			// c[2, 16-31]
			c_float_2p1 = _mm512_add_ps( selector2, c_float_2p1 );

			// c[3,0-15]
			c_float_3p0 = _mm512_add_ps( selector1, c_float_3p0 );

			// c[3, 16-31]
			c_float_3p1 = _mm512_add_ps( selector2, c_float_3p1 );

			// c[4,0-15]
			c_float_4p0 = _mm512_add_ps( selector1, c_float_4p0 );

			// c[4, 16-31]
			c_float_4p1 = _mm512_add_ps( selector2, c_float_4p1 );
		}
		else
		{
			__m512 selector3;
			__m512 selector4;
			__m512 selector5;
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_BCAST(selector1, bias_mask, 0);
				BF16_F32_BIAS_BCAST(selector2, bias_mask, 1);
				BF16_F32_BIAS_BCAST(selector3, bias_mask, 2);
				BF16_F32_BIAS_BCAST(selector4, bias_mask, 3);
				BF16_F32_BIAS_BCAST(selector5, bias_mask, 4);
			}
			else
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 1 ) );
				selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 2 ) );
				selector4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 3 ) );
				selector5 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 4 ) );
			}

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_add_ps( selector1, c_float_0p1 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );

			// c[1, 16-31]
			c_float_1p1 = _mm512_add_ps( selector2, c_float_1p1 );

			// c[2,0-15]
			c_float_2p0 = _mm512_add_ps( selector3, c_float_2p0 );

			// c[2, 16-31]
			c_float_2p1 = _mm512_add_ps( selector3, c_float_2p1 );

			// c[3,0-15]
			c_float_3p0 = _mm512_add_ps( selector4, c_float_3p0 );

			// c[3, 16-31]
			c_float_3p1 = _mm512_add_ps( selector4, c_float_3p1 );

			// c[4,0-15]
			c_float_4p0 = _mm512_add_ps( selector5, c_float_4p0 );

			// c[4, 16-31]
			c_float_4p1 = _mm512_add_ps( selector5, c_float_4p1 );
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_5x32:
	{
		selector1 = _mm512_setzero_ps();

		// c[0,0-15]
		c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

		// c[0, 16-31]
		c_float_0p1 = _mm512_max_ps( selector1, c_float_0p1 );

		// c[1,0-15]
		c_float_1p0 = _mm512_max_ps( selector1, c_float_1p0 );

		// c[1,16-31]
		c_float_1p1 = _mm512_max_ps( selector1, c_float_1p1 );

		// c[2,0-15]
		c_float_2p0 = _mm512_max_ps( selector1, c_float_2p0 );

		// c[2,16-31]
		c_float_2p1 = _mm512_max_ps( selector1, c_float_2p1 );

		// c[3,0-15]
		c_float_3p0 = _mm512_max_ps( selector1, c_float_3p0 );

		// c[3,16-31]
		c_float_3p1 = _mm512_max_ps( selector1, c_float_3p1 );

		// c[4,0-15]
		c_float_4p0 = _mm512_max_ps( selector1, c_float_4p0 );

		// c[4,16-31]
		c_float_4p1 = _mm512_max_ps( selector1, c_float_4p1 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_5x32:
	{
		selector1 = _mm512_setzero_ps();
		selector2 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__mmask16 relu_cmp_mask;

		// c[0, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_0p0)

		// c[0, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_0p1)

		// c[1, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_1p0)

		// c[1, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_1p1)

		// c[2, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_2p0)

		// c[2, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_2p1)

		// c[3, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_3p0)

		// c[3, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_3p1)

		// c[4, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_4p0)

		// c[4, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_4p1)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_TANH_5x32:
	{
		__m512 dn, z, x, r2, r, x_tanh;
		__m512i q;

		// c[0, 0-15]
		GELU_TANH_F32_AVX512(c_float_0p0, r, r2, x, z, dn, x_tanh, q)

		// c[0, 16-31]
		GELU_TANH_F32_AVX512(c_float_0p1, r, r2, x, z, dn, x_tanh, q)

		// c[1, 0-15]
		GELU_TANH_F32_AVX512(c_float_1p0, r, r2, x, z, dn, x_tanh, q)

		// c[1, 16-31]
		GELU_TANH_F32_AVX512(c_float_1p1, r, r2, x, z, dn, x_tanh, q)

		// c[2, 0-15]
		GELU_TANH_F32_AVX512(c_float_2p0, r, r2, x, z, dn, x_tanh, q)

		// c[2, 16-31]
		GELU_TANH_F32_AVX512(c_float_2p1, r, r2, x, z, dn, x_tanh, q)

		// c[3, 0-15]
		GELU_TANH_F32_AVX512(c_float_3p0, r, r2, x, z, dn, x_tanh, q)

		// c[3, 16-31]
		GELU_TANH_F32_AVX512(c_float_3p1, r, r2, x, z, dn, x_tanh, q)

		// c[4, 0-15]
		GELU_TANH_F32_AVX512(c_float_4p0, r, r2, x, z, dn, x_tanh, q)

		// c[4, 16-31]
		GELU_TANH_F32_AVX512(c_float_4p1, r, r2, x, z, dn, x_tanh, q)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_ERF_5x32:
	{
		__m512 x, r, x_erf;

		// c[0, 0-15]
		GELU_ERF_F32_AVX512(c_float_0p0, r, x, x_erf)

		// c[0, 16-31]
		GELU_ERF_F32_AVX512(c_float_0p1, r, x, x_erf)

		// c[1, 0-15]
		GELU_ERF_F32_AVX512(c_float_1p0, r, x, x_erf)

		// c[1, 16-31]
		GELU_ERF_F32_AVX512(c_float_1p1, r, x, x_erf)

		// c[2, 0-15]
		GELU_ERF_F32_AVX512(c_float_2p0, r, x, x_erf)

		// c[2, 16-31]
		GELU_ERF_F32_AVX512(c_float_2p1, r, x, x_erf)

		// c[3, 0-15]
		GELU_ERF_F32_AVX512(c_float_3p0, r, x, x_erf)

		// c[3, 16-31]
		GELU_ERF_F32_AVX512(c_float_3p1, r, x, x_erf)

		// c[4, 0-15]
		GELU_ERF_F32_AVX512(c_float_4p0, r, x, x_erf)

		// c[4, 16-31]
		GELU_ERF_F32_AVX512(c_float_4p1, r, x, x_erf)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_CLIP_5x32:
	{
		__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
		__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

		// c[0, 0-15]
		CLIP_F32_AVX512(c_float_0p0, min, max)

		// c[0, 16-31]
		CLIP_F32_AVX512(c_float_0p1, min, max)

		// c[1, 0-15]
		CLIP_F32_AVX512(c_float_1p0, min, max)

		// c[1, 16-31]
		CLIP_F32_AVX512(c_float_1p1, min, max)

		// c[2, 0-15]
		CLIP_F32_AVX512(c_float_2p0, min, max)

		// c[2, 16-31]
		CLIP_F32_AVX512(c_float_2p1, min, max)

		// c[3, 0-15]
		CLIP_F32_AVX512(c_float_3p0, min, max)

		// c[3, 16-31]
		CLIP_F32_AVX512(c_float_3p1, min, max)

		// c[4, 0-15]
		CLIP_F32_AVX512(c_float_4p0, min, max)

		// c[4, 16-31]
		CLIP_F32_AVX512(c_float_4p1, min, max)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_DOWNSCALE_5x32:
	{
		__m512 selector3 = _mm512_setzero_ps();
		__m512 selector4 = _mm512_setzero_ps();
		__m512 selector5 = _mm512_setzero_ps();

		__m512 zero_point0 = _mm512_setzero_ps();
		__m512 zero_point1 = _mm512_setzero_ps();
		__m512 zero_point2 = _mm512_setzero_ps();
		__m512 zero_point3 = _mm512_setzero_ps();
		__m512 zero_point4 = _mm512_setzero_ps();

		__mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );

		// Need to account for row vs column major swaps. For scalars
		// scale and zero point, no implications.
		// Even though different registers are used for scalar in column
		// and row major downscale path, all those registers will contain
		// the same value.
		if ( post_ops_list_temp->scale_factor_len == 1 )
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector4 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector5 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
		}

		// bf16 zero point value (scalar or vector).
		if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
		{
			zero_point0 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point1 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point2 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point3 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point4 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
		}

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				selector2 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[0, 16-31]
			SCL_MULRND_F32(c_float_0p1,selector2,zero_point1);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector1,zero_point0);

			// c[1, 16-31]
			SCL_MULRND_F32(c_float_1p1,selector2,zero_point1);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector1,zero_point0);

			// c[2, 16-31]
			SCL_MULRND_F32(c_float_2p1,selector2,zero_point1);

			// c[3, 0-15]
			SCL_MULRND_F32(c_float_3p0,selector1,zero_point0);

			// c[3, 16-31]
			SCL_MULRND_F32(c_float_3p1,selector2,zero_point1);

			// c[4, 0-15]
			SCL_MULRND_F32(c_float_4p0,selector1,zero_point0);

			// c[4, 16-31]
			SCL_MULRND_F32(c_float_4p1,selector2,zero_point1);
		}
		else
		{
			// If original output was columns major, then by the time
			// kernel sees it, the matrix would be accessed as if it were
			// transposed. Due to this the scale as well as zp array will
			// be accessed by the ic index, and each scale/zp element
			// corresponds to an entire row of the transposed output array,
			// instead of an entire column.
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 1 ) );
				selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 2 ) );
				selector4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 3 ) );
				selector5 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 4 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 0 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 1 ) ) );
				zero_point2 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 2 ) ) );
				zero_point3 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 3 ) ) );
				zero_point4 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 4 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[0, 16-31]
			SCL_MULRND_F32(c_float_0p1,selector1,zero_point0);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector2,zero_point1);

			// c[1, 16-31]
			SCL_MULRND_F32(c_float_1p1,selector2,zero_point1);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector3,zero_point2);

			// c[2, 16-31]
			SCL_MULRND_F32(c_float_2p1,selector3,zero_point2);

			// c[3, 0-15]
			SCL_MULRND_F32(c_float_3p0,selector4,zero_point3);

			// c[3, 16-31]
			SCL_MULRND_F32(c_float_3p1,selector4,zero_point3);

			// c[4, 0-15]
			SCL_MULRND_F32(c_float_4p0,selector5,zero_point4);

			// c[4, 16-31]
			SCL_MULRND_F32(c_float_4p1,selector5,zero_point4);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_ADD_5x32:
	{
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31]
			BF16_F32_MATRIX_ADD_2COL(selector1,selector2,0);

			// c[1:0-15,16-31]
			BF16_F32_MATRIX_ADD_2COL(selector1,selector2,1);

			// c[2:0-15,16-31]
			BF16_F32_MATRIX_ADD_2COL(selector1,selector2,2);

			// c[3:0-15,16-31]
			BF16_F32_MATRIX_ADD_2COL(selector1,selector2,3);

			// c[4:0-15,16-31]
			BF16_F32_MATRIX_ADD_2COL(selector1,selector2,4);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31]
			F32_F32_MATRIX_ADD_2COL(selector1,selector2,0);

			// c[1:0-15,16-31]
			F32_F32_MATRIX_ADD_2COL(selector1,selector2,1);

			// c[2:0-15,16-31]
			F32_F32_MATRIX_ADD_2COL(selector1,selector2,2);

			// c[3:0-15,16-31]
			F32_F32_MATRIX_ADD_2COL(selector1,selector2,3);

			// c[4:0-15,16-31]
			F32_F32_MATRIX_ADD_2COL(selector1,selector2,4);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_MUL_5x32:
	{
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31]
			BF16_F32_MATRIX_MUL_2COL(selector1,selector2,0);

			// c[1:0-15,16-31]
			BF16_F32_MATRIX_MUL_2COL(selector1,selector2,1);

			// c[2:0-15,16-31]
			BF16_F32_MATRIX_MUL_2COL(selector1,selector2,2);

			// c[3:0-15,16-31]
			BF16_F32_MATRIX_MUL_2COL(selector1,selector2,3);

			// c[4:0-15,16-31]
			BF16_F32_MATRIX_MUL_2COL(selector1,selector2,4);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31]
			F32_F32_MATRIX_MUL_2COL(selector1,selector2,0);

			// c[1:0-15,16-31]
			F32_F32_MATRIX_MUL_2COL(selector1,selector2,1);

			// c[2:0-15,16-31]
			F32_F32_MATRIX_MUL_2COL(selector1,selector2,2);

			// c[3:0-15,16-31]
			F32_F32_MATRIX_MUL_2COL(selector1,selector2,3);

			// c[4:0-15,16-31]
			F32_F32_MATRIX_MUL_2COL(selector1,selector2,4);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}	
POST_OPS_SWISH_5x32:
	{
		selector1 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__m512 al_in, r, r2, z, dn;
		__m512i ex_out;

		// c[0, 0-15]
		SWISH_F32_AVX512_DEF(c_float_0p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[0, 16-31]
		SWISH_F32_AVX512_DEF(c_float_0p1, selector1, al_in, r, r2, z, dn, ex_out);

		// c[1, 0-15]
		SWISH_F32_AVX512_DEF(c_float_1p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[1, 16-31]
		SWISH_F32_AVX512_DEF(c_float_1p1, selector1, al_in, r, r2, z, dn, ex_out);

		// c[2, 0-15]
		SWISH_F32_AVX512_DEF(c_float_2p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[2, 16-31]
		SWISH_F32_AVX512_DEF(c_float_2p1, selector1, al_in, r, r2, z, dn, ex_out);

		// c[3, 0-15]
		SWISH_F32_AVX512_DEF(c_float_3p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[3, 16-31]
		SWISH_F32_AVX512_DEF(c_float_3p1, selector1, al_in, r, r2, z, dn, ex_out);

		// c[4, 0-15]
		SWISH_F32_AVX512_DEF(c_float_4p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[4, 16-31]
		SWISH_F32_AVX512_DEF(c_float_4p1, selector1, al_in, r, r2, z, dn, ex_out);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_5x32_DISABLE:
	;

	if ( ( post_ops_attr.buf_downscale != NULL ) && ( post_ops_attr.is_last_k == TRUE ) )
	{
		// Generate a mask16 of all 1's.
		__m512i selector_a = _mm512_setzero_epi32();
		__m512i selector_b = _mm512_set1_epi32( 10 );
		__mmask16 mask_all1 = _mm512_cmplt_epi32_mask( selector_a, selector_b );

		// Store the results in downscaled type (int8 instead of int32).
		// c[0,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_0p0,0,0);

		// c[0, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_0p1,0,1);

		// c[1,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_1p0,1,0);

		// c[1, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_1p1,1,1);

		// c[2,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_2p0,2,0);

		// c[2, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_2p1,2,1);

		// c[3,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_3p0,3,0);

		// c[3, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_3p1,3,1);

		// c[4,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_4p0,4,0);

		// c[4, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_4p1,4,1);
	}

	else
	{
		// Store the results.
		// c[0,0-15]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 0*16 ), c_float_0p0 );

		// c[0, 16-31]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 1*16 ), c_float_0p1 );

		// c[1,0-15]
		_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 0*16 ), c_float_1p0 );

		// c[1,16-31]
		_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 1*16 ), c_float_1p1 );

		// c[2,0-15]
		_mm512_storeu_ps( c + ( rs_c * 2 ) + ( 0*16 ), c_float_2p0 );

		// c[2,16-31]
		_mm512_storeu_ps( c + ( rs_c * 2 ) + ( 1*16 ), c_float_2p1 );

		// c[3,0-15]
		_mm512_storeu_ps( c + ( rs_c * 3 ) + ( 0*16 ), c_float_3p0 );

		// c[3,16-31]
		_mm512_storeu_ps( c + ( rs_c * 3 ) + ( 1*16 ), c_float_3p1 );

		// c[4,0-15]
		_mm512_storeu_ps( c + ( rs_c * 4 ) + ( 0*16 ), c_float_4p0 );

		// c[4,16-31]
		_mm512_storeu_ps( c + ( rs_c * 4 ) + ( 1*16 ), c_float_4p1 );
	}
}

// 4x32 bf16 kernel
LPGEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_4x32)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_4x32_DISABLE,
						  &&POST_OPS_BIAS_4x32,
						  &&POST_OPS_RELU_4x32,
						  &&POST_OPS_RELU_SCALE_4x32,
						  &&POST_OPS_GELU_TANH_4x32,
						  &&POST_OPS_GELU_ERF_4x32,
						  &&POST_OPS_CLIP_4x32,
						  &&POST_OPS_DOWNSCALE_4x32,
						  &&POST_OPS_MATRIX_ADD_4x32,
						  &&POST_OPS_SWISH_4x32,
						  &&POST_OPS_MATRIX_MUL_4x32
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int16_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;
	__m512bh b1;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();
	__m512 c_float_0p1 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();
	__m512 c_float_1p1 = _mm512_setzero_ps();

	__m512 c_float_2p0 = _mm512_setzero_ps();
	__m512 c_float_2p1 = _mm512_setzero_ps();

	__m512 c_float_3p0 = _mm512_setzero_ps();
	__m512 c_float_3p1 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 1 ) );

		// Broadcast a[0,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );

		// Broadcast a[1,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 1 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-31] = a[1,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );

		// Broadcast a[2,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 2 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-31] = a[2,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
		c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );

		// Broadcast a[3,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 3 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[3,0-31] = a[3,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );
		c_float_3p1 = _mm512_dpbf16_ps( c_float_3p1, a_bf16_0, b1 );
	}
	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 1 ) );

		// Broadcast a[0,kr:kr+2].
		a_kfringe_buf = *( a + (rs_a * 0) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );

		// Broadcast a[1,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 1) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-31] = a[1,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );

		// Broadcast a[2,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 2) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-31] = a[2,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
		c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );

		// Broadcast a[3,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 3) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[3,0-31] = a[3,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );
		c_float_3p1 = _mm512_dpbf16_ps( c_float_3p1, a_bf16_0, b1 );
	}

	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	if ( alpha != 1 )
	{
		// Scale by alpha
		c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );
		c_float_0p1 = _mm512_mul_ps( selector1, c_float_0p1 );

		c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );
		c_float_1p1 = _mm512_mul_ps( selector1, c_float_1p1 );

		c_float_2p0 = _mm512_mul_ps( selector1, c_float_2p0 );
		c_float_2p1 = _mm512_mul_ps( selector1, c_float_2p1 );

		c_float_3p0 = _mm512_mul_ps( selector1, c_float_3p0 );
		c_float_3p1 = _mm512_mul_ps( selector1, c_float_3p1 );
	}

	// Scale C by beta.
	if ( beta != 0 )
	{
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
				 ( post_ops_attr.is_first_k == TRUE ) )
		{

			// c[0,0-15]
			BF16_F32_BETA_OP( c_float_0p0, 0, 0, 0, selector1, selector2 );

			// c[0, 16-31]
			BF16_F32_BETA_OP( c_float_0p1, 0, 0, 1, selector1, selector2 );

			// c[1,0-15]
			BF16_F32_BETA_OP( c_float_1p0, 0, 1, 0, selector1, selector2 );

			// c[1, 16-31]
			BF16_F32_BETA_OP( c_float_1p1, 0, 1, 1, selector1, selector2 );

			// c[2,0-15]
			BF16_F32_BETA_OP( c_float_2p0, 0, 2, 0, selector1, selector2 );

			// c[2, 16-31]
			BF16_F32_BETA_OP( c_float_2p1, 0, 2, 1, selector1, selector2 );

			// c[3,0-15]
			BF16_F32_BETA_OP( c_float_3p0, 0, 3, 0, selector1, selector2 );

			// c[3, 16-31]
			BF16_F32_BETA_OP( c_float_3p1, 0, 3, 1, selector1, selector2 );
		}
		else
		{
			// c[0,0-15]
			F32_F32_BETA_OP( c_float_0p0, 0, 0, 0, selector1, selector2 );

			// c[0, 16-31]
			F32_F32_BETA_OP( c_float_0p1, 0, 0, 1, selector1, selector2 );

			// c[1,0-15]
			F32_F32_BETA_OP( c_float_1p0, 0, 1, 0, selector1, selector2 );

			// c[1, 16-31]
			F32_F32_BETA_OP( c_float_1p1, 0, 1, 1, selector1, selector2 );

			// c[2,0-15]
			F32_F32_BETA_OP( c_float_2p0, 0, 2, 0, selector1, selector2 );

			// c[2, 16-31]
			F32_F32_BETA_OP( c_float_2p1, 0, 2, 1, selector1, selector2 );

			// c[3,0-15]
			F32_F32_BETA_OP( c_float_3p0, 0, 3, 0, selector1, selector2 );

			// c[3, 16-31]
			F32_F32_BETA_OP( c_float_3p1, 0, 3, 1, selector1, selector2 );
		}
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_4x32:
	{
		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_LOAD(selector1, bias_mask, 0);
				BF16_F32_BIAS_LOAD(selector2, bias_mask, 1);
			}
			else
			{
				selector1 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				selector2 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
			}

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_add_ps( selector2, c_float_0p1 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

			// c[1, 16-31]
			c_float_1p1 = _mm512_add_ps( selector2, c_float_1p1 );

			// c[2,0-15]
			c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );

			// c[2, 16-31]
			c_float_2p1 = _mm512_add_ps( selector2, c_float_2p1 );

			// c[3,0-15]
			c_float_3p0 = _mm512_add_ps( selector1, c_float_3p0 );

			// c[3, 16-31]
			c_float_3p1 = _mm512_add_ps( selector2, c_float_3p1 );
		}
		else
		{
			__m512 selector3;
			__m512 selector4;
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_BCAST(selector1, bias_mask, 0);
				BF16_F32_BIAS_BCAST(selector2, bias_mask, 1);
				BF16_F32_BIAS_BCAST(selector3, bias_mask, 2);
				BF16_F32_BIAS_BCAST(selector4, bias_mask, 3);
			}
			else
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 1 ) );
				selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 2 ) );
				selector4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 3 ) );
			}

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_add_ps( selector1, c_float_0p1 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );

			// c[1, 16-31]
			c_float_1p1 = _mm512_add_ps( selector2, c_float_1p1 );

			// c[2,0-15]
			c_float_2p0 = _mm512_add_ps( selector3, c_float_2p0 );

			// c[2, 16-31]
			c_float_2p1 = _mm512_add_ps( selector3, c_float_2p1 );

			// c[3,0-15]
			c_float_3p0 = _mm512_add_ps( selector4, c_float_3p0 );

			// c[3, 16-31]
			c_float_3p1 = _mm512_add_ps( selector4, c_float_3p1 );
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_4x32:
	{
		selector1 = _mm512_setzero_ps();

		// c[0,0-15]
		c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

		// c[0, 16-31]
		c_float_0p1 = _mm512_max_ps( selector1, c_float_0p1 );

		// c[1,0-15]
		c_float_1p0 = _mm512_max_ps( selector1, c_float_1p0 );

		// c[1,16-31]
		c_float_1p1 = _mm512_max_ps( selector1, c_float_1p1 );

		// c[2,0-15]
		c_float_2p0 = _mm512_max_ps( selector1, c_float_2p0 );

		// c[2,16-31]
		c_float_2p1 = _mm512_max_ps( selector1, c_float_2p1 );

		// c[3,0-15]
		c_float_3p0 = _mm512_max_ps( selector1, c_float_3p0 );

		// c[3,16-31]
		c_float_3p1 = _mm512_max_ps( selector1, c_float_3p1 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_4x32:
	{
		selector1 = _mm512_setzero_ps();
		selector2 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__mmask16 relu_cmp_mask;

		// c[0, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_0p0)

		// c[0, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_0p1)

		// c[1, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_1p0)

		// c[1, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_1p1)

		// c[2, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_2p0)

		// c[2, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_2p1)

		// c[3, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_3p0)

		// c[3, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_3p1)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_TANH_4x32:
	{
		__m512 dn, z, x, r2, r, x_tanh;
		__m512i q;

		// c[0, 0-15]
		GELU_TANH_F32_AVX512(c_float_0p0, r, r2, x, z, dn, x_tanh, q)

		// c[0, 16-31]
		GELU_TANH_F32_AVX512(c_float_0p1, r, r2, x, z, dn, x_tanh, q)

		// c[1, 0-15]
		GELU_TANH_F32_AVX512(c_float_1p0, r, r2, x, z, dn, x_tanh, q)

		// c[1, 16-31]
		GELU_TANH_F32_AVX512(c_float_1p1, r, r2, x, z, dn, x_tanh, q)

		// c[2, 0-15]
		GELU_TANH_F32_AVX512(c_float_2p0, r, r2, x, z, dn, x_tanh, q)

		// c[2, 16-31]
		GELU_TANH_F32_AVX512(c_float_2p1, r, r2, x, z, dn, x_tanh, q)

		// c[3, 0-15]
		GELU_TANH_F32_AVX512(c_float_3p0, r, r2, x, z, dn, x_tanh, q)

		// c[3, 16-31]
		GELU_TANH_F32_AVX512(c_float_3p1, r, r2, x, z, dn, x_tanh, q)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_ERF_4x32:
	{
		__m512 x, r, x_erf;

		// c[0, 0-15]
		GELU_ERF_F32_AVX512(c_float_0p0, r, x, x_erf)

		// c[0, 16-31]
		GELU_ERF_F32_AVX512(c_float_0p1, r, x, x_erf)

		// c[1, 0-15]
		GELU_ERF_F32_AVX512(c_float_1p0, r, x, x_erf)

		// c[1, 16-31]
		GELU_ERF_F32_AVX512(c_float_1p1, r, x, x_erf)

		// c[2, 0-15]
		GELU_ERF_F32_AVX512(c_float_2p0, r, x, x_erf)

		// c[2, 16-31]
		GELU_ERF_F32_AVX512(c_float_2p1, r, x, x_erf)

		// c[3, 0-15]
		GELU_ERF_F32_AVX512(c_float_3p0, r, x, x_erf)

		// c[3, 16-31]
		GELU_ERF_F32_AVX512(c_float_3p1, r, x, x_erf)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_CLIP_4x32:
	{
		__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
		__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

		// c[0, 0-15]
		CLIP_F32_AVX512(c_float_0p0, min, max)

		// c[0, 16-31]
		CLIP_F32_AVX512(c_float_0p1, min, max)

		// c[1, 0-15]
		CLIP_F32_AVX512(c_float_1p0, min, max)

		// c[1, 16-31]
		CLIP_F32_AVX512(c_float_1p1, min, max)

		// c[2, 0-15]
		CLIP_F32_AVX512(c_float_2p0, min, max)

		// c[2, 16-31]
		CLIP_F32_AVX512(c_float_2p1, min, max)

		// c[3, 0-15]
		CLIP_F32_AVX512(c_float_3p0, min, max)

		// c[3, 16-31]
		CLIP_F32_AVX512(c_float_3p1, min, max)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}

POST_OPS_DOWNSCALE_4x32:
	{
		__m512 selector3 = _mm512_setzero_ps();
		__m512 selector4 = _mm512_setzero_ps();

		__m512 zero_point0 = _mm512_setzero_ps();
		__m512 zero_point1 = _mm512_setzero_ps();
		__m512 zero_point2 = _mm512_setzero_ps();
		__m512 zero_point3 = _mm512_setzero_ps();

		__mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );

		// Need to account for row vs column major swaps. For scalars
		// scale and zero point, no implications.
		// Even though different registers are used for scalar in column
		// and row major downscale path, all those registers will contain
		// the same value.
		if ( post_ops_list_temp->scale_factor_len == 1 )
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector4 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
		}

		// bf16 zero point value (scalar or vector).
		if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
		{
			zero_point0 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point1 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point2 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point3 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
		}

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				selector2 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[0, 16-31]
			SCL_MULRND_F32(c_float_0p1,selector2,zero_point1);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector1,zero_point0);

			// c[1, 16-31]
			SCL_MULRND_F32(c_float_1p1,selector2,zero_point1);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector1,zero_point0);

			// c[2, 16-31]
			SCL_MULRND_F32(c_float_2p1,selector2,zero_point1);

			// c[3, 0-15]
			SCL_MULRND_F32(c_float_3p0,selector1,zero_point0);

			// c[3, 16-31]
			SCL_MULRND_F32(c_float_3p1,selector2,zero_point1);
		}
		else
		{
			// If original output was columns major, then by the time
			// kernel sees it, the matrix would be accessed as if it were
			// transposed. Due to this the scale as well as zp array will
			// be accessed by the ic index, and each scale/zp element
			// corresponds to an entire row of the transposed output array,
			// instead of an entire column.
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 1 ) );
				selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 2 ) );
				selector4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 3 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 0 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 1 ) ) );
				zero_point2 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 2 ) ) );
				zero_point3 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 3 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[0, 16-31]
			SCL_MULRND_F32(c_float_0p1,selector1,zero_point0);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector2,zero_point1);

			// c[1, 16-31]
			SCL_MULRND_F32(c_float_1p1,selector2,zero_point1);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector3,zero_point2);

			// c[2, 16-31]
			SCL_MULRND_F32(c_float_2p1,selector3,zero_point2);

			// c[3, 0-15]
			SCL_MULRND_F32(c_float_3p0,selector4,zero_point3);

			// c[3, 16-31]
			SCL_MULRND_F32(c_float_3p1,selector4,zero_point3);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_ADD_4x32:
	{
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31]
			BF16_F32_MATRIX_ADD_2COL(selector1,selector2,0);

			// c[1:0-15,16-31]
			BF16_F32_MATRIX_ADD_2COL(selector1,selector2,1);

			// c[2:0-15,16-31]
			BF16_F32_MATRIX_ADD_2COL(selector1,selector2,2);

			// c[3:0-15,16-31]
			BF16_F32_MATRIX_ADD_2COL(selector1,selector2,3);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31]
			F32_F32_MATRIX_ADD_2COL(selector1,selector2,0);

			// c[1:0-15,16-31]
			F32_F32_MATRIX_ADD_2COL(selector1,selector2,1);

			// c[2:0-15,16-31]
			F32_F32_MATRIX_ADD_2COL(selector1,selector2,2);

			// c[3:0-15,16-31]
			F32_F32_MATRIX_ADD_2COL(selector1,selector2,3);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_MUL_4x32:
	{
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31]
			BF16_F32_MATRIX_MUL_2COL(selector1,selector2,0);

			// c[1:0-15,16-31]
			BF16_F32_MATRIX_MUL_2COL(selector1,selector2,1);

			// c[2:0-15,16-31]
			BF16_F32_MATRIX_MUL_2COL(selector1,selector2,2);

			// c[3:0-15,16-31]
			BF16_F32_MATRIX_MUL_2COL(selector1,selector2,3);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31]
			F32_F32_MATRIX_MUL_2COL(selector1,selector2,0);

			// c[1:0-15,16-31]
			F32_F32_MATRIX_MUL_2COL(selector1,selector2,1);

			// c[2:0-15,16-31]
			F32_F32_MATRIX_MUL_2COL(selector1,selector2,2);

			// c[3:0-15,16-31]
			F32_F32_MATRIX_MUL_2COL(selector1,selector2,3);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_SWISH_4x32:
	{
		selector1 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__m512 al_in, r, r2, z, dn;
		__m512i ex_out;

		// c[0, 0-15]
		SWISH_F32_AVX512_DEF(c_float_0p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[0, 16-31]
		SWISH_F32_AVX512_DEF(c_float_0p1, selector1, al_in, r, r2, z, dn, ex_out);

		// c[1, 0-15]
		SWISH_F32_AVX512_DEF(c_float_1p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[1, 16-31]
		SWISH_F32_AVX512_DEF(c_float_1p1, selector1, al_in, r, r2, z, dn, ex_out);

		// c[2, 0-15]
		SWISH_F32_AVX512_DEF(c_float_2p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[2, 16-31]
		SWISH_F32_AVX512_DEF(c_float_2p1, selector1, al_in, r, r2, z, dn, ex_out);

		// c[3, 0-15]
		SWISH_F32_AVX512_DEF(c_float_3p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[3, 16-31]
		SWISH_F32_AVX512_DEF(c_float_3p1, selector1, al_in, r, r2, z, dn, ex_out);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_4x32_DISABLE:
	;

	if ( ( post_ops_attr.buf_downscale != NULL ) && ( post_ops_attr.is_last_k == TRUE ) )
	{
		// Generate a mask16 of all 1's.
		__m512i selector_a = _mm512_setzero_epi32();
		__m512i selector_b = _mm512_set1_epi32( 10 );
		__mmask16 mask_all1 = _mm512_cmplt_epi32_mask( selector_a, selector_b );

		// Store the results in downscaled type (int8 instead of int32).
		// c[0,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_0p0,0,0);

		// c[0, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_0p1,0,1);

		// c[1,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_1p0,1,0);

		// c[1, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_1p1,1,1);

		// c[2,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_2p0,2,0);

		// c[2, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_2p1,2,1);

		// c[3,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_3p0,3,0);

		// c[3, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_3p1,3,1);
	}

	else
	{
		// Store the results.
		// c[0,0-15]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 0*16 ), c_float_0p0 );

		// c[0, 16-31]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 1*16 ), c_float_0p1 );

		// c[1,0-15]
		_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 0*16 ), c_float_1p0 );

		// c[1,16-31]
		_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 1*16 ), c_float_1p1 );

		// c[2,0-15]
		_mm512_storeu_ps( c + ( rs_c * 2 ) + ( 0*16 ), c_float_2p0 );

		// c[2,16-31]
		_mm512_storeu_ps( c + ( rs_c * 2 ) + ( 1*16 ), c_float_2p1 );

		// c[3,0-15]
		_mm512_storeu_ps( c + ( rs_c * 3 ) + ( 0*16 ), c_float_3p0 );

		// c[3,16-31]
		_mm512_storeu_ps( c + ( rs_c * 3 ) + ( 1*16 ), c_float_3p1 );
	}
}

// 3x32 bf16 kernel
LPGEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_3x32)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_3x32_DISABLE,
						  &&POST_OPS_BIAS_3x32,
						  &&POST_OPS_RELU_3x32,
						  &&POST_OPS_RELU_SCALE_3x32,
						  &&POST_OPS_GELU_TANH_3x32,
						  &&POST_OPS_GELU_ERF_3x32,
						  &&POST_OPS_CLIP_3x32,
						  &&POST_OPS_DOWNSCALE_3x32,
						  &&POST_OPS_MATRIX_ADD_3x32,
						  &&POST_OPS_SWISH_3x32,
						  &&POST_OPS_MATRIX_MUL_3x32
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int16_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;
	__m512bh b1;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();
	__m512 c_float_0p1 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();
	__m512 c_float_1p1 = _mm512_setzero_ps();

	__m512 c_float_2p0 = _mm512_setzero_ps();
	__m512 c_float_2p1 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 1 ) );

		// Broadcast a[0,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );

		// Broadcast a[1,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 1 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-31] = a[1,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );

		// Broadcast a[2,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 2 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-31] = a[2,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
		c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );
	}
	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 1 ) );

		// Broadcast a[0,kr:kr+2].
		a_kfringe_buf = *( a + (rs_a * 0) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );

		// Broadcast a[1,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 1) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-31] = a[1,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );

		// Broadcast a[2,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 2) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-31] = a[2,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
		c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );
	}

	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	if ( alpha != 1 )
	{
		// Scale by alpha
		c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );
		c_float_0p1 = _mm512_mul_ps( selector1, c_float_0p1 );

		c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );
		c_float_1p1 = _mm512_mul_ps( selector1, c_float_1p1 );

		c_float_2p0 = _mm512_mul_ps( selector1, c_float_2p0 );
		c_float_2p1 = _mm512_mul_ps( selector1, c_float_2p1 );
	}

	// Scale C by beta.
	if ( beta != 0 )
	{
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
				 ( post_ops_attr.is_first_k == TRUE ) )
		{

			// c[0,0-15]
			BF16_F32_BETA_OP( c_float_0p0, 0, 0, 0, selector1, selector2 );

			// c[0, 16-31]
			BF16_F32_BETA_OP( c_float_0p1, 0, 0, 1, selector1, selector2 );

			// c[1,0-15]
			BF16_F32_BETA_OP( c_float_1p0, 0, 1, 0, selector1, selector2 );

			// c[1, 16-31]
			BF16_F32_BETA_OP( c_float_1p1, 0, 1, 1, selector1, selector2 );

			// c[2,0-15]
			BF16_F32_BETA_OP( c_float_2p0, 0, 2, 0, selector1, selector2 );

			// c[2, 16-31]
			BF16_F32_BETA_OP( c_float_2p1, 0, 2, 1, selector1, selector2 );
		}
		else
		{
			// c[0,0-15]
			F32_F32_BETA_OP( c_float_0p0, 0, 0, 0, selector1, selector2 );

			// c[0, 16-31]
			F32_F32_BETA_OP( c_float_0p1, 0, 0, 1, selector1, selector2 );

			// c[1,0-15]
			F32_F32_BETA_OP( c_float_1p0, 0, 1, 0, selector1, selector2 );

			// c[1, 16-31]
			F32_F32_BETA_OP( c_float_1p1, 0, 1, 1, selector1, selector2 );

			// c[2,0-15]
			F32_F32_BETA_OP( c_float_2p0, 0, 2, 0, selector1, selector2 );

			// c[2, 16-31]
			F32_F32_BETA_OP( c_float_2p1, 0, 2, 1, selector1, selector2 );
		}
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_3x32:
	{
		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_LOAD(selector1, bias_mask, 0);
				BF16_F32_BIAS_LOAD(selector2, bias_mask, 1);
			}
			else
			{
				selector1 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				selector2 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
			}

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_add_ps( selector2, c_float_0p1 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

			// c[1, 16-31]
			c_float_1p1 = _mm512_add_ps( selector2, c_float_1p1 );

			// c[2,0-15]
			c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );

			// c[2, 16-31]
			c_float_2p1 = _mm512_add_ps( selector2, c_float_2p1 );
		}
		else
		{
			__m512 selector3;
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_BCAST(selector1, bias_mask, 0);
				BF16_F32_BIAS_BCAST(selector2, bias_mask, 1);
				BF16_F32_BIAS_BCAST(selector3, bias_mask, 2);
			}
			else
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 1 ) );
				selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 2 ) );
			}

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_add_ps( selector1, c_float_0p1 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );

			// c[1, 16-31]
			c_float_1p1 = _mm512_add_ps( selector2, c_float_1p1 );

			// c[2,0-15]
			c_float_2p0 = _mm512_add_ps( selector3, c_float_2p0 );

			// c[2, 16-31]
			c_float_2p1 = _mm512_add_ps( selector3, c_float_2p1 );
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_3x32:
	{
		selector1 = _mm512_setzero_ps();

		// c[0,0-15]
		c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

		// c[0, 16-31]
		c_float_0p1 = _mm512_max_ps( selector1, c_float_0p1 );

		// c[1,0-15]
		c_float_1p0 = _mm512_max_ps( selector1, c_float_1p0 );

		// c[1,16-31]
		c_float_1p1 = _mm512_max_ps( selector1, c_float_1p1 );

		// c[2,0-15]
		c_float_2p0 = _mm512_max_ps( selector1, c_float_2p0 );

		// c[2,16-31]
		c_float_2p1 = _mm512_max_ps( selector1, c_float_2p1 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_3x32:
	{
		selector1 = _mm512_setzero_ps();
		selector2 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__mmask16 relu_cmp_mask;

		// c[0, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_0p0)

		// c[0, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_0p1)

		// c[1, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_1p0)

		// c[1, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_1p1)

		// c[2, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_2p0)

		// c[2, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_2p1)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_TANH_3x32:
	{
		__m512 dn, z, x, r2, r, x_tanh;
		__m512i q;

		// c[0, 0-15]
		GELU_TANH_F32_AVX512(c_float_0p0, r, r2, x, z, dn, x_tanh, q)

		// c[0, 16-31]
		GELU_TANH_F32_AVX512(c_float_0p1, r, r2, x, z, dn, x_tanh, q)

		// c[1, 0-15]
		GELU_TANH_F32_AVX512(c_float_1p0, r, r2, x, z, dn, x_tanh, q)

		// c[1, 16-31]
		GELU_TANH_F32_AVX512(c_float_1p1, r, r2, x, z, dn, x_tanh, q)

		// c[2, 0-15]
		GELU_TANH_F32_AVX512(c_float_2p0, r, r2, x, z, dn, x_tanh, q)

		// c[2, 16-31]
		GELU_TANH_F32_AVX512(c_float_2p1, r, r2, x, z, dn, x_tanh, q)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_ERF_3x32:
	{
		__m512 x, r, x_erf;

		// c[0, 0-15]
		GELU_ERF_F32_AVX512(c_float_0p0, r, x, x_erf)

		// c[0, 16-31]
		GELU_ERF_F32_AVX512(c_float_0p1, r, x, x_erf)

		// c[1, 0-15]
		GELU_ERF_F32_AVX512(c_float_1p0, r, x, x_erf)

		// c[1, 16-31]
		GELU_ERF_F32_AVX512(c_float_1p1, r, x, x_erf)

		// c[2, 0-15]
		GELU_ERF_F32_AVX512(c_float_2p0, r, x, x_erf)

		// c[2, 16-31]
		GELU_ERF_F32_AVX512(c_float_2p1, r, x, x_erf)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_CLIP_3x32:
	{
		__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
		__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

		// c[0, 0-15]
		CLIP_F32_AVX512(c_float_0p0, min, max)

		// c[0, 16-31]
		CLIP_F32_AVX512(c_float_0p1, min, max)

		// c[1, 0-15]
		CLIP_F32_AVX512(c_float_1p0, min, max)

		// c[1, 16-31]
		CLIP_F32_AVX512(c_float_1p1, min, max)

		// c[2, 0-15]
		CLIP_F32_AVX512(c_float_2p0, min, max)

		// c[2, 16-31]
		CLIP_F32_AVX512(c_float_2p1, min, max)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}

POST_OPS_DOWNSCALE_3x32:
	{
		__m512 selector3 = _mm512_setzero_ps();

		__m512 zero_point0 = _mm512_setzero_ps();
		__m512 zero_point1 = _mm512_setzero_ps();
		__m512 zero_point2 = _mm512_setzero_ps();

		__mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );

		// Need to account for row vs column major swaps. For scalars
		// scale and zero point, no implications.
		// Even though different registers are used for scalar in column
		// and row major downscale path, all those registers will contain
		// the same value.
		if ( post_ops_list_temp->scale_factor_len == 1 )
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
		}

		// bf16 zero point value (scalar or vector).
		if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
		{
			zero_point0 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point1 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point2 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
		}

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				selector2 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[0, 16-31]
			SCL_MULRND_F32(c_float_0p1,selector2,zero_point1);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector1,zero_point0);

			// c[1, 16-31]
			SCL_MULRND_F32(c_float_1p1,selector2,zero_point1);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector1,zero_point0);

			// c[2, 16-31]
			SCL_MULRND_F32(c_float_2p1,selector2,zero_point1);
		}
		else
		{
			// If original output was columns major, then by the time
			// kernel sees it, the matrix would be accessed as if it were
			// transposed. Due to this the scale as well as zp array will
			// be accessed by the ic index, and each scale/zp element
			// corresponds to an entire row of the transposed output array,
			// instead of an entire column.
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 1 ) );
				selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 2 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 0 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 1 ) ) );
				zero_point2 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 2 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[0, 16-31]
			SCL_MULRND_F32(c_float_0p1,selector1,zero_point0);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector2,zero_point1);

			// c[1, 16-31]
			SCL_MULRND_F32(c_float_1p1,selector2,zero_point1);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector3,zero_point2);

			// c[2, 16-31]
			SCL_MULRND_F32(c_float_2p1,selector3,zero_point2);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_ADD_3x32:
	{
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31]
			BF16_F32_MATRIX_ADD_2COL(selector1,selector2,0);

			// c[1:0-15,16-31]
			BF16_F32_MATRIX_ADD_2COL(selector1,selector2,1);

			// c[2:0-15,16-31]
			BF16_F32_MATRIX_ADD_2COL(selector1,selector2,2);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31]
			F32_F32_MATRIX_ADD_2COL(selector1,selector2,0);

			// c[1:0-15,16-31]
			F32_F32_MATRIX_ADD_2COL(selector1,selector2,1);

			// c[2:0-15,16-31]
			F32_F32_MATRIX_ADD_2COL(selector1,selector2,2);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_MUL_3x32:
	{
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31]
			BF16_F32_MATRIX_MUL_2COL(selector1,selector2,0);

			// c[1:0-15,16-31]
			BF16_F32_MATRIX_MUL_2COL(selector1,selector2,1);

			// c[2:0-15,16-31]
			BF16_F32_MATRIX_MUL_2COL(selector1,selector2,2);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31]
			F32_F32_MATRIX_MUL_2COL(selector1,selector2,0);

			// c[1:0-15,16-31]
			F32_F32_MATRIX_MUL_2COL(selector1,selector2,1);

			// c[2:0-15,16-31]
			F32_F32_MATRIX_MUL_2COL(selector1,selector2,2);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_SWISH_3x32:
	{
		selector1 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__m512 al_in, r, r2, z, dn;
		__m512i ex_out;

		// c[0, 0-15]
		SWISH_F32_AVX512_DEF(c_float_0p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[0, 16-31]
		SWISH_F32_AVX512_DEF(c_float_0p1, selector1, al_in, r, r2, z, dn, ex_out);

		// c[1, 0-15]
		SWISH_F32_AVX512_DEF(c_float_1p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[1, 16-31]
		SWISH_F32_AVX512_DEF(c_float_1p1, selector1, al_in, r, r2, z, dn, ex_out);

		// c[2, 0-15]
		SWISH_F32_AVX512_DEF(c_float_2p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[2, 16-31]
		SWISH_F32_AVX512_DEF(c_float_2p1, selector1, al_in, r, r2, z, dn, ex_out);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_3x32_DISABLE:
	;

	if ( ( post_ops_attr.buf_downscale != NULL ) && ( post_ops_attr.is_last_k == TRUE ) )
	{
		// Generate a mask16 of all 1's.
		__m512i selector_a = _mm512_setzero_epi32();
		__m512i selector_b = _mm512_set1_epi32( 10 );
		__mmask16 mask_all1 = _mm512_cmplt_epi32_mask( selector_a, selector_b );

		// Store the results in downscaled type (int8 instead of int32).
		// c[0,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_0p0,0,0);

		// c[0, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_0p1,0,1);

		// c[1,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_1p0,1,0);

		// c[1, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_1p1,1,1);

		// c[2,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_2p0,2,0);

		// c[2, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_2p1,2,1);
	}

	else
	{
		// Store the results.
		// c[0,0-15]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 0*16 ), c_float_0p0 );

		// c[0, 16-31]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 1*16 ), c_float_0p1 );

		// c[1,0-15]
		_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 0*16 ), c_float_1p0 );

		// c[1,16-31]
		_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 1*16 ), c_float_1p1 );

		// c[2,0-15]
		_mm512_storeu_ps( c + ( rs_c * 2 ) + ( 0*16 ), c_float_2p0 );

		// c[2,16-31]
		_mm512_storeu_ps( c + ( rs_c * 2 ) + ( 1*16 ), c_float_2p1 );
	}
}

// 2x32 bf16 kernel
LPGEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_2x32)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_2x32_DISABLE,
						  &&POST_OPS_BIAS_2x32,
						  &&POST_OPS_RELU_2x32,
						  &&POST_OPS_RELU_SCALE_2x32,
						  &&POST_OPS_GELU_TANH_2x32,
						  &&POST_OPS_GELU_ERF_2x32,
						  &&POST_OPS_CLIP_2x32,
						  &&POST_OPS_DOWNSCALE_2x32,
						  &&POST_OPS_MATRIX_ADD_2x32,
						  &&POST_OPS_SWISH_2x32,
						  &&POST_OPS_MATRIX_MUL_2x32
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int16_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;
	__m512bh b1;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();
	__m512 c_float_0p1 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();
	__m512 c_float_1p1 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 1 ) );

		// Broadcast a[0,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );

		// Broadcast a[1,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 1 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-31] = a[1,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );
	}
	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 1 ) );

		// Broadcast a[0,kr:kr+2].
		a_kfringe_buf = *( a + (rs_a * 0) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );

		// Broadcast a[1,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 1) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-31] = a[1,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );
	}

	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	if ( alpha != 1 )
	{
		// Scale by alpha
		c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );
		c_float_0p1 = _mm512_mul_ps( selector1, c_float_0p1 );

		c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );
		c_float_1p1 = _mm512_mul_ps( selector1, c_float_1p1 );
	}

	// Scale C by beta.
	if ( beta != 0 )
	{
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
				 ( post_ops_attr.is_first_k == TRUE ) )
		{

			// c[0,0-15]
			BF16_F32_BETA_OP( c_float_0p0, 0, 0, 0, selector1, selector2 );

			// c[0, 16-31]
			BF16_F32_BETA_OP( c_float_0p1, 0, 0, 1, selector1, selector2 );

			// c[1,0-15]
			BF16_F32_BETA_OP( c_float_1p0, 0, 1, 0, selector1, selector2 );

			// c[1, 16-31]
			BF16_F32_BETA_OP( c_float_1p1, 0, 1, 1, selector1, selector2 );
		}
		else
		{
			// c[0,0-15]
			F32_F32_BETA_OP( c_float_0p0, 0, 0, 0, selector1, selector2 );

			// c[0, 16-31]
			F32_F32_BETA_OP( c_float_0p1, 0, 0, 1, selector1, selector2 );

			// c[1,0-15]
			F32_F32_BETA_OP( c_float_1p0, 0, 1, 0, selector1, selector2 );

			// c[1, 16-31]
			F32_F32_BETA_OP( c_float_1p1, 0, 1, 1, selector1, selector2 );
		}
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_2x32:
	{
		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_LOAD(selector1, bias_mask, 0);
				BF16_F32_BIAS_LOAD(selector2, bias_mask, 1);
			}
			else
			{
				selector1 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				selector2 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
			}

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_add_ps( selector2, c_float_0p1 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

			// c[1, 16-31]
			c_float_1p1 = _mm512_add_ps( selector2, c_float_1p1 );
		}
		else
		{
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_BCAST(selector1, bias_mask, 0);
				BF16_F32_BIAS_BCAST(selector2, bias_mask, 1);
			}
			else
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 1 ) );
			}

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_add_ps( selector1, c_float_0p1 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );

			// c[1, 16-31]
			c_float_1p1 = _mm512_add_ps( selector2, c_float_1p1 );
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_2x32:
	{
		selector1 = _mm512_setzero_ps();

		// c[0,0-15]
		c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

		// c[0, 16-31]
		c_float_0p1 = _mm512_max_ps( selector1, c_float_0p1 );

		// c[1,0-15]
		c_float_1p0 = _mm512_max_ps( selector1, c_float_1p0 );

		// c[1,16-31]
		c_float_1p1 = _mm512_max_ps( selector1, c_float_1p1 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_2x32:
	{
		selector1 = _mm512_setzero_ps();
		selector2 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__mmask16 relu_cmp_mask;

		// c[0, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_0p0)

		// c[0, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_0p1)

		// c[1, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_1p0)

		// c[1, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_1p1)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_TANH_2x32:
	{
		__m512 dn, z, x, r2, r, x_tanh;
		__m512i q;

		// c[0, 0-15]
		GELU_TANH_F32_AVX512(c_float_0p0, r, r2, x, z, dn, x_tanh, q)

		// c[0, 16-31]
		GELU_TANH_F32_AVX512(c_float_0p1, r, r2, x, z, dn, x_tanh, q)

		// c[1, 0-15]
		GELU_TANH_F32_AVX512(c_float_1p0, r, r2, x, z, dn, x_tanh, q)

		// c[1, 16-31]
		GELU_TANH_F32_AVX512(c_float_1p1, r, r2, x, z, dn, x_tanh, q)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_ERF_2x32:
	{
		__m512 x, r, x_erf;

		// c[0, 0-15]
		GELU_ERF_F32_AVX512(c_float_0p0, r, x, x_erf)

		// c[0, 16-31]
		GELU_ERF_F32_AVX512(c_float_0p1, r, x, x_erf)

		// c[1, 0-15]
		GELU_ERF_F32_AVX512(c_float_1p0, r, x, x_erf)

		// c[1, 16-31]
		GELU_ERF_F32_AVX512(c_float_1p1, r, x, x_erf)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_CLIP_2x32:
	{
		__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
		__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

		// c[0, 0-15]
		CLIP_F32_AVX512(c_float_0p0, min, max)

		// c[0, 16-31]
		CLIP_F32_AVX512(c_float_0p1, min, max)

		// c[1, 0-15]
		CLIP_F32_AVX512(c_float_1p0, min, max)

		// c[1, 16-31]
		CLIP_F32_AVX512(c_float_1p1, min, max)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}

POST_OPS_DOWNSCALE_2x32:
	{
		__m512 zero_point0 = _mm512_setzero_ps();
		__m512 zero_point1 = _mm512_setzero_ps();

		__mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );

		// Need to account for row vs column major swaps. For scalars
		// scale and zero point, no implications.
		// Even though different registers are used for scalar in column
		// and row major downscale path, all those registers will contain
		// the same value.
		if ( post_ops_list_temp->scale_factor_len == 1 )
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
		}

		// bf16 zero point value (scalar or vector).
		if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
		{
			zero_point0 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point1 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
		}

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				selector2 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[0, 16-31]
			SCL_MULRND_F32(c_float_0p1,selector2,zero_point1);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector1,zero_point0);

			// c[1, 16-31]
			SCL_MULRND_F32(c_float_1p1,selector2,zero_point1);
		}
		else
		{
			// If original output was columns major, then by the time
			// kernel sees it, the matrix would be accessed as if it were
			// transposed. Due to this the scale as well as zp array will
			// be accessed by the ic index, and each scale/zp element
			// corresponds to an entire row of the transposed output array,
			// instead of an entire column.
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 1 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 0 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 1 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[0, 16-31]
			SCL_MULRND_F32(c_float_0p1,selector1,zero_point0);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector2,zero_point1);

			// c[1, 16-31]
			SCL_MULRND_F32(c_float_1p1,selector2,zero_point1);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_ADD_2x32:
	{
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31]
			BF16_F32_MATRIX_ADD_2COL(selector1,selector2,0);

			// c[1:0-15,16-31]
			BF16_F32_MATRIX_ADD_2COL(selector1,selector2,1);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31]
			F32_F32_MATRIX_ADD_2COL(selector1,selector2,0);

			// c[1:0-15,16-31]
			F32_F32_MATRIX_ADD_2COL(selector1,selector2,1);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_MUL_2x32:
	{
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31]
			BF16_F32_MATRIX_MUL_2COL(selector1,selector2,0);

			// c[1:0-15,16-31]
			BF16_F32_MATRIX_MUL_2COL(selector1,selector2,1);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31]
			F32_F32_MATRIX_MUL_2COL(selector1,selector2,0);

			// c[1:0-15,16-31]
			F32_F32_MATRIX_MUL_2COL(selector1,selector2,1);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_SWISH_2x32:
	{
		selector1 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__m512 al_in, r, r2, z, dn;
		__m512i ex_out;

		// c[0, 0-15]
		SWISH_F32_AVX512_DEF(c_float_0p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[0, 16-31]
		SWISH_F32_AVX512_DEF(c_float_0p1, selector1, al_in, r, r2, z, dn, ex_out);

		// c[1, 0-15]
		SWISH_F32_AVX512_DEF(c_float_1p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[1, 16-31]
		SWISH_F32_AVX512_DEF(c_float_1p1, selector1, al_in, r, r2, z, dn, ex_out);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_2x32_DISABLE:
	;

	if ( ( post_ops_attr.buf_downscale != NULL ) && ( post_ops_attr.is_last_k == TRUE ) )
	{
		// Generate a mask16 of all 1's.
		__m512i selector_a = _mm512_setzero_epi32();
		__m512i selector_b = _mm512_set1_epi32( 10 );
		__mmask16 mask_all1 = _mm512_cmplt_epi32_mask( selector_a, selector_b );

		// Store the results in downscaled type (int8 instead of int32).
		// c[0,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_0p0,0,0);

		// c[0, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_0p1,0,1);

		// c[1,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_1p0,1,0);

		// c[1, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_1p1,1,1);
	}
	else
	{
		// Store the results.
		// c[0,0-15]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 0*16 ), c_float_0p0 );

		// c[0, 16-31]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 1*16 ), c_float_0p1 );

		// c[1,0-15]
		_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 0*16 ), c_float_1p0 );

		// c[1,16-31]
		_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 1*16 ), c_float_1p1 );
	}
}

// 1x32 bf16 kernel
LPGEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_1x32)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_1x32_DISABLE,
						  &&POST_OPS_BIAS_1x32,
						  &&POST_OPS_RELU_1x32,
						  &&POST_OPS_RELU_SCALE_1x32,
						  &&POST_OPS_GELU_TANH_1x32,
						  &&POST_OPS_GELU_ERF_1x32,
						  &&POST_OPS_CLIP_1x32,
						  &&POST_OPS_DOWNSCALE_1x32,
						  &&POST_OPS_MATRIX_ADD_1x32,
						  &&POST_OPS_SWISH_1x32,
						  &&POST_OPS_MATRIX_MUL_1x32
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int16_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;
	__m512bh b1;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();
	__m512 c_float_0p1 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 1 ) );

		// Broadcast a[0,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
	}
	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 1 ) );

		// Broadcast a[0,kr:kr+2].
		a_kfringe_buf = *( a + (rs_a * 0) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
	}

	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	if ( alpha != 1 )
	{
		// Scale by alpha
		c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );
		c_float_0p1 = _mm512_mul_ps( selector1, c_float_0p1 );
	}

	// Scale C by beta.
	if ( beta != 0 )
	{
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
				 ( post_ops_attr.is_first_k == TRUE ) )
		{

			// c[0,0-15]
			BF16_F32_BETA_OP( c_float_0p0, 0, 0, 0, selector1, selector2 );

			// c[0, 16-31]
			BF16_F32_BETA_OP( c_float_0p1, 0, 0, 1, selector1, selector2 );
		}
		else
		{
			// c[0,0-15]
			F32_F32_BETA_OP( c_float_0p0, 0, 0, 0, selector1, selector2 );

			// c[0, 16-31]
			F32_F32_BETA_OP( c_float_0p1, 0, 0, 1, selector1, selector2 );
		}
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_1x32:
	{
		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_LOAD(selector1, bias_mask, 0);
				BF16_F32_BIAS_LOAD(selector2, bias_mask, 1);
			}
			else
			{
				selector1 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				selector2 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
			}

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_add_ps( selector2, c_float_0p1 );
		}
		else
		{
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_BCAST(selector1, bias_mask, 0);
			}
			else
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 0 ) );
			}

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_add_ps( selector1, c_float_0p1 );
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_1x32:
	{
		selector1 = _mm512_setzero_ps();

		// c[0,0-15]
		c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

		// c[0, 16-31]
		c_float_0p1 = _mm512_max_ps( selector1, c_float_0p1 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_1x32:
	{
		selector1 = _mm512_setzero_ps();
		selector2 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__mmask16 relu_cmp_mask;

		// c[0, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_0p0)

		// c[0, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_0p1)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_TANH_1x32:
	{
		__m512 dn, z, x, r2, r, x_tanh;
		__m512i q;

		// c[0, 0-15]
		GELU_TANH_F32_AVX512(c_float_0p0, r, r2, x, z, dn, x_tanh, q)

		// c[0, 16-31]
		GELU_TANH_F32_AVX512(c_float_0p1, r, r2, x, z, dn, x_tanh, q)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_ERF_1x32:
	{
		__m512 x, r, x_erf;

		// c[0, 0-15]
		GELU_ERF_F32_AVX512(c_float_0p0, r, x, x_erf)

		// c[0, 16-31]
		GELU_ERF_F32_AVX512(c_float_0p1, r, x, x_erf)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_CLIP_1x32:
	{
		__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
		__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

		// c[0, 0-15]
		CLIP_F32_AVX512(c_float_0p0, min, max)

		// c[0, 16-31]
		CLIP_F32_AVX512(c_float_0p1, min, max)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}

POST_OPS_DOWNSCALE_1x32:
	{
		__m512 zero_point0 = _mm512_setzero_ps();
		__m512 zero_point1 = _mm512_setzero_ps();

		__mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );

		// Need to account for row vs column major swaps. For scalars
		// scale and zero point, no implications.
		// Even though different registers are used for scalar in column
		// and row major downscale path, all those registers will contain
		// the same value.
		if ( post_ops_list_temp->scale_factor_len == 1 )
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
		}

		// bf16 zero point value (scalar or vector).
		if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
		{
			zero_point0 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point1 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
		}

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				selector2 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[0, 16-31]
			SCL_MULRND_F32(c_float_0p1,selector2,zero_point1);
		}
		else
		{
			// If original output was columns major, then by the time
			// kernel sees it, the matrix would be accessed as if it were
			// transposed. Due to this the scale as well as zp array will
			// be accessed by the ic index, and each scale/zp element
			// corresponds to an entire row of the transposed output array,
			// instead of an entire column.
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 0 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 0 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[0, 16-31]
			SCL_MULRND_F32(c_float_0p1,selector1,zero_point0);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_ADD_1x32:
	{
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31]
			BF16_F32_MATRIX_ADD_2COL(selector1,selector2,0);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31]
			F32_F32_MATRIX_ADD_2COL(selector1,selector2,0);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_MUL_1x32:
	{
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31]
			BF16_F32_MATRIX_MUL_2COL(selector1,selector2,0);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31]
			F32_F32_MATRIX_MUL_2COL(selector1,selector2,0);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_SWISH_1x32:
	{
		selector1 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__m512 al_in, r, r2, z, dn;
		__m512i ex_out;

		// c[0, 0-15]
		SWISH_F32_AVX512_DEF(c_float_0p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[0, 16-31]
		SWISH_F32_AVX512_DEF(c_float_0p1, selector1, al_in, r, r2, z, dn, ex_out);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_1x32_DISABLE:
	;

	if ( ( post_ops_attr.buf_downscale != NULL ) && ( post_ops_attr.is_last_k == TRUE ) )
	{
		// Generate a mask16 of all 1's.
		__m512i selector_a = _mm512_setzero_epi32();
		__m512i selector_b = _mm512_set1_epi32( 10 );
		__mmask16 mask_all1 = _mm512_cmplt_epi32_mask( selector_a, selector_b );

		// Store the results in downscaled type (int8 instead of int32).
		// c[0,0-15]
		CVT_STORE_F32_BF16_MASK(c_float_0p0,0,0);

		// c[0, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_0p1,0,1);
	}

	else
	{
		// Store the results.
		// c[0,0-15]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 0*16 ), c_float_0p0 );

		// c[0, 16-31]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 1*16 ), c_float_0p1 );
	}
}

// 5x48 bf16 kernel
LPGEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_5x48)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_5x48_DISABLE,
						  &&POST_OPS_BIAS_5x48,
						  &&POST_OPS_RELU_5x48,
						  &&POST_OPS_RELU_SCALE_5x48,
						  &&POST_OPS_GELU_TANH_5x48,
						  &&POST_OPS_GELU_ERF_5x48,
						  &&POST_OPS_CLIP_5x48,
						  &&POST_OPS_DOWNSCALE_5x48,
						  &&POST_OPS_MATRIX_ADD_5x48,
						  &&POST_OPS_SWISH_5x48,
						  &&POST_OPS_MATRIX_MUL_5x48
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int16_t a_kfringe_buf = 0;

    // B matrix storage bfloat type
	__m512bh b0;
	__m512bh b1;
	__m512bh b2;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();
	__m512 c_float_0p1 = _mm512_setzero_ps();
	__m512 c_float_0p2 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();
	__m512 c_float_1p1 = _mm512_setzero_ps();
	__m512 c_float_1p2 = _mm512_setzero_ps();

	__m512 c_float_2p0 = _mm512_setzero_ps();
	__m512 c_float_2p1 = _mm512_setzero_ps();
	__m512 c_float_2p2 = _mm512_setzero_ps();

	__m512 c_float_3p0 = _mm512_setzero_ps();
	__m512 c_float_3p1 = _mm512_setzero_ps();
	__m512 c_float_3p2 = _mm512_setzero_ps();

	__m512 c_float_4p0 = _mm512_setzero_ps();
	__m512 c_float_4p1 = _mm512_setzero_ps();
	__m512 c_float_4p2 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 1 ) );
		b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 2 ) );

		// Broadcast a[0,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-47] = a[0,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
		c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );

		// Broadcast a[1,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 1 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-47] = a[1,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );
		c_float_1p2 = _mm512_dpbf16_ps( c_float_1p2, a_bf16_0, b2 );

		// Broadcast a[2,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 2 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-47] = a[2,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
		c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );
		c_float_2p2 = _mm512_dpbf16_ps( c_float_2p2, a_bf16_0, b2 );

		// Broadcast a[3,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 3 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[3,0-47] = a[3,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );
		c_float_3p1 = _mm512_dpbf16_ps( c_float_3p1, a_bf16_0, b1 );
		c_float_3p2 = _mm512_dpbf16_ps( c_float_3p2, a_bf16_0, b2 );

		// Broadcast a[4,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 4 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[4,0-47] = a[4,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );
		c_float_4p1 = _mm512_dpbf16_ps( c_float_4p1, a_bf16_0, b1 );
		c_float_4p2 = _mm512_dpbf16_ps( c_float_4p2, a_bf16_0, b2 );
	}
	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 1 ) );
		b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 2 ) );

		// Broadcast a[0,kr:kr+2].
		a_kfringe_buf = *( a + (rs_a * 0) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-47] = a[0,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
		c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );

		// Broadcast a[1,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 1) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-47] = a[1,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );
		c_float_1p2 = _mm512_dpbf16_ps( c_float_1p2, a_bf16_0, b2 );

		// Broadcast a[2,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 2) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-47] = a[2,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
		c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );
		c_float_2p2 = _mm512_dpbf16_ps( c_float_2p2, a_bf16_0, b2 );

		// Broadcast a[3,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 3) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[3,0-47] = a[3,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );
		c_float_3p1 = _mm512_dpbf16_ps( c_float_3p1, a_bf16_0, b1 );
		c_float_3p2 = _mm512_dpbf16_ps( c_float_3p2, a_bf16_0, b2 );

		// Broadcast a[4,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 4) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[4,0-47] = a[4,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );
		c_float_4p1 = _mm512_dpbf16_ps( c_float_4p1, a_bf16_0, b1 );
		c_float_4p2 = _mm512_dpbf16_ps( c_float_4p2, a_bf16_0, b2 );
	}

	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	if ( alpha != 1 )
	{
		// Scale by alpha
		c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );
		c_float_0p1 = _mm512_mul_ps( selector1, c_float_0p1 );
		c_float_0p2 = _mm512_mul_ps( selector1, c_float_0p2 );

		c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );
		c_float_1p1 = _mm512_mul_ps( selector1, c_float_1p1 );
		c_float_1p2 = _mm512_mul_ps( selector1, c_float_1p2 );

		c_float_2p0 = _mm512_mul_ps( selector1, c_float_2p0 );
		c_float_2p1 = _mm512_mul_ps( selector1, c_float_2p1 );
		c_float_2p2 = _mm512_mul_ps( selector1, c_float_2p2 );

		c_float_3p0 = _mm512_mul_ps( selector1, c_float_3p0 );
		c_float_3p1 = _mm512_mul_ps( selector1, c_float_3p1 );
		c_float_3p2 = _mm512_mul_ps( selector1, c_float_3p2 );

		c_float_4p0 = _mm512_mul_ps( selector1, c_float_4p0 );
		c_float_4p1 = _mm512_mul_ps( selector1, c_float_4p1 );
		c_float_4p2 = _mm512_mul_ps( selector1, c_float_4p2 );
	}

	// Scale C by beta.
	if ( beta != 0 )
	{
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
				 ( post_ops_attr.is_first_k == TRUE ) )
			{
				// c[0,0-15]
				BF16_F32_BETA_OP(c_float_0p0,0,0,0,selector1,selector2)

				// c[0, 16-31]
				BF16_F32_BETA_OP(c_float_0p1,0,0,1,selector1,selector2)

				// c[0,32-47]
				BF16_F32_BETA_OP(c_float_0p2,0,0,2,selector1,selector2)

				// c[1,0-15]
				BF16_F32_BETA_OP(c_float_1p0,0,1,0,selector1,selector2)

				// c[1,16-31]
				BF16_F32_BETA_OP(c_float_1p1,0,1,1,selector1,selector2)

				// c[1,32-47]
				BF16_F32_BETA_OP(c_float_1p2,0,1,2,selector1,selector2)

				// c[2,0-15]
				BF16_F32_BETA_OP(c_float_2p0,0,2,0,selector1,selector2)

				// c[2,16-31]
				BF16_F32_BETA_OP(c_float_2p1,0,2,1,selector1,selector2)

				// c[2,32-47]
				BF16_F32_BETA_OP(c_float_2p2,0,2,2,selector1,selector2)

				// c[3,0-15]
				BF16_F32_BETA_OP(c_float_3p0,0,3,0,selector1,selector2)

				// c[3,16-31]
				BF16_F32_BETA_OP(c_float_3p1,0,3,1,selector1,selector2)

				// c[3,32-47]
				BF16_F32_BETA_OP(c_float_3p2,0,3,2,selector1,selector2)

				// c[4,0-15]
				BF16_F32_BETA_OP(c_float_4p0,0,4,0,selector1,selector2)

				// c[4,16-31]
				BF16_F32_BETA_OP(c_float_4p1,0,4,1,selector1,selector2)

				// c[4,32-47]
				BF16_F32_BETA_OP(c_float_4p2,0,4,2,selector1,selector2)
			}
			else
			{
				// c[0,0-15]
				F32_F32_BETA_OP(c_float_0p0,0,0,0,selector1,selector2)

				// c[0, 16-31]
				F32_F32_BETA_OP(c_float_0p1,0,0,1,selector1,selector2)

				// c[0,32-47]
				F32_F32_BETA_OP(c_float_0p2,0,0,2,selector1,selector2)

				// c[1,0-15]
				F32_F32_BETA_OP(c_float_1p0,0,1,0,selector1,selector2)

				// c[1,16-31]
				F32_F32_BETA_OP(c_float_1p1,0,1,1,selector1,selector2)

				// c[1,32-47]
				F32_F32_BETA_OP(c_float_1p2,0,1,2,selector1,selector2)

				// c[2,0-15]
				F32_F32_BETA_OP(c_float_2p0,0,2,0,selector1,selector2)

				// c[2,16-31]
				F32_F32_BETA_OP(c_float_2p1,0,2,1,selector1,selector2)

				// c[2,32-47]
				F32_F32_BETA_OP(c_float_2p2,0,2,2,selector1,selector2)

				// c[3,0-15]
				F32_F32_BETA_OP(c_float_3p0,0,3,0,selector1,selector2)

				// c[3,16-31]
				F32_F32_BETA_OP(c_float_3p1,0,3,1,selector1,selector2)

				// c[3,32-47]
				F32_F32_BETA_OP(c_float_3p2,0,3,2,selector1,selector2)

				// c[4,0-15]
				F32_F32_BETA_OP(c_float_4p0,0,4,0,selector1,selector2)

				// c[4,16-31]
				F32_F32_BETA_OP(c_float_4p1,0,4,1,selector1,selector2)

				// c[4,32-47]
				F32_F32_BETA_OP(c_float_4p2,0,4,2,selector1,selector2)
			}
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_5x48:
	{
		__m512 selector3;

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_LOAD(selector1, bias_mask, 0);
				BF16_F32_BIAS_LOAD(selector2, bias_mask, 1);
				BF16_F32_BIAS_LOAD(selector3, bias_mask, 2);
			}
			else
			{
				selector1 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				selector2 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				selector3 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 2 * 16 ) );
			}

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_add_ps( selector2, c_float_0p1 );

			// c[0,32-47]
			c_float_0p2 = _mm512_add_ps( selector3, c_float_0p2 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

			// c[1, 16-31]
			c_float_1p1 = _mm512_add_ps( selector2, c_float_1p1 );

			// c[1,32-47]
			c_float_1p2 = _mm512_add_ps( selector3, c_float_1p2 );

			// c[2,0-15]
			c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );

			// c[2, 16-31]
			c_float_2p1 = _mm512_add_ps( selector2, c_float_2p1 );

			// c[2,32-47]
			c_float_2p2 = _mm512_add_ps( selector3, c_float_2p2 );

			// c[3,0-15]
			c_float_3p0 = _mm512_add_ps( selector1, c_float_3p0 );

			// c[3, 16-31]
			c_float_3p1 = _mm512_add_ps( selector2, c_float_3p1 );

			// c[3,32-47]
			c_float_3p2 = _mm512_add_ps( selector3, c_float_3p2 );

			// c[4,0-15]
			c_float_4p0 = _mm512_add_ps( selector1, c_float_4p0 );

			// c[4, 16-31]
			c_float_4p1 = _mm512_add_ps( selector2, c_float_4p1 );

			// c[4,32-47]
			c_float_4p2 = _mm512_add_ps( selector3, c_float_4p2 );
		}
		else
		{
			__m512 selector4;
			__m512 selector5;
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_BCAST(selector1, bias_mask, 0);
				BF16_F32_BIAS_BCAST(selector2, bias_mask, 1);
				BF16_F32_BIAS_BCAST(selector3, bias_mask, 2);
				BF16_F32_BIAS_BCAST(selector4, bias_mask, 3);
				BF16_F32_BIAS_BCAST(selector5, bias_mask, 4);
			}
			else
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 1 ) );
				selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 2 ) );
				selector4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 3 ) );
				selector5 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 4 ) );
			}

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_add_ps( selector1, c_float_0p1 );

			// c[0,32-47]
			c_float_0p2 = _mm512_add_ps( selector1, c_float_0p2 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );

			// c[1, 16-31]
			c_float_1p1 = _mm512_add_ps( selector2, c_float_1p1 );

			// c[1,32-47]
			c_float_1p2 = _mm512_add_ps( selector2, c_float_1p2 );

			// c[2,0-15]
			c_float_2p0 = _mm512_add_ps( selector3, c_float_2p0 );

			// c[2, 16-31]
			c_float_2p1 = _mm512_add_ps( selector3, c_float_2p1 );

			// c[2,32-47]
			c_float_2p2 = _mm512_add_ps( selector3, c_float_2p2 );

			// c[3,0-15]
			c_float_3p0 = _mm512_add_ps( selector4, c_float_3p0 );

			// c[3, 16-31]
			c_float_3p1 = _mm512_add_ps( selector4, c_float_3p1 );

			// c[3,32-47]
			c_float_3p2 = _mm512_add_ps( selector4, c_float_3p2 );

			// c[4,0-15]
			c_float_4p0 = _mm512_add_ps( selector5, c_float_4p0 );

			// c[4, 16-31]
			c_float_4p1 = _mm512_add_ps( selector5, c_float_4p1 );

			// c[4,32-47]
			c_float_4p2 = _mm512_add_ps( selector5, c_float_4p2 );
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_5x48:
	{
		selector1 = _mm512_setzero_ps();

		// c[0,0-15]
		c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

		// c[0, 16-31]
		c_float_0p1 = _mm512_max_ps( selector1, c_float_0p1 );

		// c[0,32-47]
		c_float_0p2 = _mm512_max_ps( selector1, c_float_0p2 );

		// c[1,0-15]
		c_float_1p0 = _mm512_max_ps( selector1, c_float_1p0 );

		// c[1,16-31]
		c_float_1p1 = _mm512_max_ps( selector1, c_float_1p1 );

		// c[1,32-47]
		c_float_1p2 = _mm512_max_ps( selector1, c_float_1p2 );

		// c[2,0-15]
		c_float_2p0 = _mm512_max_ps( selector1, c_float_2p0 );

		// c[2,16-31]
		c_float_2p1 = _mm512_max_ps( selector1, c_float_2p1 );

		// c[2,32-47]
		c_float_2p2 = _mm512_max_ps( selector1, c_float_2p2 );

		// c[3,0-15]
		c_float_3p0 = _mm512_max_ps( selector1, c_float_3p0 );

		// c[3,16-31]
		c_float_3p1 = _mm512_max_ps( selector1, c_float_3p1 );

		// c[3,32-47]
		c_float_3p2 = _mm512_max_ps( selector1, c_float_3p2 );

		// c[4,0-15]
		c_float_4p0 = _mm512_max_ps( selector1, c_float_4p0 );

		// c[4,16-31]
		c_float_4p1 = _mm512_max_ps( selector1, c_float_4p1 );

		// c[4,32-47]
		c_float_4p2 = _mm512_max_ps( selector1, c_float_4p2 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_5x48:
	{
		selector1 = _mm512_setzero_ps();
		selector2 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__mmask16 relu_cmp_mask;

		// c[0, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_0p0)

		// c[0, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_0p1)

		// c[0, 32-47]
		RELU_SCALE_OP_F32_AVX512(c_float_0p2)

		// c[1, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_1p0)

		// c[1, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_1p1)

		// c[1, 32-47]
		RELU_SCALE_OP_F32_AVX512(c_float_1p2)

		// c[2, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_2p0)

		// c[2, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_2p1)

		// c[2, 32-47]
		RELU_SCALE_OP_F32_AVX512(c_float_2p2)

		// c[3, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_3p0)

		// c[3, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_3p1)

		// c[3, 32-47]
		RELU_SCALE_OP_F32_AVX512(c_float_3p2)

		// c[4, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_4p0)

		// c[4, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_4p1)

		// c[4, 32-47]
		RELU_SCALE_OP_F32_AVX512(c_float_4p2)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_TANH_5x48:
	{
		__m512 dn, z, x, r2, r, x_tanh;
		__m512i q;

		// c[0, 0-15]
		GELU_TANH_F32_AVX512(c_float_0p0, r, r2, x, z, dn, x_tanh, q)

		// c[0, 16-31]
		GELU_TANH_F32_AVX512(c_float_0p1, r, r2, x, z, dn, x_tanh, q)

		// c[0, 32-47]
		GELU_TANH_F32_AVX512(c_float_0p2, r, r2, x, z, dn, x_tanh, q)

		// c[1, 0-15]
		GELU_TANH_F32_AVX512(c_float_1p0, r, r2, x, z, dn, x_tanh, q)

		// c[1, 16-31]
		GELU_TANH_F32_AVX512(c_float_1p1, r, r2, x, z, dn, x_tanh, q)

		// c[1, 32-47]
		GELU_TANH_F32_AVX512(c_float_1p2, r, r2, x, z, dn, x_tanh, q)

		// c[2, 0-15]
		GELU_TANH_F32_AVX512(c_float_2p0, r, r2, x, z, dn, x_tanh, q)

		// c[2, 16-31]
		GELU_TANH_F32_AVX512(c_float_2p1, r, r2, x, z, dn, x_tanh, q)

		// c[2, 32-47]
		GELU_TANH_F32_AVX512(c_float_2p2, r, r2, x, z, dn, x_tanh, q)

		// c[3, 0-15]
		GELU_TANH_F32_AVX512(c_float_3p0, r, r2, x, z, dn, x_tanh, q)

		// c[3, 16-31]
		GELU_TANH_F32_AVX512(c_float_3p1, r, r2, x, z, dn, x_tanh, q)

		// c[3, 32-47]
		GELU_TANH_F32_AVX512(c_float_3p2, r, r2, x, z, dn, x_tanh, q)

		// c[4, 0-15]
		GELU_TANH_F32_AVX512(c_float_4p0, r, r2, x, z, dn, x_tanh, q)

		// c[4, 16-31]
		GELU_TANH_F32_AVX512(c_float_4p1, r, r2, x, z, dn, x_tanh, q)

		// c[4, 32-47]
		GELU_TANH_F32_AVX512(c_float_4p2, r, r2, x, z, dn, x_tanh, q)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_ERF_5x48:
	{
		__m512 x, r, x_erf;

		// c[0, 0-15]
		GELU_ERF_F32_AVX512(c_float_0p0, r, x, x_erf)

		// c[0, 16-31]
		GELU_ERF_F32_AVX512(c_float_0p1, r, x, x_erf)

		// c[0, 32-47]
		GELU_ERF_F32_AVX512(c_float_0p2, r, x, x_erf)

		// c[1, 0-15]
		GELU_ERF_F32_AVX512(c_float_1p0, r, x, x_erf)

		// c[1, 16-31]
		GELU_ERF_F32_AVX512(c_float_1p1, r, x, x_erf)

		// c[1, 32-47]
		GELU_ERF_F32_AVX512(c_float_1p2, r, x, x_erf)

		// c[2, 0-15]
		GELU_ERF_F32_AVX512(c_float_2p0, r, x, x_erf)

		// c[2, 16-31]
		GELU_ERF_F32_AVX512(c_float_2p1, r, x, x_erf)

		// c[2, 32-47]
		GELU_ERF_F32_AVX512(c_float_2p2, r, x, x_erf)

		// c[3, 0-15]
		GELU_ERF_F32_AVX512(c_float_3p0, r, x, x_erf)

		// c[3, 16-31]
		GELU_ERF_F32_AVX512(c_float_3p1, r, x, x_erf)

		// c[3, 32-47]
		GELU_ERF_F32_AVX512(c_float_3p2, r, x, x_erf)

		// c[4, 0-15]
		GELU_ERF_F32_AVX512(c_float_4p0, r, x, x_erf)

		// c[4, 16-31]
		GELU_ERF_F32_AVX512(c_float_4p1, r, x, x_erf)

		// c[4, 32-47]
		GELU_ERF_F32_AVX512(c_float_4p2, r, x, x_erf)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_CLIP_5x48:
	{
		__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
		__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

		// c[0, 0-15]
		CLIP_F32_AVX512(c_float_0p0, min, max)

		// c[0, 16-31]
		CLIP_F32_AVX512(c_float_0p1, min, max)

		// c[0, 32-47]
		CLIP_F32_AVX512(c_float_0p2, min, max)

		// c[1, 0-15]
		CLIP_F32_AVX512(c_float_1p0, min, max)

		// c[1, 16-31]
		CLIP_F32_AVX512(c_float_1p1, min, max)

		// c[1, 32-47]
		CLIP_F32_AVX512(c_float_1p2, min, max)

		// c[2, 0-15]
		CLIP_F32_AVX512(c_float_2p0, min, max)

		// c[2, 16-31]
		CLIP_F32_AVX512(c_float_2p1, min, max)

		// c[2, 32-47]
		CLIP_F32_AVX512(c_float_2p2, min, max)

		// c[3, 0-15]
		CLIP_F32_AVX512(c_float_3p0, min, max)

		// c[3, 16-31]
		CLIP_F32_AVX512(c_float_3p1, min, max)

		// c[3, 32-47]
		CLIP_F32_AVX512(c_float_3p2, min, max)

		// c[4, 0-15]
		CLIP_F32_AVX512(c_float_4p0, min, max)

		// c[4, 16-31]
		CLIP_F32_AVX512(c_float_4p1, min, max)

		// c[4, 32-47]
		CLIP_F32_AVX512(c_float_4p2, min, max)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}

POST_OPS_DOWNSCALE_5x48:
	{
		__m512 selector3 = _mm512_setzero_ps();
		__m512 selector4 = _mm512_setzero_ps();

		__m512 zero_point0 = _mm512_setzero_ps();
		__m512 zero_point1 = _mm512_setzero_ps();
		__m512 zero_point2 = _mm512_setzero_ps();
		__m512 zero_point3 = _mm512_setzero_ps();

		__mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );

		// Need to account for row vs column major swaps. For scalars
		// scale and zero point, no implications.
		// Even though different registers are used for scalar in column
		// and row major downscale path, all those registers will contain
		// the same value.
		if ( post_ops_list_temp->scale_factor_len == 1 )
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector4 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
		}

		// bf16 zero point value (scalar or vector).
		if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
		{
			zero_point0 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point1 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point2 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point3 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
		}

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				selector2 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				selector3 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 2 * 16 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) ) );
				zero_point2 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 2 * 16 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[0, 16-31]
			SCL_MULRND_F32(c_float_0p1,selector2,zero_point1);

			// c[0, 32-47]
			SCL_MULRND_F32(c_float_0p2,selector3,zero_point2);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector1,zero_point0);

			// c[1, 16-31]
			SCL_MULRND_F32(c_float_1p1,selector2,zero_point1);

			// c[1, 32-47]
			SCL_MULRND_F32(c_float_1p2,selector3,zero_point2);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector1,zero_point0);

			// c[2, 16-31]
			SCL_MULRND_F32(c_float_2p1,selector2,zero_point1);

			// c[2, 32-47]
			SCL_MULRND_F32(c_float_2p2,selector3,zero_point2);

			// c[3, 0-15]
			SCL_MULRND_F32(c_float_3p0,selector1,zero_point0);

			// c[3, 16-31]
			SCL_MULRND_F32(c_float_3p1,selector2,zero_point1);

			// c[3, 32-47]
			SCL_MULRND_F32(c_float_3p2,selector3,zero_point2);

			// c[4, 0-15]
			SCL_MULRND_F32(c_float_4p0,selector1,zero_point0);

			// c[4, 16-31]
			SCL_MULRND_F32(c_float_4p1,selector2,zero_point1);

			// c[4, 32-47]
			SCL_MULRND_F32(c_float_4p2,selector3,zero_point2);
		}
		else
		{
			// If original output was columns major, then by the time
			// kernel sees it, the matrix would be accessed as if it were
			// transposed. Due to this the scale as well as zp array will
			// be accessed by the ic index, and each scale/zp element
			// corresponds to an entire row of the transposed output array,
			// instead of an entire column.
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 1 ) );
				selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 2 ) );
				selector4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 3 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 0 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 1 ) ) );
				zero_point2 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 2 ) ) );
				zero_point3 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 3 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[0, 16-31]
			SCL_MULRND_F32(c_float_0p1,selector1,zero_point0);

			// c[0, 32-47]
			SCL_MULRND_F32(c_float_0p2,selector1,zero_point0);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector2,zero_point1);

			// c[1, 16-31]
			SCL_MULRND_F32(c_float_1p1,selector2,zero_point1);

			// c[1, 32-47]
			SCL_MULRND_F32(c_float_1p2,selector2,zero_point1);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector3,zero_point2);

			// c[2, 16-31]
			SCL_MULRND_F32(c_float_2p1,selector3,zero_point2);

			// c[2, 32-47]
			SCL_MULRND_F32(c_float_2p2,selector3,zero_point2);

			// c[3, 0-15]
			SCL_MULRND_F32(c_float_3p0,selector4,zero_point3);

			// c[3, 16-31]
			SCL_MULRND_F32(c_float_3p1,selector4,zero_point3);

			// c[3, 32-47]
			SCL_MULRND_F32(c_float_3p2,selector4,zero_point3);

			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 4 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 4 ) ) );
			}
			// c[4, 0-15]
			SCL_MULRND_F32(c_float_4p0,selector1,zero_point0);

			// c[4, 16-31]
			SCL_MULRND_F32(c_float_4p1,selector1,zero_point0);

			// c[4, 32-47]
			SCL_MULRND_F32(c_float_4p2,selector1,zero_point0);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_ADD_5x48:
	{
		__m512 selector3;
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47]
			BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,0);

			// c[1:0-15,16-31,32-47]
			BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,1);

			// c[2:0-15,16-31,32-47]
			BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,2);

			// c[3:0-15,16-31,32-47]
			BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,3);

			// c[4:0-15,16-31,32-47]
			BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,4);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47]
			F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,0);

			// c[1:0-15,16-31,32-47]
			F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,1);

			// c[2:0-15,16-31,32-47]
			F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,2);

			// c[3:0-15,16-31,32-47]
			F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,3);

			// c[4:0-15,16-31,32-47]
			F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,4);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_MUL_5x48:
	{
		__m512 selector3;
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47]
			BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,0);

			// c[1:0-15,16-31,32-47]
			BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,1);

			// c[2:0-15,16-31,32-47]
			BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,2);

			// c[3:0-15,16-31,32-47]
			BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,3);

			// c[4:0-15,16-31,32-47]
			BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,4);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47]
			F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,0);

			// c[1:0-15,16-31,32-47]
			F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,1);

			// c[2:0-15,16-31,32-47]
			F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,2);
			// c[3:0-15,16-31,32-47]
			F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,3);

			// c[4:0-15,16-31,32-47]
			F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,4);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_SWISH_5x48:
	{
		selector1 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__m512 al_in, r, r2, z, dn;
		__m512i ex_out;

		// c[0, 0-15]
		SWISH_F32_AVX512_DEF(c_float_0p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[0, 16-31]
		SWISH_F32_AVX512_DEF(c_float_0p1, selector1, al_in, r, r2, z, dn, ex_out);

		// c[0, 32-47]
		SWISH_F32_AVX512_DEF(c_float_0p2, selector1, al_in, r, r2, z, dn, ex_out);

		// c[1, 0-15]
		SWISH_F32_AVX512_DEF(c_float_1p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[1, 16-31]
		SWISH_F32_AVX512_DEF(c_float_1p1, selector1, al_in, r, r2, z, dn, ex_out);

		// c[1, 32-47]
		SWISH_F32_AVX512_DEF(c_float_1p2, selector1, al_in, r, r2, z, dn, ex_out);

		// c[2, 0-15]
		SWISH_F32_AVX512_DEF(c_float_2p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[2, 16-31]
		SWISH_F32_AVX512_DEF(c_float_2p1, selector1, al_in, r, r2, z, dn, ex_out);

		// c[2, 32-47]
		SWISH_F32_AVX512_DEF(c_float_2p2, selector1, al_in, r, r2, z, dn, ex_out);

		// c[3, 0-15]
		SWISH_F32_AVX512_DEF(c_float_3p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[3, 16-31]
		SWISH_F32_AVX512_DEF(c_float_3p1, selector1, al_in, r, r2, z, dn, ex_out);

		// c[3, 32-47]
		SWISH_F32_AVX512_DEF(c_float_3p2, selector1, al_in, r, r2, z, dn, ex_out);

		// c[4, 0-15]
		SWISH_F32_AVX512_DEF(c_float_4p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[4, 16-31]
		SWISH_F32_AVX512_DEF(c_float_4p1, selector1, al_in, r, r2, z, dn, ex_out);

		// c[4, 32-47]
		SWISH_F32_AVX512_DEF(c_float_4p2, selector1, al_in, r, r2, z, dn, ex_out);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_5x48_DISABLE:
	;

	if ( ( post_ops_attr.buf_downscale != NULL ) &&
			 ( post_ops_attr.is_last_k == TRUE ) )
	{
		// Generate a mask16 of all 1's.
		__m512i selector_a = _mm512_setzero_epi32();
		__m512i selector_b = _mm512_set1_epi32( 10 );
		__mmask16 mask_all1 = _mm512_cmplt_epi32_mask( selector_a, selector_b );

		// Store the results in downscaled type (bf16 instead of float).

		// c[0, 0-15]
		CVT_STORE_F32_BF16_MASK(c_float_0p0,0,0);

		// c[0, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_0p1,0,1);

		// c[0, 32-47]
		CVT_STORE_F32_BF16_MASK(c_float_0p2,0,2);

		// c[1, 0-15]
		CVT_STORE_F32_BF16_MASK(c_float_1p0,1,0);

		// c[1, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_1p1,1,1);

		// c[1, 32-47]
		CVT_STORE_F32_BF16_MASK(c_float_1p2,1,2);

		// c[2, 0-15]
		CVT_STORE_F32_BF16_MASK(c_float_2p0,2,0);

		// c[2, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_2p1,2,1);

		// c[2, 32-47]
		CVT_STORE_F32_BF16_MASK(c_float_2p2,2,2);

		// c[3, 0-15]
		CVT_STORE_F32_BF16_MASK(c_float_3p0,3,0);

		// c[3, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_3p1,3,1);

		// c[3, 32-47]
		CVT_STORE_F32_BF16_MASK(c_float_3p2,3,2);

		// c[4, 0-15]
		CVT_STORE_F32_BF16_MASK(c_float_4p0,4,0);

		// c[4, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_4p1,4,1);

		// c[4, 32-47]
		CVT_STORE_F32_BF16_MASK(c_float_4p2,4,2);
	}

	else
	{
		// Store the results.
		// c[0,0-15]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 0*16 ), c_float_0p0 );

		// c[0, 16-31]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 1*16 ), c_float_0p1 );

		// c[0,32-47]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 2*16 ), c_float_0p2 );

		// c[1,0-15]
		_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 0*16 ), c_float_1p0 );

		// c[1,16-31]
		_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 1*16 ), c_float_1p1 );

		// c[1,32-47]
		_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 2*16 ), c_float_1p2 );

		// c[2,0-15]
		_mm512_storeu_ps( c + ( rs_c * 2 ) + ( 0*16 ), c_float_2p0 );

		// c[2,16-31]
		_mm512_storeu_ps( c + ( rs_c * 2 ) + ( 1*16 ), c_float_2p1 );

		// c[2,32-47]
		_mm512_storeu_ps( c + ( rs_c * 2 ) + ( 2*16 ), c_float_2p2 );

		// c[3,0-15]
		_mm512_storeu_ps( c + ( rs_c * 3 ) + ( 0*16 ), c_float_3p0 );

		// c[3,16-31]
		_mm512_storeu_ps( c + ( rs_c * 3 ) + ( 1*16 ), c_float_3p1 );

		// c[3,32-47]
		_mm512_storeu_ps( c + ( rs_c * 3 ) + ( 2*16 ), c_float_3p2 );

		// c[4,0-15]
		_mm512_storeu_ps( c + ( rs_c * 4 ) + ( 0*16 ), c_float_4p0 );

		// c[4,16-31]
		_mm512_storeu_ps( c + ( rs_c * 4 ) + ( 1*16 ), c_float_4p1 );

		// c[4,32-47]
		_mm512_storeu_ps( c + ( rs_c * 4 ) + ( 2*16 ), c_float_4p2 );
	}
}

// 4x48 bf16 kernel
LPGEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_4x48)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_4x48_DISABLE,
						  &&POST_OPS_BIAS_4x48,
						  &&POST_OPS_RELU_4x48,
						  &&POST_OPS_RELU_SCALE_4x48,
						  &&POST_OPS_GELU_TANH_4x48,
						  &&POST_OPS_GELU_ERF_4x48,
						  &&POST_OPS_CLIP_4x48,
						  &&POST_OPS_DOWNSCALE_4x48,
						  &&POST_OPS_MATRIX_ADD_4x48,
						  &&POST_OPS_SWISH_4x48,
						  &&POST_OPS_MATRIX_MUL_4x48
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int16_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;
	__m512bh b1;
	__m512bh b2;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();
	__m512 c_float_0p1 = _mm512_setzero_ps();
	__m512 c_float_0p2 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();
	__m512 c_float_1p1 = _mm512_setzero_ps();
	__m512 c_float_1p2 = _mm512_setzero_ps();

	__m512 c_float_2p0 = _mm512_setzero_ps();
	__m512 c_float_2p1 = _mm512_setzero_ps();
	__m512 c_float_2p2 = _mm512_setzero_ps();

	__m512 c_float_3p0 = _mm512_setzero_ps();
	__m512 c_float_3p1 = _mm512_setzero_ps();
	__m512 c_float_3p2 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 1 ) );
		b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 2 ) );

		// Broadcast a[0,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-47] = a[0,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
		c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );

		// Broadcast a[1,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 1 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-47] = a[1,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );
		c_float_1p2 = _mm512_dpbf16_ps( c_float_1p2, a_bf16_0, b2 );

		// Broadcast a[2,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 2 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-47] = a[2,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
		c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );
		c_float_2p2 = _mm512_dpbf16_ps( c_float_2p2, a_bf16_0, b2 );

		// Broadcast a[3,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 3 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[3,0-47] = a[3,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );
		c_float_3p1 = _mm512_dpbf16_ps( c_float_3p1, a_bf16_0, b1 );
		c_float_3p2 = _mm512_dpbf16_ps( c_float_3p2, a_bf16_0, b2 );
	}
	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 1 ) );
		b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 2 ) );

		// Broadcast a[0,kr:kr+2].
		a_kfringe_buf = *( a + (rs_a * 0) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-47] = a[0,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
		c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );

		// Broadcast a[1,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 1) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-47] = a[1,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );
		c_float_1p2 = _mm512_dpbf16_ps( c_float_1p2, a_bf16_0, b2 );

		// Broadcast a[2,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 2) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-47] = a[2,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
		c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );
		c_float_2p2 = _mm512_dpbf16_ps( c_float_2p2, a_bf16_0, b2 );

		// Broadcast a[3,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 3) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[3,0-47] = a[3,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );
		c_float_3p1 = _mm512_dpbf16_ps( c_float_3p1, a_bf16_0, b1 );
		c_float_3p2 = _mm512_dpbf16_ps( c_float_3p2, a_bf16_0, b2 );
	}

	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	if ( alpha != 1 )
	{
		// Scale by alpha
		c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );
		c_float_0p1 = _mm512_mul_ps( selector1, c_float_0p1 );
		c_float_0p2 = _mm512_mul_ps( selector1, c_float_0p2 );

		c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );
		c_float_1p1 = _mm512_mul_ps( selector1, c_float_1p1 );
		c_float_1p2 = _mm512_mul_ps( selector1, c_float_1p2 );

		c_float_2p0 = _mm512_mul_ps( selector1, c_float_2p0 );
		c_float_2p1 = _mm512_mul_ps( selector1, c_float_2p1 );
		c_float_2p2 = _mm512_mul_ps( selector1, c_float_2p2 );

		c_float_3p0 = _mm512_mul_ps( selector1, c_float_3p0 );
		c_float_3p1 = _mm512_mul_ps( selector1, c_float_3p1 );
		c_float_3p2 = _mm512_mul_ps( selector1, c_float_3p2 );
	}

	// Scale C by beta.
	if ( beta != 0 )
	{
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
				 ( post_ops_attr.is_first_k == TRUE ) )
			{
				// c[0,0-15]
				BF16_F32_BETA_OP(c_float_0p0,0,0,0,selector1,selector2)

				// c[0, 16-31]
				BF16_F32_BETA_OP(c_float_0p1,0,0,1,selector1,selector2)

				// c[0,32-47]
				BF16_F32_BETA_OP(c_float_0p2,0,0,2,selector1,selector2)

				// c[1,0-15]
				BF16_F32_BETA_OP(c_float_1p0,0,1,0,selector1,selector2)

				// c[1,16-31]
				BF16_F32_BETA_OP(c_float_1p1,0,1,1,selector1,selector2)

				// c[1,32-47]
				BF16_F32_BETA_OP(c_float_1p2,0,1,2,selector1,selector2)

				// c[2,0-15]
				BF16_F32_BETA_OP(c_float_2p0,0,2,0,selector1,selector2)

				// c[2,16-31]
				BF16_F32_BETA_OP(c_float_2p1,0,2,1,selector1,selector2)

				// c[2,32-47]
				BF16_F32_BETA_OP(c_float_2p2,0,2,2,selector1,selector2)

				// c[3,0-15]
				BF16_F32_BETA_OP(c_float_3p0,0,3,0,selector1,selector2)

				// c[3,16-31]
				BF16_F32_BETA_OP(c_float_3p1,0,3,1,selector1,selector2)

				// c[3,32-47]
				BF16_F32_BETA_OP(c_float_3p2,0,3,2,selector1,selector2)
			}
			else
			{
				// c[0,0-15]
				F32_F32_BETA_OP(c_float_0p0,0,0,0,selector1,selector2)

				// c[0, 16-31]
				F32_F32_BETA_OP(c_float_0p1,0,0,1,selector1,selector2)

				// c[0,32-47]
				F32_F32_BETA_OP(c_float_0p2,0,0,2,selector1,selector2)

				// c[1,0-15]
				F32_F32_BETA_OP(c_float_1p0,0,1,0,selector1,selector2)

				// c[1,16-31]
				F32_F32_BETA_OP(c_float_1p1,0,1,1,selector1,selector2)

				// c[1,32-47]
				F32_F32_BETA_OP(c_float_1p2,0,1,2,selector1,selector2)

				// c[2,0-15]
				F32_F32_BETA_OP(c_float_2p0,0,2,0,selector1,selector2)

				// c[2,16-31]
				F32_F32_BETA_OP(c_float_2p1,0,2,1,selector1,selector2)

				// c[2,32-47]
				F32_F32_BETA_OP(c_float_2p2,0,2,2,selector1,selector2)

				// c[3,0-15]
				F32_F32_BETA_OP(c_float_3p0,0,3,0,selector1,selector2)

				// c[3,16-31]
				F32_F32_BETA_OP(c_float_3p1,0,3,1,selector1,selector2)

				// c[3,32-47]
				F32_F32_BETA_OP(c_float_3p2,0,3,2,selector1,selector2)
			}
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_4x48:
	{
		__m512 selector3;

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_LOAD(selector1, bias_mask, 0);
				BF16_F32_BIAS_LOAD(selector2, bias_mask, 1);
				BF16_F32_BIAS_LOAD(selector3, bias_mask, 2);
			}
			else
			{
				selector1 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				selector2 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				selector3 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 2 * 16 ) );
			}

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_add_ps( selector2, c_float_0p1 );

			// c[0,32-47]
			c_float_0p2 = _mm512_add_ps( selector3, c_float_0p2 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

			// c[1, 16-31]
			c_float_1p1 = _mm512_add_ps( selector2, c_float_1p1 );

			// c[1,32-47]
			c_float_1p2 = _mm512_add_ps( selector3, c_float_1p2 );

			// c[2,0-15]
			c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );

			// c[2, 16-31]
			c_float_2p1 = _mm512_add_ps( selector2, c_float_2p1 );

			// c[2,32-47]
			c_float_2p2 = _mm512_add_ps( selector3, c_float_2p2 );

			// c[3,0-15]
			c_float_3p0 = _mm512_add_ps( selector1, c_float_3p0 );

			// c[3, 16-31]
			c_float_3p1 = _mm512_add_ps( selector2, c_float_3p1 );

			// c[3,32-47]
			c_float_3p2 = _mm512_add_ps( selector3, c_float_3p2 );
		}
		else
		{
			__m512 selector4;
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_BCAST(selector1, bias_mask, 0);
				BF16_F32_BIAS_BCAST(selector2, bias_mask, 1);
				BF16_F32_BIAS_BCAST(selector3, bias_mask, 2);
				BF16_F32_BIAS_BCAST(selector4, bias_mask, 3);
			}
			else
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 1 ) );
				selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 2 ) );
				selector4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 3 ) );
			}

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_add_ps( selector1, c_float_0p1 );

			// c[0,32-47]
			c_float_0p2 = _mm512_add_ps( selector1, c_float_0p2 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );

			// c[1, 16-31]
			c_float_1p1 = _mm512_add_ps( selector2, c_float_1p1 );

			// c[1,32-47]
			c_float_1p2 = _mm512_add_ps( selector2, c_float_1p2 );

			// c[2,0-15]
			c_float_2p0 = _mm512_add_ps( selector3, c_float_2p0 );

			// c[2, 16-31]
			c_float_2p1 = _mm512_add_ps( selector3, c_float_2p1 );

			// c[2,32-47]
			c_float_2p2 = _mm512_add_ps( selector3, c_float_2p2 );

			// c[3,0-15]
			c_float_3p0 = _mm512_add_ps( selector4, c_float_3p0 );

			// c[3, 16-31]
			c_float_3p1 = _mm512_add_ps( selector4, c_float_3p1 );

			// c[3,32-47]
			c_float_3p2 = _mm512_add_ps( selector4, c_float_3p2 );
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_4x48:
	{
		selector1 = _mm512_setzero_ps();

		// c[0,0-15]
		c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

		// c[0, 16-31]
		c_float_0p1 = _mm512_max_ps( selector1, c_float_0p1 );

		// c[0,32-47]
		c_float_0p2 = _mm512_max_ps( selector1, c_float_0p2 );

		// c[1,0-15]
		c_float_1p0 = _mm512_max_ps( selector1, c_float_1p0 );

		// c[1,16-31]
		c_float_1p1 = _mm512_max_ps( selector1, c_float_1p1 );

		// c[1,32-47]
		c_float_1p2 = _mm512_max_ps( selector1, c_float_1p2 );

		// c[2,0-15]
		c_float_2p0 = _mm512_max_ps( selector1, c_float_2p0 );

		// c[2,16-31]
		c_float_2p1 = _mm512_max_ps( selector1, c_float_2p1 );

		// c[2,32-47]
		c_float_2p2 = _mm512_max_ps( selector1, c_float_2p2 );

		// c[3,0-15]
		c_float_3p0 = _mm512_max_ps( selector1, c_float_3p0 );

		// c[3,16-31]
		c_float_3p1 = _mm512_max_ps( selector1, c_float_3p1 );

		// c[3,32-47]
		c_float_3p2 = _mm512_max_ps( selector1, c_float_3p2 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_4x48:
	{
		selector1 = _mm512_setzero_ps();
		selector2 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__mmask16 relu_cmp_mask;

		// c[0, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_0p0)

		// c[0, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_0p1)

		// c[0, 32-47]
		RELU_SCALE_OP_F32_AVX512(c_float_0p2)

		// c[1, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_1p0)

		// c[1, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_1p1)

		// c[1, 32-47]
		RELU_SCALE_OP_F32_AVX512(c_float_1p2)

		// c[2, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_2p0)

		// c[2, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_2p1)

		// c[2, 32-47]
		RELU_SCALE_OP_F32_AVX512(c_float_2p2)

		// c[3, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_3p0)

		// c[3, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_3p1)

		// c[3, 32-47]
		RELU_SCALE_OP_F32_AVX512(c_float_3p2)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_TANH_4x48:
	{
		__m512 dn, z, x, r2, r, x_tanh;
		__m512i q;

		// c[0, 0-15]
		GELU_TANH_F32_AVX512(c_float_0p0, r, r2, x, z, dn, x_tanh, q)

		// c[0, 16-31]
		GELU_TANH_F32_AVX512(c_float_0p1, r, r2, x, z, dn, x_tanh, q)

		// c[0, 32-47]
		GELU_TANH_F32_AVX512(c_float_0p2, r, r2, x, z, dn, x_tanh, q)

		// c[1, 0-15]
		GELU_TANH_F32_AVX512(c_float_1p0, r, r2, x, z, dn, x_tanh, q)

		// c[1, 16-31]
		GELU_TANH_F32_AVX512(c_float_1p1, r, r2, x, z, dn, x_tanh, q)

		// c[1, 32-47]
		GELU_TANH_F32_AVX512(c_float_1p2, r, r2, x, z, dn, x_tanh, q)

		// c[2, 0-15]
		GELU_TANH_F32_AVX512(c_float_2p0, r, r2, x, z, dn, x_tanh, q)

		// c[2, 16-31]
		GELU_TANH_F32_AVX512(c_float_2p1, r, r2, x, z, dn, x_tanh, q)

		// c[2, 32-47]
		GELU_TANH_F32_AVX512(c_float_2p2, r, r2, x, z, dn, x_tanh, q)

		// c[3, 0-15]
		GELU_TANH_F32_AVX512(c_float_3p0, r, r2, x, z, dn, x_tanh, q)

		// c[3, 16-31]
		GELU_TANH_F32_AVX512(c_float_3p1, r, r2, x, z, dn, x_tanh, q)

		// c[3, 32-47]
		GELU_TANH_F32_AVX512(c_float_3p2, r, r2, x, z, dn, x_tanh, q)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_ERF_4x48:
	{
		__m512 x, r, x_erf;

		// c[0, 0-15]
		GELU_ERF_F32_AVX512(c_float_0p0, r, x, x_erf)

		// c[0, 16-31]
		GELU_ERF_F32_AVX512(c_float_0p1, r, x, x_erf)

		// c[0, 32-47]
		GELU_ERF_F32_AVX512(c_float_0p2, r, x, x_erf)

		// c[1, 0-15]
		GELU_ERF_F32_AVX512(c_float_1p0, r, x, x_erf)

		// c[1, 16-31]
		GELU_ERF_F32_AVX512(c_float_1p1, r, x, x_erf)

		// c[1, 32-47]
		GELU_ERF_F32_AVX512(c_float_1p2, r, x, x_erf)

		// c[2, 0-15]
		GELU_ERF_F32_AVX512(c_float_2p0, r, x, x_erf)

		// c[2, 16-31]
		GELU_ERF_F32_AVX512(c_float_2p1, r, x, x_erf)

		// c[2, 32-47]
		GELU_ERF_F32_AVX512(c_float_2p2, r, x, x_erf)

		// c[3, 0-15]
		GELU_ERF_F32_AVX512(c_float_3p0, r, x, x_erf)

		// c[3, 16-31]
		GELU_ERF_F32_AVX512(c_float_3p1, r, x, x_erf)

		// c[3, 32-47]
		GELU_ERF_F32_AVX512(c_float_3p2, r, x, x_erf)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_CLIP_4x48:
	{
		__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
		__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

		// c[0, 0-15]
		CLIP_F32_AVX512(c_float_0p0, min, max)

		// c[0, 16-31]
		CLIP_F32_AVX512(c_float_0p1, min, max)

		// c[0, 32-47]
		CLIP_F32_AVX512(c_float_0p2, min, max)

		// c[1, 0-15]
		CLIP_F32_AVX512(c_float_1p0, min, max)

		// c[1, 16-31]
		CLIP_F32_AVX512(c_float_1p1, min, max)

		// c[1, 32-47]
		CLIP_F32_AVX512(c_float_1p2, min, max)

		// c[2, 0-15]
		CLIP_F32_AVX512(c_float_2p0, min, max)

		// c[2, 16-31]
		CLIP_F32_AVX512(c_float_2p1, min, max)

		// c[2, 32-47]
		CLIP_F32_AVX512(c_float_2p2, min, max)

		// c[3, 0-15]
		CLIP_F32_AVX512(c_float_3p0, min, max)

		// c[3, 16-31]
		CLIP_F32_AVX512(c_float_3p1, min, max)

		// c[3, 32-47]
		CLIP_F32_AVX512(c_float_3p2, min, max)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_DOWNSCALE_4x48:
	{
		__m512 selector3 = _mm512_setzero_ps();
		__m512 selector4 = _mm512_setzero_ps();

		__m512 zero_point0 = _mm512_setzero_ps();
		__m512 zero_point1 = _mm512_setzero_ps();
		__m512 zero_point2 = _mm512_setzero_ps();
		__m512 zero_point3 = _mm512_setzero_ps();

		__mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );

		// Need to account for row vs column major swaps. For scalars
		// scale and zero point, no implications.
		// Even though different registers are used for scalar in column
		// and row major downscale path, all those registers will contain
		// the same value.
		if ( post_ops_list_temp->scale_factor_len == 1 )
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector4 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
		}

		// bf16 zero point value (scalar or vector).
		if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
		{
			zero_point0 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point1 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point2 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point3 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
		}

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				selector2 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				selector3 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 2 * 16 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) ) );
				zero_point2 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 2 * 16 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[0, 16-31]
			SCL_MULRND_F32(c_float_0p1,selector2,zero_point1);

			// c[0, 32-47]
			SCL_MULRND_F32(c_float_0p2,selector3,zero_point2);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector1,zero_point0);

			// c[1, 16-31]
			SCL_MULRND_F32(c_float_1p1,selector2,zero_point1);

			// c[1, 32-47]
			SCL_MULRND_F32(c_float_1p2,selector3,zero_point2);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector1,zero_point0);

			// c[2, 16-31]
			SCL_MULRND_F32(c_float_2p1,selector2,zero_point1);

			// c[2, 32-47]
			SCL_MULRND_F32(c_float_2p2,selector3,zero_point2);

			// c[3, 0-15]
			SCL_MULRND_F32(c_float_3p0,selector1,zero_point0);

			// c[3, 16-31]
			SCL_MULRND_F32(c_float_3p1,selector2,zero_point1);

			// c[3, 32-47]
			SCL_MULRND_F32(c_float_3p2,selector3,zero_point2);
		}
		else
		{
			// If original output was columns major, then by the time
			// kernel sees it, the matrix would be accessed as if it were
			// transposed. Due to this the scale as well as zp array will
			// be accessed by the ic index, and each scale/zp element
			// corresponds to an entire row of the transposed output array,
			// instead of an entire column.
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 1 ) );
				selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 2 ) );
				selector4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 3 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 0 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 1 ) ) );
				zero_point2 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 2 ) ) );
				zero_point3 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 3 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[0, 16-31]
			SCL_MULRND_F32(c_float_0p1,selector1,zero_point0);

			// c[0, 32-47]
			SCL_MULRND_F32(c_float_0p2,selector1,zero_point0);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector2,zero_point1);

			// c[1, 16-31]
			SCL_MULRND_F32(c_float_1p1,selector2,zero_point1);

			// c[1, 32-47]
			SCL_MULRND_F32(c_float_1p2,selector2,zero_point1);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector3,zero_point2);

			// c[2, 16-31]
			SCL_MULRND_F32(c_float_2p1,selector3,zero_point2);

			// c[2, 32-47]
			SCL_MULRND_F32(c_float_2p2,selector3,zero_point2);

			// c[3, 0-15]
			SCL_MULRND_F32(c_float_3p0,selector4,zero_point3);

			// c[3, 16-31]
			SCL_MULRND_F32(c_float_3p1,selector4,zero_point3);

			// c[3, 32-47]
			SCL_MULRND_F32(c_float_3p2,selector4,zero_point3);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_ADD_4x48:
	{
		__m512 selector3;
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47]
			BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,0);

			// c[1:0-15,16-31,32-47]
			BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,1);

			// c[2:0-15,16-31,32-47]
			BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,2);

			// c[3:0-15,16-31,32-47]
			BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,3);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47]
			F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,0);

			// c[1:0-15,16-31,32-47]
			F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,1);

			// c[2:0-15,16-31,32-47]
			F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,2);

			// c[3:0-15,16-31,32-47]
			F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,3);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_MUL_4x48:
	{
		__m512 selector3;
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47]
			BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,0);

			// c[1:0-15,16-31,32-47]
			BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,1);

			// c[2:0-15,16-31,32-47]
			BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,2);

			// c[3:0-15,16-31,32-47]
			BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,3);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47]
			F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,0);

			// c[1:0-15,16-31,32-47]
			F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,1);

			// c[2:0-15,16-31,32-47]
			F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,2);

			// c[3:0-15,16-31,32-47]
			F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,3);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_SWISH_4x48:
	{
		selector1 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__m512 al_in, r, r2, z, dn;
		__m512i ex_out;

		// c[0, 0-15]
		SWISH_F32_AVX512_DEF(c_float_0p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[0, 16-31]
		SWISH_F32_AVX512_DEF(c_float_0p1, selector1, al_in, r, r2, z, dn, ex_out);

		// c[0, 32-47]
		SWISH_F32_AVX512_DEF(c_float_0p2, selector1, al_in, r, r2, z, dn, ex_out);

		// c[1, 0-15]
		SWISH_F32_AVX512_DEF(c_float_1p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[1, 16-31]
		SWISH_F32_AVX512_DEF(c_float_1p1, selector1, al_in, r, r2, z, dn, ex_out);

		// c[1, 32-47]
		SWISH_F32_AVX512_DEF(c_float_1p2, selector1, al_in, r, r2, z, dn, ex_out);

		// c[2, 0-15]
		SWISH_F32_AVX512_DEF(c_float_2p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[2, 16-31]
		SWISH_F32_AVX512_DEF(c_float_2p1, selector1, al_in, r, r2, z, dn, ex_out);

		// c[2, 32-47]
		SWISH_F32_AVX512_DEF(c_float_2p2, selector1, al_in, r, r2, z, dn, ex_out);

		// c[3, 0-15]
		SWISH_F32_AVX512_DEF(c_float_3p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[3, 16-31]
		SWISH_F32_AVX512_DEF(c_float_3p1, selector1, al_in, r, r2, z, dn, ex_out);

		// c[3, 32-47]
		SWISH_F32_AVX512_DEF(c_float_3p2, selector1, al_in, r, r2, z, dn, ex_out);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_4x48_DISABLE:
	;

	if ( ( post_ops_attr.buf_downscale != NULL ) &&
			 ( post_ops_attr.is_last_k == TRUE ) )
	{
		// Generate a mask16 of all 1's.
		__m512i selector_a = _mm512_setzero_epi32();
		__m512i selector_b = _mm512_set1_epi32( 10 );
		__mmask16 mask_all1 = _mm512_cmplt_epi32_mask( selector_a, selector_b );

		// Store the results in downscaled type (bf16 instead of float).

		// c[0, 0-15]
		CVT_STORE_F32_BF16_MASK(c_float_0p0,0,0);

		// c[0, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_0p1,0,1);

		// c[0, 32-47]
		CVT_STORE_F32_BF16_MASK(c_float_0p2,0,2);

		// c[1, 0-15]
		CVT_STORE_F32_BF16_MASK(c_float_1p0,1,0);

		// c[1, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_1p1,1,1);

		// c[1, 32-47]
		CVT_STORE_F32_BF16_MASK(c_float_1p2,1,2);

		// c[2, 0-15]
		CVT_STORE_F32_BF16_MASK(c_float_2p0,2,0);

		// c[2, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_2p1,2,1);

		// c[2, 32-47]
		CVT_STORE_F32_BF16_MASK(c_float_2p2,2,2);

		// c[3, 0-15]
		CVT_STORE_F32_BF16_MASK(c_float_3p0,3,0);

		// c[3, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_3p1,3,1);

		// c[3, 32-47]
		CVT_STORE_F32_BF16_MASK(c_float_3p2,3,2);
	}

	else
	{
		// Store the results.
		// c[0,0-15]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 0*16 ), c_float_0p0 );

		// c[0, 16-31]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 1*16 ), c_float_0p1 );

		// c[0,32-47]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 2*16 ), c_float_0p2 );

		// c[1,0-15]
		_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 0*16 ), c_float_1p0 );

		// c[1,16-31]
		_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 1*16 ), c_float_1p1 );

		// c[1,32-47]
		_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 2*16 ), c_float_1p2 );

		// c[2,0-15]
		_mm512_storeu_ps( c + ( rs_c * 2 ) + ( 0*16 ), c_float_2p0 );

		// c[2,16-31]
		_mm512_storeu_ps( c + ( rs_c * 2 ) + ( 1*16 ), c_float_2p1 );

		// c[2,32-47]
		_mm512_storeu_ps( c + ( rs_c * 2 ) + ( 2*16 ), c_float_2p2 );

		// c[3,0-15]
		_mm512_storeu_ps( c + ( rs_c * 3 ) + ( 0*16 ), c_float_3p0 );

		// c[3,16-31]
		_mm512_storeu_ps( c + ( rs_c * 3 ) + ( 1*16 ), c_float_3p1 );

	// c[3,32-47]
	_mm512_storeu_ps( c + ( rs_c * 3 ) + ( 2*16 ), c_float_3p2 );
	}
}

// 3x48 bf16 kernel
LPGEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_3x48)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_3x48_DISABLE,
						  &&POST_OPS_BIAS_3x48,
						  &&POST_OPS_RELU_3x48,
						  &&POST_OPS_RELU_SCALE_3x48,
						  &&POST_OPS_GELU_TANH_3x48,
						  &&POST_OPS_GELU_ERF_3x48,
						  &&POST_OPS_CLIP_3x48,
						  &&POST_OPS_DOWNSCALE_3x48,
						  &&POST_OPS_MATRIX_ADD_3x48,
						  &&POST_OPS_SWISH_3x48,
						  &&POST_OPS_MATRIX_MUL_3x48
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int16_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;
	__m512bh b1;
	__m512bh b2;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();
	__m512 c_float_0p1 = _mm512_setzero_ps();
	__m512 c_float_0p2 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();
	__m512 c_float_1p1 = _mm512_setzero_ps();
	__m512 c_float_1p2 = _mm512_setzero_ps();

	__m512 c_float_2p0 = _mm512_setzero_ps();
	__m512 c_float_2p1 = _mm512_setzero_ps();
	__m512 c_float_2p2 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 1 ) );
		b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 2 ) );

		// Broadcast a[0,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-47] = a[0,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
		c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );

		// Broadcast a[1,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 1 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-47] = a[1,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );
		c_float_1p2 = _mm512_dpbf16_ps( c_float_1p2, a_bf16_0, b2 );

		// Broadcast a[2,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 2 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-47] = a[2,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
		c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );
		c_float_2p2 = _mm512_dpbf16_ps( c_float_2p2, a_bf16_0, b2 );
	}
	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 1 ) );
		b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 2 ) );

		// Broadcast a[0,kr:kr+2].
		a_kfringe_buf = *( a + (rs_a * 0) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-47] = a[0,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
		c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );

		// Broadcast a[1,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 1) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-47] = a[1,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );
		c_float_1p2 = _mm512_dpbf16_ps( c_float_1p2, a_bf16_0, b2 );

		// Broadcast a[2,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 2) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-47] = a[2,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
		c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );
		c_float_2p2 = _mm512_dpbf16_ps( c_float_2p2, a_bf16_0, b2 );
	}

	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	if ( alpha != 1 )
	{
		// Scale by alpha
		c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );
		c_float_0p1 = _mm512_mul_ps( selector1, c_float_0p1 );
		c_float_0p2 = _mm512_mul_ps( selector1, c_float_0p2 );

		c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );
		c_float_1p1 = _mm512_mul_ps( selector1, c_float_1p1 );
		c_float_1p2 = _mm512_mul_ps( selector1, c_float_1p2 );

		c_float_2p0 = _mm512_mul_ps( selector1, c_float_2p0 );
		c_float_2p1 = _mm512_mul_ps( selector1, c_float_2p1 );
		c_float_2p2 = _mm512_mul_ps( selector1, c_float_2p2 );
	}

	// Scale C by beta.
	if ( beta != 0 )
	{
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
				 ( post_ops_attr.is_first_k == TRUE ) )
			{
				// c[0,0-15]
				BF16_F32_BETA_OP(c_float_0p0,0,0,0,selector1,selector2)

				// c[0, 16-31]
				BF16_F32_BETA_OP(c_float_0p1,0,0,1,selector1,selector2)

				// c[0,32-47]
				BF16_F32_BETA_OP(c_float_0p2,0,0,2,selector1,selector2)

				// c[1,0-15]
				BF16_F32_BETA_OP(c_float_1p0,0,1,0,selector1,selector2)

				// c[1,16-31]
				BF16_F32_BETA_OP(c_float_1p1,0,1,1,selector1,selector2)

				// c[1,32-47]
				BF16_F32_BETA_OP(c_float_1p2,0,1,2,selector1,selector2)

				// c[2,0-15]
				BF16_F32_BETA_OP(c_float_2p0,0,2,0,selector1,selector2)

				// c[2,16-31]
				BF16_F32_BETA_OP(c_float_2p1,0,2,1,selector1,selector2)

				// c[2,32-47]
				BF16_F32_BETA_OP(c_float_2p2,0,2,2,selector1,selector2)
			}
			else
			{
				// c[0,0-15]
				F32_F32_BETA_OP(c_float_0p0,0,0,0,selector1,selector2)

				// c[0, 16-31]
				F32_F32_BETA_OP(c_float_0p1,0,0,1,selector1,selector2)

				// c[0,32-47]
				F32_F32_BETA_OP(c_float_0p2,0,0,2,selector1,selector2)

				// c[1,0-15]
				F32_F32_BETA_OP(c_float_1p0,0,1,0,selector1,selector2)

				// c[1,16-31]
				F32_F32_BETA_OP(c_float_1p1,0,1,1,selector1,selector2)

				// c[1,32-47]
				F32_F32_BETA_OP(c_float_1p2,0,1,2,selector1,selector2)

				// c[2,0-15]
				F32_F32_BETA_OP(c_float_2p0,0,2,0,selector1,selector2)

				// c[2,16-31]
				F32_F32_BETA_OP(c_float_2p1,0,2,1,selector1,selector2)

				// c[2,32-47]
				F32_F32_BETA_OP(c_float_2p2,0,2,2,selector1,selector2)
			}
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_3x48:
	{
		__m512 selector3;

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_LOAD(selector1, bias_mask, 0);
				BF16_F32_BIAS_LOAD(selector2, bias_mask, 1);
				BF16_F32_BIAS_LOAD(selector3, bias_mask, 2);
			}
			else
			{
				selector1 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				selector2 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				selector3 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 2 * 16 ) );
			}

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_add_ps( selector2, c_float_0p1 );

			// c[0,32-47]
			c_float_0p2 = _mm512_add_ps( selector3, c_float_0p2 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

			// c[1, 16-31]
			c_float_1p1 = _mm512_add_ps( selector2, c_float_1p1 );

			// c[1,32-47]
			c_float_1p2 = _mm512_add_ps( selector3, c_float_1p2 );

			// c[2,0-15]
			c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );

			// c[2, 16-31]
			c_float_2p1 = _mm512_add_ps( selector2, c_float_2p1 );

			// c[2,32-47]
			c_float_2p2 = _mm512_add_ps( selector3, c_float_2p2 );
		}
		else
		{
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_BCAST(selector1, bias_mask, 0);
				BF16_F32_BIAS_BCAST(selector2, bias_mask, 1);
				BF16_F32_BIAS_BCAST(selector3, bias_mask, 2);
			}
			else
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 1 ) );
				selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 2 ) );
			}

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_add_ps( selector1, c_float_0p1 );

			// c[0,32-47]
			c_float_0p2 = _mm512_add_ps( selector1, c_float_0p2 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );

			// c[1, 16-31]
			c_float_1p1 = _mm512_add_ps( selector2, c_float_1p1 );

			// c[1,32-47]
			c_float_1p2 = _mm512_add_ps( selector2, c_float_1p2 );

			// c[2,0-15]
			c_float_2p0 = _mm512_add_ps( selector3, c_float_2p0 );

			// c[2, 16-31]
			c_float_2p1 = _mm512_add_ps( selector3, c_float_2p1 );

			// c[2,32-47]
			c_float_2p2 = _mm512_add_ps( selector3, c_float_2p2 );
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_3x48:
	{
		selector1 = _mm512_setzero_ps();

		// c[0,0-15]
		c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

		// c[0, 16-31]
		c_float_0p1 = _mm512_max_ps( selector1, c_float_0p1 );

		// c[0,32-47]
		c_float_0p2 = _mm512_max_ps( selector1, c_float_0p2 );

		// c[1,0-15]
		c_float_1p0 = _mm512_max_ps( selector1, c_float_1p0 );

		// c[1,16-31]
		c_float_1p1 = _mm512_max_ps( selector1, c_float_1p1 );

		// c[1,32-47]
		c_float_1p2 = _mm512_max_ps( selector1, c_float_1p2 );

		// c[2,0-15]
		c_float_2p0 = _mm512_max_ps( selector1, c_float_2p0 );

		// c[2,16-31]
		c_float_2p1 = _mm512_max_ps( selector1, c_float_2p1 );

		// c[2,32-47]
		c_float_2p2 = _mm512_max_ps( selector1, c_float_2p2 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_3x48:
	{
		selector1 = _mm512_setzero_ps();
		selector2 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__mmask16 relu_cmp_mask;

		// c[0, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_0p0)

		// c[0, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_0p1)

		// c[0, 32-47]
		RELU_SCALE_OP_F32_AVX512(c_float_0p2)

		// c[1, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_1p0)

		// c[1, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_1p1)

		// c[1, 32-47]
		RELU_SCALE_OP_F32_AVX512(c_float_1p2)

		// c[2, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_2p0)

		// c[2, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_2p1)

		// c[2, 32-47]
		RELU_SCALE_OP_F32_AVX512(c_float_2p2)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_TANH_3x48:
	{
		__m512 dn, z, x, r2, r, x_tanh;
		__m512i q;

		// c[0, 0-15]
		GELU_TANH_F32_AVX512(c_float_0p0, r, r2, x, z, dn, x_tanh, q)

		// c[0, 16-31]
		GELU_TANH_F32_AVX512(c_float_0p1, r, r2, x, z, dn, x_tanh, q)

		// c[0, 32-47]
		GELU_TANH_F32_AVX512(c_float_0p2, r, r2, x, z, dn, x_tanh, q)

		// c[1, 0-15]
		GELU_TANH_F32_AVX512(c_float_1p0, r, r2, x, z, dn, x_tanh, q)

		// c[1, 16-31]
		GELU_TANH_F32_AVX512(c_float_1p1, r, r2, x, z, dn, x_tanh, q)

		// c[1, 32-47]
		GELU_TANH_F32_AVX512(c_float_1p2, r, r2, x, z, dn, x_tanh, q)

		// c[2, 0-15]
		GELU_TANH_F32_AVX512(c_float_2p0, r, r2, x, z, dn, x_tanh, q)

		// c[2, 16-31]
		GELU_TANH_F32_AVX512(c_float_2p1, r, r2, x, z, dn, x_tanh, q)

		// c[2, 32-47]
		GELU_TANH_F32_AVX512(c_float_2p2, r, r2, x, z, dn, x_tanh, q)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_ERF_3x48:
	{
		__m512 x, r, x_erf;

		// c[0, 0-15]
		GELU_ERF_F32_AVX512(c_float_0p0, r, x, x_erf)

		// c[0, 16-31]
		GELU_ERF_F32_AVX512(c_float_0p1, r, x, x_erf)

		// c[0, 32-47]
		GELU_ERF_F32_AVX512(c_float_0p2, r, x, x_erf)

		// c[1, 0-15]
		GELU_ERF_F32_AVX512(c_float_1p0, r, x, x_erf)

		// c[1, 16-31]
		GELU_ERF_F32_AVX512(c_float_1p1, r, x, x_erf)

		// c[1, 32-47]
		GELU_ERF_F32_AVX512(c_float_1p2, r, x, x_erf)

		// c[2, 0-15]
		GELU_ERF_F32_AVX512(c_float_2p0, r, x, x_erf)

		// c[2, 16-31]
		GELU_ERF_F32_AVX512(c_float_2p1, r, x, x_erf)

		// c[2, 32-47]
		GELU_ERF_F32_AVX512(c_float_2p2, r, x, x_erf)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_CLIP_3x48:
	{
		__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
		__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

		// c[0, 0-15]
		CLIP_F32_AVX512(c_float_0p0, min, max)

		// c[0, 16-31]
		CLIP_F32_AVX512(c_float_0p1, min, max)

		// c[0, 32-47]
		CLIP_F32_AVX512(c_float_0p2, min, max)

		// c[1, 0-15]
		CLIP_F32_AVX512(c_float_1p0, min, max)

		// c[1, 16-31]
		CLIP_F32_AVX512(c_float_1p1, min, max)

		// c[1, 32-47]
		CLIP_F32_AVX512(c_float_1p2, min, max)

		// c[2, 0-15]
		CLIP_F32_AVX512(c_float_2p0, min, max)

		// c[2, 16-31]
		CLIP_F32_AVX512(c_float_2p1, min, max)

		// c[2, 32-47]
		CLIP_F32_AVX512(c_float_2p2, min, max)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}

POST_OPS_DOWNSCALE_3x48:
	{
		__m512 selector3 = _mm512_setzero_ps();

		__m512 zero_point0 = _mm512_setzero_ps();
		__m512 zero_point1 = _mm512_setzero_ps();
		__m512 zero_point2 = _mm512_setzero_ps();

		__mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );

		// Need to account for row vs column major swaps. For scalars
		// scale and zero point, no implications.
		// Even though different registers are used for scalar in column
		// and row major downscale path, all those registers will contain
		// the same value.
		if ( post_ops_list_temp->scale_factor_len == 1 )
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
		}

		// bf16 zero point value (scalar or vector).
		if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
		{
			zero_point0 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point1 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point2 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
		}

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				selector2 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				selector3 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 2 * 16 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) ) );
				zero_point2 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 2 * 16 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[0, 16-31]
			SCL_MULRND_F32(c_float_0p1,selector2,zero_point1);

			// c[0, 32-47]
			SCL_MULRND_F32(c_float_0p2,selector3,zero_point2);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector1,zero_point0);

			// c[1, 16-31]
			SCL_MULRND_F32(c_float_1p1,selector2,zero_point1);

			// c[1, 32-47]
			SCL_MULRND_F32(c_float_1p2,selector3,zero_point2);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector1,zero_point0);

			// c[2, 16-31]
			SCL_MULRND_F32(c_float_2p1,selector2,zero_point1);

			// c[2, 32-47]
			SCL_MULRND_F32(c_float_2p2,selector3,zero_point2);
		}
		else
		{
			// If original output was columns major, then by the time
			// kernel sees it, the matrix would be accessed as if it were
			// transposed. Due to this the scale as well as zp array will
			// be accessed by the ic index, and each scale/zp element
			// corresponds to an entire row of the transposed output array,
			// instead of an entire column.
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 1 ) );
				selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 2 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 0 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 1 ) ) );
				zero_point2 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 2 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[0, 16-31]
			SCL_MULRND_F32(c_float_0p1,selector1,zero_point0);

			// c[0, 32-47]
			SCL_MULRND_F32(c_float_0p2,selector1,zero_point0);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector2,zero_point1);

			// c[1, 16-31]
			SCL_MULRND_F32(c_float_1p1,selector2,zero_point1);

			// c[1, 32-47]
			SCL_MULRND_F32(c_float_1p2,selector2,zero_point1);

			// c[2, 0-15]
			SCL_MULRND_F32(c_float_2p0,selector3,zero_point2);

			// c[2, 16-31]
			SCL_MULRND_F32(c_float_2p1,selector3,zero_point2);

			// c[2, 32-47]
			SCL_MULRND_F32(c_float_2p2,selector3,zero_point2);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_ADD_3x48:
	{
		__m512 selector3;
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47]
			BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,0);

			// c[1:0-15,16-31,32-47]
			BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,1);

			// c[2:0-15,16-31,32-47]
			BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,2);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47]
			F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,0);

			// c[1:0-15,16-31,32-47]
			F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,1);

			// c[2:0-15,16-31,32-47]
			F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,2);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_MUL_3x48:
	{
		__m512 selector3;
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47]
			BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,0);

			// c[1:0-15,16-31,32-47]
			BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,1);

			// c[2:0-15,16-31,32-47]
			BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,2);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47]
			F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,0);

			// c[1:0-15,16-31,32-47]
			F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,1);

			// c[2:0-15,16-31,32-47]
			F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,2);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}	
POST_OPS_SWISH_3x48:
	{
		selector1 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__m512 al_in, r, r2, z, dn;
		__m512i ex_out;

		// c[0, 0-15]
		SWISH_F32_AVX512_DEF(c_float_0p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[0, 16-31]
		SWISH_F32_AVX512_DEF(c_float_0p1, selector1, al_in, r, r2, z, dn, ex_out);

		// c[0, 32-47]
		SWISH_F32_AVX512_DEF(c_float_0p2, selector1, al_in, r, r2, z, dn, ex_out);

		// c[1, 0-15]
		SWISH_F32_AVX512_DEF(c_float_1p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[1, 16-31]
		SWISH_F32_AVX512_DEF(c_float_1p1, selector1, al_in, r, r2, z, dn, ex_out);

		// c[1, 32-47]
		SWISH_F32_AVX512_DEF(c_float_1p2, selector1, al_in, r, r2, z, dn, ex_out);

		// c[2, 0-15]
		SWISH_F32_AVX512_DEF(c_float_2p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[2, 16-31]
		SWISH_F32_AVX512_DEF(c_float_2p1, selector1, al_in, r, r2, z, dn, ex_out);

		// c[2, 32-47]
		SWISH_F32_AVX512_DEF(c_float_2p2, selector1, al_in, r, r2, z, dn, ex_out);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_3x48_DISABLE:
	;

	if ( ( post_ops_attr.buf_downscale != NULL ) &&
			 ( post_ops_attr.is_last_k == TRUE ) )
	{
		// Generate a mask16 of all 1's.
		__m512i selector_a = _mm512_setzero_epi32();
		__m512i selector_b = _mm512_set1_epi32( 10 );
		__mmask16 mask_all1 = _mm512_cmplt_epi32_mask( selector_a, selector_b );

		// Store the results in downscaled type (bf16 instead of float).

		// c[0, 0-15]
		CVT_STORE_F32_BF16_MASK(c_float_0p0,0,0);

		// c[0, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_0p1,0,1);

		// c[0, 32-47]
		CVT_STORE_F32_BF16_MASK(c_float_0p2,0,2);

		// c[1, 0-15]
		CVT_STORE_F32_BF16_MASK(c_float_1p0,1,0);

		// c[1, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_1p1,1,1);

		// c[1, 32-47]
		CVT_STORE_F32_BF16_MASK(c_float_1p2,1,2);

		// c[2, 0-15]
		CVT_STORE_F32_BF16_MASK(c_float_2p0,2,0);

		// c[2, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_2p1,2,1);

		// c[2, 32-47]
		CVT_STORE_F32_BF16_MASK(c_float_2p2,2,2);
	}

	else
	{
		// Store the results.
		// c[0,0-15]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 0*16 ), c_float_0p0 );

		// c[0, 16-31]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 1*16 ), c_float_0p1 );

		// c[0,32-47]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 2*16 ), c_float_0p2 );

		// c[1,0-15]
		_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 0*16 ), c_float_1p0 );

		// c[1,16-31]
		_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 1*16 ), c_float_1p1 );

		// c[1,32-47]
		_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 2*16 ), c_float_1p2 );

		// c[2,0-15]
		_mm512_storeu_ps( c + ( rs_c * 2 ) + ( 0*16 ), c_float_2p0 );

		// c[2,16-31]
		_mm512_storeu_ps( c + ( rs_c * 2 ) + ( 1*16 ), c_float_2p1 );

		// c[2,32-47]
		_mm512_storeu_ps( c + ( rs_c * 2 ) + ( 2*16 ), c_float_2p2 );
	}
}

// 2x48 bf16 kernel
LPGEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_2x48)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_2x48_DISABLE,
						  &&POST_OPS_BIAS_2x48,
						  &&POST_OPS_RELU_2x48,
						  &&POST_OPS_RELU_SCALE_2x48,
						  &&POST_OPS_GELU_TANH_2x48,
						  &&POST_OPS_GELU_ERF_2x48,
						  &&POST_OPS_CLIP_2x48,
						  &&POST_OPS_DOWNSCALE_2x48,
						  &&POST_OPS_MATRIX_ADD_2x48,
						  &&POST_OPS_SWISH_2x48,
						  &&POST_OPS_MATRIX_MUL_2x48
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int16_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;
	__m512bh b1;
	__m512bh b2;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();
	__m512 c_float_0p1 = _mm512_setzero_ps();
	__m512 c_float_0p2 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();
	__m512 c_float_1p1 = _mm512_setzero_ps();
	__m512 c_float_1p2 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 1 ) );
		b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 2 ) );

		// Broadcast a[0,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-47] = a[0,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
		c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );

		// Broadcast a[1,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 1 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-47] = a[1,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );
		c_float_1p2 = _mm512_dpbf16_ps( c_float_1p2, a_bf16_0, b2 );
	}
	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 1 ) );
		b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 2 ) );

		// Broadcast a[0,kr:kr+2].
		a_kfringe_buf = *( a + (rs_a * 0) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-47] = a[0,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
		c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );

		// Broadcast a[1,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 1) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-47] = a[1,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );
		c_float_1p2 = _mm512_dpbf16_ps( c_float_1p2, a_bf16_0, b2 );
	}

	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	if ( alpha != 1 )
	{
		// Scale by alpha
		c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );
		c_float_0p1 = _mm512_mul_ps( selector1, c_float_0p1 );
		c_float_0p2 = _mm512_mul_ps( selector1, c_float_0p2 );

		c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );
		c_float_1p1 = _mm512_mul_ps( selector1, c_float_1p1 );
		c_float_1p2 = _mm512_mul_ps( selector1, c_float_1p2 );
	}

	// Scale C by beta.
	if ( beta != 0 )
	{
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
				 ( post_ops_attr.is_first_k == TRUE ) )
			{
				// c[0,0-15]
				BF16_F32_BETA_OP(c_float_0p0,0,0,0,selector1,selector2)

				// c[0, 16-31]
				BF16_F32_BETA_OP(c_float_0p1,0,0,1,selector1,selector2)

				// c[0,32-47]
				BF16_F32_BETA_OP(c_float_0p2,0,0,2,selector1,selector2)

				// c[1,0-15]
				BF16_F32_BETA_OP(c_float_1p0,0,1,0,selector1,selector2)

				// c[1,16-31]
				BF16_F32_BETA_OP(c_float_1p1,0,1,1,selector1,selector2)

				// c[1,32-47]
				BF16_F32_BETA_OP(c_float_1p2,0,1,2,selector1,selector2)
			}
			else
			{
				// c[0,0-15]
				F32_F32_BETA_OP(c_float_0p0,0,0,0,selector1,selector2)

				// c[0, 16-31]
				F32_F32_BETA_OP(c_float_0p1,0,0,1,selector1,selector2)

				// c[0,32-47]
				F32_F32_BETA_OP(c_float_0p2,0,0,2,selector1,selector2)

				// c[1,0-15]
				F32_F32_BETA_OP(c_float_1p0,0,1,0,selector1,selector2)

				// c[1,16-31]
				F32_F32_BETA_OP(c_float_1p1,0,1,1,selector1,selector2)

				// c[1,32-47]
				F32_F32_BETA_OP(c_float_1p2,0,1,2,selector1,selector2)
			}
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_2x48:
	{
		__m512 selector3;

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_LOAD(selector1, bias_mask, 0);
				BF16_F32_BIAS_LOAD(selector2, bias_mask, 1);
				BF16_F32_BIAS_LOAD(selector3, bias_mask, 2);
			}
			else
			{
				selector1 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				selector2 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				selector3 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 2 * 16 ) );
			}

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_add_ps( selector2, c_float_0p1 );

			// c[0,32-47]
			c_float_0p2 = _mm512_add_ps( selector3, c_float_0p2 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

			// c[1, 16-31]
			c_float_1p1 = _mm512_add_ps( selector2, c_float_1p1 );

			// c[1,32-47]
			c_float_1p2 = _mm512_add_ps( selector3, c_float_1p2 );
		}
		else
		{
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_BCAST(selector1, bias_mask, 0);
				BF16_F32_BIAS_BCAST(selector2, bias_mask, 1);
			}
			else
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 1 ) );
			}

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_add_ps( selector1, c_float_0p1 );

			// c[0,32-47]
			c_float_0p2 = _mm512_add_ps( selector1, c_float_0p2 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );

			// c[1, 16-31]
			c_float_1p1 = _mm512_add_ps( selector2, c_float_1p1 );

			// c[1,32-47]
			c_float_1p2 = _mm512_add_ps( selector2, c_float_1p2 );
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_2x48:
	{
		selector1 = _mm512_setzero_ps();

		// c[0,0-15]
		c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

		// c[0, 16-31]
		c_float_0p1 = _mm512_max_ps( selector1, c_float_0p1 );

		// c[0,32-47]
		c_float_0p2 = _mm512_max_ps( selector1, c_float_0p2 );

		// c[1,0-15]
		c_float_1p0 = _mm512_max_ps( selector1, c_float_1p0 );

		// c[1,16-31]
		c_float_1p1 = _mm512_max_ps( selector1, c_float_1p1 );

		// c[1,32-47]
		c_float_1p2 = _mm512_max_ps( selector1, c_float_1p2 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_2x48:
	{
		selector1 = _mm512_setzero_ps();
		selector2 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__mmask16 relu_cmp_mask;

		// c[0, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_0p0)

		// c[0, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_0p1)

		// c[0, 32-47]
		RELU_SCALE_OP_F32_AVX512(c_float_0p2)

		// c[1, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_1p0)

		// c[1, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_1p1)

		// c[1, 32-47]
		RELU_SCALE_OP_F32_AVX512(c_float_1p2)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_TANH_2x48:
	{
		__m512 dn, z, x, r2, r, x_tanh;
		__m512i q;

		// c[0, 0-15]
		GELU_TANH_F32_AVX512(c_float_0p0, r, r2, x, z, dn, x_tanh, q)

		// c[0, 16-31]
		GELU_TANH_F32_AVX512(c_float_0p1, r, r2, x, z, dn, x_tanh, q)

		// c[0, 32-47]
		GELU_TANH_F32_AVX512(c_float_0p2, r, r2, x, z, dn, x_tanh, q)

		// c[1, 0-15]
		GELU_TANH_F32_AVX512(c_float_1p0, r, r2, x, z, dn, x_tanh, q)

		// c[1, 16-31]
		GELU_TANH_F32_AVX512(c_float_1p1, r, r2, x, z, dn, x_tanh, q)

		// c[1, 32-47]
		GELU_TANH_F32_AVX512(c_float_1p2, r, r2, x, z, dn, x_tanh, q)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_ERF_2x48:
	{
		__m512 x, r, x_erf;

		// c[0, 0-15]
		GELU_ERF_F32_AVX512(c_float_0p0, r, x, x_erf)

		// c[0, 16-31]
		GELU_ERF_F32_AVX512(c_float_0p1, r, x, x_erf)

		// c[0, 32-47]
		GELU_ERF_F32_AVX512(c_float_0p2, r, x, x_erf)

		// c[1, 0-15]
		GELU_ERF_F32_AVX512(c_float_1p0, r, x, x_erf)

		// c[1, 16-31]
		GELU_ERF_F32_AVX512(c_float_1p1, r, x, x_erf)

		// c[1, 32-47]
		GELU_ERF_F32_AVX512(c_float_1p2, r, x, x_erf)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_CLIP_2x48:
	{
		__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
		__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

		// c[0, 0-15]
		CLIP_F32_AVX512(c_float_0p0, min, max)

		// c[0, 16-31]
		CLIP_F32_AVX512(c_float_0p1, min, max)

		// c[0, 32-47]
		CLIP_F32_AVX512(c_float_0p2, min, max)

		// c[1, 0-15]
		CLIP_F32_AVX512(c_float_1p0, min, max)

		// c[1, 16-31]
		CLIP_F32_AVX512(c_float_1p1, min, max)

		// c[1, 32-47]
		CLIP_F32_AVX512(c_float_1p2, min, max)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_DOWNSCALE_2x48:
	{
		__m512 selector3 = _mm512_setzero_ps();

		__m512 zero_point0 = _mm512_setzero_ps();
		__m512 zero_point1 = _mm512_setzero_ps();
		__m512 zero_point2 = _mm512_setzero_ps();

		__mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );

		// Need to account for row vs column major swaps. For scalars
		// scale and zero point, no implications.
		// Even though different registers are used for scalar in column
		// and row major downscale path, all those registers will contain
		// the same value.
		if ( post_ops_list_temp->scale_factor_len == 1 )
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
		}

		// bf16 zero point value (scalar or vector).
		if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
		{
			zero_point0 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point1 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point2 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
		}

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				selector2 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				selector3 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 2 * 16 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) ) );
				zero_point2 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 2 * 16 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[0, 16-31]
			SCL_MULRND_F32(c_float_0p1,selector2,zero_point1);

			// c[0, 32-47]
			SCL_MULRND_F32(c_float_0p2,selector3,zero_point2);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector1,zero_point0);

			// c[1, 16-31]
			SCL_MULRND_F32(c_float_1p1,selector2,zero_point1);

			// c[1, 32-47]
			SCL_MULRND_F32(c_float_1p2,selector3,zero_point2);
		}
		else
		{
			// If original output was columns major, then by the time
			// kernel sees it, the matrix would be accessed as if it were
			// transposed. Due to this the scale as well as zp array will
			// be accessed by the ic index, and each scale/zp element
			// corresponds to an entire row of the transposed output array,
			// instead of an entire column.
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 1 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 0 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 1 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[0, 16-31]
			SCL_MULRND_F32(c_float_0p1,selector1,zero_point0);

			// c[0, 32-47]
			SCL_MULRND_F32(c_float_0p2,selector1,zero_point0);

			// c[1, 0-15]
			SCL_MULRND_F32(c_float_1p0,selector2,zero_point1);

			// c[1, 16-31]
			SCL_MULRND_F32(c_float_1p1,selector2,zero_point1);

			// c[1, 32-47]
			SCL_MULRND_F32(c_float_1p2,selector2,zero_point1);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_ADD_2x48:
	{
		__m512 selector3;
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47]
			BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,0);

			// c[1:0-15,16-31,32-47]
			BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,1);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47]
			F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,0);

			// c[1:0-15,16-31,32-47]
			F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,1);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_MUL_2x48:
	{
		__m512 selector3;
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47]
			BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,0);

			// c[1:0-15,16-31,32-47]
			BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,1);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47]
			F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,0);

			// c[1:0-15,16-31,32-47]
			F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,1);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}	
POST_OPS_SWISH_2x48:
	{
		selector1 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__m512 al_in, r, r2, z, dn;
		__m512i ex_out;

		// c[0, 0-15]
		SWISH_F32_AVX512_DEF(c_float_0p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[0, 16-31]
		SWISH_F32_AVX512_DEF(c_float_0p1, selector1, al_in, r, r2, z, dn, ex_out);

		// c[0, 32-47]
		SWISH_F32_AVX512_DEF(c_float_0p2, selector1, al_in, r, r2, z, dn, ex_out);

		// c[1, 0-15]
		SWISH_F32_AVX512_DEF(c_float_1p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[1, 16-31]
		SWISH_F32_AVX512_DEF(c_float_1p1, selector1, al_in, r, r2, z, dn, ex_out);

		// c[1, 32-47]
		SWISH_F32_AVX512_DEF(c_float_1p2, selector1, al_in, r, r2, z, dn, ex_out);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_2x48_DISABLE:
	;

	if ( ( post_ops_attr.buf_downscale != NULL ) &&
			 ( post_ops_attr.is_last_k == TRUE ) )
	{
		// Generate a mask16 of all 1's.
		__m512i selector_a = _mm512_setzero_epi32();
		__m512i selector_b = _mm512_set1_epi32( 10 );
		__mmask16 mask_all1 = _mm512_cmplt_epi32_mask( selector_a, selector_b );

		// Store the results in downscaled type (bf16 instead of float).

		// c[0, 0-15]
		CVT_STORE_F32_BF16_MASK(c_float_0p0,0,0);

		// c[0, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_0p1,0,1);

		// c[0, 32-47]
		CVT_STORE_F32_BF16_MASK(c_float_0p2,0,2);

		// c[1, 0-15]
		CVT_STORE_F32_BF16_MASK(c_float_1p0,1,0);

		// c[1, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_1p1,1,1);

		// c[1, 32-47]
		CVT_STORE_F32_BF16_MASK(c_float_1p2,1,2);
	}

	else
	{
		// Store the results.
		// c[0,0-15]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 0*16 ), c_float_0p0 );

		// c[0, 16-31]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 1*16 ), c_float_0p1 );

		// c[0,32-47]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 2*16 ), c_float_0p2 );

		// c[1,0-15]
		_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 0*16 ), c_float_1p0 );

		// c[1,16-31]
		_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 1*16 ), c_float_1p1 );

		// c[1,32-47]
		_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 2*16 ), c_float_1p2 );
	}
}

// 1x48 bf16 kernel
LPGEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_1x48)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_1x48_DISABLE,
						  &&POST_OPS_BIAS_1x48,
						  &&POST_OPS_RELU_1x48,
						  &&POST_OPS_RELU_SCALE_1x48,
						  &&POST_OPS_GELU_TANH_1x48,
						  &&POST_OPS_GELU_ERF_1x48,
						  &&POST_OPS_CLIP_1x48,
						  &&POST_OPS_DOWNSCALE_1x48,
						  &&POST_OPS_MATRIX_ADD_1x48,
						  &&POST_OPS_SWISH_1x48,
						  &&POST_OPS_MATRIX_MUL_1x48
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int16_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;
	__m512bh b1;
	__m512bh b2;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();
	__m512 c_float_0p1 = _mm512_setzero_ps();
	__m512 c_float_0p2 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 1 ) );
		b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 2 ) );

		// Broadcast a[0,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-47] = a[0,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
		c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );
	}
	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 1 ) );
		b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 2 ) );

		// Broadcast a[0,kr:kr+2].
		a_kfringe_buf = *( a + (rs_a * 0) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-47] = a[0,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
		c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );
	}

	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	if ( alpha != 1 )
	{
		// Scale by alpha
		c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );
		c_float_0p1 = _mm512_mul_ps( selector1, c_float_0p1 );
		c_float_0p2 = _mm512_mul_ps( selector1, c_float_0p2 );
	}

	// Scale C by beta.
	if ( beta != 0 )
	{
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
				 ( post_ops_attr.is_first_k == TRUE ) )
			{
				// c[0,0-15]
				BF16_F32_BETA_OP(c_float_0p0,0,0,0,selector1,selector2)

				// c[0, 16-31]
				BF16_F32_BETA_OP(c_float_0p1,0,0,1,selector1,selector2)

				// c[0,32-47]
				BF16_F32_BETA_OP(c_float_0p2,0,0,2,selector1,selector2)
			}
			else
			{
				// c[0,0-15]
				F32_F32_BETA_OP(c_float_0p0,0,0,0,selector1,selector2)

				// c[0, 16-31]
				F32_F32_BETA_OP(c_float_0p1,0,0,1,selector1,selector2)

				// c[0,32-47]
				F32_F32_BETA_OP(c_float_0p2,0,0,2,selector1,selector2)
			}
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_1x48:
	{
		__m512 selector3;

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_LOAD(selector1, bias_mask, 0);
				BF16_F32_BIAS_LOAD(selector2, bias_mask, 1);
				BF16_F32_BIAS_LOAD(selector3, bias_mask, 2);
			}
			else
			{
				selector1 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				selector2 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				selector3 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 2 * 16 ) );
			}

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_add_ps( selector2, c_float_0p1 );

			// c[0,32-47]
			c_float_0p2 = _mm512_add_ps( selector3, c_float_0p2 );
		}
		else
		{
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_BCAST(selector1, bias_mask, 0);
			}
			else
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_ops_attr.post_op_c_i + 0 ) );
			}

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_add_ps( selector1, c_float_0p1 );

			// c[0,32-47]
			c_float_0p2 = _mm512_add_ps( selector1, c_float_0p2 );
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_1x48:
	{
		selector1 = _mm512_setzero_ps();

		// c[0,0-15]
		c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

		// c[0, 16-31]
		c_float_0p1 = _mm512_max_ps( selector1, c_float_0p1 );

		// c[0,32-47]
		c_float_0p2 = _mm512_max_ps( selector1, c_float_0p2 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_1x48:
	{
		selector1 = _mm512_setzero_ps();
		selector2 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__mmask16 relu_cmp_mask;

		// c[0, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_0p0)

		// c[0, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_0p1)

		// c[0, 32-47]
		RELU_SCALE_OP_F32_AVX512(c_float_0p2)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_TANH_1x48:
	{
		__m512 dn, z, x, r2, r, x_tanh;
		__m512i q;

		// c[0, 0-15]
		GELU_TANH_F32_AVX512(c_float_0p0, r, r2, x, z, dn, x_tanh, q)

		// c[0, 16-31]
		GELU_TANH_F32_AVX512(c_float_0p1, r, r2, x, z, dn, x_tanh, q)

		// c[0, 32-47]
		GELU_TANH_F32_AVX512(c_float_0p2, r, r2, x, z, dn, x_tanh, q)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_GELU_ERF_1x48:
	{
		__m512 x, r, x_erf;

		// c[0, 0-15]
		GELU_ERF_F32_AVX512(c_float_0p0, r, x, x_erf)

		// c[0, 16-31]
		GELU_ERF_F32_AVX512(c_float_0p1, r, x, x_erf)

		// c[0, 32-47]
		GELU_ERF_F32_AVX512(c_float_0p2, r, x, x_erf)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_CLIP_1x48:
	{
		__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
		__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

		// c[0, 0-15]
		CLIP_F32_AVX512(c_float_0p0, min, max)

		// c[0, 16-31]
		CLIP_F32_AVX512(c_float_0p1, min, max)

		// c[0, 32-47]
		CLIP_F32_AVX512(c_float_0p2, min, max)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}

POST_OPS_DOWNSCALE_1x48:
	{
		__m512 selector3 = _mm512_setzero_ps();

		__m512 zero_point0 = _mm512_setzero_ps();
		__m512 zero_point1 = _mm512_setzero_ps();
		__m512 zero_point2 = _mm512_setzero_ps();

		__mmask16 zp_mask = _cvtu32_mask16( 0xFFFF );

		// Need to account for row vs column major swaps. For scalars
		// scale and zero point, no implications.
		// Even though different registers are used for scalar in column
		// and row major downscale path, all those registers will contain
		// the same value.
		if ( post_ops_list_temp->scale_factor_len == 1 )
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
		}

		// bf16 zero point value (scalar or vector).
		if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
		{
			zero_point0 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point1 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
			zero_point2 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_set1_epi16( zp_mask,
						*( ( bfloat16* )post_ops_list_temp->op_args1 ) ) );
		}

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				selector2 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				selector3 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 2 * 16 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
				zero_point1 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) ) );
				zero_point2 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
							( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_j + ( 2 * 16 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[0, 16-31]
			SCL_MULRND_F32(c_float_0p1,selector2,zero_point1);

			// c[0, 32-47]
			SCL_MULRND_F32(c_float_0p2,selector3,zero_point2);
		}
		else
		{
			// If original output was columns major, then by the time
			// kernel sees it, the matrix would be accessed as if it were
			// transposed. Due to this the scale as well as zp array will
			// be accessed by the ic index, and each scale/zp element
			// corresponds to an entire row of the transposed output array,
			// instead of an entire column.
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_i + 0 ) );
			}

			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_set1_epi16( zp_mask,
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +
							post_ops_attr.post_op_c_i + 0 ) ) );
			}

			// c[0, 0-15]
			SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

			// c[0, 16-31]
			SCL_MULRND_F32(c_float_0p1,selector1,zero_point0);

			// c[0, 32-47]
			SCL_MULRND_F32(c_float_0p2,selector1,zero_point0);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_ADD_1x48:
	{
		__m512 selector3;
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47]
			BF16_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,0);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47]
			F32_F32_MATRIX_ADD_3COL(selector1,selector2,selector3,0);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_MUL_1x48:
	{
		__m512 selector3;
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47]
			BF16_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,0);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47]
			F32_F32_MATRIX_MUL_3COL(selector1,selector2,selector3,0);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}	
POST_OPS_SWISH_1x48:
	{
		selector1 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__m512 al_in, r, r2, z, dn;
		__m512i ex_out;

		// c[0, 0-15]
		SWISH_F32_AVX512_DEF(c_float_0p0, selector1, al_in, r, r2, z, dn, ex_out);

		// c[0, 16-31]
		SWISH_F32_AVX512_DEF(c_float_0p1, selector1, al_in, r, r2, z, dn, ex_out);

		// c[0, 32-47]
		SWISH_F32_AVX512_DEF(c_float_0p2, selector1, al_in, r, r2, z, dn, ex_out);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_1x48_DISABLE:
	;

	if ( ( post_ops_attr.buf_downscale != NULL ) &&
			 ( post_ops_attr.is_last_k == TRUE ) )
	{
		// Generate a mask16 of all 1's.
		__m512i selector_a = _mm512_setzero_epi32();
		__m512i selector_b = _mm512_set1_epi32( 10 );
		__mmask16 mask_all1 = _mm512_cmplt_epi32_mask( selector_a, selector_b );

		// Store the results in downscaled type (bf16 instead of float).

		// c[0, 0-15]
		CVT_STORE_F32_BF16_MASK(c_float_0p0,0,0);

		// c[0, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_0p1,0,1);

		// c[0, 32-47]
		CVT_STORE_F32_BF16_MASK(c_float_0p2,0,2);
	}

	else
	{
		// Store the results.
		// c[0,0-15]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 0*16 ), c_float_0p0 );

		// c[0, 16-31]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 1*16 ), c_float_0p1 );

		// c[0,32-47]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 2*16 ), c_float_0p2 );
	}
}
#endif
#endif
