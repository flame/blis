/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#include "lpgemm_f32_kern_macros.h"

#ifndef LPGEMM_BF16_JIT

LPGEMM_ELTWISE_OPS_M_FRINGE_KERNEL(bfloat16,float,bf16of32_5x64)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_5x64_OPS_DISABLE,
						  &&POST_OPS_BIAS_5x64_OPS,
						  &&POST_OPS_RELU_5x64_OPS,
						  &&POST_OPS_RELU_SCALE_5x64_OPS,
						  &&POST_OPS_GELU_TANH_5x64_OPS,
						  &&POST_OPS_GELU_ERF_5x64_OPS,
						  &&POST_OPS_CLIP_5x64_OPS,
						  &&POST_OPS_DOWNSCALE_5x64_OPS,
						  &&POST_OPS_MATRIX_ADD_5x64_OPS,
						  &&POST_OPS_SWISH_5x64_OPS,
						  &&POST_OPS_MATRIX_MUL_5x64_OPS,
						  &&POST_OPS_TANH_5x64_OPS,
						  &&POST_OPS_SIGMOID_5x64_OPS
						};
	dim_t NR = 64;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();
	__m512 c_float_0p1 = _mm512_setzero_ps();
	__m512 c_float_0p2 = _mm512_setzero_ps();
	__m512 c_float_0p3 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();
	__m512 c_float_1p1 = _mm512_setzero_ps();
	__m512 c_float_1p2 = _mm512_setzero_ps();
	__m512 c_float_1p3 = _mm512_setzero_ps();

	__m512 c_float_2p0 = _mm512_setzero_ps();
	__m512 c_float_2p1 = _mm512_setzero_ps();
	__m512 c_float_2p2 = _mm512_setzero_ps();
	__m512 c_float_2p3 = _mm512_setzero_ps();

	__m512 c_float_3p0 = _mm512_setzero_ps();
	__m512 c_float_3p1 = _mm512_setzero_ps();
	__m512 c_float_3p2 = _mm512_setzero_ps();
	__m512 c_float_3p3 = _mm512_setzero_ps();

	__m512 c_float_4p0 = _mm512_setzero_ps();
	__m512 c_float_4p1 = _mm512_setzero_ps();
	__m512 c_float_4p2 = _mm512_setzero_ps();
	__m512 c_float_4p3 = _mm512_setzero_ps();

	__m512 selector1 = _mm512_setzero_ps();
	__m512 selector2 = _mm512_setzero_ps();
	__m512 selector3 = _mm512_setzero_ps();
	__m512 selector4 = _mm512_setzero_ps();

	__mmask16 k0 = 0xFFFF, k1 = 0xFFFF, k2 = 0xFFFF, k3 = 0xFFFF;

	dim_t NR_L = NR;
	for( dim_t jr = 0; jr < n0; jr += NR_L )
	{
		dim_t n_left = n0 - jr;
		NR_L = bli_min( NR_L, ( n_left >> 4 ) << 4 );
		if( NR_L == 0 ) { NR_L = 16; }

		dim_t nr0 = bli_min( n0 - jr, NR_L );
		if( nr0 == 64 )
		{
			// all masks are already set.
			// Nothing to modify.
		}
		else if( nr0 == 48 )
		{
			k3 = 0x0;
		}
		else if( nr0 == 32 )
		{
			k2 = k3 = 0x0;
		}
		else if( nr0 == 16 )
		{
			k1 = k2 = k3 = 0;
		}
		else if( nr0 < 16 )
		{
			k0 = (0xFFFF >> (16 - (nr0 & 0x0F)));
			k1 = k2 = k3 = 0;
		}

		// 1stx64 block.
		c_float_0p0 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k0, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 0 ) ) ) );
		c_float_0p1 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k1, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 16 ) ) ) );
		c_float_0p2 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k2, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 32 ) ) ) );
		c_float_0p3 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k3, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 48 ) ) ) );

		// 2ndx64 block.
		c_float_1p0 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k0, \
			a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 0 ) ) ) );
		c_float_1p1 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k1, \
			a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 16 ) ) ) );
		c_float_1p2 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k2, \
			a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 32 ) ) ) );
		c_float_1p3 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k3, \
			a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 48 ) ) ) );

		// 3rdx64 block.
		c_float_2p0 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k0, \
			a + ( rs_a * ( 2 ) ) + ( cs_a * ( jr + 0 ) ) ) );
		c_float_2p1 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k1, \
			a + ( rs_a * ( 2 ) ) + ( cs_a * ( jr + 16 ) ) ) );
		c_float_2p2 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k2, \
			a + ( rs_a * ( 2 ) ) + ( cs_a * ( jr + 32 ) ) ) );
		c_float_2p3 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k3, \
			a + ( rs_a * ( 2 ) ) + ( cs_a * ( jr + 48 ) ) ) );

		// 4thx64 block.
		c_float_3p0 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k0, \
			a + ( rs_a * ( 3 ) ) + ( cs_a * ( jr + 0 ) ) ) );
		c_float_3p1 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k1, \
			a + ( rs_a * ( 3 ) ) + ( cs_a * ( jr + 16 ) ) ) );
		c_float_3p2 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k2, \
			a + ( rs_a * ( 3 ) ) + ( cs_a * ( jr + 32 ) ) ) );
		c_float_3p3 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k3, \
			a + ( rs_a * ( 3 ) ) + ( cs_a * ( jr + 48 ) ) ) );

		// 5thx64 block.
		c_float_4p0 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k0, \
			a + ( rs_a * ( 4 ) ) + ( cs_a * ( jr + 0 ) ) ) );
		c_float_4p1 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k1, \
			a + ( rs_a * ( 4 ) ) + ( cs_a * ( jr + 16 ) ) ) );
		c_float_4p2 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k2, \
			a + ( rs_a * ( 4 ) ) + ( cs_a * ( jr + 32 ) ) ) );
		c_float_4p3 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k3, \
			a + ( rs_a * ( 4 ) ) + ( cs_a * ( jr + 48 ) ) ) );

		// Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_5x64_OPS:
		{
			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
				 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
				if ( post_ops_list_temp->stor_type == BF16 )
				{
					BF16_F32_BIAS_LOAD(selector1, k0, 0);
					BF16_F32_BIAS_LOAD(selector2, k1, 1);
					BF16_F32_BIAS_LOAD(selector3, k2, 2);
					BF16_F32_BIAS_LOAD(selector4, k3, 3);
				}
				else
				{
					selector1 =
					_mm512_maskz_loadu_ps( k0,
						( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					selector2 =
					_mm512_maskz_loadu_ps( k1,
						( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					selector3 =
					_mm512_maskz_loadu_ps( k2,
						( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					selector4 =
					_mm512_maskz_loadu_ps( k3,
						( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_j + ( 3 * 16 ) );
				}

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[0, 16-31]
				c_float_0p1 = _mm512_add_ps( selector2, c_float_0p1 );

				// c[0,32-47]
				c_float_0p2 = _mm512_add_ps( selector3, c_float_0p2 );

				// c[0,48-63]
				c_float_0p3 = _mm512_add_ps( selector4, c_float_0p3 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

				// c[1, 16-31]
				c_float_1p1 = _mm512_add_ps( selector2, c_float_1p1 );

				// c[1,32-47]
				c_float_1p2 = _mm512_add_ps( selector3, c_float_1p2 );

				// c[1,48-63]
				c_float_1p3 = _mm512_add_ps( selector4, c_float_1p3 );

				// c[2,0-15]
				c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );

				// c[2, 16-31]
				c_float_2p1 = _mm512_add_ps( selector2, c_float_2p1 );

				// c[2,32-47]
				c_float_2p2 = _mm512_add_ps( selector3, c_float_2p2 );

				// c[2,48-63]
				c_float_2p3 = _mm512_add_ps( selector4, c_float_2p3 );

				// c[3,0-15]
				c_float_3p0 = _mm512_add_ps( selector1, c_float_3p0 );

				// c[3, 16-31]
				c_float_3p1 = _mm512_add_ps( selector2, c_float_3p1 );

				// c[3,32-47]
				c_float_3p2 = _mm512_add_ps( selector3, c_float_3p2 );

				// c[3,48-63]
				c_float_3p3 = _mm512_add_ps( selector4, c_float_3p3 );

				// c[4,0-15]
				c_float_4p0 = _mm512_add_ps( selector1, c_float_4p0 );

				// c[4, 16-31]
				c_float_4p1 = _mm512_add_ps( selector2, c_float_4p1 );

				// c[4,32-47]
				c_float_4p2 = _mm512_add_ps( selector3, c_float_4p2 );

				// c[4,48-63]
				c_float_4p3 = _mm512_add_ps( selector4, c_float_4p3 );
			}
			else
			{
				// If original output was columns major, then by the time
				// kernel sees it, the matrix would be accessed as if it were
				// transposed. Due to this the bias array will be accessed by
				// the ic index, and each bias element corresponds to an
				// entire row of the transposed output array, instead of an
				// entire column.
				__m512 selector5;
				if ( post_ops_list_temp->stor_type == BF16 )
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
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_i + 0 ) );
					selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_i + 1 ) );
					selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_i + 2 ) );
					selector4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_i + 3 ) );
					selector5 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_i + 4 ) );
				}

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[0, 16-31]
				c_float_0p1 = _mm512_add_ps( selector1, c_float_0p1 );

				// c[0,32-47]
				c_float_0p2 = _mm512_add_ps( selector1, c_float_0p2 );

				// c[0,48-63]
				c_float_0p3 = _mm512_add_ps( selector1, c_float_0p3 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );

				// c[1, 16-31]
				c_float_1p1 = _mm512_add_ps( selector2, c_float_1p1 );

				// c[1,32-47]
				c_float_1p2 = _mm512_add_ps( selector2, c_float_1p2 );

				// c[1,48-63]
				c_float_1p3 = _mm512_add_ps( selector2, c_float_1p3 );

				// c[2,0-15]
				c_float_2p0 = _mm512_add_ps( selector3, c_float_2p0 );

				// c[2, 16-31]
				c_float_2p1 = _mm512_add_ps( selector3, c_float_2p1 );

				// c[2,32-47]
				c_float_2p2 = _mm512_add_ps( selector3, c_float_2p2 );

				// c[2,48-63]
				c_float_2p3 = _mm512_add_ps( selector3, c_float_2p3 );

				// c[3,0-15]
				c_float_3p0 = _mm512_add_ps( selector4, c_float_3p0 );

				// c[3, 16-31]
				c_float_3p1 = _mm512_add_ps( selector4, c_float_3p1 );

				// c[3,32-47]
				c_float_3p2 = _mm512_add_ps( selector4, c_float_3p2 );

				// c[3,48-63]
				c_float_3p3 = _mm512_add_ps( selector4, c_float_3p3 );

				// c[4,0-15]
				c_float_4p0 = _mm512_add_ps( selector5, c_float_4p0 );

				// c[4, 16-31]
				c_float_4p1 = _mm512_add_ps( selector5, c_float_4p1 );

				// c[4,32-47]
				c_float_4p2 = _mm512_add_ps( selector5, c_float_4p2 );

				// c[4,48-63]
				c_float_4p3 = _mm512_add_ps( selector5, c_float_4p3 );
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_5x64_OPS:
		{
			selector1 = _mm512_setzero_ps();

			// c[0,0-15]
			c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_max_ps( selector1, c_float_0p1 );

			// c[0,32-47]
			c_float_0p2 = _mm512_max_ps( selector1, c_float_0p2 );

			// c[0,48-63]
			c_float_0p3 = _mm512_max_ps( selector1, c_float_0p3 );

			// c[1,0-15]
			c_float_1p0 = _mm512_max_ps( selector1, c_float_1p0 );

			// c[1,16-31]
			c_float_1p1 = _mm512_max_ps( selector1, c_float_1p1 );

			// c[1,32-47]
			c_float_1p2 = _mm512_max_ps( selector1, c_float_1p2 );

			// c[1,48-63]
			c_float_1p3 = _mm512_max_ps( selector1, c_float_1p3 );

			// c[2,0-15]
			c_float_2p0 = _mm512_max_ps( selector1, c_float_2p0 );

			// c[2,16-31]
			c_float_2p1 = _mm512_max_ps( selector1, c_float_2p1 );

			// c[2,32-47]
			c_float_2p2 = _mm512_max_ps( selector1, c_float_2p2 );

			// c[2,48-63]
			c_float_2p3 = _mm512_max_ps( selector1, c_float_2p3 );

			// c[3,0-15]
			c_float_3p0 = _mm512_max_ps( selector1, c_float_3p0 );

			// c[3,16-31]
			c_float_3p1 = _mm512_max_ps( selector1, c_float_3p1 );

			// c[3,32-47]
			c_float_3p2 = _mm512_max_ps( selector1, c_float_3p2 );

			// c[3,48-63]
			c_float_3p3 = _mm512_max_ps( selector1, c_float_3p3 );

			// c[4,0-15]
			c_float_4p0 = _mm512_max_ps( selector1, c_float_4p0 );

			// c[4,16-31]
			c_float_4p1 = _mm512_max_ps( selector1, c_float_4p1 );

			// c[4,32-47]
			c_float_4p2 = _mm512_max_ps( selector1, c_float_4p2 );

			// c[4,48-63]
			c_float_4p3 = _mm512_max_ps( selector1, c_float_4p3 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_5x64_OPS:
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

			// c[0, 48-63]
			RELU_SCALE_OP_F32_AVX512(c_float_0p3)

			// c[1, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_1p0)

			// c[1, 16-31]
			RELU_SCALE_OP_F32_AVX512(c_float_1p1)

			// c[1, 32-47]
			RELU_SCALE_OP_F32_AVX512(c_float_1p2)

			// c[1, 48-63]
			RELU_SCALE_OP_F32_AVX512(c_float_1p3)

			// c[2, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_2p0)

			// c[2, 16-31]
			RELU_SCALE_OP_F32_AVX512(c_float_2p1)

			// c[2, 32-47]
			RELU_SCALE_OP_F32_AVX512(c_float_2p2)

			// c[2, 48-63]
			RELU_SCALE_OP_F32_AVX512(c_float_2p3)

			// c[3, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_3p0)

			// c[3, 16-31]
			RELU_SCALE_OP_F32_AVX512(c_float_3p1)

			// c[3, 32-47]
			RELU_SCALE_OP_F32_AVX512(c_float_3p2)

			// c[3, 48-63]
			RELU_SCALE_OP_F32_AVX512(c_float_3p3)

			// c[4, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_4p0)

			// c[4, 16-31]
			RELU_SCALE_OP_F32_AVX512(c_float_4p1)

			// c[4, 32-47]
			RELU_SCALE_OP_F32_AVX512(c_float_4p2)

			// c[4, 48-63]
			RELU_SCALE_OP_F32_AVX512(c_float_4p3)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_TANH_5x64_OPS:
		{
			__m512 dn, z, x, r2, r, x_tanh;
			__m512i q;

			// c[0, 0-15]
			GELU_TANH_F32_AVX512(c_float_0p0, r, r2, x, z, dn, x_tanh, q)

			// c[0, 16-31]
			GELU_TANH_F32_AVX512(c_float_0p1, r, r2, x, z, dn, x_tanh, q)

			// c[0, 32-47]
			GELU_TANH_F32_AVX512(c_float_0p2, r, r2, x, z, dn, x_tanh, q)

			// c[0, 48-63]
			GELU_TANH_F32_AVX512(c_float_0p3, r, r2, x, z, dn, x_tanh, q)

			// c[1, 0-15]
			GELU_TANH_F32_AVX512(c_float_1p0, r, r2, x, z, dn, x_tanh, q)

			// c[1, 16-31]
			GELU_TANH_F32_AVX512(c_float_1p1, r, r2, x, z, dn, x_tanh, q)

			// c[1, 32-47]
			GELU_TANH_F32_AVX512(c_float_1p2, r, r2, x, z, dn, x_tanh, q)

			// c[1, 48-63]
			GELU_TANH_F32_AVX512(c_float_1p3, r, r2, x, z, dn, x_tanh, q)

			// c[2, 0-15]
			GELU_TANH_F32_AVX512(c_float_2p0, r, r2, x, z, dn, x_tanh, q)

			// c[2, 16-31]
			GELU_TANH_F32_AVX512(c_float_2p1, r, r2, x, z, dn, x_tanh, q)

			// c[2, 32-47]
			GELU_TANH_F32_AVX512(c_float_2p2, r, r2, x, z, dn, x_tanh, q)

			// c[2, 48-63]
			GELU_TANH_F32_AVX512(c_float_2p3, r, r2, x, z, dn, x_tanh, q)

			// c[3, 0-15]
			GELU_TANH_F32_AVX512(c_float_3p0, r, r2, x, z, dn, x_tanh, q)

			// c[3, 16-31]
			GELU_TANH_F32_AVX512(c_float_3p1, r, r2, x, z, dn, x_tanh, q)

			// c[3, 32-47]
			GELU_TANH_F32_AVX512(c_float_3p2, r, r2, x, z, dn, x_tanh, q)

			// c[3, 48-63]
			GELU_TANH_F32_AVX512(c_float_3p3, r, r2, x, z, dn, x_tanh, q)

			// c[4, 0-15]
			GELU_TANH_F32_AVX512(c_float_4p0, r, r2, x, z, dn, x_tanh, q)

			// c[4, 16-31]
			GELU_TANH_F32_AVX512(c_float_4p1, r, r2, x, z, dn, x_tanh, q)

			// c[4, 32-47]
			GELU_TANH_F32_AVX512(c_float_4p2, r, r2, x, z, dn, x_tanh, q)

			// c[4, 48-63]
			GELU_TANH_F32_AVX512(c_float_4p3, r, r2, x, z, dn, x_tanh, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_ERF_5x64_OPS:
		{
			__m512 x, r, x_erf;

			// c[0, 0-15]
			GELU_ERF_F32_AVX512(c_float_0p0, r, x, x_erf)

			// c[0, 16-31]
			GELU_ERF_F32_AVX512(c_float_0p1, r, x, x_erf)

			// c[0, 32-47]
			GELU_ERF_F32_AVX512(c_float_0p2, r, x, x_erf)

			// c[0, 48-63]
			GELU_ERF_F32_AVX512(c_float_0p3, r, x, x_erf)

			// c[1, 0-15]
			GELU_ERF_F32_AVX512(c_float_1p0, r, x, x_erf)

			// c[1, 16-31]
			GELU_ERF_F32_AVX512(c_float_1p1, r, x, x_erf)

			// c[1, 32-47]
			GELU_ERF_F32_AVX512(c_float_1p2, r, x, x_erf)

			// c[1, 48-63]
			GELU_ERF_F32_AVX512(c_float_1p3, r, x, x_erf)

			// c[2, 0-15]
			GELU_ERF_F32_AVX512(c_float_2p0, r, x, x_erf)

			// c[2, 16-31]
			GELU_ERF_F32_AVX512(c_float_2p1, r, x, x_erf)

			// c[2, 32-47]
			GELU_ERF_F32_AVX512(c_float_2p2, r, x, x_erf)

			// c[2, 48-63]
			GELU_ERF_F32_AVX512(c_float_2p3, r, x, x_erf)

			// c[3, 0-15]
			GELU_ERF_F32_AVX512(c_float_3p0, r, x, x_erf)

			// c[3, 16-31]
			GELU_ERF_F32_AVX512(c_float_3p1, r, x, x_erf)

			// c[3, 32-47]
			GELU_ERF_F32_AVX512(c_float_3p2, r, x, x_erf)

			// c[3, 48-63]
			GELU_ERF_F32_AVX512(c_float_3p3, r, x, x_erf)

			// c[4, 0-15]
			GELU_ERF_F32_AVX512(c_float_4p0, r, x, x_erf)

			// c[4, 16-31]
			GELU_ERF_F32_AVX512(c_float_4p1, r, x, x_erf)

			// c[4, 32-47]
			GELU_ERF_F32_AVX512(c_float_4p2, r, x, x_erf)

			// c[4, 48-63]
			GELU_ERF_F32_AVX512(c_float_4p3, r, x, x_erf)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_CLIP_5x64_OPS:
		{
			__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
			__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

			// c[0, 0-15]
			CLIP_F32_AVX512(c_float_0p0, min, max)

			// c[0, 16-31]
			CLIP_F32_AVX512(c_float_0p1, min, max)

			// c[0, 32-47]
			CLIP_F32_AVX512(c_float_0p2, min, max)

			// c[0, 48-63]
			CLIP_F32_AVX512(c_float_0p3, min, max)

			// c[1, 0-15]
			CLIP_F32_AVX512(c_float_1p0, min, max)

			// c[1, 16-31]
			CLIP_F32_AVX512(c_float_1p1, min, max)

			// c[1, 32-47]
			CLIP_F32_AVX512(c_float_1p2, min, max)

			// c[1, 48-63]
			CLIP_F32_AVX512(c_float_1p3, min, max)

			// c[2, 0-15]
			CLIP_F32_AVX512(c_float_2p0, min, max)

			// c[2, 16-31]
			CLIP_F32_AVX512(c_float_2p1, min, max)

			// c[2, 32-47]
			CLIP_F32_AVX512(c_float_2p2, min, max)

			// c[2, 48-63]
			CLIP_F32_AVX512(c_float_2p3, min, max)

			// c[3, 0-15]
			CLIP_F32_AVX512(c_float_3p0, min, max)

			// c[3, 16-31]
			CLIP_F32_AVX512(c_float_3p1, min, max)

			// c[3, 32-47]
			CLIP_F32_AVX512(c_float_3p2, min, max)

			// c[3, 48-63]
			CLIP_F32_AVX512(c_float_3p3, min, max)

			// c[4, 0-15]
			CLIP_F32_AVX512(c_float_4p0, min, max)

			// c[4, 16-31]
			CLIP_F32_AVX512(c_float_4p1, min, max)

			// c[4, 32-47]
			CLIP_F32_AVX512(c_float_4p2, min, max)

			// c[4, 48-63]
			CLIP_F32_AVX512(c_float_4p3, min, max)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_DOWNSCALE_5x64_OPS:
		{
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
						_mm512_maskz_loadu_ps( k0,
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					selector2 =
						_mm512_maskz_loadu_ps( k1,
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					selector3 =
						_mm512_maskz_loadu_ps( k2,
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					selector4 =
						_mm512_maskz_loadu_ps( k3,
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 3 * 16 ) );
				}

				if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
				{
					zero_point0 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_loadu_epi16( k0,
						( ( bfloat16* )post_ops_list_temp->op_args1 ) +
						post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
					zero_point1 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_loadu_epi16( k1,
						( ( bfloat16* )post_ops_list_temp->op_args1 ) +
						post_ops_attr.post_op_c_j + ( 1 * 16 ) ) );
					zero_point2 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_loadu_epi16( k2,
						( ( bfloat16* )post_ops_list_temp->op_args1 ) +
						post_ops_attr.post_op_c_j + ( 2 * 16 ) ) );
					zero_point3 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_loadu_epi16( k3,
						( ( bfloat16* )post_ops_list_temp->op_args1 ) +
						post_ops_attr.post_op_c_j + ( 3 * 16 ) ) );
				}

				// c[0, 0-15]
				SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

				// c[0, 16-31]
				SCL_MULRND_F32(c_float_0p1,selector2,zero_point1);

				// c[0, 32-47]
				SCL_MULRND_F32(c_float_0p2,selector3,zero_point2);

				// c[0, 48-63]
				SCL_MULRND_F32(c_float_0p3,selector4,zero_point3);

				// c[1, 0-15]
				SCL_MULRND_F32(c_float_1p0,selector1,zero_point0);

				// c[1, 16-31]
				SCL_MULRND_F32(c_float_1p1,selector2,zero_point1);

				// c[1, 32-47]
				SCL_MULRND_F32(c_float_1p2,selector3,zero_point2);

				// c[1, 48-63]
				SCL_MULRND_F32(c_float_1p3,selector4,zero_point3);

				// c[2, 0-15]
				SCL_MULRND_F32(c_float_2p0,selector1,zero_point0);

				// c[2, 16-31]
				SCL_MULRND_F32(c_float_2p1,selector2,zero_point1);

				// c[2, 32-47]
				SCL_MULRND_F32(c_float_2p2,selector3,zero_point2);

				// c[2, 48-63]
				SCL_MULRND_F32(c_float_2p3,selector4,zero_point3);

				// c[3, 0-15]
				SCL_MULRND_F32(c_float_3p0,selector1,zero_point0);

				// c[3, 16-31]
				SCL_MULRND_F32(c_float_3p1,selector2,zero_point1);

				// c[3, 32-47]
				SCL_MULRND_F32(c_float_3p2,selector3,zero_point2);

				// c[3, 48-63]
				SCL_MULRND_F32(c_float_3p3,selector4,zero_point3);

				// c[4, 0-15]
				SCL_MULRND_F32(c_float_4p0,selector1,zero_point0);

				// c[4, 16-31]
				SCL_MULRND_F32(c_float_4p1,selector2,zero_point1);

				// c[4, 32-47]
				SCL_MULRND_F32(c_float_4p2,selector3,zero_point2);

				// c[4, 48-63]
				SCL_MULRND_F32(c_float_4p3,selector4,zero_point3);
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

				// c[0, 48-63]
				SCL_MULRND_F32(c_float_0p3,selector1,zero_point0);

				// c[1, 0-15]
				SCL_MULRND_F32(c_float_1p0,selector2,zero_point1);

				// c[1, 16-31]
				SCL_MULRND_F32(c_float_1p1,selector2,zero_point1);

				// c[1, 32-47]
				SCL_MULRND_F32(c_float_1p2,selector2,zero_point1);

				// c[1, 48-63]
				SCL_MULRND_F32(c_float_1p3,selector2,zero_point1);

				// c[2, 0-15]
				SCL_MULRND_F32(c_float_2p0,selector3,zero_point2);

				// c[2, 16-31]
				SCL_MULRND_F32(c_float_2p1,selector3,zero_point2);

				// c[2, 32-47]
				SCL_MULRND_F32(c_float_2p2,selector3,zero_point2);

				// c[2, 48-63]
				SCL_MULRND_F32(c_float_2p3,selector3,zero_point2);

				// c[3, 0-15]
				SCL_MULRND_F32(c_float_3p0,selector4,zero_point3);

				// c[3, 16-31]
				SCL_MULRND_F32(c_float_3p1,selector4,zero_point3);

				// c[3, 32-47]
				SCL_MULRND_F32(c_float_3p2,selector4,zero_point3);

				// c[3, 48-63]
				SCL_MULRND_F32(c_float_3p3,selector4,zero_point3);

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

				// c[4, 48-63]
				SCL_MULRND_F32(c_float_4p3,selector1,zero_point0);
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_ADD_5x64_OPS:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
					( ( post_ops_list_temp->stor_type == NONE ) &&
					  ( post_ops_attr.c_stor_type == BF16 ) );

			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();
			__m512 scl_fctr5 = _mm512_setzero_ps();

			// Even though different registers are used for scalar in column and
			// row major case, all those registers will contain the same value.
			if ( post_ops_list_temp->scale_factor_len == 1 )
			{
				scl_fctr1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr5 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
						_mm512_maskz_loadu_ps( k0,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					scl_fctr2 =
						_mm512_maskz_loadu_ps( k1,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					scl_fctr3 =
						_mm512_maskz_loadu_ps( k2,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					scl_fctr4 =
						_mm512_maskz_loadu_ps( k3,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 3 * 16 ) );
				}
				else
				{
					scl_fctr1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 0 ) );
					scl_fctr2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 1 ) );
					scl_fctr3 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 2 ) );
					scl_fctr4 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 3 ) );
					scl_fctr5 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 4 ) );
				}
			}

			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1);

					// c[2:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2);

					// c[3:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,3);

					// c[4:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,4);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,3);

					// c[4:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr5,scl_fctr5,scl_fctr5,scl_fctr5,4);
				}
			}
			else
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1);

					// c[2:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2);

					// c[3:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,3);

					// c[4:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,4);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,3);

					// c[4:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr5,scl_fctr5,scl_fctr5,scl_fctr5,4);
				}
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_MUL_5x64_OPS:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
					( ( post_ops_list_temp->stor_type == NONE ) &&
					  ( post_ops_attr.c_stor_type == BF16 ) );

			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();
			__m512 scl_fctr5 = _mm512_setzero_ps();

			// Even though different registers are used for scalar in column and
			// row major case, all those registers will contain the same value.
			if ( post_ops_list_temp->scale_factor_len == 1 )
			{
				scl_fctr1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr5 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
						_mm512_maskz_loadu_ps( k0,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					scl_fctr2 =
						_mm512_maskz_loadu_ps( k1,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					scl_fctr3 =
						_mm512_maskz_loadu_ps( k2,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					scl_fctr4 =
						_mm512_maskz_loadu_ps( k3,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 3 * 16 ) );
				}
				else
				{
					scl_fctr1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 0 ) );
					scl_fctr2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 1 ) );
					scl_fctr3 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 2 ) );
					scl_fctr4 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 3 ) );
					scl_fctr5 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 4 ) );
				}
			}

			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1);

					// c[2:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2);

					// c[3:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,3);

					// c[4:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,4);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,3);

					// c[4:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr5,scl_fctr5,scl_fctr5,scl_fctr5,4);
				}
			}
			else
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1);

					// c[2:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2);

					// c[3:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,3);

					// c[4:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,4);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,3);

					// c[4:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr5,scl_fctr5,scl_fctr5,scl_fctr5,4);
				}
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SWISH_5x64_OPS:
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

			// c[0, 48-63]
			SWISH_F32_AVX512_DEF(c_float_0p3, selector1, al_in, r, r2, z, dn, ex_out);

			// c[1, 0-15]
			SWISH_F32_AVX512_DEF(c_float_1p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[1, 16-31]
			SWISH_F32_AVX512_DEF(c_float_1p1, selector1, al_in, r, r2, z, dn, ex_out);

			// c[1, 32-47]
			SWISH_F32_AVX512_DEF(c_float_1p2, selector1, al_in, r, r2, z, dn, ex_out);

			// c[1, 48-63]
			SWISH_F32_AVX512_DEF(c_float_1p3, selector1, al_in, r, r2, z, dn, ex_out);

			// c[2, 0-15]
			SWISH_F32_AVX512_DEF(c_float_2p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[2, 16-31]
			SWISH_F32_AVX512_DEF(c_float_2p1, selector1, al_in, r, r2, z, dn, ex_out);

			// c[2, 32-47]
			SWISH_F32_AVX512_DEF(c_float_2p2, selector1, al_in, r, r2, z, dn, ex_out);

			// c[2, 48-63]
			SWISH_F32_AVX512_DEF(c_float_2p3, selector1, al_in, r, r2, z, dn, ex_out);

			// c[3, 0-15]
			SWISH_F32_AVX512_DEF(c_float_3p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[3, 16-31]
			SWISH_F32_AVX512_DEF(c_float_3p1, selector1, al_in, r, r2, z, dn, ex_out);

			// c[3, 32-47]
			SWISH_F32_AVX512_DEF(c_float_3p2, selector1, al_in, r, r2, z, dn, ex_out);

			// c[3, 48-63]
			SWISH_F32_AVX512_DEF(c_float_3p3, selector1, al_in, r, r2, z, dn, ex_out);

			// c[4, 0-15]
			SWISH_F32_AVX512_DEF(c_float_4p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[4, 16-31]
			SWISH_F32_AVX512_DEF(c_float_4p1, selector1, al_in, r, r2, z, dn, ex_out);

			// c[4, 32-47]
			SWISH_F32_AVX512_DEF(c_float_4p2, selector1, al_in, r, r2, z, dn, ex_out);

			// c[4, 48-63]
			SWISH_F32_AVX512_DEF(c_float_4p3, selector1, al_in, r, r2, z, dn, ex_out);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_TANH_5x64_OPS:
		{
			__m512 dn, z, x, r2, r;
			__m512i q;

			// c[0, 0-15]
			TANHF_AVX512(c_float_0p0, r, r2, x, z, dn,  q)

			// c[0, 16-31]
			TANHF_AVX512(c_float_0p1, r, r2, x, z, dn, q)

			// c[0, 32-47]
			TANHF_AVX512(c_float_0p2, r, r2, x, z, dn, q)

			// c[0, 48-63]
			TANHF_AVX512(c_float_0p3, r, r2, x, z, dn, q)

			// c[1, 0-15]
			TANHF_AVX512(c_float_1p0, r, r2, x, z, dn, q)

			// c[1, 16-31]
			TANHF_AVX512(c_float_1p1, r, r2, x, z, dn, q)

			// c[1, 32-47]
			TANHF_AVX512(c_float_1p2, r, r2, x, z, dn, q)

			// c[1, 48-63]
			TANHF_AVX512(c_float_1p3, r, r2, x, z, dn, q)

			// c[2, 0-15]
			TANHF_AVX512(c_float_2p0, r, r2, x, z, dn, q)

			// c[2, 16-31]
			TANHF_AVX512(c_float_2p1, r, r2, x, z, dn, q)

			// c[2, 32-47]
			TANHF_AVX512(c_float_2p2, r, r2, x, z, dn, q)

			// c[2, 48-63]
			TANHF_AVX512(c_float_2p3, r, r2, x, z, dn, q)

			// c[3, 0-15]
			TANHF_AVX512(c_float_3p0, r, r2, x, z, dn, q)

			// c[3, 16-31]
			TANHF_AVX512(c_float_3p1, r, r2, x, z, dn, q)

			// c[3, 32-47]
			TANHF_AVX512(c_float_3p2, r, r2, x, z, dn, q)

			// c[3, 48-63]
			TANHF_AVX512(c_float_3p3, r, r2, x, z, dn, q)

			// c[4, 0-15]
			TANHF_AVX512(c_float_4p0, r, r2, x, z, dn, q)

			// c[4, 16-31]
			TANHF_AVX512(c_float_4p1, r, r2, x, z, dn, q)

			// c[4, 32-47]
			TANHF_AVX512(c_float_4p2, r, r2, x, z, dn, q)

			// c[4, 48-63]
			TANHF_AVX512(c_float_4p3, r, r2, x, z, dn, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SIGMOID_5x64_OPS:
		{
			__m512 al_in, r, r2, z, dn;
			__m512i ex_out;

			// c[0, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_0p0, al_in, r, r2, z, dn, ex_out);

			// c[0, 16-31]
			SIGMOID_F32_AVX512_DEF(c_float_0p1, al_in, r, r2, z, dn, ex_out);

			// c[0, 32-47]
			SIGMOID_F32_AVX512_DEF(c_float_0p2, al_in, r, r2, z, dn, ex_out);

			// c[0, 48-63]
			SIGMOID_F32_AVX512_DEF(c_float_0p3, al_in, r, r2, z, dn, ex_out);

			// c[1, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_1p0, al_in, r, r2, z, dn, ex_out);

			// c[1, 16-31]
			SIGMOID_F32_AVX512_DEF(c_float_1p1, al_in, r, r2, z, dn, ex_out);

			// c[1, 32-47]
			SIGMOID_F32_AVX512_DEF(c_float_1p2, al_in, r, r2, z, dn, ex_out);

			// c[1, 48-63]
			SIGMOID_F32_AVX512_DEF(c_float_1p3, al_in, r, r2, z, dn, ex_out);

			// c[2, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_2p0, al_in, r, r2, z, dn, ex_out);

			// c[2, 16-31]
			SIGMOID_F32_AVX512_DEF(c_float_2p1, al_in, r, r2, z, dn, ex_out);

			// c[2, 32-47]
			SIGMOID_F32_AVX512_DEF(c_float_2p2, al_in, r, r2, z, dn, ex_out);

			// c[2, 48-63]
			SIGMOID_F32_AVX512_DEF(c_float_2p3, al_in, r, r2, z, dn, ex_out);

			// c[3, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_3p0, al_in, r, r2, z, dn, ex_out);

			// c[3, 16-31]
			SIGMOID_F32_AVX512_DEF(c_float_3p1, al_in, r, r2, z, dn, ex_out);

			// c[3, 32-47]
			SIGMOID_F32_AVX512_DEF(c_float_3p2, al_in, r, r2, z, dn, ex_out);

			// c[3, 48-63]
			SIGMOID_F32_AVX512_DEF(c_float_3p3, al_in, r, r2, z, dn, ex_out);

			// c[4, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_4p0, al_in, r, r2, z, dn, ex_out);

			// c[4, 16-31]
			SIGMOID_F32_AVX512_DEF(c_float_4p1, al_in, r, r2, z, dn, ex_out);

			// c[4, 32-47]
			SIGMOID_F32_AVX512_DEF(c_float_4p2, al_in, r, r2, z, dn, ex_out);

			// c[4, 48-63]
			SIGMOID_F32_AVX512_DEF(c_float_4p3, al_in, r, r2, z, dn, ex_out);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_5x64_OPS_DISABLE:
		;

		// Case where the output C matrix is bf16 (downscaled) and this is the
		// final write for a given block within C.
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			// Actually the b matrix is of type bfloat16. However
			// in order to reuse this kernel for f32, the output
			// matrix type in kernel function signature is set to
			// f32 irrespective of original output matrix type.
			bfloat16* b_q = ( bfloat16* )b;
			dim_t ir = 0;

			// Store the results in downscaled type (bf16 instead of float).
			// c[0, 0-15]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_0p0,k0,0,0);
			// c[0, 16-31]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_0p1,k1,0,16);
			// c[0, 32-47]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_0p2,k2,0,32);
			// c[0, 48-63]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_0p3,k3,0,48);

			// c[1, 0-15]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_1p0,k0,1,0);
			// c[1, 16-31]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_1p1,k1,1,16);
			// c[1, 32-47]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_1p2,k2,1,32);
			// c[1, 48-63]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_1p3,k3,1,48);

			// c[2, 0-15]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_2p0,k0,2,0);
			// c[2, 16-31]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_2p1,k1,2,16);
			// c[2, 32-47]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_2p2,k2,2,32);
			// c[2, 48-63]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_2p3,k3,2,48);

			// c[3, 0-15]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_3p0,k0,3,0);
			// c[3, 16-31]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_3p1,k1,3,16);
			// c[3, 32-47]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_3p2,k2,3,32);
			// c[3, 48-63]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_3p3,k3,3,48);

			// c[4, 0-15]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_4p0,k0,4,0);
			// c[4, 16-31]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_4p1,k1,4,16);
			// c[4, 32-47]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_4p2,k2,4,32);
			// c[4, 48-63]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_4p3,k3,4,48);
		}
		// Case where the output C matrix is float
		else
		{
			// Store the results.
			// c[0,0-15]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) + 
				( cs_b * ( jr + 0 ) ), k0, c_float_0p0 );
			// c[0,16-31]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) + 
				( cs_b * ( jr + 16 ) ), k1, c_float_0p1 );
			// c[0,32-47]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) + 
				( cs_b * ( jr + 32 ) ), k2, c_float_0p2 );
			// c[0,48-63]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) + 
				( cs_b * ( jr + 48 ) ), k3, c_float_0p3 );

			// c[1,0-15]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) + 
				( cs_b * ( jr + 0 ) ), k0, c_float_1p0 );
			// c[1,16-31]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) + 
				( cs_b * ( jr + 16 ) ), k1, c_float_1p1 );
			// c[1,32-47]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) + 
				( cs_b * ( jr + 32 ) ), k2, c_float_1p2 );
			// c[1,48-63]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) + 
				( cs_b * ( jr + 48 ) ), k3, c_float_1p3 );

			// c[2,0-15]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 2 ) ) + 
				( cs_b * ( jr + 0 ) ), k0, c_float_2p0 );
			// c[2,16-31]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 2 ) ) + 
				( cs_b * ( jr + 16 ) ), k1, c_float_2p1 );
			// c[2,32-47]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 2 ) ) + 
				( cs_b * ( jr + 32 ) ), k2, c_float_2p2 );
			// c[2,48-63]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 2 ) ) + 
				( cs_b * ( jr + 48 ) ), k3, c_float_2p3 );

			// c[3,0-15]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 3 ) ) + 
				( cs_b * ( jr + 0 ) ), k0, c_float_3p0 );
			// c[3,16-31]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 3 ) ) + 
				( cs_b * ( jr + 16 ) ), k1, c_float_3p1 );
			// c[3,32-47]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 3 ) ) + 
				( cs_b * ( jr + 32 ) ), k2, c_float_3p2 );
			// c[3,48-63]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 3 ) ) + 
				( cs_b * ( jr + 48 ) ), k3, c_float_3p3 );

			// c[4,0-15]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 4 ) ) + 
				( cs_b * ( jr + 0 ) ), k0, c_float_4p0 );
			// c[4,16-31]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 4 ) ) + 
				( cs_b * ( jr + 16 ) ), k1, c_float_4p1 );
			// c[4,32-47]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 4 ) ) + 
				( cs_b * ( jr + 32 ) ), k2, c_float_4p2 );
			// c[4,48-63]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 4 ) ) + 
				( cs_b * ( jr + 48 ) ), k3, c_float_4p3 );
		}

		post_ops_attr.post_op_c_j += NR_L;
	}
}

LPGEMM_ELTWISE_OPS_M_FRINGE_KERNEL(bfloat16,float,bf16of32_4x64)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_4x64_OPS_DISABLE,
						  &&POST_OPS_BIAS_4x64_OPS,
						  &&POST_OPS_RELU_4x64_OPS,
						  &&POST_OPS_RELU_SCALE_4x64_OPS,
						  &&POST_OPS_GELU_TANH_4x64_OPS,
						  &&POST_OPS_GELU_ERF_4x64_OPS,
						  &&POST_OPS_CLIP_4x64_OPS,
						  &&POST_OPS_DOWNSCALE_4x64_OPS,
						  &&POST_OPS_MATRIX_ADD_4x64_OPS,
						  &&POST_OPS_SWISH_4x64_OPS,
						  &&POST_OPS_MATRIX_MUL_4x64_OPS,
						  &&POST_OPS_TANH_4x64_OPS,
						  &&POST_OPS_SIGMOID_4x64_OPS
						};
	dim_t NR = 64;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();
	__m512 c_float_0p1 = _mm512_setzero_ps();
	__m512 c_float_0p2 = _mm512_setzero_ps();
	__m512 c_float_0p3 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();
	__m512 c_float_1p1 = _mm512_setzero_ps();
	__m512 c_float_1p2 = _mm512_setzero_ps();
	__m512 c_float_1p3 = _mm512_setzero_ps();

	__m512 c_float_2p0 = _mm512_setzero_ps();
	__m512 c_float_2p1 = _mm512_setzero_ps();
	__m512 c_float_2p2 = _mm512_setzero_ps();
	__m512 c_float_2p3 = _mm512_setzero_ps();

	__m512 c_float_3p0 = _mm512_setzero_ps();
	__m512 c_float_3p1 = _mm512_setzero_ps();
	__m512 c_float_3p2 = _mm512_setzero_ps();
	__m512 c_float_3p3 = _mm512_setzero_ps();

	__m512 selector1 = _mm512_setzero_ps();
	__m512 selector2 = _mm512_setzero_ps();
	__m512 selector3 = _mm512_setzero_ps();
	__m512 selector4 = _mm512_setzero_ps();

	__mmask16 k0 = 0xFFFF, k1 = 0xFFFF, k2 = 0xFFFF, k3 = 0xFFFF;

	dim_t NR_L = NR;
	for( dim_t jr = 0; jr < n0; jr += NR_L )
	{
		dim_t n_left = n0 - jr;
		NR_L = bli_min( NR_L, ( n_left >> 4 ) << 4 );
		if( NR_L == 0 ) { NR_L = 16; }

		dim_t nr0 = bli_min( n0 - jr, NR_L );
		if( nr0 == 64 )
		{
			// all masks are already set.
			// Nothing to modify.
		}
		else if( nr0 == 48 )
		{
			k3 = 0x0;
		}
		else if( nr0 == 32 )
		{
			k2 = k3 = 0x0;
		}
		else if( nr0 == 16 )
		{
			k1 = k2 = k3 = 0;
		}
		else if( nr0 < 16 )
		{
			k0 = (0xFFFF >> (16 - (nr0 & 0x0F)));
			k1 = k2 = k3 = 0;
		}

		// 1stx64 block.
		c_float_0p0 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k0, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 0 ) ) ) );
		c_float_0p1 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k1, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 16 ) ) ) );
		c_float_0p2 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k2, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 32 ) ) ) );
		c_float_0p3 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k3, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 48 ) ) ) );

		// 2ndx64 block.
		c_float_1p0 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k0, \
			a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 0 ) ) ) );
		c_float_1p1 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k1, \
			a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 16 ) ) ) );
		c_float_1p2 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k2, \
			a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 32 ) ) ) );
		c_float_1p3 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k3, \
			a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 48 ) ) ) );

		// 3rdx64 block.
		c_float_2p0 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k0, \
			a + ( rs_a * ( 2 ) ) + ( cs_a * ( jr + 0 ) ) ) );
		c_float_2p1 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k1, \
			a + ( rs_a * ( 2 ) ) + ( cs_a * ( jr + 16 ) ) ) );
		c_float_2p2 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k2, \
			a + ( rs_a * ( 2 ) ) + ( cs_a * ( jr + 32 ) ) ) );
		c_float_2p3 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k3, \
			a + ( rs_a * ( 2 ) ) + ( cs_a * ( jr + 48 ) ) ) );

		// 4thx64 block.
		c_float_3p0 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k0, \
			a + ( rs_a * ( 3 ) ) + ( cs_a * ( jr + 0 ) ) ) );
		c_float_3p1 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k1, \
			a + ( rs_a * ( 3 ) ) + ( cs_a * ( jr + 16 ) ) ) );
		c_float_3p2 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k2, \
			a + ( rs_a * ( 3 ) ) + ( cs_a * ( jr + 32 ) ) ) );
		c_float_3p3 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k3, \
			a + ( rs_a * ( 3 ) ) + ( cs_a * ( jr + 48 ) ) ) );

		// Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_4x64_OPS:
		{
			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
				 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
				if ( post_ops_list_temp->stor_type == BF16 )
				{
					BF16_F32_BIAS_LOAD(selector1, k0, 0);
					BF16_F32_BIAS_LOAD(selector2, k1, 1);
					BF16_F32_BIAS_LOAD(selector3, k2, 2);
					BF16_F32_BIAS_LOAD(selector4, k3, 3);
				}
				else
				{
					selector1 =
					_mm512_maskz_loadu_ps( k0,
						( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					selector2 =
					_mm512_maskz_loadu_ps( k1,
						( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					selector3 =
					_mm512_maskz_loadu_ps( k2,
						( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					selector4 =
					_mm512_maskz_loadu_ps( k3,
						( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_j + ( 3 * 16 ) );
				}

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[0, 16-31]
				c_float_0p1 = _mm512_add_ps( selector2, c_float_0p1 );

				// c[0,32-47]
				c_float_0p2 = _mm512_add_ps( selector3, c_float_0p2 );

				// c[0,48-63]
				c_float_0p3 = _mm512_add_ps( selector4, c_float_0p3 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

				// c[1, 16-31]
				c_float_1p1 = _mm512_add_ps( selector2, c_float_1p1 );

				// c[1,32-47]
				c_float_1p2 = _mm512_add_ps( selector3, c_float_1p2 );

				// c[1,48-63]
				c_float_1p3 = _mm512_add_ps( selector4, c_float_1p3 );

				// c[2,0-15]
				c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );

				// c[2, 16-31]
				c_float_2p1 = _mm512_add_ps( selector2, c_float_2p1 );

				// c[2,32-47]
				c_float_2p2 = _mm512_add_ps( selector3, c_float_2p2 );

				// c[2,48-63]
				c_float_2p3 = _mm512_add_ps( selector4, c_float_2p3 );

				// c[3,0-15]
				c_float_3p0 = _mm512_add_ps( selector1, c_float_3p0 );

				// c[3, 16-31]
				c_float_3p1 = _mm512_add_ps( selector2, c_float_3p1 );

				// c[3,32-47]
				c_float_3p2 = _mm512_add_ps( selector3, c_float_3p2 );

				// c[3,48-63]
				c_float_3p3 = _mm512_add_ps( selector4, c_float_3p3 );
			}
			else
			{
				// If original output was columns major, then by the time
				// kernel sees it, the matrix would be accessed as if it were
				// transposed. Due to this the bias array will be accessed by
				// the ic index, and each bias element corresponds to an
				// entire row of the transposed output array, instead of an
				// entire column.
				if ( post_ops_list_temp->stor_type == BF16 )
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
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_i + 0 ) );
					selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_i + 1 ) );
					selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_i + 2 ) );
					selector4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_i + 3 ) );
				}

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[0, 16-31]
				c_float_0p1 = _mm512_add_ps( selector1, c_float_0p1 );

				// c[0,32-47]
				c_float_0p2 = _mm512_add_ps( selector1, c_float_0p2 );

				// c[0,48-63]
				c_float_0p3 = _mm512_add_ps( selector1, c_float_0p3 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );

				// c[1, 16-31]
				c_float_1p1 = _mm512_add_ps( selector2, c_float_1p1 );

				// c[1,32-47]
				c_float_1p2 = _mm512_add_ps( selector2, c_float_1p2 );

				// c[1,48-63]
				c_float_1p3 = _mm512_add_ps( selector2, c_float_1p3 );

				// c[2,0-15]
				c_float_2p0 = _mm512_add_ps( selector3, c_float_2p0 );

				// c[2, 16-31]
				c_float_2p1 = _mm512_add_ps( selector3, c_float_2p1 );

				// c[2,32-47]
				c_float_2p2 = _mm512_add_ps( selector3, c_float_2p2 );

				// c[2,48-63]
				c_float_2p3 = _mm512_add_ps( selector3, c_float_2p3 );

				// c[3,0-15]
				c_float_3p0 = _mm512_add_ps( selector4, c_float_3p0 );

				// c[3, 16-31]
				c_float_3p1 = _mm512_add_ps( selector4, c_float_3p1 );

				// c[3,32-47]
				c_float_3p2 = _mm512_add_ps( selector4, c_float_3p2 );

				// c[3,48-63]
				c_float_3p3 = _mm512_add_ps( selector4, c_float_3p3 );
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_4x64_OPS:
		{
			selector1 = _mm512_setzero_ps();

			// c[0,0-15]
			c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_max_ps( selector1, c_float_0p1 );

			// c[0,32-47]
			c_float_0p2 = _mm512_max_ps( selector1, c_float_0p2 );

			// c[0,48-63]
			c_float_0p3 = _mm512_max_ps( selector1, c_float_0p3 );

			// c[1,0-15]
			c_float_1p0 = _mm512_max_ps( selector1, c_float_1p0 );

			// c[1,16-31]
			c_float_1p1 = _mm512_max_ps( selector1, c_float_1p1 );

			// c[1,32-47]
			c_float_1p2 = _mm512_max_ps( selector1, c_float_1p2 );

			// c[1,48-63]
			c_float_1p3 = _mm512_max_ps( selector1, c_float_1p3 );

			// c[2,0-15]
			c_float_2p0 = _mm512_max_ps( selector1, c_float_2p0 );

			// c[2,16-31]
			c_float_2p1 = _mm512_max_ps( selector1, c_float_2p1 );

			// c[2,32-47]
			c_float_2p2 = _mm512_max_ps( selector1, c_float_2p2 );

			// c[2,48-63]
			c_float_2p3 = _mm512_max_ps( selector1, c_float_2p3 );

			// c[3,0-15]
			c_float_3p0 = _mm512_max_ps( selector1, c_float_3p0 );

			// c[3,16-31]
			c_float_3p1 = _mm512_max_ps( selector1, c_float_3p1 );

			// c[3,32-47]
			c_float_3p2 = _mm512_max_ps( selector1, c_float_3p2 );

			// c[3,48-63]
			c_float_3p3 = _mm512_max_ps( selector1, c_float_3p3 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_4x64_OPS:
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

			// c[0, 48-63]
			RELU_SCALE_OP_F32_AVX512(c_float_0p3)

			// c[1, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_1p0)

			// c[1, 16-31]
			RELU_SCALE_OP_F32_AVX512(c_float_1p1)

			// c[1, 32-47]
			RELU_SCALE_OP_F32_AVX512(c_float_1p2)

			// c[1, 48-63]
			RELU_SCALE_OP_F32_AVX512(c_float_1p3)

			// c[2, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_2p0)

			// c[2, 16-31]
			RELU_SCALE_OP_F32_AVX512(c_float_2p1)

			// c[2, 32-47]
			RELU_SCALE_OP_F32_AVX512(c_float_2p2)

			// c[2, 48-63]
			RELU_SCALE_OP_F32_AVX512(c_float_2p3)

			// c[3, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_3p0)

			// c[3, 16-31]
			RELU_SCALE_OP_F32_AVX512(c_float_3p1)

			// c[3, 32-47]
			RELU_SCALE_OP_F32_AVX512(c_float_3p2)

			// c[3, 48-63]
			RELU_SCALE_OP_F32_AVX512(c_float_3p3)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_TANH_4x64_OPS:
		{
			__m512 dn, z, x, r2, r, x_tanh;
			__m512i q;

			// c[0, 0-15]
			GELU_TANH_F32_AVX512(c_float_0p0, r, r2, x, z, dn, x_tanh, q)

			// c[0, 16-31]
			GELU_TANH_F32_AVX512(c_float_0p1, r, r2, x, z, dn, x_tanh, q)

			// c[0, 32-47]
			GELU_TANH_F32_AVX512(c_float_0p2, r, r2, x, z, dn, x_tanh, q)

			// c[0, 48-63]
			GELU_TANH_F32_AVX512(c_float_0p3, r, r2, x, z, dn, x_tanh, q)

			// c[1, 0-15]
			GELU_TANH_F32_AVX512(c_float_1p0, r, r2, x, z, dn, x_tanh, q)

			// c[1, 16-31]
			GELU_TANH_F32_AVX512(c_float_1p1, r, r2, x, z, dn, x_tanh, q)

			// c[1, 32-47]
			GELU_TANH_F32_AVX512(c_float_1p2, r, r2, x, z, dn, x_tanh, q)

			// c[1, 48-63]
			GELU_TANH_F32_AVX512(c_float_1p3, r, r2, x, z, dn, x_tanh, q)

			// c[2, 0-15]
			GELU_TANH_F32_AVX512(c_float_2p0, r, r2, x, z, dn, x_tanh, q)

			// c[2, 16-31]
			GELU_TANH_F32_AVX512(c_float_2p1, r, r2, x, z, dn, x_tanh, q)

			// c[2, 32-47]
			GELU_TANH_F32_AVX512(c_float_2p2, r, r2, x, z, dn, x_tanh, q)

			// c[2, 48-63]
			GELU_TANH_F32_AVX512(c_float_2p3, r, r2, x, z, dn, x_tanh, q)

			// c[3, 0-15]
			GELU_TANH_F32_AVX512(c_float_3p0, r, r2, x, z, dn, x_tanh, q)

			// c[3, 16-31]
			GELU_TANH_F32_AVX512(c_float_3p1, r, r2, x, z, dn, x_tanh, q)

			// c[3, 32-47]
			GELU_TANH_F32_AVX512(c_float_3p2, r, r2, x, z, dn, x_tanh, q)

			// c[3, 48-63]
			GELU_TANH_F32_AVX512(c_float_3p3, r, r2, x, z, dn, x_tanh, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_ERF_4x64_OPS:
		{
			__m512 x, r, x_erf;

			// c[0, 0-15]
			GELU_ERF_F32_AVX512(c_float_0p0, r, x, x_erf)

			// c[0, 16-31]
			GELU_ERF_F32_AVX512(c_float_0p1, r, x, x_erf)

			// c[0, 32-47]
			GELU_ERF_F32_AVX512(c_float_0p2, r, x, x_erf)

			// c[0, 48-63]
			GELU_ERF_F32_AVX512(c_float_0p3, r, x, x_erf)

			// c[1, 0-15]
			GELU_ERF_F32_AVX512(c_float_1p0, r, x, x_erf)

			// c[1, 16-31]
			GELU_ERF_F32_AVX512(c_float_1p1, r, x, x_erf)

			// c[1, 32-47]
			GELU_ERF_F32_AVX512(c_float_1p2, r, x, x_erf)

			// c[1, 48-63]
			GELU_ERF_F32_AVX512(c_float_1p3, r, x, x_erf)

			// c[2, 0-15]
			GELU_ERF_F32_AVX512(c_float_2p0, r, x, x_erf)

			// c[2, 16-31]
			GELU_ERF_F32_AVX512(c_float_2p1, r, x, x_erf)

			// c[2, 32-47]
			GELU_ERF_F32_AVX512(c_float_2p2, r, x, x_erf)

			// c[2, 48-63]
			GELU_ERF_F32_AVX512(c_float_2p3, r, x, x_erf)

			// c[3, 0-15]
			GELU_ERF_F32_AVX512(c_float_3p0, r, x, x_erf)

			// c[3, 16-31]
			GELU_ERF_F32_AVX512(c_float_3p1, r, x, x_erf)

			// c[3, 32-47]
			GELU_ERF_F32_AVX512(c_float_3p2, r, x, x_erf)

			// c[3, 48-63]
			GELU_ERF_F32_AVX512(c_float_3p3, r, x, x_erf)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_CLIP_4x64_OPS:
		{
			__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
			__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

			// c[0, 0-15]
			CLIP_F32_AVX512(c_float_0p0, min, max)

			// c[0, 16-31]
			CLIP_F32_AVX512(c_float_0p1, min, max)

			// c[0, 32-47]
			CLIP_F32_AVX512(c_float_0p2, min, max)

			// c[0, 48-63]
			CLIP_F32_AVX512(c_float_0p3, min, max)

			// c[1, 0-15]
			CLIP_F32_AVX512(c_float_1p0, min, max)

			// c[1, 16-31]
			CLIP_F32_AVX512(c_float_1p1, min, max)

			// c[1, 32-47]
			CLIP_F32_AVX512(c_float_1p2, min, max)

			// c[1, 48-63]
			CLIP_F32_AVX512(c_float_1p3, min, max)

			// c[2, 0-15]
			CLIP_F32_AVX512(c_float_2p0, min, max)

			// c[2, 16-31]
			CLIP_F32_AVX512(c_float_2p1, min, max)

			// c[2, 32-47]
			CLIP_F32_AVX512(c_float_2p2, min, max)

			// c[2, 48-63]
			CLIP_F32_AVX512(c_float_2p3, min, max)

			// c[3, 0-15]
			CLIP_F32_AVX512(c_float_3p0, min, max)

			// c[3, 16-31]
			CLIP_F32_AVX512(c_float_3p1, min, max)

			// c[3, 32-47]
			CLIP_F32_AVX512(c_float_3p2, min, max)

			// c[3, 48-63]
			CLIP_F32_AVX512(c_float_3p3, min, max)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_DOWNSCALE_4x64_OPS:
		{
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
						_mm512_maskz_loadu_ps( k0,
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					selector2 =
						_mm512_maskz_loadu_ps( k1,
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					selector3 =
						_mm512_maskz_loadu_ps( k2,
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					selector4 =
						_mm512_maskz_loadu_ps( k3,
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 3 * 16 ) );
				}

				if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
				{
					zero_point0 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_loadu_epi16( k0,
						( ( bfloat16* )post_ops_list_temp->op_args1 ) +
						post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
					zero_point1 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_loadu_epi16( k1,
						( ( bfloat16* )post_ops_list_temp->op_args1 ) +
						post_ops_attr.post_op_c_j + ( 1 * 16 ) ) );
					zero_point2 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_loadu_epi16( k2,
						( ( bfloat16* )post_ops_list_temp->op_args1 ) +
						post_ops_attr.post_op_c_j + ( 2 * 16 ) ) );
					zero_point3 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_loadu_epi16( k3,
						( ( bfloat16* )post_ops_list_temp->op_args1 ) +
						post_ops_attr.post_op_c_j + ( 3 * 16 ) ) );
				}

				// c[0, 0-15]
				SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

				// c[0, 16-31]
				SCL_MULRND_F32(c_float_0p1,selector2,zero_point1);

				// c[0, 32-47]
				SCL_MULRND_F32(c_float_0p2,selector3,zero_point2);

				// c[0, 48-63]
				SCL_MULRND_F32(c_float_0p3,selector4,zero_point3);

				// c[1, 0-15]
				SCL_MULRND_F32(c_float_1p0,selector1,zero_point0);

				// c[1, 16-31]
				SCL_MULRND_F32(c_float_1p1,selector2,zero_point1);

				// c[1, 32-47]
				SCL_MULRND_F32(c_float_1p2,selector3,zero_point2);

				// c[1, 48-63]
				SCL_MULRND_F32(c_float_1p3,selector4,zero_point3);

				// c[2, 0-15]
				SCL_MULRND_F32(c_float_2p0,selector1,zero_point0);

				// c[2, 16-31]
				SCL_MULRND_F32(c_float_2p1,selector2,zero_point1);

				// c[2, 32-47]
				SCL_MULRND_F32(c_float_2p2,selector3,zero_point2);

				// c[2, 48-63]
				SCL_MULRND_F32(c_float_2p3,selector4,zero_point3);

				// c[3, 0-15]
				SCL_MULRND_F32(c_float_3p0,selector1,zero_point0);

				// c[3, 16-31]
				SCL_MULRND_F32(c_float_3p1,selector2,zero_point1);

				// c[3, 32-47]
				SCL_MULRND_F32(c_float_3p2,selector3,zero_point2);

				// c[3, 48-63]
				SCL_MULRND_F32(c_float_3p3,selector4,zero_point3);
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

				// c[0, 48-63]
				SCL_MULRND_F32(c_float_0p3,selector1,zero_point0);

				// c[1, 0-15]
				SCL_MULRND_F32(c_float_1p0,selector2,zero_point1);

				// c[1, 16-31]
				SCL_MULRND_F32(c_float_1p1,selector2,zero_point1);

				// c[1, 32-47]
				SCL_MULRND_F32(c_float_1p2,selector2,zero_point1);

				// c[1, 48-63]
				SCL_MULRND_F32(c_float_1p3,selector2,zero_point1);

				// c[2, 0-15]
				SCL_MULRND_F32(c_float_2p0,selector3,zero_point2);

				// c[2, 16-31]
				SCL_MULRND_F32(c_float_2p1,selector3,zero_point2);

				// c[2, 32-47]
				SCL_MULRND_F32(c_float_2p2,selector3,zero_point2);

				// c[2, 48-63]
				SCL_MULRND_F32(c_float_2p3,selector3,zero_point2);

				// c[3, 0-15]
				SCL_MULRND_F32(c_float_3p0,selector4,zero_point3);

				// c[3, 16-31]
				SCL_MULRND_F32(c_float_3p1,selector4,zero_point3);

				// c[3, 32-47]
				SCL_MULRND_F32(c_float_3p2,selector4,zero_point3);

				// c[3, 48-63]
				SCL_MULRND_F32(c_float_3p3,selector4,zero_point3);
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_ADD_4x64_OPS:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
					( ( post_ops_list_temp->stor_type == NONE ) &&
					  ( post_ops_attr.c_stor_type == BF16 ) );

			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();

			// Even though different registers are used for scalar in column and
			// row major case, all those registers will contain the same value.
			if ( post_ops_list_temp->scale_factor_len == 1 )
			{
				scl_fctr1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
						_mm512_maskz_loadu_ps( k0,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					scl_fctr2 =
						_mm512_maskz_loadu_ps( k1,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					scl_fctr3 =
						_mm512_maskz_loadu_ps( k2,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					scl_fctr4 =
						_mm512_maskz_loadu_ps( k3,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 3 * 16 ) );
				}
				else
				{
					scl_fctr1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 0 ) );
					scl_fctr2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 1 ) );
					scl_fctr3 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 2 ) );
					scl_fctr4 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 3 ) );
				}
			}

			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1);

					// c[2:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2);

					// c[3:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,3);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,3);
				}
			}
			else
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1);

					// c[2:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2);

					// c[3:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,3);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,3);
				}
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_MUL_4x64_OPS:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
					( ( post_ops_list_temp->stor_type == NONE ) &&
					  ( post_ops_attr.c_stor_type == BF16 ) );

			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();

			// Even though different registers are used for scalar in column and
			// row major case, all those registers will contain the same value.
			if ( post_ops_list_temp->scale_factor_len == 1 )
			{
				scl_fctr1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
						_mm512_maskz_loadu_ps( k0,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					scl_fctr2 =
						_mm512_maskz_loadu_ps( k1,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					scl_fctr3 =
						_mm512_maskz_loadu_ps( k2,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					scl_fctr4 =
						_mm512_maskz_loadu_ps( k3,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 3 * 16 ) );
				}
				else
				{
					scl_fctr1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 0 ) );
					scl_fctr2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 1 ) );
					scl_fctr3 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 2 ) );
					scl_fctr4 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 3 ) );
				}
			}

			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1);

					// c[2:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2);

					// c[3:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,3);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,3);
				}
			}
			else
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1);

					// c[2:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2);

					// c[3:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,3);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr4,scl_fctr4,scl_fctr4,scl_fctr4,3);
				}
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SWISH_4x64_OPS:
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

			// c[0, 48-63]
			SWISH_F32_AVX512_DEF(c_float_0p3, selector1, al_in, r, r2, z, dn, ex_out);

			// c[1, 0-15]
			SWISH_F32_AVX512_DEF(c_float_1p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[1, 16-31]
			SWISH_F32_AVX512_DEF(c_float_1p1, selector1, al_in, r, r2, z, dn, ex_out);

			// c[1, 32-47]
			SWISH_F32_AVX512_DEF(c_float_1p2, selector1, al_in, r, r2, z, dn, ex_out);

			// c[1, 48-63]
			SWISH_F32_AVX512_DEF(c_float_1p3, selector1, al_in, r, r2, z, dn, ex_out);

			// c[2, 0-15]
			SWISH_F32_AVX512_DEF(c_float_2p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[2, 16-31]
			SWISH_F32_AVX512_DEF(c_float_2p1, selector1, al_in, r, r2, z, dn, ex_out);

			// c[2, 32-47]
			SWISH_F32_AVX512_DEF(c_float_2p2, selector1, al_in, r, r2, z, dn, ex_out);

			// c[2, 48-63]
			SWISH_F32_AVX512_DEF(c_float_2p3, selector1, al_in, r, r2, z, dn, ex_out);

			// c[3, 0-15]
			SWISH_F32_AVX512_DEF(c_float_3p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[3, 16-31]
			SWISH_F32_AVX512_DEF(c_float_3p1, selector1, al_in, r, r2, z, dn, ex_out);

			// c[3, 32-47]
			SWISH_F32_AVX512_DEF(c_float_3p2, selector1, al_in, r, r2, z, dn, ex_out);

			// c[3, 48-63]
			SWISH_F32_AVX512_DEF(c_float_3p3, selector1, al_in, r, r2, z, dn, ex_out);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_TANH_4x64_OPS:
		{
			__m512 dn, z, x, r2, r;
			__m512i q;

			// c[0, 0-15]
			TANHF_AVX512(c_float_0p0, r, r2, x, z, dn,  q)

			// c[0, 16-31]
			TANHF_AVX512(c_float_0p1, r, r2, x, z, dn, q)

			// c[0, 32-47]
			TANHF_AVX512(c_float_0p2, r, r2, x, z, dn, q)

			// c[0, 48-63]
			TANHF_AVX512(c_float_0p3, r, r2, x, z, dn, q)

			// c[1, 0-15]
			TANHF_AVX512(c_float_1p0, r, r2, x, z, dn, q)

			// c[1, 16-31]
			TANHF_AVX512(c_float_1p1, r, r2, x, z, dn, q)

			// c[1, 32-47]
			TANHF_AVX512(c_float_1p2, r, r2, x, z, dn, q)

			// c[1, 48-63]
			TANHF_AVX512(c_float_1p3, r, r2, x, z, dn, q)

			// c[2, 0-15]
			TANHF_AVX512(c_float_2p0, r, r2, x, z, dn, q)

			// c[2, 16-31]
			TANHF_AVX512(c_float_2p1, r, r2, x, z, dn, q)

			// c[2, 32-47]
			TANHF_AVX512(c_float_2p2, r, r2, x, z, dn, q)

			// c[2, 48-63]
			TANHF_AVX512(c_float_2p3, r, r2, x, z, dn, q)

			// c[3, 0-15]
			TANHF_AVX512(c_float_3p0, r, r2, x, z, dn, q)

			// c[3, 16-31]
			TANHF_AVX512(c_float_3p1, r, r2, x, z, dn, q)

			// c[3, 32-47]
			TANHF_AVX512(c_float_3p2, r, r2, x, z, dn, q)

			// c[3, 48-63]
			TANHF_AVX512(c_float_3p3, r, r2, x, z, dn, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SIGMOID_4x64_OPS:
		{
			__m512 al_in, r, r2, z, dn;
			__m512i ex_out;

			// c[0, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_0p0, al_in, r, r2, z, dn, ex_out);

			// c[0, 16-31]
			SIGMOID_F32_AVX512_DEF(c_float_0p1, al_in, r, r2, z, dn, ex_out);

			// c[0, 32-47]
			SIGMOID_F32_AVX512_DEF(c_float_0p2, al_in, r, r2, z, dn, ex_out);

			// c[0, 48-63]
			SIGMOID_F32_AVX512_DEF(c_float_0p3, al_in, r, r2, z, dn, ex_out);

			// c[1, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_1p0, al_in, r, r2, z, dn, ex_out);

			// c[1, 16-31]
			SIGMOID_F32_AVX512_DEF(c_float_1p1, al_in, r, r2, z, dn, ex_out);

			// c[1, 32-47]
			SIGMOID_F32_AVX512_DEF(c_float_1p2, al_in, r, r2, z, dn, ex_out);

			// c[1, 48-63]
			SIGMOID_F32_AVX512_DEF(c_float_1p3, al_in, r, r2, z, dn, ex_out);

			// c[2, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_2p0, al_in, r, r2, z, dn, ex_out);

			// c[2, 16-31]
			SIGMOID_F32_AVX512_DEF(c_float_2p1, al_in, r, r2, z, dn, ex_out);

			// c[2, 32-47]
			SIGMOID_F32_AVX512_DEF(c_float_2p2, al_in, r, r2, z, dn, ex_out);

			// c[2, 48-63]
			SIGMOID_F32_AVX512_DEF(c_float_2p3, al_in, r, r2, z, dn, ex_out);

			// c[3, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_3p0, al_in, r, r2, z, dn, ex_out);

			// c[3, 16-31]
			SIGMOID_F32_AVX512_DEF(c_float_3p1, al_in, r, r2, z, dn, ex_out);

			// c[3, 32-47]
			SIGMOID_F32_AVX512_DEF(c_float_3p2, al_in, r, r2, z, dn, ex_out);

			// c[3, 48-63]
			SIGMOID_F32_AVX512_DEF(c_float_3p3, al_in, r, r2, z, dn, ex_out);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_4x64_OPS_DISABLE:
		;

		// Case where the output C matrix is bf16 (downscaled) and this is the
		// final write for a given block within C.
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			// Actually the b matrix is of type bfloat16. However
			// in order to reuse this kernel for f32, the output
			// matrix type in kernel function signature is set to
			// f32 irrespective of original output matrix type.
			bfloat16* b_q = ( bfloat16* )b;
			dim_t ir = 0;

			// Store the results in downscaled type (bf16 instead of float).
			// c[0, 0-15]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_0p0,k0,0,0);
			// c[0, 16-31]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_0p1,k1,0,16);
			// c[0, 32-47]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_0p2,k2,0,32);
			// c[0, 48-63]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_0p3,k3,0,48);

			// c[1, 0-15]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_1p0,k0,1,0);
			// c[1, 16-31]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_1p1,k1,1,16);
			// c[1, 32-47]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_1p2,k2,1,32);
			// c[1, 48-63]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_1p3,k3,1,48);

			// c[2, 0-15]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_2p0,k0,2,0);
			// c[2, 16-31]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_2p1,k1,2,16);
			// c[2, 32-47]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_2p2,k2,2,32);
			// c[2, 48-63]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_2p3,k3,2,48);

			// c[3, 0-15]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_3p0,k0,3,0);
			// c[3, 16-31]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_3p1,k1,3,16);
			// c[3, 32-47]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_3p2,k2,3,32);
			// c[3, 48-63]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_3p3,k3,3,48);
		}
		// Case where the output C matrix is float
		else
		{
			// Store the results.
			// c[0,0-15]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) + 
				( cs_b * ( jr + 0 ) ), k0, c_float_0p0 );
			// c[0,16-31]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) + 
				( cs_b * ( jr + 16 ) ), k1, c_float_0p1 );
			// c[0,32-47]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) + 
				( cs_b * ( jr + 32 ) ), k2, c_float_0p2 );
			// c[0,48-63]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) + 
				( cs_b * ( jr + 48 ) ), k3, c_float_0p3 );

			// c[1,0-15]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) + 
				( cs_b * ( jr + 0 ) ), k0, c_float_1p0 );
			// c[1,16-31]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) + 
				( cs_b * ( jr + 16 ) ), k1, c_float_1p1 );
			// c[1,32-47]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) + 
				( cs_b * ( jr + 32 ) ), k2, c_float_1p2 );
			// c[1,48-63]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) + 
				( cs_b * ( jr + 48 ) ), k3, c_float_1p3 );

			// c[2,0-15]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 2 ) ) + 
				( cs_b * ( jr + 0 ) ), k0, c_float_2p0 );
			// c[2,16-31]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 2 ) ) + 
				( cs_b * ( jr + 16 ) ), k1, c_float_2p1 );
			// c[2,32-47]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 2 ) ) + 
				( cs_b * ( jr + 32 ) ), k2, c_float_2p2 );
			// c[2,48-63]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 2 ) ) + 
				( cs_b * ( jr + 48 ) ), k3, c_float_2p3 );

			// c[3,0-15]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 3 ) ) + 
				( cs_b * ( jr + 0 ) ), k0, c_float_3p0 );
			// c[3,16-31]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 3 ) ) + 
				( cs_b * ( jr + 16 ) ), k1, c_float_3p1 );
			// c[3,32-47]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 3 ) ) + 
				( cs_b * ( jr + 32 ) ), k2, c_float_3p2 );
			// c[3,48-63]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 3 ) ) + 
				( cs_b * ( jr + 48 ) ), k3, c_float_3p3 );
		}

		post_ops_attr.post_op_c_j += NR_L;
	}
}

LPGEMM_ELTWISE_OPS_M_FRINGE_KERNEL(bfloat16,float,bf16of32_3x64)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_3x64_OPS_DISABLE,
						  &&POST_OPS_BIAS_3x64_OPS,
						  &&POST_OPS_RELU_3x64_OPS,
						  &&POST_OPS_RELU_SCALE_3x64_OPS,
						  &&POST_OPS_GELU_TANH_3x64_OPS,
						  &&POST_OPS_GELU_ERF_3x64_OPS,
						  &&POST_OPS_CLIP_3x64_OPS,
						  &&POST_OPS_DOWNSCALE_3x64_OPS,
						  &&POST_OPS_MATRIX_ADD_3x64_OPS,
						  &&POST_OPS_SWISH_3x64_OPS,
						  &&POST_OPS_MATRIX_MUL_3x64_OPS,
						  &&POST_OPS_TANH_3x64_OPS,
						  &&POST_OPS_SIGMOID_3x64_OPS
						};
	dim_t NR = 64;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();
	__m512 c_float_0p1 = _mm512_setzero_ps();
	__m512 c_float_0p2 = _mm512_setzero_ps();
	__m512 c_float_0p3 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();
	__m512 c_float_1p1 = _mm512_setzero_ps();
	__m512 c_float_1p2 = _mm512_setzero_ps();
	__m512 c_float_1p3 = _mm512_setzero_ps();

	__m512 c_float_2p0 = _mm512_setzero_ps();
	__m512 c_float_2p1 = _mm512_setzero_ps();
	__m512 c_float_2p2 = _mm512_setzero_ps();
	__m512 c_float_2p3 = _mm512_setzero_ps();

	__m512 selector1 = _mm512_setzero_ps();
	__m512 selector2 = _mm512_setzero_ps();
	__m512 selector3 = _mm512_setzero_ps();
	__m512 selector4 = _mm512_setzero_ps();

	__mmask16 k0 = 0xFFFF, k1 = 0xFFFF, k2 = 0xFFFF, k3 = 0xFFFF;

	dim_t NR_L = NR;
	for( dim_t jr = 0; jr < n0; jr += NR_L )
	{
		dim_t n_left = n0 - jr;
		NR_L = bli_min( NR_L, ( n_left >> 4 ) << 4 );
		if( NR_L == 0 ) { NR_L = 16; }

		dim_t nr0 = bli_min( n0 - jr, NR_L );
		if( nr0 == 64 )
		{
			// all masks are already set.
			// Nothing to modify.
		}
		else if( nr0 == 48 )
		{
			k3 = 0x0;
		}
		else if( nr0 == 32 )
		{
			k2 = k3 = 0x0;
		}
		else if( nr0 == 16 )
		{
			k1 = k2 = k3 = 0;
		}
		else if( nr0 < 16 )
		{
			k0 = (0xFFFF >> (16 - (nr0 & 0x0F)));
			k1 = k2 = k3 = 0;
		}

		// 1stx64 block.
		c_float_0p0 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k0, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 0 ) ) ) );
		c_float_0p1 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k1, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 16 ) ) ) );
		c_float_0p2 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k2, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 32 ) ) ) );
		c_float_0p3 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k3, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 48 ) ) ) );

		// 2ndx64 block.
		c_float_1p0 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k0, \
			a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 0 ) ) ) );
		c_float_1p1 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k1, \
			a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 16 ) ) ) );
		c_float_1p2 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k2, \
			a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 32 ) ) ) );
		c_float_1p3 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k3, \
			a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 48 ) ) ) );

		// 3rdx64 block.
		c_float_2p0 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k0, \
			a + ( rs_a * ( 2 ) ) + ( cs_a * ( jr + 0 ) ) ) );
		c_float_2p1 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k1, \
			a + ( rs_a * ( 2 ) ) + ( cs_a * ( jr + 16 ) ) ) );
		c_float_2p2 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k2, \
			a + ( rs_a * ( 2 ) ) + ( cs_a * ( jr + 32 ) ) ) );
		c_float_2p3 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k3, \
			a + ( rs_a * ( 2 ) ) + ( cs_a * ( jr + 48 ) ) ) );

		// Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_3x64_OPS:
		{
			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
				 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
				if ( post_ops_list_temp->stor_type == BF16 )
				{
					BF16_F32_BIAS_LOAD(selector1, k0, 0);
					BF16_F32_BIAS_LOAD(selector2, k1, 1);
					BF16_F32_BIAS_LOAD(selector3, k2, 2);
					BF16_F32_BIAS_LOAD(selector4, k3, 3);
				}
				else
				{
					selector1 =
					_mm512_maskz_loadu_ps( k0,
						( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					selector2 =
					_mm512_maskz_loadu_ps( k1,
						( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					selector3 =
					_mm512_maskz_loadu_ps( k2,
						( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					selector4 =
					_mm512_maskz_loadu_ps( k3,
						( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_j + ( 3 * 16 ) );
				}

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[0, 16-31]
				c_float_0p1 = _mm512_add_ps( selector2, c_float_0p1 );

				// c[0,32-47]
				c_float_0p2 = _mm512_add_ps( selector3, c_float_0p2 );

				// c[0,48-63]
				c_float_0p3 = _mm512_add_ps( selector4, c_float_0p3 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

				// c[1, 16-31]
				c_float_1p1 = _mm512_add_ps( selector2, c_float_1p1 );

				// c[1,32-47]
				c_float_1p2 = _mm512_add_ps( selector3, c_float_1p2 );

				// c[1,48-63]
				c_float_1p3 = _mm512_add_ps( selector4, c_float_1p3 );

				// c[2,0-15]
				c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );

				// c[2, 16-31]
				c_float_2p1 = _mm512_add_ps( selector2, c_float_2p1 );

				// c[2,32-47]
				c_float_2p2 = _mm512_add_ps( selector3, c_float_2p2 );

				// c[2,48-63]
				c_float_2p3 = _mm512_add_ps( selector4, c_float_2p3 );
			}
			else
			{
				// If original output was columns major, then by the time
				// kernel sees it, the matrix would be accessed as if it were
				// transposed. Due to this the bias array will be accessed by
				// the ic index, and each bias element corresponds to an
				// entire row of the transposed output array, instead of an
				// entire column.
				if ( post_ops_list_temp->stor_type == BF16 )
				{
					__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
					BF16_F32_BIAS_BCAST(selector1, bias_mask, 0);
					BF16_F32_BIAS_BCAST(selector2, bias_mask, 1);
					BF16_F32_BIAS_BCAST(selector3, bias_mask, 2);
				}
				else
				{
					selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_i + 0 ) );
					selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_i + 1 ) );
					selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_i + 2 ) );
				}

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[0, 16-31]
				c_float_0p1 = _mm512_add_ps( selector1, c_float_0p1 );

				// c[0,32-47]
				c_float_0p2 = _mm512_add_ps( selector1, c_float_0p2 );

				// c[0,48-63]
				c_float_0p3 = _mm512_add_ps( selector1, c_float_0p3 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );

				// c[1, 16-31]
				c_float_1p1 = _mm512_add_ps( selector2, c_float_1p1 );

				// c[1,32-47]
				c_float_1p2 = _mm512_add_ps( selector2, c_float_1p2 );

				// c[1,48-63]
				c_float_1p3 = _mm512_add_ps( selector2, c_float_1p3 );

				// c[2,0-15]
				c_float_2p0 = _mm512_add_ps( selector3, c_float_2p0 );

				// c[2, 16-31]
				c_float_2p1 = _mm512_add_ps( selector3, c_float_2p1 );

				// c[2,32-47]
				c_float_2p2 = _mm512_add_ps( selector3, c_float_2p2 );

				// c[2,48-63]
				c_float_2p3 = _mm512_add_ps( selector3, c_float_2p3 );
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_3x64_OPS:
		{
			selector1 = _mm512_setzero_ps();

			// c[0,0-15]
			c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_max_ps( selector1, c_float_0p1 );

			// c[0,32-47]
			c_float_0p2 = _mm512_max_ps( selector1, c_float_0p2 );

			// c[0,48-63]
			c_float_0p3 = _mm512_max_ps( selector1, c_float_0p3 );

			// c[1,0-15]
			c_float_1p0 = _mm512_max_ps( selector1, c_float_1p0 );

			// c[1,16-31]
			c_float_1p1 = _mm512_max_ps( selector1, c_float_1p1 );

			// c[1,32-47]
			c_float_1p2 = _mm512_max_ps( selector1, c_float_1p2 );

			// c[1,48-63]
			c_float_1p3 = _mm512_max_ps( selector1, c_float_1p3 );

			// c[2,0-15]
			c_float_2p0 = _mm512_max_ps( selector1, c_float_2p0 );

			// c[2,16-31]
			c_float_2p1 = _mm512_max_ps( selector1, c_float_2p1 );

			// c[2,32-47]
			c_float_2p2 = _mm512_max_ps( selector1, c_float_2p2 );

			// c[2,48-63]
			c_float_2p3 = _mm512_max_ps( selector1, c_float_2p3 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_3x64_OPS:
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

			// c[0, 48-63]
			RELU_SCALE_OP_F32_AVX512(c_float_0p3)

			// c[1, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_1p0)

			// c[1, 16-31]
			RELU_SCALE_OP_F32_AVX512(c_float_1p1)

			// c[1, 32-47]
			RELU_SCALE_OP_F32_AVX512(c_float_1p2)

			// c[1, 48-63]
			RELU_SCALE_OP_F32_AVX512(c_float_1p3)

			// c[2, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_2p0)

			// c[2, 16-31]
			RELU_SCALE_OP_F32_AVX512(c_float_2p1)

			// c[2, 32-47]
			RELU_SCALE_OP_F32_AVX512(c_float_2p2)

			// c[2, 48-63]
			RELU_SCALE_OP_F32_AVX512(c_float_2p3)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_TANH_3x64_OPS:
		{
			__m512 dn, z, x, r2, r, x_tanh;
			__m512i q;

			// c[0, 0-15]
			GELU_TANH_F32_AVX512(c_float_0p0, r, r2, x, z, dn, x_tanh, q)

			// c[0, 16-31]
			GELU_TANH_F32_AVX512(c_float_0p1, r, r2, x, z, dn, x_tanh, q)

			// c[0, 32-47]
			GELU_TANH_F32_AVX512(c_float_0p2, r, r2, x, z, dn, x_tanh, q)

			// c[0, 48-63]
			GELU_TANH_F32_AVX512(c_float_0p3, r, r2, x, z, dn, x_tanh, q)

			// c[1, 0-15]
			GELU_TANH_F32_AVX512(c_float_1p0, r, r2, x, z, dn, x_tanh, q)

			// c[1, 16-31]
			GELU_TANH_F32_AVX512(c_float_1p1, r, r2, x, z, dn, x_tanh, q)

			// c[1, 32-47]
			GELU_TANH_F32_AVX512(c_float_1p2, r, r2, x, z, dn, x_tanh, q)

			// c[1, 48-63]
			GELU_TANH_F32_AVX512(c_float_1p3, r, r2, x, z, dn, x_tanh, q)

			// c[2, 0-15]
			GELU_TANH_F32_AVX512(c_float_2p0, r, r2, x, z, dn, x_tanh, q)

			// c[2, 16-31]
			GELU_TANH_F32_AVX512(c_float_2p1, r, r2, x, z, dn, x_tanh, q)

			// c[2, 32-47]
			GELU_TANH_F32_AVX512(c_float_2p2, r, r2, x, z, dn, x_tanh, q)

			// c[2, 48-63]
			GELU_TANH_F32_AVX512(c_float_2p3, r, r2, x, z, dn, x_tanh, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_ERF_3x64_OPS:
		{
			__m512 x, r, x_erf;

			// c[0, 0-15]
			GELU_ERF_F32_AVX512(c_float_0p0, r, x, x_erf)

			// c[0, 16-31]
			GELU_ERF_F32_AVX512(c_float_0p1, r, x, x_erf)

			// c[0, 32-47]
			GELU_ERF_F32_AVX512(c_float_0p2, r, x, x_erf)

			// c[0, 48-63]
			GELU_ERF_F32_AVX512(c_float_0p3, r, x, x_erf)

			// c[1, 0-15]
			GELU_ERF_F32_AVX512(c_float_1p0, r, x, x_erf)

			// c[1, 16-31]
			GELU_ERF_F32_AVX512(c_float_1p1, r, x, x_erf)

			// c[1, 32-47]
			GELU_ERF_F32_AVX512(c_float_1p2, r, x, x_erf)

			// c[1, 48-63]
			GELU_ERF_F32_AVX512(c_float_1p3, r, x, x_erf)

			// c[2, 0-15]
			GELU_ERF_F32_AVX512(c_float_2p0, r, x, x_erf)

			// c[2, 16-31]
			GELU_ERF_F32_AVX512(c_float_2p1, r, x, x_erf)

			// c[2, 32-47]
			GELU_ERF_F32_AVX512(c_float_2p2, r, x, x_erf)

			// c[2, 48-63]
			GELU_ERF_F32_AVX512(c_float_2p3, r, x, x_erf)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_CLIP_3x64_OPS:
		{
			__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
			__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

			// c[0, 0-15]
			CLIP_F32_AVX512(c_float_0p0, min, max)

			// c[0, 16-31]
			CLIP_F32_AVX512(c_float_0p1, min, max)

			// c[0, 32-47]
			CLIP_F32_AVX512(c_float_0p2, min, max)

			// c[0, 48-63]
			CLIP_F32_AVX512(c_float_0p3, min, max)

			// c[1, 0-15]
			CLIP_F32_AVX512(c_float_1p0, min, max)

			// c[1, 16-31]
			CLIP_F32_AVX512(c_float_1p1, min, max)

			// c[1, 32-47]
			CLIP_F32_AVX512(c_float_1p2, min, max)

			// c[1, 48-63]
			CLIP_F32_AVX512(c_float_1p3, min, max)

			// c[2, 0-15]
			CLIP_F32_AVX512(c_float_2p0, min, max)

			// c[2, 16-31]
			CLIP_F32_AVX512(c_float_2p1, min, max)

			// c[2, 32-47]
			CLIP_F32_AVX512(c_float_2p2, min, max)

			// c[2, 48-63]
			CLIP_F32_AVX512(c_float_2p3, min, max)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_DOWNSCALE_3x64_OPS:
		{
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
						_mm512_maskz_loadu_ps( k0,
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					selector2 =
						_mm512_maskz_loadu_ps( k1,
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					selector3 =
						_mm512_maskz_loadu_ps( k2,
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					selector4 =
						_mm512_maskz_loadu_ps( k3,
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 3 * 16 ) );
				}

				if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
				{
					zero_point0 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_loadu_epi16( k0,
						( ( bfloat16* )post_ops_list_temp->op_args1 ) +
						post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
					zero_point1 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_loadu_epi16( k1,
						( ( bfloat16* )post_ops_list_temp->op_args1 ) +
						post_ops_attr.post_op_c_j + ( 1 * 16 ) ) );
					zero_point2 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_loadu_epi16( k2,
						( ( bfloat16* )post_ops_list_temp->op_args1 ) +
						post_ops_attr.post_op_c_j + ( 2 * 16 ) ) );
					zero_point3 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_loadu_epi16( k3,
						( ( bfloat16* )post_ops_list_temp->op_args1 ) +
						post_ops_attr.post_op_c_j + ( 3 * 16 ) ) );
				}

				// c[0, 0-15]
				SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

				// c[0, 16-31]
				SCL_MULRND_F32(c_float_0p1,selector2,zero_point1);

				// c[0, 32-47]
				SCL_MULRND_F32(c_float_0p2,selector3,zero_point2);

				// c[0, 48-63]
				SCL_MULRND_F32(c_float_0p3,selector4,zero_point3);

				// c[1, 0-15]
				SCL_MULRND_F32(c_float_1p0,selector1,zero_point0);

				// c[1, 16-31]
				SCL_MULRND_F32(c_float_1p1,selector2,zero_point1);

				// c[1, 32-47]
				SCL_MULRND_F32(c_float_1p2,selector3,zero_point2);

				// c[1, 48-63]
				SCL_MULRND_F32(c_float_1p3,selector4,zero_point3);

				// c[2, 0-15]
				SCL_MULRND_F32(c_float_2p0,selector1,zero_point0);

				// c[2, 16-31]
				SCL_MULRND_F32(c_float_2p1,selector2,zero_point1);

				// c[2, 32-47]
				SCL_MULRND_F32(c_float_2p2,selector3,zero_point2);

				// c[2, 48-63]
				SCL_MULRND_F32(c_float_2p3,selector4,zero_point3);
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

				// c[0, 48-63]
				SCL_MULRND_F32(c_float_0p3,selector1,zero_point0);

				// c[1, 0-15]
				SCL_MULRND_F32(c_float_1p0,selector2,zero_point1);

				// c[1, 16-31]
				SCL_MULRND_F32(c_float_1p1,selector2,zero_point1);

				// c[1, 32-47]
				SCL_MULRND_F32(c_float_1p2,selector2,zero_point1);

				// c[1, 48-63]
				SCL_MULRND_F32(c_float_1p3,selector2,zero_point1);

				// c[2, 0-15]
				SCL_MULRND_F32(c_float_2p0,selector3,zero_point2);

				// c[2, 16-31]
				SCL_MULRND_F32(c_float_2p1,selector3,zero_point2);

				// c[2, 32-47]
				SCL_MULRND_F32(c_float_2p2,selector3,zero_point2);

				// c[2, 48-63]
				SCL_MULRND_F32(c_float_2p3,selector3,zero_point2);
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_ADD_3x64_OPS:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
					( ( post_ops_list_temp->stor_type == NONE ) &&
					  ( post_ops_attr.c_stor_type == BF16 ) );

			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();

			// Even though different registers are used for scalar in column and
			// row major case, all those registers will contain the same value.
			if ( post_ops_list_temp->scale_factor_len == 1 )
			{
				scl_fctr1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
						_mm512_maskz_loadu_ps( k0,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					scl_fctr2 =
						_mm512_maskz_loadu_ps( k1,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					scl_fctr3 =
						_mm512_maskz_loadu_ps( k2,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					scl_fctr4 =
						_mm512_maskz_loadu_ps( k3,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 3 * 16 ) );
				}
				else
				{
					scl_fctr1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 0 ) );
					scl_fctr2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 1 ) );
					scl_fctr3 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 2 ) );
				}
			}

			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1);

					// c[2:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2);
				}
			}
			else
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1);

					// c[2:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2);
				}
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_MUL_3x64_OPS:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
					( ( post_ops_list_temp->stor_type == NONE ) &&
					  ( post_ops_attr.c_stor_type == BF16 ) );

			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();

			// Even though different registers are used for scalar in column and
			// row major case, all those registers will contain the same value.
			if ( post_ops_list_temp->scale_factor_len == 1 )
			{
				scl_fctr1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
						_mm512_maskz_loadu_ps( k0,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					scl_fctr2 =
						_mm512_maskz_loadu_ps( k1,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					scl_fctr3 =
						_mm512_maskz_loadu_ps( k2,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					scl_fctr4 =
						_mm512_maskz_loadu_ps( k3,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 3 * 16 ) );
				}
				else
				{
					scl_fctr1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 0 ) );
					scl_fctr2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 1 ) );
					scl_fctr3 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 2 ) );
				}
			}

			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1);

					// c[2:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2);
				}
			}
			else
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1);

					// c[2:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,2);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr3,scl_fctr3,scl_fctr3,scl_fctr3,2);
				}
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SWISH_3x64_OPS:
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

			// c[0, 48-63]
			SWISH_F32_AVX512_DEF(c_float_0p3, selector1, al_in, r, r2, z, dn, ex_out);

			// c[1, 0-15]
			SWISH_F32_AVX512_DEF(c_float_1p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[1, 16-31]
			SWISH_F32_AVX512_DEF(c_float_1p1, selector1, al_in, r, r2, z, dn, ex_out);

			// c[1, 32-47]
			SWISH_F32_AVX512_DEF(c_float_1p2, selector1, al_in, r, r2, z, dn, ex_out);

			// c[1, 48-63]
			SWISH_F32_AVX512_DEF(c_float_1p3, selector1, al_in, r, r2, z, dn, ex_out);

			// c[2, 0-15]
			SWISH_F32_AVX512_DEF(c_float_2p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[2, 16-31]
			SWISH_F32_AVX512_DEF(c_float_2p1, selector1, al_in, r, r2, z, dn, ex_out);

			// c[2, 32-47]
			SWISH_F32_AVX512_DEF(c_float_2p2, selector1, al_in, r, r2, z, dn, ex_out);

			// c[2, 48-63]
			SWISH_F32_AVX512_DEF(c_float_2p3, selector1, al_in, r, r2, z, dn, ex_out);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_TANH_3x64_OPS:
		{
			__m512 dn, z, x, r2, r;
			__m512i q;

			// c[0, 0-15]
			TANHF_AVX512(c_float_0p0, r, r2, x, z, dn,  q)

			// c[0, 16-31]
			TANHF_AVX512(c_float_0p1, r, r2, x, z, dn, q)

			// c[0, 32-47]
			TANHF_AVX512(c_float_0p2, r, r2, x, z, dn, q)

			// c[0, 48-63]
			TANHF_AVX512(c_float_0p3, r, r2, x, z, dn, q)

			// c[1, 0-15]
			TANHF_AVX512(c_float_1p0, r, r2, x, z, dn, q)

			// c[1, 16-31]
			TANHF_AVX512(c_float_1p1, r, r2, x, z, dn, q)

			// c[1, 32-47]
			TANHF_AVX512(c_float_1p2, r, r2, x, z, dn, q)

			// c[1, 48-63]
			TANHF_AVX512(c_float_1p3, r, r2, x, z, dn, q)

			// c[2, 0-15]
			TANHF_AVX512(c_float_2p0, r, r2, x, z, dn, q)

			// c[2, 16-31]
			TANHF_AVX512(c_float_2p1, r, r2, x, z, dn, q)

			// c[2, 32-47]
			TANHF_AVX512(c_float_2p2, r, r2, x, z, dn, q)

			// c[2, 48-63]
			TANHF_AVX512(c_float_2p3, r, r2, x, z, dn, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SIGMOID_3x64_OPS:
		{
			__m512 al_in, r, r2, z, dn;
			__m512i ex_out;

			// c[0, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_0p0, al_in, r, r2, z, dn, ex_out);

			// c[0, 16-31]
			SIGMOID_F32_AVX512_DEF(c_float_0p1, al_in, r, r2, z, dn, ex_out);

			// c[0, 32-47]
			SIGMOID_F32_AVX512_DEF(c_float_0p2, al_in, r, r2, z, dn, ex_out);

			// c[0, 48-63]
			SIGMOID_F32_AVX512_DEF(c_float_0p3, al_in, r, r2, z, dn, ex_out);

			// c[1, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_1p0, al_in, r, r2, z, dn, ex_out);

			// c[1, 16-31]
			SIGMOID_F32_AVX512_DEF(c_float_1p1, al_in, r, r2, z, dn, ex_out);

			// c[1, 32-47]
			SIGMOID_F32_AVX512_DEF(c_float_1p2, al_in, r, r2, z, dn, ex_out);

			// c[1, 48-63]
			SIGMOID_F32_AVX512_DEF(c_float_1p3, al_in, r, r2, z, dn, ex_out);

			// c[2, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_2p0, al_in, r, r2, z, dn, ex_out);

			// c[2, 16-31]
			SIGMOID_F32_AVX512_DEF(c_float_2p1, al_in, r, r2, z, dn, ex_out);

			// c[2, 32-47]
			SIGMOID_F32_AVX512_DEF(c_float_2p2, al_in, r, r2, z, dn, ex_out);

			// c[2, 48-63]
			SIGMOID_F32_AVX512_DEF(c_float_2p3, al_in, r, r2, z, dn, ex_out);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_3x64_OPS_DISABLE:
		;

		// Case where the output C matrix is bf16 (downscaled) and this is the
		// final write for a given block within C.
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			// Actually the b matrix is of type bfloat16. However
			// in order to reuse this kernel for f32, the output
			// matrix type in kernel function signature is set to
			// f32 irrespective of original output matrix type.
			bfloat16* b_q = ( bfloat16* )b;
			dim_t ir = 0;

			// Store the results in downscaled type (bf16 instead of float).
			// c[0, 0-15]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_0p0,k0,0,0);
			// c[0, 16-31]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_0p1,k1,0,16);
			// c[0, 32-47]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_0p2,k2,0,32);
			// c[0, 48-63]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_0p3,k3,0,48);

			// c[1, 0-15]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_1p0,k0,1,0);
			// c[1, 16-31]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_1p1,k1,1,16);
			// c[1, 32-47]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_1p2,k2,1,32);
			// c[1, 48-63]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_1p3,k3,1,48);

			// c[2, 0-15]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_2p0,k0,2,0);
			// c[2, 16-31]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_2p1,k1,2,16);
			// c[2, 32-47]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_2p2,k2,2,32);
			// c[2, 48-63]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_2p3,k3,2,48);
		}
		// Case where the output C matrix is float
		else
		{
			// Store the results.
			// c[0,0-15]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) + 
				( cs_b * ( jr + 0 ) ), k0, c_float_0p0 );
			// c[0,16-31]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) + 
				( cs_b * ( jr + 16 ) ), k1, c_float_0p1 );
			// c[0,32-47]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) + 
				( cs_b * ( jr + 32 ) ), k2, c_float_0p2 );
			// c[0,48-63]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) + 
				( cs_b * ( jr + 48 ) ), k3, c_float_0p3 );

			// c[1,0-15]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) + 
				( cs_b * ( jr + 0 ) ), k0, c_float_1p0 );
			// c[1,16-31]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) + 
				( cs_b * ( jr + 16 ) ), k1, c_float_1p1 );
			// c[1,32-47]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) + 
				( cs_b * ( jr + 32 ) ), k2, c_float_1p2 );
			// c[1,48-63]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) + 
				( cs_b * ( jr + 48 ) ), k3, c_float_1p3 );

			// c[2,0-15]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 2 ) ) + 
				( cs_b * ( jr + 0 ) ), k0, c_float_2p0 );
			// c[2,16-31]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 2 ) ) + 
				( cs_b * ( jr + 16 ) ), k1, c_float_2p1 );
			// c[2,32-47]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 2 ) ) + 
				( cs_b * ( jr + 32 ) ), k2, c_float_2p2 );
			// c[2,48-63]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 2 ) ) + 
				( cs_b * ( jr + 48 ) ), k3, c_float_2p3 );
		}

		post_ops_attr.post_op_c_j += NR_L;
	}
}

LPGEMM_ELTWISE_OPS_M_FRINGE_KERNEL(bfloat16,float,bf16of32_2x64)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_2x64_OPS_DISABLE,
						  &&POST_OPS_BIAS_2x64_OPS,
						  &&POST_OPS_RELU_2x64_OPS,
						  &&POST_OPS_RELU_SCALE_2x64_OPS,
						  &&POST_OPS_GELU_TANH_2x64_OPS,
						  &&POST_OPS_GELU_ERF_2x64_OPS,
						  &&POST_OPS_CLIP_2x64_OPS,
						  &&POST_OPS_DOWNSCALE_2x64_OPS,
						  &&POST_OPS_MATRIX_ADD_2x64_OPS,
						  &&POST_OPS_SWISH_2x64_OPS,
						  &&POST_OPS_MATRIX_MUL_2x64_OPS,
						  &&POST_OPS_TANH_2x64_OPS,
						  &&POST_OPS_SIGMOID_2x64_OPS
						};
	dim_t NR = 64;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();
	__m512 c_float_0p1 = _mm512_setzero_ps();
	__m512 c_float_0p2 = _mm512_setzero_ps();
	__m512 c_float_0p3 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();
	__m512 c_float_1p1 = _mm512_setzero_ps();
	__m512 c_float_1p2 = _mm512_setzero_ps();
	__m512 c_float_1p3 = _mm512_setzero_ps();

	__m512 selector1 = _mm512_setzero_ps();
	__m512 selector2 = _mm512_setzero_ps();
	__m512 selector3 = _mm512_setzero_ps();
	__m512 selector4 = _mm512_setzero_ps();

	__mmask16 k0 = 0xFFFF, k1 = 0xFFFF, k2 = 0xFFFF, k3 = 0xFFFF;

	dim_t NR_L = NR;
	for( dim_t jr = 0; jr < n0; jr += NR_L )
	{
		dim_t n_left = n0 - jr;
		NR_L = bli_min( NR_L, ( n_left >> 4 ) << 4 );
		if( NR_L == 0 ) { NR_L = 16; }

		dim_t nr0 = bli_min( n0 - jr, NR_L );
		if( nr0 == 64 )
		{
			// all masks are already set.
			// Nothing to modify.
		}
		else if( nr0 == 48 )
		{
			k3 = 0x0;
		}
		else if( nr0 == 32 )
		{
			k2 = k3 = 0x0;
		}
		else if( nr0 == 16 )
		{
			k1 = k2 = k3 = 0;
		}
		else if( nr0 < 16 )
		{
			k0 = (0xFFFF >> (16 - (nr0 & 0x0F)));
			k1 = k2 = k3 = 0;
		}

		// 1stx64 block.
		c_float_0p0 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k0, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 0 ) ) ) );
		c_float_0p1 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k1, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 16 ) ) ) );
		c_float_0p2 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k2, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 32 ) ) ) );
		c_float_0p3 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k3, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 48 ) ) ) );

		// 2ndx64 block.
		c_float_1p0 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k0, \
			a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 0 ) ) ) );
		c_float_1p1 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k1, \
			a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 16 ) ) ) );
		c_float_1p2 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k2, \
			a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 32 ) ) ) );
		c_float_1p3 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k3, \
			a + ( rs_a * ( 1 ) ) + ( cs_a * ( jr + 48 ) ) ) );

		// Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_2x64_OPS:
		{
			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
				 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
				if ( post_ops_list_temp->stor_type == BF16 )
				{
					BF16_F32_BIAS_LOAD(selector1, k0, 0);
					BF16_F32_BIAS_LOAD(selector2, k1, 1);
					BF16_F32_BIAS_LOAD(selector3, k2, 2);
					BF16_F32_BIAS_LOAD(selector4, k3, 3);
				}
				else
				{
					selector1 =
					_mm512_maskz_loadu_ps( k0,
						( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					selector2 =
					_mm512_maskz_loadu_ps( k1,
						( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					selector3 =
					_mm512_maskz_loadu_ps( k2,
						( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					selector4 =
					_mm512_maskz_loadu_ps( k3,
						( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_j + ( 3 * 16 ) );
				}

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[0, 16-31]
				c_float_0p1 = _mm512_add_ps( selector2, c_float_0p1 );

				// c[0,32-47]
				c_float_0p2 = _mm512_add_ps( selector3, c_float_0p2 );

				// c[0,48-63]
				c_float_0p3 = _mm512_add_ps( selector4, c_float_0p3 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

				// c[1, 16-31]
				c_float_1p1 = _mm512_add_ps( selector2, c_float_1p1 );

				// c[1,32-47]
				c_float_1p2 = _mm512_add_ps( selector3, c_float_1p2 );

				// c[1,48-63]
				c_float_1p3 = _mm512_add_ps( selector4, c_float_1p3 );
			}
			else
			{
				// If original output was columns major, then by the time
				// kernel sees it, the matrix would be accessed as if it were
				// transposed. Due to this the bias array will be accessed by
				// the ic index, and each bias element corresponds to an
				// entire row of the transposed output array, instead of an
				// entire column.
				if ( post_ops_list_temp->stor_type == BF16 )
				{
					__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
					BF16_F32_BIAS_BCAST(selector1, bias_mask, 0);
					BF16_F32_BIAS_BCAST(selector2, bias_mask, 1);
				}
				else
				{
					selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_i + 0 ) );
					selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_i + 1 ) );
				}

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[0, 16-31]
				c_float_0p1 = _mm512_add_ps( selector1, c_float_0p1 );

				// c[0,32-47]
				c_float_0p2 = _mm512_add_ps( selector1, c_float_0p2 );

				// c[0,48-63]
				c_float_0p3 = _mm512_add_ps( selector1, c_float_0p3 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );

				// c[1, 16-31]
				c_float_1p1 = _mm512_add_ps( selector2, c_float_1p1 );

				// c[1,32-47]
				c_float_1p2 = _mm512_add_ps( selector2, c_float_1p2 );

				// c[1,48-63]
				c_float_1p3 = _mm512_add_ps( selector2, c_float_1p3 );
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_2x64_OPS:
		{
			selector1 = _mm512_setzero_ps();

			// c[0,0-15]
			c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_max_ps( selector1, c_float_0p1 );

			// c[0,32-47]
			c_float_0p2 = _mm512_max_ps( selector1, c_float_0p2 );

			// c[0,48-63]
			c_float_0p3 = _mm512_max_ps( selector1, c_float_0p3 );

			// c[1,0-15]
			c_float_1p0 = _mm512_max_ps( selector1, c_float_1p0 );

			// c[1,16-31]
			c_float_1p1 = _mm512_max_ps( selector1, c_float_1p1 );

			// c[1,32-47]
			c_float_1p2 = _mm512_max_ps( selector1, c_float_1p2 );

			// c[1,48-63]
			c_float_1p3 = _mm512_max_ps( selector1, c_float_1p3 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_2x64_OPS:
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

			// c[0, 48-63]
			RELU_SCALE_OP_F32_AVX512(c_float_0p3)

			// c[1, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_1p0)

			// c[1, 16-31]
			RELU_SCALE_OP_F32_AVX512(c_float_1p1)

			// c[1, 32-47]
			RELU_SCALE_OP_F32_AVX512(c_float_1p2)

			// c[1, 48-63]
			RELU_SCALE_OP_F32_AVX512(c_float_1p3)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_TANH_2x64_OPS:
		{
			__m512 dn, z, x, r2, r, x_tanh;
			__m512i q;

			// c[0, 0-15]
			GELU_TANH_F32_AVX512(c_float_0p0, r, r2, x, z, dn, x_tanh, q)

			// c[0, 16-31]
			GELU_TANH_F32_AVX512(c_float_0p1, r, r2, x, z, dn, x_tanh, q)

			// c[0, 32-47]
			GELU_TANH_F32_AVX512(c_float_0p2, r, r2, x, z, dn, x_tanh, q)

			// c[0, 48-63]
			GELU_TANH_F32_AVX512(c_float_0p3, r, r2, x, z, dn, x_tanh, q)

			// c[1, 0-15]
			GELU_TANH_F32_AVX512(c_float_1p0, r, r2, x, z, dn, x_tanh, q)

			// c[1, 16-31]
			GELU_TANH_F32_AVX512(c_float_1p1, r, r2, x, z, dn, x_tanh, q)

			// c[1, 32-47]
			GELU_TANH_F32_AVX512(c_float_1p2, r, r2, x, z, dn, x_tanh, q)

			// c[1, 48-63]
			GELU_TANH_F32_AVX512(c_float_1p3, r, r2, x, z, dn, x_tanh, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_ERF_2x64_OPS:
		{
			__m512 x, r, x_erf;

			// c[0, 0-15]
			GELU_ERF_F32_AVX512(c_float_0p0, r, x, x_erf)

			// c[0, 16-31]
			GELU_ERF_F32_AVX512(c_float_0p1, r, x, x_erf)

			// c[0, 32-47]
			GELU_ERF_F32_AVX512(c_float_0p2, r, x, x_erf)

			// c[0, 48-63]
			GELU_ERF_F32_AVX512(c_float_0p3, r, x, x_erf)

			// c[1, 0-15]
			GELU_ERF_F32_AVX512(c_float_1p0, r, x, x_erf)

			// c[1, 16-31]
			GELU_ERF_F32_AVX512(c_float_1p1, r, x, x_erf)

			// c[1, 32-47]
			GELU_ERF_F32_AVX512(c_float_1p2, r, x, x_erf)

			// c[1, 48-63]
			GELU_ERF_F32_AVX512(c_float_1p3, r, x, x_erf)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_CLIP_2x64_OPS:
		{
			__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
			__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

			// c[0, 0-15]
			CLIP_F32_AVX512(c_float_0p0, min, max)

			// c[0, 16-31]
			CLIP_F32_AVX512(c_float_0p1, min, max)

			// c[0, 32-47]
			CLIP_F32_AVX512(c_float_0p2, min, max)

			// c[0, 48-63]
			CLIP_F32_AVX512(c_float_0p3, min, max)

			// c[1, 0-15]
			CLIP_F32_AVX512(c_float_1p0, min, max)

			// c[1, 16-31]
			CLIP_F32_AVX512(c_float_1p1, min, max)

			// c[1, 32-47]
			CLIP_F32_AVX512(c_float_1p2, min, max)

			// c[1, 48-63]
			CLIP_F32_AVX512(c_float_1p3, min, max)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_DOWNSCALE_2x64_OPS:
		{
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
						_mm512_maskz_loadu_ps( k0,
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					selector2 =
						_mm512_maskz_loadu_ps( k1,
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					selector3 =
						_mm512_maskz_loadu_ps( k2,
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					selector4 =
						_mm512_maskz_loadu_ps( k3,
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 3 * 16 ) );
				}

				if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
				{
					zero_point0 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_loadu_epi16( k0,
						( ( bfloat16* )post_ops_list_temp->op_args1 ) +
						post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
					zero_point1 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_loadu_epi16( k1,
						( ( bfloat16* )post_ops_list_temp->op_args1 ) +
						post_ops_attr.post_op_c_j + ( 1 * 16 ) ) );
					zero_point2 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_loadu_epi16( k2,
						( ( bfloat16* )post_ops_list_temp->op_args1 ) +
						post_ops_attr.post_op_c_j + ( 2 * 16 ) ) );
					zero_point3 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_loadu_epi16( k3,
						( ( bfloat16* )post_ops_list_temp->op_args1 ) +
						post_ops_attr.post_op_c_j + ( 3 * 16 ) ) );
				}

				// c[0, 0-15]
				SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

				// c[0, 16-31]
				SCL_MULRND_F32(c_float_0p1,selector2,zero_point1);

				// c[0, 32-47]
				SCL_MULRND_F32(c_float_0p2,selector3,zero_point2);

				// c[0, 48-63]
				SCL_MULRND_F32(c_float_0p3,selector4,zero_point3);

				// c[1, 0-15]
				SCL_MULRND_F32(c_float_1p0,selector1,zero_point0);

				// c[1, 16-31]
				SCL_MULRND_F32(c_float_1p1,selector2,zero_point1);

				// c[1, 32-47]
				SCL_MULRND_F32(c_float_1p2,selector3,zero_point2);

				// c[1, 48-63]
				SCL_MULRND_F32(c_float_1p3,selector4,zero_point3);
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

				// c[0, 48-63]
				SCL_MULRND_F32(c_float_0p3,selector1,zero_point0);

				// c[1, 0-15]
				SCL_MULRND_F32(c_float_1p0,selector2,zero_point1);

				// c[1, 16-31]
				SCL_MULRND_F32(c_float_1p1,selector2,zero_point1);

				// c[1, 32-47]
				SCL_MULRND_F32(c_float_1p2,selector2,zero_point1);

				// c[1, 48-63]
				SCL_MULRND_F32(c_float_1p3,selector2,zero_point1);
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_ADD_2x64_OPS:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
					( ( post_ops_list_temp->stor_type == NONE ) &&
					  ( post_ops_attr.c_stor_type == BF16 ) );

			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();

			// Even though different registers are used for scalar in column and
			// row major case, all those registers will contain the same value.
			if ( post_ops_list_temp->scale_factor_len == 1 )
			{
				scl_fctr1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
						_mm512_maskz_loadu_ps( k0,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					scl_fctr2 =
						_mm512_maskz_loadu_ps( k1,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					scl_fctr3 =
						_mm512_maskz_loadu_ps( k2,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					scl_fctr4 =
						_mm512_maskz_loadu_ps( k3,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 3 * 16 ) );
				}
				else
				{
					scl_fctr1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 0 ) );
					scl_fctr2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 1 ) );
				}
			}

			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1);
				}
			}
			else
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1);
				}
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_MUL_2x64_OPS:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
					( ( post_ops_list_temp->stor_type == NONE ) &&
					  ( post_ops_attr.c_stor_type == BF16 ) );

			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();

			// Even though different registers are used for scalar in column and
			// row major case, all those registers will contain the same value.
			if ( post_ops_list_temp->scale_factor_len == 1 )
			{
				scl_fctr1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
						_mm512_maskz_loadu_ps( k0,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					scl_fctr2 =
						_mm512_maskz_loadu_ps( k1,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					scl_fctr3 =
						_mm512_maskz_loadu_ps( k2,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					scl_fctr4 =
						_mm512_maskz_loadu_ps( k3,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 3 * 16 ) );
				}
				else
				{
					scl_fctr1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 0 ) );
					scl_fctr2 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 1 ) );
				}
			}

			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1);
				}
			}
			else
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,1);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr2,scl_fctr2,scl_fctr2,scl_fctr2,1);
				}
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SWISH_2x64_OPS:
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

			// c[0, 48-63]
			SWISH_F32_AVX512_DEF(c_float_0p3, selector1, al_in, r, r2, z, dn, ex_out);

			// c[1, 0-15]
			SWISH_F32_AVX512_DEF(c_float_1p0, selector1, al_in, r, r2, z, dn, ex_out);

			// c[1, 16-31]
			SWISH_F32_AVX512_DEF(c_float_1p1, selector1, al_in, r, r2, z, dn, ex_out);

			// c[1, 32-47]
			SWISH_F32_AVX512_DEF(c_float_1p2, selector1, al_in, r, r2, z, dn, ex_out);

			// c[1, 48-63]
			SWISH_F32_AVX512_DEF(c_float_1p3, selector1, al_in, r, r2, z, dn, ex_out);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_TANH_2x64_OPS:
		{
			__m512 dn, z, x, r2, r;
			__m512i q;

			// c[0, 0-15]
			TANHF_AVX512(c_float_0p0, r, r2, x, z, dn,  q)

			// c[0, 16-31]
			TANHF_AVX512(c_float_0p1, r, r2, x, z, dn, q)

			// c[0, 32-47]
			TANHF_AVX512(c_float_0p2, r, r2, x, z, dn, q)

			// c[0, 48-63]
			TANHF_AVX512(c_float_0p3, r, r2, x, z, dn, q)

			// c[1, 0-15]
			TANHF_AVX512(c_float_1p0, r, r2, x, z, dn, q)

			// c[1, 16-31]
			TANHF_AVX512(c_float_1p1, r, r2, x, z, dn, q)

			// c[1, 32-47]
			TANHF_AVX512(c_float_1p2, r, r2, x, z, dn, q)

			// c[1, 48-63]
			TANHF_AVX512(c_float_1p3, r, r2, x, z, dn, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SIGMOID_2x64_OPS:
		{
			__m512 al_in, r, r2, z, dn;
			__m512i ex_out;

			// c[0, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_0p0, al_in, r, r2, z, dn, ex_out);

			// c[0, 16-31]
			SIGMOID_F32_AVX512_DEF(c_float_0p1, al_in, r, r2, z, dn, ex_out);

			// c[0, 32-47]
			SIGMOID_F32_AVX512_DEF(c_float_0p2, al_in, r, r2, z, dn, ex_out);

			// c[0, 48-63]
			SIGMOID_F32_AVX512_DEF(c_float_0p3, al_in, r, r2, z, dn, ex_out);

			// c[1, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_1p0, al_in, r, r2, z, dn, ex_out);

			// c[1, 16-31]
			SIGMOID_F32_AVX512_DEF(c_float_1p1, al_in, r, r2, z, dn, ex_out);

			// c[1, 32-47]
			SIGMOID_F32_AVX512_DEF(c_float_1p2, al_in, r, r2, z, dn, ex_out);

			// c[1, 48-63]
			SIGMOID_F32_AVX512_DEF(c_float_1p3, al_in, r, r2, z, dn, ex_out);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_2x64_OPS_DISABLE:
		;

		// Case where the output C matrix is bf16 (downscaled) and this is the
		// final write for a given block within C.
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			// Actually the b matrix is of type bfloat16. However
			// in order to reuse this kernel for f32, the output
			// matrix type in kernel function signature is set to
			// f32 irrespective of original output matrix type.
			bfloat16* b_q = ( bfloat16* )b;
			dim_t ir = 0;

			// Store the results in downscaled type (bf16 instead of float).
			// c[0, 0-15]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_0p0,k0,0,0);
			// c[0, 16-31]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_0p1,k1,0,16);
			// c[0, 32-47]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_0p2,k2,0,32);
			// c[0, 48-63]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_0p3,k3,0,48);

			// c[1, 0-15]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_1p0,k0,1,0);
			// c[1, 16-31]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_1p1,k1,1,16);
			// c[1, 32-47]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_1p2,k2,1,32);
			// c[1, 48-63]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_1p3,k3,1,48);
		}
		// Case where the output C matrix is float
		else
		{
			// Store the results.
			// c[0,0-15]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) + 
				( cs_b * ( jr + 0 ) ), k0, c_float_0p0 );
			// c[0,16-31]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) + 
				( cs_b * ( jr + 16 ) ), k1, c_float_0p1 );
			// c[0,32-47]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) + 
				( cs_b * ( jr + 32 ) ), k2, c_float_0p2 );
			// c[0,48-63]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) + 
				( cs_b * ( jr + 48 ) ), k3, c_float_0p3 );

			// c[1,0-15]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) + 
				( cs_b * ( jr + 0 ) ), k0, c_float_1p0 );
			// c[1,16-31]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) + 
				( cs_b * ( jr + 16 ) ), k1, c_float_1p1 );
			// c[1,32-47]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) + 
				( cs_b * ( jr + 32 ) ), k2, c_float_1p2 );
			// c[1,48-63]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 1 ) ) + 
				( cs_b * ( jr + 48 ) ), k3, c_float_1p3 );
		}

		post_ops_attr.post_op_c_j += NR_L;
	}
}

LPGEMM_ELTWISE_OPS_M_FRINGE_KERNEL(bfloat16,float,bf16of32_1x64)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_1x64_OPS_DISABLE,
						  &&POST_OPS_BIAS_1x64_OPS,
						  &&POST_OPS_RELU_1x64_OPS,
						  &&POST_OPS_RELU_SCALE_1x64_OPS,
						  &&POST_OPS_GELU_TANH_1x64_OPS,
						  &&POST_OPS_GELU_ERF_1x64_OPS,
						  &&POST_OPS_CLIP_1x64_OPS,
						  &&POST_OPS_DOWNSCALE_1x64_OPS,
						  &&POST_OPS_MATRIX_ADD_1x64_OPS,
						  &&POST_OPS_SWISH_1x64_OPS,
						  &&POST_OPS_MATRIX_MUL_1x64_OPS,
						  &&POST_OPS_TANH_1x64_OPS,
						  &&POST_OPS_SIGMOID_1x64_OPS
						};
	dim_t NR = 64;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();
	__m512 c_float_0p1 = _mm512_setzero_ps();
	__m512 c_float_0p2 = _mm512_setzero_ps();
	__m512 c_float_0p3 = _mm512_setzero_ps();

	__m512 selector1 = _mm512_setzero_ps();
	__m512 selector2 = _mm512_setzero_ps();
	__m512 selector3 = _mm512_setzero_ps();
	__m512 selector4 = _mm512_setzero_ps();

	__mmask16 k0 = 0xFFFF, k1 = 0xFFFF, k2 = 0xFFFF, k3 = 0xFFFF;

	dim_t NR_L = NR;
	for( dim_t jr = 0; jr < n0; jr += NR_L )
	{
		dim_t n_left = n0 - jr;
		NR_L = bli_min( NR_L, ( n_left >> 4 ) << 4 );
		if( NR_L == 0 ) { NR_L = 16; }

		dim_t nr0 = bli_min( n0 - jr, NR_L );
		if( nr0 == 64 )
		{
			// all masks are already set.
			// Nothing to modify.
		}
		else if( nr0 == 48 )
		{
			k3 = 0x0;
		}
		else if( nr0 == 32 )
		{
			k2 = k3 = 0x0;
		}
		else if( nr0 == 16 )
		{
			k1 = k2 = k3 = 0;
		}
		else if( nr0 < 16 )
		{
			k0 = (0xFFFF >> (16 - (nr0 & 0x0F)));
			k1 = k2 = k3 = 0;
		}

		// 1stx64 block.
		c_float_0p0 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k0, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 0 ) ) ) );
		c_float_0p1 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k1, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 16 ) ) ) );
		c_float_0p2 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k2, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 32 ) ) ) );
		c_float_0p3 = CVT_BF16_F32_INT_SHIFT(_mm256_maskz_loadu_epi16( k3, \
			a + ( rs_a * ( 0 ) ) + ( cs_a * ( jr + 48 ) ) ) );

		// Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_1x64_OPS:
		{
			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
				 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
				if ( post_ops_list_temp->stor_type == BF16 )
				{
					BF16_F32_BIAS_LOAD(selector1, k0, 0);
					BF16_F32_BIAS_LOAD(selector2, k1, 1);
					BF16_F32_BIAS_LOAD(selector3, k2, 2);
					BF16_F32_BIAS_LOAD(selector4, k3, 3);
				}
				else
				{
					selector1 =
					_mm512_maskz_loadu_ps( k0,
						( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					selector2 =
					_mm512_maskz_loadu_ps( k1,
						( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					selector3 =
					_mm512_maskz_loadu_ps( k2,
						( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					selector4 =
					_mm512_maskz_loadu_ps( k3,
						( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_j + ( 3 * 16 ) );
				}

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[0, 16-31]
				c_float_0p1 = _mm512_add_ps( selector2, c_float_0p1 );

				// c[0,32-47]
				c_float_0p2 = _mm512_add_ps( selector3, c_float_0p2 );

				// c[0,48-63]
				c_float_0p3 = _mm512_add_ps( selector4, c_float_0p3 );
			}
			else
			{
				// If original output was columns major, then by the time
				// kernel sees it, the matrix would be accessed as if it were
				// transposed. Due to this the bias array will be accessed by
				// the ic index, and each bias element corresponds to an
				// entire row of the transposed output array, instead of an
				// entire column.
				if ( post_ops_list_temp->stor_type == BF16 )
				{
					__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
					BF16_F32_BIAS_BCAST(selector1, bias_mask, 0);
				}
				else
				{
					selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_i + 0 ) );
				}

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[0, 16-31]
				c_float_0p1 = _mm512_add_ps( selector1, c_float_0p1 );

				// c[0,32-47]
				c_float_0p2 = _mm512_add_ps( selector1, c_float_0p2 );

				// c[0,48-63]
				c_float_0p3 = _mm512_add_ps( selector1, c_float_0p3 );
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_1x64_OPS:
		{
			selector1 = _mm512_setzero_ps();

			// c[0,0-15]
			c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_max_ps( selector1, c_float_0p1 );

			// c[0,32-47]
			c_float_0p2 = _mm512_max_ps( selector1, c_float_0p2 );

			// c[0,48-63]
			c_float_0p3 = _mm512_max_ps( selector1, c_float_0p3 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_1x64_OPS:
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

			// c[0, 48-63]
			RELU_SCALE_OP_F32_AVX512(c_float_0p3)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_TANH_1x64_OPS:
		{
			__m512 dn, z, x, r2, r, x_tanh;
			__m512i q;

			// c[0, 0-15]
			GELU_TANH_F32_AVX512(c_float_0p0, r, r2, x, z, dn, x_tanh, q)

			// c[0, 16-31]
			GELU_TANH_F32_AVX512(c_float_0p1, r, r2, x, z, dn, x_tanh, q)

			// c[0, 32-47]
			GELU_TANH_F32_AVX512(c_float_0p2, r, r2, x, z, dn, x_tanh, q)

			// c[0, 48-63]
			GELU_TANH_F32_AVX512(c_float_0p3, r, r2, x, z, dn, x_tanh, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_ERF_1x64_OPS:
		{
			__m512 x, r, x_erf;

			// c[0, 0-15]
			GELU_ERF_F32_AVX512(c_float_0p0, r, x, x_erf)

			// c[0, 16-31]
			GELU_ERF_F32_AVX512(c_float_0p1, r, x, x_erf)

			// c[0, 32-47]
			GELU_ERF_F32_AVX512(c_float_0p2, r, x, x_erf)

			// c[0, 48-63]
			GELU_ERF_F32_AVX512(c_float_0p3, r, x, x_erf)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_CLIP_1x64_OPS:
		{
			__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
			__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

			// c[0, 0-15]
			CLIP_F32_AVX512(c_float_0p0, min, max)

			// c[0, 16-31]
			CLIP_F32_AVX512(c_float_0p1, min, max)

			// c[0, 32-47]
			CLIP_F32_AVX512(c_float_0p2, min, max)

			// c[0, 48-63]
			CLIP_F32_AVX512(c_float_0p3, min, max)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_DOWNSCALE_1x64_OPS:
		{
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
						_mm512_maskz_loadu_ps( k0,
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					selector2 =
						_mm512_maskz_loadu_ps( k1,
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					selector3 =
						_mm512_maskz_loadu_ps( k2,
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					selector4 =
						_mm512_maskz_loadu_ps( k3,
							( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 3 * 16 ) );
				}

				if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
				{
					zero_point0 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_loadu_epi16( k0,
						( ( bfloat16* )post_ops_list_temp->op_args1 ) +
						post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
					zero_point1 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_loadu_epi16( k1,
						( ( bfloat16* )post_ops_list_temp->op_args1 ) +
						post_ops_attr.post_op_c_j + ( 1 * 16 ) ) );
					zero_point2 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_loadu_epi16( k2,
						( ( bfloat16* )post_ops_list_temp->op_args1 ) +
						post_ops_attr.post_op_c_j + ( 2 * 16 ) ) );
					zero_point3 = CVT_BF16_F32_INT_SHIFT(
						_mm256_maskz_loadu_epi16( k3,
						( ( bfloat16* )post_ops_list_temp->op_args1 ) +
						post_ops_attr.post_op_c_j + ( 3 * 16 ) ) );
				}

				// c[0, 0-15]
				SCL_MULRND_F32(c_float_0p0,selector1,zero_point0);

				// c[0, 16-31]
				SCL_MULRND_F32(c_float_0p1,selector2,zero_point1);

				// c[0, 32-47]
				SCL_MULRND_F32(c_float_0p2,selector3,zero_point2);

				// c[0, 48-63]
				SCL_MULRND_F32(c_float_0p3,selector4,zero_point3);
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

				// c[0, 48-63]
				SCL_MULRND_F32(c_float_0p3,selector1,zero_point0);
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_ADD_1x64_OPS:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
					( ( post_ops_list_temp->stor_type == NONE ) &&
					  ( post_ops_attr.c_stor_type == BF16 ) );

			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();

			// Even though different registers are used for scalar in column and
			// row major case, all those registers will contain the same value.
			if ( post_ops_list_temp->scale_factor_len == 1 )
			{
				scl_fctr1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
						_mm512_maskz_loadu_ps( k0,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					scl_fctr2 =
						_mm512_maskz_loadu_ps( k1,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					scl_fctr3 =
						_mm512_maskz_loadu_ps( k2,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					scl_fctr4 =
						_mm512_maskz_loadu_ps( k3,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 3 * 16 ) );
				}
				else
				{
					scl_fctr1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 0 ) );
				}
			}

			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0);
				}
			}
			else
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0);
				}
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_MUL_1x64_OPS:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
					( ( post_ops_list_temp->stor_type == NONE ) &&
					  ( post_ops_attr.c_stor_type == BF16 ) );

			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();

			// Even though different registers are used for scalar in column and
			// row major case, all those registers will contain the same value.
			if ( post_ops_list_temp->scale_factor_len == 1 )
			{
				scl_fctr1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
						_mm512_maskz_loadu_ps( k0,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					scl_fctr2 =
						_mm512_maskz_loadu_ps( k1,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					scl_fctr3 =
						_mm512_maskz_loadu_ps( k2,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					scl_fctr4 =
						_mm512_maskz_loadu_ps( k3,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 3 * 16 ) );
				}
				else
				{
					scl_fctr1 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 0 ) );
				}
			}

			// It is expected the post-op matrix arg has the same storage
			// order as the output C matrix.
			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					BF16_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0);
				}
			}
			else
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr2,scl_fctr3,scl_fctr4,0);
				}
				else
				{
					// c[0:0-15,16-31,32-47,48-63]
					F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,\
						selector1,selector2,selector3,selector4,\
						scl_fctr1,scl_fctr1,scl_fctr1,scl_fctr1,0);
				}
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SWISH_1x64_OPS:
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

			// c[0, 48-63]
			SWISH_F32_AVX512_DEF(c_float_0p3, selector1, al_in, r, r2, z, dn, ex_out);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_TANH_1x64_OPS:
		{
			__m512 dn, z, x, r2, r;
			__m512i q;

			// c[0, 0-15]
			TANHF_AVX512(c_float_0p0, r, r2, x, z, dn,  q)

			// c[0, 16-31]
			TANHF_AVX512(c_float_0p1, r, r2, x, z, dn, q)

			// c[0, 32-47]
			TANHF_AVX512(c_float_0p2, r, r2, x, z, dn, q)

			// c[0, 48-63]
			TANHF_AVX512(c_float_0p3, r, r2, x, z, dn, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SIGMOID_1x64_OPS:
		{
			__m512 al_in, r, r2, z, dn;
			__m512i ex_out;

			// c[0, 0-15]
			SIGMOID_F32_AVX512_DEF(c_float_0p0, al_in, r, r2, z, dn, ex_out);

			// c[0, 16-31]
			SIGMOID_F32_AVX512_DEF(c_float_0p1, al_in, r, r2, z, dn, ex_out);

			// c[0, 32-47]
			SIGMOID_F32_AVX512_DEF(c_float_0p2, al_in, r, r2, z, dn, ex_out);

			// c[0, 48-63]
			SIGMOID_F32_AVX512_DEF(c_float_0p3, al_in, r, r2, z, dn, ex_out);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_1x64_OPS_DISABLE:
		;

		// Case where the output C matrix is bf16 (downscaled) and this is the
		// final write for a given block within C.
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			// Actually the b matrix is of type bfloat16. However
			// in order to reuse this kernel for f32, the output
			// matrix type in kernel function signature is set to
			// f32 irrespective of original output matrix type.
			bfloat16* b_q = ( bfloat16* )b;
			dim_t ir = 0;

			// Store the results in downscaled type (bf16 instead of float).
			// c[0, 0-15]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_0p0,k0,0,0);
			// c[0, 16-31]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_0p1,k1,0,16);
			// c[0, 32-47]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_0p2,k2,0,32);
			// c[0, 48-63]
			CVT_STORE_F32_BF16_POST_OPS_MASK(c_float_0p3,k3,0,48);
		}
		// Case where the output C matrix is float
		else
		{
			// Store the results.
			// c[0,0-15]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) + 
				( cs_b * ( jr + 0 ) ), k0, c_float_0p0 );
			// c[0,16-31]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) + 
				( cs_b * ( jr + 16 ) ), k1, c_float_0p1 );
			// c[0,32-47]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) + 
				( cs_b * ( jr + 32 ) ), k2, c_float_0p2 );
			// c[0,48-63]
			_mm512_mask_storeu_ps( b + ( rs_b * ( 0 ) ) + 
				( cs_b * ( jr + 48 ) ), k3, c_float_0p3 );
		}

		post_ops_attr.post_op_c_j += NR_L;
	}
}

#endif //LPGEMM_BF16_JIT
#endif
