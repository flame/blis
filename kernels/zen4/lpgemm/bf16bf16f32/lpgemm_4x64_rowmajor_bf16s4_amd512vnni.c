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

#include <immintrin.h>
#include "blis.h"

#ifdef BLIS_ADDON_LPGEMM

#ifndef LPGEMM_BF16_JIT

#include "lpgemm_f32_kern_macros.h"
#include "../int4_utils_avx512.h"


#define CVT_INT8_F32_SCAL_16( in, idx, scale_reg) \
    (_mm512_mul_ps( \
      _mm512_cvtepi32_ps( \
       _mm512_cvtepi8_epi32( \
        _mm512_extracti32x4_epi32( in, idx ) ) ), scale_reg ) )

#define CVT_INT8_F32_SCAL_8( in, idx, scale_reg) \
    (_mm512_mul_ps( \
      _mm512_cvtepi32_ps( \
       _mm512_cvtepi8_epi32( \
        _mm256_extracti32x4_epi32( in, idx ) ) ), scale_reg ) )

// 4x48 bf16s4 kernel
LPGEMM_M_FRINGE_KERN1( bfloat16, int8_t, float, bf16s4f32of32_4x48 )
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

	dim_t pre_op_off = post_ops_attr.pre_op_off;

	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int16_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;
	__m512bh b1;
	__m512bh b2;

	__m256i b0_s4;
	__m128i b1_s4;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	__m512i shift_idx_64;
	MULTISHIFT_32BIT_8_INT4_IDX_64ELEM(shift_idx_64);
	__m512i sign_comp = _mm512_set1_epi8(0x08);

	__m256i shift_idx_32;
	MULTISHIFT_32BIT_8_INT4_IDX_32ELEM(shift_idx_32);
	__m256i sign_comp_32 = _mm256_set1_epi8( 0x08 );

	bool signed_upscale = true;

	/* regs to store intermediate int8 values */
	__m512i b0_s8;
	__m256i b1_s8;

	/* Regs to store F32 scale values */
	__m512 scale0, scale1, scale2, scale3, scale4, scale5;
	/* Reg to store masks to interleave scale factor */
	__m512i mask_scale1, mask_scale2;

	mask_scale1 = _mm512_set_epi32( 0x17, 0x07, 0x16, 0x06, 0x15, 0x05, 0x14,
	                                0x04, 0x13, 0x03, 0x12, 0x02, 0x11, 0x01,
	                                0x10, 0x00 );

	mask_scale2 = _mm512_set_epi32( 0x1F, 0x0F, 0x1E, 0x0E, 0x1D, 0x0D, 0x1C,
	                                0x0C, 0x1B, 0x0B, 0x1A, 0x0A, 0x19, 0x09,
	                                0x18, 0x08);

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

    if( post_ops_attr.pre_op_scale_factor_len > 1 )
	{
		// load and interleave scale factor vectors
		scale0 = _mm512_loadu_ps( (float*)( post_ops_attr.pre_op_scale_factor ) +
		                                    pre_op_off);
		scale2 = _mm512_loadu_ps( (float*)( post_ops_attr.pre_op_scale_factor ) +
		                            pre_op_off + 16 );
		scale4 = _mm512_loadu_ps( (float*)( post_ops_attr.pre_op_scale_factor ) +
		                            pre_op_off + 32 );

		scale1 = _mm512_permutex2var_ps( scale0, mask_scale2, scale0 );
		scale0 = _mm512_permutex2var_ps( scale0, mask_scale1, scale0 );
		scale3 = _mm512_permutex2var_ps( scale2, mask_scale2, scale2 );
		scale2 = _mm512_permutex2var_ps( scale2, mask_scale1, scale2 );
		scale5 = _mm512_permutex2var_ps( scale4, mask_scale2, scale4 );
		scale4 = _mm512_permutex2var_ps( scale4, mask_scale1, scale4 );

	}
	else
	{
		scale0 = _mm512_set1_ps( *( ( float* )post_ops_attr.pre_op_scale_factor ) );
		scale1 = _mm512_set1_ps( *( ( float* )post_ops_attr.pre_op_scale_factor ) );
		scale2 = _mm512_set1_ps( *( ( float* )post_ops_attr.pre_op_scale_factor ) );
		scale3 = _mm512_set1_ps( *( ( float* )post_ops_attr.pre_op_scale_factor ) );
		scale4 = _mm512_set1_ps( *( ( float* )post_ops_attr.pre_op_scale_factor ) );
		scale5 = _mm512_set1_ps( *( ( float* )post_ops_attr.pre_op_scale_factor ) );
	}
	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0_s4 = _mm256_loadu_si256( (__m256i const *)( b + ( rs_b * kr ) / 2 ) );


		CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( b0_s4, b0_s8, shift_idx_64, \
		                                    sign_comp, signed_upscale);

		b0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 1, scale1 ),
		                          CVT_INT8_F32_SCAL_16( b0_s8, 0, scale0 ) );

		b1 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 3, scale3 ),
		                          CVT_INT8_F32_SCAL_16( b0_s8, 2, scale2 ) );

		b1_s4 = _mm_loadu_si128( (__m128i const *)( b + ( ( rs_b * kr ) / 2 ) + 32 ) );

		CVT_INT4_TO_INT8_32ELEM_MULTISHIFT( b1_s4, b1_s8, shift_idx_32, \
		                                    sign_comp_32, signed_upscale);

		b2 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_8( b1_s8, 1, scale5 ),
		                          CVT_INT8_F32_SCAL_8( b1_s8, 0, scale4 ) );

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
		b0_s4 = _mm256_loadu_si256( (__m256i const *)( b + ( rs_b * k_full_pieces ) / 2 ) );

		CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( b0_s4, b0_s8, shift_idx_64, \
		                                    sign_comp, signed_upscale);

		b0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 1, scale1 ),
		                          CVT_INT8_F32_SCAL_16( b0_s8, 0, scale0 ) );

		b1 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 3, scale3 ),
		                          CVT_INT8_F32_SCAL_16( b0_s8, 2, scale2 ) );

		b1_s4 = _mm_loadu_si128( (__m128i const *)( b + ( ( rs_b * k_full_pieces ) / 2 ) + 32 ) );

		CVT_INT4_TO_INT8_32ELEM_MULTISHIFT( b1_s4, b1_s8, shift_idx_32, \
		                                    sign_comp_32, signed_upscale);

		b2 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_8( b1_s8, 1, scale5 ),
		                          CVT_INT8_F32_SCAL_8( b1_s8, 0, scale4 ) );

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


// 4x32 bf16s4 kernel
LPGEMM_M_FRINGE_KERN1( bfloat16, int8_t, float, bf16s4f32of32_4x32 )
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

	dim_t pre_op_off = post_ops_attr.pre_op_off;

	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int16_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;
	__m512bh b1;

	__m256i b0_s4;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	__m512i shift_idx_64;
	MULTISHIFT_32BIT_8_INT4_IDX_64ELEM(shift_idx_64);
	__m512i sign_comp = _mm512_set1_epi8(0x08);

	bool signed_upscale = true;

	/* regs to store intermediate int8 values */
	__m512i b0_s8;

	/* Regs to store F32 scale values */
	__m512 scale0, scale1, scale2, scale3;
	/* Reg to store masks to interleave scale factor */
	__m512i mask_scale1, mask_scale2;

	mask_scale1 = _mm512_set_epi32( 0x17, 0x07, 0x16, 0x06, 0x15, 0x05, 0x14,
	                                0x04, 0x13, 0x03, 0x12, 0x02, 0x11, 0x01,
	                                0x10, 0x00 );

	mask_scale2 = _mm512_set_epi32( 0x1F, 0x0F, 0x1E, 0x0E, 0x1D, 0x0D, 0x1C,
	                                0x0C, 0x1B, 0x0B, 0x1A, 0x0A, 0x19, 0x09,
	                                0x18, 0x08);

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();
	__m512 c_float_0p1 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();
	__m512 c_float_1p1 = _mm512_setzero_ps();

	__m512 c_float_2p0 = _mm512_setzero_ps();
	__m512 c_float_2p1 = _mm512_setzero_ps();

	__m512 c_float_3p0 = _mm512_setzero_ps();
	__m512 c_float_3p1 = _mm512_setzero_ps();

	if( post_ops_attr.pre_op_scale_factor_len > 1 )
	{
		// load and interleave scale factor vectors
		scale0 = _mm512_loadu_ps( (float*)( post_ops_attr.pre_op_scale_factor ) +
		                                    pre_op_off);
		scale2 = _mm512_loadu_ps( (float*)( post_ops_attr.pre_op_scale_factor ) +
		                            pre_op_off + 16 );

		scale1 = _mm512_permutex2var_ps( scale0, mask_scale2, scale0 );
		scale0 = _mm512_permutex2var_ps( scale0, mask_scale1, scale0 );
		scale3 = _mm512_permutex2var_ps( scale2, mask_scale2, scale2 );
		scale2 = _mm512_permutex2var_ps( scale2, mask_scale1, scale2 );
	}
	else
	{
		scale0 = _mm512_set1_ps( *( ( float* )post_ops_attr.pre_op_scale_factor ) );
		scale1 = _mm512_set1_ps( *( ( float* )post_ops_attr.pre_op_scale_factor ) );
		scale2 = _mm512_set1_ps( *( ( float* )post_ops_attr.pre_op_scale_factor ) );
		scale3 = _mm512_set1_ps( *( ( float* )post_ops_attr.pre_op_scale_factor ) );
	}

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0_s4 = _mm256_loadu_si256( (__m256i const *)( b + ( rs_b * kr ) / 2 ) );


		CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( b0_s4, b0_s8, shift_idx_64, \
		                                    sign_comp, signed_upscale);

		b0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 1, scale1 ),
		                          CVT_INT8_F32_SCAL_16( b0_s8, 0, scale0 ) );

		b1 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 3, scale3 ),
		                          CVT_INT8_F32_SCAL_16( b0_s8, 2, scale2 ) );


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
		b0_s4 = _mm256_loadu_si256( (__m256i const *)( b + ( rs_b * k_full_pieces ) / 2 ) );


		CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( b0_s4, b0_s8, shift_idx_64, \
		                                    sign_comp, signed_upscale);

		b0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 1, scale1 ),
		                          CVT_INT8_F32_SCAL_16( b0_s8, 0, scale0 ) );

		b1 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 3, scale3 ),
		                          CVT_INT8_F32_SCAL_16( b0_s8, 2, scale2 ) );

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

// 4x16 bf16s4 kernel
LPGEMM_M_FRINGE_KERN1( bfloat16, int8_t, float, bf16s4f32of32_4x16 )
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

	dim_t pre_op_off = post_ops_attr.pre_op_off;

	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int16_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;

	__m128i b0_s4;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	__m256i shift_idx_32;
	MULTISHIFT_32BIT_8_INT4_IDX_32ELEM(shift_idx_32);
	__m256i sign_comp_32 = _mm256_set1_epi8( 0x08 );

	bool signed_upscale = true;

	/* regs to store intermediate int8 values */
	__m256i b0_s8;

	/* Regs to store F32 scale values */
	__m512 scale0, scale1;
	/* Reg to store masks to interleave scale factor */
	__m512i mask_scale1, mask_scale2;

	mask_scale1 = _mm512_set_epi32( 0x17, 0x07, 0x16, 0x06, 0x15, 0x05, 0x14,
	                                0x04, 0x13, 0x03, 0x12, 0x02, 0x11, 0x01,
	                                0x10, 0x00 );

	mask_scale2 = _mm512_set_epi32( 0x1F, 0x0F, 0x1E, 0x0E, 0x1D, 0x0D, 0x1C,
	                                0x0C, 0x1B, 0x0B, 0x1A, 0x0A, 0x19, 0x09,
	                                0x18, 0x08);

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();

	__m512 c_float_2p0 = _mm512_setzero_ps();

	__m512 c_float_3p0 = _mm512_setzero_ps();

	if( post_ops_attr.pre_op_scale_factor_len > 1 )
	{
		// load and interleave scale factor vectors
		scale0 = _mm512_loadu_ps( (float*)( post_ops_attr.pre_op_scale_factor ) +
		                                    pre_op_off);

		scale1 = _mm512_permutex2var_ps( scale0, mask_scale2, scale0 );
		scale0 = _mm512_permutex2var_ps( scale0, mask_scale1, scale0 );

	}
	else
	{
		scale0 = _mm512_set1_ps( *( ( float* )post_ops_attr.pre_op_scale_factor ) );
		scale1 = _mm512_set1_ps( *( ( float* )post_ops_attr.pre_op_scale_factor ) );
	}

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0_s4 = _mm_loadu_si128( (__m128i const *)( b + ( ( rs_b * kr ) / 2 ) ) );

        CVT_INT4_TO_INT8_32ELEM_MULTISHIFT( b0_s4, b0_s8, shift_idx_32, \
		                                    sign_comp_32, signed_upscale);

        b0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_8( b0_s8, 1, scale1 ),
		                          CVT_INT8_F32_SCAL_8( b0_s8, 0, scale0 ) );

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
		b0_s4 = _mm_loadu_si128( (__m128i const *)( b + ( ( rs_b * k_full_pieces ) / 2 ) ) );

        CVT_INT4_TO_INT8_32ELEM_MULTISHIFT( b0_s4, b0_s8, shift_idx_32, \
		                                    sign_comp_32, signed_upscale);

        b0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_8( b0_s8, 1, scale1 ),
		                          CVT_INT8_F32_SCAL_8( b0_s8, 0, scale0 ) );

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

// 4xlt16 bf16s4 fringe kernel
LPGEMM_N_LT_NR0_FRINGE_KERN1( bfloat16, int8_t, float, bf16s4f32of32_4xlt16 )
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

	dim_t pre_op_off = post_ops_attr.pre_op_off;

	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int16_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;

	__m128i b0_s4;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	__m256i shift_idx_32;
	MULTISHIFT_32BIT_8_INT4_IDX_32ELEM(shift_idx_32);
	__m256i sign_comp_32 = _mm256_set1_epi8( 0x08 );

	bool signed_upscale = true;

	/* regs to store intermediate int8 values */
	__m256i b0_s8;

	/* Regs to store F32 scale values */
	__m512 scale0, scale1;
	/* Reg to store masks to interleave scale factor */
	__m512i mask_scale1, mask_scale2;

	mask_scale1 = _mm512_set_epi32( 0x17, 0x07, 0x16, 0x06, 0x15, 0x05, 0x14,
	                                0x04, 0x13, 0x03, 0x12, 0x02, 0x11, 0x01,
	                                0x10, 0x00 );

	mask_scale2 = _mm512_set_epi32( 0x1F, 0x0F, 0x1E, 0x0E, 0x1D, 0x0D, 0x1C,
	                                0x0C, 0x1B, 0x0B, 0x1A, 0x0A, 0x19, 0x09,
	                                0x18, 0x08);

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();

	__m512 c_float_2p0 = _mm512_setzero_ps();

	__m512 c_float_3p0 = _mm512_setzero_ps();

	if( post_ops_attr.pre_op_scale_factor_len > 1 )
	{
		// load and interleave scale factor vectors
		scale0 = _mm512_loadu_ps( (float*)( post_ops_attr.pre_op_scale_factor ) +
		                                    pre_op_off);

		scale1 = _mm512_permutex2var_ps( scale0, mask_scale2, scale0 );
		scale0 = _mm512_permutex2var_ps( scale0, mask_scale1, scale0 );

	}
	else
	{
		scale0 = _mm512_set1_ps( *( ( float* )post_ops_attr.pre_op_scale_factor ) );
		scale1 = _mm512_set1_ps( *( ( float* )post_ops_attr.pre_op_scale_factor ) );
	}

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0_s4 = _mm_loadu_si128( (__m128i const *)( b + ( ( rs_b * kr ) / 2 ) ) );

		CVT_INT4_TO_INT8_32ELEM_MULTISHIFT( b0_s4, b0_s8, shift_idx_32, \
		                                    sign_comp_32, signed_upscale);

		b0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_8( b0_s8, 1, scale1 ),
		                          CVT_INT8_F32_SCAL_8( b0_s8, 0, scale0 ) );

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
		b0_s4 = _mm_loadu_si128( (__m128i const *)( b + ( ( rs_b * k_full_pieces ) / 2 ) ) );

		CVT_INT4_TO_INT8_32ELEM_MULTISHIFT( b0_s4, b0_s8, shift_idx_32, \
		                                    sign_comp_32, signed_upscale);

		b0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_8( b0_s8, 1, scale1 ),
		                          CVT_INT8_F32_SCAL_8( b0_s8, 0, scale0 ) );

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


// 4x64 bf16s4f32 main kernel
LPGEMM_MAIN_KERN1(bfloat16, int8_t, float, bf16s4f32of32_4x64)
{
	static void* post_ops_labels[] =
	                    {
	                      &&POST_OPS_4x64_DISABLE,
	                      &&POST_OPS_BIAS_4x64,
	                      &&POST_OPS_RELU_4x64,
	                      &&POST_OPS_RELU_SCALE_4x64,
	                      &&POST_OPS_GELU_TANH_4x64,
	                      &&POST_OPS_GELU_ERF_4x64,
	                      &&POST_OPS_CLIP_4x64,
	                      &&POST_OPS_DOWNSCALE_4x64,
	                      &&POST_OPS_MATRIX_ADD_4x64,
	                      &&POST_OPS_SWISH_4x64,
	                      &&POST_OPS_MATRIX_MUL_4x64
	                    };


	dim_t pre_op_off = post_ops_attr.pre_op_off;

	dim_t NR = 64;

	if( n0 < NR )
	{
		dim_t n0_rem = n0 % 16;

		// Split dim_to multiple smaller fringe kernels, so as to maximize
		// vectorization. Any n0 < NR(64) can be expressed as n0 = 48 + n`
		// or n0 = 32 + n` or n0 = 16 + n`, where n` < 16.
		dim_t n0_48 = n0 / 48;
		dim_t n0_32 = n0 / 32;
		dim_t n0_16 = n0 / 16;

		// KC when not multiple of 2 will have padding to make it multiple of
		// 2 in packed buffer. Also the k0 cannot be passed as the updated
		// value since A matrix is not packed and requires original k0.
		dim_t k0_updated = k0;
		k0_updated += (k0_updated & 0x1);

		if ( n0_48 == 1 )
		{
			lpgemm_rowvar_bf16s4f32of32_4x48
				(
				 k0,
				 a, rs_a, cs_a,
				 b, ( ( rs_b / 4 ) * 3 ), cs_b,
				 c, rs_c,
				 alpha, beta,
				 post_ops_list, post_ops_attr
				);

			b = b + ( 48 * k0_updated ) / 2; // k0x48 packed contiguosly.
			c = c + 48;
			post_ops_attr.post_op_c_j += 48;
			post_ops_attr.pre_op_off += 48;
		}

		else if ( n0_32 == 1 )
		{
			lpgemm_rowvar_bf16s4f32of32_4x32
				(
				 k0,
				 a, rs_a, cs_a,
				 b, ( ( rs_b / 4 ) * 2 ), cs_b,
				 c, rs_c,
				 alpha, beta,
				 post_ops_list, post_ops_attr
				);

			b = b + ( 32 * k0_updated ) / 2; // k0x32 packed contiguosly.
			c = c + 32;
			post_ops_attr.post_op_c_j += 32;
			post_ops_attr.pre_op_off += 32;
		}

		else if ( n0_16 == 1 )
		{
			lpgemm_rowvar_bf16s4f32of32_4x16
				(
				 k0,
				 a, rs_a, cs_a,
				 b, ( ( rs_b / 4 ) * 1 ), cs_b,
				 c, rs_c,
				 alpha, beta,
				 post_ops_list, post_ops_attr
				);

			b = b + ( 16 * k0_updated ) / 2; // k0x16 packed contiguosly.
			c = c + 16;
			post_ops_attr.post_op_c_j += 16;
			post_ops_attr.pre_op_off += 16;
		}

		if ( n0_rem > 0 )
		{
			lpgemm_rowvar_bf16s4f32of32_4xlt16
				(
				 k0,
				 a, rs_a, cs_a,
				 b, ( ( rs_b / 4 ) * 1 ), cs_b,
				 c, rs_c,
				 alpha, beta, n0_rem,
				 post_ops_list, post_ops_attr
				);

			// No leftover fringe after this point.
		}
		return;
    }

	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int16_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;
	__m512bh b1;
	__m512bh b2;
	__m512bh b3;

	__m256i b0_s4;
	__m256i b1_s4;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;
	__m512bh a_bf16_1;

	__m512i shift_idx_64;
	MULTISHIFT_32BIT_8_INT4_IDX_64ELEM(shift_idx_64);
	__m512i sign_comp = _mm512_set1_epi8(0x08);
	bool signed_upscale = true;

	/* regs to store intermediate int8 values */
	__m512i b0_s8, b1_s8;

	/* Regs to store F32 scale values */
	__m512 scale0, scale1, scale2, scale3, scale4, scale5, scale6, scale7;
	/* Reg to store masks to interleave scale factor */
	__m512i mask_scale1, mask_scale2;

	mask_scale1 = _mm512_set_epi32( 0x17, 0x07, 0x16, 0x06, 0x15, 0x05, 0x14,
	                                0x04, 0x13, 0x03, 0x12, 0x02, 0x11, 0x01,
	                                0x10, 0x00 );

	mask_scale2 = _mm512_set_epi32( 0x1F, 0x0F, 0x1E, 0x0E, 0x1D, 0x0D, 0x1C,
	                                0x0C, 0x1B, 0x0B, 0x1A, 0x0A, 0x19, 0x09,
	                                0x18, 0x08);

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


	if( post_ops_attr.pre_op_scale_factor_len > 1 )
	{
		// load and interleave scale factor vectors
		scale0 = _mm512_loadu_ps( (float*)( post_ops_attr.pre_op_scale_factor ) +
		                                    pre_op_off);
		scale2 = _mm512_loadu_ps( (float*)( post_ops_attr.pre_op_scale_factor ) +
		                            pre_op_off + 16 );
		scale4 = _mm512_loadu_ps( (float*)( post_ops_attr.pre_op_scale_factor ) +
		                            pre_op_off + 32 );
		scale6 = _mm512_loadu_ps( (float*)( post_ops_attr.pre_op_scale_factor ) +
		                             pre_op_off + 48 );

		scale1 = _mm512_permutex2var_ps( scale0, mask_scale2, scale0 );
		scale0 = _mm512_permutex2var_ps( scale0, mask_scale1, scale0 );
		scale3 = _mm512_permutex2var_ps( scale2, mask_scale2, scale2 );
		scale2 = _mm512_permutex2var_ps( scale2, mask_scale1, scale2 );
		scale5 = _mm512_permutex2var_ps( scale4, mask_scale2, scale4 );
		scale4 = _mm512_permutex2var_ps( scale4, mask_scale1, scale4 );
		scale7 = _mm512_permutex2var_ps( scale6, mask_scale2, scale6 );
		scale6 = _mm512_permutex2var_ps( scale6, mask_scale1, scale6 );

	}
	else
	{
		scale0 = _mm512_set1_ps( *( ( float* )post_ops_attr.pre_op_scale_factor ) );
		scale1 = _mm512_set1_ps( *( ( float* )post_ops_attr.pre_op_scale_factor ) );
		scale2 = _mm512_set1_ps( *( ( float* )post_ops_attr.pre_op_scale_factor ) );
		scale3 = _mm512_set1_ps( *( ( float* )post_ops_attr.pre_op_scale_factor ) );
		scale4 = _mm512_set1_ps( *( ( float* )post_ops_attr.pre_op_scale_factor ) );
		scale5 = _mm512_set1_ps( *( ( float* )post_ops_attr.pre_op_scale_factor ) );
		scale6 = _mm512_set1_ps( *( ( float* )post_ops_attr.pre_op_scale_factor ) );
		scale7 = _mm512_set1_ps( *( ( float* )post_ops_attr.pre_op_scale_factor ) );
	}

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		// Broadcast a[0,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

		b0_s4 = _mm256_loadu_si256( (__m256i const *)( b + ( rs_b * kr ) / 2 ) );


		CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( b0_s4, b0_s8, shift_idx_64, \
		                                    sign_comp, signed_upscale);

		b0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 1, scale1 ),
		                          CVT_INT8_F32_SCAL_16( b0_s8, 0, scale0 ) );

		b1 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 3, scale3 ),
		                          CVT_INT8_F32_SCAL_16( b0_s8, 2, scale2 ) );

		b1_s4 = _mm256_loadu_si256( (__m256i const *)( b + ( ( rs_b * kr ) / 2 ) + 32 ) );

		CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( b1_s4, b1_s8, shift_idx_64, \
		                                    sign_comp, signed_upscale);

		b2 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b1_s8, 1, scale5 ),
		                          CVT_INT8_F32_SCAL_16( b1_s8, 0, scale4 ) );

		b3 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b1_s8, 3, scale7 ),
		                          CVT_INT8_F32_SCAL_16( b1_s8, 2, scale6 ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-63] = a[0,kr:kr+4]*b[kr:kr+4,0-63]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );

		// Broadcast a[1,kr:kr+2].
		a_bf16_1 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 1 ) + ( cs_a * kr ) ) );

		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
		c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );
		c_float_0p3 = _mm512_dpbf16_ps( c_float_0p3, a_bf16_0, b3 );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-63] = a[1,kr:kr+2]*b[kr:kr+2,0-63]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_1, b0 );

		// Broadcast a[2,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 2 ) + ( cs_a * kr ) ) );

		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_1, b1 );
		c_float_1p2 = _mm512_dpbf16_ps( c_float_1p2, a_bf16_1, b2 );
		c_float_1p3 = _mm512_dpbf16_ps( c_float_1p3, a_bf16_1, b3 );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-63] = a[2,kr:kr+2]*b[kr:kr+2,0-63]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );

		// Broadcast a[3,kr:kr+2].
		a_bf16_1 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 3 ) + ( cs_a * kr ) ) );

		c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );
		c_float_2p2 = _mm512_dpbf16_ps( c_float_2p2, a_bf16_0, b2 );
		c_float_2p3 = _mm512_dpbf16_ps( c_float_2p3, a_bf16_0, b3 );

		// Perform column direction mat-mul with k = 2.
		// c[3,0-63] = a[3,kr:kr+2]*b[kr:kr+2,0-63]
		c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_1, b0 );
		c_float_3p1 = _mm512_dpbf16_ps( c_float_3p1, a_bf16_1, b1 );
		c_float_3p2 = _mm512_dpbf16_ps( c_float_3p2, a_bf16_1, b2 );
		c_float_3p3 = _mm512_dpbf16_ps( c_float_3p3, a_bf16_1, b3 );
	}

	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		// Broadcast a[0,kr:kr+2].
		a_kfringe_buf = *( a + (rs_a * 0) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		b0_s4 = _mm256_loadu_si256( (__m256i const *)( b + ( rs_b * k_full_pieces ) / 2 ) );

		CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( b0_s4, b0_s8, shift_idx_64, \
		                                    sign_comp, signed_upscale);

		b0 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 1, scale1 ),
		                          CVT_INT8_F32_SCAL_16( b0_s8, 0, scale0 ) );

		b1 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b0_s8, 3, scale3 ),
		                          CVT_INT8_F32_SCAL_16( b0_s8, 2, scale2 ) );

		b1_s4 = _mm256_loadu_si256( (__m256i const *)( b + ( ( rs_b * k_full_pieces ) / 2 ) + 32 ) );

		CVT_INT4_TO_INT8_64ELEM_MULTISHIFT( b1_s4, b1_s8, shift_idx_64, \
		                                    sign_comp, signed_upscale);

		b2 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b1_s8, 1, scale5 ),
		                          CVT_INT8_F32_SCAL_16( b1_s8, 0, scale4 ) );

		b3 = _mm512_cvtne2ps_pbh( CVT_INT8_F32_SCAL_16( b1_s8, 3, scale7 ),
		                          CVT_INT8_F32_SCAL_16( b1_s8, 2, scale6 ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-63] = a[0,kr:kr+2]*b[kr:kr+2,0-63]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );

		// Broadcast a[1,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 1) + (cs_a * ( k_full_pieces )));
		a_bf16_1 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
		c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );
		c_float_0p3 = _mm512_dpbf16_ps( c_float_0p3, a_bf16_0, b3 );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-63] = a[1,kr:kr+2]*b[kr:kr+2,0-63]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_1, b0 );

		// Broadcast a[2,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 2) + (cs_a * ( k_full_pieces )));
		a_bf16_0 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_1, b1 );
		c_float_1p2 = _mm512_dpbf16_ps( c_float_1p2, a_bf16_1, b2 );
		c_float_1p3 = _mm512_dpbf16_ps( c_float_1p3, a_bf16_1, b3 );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-63] = a[2,kr:kr+2]*b[kr:kr+2,0-63]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );

		// Broadcast a[3,kr:kr+2].
		a_kfringe_buf = *(a + (rs_a * 3) + (cs_a * ( k_full_pieces )));
		a_bf16_1 = (__m512bh)_mm512_set1_epi16( a_kfringe_buf );

		c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );
		c_float_2p2 = _mm512_dpbf16_ps( c_float_2p2, a_bf16_0, b2 );
		c_float_2p3 = _mm512_dpbf16_ps( c_float_2p3, a_bf16_0, b3 );

		// Perform column direction mat-mul with k = 2.
		// c[3,0-63] = a[3,kr:kr+2]*b[kr:kr+2,0-63]
		c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_1, b0 );
		c_float_3p1 = _mm512_dpbf16_ps( c_float_3p1, a_bf16_1, b1 );
		c_float_3p2 = _mm512_dpbf16_ps( c_float_3p2, a_bf16_1, b2 );
		c_float_3p3 = _mm512_dpbf16_ps( c_float_3p3, a_bf16_1, b3 );
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
		c_float_0p3 = _mm512_mul_ps( selector1, c_float_0p3 );

		c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );
		c_float_1p1 = _mm512_mul_ps( selector1, c_float_1p1 );
		c_float_1p2 = _mm512_mul_ps( selector1, c_float_1p2 );
		c_float_1p3 = _mm512_mul_ps( selector1, c_float_1p3 );

		c_float_2p0 = _mm512_mul_ps( selector1, c_float_2p0 );
		c_float_2p1 = _mm512_mul_ps( selector1, c_float_2p1 );
		c_float_2p2 = _mm512_mul_ps( selector1, c_float_2p2 );
		c_float_2p3 = _mm512_mul_ps( selector1, c_float_2p3 );

		c_float_3p0 = _mm512_mul_ps( selector1, c_float_3p0 );
		c_float_3p1 = _mm512_mul_ps( selector1, c_float_3p1 );
		c_float_3p2 = _mm512_mul_ps( selector1, c_float_3p2 );
		c_float_3p3 = _mm512_mul_ps( selector1, c_float_3p3 );
	}

	// Scale C by beta.
	if ( beta != 0 )
	{
		// For the downscaled api (C-bf16), the output C matrix values
		// needs to be upscaled to float to be used for beta scale.
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
				( post_ops_attr.is_first_k == TRUE ) )
		{
			// c[0,0-15]
			BF16_F32_BETA_OP(c_float_0p0,0,0,0,selector1,selector2)

			// c[0, 16-31]
			BF16_F32_BETA_OP(c_float_0p1,0,0,1,selector1,selector2)

			// c[0,32-47]
			BF16_F32_BETA_OP(c_float_0p2,0,0,2,selector1,selector2)

			// c[0,48-63]
			BF16_F32_BETA_OP(c_float_0p3,0,0,3,selector1,selector2)

			// c[1,0-15]
			BF16_F32_BETA_OP(c_float_1p0,0,1,0,selector1,selector2)

			// c[1,16-31]
			BF16_F32_BETA_OP(c_float_1p1,0,1,1,selector1,selector2)

			// c[1,32-47]
			BF16_F32_BETA_OP(c_float_1p2,0,1,2,selector1,selector2)

			// c[1,48-63]
			BF16_F32_BETA_OP(c_float_1p3,0,1,3,selector1,selector2)

			// c[2,0-15]
			BF16_F32_BETA_OP(c_float_2p0,0,2,0,selector1,selector2)

			// c[2,16-31]
			BF16_F32_BETA_OP(c_float_2p1,0,2,1,selector1,selector2)

			// c[2,32-47]
			BF16_F32_BETA_OP(c_float_2p2,0,2,2,selector1,selector2)

			// c[2,48-63]
			BF16_F32_BETA_OP(c_float_2p3,0,2,3,selector1,selector2)

			// c[3,0-15]
			BF16_F32_BETA_OP(c_float_3p0,0,3,0,selector1,selector2)

			// c[3,16-31]
			BF16_F32_BETA_OP(c_float_3p1,0,3,1,selector1,selector2)

			// c[3,32-47]
			BF16_F32_BETA_OP(c_float_3p2,0,3,2,selector1,selector2)

			// c[0,48-63]
			BF16_F32_BETA_OP(c_float_3p3,0,3,3,selector1,selector2)

		}
		else
		{
			// c[0,0-15]
			F32_F32_BETA_OP(c_float_0p0,0,0,0,selector1,selector2)

			// c[0, 16-31]
			F32_F32_BETA_OP(c_float_0p1,0,0,1,selector1,selector2)

			// c[0,32-47]
			F32_F32_BETA_OP(c_float_0p2,0,0,2,selector1,selector2)

			// c[0,48-63]
			F32_F32_BETA_OP(c_float_0p3,0,0,3,selector1,selector2)

			// c[1,0-15]
			F32_F32_BETA_OP(c_float_1p0,0,1,0,selector1,selector2)

			// c[1,16-31]
			F32_F32_BETA_OP(c_float_1p1,0,1,1,selector1,selector2)

			// c[1,32-47]
			F32_F32_BETA_OP(c_float_1p2,0,1,2,selector1,selector2)

			// c[1,48-63]
			F32_F32_BETA_OP(c_float_1p3,0,1,3,selector1,selector2)

			// c[2,0-15]
			F32_F32_BETA_OP(c_float_2p0,0,2,0,selector1,selector2)

			// c[2,16-31]
			F32_F32_BETA_OP(c_float_2p1,0,2,1,selector1,selector2)

			// c[2,32-47]
			F32_F32_BETA_OP(c_float_2p2,0,2,2,selector1,selector2)

			// c[2,48-63]
			F32_F32_BETA_OP(c_float_2p3,0,2,3,selector1,selector2)

			// c[3,0-15]
			F32_F32_BETA_OP(c_float_3p0,0,3,0,selector1,selector2)

			// c[3,16-31]
			F32_F32_BETA_OP(c_float_3p1,0,3,1,selector1,selector2)

			// c[3,32-47]
			F32_F32_BETA_OP(c_float_3p2,0,3,2,selector1,selector2)

			// c[0,48-63]
			F32_F32_BETA_OP(c_float_3p3,0,3,3,selector1,selector2)
		}
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_4x64:
	{
		__m512 selector3;
		__m512 selector4;

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			if ( post_ops_attr.c_stor_type == BF16 )
			{
				__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
				BF16_F32_BIAS_LOAD(selector1, bias_mask, 0);
				BF16_F32_BIAS_LOAD(selector2, bias_mask, 1);
				BF16_F32_BIAS_LOAD(selector3, bias_mask, 2);
				BF16_F32_BIAS_LOAD(selector4, bias_mask, 3);
			}
			else
			{
				selector1 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j );
				selector2 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				selector3 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 2 * 16 ) );
				selector4 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
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
POST_OPS_RELU_4x64:
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
POST_OPS_RELU_SCALE_4x64:
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
POST_OPS_GELU_TANH_4x64:
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
POST_OPS_GELU_ERF_4x64:
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
POST_OPS_CLIP_4x64:
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
POST_OPS_DOWNSCALE_4x64:
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
				selector4 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 3 * 16 ) );
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
				zero_point3 = CVT_BF16_F32_INT_SHIFT(
							_mm256_maskz_loadu_epi16( zp_mask,
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
POST_OPS_MATRIX_ADD_4x64:
	{
		__m512 selector3;
		__m512 selector4;
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47,48-63]
			BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,0);

			// c[1:0-15,16-31,32-47,48-63]
			BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,1);

			// c[2:0-15,16-31,32-47,48-63]
			BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,2);

			// c[3:0-15,16-31,32-47,48-63]
			BF16_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,3);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,0);

			// c[1:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,1);

			// c[2:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,2);

			// c[3:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_ADD_4COL(selector1,selector2,selector3,selector4,3);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_MUL_4x64:
	{
		__m512 selector3;
		__m512 selector4;
		dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
		if ( post_ops_attr.c_stor_type == BF16 )
		{
			bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47,48-63]
			BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,0);

			// c[1:0-15,16-31,32-47,48-63]
			BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,1);

			// c[2:0-15,16-31,32-47,48-63]
			BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,2);

			// c[3:0-15,16-31,32-47,48-63]
			BF16_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,3);
		}
		else
		{
			float* matptr = ( float* )post_ops_list_temp->op_args1;

			// c[0:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,0);

			// c[1:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,1);

			// c[2:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,2);

			// c[3:0-15,16-31,32-47,48-63]
			F32_F32_MATRIX_MUL_4COL(selector1,selector2,selector3,selector4,3);
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_SWISH_4x64:
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

POST_OPS_4x64_DISABLE:
	;

	// Case where the output C matrix is bf16 (downscaled) and this is the
	// final write for a given block within C.
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

		// c[0, 48-63]
		CVT_STORE_F32_BF16_MASK(c_float_0p3,0,3);

		// c[1, 0-15]
		CVT_STORE_F32_BF16_MASK(c_float_1p0,1,0);

		// c[1, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_1p1,1,1);

		// c[1, 32-47]
		CVT_STORE_F32_BF16_MASK(c_float_1p2,1,2);

		// c[1, 48-63]
		CVT_STORE_F32_BF16_MASK(c_float_1p3,1,3);

		// c[2, 0-15]
		CVT_STORE_F32_BF16_MASK(c_float_2p0,2,0);

		// c[2, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_2p1,2,1);

		// c[2, 32-47]
		CVT_STORE_F32_BF16_MASK(c_float_2p2,2,2);

		// c[2, 48-63]
		CVT_STORE_F32_BF16_MASK(c_float_2p3,2,3);

		// c[3, 0-15]
		CVT_STORE_F32_BF16_MASK(c_float_3p0,3,0);

		// c[3, 16-31]
		CVT_STORE_F32_BF16_MASK(c_float_3p1,3,1);

		// c[3, 32-47]
		CVT_STORE_F32_BF16_MASK(c_float_3p2,3,2);

		// c[3, 48-63]
		CVT_STORE_F32_BF16_MASK(c_float_3p3,3,3);
	}

	// Case where the output C matrix is float
	else
	{
		// Store the results.
		// c[0,0-15]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 0*16 ), c_float_0p0 );

		// c[0, 16-31]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 1*16 ), c_float_0p1 );

		// c[0,32-47]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 2*16 ), c_float_0p2 );

		// c[0,48-63]
		_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 3*16 ), c_float_0p3 );

		// c[1,0-15]
		_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 0*16 ), c_float_1p0 );

		// c[1,16-31]
		_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 1*16 ), c_float_1p1 );

		// c[1,32-47]
		_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 2*16 ), c_float_1p2 );

		// c[1,48-63]
		_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 3*16 ), c_float_1p3 );

		// c[2,0-15]
		_mm512_storeu_ps( c + ( rs_c * 2 ) + ( 0*16 ), c_float_2p0 );

		// c[2,16-31]
		_mm512_storeu_ps( c + ( rs_c * 2 ) + ( 1*16 ), c_float_2p1 );

		// c[2,32-47]
		_mm512_storeu_ps( c + ( rs_c * 2 ) + ( 2*16 ), c_float_2p2 );

		// c[2,48-63]
		_mm512_storeu_ps( c + ( rs_c * 2 ) + ( 3*16 ), c_float_2p3 );

		// c[3,0-15]
		_mm512_storeu_ps( c + ( rs_c * 3 ) + ( 0*16 ), c_float_3p0 );

		// c[3,16-31]
		_mm512_storeu_ps( c + ( rs_c * 3 ) + ( 1*16 ), c_float_3p1 );

		// c[3,32-47]
		_mm512_storeu_ps( c + ( rs_c * 3 ) + ( 2*16 ), c_float_3p2 );

		// c[3,48-63]
		_mm512_storeu_ps( c + ( rs_c * 3 ) + ( 3*16 ), c_float_3p3 );
	}
}

#endif // LPGEMM_BF16_JIT
#endif // BLIS_ADDON_LPGEMM