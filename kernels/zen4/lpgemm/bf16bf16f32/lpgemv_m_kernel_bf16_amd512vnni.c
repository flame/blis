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
#include "immintrin.h"
#include "xmmintrin.h"
#include "blis.h"

#ifdef BLIS_ADDON_LPGEMM

#include "lpgemm_f32_kern_macros.h"


#ifdef LPGEMM_BF16_JIT
LPGEMV_M_EQ1_KERN(bfloat16, bfloat16, float, bf16bf16f32of32)
{}
#else


LPGEMV_M_EQ1_KERN(bfloat16, bfloat16, float, bf16bf16f32of32)
{
	static void* post_ops_labels[] =
	                    {
	                      &&POST_OPS_6x64_DISABLE,
	                      &&POST_OPS_BIAS_6x64,
	                      &&POST_OPS_RELU_6x64,
	                      &&POST_OPS_RELU_SCALE_6x64,
	                      &&POST_OPS_GELU_TANH_6x64,
	                      &&POST_OPS_GELU_ERF_6x64,
	                      &&POST_OPS_CLIP_6x64,
	                      &&POST_OPS_DOWNSCALE_6x64,
	                      &&POST_OPS_MATRIX_ADD_6x64,
	                      &&POST_OPS_SWISH_6x64,
						  &&POST_OPS_MATRIX_MUL_6x64,
						  &&POST_OPS_TANH_6x64,
						  &&POST_OPS_SIGMOID_6x64

	                    };


	// Strides are updated based on matrix packing/reordering.
	const bfloat16 *a_use = NULL;
	const bfloat16 *b_use = NULL;

	lpgemm_post_op_attr post_ops_attr = *(post_op_attr);

	for( dim_t jr = 0; jr < n0; jr += NR )
	{

		float* c_use = c + jr * cs_c;

		dim_t n_left = n0 - jr;

		NR = bli_min( NR, ( n_left >> 4 ) << 4 );

		if( NR == 0 ) NR = 16;

		rs_b = NR * 2;

		dim_t nr0 = bli_min( n0 - jr, NR );

		__mmask16 k1 = 0xFFFF, k2 = 0xFFFF, k3 = 0xFFFF, k4 = 0xFFFF;
		__mmask32 k5 = 0xFFFFFFFF, k6 = 0xFFFFFFFF;
		__mmask32 k7 = 0xFFFFFFFF, k8 = 0xFFFFFFFF;

		if( nr0 == 64 )
		{
			// all masks are already set.
			// Nothing to modify.
		}
		else if( nr0 == 48 )
		{
			k4 = k8 = 0x0;
		}
		else if( nr0 == 32 )
		{
			k3 = k4 = k7 = k8 = 0x0;
		}
		else if( nr0 == 16 )
		{
			k2 = k3 = k4 = k6 = k7 = k8 = 0;
		}
		else if( nr0 < 16 )
		{
			k1 = (0xFFFF >> (16 - (nr0 & 0x0F)));
			k2 = k3 = k4 = k6 = k7 = k8 = 0;
		}

		__m512bh zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;
		__m512 zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14;
		__m512 zmm15, zmm16, zmm17, zmm18, zmm19, zmm20, zmm21;
		__m512 zmm22, zmm23;
		__m512bh zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31;

		// zero the accumulator registers
		ZERO_ACC_ZMM_4_REG(zmm8, zmm9, zmm10, zmm11);
		ZERO_ACC_ZMM_4_REG(zmm12, zmm13, zmm14, zmm15);
		ZERO_ACC_ZMM_4_REG(zmm16, zmm17, zmm18, zmm19);
		ZERO_ACC_ZMM_4_REG(zmm20, zmm21, zmm22, zmm23);

		for (dim_t pc = 0; pc < k; pc += KC)
		{
			dim_t kc0 = bli_min((k - pc), KC);

			// kc0 needs to be a multiple of 2 so that it can be
			// used with dpbf16_ps instruction. Padding is added in
			// cases this condition is not satisfied, and therefore
			// the kc0 offsets used for packed/reordered buffers
			// needs to be updated.
			dim_t kc0_updated = kc0;
			kc0_updated += (kc0_updated & 0x1);

			uint64_t k_iter = kc0 / 8;
			uint64_t k_rem = ( kc0 / 2) % 4;

			// No parallelization in k dim, k always starts at 0.

			// In multi-threaded scenarios, an extra offset into a given
			// packed B panel is required, since the jc loop split can
			// result in per thread start offset inside the panel, instead
			// of panel boundaries.
			b_use = b + ( n_sub_updated * pc ) +
			            ( ( jc_cur_loop_rem + jr ) * kc0_updated ) ;

			a_use = a + pc;

			for (dim_t k = 0; k < k_iter; k++)
			{

				// load first 4x32 tile from row 0-3
				zmm0 = (__m512bh)_mm512_maskz_loadu_epi16( k5, b_use );
				zmm1 = (__m512bh)_mm512_maskz_loadu_epi16( k5, b_use + rs_b );
				zmm2 = (__m512bh)_mm512_maskz_loadu_epi16( k5,
				                                           b_use + 2 * rs_b );
				zmm3 = (__m512bh)_mm512_maskz_loadu_epi16( k5,
				                                           b_use + 3 * rs_b );
				b_use += 32;

				// Broadcast col0-col3 elements of A
				zmm4 = (__m512bh)_mm512_set1_epi32(*( int32_t* )( a_use ) );
				zmm5 = (__m512bh)_mm512_set1_epi32(*( int32_t* )( a_use +
				                                                ( cs_a ) ) );
				zmm6 = (__m512bh)_mm512_set1_epi32(*( int32_t* )( a_use +
				                                                ( cs_a * 2 ) ) );
				zmm7 = (__m512bh)_mm512_set1_epi32(*( int32_t* )( a_use +
				                                                ( cs_a * 3 ) ) );

				// Load second 4x32 tile from row 0-3
				zmm24 = (__m512bh)_mm512_maskz_loadu_epi16 ( k6, b_use );
				zmm25 = (__m512bh)_mm512_maskz_loadu_epi16 ( k6, b_use + rs_b );
				zmm26 = (__m512bh)_mm512_maskz_loadu_epi16 ( k6,
				                                             b_use + 2 * rs_b );
				zmm27 = (__m512bh)_mm512_maskz_loadu_epi16 ( k6,
				                                             b_use + 3 * rs_b );
				b_use += 32;

				zmm8  = _mm512_dpbf16_ps( zmm8, zmm4, zmm0 );
				zmm9  = _mm512_dpbf16_ps( zmm9, zmm5, zmm1 );
				zmm10 = _mm512_dpbf16_ps( zmm10, zmm6, zmm2 );
				zmm11 = _mm512_dpbf16_ps( zmm11, zmm7, zmm3 );

				// load third 4x32 tile from row 0-3
				zmm0 = (__m512bh)_mm512_maskz_loadu_epi16 ( k7, b_use );
				zmm1 = (__m512bh)_mm512_maskz_loadu_epi16 ( k7, b_use + rs_b );
				zmm2 = (__m512bh)_mm512_maskz_loadu_epi16 ( k7,
				                                            b_use + 2 * rs_b );
				zmm3 = (__m512bh)_mm512_maskz_loadu_epi16 ( k7,
				                                            b_use + 3 * rs_b );
				b_use += 32;


				zmm12 = _mm512_dpbf16_ps( zmm12, zmm4, zmm24 );
				zmm13 = _mm512_dpbf16_ps( zmm13, zmm5, zmm25 );
				zmm14 = _mm512_dpbf16_ps( zmm14, zmm6, zmm26 );
				zmm15 = _mm512_dpbf16_ps( zmm15, zmm7, zmm27 );

				// Load fourth 4x32 tile from row 0-3
				zmm28 = (__m512bh)_mm512_maskz_loadu_epi16 ( k8, b_use );
				zmm29 = (__m512bh)_mm512_maskz_loadu_epi16 ( k8, b_use + rs_b );
				zmm30 = (__m512bh)_mm512_maskz_loadu_epi16 ( k8,
				                                             b_use + 2 * rs_b );
				zmm31 = (__m512bh)_mm512_maskz_loadu_epi16 ( k8,
				                                             b_use + 3 * rs_b );


				zmm16 = _mm512_dpbf16_ps( zmm16, zmm4, zmm0 );
				zmm17 = _mm512_dpbf16_ps( zmm17, zmm5, zmm1 );
				zmm18 = _mm512_dpbf16_ps( zmm18, zmm6, zmm2 );
				zmm19 = _mm512_dpbf16_ps( zmm19, zmm7, zmm3 );

				zmm20 = _mm512_dpbf16_ps( zmm20, zmm4, zmm28 );
				zmm21 = _mm512_dpbf16_ps( zmm21, zmm5, zmm29 );
				zmm22 = _mm512_dpbf16_ps( zmm22, zmm6, zmm30 );
				zmm23 = _mm512_dpbf16_ps( zmm23, zmm7, zmm31 );

				b_use -= 96; // move b point back to start of KCXNR
				b_use += (4 * rs_b);
				a_use += 4 * cs_a; // move a pointer to next col

			}

			for (dim_t kr = 0; kr < k_rem; kr++)
			{
				// load 128 elements from a row of B
				zmm0 = (__m512bh)_mm512_maskz_loadu_epi16 ( k5, b_use );
				zmm1 = (__m512bh)_mm512_maskz_loadu_epi16 ( k6,
				                                            b_use + cs_b );
				zmm2 = (__m512bh)_mm512_maskz_loadu_epi16 ( k7,
				                                            b_use + cs_b*2 );
				zmm3 = (__m512bh)_mm512_maskz_loadu_epi16 ( k8,
				                                            b_use + cs_b*3 );

				// Broadcast col0 elements of A
				zmm4 = (__m512bh)_mm512_set1_epi32(*( int32_t* )(a_use ) );

				zmm8  = _mm512_dpbf16_ps( zmm8, zmm4, zmm0  );
				zmm12 = _mm512_dpbf16_ps( zmm12, zmm4, zmm1 );
				zmm16 = _mm512_dpbf16_ps( zmm16, zmm4, zmm2 );
				zmm20 = _mm512_dpbf16_ps( zmm20, zmm4, zmm3 );

				b_use += rs_b;
				a_use += cs_a;
			}
			if( kc0 & 1 )
			{
				// load 128 elements from a row of B
				zmm0 = (__m512bh)_mm512_maskz_loadu_epi16 ( k5, b_use );
				zmm1 = (__m512bh)_mm512_maskz_loadu_epi16 ( k6, b_use + cs_b );
				zmm2 = (__m512bh)_mm512_maskz_loadu_epi16 ( k7,
				                                            b_use + cs_b*2 );
				zmm3 = (__m512bh)_mm512_maskz_loadu_epi16 ( k8,
				                                            b_use + cs_b*3 );

				// Broadcast col0 elements of A
				zmm4 = (__m512bh)_mm512_set1_epi16(*(int16_t*) a_use );

				zmm8  = _mm512_dpbf16_ps( zmm8, zmm4, zmm0  );
				zmm12 = _mm512_dpbf16_ps( zmm12, zmm4, zmm1 );
				zmm16 = _mm512_dpbf16_ps( zmm16, zmm4, zmm2 );
				zmm20 = _mm512_dpbf16_ps( zmm20, zmm4, zmm3 );

			}
		}
		// Sumup k-unroll outputs
		zmm8 = _mm512_add_ps( zmm9, zmm8 );
		zmm10 = _mm512_add_ps(zmm11, zmm10);
		zmm8 = _mm512_add_ps(zmm10, zmm8); // 32 outputs

		zmm12 = _mm512_add_ps(zmm13, zmm12);
		zmm14 = _mm512_add_ps(zmm15, zmm14);
		zmm12 = _mm512_add_ps(zmm14, zmm12); // 32 outputs

		zmm16 = _mm512_add_ps(zmm17, zmm16);
		zmm18 = _mm512_add_ps(zmm19, zmm18);
		zmm16 = _mm512_add_ps(zmm18, zmm16); // 32 outputs

		zmm20 = _mm512_add_ps(zmm21, zmm20);
		zmm22 = _mm512_add_ps(zmm23, zmm22);
		zmm20 = _mm512_add_ps(zmm22, zmm20); // 32 outputs

		__m512 selector1 = _mm512_set1_ps( alpha );
		__m512 selector2 = _mm512_set1_ps( beta );

		//Mulitply A*B output with alpha
		zmm8 = _mm512_mul_ps(selector1, zmm8);
		zmm12 = _mm512_mul_ps(selector1, zmm12);
		zmm16 = _mm512_mul_ps(selector1, zmm16);
		zmm20 = _mm512_mul_ps(selector1, zmm20);

		if (beta != 0)
		{

			// For the downscaled api (C-bf16), the output C matrix values
			// needs to be upscaled to float to be used for beta scale.
			if ( post_ops_attr.buf_downscale != NULL )
			{
				BF16_F32_BETA_OP_NLT16F_MASK( k1, zmm8,  0, 0, selector1, selector2 )
				BF16_F32_BETA_OP_NLT16F_MASK( k2, zmm12, 0, 1, selector1, selector2 )
				BF16_F32_BETA_OP_NLT16F_MASK( k3, zmm16, 0, 2, selector1, selector2 )
				BF16_F32_BETA_OP_NLT16F_MASK( k4, zmm20, 0, 3, selector1, selector2 )
			}
			else
			{
				F32_F32_BETA_OP_NLT16F_MASK( c_use, k1, zmm8,  0, 0, 0, selector1, selector2 )
				F32_F32_BETA_OP_NLT16F_MASK( c_use, k2, zmm12, 0, 0, 1, selector1, selector2 )
				F32_F32_BETA_OP_NLT16F_MASK( c_use, k3, zmm16, 0, 0, 2, selector1, selector2 )
				F32_F32_BETA_OP_NLT16F_MASK( c_use, k4, zmm20, 0, 0, 3, selector1, selector2 )
			}
		}

		post_ops_attr.is_last_k = TRUE;
		lpgemm_post_op *post_ops_list_temp = post_op;
		POST_OP_LABEL_LASTK_SAFE_JUMP

		POST_OPS_BIAS_6x64:
		{
			__m512 selector3;
			__m512 selector4;

			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
				( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
				if ( post_ops_list_temp->stor_type == BF16 )
				{
					BF16_F32_BIAS_LOAD(selector1, k1, 0);
					BF16_F32_BIAS_LOAD(selector2, k2, 1);
					BF16_F32_BIAS_LOAD(selector3, k3, 2);
					BF16_F32_BIAS_LOAD(selector4, k4, 3);
				}
				else
				{
					selector1 =
						_mm512_maskz_loadu_ps( k1,
								( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					selector2 =
						_mm512_maskz_loadu_ps( k2,
								( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					selector3 =
						_mm512_maskz_loadu_ps( k3,
								( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					selector4 =
						_mm512_maskz_loadu_ps( k4,
								( float* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 3 * 16 ) );
				}

				zmm8  = _mm512_add_ps( selector1, zmm8  );
				zmm12 = _mm512_add_ps( selector2, zmm12 );
				zmm16 = _mm512_add_ps( selector3, zmm16 );
				zmm20 = _mm512_add_ps( selector4, zmm20 );
			}
			else
			{
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

				zmm8  = _mm512_add_ps( selector1, zmm8  );
				zmm12 = _mm512_add_ps( selector1, zmm12 );
				zmm16 = _mm512_add_ps( selector1, zmm16 );
				zmm20 = _mm512_add_ps( selector1, zmm20 );
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_RELU_6x64:
		{
			selector1 = _mm512_setzero_ps();

			zmm8  = _mm512_max_ps( selector1, zmm8  );
			zmm12 = _mm512_max_ps( selector1, zmm12 );
			zmm16 = _mm512_max_ps( selector1, zmm16 );
			zmm20 = _mm512_max_ps( selector1, zmm20 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}

		POST_OPS_RELU_SCALE_6x64:
		{
			selector1 = _mm512_setzero_ps();
			selector2 =
			    _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

			__mmask16 relu_cmp_mask;

			RELU_SCALE_OP_F32_AVX512( zmm8  )
			RELU_SCALE_OP_F32_AVX512( zmm12 )
			RELU_SCALE_OP_F32_AVX512( zmm16 )
			RELU_SCALE_OP_F32_AVX512( zmm20 )

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}

		POST_OPS_GELU_TANH_6x64:
		{
			__m512 dn, z, x, r2, r, x_tanh;
			__m512i q;

			GELU_TANH_F32_AVX512( zmm8,  r, r2, x, z, dn, x_tanh, q )
			GELU_TANH_F32_AVX512( zmm12, r, r2, x, z, dn, x_tanh, q )
			GELU_TANH_F32_AVX512( zmm16, r, r2, x, z, dn, x_tanh, q )
			GELU_TANH_F32_AVX512( zmm20, r, r2, x, z, dn, x_tanh, q )

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}

		POST_OPS_GELU_ERF_6x64:
		{
			__m512 x, r, x_erf;

			GELU_ERF_F32_AVX512( zmm8,  r, x, x_erf )
			GELU_ERF_F32_AVX512( zmm12, r, x, x_erf )
			GELU_ERF_F32_AVX512( zmm16, r, x, x_erf )
			GELU_ERF_F32_AVX512( zmm20, r, x, x_erf )

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}

		POST_OPS_CLIP_6x64:
		{
			__m512 min =
			        _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
			__m512 max =
			        _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

			CLIP_F32_AVX512( zmm8,  min, max )
			CLIP_F32_AVX512( zmm12, min, max )
			CLIP_F32_AVX512( zmm16, min, max )
			CLIP_F32_AVX512( zmm20, min, max )

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}

		POST_OPS_DOWNSCALE_6x64:
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
						_mm512_maskz_loadu_ps( k1,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					selector2 =
						_mm512_maskz_loadu_ps( k2,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					selector3 =
						_mm512_maskz_loadu_ps( k3,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					selector4 =
						_mm512_maskz_loadu_ps( k4,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 3 * 16 ) );
				}

				if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
				{
					zero_point0 = CVT_BF16_F32_INT_SHIFT(
								_mm256_maskz_loadu_epi16( k1,
								( ( bfloat16* )post_ops_list_temp->op_args1 ) +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
					zero_point1 = CVT_BF16_F32_INT_SHIFT(
								_mm256_maskz_loadu_epi16( k2,
								( ( bfloat16* )post_ops_list_temp->op_args1 ) +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) ) );
					zero_point2 = CVT_BF16_F32_INT_SHIFT(
								_mm256_maskz_loadu_epi16( k3,
								( ( bfloat16* )post_ops_list_temp->op_args1 ) +
								post_ops_attr.post_op_c_j + ( 2 * 16 ) ) );
					zero_point3 = CVT_BF16_F32_INT_SHIFT(
								_mm256_maskz_loadu_epi16( k4,
								( ( bfloat16* )post_ops_list_temp->op_args1 ) +
								post_ops_attr.post_op_c_j + ( 3 * 16 ) ) );
				}

				// c[0, 0-15]
				SCL_MULRND_F32(zmm8,selector1,zero_point0);

				// c[0, 16-31]
				SCL_MULRND_F32(zmm12,selector2,zero_point1);

				// c[0, 32-47]
				SCL_MULRND_F32(zmm16,selector3,zero_point2);

				// c[0, 48-63]
				SCL_MULRND_F32(zmm20,selector4,zero_point3);
			}
			else
			{
				// If original output was columns major, then by the time
				// kernel sees it, the matrix would be accessed as if it were
				// transposed. Due to this the scale as well as zp array will
				// be accessed by the ic index, and each scale/zp element
				// corresponds to an entire row of the transposed output array,
				// instead of an entire column.
				// Scale/zp len cannot be > 1, since original n = 1 for
				// swapped m to be = 1.

				// c[0, 0-15]
				SCL_MULRND_F32(zmm8,selector1,zero_point0);

				// c[0, 16-31]
				SCL_MULRND_F32(zmm12,selector1,zero_point0);

				// c[0, 32-47]
				SCL_MULRND_F32(zmm16,selector1,zero_point0);

				// c[0, 48-63]
				SCL_MULRND_F32(zmm20,selector1,zero_point0);
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}

		POST_OPS_MATRIX_ADD_6x64:
		{
			__m512 selector3;
			__m512 selector4;

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
			// For column major, if m==1, then it means n=1 and scale_factor_len=1.
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
						_mm512_maskz_loadu_ps( k1,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					scl_fctr2 =
						_mm512_maskz_loadu_ps( k2,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					scl_fctr3 =
						_mm512_maskz_loadu_ps( k3,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					scl_fctr4 =
						_mm512_maskz_loadu_ps( k4,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 3 * 16 ) );
				}
			}

			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					BF16_F32_MATRIX_ADD_LOAD
								( k1, selector1, scl_fctr1, 0, 0 )
					BF16_F32_MATRIX_ADD_LOAD
								( k2, selector2, scl_fctr2, 0, 1 )
					BF16_F32_MATRIX_ADD_LOAD
								( k3, selector3, scl_fctr3, 0, 2 )
					BF16_F32_MATRIX_ADD_LOAD
								( k4, selector4, scl_fctr4, 0, 3 )
				}
				else
				{
					BF16_F32_MATRIX_ADD_LOAD
								( k1, selector1, scl_fctr1, 0, 0 )
					BF16_F32_MATRIX_ADD_LOAD
								( k2, selector2, scl_fctr1, 0, 1 )
					BF16_F32_MATRIX_ADD_LOAD
								( k3, selector3, scl_fctr1, 0, 2 )
					BF16_F32_MATRIX_ADD_LOAD
								( k4, selector4, scl_fctr1, 0, 3 )
				}

				zmm8  = _mm512_add_ps( selector1, zmm8  );
				zmm12 = _mm512_add_ps( selector2, zmm12 );
				zmm16 = _mm512_add_ps( selector3, zmm16 );
				zmm20 = _mm512_add_ps( selector4, zmm20 );
			}
			else
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					F32_F32_MATRIX_ADD_LOAD
								( k1, selector1, scl_fctr1, 0, 0 )
					F32_F32_MATRIX_ADD_LOAD
								( k2, selector2, scl_fctr2, 0, 1 )
					F32_F32_MATRIX_ADD_LOAD
								( k3, selector3, scl_fctr3, 0, 2 )
					F32_F32_MATRIX_ADD_LOAD
								( k4, selector4, scl_fctr4, 0, 3 )
				}
				else
				{
					F32_F32_MATRIX_ADD_LOAD
								( k1, selector1, scl_fctr1, 0, 0 )
					F32_F32_MATRIX_ADD_LOAD
								( k2, selector2, scl_fctr1, 0, 1 )
					F32_F32_MATRIX_ADD_LOAD
								( k3, selector3, scl_fctr1, 0, 2 )
					F32_F32_MATRIX_ADD_LOAD
								( k4, selector4, scl_fctr1, 0, 3 )
				}

				zmm8  = _mm512_add_ps( selector1, zmm8  );
				zmm12 = _mm512_add_ps( selector2, zmm12 );
				zmm16 = _mm512_add_ps( selector3, zmm16 );
				zmm20 = _mm512_add_ps( selector4, zmm20 );
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}

		POST_OPS_MATRIX_MUL_6x64:
		{
			__m512 selector3;
			__m512 selector4;

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
			// For column major, if m==1, then it means n=1 and scale_factor_len=1.
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
						_mm512_maskz_loadu_ps( k1,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					scl_fctr2 =
						_mm512_maskz_loadu_ps( k2,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
					scl_fctr3 =
						_mm512_maskz_loadu_ps( k3,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 2 * 16 ) );
					scl_fctr4 =
						_mm512_maskz_loadu_ps( k4,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 3 * 16 ) );
				}
			}

			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					BF16_F32_MATRIX_MUL_LOAD
								( k1, selector1, scl_fctr1, 0, 0 )
					BF16_F32_MATRIX_MUL_LOAD
								( k2, selector2, scl_fctr2, 0, 1 )
					BF16_F32_MATRIX_MUL_LOAD
								( k3, selector3, scl_fctr3, 0, 2 )
					BF16_F32_MATRIX_MUL_LOAD
								( k4, selector4, scl_fctr4, 0, 3 )
				}
				else
				{
					BF16_F32_MATRIX_MUL_LOAD
								( k1, selector1, scl_fctr1, 0, 0 )
					BF16_F32_MATRIX_MUL_LOAD
								( k2, selector2, scl_fctr1, 0, 1 )
					BF16_F32_MATRIX_MUL_LOAD
								( k3, selector3, scl_fctr1, 0, 2 )
					BF16_F32_MATRIX_MUL_LOAD
								( k4, selector4, scl_fctr1, 0, 3 )
				}

				zmm8  = _mm512_mul_ps( selector1, zmm8  );
				zmm12 = _mm512_mul_ps( selector2, zmm12 );
				zmm16 = _mm512_mul_ps( selector3, zmm16 );
				zmm20 = _mm512_mul_ps( selector4, zmm20 );
			}
			else
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					F32_F32_MATRIX_MUL_LOAD
								( k1, selector1, scl_fctr1, 0, 0 )
					F32_F32_MATRIX_MUL_LOAD
								( k2, selector2, scl_fctr2, 0, 1 )
					F32_F32_MATRIX_MUL_LOAD
								( k3, selector3, scl_fctr3, 0, 2 )
					F32_F32_MATRIX_MUL_LOAD
								( k4, selector4, scl_fctr4, 0, 3 )
				}
				else
				{
					F32_F32_MATRIX_MUL_LOAD
								( k1, selector1, scl_fctr1, 0, 0 )
					F32_F32_MATRIX_MUL_LOAD
								( k2, selector2, scl_fctr1, 0, 1 )
					F32_F32_MATRIX_MUL_LOAD
								( k3, selector3, scl_fctr1, 0, 2 )
					F32_F32_MATRIX_MUL_LOAD
								( k4, selector4, scl_fctr1, 0, 3 )
				}

				zmm8  = _mm512_mul_ps( selector1, zmm8  );
				zmm12 = _mm512_mul_ps( selector2, zmm12 );
				zmm16 = _mm512_mul_ps( selector3, zmm16 );
				zmm20 = _mm512_mul_ps( selector4, zmm20 );
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}

		POST_OPS_SWISH_6x64:
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

			__m512 al_in, r, r2, z, dn;
			__m512i ex_out;

			SWISH_F32_AVX512_DEF( zmm8,  selector1, al_in, r, r2, z, dn, ex_out );
			SWISH_F32_AVX512_DEF( zmm12, selector1, al_in, r, r2, z, dn, ex_out );
			SWISH_F32_AVX512_DEF( zmm16, selector1, al_in, r, r2, z, dn, ex_out );
			SWISH_F32_AVX512_DEF( zmm20, selector1, al_in, r, r2, z, dn, ex_out );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_TANH_6x64:
		{
			__m512 dn, z, x, r2, r;
			__m512i q;

			TANHF_AVX512(zmm8, r, r2, x, z, dn,  q)
			TANHF_AVX512(zmm12, r, r2, x, z, dn, q)
			TANHF_AVX512(zmm16, r, r2, x, z, dn, q)
			TANHF_AVX512(zmm20, r, r2, x, z, dn, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_SIGMOID_6x64:
		{
			__m512 al_in, r, r2, z, dn;
			__m512i ex_out;

			SIGMOID_F32_AVX512_DEF(zmm8, al_in, r, r2, z, dn, ex_out);
			SIGMOID_F32_AVX512_DEF(zmm12, al_in, r, r2, z, dn, ex_out);
			SIGMOID_F32_AVX512_DEF(zmm16, al_in, r, r2, z, dn, ex_out);
			SIGMOID_F32_AVX512_DEF(zmm20, al_in, r, r2, z, dn, ex_out);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_6x64_DISABLE:
		{
			// Case where the output C matrix is bf16 (downscaled)
			// and this is the final write for a given block within C.
			if ( post_ops_attr.buf_downscale != NULL )
			{
				_mm256_mask_storeu_epi16
				(
				( bfloat16* )post_ops_attr.buf_downscale +
				post_ops_attr.post_op_c_j + ( 0 * 16 ),
				k1, (__m256i) _mm512_cvtneps_pbh( zmm8 )
				);

				_mm256_mask_storeu_epi16
				(
				( bfloat16* )post_ops_attr.buf_downscale +
				post_ops_attr.post_op_c_j + ( 1 * 16 ),
				k2, (__m256i) _mm512_cvtneps_pbh( zmm12 )
				);

				_mm256_mask_storeu_epi16
				(
				( bfloat16* )post_ops_attr.buf_downscale +
				post_ops_attr.post_op_c_j + ( 2 * 16 ),
				k3, (__m256i) _mm512_cvtneps_pbh( zmm16 )
				);

				_mm256_mask_storeu_epi16
				(
				( bfloat16* )post_ops_attr.buf_downscale +
				post_ops_attr.post_op_c_j + ( 3 * 16 ),
				k4, (__m256i) _mm512_cvtneps_pbh( zmm20 )
				);
			}
			else
			{
				// Store the results.
				_mm512_mask_storeu_ps( c_use + ( 0*16 ), k1, zmm8 );
				_mm512_mask_storeu_ps( c_use + ( 1*16 ), k2, zmm12 );
				_mm512_mask_storeu_ps( c_use + ( 2*16 ), k3, zmm16 );
				_mm512_mask_storeu_ps( c_use + ( 3*16 ), k4, zmm20 );
			}
		}

		post_ops_attr.post_op_c_j += nr0;

	} // jr loop
}

#endif //  LPGEMM_BF16_JIT
#endif // BLIS_ADDON_LPGEMM
