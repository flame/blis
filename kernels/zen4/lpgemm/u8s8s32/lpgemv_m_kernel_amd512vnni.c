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
#include "immintrin.h"
#include "xmmintrin.h"
#include "blis.h"

#ifdef BLIS_ADDON_LPGEMM

#include "lpgemm_s32_kern_macros.h"
#include "lpgemm_s32_memcpy_macros.h"

LPGEMV_M_EQ1_KERN(uint8_t, int8_t, int32_t, u8s8s32os32)
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
	              &&POST_OPS_SWISH_6x64
	            };

	const uint8_t *a_use = NULL;
	const int8_t  *b_use = NULL;

	lpgemm_post_op_attr post_ops_attr = *(post_op_attr);

	for( dim_t jr = 0; jr < n0; jr += NR )
	{
		NR = bli_min( 64, ( ( n0 - jr ) / 16 ) * 16 );

		if( NR == 0 ) NR = 16;

		rs_b = NR * 4;
		dim_t nr0 = bli_min( n0 - jr, NR );

		int32_t* c_use = c + jr * cs_c;

		__mmask16 k1 = 0xFFFF, k2 = 0xFFFF, k3 = 0xFFFF, k4 = 0xFFFF;
		__mmask32 k5 = 0xFFFFFFFF, k6 = 0xFFFFFFFF;
		__mmask32  k7 = 0xFFFFFFFF, k8 = 0xFFFFFFFF;


		if( nr0 == 64 )
		{

		}
		if( nr0 == 48 )
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



		__m512i zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;
		__m512i zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14;
		__m512i zmm15, zmm16, zmm17, zmm18, zmm19, zmm20, zmm21;
		__m512i zmm22, zmm23, zmm24, zmm25, zmm26, zmm27, zmm28;
		__m512i zmm29, zmm30, zmm31;

		// zero the accumulator registers
		ZERO_ACC_ZMM_4_REG(zmm8, zmm9, zmm10, zmm11);
		ZERO_ACC_ZMM_4_REG(zmm12, zmm13, zmm14, zmm15);
		ZERO_ACC_ZMM_4_REG(zmm16, zmm17, zmm18, zmm19);
		ZERO_ACC_ZMM_4_REG(zmm20, zmm21, zmm22, zmm23);

		for (dim_t pc = 0; pc < k; pc += KC)
		{
			dim_t kc0 = bli_min((k - pc), KC);

			dim_t k_full_pieces = kc0 / 4;
			dim_t k_partial_pieces = kc0 % 4;

			dim_t k_iter = kc0 / 16;
			dim_t k_rem = k_full_pieces % 4;

			dim_t kc0_updated = kc0;

			if ( k_partial_pieces > 0 )
			{
				kc0_updated += ( 4 - k_partial_pieces );
			}

			b_use = b + (n_sub_updated * pc) +
					( ( jc_cur_loop_rem + jr ) * kc0_updated );

			a_use = a + pc;


			for( dim_t kr = 0; kr < k_iter; kr++ )
			{
				// load first 4x64 tile from row 0-3
				zmm0 = _mm512_maskz_loadu_epi16( k5, b_use );
				zmm1 = _mm512_maskz_loadu_epi16( k5, b_use + rs_b );
				zmm2 = _mm512_maskz_loadu_epi16( k5, b_use + 2 * rs_b );
				zmm3 = _mm512_maskz_loadu_epi16( k5, b_use + 3 * rs_b );
				b_use += 64;

				// Broadcast col0-col3 elements of A
				zmm4 = _mm512_set1_epi32( *( int32_t* )( a_use ) );
				zmm5 = _mm512_set1_epi32( *( int32_t* )( a_use + cs_a ) );
				zmm6 = _mm512_set1_epi32( *( int32_t* )( a_use + cs_a * 2 ) );
				zmm7 = _mm512_set1_epi32( *( int32_t* )( a_use + cs_a * 3 ) );

				// Load second 4x64 tile from row 0-3
				zmm24 = _mm512_maskz_loadu_epi16( k6, b_use );
				zmm25 = _mm512_maskz_loadu_epi16( k6, b_use + rs_b );
				zmm26 = _mm512_maskz_loadu_epi16( k6, b_use + 2 * rs_b );
				zmm27 = _mm512_maskz_loadu_epi16( k6, b_use + 3 * rs_b );
				b_use += 64;

				zmm8  = _mm512_dpbusd_epi32( zmm8,  zmm4, zmm0 );
				zmm9  = _mm512_dpbusd_epi32( zmm9,  zmm5, zmm1 );
				zmm10 = _mm512_dpbusd_epi32( zmm10, zmm6, zmm2 );
				zmm11 = _mm512_dpbusd_epi32( zmm11, zmm7, zmm3 );

				// load third 4x64 tile from row 0-3
				zmm0 = _mm512_maskz_loadu_epi16( k7, b_use );
				zmm1 = _mm512_maskz_loadu_epi16( k7, b_use + rs_b );
				zmm2 = _mm512_maskz_loadu_epi16( k7, b_use + 2 * rs_b );
				zmm3 = _mm512_maskz_loadu_epi16( k7, b_use + 3 * rs_b );
				b_use += 64;

				zmm12 = _mm512_dpbusd_epi32( zmm12, zmm4, zmm24 );
				zmm13 = _mm512_dpbusd_epi32( zmm13, zmm5, zmm25 );
				zmm14 = _mm512_dpbusd_epi32( zmm14, zmm6, zmm26 );
				zmm15 = _mm512_dpbusd_epi32( zmm15, zmm7, zmm27 );

				// load third 4x64 tile from row 0-3
				zmm28 = _mm512_maskz_loadu_epi16( k8, b_use );
				zmm29 = _mm512_maskz_loadu_epi16( k8, b_use + rs_b );
				zmm30 = _mm512_maskz_loadu_epi16( k8, b_use + 2 * rs_b );
				zmm31 = _mm512_maskz_loadu_epi16( k8, b_use + 3 * rs_b );

				zmm16 = _mm512_dpbusd_epi32( zmm16, zmm4, zmm0 );
				zmm17 = _mm512_dpbusd_epi32( zmm17, zmm5, zmm1 );
				zmm18 = _mm512_dpbusd_epi32( zmm18, zmm6, zmm2 );
				zmm19 = _mm512_dpbusd_epi32( zmm19, zmm7, zmm3 );

				zmm20 = _mm512_dpbusd_epi32( zmm20, zmm4, zmm28 );
				zmm21 = _mm512_dpbusd_epi32( zmm21, zmm5, zmm29 );
				zmm22 = _mm512_dpbusd_epi32( zmm22, zmm6, zmm30 );
				zmm23 = _mm512_dpbusd_epi32( zmm23, zmm7, zmm31 );

				b_use -= 192; // move b point back to start of KCXNR
				b_use += ( 4 * rs_b );
				a_use += 4 * cs_a; // move a pointer to next col
			}
			for( dim_t kr = 0; kr < k_rem; kr++ )
			{
				// load first 4x64 tile from row 0-3
				zmm0 = _mm512_maskz_loadu_epi16( k5, b_use );
				zmm1 = _mm512_maskz_loadu_epi16( k6, b_use + cs_b );
				zmm2 = _mm512_maskz_loadu_epi16( k7, b_use + 2 * cs_b );
				zmm3 = _mm512_maskz_loadu_epi16( k8, b_use + 3 * cs_b );

				// Broadcast col0 elements of A
				zmm4 = _mm512_set1_epi32( *( int32_t* )( a_use ) );

				zmm8  = _mm512_dpbusd_epi32( zmm8,  zmm4, zmm0 );
				zmm12 = _mm512_dpbusd_epi32( zmm12, zmm4, zmm1 );
				zmm16 = _mm512_dpbusd_epi32( zmm16, zmm4, zmm2 );
				zmm20 = _mm512_dpbusd_epi32( zmm20, zmm4, zmm3 );

				b_use += rs_b;
				a_use += cs_a; // move a pointer to next col
			}
			if( k_partial_pieces > 0 )
			{
				__m128i a_kfringe_buf;
				__mmask16 load_mask =
						_cvtu32_mask16( 0xFFFF >> ( 16 - k_partial_pieces ) );

				// load first 4x64 tile from row 0-3
				zmm0 = _mm512_maskz_loadu_epi16( k5, b_use );

				// Broadcast a[0,kr:kr+4].
				a_kfringe_buf = _mm_maskz_loadu_epi8( load_mask, a_use );
				zmm4 = _mm512_broadcastd_epi32( a_kfringe_buf );

				zmm1 = _mm512_maskz_loadu_epi16( k6, b_use + cs_b );
				zmm2 = _mm512_maskz_loadu_epi16( k7, b_use + 2 * cs_b );
				zmm3 = _mm512_maskz_loadu_epi16( k8, b_use + 3 * cs_b );

				zmm8  = _mm512_dpbusd_epi32( zmm8,  zmm4, zmm0 );
				zmm12 = _mm512_dpbusd_epi32( zmm12, zmm4, zmm1 );
				zmm16 = _mm512_dpbusd_epi32( zmm16, zmm4, zmm2 );
				zmm20 = _mm512_dpbusd_epi32( zmm20, zmm4, zmm3 );

			}

		}

		// Sumup k-unroll outputs
		zmm8 = _mm512_add_epi32( zmm9, zmm8 );
		zmm10 = _mm512_add_epi32(zmm11, zmm10);
		zmm8 = _mm512_add_epi32(zmm10, zmm8); // 64 outputs

		zmm12 = _mm512_add_epi32(zmm13, zmm12);
		zmm14 = _mm512_add_epi32(zmm15, zmm14);
		zmm12 = _mm512_add_epi32(zmm14, zmm12); // 64 outputs

		zmm16 = _mm512_add_epi32(zmm17, zmm16);
		zmm18 = _mm512_add_epi32(zmm19, zmm18);
		zmm16 = _mm512_add_epi32(zmm18, zmm16); // 64 outputs

		zmm20 = _mm512_add_epi32(zmm21, zmm20);
		zmm22 = _mm512_add_epi32(zmm23, zmm22);
		zmm20 = _mm512_add_epi32(zmm22, zmm20); // 64 outputs


		__m512i selector1 = _mm512_set1_epi32( alpha );
		__m512i selector2 = _mm512_set1_epi32( beta );

		__m512i selector3 = _mm512_setzero_epi32();
		__m512i selector4 = _mm512_setzero_epi32();

		//Mulitply A*B output with alpha
		zmm8 = _mm512_mullo_epi32(selector1, zmm8);
		zmm12 = _mm512_mullo_epi32(selector1, zmm12);
		zmm16 = _mm512_mullo_epi32(selector1, zmm16);
		zmm20 = _mm512_mullo_epi32(selector1, zmm20);

		if (beta != 0)
		{
			// For the downscaled api (C-s8), the output C matrix values
			// needs to be upscaled to s32 to be used for beta scale.
			if ( post_ops_attr.buf_downscale != NULL )
			{
				S8_S32_BETA_OP_NLT16F_MASK( k1, zmm8,  0, 0,
				                            selector1, selector2 )
				S8_S32_BETA_OP_NLT16F_MASK( k2, zmm12, 0, 1,
				                            selector1, selector2 )
				S8_S32_BETA_OP_NLT16F_MASK( k3, zmm16, 0, 2,
				                            selector1, selector2 )
				S8_S32_BETA_OP_NLT16F_MASK( k4, zmm20, 0, 3,
				                            selector1, selector2 )
			}
			else
			{
				S32_S32_BETA_OP_NLT16F_MASK( c_use, k1, zmm8,  0, 0, 0,
				                             selector1, selector2 )
				S32_S32_BETA_OP_NLT16F_MASK( c_use, k2, zmm12, 0, 0, 1,
				                             selector1, selector2 )
				S32_S32_BETA_OP_NLT16F_MASK( c_use, k3, zmm16, 0, 0, 2,
				                             selector1, selector2 )
				S32_S32_BETA_OP_NLT16F_MASK( c_use, k4, zmm20, 0, 0, 3,
				                             selector1, selector2 )
			}
		}

		post_ops_attr.is_last_k = TRUE;
		lpgemm_post_op *post_ops_list_temp = post_op;
		POST_OP_LABEL_LASTK_SAFE_JUMP

		POST_OPS_BIAS_6x64:
		{
			selector1 =
				_mm512_loadu_si512( ( int32_t* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
			selector2 =
				_mm512_loadu_si512( ( int32_t* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
			selector3 =
				_mm512_loadu_si512( ( int32_t* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 2 * 16 ) );
			selector4 =
				_mm512_loadu_si512( ( int32_t* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 3 * 16 ) );

			zmm8  = _mm512_add_epi32( selector1, zmm8 );
			zmm12 = _mm512_add_epi32( selector2, zmm12 );
			zmm16 = _mm512_add_epi32( selector3, zmm16 );
			zmm20 = _mm512_add_epi32( selector4, zmm20 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_RELU_6x64:
		{
			selector1 = _mm512_setzero_epi32();

			zmm8  = _mm512_max_epi32( selector1, zmm8 );
			zmm12 = _mm512_max_epi32( selector1, zmm12 );
			zmm16 = _mm512_max_epi32( selector1, zmm16 );
			zmm20 = _mm512_max_epi32( selector1, zmm20 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_RELU_SCALE_6x64:
		{
			selector1 = _mm512_setzero_epi32();
			selector2 =
			  _mm512_set1_epi32( *( (int32_t*)post_ops_list_temp->op_args2 ) );

			__mmask16 relu_cmp_mask;

			RELU_SCALE_OP_S32_AVX512( zmm8 )
			RELU_SCALE_OP_S32_AVX512( zmm12 )
			RELU_SCALE_OP_S32_AVX512( zmm16 )
			RELU_SCALE_OP_S32_AVX512( zmm20 )

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_GELU_TANH_6x64:
		{
			__m512 dn, z, x, r2, r, y, x_tanh;

			GELU_TANH_S32_AVX512( zmm8,  y, r, r2, x,
			                      z, dn, x_tanh, selector1 )
			GELU_TANH_S32_AVX512( zmm12, y, r, r2, x,
			                      z, dn, x_tanh, selector1 )
			GELU_TANH_S32_AVX512( zmm16, y, r, r2, x,
			                      z, dn, x_tanh, selector1 )
			GELU_TANH_S32_AVX512( zmm20, y, r, r2, x,
			                      z, dn, x_tanh, selector1 )

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_GELU_ERF_6x64:
		{
			__m512 x, r, y, x_erf;

			GELU_ERF_S32_AVX512( zmm8,  y, r, x, x_erf )
			GELU_ERF_S32_AVX512( zmm12, y, r, x, x_erf )
			GELU_ERF_S32_AVX512( zmm16, y, r, x, x_erf )
			GELU_ERF_S32_AVX512( zmm20, y, r, x, x_erf )

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR

		}
		POST_OPS_CLIP_6x64:
		{
			__m512i min = _mm512_set1_epi32(
							*( int32_t* )post_ops_list_temp->op_args2 );
			__m512i max = _mm512_set1_epi32(
							*( int32_t* )post_ops_list_temp->op_args3 );

			CLIP_S32_AVX512( zmm8,  min, max )
			CLIP_S32_AVX512( zmm12, min, max )
			CLIP_S32_AVX512( zmm16, min, max )
			CLIP_S32_AVX512( zmm20, min, max )

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_DOWNSCALE_6x64:
		{
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 =
				  _mm512_loadu_si512( (float*)post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				selector2 =
				  _mm512_loadu_si512( (float*)post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				selector3 =
				  _mm512_loadu_si512( (float*)post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 2 * 16 ) );
				selector4 =
				  _mm512_loadu_si512( (float*)post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 3 * 16 ) );
			}
			else if ( post_ops_list_temp->scale_factor_len == 1 )
			{
				selector1 = ( __m512i )_mm512_set1_ps(
							*( ( float* )post_ops_list_temp->scale_factor ) );
				selector2 = ( __m512i )_mm512_set1_ps(
							*( ( float* )post_ops_list_temp->scale_factor ) );
				selector3 = ( __m512i )_mm512_set1_ps(
							*( ( float* )post_ops_list_temp->scale_factor ) );
				selector4 = ( __m512i )_mm512_set1_ps(
							*( ( float* )post_ops_list_temp->scale_factor ) );
			}

			// Need to ensure sse not used to avoid avx512 -> sse transition.
			__m128i zero_point0 = _mm512_castsi512_si128(
				                        _mm512_setzero_si512() );
			__m128i zero_point1 = _mm512_castsi512_si128(
				                        _mm512_setzero_si512() );
			__m128i zero_point2 = _mm512_castsi512_si128(
				                        _mm512_setzero_si512() );
			__m128i zero_point3 = _mm512_castsi512_si128(
				                        _mm512_setzero_si512() );

			// int8_t zero point value.
			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point0 = _mm_loadu_si128( ( __m128i const* )
								( ( int8_t* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
				zero_point1 = _mm_loadu_si128( ( __m128i const* )
								( ( int8_t* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) ) );
				zero_point2 = _mm_loadu_si128( ( __m128i const* )
								( ( int8_t* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 2 * 16 ) ) );
				zero_point3 = _mm_loadu_si128( ( __m128i const* )
								( ( int8_t* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j + ( 3 * 16 ) ) );
			}
			else if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
			{
				zero_point0 = _mm_maskz_set1_epi8( 0xFFFF,
								*( ( int8_t* )post_ops_list_temp->op_args1 ) );
				zero_point1 = _mm_maskz_set1_epi8( 0xFFFF,
								*( ( int8_t* )post_ops_list_temp->op_args1 ) );
				zero_point2 = _mm_maskz_set1_epi8( 0xFFFF,
								*( ( int8_t* )post_ops_list_temp->op_args1 ) );
				zero_point3 = _mm_maskz_set1_epi8( 0xFFFF,
								*( ( int8_t* )post_ops_list_temp->op_args1 ) );
			}

			CVT_MULRND_CVT32(zmm8,  selector1, zero_point0 );
			CVT_MULRND_CVT32(zmm12, selector2, zero_point1 );
			CVT_MULRND_CVT32(zmm16, selector3, zero_point2 );
			CVT_MULRND_CVT32(zmm20, selector4, zero_point3 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_MATRIX_ADD_6x64:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
			if ( post_ops_attr.c_stor_type == S8 )
			{
				int8_t* matptr = ( int8_t* )post_ops_list_temp->op_args1;

				S8_S32_MATRIX_ADD_LOAD( _cvtu32_mask16( 0xFFFF ),
				                        selector1, 0, 0 );
				S8_S32_MATRIX_ADD_LOAD( _cvtu32_mask16( 0xFFFF ),
				                        selector2, 0, 1 );
				S8_S32_MATRIX_ADD_LOAD( _cvtu32_mask16( 0xFFFF ),
				                        selector3, 0, 2 );
				S8_S32_MATRIX_ADD_LOAD( _cvtu32_mask16( 0xFFFF ),
				                        selector4, 0, 3 );

			}
			else
			{
				int32_t* matptr = ( int32_t* )post_ops_list_temp->op_args1;

				S32_S32_MATRIX_ADD_LOAD( _cvtu32_mask16( 0xFFFF ),
				                         selector1, 0, 0 );
				S32_S32_MATRIX_ADD_LOAD( _cvtu32_mask16( 0xFFFF ),
				                         selector2, 0, 1 );
				S32_S32_MATRIX_ADD_LOAD( _cvtu32_mask16( 0xFFFF ),
				                         selector3, 0, 2 );
				S32_S32_MATRIX_ADD_LOAD( _cvtu32_mask16( 0xFFFF ),
				                         selector4, 0, 3 );
			}

			zmm8  = _mm512_add_epi32( selector1, zmm8 );
			zmm12 = _mm512_add_epi32( selector2, zmm12 );
			zmm16 = _mm512_add_epi32( selector3, zmm16 );
			zmm20 = _mm512_add_epi32( selector4, zmm20 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}

		POST_OPS_SWISH_6x64:
		{
			selector1 =
			  _mm512_set1_epi32( *( ( int32_t* )post_ops_list_temp->op_args2 ) );
			__m512 al = _mm512_cvtepi32_ps( selector1 );

			__m512 fl_reg, al_in, r, r2, z, dn;

			SWISH_S32_AVX512( zmm8,  fl_reg, al, al_in,
			                  r, r2, z, dn, selector2 );
			SWISH_S32_AVX512( zmm12, fl_reg, al, al_in,
			                  r, r2, z, dn, selector2 );
			SWISH_S32_AVX512( zmm16, fl_reg, al, al_in,
			                  r, r2, z, dn, selector2 );
			SWISH_S32_AVX512( zmm20, fl_reg, al, al_in,
			                  r, r2, z, dn, selector2 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_6x64_DISABLE:
		{
			if ( post_ops_attr.buf_downscale != NULL )
			{
				CVT_STORE_S32_S8_MASK( zmm8,  k1, 0, 0 );
				CVT_STORE_S32_S8_MASK( zmm12, k2, 0, 1 );
				CVT_STORE_S32_S8_MASK( zmm16, k3, 0, 2 );
				CVT_STORE_S32_S8_MASK( zmm20, k4, 0, 3 );
			}
			else
			{
				_mm512_mask_storeu_epi32( c_use + ( 0*16 ), k1, zmm8 );
				_mm512_mask_storeu_epi32( c_use + ( 1*16 ), k2, zmm12 );
				_mm512_mask_storeu_epi32( c_use + ( 2*16 ), k3, zmm16 );
				_mm512_mask_storeu_epi32( c_use + ( 3*16 ), k4, zmm20 );
			}
		}

		post_ops_attr.post_op_c_j += nr0;

	} // jr loop
}
#endif // BLIS_ADDON_LPGEMM
