/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#include "../u8s8s32/lpgemm_s32_kern_macros.h"
#include "../u8s8s32/lpgemm_s32_memcpy_macros.h"

LPGEMV_M_EQ1_KERN(int8_t,int8_t,int32_t,s8s8s32os32)
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

	const int8_t *a_use = NULL;
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
		__mmask32 k7 = 0xFFFFFFFF, k8 = 0xFFFFFFFF;


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

            uint8_t cvt_uint8 = 128;
	        __m512i vec_uint8 = _mm512_set1_epi8 (cvt_uint8);

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

                zmm4 = _mm512_add_epi8( zmm4, vec_uint8 );
                zmm5 = _mm512_add_epi8( zmm5, vec_uint8 );
                zmm6 = _mm512_add_epi8( zmm6, vec_uint8 );
                zmm7 = _mm512_add_epi8( zmm7, vec_uint8 );

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
                zmm4 = _mm512_add_epi8( zmm4, vec_uint8 );

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

				zmm0 = _mm512_maskz_loadu_epi16( k5, b_use );

				// Broadcast a[0,kr:kr+4].
				a_kfringe_buf = _mm_maskz_loadu_epi8( load_mask, a_use );

            	zmm4 = _mm512_broadcastd_epi32( a_kfringe_buf );
                zmm4 = _mm512_add_epi8( zmm4, vec_uint8 );

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

        int32_t* bsumptr = post_ops_attr.b_col_sum_vec +
		                   post_ops_attr.b_sum_offset;

        zmm0 = _mm512_maskz_loadu_epi32( k1, bsumptr );
        zmm1 = _mm512_maskz_loadu_epi32( k2, bsumptr + 16 );
        zmm2 = _mm512_maskz_loadu_epi32( k3, bsumptr + 32 );
        zmm3 = _mm512_maskz_loadu_epi32( k4, bsumptr + 48 );

		zmm8  = _mm512_sub_epi32( zmm8,  zmm0 );
        zmm12 = _mm512_sub_epi32( zmm12, zmm1 );
        zmm16 = _mm512_sub_epi32( zmm16, zmm2 );
        zmm20 = _mm512_sub_epi32( zmm20, zmm3 );

        // Load alpha and beta
		__m512i selector1 = _mm512_set1_epi32( alpha );
		__m512i selector2 = _mm512_set1_epi32( beta );

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
				if ( post_ops_attr.c_stor_type == S8 )
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
				else if ( post_ops_attr.c_stor_type == U8 )
				{
					U8_S32_BETA_OP_NLT16F_MASK( k1, zmm8,  0, 0,
												selector1, selector2 )
					U8_S32_BETA_OP_NLT16F_MASK( k2, zmm12, 0, 1,
												selector1, selector2 )
					U8_S32_BETA_OP_NLT16F_MASK( k3, zmm16, 0, 2,
												selector1, selector2 )
					U8_S32_BETA_OP_NLT16F_MASK( k4, zmm20, 0, 3,
												selector1, selector2 )
				}
				else if ( post_ops_attr.c_stor_type == BF16 )
				{
					BF16_S32_BETA_OP_NLT16F_MASK( k1, zmm8,  0, 0,
												selector1, selector2 )
					BF16_S32_BETA_OP_NLT16F_MASK( k2, zmm12, 0, 1,
												selector1, selector2 )
					BF16_S32_BETA_OP_NLT16F_MASK( k3, zmm16, 0, 2,
												selector1, selector2 )
					BF16_S32_BETA_OP_NLT16F_MASK( k4, zmm20, 0, 3,
												selector1, selector2 )
				}
				else if ( post_ops_attr.c_stor_type == F32 )
				{
					F32_S32_BETA_OP_NLT16F_MASK( k1, zmm8,  0, 0,
												selector1, selector2 )
					F32_S32_BETA_OP_NLT16F_MASK( k2, zmm12, 0, 1,
												selector1, selector2 )
					F32_S32_BETA_OP_NLT16F_MASK( k3, zmm16, 0, 2,
												selector1, selector2 )
					F32_S32_BETA_OP_NLT16F_MASK( k4, zmm20, 0, 3,
												selector1, selector2 )
				}
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

		__m512 acc_8 = _mm512_setzero_ps();
		__m512 acc_12 = _mm512_setzero_ps();
		__m512 acc_16 = _mm512_setzero_ps();
		__m512 acc_20 = _mm512_setzero_ps();

		acc_8 = _mm512_cvtepi32_ps( zmm8 );
		acc_12 = _mm512_cvtepi32_ps( zmm12 );
		acc_16 = _mm512_cvtepi32_ps( zmm16 );
		acc_20 = _mm512_cvtepi32_ps( zmm20 );

		post_ops_attr.is_last_k = TRUE;
		lpgemm_post_op *post_ops_list_temp = post_op;
		POST_OP_LABEL_LASTK_SAFE_JUMP

		POST_OPS_BIAS_6x64:
		{
			__m512 b0 = _mm512_setzero_ps();
			__m512 b1 = _mm512_setzero_ps();
			__m512 b2 = _mm512_setzero_ps();
			__m512 b3 = _mm512_setzero_ps();

			if ( post_ops_list_temp->stor_type == BF16 )
			{
				BF16_F32_BIAS_LOAD(b0, k1, 0);
				BF16_F32_BIAS_LOAD(b1, k2, 1);
				BF16_F32_BIAS_LOAD(b2, k3, 2);
				BF16_F32_BIAS_LOAD(b3, k4, 3);
			}
			else if ( post_ops_list_temp->stor_type == S8 )
			{
				S8_F32_BIAS_LOAD(b0, k1, 0);
				S8_F32_BIAS_LOAD(b1, k2, 1);
				S8_F32_BIAS_LOAD(b2, k3, 2);
				S8_F32_BIAS_LOAD(b3, k4, 3);
			}
			else if ( post_ops_list_temp->stor_type == S32 )
			{
				S32_F32_BIAS_LOAD(b0, k1, 0);
				S32_F32_BIAS_LOAD(b1, k2, 1);
				S32_F32_BIAS_LOAD(b2, k3, 2);
				S32_F32_BIAS_LOAD(b3, k4, 3);
			}
			else
			{
				b0 =
					_mm512_maskz_loadu_ps( k1,
						( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				b1 =
					_mm512_maskz_loadu_ps( k2,
						( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				b2 =
					_mm512_maskz_loadu_ps( k3,
						( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_j + ( 2 * 16 ) );
				b3 =
					_mm512_maskz_loadu_ps( k4,
						( float* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_j + ( 3 * 16 ) );
			}

			acc_8  = _mm512_add_ps( b0, acc_8 );
			acc_12 = _mm512_add_ps( b1, acc_12 );
			acc_16 = _mm512_add_ps( b2, acc_16 );
			acc_20 = _mm512_add_ps( b3, acc_20 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_RELU_6x64:
		{
			__m512 zero = _mm512_setzero_ps();

			acc_8  = _mm512_max_ps( zero, acc_8 );
			acc_12 = _mm512_max_ps( zero, acc_12 );
			acc_16 = _mm512_max_ps( zero, acc_16 );
			acc_20 = _mm512_max_ps( zero, acc_20 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_RELU_SCALE_6x64:
		{
			__m512 zero = _mm512_setzero_ps();
			__m512 scale;

			if ( ( post_ops_attr.c_stor_type == S32 ) ||
				 ( post_ops_attr.c_stor_type == U8 ) ||
			     ( post_ops_attr.c_stor_type == S8 ) )
			{
				scale = _mm512_cvtepi32_ps
						( _mm512_set1_epi32(
							*( ( int32_t* )post_ops_list_temp->op_args2 ) ) );
			}
			else
			{
				scale = _mm512_set1_ps(
						*( ( float* )post_ops_list_temp->op_args2 ) );
			}

			__mmask16 relu_cmp_mask;

			RELU_SCALE_OP_F32_AVX512( acc_8 )
			RELU_SCALE_OP_F32_AVX512( acc_12 )
			RELU_SCALE_OP_F32_AVX512( acc_16 )
			RELU_SCALE_OP_F32_AVX512( acc_20 )

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_GELU_TANH_6x64:
		{
			__m512 dn, z, x, r2, r, y;
			__m512i tmpout;

			GELU_TANH_F32_AVX512_DEF( acc_8, y, r, r2, x, z, dn, tmpout );
			GELU_TANH_F32_AVX512_DEF( acc_12, y, r, r2, x, z, dn, tmpout );
			GELU_TANH_F32_AVX512_DEF( acc_16, y, r, r2, x, z, dn, tmpout );
			GELU_TANH_F32_AVX512_DEF( acc_20, y, r, r2, x, z, dn, tmpout );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_GELU_ERF_6x64:
		{
			__m512 y, r, r2;

			GELU_ERF_F32_AVX512_DEF( acc_8, y, r, r2 );
			GELU_ERF_F32_AVX512_DEF( acc_12, y, r, r2 );
			GELU_ERF_F32_AVX512_DEF( acc_16, y, r, r2 );
			GELU_ERF_F32_AVX512_DEF( acc_20, y, r, r2 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR

		}
		POST_OPS_CLIP_6x64:
		{
			__m512 min = _mm512_setzero_ps();
			__m512 max = _mm512_setzero_ps();

			if ( ( post_ops_attr.c_stor_type == S32 ) ||
				 ( post_ops_attr.c_stor_type == U8 ) ||
				 ( post_ops_attr.c_stor_type == S8 ) )
			{
				min = _mm512_cvtepi32_ps
						(_mm512_set1_epi32( *( int32_t* )post_ops_list_temp->op_args2 ));
				max = _mm512_cvtepi32_ps
						(_mm512_set1_epi32( *( int32_t* )post_ops_list_temp->op_args3 ));
			}
			else
			{
				min = _mm512_set1_ps(
						*( ( float* )post_ops_list_temp->op_args2 ) );
				max = _mm512_set1_ps(
						*( ( float* )post_ops_list_temp->op_args3 ) );
			}

			CLIP_F32_AVX512( acc_8,  min, max )
			CLIP_F32_AVX512( acc_12, min, max )
			CLIP_F32_AVX512( acc_16, min, max )
			CLIP_F32_AVX512( acc_20, min, max )

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_DOWNSCALE_6x64:
		{
			__m512 scale0 = _mm512_setzero_ps();
			__m512 scale1 = _mm512_setzero_ps();
			__m512 scale2 = _mm512_setzero_ps();
			__m512 scale3 = _mm512_setzero_ps();

			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				scale0 =
				  _mm512_maskz_loadu_ps( k1,
						(float*)post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				scale1 =
				  _mm512_maskz_loadu_ps( k2,
						(float*)post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				scale2 =
				  _mm512_maskz_loadu_ps( k3,
						(float*)post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 2 * 16 ) );
				scale3 =
				  _mm512_maskz_loadu_ps( k4,
						(float*)post_ops_list_temp->scale_factor +
						post_ops_attr.post_op_c_j + ( 3 * 16 ) );
			}
			else if ( post_ops_list_temp->scale_factor_len == 1 )
			{
				scale0 = _mm512_set1_ps(
						*( ( float* )post_ops_list_temp->scale_factor ) );
				scale1 = _mm512_set1_ps(
						*( ( float* )post_ops_list_temp->scale_factor ) );
				scale2 = _mm512_set1_ps(
						*( ( float* )post_ops_list_temp->scale_factor ) );
				scale3 = _mm512_set1_ps(
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
				zero_point0 = _mm_maskz_loadu_epi8( k1,
							( ( int8_t* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
				zero_point1 = _mm_maskz_loadu_epi8( k2,
							( ( int8_t* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) ) );
				zero_point2 = _mm_maskz_loadu_epi8( k3,
							( ( int8_t* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 2 * 16 ) ) );
				zero_point3 = _mm_maskz_loadu_epi8( k4,
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

			CVT_MULRND_F32(acc_8, scale0, zero_point0 );
			CVT_MULRND_F32(acc_12, scale1, zero_point1 );
			CVT_MULRND_F32(acc_16, scale2, zero_point2 );
			CVT_MULRND_F32(acc_20, scale3, zero_point3 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_MATRIX_ADD_6x64:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_s8 = ( post_ops_list_temp->stor_type == S8 ) ||
					( ( post_ops_list_temp->stor_type == NONE ) &&
					  ( post_ops_attr.c_stor_type == S8 ) );
			bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 );
			bool is_f32 = ( post_ops_list_temp->stor_type == F32 );

			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();
			__m512 t0 = _mm512_setzero_ps();
			__m512 t1 = _mm512_setzero_ps();
			__m512 t2 = _mm512_setzero_ps();
			__m512 t3 = _mm512_setzero_ps();

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
					BF16_F32_MATRIX_ADD_LOAD( k1,
								t0, scl_fctr1, 0, 0 );
					BF16_F32_MATRIX_ADD_LOAD( k2,
								t1, scl_fctr2, 0, 1 );
					BF16_F32_MATRIX_ADD_LOAD( k3,
								t2, scl_fctr3, 0, 2 );
					BF16_F32_MATRIX_ADD_LOAD( k4,
								t3, scl_fctr4, 0, 3 );
				}
				else
				{
					BF16_F32_MATRIX_ADD_LOAD( k1,
								t0, scl_fctr1, 0, 0 );
					BF16_F32_MATRIX_ADD_LOAD( k2,
								t1, scl_fctr1, 0, 1 );
					BF16_F32_MATRIX_ADD_LOAD( k3,
								t2, scl_fctr1, 0, 2 );
					BF16_F32_MATRIX_ADD_LOAD( k4,
								t3, scl_fctr1, 0, 3 );
				}

				acc_8 = _mm512_add_ps( t0, acc_8 );
				acc_12 = _mm512_add_ps( t1, acc_12 );
				acc_16 = _mm512_add_ps( t2, acc_16 );
				acc_20 = _mm512_add_ps( t3, acc_20 );
			}
			else if ( is_f32 == TRUE )
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					F32_ACC_MATRIX_ADD_LOAD( k1, t0, scl_fctr1, 0, 0 );
					F32_ACC_MATRIX_ADD_LOAD( k2, t1, scl_fctr2, 0, 1 );
					F32_ACC_MATRIX_ADD_LOAD( k3, t2, scl_fctr3, 0, 2 );
					F32_ACC_MATRIX_ADD_LOAD( k4, t3, scl_fctr4, 0, 3 );
				}
				else
				{
					F32_ACC_MATRIX_ADD_LOAD( k1, t0, scl_fctr1, 0, 0 );
					F32_ACC_MATRIX_ADD_LOAD( k2, t1, scl_fctr1, 0, 1 );
					F32_ACC_MATRIX_ADD_LOAD( k3, t2, scl_fctr1, 0, 2 );
					F32_ACC_MATRIX_ADD_LOAD( k4, t3, scl_fctr1, 0, 3 );
				}

				acc_8 = _mm512_add_ps( t0, acc_8 );
				acc_12 = _mm512_add_ps( t1, acc_12 );
				acc_16 = _mm512_add_ps( t2, acc_16 );
				acc_20 = _mm512_add_ps( t3, acc_20 );
			}
			else if ( is_s8 == TRUE )
			{
				int8_t* matptr = ( int8_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					S8_F32_MATRIX_ADD_LOAD( k1, t0, scl_fctr1, 0, 0 );
					S8_F32_MATRIX_ADD_LOAD( k2, t1, scl_fctr2, 0, 1 );
					S8_F32_MATRIX_ADD_LOAD( k3, t2, scl_fctr3, 0, 2 );
					S8_F32_MATRIX_ADD_LOAD( k4, t3, scl_fctr4, 0, 3 );
				}
				else
				{
					S8_F32_MATRIX_ADD_LOAD( k1, t0, scl_fctr1, 0, 0 );
					S8_F32_MATRIX_ADD_LOAD( k2, t1, scl_fctr1, 0, 1 );
					S8_F32_MATRIX_ADD_LOAD( k3, t2, scl_fctr1, 0, 2 );
					S8_F32_MATRIX_ADD_LOAD( k4, t3, scl_fctr1, 0, 3 );
				}

				acc_8  = _mm512_add_ps( t0, acc_8 );
				acc_12 = _mm512_add_ps( t1, acc_12 );
				acc_16 = _mm512_add_ps( t2, acc_16 );
				acc_20 = _mm512_add_ps( t3, acc_20 );
			}
			else
			{
				int32_t* matptr = ( int32_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					S32_F32_MATRIX_ADD_LOAD( k1,
								 t0, scl_fctr1, 0, 0 );
					S32_F32_MATRIX_ADD_LOAD( k2,
								 t1, scl_fctr2, 0, 1 );
					S32_F32_MATRIX_ADD_LOAD( k3,
								 t2, scl_fctr3, 0, 2 );
					S32_F32_MATRIX_ADD_LOAD( k4,
								 t3, scl_fctr4, 0, 3 );
				}
				else
				{
					S32_F32_MATRIX_ADD_LOAD( k1,
								 t0, scl_fctr1, 0, 0 );
					S32_F32_MATRIX_ADD_LOAD( k2,
								 t1, scl_fctr1, 0, 1 );
					S32_F32_MATRIX_ADD_LOAD( k3,
								 t2, scl_fctr1, 0, 2 );
					S32_F32_MATRIX_ADD_LOAD( k4,
								 t3, scl_fctr1, 0, 3 );
				}

				acc_8  = _mm512_add_ps( t0, acc_8 );
				acc_12 = _mm512_add_ps( t1, acc_12 );
				acc_16 = _mm512_add_ps( t2, acc_16 );
				acc_20 = _mm512_add_ps( t3, acc_20 );
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_MATRIX_MUL_6x64:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			bool is_s8 = ( post_ops_list_temp->stor_type == S8 ) ||
					( ( post_ops_list_temp->stor_type == NONE ) &&
					  ( post_ops_attr.c_stor_type == S8 ) );
			bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 );
			bool is_f32 = ( post_ops_list_temp->stor_type == F32 );

			__m512 scl_fctr1 = _mm512_setzero_ps();
			__m512 scl_fctr2 = _mm512_setzero_ps();
			__m512 scl_fctr3 = _mm512_setzero_ps();
			__m512 scl_fctr4 = _mm512_setzero_ps();
			__m512 t0 = _mm512_setzero_ps();
			__m512 t1 = _mm512_setzero_ps();
			__m512 t2 = _mm512_setzero_ps();
			__m512 t3 = _mm512_setzero_ps();

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
					BF16_F32_MATRIX_MUL_LOAD( k1,
								t0, scl_fctr1, 0, 0 );
					BF16_F32_MATRIX_MUL_LOAD( k2,
								t1, scl_fctr2, 0, 1 );
					BF16_F32_MATRIX_MUL_LOAD( k3,
								t2, scl_fctr3, 0, 2 );
					BF16_F32_MATRIX_MUL_LOAD( k4,
								t3, scl_fctr4, 0, 3 );
				}
				else
				{
					BF16_F32_MATRIX_MUL_LOAD( k1,
								t0, scl_fctr1, 0, 0 );
					BF16_F32_MATRIX_MUL_LOAD( k2,
								t1, scl_fctr1, 0, 1 );
					BF16_F32_MATRIX_MUL_LOAD( k3,
								t2, scl_fctr1, 0, 2 );
					BF16_F32_MATRIX_MUL_LOAD( k4,
								t3, scl_fctr1, 0, 3 );
				}

				acc_8 = _mm512_mul_round_ps( t0, acc_8,
					( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
				acc_12 = _mm512_mul_round_ps( t1, acc_12,
					( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
				acc_16 = _mm512_mul_round_ps( t2, acc_16,
					( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
				acc_20 = _mm512_mul_round_ps( t3, acc_20,
					( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
			}
			else if ( is_f32 == TRUE )
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					F32_MATRIX_MUL_LOAD( k1, t0, scl_fctr1, 0, 0 );
					F32_MATRIX_MUL_LOAD( k2, t1, scl_fctr2, 0, 1 );
					F32_MATRIX_MUL_LOAD( k3, t2, scl_fctr3, 0, 2 );
					F32_MATRIX_MUL_LOAD( k4, t3, scl_fctr4, 0, 3 );
				}
				else
				{
					F32_MATRIX_MUL_LOAD( k1, t0, scl_fctr1, 0, 0 );
					F32_MATRIX_MUL_LOAD( k2, t1, scl_fctr1, 0, 1 );
					F32_MATRIX_MUL_LOAD( k3, t2, scl_fctr1, 0, 2 );
					F32_MATRIX_MUL_LOAD( k4, t3, scl_fctr1, 0, 3 );
				}

				acc_8 = _mm512_mul_round_ps( t0, acc_8,
					( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
				acc_12 = _mm512_mul_round_ps( t1, acc_12,
					( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
				acc_16 = _mm512_mul_round_ps( t2, acc_16,
					( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
				acc_20 = _mm512_mul_round_ps( t3, acc_20,
					( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
			}
			else if ( is_s8 == TRUE )
			{
				int8_t* matptr = ( int8_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					S8_F32_MATRIX_MUL_LOAD( k1, t0, scl_fctr1, 0, 0 );
					S8_F32_MATRIX_MUL_LOAD( k2, t1, scl_fctr2, 0, 1 );
					S8_F32_MATRIX_MUL_LOAD( k3, t2, scl_fctr3, 0, 2 );
					S8_F32_MATRIX_MUL_LOAD( k4, t3, scl_fctr4, 0, 3 );
				}
				else
				{
					S8_F32_MATRIX_MUL_LOAD( k1, t0, scl_fctr1, 0, 0 );
					S8_F32_MATRIX_MUL_LOAD( k2, t1, scl_fctr1, 0, 1 );
					S8_F32_MATRIX_MUL_LOAD( k3, t2, scl_fctr1, 0, 2 );
					S8_F32_MATRIX_MUL_LOAD( k4, t3, scl_fctr1, 0, 3 );
				}

				acc_8 = _mm512_mul_round_ps( t0, acc_8,
					( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
				acc_12 = _mm512_mul_round_ps( t1, acc_12,
					( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
				acc_16 = _mm512_mul_round_ps( t2, acc_16,
					( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
				acc_20 = _mm512_mul_round_ps( t3, acc_20,
					( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
			}
			else
			{
				int32_t* matptr = ( int32_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					S32_F32_MATRIX_MUL_LOAD( k1, t0, scl_fctr1, 0, 0 );
					S32_F32_MATRIX_MUL_LOAD( k2, t1, scl_fctr2, 0, 1 );
					S32_F32_MATRIX_MUL_LOAD( k3, t2, scl_fctr3, 0, 2 );
					S32_F32_MATRIX_MUL_LOAD( k4, t3, scl_fctr4, 0, 3 );
				}
				else
				{
					S32_F32_MATRIX_MUL_LOAD( k1, t0, scl_fctr1, 0, 0 );
					S32_F32_MATRIX_MUL_LOAD( k2, t1, scl_fctr1, 0, 1 );
					S32_F32_MATRIX_MUL_LOAD( k3, t2, scl_fctr1, 0, 2 );
					S32_F32_MATRIX_MUL_LOAD( k4, t3, scl_fctr1, 0, 3 );
				}

				acc_8 = _mm512_mul_round_ps( t0, acc_8,
					( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
				acc_12 = _mm512_mul_round_ps( t1, acc_12,
					( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
				acc_16 = _mm512_mul_round_ps( t2, acc_16,
					( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
				acc_20 = _mm512_mul_round_ps( t3, acc_20,
					( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_SWISH_6x64:
		{
			__m512 scale;

			if ( ( post_ops_attr.c_stor_type == S32 ) ||
				 ( post_ops_attr.c_stor_type == U8 ) ||
			     ( post_ops_attr.c_stor_type == S8 ) )
			{
				scale = _mm512_cvtepi32_ps
						(_mm512_set1_epi32(
							*( ( int32_t* )post_ops_list_temp->op_args2 ) ));
			}
			else
			{
				scale = _mm512_set1_ps(
						*( ( float* )post_ops_list_temp->op_args2 ) );
			}

			__m512 al_in, r, r2, z, dn;
			__m512i temp;

			SWISH_F32_AVX512_DEF( acc_8,  scale, al_in, r, r2, z, dn, temp );
			SWISH_F32_AVX512_DEF( acc_12, scale, al_in, r, r2, z, dn, temp );
			SWISH_F32_AVX512_DEF( acc_16, scale, al_in, r, r2, z, dn, temp );
			SWISH_F32_AVX512_DEF( acc_20, scale, al_in, r, r2, z, dn, temp );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_TANH_6x64:
		{
			__m512 dn, z, x, r2, r;
			__m512i q;

			TANHF_AVX512( acc_8, r, r2, x, z, dn, q );
			TANHF_AVX512( acc_12, r, r2, x, z, dn, q );
			TANHF_AVX512( acc_16, r, r2, x, z, dn, q );
			TANHF_AVX512( acc_20, r, r2, x, z, dn, q );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_SIGMOID_6x64:
		{
			__m512 al_in, r, r2, z, dn;
			__m512i tmpout;

			SIGMOID_F32_AVX512_DEF( acc_8,  al_in, r, r2, z, dn, tmpout );
			SIGMOID_F32_AVX512_DEF( acc_12, al_in, r, r2, z, dn, tmpout );
			SIGMOID_F32_AVX512_DEF( acc_16, al_in, r, r2, z, dn, tmpout );
			SIGMOID_F32_AVX512_DEF( acc_20, al_in, r, r2, z, dn, tmpout );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_6x64_DISABLE:
		{
			if ( post_ops_attr.buf_downscale != NULL )
			{
				if ( post_ops_attr.c_stor_type == S8 )
				{
					CVT_STORE_F32_S8_MASK( k1, acc_8, 0, 0 );
					CVT_STORE_F32_S8_MASK( k2, acc_12, 0, 1 );
					CVT_STORE_F32_S8_MASK( k3, acc_16, 0, 2 );
					CVT_STORE_F32_S8_MASK( k4, acc_20, 0, 3 );
				}
				else if ( post_ops_attr.c_stor_type == U8 )
				{
					CVT_STORE_F32_U8_MASK( k1, acc_8, 0, 0 );
					CVT_STORE_F32_U8_MASK( k2, acc_12, 0, 1 );
					CVT_STORE_F32_U8_MASK( k3, acc_16, 0, 2 );
					CVT_STORE_F32_U8_MASK( k4, acc_20, 0, 3 );
				}
				else if ( post_ops_attr.c_stor_type == BF16 )
				{
					CVT_STORE_F32_BF16_MASK( k1, acc_8, 0, 0 );
					CVT_STORE_F32_BF16_MASK( k2, acc_12, 0, 1 );
					CVT_STORE_F32_BF16_MASK( k3, acc_16, 0, 2 );
					CVT_STORE_F32_BF16_MASK( k4, acc_20, 0, 3 );
				}
				else if ( post_ops_attr.c_stor_type == F32 )
				{
					STORE_F32_MASK( k1, acc_8, 0, 0 );
					STORE_F32_MASK( k2, acc_12, 0, 1 );
					STORE_F32_MASK( k3, acc_16, 0, 2 );
					STORE_F32_MASK( k4, acc_20, 0, 3 );
				}
			}
			else
			{
				_mm512_mask_storeu_epi32( c_use + ( 0*16 ), k1,
						_mm512_cvtps_epi32( acc_8 ) );
				_mm512_mask_storeu_epi32( c_use + ( 1*16 ), k2,
						_mm512_cvtps_epi32( acc_12 ) );
				_mm512_mask_storeu_epi32( c_use + ( 2*16 ), k3,
						_mm512_cvtps_epi32( acc_16 ) );
				_mm512_mask_storeu_epi32( c_use + ( 3*16 ), k4,
						_mm512_cvtps_epi32( acc_20 ) );
			}
		}

		post_ops_attr.post_op_c_j += nr0;
		post_ops_attr.b_sum_offset += nr0;

	} // jr loop

}
#endif // BLIS_ADDON_LPGEMM
