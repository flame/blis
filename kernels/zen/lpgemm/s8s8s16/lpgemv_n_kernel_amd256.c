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

#include "../u8s8s16/lpgemm_s16_kern_macros.h"

#define LPGEMV_N_KERNEL_2_LOADS( ymm0, ymm1, paddr, stride ) \
  ymm0 = _mm256_loadu_si256( (__m256i const *)paddr ); \
  ymm1 = _mm256_loadu_si256( (__m256i const *)(paddr + stride) ); \
  ymm0 = _mm256_add_epi8( ymm0, vec_uint8 ); \
  ymm1 = _mm256_add_epi8( ymm1, vec_uint8 );

#define LPGEMV_N_KERNEL_2_FMA( a_reg1, a_reg2, b_reg, \
                               inter_reg1, inter_reg2, c_reg1, c_reg2 ) \
  inter_reg1 = _mm256_maddubs_epi16(a_reg1, b_reg); \
  c_reg1   = _mm256_add_epi16(inter_reg1, c_reg1); \
  inter_reg2 = _mm256_maddubs_epi16(a_reg2, b_reg); \
  c_reg2   = _mm256_add_epi16(inter_reg2, c_reg2);


#define LPGEMV_N_KERNEL_4_LOADS( ymm0, ymm1, ymm2, ymm3, paddr, stride ) \
  ymm0 = _mm256_loadu_si256( (__m256i const *)(paddr) ); \
  ymm1 = _mm256_loadu_si256( (__m256i const *)(paddr + stride) ); \
  ymm2 = _mm256_loadu_si256( (__m256i const *)(paddr + 2 * stride) ); \
  ymm3 = _mm256_loadu_si256( (__m256i const *)(paddr + 3 * stride) ); \
  ymm0 = _mm256_add_epi8( ymm0, vec_uint8 ); \
  ymm1 = _mm256_add_epi8( ymm1, vec_uint8 ); \
  ymm2 = _mm256_add_epi8( ymm2, vec_uint8 ); \
  ymm3 = _mm256_add_epi8( ymm3, vec_uint8 );

#define LPGEMV_N_KERNEL_4_FMA( a_reg1, a_reg2, a_reg3, a_reg4, b_reg, \
                               inter_reg1, inter_reg2, \
                               inter_reg3, inter_reg4, \
                               out_reg1, out_reg2, out_reg3, out_reg4 ) \
  inter_reg1 = _mm256_maddubs_epi16(a_reg1, b_reg); \
  out_reg1   = _mm256_add_epi16(inter_reg1, out_reg1); \
  inter_reg2 = _mm256_maddubs_epi16(a_reg2, b_reg); \
  out_reg2   = _mm256_add_epi16(inter_reg2, out_reg2); \
  inter_reg3 = _mm256_maddubs_epi16(a_reg3, b_reg); \
  out_reg3   = _mm256_add_epi16(inter_reg3, out_reg3); \
  inter_reg4 = _mm256_maddubs_epi16(a_reg4, b_reg); \
  out_reg4   = _mm256_add_epi16(inter_reg4, out_reg4);

#define LPGEMV_YMM2XMM( ymm0, ymm1, ymm2, ymm3, xmm0 ) \
  ymm0 = _mm256_hadd_epi16( ymm0, ymm1 ); \
  ymm1 = _mm256_hadd_epi16( ymm2, ymm3 ); \
  ymm0 = _mm256_hadd_epi16( ymm0, ymm1 ); \
  xmm0 = _mm_add_epi16( _mm256_extracti128_si256( ymm0, 0 ), \
                        _mm256_extracti128_si256( ymm0, 1 ) );



LPGEMV_N_EQ1_KERN(int8_t, int8_t, int16_t, s8s8s16os16)
{
	static void* post_ops_labels[] =
	            {
	              &&POST_OPS_DISABLE,
	              &&POST_OPS_BIAS,
	              &&POST_OPS_RELU,
	              &&POST_OPS_RELU_SCALE,
	              &&POST_OPS_GELU_TANH,
	              &&POST_OPS_GELU_ERF,
	              &&POST_OPS_CLIP,
	              &&POST_OPS_DOWNSCALE,
	              &&POST_OPS_MATRIX_ADD,
	              &&POST_OPS_SWISH
	            };

	int8_t *a_use = NULL;
	int8_t *b_use = NULL;
	int16_t *c_use = NULL;

	lpgemm_post_op_attr post_ops_attr = *(post_op_attr);

	// temp buffer to store output C vector
	int16_t ctemp[16];

	// temp buffers to store a, b data in k_rem case.
	int8_t buf0[32] = {0};
	int8_t buf1[32] = {0};
	int8_t buf2[32] = {0};
	int8_t buf3[32] = {0};
	int8_t buf4[32] = {0};
	int8_t buf5[32] = {0};
	int8_t buf6[32] = {0};
	int8_t buf7[32] = {0};
	int8_t buf8[32] = {0};


	uint8_t cvt_uint8 = 128;
	__m256i vec_uint8;

	int16_t* bsumptr = post_ops_attr.b_col_sum_vec_s16;

	for ( dim_t ir = 0; ir < m0; ir += MR )
	{
		dim_t mr0 = bli_min( ( m0 - ir ), MR );
		dim_t k_iter = k / 32;
		dim_t k_rem = k % 32;

		__m256i ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
		__m256i ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14;
		__m256i ymm15;

		__m128i xmm0, xmm1;

		/* zero the accumulator registers */
		ZERO_ACC_YMM_4_REG( ymm8,  ymm9,  ymm10, ymm11 )
		ZERO_ACC_YMM_4_REG( ymm12, ymm13, ymm14, ymm15 )

		//update pointers
		a_use = (int8_t*)a + ir * rs_a;
		b_use = (int8_t*)b;
		c_use = (int16_t*)c + ir * rs_c;

		if( mr0 == MR )
		{
			vec_uint8 = _mm256_set1_epi8 (cvt_uint8);

			for (dim_t k = 0; k < k_iter; k++)
			{

				ymm6 = _mm256_loadu_si256( (__m256i const *)(b_use) );
				b_use += 32;

				//Load 4x32 elements from row0-row3 of A
				LPGEMV_N_KERNEL_4_LOADS( ymm0, ymm1, ymm2, ymm3, a_use, rs_a )

				LPGEMV_N_KERNEL_4_FMA( ymm0, ymm1, ymm2, ymm3,
				                       ymm6, ymm4, ymm5, ymm7, ymm4,
				                       ymm8, ymm9, ymm10, ymm11
				                     )

				// Load 4x32 elements from row8-row11 of A
				LPGEMV_N_KERNEL_4_LOADS( ymm0, ymm1, ymm2, ymm3,
				                         ( a_use + 4 * rs_a ), rs_a
				                       )

				LPGEMV_N_KERNEL_4_FMA( ymm0, ymm1, ymm2, ymm3,
				                       ymm6, ymm4, ymm5, ymm7, ymm4,
				                       ymm12, ymm13, ymm14, ymm15
				                     )

				a_use += 32;
			}



			if( k_rem )
			{
				uint8_t buf_vec_uint8_t[32] = {0};
				int8_t* restrict a0 = (a_use);
				int8_t* restrict a1 = (a_use + rs_a );
				int8_t* restrict a2 = (a_use + 2 * rs_a );
				int8_t* restrict a3 = (a_use + 3 * rs_a );
				int8_t* restrict a4 = (a_use + 4 * rs_a );
				int8_t* restrict a5 = (a_use + 5 * rs_a );
				int8_t* restrict a6 = (a_use + 6 * rs_a );
				int8_t* restrict a7 = (a_use + 7 * rs_a );

				for( dim_t i = 0; i < k_rem; i++)
				{
					buf8[i] = b_use[i];
					buf0[i] = a0[i];
					buf1[i] = a1[i];
					buf2[i] = a2[i];
					buf3[i] = a3[i];
					buf4[i] = a4[i];
					buf5[i] = a5[i];
					buf6[i] = a6[i];
					buf7[i] = a7[i];
					buf_vec_uint8_t[i] = cvt_uint8;
				}
				ymm6 = _mm256_loadu_si256( (__m256i const *)buf8 );

				vec_uint8 = _mm256_loadu_si256( ( __m256i const *) buf_vec_uint8_t );

				//Load 4x32 elements from row0-row3 of A
				ymm0 = _mm256_loadu_si256( (__m256i const *)buf0 );
				ymm1 = _mm256_loadu_si256( (__m256i const *)buf1 );
				ymm2 = _mm256_loadu_si256( (__m256i const *)buf2 );
				ymm3 = _mm256_loadu_si256( (__m256i const *)buf3 );

				ymm0 = _mm256_add_epi8( ymm0, vec_uint8 );
				ymm1 = _mm256_add_epi8( ymm1, vec_uint8 );
				ymm2 = _mm256_add_epi8( ymm2, vec_uint8 );
				ymm3 = _mm256_add_epi8( ymm3, vec_uint8 );

				LPGEMV_N_KERNEL_4_FMA( ymm0, ymm1, ymm2, ymm3,
				                       ymm6, ymm4, ymm5, ymm7, ymm4,
				                       ymm8, ymm9, ymm10, ymm11
				                     )

				// Load 4x32 elements from row8-row11 of A
				ymm0 = _mm256_loadu_si256( (__m256i const *)buf4 );
				ymm1 = _mm256_loadu_si256( (__m256i const *)buf5 );
				ymm2 = _mm256_loadu_si256( (__m256i const *)buf6 );
				ymm3 = _mm256_loadu_si256( (__m256i const *)buf7 );

				ymm0 = _mm256_add_epi8( ymm0, vec_uint8 );
				ymm1 = _mm256_add_epi8( ymm1, vec_uint8 );
				ymm2 = _mm256_add_epi8( ymm2, vec_uint8 );
				ymm3 = _mm256_add_epi8( ymm3, vec_uint8 );

				LPGEMV_N_KERNEL_4_FMA( ymm0, ymm1, ymm2, ymm3,
				                       ymm6, ymm4, ymm5, ymm7, ymm4,
				                       ymm12, ymm13, ymm14, ymm15
				                     )

			}
			//Add the registers horizantally to get one
			LPGEMV_YMM2XMM( ymm8, ymm9, ymm10, ymm11, xmm0 )
			LPGEMV_YMM2XMM( ymm12, ymm13, ymm14, ymm15, xmm1 )

			xmm0 = _mm_hadd_epi16( xmm0, xmm1 );

			// post ops are applied on ymm register though
			// second half of the register is filled with zeroes.
			ymm8 = _mm256_setzero_si256();
			ymm8 = _mm256_inserti128_si256( ymm8, xmm0, 0);

			ymm0 = _mm256_set1_epi16( *bsumptr );
			ymm8 = _mm256_sub_epi16( ymm8, ymm0 );
		}
		else
		{
			int8_t *a_use_fringe = a_use;
			dim_t mr0_use = mr0;
			dim_t regidx = 0;

			if( mr0_use >= 4 )
			{
				vec_uint8 = _mm256_set1_epi8 (cvt_uint8);

				for (dim_t k = 0; k < k_iter; k++)
				{
					ymm6 = _mm256_loadu_si256( (__m256i const *)b_use );
					b_use += 32;

					//Load 4x32 elements from row0-row3 of A
					LPGEMV_N_KERNEL_4_LOADS( ymm0, ymm1, ymm2, ymm3,
					                         a_use, rs_a )

					LPGEMV_N_KERNEL_4_FMA( ymm0, ymm1, ymm2, ymm3,
					                       ymm6, ymm4, ymm5, ymm7, ymm4,
					                       ymm8, ymm9, ymm10, ymm11
										)

					a_use += 32;
				}

				if( k_rem )
				{
					uint8_t buf_vec_uint8_t[32] = {0};
					int8_t* restrict a0 = (a_use);
					int8_t* restrict a1 = (a_use + rs_a );
					int8_t* restrict a2 = (a_use + 2 * rs_a );
					int8_t* restrict a3 = (a_use + 3 * rs_a );

					for( dim_t i = 0; i < k_rem; i++)
					{
						buf8[i] = b_use[i];
						buf0[i] = a0[i];
						buf1[i] = a1[i];
						buf2[i] = a2[i];
						buf3[i] = a3[i];
						buf_vec_uint8_t[i] = cvt_uint8;
					}
					ymm6 = _mm256_loadu_si256( (__m256i const *)buf8 );

					vec_uint8 = _mm256_loadu_si256( (__m256i const *)buf_vec_uint8_t );
					//Load 4xk_rem elements from row0-row3 of A

					ymm0 = _mm256_loadu_si256( (__m256i const *)buf0 );
					ymm1 = _mm256_loadu_si256( (__m256i const *)buf1 );
					ymm2 = _mm256_loadu_si256( (__m256i const *)buf2 );
					ymm3 = _mm256_loadu_si256( (__m256i const *)buf3 );

					ymm0 = _mm256_add_epi8( ymm0, vec_uint8 );
					ymm1 = _mm256_add_epi8( ymm1, vec_uint8 );
					ymm2 = _mm256_add_epi8( ymm2, vec_uint8 );
					ymm3 = _mm256_add_epi8( ymm3, vec_uint8 );

					LPGEMV_N_KERNEL_4_FMA( ymm0, ymm1, ymm2, ymm3,
					                       ymm6, ymm4, ymm5, ymm7, ymm4,
					                       ymm8, ymm9, ymm10, ymm11
					                     )
				}

				//update pointers
				mr0_use -= 4;
				a_use = a_use_fringe + 4 * rs_a;
				a_use_fringe = a_use;
				b_use = (int8_t*)b;

				//Add the registers horizantally to get one
				LPGEMV_YMM2XMM( ymm8, ymm9, ymm10, ymm11, xmm0 )

				xmm0 = _mm_hadd_epi16( xmm0, xmm0 );

				__int64_t data = _mm_extract_epi64( xmm0, 0);
				//insert xmm outputs into final output reg based on regidx
				ymm8 = _mm256_setzero_si256();
				ymm8 = _mm256_insert_epi64( ymm8, data, 0 );
				regidx++;
			}

			// Dot product for  <= 3
			if ( mr0_use )
			{
				// Dot product for m = 2
				if ( mr0_use >= 2 )
				{
					vec_uint8 = _mm256_set1_epi8 (cvt_uint8);

					for ( dim_t k = 0; k < k_iter; k++ )
					{
						// Load 0-31 in b[k+0 - k+31]
						ymm6 = _mm256_loadu_si256( (__m256i const *)b_use );

						LPGEMV_N_KERNEL_2_LOADS( ymm0, ymm1, a_use, rs_a);

						LPGEMV_N_KERNEL_2_FMA( ymm0, ymm1, ymm6, ymm4,
						                       ymm5, ymm12, ymm13);
						b_use += 32; // move b pointer to next 32 elements
						a_use += 32;
					}
					if ( k_rem )
					{
						uint8_t buf_vec_uint8_t[32] = {0};
						int8_t* restrict a0 = (a_use);
						int8_t* restrict a1 = (a_use + rs_a );

						for( dim_t i = 0; i < k_rem; i++)
						{
							buf8[i] = b_use[i];
							buf0[i] = a0[i];
							buf1[i] = a1[i];
							buf_vec_uint8_t[i] = cvt_uint8;
						}
						ymm6 = _mm256_loadu_si256( (__m256i const *)buf8 );

						vec_uint8 = _mm256_loadu_si256( (__m256i const *)buf_vec_uint8_t );
						//Load 2xk_rem elements from row0-row3 of A

						ymm0 = _mm256_loadu_si256( (__m256i const *)buf0 );
						ymm1 = _mm256_loadu_si256( (__m256i const *)buf1 );

						ymm0 = _mm256_add_epi8( ymm0, vec_uint8 );
						ymm1 = _mm256_add_epi8( ymm1, vec_uint8 );

						LPGEMV_N_KERNEL_2_FMA( ymm0, ymm1, ymm6,
						                       ymm4, ymm5, ymm12, ymm13 );
					}

					mr0_use -= 2;
					a_use = a_use_fringe + 2 * rs_a;
					a_use_fringe = a_use;
					b_use = (int8_t*)b;
				}

				// Dot product for m = 1
				if ( mr0_use == 1 )
				{
					vec_uint8 = _mm256_set1_epi8 (cvt_uint8);

					for ( dim_t k = 0; k < k_iter; k++ )
					{
						// Load 0-31 in b[k+0 - k+31]
						ymm6 = _mm256_loadu_si256( (__m256i const *)b_use );

						// Load 1x32 elements from row0-row1 of A
						ymm0 = _mm256_loadu_si256( (__m256i const *)a_use );
						ymm0 = _mm256_add_epi8( ymm0, vec_uint8 );

						ymm4 = _mm256_maddubs_epi16(ymm0, ymm6);
						ymm14 = _mm256_add_epi16(ymm4, ymm14);

						b_use += 32; // move b pointer to next 32 elements
						a_use += 32;
					}
					if ( k_rem )
					{
						uint8_t buf_vec_uint8_t[32] = {0};
						int8_t* restrict a0 = (a_use);

						for( dim_t i = 0; i < k_rem; i++)
						{
							buf8[i] = b_use[i];
							buf0[i] = a0[i];
							buf_vec_uint8_t[i] = cvt_uint8;
						}
						ymm6 = _mm256_loadu_si256( (__m256i const *)buf8 );

						vec_uint8 = _mm256_loadu_si256( (__m256i const *)buf_vec_uint8_t );

						//Load 1xk_rem elements from row0-row3 of A

						ymm0 = _mm256_loadu_si256( (__m256i const *)buf0 );
						ymm0 = _mm256_add_epi8( ymm0, vec_uint8 );

						ymm4 = _mm256_maddubs_epi16(ymm0, ymm6);
						ymm14 = _mm256_add_epi16(ymm4, ymm14);
					}

					// When only fringe 1,
					// update the registers to store in order
					if ( !( mr0 & 0x2 ) )  ymm12 = ymm14;
				}

				LPGEMV_YMM2XMM( ymm12, ymm13, ymm14, ymm15, xmm0)
				xmm0 = _mm_hadd_epi16( xmm0, xmm0 );

				__int64_t data = _mm_extract_epi64( xmm0, 0);
				//insert xmm outputs into final output reg based on regidx

				if( regidx == 0 )
				{
					ymm8 = _mm256_insert_epi64( ymm8, data, 0 );
				}
				else
				{
					ymm8 = _mm256_insert_epi64( ymm8, data, 1 );
				}

			}

			int16_t buf_vec_int16_t[16] = {0};
			for( dim_t i = 0; i < mr0; i++)
				buf_vec_int16_t[i] = *bsumptr;
			ymm0 = _mm256_loadu_si256( ( __m256i const *) buf_vec_int16_t);
			ymm8 = _mm256_sub_epi16( ymm8, ymm0 );
		}

		// Load alpha and beta
		__m256i selector1 = _mm256_set1_epi16(alpha);
		__m256i selector2 = _mm256_set1_epi16(beta);

		// Scale by alpha
		ymm8 = _mm256_mullo_epi16(selector1, ymm8);

		if( beta != 0 )
		{
			if ( post_ops_attr.buf_downscale != NULL )
			{
				if( post_ops_attr.rs_c_downscale == 1 )
				{
					if( post_ops_attr.c_stor_type == S8 )
					{
						dim_t m0_rem_dscale_bytes = mr0 * sizeof( int8_t );

						S8_S16_BETA_NLT16_MEMCP_UTIL( ctemp, 0,
						                              m0_rem_dscale_bytes );

						S8_S16_BETA_OP_NLT16( ymm8, ctemp,
						                      selector1, selector2 )
					}
					else if( post_ops_attr.c_stor_type == U8 )
					{
						dim_t m0_rem_dscale_bytes = mr0 * sizeof( uint8_t );

						U8_S16_BETA_NLT16_MEMCP_UTIL( ctemp, 0,
						                              m0_rem_dscale_bytes );

						U8_S16_BETA_OP_NLT16( ymm8, ctemp,
						                      selector1, selector2 )
					}
				}
				else
				{
					if( post_ops_attr.c_stor_type == S8 )
					{
						int8_t ctemp[16];
						for( dim_t i = 0; i < mr0; i++ )
						{
							ctemp[i] = *( (int8_t*)post_ops_attr.buf_downscale
							+ ( post_ops_attr.rs_c_downscale *
							( post_ops_attr.post_op_c_i + i ) ) );
						}
						selector1 = _mm256_cvtepi8_epi32
						        ( _mm_loadu_si128( (__m128i const*)ctemp ) );
						S16_BETA_FMA( ymm8, selector1, selector2 );
					}
					else if( post_ops_attr.c_stor_type == U8 )
					{
						uint8_t ctemp[16];
						for( dim_t i = 0; i < mr0; i++ )
						{
							ctemp[i] = *( (uint8_t*)post_ops_attr.buf_downscale
							+ ( post_ops_attr.rs_c_downscale *
							( post_ops_attr.post_op_c_i + i ) ) );
						}
						selector1 = _mm256_cvtepu8_epi32
						        ( _mm_loadu_si128( (__m128i const*)ctemp ) );
						S16_BETA_FMA( ymm8, selector1, selector2 );
					}
				}
			}
			else
			{
				if( rs_c == 1 )
				{
					dim_t m0_rem_bytes = mr0 * sizeof( int16_t );
					memcpy( ctemp, c_use, m0_rem_bytes );
					S16_S16_BETA_OP_NLT16( ymm8, ctemp,
					                       selector1, selector2 )
				}
				else
				{
					for( dim_t i = 0; i < mr0; i++ )
					{
						ctemp[i] =  c_use[ i * rs_c ];
					}
					selector1 = _mm256_loadu_si256( (__m256i const *)ctemp );
					S16_BETA_FMA( ymm8, selector1, selector2 );
				}
			}
		}

		// Post Ops
		lpgemm_post_op * post_ops_list_temp = post_op;

		post_ops_attr.is_last_k = TRUE;
		POST_OP_LABEL_LASTK_SAFE_JUMP


		POST_OPS_BIAS:
		{


			selector1 =
			  _mm256_set1_epi16( *( ( int16_t* )post_ops_list_temp->op_args1) );

			ymm8 = _mm256_add_epi16( selector1, ymm8 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
	    POST_OPS_RELU:
		{
			selector1 = _mm256_setzero_si256();

			ymm8 = _mm256_max_epi16( selector1, ymm8 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_RELU_SCALE:
		{
			__m256i b0;
			selector1 = _mm256_setzero_si256();
			selector2 = _mm256_set1_epi16(
			            *( ( int16_t* )post_ops_list_temp->op_args2 ) );

			RELU_SCALE_OP_S16_AVX2( ymm8 )

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_GELU_TANH:
		{
			__m256 dn, z, x, r2, r, y1, y2, x_tanh;
			__m256i q;

			GELU_TANH_S16_AVX2( ymm8, y1, y2, r, r2, x, z, dn, x_tanh, q )

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_GELU_ERF:
		{
			__m256 x, r, y1, y2, x_erf;

			GELU_ERF_S16_AVX2(ymm8, y1, y2, r, x, x_erf)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_CLIP:
		{
			__m256i min = _mm256_set1_epi16(
			                *( int16_t* )post_ops_list_temp->op_args2 );
			__m256i max = _mm256_set1_epi16(
			                *( int16_t* )post_ops_list_temp->op_args3 );

			CLIP_S16_AVX2(ymm8, min, max)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_DOWNSCALE:
		{
			__m128i temp[2];
			__m256i temp_32[2];
			__m256 temp_float[2];
			__m256 scale_1 = _mm256_setzero_ps();
			__m256 scale_2 = _mm256_setzero_ps();
			__m128i _zero_point_0 = _mm_setzero_si128();
			__m256i zero_point_0 = _mm256_setzero_si256();
			__m256 res_1, res_2;

			scale_1 =
			  _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );

			scale_2 =
			  _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );

			_zero_point_0 = _mm_set1_epi8(
			         *( ( int8_t* )post_ops_list_temp->op_args1 ) );

			if ( post_ops_attr.c_stor_type == S8 )
			{
				zero_point_0 = _mm256_cvtepi8_epi16( _zero_point_0 );
			}
			else if ( post_ops_attr.c_stor_type == U8 )
			{
				zero_point_0 = _mm256_cvtepu8_epi16( _zero_point_0 );
			}

			// Scale first 16 columns of the 2 rows.
			CVT_MULRND_CVT16(ymm8, scale_1, scale_2, zero_point_0)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}

		POST_OPS_MATRIX_ADD:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

			if ( post_ops_attr.c_stor_type == S8 )
			{
				int8_t* matptr = ( int8_t* )post_ops_list_temp->op_args1;

				if( ldm == 1 )
				{
					memcpy
					(
					( int8_t* )ctemp,
					matptr + ( ( post_ops_attr.post_op_c_i ) * ldm ) +
					post_ops_attr.post_op_c_j + ( 0 * 16 ),
					( mr0 ) * sizeof(int8_t)
					);
					selector1 = _mm256_cvtepi8_epi16(
					            _mm_loadu_si128( ( __m128i const* )ctemp ) );
					ymm8 = _mm256_add_epi16( selector1, ymm8 );
				}
				else
				{
					int8_t ctemp[16];
					for( dim_t i = 0; i < mr0; i++ )
					{
						ctemp[i] = *( matptr +
						            ( ( post_ops_attr.post_op_c_i + i )
						                * ldm ) );
					}
					selector1 = _mm256_cvtepi8_epi16
					            ( _mm_loadu_si128( (__m128i const*)ctemp ) );
					ymm8 = _mm256_add_epi16( selector1, ymm8 );
				}
			}
			else if ( post_ops_attr.c_stor_type == U8 )
			{
				uint8_t* matptr = ( uint8_t* )post_ops_list_temp->op_args1;

				if( ldm == 1 )
				{
					memcpy
					(
					( uint8_t* )ctemp,
					matptr + ( ( post_ops_attr.post_op_c_i ) * ldm ) +
					post_ops_attr.post_op_c_j + ( 0 * 16 ),
					( mr0 ) * sizeof(uint8_t)
					);
					selector1 = _mm256_cvtepu8_epi16(
					            _mm_loadu_si128( ( __m128i const* )ctemp ) );
					ymm8 = _mm256_add_epi16( selector1, ymm8 );
				}
				else
				{
					uint8_t ctemp[16];
					for( dim_t i = 0; i < mr0; i++ )
					{
						ctemp[i] = *( matptr +
						            ( ( post_ops_attr.post_op_c_i + i )
						                * ldm ) );
					}
					selector1 = _mm256_cvtepu8_epi16
					            ( _mm_loadu_si128( (__m128i const*)ctemp ) );
					ymm8 = _mm256_add_epi16( selector1, ymm8 );
				}
			}
			else
			{
				int16_t* matptr = ( int16_t* )post_ops_list_temp->op_args1;

				if( ldm == 1 )
				{
					memcpy
					(
					( int16_t* )ctemp,
					matptr + ( ( post_ops_attr.post_op_c_i ) * ldm ) +
					post_ops_attr.post_op_c_j + ( 0 * 16 ),
					( mr0 ) * sizeof(int16_t)
					);

					selector1 = _mm256_loadu_si256( ( __m256i const* )ctemp );

					ymm8 = _mm256_add_epi16( selector1, ymm8 );
				}
				else
				{
					int16_t ctemp[16];
					for( dim_t i = 0; i < mr0; i++ )
					{
						ctemp[i] = *( matptr +
						            ( ( post_ops_attr.post_op_c_i + i )
						                * ldm ) );
					}
					selector1 = _mm256_loadu_si256( (__m256i const *)ctemp );
					ymm8 = _mm256_add_epi16( selector1, ymm8 );
				}
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_SWISH:
		{
			selector1 =
			  _mm256_set1_epi16( *( ( int16_t* )post_ops_list_temp->op_args2 ) );
			__m256 al = _mm256_cvtepi32_ps( _mm256_cvtepi16_epi32( \
							_mm256_extractf128_si256( selector1, 0 ) ) );

			__m256 al_in, tmp_reg1, tmp_reg2, r, r2, z, dn;
			__m256i ex_out;

			SWISH_S16_AVX2( ymm8, al, al_in, tmp_reg1,
			                tmp_reg2, r, r2, z, dn, ex_out );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_DISABLE:
		{
			if ( post_ops_attr.buf_downscale != NULL )
			{
				__m128i temp[2];
				__m256i zero_reg = _mm256_setzero_si256();
				if( post_ops_attr.rs_c_downscale == 1 )
				{
					if( post_ops_attr.c_stor_type == S8 )
					{
						// Store the results in downscaled type
						// (int8 instead of int16).
						CVT_STORE_S16_S8_1ROW_NLT16(ymm8, zero_reg, ctemp);

						dim_t m0_rem_dscale_bytes = mr0 * sizeof( int8_t );

						CVT_STORE_S16_S8_NLT16_MEMCP_UTIL( ctemp, 0,
						                                   m0_rem_dscale_bytes);
					}
					else if( post_ops_attr.c_stor_type == U8 )
					{
						// Store the results in downscaled type (uint8 instead of int16).
						CVT_STORE_S16_U8_1ROW_NLT16(ymm8, zero_reg, ctemp);

						dim_t m0_rem_dscale_bytes = mr0 * sizeof( uint8_t );

						CVT_STORE_S16_U8_NLT16_MEMCP_UTIL( ctemp, 0,
						                                   m0_rem_dscale_bytes);
					}
				}
				else
				{
					if( post_ops_attr.c_stor_type == S8 )
					{
						int8_t ctemp[16];

						CVT_STORE_S16_S8_1ROW_NLT16(ymm8, zero_reg, ctemp);
						for( dim_t i = 0; i < mr0; i++ )
						{
							*( ( int8_t* )post_ops_attr.buf_downscale +
							( post_ops_attr.rs_c_downscale *
							( post_ops_attr.post_op_c_i + i ) ) ) = ctemp[i];
						}
					}
					else if( post_ops_attr.c_stor_type == U8 )
					{
						uint8_t ctemp[16];

						CVT_STORE_S16_U8_1ROW_NLT16(ymm8, zero_reg, ctemp);

						for( dim_t i = 0; i < mr0; i++ )
						{
							*( ( uint8_t* )post_ops_attr.buf_downscale +
							( post_ops_attr.rs_c_downscale *
							( post_ops_attr.post_op_c_i + i ) ) ) = ctemp[i];
						}
					}
				}
			}
			else
			{
				if( rs_c == 1 )
				{
					_mm256_storeu_si256( ( __m256i* )ctemp, ymm8 );

					dim_t m0_rem_bytes = mr0 * sizeof( int16_t );

					memcpy( c_use, ctemp, m0_rem_bytes );
				}
				else
				{
					_mm256_storeu_si256( ( __m256i* )ctemp, ymm8 );

					for( dim_t i = 0; i < mr0; i++ )
					{
						c_use[i * rs_c] = ctemp[i];
					}
				}
			}

		post_ops_attr.post_op_c_i += MR;
		}
	}
}

#endif
