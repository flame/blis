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

#include "lpgemm_s32_kern_macros.h"
#include "lpgemm_s32_memcpy_macros.h"


#define LPGEMV_N_KERNEL_4_LOADS( zmm0, zmm1, zmm2, zmm3, paddr, stride ) \
  zmm0 = _mm512_loadu_si512( paddr ); \
  zmm1 = _mm512_loadu_si512( paddr + stride ); \
  zmm2 = _mm512_loadu_si512( paddr + 2 * stride ); \
  zmm3 = _mm512_loadu_si512( paddr + 3 * stride );

#define LPGEMV_N_KERNEL_4_MASKLOADS( zmm0, zmm1, zmm2, \
                                     zmm3, k1, paddr, stride ) \
  zmm0 = _mm512_maskz_loadu_epi8( k1, paddr ); \
  zmm1 = _mm512_maskz_loadu_epi8( k1, paddr + stride ); \
  zmm2 = _mm512_maskz_loadu_epi8( k1, paddr + 2 * stride ); \
  zmm3 = _mm512_maskz_loadu_epi8( k1, paddr + 3 * stride );

#define LPGEMV_N_KERNEL_4_FMA( zmm8, zmm9, zmm10, zmm11, \
                               zmm6, zmm0, zmm1, zmm2, zmm3 ) \
  zmm8  = _mm512_dpbusd_epi32( zmm8,  zmm0, zmm6 ); \
  zmm9  = _mm512_dpbusd_epi32( zmm9,  zmm1, zmm6 ); \
  zmm10 = _mm512_dpbusd_epi32( zmm10, zmm2, zmm6 ); \
  zmm11 = _mm512_dpbusd_epi32( zmm11, zmm3, zmm6 );

#define LPGEMV_ZMM2XMM( zmm0, zmm1, zmm2, zmm3, \
                        ymm0, ymm1, ymm2, ymm3, xmm0) \
  ymm0 = _mm256_add_epi32 (_mm512_extracti32x8_epi32 (zmm0, 0x0), \
                           _mm512_extracti32x8_epi32 (zmm0, 0x1)); \
  ymm1 = _mm256_add_epi32 (_mm512_extracti32x8_epi32 (zmm1, 0x0), \
                           _mm512_extracti32x8_epi32 (zmm1, 0x1)); \
  ymm0 = _mm256_hadd_epi32 (ymm0, ymm1); \
  ymm2 = _mm256_add_epi32 (_mm512_extracti32x8_epi32 (zmm2, 0x0), \
                           _mm512_extracti32x8_epi32 (zmm2, 0x1)); \
  ymm3 = _mm256_add_epi32 (_mm512_extracti32x8_epi32 (zmm3, 0x0), \
                           _mm512_extracti32x8_epi32 (zmm3, 0x1)); \
  ymm1 = _mm256_hadd_epi32 (ymm2, ymm3); \
  ymm0 = _mm256_hadd_epi32 (ymm0, ymm1); \
  xmm0 = _mm_add_epi32 ( _mm256_extracti128_si256 (ymm0, 0), \
                         _mm256_extracti128_si256 (ymm0,1));

#define CVT_STORE_S32_S8_MASK(reg,mask,m_ind,n_ind) \
  _mm512_mask_cvtsepi32_storeu_epi8 \
  ( \
    ( int8_t* )post_ops_attr.buf_downscale + \
    ( post_ops_attr.rs_c_downscale * \
	( post_ops_attr.post_op_c_i + m_ind ) ) + \
    post_ops_attr.post_op_c_j + ( n_ind * 16 ), \
    mask, reg \
  ); \

LPGEMV_N_EQ1_KERN(uint8_t, int8_t, int32_t, u8s8s32os32)
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

	const uint8_t *a_use = NULL;
	const int8_t *b_use = NULL;
	int32_t *c_use = NULL;

	lpgemm_post_op_attr post_ops_attr = *(post_op_attr);

	for ( dim_t ir = 0; ir < m0; ir += MR )
	{
		dim_t mr0 = bli_min( ( m0 - ir ), MR );
		dim_t k_iter = k/64;
		dim_t k_rem = k & 0x3F;

		//Create load mask for k fringe
		__mmask64 k1 = 0xFFFFFFFFFFFFFFFF;
		if( k_rem )
		{
			k1 = ( k1 >> ( 64 - k_rem ) );
		}

		// Create store mask for C for mr fringe
		__mmask16 k2 = 0xFFFF;
		if ( mr0 < MR )
		{
			k2 = ( 0xFFFF >> ( MR - mr0 ) );
		}

		__m512i zmm0, zmm1, zmm2, zmm3, zmm6;
		__m512i zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14;
		__m512i zmm15, zmm16, zmm17, zmm18, zmm19, zmm20, zmm21;
		__m512i zmm22, zmm23, zmm24, zmm25, zmm26, zmm27, zmm28;
		__m512i zmm29, zmm30, zmm31;

		__m256i ymm0,ymm1,ymm2,ymm3,ymm4,ymm5,ymm6;
		__m128i xmm0, xmm1, xmm2, xmm3;

		/* zero the accumulator registers */
		ZERO_ACC_ZMM_4_REG( zmm8,  zmm9,  zmm10, zmm11 )
		ZERO_ACC_ZMM_4_REG( zmm12, zmm13, zmm14, zmm15 )
		ZERO_ACC_ZMM_4_REG( zmm16, zmm17, zmm18, zmm19 )
		ZERO_ACC_ZMM_4_REG( zmm20, zmm21, zmm22, zmm23 )
		ZERO_ACC_XMM_4_REG( xmm0,  xmm1,  xmm2,  xmm3  )

		//update pointers
		a_use = a + ir * rs_a;
		b_use = b;
		c_use = c + ir * rs_c;

		if( mr0 == MR )
		{
			//Dot product kernel
			for (dim_t k = 0; k < k_iter; k++)
			{
				zmm6 = _mm512_loadu_si512( b_use );
				b_use += 64;

				//Load 4x64 elements from row0-row3 of A
				LPGEMV_N_KERNEL_4_LOADS( zmm0, zmm1, zmm2, zmm3, a_use, rs_a )
				a_use += ( 4 * rs_a );

				// Load 4x64 elements from row3-row7 of A
				LPGEMV_N_KERNEL_4_LOADS( zmm24, zmm25, zmm26,
				                         zmm27, a_use, rs_a
				                       )
				a_use += ( 4 * rs_a );

				LPGEMV_N_KERNEL_4_FMA( zmm8, zmm9, zmm10, zmm11,
				                       zmm6, zmm0, zmm1, zmm2, zmm3
				                     )

				// Load 4x64 elements from row8-row11 of A
				LPGEMV_N_KERNEL_4_LOADS( zmm28, zmm29, zmm30,
				                         zmm31, a_use, rs_a
				                       )
				a_use += ( 4 * rs_a );

				// Load 4x64 elements from row12-row15 of A
				LPGEMV_N_KERNEL_4_LOADS( zmm0, zmm1, zmm2, zmm3, a_use, rs_a )
				a_use -= ( 12 * rs_a ); //Update aptr back to move horizontally

				LPGEMV_N_KERNEL_4_FMA( zmm12, zmm13, zmm14, zmm15,
				                       zmm6, zmm24, zmm25, zmm26, zmm27
				                     )
				LPGEMV_N_KERNEL_4_FMA( zmm16, zmm17, zmm18, zmm19,
				                       zmm6, zmm28, zmm29, zmm30, zmm31
				                     )
				LPGEMV_N_KERNEL_4_FMA( zmm20, zmm21, zmm22, zmm23,
				                       zmm6, zmm0, zmm1, zmm2, zmm3
				                     )
				a_use += 64;

			} // kloop
			if( k_rem )
			{
				zmm6 = _mm512_maskz_loadu_epi8( k1, b_use );

				//Load 4x64 elements from row0-row3 of A
				LPGEMV_N_KERNEL_4_MASKLOADS( zmm0, zmm1, zmm2,
				                             zmm3, k1, a_use, rs_a
				                           )
				a_use += ( 4 * rs_a );

				// Load 4x64 elements from row3-row7 of A
				LPGEMV_N_KERNEL_4_MASKLOADS( zmm24, zmm25, zmm26,
				                             zmm27, k1, a_use, rs_a
				                           )
				a_use += ( 4 * rs_a );

				LPGEMV_N_KERNEL_4_FMA( zmm8, zmm9, zmm10, zmm11,
				                       zmm6, zmm0, zmm1, zmm2, zmm3
				                     )

				// Load 4x64 elements from row8-row11 of A
				LPGEMV_N_KERNEL_4_MASKLOADS( zmm28, zmm29, zmm30,
				                             zmm31, k1, a_use, rs_a
				                           )
				a_use += ( 4 * rs_a );

				// Load 4x64 elements from row12-row15 of A
				LPGEMV_N_KERNEL_4_MASKLOADS( zmm0, zmm1, zmm2,
				                             zmm3, k1, a_use, rs_a
				                           )
				a_use -= ( 12 * rs_a ); //Update aptr back to move horizontally


				LPGEMV_N_KERNEL_4_FMA( zmm12, zmm13, zmm14, zmm15,
				                       zmm6, zmm24, zmm25, zmm26, zmm27
				                     )
				LPGEMV_N_KERNEL_4_FMA( zmm16, zmm17, zmm18, zmm19,
				                       zmm6, zmm28, zmm29, zmm30, zmm31
				                     )
				LPGEMV_N_KERNEL_4_FMA( zmm20, zmm21, zmm22, zmm23,
				                       zmm6, zmm0, zmm1, zmm2, zmm3
				                     )
				a_use += 64;
			}

			//Add the registers horizantally to get one
			LPGEMV_ZMM2XMM( zmm8, zmm9, zmm10, zmm11,
			                ymm0, ymm1, ymm2, ymm3, xmm0
			              )
			LPGEMV_ZMM2XMM( zmm12, zmm13, zmm14, zmm15,
			                ymm4, ymm1, ymm2, ymm3, xmm1
			              )
			LPGEMV_ZMM2XMM( zmm16, zmm17, zmm18, zmm19,
			                ymm5, ymm1, ymm2, ymm3, xmm2
			              )
			LPGEMV_ZMM2XMM( zmm20, zmm21, zmm22, zmm23,
			                ymm6, ymm1, ymm2, ymm3, xmm3
			              )

			//compose outputs into one zmm to perform post-ops
			zmm8 = _mm512_inserti32x4 ( zmm8, xmm0, 0 );
			zmm8 = _mm512_inserti32x4 ( zmm8, xmm1, 1 );
			zmm8 = _mm512_inserti32x4 ( zmm8, xmm2, 2 );
			zmm8 = _mm512_inserti32x4 ( zmm8, xmm3, 3 );
		}
		else
		{
			//Handle fringe cases when mr0 < MR
			const uint8_t *a_use_fringe = a_use;
			dim_t mr0_use = mr0;
			dim_t regidx = 0;

			// Dot product for mfringe 8
			if ( mr0_use >= 8 )
			{
				// Dot product kernel for mr0 == 8
				for( dim_t k = 0; k < k_iter; k++ )
				{
					// Load 0-63 in b[k+0 - k+31]
					zmm6 = _mm512_loadu_si512( b_use );
					// move b pointer to next 64 elements
					b_use += 64;

					// Load 4x64 elements from row0-row3 of A
					LPGEMV_N_KERNEL_4_LOADS( zmm0, zmm1, zmm2,
					                         zmm3, a_use, rs_a
					                       )
					a_use += ( 4 * rs_a );

					// Load 4x64 elements from row3-row7 of A
					LPGEMV_N_KERNEL_4_LOADS( zmm24, zmm25, zmm26,
					                         zmm27, a_use, rs_a
					                       )
					a_use -= ( 4 * rs_a );

					//Perform FMA on two 4x64 block of A with 64x1
					LPGEMV_N_KERNEL_4_FMA( zmm8, zmm9, zmm10, zmm11,
					                       zmm6, zmm0, zmm1, zmm2, zmm3
					                     )
					LPGEMV_N_KERNEL_4_FMA( zmm12, zmm13, zmm14, zmm15,
					                       zmm6, zmm24, zmm25, zmm26, zmm27
					                     )
					a_use += 64;
				}

				if ( k_rem )
				{
					// Load 0-63 in b[k+0 - k+63]
					zmm6 = _mm512_maskz_loadu_epi8( k1, b_use );

					// Load 4x64 elements from row0-row3 of A
					LPGEMV_N_KERNEL_4_MASKLOADS( zmm0, zmm1, zmm2,
					                             zmm3, k1, a_use, rs_a
					                           )
					a_use += ( 4 * rs_a );
					LPGEMV_N_KERNEL_4_MASKLOADS( zmm24, zmm25, zmm26,
					                             zmm27, k1, a_use, rs_a
					                           )
					LPGEMV_N_KERNEL_4_FMA( zmm8, zmm9, zmm10, zmm11,
					                       zmm6, zmm0, zmm1, zmm2, zmm3
					                     )
					LPGEMV_N_KERNEL_4_FMA( zmm12, zmm13, zmm14, zmm15,
					                       zmm6, zmm24, zmm25, zmm26, zmm27
					                         )
				}

				// update pointers
				mr0_use -= 8;
				a_use = a_use_fringe + 8 * rs_a;
				a_use_fringe = a_use;
				b_use = b;

				// Horizontal add 8 zmm registers
				// and get output into 2 xmm registers
				LPGEMV_ZMM2XMM( zmm8, zmm9, zmm10, zmm11,
				                ymm0, ymm1, ymm2, ymm3, xmm0
				              )
				LPGEMV_ZMM2XMM( zmm12, zmm13, zmm14, zmm15,
				                ymm4, ymm1, ymm2, ymm3, xmm1
				              )

				//insert xmm outputs into final output zmm8 reg
				zmm8 = _mm512_inserti32x4( zmm8, xmm0, 0 );
				zmm8 = _mm512_inserti32x4( zmm8, xmm1, 1 );
				regidx = 2;
			}

			// Dot product for mfringe 4
			if ( mr0_use >= 4 )
			{
				// Dot product kernel for mr0 == 8
				for ( dim_t k = 0; k < k_iter; k++ )
				{
					// Load 0-63 in b[k+0 - k+63]
					zmm6 = _mm512_loadu_si512( b_use );

					// move b pointer to next 64 elements
					b_use += 64;

					// Load 4x64 elements from row0-row3 of A
					LPGEMV_N_KERNEL_4_LOADS( zmm0, zmm1, zmm2,
					                         zmm3, a_use, rs_a
					                       )
					// Perform FMA on 4x64 block of A with 64x1
					LPGEMV_N_KERNEL_4_FMA( zmm16, zmm17, zmm18, zmm19,
					                       zmm6, zmm0, zmm1, zmm2, zmm3
					                     )
					a_use += 64;
				}

				if ( k_rem )
				{
					// Load 0-63 in b[k+0 - k+63]
					zmm6 = _mm512_maskz_loadu_epi8( k1, b_use );

					// Load 4x64 elements from row0-row3 of A
					LPGEMV_N_KERNEL_4_MASKLOADS( zmm0, zmm1, zmm2,
					                             zmm3, k1, a_use, rs_a
					                           )
					LPGEMV_N_KERNEL_4_FMA( zmm16, zmm17, zmm18, zmm19,
					                       zmm6, zmm0, zmm1, zmm2, zmm3
					                     )
				}

				//update pointers
				mr0_use -= 4;
				a_use = a_use_fringe + 4 * rs_a;
				a_use_fringe = a_use;
				b_use = b;

				//Horizontal add 4 zmm reg and get the output into one xmm
				LPGEMV_ZMM2XMM( zmm16, zmm17, zmm18, zmm19,
				                ymm5, ymm1, ymm2, ymm3, xmm2
				              )

				//insert xmm outputs into final output zmm8 reg based on regidx
				if( regidx == 0 ) zmm8 = _mm512_inserti32x4( zmm8, xmm2, 0 );
				else zmm8 = _mm512_inserti32x4( zmm8, xmm2, 2 );
				regidx++;
			}

			// Dot product for  <= 3
			if ( mr0_use )
			{
				// Dot product for m = 2
				if ( mr0_use >= 2 )
				{
					for ( dim_t k = 0; k < k_iter; k++ )
					{
						// Load 0-63 in b[k+0 - k+63]
						zmm6 = _mm512_loadu_si512( b_use );

						// Load 2x64 elements from row0-row1 of A
						zmm0 = _mm512_loadu_si512( a_use );
						zmm1 = _mm512_loadu_si512( a_use + rs_a );
						zmm20 = _mm512_dpbusd_epi32( zmm20, zmm0, zmm6 );
						zmm21 = _mm512_dpbusd_epi32( zmm21, zmm1, zmm6 );

						b_use += 64; // move b pointer to next 64 elements
						a_use += 64;
					}
					if ( k_rem )
					{
						// Load 0-63 in b[k+0 - k+63]
						zmm6 = _mm512_maskz_loadu_epi8( k1, b_use );
						zmm0 = _mm512_maskz_loadu_epi8( k1, a_use );
						zmm1 = _mm512_maskz_loadu_epi8( k1, a_use + rs_a );
						zmm20 = _mm512_dpbusd_epi32( zmm20, zmm0, zmm6 );
						zmm21 = _mm512_dpbusd_epi32( zmm21, zmm1, zmm6 );
					}
					mr0_use -= 2;
					a_use = a_use_fringe + 2 * rs_a;
					a_use_fringe = a_use;
					b_use = b;
				}

				// Dot product for m = 2
				if ( mr0_use == 1 )
				{
					for ( dim_t k = 0; k < k_iter; k++ )
					{
						// Load 0-63 in b[k+0 - k+63]
						zmm6 = _mm512_loadu_si512( b_use );
						zmm0 = _mm512_loadu_si512( a_use );
						zmm22 = _mm512_dpbusd_epi32( zmm22, zmm0, zmm6 );
						b_use += 64; // move b pointer to next 64 elements
						a_use += 64;
					}

					if ( k_rem )
					{
						zmm6 = _mm512_maskz_loadu_epi8( k1, b_use );
						zmm0 = _mm512_maskz_loadu_epi8( k1, a_use );
						zmm22 = _mm512_dpbusd_epi32( zmm22, zmm0, zmm6 );
					}
					// When only fringe 1,
					// update the registers to store in order
					if ( !( mr0 & 0x2 ) )  zmm20 = zmm22;
				}

				// Horizontal add 4 zmm reg and get the output into one xmm
				LPGEMV_ZMM2XMM( zmm20, zmm21, zmm22, zmm23,
				                ymm6, ymm1, ymm2, ymm3, xmm3
				              )

				// insert xmm outputs into final output zmm8 reg based on regidx
				if( regidx == 0 )
				{
					zmm8 = _mm512_inserti32x4( zmm8, xmm3, 0 );
				}
				else if( regidx == 1 )
				{
					zmm8 = _mm512_inserti32x4( zmm8, xmm3, 1 );
				}
				else if ( regidx == 2 )
				{
					zmm8 = _mm512_inserti32x4( zmm8, xmm3, 2 );
				}
				else
				{
					zmm8 = _mm512_inserti32x4( zmm8, xmm3, 3 );
				}
			}
		}

		//Scale accumulated output with alpha
		__m512i selector1 = _mm512_set1_epi32( alpha );
		__m512i selector2 = _mm512_set1_epi32( beta );

		//Mulitply A*B output with alpha
		zmm8 = _mm512_mullo_epi32( selector1, zmm8 );

		if( beta != 0 )
		{
			if( post_ops_attr.buf_downscale != NULL )
			{
				if( post_ops_attr.rs_c_downscale == 1 )
				{
					if ( post_ops_attr.c_stor_type == S8 )
					{
						S8_S32_BETA_OP_NLT16F_MASK( k2, zmm8,  0, 0,
													selector1, selector2 )
					}
					else if ( post_ops_attr.c_stor_type == U8 )
					{
						U8_S32_BETA_OP_NLT16F_MASK( k2, zmm8,  0, 0,
													selector1, selector2 )
					}
					else if ( post_ops_attr.c_stor_type == BF16 )
					{
						BF16_S32_BETA_OP_NLT16F_MASK( k2, zmm8,  0, 0,
													selector1, selector2 )
					}
					else if ( post_ops_attr.c_stor_type == F32 )
					{
						F32_S32_BETA_OP_NLT16F_MASK( k2, zmm8,  0, 0,
													selector1, selector2 )
					}
				}
				else
				{
					if ( post_ops_attr.c_stor_type == S8 )
					{
						int8_t ctemp[16];
						for( dim_t i = 0; i < mr0; i++ )
						{
							ctemp[i] = *( ( int8_t* )post_ops_attr.buf_downscale +
							 ( post_ops_attr.rs_c_downscale *
							 ( post_ops_attr.post_op_c_i + i ) ) );
						}
						selector1 = _mm512_cvtepi8_epi32
									( _mm_maskz_loadu_epi8( 0xFFFF, ctemp ) );
					}
					else if ( post_ops_attr.c_stor_type == U8 )
					{
						uint8_t ctemp[16];
						for( dim_t i = 0; i < mr0; i++ )
						{
							ctemp[i] = *( ( uint8_t* )post_ops_attr.buf_downscale +
							 ( post_ops_attr.rs_c_downscale *
							 ( post_ops_attr.post_op_c_i + i ) ) );
						}
						selector1 = _mm512_cvtepu8_epi32
									( _mm_maskz_loadu_epi8( 0xFFFF, ctemp ) );
					}
					else if ( post_ops_attr.c_stor_type == BF16 )
					{
						bfloat16 ctemp[16];
						for( dim_t i = 0; i < mr0; i++ )
						{
							ctemp[i] = *( ( bfloat16* )post_ops_attr.buf_downscale +
							 ( post_ops_attr.rs_c_downscale *
							 ( post_ops_attr.post_op_c_i + i ) ) );
						}
						selector1 = _mm512_cvtps_epi32(
							( __m512 )_mm512_sllv_epi32(
									_mm512_cvtepi16_epi32(
										_mm256_maskz_loadu_epi16( 0xFFFF, ctemp )
									), _mm512_set1_epi32( 16 ) ) );
					}
					else if ( post_ops_attr.c_stor_type == F32 )
					{
						float ctemp[16];
						for( dim_t i = 0; i < mr0; i++ )
						{
							ctemp[i] = *( ( float* )post_ops_attr.buf_downscale +
							 ( post_ops_attr.rs_c_downscale *
							 ( post_ops_attr.post_op_c_i + i ) ) );
						}
						selector1 = _mm512_cvtps_epi32(
									_mm512_maskz_loadu_ps( 0xFFFF, ctemp ) );
					}
					S32_BETA_FMA( zmm8, selector1, selector2 );
				}
			}
			else
			{
				if( rs_c == 1)
				{
					S32_S32_BETA_OP_NLT16F_MASK( c_use, k2, zmm8, 0, 0, 0,
					                             selector1, selector2 )
				}
				else
				{
					int32_t ctemp[16];
					for( dim_t i = 0; i < mr0; i++ )
					{
						ctemp[i] =  c_use[ i * rs_c ];
					}
					selector1 = _mm512_loadu_epi32( ctemp );
					S32_BETA_FMA( zmm8, selector1, selector2 );
				}
			}
		}

		__m512 acc_8 = _mm512_setzero_ps();
		acc_8 = _mm512_cvtepi32_ps( zmm8 );

		// Post Ops
		lpgemm_post_op *post_ops_list_temp = post_op;

		post_ops_attr.is_last_k = TRUE;
		POST_OP_LABEL_LASTK_SAFE_JUMP

		POST_OPS_BIAS_6x64:
		{
			__m512 b0 = _mm512_setzero_ps();

			if ( post_ops_list_temp->stor_type == BF16 )
			{
				b0 = (__m512)_mm512_sllv_epi32(
					_mm512_cvtepi16_epi32(
						_mm256_maskz_loadu_epi16(
							_cvtu32_mask16( 0x0001 ),
							( ( bfloat16* )post_ops_list_temp->op_args1 )
						) ), _mm512_set1_epi32( 16 ) );
			}
			else if ( post_ops_list_temp->stor_type == S8 )
			{
				b0 = _mm512_cvtepi32_ps(
						_mm512_cvtepi8_epi32(
			 				_mm_maskz_loadu_epi8(
								_cvtu32_mask16( 0x0001 ),
								( ( int8_t* )post_ops_list_temp->op_args1 )
			  				) ) );
			}
			else if ( post_ops_list_temp->stor_type == S32 )
			{
				b0 = _mm512_cvtepi32_ps(
						_mm512_set1_epi32(
							*( ( int32_t* )post_ops_list_temp->op_args1) ) );
			}
			else
			{
				b0 = _mm512_maskz_loadu_ps(
							_cvtu32_mask16( 0x0001 ),
							( ( float* ) post_ops_list_temp->op_args1 ) );
			}
			acc_8 = _mm512_add_ps( b0, acc_8 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_RELU_6x64:
		{
			__m512 zero = _mm512_setzero_ps();

			acc_8 = _mm512_max_ps( zero, acc_8 );

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

			RELU_SCALE_OP_F32_AVX512(acc_8)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_GELU_TANH_6x64:
		{
			__m512 dn, z, x, r2, r, y;
			__m512i tmpout;

			GELU_TANH_F32_AVX512_DEF( acc_8, y, r, r2, x, z, dn, tmpout );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_GELU_ERF_6x64:
		{
			__m512 y, r, r2;

			GELU_ERF_F32_AVX512_DEF( acc_8, y, r, r2 );

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

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_DOWNSCALE_6x64:
		{
			__m512 scale0 = _mm512_setzero_ps();
			scale0 = _mm512_set1_ps(
			             *( ( float* )post_ops_list_temp->scale_factor ) );

			// Need to ensure sse not used to avoid avx512 -> sse transition.
			__m128i zero_point0 = _mm512_castsi512_si128(
			                          _mm512_setzero_si512() );

			zero_point0 = _mm_maskz_set1_epi8( 0xFFFF,
			              *( ( int8_t* )post_ops_list_temp->op_args1 ) );

			CVT_MULRND_F32(acc_8, scale0, zero_point0 );

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
			__m512 t0 = _mm512_setzero_ps();

			// Even though different registers are used for scalar in column and
			// row major case, all those registers will contain the same value.
			// For column major, if m==1, then it means n=1 and scale_factor_len=1.
			if ( post_ops_list_temp->scale_factor_len == 1 )
			{
				scl_fctr1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'c' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'C' ) )
				{
					scl_fctr1 =
						_mm512_maskz_loadu_ps( k2,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + ( 0 * 16 ) );
				}
			}

			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if( ldm == 1 )
				{
					BF16_F32_MATRIX_ADD_LOAD( k2, t0, scl_fctr1, 0, 0 );
					acc_8  = _mm512_add_ps( t0, acc_8 );
				}
				else
				{
					bfloat16 ctemp[16];
					for( dim_t i = 0; i < mr0; i++ )
					{
						ctemp[i] = *( matptr +
						            ( ( post_ops_attr.post_op_c_i + i )
						                * ldm ) );
					}
					t0 = (__m512)_mm512_sllv_epi32
					              (
					                _mm512_cvtepi16_epi32
					                (
					                  _mm256_maskz_loadu_epi16
					                  (
					                    k2 , ctemp
					                  )
					                ), _mm512_set1_epi32( 16 )
					              );
					t0 = _mm512_mul_ps( t0, scl_fctr1 );
					acc_8  = _mm512_add_ps( t0, acc_8 );
				}
			}
			else if ( is_f32 == TRUE )
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if( ldm == 1 )
				{
					F32_ACC_MATRIX_ADD_LOAD( k2, t0, scl_fctr1, 0, 0 );
					acc_8  = _mm512_add_ps( t0, acc_8 );
				}
				else
				{
					float ctemp[16];
					for( dim_t i = 0; i < mr0; i++ )
					{
						ctemp[i] = *( matptr +
						            ( ( post_ops_attr.post_op_c_i + i )
						                * ldm ) );
					}
					t0 = _mm512_maskz_loadu_ps( k2, ctemp );
					t0 = _mm512_mul_ps( t0, scl_fctr1 );
					acc_8  = _mm512_add_ps( t0, acc_8 );
				}
			}
			else if ( is_s8 == TRUE )
			{
				int8_t* matptr = ( int8_t* )post_ops_list_temp->op_args1;

				if( ldm == 1 )
				{
					S8_F32_MATRIX_ADD_LOAD( k2, t0, scl_fctr1, 0, 0 )
					acc_8 = _mm512_add_ps( t0, acc_8 );
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
					t0 = _mm512_cvtepi32_ps(
							_mm512_cvtepi8_epi32(
								_mm_maskz_loadu_epi8( k2, ctemp ) ) );
					t0 = _mm512_mul_round_ps( t0, scl_fctr1,
							( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC )
						);
					acc_8 = _mm512_add_ps( t0, acc_8 );
				}
			}
			else
			{
				int32_t* matptr = ( int32_t* )post_ops_list_temp->op_args1;

				if( ldm == 1 )
				{
					S32_F32_MATRIX_ADD_LOAD(k2, t0, scl_fctr1, 0, 0 );
					acc_8 = _mm512_add_ps( t0, acc_8 );
				}
				else
				{
					int32_t ctemp[16];
					for( dim_t i = 0; i < mr0; i++ )
					{
						ctemp[i] = *( matptr +
						            ( ( post_ops_attr.post_op_c_i + i )
						                * ldm ) );
					}
					t0 = _mm512_cvtepi32_ps(
							_mm512_maskz_loadu_epi32( k2, ctemp ) );
					t0 = _mm512_mul_round_ps( t0, scl_fctr1,
							( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC )
						);
					acc_8 = _mm512_add_ps( t0, acc_8 );
				}
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
			__m512 t0 = _mm512_setzero_ps();

			// Even though different registers are used for scalar in column and
			// row major case, all those registers will contain the same value.
			// For column major, if m==1, then it means n=1 and scale_factor_len=1.
			if ( post_ops_list_temp->scale_factor_len == 1 )
			{
				scl_fctr1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'c' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'C' ) )
				{
					scl_fctr1 =
						_mm512_maskz_loadu_ps( k2,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + ( 0 * 16 ) );
				}
			}

			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if( ldm == 1 )
				{
					BF16_F32_MATRIX_MUL_LOAD( k2, t0, scl_fctr1, 0, 0 );
					acc_8  = _mm512_mul_round_ps( t0, acc_8,
							( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
				}
				else
				{
					bfloat16 ctemp[16];
					for( dim_t i = 0; i < mr0; i++ )
					{
						ctemp[i] = *( matptr +
						            ( ( post_ops_attr.post_op_c_i + i )
						                * ldm ) );
					}
					t0 = (__m512)_mm512_sllv_epi32
					     (
					       _mm512_cvtepi16_epi32
					       (
					         _mm256_maskz_loadu_epi16
					         (
					           k2 , ctemp
					         )
					       ), _mm512_set1_epi32( 16 )
					     );
					t0 = _mm512_mul_ps( t0, scl_fctr1 );
					acc_8  = _mm512_mul_round_ps( t0, acc_8,
							( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
				}
			}
			else if ( is_f32 == TRUE )
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if( ldm == 1 )
				{
					F32_MATRIX_MUL_LOAD( k2, t0, scl_fctr1, 0, 0 );
					acc_8  = _mm512_mul_round_ps( t0, acc_8,
							( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
				}
				else
				{
					float ctemp[16];
					for( dim_t i = 0; i < mr0; i++ )
					{
						ctemp[i] = *( matptr +
						            ( ( post_ops_attr.post_op_c_i + i )
						                * ldm ) );
					}
					t0 = _mm512_maskz_loadu_ps( k2, ctemp );
					t0 = _mm512_mul_ps( t0, scl_fctr1 );
					acc_8  = _mm512_mul_round_ps( t0, acc_8,
							( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
				}
			}
			else if ( is_s8 == TRUE )
			{
				int8_t* matptr = ( int8_t* )post_ops_list_temp->op_args1;

				if( ldm == 1 )
				{
					S8_F32_MATRIX_MUL_LOAD( k2, t0, scl_fctr1, 0, 0 )

					acc_8 = _mm512_mul_round_ps( t0, acc_8,
						( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
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
					t0 = _mm512_cvtepi32_ps(
							_mm512_cvtepi8_epi32
					            ( _mm_maskz_loadu_epi8( k2, ctemp ) ) );
					t0 = _mm512_mul_round_ps( t0, scl_fctr1,
						( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
					acc_8 = _mm512_mul_round_ps( t0, acc_8,
						( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
				}
			}
			else
			{
				int32_t* matptr = ( int32_t* )post_ops_list_temp->op_args1;

				if( ldm == 1 )
				{
					S32_F32_MATRIX_MUL_LOAD(k2, t0, scl_fctr1, 0, 0 );

					acc_8 = _mm512_mul_round_ps( t0, acc_8,
						( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
				}
				else
				{
					int32_t ctemp[16];
					for( dim_t i = 0; i < mr0; i++ )
					{
						ctemp[i] = *( matptr +
						            ( ( post_ops_attr.post_op_c_i + i )
						                * ldm ) );
					}
					t0 = _mm512_cvtepi32_ps(
							_mm512_maskz_loadu_epi32( k2, ctemp ) );
					t0 = _mm512_mul_round_ps( t0, scl_fctr1,
						( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
					acc_8 = _mm512_mul_round_ps( t0, acc_8,
						( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) );
				}
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

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_TANH_6x64:
		{
			__m512 dn, z, x, r2, r;
			__m512i q;

			TANHF_AVX512( acc_8, r, r2, x, z, dn, q );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_SIGMOID_6x64:
		{
			__m512 al_in, r, r2, z, dn;
			__m512i tmpout;

			SIGMOID_F32_AVX512_DEF( acc_8,  al_in, r, r2, z, dn, tmpout );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
		POST_OPS_6x64_DISABLE:
		{
			// Case where the output C matrix is s8 (downscaled) and
			// this is the final write for a given block within C.
			if ( post_ops_attr.buf_downscale != NULL )
			{
				if( post_ops_attr.rs_c_downscale == 1 )
				{
					if ( post_ops_attr.c_stor_type == S8 )
					{
						CVT_STORE_F32_S8_MASK( k2, acc_8, 0, 0 );
					}
					else if ( post_ops_attr.c_stor_type == U8 )
					{
						CVT_STORE_F32_U8_MASK( k2, acc_8, 0, 0 );
					}
					else if ( post_ops_attr.c_stor_type == BF16 )
					{
						CVT_STORE_F32_BF16_MASK( k2, acc_8, 0, 0 );
					}
					else if ( post_ops_attr.c_stor_type == F32 )
					{
						STORE_F32_MASK( k2, acc_8, 0, 0 );
					}
				}
				else
				{
					if ( post_ops_attr.c_stor_type == S8 )
					{
						int8_t ctemp[16];

						_mm512_mask_cvtsepi32_storeu_epi8 ( ctemp, k2,
								_mm512_cvtps_epi32( acc_8 ) );

						for (dim_t i = 0; i < mr0; i++)
						{
							 *( ( int8_t* )post_ops_attr.buf_downscale +
							 ( post_ops_attr.rs_c_downscale *
							 ( post_ops_attr.post_op_c_i + i ) ) ) = ctemp[i];
						}
					}
					else if ( post_ops_attr.c_stor_type == U8 )
					{
						uint8_t ctemp[16];

						_mm512_mask_cvtusepi32_storeu_epi8 ( ctemp, k2,
								_mm512_cvtps_epu32(
									_mm512_max_ps( acc_8, _mm512_set1_ps( 0 ) )
								) );

						for (dim_t i = 0; i < mr0; i++)
						{
							 *( ( uint8_t* )post_ops_attr.buf_downscale +
							 ( post_ops_attr.rs_c_downscale *
							 ( post_ops_attr.post_op_c_i + i ) ) ) = ctemp[i];
						}
					}
					else if ( post_ops_attr.c_stor_type == BF16 )
					{
						bfloat16 ctemp[16];

						CVT_STORE_F32_BF16_MASK_AVX2( acc_8, k2, ctemp );

						for (dim_t i = 0; i < mr0; i++)
						{
							 *( ( bfloat16* )post_ops_attr.buf_downscale +
							 ( post_ops_attr.rs_c_downscale *
							 ( post_ops_attr.post_op_c_i + i ) ) ) = ctemp[i];
						}
					}
					else if ( post_ops_attr.c_stor_type == F32 )
					{
						float ctemp[16];

						_mm512_mask_storeu_ps( ctemp, k2, acc_8 );

						for (dim_t i = 0; i < mr0; i++)
						{
							 *( ( float* )post_ops_attr.buf_downscale +
							 ( post_ops_attr.rs_c_downscale *
							 ( post_ops_attr.post_op_c_i + i ) ) ) = ctemp[i];
						}
					}
				}
			}
			else
			{
				if(rs_c == 1)
				{
					_mm512_mask_storeu_epi32(c_use, k2,
							_mm512_cvtps_epi32( acc_8 ) );
				}
				else
				{
					// Store ZMM8 into ctemp buffer and store back
					// element by element into output buffer at strides
					int32_t ctemp[16];
					_mm512_mask_storeu_epi32(ctemp, k2,
							_mm512_cvtps_epi32( acc_8 ) );
					for (dim_t i = 0; i < mr0; i++)
					{
						c_use[i * rs_c] = ctemp[i];
					}
				}
			}
			post_ops_attr.post_op_c_i += MR;
		}
	}
}

#endif // BLIS_ADDON_LPGEMM
