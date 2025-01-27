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
#include <string.h>
#include "blis.h"

#ifdef BLIS_ADDON_LPGEMM

#include "lpgemm_s32_kern_macros.h"
#include "lpgemm_s32_memcpy_macros.h"

// This file contains micro-kernels with extended MR for n fringe kernels.
// It was observed that increasing MR resulted in better multi-thread
// performance for inputs predominantly calling n fringe kernels. However
// slight regressions were observed in single thread performance.

// 12xlt16 int8o32 fringe kernel
__attribute__((aligned(64)))
LPGEMM_N_LT_NR0_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_12xlt16)
{
	static void* post_ops_labels[] =
				{
				  &&POST_OPS_12xLT16_DISABLE,
				  &&POST_OPS_BIAS_12xLT16,
				  &&POST_OPS_RELU_12xLT16,
				  &&POST_OPS_RELU_SCALE_12xLT16,
				  &&POST_OPS_GELU_TANH_12xLT16,
				  &&POST_OPS_GELU_ERF_12xLT16,
				  &&POST_OPS_CLIP_12xLT16,
				  &&POST_OPS_DOWNSCALE_12xLT16,
				  &&POST_OPS_MATRIX_ADD_12xLT16,
				  &&POST_OPS_SWISH_12xLT16,
				  &&POST_OPS_MATRIX_MUL_12xLT16,
				  &&POST_OPS_TANH_12xLT16,
				  &&POST_OPS_SIGMOID_12xLT16

				};
	dim_t MR = 12;
	dim_t m_full_pieces = m0 / MR;
	dim_t m_full_pieces_loop_limit = m_full_pieces * MR;
	dim_t m_partial_pieces = m0 % MR;

	dim_t k_full_pieces = k0 / 4;
	dim_t k_partial_pieces = k0 % 4;

	// B matrix storage.
	__m512i b0 = _mm512_setzero_epi32();

	// A matrix storage.
	__m512i a_int32_0 = _mm512_setzero_epi32();
	__m512i a_int32_1 = _mm512_setzero_epi32();
	__m512i a_int32_2 = _mm512_setzero_epi32();
	__m512i a_int32_3 = _mm512_setzero_epi32();
	__m512i a_int32_4 = _mm512_setzero_epi32();
	__m512i a_int32_5 = _mm512_setzero_epi32();
	__m512i a_int32_6 = _mm512_setzero_epi32();
	__m512i a_int32_7 = _mm512_setzero_epi32();
	__m512i a_int32_8 = _mm512_setzero_epi32();
	__m512i a_int32_9 = _mm512_setzero_epi32();
	__m512i a_int32_10 = _mm512_setzero_epi32();
	__m512i a_int32_11 = _mm512_setzero_epi32();

	for ( dim_t ir = 0; ir < m_full_pieces_loop_limit; ir += MR )
	{
		_mm_prefetch( b, _MM_HINT_T0 );
		_mm_prefetch( a + ( MR * ps_a ) + ( 0 * 16 ), _MM_HINT_T1 );

		// Registers to use for accumulating C.
		__m512i c_int32_0p0 = _mm512_setzero_epi32();

		__m512i c_int32_1p0 = _mm512_setzero_epi32();

		__m512i c_int32_2p0 = _mm512_setzero_epi32();

		__m512i c_int32_3p0 = _mm512_setzero_epi32();

		__m512i c_int32_4p0 = _mm512_setzero_epi32();

		__m512i c_int32_5p0 = _mm512_setzero_epi32();

		__m512i c_int32_6p0 = _mm512_setzero_epi32();

		__m512i c_int32_7p0 = _mm512_setzero_epi32();

		__m512i c_int32_8p0 = _mm512_setzero_epi32();

		__m512i c_int32_9p0 = _mm512_setzero_epi32();

		__m512i c_int32_10p0 = _mm512_setzero_epi32();

		__m512i c_int32_11p0 = _mm512_setzero_epi32();

		for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
		{
			// Load 4 rows with 16 extended elements each from B to 1 ZMM
			// registers. It is to be noted that the B matrix is packed for use
			// in vnni instructions and each load to ZMM register will have 4
			// elements along k direction and 16 elements across n directions,
			// so 4x16 elements to a ZMM register.
			b0 = _mm512_loadu_si512( b + ( rs_b * kr ) + ( cs_b * 0 ) );

			// Broadcast a[0,kr:kr+4].
			a_int32_0 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

			// Broadcast a[1,kr:kr+4].
			a_int32_1 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 1 ) + ( cs_a * kr ) ) );

			// Broadcast a[2,kr:kr+4].
			a_int32_2 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 2 ) + ( cs_a * kr ) ) );

			// Broadcast a[3,kr:kr+4].
			a_int32_3 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 3 ) + ( cs_a * kr ) ) );

			// Broadcast a[4,kr:kr+4].
			a_int32_4 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 4 ) + ( cs_a * kr ) ) );

			// Broadcast a[5,kr:kr+4].
			a_int32_5 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 5 ) + ( cs_a * kr ) ) );

			// Broadcast a[6,kr:kr+4].
			a_int32_6 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 6 ) + ( cs_a * kr ) ) );

			// Broadcast a[7,kr:kr+4].
			a_int32_7 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 7 ) + ( cs_a * kr ) ) );

			// Perform column direction mat-mul with k = 4.
			// c[0,0-15] = a[0,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_0p0 = _mm512_dpbusd_epi32( c_int32_0p0, a_int32_0, b0 );

			// Perform column direction mat-mul with k = 4.
			// c[1,0-15] = a[1,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_1p0 = _mm512_dpbusd_epi32( c_int32_1p0, a_int32_1, b0 );

			// Perform column direction mat-mul with k = 4.
			// c[2,0-15] = a[2,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_2p0 = _mm512_dpbusd_epi32( c_int32_2p0, a_int32_2, b0 );

			// Perform column direction mat-mul with k = 4.
			// c[3,0-15] = a[3,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_3p0 = _mm512_dpbusd_epi32( c_int32_3p0, a_int32_3, b0 );

			// Broadcast a[8,kr:kr+4].
			a_int32_8 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 8 ) + ( cs_a * kr ) ) );

			// Broadcast a[9,kr:kr+4].
			a_int32_9 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 9 ) + ( cs_a * kr ) ) );

			// Broadcast a[10,kr:kr+4].
			a_int32_10 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 10 ) + ( cs_a * kr ) ) );

			// Broadcast a[11,kr:kr+4].
			a_int32_11 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 11 ) + ( cs_a * kr ) ) );

			// Perform column direction mat-mul with k = 4.
			// c[4,0-15] = a[4,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_4p0 = _mm512_dpbusd_epi32( c_int32_4p0, a_int32_4, b0 );

			// Perform column direction mat-mul with k = 4.
			// c[5,0-15] = a[5,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_5p0 = _mm512_dpbusd_epi32( c_int32_5p0, a_int32_5, b0 );

			// Perform column direction mat-mul with k = 4.
			// c[6,0-15] = a[6,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_6p0 = _mm512_dpbusd_epi32( c_int32_6p0, a_int32_6, b0 );

			// Perform column direction mat-mul with k = 4.
			// c[7,0-15] = a[7,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_7p0 = _mm512_dpbusd_epi32( c_int32_7p0, a_int32_7, b0 );

			// Perform column direction mat-mul with k = 4.
			// c[8,0-15] = a[8,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_8p0 = _mm512_dpbusd_epi32( c_int32_8p0, a_int32_8, b0 );

			// Perform column direction mat-mul with k = 4.
			// c[9,0-15] = a[9,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_9p0 = _mm512_dpbusd_epi32( c_int32_9p0, a_int32_9, b0 );

			// Perform column direction mat-mul with k = 4.
			// c[10,0-15] = a[10,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_10p0 = _mm512_dpbusd_epi32( c_int32_10p0, a_int32_10, b0 );

			// Perform column direction mat-mul with k = 4.
			// c[11,0-15] = a[11,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_11p0 = _mm512_dpbusd_epi32( c_int32_11p0, a_int32_11, b0 );
		}
		__asm__(".p2align 6\n");
		// Handle k remainder.
		if ( k_partial_pieces > 0 )
		{
			__m128i a_kfringe_buf;
			__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - k_partial_pieces ) );

			b0 = _mm512_loadu_si512( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );

			// Broadcast a[0,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_0 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[0,0-15] = a[0,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_0p0 = _mm512_dpbusd_epi32( c_int32_0p0, a_int32_0, b0 );

			// Broadcast a[1,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 1 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_1 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[1,0-15] = a[1,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_1p0 = _mm512_dpbusd_epi32( c_int32_1p0, a_int32_1, b0 );

			// Broadcast a[2,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 2 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_2 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[2,0-15] = a[2,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_2p0 = _mm512_dpbusd_epi32( c_int32_2p0, a_int32_2, b0 );

			// Broadcast a[3,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 3 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_3 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[3,0-15] = a[3,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_3p0 = _mm512_dpbusd_epi32( c_int32_3p0, a_int32_3, b0 );

			// Broadcast a[4,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 4 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_4 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[4,0-15] = a[4,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_4p0 = _mm512_dpbusd_epi32( c_int32_4p0, a_int32_4, b0 );

			// Broadcast a[5,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 5 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_5 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[5,0-15] = a[5,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_5p0 = _mm512_dpbusd_epi32( c_int32_5p0, a_int32_5, b0 );

			// Broadcast a[6,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 6 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_6 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[6,0-15] = a[6,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_6p0 = _mm512_dpbusd_epi32( c_int32_6p0, a_int32_6, b0 );

			// Broadcast a[7,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 7 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_7 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[7,0-15] = a[7,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_7p0 = _mm512_dpbusd_epi32( c_int32_7p0, a_int32_7, b0 );

			// Broadcast a[8,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 8 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_8 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[8,0-15] = a[8,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_8p0 = _mm512_dpbusd_epi32( c_int32_8p0, a_int32_8, b0 );

			// Broadcast a[9,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 9 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_9 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[9,0-15] = a[9,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_9p0 = _mm512_dpbusd_epi32( c_int32_9p0, a_int32_9, b0 );

			// Broadcast a[10,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 10 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_10 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[10,0-15] = a[10,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_10p0 = _mm512_dpbusd_epi32( c_int32_10p0, a_int32_10, b0 );

			// Broadcast a[11,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 11 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_11 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[11,0-15] = a[11,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_11p0 = _mm512_dpbusd_epi32( c_int32_11p0, a_int32_11, b0 );
		}

		// Load alpha and beta
		__m512i selector1 = _mm512_set1_epi32( alpha );
		__m512i selector2 = _mm512_set1_epi32( beta );

		if ( alpha != 1 )
		{
			// Scale by alpha
			c_int32_0p0 = _mm512_mullo_epi32( selector1, c_int32_0p0 );

			c_int32_1p0 = _mm512_mullo_epi32( selector1, c_int32_1p0 );

			c_int32_2p0 = _mm512_mullo_epi32( selector1, c_int32_2p0 );

			c_int32_3p0 = _mm512_mullo_epi32( selector1, c_int32_3p0 );

			c_int32_4p0 = _mm512_mullo_epi32( selector1, c_int32_4p0 );

			c_int32_5p0 = _mm512_mullo_epi32( selector1, c_int32_5p0 );

			c_int32_6p0 = _mm512_mullo_epi32( selector1, c_int32_6p0 );

			c_int32_7p0 = _mm512_mullo_epi32( selector1, c_int32_7p0 );

			c_int32_8p0 = _mm512_mullo_epi32( selector1, c_int32_8p0 );

			c_int32_9p0 = _mm512_mullo_epi32( selector1, c_int32_9p0 );

			c_int32_10p0 = _mm512_mullo_epi32( selector1, c_int32_10p0 );

			c_int32_11p0 = _mm512_mullo_epi32( selector1, c_int32_11p0 );
		}

		// Scale C by beta.
		if ( beta != 0 )
		{
			if ( ( post_ops_attr.buf_downscale != NULL ) &&
				 ( post_ops_attr.is_first_k == TRUE ) )
			{
				__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );
				if( post_ops_attr.c_stor_type == S8 )
				{
					// c[0,0-15]
					S8_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_0p0, 0, 0, \
									selector1, selector2 );

					// c[1,0-15]
					S8_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_1p0, 1, 0, \
									selector1, selector2 );

					// c[2,0-15]
					S8_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_2p0, 2, 0, \
									selector1, selector2 );

					// c[3,0-15]
					S8_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_3p0, 3, 0, \
									selector1, selector2 );

					// c[4,0-15]
					S8_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_4p0, 4, 0, \
									selector1, selector2 );

					// c[5,0-15]
					S8_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_5p0, 5, 0, \
									selector1, selector2 );

					// c[6,0-15]
					S8_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_6p0, 6, 0, \
									selector1, selector2 );

					// c[7,0-15]
					S8_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_7p0, 7, 0, \
									selector1, selector2 );

					// c[8,0-15]
					S8_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_8p0, 8, 0, \
									selector1, selector2 );

					// c[9,0-15]
					S8_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_9p0, 9, 0, \
									selector1, selector2 );

					// c[10,0-15]
					S8_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_10p0, 10, 0, \
									selector1, selector2 );

					// c[11,0-15]
					S8_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_11p0, 11, 0, \
									selector1, selector2 );
				}
				else if( post_ops_attr.c_stor_type == U8 )
				{
					// c[0,0-15]
					U8_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_0p0, 0, 0, \
									selector1, selector2 );

					// c[1,0-15]
					U8_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_1p0, 1, 0, \
									selector1, selector2 );

					// c[2,0-15]
					U8_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_2p0, 2, 0, \
									selector1, selector2 );

					// c[3,0-15]
					U8_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_3p0, 3, 0, \
									selector1, selector2 );

					// c[4,0-15]
					U8_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_4p0, 4, 0, \
									selector1, selector2 );

					// c[5,0-15]
					U8_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_5p0, 5, 0, \
									selector1, selector2 );

					// c[6,0-15]
					U8_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_6p0, 6, 0, \
									selector1, selector2 );

					// c[7,0-15]
					U8_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_7p0, 7, 0, \
									selector1, selector2 );

					// c[8,0-15]
					U8_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_8p0, 8, 0, \
									selector1, selector2 );

					// c[9,0-15]
					U8_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_9p0, 9, 0, \
									selector1, selector2 );

					// c[10,0-15]
					U8_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_10p0, 10, 0, \
									selector1, selector2 );

					// c[11,0-15]
					U8_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_11p0, 11, 0, \
									selector1, selector2 );
				}
				else if( post_ops_attr.c_stor_type == BF16 )
				{
					// c[0,0-15]
					BF16_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_0p0, 0, 0, \
									selector1, selector2 );

					// c[1,0-15]
					BF16_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_1p0, 1, 0, \
									selector1, selector2 );

					// c[2,0-15]
					BF16_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_2p0, 2, 0, \
									selector1, selector2 );

					// c[3,0-15]
					BF16_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_3p0, 3, 0, \
									selector1, selector2 );

					// c[4,0-15]
					BF16_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_4p0, 4, 0, \
									selector1, selector2 );

					// c[5,0-15]
					BF16_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_5p0, 5, 0, \
									selector1, selector2 );

					// c[6,0-15]
					BF16_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_6p0, 6, 0, \
									selector1, selector2 );

					// c[7,0-15]
					BF16_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_7p0, 7, 0, \
									selector1, selector2 );

					// c[8,0-15]
					BF16_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_8p0, 8, 0, \
									selector1, selector2 );

					// c[9,0-15]
					BF16_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_9p0, 9, 0, \
									selector1, selector2 );

					// c[10,0-15]
					BF16_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_10p0, 10, 0, \
									selector1, selector2 );

					// c[11,0-15]
					BF16_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_11p0, 11, 0, \
									selector1, selector2 );
				}
				else if( post_ops_attr.c_stor_type == F32 )
				{
					// c[0,0-15]
					F32_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_0p0, 0, 0, \
									selector1, selector2 );

					// c[1,0-15]
					F32_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_1p0, 1, 0, \
									selector1, selector2 );

					// c[2,0-15]
					F32_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_2p0, 2, 0, \
									selector1, selector2 );

					// c[3,0-15]
					F32_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_3p0, 3, 0, \
									selector1, selector2 );

					// c[4,0-15]
					F32_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_4p0, 4, 0, \
									selector1, selector2 );

					// c[5,0-15]
					F32_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_5p0, 5, 0, \
									selector1, selector2 );

					// c[6,0-15]
					F32_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_6p0, 6, 0, \
									selector1, selector2 );

					// c[7,0-15]
					F32_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_7p0, 7, 0, \
									selector1, selector2 );

					// c[8,0-15]
					F32_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_8p0, 8, 0, \
									selector1, selector2 );

					// c[9,0-15]
					F32_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_9p0, 9, 0, \
									selector1, selector2 );

					// c[10,0-15]
					F32_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_10p0, 10, 0, \
									selector1, selector2 );

					// c[11,0-15]
					F32_S32_BETA_OP_NLT16F_MASK( load_mask, c_int32_11p0, 11, 0, \
									selector1, selector2 );
				}
			}
			else
			{
				__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

				// c[0,0-15]
				S32_S32_BETA_OP_NLT16F_MASK(c, load_mask, c_int32_0p0, ir, 0, 0, \
								selector1, selector2);

				// c[1,0-15]
				S32_S32_BETA_OP_NLT16F_MASK(c, load_mask, c_int32_1p0, ir, 1, 0, \
								selector1, selector2);

				// c[2,0-15]
				S32_S32_BETA_OP_NLT16F_MASK(c, load_mask, c_int32_2p0, ir, 2, 0, \
								selector1, selector2);

				// c[3,0-15]
				S32_S32_BETA_OP_NLT16F_MASK(c, load_mask, c_int32_3p0, ir, 3, 0, \
								selector1, selector2);

				// c[4,0-15]
				S32_S32_BETA_OP_NLT16F_MASK(c, load_mask, c_int32_4p0, ir, 4, 0, \
								selector1, selector2);

				// c[5,0-15]
				S32_S32_BETA_OP_NLT16F_MASK(c, load_mask, c_int32_5p0, ir, 5, 0, \
								selector1, selector2);

				// c[6,0-15]
				S32_S32_BETA_OP_NLT16F_MASK(c, load_mask, c_int32_6p0, ir, 6, 0, \
								selector1, selector2);

				// c[7,0-15]
				S32_S32_BETA_OP_NLT16F_MASK(c, load_mask, c_int32_7p0, ir, 7, 0, \
								selector1, selector2);

				// c[8,0-15]
				S32_S32_BETA_OP_NLT16F_MASK(c, load_mask, c_int32_8p0, ir, 8, 0, \
								selector1, selector2);

				// c[9,0-15]
				S32_S32_BETA_OP_NLT16F_MASK(c, load_mask, c_int32_9p0, ir, 9, 0, \
								selector1, selector2);

				// c[10,0-15]
				S32_S32_BETA_OP_NLT16F_MASK(c, load_mask, c_int32_10p0, ir, 10, 0, \
								selector1, selector2);

				// c[11,0-15]
				S32_S32_BETA_OP_NLT16F_MASK(c, load_mask, c_int32_11p0, ir, 11, 0, \
								selector1, selector2);
			}
		}

		__m512 acc_00 = _mm512_setzero_ps();
		__m512 acc_10 = _mm512_setzero_ps();
		__m512 acc_20 = _mm512_setzero_ps();
		__m512 acc_30 = _mm512_setzero_ps();
		__m512 acc_40 = _mm512_setzero_ps();
		__m512 acc_50 = _mm512_setzero_ps();
		__m512 acc_60 = _mm512_setzero_ps();
		__m512 acc_70 = _mm512_setzero_ps();
		__m512 acc_80 = _mm512_setzero_ps();
		__m512 acc_90 = _mm512_setzero_ps();
		__m512 acc_100 = _mm512_setzero_ps();
		__m512 acc_110 = _mm512_setzero_ps();
		CVT_ACCUM_REG_INT_TO_FLOAT_12ROWS_XCOL(acc_, c_int32_, 1);

        // Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_12xLT16:
		{
			__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );
			__m512 b0 = _mm512_setzero_ps();

			if ( post_ops_list_temp->stor_type == BF16 )
			{
				BF16_F32_BIAS_LOAD(b0, bias_mask, 0);
			}
			else if ( post_ops_list_temp->stor_type == S8 )
			{
				S8_F32_BIAS_LOAD(b0, bias_mask, 0);
			}
			else if ( post_ops_list_temp->stor_type == S32 )
			{
				S32_F32_BIAS_LOAD(b0, bias_mask, 0);
			}
			else
			{
				b0 = _mm512_maskz_loadu_ps ( bias_mask,
						( ( float* )post_ops_list_temp->op_args1 ) +
						post_ops_attr.post_op_c_j + ( 0 * 16 ) );
			}

			// c[0,0-15]
			acc_00 = _mm512_add_ps( b0, acc_00 );

			// c[1,0-15]
			acc_10 = _mm512_add_ps( b0, acc_10 );

			// c[2,0-15]
			acc_20 = _mm512_add_ps( b0, acc_20 );

			// c[3,0-15]
			acc_30 = _mm512_add_ps( b0, acc_30 );

			// c[4,0-15]
			acc_40 = _mm512_add_ps( b0, acc_40 );

			// c[5,0-15]
			acc_50 = _mm512_add_ps( b0, acc_50 );

			// c[6,0-15]
			acc_60 = _mm512_add_ps( b0, acc_60 );

			// c[7,0-15]
			acc_70 = _mm512_add_ps( b0, acc_70 );

			// c[8,0-15]
			acc_80 = _mm512_add_ps( b0, acc_80 );

			// c[9,0-15]
			acc_90 = _mm512_add_ps( b0, acc_90 );

			// c[10,0-15]
			acc_100 = _mm512_add_ps( b0, acc_100 );

			// c[11,0-15]
			acc_110 = _mm512_add_ps( b0, acc_110 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_12xLT16:
		{
			__m512 zero = _mm512_setzero_ps();

			// c[0,0-15]
			acc_00 = _mm512_max_ps( zero, acc_00 );

			// c[1,0-15]
			acc_10 = _mm512_max_ps( zero, acc_10 );

			// c[2,0-15]
			acc_20 = _mm512_max_ps( zero, acc_20 );

			// c[3,0-15]
			acc_30 = _mm512_max_ps( zero, acc_30 );

			// c[4,0-15]
			acc_40 = _mm512_max_ps( zero, acc_40 );

			// c[5,0-15]
			acc_50 = _mm512_max_ps( zero, acc_50 );

			// c[6,0-15]
			acc_60 = _mm512_max_ps( zero, acc_60 );

			// c[7,0-15]
			acc_70 = _mm512_max_ps( zero, acc_70 );

			// c[8,0-15]
			acc_80 = _mm512_max_ps( zero, acc_80 );

			// c[9,0-15]
			acc_90 = _mm512_max_ps( zero, acc_90 );

			// c[10,0-15]
			acc_100 = _mm512_max_ps( zero, acc_100 );

			// c[11,0-15]
			acc_110 = _mm512_max_ps( zero, acc_110 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_12xLT16:
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

			// c[0, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_00)

			// c[1, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_10)

			// c[2, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_20)

			// c[3, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_30)

			// c[4, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_40)

			// c[5, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_50)

			// c[6, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_60)

			// c[7, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_70)

			// c[8, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_80)

			// c[9, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_90)

			// c[10, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_100)

			// c[11, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_110)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_TANH_12xLT16:
		{
			__m512 dn, z, x, r2, r, y;
			__m512i tmpout;

			// c[0, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_00, y, r, r2, x, z, dn, tmpout)

			// c[1, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_10, y, r, r2, x, z, dn, tmpout)

			// c[2, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_20, y, r, r2, x, z, dn, tmpout)

			// c[3, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_30, y, r, r2, x, z, dn, tmpout)

			// c[4, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_40, y, r, r2, x, z, dn, tmpout)

			// c[5, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_50, y, r, r2, x, z, dn, tmpout)

			// c[6, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_60, y, r, r2, x, z, dn, tmpout)

			// c[7, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_70, y, r, r2, x, z, dn, tmpout)

			// c[8, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_80, y, r, r2, x, z, dn, tmpout)

			// c[9, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_90, y, r, r2, x, z, dn, tmpout)

			// c[10, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_100, y, r, r2, x, z, dn, tmpout)

			// c[11, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_110, y, r, r2, x, z, dn, tmpout)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_ERF_12xLT16:
		{
			__m512 y, r, r2;

			// c[0, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_00, y, r, r2)

			// c[1, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_10, y, r, r2)

			// c[2, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_20, y, r, r2)

			// c[3, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_30, y, r, r2)

			// c[4, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_40, y, r, r2)

			// c[5, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_50, y, r, r2)

			// c[6, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_60, y, r, r2)

			// c[7, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_70, y, r, r2)

			// c[8, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_80, y, r, r2)

			// c[9, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_90, y, r, r2)

			// c[10, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_100, y, r, r2)

			// c[11, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_110, y, r, r2)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_CLIP_12xLT16:
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

			// c[0, 0-15]
			CLIP_F32_AVX512(acc_00, min, max)

			// c[1, 0-15]
			CLIP_F32_AVX512(acc_10, min, max)

			// c[2, 0-15]
			CLIP_F32_AVX512(acc_20, min, max)

			// c[3, 0-15]
			CLIP_F32_AVX512(acc_30, min, max)

			// c[4, 0-15]
			CLIP_F32_AVX512(acc_40, min, max)

			// c[5, 0-15]
			CLIP_F32_AVX512(acc_50, min, max)

			// c[6, 0-15]
			CLIP_F32_AVX512(acc_60, min, max)

			// c[7, 0-15]
			CLIP_F32_AVX512(acc_70, min, max)

			// c[8, 0-15]
			CLIP_F32_AVX512(acc_80, min, max)

			// c[9, 0-15]
			CLIP_F32_AVX512(acc_90, min, max)

			// c[10, 0-15]
			CLIP_F32_AVX512(acc_100, min, max)

			// c[11, 0-15]
			CLIP_F32_AVX512(acc_110, min, max)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_DOWNSCALE_12xLT16:
		{
			__m512 scale0 = _mm512_setzero_ps();
			// Typecast without data modification, safe operation.
			__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				scale0 = _mm512_maskz_loadu_ps
							(
							  load_mask,
							  ( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j )
							);
			}
			else if ( post_ops_list_temp->scale_factor_len == 1 )
			{
				scale0 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			}

			// Need to ensure sse not used to avoid avx512 -> sse transition.
			__m128i zero_point = _mm512_castsi512_si128( _mm512_setzero_si512() );
			if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
			{
				zero_point = _mm_maskz_loadu_epi8
							(
							  load_mask,
							  ( ( int8_t* )post_ops_list_temp->op_args1 +
								post_ops_attr.post_op_c_j )
							);
			}
			else if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
			{
				zero_point = _mm_maskz_set1_epi8( 0xFFFF,
							*( ( int8_t* )post_ops_list_temp->op_args1 ) );
			}

			// c[0, 0-15]
			CVT_MULRND_F32(acc_00,scale0,zero_point);

			// c[1, 0-15]
			CVT_MULRND_F32(acc_10,scale0,zero_point);

			// c[2, 0-15]
			CVT_MULRND_F32(acc_20,scale0,zero_point);

			// c[3, 0-15]
			CVT_MULRND_F32(acc_30,scale0,zero_point);

			// c[4, 0-15]
			CVT_MULRND_F32(acc_40,scale0,zero_point);

			// c[5, 0-15]
			CVT_MULRND_F32(acc_50,scale0,zero_point);

			// c[6, 0-15]
			CVT_MULRND_F32(acc_60,scale0,zero_point);

			// c[7, 0-15]
			CVT_MULRND_F32(acc_70,scale0,zero_point);

			// c[8, 0-15]
			CVT_MULRND_F32(acc_80,scale0,zero_point);

			// c[9, 0-15]
			CVT_MULRND_F32(acc_90,scale0,zero_point);

			// c[10, 0-15]
			CVT_MULRND_F32(acc_100,scale0,zero_point);

			// c[11, 0-15]
			CVT_MULRND_F32(acc_110,scale0,zero_point);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_ADD_12xLT16:
		{
			__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );
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
			__m512 scl_fctr5 = _mm512_setzero_ps();
			__m512 scl_fctr6 = _mm512_setzero_ps();
			__m512 scl_fctr7 = _mm512_setzero_ps();
			__m512 scl_fctr8 = _mm512_setzero_ps();
			__m512 scl_fctr9 = _mm512_setzero_ps();
			__m512 scl_fctr10 = _mm512_setzero_ps();
			__m512 scl_fctr11 = _mm512_setzero_ps();
			__m512 scl_fctr12 = _mm512_setzero_ps();
			__m512 t0 = _mm512_setzero_ps();

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
				scl_fctr6 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr7 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr8 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr9 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr10 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr11 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr12 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
						_mm512_maskz_loadu_ps( load_mask,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
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
					scl_fctr6 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 5 ) );
					scl_fctr7 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 6 ) );
					scl_fctr8 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 7 ) );
					scl_fctr9 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 8 ) );
					scl_fctr10 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 9 ) );
					scl_fctr11 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 10 ) );
					scl_fctr12 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 11 ) );
				}
			}

			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15]
					BF16_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,0);

					// c[1:0-15]
					BF16_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,1);

					// c[2:0-15]
					BF16_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,2);

					// c[3:0-15]
					BF16_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,3);

					// c[4:0-15]
					BF16_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,4);

					// c[5:0-15]
					BF16_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,5);

					// c[6:0-15]
					BF16_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,6);

					// c[7:0-15]
					BF16_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,7);

					// c[8:0-15]
					BF16_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,8);

					// c[9:0-15]
					BF16_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,9);

					// c[10:0-15]
					BF16_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,10);

					// c[11:0-15]
					BF16_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,11);
				}
				else
				{
					// c[0:0-15]
					BF16_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,0);

					// c[1:0-15]
					BF16_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr2,1);

					// c[2:0-15]
					BF16_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr3,2);

					// c[3:0-15]
					BF16_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr4,3);

					// c[4:0-15]
					BF16_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr5,4);

					// c[5:0-15]
					BF16_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr6,5);

					// c[6:0-15]
					BF16_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr7,6);

					// c[7:0-15]
					BF16_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr8,7);

					// c[8:0-15]
					BF16_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr9,8);

					// c[9:0-15]
					BF16_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr10,9);

					// c[10:0-15]
					BF16_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr11,10);

					// c[11:0-15]
					BF16_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr12,11);
				}
			}
			else if ( is_f32 == TRUE )
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15]
					F32_ONLY_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,0);

					// c[1:0-15]
					F32_ONLY_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,1);

					// c[2:0-15]
					F32_ONLY_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,2);

					// c[3:0-15]
					F32_ONLY_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,3);

					// c[4:0-15]
					F32_ONLY_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,4);

					// c[5:0-15]
					F32_ONLY_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,5);

					// c[6:0-15]
					F32_ONLY_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,6);

					// c[7:0-15]
					F32_ONLY_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,7);

					// c[8:0-15]
					F32_ONLY_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,8);

					// c[9:0-15]
					F32_ONLY_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,9);

					// c[10:0-15]
					F32_ONLY_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,10);

					// c[11:0-15]
					F32_ONLY_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,11);
				}
				else
				{
					// c[0:0-15]
					F32_ONLY_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,0);

					// c[1:0-15]
					F32_ONLY_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr2,1);

					// c[2:0-15]
					F32_ONLY_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr3,2);

					// c[3:0-15]
					F32_ONLY_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr4,3);

					// c[4:0-15]
					F32_ONLY_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr5,4);

					// c[5:0-15]
					F32_ONLY_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr6,5);

					// c[6:0-15]
					F32_ONLY_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr7,6);

					// c[7:0-15]
					F32_ONLY_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr8,7);

					// c[8:0-15]
					F32_ONLY_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr9,8);

					// c[9:0-15]
					F32_ONLY_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr10,9);

					// c[10:0-15]
					F32_ONLY_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr11,10);

					// c[11:0-15]
					F32_ONLY_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr12,11);
				}
			}
			else if ( is_s8 == TRUE )
			{
				int8_t* matptr = ( int8_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15]
					S8_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,0);

					// c[1:0-15]
					S8_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,1);

					// c[2:0-15]
					S8_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,2);

					// c[3:0-15]
					S8_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,3);

					// c[4:0-15]
					S8_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,4);

					// c[5:0-15]
					S8_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,5);

					// c[6:0-15]
					S8_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,6);

					// c[7:0-15]
					S8_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,7);

					// c[8:0-15]
					S8_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,8);

					// c[9:0-15]
					S8_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,9);

					// c[10:0-15]
					S8_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,10);

					// c[11:0-15]
					S8_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,11);
				}
				else
				{
					// c[0:0-15]
					S8_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,0);

					// c[1:0-15]
					S8_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr2,1);

					// c[2:0-15]
					S8_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr3,2);

					// c[3:0-15]
					S8_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr4,3);

					// c[4:0-15]
					S8_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr5,4);

					// c[5:0-15]
					S8_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr6,5);

					// c[6:0-15]
					S8_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr7,6);

					// c[7:0-15]
					S8_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr8,7);

					// c[8:0-15]
					S8_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr9,8);

					// c[9:0-15]
					S8_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr10,9);

					// c[10:0-15]
					S8_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr11,10);

					// c[11:0-15]
					S8_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr12,11);
				}
			}
			else
			{
				int32_t* matptr = ( int32_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15]
					S32_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,0);

					// c[1:0-15]
					S32_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,1);

					// c[2:0-15]
					S32_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,2);

					// c[3:0-15]
					S32_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,3);

					// c[4:0-15]
					S32_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,4);

					// c[5:0-15]
					S32_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,5);

					// c[6:0-15]
					S32_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,6);

					// c[7:0-15]
					S32_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,7);

					// c[8:0-15]
					S32_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,8);

					// c[9:0-15]
					S32_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,9);

					// c[10:0-15]
					S32_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,10);

					// c[11:0-15]
					S32_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,11);
				}
				else
				{
					// c[0:0-15]
					S32_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr1,0);

					// c[1:0-15]
					S32_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr2,1);

					// c[2:0-15]
					S32_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr3,2);

					// c[3:0-15]
					S32_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr4,3);

					// c[4:0-15]
					S32_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr5,4);

					// c[5:0-15]
					S32_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr6,5);

					// c[6:0-15]
					S32_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr7,6);

					// c[7:0-15]
					S32_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr8,7);

					// c[8:0-15]
					S32_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr9,8);

					// c[9:0-15]
					S32_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr10,9);

					// c[10:0-15]
					S32_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr10,10);

					// c[11:0-15]
					S32_F32_MATRIX_ADD_1COL_PAR(load_mask,t0,scl_fctr12,11);
				}
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_MUL_12xLT16:
		{
			__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );
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
			__m512 scl_fctr5 = _mm512_setzero_ps();
			__m512 scl_fctr6 = _mm512_setzero_ps();
			__m512 scl_fctr7 = _mm512_setzero_ps();
			__m512 scl_fctr8 = _mm512_setzero_ps();
			__m512 scl_fctr9 = _mm512_setzero_ps();
			__m512 scl_fctr10 = _mm512_setzero_ps();
			__m512 scl_fctr11 = _mm512_setzero_ps();
			__m512 scl_fctr12 = _mm512_setzero_ps();
			__m512 t0 = _mm512_setzero_ps();

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
				scl_fctr6 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr7 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr8 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr9 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr10 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr11 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr12 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
						_mm512_maskz_loadu_ps( load_mask,
								( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
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
					scl_fctr6 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 5 ) );
					scl_fctr7 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 6 ) );
					scl_fctr8 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 7 ) );
					scl_fctr9 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 8 ) );
					scl_fctr10 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 9 ) );
					scl_fctr11 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 10 ) );
					scl_fctr12 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 11 ) );
				}
			}

			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,0);

					// c[1:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,1);

					// c[2:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,2);

					// c[3:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,3);

					// c[4:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,4);

					// c[5:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,5);

					// c[6:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,6);

					// c[7:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,7);

					// c[8:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,8);

					// c[9:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,9);

					// c[10:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,10);

					// c[11:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,11);
				}
				else
				{
					// c[0:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,0);

					// c[1:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr2,1);

					// c[2:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr3,2);

					// c[3:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr4,3);

					// c[4:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr5,4);

					// c[5:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr6,5);

					// c[6:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr7,6);

					// c[7:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr8,7);

					// c[8:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr9,8);

					// c[9:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr10,9);

					// c[10:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr11,10);

					// c[11:0-15]
					BF16_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr12,11);
				}
			}
			else if ( is_f32 == TRUE )
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15]
					F32_U8S8_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,0);

					// c[1:0-15]
					F32_U8S8_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,1);

					// c[2:0-15]
					F32_U8S8_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,2);

					// c[3:0-15]
					F32_U8S8_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,3);

					// c[4:0-15]
					F32_U8S8_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,4);

					// c[5:0-15]
					F32_U8S8_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,5);

					// c[6:0-15]
					F32_U8S8_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,6);

					// c[7:0-15]
					F32_U8S8_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,7);

					// c[8:0-15]
					F32_U8S8_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,8);

					// c[9:0-15]
					F32_U8S8_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,9);

					// c[10:0-15]
					F32_U8S8_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,10);

					// c[11:0-15]
					F32_U8S8_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,11);
				}
				else
				{
					// c[0:0-15]
					F32_U8S8_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,0);

					// c[1:0-15]
					F32_U8S8_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr2,1);

					// c[2:0-15]
					F32_U8S8_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr3,2);

					// c[3:0-15]
					F32_U8S8_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr4,3);

					// c[4:0-15]
					F32_U8S8_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr5,4);

					// c[5:0-15]
					F32_U8S8_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr6,5);

					// c[6:0-15]
					F32_U8S8_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr7,6);

					// c[7:0-15]
					F32_U8S8_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr8,7);

					// c[8:0-15]
					F32_U8S8_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr9,8);

					// c[9:0-15]
					F32_U8S8_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr10,9);

					// c[10:0-15]
					F32_U8S8_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr11,10);

					// c[11:0-15]
					F32_U8S8_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr12,11);
				}
			}
			else if ( is_s8 == TRUE )
			{
				int8_t* matptr = ( int8_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15]
					S8_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,0);

					// c[1:0-15]
					S8_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,1);

					// c[2:0-15]
					S8_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,2);

					// c[3:0-15]
					S8_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,3);

					// c[4:0-15]
					S8_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,4);

					// c[5:0-15]
					S8_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,5);

					// c[6:0-15]
					S8_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,6);

					// c[7:0-15]
					S8_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,7);

					// c[8:0-15]
					S8_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,8);

					// c[9:0-15]
					S8_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,9);

					// c[10:0-15]
					S8_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,10);

					// c[11:0-15]
					S8_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,11);
				}
				else
				{
					// c[0:0-15]
					S8_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,0);

					// c[1:0-15]
					S8_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr2,1);

					// c[2:0-15]
					S8_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr3,2);

					// c[3:0-15]
					S8_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr4,3);

					// c[4:0-15]
					S8_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr5,4);

					// c[5:0-15]
					S8_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr6,5);

					// c[6:0-15]
					S8_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr7,6);

					// c[7:0-15]
					S8_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr8,7);

					// c[8:0-15]
					S8_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr9,8);

					// c[9:0-15]
					S8_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr10,9);

					// c[10:0-15]
					S8_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr11,10);

					// c[11:0-15]
					S8_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr12,11);
				}
			}
			else
			{
				int32_t* matptr = ( int32_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15]
					S32_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,0);

					// c[1:0-15]
					S32_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,1);

					// c[2:0-15]
					S32_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,2);

					// c[3:0-15]
					S32_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,3);

					// c[4:0-15]
					S32_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,4);

					// c[5:0-15]
					S32_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,5);

					// c[6:0-15]
					S32_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,6);

					// c[7:0-15]
					S32_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,7);

					// c[8:0-15]
					S32_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,8);

					// c[9:0-15]
					S32_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,9);

					// c[10:0-15]
					S32_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,10);

					// c[11:0-15]
					S32_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,11);
				}
				else
				{
					// c[0:0-15]
					S32_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr1,0);

					// c[1:0-15]
					S32_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr2,1);

					// c[2:0-15]
					S32_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr3,2);

					// c[3:0-15]
					S32_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr4,3);

					// c[4:0-15]
					S32_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr5,4);

					// c[5:0-15]
					S32_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr6,5);

					// c[6:0-15]
					S32_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr7,6);

					// c[7:0-15]
					S32_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr8,7);

					// c[8:0-15]
					S32_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr9,8);

					// c[9:0-15]
					S32_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr10,9);

					// c[10:0-15]
					S32_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr10,10);

					// c[11:0-15]
					S32_F32_MATRIX_MUL_1COL_PAR(load_mask,t0,scl_fctr12,11);
				}
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SWISH_12xLT16:
		{
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

			__m512 al_in, r, r2, z, dn;
			__m512i temp;

			// c[0, 0-15]
			SWISH_F32_AVX512_DEF(acc_00, scale, al_in, r, r2, z, dn, temp);

			// c[1, 0-15]
			SWISH_F32_AVX512_DEF(acc_10, scale, al_in, r, r2, z, dn, temp);

			// c[2, 0-15]
			SWISH_F32_AVX512_DEF(acc_20, scale, al_in, r, r2, z, dn, temp);

			// c[3, 0-15]
			SWISH_F32_AVX512_DEF(acc_30, scale, al_in, r, r2, z, dn, temp);

			// c[4, 0-15]
			SWISH_F32_AVX512_DEF(acc_40, scale, al_in, r, r2, z, dn, temp);

			// c[5, 0-15]
			SWISH_F32_AVX512_DEF(acc_50, scale, al_in, r, r2, z, dn, temp);

			// c[6, 0-15]
			SWISH_F32_AVX512_DEF(acc_60, scale, al_in, r, r2, z, dn, temp);

			// c[7, 0-15]
			SWISH_F32_AVX512_DEF(acc_70, scale, al_in, r, r2, z, dn, temp);

			// c[8, 0-15]
			SWISH_F32_AVX512_DEF(acc_80, scale, al_in, r, r2, z, dn, temp);

			// c[9, 0-15]
			SWISH_F32_AVX512_DEF(acc_90, scale, al_in, r, r2, z, dn, temp);

			// c[10, 0-15]
			SWISH_F32_AVX512_DEF(acc_100, scale, al_in, r, r2, z, dn, temp);

			// c[11, 0-15]
			SWISH_F32_AVX512_DEF(acc_110, scale, al_in, r, r2, z, dn, temp);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_TANH_12xLT16:
		{
			__m512 dn, z, x, r2, r;
			__m512i q;

			// c[0, 0-15]
			TANHF_AVX512(acc_00, r, r2, x, z, dn, q)

			// c[1, 0-15]
			TANHF_AVX512(acc_10, r, r2, x, z, dn, q)

			// c[2, 0-15]
			TANHF_AVX512(acc_20, r, r2, x, z, dn, q)

			// c[3, 0-15]
			TANHF_AVX512(acc_30, r, r2, x, z, dn, q)

			// c[4, 0-15]
			TANHF_AVX512(acc_40, r, r2, x, z, dn, q)

			// c[5, 0-15]
			TANHF_AVX512(acc_50, r, r2, x, z, dn, q)

			// c[6, 0-15]
			TANHF_AVX512(acc_60, r, r2, x, z, dn, q)

			// c[7, 0-15]
			TANHF_AVX512(acc_70, r, r2, x, z, dn, q)

			// c[8, 0-15]
			TANHF_AVX512(acc_80, r, r2, x, z, dn, q)

			// c[9, 0-15]
			TANHF_AVX512(acc_90, r, r2, x, z, dn, q)

			// c[10, 0-15]
			TANHF_AVX512(acc_100, r, r2, x, z, dn, q)

			// c[11, 0-15]
			TANHF_AVX512(acc_110, r, r2, x, z, dn, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SIGMOID_12xLT16:
		{
			__m512 al_in, r, r2, z, dn;
			__m512i tmpout;

			// c[0, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_00, al_in, r, r2, z, dn, tmpout);

			// c[1, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_10, al_in, r, r2, z, dn, tmpout);

			// c[2, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_20, al_in, r, r2, z, dn, tmpout);

			// c[3, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_30, al_in, r, r2, z, dn, tmpout);

			// c[4, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_40, al_in, r, r2, z, dn, tmpout);

			// c[5, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_50, al_in, r, r2, z, dn, tmpout);

			// c[6, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_60, al_in, r, r2, z, dn, tmpout);

			// c[7, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_70, al_in, r, r2, z, dn, tmpout);

			// c[8, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_80, al_in, r, r2, z, dn, tmpout);

			// c[9, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_90, al_in, r, r2, z, dn, tmpout);

			// c[10, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_100, al_in, r, r2, z, dn, tmpout);

			// c[11, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_110, al_in, r, r2, z, dn, tmpout);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_12xLT16_DISABLE:
		;

		// Store the results.
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
			 ( post_ops_attr.is_last_k == TRUE ) )
		{
			__mmask16 mask_all1 = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );
			if ( post_ops_attr.c_stor_type == S8)
			{
				// Store the results in downscaled type (int8 instead of int32).
				// c[0,0-15]
				CVT_STORE_F32_S8(acc_00,0,0);

				// c[1,0-15]
				CVT_STORE_F32_S8(acc_10,1,0);

				// c[2,0-15]
				CVT_STORE_F32_S8(acc_20,2,0);

				// c[3,0-15]
				CVT_STORE_F32_S8(acc_30,3,0);

				// c[4,0-15]
				CVT_STORE_F32_S8(acc_40,4,0);

				// c[5,0-15]
				CVT_STORE_F32_S8(acc_50,5,0);

				// c[6,0-15]
				CVT_STORE_F32_S8(acc_60,6,0);

				// c[7,0-15]
				CVT_STORE_F32_S8(acc_70,7,0);

				// c[8,0-15]
				CVT_STORE_F32_S8(acc_80,8,0);

				// c[9,0-15]
				CVT_STORE_F32_S8(acc_90,9,0);

				// c[10,0-15]
				CVT_STORE_F32_S8(acc_100,10,0);

				// c[11,0-15]
				CVT_STORE_F32_S8(acc_110,11,0);
			}
			else if ( post_ops_attr.c_stor_type == U8 )
			{
				// Store the results in downscaled type (uint8 instead of int32).
				// c[0,0-15]
				CVT_STORE_F32_U8(acc_00,0,0);

				// c[1,0-15]
				CVT_STORE_F32_U8(acc_10,1,0);

				// c[2,0-15]
				CVT_STORE_F32_U8(acc_20,2,0);

				// c[3,0-15]
				CVT_STORE_F32_U8(acc_30,3,0);

				// c[4,0-15]
				CVT_STORE_F32_U8(acc_40,4,0);

				// c[5,0-15]
				CVT_STORE_F32_U8(acc_50,5,0);

				// c[6,0-15]
				CVT_STORE_F32_U8(acc_60,6,0);

				// c[7,0-15]
				CVT_STORE_F32_U8(acc_70,7,0);

				// c[8,0-15]
				CVT_STORE_F32_U8(acc_80,8,0);

				// c[9,0-15]
				CVT_STORE_F32_U8(acc_90,9,0);

				// c[10,0-15]
				CVT_STORE_F32_U8(acc_100,10,0);

				// c[11,0-15]
				CVT_STORE_F32_U8(acc_110,11,0);
			}
			else if ( post_ops_attr.c_stor_type == BF16)
			{
				// Store the results in downscaled type (int8 instead of int32).
				// c[0,0-15]
				CVT_STORE_F32_BF16(acc_00,0,0);

				// c[1,0-15]
				CVT_STORE_F32_BF16(acc_10,1,0);

				// c[2,0-15]
				CVT_STORE_F32_BF16(acc_20,2,0);

				// c[3,0-15]
				CVT_STORE_F32_BF16(acc_30,3,0);

				// c[4,0-15]
				CVT_STORE_F32_BF16(acc_40,4,0);

				// c[5,0-15]
				CVT_STORE_F32_BF16(acc_50,5,0);

				// c[6,0-15]
				CVT_STORE_F32_BF16(acc_60,6,0);

				// c[7,0-15]
				CVT_STORE_F32_BF16(acc_70,7,0);

				// c[8,0-15]
				CVT_STORE_F32_BF16(acc_80,8,0);

				// c[9,0-15]
				CVT_STORE_F32_BF16(acc_90,9,0);

				// c[10,0-15]
				CVT_STORE_F32_BF16(acc_100,10,0);

				// c[11,0-15]
				CVT_STORE_F32_BF16(acc_110,11,0);
			}
			else if ( post_ops_attr.c_stor_type == F32)
			{
				// Store the results in downscaled type (int8 instead of int32).
				// c[0,0-15]
				STORE_F32(acc_00,0,0);

				// c[1,0-15]
				STORE_F32(acc_10,1,0);

				// c[2,0-15]
				STORE_F32(acc_20,2,0);

				// c[3,0-15]
				STORE_F32(acc_30,3,0);

				// c[4,0-15]
				STORE_F32(acc_40,4,0);

				// c[5,0-15]
				STORE_F32(acc_50,5,0);

				// c[6,0-15]
				STORE_F32(acc_60,6,0);

				// c[7,0-15]
				STORE_F32(acc_70,7,0);

				// c[8,0-15]
				STORE_F32(acc_80,8,0);

				// c[9,0-15]
				STORE_F32(acc_90,9,0);

				// c[10,0-15]
				STORE_F32(acc_100,10,0);

				// c[11,0-15]
				STORE_F32(acc_110,11,0);
			}
		}
		else
		{
			__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

			// Store the results.
			// c[0,0-15]
			_mm512_mask_storeu_epi32
			(
			  c + ( rs_c * ( ir + 0 ) ), load_mask, _mm512_cvtps_epi32( acc_00 )
			);

			// c[1,0-15]
			_mm512_mask_storeu_epi32
			(
			  c + ( rs_c * ( ir + 1 ) ), load_mask, _mm512_cvtps_epi32( acc_10 )
			);

			// c[2,0-15]
			_mm512_mask_storeu_epi32
			(
			  c + ( rs_c * ( ir + 2 ) ), load_mask, _mm512_cvtps_epi32( acc_20 )
			);

			// c[3,0-15]
			_mm512_mask_storeu_epi32
			(
			  c + ( rs_c * ( ir + 3 ) ), load_mask, _mm512_cvtps_epi32( acc_30 )
			);

			// c[4,0-15]
			_mm512_mask_storeu_epi32
			(
			  c + ( rs_c * ( ir + 4 ) ), load_mask, _mm512_cvtps_epi32( acc_40 )
			);

			// c[5,0-15]
			_mm512_mask_storeu_epi32
			(
			  c + ( rs_c * ( ir + 5 ) ), load_mask, _mm512_cvtps_epi32( acc_50 )
			);

			// c[6,0-15]
			_mm512_mask_storeu_epi32
			(
			  c + ( rs_c * ( ir + 6 ) ), load_mask, _mm512_cvtps_epi32( acc_60 )
			);

			// c[7,0-15]
			_mm512_mask_storeu_epi32
			(
			  c + ( rs_c * ( ir + 7 ) ), load_mask, _mm512_cvtps_epi32( acc_70 )
			);

			// c[8,0-15]
			_mm512_mask_storeu_epi32
			(
			  c + ( rs_c * ( ir + 8 ) ), load_mask, _mm512_cvtps_epi32( acc_80 )
			);

			// c[9,0-15]
			_mm512_mask_storeu_epi32
			(
			  c + ( rs_c * ( ir + 9 ) ), load_mask, _mm512_cvtps_epi32( acc_90 )
			);

			// c[10,0-15]
			_mm512_mask_storeu_epi32
			(
			  c + ( rs_c * ( ir + 10 ) ), load_mask, _mm512_cvtps_epi32( acc_100 )
			);

			// c[11,0-15]
			_mm512_mask_storeu_epi32
			(
			  c + ( rs_c * ( ir + 11 ) ), load_mask, _mm512_cvtps_epi32( acc_110 )
			);
		}

		a = a + ( MR * ps_a );
		post_ops_attr.post_op_c_i += MR;
	}

	if ( m_partial_pieces > 0 )
	{
		lpgemm_rowvar_u8s8s32o32_6xlt16
		(
		  m_partial_pieces, k0,
		  a, rs_a, cs_a, ps_a,
		  b, rs_b, cs_b,
		  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
		  alpha, beta, n0_rem,
		  post_ops_list, post_ops_attr
		);
	}
}

// 12x16 int8o32 fringe kernel
__attribute__((aligned(64)))
LPGEMM_N_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_12x16)
{
	static void* post_ops_labels[] =
				{
				  &&POST_OPS_12x16_DISABLE,
				  &&POST_OPS_BIAS_12x16,
				  &&POST_OPS_RELU_12x16,
				  &&POST_OPS_RELU_SCALE_12x16,
				  &&POST_OPS_GELU_TANH_12x16,
				  &&POST_OPS_GELU_ERF_12x16,
				  &&POST_OPS_CLIP_12x16,
				  &&POST_OPS_DOWNSCALE_12x16,
				  &&POST_OPS_MATRIX_ADD_12x16,
				  &&POST_OPS_SWISH_12x16,
				  &&POST_OPS_MATRIX_MUL_12x16,
				  &&POST_OPS_TANH_12x16,
				  &&POST_OPS_SIGMOID_12x16
				};
	dim_t MR = 12;
	dim_t m_full_pieces = m0 / MR;
	dim_t m_full_pieces_loop_limit = m_full_pieces * MR;
	dim_t m_partial_pieces = m0 % MR;

	dim_t k_full_pieces = k0 / 4;
	dim_t k_partial_pieces = k0 % 4;

	// B matrix storage.
	__m512i b0 = _mm512_setzero_epi32();

	// A matrix storage.
	__m512i a_int32_0 = _mm512_setzero_epi32();
	__m512i a_int32_1 = _mm512_setzero_epi32();
	__m512i a_int32_2 = _mm512_setzero_epi32();
	__m512i a_int32_3 = _mm512_setzero_epi32();
	__m512i a_int32_4 = _mm512_setzero_epi32();
	__m512i a_int32_5 = _mm512_setzero_epi32();
	__m512i a_int32_6 = _mm512_setzero_epi32();
	__m512i a_int32_7 = _mm512_setzero_epi32();
	__m512i a_int32_8 = _mm512_setzero_epi32();
	__m512i a_int32_9 = _mm512_setzero_epi32();
	__m512i a_int32_10 = _mm512_setzero_epi32();
	__m512i a_int32_11 = _mm512_setzero_epi32();

	for ( dim_t ir = 0; ir < m_full_pieces_loop_limit; ir += MR )
	{
		_mm_prefetch( b, _MM_HINT_T0 );
		_mm_prefetch( a + ( MR * ps_a ) + ( 0 * 16 ), _MM_HINT_T1 );

		// Registers to use for accumulating C.
		__m512i c_int32_0p0 = _mm512_setzero_epi32();

		__m512i c_int32_1p0 = _mm512_setzero_epi32();

		__m512i c_int32_2p0 = _mm512_setzero_epi32();

		__m512i c_int32_3p0 = _mm512_setzero_epi32();

		__m512i c_int32_4p0 = _mm512_setzero_epi32();

		__m512i c_int32_5p0 = _mm512_setzero_epi32();

		__m512i c_int32_6p0 = _mm512_setzero_epi32();

		__m512i c_int32_7p0 = _mm512_setzero_epi32();

		__m512i c_int32_8p0 = _mm512_setzero_epi32();

		__m512i c_int32_9p0 = _mm512_setzero_epi32();

		__m512i c_int32_10p0 = _mm512_setzero_epi32();

		__m512i c_int32_11p0 = _mm512_setzero_epi32();

		for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
		{
			// Load 4 rows with 16 elements each from B to 1 ZMM registers. It
			// is to be noted that the B matrix is packed for use in vnni
			// instructions and each load to ZMM register will have 4 elements
			// along k direction and 16 elements across n directions, so 4x16
			// elements to a ZMM register.
			b0 = _mm512_loadu_si512( b + ( rs_b * kr ) + ( cs_b * 0 ) );

			// Broadcast a[0,kr:kr+4].
			a_int32_0 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

			// Broadcast a[1,kr:kr+4].
			a_int32_1 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 1 ) + ( cs_a * kr ) ) );

			// Broadcast a[2,kr:kr+4].
			a_int32_2 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 2 ) + ( cs_a * kr ) ) );

			// Broadcast a[3,kr:kr+4].
			a_int32_3 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 3 ) + ( cs_a * kr ) ) );

			// Broadcast a[4,kr:kr+4].
			a_int32_4 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 4 ) + ( cs_a * kr ) ) );

			// Broadcast a[5,kr:kr+4].
			a_int32_5 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 5 ) + ( cs_a * kr ) ) );

			// Broadcast a[0,kr:kr+4].
			a_int32_6 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 6 ) + ( cs_a * kr ) ) );

			// Broadcast a[1,kr:kr+4].
			a_int32_7 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 7 ) + ( cs_a * kr ) ) );

			// Perform column direction mat-mul with k = 4.
			// c[0,0-15] = a[0,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_0p0 = _mm512_dpbusd_epi32( c_int32_0p0, a_int32_0, b0 );

			// Perform column direction mat-mul with k = 4.
			// c[1,0-15] = a[1,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_1p0 = _mm512_dpbusd_epi32( c_int32_1p0, a_int32_1, b0 );

			// Perform column direction mat-mul with k = 4.
			// c[2,0-15] = a[2,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_2p0 = _mm512_dpbusd_epi32( c_int32_2p0, a_int32_2, b0 );

			// Perform column direction mat-mul with k = 4.
			// c[3,0-15] = a[3,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_3p0 = _mm512_dpbusd_epi32( c_int32_3p0, a_int32_3, b0 );

			// Broadcast a[2,kr:kr+4].
			a_int32_8 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 8 ) + ( cs_a * kr ) ) );

			// Broadcast a[3,kr:kr+4].
			a_int32_9 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 9 ) + ( cs_a * kr ) ) );

			// Broadcast a[4,kr:kr+4].
			a_int32_10 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 10 ) + ( cs_a * kr ) ) );

			// Broadcast a[5,kr:kr+4].
			a_int32_11 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 11 ) + ( cs_a * kr ) ) );

			// Perform column direction mat-mul with k = 4.
			// c[4,0-15] = a[4,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_4p0 = _mm512_dpbusd_epi32( c_int32_4p0, a_int32_4, b0 );

			// Perform column direction mat-mul with k = 4.
			// c[5,0-15] = a[5,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_5p0 = _mm512_dpbusd_epi32( c_int32_5p0, a_int32_5, b0 );

			// Perform column direction mat-mul with k = 4.
			// c[6,0-15] = a[6,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_6p0 = _mm512_dpbusd_epi32( c_int32_6p0, a_int32_6, b0 );

			// Perform column direction mat-mul with k = 4.
			// c[7,0-15] = a[7,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_7p0 = _mm512_dpbusd_epi32( c_int32_7p0, a_int32_7, b0 );

			// Perform column direction mat-mul with k = 4.
			// c[8,0-15] = a[8,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_8p0 = _mm512_dpbusd_epi32( c_int32_8p0, a_int32_8, b0 );

			// Perform column direction mat-mul with k = 4.
			// c[9,0-15] = a[9,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_9p0 = _mm512_dpbusd_epi32( c_int32_9p0, a_int32_9, b0 );

			// Perform column direction mat-mul with k = 4.
			// c[10,0-15] = a[10,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_10p0 = _mm512_dpbusd_epi32( c_int32_10p0, a_int32_10, b0 );

			// Perform column direction mat-mul with k = 4.
			// c[11,0-15] = a[11,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_11p0 = _mm512_dpbusd_epi32( c_int32_11p0, a_int32_11, b0 );
		}
		__asm__(".p2align 6\n");
		// Handle k remainder.
		if ( k_partial_pieces > 0 )
		{
			__m128i a_kfringe_buf;
			__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - k_partial_pieces ) );

			b0 = _mm512_loadu_si512( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );

			// Broadcast a[0,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_0 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[0,0-15] = a[0,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_0p0 = _mm512_dpbusd_epi32( c_int32_0p0, a_int32_0, b0 );

			// Broadcast a[1,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 1 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_1 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[1,0-15] = a[1,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_1p0 = _mm512_dpbusd_epi32( c_int32_1p0, a_int32_1, b0 );

			// Broadcast a[2,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 2 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_2 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[2,0-15] = a[2,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_2p0 = _mm512_dpbusd_epi32( c_int32_2p0, a_int32_2, b0 );

			// Broadcast a[3,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 3 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_3 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[3,0-15] = a[3,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_3p0 = _mm512_dpbusd_epi32( c_int32_3p0, a_int32_3, b0 );

			// Broadcast a[4,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 4 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_4 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[4,0-15] = a[4,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_4p0 = _mm512_dpbusd_epi32( c_int32_4p0, a_int32_4, b0 );

			// Broadcast a[5,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 5 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_5 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[5,0-15] = a[5,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_5p0 = _mm512_dpbusd_epi32( c_int32_5p0, a_int32_5, b0 );

			// Broadcast a[6,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 6 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_6 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[6,0-15] = a[6,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_6p0 = _mm512_dpbusd_epi32( c_int32_6p0, a_int32_6, b0 );

			// Broadcast a[7,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 7 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_7 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[7,0-15] = a[7,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_7p0 = _mm512_dpbusd_epi32( c_int32_7p0, a_int32_7, b0 );

			// Broadcast a[8,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 8 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_8 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[8,0-15] = a[8,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_8p0 = _mm512_dpbusd_epi32( c_int32_8p0, a_int32_8, b0 );

			// Broadcast a[9,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 9 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_9 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[9,0-15] = a[9,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_9p0 = _mm512_dpbusd_epi32( c_int32_9p0, a_int32_9, b0 );

			// Broadcast a[10,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 10 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_10 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[10,0-15] = a[10,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_10p0 = _mm512_dpbusd_epi32( c_int32_10p0, a_int32_10, b0 );

			// Broadcast a[11,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 11 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_11 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[11,0-15] = a[11,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_11p0 = _mm512_dpbusd_epi32( c_int32_11p0, a_int32_11, b0 );
		}

		// Load alpha and beta
		__m512i selector1 = _mm512_set1_epi32( alpha );
		__m512i selector2 = _mm512_set1_epi32( beta );

		if ( alpha != 1 )
		{
			// Scale by alpha
			c_int32_0p0 = _mm512_mullo_epi32( selector1, c_int32_0p0 );

			c_int32_1p0 = _mm512_mullo_epi32( selector1, c_int32_1p0 );

			c_int32_2p0 = _mm512_mullo_epi32( selector1, c_int32_2p0 );

			c_int32_3p0 = _mm512_mullo_epi32( selector1, c_int32_3p0 );

			c_int32_4p0 = _mm512_mullo_epi32( selector1, c_int32_4p0 );

			c_int32_5p0 = _mm512_mullo_epi32( selector1, c_int32_5p0 );

			c_int32_6p0 = _mm512_mullo_epi32( selector1, c_int32_6p0 );

			c_int32_7p0 = _mm512_mullo_epi32( selector1, c_int32_7p0 );

			c_int32_8p0 = _mm512_mullo_epi32( selector1, c_int32_8p0 );

			c_int32_9p0 = _mm512_mullo_epi32( selector1, c_int32_9p0 );

			c_int32_10p0 = _mm512_mullo_epi32( selector1, c_int32_10p0 );

			c_int32_11p0 = _mm512_mullo_epi32( selector1, c_int32_11p0 );
		}

		// Scale C by beta.
		if ( beta != 0 )
		{
			if ( ( post_ops_attr.buf_downscale != NULL ) &&
				 ( post_ops_attr.is_first_k == TRUE ) )
			{
				if( post_ops_attr.c_stor_type == S8 )
				{
					// c[0:0-15]
					S8_S32_BETA_OP(c_int32_0p0,ir,0,0,selector1,selector2);

					// c[1:0-15]
					S8_S32_BETA_OP(c_int32_1p0,ir,1,0,selector1,selector2);

					// c[2:0-15]
					S8_S32_BETA_OP(c_int32_2p0,ir,2,0,selector1,selector2);

					// c[3:0-15]
					S8_S32_BETA_OP(c_int32_3p0,ir,3,0,selector1,selector2);

					// c[4:0-15]
					S8_S32_BETA_OP(c_int32_4p0,ir,4,0,selector1,selector2);

					// c[5:0-15]
					S8_S32_BETA_OP(c_int32_5p0,ir,5,0,selector1,selector2);

					// c[6:0-15]
					S8_S32_BETA_OP(c_int32_6p0,ir,6,0,selector1,selector2);

					// c[7:0-15]
					S8_S32_BETA_OP(c_int32_7p0,ir,7,0,selector1,selector2);

					// c[8:0-15]
					S8_S32_BETA_OP(c_int32_8p0,ir,8,0,selector1,selector2);

					// c[9:0-15]
					S8_S32_BETA_OP(c_int32_9p0,ir,9,0,selector1,selector2);

					// c[10:0-15]
					S8_S32_BETA_OP(c_int32_10p0,ir,10,0,selector1,selector2);

					// c[11:0-15]
					S8_S32_BETA_OP(c_int32_11p0,ir,11,0,selector1,selector2);
				}
				else if( post_ops_attr.c_stor_type == U8 )
				{
					// c[0:0-15]
					U8_S32_BETA_OP(c_int32_0p0,ir,0,0,selector1,selector2);

					// c[1:0-15]
					U8_S32_BETA_OP(c_int32_1p0,ir,1,0,selector1,selector2);

					// c[2:0-15]
					U8_S32_BETA_OP(c_int32_2p0,ir,2,0,selector1,selector2);

					// c[3:0-15]
					U8_S32_BETA_OP(c_int32_3p0,ir,3,0,selector1,selector2);

					// c[4:0-15]
					U8_S32_BETA_OP(c_int32_4p0,ir,4,0,selector1,selector2);

					// c[5:0-15]
					U8_S32_BETA_OP(c_int32_5p0,ir,5,0,selector1,selector2);

					// c[6:0-15]
					U8_S32_BETA_OP(c_int32_6p0,ir,6,0,selector1,selector2);

					// c[7:0-15]
					U8_S32_BETA_OP(c_int32_7p0,ir,7,0,selector1,selector2);

					// c[8:0-15]
					U8_S32_BETA_OP(c_int32_8p0,ir,8,0,selector1,selector2);

					// c[9:0-15]
					U8_S32_BETA_OP(c_int32_9p0,ir,9,0,selector1,selector2);

					// c[10:0-15]
					U8_S32_BETA_OP(c_int32_10p0,ir,10,0,selector1,selector2);

					// c[11:0-15]
					U8_S32_BETA_OP(c_int32_11p0,ir,11,0,selector1,selector2);
				}
				else if( post_ops_attr.c_stor_type == BF16 )
				{
					// c[0:0-15]
					BF16_S32_BETA_OP(c_int32_0p0,ir,0,0,selector1,selector2);

					// c[1:0-15]
					BF16_S32_BETA_OP(c_int32_1p0,ir,1,0,selector1,selector2);

					// c[2:0-15]
					BF16_S32_BETA_OP(c_int32_2p0,ir,2,0,selector1,selector2);

					// c[3:0-15]
					BF16_S32_BETA_OP(c_int32_3p0,ir,3,0,selector1,selector2);

					// c[4:0-15]
					BF16_S32_BETA_OP(c_int32_4p0,ir,4,0,selector1,selector2);

					// c[5:0-15]
					BF16_S32_BETA_OP(c_int32_5p0,ir,5,0,selector1,selector2);

					// c[6:0-15]
					BF16_S32_BETA_OP(c_int32_6p0,ir,6,0,selector1,selector2);

					// c[7:0-15]
					BF16_S32_BETA_OP(c_int32_7p0,ir,7,0,selector1,selector2);

					// c[8:0-15]
					BF16_S32_BETA_OP(c_int32_8p0,ir,8,0,selector1,selector2);

					// c[9:0-15]
					BF16_S32_BETA_OP(c_int32_9p0,ir,9,0,selector1,selector2);

					// c[10:0-15]
					BF16_S32_BETA_OP(c_int32_10p0,ir,10,0,selector1,selector2);

					// c[11:0-15]
					BF16_S32_BETA_OP(c_int32_11p0,ir,11,0,selector1,selector2);
				}
				else if( post_ops_attr.c_stor_type == F32 )
				{
					// c[0:0-15]
					F32_S32_BETA_OP(c_int32_0p0,ir,0,0,selector1,selector2);

					// c[1:0-15]
					F32_S32_BETA_OP(c_int32_1p0,ir,1,0,selector1,selector2);

					// c[2:0-15]
					F32_S32_BETA_OP(c_int32_2p0,ir,2,0,selector1,selector2);

					// c[3:0-15]
					F32_S32_BETA_OP(c_int32_3p0,ir,3,0,selector1,selector2);

					// c[4:0-15]
					F32_S32_BETA_OP(c_int32_4p0,ir,4,0,selector1,selector2);

					// c[5:0-15]
					F32_S32_BETA_OP(c_int32_5p0,ir,5,0,selector1,selector2);

					// c[6:0-15]
					F32_S32_BETA_OP(c_int32_6p0,ir,6,0,selector1,selector2);

					// c[7:0-15]
					F32_S32_BETA_OP(c_int32_7p0,ir,7,0,selector1,selector2);

					// c[8:0-15]
					F32_S32_BETA_OP(c_int32_8p0,ir,8,0,selector1,selector2);

					// c[9:0-15]
					F32_S32_BETA_OP(c_int32_9p0,ir,9,0,selector1,selector2);

					// c[10:0-15]
					F32_S32_BETA_OP(c_int32_10p0,ir,10,0,selector1,selector2);

					// c[11:0-15]
					F32_S32_BETA_OP(c_int32_11p0,ir,11,0,selector1,selector2);
				}
			}
			else
			{
				// c[0:0-15]
				S32_S32_BETA_OP(c_int32_0p0,ir,0,0,selector1,selector2);

				// c[1:0-15]
				S32_S32_BETA_OP(c_int32_1p0,ir,1,0,selector1,selector2);

				// c[2:0-15]
				S32_S32_BETA_OP(c_int32_2p0,ir,2,0,selector1,selector2);

				// c[3:0-15]
				S32_S32_BETA_OP(c_int32_3p0,ir,3,0,selector1,selector2);

				// c[4:0-15]
				S32_S32_BETA_OP(c_int32_4p0,ir,4,0,selector1,selector2);

				// c[5:0-15]
				S32_S32_BETA_OP(c_int32_5p0,ir,5,0,selector1,selector2);

				// c[6:0-15]
				S32_S32_BETA_OP(c_int32_6p0,ir,6,0,selector1,selector2);

				// c[7:0-15]
				S32_S32_BETA_OP(c_int32_7p0,ir,7,0,selector1,selector2);

				// c[8:0-15]
				S32_S32_BETA_OP(c_int32_8p0,ir,8,0,selector1,selector2);

				// c[9:0-15]
				S32_S32_BETA_OP(c_int32_9p0,ir,9,0,selector1,selector2);

				// c[10:0-15]
				S32_S32_BETA_OP(c_int32_10p0,ir,10,0,selector1,selector2);

				// c[11:0-15]
				S32_S32_BETA_OP(c_int32_11p0,ir,11,0,selector1,selector2);
			}
		}

		__m512 acc_00 = _mm512_setzero_ps();
		__m512 acc_10 = _mm512_setzero_ps();
		__m512 acc_20 = _mm512_setzero_ps();
		__m512 acc_30 = _mm512_setzero_ps();
		__m512 acc_40 = _mm512_setzero_ps();
		__m512 acc_50 = _mm512_setzero_ps();
		__m512 acc_60 = _mm512_setzero_ps();
		__m512 acc_70 = _mm512_setzero_ps();
		__m512 acc_80 = _mm512_setzero_ps();
		__m512 acc_90 = _mm512_setzero_ps();
		__m512 acc_100 = _mm512_setzero_ps();
		__m512 acc_110 = _mm512_setzero_ps();
		CVT_ACCUM_REG_INT_TO_FLOAT_12ROWS_XCOL(acc_, c_int32_, 1);

        // Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_12x16:
		{
			__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
			__m512 b0 = _mm512_setzero_ps();

			if ( post_ops_list_temp->stor_type == BF16 )
			{
				BF16_F32_BIAS_LOAD(b0, bias_mask, 0);
			}
			else if ( post_ops_list_temp->stor_type == S8 )
			{
				S8_F32_BIAS_LOAD(b0, bias_mask, 0);
			}
			else if ( post_ops_list_temp->stor_type == S32 )
			{
				S32_F32_BIAS_LOAD(b0, bias_mask, 0);
			}
			else
			{
				b0 = _mm512_maskz_loadu_ps ( bias_mask,
						( ( float* )post_ops_list_temp->op_args1 ) +
						post_ops_attr.post_op_c_j + ( 0 * 16 ) );
			}

			// c[0,0-15]
			acc_00 = _mm512_add_ps( b0, acc_00 );

			// c[1,0-15]
			acc_10 = _mm512_add_ps( b0, acc_10 );

			// c[2,0-15]
			acc_20 = _mm512_add_ps( b0, acc_20 );

			// c[3,0-15]
			acc_30 = _mm512_add_ps( b0, acc_30 );

			// c[4,0-15]
			acc_40 = _mm512_add_ps( b0, acc_40 );

			// c[5,0-15]
			acc_50 = _mm512_add_ps( b0, acc_50 );

			// c[6,0-15]
			acc_60 = _mm512_add_ps( b0, acc_60 );

			// c[7,0-15]
			acc_70 = _mm512_add_ps( b0, acc_70 );

			// c[8,0-15]
			acc_80 = _mm512_add_ps( b0, acc_80 );

			// c[9,0-15]
			acc_90 = _mm512_add_ps( b0, acc_90 );

			// c[10,0-15]
			acc_100 = _mm512_add_ps( b0, acc_100 );

			// c[11,0-15]
			acc_110 = _mm512_add_ps( b0, acc_110 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_12x16:
		{
			__m512 zero = _mm512_setzero_ps();

			// c[0,0-15]
			acc_00 = _mm512_max_ps( zero, acc_00 );

			// c[1,0-15]
			acc_10 = _mm512_max_ps( zero, acc_10 );

			// c[2,0-15]
			acc_20 = _mm512_max_ps( zero, acc_20 );

			// c[3,0-15]
			acc_30 = _mm512_max_ps( zero, acc_30 );

			// c[4,0-15]
			acc_40 = _mm512_max_ps( zero, acc_40 );

			// c[5,0-15]
			acc_50 = _mm512_max_ps( zero, acc_50 );

			// c[6,0-15]
			acc_60 = _mm512_max_ps( zero, acc_60 );

			// c[7,0-15]
			acc_70 = _mm512_max_ps( zero, acc_70 );

			// c[8,0-15]
			acc_80 = _mm512_max_ps( zero, acc_80 );

			// c[9,0-15]
			acc_90 = _mm512_max_ps( zero, acc_90 );

			// c[10,0-15]
			acc_100 = _mm512_max_ps( zero, acc_100 );

			// c[11,0-15]
			acc_110 = _mm512_max_ps( zero, acc_110 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_12x16:
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

			// c[0, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_00)

			// c[1, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_10)

			// c[2, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_20)

			// c[3, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_30)

			// c[4, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_40)

			// c[5, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_50)

			// c[6, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_60)

			// c[7, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_70)

			// c[8, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_80)

			// c[9, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_90)

			// c[10, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_100)

			// c[11, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_110)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_TANH_12x16:
		{
			__m512 dn, z, x, r2, r, y;
			__m512i tmpout;

			// c[0, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_00, y, r, r2, x, z, dn, tmpout)

			// c[1, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_10, y, r, r2, x, z, dn, tmpout)

			// c[2, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_20, y, r, r2, x, z, dn, tmpout)

			// c[3, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_30, y, r, r2, x, z, dn, tmpout)

			// c[4, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_40, y, r, r2, x, z, dn, tmpout)

			// c[5, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_50, y, r, r2, x, z, dn, tmpout)

			// c[6, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_60, y, r, r2, x, z, dn, tmpout)

			// c[7, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_70, y, r, r2, x, z, dn, tmpout)

			// c[8, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_80, y, r, r2, x, z, dn, tmpout)

			// c[9, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_90, y, r, r2, x, z, dn, tmpout)

			// c[10, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_100, y, r, r2, x, z, dn, tmpout)

			// c[11, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_110, y, r, r2, x, z, dn, tmpout)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_ERF_12x16:
		{
			__m512 y, r, r2;

			// c[0, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_00, y, r, r2)

			// c[1, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_10, y, r, r2)

			// c[2, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_20, y, r, r2)

			// c[3, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_30, y, r, r2)

			// c[4, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_40, y, r, r2)

			// c[5, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_50, y, r, r2)

			// c[6, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_60, y, r, r2)

			// c[7, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_70, y, r, r2)

			// c[8, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_80, y, r, r2)

			// c[9, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_90, y, r, r2)

			// c[10, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_100, y, r, r2)

			// c[11, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_110, y, r, r2)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_CLIP_12x16:
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

			// c[0, 0-15]
			CLIP_F32_AVX512(acc_00, min, max)

			// c[1, 0-15]
			CLIP_F32_AVX512(acc_10, min, max)

			// c[2, 0-15]
			CLIP_F32_AVX512(acc_20, min, max)

			// c[3, 0-15]
			CLIP_F32_AVX512(acc_30, min, max)

			// c[4, 0-15]
			CLIP_F32_AVX512(acc_40, min, max)

			// c[5, 0-15]
			CLIP_F32_AVX512(acc_50, min, max)

			// c[6, 0-15]
			CLIP_F32_AVX512(acc_60, min, max)

			// c[7, 0-15]
			CLIP_F32_AVX512(acc_70, min, max)

			// c[8, 0-15]
			CLIP_F32_AVX512(acc_80, min, max)

			// c[9, 0-15]
			CLIP_F32_AVX512(acc_90, min, max)

			// c[10, 0-15]
			CLIP_F32_AVX512(acc_100, min, max)

			// c[11, 0-15]
			CLIP_F32_AVX512(acc_110, min, max)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_DOWNSCALE_12x16:
	{
		__m512 scale0 = _mm512_setzero_ps();
		if ( post_ops_list_temp->scale_factor_len > 1 )
		{
			scale0 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
					post_ops_attr.post_op_c_j + ( 0 * 16 ) );
		}
		else if ( post_ops_list_temp->scale_factor_len == 1 )
		{
			scale0 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
		}

		// Need to ensure sse not used to avoid avx512 -> sse transition.
		__m128i zero_point0 = _mm512_castsi512_si128( _mm512_setzero_si512() );
		if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
		{
			zero_point0 = _mm_loadu_si128( ( __m128i const* )
							( ( int8_t* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
		}
		else if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
		{
			zero_point0 = _mm_maskz_set1_epi8( 0xFFFF,
							*( ( int8_t* )post_ops_list_temp->op_args1 ) );
		}

		// c[0, 0-15]
		CVT_MULRND_F32(acc_00,scale0,zero_point0);

		// c[1, 0-15]
		CVT_MULRND_F32(acc_10,scale0,zero_point0);

		// c[2, 0-15]
		CVT_MULRND_F32(acc_20,scale0,zero_point0);

		// c[3, 0-15]
		CVT_MULRND_F32(acc_30,scale0,zero_point0);

		// c[4, 0-15]
		CVT_MULRND_F32(acc_40,scale0,zero_point0);

		// c[5, 0-15]
		CVT_MULRND_F32(acc_50,scale0,zero_point0);

		// c[6, 0-15]
		CVT_MULRND_F32(acc_60,scale0,zero_point0);

		// c[7, 0-15]
		CVT_MULRND_F32(acc_70,scale0,zero_point0);

		// c[8, 0-15]
		CVT_MULRND_F32(acc_80,scale0,zero_point0);

		// c[9, 0-15]
		CVT_MULRND_F32(acc_90,scale0,zero_point0);

		// c[10, 0-15]
		CVT_MULRND_F32(acc_100,scale0,zero_point0);

		// c[11, 0-15]
		CVT_MULRND_F32(acc_110,scale0,zero_point0);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_ADD_12x16:
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
			__m512 scl_fctr5 = _mm512_setzero_ps();
			__m512 scl_fctr6 = _mm512_setzero_ps();
			__m512 scl_fctr7 = _mm512_setzero_ps();
			__m512 scl_fctr8 = _mm512_setzero_ps();
			__m512 scl_fctr9 = _mm512_setzero_ps();
			__m512 scl_fctr10 = _mm512_setzero_ps();
			__m512 scl_fctr11 = _mm512_setzero_ps();
			__m512 scl_fctr12 = _mm512_setzero_ps();
			__m512 t0 = _mm512_setzero_ps();

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
				scl_fctr6 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr7 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr8 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr9 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr10 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr11 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr12 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
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
					scl_fctr6 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 5 ) );
					scl_fctr7 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 6 ) );
					scl_fctr8 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 7 ) );
					scl_fctr9 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 8 ) );
					scl_fctr10 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 9 ) );
					scl_fctr11 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 10 ) );
					scl_fctr12 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 11 ) );
				}
			}

			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15]
					BF16_MATRIX_ADD_1COL(t0,scl_fctr1,0);

					// c[1:0-15]
					BF16_MATRIX_ADD_1COL(t0,scl_fctr1,1);

					// c[2:0-15]
					BF16_MATRIX_ADD_1COL(t0,scl_fctr1,2);

					// c[3:0-15]
					BF16_MATRIX_ADD_1COL(t0,scl_fctr1,3);

					// c[4:0-15]
					BF16_MATRIX_ADD_1COL(t0,scl_fctr1,4);

					// c[5:0-15]
					BF16_MATRIX_ADD_1COL(t0,scl_fctr1,5);

					// c[6:0-15]
					BF16_MATRIX_ADD_1COL(t0,scl_fctr1,6);

					// c[7:0-15]
					BF16_MATRIX_ADD_1COL(t0,scl_fctr1,7);

					// c[8:0-15]
					BF16_MATRIX_ADD_1COL(t0,scl_fctr1,8);

					// c[9:0-15]
					BF16_MATRIX_ADD_1COL(t0,scl_fctr1,9);

					// c[10:0-15]
					BF16_MATRIX_ADD_1COL(t0,scl_fctr1,10);

					// c[11:0-15]
					BF16_MATRIX_ADD_1COL(t0,scl_fctr1,11);
				}
				else
				{
					// c[0:0-15]
					BF16_MATRIX_ADD_1COL(t0,scl_fctr1,0);

					// c[1:0-15]
					BF16_MATRIX_ADD_1COL(t0,scl_fctr2,1);

					// c[2:0-15]
					BF16_MATRIX_ADD_1COL(t0,scl_fctr3,2);

					// c[3:0-15]
					BF16_MATRIX_ADD_1COL(t0,scl_fctr4,3);

					// c[4:0-15]
					BF16_MATRIX_ADD_1COL(t0,scl_fctr5,4);

					// c[5:0-15]
					BF16_MATRIX_ADD_1COL(t0,scl_fctr6,5);

					// c[6:0-15]
					BF16_MATRIX_ADD_1COL(t0,scl_fctr7,6);

					// c[7:0-15]
					BF16_MATRIX_ADD_1COL(t0,scl_fctr8,7);

					// c[8:0-15]
					BF16_MATRIX_ADD_1COL(t0,scl_fctr9,8);

					// c[9:0-15]
					BF16_MATRIX_ADD_1COL(t0,scl_fctr10,9);

					// c[10:0-15]
					BF16_MATRIX_ADD_1COL(t0,scl_fctr11,10);

					// c[11:0-15]
					BF16_MATRIX_ADD_1COL(t0,scl_fctr12,11);
				}
			}
			else if ( is_f32 == TRUE )
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15]
					F32_ONLY_MATRIX_ADD_1COL(t0,scl_fctr1,0);

					// c[1:0-15]
					F32_ONLY_MATRIX_ADD_1COL(t0,scl_fctr1,1);

					// c[2:0-15]
					F32_ONLY_MATRIX_ADD_1COL(t0,scl_fctr1,2);

					// c[3:0-15]
					F32_ONLY_MATRIX_ADD_1COL(t0,scl_fctr1,3);

					// c[4:0-15]
					F32_ONLY_MATRIX_ADD_1COL(t0,scl_fctr1,4);

					// c[5:0-15]
					F32_ONLY_MATRIX_ADD_1COL(t0,scl_fctr1,5);

					// c[6:0-15]
					F32_ONLY_MATRIX_ADD_1COL(t0,scl_fctr1,6);

					// c[7:0-15]
					F32_ONLY_MATRIX_ADD_1COL(t0,scl_fctr1,7);

					// c[8:0-15]
					F32_ONLY_MATRIX_ADD_1COL(t0,scl_fctr1,8);

					// c[9:0-15]
					F32_ONLY_MATRIX_ADD_1COL(t0,scl_fctr1,9);

					// c[10:0-15]
					F32_ONLY_MATRIX_ADD_1COL(t0,scl_fctr1,10);

					// c[11:0-15]
					F32_ONLY_MATRIX_ADD_1COL(t0,scl_fctr1,11);
				}
				else
				{
					// c[0:0-15]
					F32_ONLY_MATRIX_ADD_1COL(t0,scl_fctr1,0);

					// c[1:0-15]
					F32_ONLY_MATRIX_ADD_1COL(t0,scl_fctr2,1);

					// c[2:0-15]
					F32_ONLY_MATRIX_ADD_1COL(t0,scl_fctr3,2);

					// c[3:0-15]
					F32_ONLY_MATRIX_ADD_1COL(t0,scl_fctr4,3);

					// c[4:0-15]
					F32_ONLY_MATRIX_ADD_1COL(t0,scl_fctr5,4);

					// c[5:0-15]
					F32_ONLY_MATRIX_ADD_1COL(t0,scl_fctr6,5);

					// c[6:0-15]
					F32_ONLY_MATRIX_ADD_1COL(t0,scl_fctr7,6);

					// c[7:0-15]
					F32_ONLY_MATRIX_ADD_1COL(t0,scl_fctr8,7);

					// c[8:0-15]
					F32_ONLY_MATRIX_ADD_1COL(t0,scl_fctr9,8);

					// c[9:0-15]
					F32_ONLY_MATRIX_ADD_1COL(t0,scl_fctr10,9);

					// c[10:0-15]
					F32_ONLY_MATRIX_ADD_1COL(t0,scl_fctr11,10);

					// c[11:0-15]
					F32_ONLY_MATRIX_ADD_1COL(t0,scl_fctr12,11);
				}
			}
			else if ( is_s8 == TRUE )
			{
				int8_t* matptr = ( int8_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15]
					S8_F32_MATRIX_ADD_1COL(t0,scl_fctr1,0);

					// c[1:0-15]
					S8_F32_MATRIX_ADD_1COL(t0,scl_fctr1,1);

					// c[2:0-15]
					S8_F32_MATRIX_ADD_1COL(t0,scl_fctr1,2);

					// c[3:0-15]
					S8_F32_MATRIX_ADD_1COL(t0,scl_fctr1,3);

					// c[4:0-15]
					S8_F32_MATRIX_ADD_1COL(t0,scl_fctr1,4);

					// c[5:0-15]
					S8_F32_MATRIX_ADD_1COL(t0,scl_fctr1,5);

					// c[6:0-15]
					S8_F32_MATRIX_ADD_1COL(t0,scl_fctr1,6);

					// c[7:0-15]
					S8_F32_MATRIX_ADD_1COL(t0,scl_fctr1,7);

					// c[8:0-15]
					S8_F32_MATRIX_ADD_1COL(t0,scl_fctr1,8);

					// c[9:0-15]
					S8_F32_MATRIX_ADD_1COL(t0,scl_fctr1,9);

					// c[10:0-15]
					S8_F32_MATRIX_ADD_1COL(t0,scl_fctr1,10);

					// c[11:0-15]
					S8_F32_MATRIX_ADD_1COL(t0,scl_fctr1,11);
				}
				else
				{
					// c[0:0-15]
					S8_F32_MATRIX_ADD_1COL(t0,scl_fctr1,0);

					// c[1:0-15]
					S8_F32_MATRIX_ADD_1COL(t0,scl_fctr2,1);

					// c[2:0-15]
					S8_F32_MATRIX_ADD_1COL(t0,scl_fctr3,2);

					// c[3:0-15]
					S8_F32_MATRIX_ADD_1COL(t0,scl_fctr4,3);

					// c[4:0-15]
					S8_F32_MATRIX_ADD_1COL(t0,scl_fctr5,4);

					// c[5:0-15]
					S8_F32_MATRIX_ADD_1COL(t0,scl_fctr6,5);

					// c[6:0-15]
					S8_F32_MATRIX_ADD_1COL(t0,scl_fctr7,6);

					// c[7:0-15]
					S8_F32_MATRIX_ADD_1COL(t0,scl_fctr8,7);

					// c[8:0-15]
					S8_F32_MATRIX_ADD_1COL(t0,scl_fctr9,8);

					// c[9:0-15]
					S8_F32_MATRIX_ADD_1COL(t0,scl_fctr10,9);

					// c[10:0-15]
					S8_F32_MATRIX_ADD_1COL(t0,scl_fctr11,10);

					// c[11:0-15]
					S8_F32_MATRIX_ADD_1COL(t0,scl_fctr12,11);
				}
			}
			else
			{
				int32_t* matptr = ( int32_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15]
					S32_F32_MATRIX_ADD_1COL(t0,scl_fctr1,0);

					// c[1:0-15]
					S32_F32_MATRIX_ADD_1COL(t0,scl_fctr1,1);

					// c[2:0-15]
					S32_F32_MATRIX_ADD_1COL(t0,scl_fctr1,2);

					// c[3:0-15]
					S32_F32_MATRIX_ADD_1COL(t0,scl_fctr1,3);

					// c[4:0-15]
					S32_F32_MATRIX_ADD_1COL(t0,scl_fctr1,4);

					// c[5:0-15]
					S32_F32_MATRIX_ADD_1COL(t0,scl_fctr1,5);

					// c[6:0-15]
					S32_F32_MATRIX_ADD_1COL(t0,scl_fctr1,6);

					// c[7:0-15]
					S32_F32_MATRIX_ADD_1COL(t0,scl_fctr1,7);

					// c[8:0-15]
					S32_F32_MATRIX_ADD_1COL(t0,scl_fctr1,8);

					// c[9:0-15]
					S32_F32_MATRIX_ADD_1COL(t0,scl_fctr1,9);

					// c[10:0-15]
					S32_F32_MATRIX_ADD_1COL(t0,scl_fctr1,10);

					// c[11:0-15]
					S32_F32_MATRIX_ADD_1COL(t0,scl_fctr1,11);
				}
				else
				{
					// c[0:0-15]
					S32_F32_MATRIX_ADD_1COL(t0,scl_fctr1,0);

					// c[1:0-15]
					S32_F32_MATRIX_ADD_1COL(t0,scl_fctr2,1);

					// c[2:0-15]
					S32_F32_MATRIX_ADD_1COL(t0,scl_fctr3,2);

					// c[3:0-15]
					S32_F32_MATRIX_ADD_1COL(t0,scl_fctr4,3);

					// c[4:0-15]
					S32_F32_MATRIX_ADD_1COL(t0,scl_fctr5,4);

					// c[5:0-15]
					S32_F32_MATRIX_ADD_1COL(t0,scl_fctr6,5);

					// c[6:0-15]
					S32_F32_MATRIX_ADD_1COL(t0,scl_fctr7,6);

					// c[7:0-15]
					S32_F32_MATRIX_ADD_1COL(t0,scl_fctr8,7);

					// c[8:0-15]
					S32_F32_MATRIX_ADD_1COL(t0,scl_fctr9,8);

					// c[9:0-15]
					S32_F32_MATRIX_ADD_1COL(t0,scl_fctr10,9);

					// c[10:0-15]
					S32_F32_MATRIX_ADD_1COL(t0,scl_fctr10,10);

					// c[11:0-15]
					S32_F32_MATRIX_ADD_1COL(t0,scl_fctr12,11);
				}
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_MUL_12x16:
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
			__m512 scl_fctr5 = _mm512_setzero_ps();
			__m512 scl_fctr6 = _mm512_setzero_ps();
			__m512 scl_fctr7 = _mm512_setzero_ps();
			__m512 scl_fctr8 = _mm512_setzero_ps();
			__m512 scl_fctr9 = _mm512_setzero_ps();
			__m512 scl_fctr10 = _mm512_setzero_ps();
			__m512 scl_fctr11 = _mm512_setzero_ps();
			__m512 scl_fctr12 = _mm512_setzero_ps();
			__m512 t0 = _mm512_setzero_ps();

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
				scl_fctr6 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr7 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr8 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr9 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr10 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr11 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr12 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
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
					scl_fctr6 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 5 ) );
					scl_fctr7 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 6 ) );
					scl_fctr8 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 7 ) );
					scl_fctr9 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 8 ) );
					scl_fctr10 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 9 ) );
					scl_fctr11 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 10 ) );
					scl_fctr12 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 11 ) );
				}
			}

			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15]
					BF16_F32_MATRIX_MUL_1COL(t0,scl_fctr1,0);

					// c[1:0-15]
					BF16_F32_MATRIX_MUL_1COL(t0,scl_fctr1,1);

					// c[2:0-15]
					BF16_F32_MATRIX_MUL_1COL(t0,scl_fctr1,2);

					// c[3:0-15]
					BF16_F32_MATRIX_MUL_1COL(t0,scl_fctr1,3);

					// c[4:0-15]
					BF16_F32_MATRIX_MUL_1COL(t0,scl_fctr1,4);

					// c[5:0-15]
					BF16_F32_MATRIX_MUL_1COL(t0,scl_fctr1,5);

					// c[6:0-15]
					BF16_F32_MATRIX_MUL_1COL(t0,scl_fctr1,6);

					// c[7:0-15]
					BF16_F32_MATRIX_MUL_1COL(t0,scl_fctr1,7);

					// c[8:0-15]
					BF16_F32_MATRIX_MUL_1COL(t0,scl_fctr1,8);

					// c[9:0-15]
					BF16_F32_MATRIX_MUL_1COL(t0,scl_fctr1,9);

					// c[10:0-15]
					BF16_F32_MATRIX_MUL_1COL(t0,scl_fctr1,10);

					// c[11:0-15]
					BF16_F32_MATRIX_MUL_1COL(t0,scl_fctr1,11);
				}
				else
				{
					// c[0:0-15]
					BF16_F32_MATRIX_MUL_1COL(t0,scl_fctr1,0);

					// c[1:0-15]
					BF16_F32_MATRIX_MUL_1COL(t0,scl_fctr2,1);

					// c[2:0-15]
					BF16_F32_MATRIX_MUL_1COL(t0,scl_fctr3,2);

					// c[3:0-15]
					BF16_F32_MATRIX_MUL_1COL(t0,scl_fctr4,3);

					// c[4:0-15]
					BF16_F32_MATRIX_MUL_1COL(t0,scl_fctr5,4);

					// c[5:0-15]
					BF16_F32_MATRIX_MUL_1COL(t0,scl_fctr6,5);

					// c[6:0-15]
					BF16_F32_MATRIX_MUL_1COL(t0,scl_fctr7,6);

					// c[7:0-15]
					BF16_F32_MATRIX_MUL_1COL(t0,scl_fctr8,7);

					// c[8:0-15]
					BF16_F32_MATRIX_MUL_1COL(t0,scl_fctr9,8);

					// c[9:0-15]
					BF16_F32_MATRIX_MUL_1COL(t0,scl_fctr10,9);

					// c[10:0-15]
					BF16_F32_MATRIX_MUL_1COL(t0,scl_fctr11,10);

					// c[11:0-15]
					BF16_F32_MATRIX_MUL_1COL(t0,scl_fctr12,11);
				}
			}
			else if ( is_f32 == TRUE )
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15]
					F32_U8S8_MATRIX_MUL_1COL(t0,scl_fctr1,0);

					// c[1:0-15]
					F32_U8S8_MATRIX_MUL_1COL(t0,scl_fctr1,1);

					// c[2:0-15]
					F32_U8S8_MATRIX_MUL_1COL(t0,scl_fctr1,2);

					// c[3:0-15]
					F32_U8S8_MATRIX_MUL_1COL(t0,scl_fctr1,3);

					// c[4:0-15]
					F32_U8S8_MATRIX_MUL_1COL(t0,scl_fctr1,4);

					// c[5:0-15]
					F32_U8S8_MATRIX_MUL_1COL(t0,scl_fctr1,5);

					// c[6:0-15]
					F32_U8S8_MATRIX_MUL_1COL(t0,scl_fctr1,6);

					// c[7:0-15]
					F32_U8S8_MATRIX_MUL_1COL(t0,scl_fctr1,7);

					// c[8:0-15]
					F32_U8S8_MATRIX_MUL_1COL(t0,scl_fctr1,8);

					// c[9:0-15]
					F32_U8S8_MATRIX_MUL_1COL(t0,scl_fctr1,9);

					// c[10:0-15]
					F32_U8S8_MATRIX_MUL_1COL(t0,scl_fctr1,10);

					// c[11:0-15]
					F32_U8S8_MATRIX_MUL_1COL(t0,scl_fctr1,11);
				}
				else
				{
					// c[0:0-15]
					F32_U8S8_MATRIX_MUL_1COL(t0,scl_fctr1,0);

					// c[1:0-15]
					F32_U8S8_MATRIX_MUL_1COL(t0,scl_fctr2,1);

					// c[2:0-15]
					F32_U8S8_MATRIX_MUL_1COL(t0,scl_fctr3,2);

					// c[3:0-15]
					F32_U8S8_MATRIX_MUL_1COL(t0,scl_fctr4,3);

					// c[4:0-15]
					F32_U8S8_MATRIX_MUL_1COL(t0,scl_fctr5,4);

					// c[5:0-15]
					F32_U8S8_MATRIX_MUL_1COL(t0,scl_fctr6,5);

					// c[6:0-15]
					F32_U8S8_MATRIX_MUL_1COL(t0,scl_fctr7,6);

					// c[7:0-15]
					F32_U8S8_MATRIX_MUL_1COL(t0,scl_fctr8,7);

					// c[8:0-15]
					F32_U8S8_MATRIX_MUL_1COL(t0,scl_fctr9,8);

					// c[9:0-15]
					F32_U8S8_MATRIX_MUL_1COL(t0,scl_fctr10,9);

					// c[10:0-15]
					F32_U8S8_MATRIX_MUL_1COL(t0,scl_fctr11,10);

					// c[11:0-15]
					F32_U8S8_MATRIX_MUL_1COL(t0,scl_fctr12,11);
				}
			}
			else if ( is_s8 == TRUE )
			{
				int8_t* matptr = ( int8_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15]
					S8_F32_MATRIX_MUL_1COL(t0,scl_fctr1,0);

					// c[1:0-15]
					S8_F32_MATRIX_MUL_1COL(t0,scl_fctr1,1);

					// c[2:0-15]
					S8_F32_MATRIX_MUL_1COL(t0,scl_fctr1,2);

					// c[3:0-15]
					S8_F32_MATRIX_MUL_1COL(t0,scl_fctr1,3);

					// c[4:0-15]
					S8_F32_MATRIX_MUL_1COL(t0,scl_fctr1,4);

					// c[5:0-15]
					S8_F32_MATRIX_MUL_1COL(t0,scl_fctr1,5);

					// c[6:0-15]
					S8_F32_MATRIX_MUL_1COL(t0,scl_fctr1,6);

					// c[7:0-15]
					S8_F32_MATRIX_MUL_1COL(t0,scl_fctr1,7);

					// c[8:0-15]
					S8_F32_MATRIX_MUL_1COL(t0,scl_fctr1,8);

					// c[9:0-15]
					S8_F32_MATRIX_MUL_1COL(t0,scl_fctr1,9);

					// c[10:0-15]
					S8_F32_MATRIX_MUL_1COL(t0,scl_fctr1,10);

					// c[11:0-15]
					S8_F32_MATRIX_MUL_1COL(t0,scl_fctr1,11);
				}
				else
				{
					// c[0:0-15]
					S8_F32_MATRIX_MUL_1COL(t0,scl_fctr1,0);

					// c[1:0-15]
					S8_F32_MATRIX_MUL_1COL(t0,scl_fctr2,1);

					// c[2:0-15]
					S8_F32_MATRIX_MUL_1COL(t0,scl_fctr3,2);

					// c[3:0-15]
					S8_F32_MATRIX_MUL_1COL(t0,scl_fctr4,3);

					// c[4:0-15]
					S8_F32_MATRIX_MUL_1COL(t0,scl_fctr5,4);

					// c[5:0-15]
					S8_F32_MATRIX_MUL_1COL(t0,scl_fctr6,5);

					// c[6:0-15]
					S8_F32_MATRIX_MUL_1COL(t0,scl_fctr7,6);

					// c[7:0-15]
					S8_F32_MATRIX_MUL_1COL(t0,scl_fctr8,7);

					// c[8:0-15]
					S8_F32_MATRIX_MUL_1COL(t0,scl_fctr9,8);

					// c[9:0-15]
					S8_F32_MATRIX_MUL_1COL(t0,scl_fctr10,9);

					// c[10:0-15]
					S8_F32_MATRIX_MUL_1COL(t0,scl_fctr11,10);

					// c[11:0-15]
					S8_F32_MATRIX_MUL_1COL(t0,scl_fctr12,11);
				}
			}
			else
			{
				int32_t* matptr = ( int32_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15]
					S32_F32_MATRIX_MUL_1COL(t0,scl_fctr1,0);

					// c[1:0-15]
					S32_F32_MATRIX_MUL_1COL(t0,scl_fctr1,1);

					// c[2:0-15]
					S32_F32_MATRIX_MUL_1COL(t0,scl_fctr1,2);

					// c[3:0-15]
					S32_F32_MATRIX_MUL_1COL(t0,scl_fctr1,3);

					// c[4:0-15]
					S32_F32_MATRIX_MUL_1COL(t0,scl_fctr1,4);

					// c[5:0-15]
					S32_F32_MATRIX_MUL_1COL(t0,scl_fctr1,5);

					// c[6:0-15]
					S32_F32_MATRIX_MUL_1COL(t0,scl_fctr1,6);

					// c[7:0-15]
					S32_F32_MATRIX_MUL_1COL(t0,scl_fctr1,7);

					// c[8:0-15]
					S32_F32_MATRIX_MUL_1COL(t0,scl_fctr1,8);

					// c[9:0-15]
					S32_F32_MATRIX_MUL_1COL(t0,scl_fctr1,9);

					// c[10:0-15]
					S32_F32_MATRIX_MUL_1COL(t0,scl_fctr1,10);

					// c[11:0-15]
					S32_F32_MATRIX_MUL_1COL(t0,scl_fctr1,11);
				}
				else
				{
					// c[0:0-15]
					S32_F32_MATRIX_MUL_1COL(t0,scl_fctr1,0);

					// c[1:0-15]
					S32_F32_MATRIX_MUL_1COL(t0,scl_fctr2,1);

					// c[2:0-15]
					S32_F32_MATRIX_MUL_1COL(t0,scl_fctr3,2);

					// c[3:0-15]
					S32_F32_MATRIX_MUL_1COL(t0,scl_fctr4,3);

					// c[4:0-15]
					S32_F32_MATRIX_MUL_1COL(t0,scl_fctr5,4);

					// c[5:0-15]
					S32_F32_MATRIX_MUL_1COL(t0,scl_fctr6,5);

					// c[6:0-15]
					S32_F32_MATRIX_MUL_1COL(t0,scl_fctr7,6);

					// c[7:0-15]
					S32_F32_MATRIX_MUL_1COL(t0,scl_fctr8,7);

					// c[8:0-15]
					S32_F32_MATRIX_MUL_1COL(t0,scl_fctr9,8);

					// c[9:0-15]
					S32_F32_MATRIX_MUL_1COL(t0,scl_fctr10,9);

					// c[10:0-15]
					S32_F32_MATRIX_MUL_1COL(t0,scl_fctr10,10);

					// c[11:0-15]
					S32_F32_MATRIX_MUL_1COL(t0,scl_fctr12,11);
				}
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SWISH_12x16:
		{
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

			__m512 al_in, r, r2, z, dn;
			__m512i temp;

			// c[0, 0-15]
			SWISH_F32_AVX512_DEF(acc_00, scale, al_in, r, r2, z, dn, temp);

			// c[1, 0-15]
			SWISH_F32_AVX512_DEF(acc_10, scale, al_in, r, r2, z, dn, temp);

			// c[2, 0-15]
			SWISH_F32_AVX512_DEF(acc_20, scale, al_in, r, r2, z, dn, temp);

			// c[3, 0-15]
			SWISH_F32_AVX512_DEF(acc_30, scale, al_in, r, r2, z, dn, temp);

			// c[4, 0-15]
			SWISH_F32_AVX512_DEF(acc_40, scale, al_in, r, r2, z, dn, temp);

			// c[5, 0-15]
			SWISH_F32_AVX512_DEF(acc_50, scale, al_in, r, r2, z, dn, temp);

			// c[6, 0-15]
			SWISH_F32_AVX512_DEF(acc_60, scale, al_in, r, r2, z, dn, temp);

			// c[7, 0-15]
			SWISH_F32_AVX512_DEF(acc_70, scale, al_in, r, r2, z, dn, temp);

			// c[8, 0-15]
			SWISH_F32_AVX512_DEF(acc_80, scale, al_in, r, r2, z, dn, temp);

			// c[9, 0-15]
			SWISH_F32_AVX512_DEF(acc_90, scale, al_in, r, r2, z, dn, temp);

			// c[10, 0-15]
			SWISH_F32_AVX512_DEF(acc_100, scale, al_in, r, r2, z, dn, temp);

			// c[11, 0-15]
			SWISH_F32_AVX512_DEF(acc_110, scale, al_in, r, r2, z, dn, temp);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_TANH_12x16:
		{
			__m512 dn, z, x, r2, r;
			__m512i q;

			// c[0, 0-15]
			TANHF_AVX512(acc_00, r, r2, x, z, dn, q)

			// c[1, 0-15]
			TANHF_AVX512(acc_10, r, r2, x, z, dn, q)

			// c[2, 0-15]
			TANHF_AVX512(acc_20, r, r2, x, z, dn, q)

			// c[3, 0-15]
			TANHF_AVX512(acc_30, r, r2, x, z, dn, q)

			// c[4, 0-15]
			TANHF_AVX512(acc_40, r, r2, x, z, dn, q)

			// c[5, 0-15]
			TANHF_AVX512(acc_50, r, r2, x, z, dn, q)

			// c[6, 0-15]
			TANHF_AVX512(acc_60, r, r2, x, z, dn, q)

			// c[7, 0-15]
			TANHF_AVX512(acc_70, r, r2, x, z, dn, q)

			// c[8, 0-15]
			TANHF_AVX512(acc_80, r, r2, x, z, dn, q)

			// c[9, 0-15]
			TANHF_AVX512(acc_90, r, r2, x, z, dn, q)

			// c[10, 0-15]
			TANHF_AVX512(acc_100, r, r2, x, z, dn, q)

			// c[11, 0-15]
			TANHF_AVX512(acc_110, r, r2, x, z, dn, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SIGMOID_12x16:
		{
			__m512 al_in, r, r2, z, dn;
			__m512i tmpout;

			// c[0, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_00, al_in, r, r2, z, dn, tmpout);

			// c[1, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_10, al_in, r, r2, z, dn, tmpout);

			// c[2, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_20, al_in, r, r2, z, dn, tmpout);

			// c[3, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_30, al_in, r, r2, z, dn, tmpout);

			// c[4, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_40, al_in, r, r2, z, dn, tmpout);

			// c[5, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_50, al_in, r, r2, z, dn, tmpout);

			// c[6, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_60, al_in, r, r2, z, dn, tmpout);

			// c[7, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_70, al_in, r, r2, z, dn, tmpout);

			// c[8, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_80, al_in, r, r2, z, dn, tmpout);

			// c[9, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_90, al_in, r, r2, z, dn, tmpout);

			// c[10, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_100, al_in, r, r2, z, dn, tmpout);

			// c[11, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_110, al_in, r, r2, z, dn, tmpout);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_12x16_DISABLE:
		;

		if ( ( post_ops_attr.buf_downscale != NULL ) && ( post_ops_attr.is_last_k == TRUE ) )
		{
			// Generate a mask16 of all 1's.
			selector1 = _mm512_setzero_epi32();
			selector2 = _mm512_set1_epi32( 10 );
			__mmask16 mask_all1 = _mm512_cmplt_epi32_mask( selector1, selector2 );

			if ( post_ops_attr.c_stor_type == S8)
			{
				// Store the results in downscaled type (int8 instead of int32).
				// c[0,0-15]
				CVT_STORE_F32_S8(acc_00,0,0);

				// c[1,0-15]
				CVT_STORE_F32_S8(acc_10,1,0);

				// c[2,0-15]
				CVT_STORE_F32_S8(acc_20,2,0);

				// c[3,0-15]
				CVT_STORE_F32_S8(acc_30,3,0);

				// c[4,0-15]
				CVT_STORE_F32_S8(acc_40,4,0);

				// c[5,0-15]
				CVT_STORE_F32_S8(acc_50,5,0);

				// c[6,0-15]
				CVT_STORE_F32_S8(acc_60,6,0);

				// c[7,0-15]
				CVT_STORE_F32_S8(acc_70,7,0);

				// c[8,0-15]
				CVT_STORE_F32_S8(acc_80,8,0);

				// c[9,0-15]
				CVT_STORE_F32_S8(acc_90,9,0);

				// c[10,0-15]
				CVT_STORE_F32_S8(acc_100,10,0);

				// c[11,0-15]
				CVT_STORE_F32_S8(acc_110,11,0);
			}
			else if ( post_ops_attr.c_stor_type == U8 )
			{
				// Store the results in downscaled type (uint8 instead of int32).
				// c[0,0-15]
				CVT_STORE_F32_U8(acc_00,0,0);

				// c[1,0-15]
				CVT_STORE_F32_U8(acc_10,1,0);

				// c[2,0-15]
				CVT_STORE_F32_U8(acc_20,2,0);

				// c[3,0-15]
				CVT_STORE_F32_U8(acc_30,3,0);

				// c[4,0-15]
				CVT_STORE_F32_U8(acc_40,4,0);

				// c[5,0-15]
				CVT_STORE_F32_U8(acc_50,5,0);

				// c[6,0-15]
				CVT_STORE_F32_U8(acc_60,6,0);

				// c[7,0-15]
				CVT_STORE_F32_U8(acc_70,7,0);

				// c[8,0-15]
				CVT_STORE_F32_U8(acc_80,8,0);

				// c[9,0-15]
				CVT_STORE_F32_U8(acc_90,9,0);

				// c[10,0-15]
				CVT_STORE_F32_U8(acc_100,10,0);

				// c[11,0-15]
				CVT_STORE_F32_U8(acc_110,11,0);
			}
			else if ( post_ops_attr.c_stor_type == BF16)
			{
				// Store the results in downscaled type (int8 instead of int32).
				// c[0,0-15]
				CVT_STORE_F32_BF16(acc_00,0,0);

				// c[1,0-15]
				CVT_STORE_F32_BF16(acc_10,1,0);

				// c[2,0-15]
				CVT_STORE_F32_BF16(acc_20,2,0);

				// c[3,0-15]
				CVT_STORE_F32_BF16(acc_30,3,0);

				// c[4,0-15]
				CVT_STORE_F32_BF16(acc_40,4,0);

				// c[5,0-15]
				CVT_STORE_F32_BF16(acc_50,5,0);

				// c[6,0-15]
				CVT_STORE_F32_BF16(acc_60,6,0);

				// c[7,0-15]
				CVT_STORE_F32_BF16(acc_70,7,0);

				// c[8,0-15]
				CVT_STORE_F32_BF16(acc_80,8,0);

				// c[9,0-15]
				CVT_STORE_F32_BF16(acc_90,9,0);

				// c[10,0-15]
				CVT_STORE_F32_BF16(acc_100,10,0);

				// c[11,0-15]
				CVT_STORE_F32_BF16(acc_110,11,0);
			}
			else if ( post_ops_attr.c_stor_type == F32)
			{
				// Store the results in downscaled type (int8 instead of int32).
				// c[0,0-15]
				STORE_F32(acc_00,0,0);

				// c[1,0-15]
				STORE_F32(acc_10,1,0);

				// c[2,0-15]
				STORE_F32(acc_20,2,0);

				// c[3,0-15]
				STORE_F32(acc_30,3,0);

				// c[4,0-15]
				STORE_F32(acc_40,4,0);

				// c[5,0-15]
				STORE_F32(acc_50,5,0);

				// c[6,0-15]
				STORE_F32(acc_60,6,0);

				// c[7,0-15]
				STORE_F32(acc_70,7,0);

				// c[8,0-15]
				STORE_F32(acc_80,8,0);

				// c[9,0-15]
				STORE_F32(acc_90,9,0);

				// c[10,0-15]
				STORE_F32(acc_100,10,0);

				// c[11,0-15]
				STORE_F32(acc_110,11,0);
			}
		}
		else
		{
			// Store the results.
			// c[0,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 0 ) ) + ( 0*16 ),
				_mm512_cvtps_epi32( acc_00 ) );

			// c[1,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 1 ) ) + ( 0*16 ),
				_mm512_cvtps_epi32( acc_10 ) );

			// c[2,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 2 ) ) + ( 0*16 ),
				_mm512_cvtps_epi32( acc_20 ) );

			// c[3,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 3 ) ) + ( 0*16 ),
				_mm512_cvtps_epi32( acc_30 ) );

			// c[4,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 4 ) ) + ( 0*16 ),
				_mm512_cvtps_epi32( acc_40 ) );

			// c[5,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 5 ) ) + ( 0*16 ),
				_mm512_cvtps_epi32( acc_50 ) );

			// c[6,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 6 ) ) + ( 0*16 ),
				_mm512_cvtps_epi32( acc_60 ) );

			// c[7,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 7 ) ) + ( 0*16 ),
				_mm512_cvtps_epi32( acc_70 ) );

			// c[8,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 8 ) ) + ( 0*16 ),
				_mm512_cvtps_epi32( acc_80 ) );

			// c[9,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 9 ) ) + ( 0*16 ),
				_mm512_cvtps_epi32( acc_90 ) );

			// c[10,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 10 ) ) + ( 0*16 ),
				_mm512_cvtps_epi32( acc_100 ) );

			// c[11,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 11 ) ) + ( 0*16 ),
				_mm512_cvtps_epi32( acc_110 ) );
		}

		a = a + ( MR * ps_a );
		post_ops_attr.post_op_c_i += MR;
	}

	if ( m_partial_pieces > 0 )
	{
		lpgemm_rowvar_u8s8s32o32_6x16
		(
		  m_partial_pieces, k0,
		  a, rs_a, cs_a, ps_a,
		  b, rs_b, cs_b,
		  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
		  alpha, beta,
		  post_ops_list, post_ops_attr
		);
	}
}

// 9x32 int8o32 fringe kernel
__attribute__((aligned(64)))
LPGEMM_N_FRINGE_KERN(uint8_t,int8_t,int32_t,u8s8s32o32_9x32)
{
	static void* post_ops_labels[] =
				{
				  &&POST_OPS_9x32_DISABLE,
				  &&POST_OPS_BIAS_9x32,
				  &&POST_OPS_RELU_9x32,
				  &&POST_OPS_RELU_SCALE_9x32,
				  &&POST_OPS_GELU_TANH_9x32,
				  &&POST_OPS_GELU_ERF_9x32,
				  &&POST_OPS_CLIP_9x32,
				  &&POST_OPS_DOWNSCALE_9x32,
				  &&POST_OPS_MATRIX_ADD_9x32,
				  &&POST_OPS_SWISH_9x32,
				  &&POST_OPS_MATRIX_MUL_9x32,
				  &&POST_OPS_TANH_9x32,
				  &&POST_OPS_SIGMOID_9x32
				};
	dim_t MR = 9;
	dim_t m_full_pieces = m0 / MR;
	dim_t m_full_pieces_loop_limit = m_full_pieces * MR;
	dim_t m_partial_pieces = m0 % MR;

	dim_t k_full_pieces = k0 / 4;
	dim_t k_partial_pieces = k0 % 4;

	// B matrix storage.
	__m512i b0 = _mm512_setzero_epi32();
	__m512i b1 = _mm512_setzero_epi32();

	// A matrix storage.
	__m512i a_int32_0 = _mm512_setzero_epi32();
	__m512i a_int32_1 = _mm512_setzero_epi32();
	__m512i a_int32_2 = _mm512_setzero_epi32();
	__m512i a_int32_3 = _mm512_setzero_epi32();

	__m512i selector1;
	__m512i selector2;

	for ( dim_t ir = 0; ir < m_full_pieces_loop_limit; ir += MR )
	{
		_mm_prefetch( b, _MM_HINT_T0 );
		_mm_prefetch( a + ( MR * ps_a ) + ( 0 * 16 ), _MM_HINT_T1 );

		// Registers to use for accumulating C.
		__m512i c_int32_0p0 = _mm512_setzero_epi32();
		__m512i c_int32_0p1 = _mm512_setzero_epi32();

		__m512i c_int32_1p0 = _mm512_setzero_epi32();
		__m512i c_int32_1p1 = _mm512_setzero_epi32();

		__m512i c_int32_2p0 = _mm512_setzero_epi32();
		__m512i c_int32_2p1 = _mm512_setzero_epi32();

		__m512i c_int32_3p0 = _mm512_setzero_epi32();
		__m512i c_int32_3p1 = _mm512_setzero_epi32();

		__m512i c_int32_4p0 = _mm512_setzero_epi32();
		__m512i c_int32_4p1 = _mm512_setzero_epi32();

		__m512i c_int32_5p0 = _mm512_setzero_epi32();
		__m512i c_int32_5p1 = _mm512_setzero_epi32();

		__m512i c_int32_6p0 = _mm512_setzero_epi32();
		__m512i c_int32_6p1 = _mm512_setzero_epi32();

		__m512i c_int32_7p0 = _mm512_setzero_epi32();
		__m512i c_int32_7p1 = _mm512_setzero_epi32();

		__m512i c_int32_8p0 = _mm512_setzero_epi32();
		__m512i c_int32_8p1 = _mm512_setzero_epi32();

		for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
		{
			// Load 4 rows with 32 elements each from B to 2 ZMM registers. It
			// is to be noted that the B matrix is packed for use in vnni
			// instructions and each load to ZMM register will have 4 elements
			// along k direction and 16 elements across n directions, so 4x16
			// elements to a ZMM register.
			b0 = _mm512_loadu_si512( b + ( rs_b * kr ) + ( cs_b * 0 ) );
			b1 = _mm512_loadu_si512( b + ( rs_b * kr ) + ( cs_b * 1 ) );

			// Broadcast a[0,kr:kr+4].
			a_int32_0 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

			// Broadcast a[1,kr:kr+4].
			a_int32_1 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 1 ) + ( cs_a * kr ) ) );

			// Broadcast a[2,kr:kr+4].
			a_int32_2 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 2 ) + ( cs_a * kr ) ) );

			// Broadcast a[3,kr:kr+4].
			a_int32_3 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 3 ) + ( cs_a * kr ) ) );

			// Broadcast a[4,kr:kr+4].
			selector1 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 4 ) + ( cs_a * kr ) ) );

			// Broadcast a[5,kr:kr+4].
			selector2 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 5 ) + ( cs_a * kr ) ) );

			// Perform column direction mat-mul with k = 4.
			// c[0,0-31] = a[0,kr:kr+4]*b[kr:kr+4,0-31]
			c_int32_0p0 = _mm512_dpbusd_epi32( c_int32_0p0, a_int32_0, b0 );
			c_int32_0p1 = _mm512_dpbusd_epi32( c_int32_0p1, a_int32_0, b1 );

			// Perform column direction mat-mul with k = 4.
			// c[1,0-31] = a[1,kr:kr+4]*b[kr:kr+4,0-31]
			c_int32_1p0 = _mm512_dpbusd_epi32( c_int32_1p0, a_int32_1, b0 );
			c_int32_1p1 = _mm512_dpbusd_epi32( c_int32_1p1, a_int32_1, b1 );

			// Perform column direction mat-mul with k = 4.
			// c[2,0-31] = a[2,kr:kr+4]*b[kr:kr+4,0-31]
			c_int32_2p0 = _mm512_dpbusd_epi32( c_int32_2p0, a_int32_2, b0 );
			c_int32_2p1 = _mm512_dpbusd_epi32( c_int32_2p1, a_int32_2, b1 );

			// Perform column direction mat-mul with k = 4.
			// c[3,0-31] = a[3,kr:kr+4]*b[kr:kr+4,0-31]
			c_int32_3p0 = _mm512_dpbusd_epi32( c_int32_3p0, a_int32_3, b0 );
			c_int32_3p1 = _mm512_dpbusd_epi32( c_int32_3p1, a_int32_3, b1 );

			// Perform column direction mat-mul with k = 4.
			// c[4,0-31] = a[4,kr:kr+4]*b[kr:kr+4,0-31]
			c_int32_4p0 = _mm512_dpbusd_epi32( c_int32_4p0, selector1, b0 );
			c_int32_4p1 = _mm512_dpbusd_epi32( c_int32_4p1, selector1, b1 );

			// Perform column direction mat-mul with k = 4.
			// c[5,0-31] = a[5,kr:kr+4]*b[kr:kr+4,0-31]
			c_int32_5p0 = _mm512_dpbusd_epi32( c_int32_5p0, selector2, b0 );
			c_int32_5p1 = _mm512_dpbusd_epi32( c_int32_5p1, selector2, b1 );

			// Broadcast a[6,kr:kr+4].
			a_int32_0 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 6 ) + ( cs_a * kr ) ) );

			// Broadcast a[7,kr:kr+4].
			a_int32_1 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 7 ) + ( cs_a * kr ) ) );

			// Broadcast a[8,kr:kr+4].
			a_int32_2 = _mm512_set1_epi32( *( uint32_t* )( a + ( rs_a * 8 ) + ( cs_a * kr ) ) );

			// Perform column direction mat-mul with k = 4.
			// c[6,0-31] = a[6,kr:kr+4]*b[kr:kr+4,0-31]
			c_int32_6p0 = _mm512_dpbusd_epi32( c_int32_6p0, a_int32_0, b0 );
			c_int32_6p1 = _mm512_dpbusd_epi32( c_int32_6p1, a_int32_0, b1 );

			// Perform column direction mat-mul with k = 4.
			// c[7,0-31] = a[7,kr:kr+4]*b[kr:kr+4,0-31]
			c_int32_7p0 = _mm512_dpbusd_epi32( c_int32_7p0, a_int32_1, b0 );
			c_int32_7p1 = _mm512_dpbusd_epi32( c_int32_7p1, a_int32_1, b1 );

			// Perform column direction mat-mul with k = 4.
			// c[8,0-31] = a[8,kr:kr+4]*b[kr:kr+4,0-31]
			c_int32_8p0 = _mm512_dpbusd_epi32( c_int32_8p0, a_int32_2, b0 );
			c_int32_8p1 = _mm512_dpbusd_epi32( c_int32_8p1, a_int32_2, b1 );
		}
		__asm__(".p2align 6\n");
		// Handle k remainder.
		if ( k_partial_pieces > 0 )
		{
			__m128i a_kfringe_buf;
			__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - k_partial_pieces ) );

			b0 = _mm512_loadu_si512( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );
			b1 = _mm512_loadu_si512( b + ( rs_b * k_full_pieces ) + ( cs_b * 1 ) );

			// Broadcast a[0,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_0 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[0,0-31] = a[0,kr:kr+4]*b[kr:kr+4,0-31]
			c_int32_0p0 = _mm512_dpbusd_epi32( c_int32_0p0, a_int32_0, b0 );
			c_int32_0p1 = _mm512_dpbusd_epi32( c_int32_0p1, a_int32_0, b1 );

			// Broadcast a[1,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 1 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_1 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[1,0-31] = a[1,kr:kr+4]*b[kr:kr+4,0-31]
			c_int32_1p0 = _mm512_dpbusd_epi32( c_int32_1p0, a_int32_1, b0 );
			c_int32_1p1 = _mm512_dpbusd_epi32( c_int32_1p1, a_int32_1, b1 );

			// Broadcast a[2,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 2 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_2 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[2,0-31] = a[2,kr:kr+4]*b[kr:kr+4,0-31]
			c_int32_2p0 = _mm512_dpbusd_epi32( c_int32_2p0, a_int32_2, b0 );
			c_int32_2p1 = _mm512_dpbusd_epi32( c_int32_2p1, a_int32_2, b1 );

			// Broadcast a[3,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 3 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_3 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[3,0-31] = a[3,kr:kr+4]*b[kr:kr+4,0-31]
			c_int32_3p0 = _mm512_dpbusd_epi32( c_int32_3p0, a_int32_3, b0 );
			c_int32_3p1 = _mm512_dpbusd_epi32( c_int32_3p1, a_int32_3, b1 );

			// Broadcast a[4,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 4 ) + ( cs_a * k_full_pieces ) )
			);
			selector1 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[4,0-31] = a[4,kr:kr+4]*b[kr:kr+4,0-31]
			c_int32_4p0 = _mm512_dpbusd_epi32( c_int32_4p0, selector1, b0 );
			c_int32_4p1 = _mm512_dpbusd_epi32( c_int32_4p1, selector1, b1 );

			// Broadcast a[5,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 5 ) + ( cs_a * k_full_pieces ) )
			);
			selector2 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[5,0-31] = a[5,kr:kr+4]*b[kr:kr+4,0-31]
			c_int32_5p0 = _mm512_dpbusd_epi32( c_int32_5p0, selector2, b0 );
			c_int32_5p1 = _mm512_dpbusd_epi32( c_int32_5p1, selector2, b1 );

			// Broadcast a[6,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 6 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_0 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[6,0-31] = a[6,kr:kr+4]*b[kr:kr+4,0-31]
			c_int32_6p0 = _mm512_dpbusd_epi32( c_int32_6p0, a_int32_0, b0 );
			c_int32_6p1 = _mm512_dpbusd_epi32( c_int32_6p1, a_int32_0, b1 );

			// Broadcast a[7,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 7 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_1 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[7,0-31] = a[7,kr:kr+4]*b[kr:kr+4,0-31]
			c_int32_7p0 = _mm512_dpbusd_epi32( c_int32_7p0, a_int32_1, b0 );
			c_int32_7p1 = _mm512_dpbusd_epi32( c_int32_7p1, a_int32_1, b1 );

			// Broadcast a[8,kr:kr+4].
			a_kfringe_buf = _mm_maskz_loadu_epi8
			(
			  load_mask,
			  ( a + ( rs_a * 8 ) + ( cs_a * k_full_pieces ) )
			);
			a_int32_2 = _mm512_broadcastd_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[8,0-31] = a[8,kr:kr+4]*b[kr:kr+4,0-31]
			c_int32_8p0 = _mm512_dpbusd_epi32( c_int32_8p0, a_int32_2, b0 );
			c_int32_8p1 = _mm512_dpbusd_epi32( c_int32_8p1, a_int32_2, b1 );
		}

		// Load alpha and beta
		selector1 = _mm512_set1_epi32( alpha );
		selector2 = _mm512_set1_epi32( beta );

		if ( alpha != 1 )
		{
			// Scale by alpha
			c_int32_0p0 = _mm512_mullo_epi32( selector1, c_int32_0p0 );
			c_int32_0p1 = _mm512_mullo_epi32( selector1, c_int32_0p1 );

			c_int32_1p0 = _mm512_mullo_epi32( selector1, c_int32_1p0 );
			c_int32_1p1 = _mm512_mullo_epi32( selector1, c_int32_1p1 );

			c_int32_2p0 = _mm512_mullo_epi32( selector1, c_int32_2p0 );
			c_int32_2p1 = _mm512_mullo_epi32( selector1, c_int32_2p1 );

			c_int32_3p0 = _mm512_mullo_epi32( selector1, c_int32_3p0 );
			c_int32_3p1 = _mm512_mullo_epi32( selector1, c_int32_3p1 );

			c_int32_4p0 = _mm512_mullo_epi32( selector1, c_int32_4p0 );
			c_int32_4p1 = _mm512_mullo_epi32( selector1, c_int32_4p1 );

			c_int32_5p0 = _mm512_mullo_epi32( selector1, c_int32_5p0 );
			c_int32_5p1 = _mm512_mullo_epi32( selector1, c_int32_5p1 );

			c_int32_6p0 = _mm512_mullo_epi32( selector1, c_int32_6p0 );
			c_int32_6p1 = _mm512_mullo_epi32( selector1, c_int32_6p1 );

			c_int32_7p0 = _mm512_mullo_epi32( selector1, c_int32_7p0 );
			c_int32_7p1 = _mm512_mullo_epi32( selector1, c_int32_7p1 );

			c_int32_8p0 = _mm512_mullo_epi32( selector1, c_int32_8p0 );
			c_int32_8p1 = _mm512_mullo_epi32( selector1, c_int32_8p1 );
		}

		// Scale C by beta.
		if ( beta != 0 )
		{
			if ( ( post_ops_attr.buf_downscale != NULL ) &&
				 ( post_ops_attr.is_first_k == TRUE ) )
			{
				if( post_ops_attr.c_stor_type == S8 )
				{
					// c[0:0-15,16-31]
					S8_S32_BETA_OP2(ir,0,selector1,selector2);

					// c[1:0-15,16-31]
					S8_S32_BETA_OP2(ir,1,selector1,selector2);

					// c[2:0-15,16-31]
					S8_S32_BETA_OP2(ir,2,selector1,selector2);

					// c[3:0-15,16-31]
					S8_S32_BETA_OP2(ir,3,selector1,selector2);

					// c[4:0-15,16-31]
					S8_S32_BETA_OP2(ir,4,selector1,selector2);

					// c[5:0-15,16-31]
					S8_S32_BETA_OP2(ir,5,selector1,selector2);

					// c[6:0-15,16-31]
					S8_S32_BETA_OP2(ir,6,selector1,selector2);

					// c[7:0-15,16-31]
					S8_S32_BETA_OP2(ir,7,selector1,selector2);

					// c[8:0-15,16-31]
					S8_S32_BETA_OP2(ir,8,selector1,selector2);
				}
				else if( post_ops_attr.c_stor_type == U8 )
				{
					// c[0:0-15,16-31]
					U8_S32_BETA_OP2(ir,0,selector1,selector2);

					// c[1:0-15,16-31]
					U8_S32_BETA_OP2(ir,1,selector1,selector2);

					// c[2:0-15,16-31]
					U8_S32_BETA_OP2(ir,2,selector1,selector2);

					// c[3:0-15,16-31]
					U8_S32_BETA_OP2(ir,3,selector1,selector2);

					// c[4:0-15,16-31]
					U8_S32_BETA_OP2(ir,4,selector1,selector2);

					// c[5:0-15,16-31]
					U8_S32_BETA_OP2(ir,5,selector1,selector2);

					// c[6:0-15,16-31]
					U8_S32_BETA_OP2(ir,6,selector1,selector2);

					// c[7:0-15,16-31]
					U8_S32_BETA_OP2(ir,7,selector1,selector2);

					// c[8:0-15,16-31]
					U8_S32_BETA_OP2(ir,8,selector1,selector2);
				}
				else if( post_ops_attr.c_stor_type == BF16 )
				{
					// c[0:0-15,16-31]
					BF16_S32_BETA_OP2(ir,0,selector1,selector2);

					// c[1:0-15,16-31]
					BF16_S32_BETA_OP2(ir,1,selector1,selector2);

					// c[2:0-15,16-31]
					BF16_S32_BETA_OP2(ir,2,selector1,selector2);

					// c[3:0-15,16-31]
					BF16_S32_BETA_OP2(ir,3,selector1,selector2);

					// c[4:0-15,16-31]
					BF16_S32_BETA_OP2(ir,4,selector1,selector2);

					// c[5:0-15,16-31]
					BF16_S32_BETA_OP2(ir,5,selector1,selector2);

					// c[6:0-15,16-31]
					BF16_S32_BETA_OP2(ir,6,selector1,selector2);

					// c[7:0-15,16-31]
					BF16_S32_BETA_OP2(ir,7,selector1,selector2);

					// c[8:0-15,16-31]
					BF16_S32_BETA_OP2(ir,8,selector1,selector2);
				}
				else if( post_ops_attr.c_stor_type == F32 )
				{
					// c[0:0-15,16-31]
					F32_S32_BETA_OP2(ir,0,selector1,selector2);

					// c[1:0-15,16-31]
					F32_S32_BETA_OP2(ir,1,selector1,selector2);

					// c[2:0-15,16-31]
					F32_S32_BETA_OP2(ir,2,selector1,selector2);

					// c[3:0-15,16-31]
					F32_S32_BETA_OP2(ir,3,selector1,selector2);

					// c[4:0-15,16-31]
					F32_S32_BETA_OP2(ir,4,selector1,selector2);

					// c[5:0-15,16-31]
					F32_S32_BETA_OP2(ir,5,selector1,selector2);

					// c[6:0-15,16-31]
					F32_S32_BETA_OP2(ir,6,selector1,selector2);

					// c[7:0-15,16-31]
					F32_S32_BETA_OP2(ir,7,selector1,selector2);

					// c[8:0-15,16-31]
					F32_S32_BETA_OP2(ir,8,selector1,selector2);
				}
			}
			else
			{
				// c[0:0-15,16-31]
				S32_S32_BETA_OP2(ir,0,selector1,selector2);

				// c[1:0-15,16-31]
				S32_S32_BETA_OP2(ir,1,selector1,selector2);

				// c[2:0-15,16-31]
				S32_S32_BETA_OP2(ir,2,selector1,selector2);

				// c[3:0-15,16-31]
				S32_S32_BETA_OP2(ir,3,selector1,selector2);

				// c[4:0-15,16-31]
				S32_S32_BETA_OP2(ir,4,selector1,selector2);

				// c[5:0-15,16-31]
				S32_S32_BETA_OP2(ir,5,selector1,selector2);

				// c[6:0-15,16-31]
				S32_S32_BETA_OP2(ir,6,selector1,selector2);

				// c[7:0-15,16-31]
				S32_S32_BETA_OP2(ir,7,selector1,selector2);

				// c[8:0-15,16-31]
				S32_S32_BETA_OP2(ir,8,selector1,selector2);
			}
		}

		__m512 acc_00 = _mm512_setzero_ps();
		__m512 acc_01 = _mm512_setzero_ps();
		__m512 acc_10 = _mm512_setzero_ps();
		__m512 acc_11 = _mm512_setzero_ps();
		__m512 acc_20 = _mm512_setzero_ps();
		__m512 acc_21 = _mm512_setzero_ps();
		__m512 acc_30 = _mm512_setzero_ps();
		__m512 acc_31 = _mm512_setzero_ps();
		__m512 acc_40 = _mm512_setzero_ps();
		__m512 acc_41 = _mm512_setzero_ps();
		__m512 acc_50 = _mm512_setzero_ps();
		__m512 acc_51 = _mm512_setzero_ps();
		__m512 acc_60 = _mm512_setzero_ps();
		__m512 acc_61 = _mm512_setzero_ps();
		__m512 acc_70 = _mm512_setzero_ps();
		__m512 acc_71 = _mm512_setzero_ps();
		__m512 acc_80 = _mm512_setzero_ps();
		__m512 acc_81 = _mm512_setzero_ps();
		CVT_ACCUM_REG_INT_TO_FLOAT_9ROWS_XCOL(acc_, c_int32_, 2);

        // Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_9x32:
		{
			__mmask16 bias_mask = _cvtu32_mask16( 0xFFFF );
			__m512 b0 = _mm512_setzero_ps();
			__m512 b1 = _mm512_setzero_ps();

			if ( post_ops_list_temp->stor_type == BF16 )
			{
				BF16_F32_BIAS_LOAD(b0, bias_mask, 0);
				BF16_F32_BIAS_LOAD(b1, bias_mask, 1);
			}
			else if ( post_ops_list_temp->stor_type == S8 )
			{
				S8_F32_BIAS_LOAD(b0, bias_mask, 0);
				S8_F32_BIAS_LOAD(b1, bias_mask, 1);
			}
			else if ( post_ops_list_temp->stor_type == S32 )
			{
				S32_F32_BIAS_LOAD(b0, bias_mask, 0);
				S32_F32_BIAS_LOAD(b1, bias_mask, 1);
			}
			else
			{
				b0 = _mm512_maskz_loadu_ps ( bias_mask,
						( ( float* )post_ops_list_temp->op_args1 ) +
						post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				b1 = _mm512_maskz_loadu_ps ( bias_mask,
						( ( float* )post_ops_list_temp->op_args1 ) +
						post_ops_attr.post_op_c_j + ( 1 * 16 ) );
			}

			// c[0,0-15]
			acc_00 = _mm512_add_ps( b0, acc_00 );

			// c[0, 16-31]
			acc_01 = _mm512_add_ps( b1, acc_01 );

			// c[1,0-15]
			acc_10 = _mm512_add_ps( b0, acc_10 );

			// c[1, 16-31]
			acc_11 = _mm512_add_ps( b1, acc_11 );

			// c[2,0-15]
			acc_20 = _mm512_add_ps( b0, acc_20 );

			// c[2, 16-31]
			acc_21 = _mm512_add_ps( b1, acc_21 );

			// c[3,0-15]
			acc_30 = _mm512_add_ps( b0, acc_30 );

			// c[3, 16-31]
			acc_31 = _mm512_add_ps( b1, acc_31 );

			// c[4,0-15]
			acc_40 = _mm512_add_ps( b0, acc_40 );

			// c[4, 16-31]
			acc_41 = _mm512_add_ps( b1, acc_41 );

			// c[5,0-15]
			acc_50 = _mm512_add_ps( b0, acc_50 );

			// c[5, 16-31]
			acc_51 = _mm512_add_ps( b1, acc_51 );

			// c[6,0-15]
			acc_60 = _mm512_add_ps( b0, acc_60 );

			// c[6, 16-31]
			acc_61 = _mm512_add_ps( b1, acc_61 );

			// c[7,0-15]
			acc_70 = _mm512_add_ps( b0, acc_70 );

			// c[7, 16-31]
			acc_71 = _mm512_add_ps( b1, acc_71 );

			// c[8,0-15]
			acc_80 = _mm512_add_ps( b0, acc_80 );

			// c[8, 16-31]
			acc_81 = _mm512_add_ps( b1, acc_81 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_9x32:
		{
			__m512 zero = _mm512_setzero_ps();

			// c[0,0-15]
			acc_00 = _mm512_max_ps( zero, acc_00 );

			// c[0, 16-31]
			acc_01 = _mm512_max_ps( zero, acc_01 );

			// c[1,0-15]
			acc_10 = _mm512_max_ps( zero, acc_10 );

			// c[1,16-31]
			acc_11 = _mm512_max_ps( zero, acc_11 );

			// c[2,0-15]
			acc_20 = _mm512_max_ps( zero, acc_20 );

			// c[2,16-31]
			acc_21 = _mm512_max_ps( zero, acc_21 );

			// c[3,0-15]
			acc_30 = _mm512_max_ps( zero, acc_30 );

			// c[3,16-31]
			acc_31 = _mm512_max_ps( zero, acc_31 );

			// c[4,0-15]
			acc_40 = _mm512_max_ps( zero, acc_40 );

			// c[4,16-31]
			acc_41 = _mm512_max_ps( zero, acc_41 );

			// c[5,0-15]
			acc_50 = _mm512_max_ps( zero, acc_50 );

			// c[5,16-31]
			acc_51 = _mm512_max_ps( zero, acc_51 );

			// c[6,0-15]
			acc_60 = _mm512_max_ps( zero, acc_60 );

			// c[6, 16-31]
			acc_61 = _mm512_max_ps( zero, acc_61 );

			// c[7,0-15]
			acc_70 = _mm512_max_ps( zero, acc_70 );

			// c[7,16-31]
			acc_71 = _mm512_max_ps( zero, acc_71 );

			// c[8,0-15]
			acc_80 = _mm512_max_ps( zero, acc_80 );

			// c[8,16-31]
			acc_81 = _mm512_max_ps( zero, acc_81 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_9x32:
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

			// c[0, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_00)

			// c[0, 16-31]
			RELU_SCALE_OP_F32_AVX512(acc_01)

			// c[1, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_10)

			// c[1, 16-31]
			RELU_SCALE_OP_F32_AVX512(acc_11)

			// c[2, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_20)

			// c[2, 16-31]
			RELU_SCALE_OP_F32_AVX512(acc_21)

			// c[3, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_30)

			// c[3, 16-31]
			RELU_SCALE_OP_F32_AVX512(acc_31)

			// c[4, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_40)

			// c[4, 16-31]
			RELU_SCALE_OP_F32_AVX512(acc_41)

			// c[5, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_50)

			// c[5, 16-31]
			RELU_SCALE_OP_F32_AVX512(acc_51)

			// c[6, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_60)

			// c[6, 16-31]
			RELU_SCALE_OP_F32_AVX512(acc_61)

			// c[7, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_70)

			// c[7, 16-31]
			RELU_SCALE_OP_F32_AVX512(acc_71)

			// c[8, 0-15]
			RELU_SCALE_OP_F32_AVX512(acc_80)

			// c[8, 16-31]
			RELU_SCALE_OP_F32_AVX512(acc_81)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_TANH_9x32:
		{
			__m512 dn, z, x, r2, r, y;
			__m512i tmpout;

			// c[0, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_00, y, r, r2, x, z, dn, tmpout)

			// c[0, 16-31]
			GELU_TANH_F32_AVX512_DEF(acc_01, y, r, r2, x, z, dn, tmpout)

			// c[1, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_10, y, r, r2, x, z, dn, tmpout)

			// c[1, 16-31]
			GELU_TANH_F32_AVX512_DEF(acc_11, y, r, r2, x, z, dn, tmpout)

			// c[2, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_20, y, r, r2, x, z, dn, tmpout)

			// c[2, 16-31]
			GELU_TANH_F32_AVX512_DEF(acc_21, y, r, r2, x, z, dn, tmpout)

			// c[3, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_30, y, r, r2, x, z, dn, tmpout)

			// c[3, 16-31]
			GELU_TANH_F32_AVX512_DEF(acc_31, y, r, r2, x, z, dn, tmpout)

			// c[4, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_40, y, r, r2, x, z, dn, tmpout)

			// c[4, 16-31]
			GELU_TANH_F32_AVX512_DEF(acc_41, y, r, r2, x, z, dn, tmpout)

			// c[5, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_50, y, r, r2, x, z, dn, tmpout)

			// c[5, 16-31]
			GELU_TANH_F32_AVX512_DEF(acc_51, y, r, r2, x, z, dn, tmpout)

			// c[6, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_60, y, r, r2, x, z, dn, tmpout)

			// c[6, 16-31]
			GELU_TANH_F32_AVX512_DEF(acc_61, y, r, r2, x, z, dn, tmpout)

			// c[7, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_70, y, r, r2, x, z, dn, tmpout)

			// c[7, 16-31]
			GELU_TANH_F32_AVX512_DEF(acc_71, y, r, r2, x, z, dn, tmpout)

			// c[8, 0-15]
			GELU_TANH_F32_AVX512_DEF(acc_80, y, r, r2, x, z, dn, tmpout)

			// c[8, 16-31]
			GELU_TANH_F32_AVX512_DEF(acc_81, y, r, r2, x, z, dn, tmpout)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_ERF_9x32:
		{
			__m512 y, r, r2;

			// c[0, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_00, y, r, r2)

			// c[0, 16-31]
			GELU_ERF_F32_AVX512_DEF(acc_01, y, r, r2)

			// c[1, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_10, y, r, r2)

			// c[1, 16-31]
			GELU_ERF_F32_AVX512_DEF(acc_11, y, r, r2)

			// c[2, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_20, y, r, r2)

			// c[2, 16-31]
			GELU_ERF_F32_AVX512_DEF(acc_21, y, r, r2)

			// c[3, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_30, y, r, r2)

			// c[3, 16-31]
			GELU_ERF_F32_AVX512_DEF(acc_31, y, r, r2)

			// c[4, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_40, y, r, r2)

			// c[4, 16-31]
			GELU_ERF_F32_AVX512_DEF(acc_41, y, r, r2)

			// c[5, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_50, y, r, r2)

			// c[5, 16-31]
			GELU_ERF_F32_AVX512_DEF(acc_51, y, r, r2)

			// c[6, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_60, y, r, r2)

			// c[6, 16-31]
			GELU_ERF_F32_AVX512_DEF(acc_61, y, r, r2)

			// c[7, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_70, y, r, r2)

			// c[7, 16-31]
			GELU_ERF_F32_AVX512_DEF(acc_71, y, r, r2)

			// c[8, 0-15]
			GELU_ERF_F32_AVX512_DEF(acc_80, y, r, r2)

			// c[8, 16-31]
			GELU_ERF_F32_AVX512_DEF(acc_81, y, r, r2)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_CLIP_9x32:
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

			// c[0, 0-15]
			CLIP_F32_AVX512(acc_00, min, max)

			// c[0, 16-31]
			CLIP_F32_AVX512(acc_01, min, max)

			// c[1, 0-15]
			CLIP_F32_AVX512(acc_10, min, max)

			// c[1, 16-31]
			CLIP_F32_AVX512(acc_11, min, max)

			// c[2, 0-15]
			CLIP_F32_AVX512(acc_20, min, max)

			// c[2, 16-31]
			CLIP_F32_AVX512(acc_21, min, max)

			// c[3, 0-15]
			CLIP_F32_AVX512(acc_30, min, max)

			// c[3, 16-31]
			CLIP_F32_AVX512(acc_31, min, max)

			// c[4, 0-15]
			CLIP_F32_AVX512(acc_40, min, max)

			// c[4, 16-31]
			CLIP_F32_AVX512(acc_41, min, max)

			// c[5, 0-15]
			CLIP_F32_AVX512(acc_50, min, max)

			// c[5, 16-31]
			CLIP_F32_AVX512(acc_51, min, max)

			// c[6, 0-15]
			CLIP_F32_AVX512(acc_60, min, max)

			// c[6, 16-31]
			CLIP_F32_AVX512(acc_61, min, max)

			// c[7, 0-15]
			CLIP_F32_AVX512(acc_70, min, max)

			// c[7, 16-31]
			CLIP_F32_AVX512(acc_71, min, max)

			// c[8, 0-15]
			CLIP_F32_AVX512(acc_80, min, max)

			// c[8, 16-31]
			CLIP_F32_AVX512(acc_81, min, max)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_DOWNSCALE_9x32:
	{
		__m512 scale0 = _mm512_setzero_ps();
		__m512 scale1 = _mm512_setzero_ps();
		if ( post_ops_list_temp->scale_factor_len > 1 )
		{
			scale0 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
			scale1 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
		}
		else if ( post_ops_list_temp->scale_factor_len == 1 )
		{
			scale0 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			scale1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
		}

		// Need to ensure sse not used to avoid avx512 -> sse transition.
		__m128i zero_point0 = _mm512_castsi512_si128( _mm512_setzero_si512() );
		__m128i zero_point1 = _mm512_castsi512_si128( _mm512_setzero_si512() );
		if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
		{
			zero_point0 = _mm_loadu_si128( ( __m128i const* )
							( ( int8_t* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) ) );
			zero_point1 = _mm_loadu_si128( ( __m128i const* )
							( ( int8_t* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) ) );
		}
		else if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) == 1 )
		{
			zero_point0 = _mm_maskz_set1_epi8( 0xFFFF,
							*( ( int8_t* )post_ops_list_temp->op_args1 ) );
			zero_point1 = _mm_maskz_set1_epi8( 0xFFFF,
							*( ( int8_t* )post_ops_list_temp->op_args1 ) );
		}

		// c[0, 0-15]
		CVT_MULRND_F32(acc_00,scale0,zero_point0);

		// c[0, 16-31]
		CVT_MULRND_F32(acc_01,scale1,zero_point1);

		// c[1, 0-15]
		CVT_MULRND_F32(acc_10,scale0,zero_point0);

		// c[1, 16-31]
		CVT_MULRND_F32(acc_11,scale1,zero_point1);

		// c[2, 0-15]
		CVT_MULRND_F32(acc_20,scale0,zero_point0);

		// c[2, 16-31]
		CVT_MULRND_F32(acc_21,scale1,zero_point1);

		// c[3, 0-15]
		CVT_MULRND_F32(acc_30,scale0,zero_point0);

		// c[3, 16-31]
		CVT_MULRND_F32(acc_31,scale1,zero_point1);

		// c[4, 0-15]
		CVT_MULRND_F32(acc_40,scale0,zero_point0);

		// c[4, 16-31]
		CVT_MULRND_F32(acc_41,scale1,zero_point1);

		// c[5, 0-15]
		CVT_MULRND_F32(acc_50,scale0,zero_point0);

		// c[5, 16-31]
		CVT_MULRND_F32(acc_51,scale1,zero_point1);

		// c[6, 0-15]
		CVT_MULRND_F32(acc_60,scale0,zero_point0);

		// c[6, 16-31]
		CVT_MULRND_F32(acc_61,scale1,zero_point1);

		// c[7, 0-15]
		CVT_MULRND_F32(acc_70,scale0,zero_point0);

		// c[7, 16-31]
		CVT_MULRND_F32(acc_71,scale1,zero_point1);

		// c[8, 0-15]
		CVT_MULRND_F32(acc_80,scale0,zero_point0);

		// c[8, 16-31]
		CVT_MULRND_F32(acc_81,scale1,zero_point1);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_ADD_9x32:
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
			__m512 scl_fctr5 = _mm512_setzero_ps();
			__m512 scl_fctr6 = _mm512_setzero_ps();
			__m512 scl_fctr7 = _mm512_setzero_ps();
			__m512 scl_fctr8 = _mm512_setzero_ps();
			__m512 scl_fctr9 = _mm512_setzero_ps();
			__m512 t0 = _mm512_setzero_ps();
			__m512 t1 = _mm512_setzero_ps();

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
				scl_fctr6 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr7 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr8 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr9 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					scl_fctr2 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
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
					scl_fctr6 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 5 ) );
					scl_fctr7 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 6 ) );
					scl_fctr8 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 7 ) );
					scl_fctr9 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 8 ) );
				}
			}

			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31]
					BF16_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,0);

					// c[1:0-15,16-31]
					BF16_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,1);

					// c[2:0-15,16-31]
					BF16_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,2);

					// c[3:0-15,16-31]
					BF16_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,3);

					// c[4:0-15,16-31]
					BF16_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,4);

					// c[5:0-15,16-31]
					BF16_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,5);

					// c[6:0-15,16-31]
					BF16_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,6);

					// c[7:0-15,16-31]
					BF16_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,7);

					// c[8:0-15,16-31]
					BF16_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,8);
				}
				else
				{
					// c[0:0-15,16-31]
					BF16_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31]
					BF16_MATRIX_ADD_2COL(t0,t1,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31]
					BF16_MATRIX_ADD_2COL(t0,t1,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31]
					BF16_MATRIX_ADD_2COL(t0,t1,scl_fctr4,scl_fctr4,3);

					// c[4:0-15,16-31]
					BF16_MATRIX_ADD_2COL(t0,t1,scl_fctr5,scl_fctr5,4);

					// c[5:0-15,16-31]
					BF16_MATRIX_ADD_2COL(t0,t1,scl_fctr6,scl_fctr6,5);

					// c[6:0-15,16-31]
					BF16_MATRIX_ADD_2COL(t0,t1,scl_fctr7,scl_fctr7,6);

					// c[7:0-15,16-31]
					BF16_MATRIX_ADD_2COL(t0,t1,scl_fctr8,scl_fctr8,7);

					// c[8:0-15,16-31]
					BF16_MATRIX_ADD_2COL(t0,t1,scl_fctr9,scl_fctr9,8);
				}
			}
			else if ( is_f32 == TRUE )
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31]
					F32_ONLY_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,0);

					// c[1:0-15,16-31]
					F32_ONLY_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,1);

					// c[2:0-15,16-31]
					F32_ONLY_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,2);

					// c[3:0-15,16-31]
					F32_ONLY_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,3);

					// c[4:0-15,16-31]
					F32_ONLY_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,4);

					// c[5:0-15,16-31]
					F32_ONLY_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,5);

					// c[6:0-15,16-31]
					F32_ONLY_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,6);

					// c[7:0-15,16-31]
					F32_ONLY_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,7);

					// c[8:0-15,16-31]
					F32_ONLY_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,8);
				}
				else
				{
					// c[0:0-15,16-31]
					F32_ONLY_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31]
					F32_ONLY_MATRIX_ADD_2COL(t0,t1,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31]
					F32_ONLY_MATRIX_ADD_2COL(t0,t1,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31]
					F32_ONLY_MATRIX_ADD_2COL(t0,t1,scl_fctr4,scl_fctr4,3);

					// c[4:0-15,16-31]
					F32_ONLY_MATRIX_ADD_2COL(t0,t1,scl_fctr5,scl_fctr5,4);

					// c[5:0-15,16-31]
					F32_ONLY_MATRIX_ADD_2COL(t0,t1,scl_fctr6,scl_fctr6,5);

					// c[6:0-15,16-31]
					F32_ONLY_MATRIX_ADD_2COL(t0,t1,scl_fctr7,scl_fctr7,6);

					// c[7:0-15,16-31]
					F32_ONLY_MATRIX_ADD_2COL(t0,t1,scl_fctr8,scl_fctr8,7);

					// c[8:0-15,16-31]
					F32_ONLY_MATRIX_ADD_2COL(t0,t1,scl_fctr9,scl_fctr9,8);
				}
			}
			else if ( is_s8 == TRUE )
			{
				int8_t* matptr = ( int8_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31]
					S8_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,0);

					// c[1:0-15,16-31]
					S8_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,1);

					// c[2:0-15,16-31]
					S8_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,2);

					// c[3:0-15,16-31]
					S8_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,3);

					// c[4:0-15,16-31]
					S8_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,4);

					// c[5:0-15,16-31]
					S8_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,5);

					// c[6:0-15,16-31]
					S8_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,6);

					// c[7:0-15,16-31]
					S8_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,7);

					// c[8:0-15,16-31]
					S8_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,8);
				}
				else
				{
					// c[0:0-15,16-31]
					S8_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31]
					S8_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31]
					S8_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31]
					S8_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr4,scl_fctr4,3);

					// c[4:0-15,16-31]
					S8_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr5,scl_fctr5,4);

					// c[5:0-15,16-31]
					S8_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr6,scl_fctr6,5);

					// c[6:0-15,16-31]
					S8_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr7,scl_fctr7,6);

					// c[7:0-15,16-31]
					S8_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr8,scl_fctr8,7);

					// c[8:0-15,16-31]
					S8_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr9,scl_fctr9,8);
				}
			}
			else
			{
				int32_t* matptr = ( int32_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31]
					S32_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,0);

					// c[1:0-15,16-31]
					S32_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,1);

					// c[2:0-15,16-31]
					S32_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,2);

					// c[3:0-15,16-31]
					S32_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,3);

					// c[4:0-15,16-31]
					S32_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,4);

					// c[5:0-15,16-31]
					S32_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,5);

					// c[6:0-15,16-31]
					S32_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,6);

					// c[7:0-15,16-31]
					S32_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,7);

					// c[8:0-15,16-31]
					S32_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr2,8);
				}
				else
				{
					// c[0:0-15,16-31]
					S32_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31]
					S32_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31]
					S32_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31]
					S32_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr4,scl_fctr4,3);

					// c[4:0-15,16-31]
					S32_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr5,scl_fctr5,4);

					// c[5:0-15,16-31]
					S32_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr6,scl_fctr6,5);

					// c[6:0-15,16-31]
					S32_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr7,scl_fctr7,6);

					// c[7:0-15,16-31]
					S32_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr8,scl_fctr8,7);

					// c[8:0-15,16-31]
					S32_F32_MATRIX_ADD_2COL(t0,t1,scl_fctr9,scl_fctr9,8);
				}
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_MUL_9x32:
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
			__m512 scl_fctr5 = _mm512_setzero_ps();
			__m512 scl_fctr6 = _mm512_setzero_ps();
			__m512 scl_fctr7 = _mm512_setzero_ps();
			__m512 scl_fctr8 = _mm512_setzero_ps();
			__m512 scl_fctr9 = _mm512_setzero_ps();
			__m512 t0 = _mm512_setzero_ps();
			__m512 t1 = _mm512_setzero_ps();

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
				scl_fctr6 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr7 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr8 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
				scl_fctr9 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			}
			else
			{
				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					scl_fctr1 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
					scl_fctr2 =
						_mm512_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
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
					scl_fctr6 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 5 ) );
					scl_fctr7 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 6 ) );
					scl_fctr8 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 7 ) );
					scl_fctr9 =
						_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_i + 8 ) );
				}
			}

			if ( is_bf16 == TRUE )
			{
				bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31]
					BF16_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,0);

					// c[1:0-15,16-31]
					BF16_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,1);

					// c[2:0-15,16-31]
					BF16_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,2);

					// c[3:0-15,16-31]
					BF16_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,3);

					// c[4:0-15,16-31]
					BF16_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,4);

					// c[5:0-15,16-31]
					BF16_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,5);

					// c[6:0-15,16-31]
					BF16_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,6);

					// c[7:0-15,16-31]
					BF16_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,7);

					// c[8:0-15,16-31]
					BF16_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,8);
				}
				else
				{
					// c[0:0-15,16-31]
					BF16_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31]
					BF16_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31]
					BF16_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31]
					BF16_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr4,scl_fctr4,3);

					// c[4:0-15,16-31]
					BF16_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr5,scl_fctr5,4);

					// c[5:0-15,16-31]
					BF16_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr6,scl_fctr6,5);

					// c[6:0-15,16-31]
					BF16_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr7,scl_fctr7,6);

					// c[7:0-15,16-31]
					BF16_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr8,scl_fctr8,7);

					// c[8:0-15,16-31]
					BF16_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr9,scl_fctr9,8);
				}
			}
			else if ( is_f32 == TRUE )
			{
				float* matptr = ( float* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31]
					F32_U8S8_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,0);

					// c[1:0-15,16-31]
					F32_U8S8_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,1);

					// c[2:0-15,16-31]
					F32_U8S8_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,2);

					// c[3:0-15,16-31]
					F32_U8S8_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,3);

					// c[4:0-15,16-31]
					F32_U8S8_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,4);

					// c[5:0-15,16-31]
					F32_U8S8_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,5);

					// c[6:0-15,16-31]
					F32_U8S8_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,6);

					// c[7:0-15,16-31]
					F32_U8S8_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,7);

					// c[8:0-15,16-31]
					F32_U8S8_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,8);
				}
				else
				{
					// c[0:0-15,16-31]
					F32_U8S8_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31]
					F32_U8S8_MATRIX_MUL_2COL(t0,t1,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31]
					F32_U8S8_MATRIX_MUL_2COL(t0,t1,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31]
					F32_U8S8_MATRIX_MUL_2COL(t0,t1,scl_fctr4,scl_fctr4,3);

					// c[4:0-15,16-31]
					F32_U8S8_MATRIX_MUL_2COL(t0,t1,scl_fctr5,scl_fctr5,4);

					// c[5:0-15,16-31]
					F32_U8S8_MATRIX_MUL_2COL(t0,t1,scl_fctr6,scl_fctr6,5);

					// c[6:0-15,16-31]
					F32_U8S8_MATRIX_MUL_2COL(t0,t1,scl_fctr7,scl_fctr7,6);

					// c[7:0-15,16-31]
					F32_U8S8_MATRIX_MUL_2COL(t0,t1,scl_fctr8,scl_fctr8,7);

					// c[8:0-15,16-31]
					F32_U8S8_MATRIX_MUL_2COL(t0,t1,scl_fctr9,scl_fctr9,8);
				}
			}
			else if ( is_s8 == TRUE )
			{
				int8_t* matptr = ( int8_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31]
					S8_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,0);

					// c[1:0-15,16-31]
					S8_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,1);

					// c[2:0-15,16-31]
					S8_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,2);

					// c[3:0-15,16-31]
					S8_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,3);

					// c[4:0-15,16-31]
					S8_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,4);

					// c[5:0-15,16-31]
					S8_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,5);

					// c[6:0-15,16-31]
					S8_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,6);

					// c[7:0-15,16-31]
					S8_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,7);

					// c[8:0-15,16-31]
					S8_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,8);
				}
				else
				{
					// c[0:0-15,16-31]
					S8_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31]
					S8_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31]
					S8_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31]
					S8_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr4,scl_fctr4,3);

					// c[4:0-15,16-31]
					S8_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr5,scl_fctr5,4);

					// c[5:0-15,16-31]
					S8_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr6,scl_fctr6,5);

					// c[6:0-15,16-31]
					S8_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr7,scl_fctr7,6);

					// c[7:0-15,16-31]
					S8_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr8,scl_fctr8,7);

					// c[8:0-15,16-31]
					S8_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr9,scl_fctr9,8);
				}
			}
			else
			{
				int32_t* matptr = ( int32_t* )post_ops_list_temp->op_args1;

				if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
					 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
				{
					// c[0:0-15,16-31]
					S32_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,0);

					// c[1:0-15,16-31]
					S32_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,1);

					// c[2:0-15,16-31]
					S32_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,2);

					// c[3:0-15,16-31]
					S32_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,3);

					// c[4:0-15,16-31]
					S32_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,4);

					// c[5:0-15,16-31]
					S32_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,5);

					// c[6:0-15,16-31]
					S32_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,6);

					// c[7:0-15,16-31]
					S32_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,7);

					// c[8:0-15,16-31]
					S32_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr2,8);
				}
				else
				{
					// c[0:0-15,16-31]
					S32_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr1,scl_fctr1,0);

					// c[1:0-15,16-31]
					S32_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr2,scl_fctr2,1);

					// c[2:0-15,16-31]
					S32_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr3,scl_fctr3,2);

					// c[3:0-15,16-31]
					S32_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr4,scl_fctr4,3);

					// c[4:0-15,16-31]
					S32_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr5,scl_fctr5,4);

					// c[5:0-15,16-31]
					S32_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr6,scl_fctr6,5);

					// c[6:0-15,16-31]
					S32_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr7,scl_fctr7,6);

					// c[7:0-15,16-31]
					S32_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr8,scl_fctr8,7);

					// c[8:0-15,16-31]
					S32_F32_MATRIX_MUL_2COL(t0,t1,scl_fctr9,scl_fctr9,8);
				}
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SWISH_9x32:
		{
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

			__m512 al_in, r, r2, z, dn;
			__m512i temp;

			// c[0, 0-15]
			SWISH_F32_AVX512_DEF(acc_00, scale, al_in, r, r2, z, dn, temp);

			// c[0, 16-31]
			SWISH_F32_AVX512_DEF(acc_01, scale, al_in, r, r2, z, dn, temp);

			// c[1, 0-15]
			SWISH_F32_AVX512_DEF(acc_10, scale, al_in, r, r2, z, dn, temp);

			// c[1, 16-31]
			SWISH_F32_AVX512_DEF(acc_11, scale, al_in, r, r2, z, dn, temp);

			// c[2, 0-15]
			SWISH_F32_AVX512_DEF(acc_20, scale, al_in, r, r2, z, dn, temp);

			// c[2, 16-31]
			SWISH_F32_AVX512_DEF(acc_21, scale, al_in, r, r2, z, dn, temp);

			// c[3, 0-15]
			SWISH_F32_AVX512_DEF(acc_30, scale, al_in, r, r2, z, dn, temp);

			// c[3, 16-31]
			SWISH_F32_AVX512_DEF(acc_31, scale, al_in, r, r2, z, dn, temp);

			// c[4, 0-15]
			SWISH_F32_AVX512_DEF(acc_40, scale, al_in, r, r2, z, dn, temp);

			// c[4, 16-31]
			SWISH_F32_AVX512_DEF(acc_41, scale, al_in, r, r2, z, dn, temp);

			// c[5, 0-15]
			SWISH_F32_AVX512_DEF(acc_50, scale, al_in, r, r2, z, dn, temp);

			// c[5, 16-31]
			SWISH_F32_AVX512_DEF(acc_51, scale, al_in, r, r2, z, dn, temp);

			// c[6, 0-15]
			SWISH_F32_AVX512_DEF(acc_60, scale, al_in, r, r2, z, dn, temp);

			// c[6, 16-31]
			SWISH_F32_AVX512_DEF(acc_61, scale, al_in, r, r2, z, dn, temp);

			// c[7, 0-15]
			SWISH_F32_AVX512_DEF(acc_70, scale, al_in, r, r2, z, dn, temp);

			// c[7, 16-31]
			SWISH_F32_AVX512_DEF(acc_71, scale, al_in, r, r2, z, dn, temp);

			// c[8, 0-15]
			SWISH_F32_AVX512_DEF(acc_80, scale, al_in, r, r2, z, dn, temp);

			// c[8, 16-31]
			SWISH_F32_AVX512_DEF(acc_81, scale, al_in, r, r2, z, dn, temp);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_TANH_9x32:
		{
			__m512 dn, z, x, r2, r;
			__m512i q;

			// c[0, 0-15]
			TANHF_AVX512(acc_00, r, r2, x, z, dn, q)

			// c[0, 16-31]
			TANHF_AVX512(acc_01, r, r2, x, z, dn, q)

			// c[1, 0-15]
			TANHF_AVX512(acc_10, r, r2, x, z, dn, q)

			// c[1, 16-31]
			TANHF_AVX512(acc_11, r, r2, x, z, dn, q)

			// c[2, 0-15]
			TANHF_AVX512(acc_20, r, r2, x, z, dn, q)

			// c[2, 16-31]
			TANHF_AVX512(acc_21, r, r2, x, z, dn, q)

			// c[3, 0-15]
			TANHF_AVX512(acc_30, r, r2, x, z, dn, q)

			// c[3, 16-31]
			TANHF_AVX512(acc_31, r, r2, x, z, dn, q)

			// c[4, 0-15]
			TANHF_AVX512(acc_40, r, r2, x, z, dn, q)

			// c[4, 16-31]
			TANHF_AVX512(acc_41, r, r2, x, z, dn, q)

			// c[5, 0-15]
			TANHF_AVX512(acc_50, r, r2, x, z, dn, q)

			// c[5, 16-31]
			TANHF_AVX512(acc_51, r, r2, x, z, dn, q)

			// c[6, 0-15]
			TANHF_AVX512(acc_60, r, r2, x, z, dn, q)

			// c[6, 16-31]
			TANHF_AVX512(acc_61, r, r2, x, z, dn, q)

			// c[7, 0-15]
			TANHF_AVX512(acc_70, r, r2, x, z, dn, q)

			// c[7, 16-31]
			TANHF_AVX512(acc_71, r, r2, x, z, dn, q)

			// c[8, 0-15]
			TANHF_AVX512(acc_80, r, r2, x, z, dn, q)

			// c[8, 16-31]
			TANHF_AVX512(acc_81, r, r2, x, z, dn, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SIGMOID_9x32:
		{
			__m512 al_in, r, r2, z, dn;
			__m512i tmpout;

			// c[0, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_00, al_in, r, r2, z, dn, tmpout);

			// c[0, 16-31]
			SIGMOID_F32_AVX512_DEF(acc_01, al_in, r, r2, z, dn, tmpout);

			// c[1, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_10, al_in, r, r2, z, dn, tmpout);

			// c[1, 16-31]
			SIGMOID_F32_AVX512_DEF(acc_11, al_in, r, r2, z, dn, tmpout);

			// c[2, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_20, al_in, r, r2, z, dn, tmpout);

			// c[2, 16-31]
			SIGMOID_F32_AVX512_DEF(acc_21, al_in, r, r2, z, dn, tmpout);

			// c[3, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_30, al_in, r, r2, z, dn, tmpout);

			// c[3, 16-31]
			SIGMOID_F32_AVX512_DEF(acc_31, al_in, r, r2, z, dn, tmpout);

			// c[4, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_40, al_in, r, r2, z, dn, tmpout);

			// c[4, 16-31]
			SIGMOID_F32_AVX512_DEF(acc_41, al_in, r, r2, z, dn, tmpout);

			// c[5, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_50, al_in, r, r2, z, dn, tmpout);

			// c[5, 16-31]
			SIGMOID_F32_AVX512_DEF(acc_51, al_in, r, r2, z, dn, tmpout);

			// c[6, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_60, al_in, r, r2, z, dn, tmpout);

			// c[6, 16-31]
			SIGMOID_F32_AVX512_DEF(acc_61, al_in, r, r2, z, dn, tmpout);

			// c[7, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_70, al_in, r, r2, z, dn, tmpout);

			// c[7, 16-31]
			SIGMOID_F32_AVX512_DEF(acc_71, al_in, r, r2, z, dn, tmpout);

			// c[8, 0-15]
			SIGMOID_F32_AVX512_DEF(acc_80, al_in, r, r2, z, dn, tmpout);

			// c[8, 16-31]
			SIGMOID_F32_AVX512_DEF(acc_81, al_in, r, r2, z, dn, tmpout);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_9x32_DISABLE:
		;

		if ( ( post_ops_attr.buf_downscale != NULL ) && ( post_ops_attr.is_last_k == TRUE ) )
		{
			// Generate a mask16 of all 1's.
			selector1 = _mm512_setzero_epi32();
			selector2 = _mm512_set1_epi32( 10 );
			__mmask16 mask_all1 = _mm512_cmplt_epi32_mask( selector1, selector2 );
			if ( post_ops_attr.c_stor_type == S8)
			{
				// Store the results in downscaled type (int8 instead of int32).
				// c[0,0-15]
				CVT_STORE_F32_S8(acc_00,0,0);

				// c[0,16-31]
				CVT_STORE_F32_S8(acc_01,0,1);

				// c[1,0-15]
				CVT_STORE_F32_S8(acc_10,1,0);

				// c[1,16-31]
				CVT_STORE_F32_S8(acc_11,1,1);

				// c[2,0-15]
				CVT_STORE_F32_S8(acc_20,2,0);

				// c[2,16-31]
				CVT_STORE_F32_S8(acc_21,2,1);

				// c[3,0-15]
				CVT_STORE_F32_S8(acc_30,3,0);

				// c[3,16-31]
				CVT_STORE_F32_S8(acc_31,3,1);

				// c[4,0-15]
				CVT_STORE_F32_S8(acc_40,4,0);

				// c[4,16-31]
				CVT_STORE_F32_S8(acc_41,4,1);

				// c[5,0-15]
				CVT_STORE_F32_S8(acc_50,5,0);

				// c[5,16-31]
				CVT_STORE_F32_S8(acc_51,5,1);

				// c[6,0-15]
				CVT_STORE_F32_S8(acc_60,6,0);

				// c[6,16-31]
				CVT_STORE_F32_S8(acc_61,6,1);

				// c[7,0-15]
				CVT_STORE_F32_S8(acc_70,7,0);

				// c[7,16-31]
				CVT_STORE_F32_S8(acc_71,7,1);

				// c[8,0-15]
				CVT_STORE_F32_S8(acc_80,8,0);

				// c[8,16-31]
				CVT_STORE_F32_S8(acc_81,8,1);
			}
			else if ( post_ops_attr.c_stor_type == U8 )
			{
				// Store the results in downscaled type (uint8 instead of int32).
				// c[0,0-15]
				CVT_STORE_F32_U8(acc_00,0,0);

				// c[0,16-31]
				CVT_STORE_F32_U8(acc_01,0,1);

				// c[1,0-15]
				CVT_STORE_F32_U8(acc_10,1,0);

				// c[1,16-31]
				CVT_STORE_F32_U8(acc_11,1,1);

				// c[2,0-15]
				CVT_STORE_F32_U8(acc_20,2,0);

				// c[2,16-31]
				CVT_STORE_F32_U8(acc_21,2,1);

				// c[3,0-15]
				CVT_STORE_F32_U8(acc_30,3,0);

				// c[3,16-31]
				CVT_STORE_F32_U8(acc_31,3,1);

				// c[4,0-15]
				CVT_STORE_F32_U8(acc_40,4,0);

				// c[4,16-31]
				CVT_STORE_F32_U8(acc_41,4,1);

				// c[5,0-15]
				CVT_STORE_F32_U8(acc_50,5,0);

				// c[5,16-31]
				CVT_STORE_F32_U8(acc_51,5,1);

				// c[6,0-15]
				CVT_STORE_F32_U8(acc_60,6,0);

				// c[6,16-31]
				CVT_STORE_F32_U8(acc_61,6,1);

				// c[7,0-15]
				CVT_STORE_F32_U8(acc_70,7,0);

				// c[7,16-31]
				CVT_STORE_F32_U8(acc_71,7,1);

				// c[8,0-15]
				CVT_STORE_F32_U8(acc_80,8,0);

				// c[8,16-31]
				CVT_STORE_F32_U8(acc_81,8,1);
			}
			else if ( post_ops_attr.c_stor_type == BF16)
			{
					// c[0,0-15]
				CVT_STORE_F32_BF16(acc_00,0,0);

				// c[0,16-31]
				CVT_STORE_F32_BF16(acc_01,0,1);

				// c[1,0-15]
				CVT_STORE_F32_BF16(acc_10,1,0);

				// c[1,16-31]
				CVT_STORE_F32_BF16(acc_11,1,1);

				// c[2,0-15]
				CVT_STORE_F32_BF16(acc_20,2,0);

				// c[2,16-31]
				CVT_STORE_F32_BF16(acc_21,2,1);

				// c[3,0-15]
				CVT_STORE_F32_BF16(acc_30,3,0);

				// c[3,16-31]
				CVT_STORE_F32_BF16(acc_31,3,1);

				// c[4,0-15]
				CVT_STORE_F32_BF16(acc_40,4,0);

				// c[4,16-31]
				CVT_STORE_F32_BF16(acc_41,4,1);

				// c[5,0-15]
				CVT_STORE_F32_BF16(acc_50,5,0);

				// c[5,16-31]
				CVT_STORE_F32_BF16(acc_51,5,1);

				// c[6,0-15]
				CVT_STORE_F32_BF16(acc_60,6,0);

				// c[6,16-31]
				CVT_STORE_F32_BF16(acc_61,6,1);

				// c[7,0-15]
				CVT_STORE_F32_BF16(acc_70,7,0);

				// c[7,16-31]
				CVT_STORE_F32_BF16(acc_71,7,1);

				// c[8,0-15]
				CVT_STORE_F32_BF16(acc_80,8,0);

				// c[8,16-31]
				CVT_STORE_F32_BF16(acc_81,8,1);
			}
			else if ( post_ops_attr.c_stor_type == F32)
			{
					// c[0,0-15]
				STORE_F32(acc_00,0,0);

				// c[0,16-31]
				STORE_F32(acc_01,0,1);

				// c[1,0-15]
				STORE_F32(acc_10,1,0);

				// c[1,16-31]
				STORE_F32(acc_11,1,1);

				// c[2,0-15]
				STORE_F32(acc_20,2,0);

				// c[2,16-31]
				STORE_F32(acc_21,2,1);

				// c[3,0-15]
				STORE_F32(acc_30,3,0);

				// c[3,16-31]
				STORE_F32(acc_31,3,1);

				// c[4,0-15]
				STORE_F32(acc_40,4,0);

				// c[4,16-31]
				STORE_F32(acc_41,4,1);

				// c[5,0-15]
				STORE_F32(acc_50,5,0);

				// c[5,16-31]
				STORE_F32(acc_51,5,1);

				// c[6,0-15]
				STORE_F32(acc_60,6,0);

				// c[6,16-31]
				STORE_F32(acc_61,6,1);

				// c[7,0-15]
				STORE_F32(acc_70,7,0);

				// c[7,16-31]
				STORE_F32(acc_71,7,1);

				// c[8,0-15]
				STORE_F32(acc_80,8,0);

				// c[8,16-31]
				STORE_F32(acc_81,8,1);
			}
		}
		else
		{
			// Store the results.
			// c[0,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 0 ) ) + ( 0*16 ),
				_mm512_cvtps_epi32( acc_00 ) );

			// c[0, 16-31]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 0 ) ) + ( 1*16 ),
				_mm512_cvtps_epi32( acc_01 ) );

			// c[1,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 1 ) ) + ( 0*16 ),
				_mm512_cvtps_epi32( acc_10 ) );

			// c[1,16-31]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 1 ) ) + ( 1*16 ),
				_mm512_cvtps_epi32( acc_11 ) );

			// c[2,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 2 ) ) + ( 0*16 ),
				_mm512_cvtps_epi32( acc_20 ) );

			// c[2,16-31]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 2 ) ) + ( 1*16 ),
				_mm512_cvtps_epi32( acc_21 ) );

			// c[3,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 3 ) ) + ( 0*16 ),
				_mm512_cvtps_epi32( acc_30 ) );

			// c[3,16-31]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 3 ) ) + ( 1*16 ),
				_mm512_cvtps_epi32( acc_31 ) );

			// c[4,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 4 ) ) + ( 0*16 ),
				_mm512_cvtps_epi32( acc_40 ) );

			// c[4,16-31]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 4 ) ) + ( 1*16 ),
				_mm512_cvtps_epi32( acc_41 ) );

			// c[5,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 5 ) ) + ( 0*16 ),
				_mm512_cvtps_epi32( acc_50 ) );

			// c[5,16-31]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 5 ) ) + ( 1*16 ),
				_mm512_cvtps_epi32( acc_51 ) );

			// c[6,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 6 ) ) + ( 0*16 ),
				_mm512_cvtps_epi32( acc_60 ) );

			// c[6, 16-31]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 6 ) ) + ( 1*16 ),
				_mm512_cvtps_epi32( acc_61 ) );

			// c[7,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 7 ) ) + ( 0*16 ),
				_mm512_cvtps_epi32( acc_70 ) );

			// c[7,16-31]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 7 ) ) + ( 1*16 ),
				_mm512_cvtps_epi32( acc_71 ) );

			// c[8,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 8 ) ) + ( 0*16 ),
				_mm512_cvtps_epi32( acc_80 ) );

			// c[8,16-31]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 8 ) ) + ( 1*16 ),
				_mm512_cvtps_epi32( acc_81 ) );
		}

		a = a + ( MR * ps_a );
		post_ops_attr.post_op_c_i += MR;
	}

	if ( m_partial_pieces > 0 )
	{
		lpgemm_rowvar_u8s8s32o32_6x32
		(
		  m_partial_pieces, k0,
		  a, rs_a, cs_a, ps_a,
		  b, rs_b, cs_b,
		  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
		  alpha, beta,
		  post_ops_list, post_ops_attr
		);
	}
}

#endif
