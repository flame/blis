/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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
				  &&POST_OPS_SWISH_12xLT16
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

        // Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_12xLT16:
		{
			__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );
			selector1 = _mm512_maskz_loadu_epi32
			(
			  load_mask,
			  ( ( int32_t* )post_ops_list_temp->op_args1 +
				post_ops_attr.post_op_c_j )
			);

			// c[0,0-15]
			c_int32_0p0 = _mm512_add_epi32( selector1, c_int32_0p0 );

			// c[1,0-15]
			c_int32_1p0 = _mm512_add_epi32( selector1, c_int32_1p0 );

			// c[2,0-15]
			c_int32_2p0 = _mm512_add_epi32( selector1, c_int32_2p0 );

			// c[3,0-15]
			c_int32_3p0 = _mm512_add_epi32( selector1, c_int32_3p0 );

			// c[4,0-15]
			c_int32_4p0 = _mm512_add_epi32( selector1, c_int32_4p0 );

			// c[5,0-15]
			c_int32_5p0 = _mm512_add_epi32( selector1, c_int32_5p0 );

			// c[6,0-15]
			c_int32_6p0 = _mm512_add_epi32( selector1, c_int32_6p0 );

			// c[7,0-15]
			c_int32_7p0 = _mm512_add_epi32( selector1, c_int32_7p0 );

			// c[8,0-15]
			c_int32_8p0 = _mm512_add_epi32( selector1, c_int32_8p0 );

			// c[9,0-15]
			c_int32_9p0 = _mm512_add_epi32( selector1, c_int32_9p0 );

			// c[10,0-15]
			c_int32_10p0 = _mm512_add_epi32( selector1, c_int32_10p0 );

			// c[11,0-15]
			c_int32_11p0 = _mm512_add_epi32( selector1, c_int32_11p0 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_12xLT16:
		{
			selector1 = _mm512_setzero_epi32();

			// c[0,0-15]
			c_int32_0p0 = _mm512_max_epi32( selector1, c_int32_0p0 );

			// c[1,0-15]
			c_int32_1p0 = _mm512_max_epi32( selector1, c_int32_1p0 );

			// c[2,0-15]
			c_int32_2p0 = _mm512_max_epi32( selector1, c_int32_2p0 );

			// c[3,0-15]
			c_int32_3p0 = _mm512_max_epi32( selector1, c_int32_3p0 );

			// c[4,0-15]
			c_int32_4p0 = _mm512_max_epi32( selector1, c_int32_4p0 );

			// c[5,0-15]
			c_int32_5p0 = _mm512_max_epi32( selector1, c_int32_5p0 );

			// c[6,0-15]
			c_int32_6p0 = _mm512_max_epi32( selector1, c_int32_6p0 );

			// c[7,0-15]
			c_int32_7p0 = _mm512_max_epi32( selector1, c_int32_7p0 );

			// c[8,0-15]
			c_int32_8p0 = _mm512_max_epi32( selector1, c_int32_8p0 );

			// c[9,0-15]
			c_int32_9p0 = _mm512_max_epi32( selector1, c_int32_9p0 );

			// c[10,0-15]
			c_int32_10p0 = _mm512_max_epi32( selector1, c_int32_10p0 );

			// c[11,0-15]
			c_int32_11p0 = _mm512_max_epi32( selector1, c_int32_11p0 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_12xLT16:
		{
			selector1 = _mm512_setzero_epi32();
			selector2 =
				_mm512_set1_epi32( *( ( int32_t* )post_ops_list_temp->op_args2 ) );

			__mmask16 relu_cmp_mask;

			// c[0, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_0p0)

			// c[1, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_1p0)

			// c[2, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_2p0)

			// c[3, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_3p0)

			// c[4, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_4p0)

			// c[5, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_5p0)

			// c[6, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_6p0)

			// c[7, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_7p0)

			// c[8, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_8p0)

			// c[9, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_9p0)

			// c[10, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_10p0)

			// c[11, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_11p0)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_TANH_12xLT16:
		{
			__m512 dn, z, x, r2, r, y, x_tanh;
			__m512i q;

			// c[0, 0-15]
			GELU_TANH_S32_AVX512(c_int32_0p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[1, 0-15]
			GELU_TANH_S32_AVX512(c_int32_1p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[2, 0-15]
			GELU_TANH_S32_AVX512(c_int32_2p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[3, 0-15]
			GELU_TANH_S32_AVX512(c_int32_3p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[4, 0-15]
			GELU_TANH_S32_AVX512(c_int32_4p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[5, 0-15]
			GELU_TANH_S32_AVX512(c_int32_5p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[6, 0-15]
			GELU_TANH_S32_AVX512(c_int32_6p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[7, 0-15]
			GELU_TANH_S32_AVX512(c_int32_7p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[8, 0-15]
			GELU_TANH_S32_AVX512(c_int32_8p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[9, 0-15]
			GELU_TANH_S32_AVX512(c_int32_9p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[10, 0-15]
			GELU_TANH_S32_AVX512(c_int32_10p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[11, 0-15]
			GELU_TANH_S32_AVX512(c_int32_11p0, y, r, r2, x, z, dn, x_tanh, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_ERF_12xLT16:
		{
			__m512 x, r, y, x_erf;

			// c[0, 0-15]
			GELU_ERF_S32_AVX512(c_int32_0p0, y, r, x, x_erf)

			// c[1, 0-15]
			GELU_ERF_S32_AVX512(c_int32_1p0, y, r, x, x_erf)

			// c[2, 0-15]
			GELU_ERF_S32_AVX512(c_int32_2p0, y, r, x, x_erf)

			// c[3, 0-15]
			GELU_ERF_S32_AVX512(c_int32_3p0, y, r, x, x_erf)

			// c[4, 0-15]
			GELU_ERF_S32_AVX512(c_int32_4p0, y, r, x, x_erf)

			// c[5, 0-15]
			GELU_ERF_S32_AVX512(c_int32_5p0, y, r, x, x_erf)

			// c[6, 0-15]
			GELU_ERF_S32_AVX512(c_int32_6p0, y, r, x, x_erf)

			// c[7, 0-15]
			GELU_ERF_S32_AVX512(c_int32_7p0, y, r, x, x_erf)

			// c[8, 0-15]
			GELU_ERF_S32_AVX512(c_int32_8p0, y, r, x, x_erf)

			// c[9, 0-15]
			GELU_ERF_S32_AVX512(c_int32_9p0, y, r, x, x_erf)

			// c[10, 0-15]
			GELU_ERF_S32_AVX512(c_int32_10p0, y, r, x, x_erf)

			// c[11, 0-15]
			GELU_ERF_S32_AVX512(c_int32_11p0, y, r, x, x_erf)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_CLIP_12xLT16:
		{
			__m512i min = _mm512_set1_epi32( *( int32_t* )post_ops_list_temp->op_args2 );
			__m512i max = _mm512_set1_epi32( *( int32_t* )post_ops_list_temp->op_args3 );

			// c[0, 0-15]
			CLIP_S32_AVX512(c_int32_0p0, min, max)

			// c[1, 0-15]
			CLIP_S32_AVX512(c_int32_1p0, min, max)

			// c[2, 0-15]
			CLIP_S32_AVX512(c_int32_2p0, min, max)

			// c[3, 0-15]
			CLIP_S32_AVX512(c_int32_3p0, min, max)

			// c[4, 0-15]
			CLIP_S32_AVX512(c_int32_4p0, min, max)

			// c[5, 0-15]
			CLIP_S32_AVX512(c_int32_5p0, min, max)

			// c[6, 0-15]
			CLIP_S32_AVX512(c_int32_6p0, min, max)

			// c[7, 0-15]
			CLIP_S32_AVX512(c_int32_7p0, min, max)

			// c[8, 0-15]
			CLIP_S32_AVX512(c_int32_8p0, min, max)

			// c[9, 0-15]
			CLIP_S32_AVX512(c_int32_9p0, min, max)

			// c[10, 0-15]
			CLIP_S32_AVX512(c_int32_10p0, min, max)

			// c[11, 0-15]
			CLIP_S32_AVX512(c_int32_11p0, min, max)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_DOWNSCALE_12xLT16:
		{
			// Typecast without data modification, safe operation.
			__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );
			if ( post_ops_list_temp->scale_factor_len > 1 )
			{
				selector1 = _mm512_maskz_loadu_epi32
							(
							  load_mask,
							  ( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j )
							);
			}
			else if ( post_ops_list_temp->scale_factor_len == 1 )
			{
				selector1 =
					( __m512i )_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
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
			CVT_MULRND_CVT32_LT16(c_int32_0p0,selector1,zero_point);

			// c[1, 0-15]
			CVT_MULRND_CVT32_LT16(c_int32_1p0,selector1,zero_point);

			// c[2, 0-15]
			CVT_MULRND_CVT32_LT16(c_int32_2p0,selector1,zero_point);

			// c[3, 0-15]
			CVT_MULRND_CVT32_LT16(c_int32_3p0,selector1,zero_point);

			// c[4, 0-15]
			CVT_MULRND_CVT32_LT16(c_int32_4p0,selector1,zero_point);

			// c[5, 0-15]
			CVT_MULRND_CVT32_LT16(c_int32_5p0,selector1,zero_point);

			// c[6, 0-15]
			CVT_MULRND_CVT32_LT16(c_int32_6p0,selector1,zero_point);

			// c[7, 0-15]
			CVT_MULRND_CVT32_LT16(c_int32_7p0,selector1,zero_point);

			// c[8, 0-15]
			CVT_MULRND_CVT32_LT16(c_int32_8p0,selector1,zero_point);

			// c[9, 0-15]
			CVT_MULRND_CVT32_LT16(c_int32_9p0,selector1,zero_point);

			// c[10, 0-15]
			CVT_MULRND_CVT32_LT16(c_int32_10p0,selector1,zero_point);

			// c[11, 0-15]
			CVT_MULRND_CVT32_LT16(c_int32_11p0,selector1,zero_point);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_MATRIX_ADD_12xLT16:
		{
			__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
			if ( post_ops_attr.c_stor_type == S8 )
			{
				int8_t* matptr = ( int8_t* )post_ops_list_temp->op_args1;

				// c[0:0-15]
				S8_S32_MATRIX_ADD_1COL_PAR(load_mask,selector1,0);

				// c[1:0-15]
				S8_S32_MATRIX_ADD_1COL_PAR(load_mask,selector1,1);

				// c[2:0-15]
				S8_S32_MATRIX_ADD_1COL_PAR(load_mask,selector1,2);

				// c[3:0-15]
				S8_S32_MATRIX_ADD_1COL_PAR(load_mask,selector1,3);

				// c[4:0-15]
				S8_S32_MATRIX_ADD_1COL_PAR(load_mask,selector1,4);

				// c[5:0-15]
				S8_S32_MATRIX_ADD_1COL_PAR(load_mask,selector1,5);

				// c[6:0-15]
				S8_S32_MATRIX_ADD_1COL_PAR(load_mask,selector1,6);

				// c[7:0-15]
				S8_S32_MATRIX_ADD_1COL_PAR(load_mask,selector1,7);

				// c[8:0-15]
				S8_S32_MATRIX_ADD_1COL_PAR(load_mask,selector1,8);

				// c[9:0-15]
				S8_S32_MATRIX_ADD_1COL_PAR(load_mask,selector1,9);

				// c[10:0-15]
				S8_S32_MATRIX_ADD_1COL_PAR(load_mask,selector1,10);

				// c[11:0-15]
				S8_S32_MATRIX_ADD_1COL_PAR(load_mask,selector1,11);
			}
			else
			{
				int32_t* matptr = ( int32_t* )post_ops_list_temp->op_args1;

				// c[0:0-15]
				S32_S32_MATRIX_ADD_1COL_PAR(load_mask,selector1,0);

				// c[1:0-15]
				S32_S32_MATRIX_ADD_1COL_PAR(load_mask,selector1,1);

				// c[2:0-15]
				S32_S32_MATRIX_ADD_1COL_PAR(load_mask,selector1,2);

				// c[3:0-15]
				S32_S32_MATRIX_ADD_1COL_PAR(load_mask,selector1,3);

				// c[4:0-15]
				S32_S32_MATRIX_ADD_1COL_PAR(load_mask,selector1,4);

				// c[5:0-15]
				S32_S32_MATRIX_ADD_1COL_PAR(load_mask,selector1,5);

				// c[6:0-15]
				S32_S32_MATRIX_ADD_1COL_PAR(load_mask,selector1,6);

				// c[7:0-15]
				S32_S32_MATRIX_ADD_1COL_PAR(load_mask,selector1,7);

				// c[8:0-15]
				S32_S32_MATRIX_ADD_1COL_PAR(load_mask,selector1,8);

				// c[9:0-15]
				S32_S32_MATRIX_ADD_1COL_PAR(load_mask,selector1,9);

				// c[10:0-15]
				S32_S32_MATRIX_ADD_1COL_PAR(load_mask,selector1,10);

				// c[11:0-15]
				S32_S32_MATRIX_ADD_1COL_PAR(load_mask,selector1,11);
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SWISH_12xLT16:
		{
			selector1 =
				_mm512_set1_epi32( *( ( int32_t* )post_ops_list_temp->op_args2 ) );
			__m512 al = _mm512_cvtepi32_ps( selector1 );

			__m512 fl_reg, al_in, r, r2, z, dn;

			// c[0, 0-15]
			SWISH_S32_AVX512(c_int32_0p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[1, 0-15]
			SWISH_S32_AVX512(c_int32_1p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[2, 0-15]
			SWISH_S32_AVX512(c_int32_2p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[3, 0-15]
			SWISH_S32_AVX512(c_int32_3p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[4, 0-15]
			SWISH_S32_AVX512(c_int32_4p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[5, 0-15]
			SWISH_S32_AVX512(c_int32_5p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[6, 0-15]
			SWISH_S32_AVX512(c_int32_6p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[7, 0-15]
			SWISH_S32_AVX512(c_int32_7p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[8, 0-15]
			SWISH_S32_AVX512(c_int32_8p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[9, 0-15]
			SWISH_S32_AVX512(c_int32_9p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[10, 0-15]
			SWISH_S32_AVX512(c_int32_10p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[11, 0-15]
			SWISH_S32_AVX512(c_int32_11p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_12xLT16_DISABLE:
		;

		// Store the results.
		if ( ( post_ops_attr.buf_downscale != NULL ) &&
			 ( post_ops_attr.is_last_k == TRUE ) )
		{
			__mmask16 mask_all1 = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

			// Store the results in downscaled type (int8 instead of int32).
			// c[0,0-15]
			CVT_STORE_S32_S8(c_int32_0p0,0,0);

			// c[1,0-15]
			CVT_STORE_S32_S8(c_int32_1p0,1,0);

			// c[2,0-15]
			CVT_STORE_S32_S8(c_int32_2p0,2,0);

			// c[3,0-15]
			CVT_STORE_S32_S8(c_int32_3p0,3,0);

			// c[4,0-15]
			CVT_STORE_S32_S8(c_int32_4p0,4,0);

			// c[5,0-15]
			CVT_STORE_S32_S8(c_int32_5p0,5,0);

			// c[6,0-15]
			CVT_STORE_S32_S8(c_int32_6p0,6,0);

			// c[7,0-15]
			CVT_STORE_S32_S8(c_int32_7p0,7,0);

			// c[8,0-15]
			CVT_STORE_S32_S8(c_int32_8p0,8,0);

			// c[9,0-15]
			CVT_STORE_S32_S8(c_int32_9p0,9,0);

			// c[10,0-15]
			CVT_STORE_S32_S8(c_int32_10p0,10,0);

			// c[11,0-15]
			CVT_STORE_S32_S8(c_int32_11p0,11,0);
		}
		else
		{
			__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

			// Store the results.
			// c[0,0-15]
			_mm512_mask_storeu_epi32
			(
			  c + ( rs_c * ( ir + 0 ) ), load_mask, c_int32_0p0
			);

			// c[1,0-15]
			_mm512_mask_storeu_epi32
			(
			  c + ( rs_c * ( ir + 1 ) ), load_mask, c_int32_1p0
			);

			// c[2,0-15]
			_mm512_mask_storeu_epi32
			(
			  c + ( rs_c * ( ir + 2 ) ), load_mask, c_int32_2p0
			);

			// c[3,0-15]
			_mm512_mask_storeu_epi32
			(
			  c + ( rs_c * ( ir + 3 ) ), load_mask, c_int32_3p0
			);

			// c[4,0-15]
			_mm512_mask_storeu_epi32
			(
			  c + ( rs_c * ( ir + 4 ) ), load_mask, c_int32_4p0
			);

			// c[5,0-15]
			_mm512_mask_storeu_epi32
			(
			  c + ( rs_c * ( ir + 5 ) ), load_mask, c_int32_5p0
			);

			// c[6,0-15]
			_mm512_mask_storeu_epi32
			(
			  c + ( rs_c * ( ir + 6 ) ), load_mask, c_int32_6p0
			);

			// c[7,0-15]
			_mm512_mask_storeu_epi32
			(
			  c + ( rs_c * ( ir + 7 ) ), load_mask, c_int32_7p0
			);

			// c[8,0-15]
			_mm512_mask_storeu_epi32
			(
			  c + ( rs_c * ( ir + 8 ) ), load_mask, c_int32_8p0
			);

			// c[9,0-15]
			_mm512_mask_storeu_epi32
			(
			  c + ( rs_c * ( ir + 9 ) ), load_mask, c_int32_9p0
			);

			// c[10,0-15]
			_mm512_mask_storeu_epi32
			(
			  c + ( rs_c * ( ir + 10 ) ), load_mask, c_int32_10p0
			);

			// c[11,0-15]
			_mm512_mask_storeu_epi32
			(
			  c + ( rs_c * ( ir + 11 ) ), load_mask, c_int32_11p0
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
				  &&POST_OPS_SWISH_12x16
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

        // Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_12x16:
		{
			selector1 =
					_mm512_loadu_si512( ( int32_t* )post_ops_list_temp->op_args1 +
									post_ops_attr.post_op_c_j );

			// c[0,0-15]
			c_int32_0p0 = _mm512_add_epi32( selector1, c_int32_0p0 );

			// c[1,0-15]
			c_int32_1p0 = _mm512_add_epi32( selector1, c_int32_1p0 );

			// c[2,0-15]
			c_int32_2p0 = _mm512_add_epi32( selector1, c_int32_2p0 );

			// c[3,0-15]
			c_int32_3p0 = _mm512_add_epi32( selector1, c_int32_3p0 );

			// c[4,0-15]
			c_int32_4p0 = _mm512_add_epi32( selector1, c_int32_4p0 );

			// c[5,0-15]
			c_int32_5p0 = _mm512_add_epi32( selector1, c_int32_5p0 );

			// c[6,0-15]
			c_int32_6p0 = _mm512_add_epi32( selector1, c_int32_6p0 );

			// c[7,0-15]
			c_int32_7p0 = _mm512_add_epi32( selector1, c_int32_7p0 );

			// c[8,0-15]
			c_int32_8p0 = _mm512_add_epi32( selector1, c_int32_8p0 );

			// c[9,0-15]
			c_int32_9p0 = _mm512_add_epi32( selector1, c_int32_9p0 );

			// c[10,0-15]
			c_int32_10p0 = _mm512_add_epi32( selector1, c_int32_10p0 );

			// c[11,0-15]
			c_int32_11p0 = _mm512_add_epi32( selector1, c_int32_11p0 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_12x16:
		{
			selector1 = _mm512_setzero_epi32();

			// c[0,0-15]
			c_int32_0p0 = _mm512_max_epi32( selector1, c_int32_0p0 );

			// c[1,0-15]
			c_int32_1p0 = _mm512_max_epi32( selector1, c_int32_1p0 );

			// c[2,0-15]
			c_int32_2p0 = _mm512_max_epi32( selector1, c_int32_2p0 );

			// c[3,0-15]
			c_int32_3p0 = _mm512_max_epi32( selector1, c_int32_3p0 );

			// c[4,0-15]
			c_int32_4p0 = _mm512_max_epi32( selector1, c_int32_4p0 );

			// c[5,0-15]
			c_int32_5p0 = _mm512_max_epi32( selector1, c_int32_5p0 );

			// c[6,0-15]
			c_int32_6p0 = _mm512_max_epi32( selector1, c_int32_6p0 );

			// c[7,0-15]
			c_int32_7p0 = _mm512_max_epi32( selector1, c_int32_7p0 );

			// c[8,0-15]
			c_int32_8p0 = _mm512_max_epi32( selector1, c_int32_8p0 );

			// c[9,0-15]
			c_int32_9p0 = _mm512_max_epi32( selector1, c_int32_9p0 );

			// c[10,0-15]
			c_int32_10p0 = _mm512_max_epi32( selector1, c_int32_10p0 );

			// c[11,0-15]
			c_int32_11p0 = _mm512_max_epi32( selector1, c_int32_11p0 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_12x16:
		{
			selector1 = _mm512_setzero_epi32();
			selector2 =
				_mm512_set1_epi32( *( ( int32_t* )post_ops_list_temp->op_args2 ) );

			__mmask16 relu_cmp_mask;

			// c[0, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_0p0)

			// c[1, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_1p0)

			// c[2, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_2p0)

			// c[3, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_3p0)

			// c[4, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_4p0)

			// c[5, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_5p0)

			// c[6, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_6p0)

			// c[7, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_7p0)

			// c[8, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_8p0)

			// c[9, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_9p0)

			// c[10, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_10p0)

			// c[11, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_11p0)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_TANH_12x16:
		{
			__m512 dn, z, x, r2, r, y, x_tanh;
			__m512i q;

			// c[0, 0-15]
			GELU_TANH_S32_AVX512(c_int32_0p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[1, 0-15]
			GELU_TANH_S32_AVX512(c_int32_1p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[2, 0-15]
			GELU_TANH_S32_AVX512(c_int32_2p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[3, 0-15]
			GELU_TANH_S32_AVX512(c_int32_3p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[4, 0-15]
			GELU_TANH_S32_AVX512(c_int32_4p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[5, 0-15]
			GELU_TANH_S32_AVX512(c_int32_5p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[6, 0-15]
			GELU_TANH_S32_AVX512(c_int32_6p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[7, 0-15]
			GELU_TANH_S32_AVX512(c_int32_7p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[8, 0-15]
			GELU_TANH_S32_AVX512(c_int32_8p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[9, 0-15]
			GELU_TANH_S32_AVX512(c_int32_9p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[10, 0-15]
			GELU_TANH_S32_AVX512(c_int32_10p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[11, 0-15]
			GELU_TANH_S32_AVX512(c_int32_11p0, y, r, r2, x, z, dn, x_tanh, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_ERF_12x16:
		{
			__m512 x, r, y, x_erf;

			// c[0, 0-15]
			GELU_ERF_S32_AVX512(c_int32_0p0, y, r, x, x_erf)

			// c[1, 0-15]
			GELU_ERF_S32_AVX512(c_int32_1p0, y, r, x, x_erf)

			// c[2, 0-15]
			GELU_ERF_S32_AVX512(c_int32_2p0, y, r, x, x_erf)

			// c[3, 0-15]
			GELU_ERF_S32_AVX512(c_int32_3p0, y, r, x, x_erf)

			// c[4, 0-15]
			GELU_ERF_S32_AVX512(c_int32_4p0, y, r, x, x_erf)

			// c[5, 0-15]
			GELU_ERF_S32_AVX512(c_int32_5p0, y, r, x, x_erf)

			// c[6, 0-15]
			GELU_ERF_S32_AVX512(c_int32_6p0, y, r, x, x_erf)

			// c[7, 0-15]
			GELU_ERF_S32_AVX512(c_int32_7p0, y, r, x, x_erf)

			// c[8, 0-15]
			GELU_ERF_S32_AVX512(c_int32_8p0, y, r, x, x_erf)

			// c[9, 0-15]
			GELU_ERF_S32_AVX512(c_int32_9p0, y, r, x, x_erf)

			// c[10, 0-15]
			GELU_ERF_S32_AVX512(c_int32_10p0, y, r, x, x_erf)

			// c[11, 0-15]
			GELU_ERF_S32_AVX512(c_int32_11p0, y, r, x, x_erf)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_CLIP_12x16:
		{
			__m512i min = _mm512_set1_epi32( *( int32_t* )post_ops_list_temp->op_args2 );
			__m512i max = _mm512_set1_epi32( *( int32_t* )post_ops_list_temp->op_args3 );

			// c[0, 0-15]
			CLIP_S32_AVX512(c_int32_0p0, min, max)

			// c[1, 0-15]
			CLIP_S32_AVX512(c_int32_1p0, min, max)

			// c[2, 0-15]
			CLIP_S32_AVX512(c_int32_2p0, min, max)

			// c[3, 0-15]
			CLIP_S32_AVX512(c_int32_3p0, min, max)

			// c[4, 0-15]
			CLIP_S32_AVX512(c_int32_4p0, min, max)

			// c[5, 0-15]
			CLIP_S32_AVX512(c_int32_5p0, min, max)

			// c[6, 0-15]
			CLIP_S32_AVX512(c_int32_6p0, min, max)

			// c[7, 0-15]
			CLIP_S32_AVX512(c_int32_7p0, min, max)

			// c[8, 0-15]
			CLIP_S32_AVX512(c_int32_8p0, min, max)

			// c[9, 0-15]
			CLIP_S32_AVX512(c_int32_9p0, min, max)

			// c[10, 0-15]
			CLIP_S32_AVX512(c_int32_10p0, min, max)

			// c[11, 0-15]
			CLIP_S32_AVX512(c_int32_11p0, min, max)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_DOWNSCALE_12x16:
	{
		if ( post_ops_list_temp->scale_factor_len > 1 )
		{
			selector1 =
				_mm512_loadu_si512( ( float* )post_ops_list_temp->scale_factor +
					post_ops_attr.post_op_c_j + ( 0 * 16 ) );
		}
		else if ( post_ops_list_temp->scale_factor_len == 1 )
		{
			selector1 =
				( __m512i )_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
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
		CVT_MULRND_CVT32(c_int32_0p0,selector1,zero_point0);

		// c[1, 0-15]
		CVT_MULRND_CVT32(c_int32_1p0,selector1,zero_point0);

		// c[2, 0-15]
		CVT_MULRND_CVT32(c_int32_2p0,selector1,zero_point0);

		// c[3, 0-15]
		CVT_MULRND_CVT32(c_int32_3p0,selector1,zero_point0);

		// c[4, 0-15]
		CVT_MULRND_CVT32(c_int32_4p0,selector1,zero_point0);

		// c[5, 0-15]
		CVT_MULRND_CVT32(c_int32_5p0,selector1,zero_point0);

		// c[6, 0-15]
		CVT_MULRND_CVT32(c_int32_6p0,selector1,zero_point0);

		// c[7, 0-15]
		CVT_MULRND_CVT32(c_int32_7p0,selector1,zero_point0);

		// c[8, 0-15]
		CVT_MULRND_CVT32(c_int32_8p0,selector1,zero_point0);

		// c[9, 0-15]
		CVT_MULRND_CVT32(c_int32_9p0,selector1,zero_point0);

		// c[10, 0-15]
		CVT_MULRND_CVT32(c_int32_10p0,selector1,zero_point0);

		// c[11, 0-15]
		CVT_MULRND_CVT32(c_int32_11p0,selector1,zero_point0);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_ADD_12x16:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
			if ( post_ops_attr.c_stor_type == S8 )
			{
				int8_t* matptr = ( int8_t* )post_ops_list_temp->op_args1;

				// c[0:0-15]
				S8_S32_MATRIX_ADD_1COL(selector1,0);

				// c[1:0-15]
				S8_S32_MATRIX_ADD_1COL(selector1,1);

				// c[2:0-15]
				S8_S32_MATRIX_ADD_1COL(selector1,2);

				// c[3:0-15]
				S8_S32_MATRIX_ADD_1COL(selector1,3);

				// c[4:0-15]
				S8_S32_MATRIX_ADD_1COL(selector1,4);

				// c[5:0-15]
				S8_S32_MATRIX_ADD_1COL(selector1,5);

				// c[6:0-15]
				S8_S32_MATRIX_ADD_1COL(selector1,6);

				// c[7:0-15]
				S8_S32_MATRIX_ADD_1COL(selector1,7);

				// c[8:0-15]
				S8_S32_MATRIX_ADD_1COL(selector1,8);

				// c[9:0-15]
				S8_S32_MATRIX_ADD_1COL(selector1,9);

				// c[10:0-15]
				S8_S32_MATRIX_ADD_1COL(selector1,10);

				// c[11:0-15]
				S8_S32_MATRIX_ADD_1COL(selector1,11);
			}
			else
			{
				int32_t* matptr = ( int32_t* )post_ops_list_temp->op_args1;

				// c[0:0-15]
				S32_S32_MATRIX_ADD_1COL(selector1,0);

				// c[1:0-15]
				S32_S32_MATRIX_ADD_1COL(selector1,1);

				// c[2:0-15]
				S32_S32_MATRIX_ADD_1COL(selector1,2);

				// c[3:0-15]
				S32_S32_MATRIX_ADD_1COL(selector1,3);

				// c[4:0-15]
				S32_S32_MATRIX_ADD_1COL(selector1,4);

				// c[5:0-15]
				S32_S32_MATRIX_ADD_1COL(selector1,5);

				// c[6:0-15]
				S32_S32_MATRIX_ADD_1COL(selector1,6);

				// c[7:0-15]
				S32_S32_MATRIX_ADD_1COL(selector1,7);

				// c[8:0-15]
				S32_S32_MATRIX_ADD_1COL(selector1,8);

				// c[9:0-15]
				S32_S32_MATRIX_ADD_1COL(selector1,9);

				// c[10:0-15]
				S32_S32_MATRIX_ADD_1COL(selector1,10);

				// c[11:0-15]
				S32_S32_MATRIX_ADD_1COL(selector1,11);
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SWISH_12x16:
		{
			selector1 =
				_mm512_set1_epi32( *( ( int32_t* )post_ops_list_temp->op_args2 ) );
			__m512 al = _mm512_cvtepi32_ps( selector1 );

			__m512 fl_reg, al_in, r, r2, z, dn;

			// c[0, 0-15]
			SWISH_S32_AVX512(c_int32_0p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[1, 0-15]
			SWISH_S32_AVX512(c_int32_1p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[2, 0-15]
			SWISH_S32_AVX512(c_int32_2p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[3, 0-15]
			SWISH_S32_AVX512(c_int32_3p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[4, 0-15]
			SWISH_S32_AVX512(c_int32_4p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[5, 0-15]
			SWISH_S32_AVX512(c_int32_5p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[6, 0-15]
			SWISH_S32_AVX512(c_int32_6p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[7, 0-15]
			SWISH_S32_AVX512(c_int32_7p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[8, 0-15]
			SWISH_S32_AVX512(c_int32_8p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[9, 0-15]
			SWISH_S32_AVX512(c_int32_9p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[10, 0-15]
			SWISH_S32_AVX512(c_int32_10p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[11, 0-15]
			SWISH_S32_AVX512(c_int32_11p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

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

			// Store the results in downscaled type (int8 instead of int32).
			// c[0,0-15]
			CVT_STORE_S32_S8(c_int32_0p0,0,0);

			// c[1,0-15]
			CVT_STORE_S32_S8(c_int32_1p0,1,0);

			// c[2,0-15]
			CVT_STORE_S32_S8(c_int32_2p0,2,0);

			// c[3,0-15]
			CVT_STORE_S32_S8(c_int32_3p0,3,0);

			// c[4,0-15]
			CVT_STORE_S32_S8(c_int32_4p0,4,0);

			// c[5,0-15]
			CVT_STORE_S32_S8(c_int32_5p0,5,0);

			// c[6,0-15]
			CVT_STORE_S32_S8(c_int32_6p0,6,0);

			// c[7,0-15]
			CVT_STORE_S32_S8(c_int32_7p0,7,0);

			// c[8,0-15]
			CVT_STORE_S32_S8(c_int32_8p0,8,0);

			// c[9,0-15]
			CVT_STORE_S32_S8(c_int32_9p0,9,0);

			// c[10,0-15]
			CVT_STORE_S32_S8(c_int32_10p0,10,0);

			// c[11,0-15]
			CVT_STORE_S32_S8(c_int32_11p0,11,0);
		}
		else
		{
			// Store the results.
			// c[0,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 0 ) ) + ( 0*16 ), c_int32_0p0 );

			// c[1,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 1 ) ) + ( 0*16 ), c_int32_1p0 );

			// c[2,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 2 ) ) + ( 0*16 ), c_int32_2p0 );

			// c[3,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 3 ) ) + ( 0*16 ), c_int32_3p0 );

			// c[4,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 4 ) ) + ( 0*16 ), c_int32_4p0 );

			// c[5,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 5 ) ) + ( 0*16 ), c_int32_5p0 );

			// c[6,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 6 ) ) + ( 0*16 ), c_int32_6p0 );

			// c[7,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 7 ) ) + ( 0*16 ), c_int32_7p0 );

			// c[8,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 8 ) ) + ( 0*16 ), c_int32_8p0 );

			// c[9,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 9 ) ) + ( 0*16 ), c_int32_9p0 );

			// c[10,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 10 ) ) + ( 0*16 ), c_int32_10p0 );

			// c[11,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 11 ) ) + ( 0*16 ), c_int32_11p0 );
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
				  &&POST_OPS_SWISH_9x32
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

        // Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_9x32:
		{
			selector1 =
					_mm512_loadu_si512( ( int32_t* )post_ops_list_temp->op_args1 +
									post_ops_attr.post_op_c_j + ( 0 * 16 ) );
			selector2 =
					_mm512_loadu_si512( ( int32_t* )post_ops_list_temp->op_args1 +
									post_ops_attr.post_op_c_j + ( 1 * 16 ) );

			// c[0,0-15]
			c_int32_0p0 = _mm512_add_epi32( selector1, c_int32_0p0 );

			// c[0, 16-31]
			c_int32_0p1 = _mm512_add_epi32( selector2, c_int32_0p1 );

			// c[1,0-15]
			c_int32_1p0 = _mm512_add_epi32( selector1, c_int32_1p0 );

			// c[1, 16-31]
			c_int32_1p1 = _mm512_add_epi32( selector2, c_int32_1p1 );

			// c[2,0-15]
			c_int32_2p0 = _mm512_add_epi32( selector1, c_int32_2p0 );

			// c[2, 16-31]
			c_int32_2p1 = _mm512_add_epi32( selector2, c_int32_2p1 );

			// c[3,0-15]
			c_int32_3p0 = _mm512_add_epi32( selector1, c_int32_3p0 );

			// c[3, 16-31]
			c_int32_3p1 = _mm512_add_epi32( selector2, c_int32_3p1 );

			// c[4,0-15]
			c_int32_4p0 = _mm512_add_epi32( selector1, c_int32_4p0 );

			// c[4, 16-31]
			c_int32_4p1 = _mm512_add_epi32( selector2, c_int32_4p1 );

			// c[5,0-15]
			c_int32_5p0 = _mm512_add_epi32( selector1, c_int32_5p0 );

			// c[5, 16-31]
			c_int32_5p1 = _mm512_add_epi32( selector2, c_int32_5p1 );

			// c[6,0-15]
			c_int32_6p0 = _mm512_add_epi32( selector1, c_int32_6p0 );

			// c[6, 16-31]
			c_int32_6p1 = _mm512_add_epi32( selector2, c_int32_6p1 );

			// c[7,0-15]
			c_int32_7p0 = _mm512_add_epi32( selector1, c_int32_7p0 );

			// c[7, 16-31]
			c_int32_7p1 = _mm512_add_epi32( selector2, c_int32_7p1 );

			// c[8,0-15]
			c_int32_8p0 = _mm512_add_epi32( selector1, c_int32_8p0 );

			// c[8, 16-31]
			c_int32_8p1 = _mm512_add_epi32( selector2, c_int32_8p1 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_9x32:
		{
			selector1 = _mm512_setzero_epi32();

			// c[0,0-15]
			c_int32_0p0 = _mm512_max_epi32( selector1, c_int32_0p0 );

			// c[0, 16-31]
			c_int32_0p1 = _mm512_max_epi32( selector1, c_int32_0p1 );

			// c[1,0-15]
			c_int32_1p0 = _mm512_max_epi32( selector1, c_int32_1p0 );

			// c[1,16-31]
			c_int32_1p1 = _mm512_max_epi32( selector1, c_int32_1p1 );

			// c[2,0-15]
			c_int32_2p0 = _mm512_max_epi32( selector1, c_int32_2p0 );

			// c[2,16-31]
			c_int32_2p1 = _mm512_max_epi32( selector1, c_int32_2p1 );

			// c[3,0-15]
			c_int32_3p0 = _mm512_max_epi32( selector1, c_int32_3p0 );

			// c[3,16-31]
			c_int32_3p1 = _mm512_max_epi32( selector1, c_int32_3p1 );

			// c[4,0-15]
			c_int32_4p0 = _mm512_max_epi32( selector1, c_int32_4p0 );

			// c[4,16-31]
			c_int32_4p1 = _mm512_max_epi32( selector1, c_int32_4p1 );

			// c[5,0-15]
			c_int32_5p0 = _mm512_max_epi32( selector1, c_int32_5p0 );

			// c[5,16-31]
			c_int32_5p1 = _mm512_max_epi32( selector1, c_int32_5p1 );

			// c[6,0-15]
			c_int32_6p0 = _mm512_max_epi32( selector1, c_int32_6p0 );

			// c[6, 16-31]
			c_int32_6p1 = _mm512_max_epi32( selector1, c_int32_6p1 );

			// c[7,0-15]
			c_int32_7p0 = _mm512_max_epi32( selector1, c_int32_7p0 );

			// c[7,16-31]
			c_int32_7p1 = _mm512_max_epi32( selector1, c_int32_7p1 );

			// c[8,0-15]
			c_int32_8p0 = _mm512_max_epi32( selector1, c_int32_8p0 );

			// c[8,16-31]
			c_int32_8p1 = _mm512_max_epi32( selector1, c_int32_8p1 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_9x32:
		{
			selector1 = _mm512_setzero_epi32();
			selector2 =
				_mm512_set1_epi32( *( ( int32_t* )post_ops_list_temp->op_args2 ) );

			__mmask16 relu_cmp_mask;

			// c[0, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_0p0)

			// c[0, 16-31]
			RELU_SCALE_OP_S32_AVX512(c_int32_0p1)

			// c[1, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_1p0)

			// c[1, 16-31]
			RELU_SCALE_OP_S32_AVX512(c_int32_1p1)

			// c[2, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_2p0)

			// c[2, 16-31]
			RELU_SCALE_OP_S32_AVX512(c_int32_2p1)

			// c[3, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_3p0)

			// c[3, 16-31]
			RELU_SCALE_OP_S32_AVX512(c_int32_3p1)

			// c[4, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_4p0)

			// c[4, 16-31]
			RELU_SCALE_OP_S32_AVX512(c_int32_4p1)

			// c[5, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_5p0)

			// c[5, 16-31]
			RELU_SCALE_OP_S32_AVX512(c_int32_5p1)

			// c[6, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_6p0)

			// c[6, 16-31]
			RELU_SCALE_OP_S32_AVX512(c_int32_6p1)

			// c[7, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_7p0)

			// c[7, 16-31]
			RELU_SCALE_OP_S32_AVX512(c_int32_7p1)

			// c[8, 0-15]
			RELU_SCALE_OP_S32_AVX512(c_int32_8p0)

			// c[8, 16-31]
			RELU_SCALE_OP_S32_AVX512(c_int32_8p1)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_TANH_9x32:
		{
			__m512 dn, z, x, r2, r, y, x_tanh;
			__m512i q;

			// c[0, 0-15]
			GELU_TANH_S32_AVX512(c_int32_0p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[0, 16-31]
			GELU_TANH_S32_AVX512(c_int32_0p1, y, r, r2, x, z, dn, x_tanh, q)

			// c[1, 0-15]
			GELU_TANH_S32_AVX512(c_int32_1p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[1, 16-31]
			GELU_TANH_S32_AVX512(c_int32_1p1, y, r, r2, x, z, dn, x_tanh, q)

			// c[2, 0-15]
			GELU_TANH_S32_AVX512(c_int32_2p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[2, 16-31]
			GELU_TANH_S32_AVX512(c_int32_2p1, y, r, r2, x, z, dn, x_tanh, q)

			// c[3, 0-15]
			GELU_TANH_S32_AVX512(c_int32_3p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[3, 16-31]
			GELU_TANH_S32_AVX512(c_int32_3p1, y, r, r2, x, z, dn, x_tanh, q)

			// c[4, 0-15]
			GELU_TANH_S32_AVX512(c_int32_4p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[4, 16-31]
			GELU_TANH_S32_AVX512(c_int32_4p1, y, r, r2, x, z, dn, x_tanh, q)

			// c[5, 0-15]
			GELU_TANH_S32_AVX512(c_int32_5p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[5, 16-31]
			GELU_TANH_S32_AVX512(c_int32_5p1, y, r, r2, x, z, dn, x_tanh, q)

			// c[6, 0-15]
			GELU_TANH_S32_AVX512(c_int32_6p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[6, 16-31]
			GELU_TANH_S32_AVX512(c_int32_6p1, y, r, r2, x, z, dn, x_tanh, q)

			// c[7, 0-15]
			GELU_TANH_S32_AVX512(c_int32_7p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[7, 16-31]
			GELU_TANH_S32_AVX512(c_int32_7p1, y, r, r2, x, z, dn, x_tanh, q)

			// c[8, 0-15]
			GELU_TANH_S32_AVX512(c_int32_8p0, y, r, r2, x, z, dn, x_tanh, q)

			// c[8, 16-31]
			GELU_TANH_S32_AVX512(c_int32_8p1, y, r, r2, x, z, dn, x_tanh, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_ERF_9x32:
		{
			__m512 x, r, y, x_erf;

			// c[0, 0-15]
			GELU_ERF_S32_AVX512(c_int32_0p0, y, r, x, x_erf)

			// c[0, 16-31]
			GELU_ERF_S32_AVX512(c_int32_0p1, y, r, x, x_erf)

			// c[1, 0-15]
			GELU_ERF_S32_AVX512(c_int32_1p0, y, r, x, x_erf)

			// c[1, 16-31]
			GELU_ERF_S32_AVX512(c_int32_1p1, y, r, x, x_erf)

			// c[2, 0-15]
			GELU_ERF_S32_AVX512(c_int32_2p0, y, r, x, x_erf)

			// c[2, 16-31]
			GELU_ERF_S32_AVX512(c_int32_2p1, y, r, x, x_erf)

			// c[3, 0-15]
			GELU_ERF_S32_AVX512(c_int32_3p0, y, r, x, x_erf)

			// c[3, 16-31]
			GELU_ERF_S32_AVX512(c_int32_3p1, y, r, x, x_erf)

			// c[4, 0-15]
			GELU_ERF_S32_AVX512(c_int32_4p0, y, r, x, x_erf)

			// c[4, 16-31]
			GELU_ERF_S32_AVX512(c_int32_4p1, y, r, x, x_erf)

			// c[5, 0-15]
			GELU_ERF_S32_AVX512(c_int32_5p0, y, r, x, x_erf)

			// c[5, 16-31]
			GELU_ERF_S32_AVX512(c_int32_5p1, y, r, x, x_erf)

			// c[6, 0-15]
			GELU_ERF_S32_AVX512(c_int32_6p0, y, r, x, x_erf)

			// c[6, 16-31]
			GELU_ERF_S32_AVX512(c_int32_6p1, y, r, x, x_erf)

			// c[7, 0-15]
			GELU_ERF_S32_AVX512(c_int32_7p0, y, r, x, x_erf)

			// c[7, 16-31]
			GELU_ERF_S32_AVX512(c_int32_7p1, y, r, x, x_erf)

			// c[8, 0-15]
			GELU_ERF_S32_AVX512(c_int32_8p0, y, r, x, x_erf)

			// c[8, 16-31]
			GELU_ERF_S32_AVX512(c_int32_8p1, y, r, x, x_erf)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_CLIP_9x32:
		{
			__m512i min = _mm512_set1_epi32( *( int32_t* )post_ops_list_temp->op_args2 );
			__m512i max = _mm512_set1_epi32( *( int32_t* )post_ops_list_temp->op_args3 );

			// c[0, 0-15]
			CLIP_S32_AVX512(c_int32_0p0, min, max)

			// c[0, 16-31]
			CLIP_S32_AVX512(c_int32_0p1, min, max)

			// c[1, 0-15]
			CLIP_S32_AVX512(c_int32_1p0, min, max)

			// c[1, 16-31]
			CLIP_S32_AVX512(c_int32_1p1, min, max)

			// c[2, 0-15]
			CLIP_S32_AVX512(c_int32_2p0, min, max)

			// c[2, 16-31]
			CLIP_S32_AVX512(c_int32_2p1, min, max)

			// c[3, 0-15]
			CLIP_S32_AVX512(c_int32_3p0, min, max)

			// c[3, 16-31]
			CLIP_S32_AVX512(c_int32_3p1, min, max)

			// c[4, 0-15]
			CLIP_S32_AVX512(c_int32_4p0, min, max)

			// c[4, 16-31]
			CLIP_S32_AVX512(c_int32_4p1, min, max)

			// c[5, 0-15]
			CLIP_S32_AVX512(c_int32_5p0, min, max)

			// c[5, 16-31]
			CLIP_S32_AVX512(c_int32_5p1, min, max)

			// c[6, 0-15]
			CLIP_S32_AVX512(c_int32_6p0, min, max)

			// c[6, 16-31]
			CLIP_S32_AVX512(c_int32_6p1, min, max)

			// c[7, 0-15]
			CLIP_S32_AVX512(c_int32_7p0, min, max)

			// c[7, 16-31]
			CLIP_S32_AVX512(c_int32_7p1, min, max)

			// c[8, 0-15]
			CLIP_S32_AVX512(c_int32_8p0, min, max)

			// c[8, 16-31]
			CLIP_S32_AVX512(c_int32_8p1, min, max)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_DOWNSCALE_9x32:
	{
		if ( post_ops_list_temp->scale_factor_len > 1 )
		{
			selector1 =
				_mm512_loadu_si512( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 0 * 16 ) );
			selector2 =
				_mm512_loadu_si512( ( float* )post_ops_list_temp->scale_factor +
								post_ops_attr.post_op_c_j + ( 1 * 16 ) );
		}
		else if ( post_ops_list_temp->scale_factor_len == 1 )
		{
			selector1 =
				( __m512i )_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
			selector2 =
				( __m512i )_mm512_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
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
		CVT_MULRND_CVT32(c_int32_0p0,selector1,zero_point0);

		// c[0, 16-31]
		CVT_MULRND_CVT32(c_int32_0p1,selector2,zero_point1);

		// c[1, 0-15]
		CVT_MULRND_CVT32(c_int32_1p0,selector1,zero_point0);

		// c[1, 16-31]
		CVT_MULRND_CVT32(c_int32_1p1,selector2,zero_point1);

		// c[2, 0-15]
		CVT_MULRND_CVT32(c_int32_2p0,selector1,zero_point0);

		// c[2, 16-31]
		CVT_MULRND_CVT32(c_int32_2p1,selector2,zero_point1);

		// c[3, 0-15]
		CVT_MULRND_CVT32(c_int32_3p0,selector1,zero_point0);

		// c[3, 16-31]
		CVT_MULRND_CVT32(c_int32_3p1,selector2,zero_point1);

		// c[4, 0-15]
		CVT_MULRND_CVT32(c_int32_4p0,selector1,zero_point0);

		// c[4, 16-31]
		CVT_MULRND_CVT32(c_int32_4p1,selector2,zero_point1);

		// c[5, 0-15]
		CVT_MULRND_CVT32(c_int32_5p0,selector1,zero_point0);

		// c[5, 16-31]
		CVT_MULRND_CVT32(c_int32_5p1,selector2,zero_point1);

		// c[6, 0-15]
		CVT_MULRND_CVT32(c_int32_6p0,selector1,zero_point0);

		// c[6, 16-31]
		CVT_MULRND_CVT32(c_int32_6p1,selector2,zero_point1);

		// c[7, 0-15]
		CVT_MULRND_CVT32(c_int32_7p0,selector1,zero_point0);

		// c[7, 16-31]
		CVT_MULRND_CVT32(c_int32_7p1,selector2,zero_point1);

		// c[8, 0-15]
		CVT_MULRND_CVT32(c_int32_8p0,selector1,zero_point0);

		// c[8, 16-31]
		CVT_MULRND_CVT32(c_int32_8p1,selector2,zero_point1);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_MATRIX_ADD_9x32:
		{
			dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
			if ( post_ops_attr.c_stor_type == S8 )
			{
				int8_t* matptr = ( int8_t* )post_ops_list_temp->op_args1;

				// c[0:0-15,16-31]
				S8_S32_MATRIX_ADD_2COL(selector1,selector2,0);

				// c[1:0-15,16-31]
				S8_S32_MATRIX_ADD_2COL(selector1,selector2,1);

				// c[2:0-15,16-31]
				S8_S32_MATRIX_ADD_2COL(selector1,selector2,2);

				// c[3:0-15,16-31]
				S8_S32_MATRIX_ADD_2COL(selector1,selector2,3);

				// c[4:0-15,16-31]
				S8_S32_MATRIX_ADD_2COL(selector1,selector2,4);

				// c[5:0-15,16-31]
				S8_S32_MATRIX_ADD_2COL(selector1,selector2,5);

				// c[6:0-15,16-31]
				S8_S32_MATRIX_ADD_2COL(selector1,selector2,6);

				// c[7:0-15,16-31]
				S8_S32_MATRIX_ADD_2COL(selector1,selector2,7);

				// c[8:0-15,16-31]
				S8_S32_MATRIX_ADD_2COL(selector1,selector2,8);
			}
			else
			{
				int32_t* matptr = ( int32_t* )post_ops_list_temp->op_args1;

				// c[0:0-15,16-31]
				S32_S32_MATRIX_ADD_2COL(selector1,selector2,0);

				// c[1:0-15,16-31]
				S32_S32_MATRIX_ADD_2COL(selector1,selector2,1);

				// c[2:0-15,16-31]
				S32_S32_MATRIX_ADD_2COL(selector1,selector2,2);

				// c[3:0-15,16-31]
				S32_S32_MATRIX_ADD_2COL(selector1,selector2,3);

				// c[4:0-15,16-31]
				S32_S32_MATRIX_ADD_2COL(selector1,selector2,4);

				// c[5:0-15,16-31]
				S32_S32_MATRIX_ADD_2COL(selector1,selector2,5);

				// c[6:0-15,16-31]
				S32_S32_MATRIX_ADD_2COL(selector1,selector2,6);

				// c[7:0-15,16-31]
				S32_S32_MATRIX_ADD_2COL(selector1,selector2,7);

				// c[8:0-15,16-31]
				S32_S32_MATRIX_ADD_2COL(selector1,selector2,8);
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_SWISH_9x32:
		{
			selector1 =
				_mm512_set1_epi32( *( ( int32_t* )post_ops_list_temp->op_args2 ) );
			__m512 al = _mm512_cvtepi32_ps( selector1 );

			__m512 fl_reg, al_in, r, r2, z, dn;

			// c[0, 0-15]
			SWISH_S32_AVX512(c_int32_0p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[0, 16-31]
			SWISH_S32_AVX512(c_int32_0p1, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[1, 0-15]
			SWISH_S32_AVX512(c_int32_1p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[1, 16-31]
			SWISH_S32_AVX512(c_int32_1p1, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[2, 0-15]
			SWISH_S32_AVX512(c_int32_2p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[2, 16-31]
			SWISH_S32_AVX512(c_int32_2p1, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[3, 0-15]
			SWISH_S32_AVX512(c_int32_3p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[3, 16-31]
			SWISH_S32_AVX512(c_int32_3p1, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[4, 0-15]
			SWISH_S32_AVX512(c_int32_4p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[4, 16-31]
			SWISH_S32_AVX512(c_int32_4p1, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[5, 0-15]
			SWISH_S32_AVX512(c_int32_5p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[5, 16-31]
			SWISH_S32_AVX512(c_int32_5p1, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[6, 0-15]
			SWISH_S32_AVX512(c_int32_6p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[6, 16-31]
			SWISH_S32_AVX512(c_int32_6p1, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[7, 0-15]
			SWISH_S32_AVX512(c_int32_7p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[7, 16-31]
			SWISH_S32_AVX512(c_int32_7p1, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[8, 0-15]
			SWISH_S32_AVX512(c_int32_8p0, fl_reg, al, al_in, r, r2, z, dn, selector2);

			// c[8, 16-31]
			SWISH_S32_AVX512(c_int32_8p1, fl_reg, al, al_in, r, r2, z, dn, selector2);

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

			// Store the results in downscaled type (int8 instead of int32).
			// c[0,0-15]
			CVT_STORE_S32_S8(c_int32_0p0,0,0);

			// c[0,16-31]
			CVT_STORE_S32_S8(c_int32_0p1,0,1);

			// c[1,0-15]
			CVT_STORE_S32_S8(c_int32_1p0,1,0);

			// c[1,16-31]
			CVT_STORE_S32_S8(c_int32_1p1,1,1);

			// c[2,0-15]
			CVT_STORE_S32_S8(c_int32_2p0,2,0);

			// c[2,16-31]
			CVT_STORE_S32_S8(c_int32_2p1,2,1);

			// c[3,0-15]
			CVT_STORE_S32_S8(c_int32_3p0,3,0);

			// c[3,16-31]
			CVT_STORE_S32_S8(c_int32_3p1,3,1);

			// c[4,0-15]
			CVT_STORE_S32_S8(c_int32_4p0,4,0);

			// c[4,16-31]
			CVT_STORE_S32_S8(c_int32_4p1,4,1);

			// c[5,0-15]
			CVT_STORE_S32_S8(c_int32_5p0,5,0);

			// c[5,16-31]
			CVT_STORE_S32_S8(c_int32_5p1,5,1);

			// c[6,0-15]
			CVT_STORE_S32_S8(c_int32_6p0,6,0);

			// c[6,16-31]
			CVT_STORE_S32_S8(c_int32_6p1,6,1);

			// c[7,0-15]
			CVT_STORE_S32_S8(c_int32_7p0,7,0);

			// c[7,16-31]
			CVT_STORE_S32_S8(c_int32_7p1,7,1);

			// c[8,0-15]
			CVT_STORE_S32_S8(c_int32_8p0,8,0);

			// c[8,16-31]
			CVT_STORE_S32_S8(c_int32_8p1,8,1);
		}
		else
		{
			// Store the results.
			// c[0,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 0 ) ) + ( 0*16 ), c_int32_0p0 );

			// c[0, 16-31]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 0 ) ) + ( 1*16 ), c_int32_0p1 );

			// c[1,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 1 ) ) + ( 0*16 ), c_int32_1p0 );

			// c[1,16-31]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 1 ) ) + ( 1*16 ), c_int32_1p1 );

			// c[2,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 2 ) ) + ( 0*16 ), c_int32_2p0 );

			// c[2,16-31]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 2 ) ) + ( 1*16 ), c_int32_2p1 );

			// c[3,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 3 ) ) + ( 0*16 ), c_int32_3p0 );

			// c[3,16-31]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 3 ) ) + ( 1*16 ), c_int32_3p1 );

			// c[4,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 4 ) ) + ( 0*16 ), c_int32_4p0 );

			// c[4,16-31]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 4 ) ) + ( 1*16 ), c_int32_4p1 );

			// c[5,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 5 ) ) + ( 0*16 ), c_int32_5p0 );

			// c[5,16-31]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 5 ) ) + ( 1*16 ), c_int32_5p1 );

			// c[6,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 6 ) ) + ( 0*16 ), c_int32_6p0 );

			// c[6, 16-31]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 6 ) ) + ( 1*16 ), c_int32_6p1 );

			// c[7,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 7 ) ) + ( 0*16 ), c_int32_7p0 );

			// c[7,16-31]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 7 ) ) + ( 1*16 ), c_int32_7p1 );

			// c[8,0-15]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 8 ) ) + ( 0*16 ), c_int32_8p0 );

			// c[8,16-31]
			_mm512_storeu_si512( c + ( rs_c * ( ir + 8 ) ) + ( 1*16 ), c_int32_8p1 );
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
