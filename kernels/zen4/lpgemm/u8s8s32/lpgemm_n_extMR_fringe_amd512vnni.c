/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

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
				  &&POST_OPS_DOWNSCALE_12xLT16
				};
	dim_t MR = 12;
	dim_t m_full_pieces = m0 / MR;
	dim_t m_full_pieces_loop_limit = m_full_pieces * MR;
	dim_t m_partial_pieces = m0 % MR;

	dim_t k_full_pieces = k0 / 4;
	dim_t k_partial_pieces = k0 % 4;

	uint32_t a_kfringe_buf = 0;

	// B matrix storage.
	__m512i b0;

	// A matrix storage.
	__m512i a_int32_0;
	__m512i a_int32_1;
	__m512i a_int32_2;
	__m512i a_int32_3;
	__m512i a_int32_4;
	__m512i a_int32_5;
	__m512i a_int32_6;
	__m512i a_int32_7;
	__m512i a_int32_8;
	__m512i a_int32_9;
	__m512i a_int32_10;
	__m512i a_int32_11;

	// For corner cases.
	int32_t buf0[16];
	int32_t buf1[16];
	int32_t buf2[16];
	int32_t buf3[16];
	int32_t buf4[16];
	int32_t buf5[16];

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
			b0 = _mm512_loadu_epi8( b + ( rs_b * kr ) + ( cs_b * 0 ) );

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
			b0 = _mm512_loadu_epi8( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );

			// Broadcast a[0,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_0 = _mm512_set1_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[0,0-15] = a[0,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_0p0 = _mm512_dpbusd_epi32( c_int32_0p0, a_int32_0, b0 );

			// Broadcast a[1,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 1 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_1 = _mm512_set1_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[1,0-15] = a[1,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_1p0 = _mm512_dpbusd_epi32( c_int32_1p0, a_int32_1, b0 );

			// Broadcast a[2,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 2 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_2 = _mm512_set1_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[2,0-15] = a[2,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_2p0 = _mm512_dpbusd_epi32( c_int32_2p0, a_int32_2, b0 );

			// Broadcast a[3,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 3 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_3 = _mm512_set1_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[3,0-15] = a[3,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_3p0 = _mm512_dpbusd_epi32( c_int32_3p0, a_int32_3, b0 );

			// Broadcast a[4,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 4 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_4 = _mm512_set1_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[4,0-15] = a[4,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_4p0 = _mm512_dpbusd_epi32( c_int32_4p0, a_int32_4, b0 );

			// Broadcast a[5,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 5 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_5 = _mm512_set1_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[5,0-15] = a[5,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_5p0 = _mm512_dpbusd_epi32( c_int32_5p0, a_int32_5, b0 );

			// Broadcast a[6,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 6 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_6 = _mm512_set1_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[6,0-15] = a[6,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_6p0 = _mm512_dpbusd_epi32( c_int32_6p0, a_int32_6, b0 );

			// Broadcast a[7,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 7 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_7 = _mm512_set1_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[7,0-15] = a[7,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_7p0 = _mm512_dpbusd_epi32( c_int32_7p0, a_int32_7, b0 );

			// Broadcast a[8,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 8 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_8 = _mm512_set1_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[8,0-15] = a[8,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_8p0 = _mm512_dpbusd_epi32( c_int32_8p0, a_int32_8, b0 );

			// Broadcast a[9,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 9 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_9 = _mm512_set1_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[9,0-15] = a[9,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_9p0 = _mm512_dpbusd_epi32( c_int32_9p0, a_int32_9, b0 );

			// Broadcast a[10,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 10 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_10 = _mm512_set1_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[10,0-15] = a[10,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_10p0 = _mm512_dpbusd_epi32( c_int32_10p0, a_int32_10, b0 );

			// Broadcast a[11,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 11 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_11 = _mm512_set1_epi32( a_kfringe_buf );

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
				MEMCPY_S32_LT16_INIT(n0_rem);

				int8_t* _p0 = ( ( int8_t* )post_ops_attr.buf_downscale ) + \
					( post_ops_attr.rs_c_downscale * \
					( post_ops_attr.post_op_c_i + 0 ) ) + post_ops_attr.post_op_c_j;
				int8_t* _p1 = ( ( int8_t* )post_ops_attr.buf_downscale ) + \
					( post_ops_attr.rs_c_downscale * \
					( post_ops_attr.post_op_c_i + 1 ) ) + post_ops_attr.post_op_c_j;
				int8_t* _p2 = ( ( int8_t* )post_ops_attr.buf_downscale ) + \
					( post_ops_attr.rs_c_downscale * \
					( post_ops_attr.post_op_c_i + 2 ) ) + post_ops_attr.post_op_c_j;
				int8_t* _p3 = ( ( int8_t* )post_ops_attr.buf_downscale ) + \
					( post_ops_attr.rs_c_downscale * \
					( post_ops_attr.post_op_c_i + 3 ) ) + post_ops_attr.post_op_c_j;
				int8_t* _p4 = ( ( int8_t* )post_ops_attr.buf_downscale ) + \
					( post_ops_attr.rs_c_downscale * \
					( post_ops_attr.post_op_c_i + 4 ) ) + post_ops_attr.post_op_c_j;
				int8_t* _p5 = ( ( int8_t* )post_ops_attr.buf_downscale ) + \
					( post_ops_attr.rs_c_downscale * \
					( post_ops_attr.post_op_c_i + 5 ) ) + post_ops_attr.post_op_c_j;

				MEMCPY_S32_LT16_INT8(6,int64_t,int32_t,int16_t,int8_t,buf,_p);

				// c[0,0-15]
				S8_S32_BETA_OP_NLT16F(c_int32_0p0,buf0,selector1,selector2);

				// c[1,0-15]
				S8_S32_BETA_OP_NLT16F(c_int32_1p0,buf1,selector1,selector2);

				// c[2,0-15]
				S8_S32_BETA_OP_NLT16F(c_int32_2p0,buf2,selector1,selector2);

				// c[3,0-15]
				S8_S32_BETA_OP_NLT16F(c_int32_3p0,buf3,selector1,selector2);

				// c[4,0-15]
				S8_S32_BETA_OP_NLT16F(c_int32_4p0,buf4,selector1,selector2);

				// c[5,0-15]
				S8_S32_BETA_OP_NLT16F(c_int32_5p0,buf5,selector1,selector2);

				MEMCPY_S32_LT16_REINIT(n0_rem);

				_p0 = ( ( int8_t* )post_ops_attr.buf_downscale ) + \
					( post_ops_attr.rs_c_downscale * \
					( post_ops_attr.post_op_c_i + 6 ) ) + post_ops_attr.post_op_c_j;
				_p1 = ( ( int8_t* )post_ops_attr.buf_downscale ) + \
					( post_ops_attr.rs_c_downscale * \
					( post_ops_attr.post_op_c_i + 7 ) ) + post_ops_attr.post_op_c_j;
				_p2 = ( ( int8_t* )post_ops_attr.buf_downscale ) + \
					( post_ops_attr.rs_c_downscale * \
					( post_ops_attr.post_op_c_i + 8 ) ) + post_ops_attr.post_op_c_j;
				_p3 = ( ( int8_t* )post_ops_attr.buf_downscale ) + \
					( post_ops_attr.rs_c_downscale * \
					( post_ops_attr.post_op_c_i + 9 ) ) + post_ops_attr.post_op_c_j;
				_p4 = ( ( int8_t* )post_ops_attr.buf_downscale ) + \
					( post_ops_attr.rs_c_downscale * \
					( post_ops_attr.post_op_c_i + 10 ) ) + post_ops_attr.post_op_c_j;
				_p5 = ( ( int8_t* )post_ops_attr.buf_downscale ) + \
					( post_ops_attr.rs_c_downscale * \
					( post_ops_attr.post_op_c_i + 11 ) ) + post_ops_attr.post_op_c_j;

				MEMCPY_S32_LT16_INT8(6,int64_t,int32_t,int16_t,int8_t,buf,_p);

				// c[6,0-15]
				S8_S32_BETA_OP_NLT16F(c_int32_6p0,buf0,selector1,selector2);

				// c[7,0-15]
				S8_S32_BETA_OP_NLT16F(c_int32_7p0,buf1,selector1,selector2);

				// c[8,0-15]
				S8_S32_BETA_OP_NLT16F(c_int32_8p0,buf2,selector1,selector2);

				// c[9,0-15]
				S8_S32_BETA_OP_NLT16F(c_int32_9p0,buf3,selector1,selector2);

				// c[10,0-15]
				S8_S32_BETA_OP_NLT16F(c_int32_10p0,buf4,selector1,selector2);

				// c[11,0-15]
				S8_S32_BETA_OP_NLT16F(c_int32_11p0,buf5,selector1,selector2);
			}
			else
			{
				MEMCPY_S32_LT16_INIT(n0_rem);

				int32_t* _c0 = c + ( rs_c * ( ir + 0 ) ) + ( 0 * 16 );
				int32_t* _c1 = c + ( rs_c * ( ir + 1 ) ) + ( 0 * 16 );
				int32_t* _c2 = c + ( rs_c * ( ir + 2 ) ) + ( 0 * 16 );
				int32_t* _c3 = c + ( rs_c * ( ir + 3 ) ) + ( 0 * 16 );
				int32_t* _c4 = c + ( rs_c * ( ir + 4 ) ) + ( 0 * 16 );
				int32_t* _c5 = c + ( rs_c * ( ir + 5 ) ) + ( 0 * 16 );

				MEMCPY_S32_LT16_INT32(6,int64_t,int32_t,buf,_c);

				// c[0,0-15]
				S32_S32_BETA_OP_NLT16F(c_int32_0p0,buf0,selector1,selector2);

				// c[1,0-15]
				S32_S32_BETA_OP_NLT16F(c_int32_1p0,buf1,selector1,selector2);

				// c[2,0-15]
				S32_S32_BETA_OP_NLT16F(c_int32_2p0,buf2,selector1,selector2);

				// c[3,0-15]
				S32_S32_BETA_OP_NLT16F(c_int32_3p0,buf3,selector1,selector2);

				// c[4,0-15]
				S32_S32_BETA_OP_NLT16F(c_int32_4p0,buf4,selector1,selector2);

				// c[5,0-15]
				S32_S32_BETA_OP_NLT16F(c_int32_5p0,buf5,selector1,selector2);

				MEMCPY_S32_LT16_REINIT(n0_rem);

				_c0 = c + ( rs_c * ( ir + 6 ) ) + ( 0 * 16 );
				_c1 = c + ( rs_c * ( ir + 7 ) ) + ( 0 * 16 );
				_c2 = c + ( rs_c * ( ir + 8 ) ) + ( 0 * 16 );
				_c3 = c + ( rs_c * ( ir + 9 ) ) + ( 0 * 16 );
				_c4 = c + ( rs_c * ( ir + 10 ) ) + ( 0 * 16 );
				_c5 = c + ( rs_c * ( ir + 11 ) ) + ( 0 * 16 );

				MEMCPY_S32_LT16_INT32(6,int64_t,int32_t,buf,_c);

				// c[6,0-15]
				S32_S32_BETA_OP_NLT16F(c_int32_6p0,buf0,selector1,selector2);

				// c[7,0-15]
				S32_S32_BETA_OP_NLT16F(c_int32_7p0,buf1,selector1,selector2);

				// c[8,0-15]
				S32_S32_BETA_OP_NLT16F(c_int32_8p0,buf2,selector1,selector2);

				// c[9,0-15]
				S32_S32_BETA_OP_NLT16F(c_int32_9p0,buf3,selector1,selector2);

				// c[10,0-15]
				S32_S32_BETA_OP_NLT16F(c_int32_10p0,buf4,selector1,selector2);

				// c[11,0-15]
				S32_S32_BETA_OP_NLT16F(c_int32_11p0,buf5,selector1,selector2);
			}
		}

        // Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_12xLT16:
		{
			memcpy( buf0, ( ( int32_t* )post_ops_list_temp->op_args1 +
						post_ops_attr.post_op_c_j ), ( n0_rem * sizeof( int32_t ) ) );
			selector1 = _mm512_loadu_epi32( buf0 );

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

POST_OPS_DOWNSCALE_12xLT16:
		{
			// Typecast without data modification, safe operation.
			float* _buf0 = ( float* )buf0;
			float* _p0 = ( ( float* )post_ops_list_temp->scale_factor + \
							post_ops_attr.post_op_c_j );
			MEMCPY_S32_LT16_INIT(n0_rem);
			MEMCPY_S32_LT16_FLOAT(1,double,float,_buf,_p);

			selector1 = _mm512_loadu_epi32( buf0 );

			// c[0, 0-15]
			CVT_MULRND_CVT32_LT16(c_int32_0p0,selector1);

			// c[1, 0-15]
			CVT_MULRND_CVT32_LT16(c_int32_1p0,selector1);

			// c[2, 0-15]
			CVT_MULRND_CVT32_LT16(c_int32_2p0,selector1);

			// c[3, 0-15]
			CVT_MULRND_CVT32_LT16(c_int32_3p0,selector1);

			// c[4, 0-15]
			CVT_MULRND_CVT32_LT16(c_int32_4p0,selector1);

			// c[5, 0-15]
			CVT_MULRND_CVT32_LT16(c_int32_5p0,selector1);

			// c[6, 0-15]
			CVT_MULRND_CVT32_LT16(c_int32_6p0,selector1);

			// c[7, 0-15]
			CVT_MULRND_CVT32_LT16(c_int32_7p0,selector1);

			// c[8, 0-15]
			CVT_MULRND_CVT32_LT16(c_int32_8p0,selector1);

			// c[9, 0-15]
			CVT_MULRND_CVT32_LT16(c_int32_9p0,selector1);

			// c[10, 0-15]
			CVT_MULRND_CVT32_LT16(c_int32_10p0,selector1);

			// c[11, 0-15]
			CVT_MULRND_CVT32_LT16(c_int32_11p0,selector1);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_12xLT16_DISABLE:
		;

		// Store the results.
		if ( ( post_ops_attr.buf_downscale != NULL ) && ( post_ops_attr.is_last_k == TRUE ) )
		{
			// Need buffers to copy the interim results for all registers
			// before the vzeroupper.
			int32_t buf_tmp0[16];
			int32_t buf_tmp1[16];
			int32_t buf_tmp2[16];
			int32_t buf_tmp3[16];
			int32_t buf_tmp4[16];
			int32_t buf_tmp5[16];

			// Generate a mask16 of all 1's.
			selector1 = _mm512_setzero_epi32();
			selector2 = _mm512_set1_epi32( 10 );
			__mmask16 mask_all1 = _mm512_cmplt_epi32_mask( selector1, selector2 );

			_mm512_mask_cvtsepi32_storeu_epi8( ( int8_t* )buf0, mask_all1, c_int32_0p0 );
			_mm512_mask_cvtsepi32_storeu_epi8( ( int8_t* )buf1, mask_all1, c_int32_1p0 );
			_mm512_mask_cvtsepi32_storeu_epi8( ( int8_t* )buf2, mask_all1, c_int32_2p0 );
			_mm512_mask_cvtsepi32_storeu_epi8( ( int8_t* )buf3, mask_all1, c_int32_3p0 );
			_mm512_mask_cvtsepi32_storeu_epi8( ( int8_t* )buf4, mask_all1, c_int32_4p0 );
			_mm512_mask_cvtsepi32_storeu_epi8( ( int8_t* )buf5, mask_all1, c_int32_5p0 );
			_mm512_mask_cvtsepi32_storeu_epi8( ( int8_t* )buf_tmp0, mask_all1, c_int32_6p0 );
			_mm512_mask_cvtsepi32_storeu_epi8( ( int8_t* )buf_tmp1, mask_all1, c_int32_7p0 );
			_mm512_mask_cvtsepi32_storeu_epi8( ( int8_t* )buf_tmp2, mask_all1, c_int32_8p0 );
			_mm512_mask_cvtsepi32_storeu_epi8( ( int8_t* )buf_tmp3, mask_all1, c_int32_9p0 );
			_mm512_mask_cvtsepi32_storeu_epi8( ( int8_t* )buf_tmp4, mask_all1, c_int32_10p0 );
			_mm512_mask_cvtsepi32_storeu_epi8( ( int8_t* )buf_tmp5, mask_all1, c_int32_11p0 );

			MEMCPY_S32_LT16_INIT(n0_rem);

			_mm256_zeroupper();

			int8_t* _p0 = ( ( int8_t* )post_ops_attr.buf_downscale ) + \
				( post_ops_attr.rs_c_downscale * \
				( post_ops_attr.post_op_c_i + 0 ) ) + post_ops_attr.post_op_c_j;
			int8_t* _p1 = ( ( int8_t* )post_ops_attr.buf_downscale ) + \
				( post_ops_attr.rs_c_downscale * \
				( post_ops_attr.post_op_c_i + 1 ) ) + post_ops_attr.post_op_c_j;
			int8_t* _p2 = ( ( int8_t* )post_ops_attr.buf_downscale ) + \
				( post_ops_attr.rs_c_downscale * \
				( post_ops_attr.post_op_c_i + 2 ) ) + post_ops_attr.post_op_c_j;
			int8_t* _p3 = ( ( int8_t* )post_ops_attr.buf_downscale ) + \
				( post_ops_attr.rs_c_downscale * \
				( post_ops_attr.post_op_c_i + 3 ) ) + post_ops_attr.post_op_c_j;
			int8_t* _p4 = ( ( int8_t* )post_ops_attr.buf_downscale ) + \
				( post_ops_attr.rs_c_downscale * \
				( post_ops_attr.post_op_c_i + 4 ) ) + post_ops_attr.post_op_c_j;
			int8_t* _p5 = ( ( int8_t* )post_ops_attr.buf_downscale ) + \
				( post_ops_attr.rs_c_downscale * \
				( post_ops_attr.post_op_c_i + 5 ) ) + post_ops_attr.post_op_c_j;

			MEMCPY_S32_LT16_INT8(6,int64_t,int32_t,int16_t,int8_t,_p,buf) ;

			MEMCPY_S32_LT16_REINIT(n0_rem);

			_p0 = ( ( int8_t* )post_ops_attr.buf_downscale ) + \
				( post_ops_attr.rs_c_downscale * \
				( post_ops_attr.post_op_c_i + 6 ) ) + post_ops_attr.post_op_c_j;
			_p1 = ( ( int8_t* )post_ops_attr.buf_downscale ) + \
				( post_ops_attr.rs_c_downscale * \
				( post_ops_attr.post_op_c_i + 7 ) ) + post_ops_attr.post_op_c_j;
			_p2 = ( ( int8_t* )post_ops_attr.buf_downscale ) + \
				( post_ops_attr.rs_c_downscale * \
				( post_ops_attr.post_op_c_i + 8 ) ) + post_ops_attr.post_op_c_j;
			_p3 = ( ( int8_t* )post_ops_attr.buf_downscale ) + \
				( post_ops_attr.rs_c_downscale * \
				( post_ops_attr.post_op_c_i + 9 ) ) + post_ops_attr.post_op_c_j;
			_p4 = ( ( int8_t* )post_ops_attr.buf_downscale ) + \
				( post_ops_attr.rs_c_downscale * \
				( post_ops_attr.post_op_c_i + 10 ) ) + post_ops_attr.post_op_c_j;
			_p5 = ( ( int8_t* )post_ops_attr.buf_downscale ) + \
				( post_ops_attr.rs_c_downscale * \
				( post_ops_attr.post_op_c_i + 11 ) ) + post_ops_attr.post_op_c_j;

			MEMCPY_S32_LT16_INT8(6,int64_t,int32_t,int16_t,int8_t,_p,buf_tmp) ;
		}
		else
		{
			// Need buffers to copy the interim results for all registers
			// before the vzeroupper.
			int32_t buf_tmp0[16];
			int32_t buf_tmp1[16];
			int32_t buf_tmp2[16];
			int32_t buf_tmp3[16];
			int32_t buf_tmp4[16];
			int32_t buf_tmp5[16];

			_mm512_storeu_epi32( buf0, c_int32_0p0 );
			_mm512_storeu_epi32( buf1, c_int32_1p0 );
			_mm512_storeu_epi32( buf2, c_int32_2p0 );
			_mm512_storeu_epi32( buf3, c_int32_3p0 );
			_mm512_storeu_epi32( buf4, c_int32_4p0 );
			_mm512_storeu_epi32( buf5, c_int32_5p0 );
			_mm512_storeu_epi32( buf_tmp0, c_int32_6p0 );
			_mm512_storeu_epi32( buf_tmp1, c_int32_7p0 );
			_mm512_storeu_epi32( buf_tmp2, c_int32_8p0 );
			_mm512_storeu_epi32( buf_tmp3, c_int32_9p0 );
			_mm512_storeu_epi32( buf_tmp4, c_int32_10p0 );
			_mm512_storeu_epi32( buf_tmp5, c_int32_11p0 );

			MEMCPY_S32_LT16_INIT(n0_rem);

			_mm256_zeroupper();

			int32_t* _c0 = c + ( rs_c * ( ir + 0 ) );
			int32_t* _c1 = c + ( rs_c * ( ir + 1 ) );
			int32_t* _c2 = c + ( rs_c * ( ir + 2 ) );
			int32_t* _c3 = c + ( rs_c * ( ir + 3 ) );
			int32_t* _c4 = c + ( rs_c * ( ir + 4 ) );
			int32_t* _c5 = c + ( rs_c * ( ir + 5 ) );

			MEMCPY_S32_LT16_INT32(6,int64_t,int32_t,_c,buf);

			MEMCPY_S32_LT16_REINIT(n0_rem);

			_c0 = c + ( rs_c * ( ir + 6 ) );
			_c1 = c + ( rs_c * ( ir + 7 ) );
			_c2 = c + ( rs_c * ( ir + 8 ) );
			_c3 = c + ( rs_c * ( ir + 9 ) );
			_c4 = c + ( rs_c * ( ir + 10 ) );
			_c5 = c + ( rs_c * ( ir + 11 ) );

			MEMCPY_S32_LT16_INT32(6,int64_t,int32_t,_c,buf_tmp);
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
				  &&POST_OPS_DOWNSCALE_12x16
				};
	dim_t MR = 12;
	dim_t m_full_pieces = m0 / MR;
	dim_t m_full_pieces_loop_limit = m_full_pieces * MR;
	dim_t m_partial_pieces = m0 % MR;

	dim_t k_full_pieces = k0 / 4;
	dim_t k_partial_pieces = k0 % 4;

	uint32_t a_kfringe_buf = 0;

	// B matrix storage.
	__m512i b0;

	// A matrix storage.
	__m512i a_int32_0;
	__m512i a_int32_1;
	__m512i a_int32_2;
	__m512i a_int32_3;
	__m512i a_int32_4;
	__m512i a_int32_5;
	__m512i a_int32_6;
	__m512i a_int32_7;
	__m512i a_int32_8;
	__m512i a_int32_9;
	__m512i a_int32_10;
	__m512i a_int32_11;

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
			b0 = _mm512_loadu_epi8( b + ( rs_b * kr ) + ( cs_b * 0 ) );

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
			b0 = _mm512_loadu_epi8( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );

			// Broadcast a[0,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_0 = _mm512_set1_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[0,0-15] = a[0,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_0p0 = _mm512_dpbusd_epi32( c_int32_0p0, a_int32_0, b0 );

			// Broadcast a[1,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 1 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_1 = _mm512_set1_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[1,0-15] = a[1,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_1p0 = _mm512_dpbusd_epi32( c_int32_1p0, a_int32_1, b0 );

			// Broadcast a[2,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 2 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_2 = _mm512_set1_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[2,0-15] = a[2,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_2p0 = _mm512_dpbusd_epi32( c_int32_2p0, a_int32_2, b0 );

			// Broadcast a[3,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 3 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_3 = _mm512_set1_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[3,0-15] = a[3,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_3p0 = _mm512_dpbusd_epi32( c_int32_3p0, a_int32_3, b0 );

			// Broadcast a[4,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 4 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_4 = _mm512_set1_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[4,0-15] = a[4,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_4p0 = _mm512_dpbusd_epi32( c_int32_4p0, a_int32_4, b0 );

			// Broadcast a[5,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 5 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_5 = _mm512_set1_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[5,0-15] = a[5,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_5p0 = _mm512_dpbusd_epi32( c_int32_5p0, a_int32_5, b0 );

			// Broadcast a[6,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 6 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_6 = _mm512_set1_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[6,0-15] = a[6,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_6p0 = _mm512_dpbusd_epi32( c_int32_6p0, a_int32_6, b0 );

			// Broadcast a[7,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 7 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_7 = _mm512_set1_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[7,0-15] = a[7,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_7p0 = _mm512_dpbusd_epi32( c_int32_7p0, a_int32_7, b0 );

			// Broadcast a[8,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 8 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_8 = _mm512_set1_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[8,0-15] = a[8,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_8p0 = _mm512_dpbusd_epi32( c_int32_8p0, a_int32_8, b0 );

			// Broadcast a[9,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 9 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_9 = _mm512_set1_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[9,0-15] = a[9,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_9p0 = _mm512_dpbusd_epi32( c_int32_9p0, a_int32_9, b0 );

			// Broadcast a[10,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 10 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_10 = _mm512_set1_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[10,0-15] = a[10,kr:kr+4]*b[kr:kr+4,0-15]
			c_int32_10p0 = _mm512_dpbusd_epi32( c_int32_10p0, a_int32_10, b0 );

			// Broadcast a[11,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 11 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_11 = _mm512_set1_epi32( a_kfringe_buf );

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
					_mm512_loadu_epi32( ( int32_t* )post_ops_list_temp->op_args1 +
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

POST_OPS_DOWNSCALE_12x16:
	{
		selector1 =
			_mm512_loadu_epi32( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );

		// c[0, 0-15]
		CVT_MULRND_CVT32(c_int32_0p0,selector1);

		// c[1, 0-15]
		CVT_MULRND_CVT32(c_int32_1p0,selector1);

		// c[2, 0-15]
		CVT_MULRND_CVT32(c_int32_2p0,selector1);

		// c[3, 0-15]
		CVT_MULRND_CVT32(c_int32_3p0,selector1);

		// c[4, 0-15]
		CVT_MULRND_CVT32(c_int32_4p0,selector1);

		// c[5, 0-15]
		CVT_MULRND_CVT32(c_int32_5p0,selector1);

		// c[6, 0-15]
		CVT_MULRND_CVT32(c_int32_6p0,selector1);

		// c[7, 0-15]
		CVT_MULRND_CVT32(c_int32_7p0,selector1);

		// c[8, 0-15]
		CVT_MULRND_CVT32(c_int32_8p0,selector1);

		// c[9, 0-15]
		CVT_MULRND_CVT32(c_int32_9p0,selector1);

		// c[10, 0-15]
		CVT_MULRND_CVT32(c_int32_10p0,selector1);

		// c[11, 0-15]
		CVT_MULRND_CVT32(c_int32_11p0,selector1);

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
			_mm512_storeu_epi32( c + ( rs_c * ( ir + 0 ) ) + ( 0*16 ), c_int32_0p0 );

			// c[1,0-15]
			_mm512_storeu_epi32( c + ( rs_c * ( ir + 1 ) ) + ( 0*16 ), c_int32_1p0 );

			// c[2,0-15]
			_mm512_storeu_epi32( c + ( rs_c * ( ir + 2 ) ) + ( 0*16 ), c_int32_2p0 );

			// c[3,0-15]
			_mm512_storeu_epi32( c + ( rs_c * ( ir + 3 ) ) + ( 0*16 ), c_int32_3p0 );

			// c[4,0-15]
			_mm512_storeu_epi32( c + ( rs_c * ( ir + 4 ) ) + ( 0*16 ), c_int32_4p0 );

			// c[5,0-15]
			_mm512_storeu_epi32( c + ( rs_c * ( ir + 5 ) ) + ( 0*16 ), c_int32_5p0 );

			// c[6,0-15]
			_mm512_storeu_epi32( c + ( rs_c * ( ir + 6 ) ) + ( 0*16 ), c_int32_6p0 );

			// c[7,0-15]
			_mm512_storeu_epi32( c + ( rs_c * ( ir + 7 ) ) + ( 0*16 ), c_int32_7p0 );

			// c[8,0-15]
			_mm512_storeu_epi32( c + ( rs_c * ( ir + 8 ) ) + ( 0*16 ), c_int32_8p0 );

			// c[9,0-15]
			_mm512_storeu_epi32( c + ( rs_c * ( ir + 9 ) ) + ( 0*16 ), c_int32_9p0 );

			// c[10,0-15]
			_mm512_storeu_epi32( c + ( rs_c * ( ir + 10 ) ) + ( 0*16 ), c_int32_10p0 );

			// c[11,0-15]
			_mm512_storeu_epi32( c + ( rs_c * ( ir + 11 ) ) + ( 0*16 ), c_int32_11p0 );
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
				  &&POST_OPS_DOWNSCALE_9x32
				};
	dim_t MR = 9;
	dim_t m_full_pieces = m0 / MR;
	dim_t m_full_pieces_loop_limit = m_full_pieces * MR;
	dim_t m_partial_pieces = m0 % MR;

	dim_t k_full_pieces = k0 / 4;
	dim_t k_partial_pieces = k0 % 4;

	uint32_t a_kfringe_buf = 0;

	// B matrix storage.
	__m512i b0;
	__m512i b1;

	// A matrix storage.
	__m512i a_int32_0;
	__m512i a_int32_1;
	__m512i a_int32_2;
	__m512i a_int32_3;

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
			b0 = _mm512_loadu_epi8( b + ( rs_b * kr ) + ( cs_b * 0 ) );
			b1 = _mm512_loadu_epi8( b + ( rs_b * kr ) + ( cs_b * 1 ) );

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
			b0 = _mm512_loadu_epi8( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );
			b1 = _mm512_loadu_epi8( b + ( rs_b * k_full_pieces ) + ( cs_b * 1 ) );

			// Broadcast a[0,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_0 = _mm512_set1_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[0,0-31] = a[0,kr:kr+4]*b[kr:kr+4,0-31]
			c_int32_0p0 = _mm512_dpbusd_epi32( c_int32_0p0, a_int32_0, b0 );
			c_int32_0p1 = _mm512_dpbusd_epi32( c_int32_0p1, a_int32_0, b1 );

			// Broadcast a[1,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 1 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_1 = _mm512_set1_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[1,0-31] = a[1,kr:kr+4]*b[kr:kr+4,0-31]
			c_int32_1p0 = _mm512_dpbusd_epi32( c_int32_1p0, a_int32_1, b0 );
			c_int32_1p1 = _mm512_dpbusd_epi32( c_int32_1p1, a_int32_1, b1 );

			// Broadcast a[2,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 2 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_2 = _mm512_set1_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[2,0-31] = a[2,kr:kr+4]*b[kr:kr+4,0-31]
			c_int32_2p0 = _mm512_dpbusd_epi32( c_int32_2p0, a_int32_2, b0 );
			c_int32_2p1 = _mm512_dpbusd_epi32( c_int32_2p1, a_int32_2, b1 );

			// Broadcast a[3,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 3 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_3 = _mm512_set1_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[3,0-31] = a[3,kr:kr+4]*b[kr:kr+4,0-31]
			c_int32_3p0 = _mm512_dpbusd_epi32( c_int32_3p0, a_int32_3, b0 );
			c_int32_3p1 = _mm512_dpbusd_epi32( c_int32_3p1, a_int32_3, b1 );

			// Broadcast a[4,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 4 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			selector1 = _mm512_set1_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[4,0-31] = a[4,kr:kr+4]*b[kr:kr+4,0-31]
			c_int32_4p0 = _mm512_dpbusd_epi32( c_int32_4p0, selector1, b0 );
			c_int32_4p1 = _mm512_dpbusd_epi32( c_int32_4p1, selector1, b1 );

			// Broadcast a[5,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 5 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			selector2 = _mm512_set1_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[5,0-31] = a[5,kr:kr+4]*b[kr:kr+4,0-31]
			c_int32_5p0 = _mm512_dpbusd_epi32( c_int32_5p0, selector2, b0 );
			c_int32_5p1 = _mm512_dpbusd_epi32( c_int32_5p1, selector2, b1 );

			// Broadcast a[6,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 6 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_0 = _mm512_set1_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[6,0-31] = a[6,kr:kr+4]*b[kr:kr+4,0-31]
			c_int32_6p0 = _mm512_dpbusd_epi32( c_int32_6p0, a_int32_0, b0 );
			c_int32_6p1 = _mm512_dpbusd_epi32( c_int32_6p1, a_int32_0, b1 );

			// Broadcast a[7,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 7 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_1 = _mm512_set1_epi32( a_kfringe_buf );

			// Perform column direction mat-mul with k = 4.
			// c[7,0-31] = a[7,kr:kr+4]*b[kr:kr+4,0-31]
			c_int32_7p0 = _mm512_dpbusd_epi32( c_int32_7p0, a_int32_1, b0 );
			c_int32_7p1 = _mm512_dpbusd_epi32( c_int32_7p1, a_int32_1, b1 );

			// Broadcast a[8,kr:kr+4].
			MEMCPY_S32GM_LT4_UINT8
			(
			  &a_kfringe_buf,
			  ( a + ( rs_a * 8 ) + ( cs_a * k_full_pieces ) ),
			  ( k_partial_pieces )
			);
			a_int32_2 = _mm512_set1_epi32( a_kfringe_buf );

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
					_mm512_loadu_epi32( ( int32_t* )post_ops_list_temp->op_args1 +
									post_ops_attr.post_op_c_j + ( 0 * 16 ) );
			selector2 =
					_mm512_loadu_epi32( ( int32_t* )post_ops_list_temp->op_args1 +
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
POST_OPS_DOWNSCALE_9x32:
	{
		selector1 =
			_mm512_loadu_epi32( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
		selector2 =
			_mm512_loadu_epi32( ( float* )post_ops_list_temp->scale_factor +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );

		// c[0, 0-15]
		CVT_MULRND_CVT32(c_int32_0p0,selector1);

		// c[0, 16-31]
		CVT_MULRND_CVT32(c_int32_0p1,selector2);

		// c[1, 0-15]
		CVT_MULRND_CVT32(c_int32_1p0,selector1);

		// c[1, 16-31]
		CVT_MULRND_CVT32(c_int32_1p1,selector2);

		// c[2, 0-15]
		CVT_MULRND_CVT32(c_int32_2p0,selector1);

		// c[2, 16-31]
		CVT_MULRND_CVT32(c_int32_2p1,selector2);

		// c[3, 0-15]
		CVT_MULRND_CVT32(c_int32_3p0,selector1);

		// c[3, 16-31]
		CVT_MULRND_CVT32(c_int32_3p1,selector2);

		// c[4, 0-15]
		CVT_MULRND_CVT32(c_int32_4p0,selector1);

		// c[4, 16-31]
		CVT_MULRND_CVT32(c_int32_4p1,selector2);

		// c[5, 0-15]
		CVT_MULRND_CVT32(c_int32_5p0,selector1);

		// c[5, 16-31]
		CVT_MULRND_CVT32(c_int32_5p1,selector2);

		// c[6, 0-15]
		CVT_MULRND_CVT32(c_int32_6p0,selector1);

		// c[6, 16-31]
		CVT_MULRND_CVT32(c_int32_6p1,selector2);

		// c[7, 0-15]
		CVT_MULRND_CVT32(c_int32_7p0,selector1);

		// c[7, 16-31]
		CVT_MULRND_CVT32(c_int32_7p1,selector2);

		// c[8, 0-15]
		CVT_MULRND_CVT32(c_int32_8p0,selector1);

		// c[8, 16-31]
		CVT_MULRND_CVT32(c_int32_8p1,selector2);

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
			_mm512_storeu_epi32( c + ( rs_c * ( ir + 0 ) ) + ( 0*16 ), c_int32_0p0 );

			// c[0, 16-31]
			_mm512_storeu_epi32( c + ( rs_c * ( ir + 0 ) ) + ( 1*16 ), c_int32_0p1 );

			// c[1,0-15]
			_mm512_storeu_epi32( c + ( rs_c * ( ir + 1 ) ) + ( 0*16 ), c_int32_1p0 );

			// c[1,16-31]
			_mm512_storeu_epi32( c + ( rs_c * ( ir + 1 ) ) + ( 1*16 ), c_int32_1p1 );

			// c[2,0-15]
			_mm512_storeu_epi32( c + ( rs_c * ( ir + 2 ) ) + ( 0*16 ), c_int32_2p0 );

			// c[2,16-31]
			_mm512_storeu_epi32( c + ( rs_c * ( ir + 2 ) ) + ( 1*16 ), c_int32_2p1 );

			// c[3,0-15]
			_mm512_storeu_epi32( c + ( rs_c * ( ir + 3 ) ) + ( 0*16 ), c_int32_3p0 );

			// c[3,16-31]
			_mm512_storeu_epi32( c + ( rs_c * ( ir + 3 ) ) + ( 1*16 ), c_int32_3p1 );

			// c[4,0-15]
			_mm512_storeu_epi32( c + ( rs_c * ( ir + 4 ) ) + ( 0*16 ), c_int32_4p0 );

			// c[4,16-31]
			_mm512_storeu_epi32( c + ( rs_c * ( ir + 4 ) ) + ( 1*16 ), c_int32_4p1 );

			// c[5,0-15]
			_mm512_storeu_epi32( c + ( rs_c * ( ir + 5 ) ) + ( 0*16 ), c_int32_5p0 );

			// c[5,16-31]
			_mm512_storeu_epi32( c + ( rs_c * ( ir + 5 ) ) + ( 1*16 ), c_int32_5p1 );

			// c[6,0-15]
			_mm512_storeu_epi32( c + ( rs_c * ( ir + 6 ) ) + ( 0*16 ), c_int32_6p0 );

			// c[6, 16-31]
			_mm512_storeu_epi32( c + ( rs_c * ( ir + 6 ) ) + ( 1*16 ), c_int32_6p1 );

			// c[7,0-15]
			_mm512_storeu_epi32( c + ( rs_c * ( ir + 7 ) ) + ( 0*16 ), c_int32_7p0 );

			// c[7,16-31]
			_mm512_storeu_epi32( c + ( rs_c * ( ir + 7 ) ) + ( 1*16 ), c_int32_7p1 );

			// c[8,0-15]
			_mm512_storeu_epi32( c + ( rs_c * ( ir + 8 ) ) + ( 0*16 ), c_int32_8p0 );

			// c[8,16-31]
			_mm512_storeu_epi32( c + ( rs_c * ( ir + 8 ) ) + ( 1*16 ), c_int32_8p1 );
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
