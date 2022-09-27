/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.

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
#include "lpgemm_kernels.h"
#include "lpgemm_f32_kern_macros.h"

#ifdef BLIS_KERNELS_ZEN4
// 5x64 bf16 kernel
LPGEMM_M_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_5x64)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_5x64_DISABLE,
						  &&POST_OPS_BIAS_5x64,
						  &&POST_OPS_RELU_5x64,
						  &&POST_OPS_RELU_SCALE_5x64,
						  &&POST_OPS_DOWNSCALE_5x64
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int32_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;
	__m512bh b1;
	__m512bh b2;
	__m512bh b3;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;
	__m512bh a_bf16_1;
	
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

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 1 ) );
		b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 2 ) );
		b3 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 3 ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-63] = a[0,kr:kr+2]*b[kr:kr+2,0-63]
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

		c_float_2p1 =  _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );
		c_float_2p2 =  _mm512_dpbf16_ps( c_float_2p2, a_bf16_0, b2 );
		c_float_2p3 =  _mm512_dpbf16_ps( c_float_2p3, a_bf16_0, b3 );

		// Perform column direction mat-mul with k = 2.
		// c[3,0-63] = a[3,kr:kr+2]*b[kr:kr+2,0-63]
		c_float_3p0 =  _mm512_dpbf16_ps( c_float_3p0, a_bf16_1, b0 );
		
		// Broadcast a[4,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 4 ) + ( cs_a * kr ) ) );

		c_float_3p1 = _mm512_dpbf16_ps( c_float_3p1, a_bf16_1, b1 );
		c_float_3p2 = _mm512_dpbf16_ps( c_float_3p2, a_bf16_1, b2 );
		c_float_3p3 = _mm512_dpbf16_ps( c_float_3p3, a_bf16_1, b3 );

		// Perform column direction mat-mul with k = 2.
		// c[4,0-63] = a[4,kr:kr+2]*b[kr:kr+2,0-63]
		c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );
		c_float_4p1 = _mm512_dpbf16_ps( c_float_4p1, a_bf16_0, b1 );
		c_float_4p2 = _mm512_dpbf16_ps( c_float_4p2, a_bf16_0, b2 );
		c_float_4p3 = _mm512_dpbf16_ps( c_float_4p3, a_bf16_0, b3 );
	}	
	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+4].
		memcpy
		(
		  &a_kfringe_buf,
		  ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) ),
		  ( k_partial_pieces * sizeof( bfloat16 ) )
		);
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 1 ) );
		b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 2 ) );
		b3 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 3 ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-63] = a[0,kr:kr+2]*b[kr:kr+2,0-63]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		
		// Broadcast a[1,kr:kr+2].
		memcpy
		(
		  &a_kfringe_buf,
		  ( a + ( rs_a * 1 ) + ( cs_a * k_full_pieces ) ),
		  ( k_partial_pieces * sizeof( bfloat16 ) )
		);
		a_bf16_1 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
		c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );
		c_float_0p3 = _mm512_dpbf16_ps( c_float_0p3, a_bf16_0, b3 );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-63] = a[1,kr:kr+2]*b[kr:kr+2,0-63]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_1, b0 );
		
		// Broadcast a[2,kr:kr+2].
		memcpy
		(
		  &a_kfringe_buf,
		  ( a + ( rs_a * 2 ) + ( cs_a * k_full_pieces ) ),
		  ( k_partial_pieces * sizeof( bfloat16 ) )
		);
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_1, b1 );
		c_float_1p2 = _mm512_dpbf16_ps( c_float_1p2, a_bf16_1, b2 );
		c_float_1p3 = _mm512_dpbf16_ps( c_float_1p3, a_bf16_1, b3 );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-63] = a[2,kr:kr+2]*b[kr:kr+2,0-63]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
		
		// Broadcast a[3,kr:kr+4].
		memcpy
		(
		  &a_kfringe_buf,
		  ( a + ( rs_a * 3 ) + ( cs_a * k_full_pieces ) ),
		  ( k_partial_pieces * sizeof( bfloat16 ) )
		);
		a_bf16_1 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );
		c_float_2p2 = _mm512_dpbf16_ps( c_float_2p2, a_bf16_0, b2 );
		c_float_2p3 = _mm512_dpbf16_ps( c_float_2p3, a_bf16_0, b3 );

		// Perform column direction mat-mul with k = 2.
		// c[3,0-63] = a[3,kr:kr+2]*b[kr:kr+2,0-63]
		c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_1, b0 );
		
		// Broadcast a[4,kr:kr+2].
		memcpy
		(
		  &a_kfringe_buf,
		  ( a + ( rs_a * 4 ) + ( cs_a * k_full_pieces ) ),
		  ( k_partial_pieces * sizeof( bfloat16 ) )
		);
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		c_float_3p1 = _mm512_dpbf16_ps( c_float_3p1, a_bf16_1, b1 );
		c_float_3p2 = _mm512_dpbf16_ps( c_float_3p2, a_bf16_1, b2 );
		c_float_3p3 = _mm512_dpbf16_ps( c_float_3p3, a_bf16_1, b3 );

		// Perform column direction mat-mul with k = 2.
		// c[4,0-63] = a[4,kr:kr+2]*b[kr:kr+2,0-63]
		c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );
		c_float_4p1 = _mm512_dpbf16_ps( c_float_4p1, a_bf16_0, b1 );
		c_float_4p2 = _mm512_dpbf16_ps( c_float_4p2, a_bf16_0, b2 );
		c_float_4p3 = _mm512_dpbf16_ps( c_float_4p3, a_bf16_0, b3 );
	}
	
	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

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
	
	c_float_4p0 = _mm512_mul_ps( selector1, c_float_4p0 );
	c_float_4p1 = _mm512_mul_ps( selector1, c_float_4p1 );
	c_float_4p2 = _mm512_mul_ps( selector1, c_float_4p2 );
	c_float_4p3 = _mm512_mul_ps( selector1, c_float_4p3 );

	// Scale C by beta.
	if ( beta != 0 )
	{
		// c[0,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 0 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

		// c[0, 16-31]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 0 ) + ( 1*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_0p1 = _mm512_add_ps( selector1, c_float_0p1 );

		// c[0,32-47]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 0 ) + ( 2*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_0p2 = _mm512_add_ps( selector1, c_float_0p2 );

		// c[0,48-63]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 0 ) + ( 3*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_0p3 = _mm512_add_ps( selector1, c_float_0p3 );

		// c[1,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 1 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

		// c[1,16-31]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 1 ) + ( 1*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p1 = _mm512_add_ps( selector1, c_float_1p1 );

		// c[1,32-47]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 1 ) + ( 2*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p2 = _mm512_add_ps( selector1, c_float_1p2 );

		// c[1,48-63]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 1 ) + ( 3*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p3 = _mm512_add_ps( selector1, c_float_1p3 );

		// c[2,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 2 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );

		// c[2,16-31]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 2 ) + ( 1*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_2p1 = _mm512_add_ps( selector1, c_float_2p1 );

		// c[2,32-47]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 2 ) + ( 2*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_2p2 = _mm512_add_ps( selector1, c_float_2p2 );

		// c[2,48-63]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 2 ) + ( 3*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_2p3 = _mm512_add_ps( selector1, c_float_2p3 );

		// c[3,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 3 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_3p0 = _mm512_add_ps( selector1, c_float_3p0 );

		// c[3,16-31]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 3 ) + ( 1*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_3p1 = _mm512_add_ps( selector1, c_float_3p1 );

		// c[3,32-47]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 3 ) + ( 2*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_3p2 = _mm512_add_ps( selector1, c_float_3p2 );

		// c[3,48-63]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 3 ) + ( 3*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_3p3 = _mm512_add_ps( selector1, c_float_3p3 );

		// c[4,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 4 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_4p0 = _mm512_add_ps( selector1, c_float_4p0 );

		// c[4,16-31]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 4 ) + ( 1*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_4p1 = _mm512_add_ps( selector1, c_float_4p1 );

		// c[4,32-47]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 4 ) + ( 2*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_4p2 = _mm512_add_ps( selector1, c_float_4p2 );

		// c[4,48-63]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 4 ) + ( 3*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_4p3 = _mm512_add_ps( selector1, c_float_4p3 );
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_5x64:
	{
		__m512 selector3;
		__m512 selector4;

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			selector1 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j );
			selector2 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 1 * 16 ) );
			selector3 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 2 * 16 ) );
			selector4 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 3 * 16 ) );

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
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_i + 0 ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_i + 1 ) );
			selector3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_i + 2 ) );
			selector4 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_i + 3 ) );
			__m512 selector5 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_i + 4 ) );

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
POST_OPS_RELU_5x64:
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
POST_OPS_RELU_SCALE_5x64:
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
POST_OPS_DOWNSCALE_5x64:
	{
		// c[0, 0-15]
		CVT_F32_BF16(c_float_0p0,0,0);

		// c[0, 16-31]
		CVT_F32_BF16(c_float_0p1,0,1);

		// c[0, 32-47]
		CVT_F32_BF16(c_float_0p2,0,2);

		// c[0, 48-63]
		CVT_F32_BF16(c_float_0p3,0,3);

		// c[1, 0-15]
		CVT_F32_BF16(c_float_1p0,1,0);

		// c[1, 16-31]
		CVT_F32_BF16(c_float_1p1,1,1);

		// c[1, 32-47]
		CVT_F32_BF16(c_float_1p2,1,2);

		// c[1, 48-63]
		CVT_F32_BF16(c_float_1p3,1,3);

		// c[2, 0-15]
		CVT_F32_BF16(c_float_2p0,2,0);

		// c[2, 16-31]
		CVT_F32_BF16(c_float_2p1,2,1);

		// c[2, 32-47]
		CVT_F32_BF16(c_float_2p2,2,2);

		// c[2, 48-63]
		CVT_F32_BF16(c_float_2p3,2,3);

		// c[3, 0-15]
		CVT_F32_BF16(c_float_3p0,3,0);

		// c[3, 16-31]
		CVT_F32_BF16(c_float_3p1,3,1);

		// c[3, 32-47]
		CVT_F32_BF16(c_float_3p2,3,2);

		// c[3, 48-63]
		CVT_F32_BF16(c_float_3p3,3,3);

		// c[4, 0-15]
		CVT_F32_BF16(c_float_4p0,4,0);

		// c[4, 16-31]
		CVT_F32_BF16(c_float_4p1,4,1);

		// c[4, 32-47]
		CVT_F32_BF16(c_float_4p2,4,2);

		// c[4, 48-63]
		CVT_F32_BF16(c_float_4p3,4,3);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}		
POST_OPS_5x64_DISABLE:
	;
	
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

	// c[4,0-15]
	_mm512_storeu_ps( c + ( rs_c * 4 ) + ( 0*16 ), c_float_4p0 );

	// c[4,16-31]
	_mm512_storeu_ps( c + ( rs_c * 4 ) + ( 1*16 ), c_float_4p1 );

	// c[4,32-47]
	_mm512_storeu_ps( c + ( rs_c * 4 ) + ( 2*16 ), c_float_4p2 );

	// c[4,48-63]
	_mm512_storeu_ps( c + ( rs_c * 4 ) + ( 3*16 ), c_float_4p3 );
}

// 4x64 bf16 kernel
LPGEMM_M_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_4x64)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_4x64_DISABLE,
						  &&POST_OPS_BIAS_4x64,
						  &&POST_OPS_RELU_4x64,
						  &&POST_OPS_RELU_SCALE_4x64,
						  &&POST_OPS_DOWNSCALE_4x64
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int32_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;
	__m512bh b1;
	__m512bh b2;
	__m512bh b3;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;
	__m512bh a_bf16_1;

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

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 1 ) );
		b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 2 ) );
		b3 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 3 ) );

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
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		memcpy
		(
		  &a_kfringe_buf,
		  ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) ),
		  ( k_partial_pieces * sizeof( bfloat16) )
		);
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 1 ) );
		b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 2 ) );
		b3 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 3 ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-63] = a[0,kr:kr+2]*b[kr:kr+2,0-63]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		
		// Broadcast a[1,kr:kr+2].
		memcpy
		(
		  &a_kfringe_buf,
		  ( a + ( rs_a * 1 ) + ( cs_a * k_full_pieces ) ),
		  ( k_partial_pieces * sizeof( bfloat16 ) )
		);
		a_bf16_1 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
		c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );
		c_float_0p3 = _mm512_dpbf16_ps( c_float_0p3, a_bf16_0, b3 );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-63] = a[1,kr:kr+2]*b[kr:kr+2,0-63]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_1, b0 );
		
		// Broadcast a[2,kr:kr+2].
		memcpy
		(
		  &a_kfringe_buf,
		  ( a + ( rs_a * 2 ) + ( cs_a * k_full_pieces ) ),
		  ( k_partial_pieces * sizeof( bfloat16 ) )
		);
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_1, b1 );
		c_float_1p2 = _mm512_dpbf16_ps( c_float_1p2, a_bf16_1, b2 );
		c_float_1p3 = _mm512_dpbf16_ps( c_float_1p3, a_bf16_1, b3 );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-63] = a[2,kr:kr+2]*b[kr:kr+2,0-63]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
		
		// Broadcast a[3,kr:kr+2].
		memcpy
		(
		  &a_kfringe_buf,
		  ( a + ( rs_a * 3 ) + ( cs_a * k_full_pieces ) ),
		  ( k_partial_pieces * sizeof( bfloat16 ) )
		);
		a_bf16_1 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );
		
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

	// Scale C by beta.
	if ( beta != 0 )
	{
		// c[0,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 0 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

		// c[0, 16-31]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 0 ) + ( 1*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_0p1 = _mm512_add_ps( selector1, c_float_0p1 );

		// c[0,32-47]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 0 ) + ( 2*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_0p2 = _mm512_add_ps( selector1, c_float_0p2 );

		// c[0,48-63]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 0 ) + ( 3*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_0p3 = _mm512_add_ps( selector1, c_float_0p3 );

		// c[1,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 1 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

		// c[1,16-31]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 1 ) + ( 1*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p1 = _mm512_add_ps( selector1, c_float_1p1 );

		// c[1,32-47]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 1 ) + ( 2*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p2 = _mm512_add_ps( selector1, c_float_1p2 );

		// c[1,48-63]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 1 ) + ( 3*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p3 = _mm512_add_ps( selector1, c_float_1p3 );

		// c[2,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 2 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );

		// c[2,16-31]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 2 ) + ( 1*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_2p1 = _mm512_add_ps( selector1, c_float_2p1 );

		// c[2,32-47]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 2 ) + ( 2*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_2p2 = _mm512_add_ps( selector1, c_float_2p2 );

		// c[2,48-63]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 2 ) + ( 3*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_2p3 = _mm512_add_ps( selector1, c_float_2p3 );

		// c[3,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 3 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_3p0 = _mm512_add_ps( selector1, c_float_3p0 );

		// c[3,16-31]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 3 ) + ( 1*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_3p1 = _mm512_add_ps( selector1, c_float_3p1 );

		// c[3,32-47]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 3 ) + ( 2*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_3p2 = _mm512_add_ps( selector1, c_float_3p2 );

		// c[3,48-63]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 3 ) + ( 3*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_3p3 = _mm512_add_ps( selector1, c_float_3p3 );
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
			selector1 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j );
			selector2 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 1 * 16 ) );
			selector3 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 2 * 16 ) );
			selector4 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 3 * 16 ) );

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
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_i + 0 ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_i + 1 ) );
			selector3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_i + 2 ) );
			selector4 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_i + 3 ) );

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
POST_OPS_DOWNSCALE_4x64:
	{
		// c[0, 0-15]
		CVT_F32_BF16(c_float_0p0,0,0);

		// c[0, 16-31]
		CVT_F32_BF16(c_float_0p1,0,1);

		// c[0, 32-47]
		CVT_F32_BF16(c_float_0p2,0,2);

		// c[0, 48-63]
		CVT_F32_BF16(c_float_0p3,0,3);

		// c[1, 0-15]
		CVT_F32_BF16(c_float_1p0,1,0);

		// c[1, 16-31]
		CVT_F32_BF16(c_float_1p1,1,1);

		// c[1, 32-47]
		CVT_F32_BF16(c_float_1p2,1,2);

		// c[1, 48-63]
		CVT_F32_BF16(c_float_1p3,1,3);

		// c[2, 0-15]
		CVT_F32_BF16(c_float_2p0,2,0);

		// c[2, 16-31]
		CVT_F32_BF16(c_float_2p1,2,1);

		// c[2, 32-47]
		CVT_F32_BF16(c_float_2p2,2,2);

		// c[2, 48-63]
		CVT_F32_BF16(c_float_2p3,2,3);

		// c[3, 0-15]
		CVT_F32_BF16(c_float_3p0,3,0);

		// c[3, 16-31]
		CVT_F32_BF16(c_float_3p1,3,1);

		// c[3, 32-47]
		CVT_F32_BF16(c_float_3p2,3,2);

		// c[3, 48-63]
		CVT_F32_BF16(c_float_3p3,3,3);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}

POST_OPS_4x64_DISABLE:
	;
	
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

// 3x64 bf16 kernel
LPGEMM_M_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_3x64)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_3x64_DISABLE,
						  &&POST_OPS_BIAS_3x64,
						  &&POST_OPS_RELU_3x64,
						  &&POST_OPS_RELU_SCALE_3x64,
						  &&POST_OPS_DOWNSCALE_3x64
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int32_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;
	__m512bh b1;
	__m512bh b2;
	__m512bh b3;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;
	__m512bh a_bf16_1;

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

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a *  0 ) + ( cs_a * kr ) ) );

		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 1 ) );
		b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 2 ) );
		b3 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 3 ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-63] = a[0,kr:kr+2]*b[kr:kr+2,0-63]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		
		// Broadcast a[1,kr:kr+2].
		a_bf16_1 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 1 ) + ( cs_a * kr ) ) );

		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
		c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );
		c_float_0p3 = _mm512_dpbf16_ps( c_float_0p3, a_bf16_0, b3 );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-63] = a[1,kr:kr+2]*b[kr:kr+2,0-63]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_1, b0 );
		
		// Broadcast a[2,kr:kr+4].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 2 ) + ( cs_a * kr ) ) );

		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_1, b1 );
		c_float_1p2 = _mm512_dpbf16_ps( c_float_1p2, a_bf16_1, b2 );
		c_float_1p3 = _mm512_dpbf16_ps( c_float_1p3, a_bf16_1, b3 );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-63] = a[2,kr:kr+2]*b[kr:kr+2,0-63]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
		c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );
		c_float_2p2 = _mm512_dpbf16_ps( c_float_2p2, a_bf16_0, b2 );
		c_float_2p3 = _mm512_dpbf16_ps( c_float_2p3, a_bf16_0, b3 );
	}
	
	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		memcpy
		(
		  &a_kfringe_buf,
		  ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) ),
		  ( k_partial_pieces * sizeof( bfloat16 ) )
		);
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 1 ) );
		b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 2 ) );
		b3 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 3 ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-63] = a[0,kr:kr+2]*b[kr:kr+2,0-63]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		
		// Broadcast a[1,kr:kr+2].
		memcpy
		(
		  &a_kfringe_buf,
		  ( a + ( rs_a * 1 ) + ( cs_a * k_full_pieces ) ),
		  ( k_partial_pieces * sizeof( bfloat16 ) )
		);
		a_bf16_1 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
		c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );
		c_float_0p3 = _mm512_dpbf16_ps( c_float_0p3, a_bf16_0, b3 );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-63] = a[1,kr:kr+2]*b[kr:kr+2,0-63]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_1, b0 );
		
		// Broadcast a[2,kr:kr+2].
		memcpy
		(
		  &a_kfringe_buf,
		  ( a + ( rs_a * 2 ) + ( cs_a * k_full_pieces ) ),
		  ( k_partial_pieces * sizeof( bfloat16 ) )
		);
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_1, b1 );
		c_float_1p2 = _mm512_dpbf16_ps( c_float_1p2, a_bf16_1, b2 );
		c_float_1p3 = _mm512_dpbf16_ps( c_float_1p3, a_bf16_1, b3 );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-63] = a[2,kr:kr+2]*b[kr:kr+2,0-63]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
		c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );
		c_float_2p2 = _mm512_dpbf16_ps( c_float_2p2, a_bf16_0, b2 );
		c_float_2p3 = _mm512_dpbf16_ps( c_float_2p3, a_bf16_0, b3 );
	}
    
	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

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

	// Scale C by beta.
	if ( beta != 0 )
	{
		// c[0,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 0 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

		// c[0, 16-31]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 0 ) + ( 1*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_0p1 = _mm512_add_ps( selector1, c_float_0p1 );

		// c[0,32-47]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 0 ) + ( 2*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_0p2 = _mm512_add_ps( selector1, c_float_0p2 );

		// c[0,48-63]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 0 ) + ( 3*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_0p3 = _mm512_add_ps( selector1, c_float_0p3 );

		// c[1,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 1 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

		// c[1,16-31]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 1 ) + ( 1*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p1 = _mm512_add_ps( selector1, c_float_1p1 );

		// c[1,32-47]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 1 ) + ( 2*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p2 = _mm512_add_ps( selector1, c_float_1p2 );

		// c[1,48-63]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 1 ) + ( 3*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p3 = _mm512_add_ps( selector1, c_float_1p3 );

		// c[2,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 2 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );

		// c[2,16-31]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 2 ) + ( 1*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_2p1 = _mm512_add_ps( selector1, c_float_2p1 );

		// c[2,32-47]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 2 ) + ( 2*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_2p2 = _mm512_add_ps( selector1, c_float_2p2 );

		// c[2,48-63]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 2 ) + ( 3*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_2p3 = _mm512_add_ps( selector1, c_float_2p3 );
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_3x64:
	{
		__m512 selector3;
		__m512 selector4;

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			selector1 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j );
			selector2 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 1 * 16 ) );
			selector3 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 2 * 16 ) );
			selector4 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 3 * 16 ) );

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
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_i + 0 ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_i + 1 ) );
			selector3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_i + 2 ) );

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
POST_OPS_RELU_3x64:
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
POST_OPS_RELU_SCALE_3x64:
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
POST_OPS_DOWNSCALE_3x64:
	{
		// c[0, 0-15]
		CVT_F32_BF16(c_float_0p0,0,0);

		// c[0, 16-31]
		CVT_F32_BF16(c_float_0p1,0,1);

		// c[0, 32-47]
		CVT_F32_BF16(c_float_0p2,0,2);

		// c[0, 48-63]
		CVT_F32_BF16(c_float_0p3,0,3);

		// c[1, 0-15]
		CVT_F32_BF16(c_float_1p0,1,0);

		// c[1, 16-31]
		CVT_F32_BF16(c_float_1p1,1,1);

		// c[1, 32-47]
		CVT_F32_BF16(c_float_1p2,1,2);

		// c[1, 48-63]
		CVT_F32_BF16(c_float_1p3,1,3);

		// c[2, 0-15]
		CVT_F32_BF16(c_float_2p0,2,0);

		// c[2, 16-31]
		CVT_F32_BF16(c_float_2p1,2,1);

		// c[2, 32-47]
		CVT_F32_BF16(c_float_2p2,2,2);

		// c[2, 48-63]
		CVT_F32_BF16(c_float_2p3,2,3);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}	
POST_OPS_3x64_DISABLE:
	;
	
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
}

// 2x64 bf16 kernel
LPGEMM_M_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_2x64)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_2x64_DISABLE,
						  &&POST_OPS_BIAS_2x64,
						  &&POST_OPS_RELU_2x64,
						  &&POST_OPS_RELU_SCALE_2x64,
						  &&POST_OPS_DOWNSCALE_2x64
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int32_t a_kfringe_buf = 0;
	// B matrix storage bfloat type
	__m512bh b0;
	__m512bh b1;
	__m512bh b2;
	__m512bh b3;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;
	__m512bh a_bf16_1;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();
	__m512 c_float_0p1 = _mm512_setzero_ps();
	__m512 c_float_0p2 = _mm512_setzero_ps();
	__m512 c_float_0p3 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();
	__m512 c_float_1p1 = _mm512_setzero_ps();
	__m512 c_float_1p2 = _mm512_setzero_ps();
	__m512 c_float_1p3 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 1 ) );
		b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 2 ) );
		b3 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 3 ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-63] = a[0,kr:kr+2]*b[kr:kr+2,0-63]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		
		// Broadcast a[1,kr:kr+2].
		a_bf16_1 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 1 ) + ( cs_a * kr ) ) );
		
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
		c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );
		c_float_0p3 = _mm512_dpbf16_ps( c_float_0p3, a_bf16_0, b3 );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-63] = a[1,kr:kr+2]*b[kr:kr+2,0-63]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_1, b0 );
		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_1, b1 );
		c_float_1p2 = _mm512_dpbf16_ps( c_float_1p2, a_bf16_1, b2 );
		c_float_1p3 = _mm512_dpbf16_ps( c_float_1p3, a_bf16_1, b3 );
	}
	
	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		memcpy
		(
		  &a_kfringe_buf,
		  ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) ),
		  ( k_partial_pieces * sizeof( bfloat16 ) )
		);
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 1 ) );
		b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 2 ) );
		b3 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 3 ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-63] = a[0,kr:kr+2]*b[kr:kr+2,0-63]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		
		// Broadcast a[1,kr:kr+2].
		memcpy
		(
		  &a_kfringe_buf,
		  ( a + ( rs_a * 1 ) + ( cs_a * k_full_pieces ) ),
		  ( k_partial_pieces * sizeof( bfloat16 ) )
		);
		a_bf16_1 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
		c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );
		c_float_0p3 = _mm512_dpbf16_ps( c_float_0p3, a_bf16_0, b3 );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-63] = a[1,kr:kr+2]*b[kr:kr+2,0-63]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_1, b0 );
		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_1, b1 );
		c_float_1p2 = _mm512_dpbf16_ps( c_float_1p2, a_bf16_1, b2 );
		c_float_1p3 = _mm512_dpbf16_ps( c_float_1p3, a_bf16_1, b3 );
	}
	

	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	// Scale by alpha
	c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );
	c_float_0p1 = _mm512_mul_ps( selector1, c_float_0p1 );
	c_float_0p2 = _mm512_mul_ps( selector1, c_float_0p2 );
	c_float_0p3 = _mm512_mul_ps( selector1, c_float_0p3 );

	c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );
	c_float_1p1 = _mm512_mul_ps( selector1, c_float_1p1 );
	c_float_1p2 = _mm512_mul_ps( selector1, c_float_1p2 );
	c_float_1p3 = _mm512_mul_ps( selector1, c_float_1p3 );

	// Scale C by beta.
	if ( beta != 0 )
	{
		// c[0,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 0 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

		// c[0, 16-31]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 0 ) + ( 1*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_0p1 = _mm512_add_ps( selector1, c_float_0p1 );

		// c[0,32-47]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 0 ) + ( 2*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_0p2 = _mm512_add_ps( selector1, c_float_0p2 );

		// c[0,48-63]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 0 ) + ( 3*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_0p3 = _mm512_add_ps( selector1, c_float_0p3 );

		// c[1,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 1 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

		// c[1,16-31]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 1 ) + ( 1*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p1 = _mm512_add_ps( selector1, c_float_1p1 );

		// c[1,32-47]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 1 ) + ( 2*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p2 = _mm512_add_ps( selector1, c_float_1p2 );

		// c[1,48-63]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 1 ) + ( 3*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p3 = _mm512_add_ps( selector1, c_float_1p3 );
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_2x64:
	{
		__m512 selector3;
		__m512 selector4;

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			selector1 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j );
			selector2 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 1 * 16 ) );
			selector3 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 2 * 16 ) );
			selector4 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 3 * 16 ) );

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
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_i + 0 ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_i + 1 ) );

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
POST_OPS_RELU_2x64:
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
POST_OPS_RELU_SCALE_2x64:
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
	POST_OPS_DOWNSCALE_2x64:
	{
		// c[0, 0-15]
		CVT_F32_BF16(c_float_0p0,0,0);

		// c[0, 16-31]
		CVT_F32_BF16(c_float_0p1,0,1);

		// c[0, 32-47]
		CVT_F32_BF16(c_float_0p2,0,2);

		// c[0, 48-63]
		CVT_F32_BF16(c_float_0p3,0,3);

		// c[1, 0-15]
		CVT_F32_BF16(c_float_1p0,1,0);

		// c[1, 16-31]
		CVT_F32_BF16(c_float_1p1,1,1);

		// c[1, 32-47]
		CVT_F32_BF16(c_float_1p2,1,2);

		// c[1, 48-63]
		CVT_F32_BF16(c_float_1p3,1,3);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_2x64_DISABLE:
	;
	
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
}

// 1x64 bf16 kernel
LPGEMM_M_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_1x64)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_1x64_DISABLE,
						  &&POST_OPS_BIAS_1x64,
						  &&POST_OPS_RELU_1x64,
						  &&POST_OPS_RELU_SCALE_1x64,
						  &&POST_OPS_DOWNSCALE_1x64
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int32_t a_kfringe_buf = 0;

	//  Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();
	__m512 c_float_0p1 = _mm512_setzero_ps();
	__m512 c_float_0p2 = _mm512_setzero_ps();
	__m512 c_float_0p3 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		__m512bh b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr]
		__m512bh a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

		__m512bh b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 1 ) );
		__m512bh b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 2 ) );
		__m512bh b3 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 3 ) );

		// Perform column direction mat-mul with k = 2.
        // c[0,0-63] = a[0,kr:kr+2]*b[kr:kr+2,0-63]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
		c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );
		c_float_0p3 = _mm512_dpbf16_ps( c_float_0p3, a_bf16_0, b3 );
	}
	
	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		__m512bh b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		memcpy
		(
		  &a_kfringe_buf,
		  ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) ),
		  ( k_partial_pieces * sizeof( bfloat16 ) )
		);
		__m512bh a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		__m512bh b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 1 ) );
		__m512bh b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 2 ) );
		__m512bh b3 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 3 ) );

		// Perform column direction mat-mul with k = 2.
        // c[0,0-63] = a[0,kr:kr+2]*b[kr:kr+2,0-63]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
		c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );
		c_float_0p3 = _mm512_dpbf16_ps( c_float_0p3, a_bf16_0, b3 );
	}
	
	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	// Scale by alpha
	c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );
	c_float_0p1 = _mm512_mul_ps( selector1, c_float_0p1 );
	c_float_0p2 = _mm512_mul_ps( selector1, c_float_0p2 );
	c_float_0p3 = _mm512_mul_ps( selector1, c_float_0p3 );
	
	// Scale C by beta.
	if ( beta != 0)
	{
		// c[0,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 0 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

		// c[0, 16-31]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 0 ) + ( 1*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_0p1 = _mm512_add_ps( selector1, c_float_0p1 );

		// c[0,32-47]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 0 ) + ( 2*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_0p2 = _mm512_add_ps( selector1, c_float_0p2 );

		// c[0,48-63]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 0 ) + ( 3*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_0p3 = _mm512_add_ps( selector1, c_float_0p3 );
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_1x64:
	{
		__m512 selector3;
		__m512 selector4;

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			selector1 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j );
			selector2 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 1 * 16 ) );
			selector3 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 2 * 16 ) );
			selector4 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 3 * 16 ) );

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
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_i + 0 ) );

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
POST_OPS_RELU_1x64:
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
POST_OPS_RELU_SCALE_1x64:
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
POST_OPS_DOWNSCALE_1x64:
	{
		// c[0, 0-15]
		CVT_F32_BF16(c_float_0p0,0,0);

		// c[0, 16-31]
		CVT_F32_BF16(c_float_0p1,0,1);

		// c[0, 32-47]
		CVT_F32_BF16(c_float_0p2,0,2);

		// c[0, 48-63]
		CVT_F32_BF16(c_float_0p3,0,3);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}	
POST_OPS_1x64_DISABLE:
	;
    
	// Store the accumulated results.
	// c[0,0-15]
	_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 0*16 ), c_float_0p0 );

	// c[0, 16-31]
	_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 1*16 ), c_float_0p1 );

	// c[0,32-47]
	_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 2*16 ), c_float_0p2 );

	// c[0,48-63]
	_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 3*16 ), c_float_0p3 );
}
#endif
