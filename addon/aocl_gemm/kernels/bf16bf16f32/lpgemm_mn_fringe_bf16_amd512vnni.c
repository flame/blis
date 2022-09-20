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

// 5xlt16 bf16 fringe kernel
LPGEMM_MN_LT_NR0_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_5xlt16)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_5xLT16_DISABLE,
						  &&POST_OPS_BIAS_5xLT16,
						  &&POST_OPS_RELU_5xLT16,
						  &&POST_OPS_RELU_SCALE_5xLT16,
						  &&POST_OPS_DOWNSCALE_5xLT16
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int32_t a_kfringe_buf = 0;

    // B matrix storage bfloat type
	__m512bh b0;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// For corner cases.
	float buf0[16];
	float buf1[16];
	float buf2[16];
	float buf3[16];
	float buf4[16];

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();

	__m512 c_float_2p0 = _mm512_setzero_ps();
		
	__m512 c_float_3p0 = _mm512_setzero_ps();

	__m512 c_float_4p0 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );

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
			
		// Broadcast a[4,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 4 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[4,0-15] = a[4,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );
	}
	// Handle k remainder.        
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
			
		// Broadcast a[1,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 1 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
			
		// Broadcast a[2,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 2 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-15] = a[2,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
			
		// Broadcast a[3,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 3 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[3,0-15] = a[3,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );
			
		// Broadcast a[4,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 4 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[4,0-15] = a[4,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );
	}
        
	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	// Scale by alpha
	c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );

	c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );
	
	c_float_2p0 = _mm512_mul_ps( selector1, c_float_2p0 );
	
	c_float_3p0 = _mm512_mul_ps( selector1, c_float_3p0 );
	
	c_float_4p0 = _mm512_mul_ps( selector1, c_float_4p0 );
		
	// Scale C by beta.
	if ( beta != 0 )
	{
		memcpy( buf0, ( c + ( rs_c * 0 ) ), ( n0_rem * sizeof( float ) ) );
		memcpy( buf1, ( c + ( rs_c * 1 ) ), ( n0_rem * sizeof( float ) ) );
		memcpy( buf2, ( c + ( rs_c * 2 ) ), ( n0_rem * sizeof( float ) ) );
		memcpy( buf3, ( c + ( rs_c * 3 ) ), ( n0_rem * sizeof( float ) ) );
		memcpy( buf4, ( c + ( rs_c * 4 ) ), ( n0_rem * sizeof( float ) ) );
		
		// c[0,0-15]
		selector1 = _mm512_loadu_ps( buf0 );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

		// c[1,0-15]
		selector1 = _mm512_loadu_ps( buf1 );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

		// c[2,0-15]
		selector1 = _mm512_loadu_ps( buf2 );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );

		// c[3,0-15]
		selector1 = _mm512_loadu_ps( buf3 );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_3p0 = _mm512_add_ps( selector1, c_float_3p0 );

		// c[4,0-15]
		selector1 = _mm512_loadu_ps( buf4 );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_4p0 = _mm512_add_ps( selector1, c_float_4p0 );
	}
	// Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_5xLT16:
		{
			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
				 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
				memcpy( buf0, ( ( float* )post_ops_list_temp->op_args1 +
						post_op_c_j ), ( n0_rem * sizeof( float ) ) );
				selector1 = _mm512_loadu_ps( buf0 );

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

				// c[2,0-15]
				c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );

				// c[3,0-15]
				c_float_3p0 = _mm512_add_ps( selector1, c_float_3p0 );

				// c[4,0-15]
				c_float_4p0 = _mm512_add_ps( selector1, c_float_4p0 );
			}
			else
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_op_c_i + 1 ) );
				__m512 selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_op_c_i + 2 ) );
				__m512 selector4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_op_c_i + 3 ) );
				a_bf16_0 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_op_c_i + 4 ) );

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );

				// c[2,0-15]
				c_float_2p0 = _mm512_add_ps( selector3, c_float_2p0 );

				// c[3,0-15]
				c_float_3p0 = _mm512_add_ps( selector4, c_float_3p0 );

				// c[4,0-15]
				c_float_4p0 = _mm512_add_ps( a_bf16_0, c_float_4p0 );
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_5xLT16:
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

			// c[4,0-15]
			c_float_4p0 = _mm512_max_ps( selector1, c_float_4p0 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_5xLT16:
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

			// c[4, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_4p0)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_DOWNSCALE_5xLT16:
	{
		// c[0, 0-15]
		CVT_F32_BF16_LT16(c_float_0p0,0,0);

		// c[1, 0-15]
		CVT_F32_BF16_LT16(c_float_1p0,1,0);

		// c[2, 0-15]
		CVT_F32_BF16_LT16(c_float_2p0,2,0);

		// c[3, 0-15]
		CVT_F32_BF16_LT16(c_float_3p0,3,0);

		// c[4, 0-15]
		CVT_F32_BF16_LT16(c_float_4p0,4,0);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}		
POST_OPS_5xLT16_DISABLE:
		;
		
	// Store the results.
	// c[0,0-15]
	_mm512_storeu_ps( buf0, c_float_0p0 );

	// c[1,0-15]
	_mm512_storeu_ps( buf1, c_float_1p0 );

	// c[2,0-15]
	_mm512_storeu_ps( buf2, c_float_2p0 );

	// c[3,0-15]
	_mm512_storeu_ps( buf3, c_float_3p0 );

	// c[4,0-15]
	_mm512_storeu_ps( buf4, c_float_4p0 );

	// Memcpy partial parts.
	// c[0,0-15]
	memcpy( c + ( rs_c * 0 ) + ( 0*16 ), buf0, ( n0_rem * sizeof( float ) ) );

	// c[1,0-15]
	memcpy( c + ( rs_c * 1 ) + ( 0*16 ), buf1, ( n0_rem * sizeof( float ) ) );

	// c[2,0-15]
	memcpy( c + ( rs_c * 2 ) + ( 0*16 ), buf2, ( n0_rem * sizeof( float ) ) );

	// c[3,0-15]
	memcpy( c + ( rs_c * 3 ) + ( 0*16 ), buf3, ( n0_rem * sizeof( float ) ) );

	// c[4,0-15]
	memcpy( c + ( rs_c * 4 ) + ( 0*16 ), buf4, ( n0_rem * sizeof( float ) ) );
	
}

// 4xlt16 bf16 fringe kernel
LPGEMM_MN_LT_NR0_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_4xlt16)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_4xLT16_DISABLE,
						  &&POST_OPS_BIAS_4xLT16,
						  &&POST_OPS_RELU_4xLT16,
						  &&POST_OPS_RELU_SCALE_4xLT16,
						  &&POST_OPS_DOWNSCALE_4xLT16
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int32_t a_kfringe_buf = 0;

    // B matrix storage bfloat type
	__m512bh b0;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// For corner cases.
	float buf0[16];
	float buf1[16];
	float buf2[16];
	float buf3[16];

	
	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();

	__m512 c_float_2p0 = _mm512_setzero_ps();
	
	__m512 c_float_3p0 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );

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
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		
		// Broadcast a[1,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 1 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
		
		// Broadcast a[2,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 2 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-15] = a[2,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
		
		// Broadcast a[3,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 3 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[3,0-15] = a[3,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );
	}
	
	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	// Scale by alpha
	c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );

	c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );
	
	c_float_2p0 = _mm512_mul_ps( selector1, c_float_2p0 );
	
	c_float_3p0 = _mm512_mul_ps( selector1, c_float_3p0 );
		
	// Scale C by beta.
	if ( beta != 0 )
	{
		memcpy( buf0, ( c + ( rs_c * 0 ) ), ( n0_rem * sizeof( float ) ) );
		memcpy( buf1, ( c + ( rs_c * 1 ) ), ( n0_rem * sizeof( float ) ) );
		memcpy( buf2, ( c + ( rs_c * 2 ) ), ( n0_rem * sizeof( float ) ) );
		memcpy( buf3, ( c + ( rs_c * 3 ) ), ( n0_rem * sizeof( float ) ) );
		
		// c[0,0-15]
		selector1 = _mm512_loadu_ps( buf0 );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

		// c[1,0-15]
		selector1 = _mm512_loadu_ps( buf1 );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

		// c[2,0-15]
		selector1 = _mm512_loadu_ps( buf2 );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );

		// c[3,0-15]
		selector1 = _mm512_loadu_ps( buf3 );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_3p0 = _mm512_add_ps( selector1, c_float_3p0 );
	}
	// Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_4xLT16:
		{
			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
				 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
				memcpy( buf0, ( ( float* )post_ops_list_temp->op_args1 +
						post_op_c_j ), ( n0_rem * sizeof( float ) ) );
				selector1 = _mm512_loadu_ps( buf0 );

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
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_op_c_i + 1 ) );
				__m512 selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_op_c_i + 2 ) );
				a_bf16_0 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_op_c_i + 3 ) );

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );

				// c[2,0-15]
				c_float_2p0 = _mm512_add_ps( selector3, c_float_2p0 );

				// c[3,0-15]
				c_float_3p0 = _mm512_add_ps( a_bf16_0, c_float_3p0 );
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
POST_OPS_DOWNSCALE_4xLT16:
	{
		// c[0, 0-15]
		CVT_F32_BF16_LT16(c_float_0p0,0,0);

		// c[1, 0-15]
		CVT_F32_BF16_LT16(c_float_1p0,1,0);

		// c[2, 0-15]
		CVT_F32_BF16_LT16(c_float_2p0,2,0);

		// c[3, 0-15]
		CVT_F32_BF16_LT16(c_float_3p0,3,0);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}		
POST_OPS_4xLT16_DISABLE:
		;

	// Store the results.
	// c[0,0-15]
	_mm512_storeu_ps( buf0, c_float_0p0 );

	// c[1,0-15]
	_mm512_storeu_ps( buf1, c_float_1p0 );

	// c[2,0-15]
	_mm512_storeu_ps( buf2, c_float_2p0 );

	// c[3,0-15]
	_mm512_storeu_ps( buf3, c_float_3p0 );

	// Memcpy partial parts.
	// c[0,0-15]
	memcpy( c + ( rs_c * 0 ) + ( 0*16 ), buf0, ( n0_rem * sizeof( float ) ) );

	// c[1,0-15]
	memcpy( c + ( rs_c * 1 ) + ( 0*16 ), buf1, ( n0_rem * sizeof( float ) ) );

	// c[2,0-15]
	memcpy( c + ( rs_c * 2 ) + ( 0*16 ), buf2, ( n0_rem * sizeof( float ) ) );

	// c[3,0-15]
	memcpy( c + ( rs_c * 3 ) + ( 0*16 ), buf3, ( n0_rem * sizeof( float ) ) );
	
}

// 3xlt16 bf16 fringe kernel
LPGEMM_MN_LT_NR0_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_3xlt16)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_3xLT16_DISABLE,
						  &&POST_OPS_BIAS_3xLT16,
						  &&POST_OPS_RELU_3xLT16,
						  &&POST_OPS_RELU_SCALE_3xLT16,
						  &&POST_OPS_DOWNSCALE_3xLT16
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int32_t a_kfringe_buf = 0;

    // B matrix storage bfloat type
	__m512bh b0;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// For corner cases.
	float buf0[16];
	float buf1[16];
	float buf2[16];

	
	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();

	__m512 c_float_2p0 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );

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
	}
	// Handle k remainder.		
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		
		// Broadcast a[1,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 1 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
		
		// Broadcast a[2,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 2 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-15] = a[2,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
	}
	
	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	// Scale by alpha
	c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );

	c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );
	
	c_float_2p0 = _mm512_mul_ps( selector1, c_float_2p0 );
	
	// Scale C by beta.
	if ( beta != 0 )
	{
		memcpy( buf0, ( c + ( rs_c * 0 ) ), ( n0_rem * sizeof( float ) ) );
		memcpy( buf1, ( c + ( rs_c * 1 ) ), ( n0_rem * sizeof( float ) ) );
		memcpy( buf2, ( c + ( rs_c * 2 ) ), ( n0_rem * sizeof( float) ) );
		
		// c[0,0-15]
		selector1 = _mm512_loadu_ps( buf0 );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

		// c[1,0-15]
		selector1 = _mm512_loadu_ps( buf1 );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

		// c[2,0-15]
		selector1 = _mm512_loadu_ps( buf2 );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );
	}
	// Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_3xLT16:
		{
			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
				 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
				memcpy( buf0, ( ( float* )post_ops_list_temp->op_args1 +
						post_op_c_j ), ( n0_rem * sizeof( float ) ) );
				selector1 = _mm512_loadu_ps( buf0 );

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

				// c[2,0-15]
				c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );
			}
			else
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_op_c_i + 1 ) );
				a_bf16_0 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_op_c_i + 2 ) );

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );

				// c[2,0-15]
				c_float_2p0 = _mm512_add_ps( a_bf16_0, c_float_2p0 );
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_3xLT16:
		{
			selector1 = _mm512_setzero_ps();

			// c[0,0-15]
			c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

			// c[1,0-15]
			c_float_1p0 = _mm512_max_ps( selector1, c_float_1p0 );

			// c[2,0-15]
			c_float_2p0 = _mm512_max_ps( selector1, c_float_2p0 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_3xLT16:
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

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_DOWNSCALE_3xLT16:
	{
		// c[0, 0-15]
		CVT_F32_BF16_LT16(c_float_0p0,0,0);

		// c[1, 0-15]
		CVT_F32_BF16_LT16(c_float_1p0,1,0);

		// c[2, 0-15]
		CVT_F32_BF16_LT16(c_float_2p0,2,0);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}		
POST_OPS_3xLT16_DISABLE:
		;
	
	// Store the results.
	// c[0,0-15]
	_mm512_storeu_ps( buf0, c_float_0p0 );

	// c[1,0-15]
	_mm512_storeu_ps( buf1, c_float_1p0 );

	// c[2,0-15]
	_mm512_storeu_ps( buf2, c_float_2p0 );

	// Memcpy partial parts.
	// c[0,0-15]
	memcpy( c + ( rs_c * 0 ) + ( 0*16 ), buf0, ( n0_rem * sizeof( float ) ) );

	// c[1,0-15]
	memcpy( c + ( rs_c * 1 ) + ( 0*16 ), buf1, ( n0_rem * sizeof( float ) ) );

	// c[2,0-15]
	memcpy( c + ( rs_c * 2 ) + ( 0*16 ), buf2, ( n0_rem * sizeof( float ) ) );
	
}

// 2xlt16 bf16 fringe kernel
LPGEMM_MN_LT_NR0_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_2xlt16)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_2xLT16_DISABLE,
						  &&POST_OPS_BIAS_2xLT16,
						  &&POST_OPS_RELU_2xLT16,
						  &&POST_OPS_RELU_SCALE_2xLT16,
						  &&POST_OPS_DOWNSCALE_2xLT16
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int32_t a_kfringe_buf = 0;

    // B matrix storage bfloat type
	__m512bh b0;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// For corner cases.
	float buf0[16];
	float buf1[16];

	
	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );

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
	}
	// Handle k remainder.		
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		
		// Broadcast a[1,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 1 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
	}
	
	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	// Scale by alpha
	c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );

	c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );
	
	// Scale C by beta.
	if ( beta != 0 )
	{
		memcpy( buf0, ( c + ( rs_c * 0 ) ), ( n0_rem * sizeof( float ) ) );
		memcpy( buf1, ( c + ( rs_c * 1 ) ), ( n0_rem * sizeof( float) ) );
		
		// c[0,0-15]
		selector1 = _mm512_loadu_ps( buf0 );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

		// c[1,0-15]
		selector1 = _mm512_loadu_ps( buf1 );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );
	}
	// Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_2xLT16:
		{
			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
				 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
				memcpy( buf0, ( ( float* )post_ops_list_temp->op_args1 +
						post_op_c_j ), ( n0_rem * sizeof( float ) ) );
				selector1 = _mm512_loadu_ps( buf0 );

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );
			}
			else
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_op_c_i + 1 ) );

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

				// c[1,0-15]
				c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_2xLT16:
		{
			selector1 = _mm512_setzero_ps();

			// c[0,0-15]
			c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

			// c[1,0-15]
			c_float_1p0 = _mm512_max_ps( selector1, c_float_1p0 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_2xLT16:
		{
			selector1 = _mm512_setzero_ps();
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

			__mmask16 relu_cmp_mask;

			// c[0, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_0p0)

			// c[1, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_1p0)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_DOWNSCALE_2xLT16:
	{
		// c[0, 0-15]
		CVT_F32_BF16_LT16(c_float_0p0,0,0);

		// c[1, 0-15]
		CVT_F32_BF16_LT16(c_float_1p0,1,0);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}		
POST_OPS_2xLT16_DISABLE:
		;
	
	// Store the results.
	// c[0,0-15]
	_mm512_storeu_ps( buf0, c_float_0p0 );

	// c[1,0-15]
	_mm512_storeu_ps( buf1, c_float_1p0 );

	// Memcpy partial parts.
	// c[0,0-15]
	memcpy( c + ( rs_c * 0 ) + ( 0*16 ), buf0, ( n0_rem * sizeof( float ) ) );

	// c[1,0-15]
	memcpy( c + ( rs_c * 1 ) + ( 0*16 ), buf1, ( n0_rem * sizeof( float ) ) );
	
}

// 1xlt16 bf16 fringe kernel
LPGEMM_MN_LT_NR0_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_1xlt16)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_1xLT16_DISABLE,
						  &&POST_OPS_BIAS_1xLT16,
						  &&POST_OPS_RELU_1xLT16,
						  &&POST_OPS_RELU_SCALE_1xLT16,
						  &&POST_OPS_DOWNSCALE_1xLT16
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int32_t a_kfringe_buf = 0;

    // B matrix storage bfloat type
	__m512bh b0;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// For corner cases.
	float buf0[16];

	
	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
	}
	// Handle k remainder.		
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
	}
	
	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	// Scale by alpha
	c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );
	
	// Scale C by beta.
	if ( beta != 0 )
	{
		memcpy( buf0, ( c + ( rs_c * 0 ) ), ( n0_rem * sizeof( float ) ) );
		
		// c[0,0-15]
		selector1 = _mm512_loadu_ps( buf0 );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );
	}
	// Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_1xLT16:
		{
			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
				 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
				memcpy( buf0, ( ( float* )post_ops_list_temp->op_args1 +
						post_op_c_j ), ( n0_rem * sizeof( float ) ) );
				selector1 = _mm512_loadu_ps( buf0 );

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );
			}
			else
			{
				selector1 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
								+ post_op_c_i + 0 ) );

				// c[0,0-15]
				c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_1xLT16:
		{
			selector1 = _mm512_setzero_ps();

			// c[0,0-15]
			c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_1xLT16:
		{
			selector1 = _mm512_setzero_ps();
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

			__mmask16 relu_cmp_mask;

			// c[0, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_0p0)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_DOWNSCALE_1xLT16:
	{
		// c[0, 0-15]
		CVT_F32_BF16_LT16(c_float_0p0,0,0);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}		
POST_OPS_1xLT16_DISABLE:
		;
	
	// Store the results.
	// c[0,0-15]
	_mm512_storeu_ps( buf0, c_float_0p0 );

	// Memcpy partial parts.
	// c[0,0-15]
	memcpy( c + ( rs_c * 0 ) + ( 0*16 ), buf0, ( n0_rem * sizeof( float ) ) );
	
}

// 5x16 bf16 kernel
LPGEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_5x16)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_5x16_DISABLE,
						  &&POST_OPS_BIAS_5x16,
						  &&POST_OPS_RELU_5x16,
						  &&POST_OPS_RELU_SCALE_5x16,
						  &&POST_OPS_DOWNSCALE_5x16
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int32_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();

	__m512 c_float_2p0 = _mm512_setzero_ps();
	
	__m512 c_float_3p0 = _mm512_setzero_ps();

	__m512 c_float_4p0 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );

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
		
		// Broadcast a[4,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 4 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[4,0-15] = a[4,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );
	}
	// Handle k remainder.	
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		
		// Broadcast a[1,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 1 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
		
		// Broadcast a[2,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 2 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-15] = a[2,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
		
		// Broadcast a[3,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 3 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[3,0-15] = a[3,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );
		
		// Broadcast a[4,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 4 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[4,0-15] = a[4,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );
	}
	
	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	// Scale by alpha
	c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );

	c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );
	
	c_float_2p0 = _mm512_mul_ps( selector1, c_float_2p0 );
	
	c_float_3p0 = _mm512_mul_ps( selector1, c_float_3p0 );
	
	c_float_4p0 = _mm512_mul_ps( selector1, c_float_4p0 );

	// Scale C by beta.
	if ( beta != 0 )
	{
		// c[0,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 0 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

		// c[1,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 1 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

		// c[2,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 2 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );

		// c[3,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 3 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_3p0 = _mm512_add_ps( selector1, c_float_3p0 );

		// c[4,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 4 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_4p0 = _mm512_add_ps( selector1, c_float_4p0 );
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_5x16:
	{
		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			selector1 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j );

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

			// c[2,0-15]
			c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );

			// c[3,0-15]
			c_float_3p0 = _mm512_add_ps( selector1, c_float_3p0 );

			// c[4,0-15]
			c_float_4p0 = _mm512_add_ps( selector1, c_float_4p0 );
		}
		else
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 0 ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 1 ) );
			__m512 selector3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 2 ) );
			__m512 selector4 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 3 ) );
			__m512 selector5 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 4 ) );

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );

			// c[2,0-15]
			c_float_2p0 = _mm512_add_ps( selector3, c_float_2p0 );

			// c[3,0-15]
			c_float_3p0 = _mm512_add_ps( selector4, c_float_3p0 );

			// c[4,0-15]
			c_float_4p0 = _mm512_add_ps( selector5, c_float_4p0 );
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_5x16:
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

		// c[4,0-15]
		c_float_4p0 = _mm512_max_ps( selector1, c_float_4p0 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_5x16:
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

		// c[4, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_4p0)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_DOWNSCALE_5x16:
	{
		// c[0, 0-15]
		CVT_F32_BF16(c_float_0p0,0,0);

		// c[1, 0-15]
		CVT_F32_BF16(c_float_1p0,1,0);

		// c[2, 0-15]
		CVT_F32_BF16(c_float_2p0,2,0);

		// c[3, 0-15]
		CVT_F32_BF16(c_float_3p0,3,0);

		// c[4, 0-15]
		CVT_F32_BF16(c_float_4p0,4,0);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}	
POST_OPS_5x16_DISABLE:
	;
	
	// Store the results.
	// c[0,0-15]
	_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 0*16 ), c_float_0p0 );

	// c[1,0-15]
	_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 0*16 ), c_float_1p0 );

	// c[2,0-15]
	_mm512_storeu_ps( c + ( rs_c * 2 ) + ( 0*16 ), c_float_2p0 );

	// c[3,0-15]
	_mm512_storeu_ps( c + ( rs_c * 3 ) + ( 0*16 ), c_float_3p0 );

	// c[4,0-15]
	_mm512_storeu_ps( c + ( rs_c * 4 ) + ( 0*16 ), c_float_4p0 );
}

// 4x16 bf16 kernel
LPGEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_4x16)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_4x16_DISABLE,
						  &&POST_OPS_BIAS_4x16,
						  &&POST_OPS_RELU_4x16,
						  &&POST_OPS_RELU_SCALE_4x16,
						  &&POST_OPS_DOWNSCALE_4x16
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int32_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();

	__m512 c_float_2p0 = _mm512_setzero_ps();
	
	__m512 c_float_3p0 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );

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
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		
		// Broadcast a[1,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 1 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
		
		// Broadcast a[2,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 2 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-15] = a[2,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
		
		// Broadcast a[3,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 3 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[3,0-15] = a[3,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );
	}
	
	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	// Scale by alpha
	c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );

	c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );
	
	c_float_2p0 = _mm512_mul_ps( selector1, c_float_2p0 );
	
	c_float_3p0 = _mm512_mul_ps( selector1, c_float_3p0 );

	// Scale C by beta.
	if ( beta != 0 )
	{
		// c[0,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 0 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

		// c[1,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 1 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

		// c[2,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 2 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );

		// c[3,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 3 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_3p0 = _mm512_add_ps( selector1, c_float_3p0 );
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_4x16:
	{
		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			selector1 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j );

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
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 0 ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 1 ) );
			__m512 selector3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 2 ) );
			__m512 selector4 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 3 ) );

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
POST_OPS_DOWNSCALE_4x16:
	{
		// c[0, 0-15]
		CVT_F32_BF16(c_float_0p0,0,0);

		// c[1, 0-15]
		CVT_F32_BF16(c_float_1p0,1,0);

		// c[2, 0-15]
		CVT_F32_BF16(c_float_2p0,2,0);

		// c[3, 0-15]
		CVT_F32_BF16(c_float_3p0,3,0);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}	
POST_OPS_4x16_DISABLE:
	;
	
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

// 3x16 bf16 kernel
LPGEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_3x16)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_3x16_DISABLE,
						  &&POST_OPS_BIAS_3x16,
						  &&POST_OPS_RELU_3x16,
						  &&POST_OPS_RELU_SCALE_3x16,
						  &&POST_OPS_DOWNSCALE_3x16
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int32_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();

	__m512 c_float_2p0 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );

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
	}
	// Handle k remainder.	
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		
		// Broadcast a[1,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 1 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
		
		// Broadcast a[2,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 2 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-15] = a[2,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
	}
	
	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	// Scale by alpha
	c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );

	c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );
	
	c_float_2p0 = _mm512_mul_ps( selector1, c_float_2p0 );

	// Scale C by beta.
	if ( beta != 0 )
	{
		// c[0,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 0 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

		// c[1,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 1 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

		// c[2,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 2 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_3x16:
	{
		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			selector1 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j );

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

			// c[2,0-15]
			c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );
		}
		else
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 0 ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 1 ) );
			__m512 selector3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 2 ) );

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );

			// c[2,0-15]
			c_float_2p0 = _mm512_add_ps( selector3, c_float_2p0 );
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_3x16:
	{
		selector1 = _mm512_setzero_ps();

		// c[0,0-15]
		c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

		// c[1,0-15]
		c_float_1p0 = _mm512_max_ps( selector1, c_float_1p0 );

		// c[2,0-15]
		c_float_2p0 = _mm512_max_ps( selector1, c_float_2p0 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_3x16:
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

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_DOWNSCALE_3x16:
	{
		// c[0, 0-15]
		CVT_F32_BF16(c_float_0p0,0,0);

		// c[1, 0-15]
		CVT_F32_BF16(c_float_1p0,1,0);

		// c[2, 0-15]
		CVT_F32_BF16(c_float_2p0,2,0);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}	
POST_OPS_3x16_DISABLE:
	;


	// Store the results.
	// c[0,0-15]
	_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 0*16 ), c_float_0p0 );

	// c[1,0-15]
	_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 0*16 ), c_float_1p0 );

	// c[2,0-15]
	_mm512_storeu_ps( c + ( rs_c * 2 ) + ( 0*16 ), c_float_2p0 );
}

// 2x16 bf16 kernel
LPGEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_2x16)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_2x16_DISABLE,
						  &&POST_OPS_BIAS_2x16,
						  &&POST_OPS_RELU_2x16,
						  &&POST_OPS_RELU_SCALE_2x16,
						  &&POST_OPS_DOWNSCALE_2x16
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int32_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );

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
	}
	// Handle k remainder.	
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		
		// Broadcast a[1,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 1 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-15] = a[1,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
	}
	
	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	// Scale by alpha
	c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );

	c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );

	// Scale C by beta.
	if ( beta != 0 )
	{
		// c[0,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 0 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

		// c[1,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 1 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_2x16:
	{
		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			selector1 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j );

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );
		}
		else
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 0 ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 1 ) );

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_2x16:
	{
		selector1 = _mm512_setzero_ps();

		// c[0,0-15]
		c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

		// c[1,0-15]
		c_float_1p0 = _mm512_max_ps( selector1, c_float_1p0 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_2x16:
	{
		selector1 = _mm512_setzero_ps();
		selector2 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__mmask16 relu_cmp_mask;

		// c[0, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_0p0)

		// c[1, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_1p0)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_DOWNSCALE_2x16:
	{
		// c[0, 0-15]
		CVT_F32_BF16(c_float_0p0,0,0);

		// c[1, 0-15]
		CVT_F32_BF16(c_float_1p0,1,0);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}	
POST_OPS_2x16_DISABLE:
	;
	
	// Store the results.
	// c[0,0-15]
	_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 0*16 ), c_float_0p0 );

	// c[1,0-15]
	_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 0*16 ), c_float_1p0 );
}

// 1x16 bf16 kernel
LPGEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_1x16)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_1x16_DISABLE,
						  &&POST_OPS_BIAS_1x16,
						  &&POST_OPS_RELU_1x16,
						  &&POST_OPS_RELU_SCALE_1x16,
						  &&POST_OPS_DOWNSCALE_1x16
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int32_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
	}
	// Handle k remainder.	
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );

		// Broadcast a[0,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-15] = a[0,kr:kr+2]*b[kr:kr+2,0-15]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
	}
    
	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	// Scale by alpha
	c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );

	// Scale C by beta.
	if ( beta != 0 )
	{
		// c[0,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 0 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_1x16:
	{
		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			selector1 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j );

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );
		}
		else
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 0 ) );

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_1x16:
	{
		selector1 = _mm512_setzero_ps();

		// c[0,0-15]
		c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_1x16:
	{
		selector1 = _mm512_setzero_ps();
		selector2 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__mmask16 relu_cmp_mask;

		// c[0, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_0p0)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_DOWNSCALE_1x16:
	{
		// c[0, 0-15]
		CVT_F32_BF16(c_float_0p0,0,0);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}	
POST_OPS_1x16_DISABLE:
	;
	
	// Store the results.
	// c[0,0-15]
	_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 0*16 ), c_float_0p0 );
}

// 5x32 bf16 kernel
LPGEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_5x32)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_5x32_DISABLE,
						  &&POST_OPS_BIAS_5x32,
						  &&POST_OPS_RELU_5x32,
						  &&POST_OPS_RELU_SCALE_5x32,
						  &&POST_OPS_DOWNSCALE_5x32
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int32_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;
	__m512bh b1;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();
	__m512 c_float_0p1 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();
	__m512 c_float_1p1 = _mm512_setzero_ps();

	__m512 c_float_2p0 = _mm512_setzero_ps();
	__m512 c_float_2p1 = _mm512_setzero_ps();
	
	__m512 c_float_3p0 = _mm512_setzero_ps();
	__m512 c_float_3p1 = _mm512_setzero_ps();

	__m512 c_float_4p0 = _mm512_setzero_ps();
	__m512 c_float_4p1 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 1 ) );

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
		
		// Broadcast a[4,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 4 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[4,0-31] = a[4,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );
		c_float_4p1 = _mm512_dpbf16_ps( c_float_4p1, a_bf16_0, b1 );
	}	
	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 1 ) );

		// Broadcast a[0,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
		
		// Broadcast a[1,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 1 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-31] = a[1,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );
		
		// Broadcast a[2,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 2 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-31] = a[2,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
		c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );
		
		// Broadcast a[3,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 3 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[3,0-31] = a[3,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );
		c_float_3p1 = _mm512_dpbf16_ps( c_float_3p1, a_bf16_0, b1 );
		
		// Broadcast a[4,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 4 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[4,0-31] = a[4,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );
		c_float_4p1 = _mm512_dpbf16_ps( c_float_4p1, a_bf16_0, b1 );
	}
	
	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	// Scale by alpha
	c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );
	c_float_0p1 = _mm512_mul_ps( selector1, c_float_0p1 );

	c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );
	c_float_1p1 = _mm512_mul_ps( selector1, c_float_1p1 );
	
	c_float_2p0 = _mm512_mul_ps( selector1, c_float_2p0 );
	c_float_2p1 = _mm512_mul_ps( selector1, c_float_2p1 );
	
	c_float_3p0 = _mm512_mul_ps( selector1, c_float_3p0 );
	c_float_3p1 = _mm512_mul_ps( selector1, c_float_3p1 );
	
	c_float_4p0 = _mm512_mul_ps( selector1, c_float_4p0 );
	c_float_4p1 = _mm512_mul_ps( selector1, c_float_4p1 );

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

		// c[1,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 1 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

		// c[1,16-31]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 1 ) + ( 1*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p1 = _mm512_add_ps( selector1, c_float_1p1 );

		// c[2,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 2 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );

		// c[2,16-31]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 2 ) + ( 1*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_2p1 = _mm512_add_ps( selector1, c_float_2p1 );

		// c[3,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 3 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_3p0 = _mm512_add_ps( selector1, c_float_3p0 );

		// c[3,16-31]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 3 ) + ( 1*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_3p1 = _mm512_add_ps( selector1, c_float_3p1 );

		// c[4,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 4 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_4p0 = _mm512_add_ps( selector1, c_float_4p0 );

		// c[4,16-31]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 4 ) + ( 1*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_4p1 = _mm512_add_ps( selector1, c_float_4p1 );
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_5x32:
	{
		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			selector1 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 0 * 16 ) );
			selector2 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 1 * 16 ) );

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

			// c[4,0-15]
			c_float_4p0 = _mm512_add_ps( selector1, c_float_4p0 );

			// c[4, 16-31]
			c_float_4p1 = _mm512_add_ps( selector2, c_float_4p1 );
		}
		else
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 0 ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 1 ) );
			__m512 selector3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 2 ) );
			__m512 selector4 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 3 ) );
			__m512 selector5 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 4 ) );

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

			// c[4,0-15]
			c_float_4p0 = _mm512_add_ps( selector5, c_float_4p0 );

			// c[4, 16-31]
			c_float_4p1 = _mm512_add_ps( selector5, c_float_4p1 );
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_5x32:
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

		// c[4,0-15]
		c_float_4p0 = _mm512_max_ps( selector1, c_float_4p0 );

		// c[4,16-31]
		c_float_4p1 = _mm512_max_ps( selector1, c_float_4p1 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_5x32:
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

		// c[4, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_4p0)

		// c[4, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_4p1)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_DOWNSCALE_5x32:
	{
		// c[0, 0-15]
		CVT_F32_BF16(c_float_0p0,0,0);

		// c[0, 16-31]
		CVT_F32_BF16(c_float_0p1,0,1);

		// c[1, 0-15]
		CVT_F32_BF16(c_float_1p0,1,0);

		// c[1, 16-31]
		CVT_F32_BF16(c_float_1p1,1,1);

		// c[2, 0-15]
		CVT_F32_BF16(c_float_2p0,2,0);

		// c[2, 16-31]
		CVT_F32_BF16(c_float_2p1,2,1);

		// c[3, 0-15]
		CVT_F32_BF16(c_float_3p0,3,0);

		// c[3, 16-31]
		CVT_F32_BF16(c_float_3p1,3,1);

		// c[4, 0-15]
		CVT_F32_BF16(c_float_4p0,4,0);

		// c[4, 16-31]
		CVT_F32_BF16(c_float_4p1,4,1);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}	
POST_OPS_5x32_DISABLE:
	;
	
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

	// c[4,0-15]
	_mm512_storeu_ps( c + ( rs_c * 4 ) + ( 0*16 ), c_float_4p0 );

	// c[4,16-31]
	_mm512_storeu_ps( c + ( rs_c * 4 ) + ( 1*16 ), c_float_4p1 );
}

// 4x32 bf16 kernel
LPGEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_4x32)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_4x32_DISABLE,
						  &&POST_OPS_BIAS_4x32,
						  &&POST_OPS_RELU_4x32,
						  &&POST_OPS_RELU_SCALE_4x32,
						  &&POST_OPS_DOWNSCALE_4x32
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int32_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;
	__m512bh b1;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();
	__m512 c_float_0p1 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();
	__m512 c_float_1p1 = _mm512_setzero_ps();

	__m512 c_float_2p0 = _mm512_setzero_ps();
	__m512 c_float_2p1 = _mm512_setzero_ps();
	
	__m512 c_float_3p0 = _mm512_setzero_ps();
	__m512 c_float_3p1 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 1 ) );

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
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 1 ) );

		// Broadcast a[0,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
		
		// Broadcast a[1,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 1 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-31] = a[1,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );
		
		// Broadcast a[2,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 2 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-31] = a[2,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
		c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );
		
		// Broadcast a[3,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 3 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[3,0-31] = a[3,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );
		c_float_3p1 = _mm512_dpbf16_ps( c_float_3p1, a_bf16_0, b1 );
	}
	
	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	// Scale by alpha
	c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );
	c_float_0p1 = _mm512_mul_ps( selector1, c_float_0p1 );

	c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );
	c_float_1p1 = _mm512_mul_ps( selector1, c_float_1p1 );
	
	c_float_2p0 = _mm512_mul_ps( selector1, c_float_2p0 );
	c_float_2p1 = _mm512_mul_ps( selector1, c_float_2p1 );
	
	c_float_3p0 = _mm512_mul_ps( selector1, c_float_3p0 );
	c_float_3p1 = _mm512_mul_ps( selector1, c_float_3p1 );

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

		// c[1,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 1 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

		// c[1,16-31]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 1 ) + ( 1*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p1 = _mm512_add_ps( selector1, c_float_1p1 );

		// c[2,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 2 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );

		// c[2,16-31]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 2 ) + ( 1*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_2p1 = _mm512_add_ps( selector1, c_float_2p1 );

		// c[3,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 3 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_3p0 = _mm512_add_ps( selector1, c_float_3p0 );

		// c[3,16-31]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 3 ) + ( 1*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_3p1 = _mm512_add_ps( selector1, c_float_3p1 );
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_4x32:
	{
		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			selector1 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 0 * 16 ) );
			selector2 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 1 * 16 ) );

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
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 0 ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 1 ) );
			__m512 selector3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 2 ) );
			__m512 selector4 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 3 ) );

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
POST_OPS_DOWNSCALE_4x32:
	{
		// c[0, 0-15]
		CVT_F32_BF16(c_float_0p0,0,0);

		// c[0, 16-31]
		CVT_F32_BF16(c_float_0p1,0,1);

		// c[1, 0-15]
		CVT_F32_BF16(c_float_1p0,1,0);

		// c[1, 16-31]
		CVT_F32_BF16(c_float_1p1,1,1);

		// c[2, 0-15]
		CVT_F32_BF16(c_float_2p0,2,0);

		// c[2, 16-31]
		CVT_F32_BF16(c_float_2p1,2,1);

		// c[3, 0-15]
		CVT_F32_BF16(c_float_3p0,3,0);

		// c[3, 16-31]
		CVT_F32_BF16(c_float_3p1,3,1);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}	
POST_OPS_4x32_DISABLE:
	;
	
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

// 3x32 bf16 kernel
LPGEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_3x32)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_3x32_DISABLE,
						  &&POST_OPS_BIAS_3x32,
						  &&POST_OPS_RELU_3x32,
						  &&POST_OPS_RELU_SCALE_3x32,
						  &&POST_OPS_DOWNSCALE_3x32
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int32_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;
	__m512bh b1;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();
	__m512 c_float_0p1 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();
	__m512 c_float_1p1 = _mm512_setzero_ps();

	__m512 c_float_2p0 = _mm512_setzero_ps();
	__m512 c_float_2p1 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 1 ) );

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
	}
	// Handle k remainder.	
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 1 ) );

		// Broadcast a[0,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
		
		// Broadcast a[1,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 1 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-31] = a[1,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );
		
		// Broadcast a[2,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 2 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-31] = a[2,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
		c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );
	}
	
	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	// Scale by alpha
	c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );
	c_float_0p1 = _mm512_mul_ps( selector1, c_float_0p1 );

	c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );
	c_float_1p1 = _mm512_mul_ps( selector1, c_float_1p1 );
	
	c_float_2p0 = _mm512_mul_ps( selector1, c_float_2p0 );
	c_float_2p1 = _mm512_mul_ps( selector1, c_float_2p1 );

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

		// c[1,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 1 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

		// c[1,16-31]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 1 ) + ( 1*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p1 = _mm512_add_ps( selector1, c_float_1p1 );

		// c[2,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 2 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );

		// c[2,16-31]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 2 ) + ( 1*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_2p1 = _mm512_add_ps( selector1, c_float_2p1 );
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_3x32:
	{
		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			selector1 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 0 * 16 ) );
			selector2 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 1 * 16 ) );

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
		}
		else
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 0 ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 1 ) );
			__m512 selector3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 2 ) );

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
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_3x32:
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

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_3x32:
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

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_DOWNSCALE_3x32:
	{
		// c[0, 0-15]
		CVT_F32_BF16(c_float_0p0,0,0);

		// c[0, 16-31]
		CVT_F32_BF16(c_float_0p1,0,1);

		// c[1, 0-15]
		CVT_F32_BF16(c_float_1p0,1,0);

		// c[1, 16-31]
		CVT_F32_BF16(c_float_1p1,1,1);

		// c[2, 0-15]
		CVT_F32_BF16(c_float_2p0,2,0);

		// c[2, 16-31]
		CVT_F32_BF16(c_float_2p1,2,1);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}	
POST_OPS_3x32_DISABLE:
	;
	
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
}

// 2x32 bf16 kernel
LPGEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_2x32)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_2x32_DISABLE,
						  &&POST_OPS_BIAS_2x32,
						  &&POST_OPS_RELU_2x32,
						  &&POST_OPS_RELU_SCALE_2x32,
						  &&POST_OPS_DOWNSCALE_2x32
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int32_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;
	__m512bh b1;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();
	__m512 c_float_0p1 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();
	__m512 c_float_1p1 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 1 ) );

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
	}
	// Handle k remainder.	
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 1 ) );

		// Broadcast a[0,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
		
		// Broadcast a[1,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 1 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-31] = a[1,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );
	}
	
	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	// Scale by alpha
	c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );
	c_float_0p1 = _mm512_mul_ps( selector1, c_float_0p1 );

	c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );
	c_float_1p1 = _mm512_mul_ps( selector1, c_float_1p1 );

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

		// c[1,0-15]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 1 ) + ( 0*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

		// c[1,16-31]
		selector1 = _mm512_loadu_ps( c + ( rs_c * 1 ) + ( 1*16 ) );
		selector1 = _mm512_mul_ps( selector2, selector1 );
		c_float_1p1 = _mm512_add_ps( selector1, c_float_1p1 );
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_2x32:
	{
		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			selector1 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 0 * 16 ) );
			selector2 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 1 * 16 ) );

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_add_ps( selector2, c_float_0p1 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

			// c[1, 16-31]
			c_float_1p1 = _mm512_add_ps( selector2, c_float_1p1 );
		}
		else
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 0 ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 1 ) );

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_add_ps( selector1, c_float_0p1 );

			// c[1,0-15]
			c_float_1p0 = _mm512_add_ps( selector2, c_float_1p0 );

			// c[1, 16-31]
			c_float_1p1 = _mm512_add_ps( selector2, c_float_1p1 );
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_2x32:
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

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_2x32:
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

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_DOWNSCALE_2x32:
	{
		// c[0, 0-15]
		CVT_F32_BF16(c_float_0p0,0,0);

		// c[0, 16-31]
		CVT_F32_BF16(c_float_0p1,0,1);

		// c[1, 0-15]
		CVT_F32_BF16(c_float_1p0,1,0);

		// c[1, 16-31]
		CVT_F32_BF16(c_float_1p1,1,1);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}	
POST_OPS_2x32_DISABLE:
	;
	
	// Store the results.
	// c[0,0-15]
	_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 0*16 ), c_float_0p0 );

	// c[0, 16-31]
	_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 1*16 ), c_float_0p1 );

	// c[1,0-15]
	_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 0*16 ), c_float_1p0 );

	// c[1,16-31]
	_mm512_storeu_ps( c + ( rs_c * 1 ) + ( 1*16 ), c_float_1p1 );
}

// 1x32 bf16 kernel
LPGEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_1x32)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_1x32_DISABLE,
						  &&POST_OPS_BIAS_1x32,
						  &&POST_OPS_RELU_1x32,
						  &&POST_OPS_RELU_SCALE_1x32,
						  &&POST_OPS_DOWNSCALE_1x32
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int32_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;
	__m512bh b1;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();
	__m512 c_float_0p1 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 1 ) );

		// Broadcast a[0,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
	}
	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 1 ) );

		// Broadcast a[0,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-31] = a[0,kr:kr+2]*b[kr:kr+2,0-31]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
	}
	
	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	// Scale by alpha
	c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );
	c_float_0p1 = _mm512_mul_ps( selector1, c_float_0p1 );

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
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_1x32:
	{
		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			selector1 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 0 * 16 ) );
			selector2 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 1 * 16 ) );

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_add_ps( selector2, c_float_0p1 );
		}
		else
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 0 ) );

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_add_ps( selector1, c_float_0p1 );
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_1x32:
	{
		selector1 = _mm512_setzero_ps();

		// c[0,0-15]
		c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

		// c[0, 16-31]
		c_float_0p1 = _mm512_max_ps( selector1, c_float_0p1 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_1x32:
	{
		selector1 = _mm512_setzero_ps();
		selector2 =
			_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

		__mmask16 relu_cmp_mask;

		// c[0, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_0p0)

		// c[0, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_0p1)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_DOWNSCALE_1x32:
	{
		// c[0, 0-15]
		CVT_F32_BF16(c_float_0p0,0,0);

		// c[0, 16-31]
		CVT_F32_BF16(c_float_0p1,0,1);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}	
POST_OPS_1x32_DISABLE:
	;
	
	// Store the results.
	// c[0,0-15]
	_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 0*16 ), c_float_0p0 );

	// c[0, 16-31]
	_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 1*16 ), c_float_0p1 );
}

// 5x48 bf16 kernel
LPGEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_5x48)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_5x48_DISABLE,
						  &&POST_OPS_BIAS_5x48,
						  &&POST_OPS_RELU_5x48,
						  &&POST_OPS_RELU_SCALE_5x48,
						  &&POST_OPS_DOWNSCALE_5x48
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int32_t a_kfringe_buf = 0;

    // B matrix storage bfloat type
	__m512bh b0;
	__m512bh b1;
	__m512bh b2;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

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

	__m512 c_float_4p0 = _mm512_setzero_ps();
	__m512 c_float_4p1 = _mm512_setzero_ps();
	__m512 c_float_4p2 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 1 ) );
		b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 2 ) );

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
		
		// Broadcast a[4,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 4 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[4,0-47] = a[4,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );
		c_float_4p1 = _mm512_dpbf16_ps( c_float_4p1, a_bf16_0, b1 );
		c_float_4p2 = _mm512_dpbf16_ps( c_float_4p2, a_bf16_0, b2 );
	}
	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 1 ) );
		b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 2 ) );

		// Broadcast a[0,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-47] = a[0,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
		c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );
		
		// Broadcast a[1,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 1 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-47] = a[1,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );
		c_float_1p2 = _mm512_dpbf16_ps( c_float_1p2, a_bf16_0, b2 );
		
		// Broadcast a[2,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 2 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-47] = a[2,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
		c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );
		c_float_2p2 = _mm512_dpbf16_ps( c_float_2p2, a_bf16_0, b2 );
		
		// Broadcast a[3,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 3 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[3,0-47] = a[3,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );
		c_float_3p1 = _mm512_dpbf16_ps( c_float_3p1, a_bf16_0, b1 );
		c_float_3p2 = _mm512_dpbf16_ps( c_float_3p2, a_bf16_0, b2 );
		
		// Broadcast a[4,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 4 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[4,0-47] = a[4,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );
		c_float_4p1 = _mm512_dpbf16_ps( c_float_4p1, a_bf16_0, b1 );
		c_float_4p2 = _mm512_dpbf16_ps( c_float_4p2, a_bf16_0, b2 );
	}
    
	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

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
	
	c_float_4p0 = _mm512_mul_ps( selector1, c_float_4p0 );
	c_float_4p1 = _mm512_mul_ps( selector1, c_float_4p1 );
	c_float_4p2 = _mm512_mul_ps( selector1, c_float_4p2 );

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
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_5x48:
	{
		__m512 selector3;

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			selector1 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 0 * 16 ) );
			selector2 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 1 * 16 ) );
			selector3 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 2 * 16 ) );

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

			// c[4,0-15]
			c_float_4p0 = _mm512_add_ps( selector1, c_float_4p0 );

			// c[4, 16-31]
			c_float_4p1 = _mm512_add_ps( selector2, c_float_4p1 );

			// c[4,32-47]
			c_float_4p2 = _mm512_add_ps( selector3, c_float_4p2 );
		}
		else
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 0 ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 1 ) );
			selector3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 2 ) );
			__m512 selector4 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 3 ) );
			__m512 selector5 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 4 ) );

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

			// c[4,0-15]
			c_float_4p0 = _mm512_add_ps( selector5, c_float_4p0 );

			// c[4, 16-31]
			c_float_4p1 = _mm512_add_ps( selector5, c_float_4p1 );

			// c[4,32-47]
			c_float_4p2 = _mm512_add_ps( selector5, c_float_4p2 );
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_5x48:
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

		// c[4,0-15]
		c_float_4p0 = _mm512_max_ps( selector1, c_float_4p0 );

		// c[4,16-31]
		c_float_4p1 = _mm512_max_ps( selector1, c_float_4p1 );

		// c[4,32-47]
		c_float_4p2 = _mm512_max_ps( selector1, c_float_4p2 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_5x48:
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

		// c[4, 0-15]
		RELU_SCALE_OP_F32_AVX512(c_float_4p0)

		// c[4, 16-31]
		RELU_SCALE_OP_F32_AVX512(c_float_4p1)

		// c[4, 32-47]
		RELU_SCALE_OP_F32_AVX512(c_float_4p2)

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_DOWNSCALE_5x48:
	{
		// c[0, 0-15]
		CVT_F32_BF16(c_float_0p0,0,0);

		// c[0, 16-31]
		CVT_F32_BF16(c_float_0p1,0,1);

		// c[0, 32-47]
		CVT_F32_BF16(c_float_0p2,0,2);

		// c[1, 0-15]
		CVT_F32_BF16(c_float_1p0,1,0);

		// c[1, 16-31]
		CVT_F32_BF16(c_float_1p1,1,1);

		// c[1, 32-47]
		CVT_F32_BF16(c_float_1p2,1,2);

		// c[2, 0-15]
		CVT_F32_BF16(c_float_2p0,2,0);

		// c[2, 16-31]
		CVT_F32_BF16(c_float_2p1,2,1);

		// c[2, 32-47]
		CVT_F32_BF16(c_float_2p2,2,2);

		// c[3, 0-15]
		CVT_F32_BF16(c_float_3p0,3,0);

		// c[3, 16-31]
		CVT_F32_BF16(c_float_3p1,3,1);

		// c[3, 32-47]
		CVT_F32_BF16(c_float_3p2,3,2);

		// c[4, 0-15]
		CVT_F32_BF16(c_float_4p0,4,0);

		// c[4, 16-31]
		CVT_F32_BF16(c_float_4p1,4,1);

		// c[4, 32-47]
		CVT_F32_BF16(c_float_4p2,4,2);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}	
POST_OPS_5x48_DISABLE:
	;
	
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

	// c[4,0-15]
	_mm512_storeu_ps( c + ( rs_c * 4 ) + ( 0*16 ), c_float_4p0 );

	// c[4,16-31]
	_mm512_storeu_ps( c + ( rs_c * 4 ) + ( 1*16 ), c_float_4p1 );

	// c[4,32-47]
	_mm512_storeu_ps( c + ( rs_c * 4 ) + ( 2*16 ), c_float_4p2 );
}

// 4x48 bf16 kernel
LPGEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_4x48)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_4x48_DISABLE,
						  &&POST_OPS_BIAS_4x48,
						  &&POST_OPS_RELU_4x48,
						  &&POST_OPS_RELU_SCALE_4x48,
						  &&POST_OPS_DOWNSCALE_4x48
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int32_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;
	__m512bh b1;
	__m512bh b2;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

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

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 1 ) );
		b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 2 ) );

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
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 1 ) );
		b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 2 ) );

		// Broadcast a[0,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-47] = a[0,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
		c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );
		
		// Broadcast a[1,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 1 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-47] = a[1,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );
		c_float_1p2 = _mm512_dpbf16_ps( c_float_1p2, a_bf16_0, b2 );
		
		// Broadcast a[2,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 2 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-47] = a[2,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
		c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );
		c_float_2p2 = _mm512_dpbf16_ps( c_float_2p2, a_bf16_0, b2 );
		
		// Broadcast a[3,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 3 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[3,0-47] = a[3,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_0, b0 );
		c_float_3p1 = _mm512_dpbf16_ps( c_float_3p1, a_bf16_0, b1 );
		c_float_3p2 = _mm512_dpbf16_ps( c_float_3p2, a_bf16_0, b2 );
	}
	
	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

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
			selector1 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 0 * 16 ) );
			selector2 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 1 * 16 ) );
			selector3 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 2 * 16 ) );

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
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 0 ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 1 ) );
			selector3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 2 ) );
			__m512 selector4 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 3 ) );

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
POST_OPS_DOWNSCALE_4x48:
	{
		// c[0, 0-15]
		CVT_F32_BF16(c_float_0p0,0,0);

		// c[0, 16-31]
		CVT_F32_BF16(c_float_0p1,0,1);

		// c[0, 32-47]
		CVT_F32_BF16(c_float_0p2,0,2);

		// c[1, 0-15]
		CVT_F32_BF16(c_float_1p0,1,0);

		// c[1, 16-31]
		CVT_F32_BF16(c_float_1p1,1,1);

		// c[1, 32-47]
		CVT_F32_BF16(c_float_1p2,1,2);

		// c[2, 0-15]
		CVT_F32_BF16(c_float_2p0,2,0);

		// c[2, 16-31]
		CVT_F32_BF16(c_float_2p1,2,1);

		// c[2, 32-47]
		CVT_F32_BF16(c_float_2p2,2,2);

		// c[3, 0-15]
		CVT_F32_BF16(c_float_3p0,3,0);

		// c[3, 16-31]
		CVT_F32_BF16(c_float_3p1,3,1);

		// c[3, 32-47]
		CVT_F32_BF16(c_float_3p2,3,2);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}	
POST_OPS_4x48_DISABLE:
	;
	
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

// 3x48 bf16 kernel
LPGEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_3x48)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_3x48_DISABLE,
						  &&POST_OPS_BIAS_3x48,
						  &&POST_OPS_RELU_3x48,
						  &&POST_OPS_RELU_SCALE_3x48,
						  &&POST_OPS_DOWNSCALE_3x48
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int32_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;
	__m512bh b1;
	__m512bh b2;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

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

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 1 ) );
		b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 2 ) );

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
	}
	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 1 ) );
		b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 2 ) );

		// Broadcast a[0,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-47] = a[0,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
		c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );
		
		// Broadcast a[1,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 1 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-47] = a[1,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );
		c_float_1p2 = _mm512_dpbf16_ps( c_float_1p2, a_bf16_0, b2 );
		
		// Broadcast a[2,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 2 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[2,0-47] = a[2,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );
		c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );
		c_float_2p2 = _mm512_dpbf16_ps( c_float_2p2, a_bf16_0, b2 );
	}

	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

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
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_3x48:
	{
		__m512 selector3;

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			selector1 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 0 * 16 ) );
			selector2 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 1 * 16 ) );
			selector3 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 2 * 16 ) );

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
		}
		else
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 0 ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 1 ) );
			selector3 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 2 ) );

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
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_3x48:
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

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_3x48:
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

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_DOWNSCALE_3x48:
	{
		// c[0, 0-15]
		CVT_F32_BF16(c_float_0p0,0,0);

		// c[0, 16-31]
		CVT_F32_BF16(c_float_0p1,0,1);

		// c[0, 32-47]
		CVT_F32_BF16(c_float_0p2,0,2);

		// c[1, 0-15]
		CVT_F32_BF16(c_float_1p0,1,0);

		// c[1, 16-31]
		CVT_F32_BF16(c_float_1p1,1,1);

		// c[1, 32-47]
		CVT_F32_BF16(c_float_1p2,1,2);

		// c[2, 0-15]
		CVT_F32_BF16(c_float_2p0,2,0);

		// c[2, 16-31]
		CVT_F32_BF16(c_float_2p1,2,1);

		// c[2, 32-47]
		CVT_F32_BF16(c_float_2p2,2,2);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}	
POST_OPS_3x48_DISABLE:
	;
	
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
}

// 2x48 bf16 kernel
LPGEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_2x48)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_2x48_DISABLE,
						  &&POST_OPS_BIAS_2x48,
						  &&POST_OPS_RELU_2x48,
						  &&POST_OPS_RELU_SCALE_2x48,
						  &&POST_OPS_DOWNSCALE_2x48
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int32_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;
	__m512bh b1;
	__m512bh b2;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();
	__m512 c_float_0p1 = _mm512_setzero_ps();
	__m512 c_float_0p2 = _mm512_setzero_ps();

	__m512 c_float_1p0 = _mm512_setzero_ps();
	__m512 c_float_1p1 = _mm512_setzero_ps();
	__m512 c_float_1p2 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 1 ) );
		b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 2 ) );

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
	}
	// Handle k remainder.	
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 1 ) );
		b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 2 ) );

		// Broadcast a[0,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-47] = a[0,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
		c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );
		
		// Broadcast a[1,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 1 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[1,0-47] = a[1,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_0, b0 );
		c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_0, b1 );
		c_float_1p2 = _mm512_dpbf16_ps( c_float_1p2, a_bf16_0, b2 );
	}
	
	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	// Scale by alpha
	c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );
	c_float_0p1 = _mm512_mul_ps( selector1, c_float_0p1 );
	c_float_0p2 = _mm512_mul_ps( selector1, c_float_0p2 );

	c_float_1p0 = _mm512_mul_ps( selector1, c_float_1p0 );
	c_float_1p1 = _mm512_mul_ps( selector1, c_float_1p1 );
	c_float_1p2 = _mm512_mul_ps( selector1, c_float_1p2 );

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
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_2x48:
	{
		__m512 selector3;

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			selector1 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 0 * 16 ) );
			selector2 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 1 * 16 ) );
			selector3 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 2 * 16 ) );

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
		}
		else
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 0 ) );
			selector2 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 1 ) );

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
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_2x48:
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

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_2x48:
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

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_DOWNSCALE_2x48:
	{
		// c[0, 0-15]
		CVT_F32_BF16(c_float_0p0,0,0);

		// c[0, 16-31]
		CVT_F32_BF16(c_float_0p1,0,1);

		// c[0, 32-47]
		CVT_F32_BF16(c_float_0p2,0,2);

		// c[1, 0-15]
		CVT_F32_BF16(c_float_1p0,1,0);

		// c[1, 16-31]
		CVT_F32_BF16(c_float_1p1,1,1);

		// c[1, 32-47]
		CVT_F32_BF16(c_float_1p2,1,2);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}	
POST_OPS_2x48_DISABLE:
	;
	
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
}

// 1x48 bf16 kernel
LPGEMM_MN_FRINGE_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_1x48)
{
	static void* post_ops_labels[] =
						{
						  &&POST_OPS_1x48_DISABLE,
						  &&POST_OPS_BIAS_1x48,
						  &&POST_OPS_RELU_1x48,
						  &&POST_OPS_RELU_SCALE_1x48,
						  &&POST_OPS_DOWNSCALE_1x48
						};
	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int32_t a_kfringe_buf = 0;

	// B matrix storage bfloat type
	__m512bh b0;
	__m512bh b1;
	__m512bh b2;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;

	// Registers to use for accumulating C.
	__m512 c_float_0p0 = _mm512_setzero_ps();
	__m512 c_float_0p1 = _mm512_setzero_ps();
	__m512 c_float_0p2 = _mm512_setzero_ps();

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 1 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 1 ) );
		b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 2 ) );

		// Broadcast a[0,kr:kr+2].
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( *( int32_t* )( a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-47] = a[0,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
		c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );
	}
	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 0 ) );
		b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 1 ) );
		b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * k_full_pieces ) + ( cs_b * 2 ) );

		// Broadcast a[0,kr:kr+2].
		memcpy( &a_kfringe_buf, ( a + ( rs_a * 0 ) + ( cs_a * k_full_pieces ) ), ( k_partial_pieces * sizeof( bfloat16 ) ) );
		a_bf16_0 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

		// Perform column direction mat-mul with k = 2.
		// c[0,0-47] = a[0,kr:kr+2]*b[kr:kr+2,0-47]
		c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );
		c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
		c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );
	}
    
	// Load alpha and beta
	__m512 selector1 = _mm512_set1_ps( alpha );
	__m512 selector2 = _mm512_set1_ps( beta );

	// Scale by alpha
	c_float_0p0 = _mm512_mul_ps( selector1, c_float_0p0 );
	c_float_0p1 = _mm512_mul_ps( selector1, c_float_0p1 );
	c_float_0p2 = _mm512_mul_ps( selector1, c_float_0p2 );

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
	}
	// Post Ops
	lpgemm_post_op* post_ops_list_temp = post_ops_list;
	POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_1x48:
	{
		__m512 selector3;

		if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
			 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
		{
			selector1 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 0 * 16 ) );
			selector2 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 1 * 16 ) );
			selector3 =
				_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_op_c_j + ( 2 * 16 ) );

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_add_ps( selector2, c_float_0p1 );

			// c[0,32-47]
			c_float_0p2 = _mm512_add_ps( selector3, c_float_0p2 );
		}
		else
		{
			selector1 =
				_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1
							+ post_op_c_i + 0 ) );

			// c[0,0-15]
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			c_float_0p1 = _mm512_add_ps( selector1, c_float_0p1 );

			// c[0,32-47]
			c_float_0p2 = _mm512_add_ps( selector1, c_float_0p2 );
		}

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_1x48:
	{
		selector1 = _mm512_setzero_ps();

		// c[0,0-15]
		c_float_0p0 = _mm512_max_ps( selector1, c_float_0p0 );

		// c[0, 16-31]
		c_float_0p1 = _mm512_max_ps( selector1, c_float_0p1 );

		// c[0,32-47]
		c_float_0p2 = _mm512_max_ps( selector1, c_float_0p2 );

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_RELU_SCALE_1x48:
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

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}
POST_OPS_DOWNSCALE_1x48:
	{
		// c[0, 0-15]
		CVT_F32_BF16(c_float_0p0,0,0);

		// c[0, 16-31]
		CVT_F32_BF16(c_float_0p1,0,1);

		// c[0, 32-47]
		CVT_F32_BF16(c_float_0p2,0,2);

		POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
	}	
POST_OPS_1x48_DISABLE:
	;
	
	// Store the results.
	// c[0,0-15]
	_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 0*16 ), c_float_0p0 );

	// c[0, 16-31]
	_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 1*16 ), c_float_0p1 );

	// c[0,32-47]
	_mm512_storeu_ps( c + ( rs_c * 0 ) + ( 2*16 ), c_float_0p2 );
}
