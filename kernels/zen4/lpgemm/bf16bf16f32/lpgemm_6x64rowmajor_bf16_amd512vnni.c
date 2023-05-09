/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022-23, Advanced Micro Devices, Inc. All rights reserved.

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
   DATA, OR PROFITS; OR BUSINESS dim_tERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include <immintrin.h>
#include "blis.h"

#ifdef BLIS_ADDON_LPGEMM

#include "lpgemm_f32_kern_macros.h"

#ifdef LPGEMM_BF16_NOT_SUPPORTED

// BF16 ISA is not supported by gcc < 10. Use a dummy kernel here.
LPGEMM_MAIN_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_6x64)
{}

#else

// 6x64 bf16 kernel
LPGEMM_MAIN_KERN(bfloat16, bfloat16, float, bf16bf16f32of32_6x64)
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
						  &&POST_OPS_DOWNSCALE_6x64
						};
	dim_t MR = 6;
	dim_t NR = 64;

	dim_t m_full_pieces = m0 / MR;
	dim_t m_full_pieces_loop_limit = m_full_pieces * MR;
	dim_t m_partial_pieces = m0 % MR;

	dim_t k_full_pieces = k0 / 2;
	dim_t k_partial_pieces = k0 % 2;

	int32_t a_kfringe_buf = 0;

	if ( n0 < NR )
	{
		dim_t n0_rem = n0 % 16;

		// Split dim_to multiple smaller fringe kernels, so as to maximize
		// vectorization. Any n0 < NR(64) can be expressed as n0 = 48 + n`
		// or n0 = 32 + n` or n0 = 16 + n`, where n` < 16.
		dim_t n0_48 = n0 / 48;
		dim_t n0_32 = n0 / 32;
		dim_t n0_16 = n0 / 16;

		// KC when not multiple of 2 will have padding to make it multiple of
		// 2 in packed buffer. Also the k0 cannot be passed as the updated
		// value since A matrix is not packed and requires original k0.
		dim_t k0_updated = k0;
		k0_updated += (k0_updated & 0x1);

		if ( n0_48 == 1 )
		{
			lpgemm_rowvar_bf16bf16f32of32_6x48
				(
				 m0, k0,
				 a, rs_a, cs_a, ps_a,
				 b, ( ( rs_b / 4 ) * 3 ), cs_b,
				 c, rs_c,
				 alpha, beta,
			     post_ops_list, post_ops_attr
				);

			b = b + ( 48 * k0_updated ); // k0x48 packed contiguosly.
			c = c + 48;
			post_ops_attr.post_op_c_j += 48;
		}

		else if ( n0_32 == 1 )
		{
			lpgemm_rowvar_bf16bf16f32of32_6x32
				(
				 m0, k0,
				 a, rs_a, cs_a, ps_a,
				 b, ( ( rs_b / 4 ) * 2 ), cs_b,
				 c, rs_c,
				 alpha, beta,
			     post_ops_list, post_ops_attr
				);

			b = b + ( 32 * k0_updated ); // k0x32 packed contiguosly.
			c = c + 32;
			post_ops_attr.post_op_c_j += 32;
		}

		else if ( n0_16 == 1 )
		{
			lpgemm_rowvar_bf16bf16f32of32_6x16
				(
				 m0, k0,
				 a, rs_a, cs_a, ps_a,
				 b, ( ( rs_b / 4 ) * 1 ), cs_b,
				 c, rs_c,
				 alpha, beta,
			     post_ops_list, post_ops_attr
				);

			b = b + ( 16 * k0_updated ); // k0x16 packed contiguosly.
			c = c + 16;
			post_ops_attr.post_op_c_j += 16;
		}

		if ( n0_rem > 0 )
		{
			lpgemm_rowvar_bf16bf16f32of32_6xlt16
				(
				 m0, k0,
				 a, rs_a, cs_a, ps_a,
				 b, ( ( rs_b / 4 ) * 1 ), cs_b,
				 c, rs_c,
				 alpha, beta, n0_rem,
			     post_ops_list, post_ops_attr
				);

			// No leftover fringe after this podint.
		}
		return;
	}

	// B matrix storage bfloat type
	__m512bh b0;
	__m512bh b1;
	__m512bh b2;
	__m512bh b3;

	// A matrix storage bfloat type
	__m512bh a_bf16_0;
	__m512bh a_bf16_1;

	dim_t value;

	if(k_full_pieces > 40)
	{
		value = 40;
	}
	else
	{
		value = 0;
	}

	for ( dim_t ir = 0; ir < m_full_pieces_loop_limit; ir += MR )
	{
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

		__m512 c_float_5p0 = _mm512_setzero_ps();
		__m512 c_float_5p1 = _mm512_setzero_ps();
		__m512 c_float_5p2 = _mm512_setzero_ps();
		__m512 c_float_5p3 = _mm512_setzero_ps();

		for ( dim_t kr = 0; kr < k_full_pieces - value; kr += 1 )
		{
			// The instructions are arranged in a mixed way to reduce data
			// chain dependencies.

			b0 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 0 ) );

			// Broadcast a[0,kr:kr+2]
			a_bf16_0 = (__m512bh)_mm512_set1_epi32(
					*( int32_t* )(a + ( rs_a * 0 ) + ( cs_a * kr ) ) );

			b1 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 1 ) );
			b2 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 2 ) );
			b3 = (__m512bh)_mm512_loadu_epi16( b + ( rs_b * kr ) + ( cs_b * 3 ) );

			// Perform column direction mat-mul with k = 2.
			// c[0,0-63] = a[0,kr:kr+2]*b[kr:kr+2,0-63]
			c_float_0p0 = _mm512_dpbf16_ps( c_float_0p0, a_bf16_0, b0 );

			// Broadcast a[1,kr:kr+2].
			a_bf16_1 = (__m512bh)_mm512_set1_epi32(
					*( int32_t* )( a + ( rs_a * 1 ) + ( cs_a * kr ) ) );

			c_float_0p1 = _mm512_dpbf16_ps( c_float_0p1, a_bf16_0, b1 );
			c_float_0p2 = _mm512_dpbf16_ps( c_float_0p2, a_bf16_0, b2 );
			c_float_0p3 = _mm512_dpbf16_ps( c_float_0p3, a_bf16_0, b3 );

			// Perform column direction mat-mul with k = 2.
			// c[1,0-63] = a[1,kr:kr+2]*b[kr:kr+2,0-63]
			c_float_1p0 = _mm512_dpbf16_ps( c_float_1p0, a_bf16_1, b0 );

			// Broadcast a[2,kr:kr+2].
			a_bf16_0 = (__m512bh)_mm512_set1_epi32(
					*( int32_t* )( a + ( rs_a * 2 ) + ( cs_a * kr ) ) );

			c_float_1p1 = _mm512_dpbf16_ps( c_float_1p1, a_bf16_1, b1 );
			c_float_1p2 = _mm512_dpbf16_ps( c_float_1p2, a_bf16_1, b2 );
			c_float_1p3 = _mm512_dpbf16_ps( c_float_1p3, a_bf16_1, b3 );

			// Perform column direction mat-mul with k = 2.
			// c[2,0-63] = a[2,kr:kr+2]*b[kr:kr+2,0-63]
			c_float_2p0 = _mm512_dpbf16_ps( c_float_2p0, a_bf16_0, b0 );

			// Broadcast a[3,kr:kr+2].
			a_bf16_1 = (__m512bh)_mm512_set1_epi32(
					*( int32_t* )( a + ( rs_a * 3 ) + ( cs_a * kr ) ) );

			c_float_2p1 = _mm512_dpbf16_ps( c_float_2p1, a_bf16_0, b1 );
			c_float_2p2 = _mm512_dpbf16_ps( c_float_2p2, a_bf16_0, b2 );
			c_float_2p3 = _mm512_dpbf16_ps( c_float_2p3, a_bf16_0, b3 );

			// Perform column direction mat-mul with k = 2.
			// c[3,0-63] = a[3,kr:kr+2]*b[kr:kr+2,0-63]
			c_float_3p0 = _mm512_dpbf16_ps( c_float_3p0, a_bf16_1, b0 );

			// Broadcast a[4,kr:kr+2].
			a_bf16_0 = (__m512bh)_mm512_set1_epi32(
					*( int32_t* )( a + ( rs_a * 4 ) + ( cs_a * kr ) ) );

			c_float_3p1 = _mm512_dpbf16_ps( c_float_3p1, a_bf16_1, b1 );
			c_float_3p2 = _mm512_dpbf16_ps( c_float_3p2, a_bf16_1, b2 );
			c_float_3p3 = _mm512_dpbf16_ps( c_float_3p3, a_bf16_1, b3 );

			// Perform column direction mat-mul with k = 2.
			// c[4,0-63] = a[4,kr:kr+2]*b[kr:kr+2,0-63]
			c_float_4p0 = _mm512_dpbf16_ps( c_float_4p0, a_bf16_0, b0 );

			// Broadcast a[5,kr:kr+2].
			a_bf16_1 = (__m512bh)_mm512_set1_epi32(
					*( int32_t* )( a + ( rs_a * 5 ) + ( cs_a * kr ) ) );

			c_float_4p1 = _mm512_dpbf16_ps( c_float_4p1, a_bf16_0, b1 );
			c_float_4p2 = _mm512_dpbf16_ps( c_float_4p2, a_bf16_0, b2 );
			c_float_4p3 = _mm512_dpbf16_ps( c_float_4p3, a_bf16_0, b3 );

			// Perform column direction mat-mul with k = 2.
			// c[5,0-63] = a[5,kr:kr+2]*b[kr:kr+2,0-63]
			c_float_5p0 = _mm512_dpbf16_ps( c_float_5p0, a_bf16_1, b0 );
			c_float_5p1 = _mm512_dpbf16_ps( c_float_5p1, a_bf16_1, b1 );
			c_float_5p2 = _mm512_dpbf16_ps( c_float_5p2, a_bf16_1, b2 );
			c_float_5p3 = _mm512_dpbf16_ps( c_float_5p3, a_bf16_1, b3 );
		}

		_mm_prefetch(c + (rs_c * (ir + 0)) + (0 * 16), _MM_HINT_T1);
		_mm_prefetch(c + (rs_c * (ir + 0)) + (1 * 16), _MM_HINT_T1);
		_mm_prefetch(c + (rs_c * (ir + 0)) + (2 * 16), _MM_HINT_T1);
		_mm_prefetch(c + (rs_c * (ir + 0)) + (3 * 16), _MM_HINT_T1);

		_mm_prefetch(c + (rs_c * (ir + 1)) + (0 * 16), _MM_HINT_T1);
		_mm_prefetch(c + (rs_c * (ir + 1)) + (1 * 16), _MM_HINT_T1);
		_mm_prefetch(c + (rs_c * (ir + 1)) + (2 * 16), _MM_HINT_T1);
		_mm_prefetch(c + (rs_c * (ir + 1)) + (3 * 16), _MM_HINT_T1);

		_mm_prefetch(c + (rs_c * (ir + 2)) + (0 * 16), _MM_HINT_T1);
		_mm_prefetch(c + (rs_c * (ir + 2)) + (1 * 16), _MM_HINT_T1);
		_mm_prefetch(c + (rs_c * (ir + 2)) + (2 * 16), _MM_HINT_T1);
		_mm_prefetch(c + (rs_c * (ir + 2)) + (3 * 16), _MM_HINT_T1);

		_mm_prefetch(c + (rs_c * (ir + 3)) + (0 * 16), _MM_HINT_T1);
		_mm_prefetch(c + (rs_c * (ir + 3)) + (1 * 16), _MM_HINT_T1);
		_mm_prefetch(c + (rs_c * (ir + 3)) + (2 * 16), _MM_HINT_T1);
		_mm_prefetch(c + (rs_c * (ir + 3)) + (3 * 16), _MM_HINT_T1);

		_mm_prefetch(c + (rs_c * (ir + 4)) + (0 * 16), _MM_HINT_T1);
		_mm_prefetch(c + (rs_c * (ir + 4)) + (1 * 16), _MM_HINT_T1);
		_mm_prefetch(c + (rs_c * (ir + 4)) + (2 * 16), _MM_HINT_T1);
		_mm_prefetch(c + (rs_c * (ir + 4)) + (3 * 16), _MM_HINT_T1);

		_mm_prefetch(c + (rs_c * (ir + 5)) + (0 * 16), _MM_HINT_T1);
		_mm_prefetch(c + (rs_c * (ir + 5)) + (1 * 16), _MM_HINT_T1);
		_mm_prefetch(c + (rs_c * (ir + 5)) + (2 * 16), _MM_HINT_T1);
		_mm_prefetch(c + (rs_c * (ir + 5)) + (3 * 16), _MM_HINT_T1);

		for (dim_t kr = k_full_pieces - value; kr < k_full_pieces; kr += 1)
		{
			// The instructions are arranged in a mixed way to reduce data
			// chain dependencies.

			b0 = (__m512bh)_mm512_loadu_epi16(b + (rs_b * kr) + (cs_b * 0));

			// Broadcast a[0,kr:kr+2]
			a_bf16_0 = (__m512bh)_mm512_set1_epi32(
				*(int32_t *)(a + (rs_a * 0) + (cs_a * kr)));

			b1 = (__m512bh)_mm512_loadu_epi16(b + (rs_b * kr) + (cs_b * 1));
			b2 = (__m512bh)_mm512_loadu_epi16(b + (rs_b * kr) + (cs_b * 2));
			b3 = (__m512bh)_mm512_loadu_epi16(b + (rs_b * kr) + (cs_b * 3));

			// Perform column direction mat-mul with k = 2.
			// c[0,0-63] = a[0,kr:kr+2]*b[kr:kr+2,0-63]
			c_float_0p0 = _mm512_dpbf16_ps(c_float_0p0, a_bf16_0, b0);

			// Broadcast a[1,kr:kr+2].
			a_bf16_1 = (__m512bh)_mm512_set1_epi32(
				*(int32_t *)(a + (rs_a * 1) + (cs_a * kr)));

			c_float_0p1 = _mm512_dpbf16_ps(c_float_0p1, a_bf16_0, b1);
			c_float_0p2 = _mm512_dpbf16_ps(c_float_0p2, a_bf16_0, b2);
			c_float_0p3 = _mm512_dpbf16_ps(c_float_0p3, a_bf16_0, b3);

			// Perform column direction mat-mul with k = 2.
			// c[1,0-63] = a[1,kr:kr+2]*b[kr:kr+2,0-63]
			c_float_1p0 = _mm512_dpbf16_ps(c_float_1p0, a_bf16_1, b0);

			// Broadcast a[2,kr:kr+2].
			a_bf16_0 = (__m512bh)_mm512_set1_epi32(
				*(int32_t *)(a + (rs_a * 2) + (cs_a * kr)));

			c_float_1p1 = _mm512_dpbf16_ps(c_float_1p1, a_bf16_1, b1);
			c_float_1p2 = _mm512_dpbf16_ps(c_float_1p2, a_bf16_1, b2);
			c_float_1p3 = _mm512_dpbf16_ps(c_float_1p3, a_bf16_1, b3);

			// Perform column direction mat-mul with k = 2.
			// c[2,0-63] = a[2,kr:kr+2]*b[kr:kr+2,0-63]
			c_float_2p0 = _mm512_dpbf16_ps(c_float_2p0, a_bf16_0, b0);

			// Broadcast a[3,kr:kr+2].
			a_bf16_1 = (__m512bh)_mm512_set1_epi32(
				*(int32_t *)(a + (rs_a * 3) + (cs_a * kr)));

			c_float_2p1 = _mm512_dpbf16_ps(c_float_2p1, a_bf16_0, b1);
			c_float_2p2 = _mm512_dpbf16_ps(c_float_2p2, a_bf16_0, b2);
			c_float_2p3 = _mm512_dpbf16_ps(c_float_2p3, a_bf16_0, b3);

			// Perform column direction mat-mul with k = 2.
			// c[3,0-63] = a[3,kr:kr+2]*b[kr:kr+2,0-63]
			c_float_3p0 = _mm512_dpbf16_ps(c_float_3p0, a_bf16_1, b0);

			// Broadcast a[4,kr:kr+2].
			a_bf16_0 = (__m512bh)_mm512_set1_epi32(
				*(int32_t *)(a + (rs_a * 4) + (cs_a * kr)));

			c_float_3p1 = _mm512_dpbf16_ps(c_float_3p1, a_bf16_1, b1);
			c_float_3p2 = _mm512_dpbf16_ps(c_float_3p2, a_bf16_1, b2);
			c_float_3p3 = _mm512_dpbf16_ps(c_float_3p3, a_bf16_1, b3);

			// Perform column direction mat-mul with k = 2.
			// c[4,0-63] = a[4,kr:kr+2]*b[kr:kr+2,0-63]
			c_float_4p0 = _mm512_dpbf16_ps(c_float_4p0, a_bf16_0, b0);

			// Broadcast a[5,kr:kr+2].
			a_bf16_1 = (__m512bh)_mm512_set1_epi32(
				*(int32_t *)(a + (rs_a * 5) + (cs_a * kr)));

			c_float_4p1 = _mm512_dpbf16_ps(c_float_4p1, a_bf16_0, b1);
			c_float_4p2 = _mm512_dpbf16_ps(c_float_4p2, a_bf16_0, b2);
			c_float_4p3 = _mm512_dpbf16_ps(c_float_4p3, a_bf16_0, b3);

			// Perform column direction mat-mul with k = 2.
			// c[5,0-63] = a[5,kr:kr+2]*b[kr:kr+2,0-63]
			c_float_5p0 = _mm512_dpbf16_ps(c_float_5p0, a_bf16_1, b0);
			c_float_5p1 = _mm512_dpbf16_ps(c_float_5p1, a_bf16_1, b1);
			c_float_5p2 = _mm512_dpbf16_ps(c_float_5p2, a_bf16_1, b2);
			c_float_5p3 = _mm512_dpbf16_ps(c_float_5p3, a_bf16_1, b3);
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

			// Broadcast a[5,kr:kr+2].
			memcpy
				(
				 &a_kfringe_buf,
				 ( a + ( rs_a * 5 ) + ( cs_a * k_full_pieces ) ),
				 ( k_partial_pieces * sizeof( bfloat16 ) )
				);
			a_bf16_1 = (__m512bh)_mm512_set1_epi32( a_kfringe_buf );

			c_float_4p1 = _mm512_dpbf16_ps( c_float_4p1, a_bf16_0, b1 );
			c_float_4p2 = _mm512_dpbf16_ps( c_float_4p2, a_bf16_0, b2 );
			c_float_4p3 = _mm512_dpbf16_ps( c_float_4p3, a_bf16_0, b3 );

			// Perform column direction mat-mul with k = 2.
			// c[5,0-63] = a[5,kr:kr+2]*b[kr:kr+2,0-63]
			c_float_5p0 = _mm512_dpbf16_ps( c_float_5p0, a_bf16_1, b0 );
			c_float_5p1 = _mm512_dpbf16_ps( c_float_5p1, a_bf16_1, b1 );
			c_float_5p2 = _mm512_dpbf16_ps( c_float_5p2, a_bf16_1, b2 );
			c_float_5p3 = _mm512_dpbf16_ps( c_float_5p3, a_bf16_1, b3 );
		}

		// Load alpha and beta
		__m512 selector1 = _mm512_set1_ps ( alpha );
		__m512 selector2 = _mm512_set1_ps ( beta );

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

		c_float_5p0 = _mm512_mul_ps( selector1, c_float_5p0 );
		c_float_5p1 = _mm512_mul_ps( selector1, c_float_5p1 );
		c_float_5p2 = _mm512_mul_ps( selector1, c_float_5p2 );
		c_float_5p3 = _mm512_mul_ps( selector1, c_float_5p3 );

		// Scale C by beta.
		if ( beta != 0 )
		{
			// c[0,0-15]
			selector1 = _mm512_loadu_ps( c + ( rs_c * ( ir + 0 ) ) + ( 0*16 ) );
			selector1 = _mm512_mul_ps( selector2, selector1 );
			c_float_0p0 = _mm512_add_ps( selector1, c_float_0p0 );

			// c[0, 16-31]
			selector1 = _mm512_loadu_ps( c + ( rs_c * ( ir + 0 ) ) + ( 1*16 ) );
			selector1 = _mm512_mul_ps( selector2, selector1 );
			c_float_0p1 = _mm512_add_ps( selector1, c_float_0p1 );

			// c[0,32-47]
			selector1 = _mm512_loadu_ps( c + ( rs_c * ( ir + 0 ) ) + ( 2*16 ) );
			selector1 = _mm512_mul_ps( selector2, selector1 );
			c_float_0p2 = _mm512_add_ps( selector1, c_float_0p2 );

			// c[0,48-63]
			selector1 = _mm512_loadu_ps( c + ( rs_c * ( ir + 0 ) ) + ( 3*16 ) );
			selector1 = _mm512_mul_ps( selector2, selector1 );
			c_float_0p3 = _mm512_add_ps( selector1, c_float_0p3 );

			// c[1,0-15]
			selector1 = _mm512_loadu_ps( c + ( rs_c * ( ir + 1 ) ) + ( 0*16 ) );
			selector1 = _mm512_mul_ps( selector2, selector1 );
			c_float_1p0 = _mm512_add_ps( selector1, c_float_1p0 );

			// c[1,16-31]
			selector1 = _mm512_loadu_ps( c + ( rs_c * ( ir + 1 ) ) + ( 1*16 ) );
			selector1 = _mm512_mul_ps( selector2, selector1 );
			c_float_1p1 = _mm512_add_ps( selector1, c_float_1p1 );

			// c[1,32-47]
			selector1 = _mm512_loadu_ps( c + ( rs_c * ( ir + 1 ) ) + ( 2*16 ) );
			selector1 = _mm512_mul_ps( selector2, selector1 );
			c_float_1p2 = _mm512_add_ps( selector1, c_float_1p2 );

			// c[1,48-63]
			selector1 = _mm512_loadu_ps( c + ( rs_c * ( ir + 1 ) ) + ( 3*16 ) );
			selector1 = _mm512_mul_ps( selector2, selector1 );
			c_float_1p3 = _mm512_add_ps( selector1, c_float_1p3 );

			// c[2,0-15]
			selector1 = _mm512_loadu_ps( c + ( rs_c * ( ir + 2 ) ) + ( 0*16 ) );
			selector1 = _mm512_mul_ps( selector2, selector1 );
			c_float_2p0 = _mm512_add_ps( selector1, c_float_2p0 );

			// c[2,16-31]
			selector1 = _mm512_loadu_ps( c + ( rs_c * ( ir + 2 ) ) + ( 1*16 ) );
			selector1 = _mm512_mul_ps( selector2, selector1 );
			c_float_2p1 = _mm512_add_ps( selector1, c_float_2p1 );

			// c[2,32-47]
			selector1 = _mm512_loadu_ps( c + ( rs_c * ( ir + 2 ) ) + ( 2*16 ) );
			selector1 = _mm512_mul_ps( selector2, selector1 );
			c_float_2p2 = _mm512_add_ps( selector1, c_float_2p2 );

			// c[2,48-63]
			selector1 = _mm512_loadu_ps( c + ( rs_c * ( ir + 2 ) ) + ( 3*16 ) );
			selector1 = _mm512_mul_ps( selector2, selector1 );
			c_float_2p3 = _mm512_add_ps( selector1, c_float_2p3 );

			// c[3,0-15]
			selector1 = _mm512_loadu_ps( c + ( rs_c * ( ir + 3 ) ) + ( 0*16 ) );
			selector1 = _mm512_mul_ps( selector2, selector1 );
			c_float_3p0 = _mm512_add_ps( selector1, c_float_3p0 );

			// c[3,16-31]
			selector1 = _mm512_loadu_ps( c + ( rs_c * ( ir + 3 ) ) + ( 1*16 ) );
			selector1 = _mm512_mul_ps( selector2, selector1 );
			c_float_3p1 = _mm512_add_ps( selector1, c_float_3p1 );

			// c[3,32-47]
			selector1 = _mm512_loadu_ps( c + ( rs_c * ( ir + 3 ) ) + ( 2*16 ) );
			selector1 = _mm512_mul_ps( selector2, selector1 );
			c_float_3p2 = _mm512_add_ps( selector1, c_float_3p2 );

			// c[3,48-63]
			selector1 = _mm512_loadu_ps( c + ( rs_c * ( ir + 3 ) ) + ( 3*16 ) );
			selector1 = _mm512_mul_ps( selector2, selector1 );
			c_float_3p3 = _mm512_add_ps( selector1, c_float_3p3 );

			// c[4,0-15]
			selector1 = _mm512_loadu_ps( c + ( rs_c * ( ir + 4 ) ) + ( 0*16 ) );
			selector1 = _mm512_mul_ps( selector2, selector1 );
			c_float_4p0 = _mm512_add_ps( selector1, c_float_4p0 );

			// c[4,16-31]
			selector1 = _mm512_loadu_ps( c + ( rs_c * ( ir + 4 ) ) + ( 1*16 ) );
			selector1 = _mm512_mul_ps( selector2, selector1 );
			c_float_4p1 = _mm512_add_ps( selector1, c_float_4p1 );

			// c[4,32-47]
			selector1 = _mm512_loadu_ps( c + ( rs_c * ( ir + 4 ) ) + ( 2*16 ) );
			selector1 = _mm512_mul_ps( selector2, selector1 );
			c_float_4p2 = _mm512_add_ps( selector1, c_float_4p2 );

			// c[4,48-63]
			selector1 = _mm512_loadu_ps( c + ( rs_c * ( ir + 4 ) ) + ( 3*16 ) );
			selector1 = _mm512_mul_ps( selector2, selector1 );
			c_float_4p3 = _mm512_add_ps( selector1, c_float_4p3 );

			// c[5,0-15]
			selector1 = _mm512_loadu_ps( c + ( rs_c * ( ir + 5 ) ) + ( 0*16 ) );
			selector1 = _mm512_mul_ps( selector2, selector1 );
			c_float_5p0 = _mm512_add_ps( selector1, c_float_5p0 );

			// c[5,16-31]
			selector1 = _mm512_loadu_ps( c + ( rs_c * ( ir + 5 ) ) + ( 1*16 ) );
			selector1 = _mm512_mul_ps( selector2, selector1 );
			c_float_5p1 = _mm512_add_ps( selector1, c_float_5p1 );

			// c[5,32-47]
			selector1 = _mm512_loadu_ps( c + ( rs_c * ( ir + 5 ) ) + ( 2*16 ) );
			selector1 = _mm512_mul_ps( selector2, selector1 );
			c_float_5p2 = _mm512_add_ps( selector1, c_float_5p2 );

			// c[5,48-63]
			selector1 = _mm512_loadu_ps( c + ( rs_c * ( ir + 5 ) ) + ( 3*16 ) );
			selector1 = _mm512_mul_ps( selector2, selector1 );
			c_float_5p3 = _mm512_add_ps( selector1, c_float_5p3 );
		}
		// Post Ops
		lpgemm_post_op* post_ops_list_temp = post_ops_list;
		POST_OP_LABEL_LASTK_SAFE_JUMP
POST_OPS_BIAS_6x64:
		{
			__m512 selector3;
			__m512 selector4;

			if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
				 ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
			{
				selector1 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 0 * 16 ) );
				selector2 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 1 * 16 ) );
				selector3 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 2 * 16 ) );
				selector4 =
					_mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_j + ( 3 * 16 ) );

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

				// c[5,0-15]
				c_float_5p0 = _mm512_add_ps( selector1, c_float_5p0 );

				// c[5, 16-31]
				c_float_5p1 = _mm512_add_ps( selector2, c_float_5p1 );

				// c[5,32-47]
				c_float_5p2 = _mm512_add_ps( selector3, c_float_5p2 );

				// c[5,48-63]
				c_float_5p3 = _mm512_add_ps( selector4, c_float_5p3 );
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
							post_ops_attr.post_op_c_i + 0 ) );
				selector2 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_i + 1 ) );
				selector3 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_i + 2 ) );
				selector4 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_i + 3 ) );
				__m512 selector5 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_i + 4 ) );
				__m512 selector6 =
					_mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
							post_ops_attr.post_op_c_i + 5 ) );

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

				// c[5,0-15]
				c_float_5p0 = _mm512_add_ps( selector6, c_float_5p0 );

				// c[5, 16-31]
				c_float_5p1 = _mm512_add_ps( selector6, c_float_5p1 );

				// c[5,32-47]
				c_float_5p2 = _mm512_add_ps( selector6, c_float_5p2 );

				// c[5,48-63]
				c_float_5p3 = _mm512_add_ps( selector6, c_float_5p3 );
			}

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_6x64:
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

			// c[5,0-15]
			c_float_5p0 = _mm512_max_ps( selector1, c_float_5p0 );

			// c[5,16-31]
			c_float_5p1 = _mm512_max_ps( selector1, c_float_5p1 );

			// c[5,32-47]
			c_float_5p2 = _mm512_max_ps( selector1, c_float_5p2 );

			// c[5,48-63]
			c_float_5p3 = _mm512_max_ps( selector1, c_float_5p3 );

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_RELU_SCALE_6x64:
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

			// c[5, 0-15]
			RELU_SCALE_OP_F32_AVX512(c_float_5p0)

			// c[5, 16-31]
			RELU_SCALE_OP_F32_AVX512(c_float_5p1)

			// c[5, 32-47]
			RELU_SCALE_OP_F32_AVX512(c_float_5p2)

			// c[5, 48-63]
			RELU_SCALE_OP_F32_AVX512(c_float_5p3)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_TANH_6x64:
		{
			__m512 dn, z, x, r2, r, x_tanh;
			__m512i q;

			// c[0, 0-15]
			GELU_TANH_F32_AVX512(c_float_0p0, r, r2, x, z, dn, x_tanh, q)

			// c[0, 16-31]
			GELU_TANH_F32_AVX512(c_float_0p1, r, r2, x, z, dn, x_tanh, q)

			// c[0, 32-47]
			GELU_TANH_F32_AVX512(c_float_0p2, r, r2, x, z, dn, x_tanh, q)

			// c[0, 48-63]
			GELU_TANH_F32_AVX512(c_float_0p3, r, r2, x, z, dn, x_tanh, q)

			// c[1, 0-15]
			GELU_TANH_F32_AVX512(c_float_1p0, r, r2, x, z, dn, x_tanh, q)

			// c[1, 16-31]
			GELU_TANH_F32_AVX512(c_float_1p1, r, r2, x, z, dn, x_tanh, q)

			// c[1, 32-47]
			GELU_TANH_F32_AVX512(c_float_1p2, r, r2, x, z, dn, x_tanh, q)

			// c[1, 48-63]
			GELU_TANH_F32_AVX512(c_float_1p3, r, r2, x, z, dn, x_tanh, q)

			// c[2, 0-15]
			GELU_TANH_F32_AVX512(c_float_2p0, r, r2, x, z, dn, x_tanh, q)

			// c[2, 16-31]
			GELU_TANH_F32_AVX512(c_float_2p1, r, r2, x, z, dn, x_tanh, q)

			// c[2, 32-47]
			GELU_TANH_F32_AVX512(c_float_2p2, r, r2, x, z, dn, x_tanh, q)

			// c[2, 48-63]
			GELU_TANH_F32_AVX512(c_float_2p3, r, r2, x, z, dn, x_tanh, q)

			// c[3, 0-15]
			GELU_TANH_F32_AVX512(c_float_3p0, r, r2, x, z, dn, x_tanh, q)

			// c[3, 16-31]
			GELU_TANH_F32_AVX512(c_float_3p1, r, r2, x, z, dn, x_tanh, q)

			// c[3, 32-47]
			GELU_TANH_F32_AVX512(c_float_3p2, r, r2, x, z, dn, x_tanh, q)

			// c[3, 48-63]
			GELU_TANH_F32_AVX512(c_float_3p3, r, r2, x, z, dn, x_tanh, q)

			// c[4, 0-15]
			GELU_TANH_F32_AVX512(c_float_4p0, r, r2, x, z, dn, x_tanh, q)

			// c[4, 16-31]
			GELU_TANH_F32_AVX512(c_float_4p1, r, r2, x, z, dn, x_tanh, q)

			// c[4, 32-47]
			GELU_TANH_F32_AVX512(c_float_4p2, r, r2, x, z, dn, x_tanh, q)

			// c[4, 48-63]
			GELU_TANH_F32_AVX512(c_float_4p3, r, r2, x, z, dn, x_tanh, q)

			// c[5, 0-15]
			GELU_TANH_F32_AVX512(c_float_5p0, r, r2, x, z, dn, x_tanh, q)

			// c[5, 16-31]
			GELU_TANH_F32_AVX512(c_float_5p1, r, r2, x, z, dn, x_tanh, q)

			// c[5, 32-47]
			GELU_TANH_F32_AVX512(c_float_5p2, r, r2, x, z, dn, x_tanh, q)

			// c[5, 48-63]
			GELU_TANH_F32_AVX512(c_float_5p3, r, r2, x, z, dn, x_tanh, q)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}
POST_OPS_GELU_ERF_6x64:
		{
			__m512 x, r, x_erf;

			// c[0, 0-15]
			GELU_ERF_F32_AVX512(c_float_0p0, r, x, x_erf)

			// c[0, 16-31]
			GELU_ERF_F32_AVX512(c_float_0p1, r, x, x_erf)

			// c[0, 32-47]
			GELU_ERF_F32_AVX512(c_float_0p2, r, x, x_erf)

			// c[0, 48-63]
			GELU_ERF_F32_AVX512(c_float_0p3, r, x, x_erf)

			// c[1, 0-15]
			GELU_ERF_F32_AVX512(c_float_1p0, r, x, x_erf)

			// c[1, 16-31]
			GELU_ERF_F32_AVX512(c_float_1p1, r, x, x_erf)

			// c[1, 32-47]
			GELU_ERF_F32_AVX512(c_float_1p2, r, x, x_erf)

			// c[1, 48-63]
			GELU_ERF_F32_AVX512(c_float_1p3, r, x, x_erf)

			// c[2, 0-15]
			GELU_ERF_F32_AVX512(c_float_2p0, r, x, x_erf)

			// c[2, 16-31]
			GELU_ERF_F32_AVX512(c_float_2p1, r, x, x_erf)

			// c[2, 32-47]
			GELU_ERF_F32_AVX512(c_float_2p2, r, x, x_erf)

			// c[2, 48-63]
			GELU_ERF_F32_AVX512(c_float_2p3, r, x, x_erf)

			// c[3, 0-15]
			GELU_ERF_F32_AVX512(c_float_3p0, r, x, x_erf)

			// c[3, 16-31]
			GELU_ERF_F32_AVX512(c_float_3p1, r, x, x_erf)

			// c[3, 32-47]
			GELU_ERF_F32_AVX512(c_float_3p2, r, x, x_erf)

			// c[3, 48-63]
			GELU_ERF_F32_AVX512(c_float_3p3, r, x, x_erf)

			// c[4, 0-15]
			GELU_ERF_F32_AVX512(c_float_4p0, r, x, x_erf)

			// c[4, 16-31]
			GELU_ERF_F32_AVX512(c_float_4p1, r, x, x_erf)

			// c[4, 32-47]
			GELU_ERF_F32_AVX512(c_float_4p2, r, x, x_erf)

			// c[4, 48-63]
			GELU_ERF_F32_AVX512(c_float_4p3, r, x, x_erf)

			// c[5, 0-15]
			GELU_ERF_F32_AVX512(c_float_5p0, r, x, x_erf)

			// c[5, 16-31]
			GELU_ERF_F32_AVX512(c_float_5p1, r, x, x_erf)

			// c[5, 32-47]
			GELU_ERF_F32_AVX512(c_float_5p2, r, x, x_erf)

			// c[5, 48-63]
			GELU_ERF_F32_AVX512(c_float_5p3, r, x, x_erf)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}

POST_OPS_CLIP_6x64:
		{
			__m512 min = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
			__m512 max = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

			// c[0, 0-15]
			CLIP_F32_AVX512(c_float_0p0, min, max)

			// c[0, 16-31]
			CLIP_F32_AVX512(c_float_0p1, min, max)

			// c[0, 32-47]
			CLIP_F32_AVX512(c_float_0p2, min, max)

			// c[0, 48-63]
			CLIP_F32_AVX512(c_float_0p3, min, max)

			// c[1, 0-15]
			CLIP_F32_AVX512(c_float_1p0, min, max)

			// c[1, 16-31]
			CLIP_F32_AVX512(c_float_1p1, min, max)

			// c[1, 32-47]
			CLIP_F32_AVX512(c_float_1p2, min, max)

			// c[1, 48-63]
			CLIP_F32_AVX512(c_float_1p3, min, max)

			// c[2, 0-15]
			CLIP_F32_AVX512(c_float_2p0, min, max)

			// c[2, 16-31]
			CLIP_F32_AVX512(c_float_2p1, min, max)

			// c[2, 32-47]
			CLIP_F32_AVX512(c_float_2p2, min, max)

			// c[2, 48-63]
			CLIP_F32_AVX512(c_float_2p3, min, max)

			// c[3, 0-15]
			CLIP_F32_AVX512(c_float_3p0, min, max)

			// c[3, 16-31]
			CLIP_F32_AVX512(c_float_3p1, min, max)

			// c[3, 32-47]
			CLIP_F32_AVX512(c_float_3p2, min, max)

			// c[3, 48-63]
			CLIP_F32_AVX512(c_float_3p3, min, max)

			// c[4, 0-15]
			CLIP_F32_AVX512(c_float_4p0, min, max)

			// c[4, 16-31]
			CLIP_F32_AVX512(c_float_4p1, min, max)

			// c[4, 32-47]
			CLIP_F32_AVX512(c_float_4p2, min, max)

			// c[4, 48-63]
			CLIP_F32_AVX512(c_float_4p3, min, max)

			// c[5, 0-15]
			CLIP_F32_AVX512(c_float_5p0, min, max)

			// c[5, 16-31]
			CLIP_F32_AVX512(c_float_5p1, min, max)

			// c[5, 32-47]
			CLIP_F32_AVX512(c_float_5p2, min, max)

			// c[5, 48-63]
			CLIP_F32_AVX512(c_float_5p3, min, max)

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
		}

POST_OPS_DOWNSCALE_6x64:
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

			// c[5, 0-15]
			CVT_F32_BF16(c_float_5p0,5,0);

			// c[5, 16-31]
			CVT_F32_BF16(c_float_5p1,5,1);

			// c[5, 32-47]
			CVT_F32_BF16(c_float_5p2,5,2);

			// c[5, 48-63]
			CVT_F32_BF16(c_float_5p3,5,3);

			POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
}

POST_OPS_6x64_DISABLE:
		;

		// Store the results.
		// c[0,0-15]
		_mm512_storeu_ps( c + ( rs_c * ( ir + 0 ) ) + ( 0*16 ), c_float_0p0 );

		// c[0, 16-31]
		_mm512_storeu_ps( c + ( rs_c * ( ir + 0 ) ) + ( 1*16 ), c_float_0p1 );

		// c[0,32-47]
		_mm512_storeu_ps( c + ( rs_c * ( ir + 0 ) ) + ( 2*16 ), c_float_0p2 );

		// c[0,48-63]
		_mm512_storeu_ps( c + ( rs_c * ( ir + 0 ) ) + ( 3*16 ), c_float_0p3 );

		// c[1,0-15]
		_mm512_storeu_ps( c + ( rs_c * ( ir + 1 ) ) + ( 0*16 ), c_float_1p0 );

		// c[1,16-31]
		_mm512_storeu_ps( c + ( rs_c * ( ir + 1 ) ) + ( 1*16 ), c_float_1p1 );

		// c[1,32-47]
		_mm512_storeu_ps( c + ( rs_c * ( ir + 1 ) ) + ( 2*16 ), c_float_1p2 );

		// c[1,48-63]
		_mm512_storeu_ps( c + ( rs_c * ( ir + 1 ) ) + ( 3*16 ), c_float_1p3 );

		// c[2,0-15]
		_mm512_storeu_ps( c + ( rs_c * ( ir + 2 ) ) + ( 0*16 ), c_float_2p0 );

		// c[2,16-31]
		_mm512_storeu_ps( c + ( rs_c * ( ir + 2 ) ) + ( 1*16 ), c_float_2p1 );

		// c[2,32-47]
		_mm512_storeu_ps( c + ( rs_c * ( ir + 2 ) ) + ( 2*16 ), c_float_2p2 );

		// c[2,48-63]
		_mm512_storeu_ps( c + ( rs_c * ( ir + 2 ) ) + ( 3*16 ), c_float_2p3 );

		// c[3,0-15]
		_mm512_storeu_ps( c + ( rs_c * ( ir + 3 ) ) + ( 0*16 ), c_float_3p0 );

		// c[3,16-31]
		_mm512_storeu_ps( c + ( rs_c * ( ir + 3 ) ) + ( 1*16 ), c_float_3p1 );

		// c[3,32-47]
		_mm512_storeu_ps( c + ( rs_c * ( ir + 3 ) ) + ( 2*16 ), c_float_3p2 );

		// c[3,48-63]
		_mm512_storeu_ps( c + ( rs_c * ( ir + 3 ) ) + ( 3*16 ), c_float_3p3 );

		// c[4,0-15]
		_mm512_storeu_ps( c + ( rs_c * ( ir + 4 ) ) + ( 0*16 ), c_float_4p0 );

		// c[4,16-31]
		_mm512_storeu_ps( c + ( rs_c * ( ir + 4 ) ) + ( 1*16 ), c_float_4p1 );

		// c[4,32-47]
		_mm512_storeu_ps( c + ( rs_c * ( ir + 4 ) ) + ( 2*16 ), c_float_4p2 );

		// c[4,48-63]
		_mm512_storeu_ps( c + ( rs_c * ( ir + 4 ) ) + ( 3*16 ), c_float_4p3 );

		// c[5,0-15]
		_mm512_storeu_ps( c + ( rs_c * ( ir + 5 ) ) + ( 0*16 ), c_float_5p0 );

		// c[5,16-31]
		_mm512_storeu_ps( c + ( rs_c * ( ir + 5 ) ) + ( 1*16 ), c_float_5p1 );

		// c[5,32-47]
		_mm512_storeu_ps( c + ( rs_c * ( ir + 5 ) ) + ( 2*16 ), c_float_5p2 );

		// c[5,48-63]
		_mm512_storeu_ps( c + ( rs_c * ( ir + 5 ) ) + ( 3*16 ), c_float_5p3 );

		a = a + ( MR * ps_a );
		post_ops_attr.post_op_c_i += MR;
	}

	if ( m_partial_pieces > 0 )
	{
		if ( m_partial_pieces == 5 )
		{
			// In cases where A matrix is packed cs_a is set to 12, since the
			// next column in a given row is accessed after 2*6 elements, where
			// 6 is MR and 2 elements are broadcasted each time from A (bf16).
			// In fringe case, where m < MR, the next column will be after m'*2
			// elements, and subsequently following adjustment of cs_a is
			// required before calling m fringe kernels.
			dim_t cs_a_use = ( cs_a == 2 ) ? 2 : ( ( cs_a / 6 ) * 5 );
			lpgemm_rowvar_bf16bf16f32of32_5x64
			(
			  k0,
			  a, rs_a, cs_a_use,
			  b, rs_b, cs_b,
			  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
			  alpha, beta,
			  post_ops_list, post_ops_attr
			);
		}
		else if ( m_partial_pieces == 4 )
		{
			dim_t cs_a_use = ( cs_a == 2 ) ? 2 : ( ( cs_a / 6 ) * 4 );
			lpgemm_rowvar_bf16bf16f32of32_4x64
			(
			  k0,
			  a, rs_a, cs_a_use,
			  b, rs_b, cs_b,
			  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
			  alpha, beta,
			  post_ops_list, post_ops_attr
			);
		}
		else if ( m_partial_pieces == 3 )
		{
			dim_t cs_a_use = ( cs_a == 2 ) ? 2 : ( ( cs_a / 6 ) * 3 );
			lpgemm_rowvar_bf16bf16f32of32_3x64
			(
			  k0,
			  a, rs_a, cs_a_use,
			  b, rs_b, cs_b,
			  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
			  alpha, beta,
			  post_ops_list, post_ops_attr
			);
		}
		else if ( m_partial_pieces == 2 )
		{
			dim_t cs_a_use = ( cs_a == 2 ) ? 2 : ( ( cs_a / 6 ) * 2 );
			lpgemm_rowvar_bf16bf16f32of32_2x64
			(
			  k0,
			  a, rs_a, cs_a_use,
			  b, rs_b, cs_b,
			  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
			  alpha, beta,
			  post_ops_list, post_ops_attr
			);
		}
		else if ( m_partial_pieces == 1 )
		{
			dim_t cs_a_use = ( cs_a == 2 ) ? 2 : ( ( cs_a / 6 ) * 1 );
			lpgemm_rowvar_bf16bf16f32of32_1x64
			(
			  k0,
			  a, rs_a, cs_a_use,
			  b, rs_b, cs_b,
			  ( c + ( rs_c * m_full_pieces_loop_limit ) ), rs_c,
			  alpha, beta,
			  post_ops_list, post_ops_attr
			);
		}
	}
}

#endif //LPGEMM_BF16_NOT_SUPPORTED
#endif
