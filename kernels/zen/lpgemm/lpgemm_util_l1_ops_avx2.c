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
#include "blis.h"

#ifdef BLIS_ADDON_LPGEMM

#include "math_utils_avx2.h"
#include "gelu_avx2.h"

// TANH GeLU (x) = 0.5 * x * ( 1 + tanh ( 0.797884 * ( x + ( 0.044715 * x^3 ) ) ) )
#define GELU_TANH_NONVEC(in_val) \
	( in_val ) = 0.5 * ( ( double )( in_val ) ) * \
	( \
	  1 + tanhf \
	  ( \
	    0.797884 * \
	    ( \
	   	  ( double )( in_val ) + \
		  ( \
		    0.044715 * \
		    ( ( double )( in_val ) * ( double )( in_val ) * ( double )( in_val ) ) \
		  ) \
	    ) \
	  ) \
	); \

/* ERF GeLU (x) = 0.5* x * (1 + erf (x * 0.707107 ))  */
#define GELU_ERF_NONVEC(in_val) \
	( in_val ) = 0.5 * ( double )( in_val ) * \
		( 1 + erff( ( double )( in_val ) * 0.707107 ) ); \

LPGEMM_UTIL_L1_OP_KERNEL(float,f32_gelu_tanh_avx2)
{
	if ( incx == 1 )
	{
		// Break the input into avx2 + sse + non-vetorized blocks.
		dim_t n_part8 = ( n / 8 ) * 8;
		dim_t n_part8_rem = n - n_part8;
		dim_t n_part4 = n_part8_rem / 4;
		dim_t n_part4_rem = n_part8_rem - ( n_part4 * 4 );

		dim_t idx = 0;
		__m256 ymm0, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;
		__m256i ymm10i;
		// avx2 block loop.
		for ( idx = 0; idx < n_part8; idx += 8 )
		{
			ymm0 = _mm256_loadu_ps( x + idx );

			GELU_TANH_F32_AVX2_DEF(ymm0, ymm10, ymm11, ymm12, \
							ymm13, ymm14, ymm15, ymm10i);

			_mm256_storeu_ps( x + idx, ymm0 );
		}

		// sse remainder block.
		if ( n_part4 == 1 )
		{
			__m128 xmm0, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15;
			__m128i xmm10i;

			xmm0 = _mm_loadu_ps( x + idx );

			GELU_TANH_F32_SSE_DEF(xmm0, xmm10, xmm11, xmm12, \
							xmm13, xmm14, xmm15, xmm10i);

			_mm_storeu_ps( x + idx, xmm0 );

			idx = idx + 4;
		}
		// non vector remainder block.
		if ( n_part4_rem > 0 )
		{
			for ( dim_t rem_idx = 0; rem_idx < n_part4_rem; ++rem_idx )
			{
				float temp_val = *( x + idx + rem_idx );
				*( x + idx + rem_idx ) = GELU_TANH_NONVEC(temp_val);
			}
		}
	}
	// For non unit increment, use non-vectorized code.
	else
	{
		dim_t n_incx = n * incx;
		for ( dim_t idx = 0; idx < n_incx; idx += incx )
		{
			float temp_val = *( x + idx );
			*( x + idx ) = GELU_TANH_NONVEC(temp_val);
		}
	}
}

LPGEMM_UTIL_L1_OP_KERNEL(float,f32_gelu_erf_avx2)
{
	if ( incx == 1 )
	{
		// Break the input into avx2 + sse + non-vetorized blocks.
		dim_t n_part8 = ( n / 8 ) * 8;
		dim_t n_part8_rem = n - n_part8;
		dim_t n_part4 = n_part8_rem / 4;
		dim_t n_part4_rem = n_part8_rem - ( n_part4 * 4 );

		dim_t idx = 0;
		__m256 ymm0, ymm10, ymm11, ymm12;
		// avx2 block loop.
		for ( idx = 0; idx < n_part8; idx += 8 )
		{
			ymm0 = _mm256_loadu_ps( x + idx );

			GELU_ERF_F32_AVX2_DEF(ymm0, ymm10, ymm11, ymm12);

			_mm256_storeu_ps( x + idx, ymm0 );
		}

		// sse remainder block.
		if ( n_part4 == 1 )
		{
			__m128 xmm0, xmm10, xmm11, xmm12;

			xmm0 = _mm_loadu_ps( x + idx );

			GELU_ERF_F32_SSE_DEF(xmm0, xmm10, xmm11, xmm12);

			_mm_storeu_ps( x + idx, xmm0 );

			idx = idx + 4;
		}
		// non vector remainder block.
		if ( n_part4_rem > 0 )
		{
			for ( dim_t rem_idx = 0; rem_idx < n_part4_rem; ++rem_idx )
			{
				float temp_val = *( x + idx + rem_idx );
				*( x + idx + rem_idx ) = GELU_ERF_NONVEC(temp_val);
			}
		}
	}
	// For non unit increment, use non-vectorized code.
	else
	{
		dim_t n_incx = n * incx;
		for ( dim_t idx = 0; idx < n_incx; idx += incx )
		{
			float temp_val = *( x + idx );
			*( x + idx ) = GELU_ERF_NONVEC(temp_val);
		}
	}
}

LPGEMM_UTIL_L1_OP_KERNEL(float,f32_softmax_avx2)
{
	if ( incx == 1 )
	{
		double exp_sum[2] = { 0.0 };

		// Break the input into avx2 + sse + non-vetorized blocks.
		dim_t n_part8 = ( n / 8 ) * 8;
		dim_t n_part8_rem = n - n_part8;
		dim_t n_part4 = n_part8_rem / 4;
		dim_t n_part4_rem = n_part8_rem - ( n_part4 * 4 );

		dim_t idx = 0;
		__m256 ymm0, ymm10, ymm11, ymm12, ymm13, ymm10out;
		__m256i ymm10outi;
		__m128 xmm0, xmm1;
		__m256d ymm0d, ymm1d;
		__m128d xmm0d, xmm1d;

		// Exp reduction of the array - avx2 block.
		for ( idx = 0; idx < n_part8; idx += 8 )
		{
			ymm0 = _mm256_loadu_ps( x + idx );

			EXPF_AVX2(ymm0, ymm10, ymm11, ymm12, ymm13, ymm10outi); // zmm10out is the output
			ymm10out = _mm256_castsi256_ps( ymm10outi );

			// Reduction to be done as double data type.
			xmm0 = _mm256_castps256_ps128( ymm10out );
			xmm1 = _mm256_extractf128_ps( ymm10out, 0x1 );
			ymm0d = _mm256_cvtps_pd( xmm0 );
			ymm1d = _mm256_cvtps_pd( xmm1 );
			ymm0d = _mm256_add_pd( ymm0d, ymm1d );

			xmm0d = _mm256_castpd256_pd128( ymm0d );
			xmm1d = _mm256_extractf128_pd( ymm0d, 0x1 );
			xmm0d = _mm_add_pd( xmm0d, xmm1d );

			xmm1d = _mm_permute_pd( xmm0d, 0x01);
			xmm0d = _mm_add_pd( xmm0d, xmm1d );
			exp_sum[1] = _mm_cvtsd_f64( xmm0d );
			exp_sum[0] += exp_sum[1];
		}
		// sse remainder block.
		if ( n_part4 == 1 )
		{
			__m128 xmm10, xmm11, xmm12, xmm10out;
			__m128i xmm10outi;

			xmm0 = _mm_loadu_ps( x + idx );

			EXPF_SSE(xmm0, xmm1, xmm10, xmm11, xmm12, xmm10outi);
			xmm10out = _mm_castsi128_ps( xmm10outi );

			xmm0d = _mm_cvtps_pd( xmm10out );
			xmm1d = _mm_cvtps_pd( _mm_permute_ps( xmm10out, 0x4E ) ); //0 1 2 3 -> 2 3 0 1
			xmm0d = _mm_add_pd( xmm0d, xmm1d );

			xmm1d = _mm_permute_pd( xmm0d, 0x01);
			xmm0d = _mm_add_pd( xmm0d, xmm1d );
			exp_sum[1] = _mm_cvtsd_f64( xmm0d );
			exp_sum[0] += exp_sum[1];

			idx = idx + 4;
		}
		// non vector remainder block.
		if ( n_part4_rem > 0 )
		{
			float temp_fl_buf[4] = { 0.0 };
			memcpy( temp_fl_buf, x + idx, n_part4_rem * sizeof( float ) );

			__m128 xmm10, xmm11, xmm12, xmm10out;
			__m128i xmm10outi;

			xmm0 = _mm_loadu_ps( temp_fl_buf );

			EXPF_SSE(xmm0, xmm1, xmm10, xmm11, xmm12, xmm10outi);
			xmm10out = _mm_castsi128_ps( xmm10outi );

			xmm0d = _mm_cvtps_pd( xmm10out );
			xmm1d = _mm_cvtps_pd( _mm_permute_ps( xmm10out, 0x4E ) ); //0 1 2 3 -> 2 3 0 1
			xmm0d = _mm_add_pd( xmm0d, xmm1d );

			xmm1d = _mm_permute_pd( xmm0d, 0x01);
			xmm0d = _mm_add_pd( xmm0d, xmm1d );
			exp_sum[1] = _mm_cvtsd_f64( xmm0d );
			exp_sum[0] += exp_sum[1];
			// Only n_part_rem4 elems are valid, need to zero out rest.
			// This is because exp(0)=1;
			exp_sum[0] -= ( 4 - n_part4_rem );
		}

		// Broadcast the double exp sum.
		__m256d exp_red_ymm0;
		__m128d exp_red_xmm0;
		exp_sum[1] = exp_sum[0];
		exp_red_xmm0 = _mm_loadu_pd( exp_sum );
		exp_red_ymm0 = _mm256_broadcastsd_pd( exp_red_xmm0 );

		// Exp division of the array - avx2 block.
		for ( idx = 0; idx < n_part8; idx += 8 )
		{
			ymm0 = _mm256_loadu_ps( x + idx );

			// Convert to double
			xmm0 = _mm256_castps256_ps128( ymm0 );
			xmm1 = _mm256_extractf128_ps( ymm0, 0x1 );
			ymm0d = _mm256_cvtps_pd( xmm0 );
			ymm1d = _mm256_cvtps_pd( xmm1 );

			// Divide at double level
			ymm0d = _mm256_div_pd( ymm0d, exp_red_ymm0 );
			ymm1d = _mm256_div_pd( ymm1d, exp_red_ymm0 );

			xmm0 = _mm256_cvtpd_ps( ymm0d );
			xmm1 = _mm256_cvtpd_ps( ymm1d );

			_mm_storeu_ps( x + idx, xmm0 );
			_mm_storeu_ps( x + idx + 4, xmm1 );
		}
		// sse remainder block.
		if ( n_part4 == 1 )
		{
			xmm0 = _mm_loadu_ps( x + idx );

			// Convert to double
			xmm0d = _mm_cvtps_pd( xmm0 );
			xmm1d = _mm_cvtps_pd( _mm_permute_ps( xmm0, 0x4E ) ); //0 1 2 3 -> 2 3 0 1

			// Divide at double level
			xmm0d = _mm_div_pd( xmm0d, exp_red_xmm0 );
			xmm1d = _mm_div_pd( xmm1d, exp_red_xmm0 );

			xmm0 = _mm_cvtpd_ps( xmm0d );
			xmm1 = _mm_cvtpd_ps( xmm1d );
			xmm1 = _mm_permute_ps( xmm1, 0x4E );
			xmm0 = _mm_blend_ps( xmm0, xmm1, 0xC); // Combine outputs from 2 registers.

			_mm_storeu_ps( x + idx, xmm0 );

			idx = idx + 4;
		}
		// non vector remainder block.
		if ( n_part4_rem > 0 )
		{
			float temp_fl_buf[4] = { 0.0 };
			memcpy( temp_fl_buf, x + idx, n_part4_rem * sizeof( float ) );

			xmm0 = _mm_loadu_ps( temp_fl_buf );

			// Convert to double
			xmm0d = _mm_cvtps_pd( xmm0 );
			xmm1d = _mm_cvtps_pd( _mm_permute_ps( xmm0, 0x4E ) ); //0 1 2 3 -> 2 3 0 1

			// Divide at double level
			xmm0d = _mm_div_pd( xmm0d, exp_red_xmm0 );
			xmm1d = _mm_div_pd( xmm1d, exp_red_xmm0 );

			xmm0 = _mm_cvtpd_ps( xmm0d );
			xmm1 = _mm_cvtpd_ps( xmm1d );
			xmm1 = _mm_permute_ps( xmm1, 0x4E );
			xmm0 = _mm_blend_ps( xmm0, xmm1, 0xC);

			_mm_storeu_ps( temp_fl_buf, xmm0 );
			memcpy( x + idx, temp_fl_buf, n_part4_rem * sizeof( float ) );
		}
	}
	// For non unit increment, use non-vectorized code.
	else
	{
		double exp_sum = 0.0;

		dim_t n_incx = n * incx;

		// Exp reduction of the array.
		for ( dim_t idx = 0; idx < n_incx; idx += incx )
		{
			float temp_val = *( x + idx );
			exp_sum += (double)( expf( temp_val ) );
		}
		// Exp division of the array.
		for ( dim_t idx = 0; idx < n_incx; idx += incx )
		{
			float temp_val = *( x + idx );
			*( x + idx ) = ( float )( ( double ) temp_val / exp_sum );
		}
	}
}
#endif
