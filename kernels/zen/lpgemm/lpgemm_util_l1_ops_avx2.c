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

// This kernels only works on arrays with inc=1.
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

// This kernels only works on arrays with inc=1.
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
#endif
