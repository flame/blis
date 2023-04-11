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

#include "math_utils_avx512.h"
#include "gelu_avx512.h"

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
LPGEMM_UTIL_L1_OP_KERNEL(float,f32_gelu_tanh_avx512)
{
	if ( incx == 1 )
	{
		dim_t n_part16 = ( n / 16 ) * 16;
		dim_t n_part16_rem = n - n_part16;

		dim_t idx = 0;
		__m512 zmm0, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;
		__m512i zmm10i;
		// avx512 block loop.
		for ( idx = 0; idx < n_part16; idx += 16 )
		{
			zmm0 = _mm512_loadu_ps( x + idx );

			GELU_TANH_F32_AVX512_DEF(zmm0, zmm10, zmm11, zmm12, \
							zmm13, zmm14, zmm15, zmm10i);

			_mm512_storeu_ps( x + idx, zmm0 );
		}

		// Process remainder using masked load.
		__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n_part16_rem ) );
		zmm0 = _mm512_maskz_loadu_ps( load_mask, x + idx );

		GELU_TANH_F32_AVX512_DEF(zmm0, zmm10, zmm11, zmm12, \
						zmm13, zmm14, zmm15, zmm10i);

		_mm512_mask_storeu_ps( x + idx, load_mask, zmm0 );
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
LPGEMM_UTIL_L1_OP_KERNEL(float,f32_gelu_erf_avx512)
{
	if ( incx == 1 )
	{
		dim_t n_part16 = ( n / 16 ) * 16;
		dim_t n_part16_rem = n - n_part16;

		dim_t idx = 0;
		__m512 zmm0, zmm10, zmm11, zmm12;
		// avx512 block loop.
		for ( idx = 0; idx < n_part16; idx += 16 )
		{
			zmm0 = _mm512_loadu_ps( x + idx );

			GELU_ERF_F32_AVX512_DEF(zmm0, zmm10, zmm11, zmm12);

			_mm512_storeu_ps( x + idx, zmm0 );
		}

		// Process remainder using masked load.
		__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n_part16_rem ) );
		zmm0 = _mm512_maskz_loadu_ps( load_mask, x + idx );

		GELU_ERF_F32_AVX512_DEF(zmm0, zmm10, zmm11, zmm12);

		_mm512_mask_storeu_ps( x + idx, load_mask, zmm0 );
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
