/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2020 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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
#include "blis.h"


/* Union data structure to access AVX registers
   One 256-bit AVX register holds 8 SP elements. */
typedef union
{
	__m256  v;
	float   f[8] __attribute__((aligned(64)));
} v8sf_t;

/* Union data structure to access AVX registers
*  One 256-bit AVX register holds 4 DP elements. */
typedef union
{
	__m256d v;
	double  d[4] __attribute__((aligned(64)));
} v4df_t;

// -----------------------------------------------------------------------------

void bli_sswapv_zen_int8
     (
       dim_t            n,
       float*  restrict x, inc_t incx,
       float*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{

	const dim_t     n_elem_per_reg = 8;
	dim_t           i = 0;

	float* restrict x0;
	float* restrict y0;

	__m256          xv[8];
	__m256          yv[8];

	// If the vector dimension is zero, return early.
	if ( bli_zero_dim1( n ) ) return;

	x0 = x;
	y0 = y;

	if ( incx == 1 && incy == 1 )
	{
		for ( i = 0; ( i + 63 ) < n; i += 64 )
		{
            //Load the input values
			xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_ps( x0 + 2*n_elem_per_reg );
			xv[3] = _mm256_loadu_ps( x0 + 3*n_elem_per_reg );
			xv[4] = _mm256_loadu_ps( x0 + 4*n_elem_per_reg );
			xv[5] = _mm256_loadu_ps( x0 + 5*n_elem_per_reg );
			xv[6] = _mm256_loadu_ps( x0 + 6*n_elem_per_reg );
			xv[7] = _mm256_loadu_ps( x0 + 7*n_elem_per_reg );

			yv[0] = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );
			yv[1] = _mm256_loadu_ps( y0 + 1*n_elem_per_reg );
			yv[2] = _mm256_loadu_ps( y0 + 2*n_elem_per_reg );
			yv[3] = _mm256_loadu_ps( y0 + 3*n_elem_per_reg );
			yv[4] = _mm256_loadu_ps( y0 + 4*n_elem_per_reg );
			yv[5] = _mm256_loadu_ps( y0 + 5*n_elem_per_reg );
			yv[6] = _mm256_loadu_ps( y0 + 6*n_elem_per_reg );
			yv[7] = _mm256_loadu_ps( y0 + 7*n_elem_per_reg );

			_mm256_storeu_ps( (x0 + 0*n_elem_per_reg), yv[0]);
			_mm256_storeu_ps( (x0 + 1*n_elem_per_reg), yv[1]);
			_mm256_storeu_ps( (x0 + 2*n_elem_per_reg), yv[2]);
			_mm256_storeu_ps( (x0 + 3*n_elem_per_reg), yv[3]);
			_mm256_storeu_ps( (x0 + 4*n_elem_per_reg), yv[4]);
			_mm256_storeu_ps( (x0 + 5*n_elem_per_reg), yv[5]);
			_mm256_storeu_ps( (x0 + 6*n_elem_per_reg), yv[6]);
			_mm256_storeu_ps( (x0 + 7*n_elem_per_reg), yv[7]);

			_mm256_storeu_ps( (y0 + 0*n_elem_per_reg), xv[0]);
			_mm256_storeu_ps( (y0 + 1*n_elem_per_reg), xv[1]);
			_mm256_storeu_ps( (y0 + 2*n_elem_per_reg), xv[2]);
			_mm256_storeu_ps( (y0 + 3*n_elem_per_reg), xv[3]);
			_mm256_storeu_ps( (y0 + 4*n_elem_per_reg), xv[4]);
			_mm256_storeu_ps( (y0 + 5*n_elem_per_reg), xv[5]);
			_mm256_storeu_ps( (y0 + 6*n_elem_per_reg), xv[6]);
			_mm256_storeu_ps( (y0 + 7*n_elem_per_reg), xv[7]);

			x0 += 8*n_elem_per_reg;
			y0 += 8*n_elem_per_reg;
		}

		for ( ; ( i + 31 ) < n; i += 32 )
		{
            //Load the input values
			xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_ps( x0 + 2*n_elem_per_reg );
			xv[3] = _mm256_loadu_ps( x0 + 3*n_elem_per_reg );

			yv[0] = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );
			yv[1] = _mm256_loadu_ps( y0 + 1*n_elem_per_reg );
			yv[2] = _mm256_loadu_ps( y0 + 2*n_elem_per_reg );
			yv[3] = _mm256_loadu_ps( y0 + 3*n_elem_per_reg );

			_mm256_storeu_ps( (y0 + 0*n_elem_per_reg), xv[0]);
			_mm256_storeu_ps( (y0 + 1*n_elem_per_reg), xv[1]);
			_mm256_storeu_ps( (y0 + 2*n_elem_per_reg), xv[2]);
			_mm256_storeu_ps( (y0 + 3*n_elem_per_reg), xv[3]);

			_mm256_storeu_ps( (x0 + 0*n_elem_per_reg), yv[0]);
			_mm256_storeu_ps( (x0 + 1*n_elem_per_reg), yv[1]);
			_mm256_storeu_ps( (x0 + 2*n_elem_per_reg), yv[2]);
			_mm256_storeu_ps( (x0 + 3*n_elem_per_reg), yv[3]);

			x0 += 4*n_elem_per_reg;
			y0 += 4*n_elem_per_reg;
		}

		for ( ; ( i + 15 ) < n; i += 16 )
		{
			xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );

			yv[0] = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );
			yv[1] = _mm256_loadu_ps( y0 + 1*n_elem_per_reg );

			_mm256_storeu_ps( (y0 + 0*n_elem_per_reg), xv[0]);
			_mm256_storeu_ps( (y0 + 1*n_elem_per_reg), xv[1]);

			_mm256_storeu_ps( (x0 + 0*n_elem_per_reg), yv[0]);
			_mm256_storeu_ps( (x0 + 1*n_elem_per_reg), yv[1]);

			x0 += 2*n_elem_per_reg;
			y0 += 2*n_elem_per_reg;
		}

		for ( ; ( i + 7 ) < n; i += 8 )
		{
			xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );

			yv[0] = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );

			_mm256_storeu_ps( (x0 + 0*n_elem_per_reg), yv[0]);

			_mm256_storeu_ps( (y0 + 0*n_elem_per_reg), xv[0]);

			x0 += 1*n_elem_per_reg;
			y0 += 1*n_elem_per_reg;
		}

		for ( ; (i + 0) < n; i += 1 )
		{
			PASTEMAC(s,swaps)( x[i], y[i] );
		}
	}
	else
	{
		for ( i = 0; i < n; ++i )
		{
			PASTEMAC(s,swaps)( (*x0), (*y0) );

			x0 += incx;
			y0 += incy;
		}
	}

}

//--------------------------------------------------------------------------------

void bli_dswapv_zen_int8
     (
       dim_t            n,
       double* restrict x, inc_t incx,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
	const dim_t      n_elem_per_reg = 4;
	dim_t            i = 0;

	double* restrict x0;
	double* restrict y0;

	__m256d          xv[8];
	__m256d          yv[8];

	// If the vector dimension is zero, return early.
	if ( bli_zero_dim1( n ) ) return;

	x0 = x;
	y0 = y;

	if ( incx == 1 && incy == 1 )
	{
		for ( ; ( i + 31 ) < n; i += 32 )
		{
			xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_pd( x0 + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_pd( x0 + 2*n_elem_per_reg );
			xv[3] = _mm256_loadu_pd( x0 + 3*n_elem_per_reg );
			xv[4] = _mm256_loadu_pd( x0 + 4*n_elem_per_reg );
			xv[5] = _mm256_loadu_pd( x0 + 5*n_elem_per_reg );
			xv[6] = _mm256_loadu_pd( x0 + 6*n_elem_per_reg );
			xv[7] = _mm256_loadu_pd( x0 + 7*n_elem_per_reg );

			yv[0] = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
			yv[1] = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );
			yv[2] = _mm256_loadu_pd( y0 + 2*n_elem_per_reg );
			yv[3] = _mm256_loadu_pd( y0 + 3*n_elem_per_reg );
			yv[4] = _mm256_loadu_pd( y0 + 4*n_elem_per_reg );
			yv[5] = _mm256_loadu_pd( y0 + 5*n_elem_per_reg );
			yv[6] = _mm256_loadu_pd( y0 + 6*n_elem_per_reg );
			yv[7] = _mm256_loadu_pd( y0 + 7*n_elem_per_reg );

			_mm256_storeu_pd( (x0 + 0*n_elem_per_reg), yv[0]);
			_mm256_storeu_pd( (x0 + 1*n_elem_per_reg), yv[1]);
			_mm256_storeu_pd( (x0 + 2*n_elem_per_reg), yv[2]);
			_mm256_storeu_pd( (x0 + 3*n_elem_per_reg), yv[3]);
			_mm256_storeu_pd( (x0 + 4*n_elem_per_reg), yv[4]);
			_mm256_storeu_pd( (x0 + 5*n_elem_per_reg), yv[5]);
			_mm256_storeu_pd( (x0 + 6*n_elem_per_reg), yv[6]);
			_mm256_storeu_pd( (x0 + 7*n_elem_per_reg), yv[7]);

			_mm256_storeu_pd( (y0 + 0*n_elem_per_reg), xv[0]);
			_mm256_storeu_pd( (y0 + 1*n_elem_per_reg), xv[1]);
			_mm256_storeu_pd( (y0 + 2*n_elem_per_reg), xv[2]);
			_mm256_storeu_pd( (y0 + 3*n_elem_per_reg), xv[3]);
			_mm256_storeu_pd( (y0 + 4*n_elem_per_reg), xv[4]);
			_mm256_storeu_pd( (y0 + 5*n_elem_per_reg), xv[5]);
			_mm256_storeu_pd( (y0 + 6*n_elem_per_reg), xv[6]);
			_mm256_storeu_pd( (y0 + 7*n_elem_per_reg), xv[7]);

			x0 += 8*n_elem_per_reg;
			y0 += 8*n_elem_per_reg;
		}

		for ( ; ( i + 15 ) < n; i += 16 )
		{
			xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_pd( x0 + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_pd( x0 + 2*n_elem_per_reg );
			xv[3] = _mm256_loadu_pd( x0 + 3*n_elem_per_reg );

			yv[0] = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
			yv[1] = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );
			yv[2] = _mm256_loadu_pd( y0 + 2*n_elem_per_reg );
			yv[3] = _mm256_loadu_pd( y0 + 3*n_elem_per_reg );

			_mm256_storeu_pd( (y0 + 0*n_elem_per_reg), xv[0]);
			_mm256_storeu_pd( (y0 + 1*n_elem_per_reg), xv[1]);
			_mm256_storeu_pd( (y0 + 2*n_elem_per_reg), xv[2]);
			_mm256_storeu_pd( (y0 + 3*n_elem_per_reg), xv[3]);

			_mm256_storeu_pd( (x0 + 0*n_elem_per_reg), yv[0]);
			_mm256_storeu_pd( (x0 + 1*n_elem_per_reg), yv[1]);
			_mm256_storeu_pd( (x0 + 2*n_elem_per_reg), yv[2]);
			_mm256_storeu_pd( (x0 + 3*n_elem_per_reg), yv[3]);

			x0 += 4*n_elem_per_reg;
			y0 += 4*n_elem_per_reg;
		}

		for ( ; ( i + 7 ) < n; i += 8 )
		{
			xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_pd( x0 + 1*n_elem_per_reg );

			yv[0] = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
			yv[1] = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );

			_mm256_storeu_pd( (y0 + 0*n_elem_per_reg), xv[0]);
			_mm256_storeu_pd( (y0 + 1*n_elem_per_reg), xv[1]);

			_mm256_storeu_pd( (x0 + 0*n_elem_per_reg), yv[0]);
			_mm256_storeu_pd( (x0 + 1*n_elem_per_reg), yv[1]);

			x0 += 2*n_elem_per_reg;
			y0 += 2*n_elem_per_reg;
		}

		for ( ; ( i + 3 ) < n; i += 4 )
		{
			xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );

			yv[0] = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );

			_mm256_storeu_pd( (y0 + 0*n_elem_per_reg), xv[0]);

			_mm256_storeu_pd( (x0 + 0*n_elem_per_reg), yv[0]);

			x0 += 1*n_elem_per_reg;
			y0 += 1*n_elem_per_reg;
		}

		for ( ; (i + 0) < n; i += 1 )
		{
			PASTEMAC(d,swaps)( x[i], y[i] );
		}
	}
	else
	{
		for ( i = 0; i < n; ++i )
		{
			PASTEMAC(d,swaps)( (*x0), (*y0) );

			x0 += incx;
			y0 += incy;
		}
	}
}

