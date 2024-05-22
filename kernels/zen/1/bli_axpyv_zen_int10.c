/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2016 - 2019, Advanced Micro Devices, Inc.
   Copyright (C) 2018, The University of Texas at Austin

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

void bli_saxpyv_zen_int10
     (
             conj_t  conjx,
             dim_t   n,
       const void*   alpha0,
       const void*   x0, inc_t incx,
             void*   y0, inc_t incy,
       const cntx_t* cntx
     )
{
	const float*     alpha = alpha0;
	const float*     x     = x0;
	      float*     y     = y0;

	const dim_t      n_elem_per_reg = 8;

	dim_t            i;

	__m256           alphav;
	__m256           xv[10];
	__m256           yv[10];
	__m256           zv[10];

	// If the vector dimension is zero, or if alpha is zero, return early.
	if ( bli_zero_dim1( n ) || PASTEMAC(s,eq0)( *alpha ) ) return;

	// Initialize local pointers.
	const float* restrict xp = x;
	      float* restrict yp = y;

	if ( incx == 1 && incy == 1 )
	{
		// Broadcast the alpha scalar to all elements of a vector register.
		alphav = _mm256_broadcast_ss( alpha );

		for ( i = 0; (i + 79) < n; i += 80 )
		{
			// 80 elements will be processed per loop; 10 FMAs will run per loop.
			xv[0] = _mm256_loadu_ps( xp + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_ps( xp + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_ps( xp + 2*n_elem_per_reg );
			xv[3] = _mm256_loadu_ps( xp + 3*n_elem_per_reg );
			xv[4] = _mm256_loadu_ps( xp + 4*n_elem_per_reg );
			xv[5] = _mm256_loadu_ps( xp + 5*n_elem_per_reg );
			xv[6] = _mm256_loadu_ps( xp + 6*n_elem_per_reg );
			xv[7] = _mm256_loadu_ps( xp + 7*n_elem_per_reg );
			xv[8] = _mm256_loadu_ps( xp + 8*n_elem_per_reg );
			xv[9] = _mm256_loadu_ps( xp + 9*n_elem_per_reg );

			yv[0] = _mm256_loadu_ps( yp + 0*n_elem_per_reg );
			yv[1] = _mm256_loadu_ps( yp + 1*n_elem_per_reg );
			yv[2] = _mm256_loadu_ps( yp + 2*n_elem_per_reg );
			yv[3] = _mm256_loadu_ps( yp + 3*n_elem_per_reg );
			yv[4] = _mm256_loadu_ps( yp + 4*n_elem_per_reg );
			yv[5] = _mm256_loadu_ps( yp + 5*n_elem_per_reg );
			yv[6] = _mm256_loadu_ps( yp + 6*n_elem_per_reg );
			yv[7] = _mm256_loadu_ps( yp + 7*n_elem_per_reg );
			yv[8] = _mm256_loadu_ps( yp + 8*n_elem_per_reg );
			yv[9] = _mm256_loadu_ps( yp + 9*n_elem_per_reg );

			zv[0] = _mm256_fmadd_ps( xv[0], alphav, yv[0] );
			zv[1] = _mm256_fmadd_ps( xv[1], alphav, yv[1] );
			zv[2] = _mm256_fmadd_ps( xv[2], alphav, yv[2] );
			zv[3] = _mm256_fmadd_ps( xv[3], alphav, yv[3] );
			zv[4] = _mm256_fmadd_ps( xv[4], alphav, yv[4] );
			zv[5] = _mm256_fmadd_ps( xv[5], alphav, yv[5] );
			zv[6] = _mm256_fmadd_ps( xv[6], alphav, yv[6] );
			zv[7] = _mm256_fmadd_ps( xv[7], alphav, yv[7] );
			zv[8] = _mm256_fmadd_ps( xv[8], alphav, yv[8] );
			zv[9] = _mm256_fmadd_ps( xv[9], alphav, yv[9] );

			_mm256_storeu_ps( (yp + 0*n_elem_per_reg), zv[0] );
			_mm256_storeu_ps( (yp + 1*n_elem_per_reg), zv[1] );
			_mm256_storeu_ps( (yp + 2*n_elem_per_reg), zv[2] );
			_mm256_storeu_ps( (yp + 3*n_elem_per_reg), zv[3] );
			_mm256_storeu_ps( (yp + 4*n_elem_per_reg), zv[4] );
			_mm256_storeu_ps( (yp + 5*n_elem_per_reg), zv[5] );
			_mm256_storeu_ps( (yp + 6*n_elem_per_reg), zv[6] );
			_mm256_storeu_ps( (yp + 7*n_elem_per_reg), zv[7] );
			_mm256_storeu_ps( (yp + 8*n_elem_per_reg), zv[8] );
			_mm256_storeu_ps( (yp + 9*n_elem_per_reg), zv[9] );

			xp += 10*n_elem_per_reg;
			yp += 10*n_elem_per_reg;
		}

		for ( ; (i + 39) < n; i += 40 )
		{
			xv[0] = _mm256_loadu_ps( xp + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_ps( xp + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_ps( xp + 2*n_elem_per_reg );
			xv[3] = _mm256_loadu_ps( xp + 3*n_elem_per_reg );
			xv[4] = _mm256_loadu_ps( xp + 4*n_elem_per_reg );

			yv[0] = _mm256_loadu_ps( yp + 0*n_elem_per_reg );
			yv[1] = _mm256_loadu_ps( yp + 1*n_elem_per_reg );
			yv[2] = _mm256_loadu_ps( yp + 2*n_elem_per_reg );
			yv[3] = _mm256_loadu_ps( yp + 3*n_elem_per_reg );
			yv[4] = _mm256_loadu_ps( yp + 4*n_elem_per_reg );

			zv[0] = _mm256_fmadd_ps( xv[0], alphav, yv[0] );
			zv[1] = _mm256_fmadd_ps( xv[1], alphav, yv[1] );
			zv[2] = _mm256_fmadd_ps( xv[2], alphav, yv[2] );
			zv[3] = _mm256_fmadd_ps( xv[3], alphav, yv[3] );
			zv[4] = _mm256_fmadd_ps( xv[4], alphav, yv[4] );

			_mm256_storeu_ps( (yp + 0*n_elem_per_reg), zv[0] );
			_mm256_storeu_ps( (yp + 1*n_elem_per_reg), zv[1] );
			_mm256_storeu_ps( (yp + 2*n_elem_per_reg), zv[2] );
			_mm256_storeu_ps( (yp + 3*n_elem_per_reg), zv[3] );
			_mm256_storeu_ps( (yp + 4*n_elem_per_reg), zv[4] );

			xp += 5*n_elem_per_reg;
			yp += 5*n_elem_per_reg;
		}

		for ( ; (i + 31) < n; i += 32 )
		{
			xv[0] = _mm256_loadu_ps( xp + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_ps( xp + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_ps( xp + 2*n_elem_per_reg );
			xv[3] = _mm256_loadu_ps( xp + 3*n_elem_per_reg );

			yv[0] = _mm256_loadu_ps( yp + 0*n_elem_per_reg );
			yv[1] = _mm256_loadu_ps( yp + 1*n_elem_per_reg );
			yv[2] = _mm256_loadu_ps( yp + 2*n_elem_per_reg );
			yv[3] = _mm256_loadu_ps( yp + 3*n_elem_per_reg );

			zv[0] = _mm256_fmadd_ps( xv[0], alphav, yv[0] );
			zv[1] = _mm256_fmadd_ps( xv[1], alphav, yv[1] );
			zv[2] = _mm256_fmadd_ps( xv[2], alphav, yv[2] );
			zv[3] = _mm256_fmadd_ps( xv[3], alphav, yv[3] );

			_mm256_storeu_ps( (yp + 0*n_elem_per_reg), zv[0] );
			_mm256_storeu_ps( (yp + 1*n_elem_per_reg), zv[1] );
			_mm256_storeu_ps( (yp + 2*n_elem_per_reg), zv[2] );
			_mm256_storeu_ps( (yp + 3*n_elem_per_reg), zv[3] );

			xp += 4*n_elem_per_reg;
			yp += 4*n_elem_per_reg;
		}

		for ( ; (i + 15) < n; i += 16 )
		{
			xv[0] = _mm256_loadu_ps( xp + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_ps( xp + 1*n_elem_per_reg );

			yv[0] = _mm256_loadu_ps( yp + 0*n_elem_per_reg );
			yv[1] = _mm256_loadu_ps( yp + 1*n_elem_per_reg );

			zv[0] = _mm256_fmadd_ps( xv[0], alphav, yv[0] );
			zv[1] = _mm256_fmadd_ps( xv[1], alphav, yv[1] );

			_mm256_storeu_ps( (yp + 0*n_elem_per_reg), zv[0] );
			_mm256_storeu_ps( (yp + 1*n_elem_per_reg), zv[1] );

			xp += 2*n_elem_per_reg;
			yp += 2*n_elem_per_reg;
		}

		for ( ; (i + 7) < n; i += 8 )
		{
			xv[0] = _mm256_loadu_ps( xp + 0*n_elem_per_reg );

			yv[0] = _mm256_loadu_ps( yp + 0*n_elem_per_reg );

			zv[0] = _mm256_fmadd_ps( xv[0], alphav, yv[0] );

			_mm256_storeu_ps( (yp + 0*n_elem_per_reg), zv[0] );

			xp += 1*n_elem_per_reg;
			yp += 1*n_elem_per_reg;
		}

		// Issue vzeroupper instruction to clear upper lanes of ymm registers.
		// This avoids a performance penalty caused by false dependencies when
		// transitioning from from AVX to SSE instructions (which may occur
		// as soon as the n_left cleanup loop below if BLIS is compiled with
		// -mfpmath=sse).
		_mm256_zeroupper();

		for ( ; (i + 0) < n; i += 1 )
		{
			*yp += (*alpha) * (*xp);

			xp += 1;
			yp += 1;
		}
	}
	else
	{
		const float alphac = *alpha;

		for ( i = 0; i < n; ++i )
		{
			const float xpc = *xp;

			*yp += alphac * xpc;

			xp += incx;
			yp += incy;
		}
	}
}

// -----------------------------------------------------------------------------

void bli_daxpyv_zen_int10
     (
             conj_t  conjx,
             dim_t   n,
       const void*   alpha0,
       const void*   x0, inc_t incx,
             void*   y0, inc_t incy,
       const cntx_t* cntx
     )
{
	const double*    alpha = alpha0;
	const double*    x     = x0;
	      double*    y     = y0;

	const dim_t      n_elem_per_reg = 4;

	dim_t            i;

	__m256d          alphav;
	__m256d          xv[10];
	__m256d          yv[10];
	__m256d          zv[10];

	// If the vector dimension is zero, or if alpha is zero, return early.
	if ( bli_zero_dim1( n ) || PASTEMAC(d,eq0)( *alpha ) ) return;

	// Initialize local pointers.
	const double* restrict xp = x;
	      double* restrict yp = y;

	if ( incx == 1 && incy == 1 )
	{
		// Broadcast the alpha scalar to all elements of a vector register.
		alphav = _mm256_broadcast_sd( alpha );

		for ( i = 0; (i + 39) < n; i += 40 )
		{
			// 40 elements will be processed per loop; 10 FMAs will run per loop.
			xv[0] = _mm256_loadu_pd( xp + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_pd( xp + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_pd( xp + 2*n_elem_per_reg );
			xv[3] = _mm256_loadu_pd( xp + 3*n_elem_per_reg );
			xv[4] = _mm256_loadu_pd( xp + 4*n_elem_per_reg );
			xv[5] = _mm256_loadu_pd( xp + 5*n_elem_per_reg );
			xv[6] = _mm256_loadu_pd( xp + 6*n_elem_per_reg );
			xv[7] = _mm256_loadu_pd( xp + 7*n_elem_per_reg );
			xv[8] = _mm256_loadu_pd( xp + 8*n_elem_per_reg );
			xv[9] = _mm256_loadu_pd( xp + 9*n_elem_per_reg );

			yv[0] = _mm256_loadu_pd( yp + 0*n_elem_per_reg );
			yv[1] = _mm256_loadu_pd( yp + 1*n_elem_per_reg );
			yv[2] = _mm256_loadu_pd( yp + 2*n_elem_per_reg );
			yv[3] = _mm256_loadu_pd( yp + 3*n_elem_per_reg );
			yv[4] = _mm256_loadu_pd( yp + 4*n_elem_per_reg );
			yv[5] = _mm256_loadu_pd( yp + 5*n_elem_per_reg );
			yv[6] = _mm256_loadu_pd( yp + 6*n_elem_per_reg );
			yv[7] = _mm256_loadu_pd( yp + 7*n_elem_per_reg );
			yv[8] = _mm256_loadu_pd( yp + 8*n_elem_per_reg );
			yv[9] = _mm256_loadu_pd( yp + 9*n_elem_per_reg );

			zv[0] = _mm256_fmadd_pd( xv[0], alphav, yv[0] );
			zv[1] = _mm256_fmadd_pd( xv[1], alphav, yv[1] );
			zv[2] = _mm256_fmadd_pd( xv[2], alphav, yv[2] );
			zv[3] = _mm256_fmadd_pd( xv[3], alphav, yv[3] );
			zv[4] = _mm256_fmadd_pd( xv[4], alphav, yv[4] );
			zv[5] = _mm256_fmadd_pd( xv[5], alphav, yv[5] );
			zv[6] = _mm256_fmadd_pd( xv[6], alphav, yv[6] );
			zv[7] = _mm256_fmadd_pd( xv[7], alphav, yv[7] );
			zv[8] = _mm256_fmadd_pd( xv[8], alphav, yv[8] );
			zv[9] = _mm256_fmadd_pd( xv[9], alphav, yv[9] );

			_mm256_storeu_pd( (yp + 0*n_elem_per_reg), zv[0] );
			_mm256_storeu_pd( (yp + 1*n_elem_per_reg), zv[1] );
			_mm256_storeu_pd( (yp + 2*n_elem_per_reg), zv[2] );
			_mm256_storeu_pd( (yp + 3*n_elem_per_reg), zv[3] );
			_mm256_storeu_pd( (yp + 4*n_elem_per_reg), zv[4] );
			_mm256_storeu_pd( (yp + 5*n_elem_per_reg), zv[5] );
			_mm256_storeu_pd( (yp + 6*n_elem_per_reg), zv[6] );
			_mm256_storeu_pd( (yp + 7*n_elem_per_reg), zv[7] );
			_mm256_storeu_pd( (yp + 8*n_elem_per_reg), zv[8] );
			_mm256_storeu_pd( (yp + 9*n_elem_per_reg), zv[9] );

			xp += 10*n_elem_per_reg;
			yp += 10*n_elem_per_reg;
		}

		for ( ; (i + 19) < n; i += 20 )
		{
			xv[0] = _mm256_loadu_pd( xp + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_pd( xp + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_pd( xp + 2*n_elem_per_reg );
			xv[3] = _mm256_loadu_pd( xp + 3*n_elem_per_reg );
			xv[4] = _mm256_loadu_pd( xp + 4*n_elem_per_reg );

			yv[0] = _mm256_loadu_pd( yp + 0*n_elem_per_reg );
			yv[1] = _mm256_loadu_pd( yp + 1*n_elem_per_reg );
			yv[2] = _mm256_loadu_pd( yp + 2*n_elem_per_reg );
			yv[3] = _mm256_loadu_pd( yp + 3*n_elem_per_reg );
			yv[4] = _mm256_loadu_pd( yp + 4*n_elem_per_reg );

			zv[0] = _mm256_fmadd_pd( xv[0], alphav, yv[0] );
			zv[1] = _mm256_fmadd_pd( xv[1], alphav, yv[1] );
			zv[2] = _mm256_fmadd_pd( xv[2], alphav, yv[2] );
			zv[3] = _mm256_fmadd_pd( xv[3], alphav, yv[3] );
			zv[4] = _mm256_fmadd_pd( xv[4], alphav, yv[4] );

			_mm256_storeu_pd( (yp + 0*n_elem_per_reg), zv[0] );
			_mm256_storeu_pd( (yp + 1*n_elem_per_reg), zv[1] );
			_mm256_storeu_pd( (yp + 2*n_elem_per_reg), zv[2] );
			_mm256_storeu_pd( (yp + 3*n_elem_per_reg), zv[3] );
			_mm256_storeu_pd( (yp + 4*n_elem_per_reg), zv[4] );

			xp += 5*n_elem_per_reg;
			yp += 5*n_elem_per_reg;
		}

		for ( ; (i + 15) < n; i += 16 )
		{
			xv[0] = _mm256_loadu_pd( xp + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_pd( xp + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_pd( xp + 2*n_elem_per_reg );
			xv[3] = _mm256_loadu_pd( xp + 3*n_elem_per_reg );

			yv[0] = _mm256_loadu_pd( yp + 0*n_elem_per_reg );
			yv[1] = _mm256_loadu_pd( yp + 1*n_elem_per_reg );
			yv[2] = _mm256_loadu_pd( yp + 2*n_elem_per_reg );
			yv[3] = _mm256_loadu_pd( yp + 3*n_elem_per_reg );

			zv[0] = _mm256_fmadd_pd( xv[0], alphav, yv[0] );
			zv[1] = _mm256_fmadd_pd( xv[1], alphav, yv[1] );
			zv[2] = _mm256_fmadd_pd( xv[2], alphav, yv[2] );
			zv[3] = _mm256_fmadd_pd( xv[3], alphav, yv[3] );

			_mm256_storeu_pd( (yp + 0*n_elem_per_reg), zv[0] );
			_mm256_storeu_pd( (yp + 1*n_elem_per_reg), zv[1] );
			_mm256_storeu_pd( (yp + 2*n_elem_per_reg), zv[2] );
			_mm256_storeu_pd( (yp + 3*n_elem_per_reg), zv[3] );

			xp += 4*n_elem_per_reg;
			yp += 4*n_elem_per_reg;
		}

		for ( ; i + 7 < n; i += 8 )
		{
			xv[0] = _mm256_loadu_pd( xp + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_pd( xp + 1*n_elem_per_reg );

			yv[0] = _mm256_loadu_pd( yp + 0*n_elem_per_reg );
			yv[1] = _mm256_loadu_pd( yp + 1*n_elem_per_reg );

			zv[0] = _mm256_fmadd_pd( xv[0], alphav, yv[0] );
			zv[1] = _mm256_fmadd_pd( xv[1], alphav, yv[1] );

			_mm256_storeu_pd( (yp + 0*n_elem_per_reg), zv[0] );
			_mm256_storeu_pd( (yp + 1*n_elem_per_reg), zv[1] );

			xp += 2*n_elem_per_reg;
			yp += 2*n_elem_per_reg;
		}

		for ( ; i + 3 < n; i += 4 )
		{
			xv[0] = _mm256_loadu_pd( xp + 0*n_elem_per_reg );

			yv[0] = _mm256_loadu_pd( yp + 0*n_elem_per_reg );

			zv[0] = _mm256_fmadd_pd( xv[0], alphav, yv[0] );

			_mm256_storeu_pd( (yp + 0*n_elem_per_reg), zv[0] );

			xp += 1*n_elem_per_reg;
			yp += 1*n_elem_per_reg;
		}

		// Issue vzeroupper instruction to clear upper lanes of ymm registers.
		// This avoids a performance penalty caused by false dependencies when
		// transitioning from from AVX to SSE instructions (which may occur
		// as soon as the n_left cleanup loop below if BLIS is compiled with
		// -mfpmath=sse).
		_mm256_zeroupper();

		for ( ; i < n; i += 1 )
		{
			*yp += (*alpha) * (*xp);

			yp += 1;
			xp += 1;
		}
	}
	else
	{
		const double alphac = *alpha;

		for ( i = 0; i < n; ++i )
		{
			const double xpc = *xp;

			*yp += alphac * xpc;

			xp += incx;
			yp += incy;
		}
	}
}

