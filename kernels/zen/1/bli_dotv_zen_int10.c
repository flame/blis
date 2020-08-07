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

void bli_sdotv_zen_int10
     (
       conj_t           conjx,
       conj_t           conjy,
       dim_t            n,
       float*  restrict x, inc_t incx,
       float*  restrict y, inc_t incy,
       float*  restrict rho,
       cntx_t* restrict cntx
     )
{
	const dim_t      n_elem_per_reg = 8;

	dim_t            i;

	float*  restrict x0;
	float*  restrict y0;

	float            rho0;

	__m256           xv[10];
	__m256           yv[10];
	v8sf_t           rhov[2];

	// If the vector dimension is zero, or if alpha is zero, return early.
	if ( bli_zero_dim1( n ) )
	{
		PASTEMAC(s,set0s)( *rho );
		return;
	}

	// Initialize local pointers.
	x0 = x;
	y0 = y;

	PASTEMAC(s,set0s)( rho0 );

	if ( incx == 1 && incy == 1 )
	{
		rhov[0].v = _mm256_setzero_ps();
		rhov[1].v = _mm256_setzero_ps();

		for ( i = 0; (i + 79) < n; i += 80 )
		{
			// 80 elements will be processed per loop; 10 FMAs will run per loop.
			xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_ps( x0 + 2*n_elem_per_reg );
			xv[3] = _mm256_loadu_ps( x0 + 3*n_elem_per_reg );
			xv[4] = _mm256_loadu_ps( x0 + 4*n_elem_per_reg );
			xv[5] = _mm256_loadu_ps( x0 + 5*n_elem_per_reg );
			xv[6] = _mm256_loadu_ps( x0 + 6*n_elem_per_reg );
			xv[7] = _mm256_loadu_ps( x0 + 7*n_elem_per_reg );
			xv[8] = _mm256_loadu_ps( x0 + 8*n_elem_per_reg );
			xv[9] = _mm256_loadu_ps( x0 + 9*n_elem_per_reg );

			yv[0] = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );
			yv[1] = _mm256_loadu_ps( y0 + 1*n_elem_per_reg );
			yv[2] = _mm256_loadu_ps( y0 + 2*n_elem_per_reg );
			yv[3] = _mm256_loadu_ps( y0 + 3*n_elem_per_reg );
			yv[4] = _mm256_loadu_ps( y0 + 4*n_elem_per_reg );
			yv[5] = _mm256_loadu_ps( y0 + 5*n_elem_per_reg );
			yv[6] = _mm256_loadu_ps( y0 + 6*n_elem_per_reg );
			yv[7] = _mm256_loadu_ps( y0 + 7*n_elem_per_reg );
			yv[8] = _mm256_loadu_ps( y0 + 8*n_elem_per_reg );
			yv[9] = _mm256_loadu_ps( y0 + 9*n_elem_per_reg );

			rhov[0].v = _mm256_fmadd_ps( xv[0], yv[0], rhov[0].v );
			rhov[1].v = _mm256_fmadd_ps( xv[1], yv[1], rhov[1].v );
			rhov[0].v = _mm256_fmadd_ps( xv[2], yv[2], rhov[0].v );
			rhov[1].v = _mm256_fmadd_ps( xv[3], yv[3], rhov[1].v );
			rhov[0].v = _mm256_fmadd_ps( xv[4], yv[4], rhov[0].v );
			rhov[1].v = _mm256_fmadd_ps( xv[5], yv[5], rhov[1].v );
			rhov[0].v = _mm256_fmadd_ps( xv[6], yv[6], rhov[0].v );
			rhov[1].v = _mm256_fmadd_ps( xv[7], yv[7], rhov[1].v );
			rhov[0].v = _mm256_fmadd_ps( xv[8], yv[8], rhov[0].v );
			rhov[1].v = _mm256_fmadd_ps( xv[9], yv[9], rhov[1].v );

			x0 += 10*n_elem_per_reg;
			y0 += 10*n_elem_per_reg;
		}

		for ( ; (i + 39) < n; i += 40 )
		{
			xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_ps( x0 + 2*n_elem_per_reg );
			xv[3] = _mm256_loadu_ps( x0 + 3*n_elem_per_reg );
			xv[4] = _mm256_loadu_ps( x0 + 4*n_elem_per_reg );

			yv[0] = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );
			yv[1] = _mm256_loadu_ps( y0 + 1*n_elem_per_reg );
			yv[2] = _mm256_loadu_ps( y0 + 2*n_elem_per_reg );
			yv[3] = _mm256_loadu_ps( y0 + 3*n_elem_per_reg );
			yv[4] = _mm256_loadu_ps( y0 + 4*n_elem_per_reg );

			rhov[0].v = _mm256_fmadd_ps( xv[0], yv[0], rhov[0].v );
			rhov[1].v = _mm256_fmadd_ps( xv[1], yv[1], rhov[1].v );
			rhov[0].v = _mm256_fmadd_ps( xv[2], yv[2], rhov[0].v );
			rhov[1].v = _mm256_fmadd_ps( xv[3], yv[3], rhov[1].v );
			rhov[0].v = _mm256_fmadd_ps( xv[4], yv[4], rhov[0].v );

			x0 += 5*n_elem_per_reg;
			y0 += 5*n_elem_per_reg;
		}

		for ( ; (i + 31) < n; i += 32 )
		{
			xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_ps( x0 + 2*n_elem_per_reg );
			xv[3] = _mm256_loadu_ps( x0 + 3*n_elem_per_reg );

			yv[0] = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );
			yv[1] = _mm256_loadu_ps( y0 + 1*n_elem_per_reg );
			yv[2] = _mm256_loadu_ps( y0 + 2*n_elem_per_reg );
			yv[3] = _mm256_loadu_ps( y0 + 3*n_elem_per_reg );

			rhov[0].v = _mm256_fmadd_ps( xv[0], yv[0], rhov[0].v );
			rhov[1].v = _mm256_fmadd_ps( xv[1], yv[1], rhov[1].v );
			rhov[0].v = _mm256_fmadd_ps( xv[2], yv[2], rhov[0].v );
			rhov[1].v = _mm256_fmadd_ps( xv[3], yv[3], rhov[1].v );

			x0 += 4*n_elem_per_reg;
			y0 += 4*n_elem_per_reg;
		}

		for ( ; (i + 15) < n; i += 16 )
		{
			xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );

			yv[0] = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );
			yv[1] = _mm256_loadu_ps( y0 + 1*n_elem_per_reg );

			rhov[0].v = _mm256_fmadd_ps( xv[0], yv[0], rhov[0].v );
			rhov[1].v = _mm256_fmadd_ps( xv[1], yv[1], rhov[1].v );

			x0 += 2*n_elem_per_reg;
			y0 += 2*n_elem_per_reg;
		}

		for ( ; (i + 7) < n; i += 8 )
		{
			xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );

			yv[0] = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );

			rhov[0].v = _mm256_fmadd_ps( xv[0], yv[0], rhov[0].v );

			x0 += 1*n_elem_per_reg;
			y0 += 1*n_elem_per_reg;
		}

		for ( ; (i + 0) < n; i += 1 )
		{
			rhov[0].f[0] += x0[i] * y0[i];
		}

		v8sf_t onev;

		onev.v = _mm256_set1_ps( 1.0f );

		rhov[0].v = _mm256_dp_ps( rhov[0].v, onev.v, 0xf1 );
        rhov[1].v = _mm256_dp_ps( rhov[1].v, onev.v, 0xf1 );

		// Manually add the results from above to finish the sum.
		rho0   += rhov[0].f[0] + rhov[0].f[4];
		rho0   += rhov[1].f[0] + rhov[1].f[4];

		// Issue vzeroupper instruction to clear upper lanes of ymm registers.
		// This avoids a performance penalty caused by false dependencies when
		// transitioning from from AVX to SSE instructions (which may occur
		// later, especially if BLIS is compiled with -mfpmath=sse).
		_mm256_zeroupper();
	}
	else
	{
		for ( i = 0; i < n; ++i )
		{
			const float x0c = *x0;
			const float y0c = *y0;

			rho0 += x0c * y0c;

			x0 += incx;
			y0 += incy;
		}
	}

	// Copy the final result into the output variable.
	PASTEMAC(s,copys)( rho0, *rho );
}

// -----------------------------------------------------------------------------

void bli_ddotv_zen_int10
     (
       conj_t           conjx,
       conj_t           conjy,
       dim_t            n,
       double* restrict x, inc_t incx,
       double* restrict y, inc_t incy,
       double* restrict rho,
       cntx_t* restrict cntx
     )
{
	const dim_t      n_elem_per_reg = 4;

	dim_t            i;

	double* restrict x0;
	double* restrict y0;

	double           rho0;

	__m256d          xv[10];
	__m256d          yv[10];
	v4df_t           rhov[2];

	// If the vector dimension is zero, or if alpha is zero, return early.
	if ( bli_zero_dim1( n ) )
	{
		PASTEMAC(d,set0s)( *rho );
		return;
	}

	// Initialize local pointers.
	x0 = x;
	y0 = y;

	PASTEMAC(d,set0s)( rho0 );

	if ( incx == 1 && incy == 1 )
	{
		rhov[0].v = _mm256_setzero_pd();
		rhov[1].v = _mm256_setzero_pd();

		for ( i = 0; (i + 39) < n; i += 40 )
		{
			// 80 elements will be processed per loop; 10 FMAs will run per loop.
			xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_pd( x0 + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_pd( x0 + 2*n_elem_per_reg );
			xv[3] = _mm256_loadu_pd( x0 + 3*n_elem_per_reg );
			xv[4] = _mm256_loadu_pd( x0 + 4*n_elem_per_reg );
			xv[5] = _mm256_loadu_pd( x0 + 5*n_elem_per_reg );
			xv[6] = _mm256_loadu_pd( x0 + 6*n_elem_per_reg );
			xv[7] = _mm256_loadu_pd( x0 + 7*n_elem_per_reg );
			xv[8] = _mm256_loadu_pd( x0 + 8*n_elem_per_reg );
			xv[9] = _mm256_loadu_pd( x0 + 9*n_elem_per_reg );

			yv[0] = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
			yv[1] = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );
			yv[2] = _mm256_loadu_pd( y0 + 2*n_elem_per_reg );
			yv[3] = _mm256_loadu_pd( y0 + 3*n_elem_per_reg );
			yv[4] = _mm256_loadu_pd( y0 + 4*n_elem_per_reg );
			yv[5] = _mm256_loadu_pd( y0 + 5*n_elem_per_reg );
			yv[6] = _mm256_loadu_pd( y0 + 6*n_elem_per_reg );
			yv[7] = _mm256_loadu_pd( y0 + 7*n_elem_per_reg );
			yv[8] = _mm256_loadu_pd( y0 + 8*n_elem_per_reg );
			yv[9] = _mm256_loadu_pd( y0 + 9*n_elem_per_reg );

			rhov[0].v = _mm256_fmadd_pd( xv[0], yv[0], rhov[0].v );
			rhov[1].v = _mm256_fmadd_pd( xv[1], yv[1], rhov[1].v );
			rhov[0].v = _mm256_fmadd_pd( xv[2], yv[2], rhov[0].v );
			rhov[1].v = _mm256_fmadd_pd( xv[3], yv[3], rhov[1].v );
			rhov[0].v = _mm256_fmadd_pd( xv[4], yv[4], rhov[0].v );
			rhov[1].v = _mm256_fmadd_pd( xv[5], yv[5], rhov[1].v );
			rhov[0].v = _mm256_fmadd_pd( xv[6], yv[6], rhov[0].v );
			rhov[1].v = _mm256_fmadd_pd( xv[7], yv[7], rhov[1].v );
			rhov[0].v = _mm256_fmadd_pd( xv[8], yv[8], rhov[0].v );
			rhov[1].v = _mm256_fmadd_pd( xv[9], yv[9], rhov[1].v );

			x0 += 10*n_elem_per_reg;
			y0 += 10*n_elem_per_reg;
		}

		for ( ; (i + 19) < n; i += 20 )
		{
			xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_pd( x0 + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_pd( x0 + 2*n_elem_per_reg );
			xv[3] = _mm256_loadu_pd( x0 + 3*n_elem_per_reg );
			xv[4] = _mm256_loadu_pd( x0 + 4*n_elem_per_reg );

			yv[0] = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
			yv[1] = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );
			yv[2] = _mm256_loadu_pd( y0 + 2*n_elem_per_reg );
			yv[3] = _mm256_loadu_pd( y0 + 3*n_elem_per_reg );
			yv[4] = _mm256_loadu_pd( y0 + 4*n_elem_per_reg );

			rhov[0].v = _mm256_fmadd_pd( xv[0], yv[0], rhov[0].v );
			rhov[1].v = _mm256_fmadd_pd( xv[1], yv[1], rhov[1].v );
			rhov[0].v = _mm256_fmadd_pd( xv[2], yv[2], rhov[0].v );
			rhov[1].v = _mm256_fmadd_pd( xv[3], yv[3], rhov[1].v );
			rhov[0].v = _mm256_fmadd_pd( xv[4], yv[4], rhov[0].v );

			x0 += 5*n_elem_per_reg;
			y0 += 5*n_elem_per_reg;
		}

		for ( ; (i + 15) < n; i += 16 )
		{
			xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_pd( x0 + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_pd( x0 + 2*n_elem_per_reg );
			xv[3] = _mm256_loadu_pd( x0 + 3*n_elem_per_reg );

			yv[0] = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
			yv[1] = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );
			yv[2] = _mm256_loadu_pd( y0 + 2*n_elem_per_reg );
			yv[3] = _mm256_loadu_pd( y0 + 3*n_elem_per_reg );

			rhov[0].v = _mm256_fmadd_pd( xv[0], yv[0], rhov[0].v );
			rhov[1].v = _mm256_fmadd_pd( xv[1], yv[1], rhov[1].v );
			rhov[0].v = _mm256_fmadd_pd( xv[2], yv[2], rhov[0].v );
			rhov[1].v = _mm256_fmadd_pd( xv[3], yv[3], rhov[1].v );

			x0 += 4*n_elem_per_reg;
			y0 += 4*n_elem_per_reg;
		}

		for ( ; (i + 7) < n; i += 8 )
		{
			xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_pd( x0 + 1*n_elem_per_reg );

			yv[0] = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
			yv[1] = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );

			rhov[0].v = _mm256_fmadd_pd( xv[0], yv[0], rhov[0].v );
			rhov[1].v = _mm256_fmadd_pd( xv[1], yv[1], rhov[1].v );

			x0 += 2*n_elem_per_reg;
			y0 += 2*n_elem_per_reg;
		}

		for ( ; (i + 3) < n; i += 4 )
		{
			xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );

			yv[0] = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );

			rhov[0].v = _mm256_fmadd_pd( xv[0], yv[0], rhov[0].v );

			x0 += 1*n_elem_per_reg;
			y0 += 1*n_elem_per_reg;
		}

		for ( ; (i + 0) < n; i += 1 )
		{
			rhov[0].d[0] += x0[i] * y0[i];
		}

		// Manually add the results from above to finish the sum.
		rho0   += rhov[0].d[0] + rhov[0].d[1] + rhov[0].d[2] + rhov[0].d[3];
		rho0   += rhov[1].d[0] + rhov[1].d[1] + rhov[1].d[2] + rhov[1].d[3];

		// Issue vzeroupper instruction to clear upper lanes of ymm registers.
		// This avoids a performance penalty caused by false dependencies when
		// transitioning from from AVX to SSE instructions (which may occur
		// later, especially if BLIS is compiled with -mfpmath=sse).
		_mm256_zeroupper();
	}
	else
	{
		for ( i = 0; i < n; ++i )
		{
			const double x0c = *x0;
			const double y0c = *y0;

			rho0 += x0c * y0c;

			x0 += incx;
			y0 += incy;
		}
	}

	// Copy the final result into the output variable.
	PASTEMAC(d,copys)( rho0, *rho );
}

