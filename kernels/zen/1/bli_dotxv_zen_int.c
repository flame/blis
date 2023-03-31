/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2016-2023, Advanced Micro Devices, Inc. All rights reserved.
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
   One 128-bit AVX register holds 8 SP elements. */
typedef union
{
	__m128  v;
	float   f[4] __attribute__((aligned(64)));
} v4sf_t;

/* Union data structure to access AVX registers
   One 256-bit AVX register holds 8 SP elements. */
typedef union
{
	__m256  v;
	float   f[8] __attribute__((aligned(64)));
} v8sf_t;

/* Union data structure to access AVX registers
*  One 128-bit AVX register holds 4 DP elements. */
typedef union
{
	__m128d v;
	double  d[2] __attribute__((aligned(64)));
} v2df_t;

/* Union data structure to access AVX registers
*  One 256-bit AVX register holds 4 DP elements. */
typedef union
{
	__m256d v;
	double  d[4] __attribute__((aligned(64)));
} v4df_t;

// -----------------------------------------------------------------------------

void bli_sdotxv_zen_int
     (
       conj_t           conjx,
       conj_t           conjy,
       dim_t            n,
       float*  restrict alpha,
       float*  restrict x, inc_t incx,
       float*  restrict y, inc_t incy,
       float*  restrict beta,
       float*  restrict rho,
       cntx_t* restrict cntx
     )
{
	const dim_t      n_elem_per_reg = 8;
	const dim_t      n_iter_unroll  = 4;

	dim_t            i;
	dim_t            n_viter;
	dim_t            n_left;

	float*  restrict x0;
	float*  restrict y0;
	float            rho0;

	v8sf_t           rhov[4], xv[4], yv[4];

	// If beta is zero, initialize rho1 to zero instead of scaling
	// rho by beta (in case rho contains NaN or Inf).
	if ( PASTEMAC(s,eq0)( *beta ) )
	{
		PASTEMAC(s,set0s)( *rho );
	}
	else
	{
		PASTEMAC(s,scals)( *beta, *rho );
	}

	// If the vector dimension is zero, output rho and return early.
	if ( bli_zero_dim1( n ) || PASTEMAC(s,eq0)( *alpha ) ) return;

	// Use the unrolling factor and the number of elements per register
	// to compute the number of vectorized and leftover iterations.
	n_viter = ( n ) / ( n_elem_per_reg * n_iter_unroll );
	n_left  = ( n ) % ( n_elem_per_reg * n_iter_unroll );

	// If there is anything that would interfere with our use of contiguous
	// vector loads/stores, override n_viter and n_left to use scalar code
	// for all iterations.
	if ( incx != 1 || incy != 1 )
	{
		n_viter = 0;
		n_left  = n;
	}

	// Initialize local pointers.
	x0 = x;
	y0 = y;

	// Initialize the unrolled iterations' rho vectors to zero.
	rhov[0].v = _mm256_setzero_ps();
	rhov[1].v = _mm256_setzero_ps();
	rhov[2].v = _mm256_setzero_ps();
	rhov[3].v = _mm256_setzero_ps();

	for ( i = 0; i < n_viter; ++i )
	{
		// Load the x and y input vector elements.
		xv[0].v = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
		yv[0].v = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );

		xv[1].v = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );
		yv[1].v = _mm256_loadu_ps( y0 + 1*n_elem_per_reg );

		xv[2].v = _mm256_loadu_ps( x0 + 2*n_elem_per_reg );
		yv[2].v = _mm256_loadu_ps( y0 + 2*n_elem_per_reg );

		xv[3].v = _mm256_loadu_ps( x0 + 3*n_elem_per_reg );
		yv[3].v = _mm256_loadu_ps( y0 + 3*n_elem_per_reg );

		// Compute the element-wise product of the x and y vectors,
		// storing in the corresponding rho vectors.
		rhov[0].v = _mm256_fmadd_ps( xv[0].v, yv[0].v, rhov[0].v );
		rhov[1].v = _mm256_fmadd_ps( xv[1].v, yv[1].v, rhov[1].v );
		rhov[2].v = _mm256_fmadd_ps( xv[2].v, yv[2].v, rhov[2].v );
		rhov[3].v = _mm256_fmadd_ps( xv[3].v, yv[3].v, rhov[3].v );

		x0 += ( n_elem_per_reg * n_iter_unroll );
		y0 += ( n_elem_per_reg * n_iter_unroll );
	}

	// Accumulate the unrolled rho vectors into a single vector.
	rhov[0].v = _mm256_add_ps(rhov[0].v,rhov[1].v);
	rhov[0].v = _mm256_add_ps(rhov[0].v,rhov[2].v);
	rhov[0].v = _mm256_add_ps(rhov[0].v,rhov[3].v);

	v4sf_t inter0, inter1;

	inter0.v = _mm256_extractf128_ps(rhov[0].v,0);
	inter1.v = _mm256_extractf128_ps(rhov[0].v,1);

	inter0.v = _mm_add_ps(inter0.v, inter1.v);

	inter1.v = _mm_permute_ps(inter0.v, 14);

	inter0.v = _mm_add_ps(inter0.v,inter1.v);

	// Accumulate the final rho vector into a single scalar result.
	rho0 = inter0.f[0] + inter0.f[1];

	// Issue vzeroupper instruction to clear upper lanes of ymm registers.
	// This avoids a performance penalty caused by false dependencies when
	// transitioning from AVX to SSE instructions (which may occur as soon
	// as the n_left cleanup loop below if BLIS is compiled with
	// -mfpmath=sse).
	_mm256_zeroupper();

	// If there are leftover iterations, perform them with scalar code.
	for ( i = 0; i < n_left; ++i )
	{
		const float x0c = *x0;
		const float y0c = *y0;

		rho0 += x0c * y0c;

		x0 += incx;
		y0 += incy;
	}

	// Accumulate the final result into the output variable.
	PASTEMAC(s,axpys)( *alpha, rho0, *rho );
}

// -----------------------------------------------------------------------------

void bli_ddotxv_zen_int
     (
       conj_t           conjx,
       conj_t           conjy,
       dim_t            n,
       double* restrict alpha,
       double* restrict x, inc_t incx,
       double* restrict y, inc_t incy,
       double* restrict beta,
       double* restrict rho,
       cntx_t* restrict cntx
     )
{
	const dim_t      n_elem_per_reg = 4;
	const dim_t      n_iter_unroll  = 4;

	dim_t            i;
	dim_t            n_viter;
	dim_t            n_left;

	double* restrict x0;
	double* restrict y0;
	double           rho0;

	v4df_t           rhov[4], xv[4], yv[4];
	
	// If beta is zero, initialize rho1 to zero instead of scaling
	// rho by beta (in case rho contains NaN or Inf).
	if ( PASTEMAC(d,eq0)( *beta ) )
	{
		PASTEMAC(d,set0s)( *rho );
	}
	else
	{
		PASTEMAC(d,scals)( *beta, *rho );
	}

	// If the vector dimension is zero, output rho and return early.
	if ( bli_zero_dim1( n ) || PASTEMAC(d,eq0)( *alpha ) ) return;

	// Use the unrolling factor and the number of elements per register
	// to compute the number of vectorized and leftover iterations.
	n_viter = ( n ) / ( n_elem_per_reg * n_iter_unroll );
	n_left  = ( n ) % ( n_elem_per_reg * n_iter_unroll );

	// If there is anything that would interfere with our use of contiguous
	// vector loads/stores, override n_viter and n_left to use scalar code
	// for all iterations.
	if ( incx != 1 || incy != 1 )
	{
		n_viter = 0;
		n_left  = n;
	}

	// Initialize local pointers.
	x0 = x;
	y0 = y;

	// Initialize the unrolled iterations' rho vectors to zero.
	rhov[0].v = _mm256_setzero_pd();
	rhov[1].v = _mm256_setzero_pd();
	rhov[2].v = _mm256_setzero_pd();
	rhov[3].v = _mm256_setzero_pd();

	for ( i = 0; i < n_viter; ++i )
	{
		// Load the x and y input vector elements.
		xv[0].v = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
		yv[0].v = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );

		xv[1].v = _mm256_loadu_pd( x0 + 1*n_elem_per_reg );
		yv[1].v = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );

		xv[2].v = _mm256_loadu_pd( x0 + 2*n_elem_per_reg );
		yv[2].v = _mm256_loadu_pd( y0 + 2*n_elem_per_reg );

		xv[3].v = _mm256_loadu_pd( x0 + 3*n_elem_per_reg );
		yv[3].v = _mm256_loadu_pd( y0 + 3*n_elem_per_reg );
		
		// Compute the element-wise product of the x and y vectors,
		// storing in the corresponding rho vectors.
		rhov[0].v = _mm256_fmadd_pd( xv[0].v, yv[0].v, rhov[0].v );
		rhov[1].v = _mm256_fmadd_pd( xv[1].v, yv[1].v, rhov[1].v );
		rhov[2].v = _mm256_fmadd_pd( xv[2].v, yv[2].v, rhov[2].v );
		rhov[3].v = _mm256_fmadd_pd( xv[3].v, yv[3].v, rhov[3].v );

		x0 += ( n_elem_per_reg * n_iter_unroll );
		y0 += ( n_elem_per_reg * n_iter_unroll );
	}

	// Accumulate the unrolled rho vectors into a single vector.
	rhov[0].v = _mm256_add_pd(rhov[1].v,rhov[0].v);
	rhov[0].v = _mm256_add_pd(rhov[2].v,rhov[0].v);
	rhov[0].v = _mm256_add_pd(rhov[3].v,rhov[0].v);

	v2df_t inter1, inter2;

	inter1.v = _mm256_extractf128_pd(rhov[0].v,1);
	inter2.v = _mm256_extractf128_pd(rhov[0].v,0);

	inter1.v = _mm_add_pd(inter1.v, inter2.v);

	// Accumulate the final rho vector into a single scalar result.
	rho0 = inter1.d[0] + inter1.d[1];

	// Issue vzeroupper instruction to clear upper lanes of ymm registers.
	// This avoids a performance penalty caused by false dependencies when
	// transitioning from AVX to SSE instructions (which may occur as soon
	// as the n_left cleanup loop below if BLIS is compiled with
	// -mfpmath=sse).
	_mm256_zeroupper();

	// If there are leftover iterations, perform them with scalar code.
	for ( i = 0; i < n_left; ++i )
	{
		const double x0c = *x0;
		const double y0c = *y0;

		rho0 += x0c * y0c;

		x0 += incx;
		y0 += incy;
	}

	// Accumulate the final result into the output variable.
	PASTEMAC(d,axpys)( *alpha, rho0, *rho );
}



void bli_zdotxv_zen_int
     (
       conj_t           conjx,
       conj_t           conjy,
       dim_t            n,
       dcomplex* restrict alpha,
       dcomplex* restrict x, inc_t incx,
       dcomplex* restrict y, inc_t incy,
       dcomplex* restrict beta,
       dcomplex* restrict rho,
       cntx_t* restrict cntx
     )
{
	const dim_t      n_elem_per_reg = 2;
	const dim_t      n_iter_unroll  = 4;

	dim_t            i;
	dim_t            n_viter;
	dim_t            n_left;

	dcomplex* restrict x0;
	dcomplex* restrict y0;
	dcomplex           rho0;

	v4df_t           rhov[8], xv[4], yv[8];

	conj_t conjx_use = conjx;
	if ( bli_is_conj( conjy ) )
	{
		bli_toggle_conj( &conjx_use );
	}
	// If beta is zero, initialize rho1 to zero instead of scaling
	// rho by beta (in case rho contains NaN or Inf).
	if ( PASTEMAC(z,eq0)( *beta ) )
	{
		PASTEMAC(z,set0s)( *rho );
	}
	else
	{
		PASTEMAC(z,scals)( *beta, *rho );
	}

	// If the vector dimension is zero, output rho and return early.
	if ( bli_zero_dim1( n ) || PASTEMAC(z,eq0)( *alpha ) ) return;

	// Use the unrolling factor and the number of elements per register
	// to compute the number of vectorized and leftover iterations.
	n_viter = ( n ) / ( n_elem_per_reg * n_iter_unroll );
	n_left  = ( n ) % ( n_elem_per_reg * n_iter_unroll );

	// If there is anything that would interfere with our use of contiguous
	// vector loads/stores, override n_viter and n_left to use scalar code
	// for all iterations.
	if ( incx != 1 || incy != 1 )
	{
		n_viter = 0;
		n_left  = n;
	}

	// Initialize local pointers.
	x0 = x;
	y0 = y;

	// Initialize the unrolled iterations' rho vectors to zero.
	rhov[0].v = _mm256_setzero_pd();
	rhov[1].v = _mm256_setzero_pd();
	rhov[2].v = _mm256_setzero_pd();
	rhov[3].v = _mm256_setzero_pd();

	rhov[4].v = _mm256_setzero_pd();
	rhov[5].v = _mm256_setzero_pd();
	rhov[6].v = _mm256_setzero_pd();
	rhov[7].v = _mm256_setzero_pd();

	if ( bli_is_conj( conjx_use ) )
        {
		__m256d conju = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
		for ( i = 0; i < n_viter; ++i )
		{
			// Load the x and y input vector elements.
			xv[0].v = _mm256_loadu_pd((double *) (x0 + 0*n_elem_per_reg) );
			yv[0].v = _mm256_loadu_pd((double *) (y0 + 0*n_elem_per_reg) );

			xv[1].v = _mm256_loadu_pd((double *) (x0 + 1*n_elem_per_reg) );
			yv[1].v = _mm256_loadu_pd((double *) (y0 + 1*n_elem_per_reg) );

			xv[2].v = _mm256_loadu_pd((double *) (x0 + 2*n_elem_per_reg) );
			yv[2].v = _mm256_loadu_pd((double *) (y0 + 2*n_elem_per_reg) );

			xv[3].v = _mm256_loadu_pd((double *) (x0 + 3*n_elem_per_reg) );
			yv[3].v = _mm256_loadu_pd((double *) (y0 + 3*n_elem_per_reg) );

			yv[0].v = _mm256_mul_pd(yv[0].v, conju);
			yv[1].v = _mm256_mul_pd(yv[1].v, conju);
			yv[2].v = _mm256_mul_pd(yv[2].v, conju);
			yv[3].v = _mm256_mul_pd(yv[3].v, conju);
			//yi0 yi0 yi1 yi1
			//xr0 xi0 xr1 xi1
			//after permute of vector registers
			//yi0*xr0 yi0*xi0 yi1*xr1 yi1*xi1
			yv[4].v = _mm256_permute_pd( yv[0].v, 15 );
			yv[5].v = _mm256_permute_pd( yv[1].v, 15 );
			yv[6].v = _mm256_permute_pd( yv[2].v, 15 );
			yv[7].v = _mm256_permute_pd( yv[3].v, 15 );

			//yr0 yr0 yr1 yr1
			//xr0 xi0 xr1 xi1
			//after permute of vector registers
			//yr0*xr0 yr0*xi0 yr1*xr1 yr1*xi1
			yv[0].v = _mm256_permute_pd( yv[0].v, 0 );
			yv[1].v = _mm256_permute_pd( yv[1].v, 0 );
			yv[2].v = _mm256_permute_pd( yv[2].v, 0 );
			yv[3].v = _mm256_permute_pd( yv[3].v, 0 );

			// Compute the element-wise product of the x and y vectors,
			// storing in the corresponding rho vectors.
			rhov[0].v = _mm256_fmadd_pd( xv[0].v, yv[0].v, rhov[0].v );
			rhov[1].v = _mm256_fmadd_pd( xv[1].v, yv[1].v, rhov[1].v );
			rhov[2].v = _mm256_fmadd_pd( xv[2].v, yv[2].v, rhov[2].v );
			rhov[3].v = _mm256_fmadd_pd( xv[3].v, yv[3].v, rhov[3].v );

			rhov[4].v = _mm256_fmadd_pd( xv[0].v, yv[4].v, rhov[4].v );
			rhov[5].v = _mm256_fmadd_pd( xv[1].v, yv[5].v, rhov[5].v );
			rhov[6].v = _mm256_fmadd_pd( xv[2].v, yv[6].v, rhov[6].v );
			rhov[7].v = _mm256_fmadd_pd( xv[3].v, yv[7].v, rhov[7].v );

			x0 += ( n_elem_per_reg * n_iter_unroll );
			y0 += ( n_elem_per_reg * n_iter_unroll );
		}
	}
	else
	{
		for ( i = 0; i < n_viter; ++i )
		{
			// Load the x and y input vector elements.
			xv[0].v = _mm256_loadu_pd((double *) (x0 + 0*n_elem_per_reg) );
			yv[0].v = _mm256_loadu_pd((double *) (y0 + 0*n_elem_per_reg) );

			xv[1].v = _mm256_loadu_pd((double *) (x0 + 1*n_elem_per_reg) );
			yv[1].v = _mm256_loadu_pd((double *) (y0 + 1*n_elem_per_reg) );

			xv[2].v = _mm256_loadu_pd((double *) (x0 + 2*n_elem_per_reg) );
			yv[2].v = _mm256_loadu_pd((double *) (y0 + 2*n_elem_per_reg) );

			xv[3].v = _mm256_loadu_pd((double *) (x0 + 3*n_elem_per_reg) );
			yv[3].v = _mm256_loadu_pd((double *) (y0 + 3*n_elem_per_reg) );

			//yi0 yi0 yi1 yi1
			//xr0 xi0 xr1 xi1
			//---------------
			//yi0*xr0 yi0*xi0 yi1*xr1 yi1*xi1
			yv[4].v = _mm256_permute_pd( yv[0].v, 15 );
			yv[5].v = _mm256_permute_pd( yv[1].v, 15 );
			yv[6].v = _mm256_permute_pd( yv[2].v, 15 );
			yv[7].v = _mm256_permute_pd( yv[3].v, 15 );

			//yr0 yr0 yr1 yr1
			//xr0 xi0 xr1 xi1
			//----------------
			//yr0*xr0 yr0*xi0 yr1*xr1 yr1*xi1
			yv[0].v = _mm256_permute_pd( yv[0].v, 0 );
			yv[1].v = _mm256_permute_pd( yv[1].v, 0 );
			yv[2].v = _mm256_permute_pd( yv[2].v, 0 );
			yv[3].v = _mm256_permute_pd( yv[3].v, 0 );

			// Compute the element-wise product of the x and y vectors,
			// storing in the corresponding rho vectors.
			rhov[0].v = _mm256_fmadd_pd( xv[0].v, yv[0].v, rhov[0].v );
			rhov[1].v = _mm256_fmadd_pd( xv[1].v, yv[1].v, rhov[1].v );
			rhov[2].v = _mm256_fmadd_pd( xv[2].v, yv[2].v, rhov[2].v );
			rhov[3].v = _mm256_fmadd_pd( xv[3].v, yv[3].v, rhov[3].v );

			rhov[4].v = _mm256_fmadd_pd( xv[0].v, yv[4].v, rhov[4].v );
			rhov[5].v = _mm256_fmadd_pd( xv[1].v, yv[5].v, rhov[5].v );
			rhov[6].v = _mm256_fmadd_pd( xv[2].v, yv[6].v, rhov[6].v );
			rhov[7].v = _mm256_fmadd_pd( xv[3].v, yv[7].v, rhov[7].v );

			x0 += ( n_elem_per_reg * n_iter_unroll );
			y0 += ( n_elem_per_reg * n_iter_unroll );
		}
	}

	//yr0*xr0 yr0*xi0 yr1*xr1 yr1*xi1
	//   -      +        -      +
	//yi0*xi0 yi0*xr0 yi1*xi1 yi1*xr1
	rhov[4].v = _mm256_permute_pd(rhov[4].v, 0x05);
	rhov[5].v = _mm256_permute_pd(rhov[5].v, 0x05);
	rhov[6].v = _mm256_permute_pd(rhov[6].v, 0x05);
	rhov[7].v = _mm256_permute_pd(rhov[7].v, 0x05);

	rhov[0].v = _mm256_addsub_pd(rhov[0].v, rhov[4].v);
	rhov[1].v = _mm256_addsub_pd(rhov[1].v, rhov[5].v);
	rhov[2].v = _mm256_addsub_pd(rhov[2].v, rhov[6].v);
	rhov[3].v = _mm256_addsub_pd(rhov[3].v, rhov[7].v);

	// Accumulate the unrolled rho vectors into a single vector.
	rhov[0].v = _mm256_add_pd(rhov[1].v,rhov[0].v);
	rhov[0].v = _mm256_add_pd(rhov[2].v,rhov[0].v);
	rhov[0].v = _mm256_add_pd(rhov[3].v,rhov[0].v);

	v2df_t inter1, inter2;

	inter1.v = _mm256_extractf128_pd(rhov[0].v,1);
	inter2.v = _mm256_extractf128_pd(rhov[0].v,0);

	inter1.v = _mm_add_pd(inter1.v, inter2.v);

	// Accumulate the final rho vector into a single scalar result.
	rho0.real = inter1.d[0];
	rho0.imag = inter1.d[1];

	/* Negate sign of imaginary value when vector y is conjugate */
	if ( bli_is_conj(conjx_use))
            rho0.imag = -rho0.imag;

	// Issue vzeroupper instruction to clear upper lanes of ymm registers.
	// This avoids a performance penalty caused by false dependencies when
	// transitioning from AVX to SSE instructions (which may occur as soon
	// as the n_left cleanup loop below if BLIS is compiled with
	// -mfpmath=sse).
	_mm256_zeroupper();

	// If there are leftover iterations, perform them with scalar code.
	if ( bli_is_conj( conjx_use ) )
	{
		for ( i = 0; i < n_left; ++i )
		{
			PASTEMAC(z,dotjs)( *x0, *y0, rho0 );
			x0 += incx;
			y0 += incy;
		}
	}
	else
	{
		for ( i = 0; i < n_left; ++i )
		{
			PASTEMAC(z,dots)( *x0, *y0, rho0 );
			x0 += incx;
			y0 += incy;
		}
	}

	if ( bli_is_conj( conjy ) )
		PASTEMAC(z,conjs)( rho0 );

	// Accumulate the final result into the output variable.
	PASTEMAC(z,axpys)( *alpha, rho0, *rho );
}

void bli_cdotxv_zen_int
     (
       conj_t           conjx,
       conj_t           conjy,
       dim_t            n,
       scomplex* restrict alpha,
       scomplex* restrict x, inc_t incx,
       scomplex* restrict y, inc_t incy,
       scomplex* restrict beta,
       scomplex* restrict rho,
       cntx_t* restrict cntx
     )
{
	const dim_t      n_elem_per_reg = 4;
	const dim_t      n_iter_unroll  = 4;

	dim_t            i;
	dim_t            n_viter;
	dim_t            n_left;

	scomplex* restrict x0;
	scomplex* restrict y0;
	scomplex           rho0;

	v8sf_t           rhov[8], xv[4], yv[8];

	conj_t conjx_use = conjx;
	if ( bli_is_conj( conjy ) )
	{
		bli_toggle_conj( &conjx_use );
	}
	// If beta is zero, initialize rho1 to zero instead of scaling
	// rho by beta (in case rho contains NaN or Inf).
	if ( PASTEMAC(c,eq0)( *beta ) )
	{
		PASTEMAC(c,set0s)( *rho );
	}
	else
	{
		PASTEMAC(c,scals)( *beta, *rho );
	}

	// If the vector dimension is zero, output rho and return early.
	if ( bli_zero_dim1( n ) || PASTEMAC(c,eq0)( *alpha ) ) return;

	// Use the unrolling factor and the number of elements per register
	// to compute the number of vectorized and leftover iterations.
	n_viter = ( n ) / ( n_elem_per_reg * n_iter_unroll );
	n_left  = ( n ) % ( n_elem_per_reg * n_iter_unroll );

	// If there is anything that would interfere with our use of contiguous
	// vector loads/stores, override n_viter and n_left to use scalar code
	// for all iterations.
	if ( incx != 1 || incy != 1 )
	{
		n_viter = 0;
		n_left  = n;
	}

	// Initialize local pointers.
	x0 = x;
	y0 = y;

	// Initialize the unrolled iterations' rho vectors to zero.
	rhov[0].v = _mm256_setzero_ps();
	rhov[1].v = _mm256_setzero_ps();
	rhov[2].v = _mm256_setzero_ps();
	rhov[3].v = _mm256_setzero_ps();

	rhov[4].v = _mm256_setzero_ps();
	rhov[5].v = _mm256_setzero_ps();
	rhov[6].v = _mm256_setzero_ps();
	rhov[7].v = _mm256_setzero_ps();

	if ( bli_is_conj( conjx_use ) )
        {
		__m256 conju = _mm256_setr_ps(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
		for ( i = 0; i < n_viter; ++i )
		{
			// Load the x and y input vector elements.
			xv[0].v = _mm256_loadu_ps((float *) (x0 + 0*n_elem_per_reg) );
			yv[0].v = _mm256_loadu_ps((float *) (y0 + 0*n_elem_per_reg) );

			xv[1].v = _mm256_loadu_ps((float *) (x0 + 1*n_elem_per_reg) );
			yv[1].v = _mm256_loadu_ps((float *) (y0 + 1*n_elem_per_reg) );

			xv[2].v = _mm256_loadu_ps((float *) (x0 + 2*n_elem_per_reg) );
			yv[2].v = _mm256_loadu_ps((float *) (y0 + 2*n_elem_per_reg) );

			xv[3].v = _mm256_loadu_ps((float *) (x0 + 3*n_elem_per_reg) );
			yv[3].v = _mm256_loadu_ps((float *) (y0 + 3*n_elem_per_reg) );

			yv[0].v = _mm256_mul_ps(yv[0].v, conju);
			yv[1].v = _mm256_mul_ps(yv[1].v, conju);
			yv[2].v = _mm256_mul_ps(yv[2].v, conju);
			yv[3].v = _mm256_mul_ps(yv[3].v, conju);
			//yi0 yi0 yi1 yi1
			//xr0 xi0 xr1 xi1
			//after permute of vector registers
			//yi0*xr0 yi0*xi0 yi1*xr1 yi1*xi1
			yv[4].v = _mm256_permute_ps( yv[0].v, 0xf5 );
			yv[5].v = _mm256_permute_ps( yv[1].v, 0xf5 );
			yv[6].v = _mm256_permute_ps( yv[2].v, 0xf5 );
			yv[7].v = _mm256_permute_ps( yv[3].v, 0xf5 );

			//yr0 yr0 yr1 yr1
			//xr0 xi0 xr1 xi1
			//after permute of vector registers
			//yr0*xr0 yr0*xi0 yr1*xr1 yr1*xi1
			yv[0].v = _mm256_permute_ps( yv[0].v, 0xa0 );
			yv[1].v = _mm256_permute_ps( yv[1].v, 0xa0 );
			yv[2].v = _mm256_permute_ps( yv[2].v, 0xa0 );
			yv[3].v = _mm256_permute_ps( yv[3].v, 0xa0 );

			// Compute the element-wise product of the x and y vectors,
			// storing in the corresponding rho vectors.
			rhov[0].v = _mm256_fmadd_ps( xv[0].v, yv[0].v, rhov[0].v );
			rhov[1].v = _mm256_fmadd_ps( xv[1].v, yv[1].v, rhov[1].v );
			rhov[2].v = _mm256_fmadd_ps( xv[2].v, yv[2].v, rhov[2].v );
			rhov[3].v = _mm256_fmadd_ps( xv[3].v, yv[3].v, rhov[3].v );

			rhov[4].v = _mm256_fmadd_ps( xv[0].v, yv[4].v, rhov[4].v );
			rhov[5].v = _mm256_fmadd_ps( xv[1].v, yv[5].v, rhov[5].v );
			rhov[6].v = _mm256_fmadd_ps( xv[2].v, yv[6].v, rhov[6].v );
			rhov[7].v = _mm256_fmadd_ps( xv[3].v, yv[7].v, rhov[7].v );

			x0 += ( n_elem_per_reg * n_iter_unroll );
			y0 += ( n_elem_per_reg * n_iter_unroll );
		}
	}
	else
	{
		for ( i = 0; i < n_viter; ++i )
		{
			// Load the x and y input vector elements.
			xv[0].v = _mm256_loadu_ps((float *) (x0 + 0*n_elem_per_reg) );
			yv[0].v = _mm256_loadu_ps((float *) (y0 + 0*n_elem_per_reg) );

			xv[1].v = _mm256_loadu_ps((float *) (x0 + 1*n_elem_per_reg) );
			yv[1].v = _mm256_loadu_ps((float *) (y0 + 1*n_elem_per_reg) );

			xv[2].v = _mm256_loadu_ps((float *) (x0 + 2*n_elem_per_reg) );
			yv[2].v = _mm256_loadu_ps((float *) (y0 + 2*n_elem_per_reg) );

			xv[3].v = _mm256_loadu_ps((float *) (x0 + 3*n_elem_per_reg) );
			yv[3].v = _mm256_loadu_ps((float *) (y0 + 3*n_elem_per_reg) );

			//yi0 yi0 yi1 yi1
			//xr0 xi0 xr1 xi1
			//---------------
			//yi0*xr0 yi0*xi0 yi1*xr1 yi1*xi1
			yv[4].v = _mm256_permute_ps( yv[0].v, 0xf5 );
			yv[5].v = _mm256_permute_ps( yv[1].v, 0xf5 );
			yv[6].v = _mm256_permute_ps( yv[2].v, 0xf5 );
			yv[7].v = _mm256_permute_ps( yv[3].v, 0xf5 );

			//yr0 yr0 yr1 yr1
			//xr0 xi0 xr1 xi1
			//----------------
			//yr0*xr0 yr0*xi0 yr1*xr1 yr1*xi1
			yv[0].v = _mm256_permute_ps( yv[0].v, 0xa0 );
			yv[1].v = _mm256_permute_ps( yv[1].v, 0xa0 );
			yv[2].v = _mm256_permute_ps( yv[2].v, 0xa0 );
			yv[3].v = _mm256_permute_ps( yv[3].v, 0xa0 );

			// Compute the element-wise product of the x and y vectors,
			// storing in the corresponding rho vectors.
			rhov[0].v = _mm256_fmadd_ps( xv[0].v, yv[0].v, rhov[0].v );
			rhov[1].v = _mm256_fmadd_ps( xv[1].v, yv[1].v, rhov[1].v );
			rhov[2].v = _mm256_fmadd_ps( xv[2].v, yv[2].v, rhov[2].v );
			rhov[3].v = _mm256_fmadd_ps( xv[3].v, yv[3].v, rhov[3].v );

			rhov[4].v = _mm256_fmadd_ps( xv[0].v, yv[4].v, rhov[4].v );
			rhov[5].v = _mm256_fmadd_ps( xv[1].v, yv[5].v, rhov[5].v );
			rhov[6].v = _mm256_fmadd_ps( xv[2].v, yv[6].v, rhov[6].v );
			rhov[7].v = _mm256_fmadd_ps( xv[3].v, yv[7].v, rhov[7].v );

			x0 += ( n_elem_per_reg * n_iter_unroll );
			y0 += ( n_elem_per_reg * n_iter_unroll );
		}
	}

	//yr0*xr0 yr0*xi0 yr1*xr1 yr1*xi1
	//   -      +        -      +
	//yi0*xi0 yi0*xr0 yi1*xi1 yi1*xr1
	rhov[4].v = _mm256_permute_ps(rhov[4].v, 0xb1);
	rhov[5].v = _mm256_permute_ps(rhov[5].v, 0xb1);
	rhov[6].v = _mm256_permute_ps(rhov[6].v, 0xb1);
	rhov[7].v = _mm256_permute_ps(rhov[7].v, 0xb1);

	rhov[0].v = _mm256_addsub_ps(rhov[0].v, rhov[4].v);
	rhov[1].v = _mm256_addsub_ps(rhov[1].v, rhov[5].v);
	rhov[2].v = _mm256_addsub_ps(rhov[2].v, rhov[6].v);
	rhov[3].v = _mm256_addsub_ps(rhov[3].v, rhov[7].v);

	// Accumulate the unrolled rho vectors into a single vector.
	rhov[0].v = _mm256_add_ps(rhov[1].v,rhov[0].v);
	rhov[0].v = _mm256_add_ps(rhov[2].v,rhov[0].v);
	rhov[0].v = _mm256_add_ps(rhov[3].v,rhov[0].v);

	v4sf_t inter1, inter2;

	inter1.v = _mm256_extractf128_ps(rhov[0].v,1);
	inter2.v = _mm256_extractf128_ps(rhov[0].v,0);

	inter1.v = _mm_add_ps(inter1.v, inter2.v);

	// Accumulate the final rho vector into a single scalar result.
	rho0.real = inter1.f[0] + inter1.f[2];
	rho0.imag = inter1.f[1] + inter1.f[3];

	/* Negate sign of imaginary value when vector y is conjugate */
	if ( bli_is_conj(conjx_use))
            rho0.imag = -rho0.imag;

	// Issue vzeroupper instruction to clear upper lanes of ymm registers.
	// This avoids a performance penalty caused by false dependencies when
	// transitioning from AVX to SSE instructions (which may occur as soon
	// as the n_left cleanup loop below if BLIS is compiled with
	// -mfpmath=sse).
	_mm256_zeroupper();

	// If there are leftover iterations, perform them with scalar code.
	if ( bli_is_conj( conjx_use ) )
	{
		for ( i = 0; i < n_left; ++i )
		{
			PASTEMAC(c,dotjs)( *x0, *y0, rho0 );
			x0 += incx;
			y0 += incy;
		}
	}
	else
	{
		for ( i = 0; i < n_left; ++i )
		{
			PASTEMAC(c,dots)( *x0, *y0, rho0 );
			x0 += incx;
			y0 += incy;
		}
	}

	if ( bli_is_conj( conjy ) )
		PASTEMAC(c,conjs)( rho0 );

	// Accumulate the final result into the output variable.
	PASTEMAC(c,axpys)( *alpha, rho0, *rho );
}
