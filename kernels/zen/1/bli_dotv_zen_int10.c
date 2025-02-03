/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2016 - 2025, Advanced Micro Devices, Inc. All rights reserved.
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


//Loads lower 3 64-bit double precision elements into ymm register
static int64_t mask_3[4] = {-1, -1, -1, 0};
//Loads lower 2 64-bit double precision elements into ymm register
static int64_t mask_2[4] = {-1, -1, 0, 0};
//Loads lower 1 64-bit double precision elements into ymm register
static int64_t mask_1[4] = {-1, 0, 0, 0};
//Loads 4 64-bit double precision elements into ymm register
static int64_t mask_0[4] = {0, 0, 0, 0};

static int64_t *mask_ptr[] = {mask_0, mask_1, mask_2, mask_3};
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

	float            rho0 = 0.0;

	__m256           xv[10];
	__m256           yv[10];
	v8sf_t           rhov[10];

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
		rhov[2].v = _mm256_setzero_ps();
		rhov[3].v = _mm256_setzero_ps();
		rhov[4].v = _mm256_setzero_ps();
		rhov[5].v = _mm256_setzero_ps();
		rhov[6].v = _mm256_setzero_ps();
		rhov[7].v = _mm256_setzero_ps();
		rhov[8].v = _mm256_setzero_ps();
		rhov[9].v = _mm256_setzero_ps();

		for ( i = 0 ; (i + 79) < n; i += 80 )
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
			rhov[2].v = _mm256_fmadd_ps( xv[2], yv[2], rhov[2].v );
			rhov[3].v = _mm256_fmadd_ps( xv[3], yv[3], rhov[3].v );
			rhov[4].v = _mm256_fmadd_ps( xv[4], yv[4], rhov[4].v );
			rhov[5].v = _mm256_fmadd_ps( xv[5], yv[5], rhov[5].v );
			rhov[6].v = _mm256_fmadd_ps( xv[6], yv[6], rhov[6].v );
			rhov[7].v = _mm256_fmadd_ps( xv[7], yv[7], rhov[7].v );
			rhov[8].v = _mm256_fmadd_ps( xv[8], yv[8], rhov[8].v );
			rhov[9].v = _mm256_fmadd_ps( xv[9], yv[9], rhov[9].v );

			x0 += 10*n_elem_per_reg;
			y0 += 10*n_elem_per_reg;
		}

		rhov[0].v += rhov[5].v;
		rhov[1].v += rhov[6].v;
		rhov[2].v += rhov[7].v;
		rhov[3].v += rhov[8].v;
		rhov[4].v += rhov[9].v;

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
			rhov[2].v = _mm256_fmadd_ps( xv[2], yv[2], rhov[2].v );
			rhov[3].v = _mm256_fmadd_ps( xv[3], yv[3], rhov[3].v );
			rhov[4].v = _mm256_fmadd_ps( xv[4], yv[4], rhov[4].v );

			x0 += 5*n_elem_per_reg;
			y0 += 5*n_elem_per_reg;
		}

		rhov[0].v += rhov[2].v;
		rhov[1].v += rhov[3].v;
		rhov[0].v += rhov[4].v;

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

		rhov[0].v += rhov[1].v;

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
			rho0 += (*x0) * (*y0);
			x0 += 1;
			y0 += 1;
		}

		rho0 += rhov[0].f[0] + rhov[0].f[1] +
		        rhov[0].f[2] + rhov[0].f[3] +
		        rhov[0].f[4] + rhov[0].f[5] +
		        rhov[0].f[6] + rhov[0].f[7];

		// Issue vzeroupper instruction to clear upper lanes of ymm registers.
		// This avoids a performance penalty caused by false dependencies when
		// transitioning from AVX to SSE instructions (which may occur later,
		// especially if BLIS is compiled with -mfpmath=sse).
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

	double           rho0 = 0.0;

	__m256d          xv[5];
	__m256d          yv[5];
	__m256d          rhov[5];
	v4df_t           rh;

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
		rhov[0] = _mm256_setzero_pd();
		rhov[1] = _mm256_setzero_pd();
		rhov[2] = _mm256_setzero_pd();
		rhov[3] = _mm256_setzero_pd();
		rhov[4] = _mm256_setzero_pd();

		for ( i = 0; (i + 19) < n; i += 20 )
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

			rhov[0] = _mm256_fmadd_pd( xv[0], yv[0], rhov[0] );
			rhov[1] = _mm256_fmadd_pd( xv[1], yv[1], rhov[1] );
			rhov[2] = _mm256_fmadd_pd( xv[2], yv[2], rhov[2] );
			rhov[3] = _mm256_fmadd_pd( xv[3], yv[3], rhov[3] );
			rhov[4] = _mm256_fmadd_pd( xv[4], yv[4], rhov[4] );

			x0 += 5*n_elem_per_reg;
			y0 += 5*n_elem_per_reg;
		}

		rhov[0] = _mm256_add_pd( rhov[3], rhov[0]) ;
		rhov[1] = _mm256_add_pd( rhov[4], rhov[1]) ;

		if ( (i + 11) < n )
		{
			xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_pd( x0 + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_pd( x0 + 2*n_elem_per_reg );

			yv[0] = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
			yv[1] = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );
			yv[2] = _mm256_loadu_pd( y0 + 2*n_elem_per_reg );

			rhov[0] = _mm256_fmadd_pd( xv[0], yv[0], rhov[0] );
			rhov[1] = _mm256_fmadd_pd( xv[1], yv[1], rhov[1] );
			rhov[2] = _mm256_fmadd_pd( xv[2], yv[2], rhov[2] );

			x0 += 3*n_elem_per_reg;
			y0 += 3*n_elem_per_reg;
			i  += 3*n_elem_per_reg;
		}

		rhov[0] = _mm256_add_pd( rhov[2], rhov[0]) ;

		if ( (i + 7) < n )
		{
			xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_pd( x0 + 1*n_elem_per_reg );

			yv[0] = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
			yv[1] = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );

			rhov[0] = _mm256_fmadd_pd( xv[0], yv[0], rhov[0] );
			rhov[1] = _mm256_fmadd_pd( xv[1], yv[1], rhov[1] );

			x0 += 2*n_elem_per_reg;
			y0 += 2*n_elem_per_reg;
			i  += 2*n_elem_per_reg;
		}

		rhov[0] = _mm256_add_pd( rhov[1], rhov[0]) ;

		if ( (i + 3) < n )
		{
			xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );

			yv[0] = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );

			rhov[0] = _mm256_fmadd_pd( xv[0], yv[0], rhov[0] );

			x0 += n_elem_per_reg;
			y0 += n_elem_per_reg;
			i  += n_elem_per_reg;
		}

		if( i < n )
		{
			__m256i maskVec = _mm256_loadu_si256( (__m256i *)mask_ptr[(n - i)]);

			xv[0] = _mm256_maskload_pd( x0, maskVec );
			yv[0] = _mm256_maskload_pd( y0, maskVec );

			rhov[0] = _mm256_fmadd_pd( xv[0], yv[0], rhov[0] );
			i = n;
		}

		// Perform horizontal addition of the elements in the vector.
		rh.v = _mm256_hadd_pd( rhov[0], rhov[0] );

		// Manually add the first and third element from above vector to finish the sum.
		rho0 += rh.d[0]  + rh.d[2];

		// Issue vzeroupper instruction to clear upper lanes of ymm registers.
		// This avoids a performance penalty caused by false dependencies when
		// transitioning from AVX to SSE instructions (which may occur later,
		// especially if BLIS is compiled with -mfpmath=sse).
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

// -----------------------------------------------------------------------------


void bli_cdotv_zen_int5
     (
       conj_t           conjx,
       conj_t           conjy,
       dim_t            n,
       scomplex*  restrict x, inc_t incx,
       scomplex*  restrict y, inc_t incy,
       scomplex*  restrict rho,
       cntx_t* restrict cntx
     )
{
    const dim_t      n_elem_per_reg = 8;

    dim_t            i;

    float*  restrict x0;
    float*  restrict y0;

    scomplex    rho0 ;
    rho0.real = 0.0;
    rho0.imag = 0.0;

    __m256           xv[5];
    __m256           yv[5];
    __m256           zv[5];
    v8sf_t           rhov[10];

    conj_t conjx_use = conjx;
    /* If y must be conjugated, we do so indirectly by first toggling the
        effective conjugation of x and then conjugating the resulting dot
        product. */
    if ( bli_is_conj( conjy ) )
        bli_toggle_conj( &conjx_use );

    // If the vector dimension is zero, or if alpha is zero, return early.
    if ( bli_zero_dim1( n ) )
    {
        PASTEMAC(c,set0s)( *rho );
        return;
    }

    // Initialize local pointers.
    x0 = (float*) x;
    y0 = (float*) y;

    PASTEMAC(c,set0s)( rho0 );

    /*
     * Computing dot product of 2 complex vectors
     * dotProd = Σ (xr + i xi) * (yr - i yi)
     * dotProdReal = xr1 * yr1 + xi1 * yi1 +  xr2 * yr2 + xi2 * yi2 + .... +  xrn * yrn + xin * yin
     * dotProdImag = -( xr1 * yi1 - xi1 * yr1 +  xr2 * yi2 - xi2 * yr2 + .... +  xrn * yin - xin * yrn)
     * Product of vectors are carried out using intrinsics code _mm256_fmadd_ps with 256bit register
     * Each element of 256bit register is added/subtracted based on element position
     */
    if ( incx == 1 && incy == 1 )
    {
        /* Set of registers used to compute real value of dot product */
        rhov[0].v = _mm256_setzero_ps();
        rhov[1].v = _mm256_setzero_ps();
        rhov[2].v = _mm256_setzero_ps();
        rhov[3].v = _mm256_setzero_ps();
        rhov[4].v = _mm256_setzero_ps();
        /* set of registers used to compute imag value of dot product */
        rhov[5].v = _mm256_setzero_ps();
        rhov[6].v = _mm256_setzero_ps();
        rhov[7].v = _mm256_setzero_ps();
        rhov[8].v = _mm256_setzero_ps();
        rhov[9].v = _mm256_setzero_ps();

        /*
         * Compute of 1-256bit register
         * xv = xr1  xi1  xr2  xi2  xr3  xi3  xr4  xi4
         * yv = yr1  yi1  yr2  yi2  yr3  yi3  yr4  yi4
         * zv = yi1  yr1  yi2  yr2  yi3  yr3  yi4  yr4
         * rhov0(real) = xr1*yr1, xi1*yi1, xr2*yr2, xi2*yi2, xr3*yr3, xi3*yi3, xr4*yr4, xi4*yi4
         * rhov5(imag) = xr1*yi1, xi1*yr1, xr2*yi2, xi2*yr2, xr3*yi3, xi3*yr3, xr4*yi4, xi4*yr4
         */
        for (i=0 ; (i + 19) < n; i += 20 )
        {
            // 20 elements will be processed per loop; 10 FMAs will run per loop.
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

            /* Permute step is swapping real and imaginary values.
            yv = yr1 yi1 yr2 yi2 yr3 yi3 yr4 yi4
            zv = yi1 yr1 yi2 yr2 yi3 yr3 yi4 yr4
            zv is required to compute imaginary values */
            zv[0] = _mm256_permute_ps( yv[0], 0xB1 );
            zv[1] = _mm256_permute_ps( yv[1], 0xB1 );
            zv[2] = _mm256_permute_ps( yv[2], 0xB1 );
            zv[3] = _mm256_permute_ps( yv[3], 0xB1 );
            zv[4] = _mm256_permute_ps( yv[4], 0xB1 );

            /* Compute real values */
            rhov[0].v = _mm256_fmadd_ps( xv[0], yv[0], rhov[0].v );
            rhov[1].v = _mm256_fmadd_ps( xv[1], yv[1], rhov[1].v );
            rhov[2].v = _mm256_fmadd_ps( xv[2], yv[2], rhov[2].v );
            rhov[3].v = _mm256_fmadd_ps( xv[3], yv[3], rhov[3].v );
            rhov[4].v = _mm256_fmadd_ps( xv[4], yv[4], rhov[4].v );

            /* Compute imaginary values*/
            rhov[5].v = _mm256_fmadd_ps( xv[0], zv[0], rhov[5].v );
            rhov[6].v = _mm256_fmadd_ps( xv[1], zv[1], rhov[6].v );
            rhov[7].v = _mm256_fmadd_ps( xv[2], zv[2], rhov[7].v );
            rhov[8].v = _mm256_fmadd_ps( xv[3], zv[3], rhov[8].v );
            rhov[9].v = _mm256_fmadd_ps( xv[4], zv[4], rhov[9].v );

            x0 += 5 * n_elem_per_reg;
            y0 += 5 * n_elem_per_reg;
        }

        /* Real value computation: rhov[0] & rhov[1] used in below
           for loops hence adding up other register values */
        rhov[0].v += rhov[2].v;
        rhov[1].v += rhov[3].v;
        rhov[0].v += rhov[4].v;

        /* Imag value computation: rhov[5] & rhov[6] used in below
           for loops hence adding up other register values */
        rhov[5].v += rhov[7].v;
        rhov[6].v += rhov[8].v;
        rhov[5].v += rhov[9].v;

        for ( ; (i + 7) < n; i += 8 )
        {
            xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
            xv[1] = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );

            yv[0] = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );
            yv[1] = _mm256_loadu_ps( y0 + 1*n_elem_per_reg );

            /* Permute step is swapping real and imaginary values.
            yv = yr1 yi1 yr2 yi2 yr3 yi3 yr4 yi4
            zv = yi1 yr1 yi2 yr2 yi3 yr3 yi4 yr4
            zv is required to compute imaginary values */
            zv[0] = _mm256_permute_ps( yv[0], 0xB1 );
            zv[1] = _mm256_permute_ps( yv[1], 0xB1 );

            /* Compute real values */
            rhov[0].v = _mm256_fmadd_ps( xv[0], yv[0], rhov[0].v );
            rhov[1].v = _mm256_fmadd_ps( xv[1], yv[1], rhov[1].v );

            /* Compute imaginary values*/
            rhov[5].v = _mm256_fmadd_ps( xv[0], zv[0], rhov[5].v );
            rhov[6].v = _mm256_fmadd_ps( xv[1], zv[1], rhov[6].v );

            x0 += 2 * n_elem_per_reg;
            y0 += 2 * n_elem_per_reg;
        }

        /*Accumalte real values in to rhov[0]*/
        rhov[0].v += rhov[1].v;

        /*Accumalte imaginary values in to rhov[5]*/
        rhov[5].v += rhov[6].v;

        for ( ; (i + 3) < n; i += 4 )
        {
            xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );

            yv[0] = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );

           /* Permute step is swapping real and imaginary values.
            yv = yr1 yi1 yr2 yi2 yr3 yi3 yr4 yi4
            zv = yi1 yr1 yi2 yr2 yi3 yr3 yi4 yr4
            zv is required to compute imaginary values */
            zv[0] = _mm256_permute_ps( yv[0], 0xB1 );

            /* Compute real values */
            rhov[0].v = _mm256_fmadd_ps( xv[0], yv[0], rhov[0].v );

            /* Compute imaginary values*/
            rhov[5].v = _mm256_fmadd_ps( xv[0], zv[0], rhov[5].v );

            x0 += 1 * n_elem_per_reg;
            y0 += 1 * n_elem_per_reg;
        }


        /* Residual values are calculated here
           rho := conjx(x)^T * conjy(y)
            n = 1, When no conjugate for x or y vector
            rho = conj(xr + xi) * conj(yr + yi)
            rho = (xr - xi) * (yr -yi)
            rho.real = xr*yr + xi*yi
            rho.imag = -(xi*yr - xr *yi)
            -ve sign of imaginary value is taken care at the end of function
           When vector x/y to be conjugated, imaginary values(xi and yi) to be negated
        */
        if ( !bli_is_conj(conjx_use) )
        {
            for ( ; (i + 0) < n; i += 1 )
            {
                rho0.real += (*x0) * (*y0) - (*(x0+1)) * (*(y0+1));
                rho0.imag += (*x0) * (*(y0+1)) + (*(x0+1)) * (*y0);
                x0 += 2;
                y0 += 2;
            }
        }
        else
        {
            for ( ; (i + 0) < n; i += 1 )
            {
                rho0.real += (*x0) * (*y0) + (*(x0+1)) * (*(y0+1));
                rho0.imag += (*x0) * (*(y0+1)) - (*(x0+1)) * (*y0);
                x0 += 2;
                y0 += 2;
            }
        }

        /* Find dot product by summing up all elements */
        if ( !bli_is_conj(conjx_use) )
        {
                rho0.real += rhov[0].f[0] - rhov[0].f[1] +
                             rhov[0].f[2] - rhov[0].f[3] +
                             rhov[0].f[4] - rhov[0].f[5] +
                             rhov[0].f[6] - rhov[0].f[7];

                rho0.imag += rhov[5].f[0] + rhov[5].f[1] +
                             rhov[5].f[2] + rhov[5].f[3] +
                             rhov[5].f[4] + rhov[5].f[5] +
                             rhov[5].f[6] + rhov[5].f[7];
        }
        else
        {
               rho0.real += rhov[0].f[0] + rhov[0].f[1] +
                            rhov[0].f[2] + rhov[0].f[3] +
                            rhov[0].f[4] + rhov[0].f[5] +
                            rhov[0].f[6] + rhov[0].f[7];

               rho0.imag += rhov[5].f[0] - rhov[5].f[1] +
                            rhov[5].f[2] - rhov[5].f[3] +
                            rhov[5].f[4] - rhov[5].f[5] +
                            rhov[5].f[6] - rhov[5].f[7];
        }

        /* Negate sign of imaginary value when vector y is conjugate */
        if ( bli_is_conj(conjy) ) {
            rho0.imag = -rho0.imag;
        }
        // Issue vzeroupper instruction to clear upper lanes of ymm registers.
        // This avoids a performance penalty caused by false dependencies when
        // transitioning from AVX to SSE instructions (which may occur later,
        // especially if BLIS is compiled with -mfpmath=sse).
        _mm256_zeroupper();
    }
    else
    {
        /* rho := conjx(x)^T * conjy(y)
           n = 1, When no conjugate for x or y vector
                rho = conj(xr + xi) * conj(yr + yi)
                rho = (xr - xi) * (yr -yi)
                rho.real = xr*yr + xi*yi
                rho.imag = -(xi*yr - xr *yi)
                -ve sign of imaginary value is taken care at the end of function
           When vector x/y to be conjugated, imaginary values(xi and yi) to be negated
        */
        if ( !bli_is_conj(conjx_use) )
        {
            for ( i = 0; i < n; ++i )
            {
                const float x0c = *x0;
                const float y0c = *y0;

                const float x1c = *( x0+1 );
                const float y1c = *( y0+1 );

                rho0.real += x0c * y0c - x1c * y1c;
                rho0.imag += x0c * y1c + x1c * y0c;

                x0 += incx * 2;
                y0 += incy * 2;
            }
        }
        else
        {
            for ( i = 0; i < n; ++i )
            {
                const float x0c = *x0;
                const float y0c = *y0;

                const float x1c = *( x0+1 );
                const float y1c = *( y0+1 );

                rho0.real += x0c * y0c + x1c * y1c;
                rho0.imag += x0c * y1c - x1c * y0c;

                x0+= incx * 2;
                y0+= incy * 2;
            }
        }

        /* Negate sign of imaginary value when vector y is conjugate */
        if( bli_is_conj(conjy) )
            rho0.imag = -rho0.imag;
    }

    // Copy the final result into the output variable.
    PASTEMAC(c,copys)( rho0, *rho );
}


// -----------------------------------------------------------------------------

void bli_zdotv_zen_int5
    (
      conj_t           conjx,
      conj_t           conjy,
      dim_t            n,
      dcomplex* restrict x, inc_t incx,
      dcomplex* restrict y, inc_t incy,
      dcomplex* restrict rho,
      cntx_t* restrict cntx
    )
{
    const dim_t      n_elem_per_reg = 4;

    dim_t            i;

    double* restrict x0;
    double* restrict y0;

    dcomplex           rho0 ;

    rho0.real = 0.0;
    rho0.imag = 0.0;

    __m256d          xv[5];
    __m256d          yv[5];
    __m256d          zv[5];
    v4df_t           rhov[10];

    conj_t conjx_use = conjx;
    /* If y must be conjugated, we do so indirectly by first toggling the
        effective conjugation of x and then conjugating the resulting dot
        product. */
    if ( bli_is_conj( conjy ) )
        bli_toggle_conj( &conjx_use );

    // If the vector dimension is zero, or if alpha is zero, return early.
    if ( bli_zero_dim1( n ) )
    {
        PASTEMAC(z,set0s)( *rho );
        return;
    }

    // Initialize local pointers.
    x0 = (double *) x;
    y0 = (double *) y;

    PASTEMAC(z,set0s)( rho0 );

     /*
     * Computing dot product of 2 complex vectors
     * dotProd = Σ (xr + i xi) * (yr - iyi)
     * dotProdReal = xr1 * yr1 + xi1 * yi1 +  xr2 * yr2 + xi2 * yi2 + .... +  xrn * yrn + xin * yin
     * dotProdImag = xi1 * yr1 - xr1 * yi1 +  xi2 * yr2 - xr2 * yi2 + .... +  xin * yrn - xrn * yin
     * Product of vectors are carried out using intrinsics code _mm256_fmadd_ps with 256bit register
     * Each element of 256bit register is added/subtracted based on element position
     */
    if ( incx == 1 && incy == 1 )
    {
        /* Set of registers used to compute real value of dot product */
        rhov[0].v = _mm256_setzero_pd();
        rhov[1].v = _mm256_setzero_pd();
        rhov[2].v = _mm256_setzero_pd();
        rhov[3].v = _mm256_setzero_pd();
        rhov[4].v = _mm256_setzero_pd();
        /* Set of registers used to compute real value of dot product */
        rhov[5].v = _mm256_setzero_pd();
        rhov[6].v = _mm256_setzero_pd();
        rhov[7].v = _mm256_setzero_pd();
        rhov[8].v = _mm256_setzero_pd();
        rhov[9].v = _mm256_setzero_pd();

        /*
         * Compute of 1-256bit register
         * xv = xr1  xi1  xr2  xi2
         * yv = yr1  yi1  yr1  yi2
         * zv = yi1  yr1  yi1  yr2
         * rhov0(real) = xr1*yr1, xi1*yi1, xr2*yr2, xi2*yi2
         * rhov5(imag) = xr1*yi1, xi1*yr1, xr2*yi2, xi2*yr2
         */
        for ( i = 0; (i + 9) < n; i += 10 )
        {
            // 10 elements will be processed per loop; 10 FMAs will run per loop.
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

            /* Permute step is swapping real and imaginary values.
            yv = yr1 yi1 yr2 yi2
            zv = yi1 yr1 yi2 yr2
            zv is required to compute imaginary values */
            zv[0] = _mm256_permute_pd( yv[0], 5 );
            zv[1] = _mm256_permute_pd( yv[1], 5 );
            zv[2] = _mm256_permute_pd( yv[2], 5 );
            zv[3] = _mm256_permute_pd( yv[3], 5 );
            zv[4] = _mm256_permute_pd( yv[4], 5 );

            rhov[0].v = _mm256_fmadd_pd( xv[0], yv[0], rhov[0].v );
            rhov[1].v = _mm256_fmadd_pd( xv[1], yv[1], rhov[1].v );
            rhov[2].v = _mm256_fmadd_pd( xv[2], yv[2], rhov[2].v );
            rhov[3].v = _mm256_fmadd_pd( xv[3], yv[3], rhov[3].v );
            rhov[4].v = _mm256_fmadd_pd( xv[4], yv[4], rhov[4].v );
            rhov[5].v = _mm256_fmadd_pd( xv[0], zv[0], rhov[5].v );
            rhov[6].v = _mm256_fmadd_pd( xv[1], zv[1], rhov[6].v );
            rhov[7].v = _mm256_fmadd_pd( xv[2], zv[2], rhov[7].v );
            rhov[8].v = _mm256_fmadd_pd( xv[3], zv[3], rhov[8].v );
            rhov[9].v = _mm256_fmadd_pd( xv[4], zv[4], rhov[9].v );

            x0 += 5*n_elem_per_reg;
            y0 += 5*n_elem_per_reg;
        }

        /* Real value computation: rhov[0] & rhov[1] used in below
           for loops hence adding up other register values */
        rhov[0].v += rhov[2].v;
        rhov[1].v += rhov[3].v;
        rhov[0].v += rhov[4].v;

        /* Imag value computation: rhov[5] & rhov[6] used in below
           for loops hence adding up other register values */
        rhov[5].v += rhov[7].v;
        rhov[6].v += rhov[8].v;
        rhov[5].v += rhov[9].v;

        for ( ; (i + 3) < n; i += 4 )
        {
            xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
            xv[1] = _mm256_loadu_pd( x0 + 1*n_elem_per_reg );

            yv[0] = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
            yv[1] = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );

            /* Permute step is swapping real and imaginary values.
            yv = yr1 yi1 yr2 yi2
            zv = yi1 yr1 yi2 yr2
            zv is required to compute imaginary values */
            zv[0] = _mm256_permute_pd( yv[0], 5 );
            zv[1] = _mm256_permute_pd( yv[1], 5 );

            rhov[0].v = _mm256_fmadd_pd( xv[0], yv[0], rhov[0].v );
            rhov[1].v = _mm256_fmadd_pd( xv[1], yv[1], rhov[1].v );
            rhov[5].v = _mm256_fmadd_pd( xv[0], zv[0], rhov[5].v );
            rhov[6].v = _mm256_fmadd_pd( xv[1], zv[1], rhov[6].v );

            x0 += 2*n_elem_per_reg;
            y0 += 2*n_elem_per_reg;
        }

        /*Accumulate real values*/
        rhov[0].v += rhov[1].v;
        /*Accumulate imaginary values*/
        rhov[5].v += rhov[6].v;

        for ( ; (i + 3) < n; i += 2 )
        {
            xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );

            yv[0] = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );

        /* Permute step is swapping real and imaginary values.
            yv = yr1 yi1 yr2 yi2
            zv = yi1 yr1 yi2 yr2
            zv is required to compute imaginary values */
            zv[0] = _mm256_permute_pd( yv[0], 5 );

            rhov[0].v = _mm256_fmadd_pd( xv[0], yv[0], rhov[0].v );
            rhov[5].v = _mm256_fmadd_pd( xv[0], zv[0], rhov[5].v );

            x0 += 1*n_elem_per_reg;
            y0 += 1*n_elem_per_reg;
        }

        /* Residual values are calculated here
           rho := conjx(x)^T * conjy(y)
           n = 1, When no conjugate for x or y vector
                rho = conj(xr + xi) * conj(yr + yi)
                rho = (xr - xi) * (yr -yi)
                rho.real = xr*yr + xi*yi
                rho.imag = -(xi*yr - xr *yi)
                -ve sign of imaginary value is taken care at the end of function
          When vector x/y to be conjugated, imaginary values(xi and yi) to be negated
        */
        if ( !bli_is_conj(conjx_use) )
        {
            for ( ; (i + 0) < n; i += 1 )
            {
                rho0.real += ( *x0 ) * ( *y0 ) - ( *(x0+1) ) * ( *( y0+1 ) );
                rho0.imag += ( *x0 ) * ( *(y0+1) ) + ( *(x0+1) ) * ( *y0 );
                x0 += 2;
                y0 += 2;
            }
        }
        else
        {
            for ( ; (i + 0) < n; i += 1 )
            {
                rho0.real += ( *x0 ) * ( *y0 ) + ( *( x0+1 ) ) * ( *( y0+1 ) );
                rho0.imag += ( *x0 ) * ( *( y0+1 ) ) - ( *( x0+1 ) ) * ( *y0 );
                x0 += 2;
                y0 += 2;
            }
        }

        /* Find dot product by summing up all elements */
        if ( !bli_is_conj(conjx_use) )
        {
            rho0.real += rhov[0].d[0] - rhov[0].d[1] + rhov[0].d[2] - rhov[0].d[3];
            rho0.imag += rhov[5].d[0] + rhov[5].d[1] + rhov[5].d[2] + rhov[5].d[3];
        }
        else
        {
            rho0.real += rhov[0].d[0] + rhov[0].d[1] + rhov[0].d[2] + rhov[0].d[3];
            rho0.imag += rhov[5].d[0] - rhov[5].d[1] + rhov[5].d[2] - rhov[5].d[3];
        }
        /* Negate sign of imaginary value when vector y is conjugate */
        if ( bli_is_conj(conjy) )
            rho0.imag = -rho0.imag;

        // Issue vzeroupper instruction to clear upper lanes of ymm registers.
        // This avoids a performance penalty caused by false dependencies when
        // transitioning from AVX to SSE instructions (which may occur later,
        // especially if BLIS is compiled with -mfpmath=sse).
        _mm256_zeroupper();
    }
    else
    {
        /*  rho := conjx(x)^T * conjy(y)
            n = 1, When no conjugate for x or y vector
                rho = conj(xr + xi) * conj(yr + yi)
                rho = (xr - xi) * (yr -yi)
                rho.real = xr*yr + xi*yi
                rho.imag = -(xi*yr - xr *yi)
                -ve sign of imaginary value is taken care at the end of function
           When vector x/y to be conjugated, imaginary values(xi and yi) to be negated
        */
        if ( !bli_is_conj(conjx_use) )
        {
            for ( i = 0; i < n; ++i )
            {
                const double x0c = *x0;
                const double y0c = *y0;

                const double x1c = *( x0 + 1 );
                const double y1c = *( y0 + 1 );

                rho0.real += x0c * y0c - x1c * y1c;
                rho0.imag += x0c * y1c + x1c * y0c;

                x0 += incx * 2;
                y0 += incy * 2;
            }
        }
        else
        {
            for ( i = 0; i < n; ++i )
            {
                const double x0c = *x0;
                const double y0c = *y0;

                const double x1c = *( x0 + 1 );
                const double y1c = *( y0 + 1 );

                rho0.real += x0c * y0c + x1c * y1c;
                rho0.imag += x0c * y1c - x1c * y0c;

                x0 += incx * 2;
                y0 += incy * 2;
            }
        }
        /* Negate sign of imaginary value when vector y is conjugate */
        if ( bli_is_conj(conjy) )
            rho0.imag = -rho0.imag;
    }

    // Copy the final result into the output variable.
    PASTEMAC(z,copys)( rho0, *rho );
}
