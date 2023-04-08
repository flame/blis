/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2018, The University of Texas at Austin
   Copyright (C) 2016 - 2018, Advanced Micro Devices, Inc.

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

void bli_saxpyf_zen_int_8
     (
             conj_t  conja,
             conj_t  conjx,
             dim_t   m,
             dim_t   b_n,
       const void*   alpha0,
       const void*   a0, inc_t inca, inc_t lda,
       const void*   x0, inc_t incx,
             void*   y0, inc_t incy,
       const cntx_t* cntx
     )
{
	const float* restrict alpha = alpha0;
	const float* restrict a     = a0;
	const float* restrict x     = x0;
	      float* restrict y     = y0;

	const dim_t      fuse_fac       = 8;

	const dim_t      n_elem_per_reg = 8;
	const dim_t      n_iter_unroll  = 1;

	dim_t            i;
	dim_t            m_viter;
	dim_t            m_left;

	v8sf_t           chi0v, chi1v, chi2v, chi3v;
	v8sf_t           chi4v, chi5v, chi6v, chi7v;

	v8sf_t           a0v, a1v, a2v, a3v;
	v8sf_t           a4v, a5v, a6v, a7v;
	v8sf_t           y0v;

	float            chi0, chi1, chi2, chi3;
	float            chi4, chi5, chi6, chi7;

	// If either dimension is zero, or if alpha is zero, return early.
	if ( bli_zero_dim2( m, b_n ) || PASTEMAC(s,eq0)( *alpha ) ) return;

	// If b_n is not equal to the fusing factor, then perform the entire
	// operation as a loop over axpyv.
	if ( b_n != fuse_fac )
	{
		axpyv_ker_ft f = bli_cntx_get_ukr_dt( BLIS_FLOAT, BLIS_AXPYV_KER, cntx );

		for ( i = 0; i < b_n; ++i )
		{
			const float* restrict a1   = a + (0  )*inca + (i  )*lda;
			const float* restrict chi1 = x + (i  )*incx;
			      float* restrict y1   = y + (0  )*incy;
			      float           alpha_chi1;

			PASTEMAC(s,copycjs)( conjx, *chi1, alpha_chi1 );
			PASTEMAC(s,scals)( *alpha, alpha_chi1 );

			f
			(
			  conja,
			  m,
			  &alpha_chi1,
			  a1, inca,
			  y1, incy,
			  cntx
			);
		}

		return;
	}

	// At this point, we know that b_n is exactly equal to the fusing factor.

	// Use the unrolling factor and the number of elements per register
	// to compute the number of vectorized and leftover iterations.
	m_viter = ( m ) / ( n_elem_per_reg * n_iter_unroll );
	m_left  = ( m ) % ( n_elem_per_reg * n_iter_unroll );

	// If there is anything that would interfere with our use of contiguous
	// vector loads/stores, override m_viter and m_left to use scalar code
	// for all iterations.
	if ( inca != 1 || incy != 1 )
	{
		m_viter = 0;
		m_left  = m;
	}

	const float* restrict ap0   = a + 0*lda;
	const float* restrict ap1   = a + 1*lda;
	const float* restrict ap2   = a + 2*lda;
	const float* restrict ap3   = a + 3*lda;
	const float* restrict ap4   = a + 4*lda;
	const float* restrict ap5   = a + 5*lda;
	const float* restrict ap6   = a + 6*lda;
	const float* restrict ap7   = a + 7*lda;
	      float* restrict yp0   = y;

	chi0 = *( x + 0*incx );
	chi1 = *( x + 1*incx );
	chi2 = *( x + 2*incx );
	chi3 = *( x + 3*incx );
	chi4 = *( x + 4*incx );
	chi5 = *( x + 5*incx );
	chi6 = *( x + 6*incx );
	chi7 = *( x + 7*incx );

	// Scale each chi scalar by alpha.
	PASTEMAC(s,scals)( *alpha, chi0 );
	PASTEMAC(s,scals)( *alpha, chi1 );
	PASTEMAC(s,scals)( *alpha, chi2 );
	PASTEMAC(s,scals)( *alpha, chi3 );
	PASTEMAC(s,scals)( *alpha, chi4 );
	PASTEMAC(s,scals)( *alpha, chi5 );
	PASTEMAC(s,scals)( *alpha, chi6 );
	PASTEMAC(s,scals)( *alpha, chi7 );

	// Broadcast the (alpha*chi?) scalars to all elements of vector registers.
	chi0v.v = _mm256_broadcast_ss( &chi0 );
	chi1v.v = _mm256_broadcast_ss( &chi1 );
	chi2v.v = _mm256_broadcast_ss( &chi2 );
	chi3v.v = _mm256_broadcast_ss( &chi3 );
	chi4v.v = _mm256_broadcast_ss( &chi4 );
	chi5v.v = _mm256_broadcast_ss( &chi5 );
	chi6v.v = _mm256_broadcast_ss( &chi6 );
	chi7v.v = _mm256_broadcast_ss( &chi7 );

	// If there are vectorized iterations, perform them with vector
	// instructions.
	for ( i = 0; i < m_viter; ++i )
	{
		// Load the input values.
		y0v.v = _mm256_loadu_ps( yp0 + 0*n_elem_per_reg );
		a0v.v = _mm256_loadu_ps( ap0 + 0*n_elem_per_reg );
		a1v.v = _mm256_loadu_ps( ap1 + 0*n_elem_per_reg );
		a2v.v = _mm256_loadu_ps( ap2 + 0*n_elem_per_reg );
		a3v.v = _mm256_loadu_ps( ap3 + 0*n_elem_per_reg );
		a4v.v = _mm256_loadu_ps( ap4 + 0*n_elem_per_reg );
		a5v.v = _mm256_loadu_ps( ap5 + 0*n_elem_per_reg );
		a6v.v = _mm256_loadu_ps( ap6 + 0*n_elem_per_reg );
		a7v.v = _mm256_loadu_ps( ap7 + 0*n_elem_per_reg );

		// perform : y += alpha * x;
		y0v.v = _mm256_fmadd_ps( a0v.v, chi0v.v, y0v.v );
		y0v.v = _mm256_fmadd_ps( a1v.v, chi1v.v, y0v.v );
		y0v.v = _mm256_fmadd_ps( a2v.v, chi2v.v, y0v.v );
		y0v.v = _mm256_fmadd_ps( a3v.v, chi3v.v, y0v.v );
		y0v.v = _mm256_fmadd_ps( a4v.v, chi4v.v, y0v.v );
		y0v.v = _mm256_fmadd_ps( a5v.v, chi5v.v, y0v.v );
		y0v.v = _mm256_fmadd_ps( a6v.v, chi6v.v, y0v.v );
		y0v.v = _mm256_fmadd_ps( a7v.v, chi7v.v, y0v.v );

		// Store the output.
		_mm256_storeu_ps( (yp0 + 0*n_elem_per_reg), y0v.v );

		yp0 += n_elem_per_reg;
		ap0 += n_elem_per_reg;
		ap1 += n_elem_per_reg;
		ap2 += n_elem_per_reg;
		ap3 += n_elem_per_reg;
		ap4 += n_elem_per_reg;
		ap5 += n_elem_per_reg;
		ap6 += n_elem_per_reg;
		ap7 += n_elem_per_reg;
	}

	// If there are leftover iterations, perform them with scalar code.
	for ( i = 0; i < m_left ; ++i )
	{
		float       y0c = *yp0;

		const float a0c = *ap0;
		const float a1c = *ap1;
		const float a2c = *ap2;
		const float a3c = *ap3;
		const float a4c = *ap4;
		const float a5c = *ap5;
		const float a6c = *ap6;
		const float a7c = *ap7;

		y0c += chi0 * a0c;
		y0c += chi1 * a1c;
		y0c += chi2 * a2c;
		y0c += chi3 * a3c;
		y0c += chi4 * a4c;
		y0c += chi5 * a5c;
		y0c += chi6 * a6c;
		y0c += chi7 * a7c;

		*yp0 = y0c;

		ap0 += inca;
		ap1 += inca;
		ap2 += inca;
		ap3 += inca;
		ap4 += inca;
		ap5 += inca;
		ap6 += inca;
		ap7 += inca;
		yp0 += incy;
	}
}

// -----------------------------------------------------------------------------

void bli_daxpyf_zen_int_8
     (
             conj_t  conja,
             conj_t  conjx,
             dim_t   m,
             dim_t   b_n,
       const void*   alpha0,
       const void*   a0, inc_t inca, inc_t lda,
       const void*   x0, inc_t incx,
             void*   y0, inc_t incy,
       const cntx_t* cntx
     )
{
	const double* restrict alpha = alpha0;
	const double* restrict a     = a0;
	const double* restrict x     = x0;
	      double* restrict y     = y0;

	const dim_t      fuse_fac       = 8;

	const dim_t      n_elem_per_reg = 4;
	const dim_t      n_iter_unroll  = 1;

	dim_t            i;
	dim_t            m_viter;
	dim_t            m_left;

	v4df_t           chi0v, chi1v, chi2v, chi3v;
	v4df_t           chi4v, chi5v, chi6v, chi7v;

	v4df_t           a0v, a1v, a2v, a3v;
	v4df_t           a4v, a5v, a6v, a7v;
	v4df_t           y0v;

	double           chi0, chi1, chi2, chi3;
	double           chi4, chi5, chi6, chi7;

	// If either dimension is zero, or if alpha is zero, return early.
	if ( bli_zero_dim2( m, b_n ) || PASTEMAC(d,eq0)( *alpha ) ) return;

	// If b_n is not equal to the fusing factor, then perform the entire
	// operation as a loop over axpyv.
	if ( b_n != fuse_fac )
	{
		axpyv_ker_ft f = bli_cntx_get_ukr_dt( BLIS_DOUBLE, BLIS_AXPYV_KER, cntx );

		for ( i = 0; i < b_n; ++i )
		{
			const double* restrict a1   = a + (0  )*inca + (i  )*lda;
			const double* restrict chi1 = x + (i  )*incx;
			      double* restrict y1   = y + (0  )*incy;
			      double           alpha_chi1;

			PASTEMAC(d,copycjs)( conjx, *chi1, alpha_chi1 );
			PASTEMAC(d,scals)( *alpha, alpha_chi1 );

			f
			(
			  conja,
			  m,
			  &alpha_chi1,
			  a1, inca,
			  y1, incy,
			  cntx
			);
		}

		return;
	}

	// At this point, we know that b_n is exactly equal to the fusing factor.

	// Use the unrolling factor and the number of elements per register
	// to compute the number of vectorized and leftover iterations.
	m_viter = ( m ) / ( n_elem_per_reg * n_iter_unroll );
	m_left  = ( m ) % ( n_elem_per_reg * n_iter_unroll );

	// If there is anything that would interfere with our use of contiguous
	// vector loads/stores, override m_viter and m_left to use scalar code
	// for all iterations.
	if ( inca != 1 || incy != 1 )
	{
		m_viter = 0;
		m_left  = m;
	}

	const double* restrict ap0   = a + 0*lda;
	const double* restrict ap1   = a + 1*lda;
	const double* restrict ap2   = a + 2*lda;
	const double* restrict ap3   = a + 3*lda;
	const double* restrict ap4   = a + 4*lda;
	const double* restrict ap5   = a + 5*lda;
	const double* restrict ap6   = a + 6*lda;
	const double* restrict ap7   = a + 7*lda;
	      double* restrict yp0   = y;

	chi0 = *( x + 0*incx );
	chi1 = *( x + 1*incx );
	chi2 = *( x + 2*incx );
	chi3 = *( x + 3*incx );
	chi4 = *( x + 4*incx );
	chi5 = *( x + 5*incx );
	chi6 = *( x + 6*incx );
	chi7 = *( x + 7*incx );

	// Scale each chi scalar by alpha.
	PASTEMAC(d,scals)( *alpha, chi0 );
	PASTEMAC(d,scals)( *alpha, chi1 );
	PASTEMAC(d,scals)( *alpha, chi2 );
	PASTEMAC(d,scals)( *alpha, chi3 );
	PASTEMAC(d,scals)( *alpha, chi4 );
	PASTEMAC(d,scals)( *alpha, chi5 );
	PASTEMAC(d,scals)( *alpha, chi6 );
	PASTEMAC(d,scals)( *alpha, chi7 );

	// Broadcast the (alpha*chi?) scalars to all elements of vector registers.
	chi0v.v = _mm256_broadcast_sd( &chi0 );
	chi1v.v = _mm256_broadcast_sd( &chi1 );
	chi2v.v = _mm256_broadcast_sd( &chi2 );
	chi3v.v = _mm256_broadcast_sd( &chi3 );
	chi4v.v = _mm256_broadcast_sd( &chi4 );
	chi5v.v = _mm256_broadcast_sd( &chi5 );
	chi6v.v = _mm256_broadcast_sd( &chi6 );
	chi7v.v = _mm256_broadcast_sd( &chi7 );

	// If there are vectorized iterations, perform them with vector
	// instructions.
	for ( i = 0; i < m_viter; ++i )
	{
		// Load the input values.
		y0v.v = _mm256_loadu_pd( yp0 + 0*n_elem_per_reg );
		a0v.v = _mm256_loadu_pd( ap0 + 0*n_elem_per_reg );
		a1v.v = _mm256_loadu_pd( ap1 + 0*n_elem_per_reg );
		a2v.v = _mm256_loadu_pd( ap2 + 0*n_elem_per_reg );
		a3v.v = _mm256_loadu_pd( ap3 + 0*n_elem_per_reg );
		a4v.v = _mm256_loadu_pd( ap4 + 0*n_elem_per_reg );
		a5v.v = _mm256_loadu_pd( ap5 + 0*n_elem_per_reg );
		a6v.v = _mm256_loadu_pd( ap6 + 0*n_elem_per_reg );
		a7v.v = _mm256_loadu_pd( ap7 + 0*n_elem_per_reg );

		// perform : y += alpha * x;
		y0v.v = _mm256_fmadd_pd( a0v.v, chi0v.v, y0v.v );
		y0v.v = _mm256_fmadd_pd( a1v.v, chi1v.v, y0v.v );
		y0v.v = _mm256_fmadd_pd( a2v.v, chi2v.v, y0v.v );
		y0v.v = _mm256_fmadd_pd( a3v.v, chi3v.v, y0v.v );
		y0v.v = _mm256_fmadd_pd( a4v.v, chi4v.v, y0v.v );
		y0v.v = _mm256_fmadd_pd( a5v.v, chi5v.v, y0v.v );
		y0v.v = _mm256_fmadd_pd( a6v.v, chi6v.v, y0v.v );
		y0v.v = _mm256_fmadd_pd( a7v.v, chi7v.v, y0v.v );

		// Store the output.
		_mm256_storeu_pd( (yp0 + 0*n_elem_per_reg), y0v.v );

		yp0 += n_elem_per_reg;
		ap0 += n_elem_per_reg;
		ap1 += n_elem_per_reg;
		ap2 += n_elem_per_reg;
		ap3 += n_elem_per_reg;
		ap4 += n_elem_per_reg;
		ap5 += n_elem_per_reg;
		ap6 += n_elem_per_reg;
		ap7 += n_elem_per_reg;
	}

	// If there are leftover iterations, perform them with scalar code.
	for ( i = 0; i < m_left ; ++i )
	{
		double       y0c = *yp0;

		const double a0c = *ap0;
		const double a1c = *ap1;
		const double a2c = *ap2;
		const double a3c = *ap3;
		const double a4c = *ap4;
		const double a5c = *ap5;
		const double a6c = *ap6;
		const double a7c = *ap7;

		y0c += chi0 * a0c;
		y0c += chi1 * a1c;
		y0c += chi2 * a2c;
		y0c += chi3 * a3c;
		y0c += chi4 * a4c;
		y0c += chi5 * a5c;
		y0c += chi6 * a6c;
		y0c += chi7 * a7c;

		*yp0 = y0c;

		ap0 += inca;
		ap1 += inca;
		ap2 += inca;
		ap3 += inca;
		ap4 += inca;
		ap5 += inca;
		ap6 += inca;
		ap7 += inca;
		yp0 += incy;
	}
}

