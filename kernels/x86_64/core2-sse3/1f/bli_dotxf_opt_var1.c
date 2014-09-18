/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"


#include "pmmintrin.h"
typedef union
{
    __m128d v;
    double  d[2];
} v2df_t;


void bli_ddotxf_opt_var1(
                          conj_t             conjat,
                          conj_t             conjx,
                          dim_t              m,
                          dim_t              b_n,
                          double*   restrict alpha,
                          double*   restrict a, inc_t inca, inc_t lda,
                          double*   restrict x, inc_t incx,
                          double*   restrict beta,
                          double*   restrict y, inc_t incy
                        )
{ 
	double*  restrict alpha_cast = alpha; 
	double*  restrict beta_cast = beta; 
	double*  restrict a_cast = a; 
	double*  restrict x_cast = x; 
	double*  restrict y_cast = y; 
	dim_t             i; 

	const dim_t       n_elem_per_reg = 2;
	const dim_t       n_iter_unroll  = 4;

	dim_t             m_pre;
	dim_t             m_run;
	dim_t             m_left;

	double*  restrict x0;
	double*  restrict x1;
	double*  restrict x2;
	double*  restrict x3;
	double*  restrict y0;
	double            rho0, rho1, rho2, rho3;
	double            x0c, x1c, x2c, x3c, y0c;

	v2df_t            rho0v, rho1v, rho2v, rho3v;
	v2df_t            x0v, x1v, x2v, x3v, y0v, betav, alphav;

	bool_t            use_ref = FALSE;


	if ( bli_zero_dim1( b_n ) ) return;

	// If the vector lengths are zero, scale r by beta and return.
	if ( bli_zero_dim1( m ) ) 
	{ 
		PASTEMAC2(d,d,scalv)( BLIS_NO_CONJUGATE,
		                      b_n,
		                      beta_cast,
		                      y_cast, incy );
		return; 
	} 

    m_pre = 0;

    // If there is anything that would interfere with our use of aligned
    // vector loads/stores, call the reference implementation.
	if ( b_n < PASTEMAC(d,dotxf_fusefac) )
	{
		use_ref = TRUE;
	}
    else if ( inca != 1 || incx != 1 || incy != 1 ||
	          bli_is_unaligned_to( lda*sizeof(double), 16 ) )
    {
        use_ref = TRUE;
    }
	else if ( bli_is_unaligned_to( a, 16 ) ||
	          bli_is_unaligned_to( x, 16 ) ||
	          bli_is_unaligned_to( y, 16 ) )
	{
		use_ref = TRUE;

		if ( bli_is_unaligned_to( a, 16 ) &&
		     bli_is_unaligned_to( x, 16 ) &&
		     bli_is_aligned_to( y, 16 ) ) // Note: r is not affected by x and y being unaligned. 
		{
			use_ref = FALSE;
			m_pre   = 1;
		}
	}

	// Call the reference implementation if needed.
	if ( use_ref == TRUE )
	{
		BLIS_DDOTXF_KERNEL_REF( conjat,
		                        conjx,
		                        m,
		                        b_n,
		                        alpha_cast,
		                        a_cast, inca, lda,
		                        x_cast, incx,
		                        beta_cast,
		                        y_cast, incy );
		return;
	}


	m_run       = ( m - m_pre ) / ( n_elem_per_reg * n_iter_unroll );
	m_left      = ( m - m_pre ) % ( n_elem_per_reg * n_iter_unroll );

	x0 = a_cast;
	x1 = a_cast +   lda;
	x2 = a_cast + 2*lda;
	x3 = a_cast + 3*lda;
	y0 = x_cast;

	PASTEMAC(d,set0s)( rho0 ); 
	PASTEMAC(d,set0s)( rho1 ); 
	PASTEMAC(d,set0s)( rho2 ); 
	PASTEMAC(d,set0s)( rho3 ); 

	if ( m_pre == 1 )
	{
		x0c = *x0;
		x1c = *x1;
		x2c = *x2;
		x3c = *x3;
		y0c = *y0;

		rho0 += x0c * y0c;
		rho1 += x1c * y0c;
		rho2 += x2c * y0c;
		rho3 += x3c * y0c;

		x0 += inca;
		x1 += inca;
		x2 += inca;
		x3 += inca;
		y0 += incx;
	}

	rho0v.v = _mm_setzero_pd();
	rho1v.v = _mm_setzero_pd();
	rho2v.v = _mm_setzero_pd();
	rho3v.v = _mm_setzero_pd();

	for ( i = 0; i < m_run; ++i )
	{
		x0v.v = _mm_load_pd( ( double* )(x0 + 0*n_elem_per_reg) );
		x1v.v = _mm_load_pd( ( double* )(x1 + 0*n_elem_per_reg) );
		x2v.v = _mm_load_pd( ( double* )(x2 + 0*n_elem_per_reg) );
		x3v.v = _mm_load_pd( ( double* )(x3 + 0*n_elem_per_reg) );
		y0v.v = _mm_load_pd( ( double* )(y0 + 0*n_elem_per_reg) );

		rho0v.v += x0v.v * y0v.v;
		rho1v.v += x1v.v * y0v.v;
		rho2v.v += x2v.v * y0v.v;
		rho3v.v += x3v.v * y0v.v;

		x0v.v = _mm_load_pd( ( double* )(x0 + 1*n_elem_per_reg) );
		x1v.v = _mm_load_pd( ( double* )(x1 + 1*n_elem_per_reg) );
		x2v.v = _mm_load_pd( ( double* )(x2 + 1*n_elem_per_reg) );
		x3v.v = _mm_load_pd( ( double* )(x3 + 1*n_elem_per_reg) );
		y0v.v = _mm_load_pd( ( double* )(y0 + 1*n_elem_per_reg) );

		rho0v.v += x0v.v * y0v.v;
		rho1v.v += x1v.v * y0v.v;
		rho2v.v += x2v.v * y0v.v;
		rho3v.v += x3v.v * y0v.v;

		x0v.v = _mm_load_pd( ( double* )(x0 + 2*n_elem_per_reg) );
		x1v.v = _mm_load_pd( ( double* )(x1 + 2*n_elem_per_reg) );
		x2v.v = _mm_load_pd( ( double* )(x2 + 2*n_elem_per_reg) );
		x3v.v = _mm_load_pd( ( double* )(x3 + 2*n_elem_per_reg) );
		y0v.v = _mm_load_pd( ( double* )(y0 + 2*n_elem_per_reg) );

		rho0v.v += x0v.v * y0v.v;
		rho1v.v += x1v.v * y0v.v;
		rho2v.v += x2v.v * y0v.v;
		rho3v.v += x3v.v * y0v.v;

		x0v.v = _mm_load_pd( ( double* )(x0 + 3*n_elem_per_reg) );
		x1v.v = _mm_load_pd( ( double* )(x1 + 3*n_elem_per_reg) );
		x2v.v = _mm_load_pd( ( double* )(x2 + 3*n_elem_per_reg) );
		x3v.v = _mm_load_pd( ( double* )(x3 + 3*n_elem_per_reg) );
		y0v.v = _mm_load_pd( ( double* )(y0 + 3*n_elem_per_reg) );

		rho0v.v += x0v.v * y0v.v;
		rho1v.v += x1v.v * y0v.v;
		rho2v.v += x2v.v * y0v.v;
		rho3v.v += x3v.v * y0v.v;


		x0 += n_elem_per_reg * n_iter_unroll;
		x1 += n_elem_per_reg * n_iter_unroll;
		x2 += n_elem_per_reg * n_iter_unroll;
		x3 += n_elem_per_reg * n_iter_unroll;
		y0 += n_elem_per_reg * n_iter_unroll;
	}

	rho0 += rho0v.d[0] + rho0v.d[1];
	rho1 += rho1v.d[0] + rho1v.d[1];
	rho2 += rho2v.d[0] + rho2v.d[1];
	rho3 += rho3v.d[0] + rho3v.d[1];

	if ( m_left > 0 )
	{
		for ( i = 0; i < m_left; ++i )
		{
			x0c = *x0;
			x1c = *x1;
			x2c = *x2;
			x3c = *x3;
			y0c = *y0;

			rho0 += x0c * y0c;
			rho1 += x1c * y0c;
			rho2 += x2c * y0c;
			rho3 += x3c * y0c;

			x0 += inca;
			x1 += inca;
			x2 += inca;
			x3 += inca;
			y0 += incx;
		}
	}
/*
	PASTEMAC2(d,d,scals)( *beta_cast, *(y_cast  ) ); \
	PASTEMAC2(d,d,scals)( *beta_cast, *(y_cast+1) ); \
	PASTEMAC2(d,d,scals)( *beta_cast, *(y_cast+2) ); \
	PASTEMAC2(d,d,scals)( *beta_cast, *(y_cast+3) ); \

	PASTEMAC3(d,d,d,axpys)( *alpha_cast, rho1, *(y_cast  ) ); \
	PASTEMAC3(d,d,d,axpys)( *alpha_cast, rho2, *(y_cast+1) ); \
	PASTEMAC3(d,d,d,axpys)( *alpha_cast, rho3, *(y_cast+2) ); \
	PASTEMAC3(d,d,d,axpys)( *alpha_cast, rho4, *(y_cast+3) ); \
*/

	rho1v.d[0] = rho0;
	rho1v.d[1] = rho1;
	rho3v.d[0] = rho2;
	rho3v.d[1] = rho3;

	betav.v  = _mm_loaddup_pd( ( double* ) beta_cast );
	alphav.v = _mm_loaddup_pd( ( double* ) alpha_cast );

	rho0v.v = _mm_load_pd( ( double* )(y_cast + 0*n_elem_per_reg) );
	rho2v.v = _mm_load_pd( ( double* )(y_cast + 1*n_elem_per_reg) );

	rho0v.v *= betav.v;
	rho2v.v *= betav.v;

	rho0v.v += alphav.v * rho1v.v;
	rho2v.v += alphav.v * rho3v.v;

	_mm_store_pd( ( double* )(y_cast + 0*n_elem_per_reg), rho0v.v );
	_mm_store_pd( ( double* )(y_cast + 1*n_elem_per_reg), rho2v.v );

}

