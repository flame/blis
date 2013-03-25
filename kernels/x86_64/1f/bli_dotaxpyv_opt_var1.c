/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2013, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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
   THEORY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"


#undef  GENTFUNC3U12
#define GENTFUNC3U12( ctype_x, ctype_y, ctype_z, ctype_xy, chx, chy, chz, chxy, opname, varname ) \
\
void PASTEMAC3(chx,chy,chz,varname)( \
                                     conj_t conjxt, \
                                     conj_t conjx, \
                                     conj_t conjy, \
                                     dim_t  n, \
                                     void*  alpha, \
                                     void*  x, inc_t incx, \
                                     void*  y, inc_t incy, \
                                     void*  rho, \
                                     void*  z, inc_t incz \
                                   ) \
{ \
	ctype_xy* one        = PASTEMAC(chxy,1); \
	ctype_xy* zero       = PASTEMAC(chxy,0); \
	ctype_xy* alpha_cast = alpha; \
	ctype_x*  x_cast     = x; \
	ctype_y*  y_cast     = y; \
	ctype_xy* rho_cast   = rho; \
	ctype_z*  z_cast     = z; \
\
	PASTEMAC3(chx,chy,chxy,dotxv)( conjxt, \
	                               conjy, \
	                               n, \
	                               one, \
	                               x_cast, incx, \
	                               y_cast, incy, \
	                               zero, \
	                               rho_cast ); \
	PASTEMAC3(chxy,chx,chz,axpyv)( conjx, \
	                               n, \
	                               alpha_cast, \
	                               x_cast, incx, \
	                               z_cast, incz ); \
}

// Define the basic set of functions unconditionally, and then also some
// mixed datatype functions if requested.
//INSERT_GENTFUNC3U12_BASIC( dotaxpyv, dotaxpyv_opt_var1 )
GENTFUNC3U12( float,    float,    float,    float,    s, s, s, s, dotaxpyv, dotaxpyv_opt_var1 )
//GENTFUNC3U12( double,   double,   double,   double,   d, d, d, d, dotaxpyv, dotaxpyv_opt_var1 )
GENTFUNC3U12( scomplex, scomplex, scomplex, scomplex, c, c, c, c, dotaxpyv, dotaxpyv_opt_var1 )
GENTFUNC3U12( dcomplex, dcomplex, dcomplex, dcomplex, z, z, z, z, dotaxpyv, dotaxpyv_opt_var1 )


#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTFUNC3U12_MIX_D( dotaxpyv, dotaxpyv_opt_var1 )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTFUNC3U12_MIX_P( dotaxpyv, dotaxpyv_opt_var1 )
#endif


#include "pmmintrin.h"
typedef union
{
	__m128d v;
	double  d[2];
} v2df_t;


void bli_ddddotaxpyv_opt_var1(
                               conj_t conjxt,
                               conj_t conjx,
                               conj_t conjy,
                               dim_t  n,
                               void*  alpha,
                               void*  x, inc_t incx,
                               void*  y, inc_t incy,
                               void*  rho,
                               void*  z, inc_t incz
                             )
{
	double*  restrict alpha_cast = alpha;
	double*  restrict x_cast     = x;
	double*  restrict y_cast     = y;
	double*  restrict rho_cast   = rho;
	double*  restrict z_cast     = z;

	dim_t             n_pre;
	dim_t             n_run;
	dim_t             n_left;

	double*  restrict chi1;
	double*  restrict psi1;
	double*  restrict zeta1;
	double            alpha1c, chi1c, psi1c, rho1c;
	dim_t             i;
	inc_t             stepx, stepy, stepz;

	v2df_t    alphav, rhov;
	v2df_t    x1v, y1v, z1v;

	if ( bli_zero_dim1( n ) )
	{
		PASTEMAC(d,set0)( *rho_cast );
		return;
	}

   n_pre = 0;
	if ( ( unsigned long ) x % 16 != 0 )
	{
		if ( ( unsigned long ) y % 16 == 0 ||
		     ( unsigned long ) z % 16 == 0 ) bli_abort();

		n_pre = 1;
	}

	n_run       = ( n - n_pre ) / ( 2 * 1 );
	n_left      = ( n - n_pre ) % ( 2 * 1 );

	stepx       = 2 * incx;
	stepy       = 2 * incy;
	stepz       = 2 * incz;

	PASTEMAC(d,set0)( rho1c );

	alpha1c = *alpha_cast;

	chi1  = x_cast;
	psi1  = y_cast;
	zeta1 = z_cast;

	if ( n_pre == 1 )
	{
		chi1c  = *chi1;
		psi1c  = *psi1;

		rho1c  += chi1c * psi1c;
		*zeta1 += alpha1c * chi1c;

		chi1  += incx;
		psi1  += incy;
		zeta1 += incz;
	}

	rhov.v = _mm_setzero_pd();

	alphav.v = _mm_loaddup_pd( ( double* )alpha_cast );

	for ( i = 0; i < n_run; ++i )
	{
		x1v.v = _mm_load_pd( ( double* )chi1 );
		y1v.v = _mm_load_pd( ( double* )psi1 );
		z1v.v = _mm_load_pd( ( double* )zeta1 );
		//y1v.v = _mm_setr_pd( *psi1,  *(psi1  + incy) );
		//z1v.v = _mm_setr_pd( *zeta1, *(zeta1 + incz) );

		rhov.v += x1v.v * y1v.v;
		z1v.v  += alphav.v * x1v.v;

		_mm_store_pd( ( double* )zeta1, z1v.v );

		//chi1  += stepx;
		//psi1  += stepy;
		//zeta1 += stepz;
		chi1  += 2;
		psi1  += 2;
		zeta1 += 2;
	}

	if ( n_left > 0 )
	{
		for ( i = 0; i < n_left; ++i )
		{
			chi1c  = *chi1;
			psi1c  = *psi1;

			rho1c += chi1c * psi1c;
			*zeta1 += alpha1c * chi1c;

			chi1  += incx;
			psi1  += incy;
			zeta1 += incz;
		}
	}

	rho1c += rhov.d[0] + rhov.d[1];

	*rho_cast = rho1c;
}

