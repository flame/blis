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

/*
#define FUNCPTR_T axpy2v_fp

typedef void (*FUNCPTR_T)(
                           conj_t conjx,
                           conj_t conjy,
                           dim_t  n,
                           void*  alpha,
                           void*  x, inc_t incx,
                           void*  y, inc_t incy
                         );

// If some mixed datatype functions will not be compiled, we initialize
// the corresponding elements of the function array to NULL.
#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
static FUNCPTR_T GENARRAY3_ALL(ftypes,axpy2v_unb_var1);
#else
#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
static FUNCPTR_T GENARRAY3_EXT(ftypes,axpy2v_unb_var1);
#else
static FUNCPTR_T GENARRAY3_MIN(ftypes,axpy2v_unb_var1);
#endif
#endif


void bli_axpy2v_unb_var1( obj_t*  alpha,
                         obj_t*  x,
                         obj_t*  y )
{
	num_t     dt_x      = bli_obj_datatype( *x );
	num_t     dt_y      = bli_obj_datatype( *y );

	conj_t    conjx     = bli_obj_conj_status( *x );
	conj_t    conjy     = bli_obj_conj_status( *y );
	dim_t     n         = bli_obj_vector_dim( *x );

	inc_t     inc_x     = bli_obj_vector_inc( *x );
	void*     buf_x     = bli_obj_buffer_at_off( *x );

	inc_t     inc_y     = bli_obj_vector_inc( *y );
	void*     buf_y     = bli_obj_buffer_at_off( *y );

	num_t     dt_alpha;
	void*     buf_alpha;

	FUNCPTR_T f;

	// If alpha is a scalar constant, use dt_x to extract the address of the
	// corresponding constant value; otherwise, use the datatype encoded
	// within the alpha object and extract the buffer at the alpha offset.
	bli_set_scalar_dt_buffer( alpha, dt_x, dt_alpha, buf_alpha );

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_alpha][dt_x][dt_y];

	// Invoke the function.
	f( conjx,
	   conjy,
	   n,
	   buf_alpha,
	   buf_x, inc_x,
	   buf_y, inc_y );
}
*/


#undef  GENTFUNC3U12
#define GENTFUNC3U12( ctype_x, ctype_y, ctype_z, ctype_xy, chx, chy, chz, chxy, opname, varname ) \
\
void PASTEMAC3(chx,chy,chz,varname)( \
                                     conj_t conjx, \
                                     conj_t conjy, \
                                     dim_t  n, \
                                     void*  alpha1, \
                                     void*  alpha2, \
                                     void*  x, inc_t incx, \
                                     void*  y, inc_t incy, \
                                     void*  z,  inc_t incz \
                                   ) \
{ \
	ctype_xy* alpha1_cast = alpha1; \
	ctype_xy* alpha2_cast = alpha2; \
	ctype_x*  x_cast      = x; \
	ctype_y*  y_cast      = y; \
	ctype_z*  z_cast      = z; \
\
	PASTEMAC3(chxy,chx,chz,axpyv)( conjx, \
	                               n, \
	                               alpha1_cast, \
	                               x_cast, incx, \
	                               z_cast, incz ); \
	PASTEMAC3(chxy,chy,chz,axpyv)( conjy, \
	                               n, \
	                               alpha2_cast, \
	                               y_cast, incy, \
	                               z_cast, incz ); \
}

// Define the basic set of functions unconditionally, and then also some
// mixed datatype functions if requested.
//INSERT_GENTFUNC3_BASIC( axpy2v, axpy2v_opt_var1 )
GENTFUNC3U12( float,    float,    float,    float,    s, s, s, s, axpy2v, axpy2v_opt_var1 )
//GENTFUNC3U12( double,   double,   double,   double,   d, d, d, d, axpy2v, axpy2v_opt_var1 )
GENTFUNC3U12( scomplex, scomplex, scomplex, scomplex, c, c, c, c, axpy2v, axpy2v_opt_var1 )
GENTFUNC3U12( dcomplex, dcomplex, dcomplex, dcomplex, z, z, z, z, axpy2v, axpy2v_opt_var1 )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTFUNC3U12_MIX_D( axpy2v, axpy2v_opt_var1 )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTFUNC3U12_MIX_P( axpy2v, axpy2v_opt_var1 )
#endif


#include "pmmintrin.h"
typedef union
{
	__m128d v;
	double  d[2];
} v2df_t;


void bli_dddaxpy2v_opt_var1( 
                            conj_t conjx,
                            conj_t conjy,
                            dim_t  n,
                            void*  alpha,
                            void*  beta,
                            void*  x, inc_t incx,
                            void*  y, inc_t incy,
                            void*  z,  inc_t incz
                          )
{
	double*  restrict alpha_cast  = alpha;
	double*  restrict beta_cast   = beta;
	double*  restrict x_cast      = x;
	double*  restrict y_cast      = y;
	double*  restrict z_cast      = z;
	dim_t             i;

	const dim_t       n_elem_per_reg = 2;
	const dim_t       n_iter_unroll  = 4;

	dim_t             n_pre;
	dim_t             n_run;
	dim_t             n_left;

	double*  restrict x1;
	double*  restrict y1;
	double*  restrict z1;
	double            alphac, betac, x1c, y1c;

	v2df_t            alphav, betav;
	v2df_t            x1v, y1v, z1v;
	v2df_t            x2v, y2v, z2v;

	if ( bli_zero_dim1( n ) ) return;

	if ( incx != 1 ||
	     incy != 1 ||
		 incz != 1 ) bli_abort();

	n_pre = 0;
	if ( ( unsigned long ) x % 16 != 0 )
	{
		if ( ( unsigned long ) y % 16 == 0 ||
		     ( unsigned long ) z % 16 == 0 ) bli_abort();

		n_pre = 1;
	}

	n_run       = ( n - n_pre ) / ( n_elem_per_reg * n_iter_unroll );
	n_left      = ( n - n_pre ) % ( n_elem_per_reg * n_iter_unroll );

	alphac = *alpha_cast;
	betac  = *beta_cast;

	x1 = x_cast;
	y1 = y_cast;
	z1 = z_cast;

	if ( n_pre == 1 )
	{
		x1c = *x1;
		y1c = *y1;

		*z1 += alphac * x1c +
		       betac  * y1c;

		x1 += incx;
		y1 += incy;
		z1 += incz;
	}

	alphav.v = _mm_loaddup_pd( ( double* )alpha_cast );
	betav.v  = _mm_loaddup_pd( ( double* )beta_cast );

	for ( i = 0; i < n_run; ++i )
	{
/*
		z1v.v = _mm_load_pd( ( double* )z1 + 0*n_elem_per_reg );
		x1v.v = _mm_load_pd( ( double* )x1 + 0*n_elem_per_reg );
		y1v.v = _mm_load_pd( ( double* )y1 + 0*n_elem_per_reg );

		z1v.v += alphav.v * x1v.v;
		z1v.v += betav.v  * y1v.v;

		_mm_store_pd( ( double* )(z1 + 0*n_elem_per_reg ), z1v.v );

		z1v.v = _mm_load_pd( ( double* )z1 + 1*n_elem_per_reg );
		x1v.v = _mm_load_pd( ( double* )x1 + 1*n_elem_per_reg );
		y1v.v = _mm_load_pd( ( double* )y1 + 1*n_elem_per_reg );

		z1v.v += alphav.v * x1v.v;
		z1v.v += betav.v  * y1v.v;

		_mm_store_pd( ( double* )(z1 + 1*n_elem_per_reg ), z1v.v );
*/
/*
		z1v.v = _mm_load_pd( ( double* )z1 + 0*n_elem_per_reg );
		x1v.v = _mm_load_pd( ( double* )x1 + 0*n_elem_per_reg );
		y1v.v = _mm_load_pd( ( double* )y1 + 0*n_elem_per_reg );

		z2v.v = _mm_load_pd( ( double* )z1 + 1*n_elem_per_reg );
		x2v.v = _mm_load_pd( ( double* )x1 + 1*n_elem_per_reg );
		y2v.v = _mm_load_pd( ( double* )y1 + 1*n_elem_per_reg );

		z1v.v += alphav.v * x1v.v;
		z1v.v += betav.v  * y1v.v;

		_mm_store_pd( ( double* )(z1 + 0*n_elem_per_reg ), z1v.v );

		z2v.v += alphav.v * x2v.v;
		z2v.v += betav.v  * y2v.v;

		_mm_store_pd( ( double* )(z1 + 1*n_elem_per_reg ), z2v.v );
*/
		z1v.v = _mm_load_pd( ( double* )z1 + 0*n_elem_per_reg );
		x1v.v = _mm_load_pd( ( double* )x1 + 0*n_elem_per_reg );
		y1v.v = _mm_load_pd( ( double* )y1 + 0*n_elem_per_reg );

		z2v.v = _mm_load_pd( ( double* )z1 + 1*n_elem_per_reg );
		x2v.v = _mm_load_pd( ( double* )x1 + 1*n_elem_per_reg );
		y2v.v = _mm_load_pd( ( double* )y1 + 1*n_elem_per_reg );

		z1v.v += alphav.v * x1v.v;
		z1v.v += betav.v  * y1v.v;

		_mm_store_pd( ( double* )(z1 + 0*n_elem_per_reg ), z1v.v );

		z1v.v = _mm_load_pd( ( double* )z1 + 2*n_elem_per_reg );
		x1v.v = _mm_load_pd( ( double* )x1 + 2*n_elem_per_reg );
		y1v.v = _mm_load_pd( ( double* )y1 + 2*n_elem_per_reg );

		z2v.v += alphav.v * x2v.v;
		z2v.v += betav.v  * y2v.v;

		_mm_store_pd( ( double* )(z1 + 1*n_elem_per_reg ), z2v.v );

		z2v.v = _mm_load_pd( ( double* )z1 + 3*n_elem_per_reg );
		x2v.v = _mm_load_pd( ( double* )x1 + 3*n_elem_per_reg );
		y2v.v = _mm_load_pd( ( double* )y1 + 3*n_elem_per_reg );

		z1v.v += alphav.v * x1v.v;
		z1v.v += betav.v  * y1v.v;

		_mm_store_pd( ( double* )(z1 + 2*n_elem_per_reg ), z1v.v );

		z2v.v += alphav.v * x2v.v;
		z2v.v += betav.v  * y2v.v;

		_mm_store_pd( ( double* )(z1 + 3*n_elem_per_reg ), z2v.v );



		x1 += n_elem_per_reg * n_iter_unroll;
		y1 += n_elem_per_reg * n_iter_unroll;
		z1 += n_elem_per_reg * n_iter_unroll;
	}

	if ( n_left > 0 )
	{
		for ( i = 0; i < n_left; ++i )
		{
			x1c = *x1;
			y1c = *y1;

			*z1 += alphac * x1c +
			       betac  * y1c;

			x1 += incx;
			y1 += incy;
			z1 += incz;
		}
	}
}
