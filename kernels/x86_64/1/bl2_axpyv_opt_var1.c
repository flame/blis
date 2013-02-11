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

#include "blis2.h"

#define FUNCPTR_T axpyv_fp

typedef void (*FUNCPTR_T)(
                           conj_t conjx,
                           dim_t  n,
                           void*  alpha,
                           void*  x, inc_t incx,
                           void*  y, inc_t incy
                         );

// If some mixed datatype functions will not be compiled, we initialize
// the corresponding elements of the function array to NULL.
#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
static FUNCPTR_T GENARRAY3_ALL(ftypes,axpyv_opt_var1);
#else
#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
static FUNCPTR_T GENARRAY3_EXT(ftypes,axpyv_opt_var1);
#else
static FUNCPTR_T GENARRAY3_MIN(ftypes,axpyv_opt_var1);
#endif
#endif


void bl2_axpyv_opt_var1( obj_t*  alpha,
                         obj_t*  x,
                         obj_t*  y )
{
	num_t     dt_x      = bl2_obj_datatype( *x );
	num_t     dt_y      = bl2_obj_datatype( *y );

	conj_t    conjx     = bl2_obj_conj_status( *x );
	dim_t     n         = bl2_obj_vector_dim( *x );

	inc_t     inc_x     = bl2_obj_vector_inc( *x );
	void*     buf_x     = bl2_obj_buffer_at_off( *x );

	inc_t     inc_y     = bl2_obj_vector_inc( *y );
	void*     buf_y     = bl2_obj_buffer_at_off( *y );

	num_t     dt_alpha;
	void*     buf_alpha;

	FUNCPTR_T f;

	// If alpha is a scalar constant, use dt_x to extract the address of the
	// corresponding constant value; otherwise, use the datatype encoded
	// within the alpha object and extract the buffer at the alpha offset.
	bl2_set_scalar_dt_buffer( alpha, dt_x, dt_alpha, buf_alpha );

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_alpha][dt_x][dt_y];

	// Invoke the function.
	f( conjx,
	   n,
	   buf_alpha,
	   buf_x, inc_x,
	   buf_y, inc_y );
}


#undef  GENTFUNC3
#define GENTFUNC3( ctype_a, ctype_x, ctype_y, cha, chx, chy, opname, varname ) \
\
void PASTEMAC3(cha,chx,chy,varname)( \
                                     conj_t conjx, \
                                     dim_t  n, \
                                     void*  alpha, \
                                     void*  x, inc_t incx, \
                                     void*  y, inc_t incy \
                                   ) \
{ \
	ctype_a* alpha_cast = alpha; \
	ctype_x* x_cast     = x; \
	ctype_y* y_cast     = y; \
	ctype_x* chi1; \
	ctype_y* psi1; \
	dim_t    i; \
\
	if ( bl2_zero_dim1( n ) ) return; \
\
	chi1 = x_cast; \
	psi1 = y_cast; \
\
	if ( bl2_is_conj( conjx ) ) \
	{ \
		for ( i = 0; i < n; ++i ) \
		{ \
			PASTEMAC3(cha,chx,chy,axpyjs)( *alpha_cast, *chi1, *psi1 ); \
\
			chi1 += incx; \
			psi1 += incy; \
		} \
	} \
	else \
	{ \
		for ( i = 0; i < n; ++i ) \
		{ \
			PASTEMAC3(cha,chx,chy,axpys)( *alpha_cast, *chi1, *psi1 ); \
\
			chi1 += incx; \
			psi1 += incy; \
		} \
	} \
}

// Define the basic set of functions unconditionally, and then also some
// mixed datatype functions if requested.
//INSERT_GENTFUNC3_BASIC( axpyv, axpyv_opt_var1 )
GENTFUNC3( float,    float,    float,    s, s, s, axpyv, axpyv_opt_var1 )
//GENTFUNC3( double,   double,   double,   d, d, d, axpyv, axpyv_opt_var1 )
GENTFUNC3( scomplex, scomplex, scomplex, c, c, c, axpyv, axpyv_opt_var1 )
GENTFUNC3( dcomplex, dcomplex, dcomplex, z, z, z, axpyv, axpyv_opt_var1 )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTFUNC3_MIX_D( axpyv, axpyv_opt_var1 )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTFUNC3_MIX_P( axpyv, axpyv_opt_var1 )
#endif


#include "pmmintrin.h"
typedef union
{
	__m128d v;
	double  d[2];
} v2df_t;


void bl2_dddaxpyv_opt_var1( 
                            conj_t conjx,
                            dim_t  n,
                            void*  alpha,
                            void*  x, inc_t incx,
                            void*  y, inc_t incy
                          )
{
	double*  restrict alpha_cast = alpha;
	double*  restrict x_cast = x;
	double*  restrict y_cast = y;
	dim_t             i;

	const dim_t       n_elem_per_reg = 2;
	const dim_t       n_iter_unroll  = 4;

	dim_t             n_pre;
	dim_t             n_run;
	dim_t             n_left;

	double*  restrict x1;
	double*  restrict y1;
	double            alpha1c, x1c;

	v2df_t            alpha1v;
	v2df_t            x1v, x2v, x3v, x4v;
	v2df_t            y1v, y2v, y3v, y4v;

	if ( bl2_zero_dim1( n ) ) return;

	if ( incx != 1 || incy != 1 )
	{
		bl2_dddaxpyv_unb_var1( conjx,
		                       n,
		                       alpha,
		                       x, incx,
		                       y, incy );
		return;
	}

	n_pre = 0;
	if ( ( unsigned long ) x % 16 != 0 )
	{
		if ( ( unsigned long ) y % 16 == 0 ) bl2_abort();

		n_pre = 1;
	}

	n_run       = ( n - n_pre ) / ( n_elem_per_reg * n_iter_unroll );
	n_left      = ( n - n_pre ) % ( n_elem_per_reg * n_iter_unroll );

	alpha1c = *alpha_cast;

	x1 = x_cast;
	y1 = y_cast;

	if ( n_pre == 1 )
	{
		x1c = *x1;

		*y1 += alpha1c * x1c;

		x1 += incx;
		y1 += incy;
	}

	alpha1v.v = _mm_loaddup_pd( ( double* )&alpha1c );

	for ( i = 0; i < n_run; ++i )
	{
		y1v.v = _mm_load_pd( ( double* )y1 );
		x1v.v = _mm_load_pd( ( double* )x1 );

		y1v.v += alpha1v.v * x1v.v;

		_mm_store_pd( ( double* )(y1    ), y1v.v );

		y2v.v = _mm_load_pd( ( double* )(y1 + 2) );
		x2v.v = _mm_load_pd( ( double* )(x1 + 2) );

		y2v.v += alpha1v.v * x2v.v;

		_mm_store_pd( ( double* )(y1 + 2), y2v.v );

		y3v.v = _mm_load_pd( ( double* )(y1 + 4) );
		x3v.v = _mm_load_pd( ( double* )(x1 + 4) );

		y3v.v += alpha1v.v * x3v.v;

		_mm_store_pd( ( double* )(y1 + 4), y3v.v );

		y4v.v = _mm_load_pd( ( double* )(y1 + 6) );
		x4v.v = _mm_load_pd( ( double* )(x1 + 6) );

		y4v.v += alpha1v.v * x4v.v;

		_mm_store_pd( ( double* )(y1 + 6), y4v.v );


		x1 += n_elem_per_reg * n_iter_unroll;
		y1 += n_elem_per_reg * n_iter_unroll;
	}

	if ( n_left > 0 )
	{
		for ( i = 0; i < n_left; ++i )
		{
			x1c = *x1;

			*y1 += alpha1c * x1c;

			x1 += incx;
			y1 += incy;
		}
	}
}
