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
#define FUNCPTR_T axpyf_fp

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
static FUNCPTR_T GENARRAY3_ALL(ftypes,axpyf_opt_var1);
#else
#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
static FUNCPTR_T GENARRAY3_EXT(ftypes,axpyf_opt_var1);
#else
static FUNCPTR_T GENARRAY3_MIN(ftypes,axpyf_opt_var1);
#endif
#endif


void bli_axpyf_opt_var1( obj_t*  alpha,
                         obj_t*  x,
                         obj_t*  y )
{
	num_t     dt_x      = bli_obj_datatype( *x );
	num_t     dt_y      = bli_obj_datatype( *y );

	conj_t    conjx     = bli_obj_conj_status( *x );
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
	   n,
	   buf_alpha,
	   buf_x, inc_x,
	   buf_y, inc_y );
}
*/

#undef  GENTFUNC3U12
#define GENTFUNC3U12( ctype_a, ctype_x, ctype_y, ctype_ax, cha, chx, chy, chax, opname, varname ) \
\
void PASTEMAC3(cha,chx,chy,varname)( \
                                     conj_t conja, \
                                     conj_t conjx, \
                                     dim_t  m, \
                                     dim_t  b_n, \
                                     void*  alpha, \
                                     void*  a, inc_t inca, inc_t lda, \
                                     void*  x, inc_t incx, \
                                     void*  y, inc_t incy \
                                   ) \
{ \
	ctype_ax* alpha_cast = alpha; \
	ctype_a*  a_cast     = a; \
	ctype_x*  x_cast     = x; \
	ctype_y*  y_cast     = y; \
	ctype_a*  a1; \
	ctype_x*  chi1; \
	ctype_y*  y1; \
	ctype_ax  alpha_chi1; \
	dim_t     i; \
\
	for ( i = 0; i < b_n; ++i ) \
	{ \
		a1   = a_cast + (0  )*inca + (i  )*lda; \
		chi1 = x_cast + (i  )*incx; \
		y1   = y_cast + (0  )*incy; \
\
		PASTEMAC2(chx,chax,copycjs)( conjx, *chi1, alpha_chi1 ); \
		PASTEMAC2(chax,chax,scals)( *alpha_cast, alpha_chi1 ); \
\
		PASTEMAC3(chax,cha,chy,axpyv)( conja, \
		                               m, \
		                               &alpha_chi1, \
		                               a1, inca, \
		                               y1, incy ); \
	} \
}

// Define the basic set of functions unconditionally, and then also some
// mixed datatype functions if requested.
//INSERT_GENTFUNC3U12_BASIC( axpyf, axpyf_opt_var1 )
GENTFUNC3U12( float,    float,    float,    float,    s, s, s, s, axpyf, axpyf_opt_var1 )
//GENTFUNC3U12( double,   double,   double,   double,   d, d, d, d, axpyf, axpyf_opt_var1 )
GENTFUNC3U12( scomplex, scomplex, scomplex, scomplex, c, c, c, c, axpyf, axpyf_opt_var1 )
GENTFUNC3U12( dcomplex, dcomplex, dcomplex, dcomplex, z, z, z, z, axpyf, axpyf_opt_var1 )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTFUNC3U12_MIX_D( axpyf, axpyf_opt_var1 )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTFUNC3U12_MIX_P( axpyf, axpyf_opt_var1 )
#endif


void bli_dddaxpyf_opt_var1(
                            conj_t conja,
                            conj_t conjx,
                            dim_t  m,
                            dim_t  b_n,
                            void*  alpha,
                            void*  a, inc_t inca, inc_t lda,
                            void*  x, inc_t incx,
                            void*  y, inc_t incy
                          )
{
	double*  restrict alpha_cast = alpha;
	double*  restrict a_cast = a;
	double*  restrict x_cast = x;
	double*  restrict y_cast = y;
	
    if ( bli_zero_dim2( m, b_n ) ) return;

	bool_t            use_ref = FALSE;
//    printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\n", b_n, PASTEMAC(d, axpyf_fusefac), inca, incx, incy, bli_is_unaligned_to(a, 32), bli_is_unaligned_to( y, 32));
	// If there is anything that would interfere with our use of aligned
	// vector loads/stores, call the reference implementation.
	if ( b_n < PASTEMAC(d,axpyf_fusefac) || inca != 1 || incx != 1 || incy != 1 || bli_is_unaligned_to( a, 32 ) || bli_is_unaligned_to( y, 32 ) )
		use_ref = TRUE;
	// Call the reference implementation if needed.
	if ( use_ref == TRUE )
	{   
//        printf("%d\t%d\t%d\t%d\t%d\t%d\n", PASTEMAC(d, axpyf_fusefac), inca, incx, incy, bli_is_unaligned_to(a, 32), bli_is_unaligned_to( y, 32));
//        printf("DEFAULTING TO REFERENCE IMPLEMENTATION\n");
		PASTEMAC3(d,d,d,axpyf_unb_var1)( conja, conjx, m, b_n, alpha_cast, a_cast, inca, lda, x_cast, incx, y_cast, incy );
		return;
	}

	dim_t m_run       =  m / 4;
	dim_t m_left      =  m % 4;

	double * a0   = a_cast + 0*lda;
	double * a1   = a_cast + 1*lda;
	double * a2   = a_cast + 2*lda;
	double * a3   = a_cast + 3*lda;
	double * a4   = a_cast + 4*lda;
	double * a5   = a_cast + 5*lda;
	double * a6   = a_cast + 6*lda;
	double * a7   = a_cast + 7*lda;
	double * y0   = y_cast;

	double chi0 = *(x_cast + 0*incx);
	double chi1 = *(x_cast + 1*incx);
	double chi2 = *(x_cast + 2*incx);
	double chi3 = *(x_cast + 3*incx);
	double chi4 = *(x_cast + 4*incx);
	double chi5 = *(x_cast + 5*incx);
	double chi6 = *(x_cast + 6*incx);
	double chi7 = *(x_cast + 7*incx);

	PASTEMAC2(d,d,scals)( *alpha_cast, chi0 );
	PASTEMAC2(d,d,scals)( *alpha_cast, chi1 );
	PASTEMAC2(d,d,scals)( *alpha_cast, chi2 );
	PASTEMAC2(d,d,scals)( *alpha_cast, chi3 );
	PASTEMAC2(d,d,scals)( *alpha_cast, chi4 );
	PASTEMAC2(d,d,scals)( *alpha_cast, chi5 );
	PASTEMAC2(d,d,scals)( *alpha_cast, chi6 );
	PASTEMAC2(d,d,scals)( *alpha_cast, chi7 );

	vector4double   a0v, a1v, a2v, a3v, a4v, a5v, a6v, a7v;
    vector4double   yv;
	vector4double   chi0v, chi1v, chi2v, chi3v, chi4v, chi5v, chi6v, chi7v;
	chi0v = vec_splats( chi0 );
	chi1v = vec_splats( chi1 );
	chi2v = vec_splats( chi2 );
	chi3v = vec_splats( chi3 );
	chi4v = vec_splats( chi4 );
	chi5v = vec_splats( chi5 );
	chi6v = vec_splats( chi6 );
	chi7v = vec_splats( chi7 );

    for ( dim_t i = 0; i < m_run; i += 1 )
	{
		yv  = vec_lda( 0 * sizeof(double), &y0[i*4]);

		a0v = vec_lda( 0 * sizeof(double), &a0[i*4]);
		a1v = vec_lda( 0 * sizeof(double), &a1[i*4]);
		a2v = vec_lda( 0 * sizeof(double), &a2[i*4]);
		a3v = vec_lda( 0 * sizeof(double), &a3[i*4]);
		a4v = vec_lda( 0 * sizeof(double), &a4[i*4]);
		a5v = vec_lda( 0 * sizeof(double), &a5[i*4]);
		a6v = vec_lda( 0 * sizeof(double), &a6[i*4]);
		a7v = vec_lda( 0 * sizeof(double), &a7[i*4]);

        yv = vec_madd( chi0v, a0v, yv );
        yv = vec_madd( chi1v, a1v, yv );
        yv = vec_madd( chi2v, a2v, yv );
        yv = vec_madd( chi3v, a3v, yv );
        yv = vec_madd( chi4v, a4v, yv );
        yv = vec_madd( chi5v, a5v, yv );
        yv = vec_madd( chi6v, a6v, yv );
        yv = vec_madd( chi7v, a7v, yv );

        vec_sta( yv, 0 * sizeof(double), &y0[i*4]);
	}
    
    for ( dim_t i = 0; i < m_left; ++i )
    {
        y0[4*m_run + i] += chi0 * a0[4*m_run + i]
                      +  chi1 * a1[4*m_run + i]
                      +  chi2 * a2[4*m_run + i]
                      +  chi3 * a3[4*m_run + i]
                      +  chi4 * a4[4*m_run + i]
                      +  chi5 * a5[4*m_run + i]
                      +  chi6 * a6[4*m_run + i]
                      +  chi7 * a7[4*m_run + i];
    }
}

