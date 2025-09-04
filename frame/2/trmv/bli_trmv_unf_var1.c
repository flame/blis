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

#include "blis.h"

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       uplo_t  uploa, \
       trans_t transa, \
       diag_t  diaga, \
       dim_t   m, \
       ctype*  alpha, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       ctype*  x, inc_t incx, \
       cntx_t* cntx  \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	ctype*  one        = PASTEMAC(ch,1); \
	ctype*  A10; \
	ctype*  A11; \
	ctype*  A12; \
	ctype*  a10t; \
	ctype*  alpha11; \
	ctype*  a12t; \
	ctype*  x0; \
	ctype*  x1; \
	ctype*  x2; \
	ctype*  x01; \
	ctype*  chi11; \
	ctype*  x21; \
	ctype   alpha_alpha11_conj; \
	ctype   rho1; \
	dim_t   iter, i, k, j, l; \
	dim_t   b_fuse, f; \
	dim_t   n_ahead, f_ahead; \
	inc_t   rs_at, cs_at; \
	uplo_t  uploa_trans; \
	conj_t  conja; \
\
	if      ( bli_does_notrans( transa ) ) \
	{ \
		rs_at = rs_a; \
		cs_at = cs_a; \
		uploa_trans = uploa; \
	} \
	else /* if ( bli_does_trans( transa ) ) */ \
	{ \
		rs_at = cs_a; \
		cs_at = rs_a; \
		uploa_trans = bli_uplo_toggled( uploa ); \
	} \
\
	conja = bli_extract_conj( transa ); \
\
	PASTECH(ch,dotxf_ker_ft) kfp_df; \
\
	/* Query the context for the kernel function pointer and fusing factor. */ \
	kfp_df = bli_cntx_get_l1f_ker_dt( dt, BLIS_DOTXF_KER, cntx ); \
	b_fuse = bli_cntx_get_blksz_def_dt( dt, BLIS_DF, cntx ); \
\
	/* We reduce all of the possible cases down to just lower/upper. */ \
	if      ( bli_is_upper( uploa_trans ) ) \
	{ \
		for ( iter = 0; iter < m; iter += f ) \
		{ \
			f        = bli_determine_blocksize_dim_f( iter, m, b_fuse ); \
			i        = iter; \
			n_ahead  = m - iter - f; \
			A11      = a + (i  )*rs_at + (i  )*cs_at; \
			A12      = a + (i  )*rs_at + (i+f)*cs_at; \
			x1       = x + (i  )*incx; \
			x2       = x + (i+f)*incx; \
\
			/* x1 = alpha * A11 * x1; */ \
			for ( k = 0; k < f; ++k ) \
			{ \
				l        = k; \
				f_ahead  = f - l - 1; \
				alpha11  = A11 + (l  )*rs_at + (l  )*cs_at; \
				a12t     = A11 + (l  )*rs_at + (l+1)*cs_at; \
				chi11    = x1  + (l  )*incx; \
				x21      = x1  + (l+1)*incx; \
\
				/* chi11 = alpha * alpha11 * chi11; */ \
				PASTEMAC(ch,copys)( *alpha, alpha_alpha11_conj ); \
				if ( bli_is_nonunit_diag( diaga ) ) \
					PASTEMAC(ch,scalcjs)( conja, *alpha11, alpha_alpha11_conj ); \
				PASTEMAC(ch,scals)( alpha_alpha11_conj, *chi11 ); \
\
				/* chi11 = chi11 + alpha * a12t * x21; */ \
				PASTEMAC(ch,set0s)( rho1 ); \
				if ( bli_is_conj( conja ) ) \
				{ \
					for ( j = 0; j < f_ahead; ++j ) \
						PASTEMAC(ch,dotjs)( *(a12t + j*cs_at), *(x21 + j*incx), rho1 ); \
				} \
				else \
				{ \
					for ( j = 0; j < f_ahead; ++j ) \
						PASTEMAC(ch,dots)( *(a12t + j*cs_at), *(x21 + j*incx), rho1 ); \
				} \
				PASTEMAC(ch,axpys)( *alpha, rho1, *chi11 ); \
			} \
\
			/* x1 = x1 + alpha * A12 * x2; */ \
			kfp_df \
			( \
			  conja, \
			  BLIS_NO_CONJUGATE, \
			  n_ahead, \
			  f, \
			  alpha, \
			  A12, cs_at, rs_at, \
			  x2,  incx, \
			  one, \
			  x1,  incx, \
			  cntx  \
			); \
		} \
	} \
	else /* if ( bli_is_lower( uploa_trans ) ) */ \
	{ \
		for ( iter = 0; iter < m; iter += f ) \
		{ \
			f        = bli_determine_blocksize_dim_b( iter, m, b_fuse ); \
			i        = m - iter - f; \
			n_ahead  = i; \
			A11      = a + (i  )*rs_at + (i  )*cs_at; \
			A10      = a + (i  )*rs_at + (0  )*cs_at; \
			x1       = x + (i  )*incx; \
			x0       = x + (0  )*incx; \
\
			/* x1 = alpha * A11 * x1; */ \
			for ( k = 0; k < f; ++k ) \
			{ \
				l        = f - k - 1; \
				f_ahead  = l; \
				alpha11  = A11 + (l  )*rs_at + (l  )*cs_at; \
				a10t     = A11 + (l  )*rs_at + (0  )*cs_at; \
				chi11    = x1  + (l  )*incx; \
				x01      = x1  + (0  )*incx; \
\
				/* chi11 = alpha * alpha11 * chi11; */ \
				PASTEMAC(ch,copys)( *alpha, alpha_alpha11_conj ); \
				if ( bli_is_nonunit_diag( diaga ) ) \
					PASTEMAC(ch,scalcjs)( conja, *alpha11, alpha_alpha11_conj ); \
				PASTEMAC(ch,scals)( alpha_alpha11_conj, *chi11 ); \
\
				/* chi11 = chi11 + alpha * a10t * x01; */ \
				PASTEMAC(ch,set0s)( rho1 ); \
				if ( bli_is_conj( conja ) ) \
				{ \
					for ( j = 0; j < f_ahead; ++j ) \
						PASTEMAC(ch,dotjs)( *(a10t + j*cs_at), *(x01 + j*incx), rho1 ); \
				} \
				else \
				{ \
					for ( j = 0; j < f_ahead; ++j ) \
						PASTEMAC(ch,dots)( *(a10t + j*cs_at), *(x01 + j*incx), rho1 ); \
				} \
				PASTEMAC(ch,axpys)( *alpha, rho1, *chi11 ); \
			} \
\
			/* x1 = x1 + alpha * A10 * x0; */ \
			kfp_df \
			( \
			  conja, \
			  BLIS_NO_CONJUGATE, \
			  n_ahead, \
			  f, \
			  alpha, \
			  A10, cs_at, rs_at, \
			  x0,  incx, \
			  one, \
			  x1,  incx, \
			  cntx  \
			); \
		} \
	} \
}

INSERT_GENTFUNC_BASIC0( trmv_unf_var1 )

