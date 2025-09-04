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
       uplo_t  uplo, \
       conj_t  conja, \
       conj_t  conjx, \
       conj_t  conjh, \
       dim_t   m, \
       ctype*  alpha, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       ctype*  x, inc_t incx, \
       ctype*  beta, \
       ctype*  y, inc_t incy, \
       cntx_t* cntx  \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	ctype*  one        = PASTEMAC(ch,1); \
	ctype*  zero       = PASTEMAC(ch,0); \
	ctype*  a10t; \
	ctype*  alpha11; \
	ctype*  x0; \
	ctype*  chi1; \
	ctype*  y0; \
	ctype*  psi1; \
	ctype   conjx_chi1; \
	ctype   alpha_chi1; \
	ctype   alpha11_temp; \
	dim_t   i; \
	dim_t   n_behind; \
	inc_t   rs_at, cs_at; \
	conj_t  conj0, conj1; \
\
	/* The algorithm will be expressed in terms of the lower triangular case;
	   the upper triangular case is supported by swapping the row and column
	   strides of A and toggling some conj parameters. */ \
	if      ( bli_is_lower( uplo ) ) \
	{ \
		rs_at = rs_a; \
		cs_at = cs_a; \
\
		conj0 = bli_apply_conj( conjh, conja ); \
		conj1 = conja; \
	} \
	else /* if ( bli_is_upper( uplo ) ) */ \
	{ \
		rs_at = cs_a; \
		cs_at = rs_a; \
\
		conj0 = conja; \
		conj1 = bli_apply_conj( conjh, conja ); \
	} \
\
	/* If beta is zero, use setv. Otherwise, scale by beta. */ \
	if ( PASTEMAC(ch,eq0)( *beta ) ) \
	{ \
		/* y = 0; */ \
		PASTEMAC2(ch,setv,BLIS_TAPI_EX_SUF) \
		( \
		  BLIS_NO_CONJUGATE, \
		  m, \
		  zero, \
		  y, incy, \
		  cntx, \
		  NULL  \
		); \
	} \
	else \
	{ \
		/* y = beta * y; */ \
		PASTEMAC2(ch,scalv,BLIS_TAPI_EX_SUF) \
		( \
		  BLIS_NO_CONJUGATE, \
		  m, \
		  beta, \
		  y, incy, \
		  cntx, \
		  NULL  \
		); \
	} \
\
	PASTECH(ch,axpyv_ker_ft) kfp_av; \
	PASTECH(ch,dotxv_ker_ft) kfp_dv; \
\
	/* Query the context for the kernel function pointers. */ \
	kfp_av = bli_cntx_get_l1v_ker_dt( dt, BLIS_AXPYV_KER, cntx ); \
	kfp_dv = bli_cntx_get_l1v_ker_dt( dt, BLIS_DOTXV_KER, cntx ); \
\
	for ( i = 0; i < m; ++i ) \
	{ \
		n_behind = i; \
		a10t     = a + (i  )*rs_at + (0  )*cs_at; \
		alpha11  = a + (i  )*rs_at + (i  )*cs_at; \
		x0       = x + (0  )*incx; \
		chi1     = x + (i  )*incx; \
		y0       = y + (0  )*incy; \
		psi1     = y + (i  )*incy; \
\
		/* Apply conjx to chi1 and and scale by alpha. */ \
		PASTEMAC(ch,copycjs)( conjx, *chi1, conjx_chi1 ); \
		PASTEMAC(ch,scal2s)( *alpha, conjx_chi1, alpha_chi1 ); \
\
		/* y0 = y0 + alpha * a10t' * chi1; */ \
		kfp_av \
		( \
		  conj0, \
		  n_behind, \
		  &alpha_chi1, \
		  a10t, cs_at, \
		  y0,   incy, \
		  cntx  \
		); \
\
		/* psi1 = psi1 + alpha * a10t * x0; */ \
		kfp_dv \
		( \
		  conj1, \
		  conjx, \
		  n_behind, \
		  alpha, \
		  a10t, cs_at, \
		  x0,   incx, \
		  one, \
		  psi1, \
		  cntx  \
		); \
\
		/* For hemv, explicitly set the imaginary component of alpha11 to
		   zero. */ \
		PASTEMAC(ch,copycjs)( conja, *alpha11, alpha11_temp ); \
		if ( bli_is_conj( conjh ) ) \
			PASTEMAC(ch,seti0s)( alpha11_temp ); \
\
		/* psi1 = psi1 + alpha * alpha11 * chi1; */ \
		PASTEMAC(ch,axpys)( alpha_chi1, alpha11_temp, *psi1 ); \
\
	} \
}

INSERT_GENTFUNC_BASIC0( hemv_unb_var1 )

