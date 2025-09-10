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
       trans_t transa, \
       conj_t  conjx, \
       dim_t   m, \
       dim_t   n, \
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
	ctype*  zero       = PASTEMAC(ch,0); \
	ctype*  a1; \
	ctype*  chi1; \
	ctype*  y1; \
	ctype   alpha_chi1; \
	dim_t   i; \
	dim_t   n_elem, n_iter; \
	inc_t   rs_at, cs_at; \
	conj_t  conja; \
\
	bli_set_dims_incs_with_trans( transa, \
	                              m, n, rs_a, cs_a, \
	                              &n_elem, &n_iter, &rs_at, &cs_at ); \
\
	conja = bli_extract_conj( transa ); \
\
	/* If beta is zero, use setv. Otherwise, scale by beta. */ \
	if ( PASTEMAC(ch,eq0)( *beta ) ) \
	{ \
		/* y = 0; */ \
		PASTEMAC2(ch,setv,BLIS_TAPI_EX_SUF) \
		( \
		  BLIS_NO_CONJUGATE, \
		  n_elem, \
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
		  n_elem, \
		  beta, \
		  y, incy, \
		  cntx, \
		  NULL  \
		); \
	} \
\
	PASTECH(ch,axpyv_ker_ft) kfp_av; \
\
	/* Query the context for the kernel function pointer. */ \
	kfp_av = bli_cntx_get_l1v_ker_dt( dt, BLIS_AXPYV_KER, cntx ); \
\
	for ( i = 0; i < n_iter; ++i ) \
	{ \
		a1   = a + (0  )*rs_at + (i  )*cs_at; \
		chi1 = x + (i  )*incx; \
		y1   = y + (0  )*incy; \
\
		/* y = y + alpha * chi1 * a1; */ \
		PASTEMAC(ch,copycjs)( conjx, *chi1, alpha_chi1 ); \
		PASTEMAC(ch,scals)( *alpha, alpha_chi1 ); \
\
		kfp_av \
		( \
		  conja, \
		  n_elem, \
		  &alpha_chi1, \
		  a1, rs_at, \
		  y1, incy, \
		  cntx  \
		); \
	} \
}

INSERT_GENTFUNC_BASIC0( gemv_unb_var2 )

