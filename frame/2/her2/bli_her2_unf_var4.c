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
       struc_t struc, \
       uplo_t  uplo, \
       conj_t  conjx, \
       conj_t  conjy, \
       dim_t   m, \
       ctype*  alpha, \
       ctype*  x, inc_t incx, \
       ctype*  y, inc_t incy, \
       ctype*  c, inc_t rs_c, inc_t cs_c, \
       cntx_t* cntx  \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	ctype*  chi1; \
	ctype*  x2; \
	ctype*  psi1; \
	ctype*  y2; \
	ctype*  gamma11; \
	ctype*  c21; \
	ctype   alpha0; \
	ctype   alpha1; \
	ctype   alpha0_psi1; \
	ctype   alpha1_chi1; \
	ctype   conjx0_chi1; \
	ctype   conjx1_chi1; \
	ctype   conjy0_psi1; \
	ctype   conjy1_psi1; \
	dim_t   i; \
	dim_t   n_ahead; \
	inc_t   rs_ct, cs_ct; \
	conj_t  conjh, conjx0, conjy0, conjx1, conjy1; \
\
	rs_ct = rs_c; \
	cs_ct = cs_c; \
\
	conjh = ( bli_is_hermitian( struc ) || bli_is_skew_hermitian( struc ) ) ? BLIS_CONJUGATE : BLIS_NO_CONJUGATE; \
	conjx0 = conjx; \
	conjy1 = conjy; \
\
	/* The algorithm will be expressed in terms of the lower triangular case;
	   the upper triangular case is supported by swapping the row and column
	   strides of A and toggling some conj parameters. */ \
	if      ( bli_is_lower( uplo ) ) \
	{ \
		PASTEMAC(ch,copys)( *alpha, alpha0 ); \
		PASTEMAC(ch,copycjs)( conjh, *alpha, alpha1 ); \
\
		if ( bli_is_skew_symmetric( struc ) || bli_is_skew_hermitian( struc ) ) \
			PASTEMAC(ch,negs)( alpha1 ); \
	} \
	else /* if ( bli_is_upper( uplo ) ) */ \
	{ \
		bli_swap_incs( &rs_ct, &cs_ct ); \
\
		/* Toggle conjugation of conjx/conjy, but only if we are being invoked
		   as her2; for syr2, conjx/conjy are unchanged. */ \
		conjx0 = bli_apply_conj( conjh, conjx0 ); \
		conjy1 = bli_apply_conj( conjh, conjy1 ); \
\
		PASTEMAC(ch,copycjs)( conjh, *alpha, alpha0 ); \
		PASTEMAC(ch,copys)( *alpha, alpha1 ); \
\
		if ( bli_is_skew_symmetric( struc ) || bli_is_skew_hermitian( struc ) ) \
			PASTEMAC(ch,negs)( alpha0 ); \
	} \
\
	/* Apply conjh (which carries the conjugation component of the Hermitian
	   transpose, if applicable) to conjx and/or conjy as needed to arrive at
	   the effective conjugation for the vector subproblems. */ \
	conjx1 = bli_apply_conj( conjh, conjx0 ); \
	conjy0 = bli_apply_conj( conjh, conjy1 ); \
\
	/* Query the context for the kernel function pointer. */ \
	axpy2v_ker_ft kfp_2v = bli_cntx_get_ukr_dt( dt, BLIS_AXPY2V_KER, cntx ); \
\
	for ( i = 0; i < m; ++i ) \
	{ \
		n_ahead  = m - i - 1; \
		chi1     = x + (i  )*incx; \
		x2       = x + (i+1)*incx; \
		psi1     = y + (i  )*incy; \
		y2       = y + (i+1)*incy; \
		gamma11  = c + (i  )*rs_ct + (i  )*cs_ct; \
		c21      = c + (i+1)*rs_ct + (i  )*cs_ct; \
\
		/* Apply conjx and/or conjy to chi1 and/or psi1. */ \
		PASTEMAC(ch,copycjs)( conjx0, *chi1, conjx0_chi1 ); \
		PASTEMAC(ch,copycjs)( conjx1, *chi1, conjx1_chi1 ); \
		PASTEMAC(ch,copycjs)( conjy0, *psi1, conjy0_psi1 ); \
		PASTEMAC(ch,copycjs)( conjy1, *psi1, conjy1_psi1 ); \
\
		/* Compute scalars for vector subproblems. */ \
		PASTEMAC(ch,scal2s)( alpha0, conjy0_psi1, alpha0_psi1 ); \
		PASTEMAC(ch,scal2s)( alpha1, conjx1_chi1, alpha1_chi1 ); \
\
		/* c21 = c21 +      alpha  * x2 * conj(psi1); */ \
		/* c21 = c21 + conj(alpha) * y2 * conj(chi1); */ \
		kfp_2v \
		( \
		  conjx0, \
		  conjy1, \
		  n_ahead, \
		  &alpha0_psi1, \
		  &alpha1_chi1, \
		  x2,  incx, \
		  y2,  incy, \
		  c21, rs_ct, \
		  cntx  \
		); \
\
		/* gamma11 = gamma11 +        alpha  * chi1 * conj(psi1) \
		                     +/- conj(alpha) * psi1 * conj(chi1); */ \
		PASTEMAC(ch,axpys)( alpha0_psi1, conjx0_chi1, *gamma11 ); \
		PASTEMAC(ch,axpys)( alpha1_chi1, conjy1_psi1, *gamma11 ); \
\
		/* For her2, explicitly set the imaginary component of gamma11 to
           zero. */ \
		if ( bli_is_hermitian( struc ) ) \
			PASTEMAC(ch,seti0s)( *gamma11 ); \
		if ( bli_is_skew_hermitian( struc ) ) \
			PASTEMAC(ch,setr0s)( *gamma11 ); \
	} \
}

INSERT_GENTFUNC_BASIC( her2_unf_var4 )

