/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, The University of Texas at Austin

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

//
// Define BLAS-like interfaces with typed operands.
//

#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, opname ) \
\
err_t PASTEMAC(ch,opname) \
     ( \
             double   thresh, \
             uplo_t   uploa, \
             dim_t    m, \
       const ctype*   a, inc_t rs_a, inc_t cs_a, \
             ctype*   p, inc_t rs_p, inc_t cs_p  \
     ) \
{ \
	return PASTEMAC2(ch,opname,BLIS_TAPI_EX_SUF) \
	( \
	  thresh, \
	  uploa, \
	  m, \
	  a, rs_a, cs_a, \
	  p, rs_p, cs_p, \
	  NULL, \
	  NULL  \
	); \
} \

INSERT_GENTFUNCR_BASIC0( hevpinv )


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, opname ) \
\
err_t PASTEMAC2(ch,opname,BLIS_TAPI_EX_SUF) \
     ( \
             double   thresh, \
             uplo_t   uploa, \
             dim_t    m, \
       const ctype*   a, inc_t rs_a, inc_t cs_a, \
             ctype*   p, inc_t rs_p, inc_t cs_p, \
       const cntx_t*  cntx, \
       const rntm_t*  rntm  \
     ) \
{ \
	err_t r_val; \
\
	bli_init_once(); \
\
	/* Initialize some strides to use for a temporary matrix V and temporary
	   eigenvalue vector e. */ \
	inc_t rs_v = 0; /* rs = cs = 0 requests default (column-major) storage. */ \
	inc_t cs_v = 0; \
	inc_t is_v = 1; /* Not used but required for bli_adjust_strides() API. */ \
	inc_t ince = 1; \
\
	/* Interpret the strides above and perform leading dimension alignment. */ \
	bli_adjust_strides( m, m, sizeof( ctype ), &rs_v, &cs_v, &is_v ); \
\
	/* Make sure bli_adjust_strides() gave us strides for column storage. */ \
	if ( !bli_is_col_stored( rs_v, cs_v ) ) bli_abort(); \
\
	/* Allocate memory for a complex matrix V and a real vector e. */ \
	ctype*   v = bli_malloc_intl( cs_v * m * sizeof( ctype   ), &r_val ); \
	if ( r_val != BLIS_SUCCESS ) bli_abort(); \
\
	ctype_r* e = bli_malloc_intl( ince * m * sizeof( ctype_r ), &r_val ); \
	if ( r_val != BLIS_SUCCESS ) bli_abort(); \
\
	/* Perform a Hermitian EVD on A, storing eigenvectors to V and eigenvalues
	   to e. */ \
	r_val = PASTEMAC2(ch,hevd,BLIS_TAPI_EX_SUF) \
	( \
	  TRUE,  /* Always compute eigenvectors. */ \
	  uploa, \
	  m, \
	  a, rs_a, cs_a, \
	  v, rs_v, cs_v, \
	  e, ince, \
	  NULL,  /* work:  Request optimal size allocation for work array. */ \
	  0,     /* lwork: ignored if work == NULL. */ \
	  NULL,  /* rwork: Request optimal size allocation for rwork array. */ \
	  cntx, \
	  rntm  \
	); \
	if ( r_val != BLIS_SUCCESS ) bli_abort(); \
\
	/* Invert each eigenvalue, unless it falls below thresh, in which case it
	   gets set to zero. */ \
	PASTEMAC(chr,inverttv) \
	( \
	  thresh, \
	  m, \
	  e, ince  \
	); \
\
	/* Perform a reverse Hermitian EVD with the inverted eigenvalues to
	   compute the pseudo-inverse, storing the result to P. */ \
	r_val = PASTEMAC2(ch,rhevd,BLIS_TAPI_EX_SUF) \
	( \
	  uploa, \
	  m, \
	  v, rs_v, cs_v, \
	  e, ince, \
	  p, rs_p, cs_p, \
	  cntx, \
	  rntm  \
	); \
	if ( r_val != BLIS_SUCCESS ) bli_abort(); \
\
	/* Make P explicitly/densely Hermitian. */ \
	PASTEMAC2(ch,mkherm,BLIS_TAPI_EX_SUF) \
	( \
	  uploa, \
	  m, \
	  p, rs_p, cs_p, \
	  cntx, \
	  rntm  \
	); \
\
	/* Free the temporary matrix V and vector e. */ \
	bli_free_intl( v ); \
	bli_free_intl( e ); \
\
	return BLIS_SUCCESS; \
}

INSERT_GENTFUNCR_BASIC0( hevpinv )

