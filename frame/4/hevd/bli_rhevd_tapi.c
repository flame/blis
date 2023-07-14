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
             uplo_t   uploa, \
             dim_t    m, \
       const ctype*   v, inc_t rs_v, inc_t cs_v, \
       const ctype_r* e, inc_t ince, \
             ctype*   a, inc_t rs_a, inc_t cs_a  \
     ) \
{ \
	return PASTEMAC2(ch,opname,BLIS_TAPI_EX_SUF) \
	( \
	  uploa, \
	  m, \
	  v,  rs_v,  cs_v, \
	  e,  ince, \
	  a,  rs_a,  cs_a, \
	  NULL, \
	  NULL  \
	); \
}

INSERT_GENTFUNCR_BASIC0( rhevd )


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, opname ) \
\
err_t PASTEMAC2(ch,opname,BLIS_TAPI_EX_SUF) \
     ( \
             uplo_t   uploa, \
             dim_t    m, \
       const ctype*   v, inc_t rs_v, inc_t cs_v, \
       const ctype_r* e, inc_t ince, \
             ctype*   a, inc_t rs_a, inc_t cs_a, \
       const cntx_t*  cntx, \
       const rntm_t*  rntm  \
     ) \
{ \
	err_t r_val; \
\
	bli_init_once(); \
\
	/* Acquire a num_t for the datatype so we can branch on it later. */ \
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Initialize some strides to use for a temporary "Ve" matrix. */ \
	inc_t rs_ve = 0; /* rs = cs = 0 requests default (column-major) storage. */ \
	inc_t cs_ve = 0; \
	inc_t is_ve = 1; /* Not used but required for bli_adjust_strides() API. */ \
\
	/* Interpret the strides above and perform leading dimension alignment. */ \
	bli_adjust_strides( m, m, sizeof( ctype ), &rs_ve, &cs_ve, &is_ve ); \
\
	/* Make sure bli_adjust_strides() gave us strides for column storage.
	   Column storage is assumed in:
	   - the allocation of Ve;
	   - the scaling of the columns of Ve (which may be complex) by the
	     corresponding elements of e (which are always real). */ \
	if ( !bli_is_col_stored( rs_ve, cs_ve ) ) bli_abort(); \
\
	/* Allocate memory for the Ve matrix. */ \
	ctype* ve = bli_malloc_intl( cs_ve * m * sizeof( ctype ), &r_val ); \
\
	/* Make sure nothing went wrong. */ \
	if ( r_val != BLIS_SUCCESS ) bli_abort(); \
\
	/* Copy V to Ve. */ \
	PASTEMAC2(ch,copym,BLIS_TAPI_EX_SUF) \
	( \
	  0, \
	  BLIS_NONUNIT_DIAG, \
	  BLIS_DENSE, \
	  BLIS_NO_TRANSPOSE, \
	  m, m, \
	  v,  rs_v,  cs_v, \
	  ve, rs_ve, cs_ve, \
	  cntx, \
	  rntm  \
	); \
\
	/* Scale the columns of Ve (with contains only V so far) by the
	   corresponding elements of e. Notice that we leverage the column storage
	   of Ve to use real-domain scalv to scale both real and imaginary elements
	   when Ve is complex (by pretending each column is twice as long as it
	   actually is), which avoids extraneous flops that would otherwise be
	   incurred by complex-domain scalv when computing with the implicit zero
	   imaginary parts of the elements of e. */ \
	for ( dim_t j = 0; j < m; ++j ) \
	{ \
		ctype_r* vej_r = ( ctype_r* )( ve + j*cs_ve ); \
		ctype_r* ej    =               e  + j*ince; \
		dim_t    scl   = sizeof( ctype )/sizeof( ctype_r ); \
\
		PASTEMAC2(chr,scalv,BLIS_TAPI_EX_SUF) \
		( \
		  BLIS_NO_CONJUGATE, \
		  scl * m, \
		  ej, \
		  vej_r, 1, \
		  cntx, \
		  rntm  \
		); \
	} \
\
	ctype* one  = PASTEMAC(ch,1); \
	ctype* zero = PASTEMAC(ch,0); \
\
	/* Use gemmt to compute A = Ve * V^H, storing only the triangle specified
	   by uploa. */ \
	PASTEMAC2(ch,gemmt,BLIS_TAPI_EX_SUF) \
	( \
	  uploa, \
	  BLIS_NO_TRANSPOSE, \
	  BLIS_CONJ_TRANSPOSE, \
	  m, \
	  m, \
	  one, \
	  ve, rs_ve, cs_ve, \
	  v,  rs_v,  cs_v, \
	  zero, \
	  a,  rs_a,  cs_a, \
	  cntx, \
	  rntm  \
	); \
\
	ctype_r* zero_r = PASTEMAC(chr,0); \
\
	/* For complex domain computation, set the imaginary part of A to zero. */ \
	PASTEMAC2(ch,setid,BLIS_TAPI_EX_SUF) \
	( \
	  0, \
	  m, m, \
	  zero_r, \
	  a, rs_a, cs_a, \
	  cntx, \
	  rntm  \
	); \
\
	/* NOTE: Matrix A now contains V * Ie * V^H, but *only* in the triangle
	   specified by uploa. The caller may wish to make the matrix explicitly
	   Hermitian by calling bli_?mkherm() (or its object API analogue). */ \
\
	/* Free the secondary V matrix. */ \
	bli_free_intl( ve ); \
\
	return BLIS_SUCCESS; \
}

INSERT_GENTFUNCR_BASIC0( rhevd )

