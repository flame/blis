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
             bool     comp_evecs, \
             uplo_t   uploa, \
             dim_t    m, \
       const ctype*   a, inc_t rs_a, inc_t cs_a, \
             ctype*   v, inc_t rs_v, inc_t cs_v, \
             ctype_r* e, inc_t ince, \
             ctype*   work, \
             dim_t    lwork, \
             ctype_r* rwork  \
     ) \
{ \
	return PASTEMAC2(ch,opname,BLIS_TAPI_EX_SUF) \
	( \
	  comp_evecs, \
	  uploa, \
	  m, \
	  a,  rs_a,  cs_a, \
	  v,  rs_v,  cs_v, \
	  e,  ince, \
	  work, \
	  lwork, \
	  rwork, \
	  NULL, \
	  NULL  \
	); \
}

INSERT_GENTFUNCR_BASIC0( hevd )


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, opname, laname ) \
\
err_t PASTEMAC2(ch,opname,BLIS_TAPI_EX_SUF) \
     ( \
             bool     comp_evecs, \
             uplo_t   uploa, \
             dim_t    m, \
       const ctype*   a, inc_t rs_a, inc_t cs_a, \
             ctype*   v, inc_t rs_v, inc_t cs_v, \
             ctype_r* e, inc_t ince, \
             ctype*   work, \
             dim_t    lwork, \
             ctype_r* rwork, \
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
	const bool e_unit     = ( ince == 1 ); \
	const bool work_null  = ( work  == NULL ); \
	const bool rwork_null = ( rwork == NULL ); \
\
	/* Copy A to V so that the original matrix is preserved. */ \
	PASTEMAC(ch,copym) \
	( \
	  0, \
	  BLIS_NONUNIT_DIAG, \
	  uploa, \
	  BLIS_NO_TRANSPOSE, \
	  m, m, \
	  a, rs_a, cs_a, \
	  v, rs_v, cs_v  \
	); \
\
	/* If matrix V is row-stored, induce a Hermitian transpose to get it into
	   column storage since that is what LAPACK code expects. This will
	   effectively transform a Hermitian matrix stored in the lower triangle to
	   one that is stored in the upper triangle (and vice versa). */ \
	if ( bli_is_row_stored( rs_v, cs_v ) ) \
	{ \
		PASTEMAC(ch,hevd_induce_conjtrans)( &uploa, m, v, &rs_v, &cs_v ); \
	} \
\
	ctype_r* eu = e; \
\
	/* If the caller passed in an eigenvalue array that has non-unit stride,
	   we'll need to allocate a temporary array since the underlying LAPACK
	   implementation requires unit stride for that operand. */ \
	if ( !e_unit ) \
	{ \
		eu = bli_malloc_intl( ( size_t )m * sizeof( ctype_r ), &r_val ); \
\
		/* Make sure nothing went wrong. */ \
		if ( r_val != BLIS_SUCCESS ) bli_abort(); \
	} \
\
	/* Declare the LAPACK parameters needed to call the underlying LAPACK
	   operation. */ \
	f77_char jobz_la  = ( comp_evecs            ? 'v' : 'n' ); \
	f77_char uplo_la  = ( bli_is_lower( uploa ) ? 'l' : 'u' ); \
	f77_int  n_la     = ( f77_int  )m; \
	ctype*   a_la     = ( ctype*   )v; \
	f77_int  lda_la   = ( f77_int  )cs_v; \
	ctype_r* w_la     = ( ctype_r* )eu; \
\
	ctype_r  work_qy[ 2 ]; \
	f77_int  lwork_qy = -1; /* lwork == -1 triggers a workspace query. */ \
\
	ctype*   work_la  = ( ctype*   )work; \
	f77_int  lwork_la = ( f77_int  )lwork; \
\
	ctype_r* rwork_la = ( ctype_r* )rwork; \
\
	f77_int  info_la; \
	f77_int  jobz_len = 1; \
	f77_int  uplo_len = 1; \
\
	/* Query the optimal value for lwork. */ \
	PASTEF2C(ch,laname) \
	( \
	  &jobz_la, &uplo_la, \
	  &n_la, \
	  a_la, &lda_la, \
	  w_la, \
	  ( ctype* )work_qy, &lwork_qy, \
	  rwork_la, \
	  &info_la, \
	  jobz_len, uplo_len  \
	); \
\
	/* Make sure nothing went wrong in the workspace query. */ \
	if ( info_la != 0 ) bli_abort(); \
\
	/* Recover the optimal value for lwork. */ \
	f77_int lwork_op = ( f77_int )work_qy[0]; \
\
	/* If it turns out that lwork is not optimal, use the optimal value
	   queried above to allocate local workspace. */ \
	if ( lwork < lwork_op || work_null ) \
	{ \
		work     = bli_malloc_intl( ( size_t )lwork_op * sizeof( ctype ), &r_val ); \
		work_la  = work; \
		lwork_la = lwork_op; \
\
		/* Make sure nothing went wrong. */ \
		if ( r_val != BLIS_SUCCESS ) bli_abort(); \
	} \
\
	/* If rwork is NULL, allocate it. */ \
	if ( bli_is_complex( dt ) && rwork_null ) \
	{ \
		rwork    = bli_malloc_intl( 3 * ( size_t )m * sizeof( ctype_r ), &r_val ); \
		rwork_la = rwork; \
\
		/* Make sure nothing went wrong. */ \
		if ( r_val != BLIS_SUCCESS ) bli_abort(); \
	} \
\
	/* Call the LAPACK function, this time to perform the computation. */ \
	PASTEF2C(ch,laname) \
	( \
	  &jobz_la, &uplo_la, \
	  &n_la, \
	  a_la, &lda_la, \
	  w_la, \
	  work_la, &lwork_la, \
	  rwork_la, \
	  &info_la, \
	  jobz_len, uplo_len  \
	); \
\
	/* If we allocated a local rwork array, free it. */ \
	if ( bli_is_complex( dt ) && rwork_null ) bli_free_intl( rwork ); \
\
	/* If we allocated a local work array, free it. */ \
	if ( lwork < lwork_op || work_null ) bli_free_intl( work ); \
\
	/* If we used a temporary array for the eigenvalues, save the values to
	   the non-unit stride array that the caller passed in and then free the
	   temporary array. */ \
	if ( !e_unit ) \
	{ \
		PASTEMAC(chr,copyv) \
		( \
		  BLIS_NO_CONJUGATE, \
		  m, \
		  eu, 1, \
		  e,  ince  \
		); \
\
		bli_free_intl( eu ); \
	} \
\
	/* Make sure nothing went wrong in the computation. */ \
	if ( info_la != 0 ) \
		return info_la; \
\
	return BLIS_SUCCESS; \
}

GENTFUNCR( float,    float,  s, s, hevd, syev )
GENTFUNCR( double,   double, d, d, hevd, syev )
GENTFUNCR( scomplex, float,  c, s, hevd, heev )
GENTFUNCR( dcomplex, double, z, d, hevd, heev )


// -- Helper functions ---------------------------------------------------------

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       uplo_t*  uploa, \
       dim_t    m, \
       ctype*   a, inc_t* rs_a, inc_t* cs_a  \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Induce a transposition. */ \
	bli_swap_incs( rs_a, cs_a ); \
	bli_toggle_uplo( uploa ); \
\
	/* If the matrix is complex, we must also explicitly conjugate the stored
	   triangle (minus the diagonal, since its imaginary components are zero). */ \
	if ( bli_is_complex( dt ) ) \
	{ \
		const dim_t rs = *rs_a; \
		const dim_t cs = *cs_a; \
\
		if ( bli_is_lower( *uploa ) ) \
		{ \
			for ( dim_t j = 0; j < m; ++j ) \
			for ( dim_t i = 0; i < m; ++i ) \
			if ( j - i < 0 ) { PASTEMAC(ch,conjs)( *( a + i*rs + j*cs ) ); } \
		} \
		else /* bli_is_upper( *uploa ) */ \
		{ \
			for ( dim_t j = 0; j < m; ++j ) \
			for ( dim_t i = 0; i < m; ++i ) \
			if ( j - i > 0 ) { PASTEMAC(ch,conjs)( *( a + i*rs + j*cs ) ); } \
		} \
	} \
}

INSERT_GENTFUNC_BASIC0( hevd_induce_conjtrans )

