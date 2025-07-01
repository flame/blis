/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2019 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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
// Define BLAS-to-BLIS interfaces.
//
#define ENABLE_INDUCED_METHOD 0
#ifdef BLIS_BLAS3_CALLS_TAPI

#undef  GENTFUNC
#define GENTFUNC( ftype, ch, blasname, blisname ) \
\
void PASTEF77S(ch,blasname) \
     ( \
       const f77_char* transa, \
       const f77_char* transb, \
       const f77_int*  m, \
       const f77_int*  n, \
       const f77_int*  k, \
       const ftype*    alpha, \
       const ftype*    a, const f77_int* lda, \
       const ftype*    b, const f77_int* ldb, \
       const ftype*    beta, \
	     ftype*    c, const f77_int* ldc  \
     ) \
{ \
	trans_t blis_transa; \
	trans_t blis_transb; \
	dim_t   m0, n0, k0; \
	inc_t   rs_a, cs_a; \
	inc_t   rs_b, cs_b; \
	inc_t   rs_c, cs_c; \
\
	/* Initialize BLIS. */ \
	bli_init_auto(); \
\
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1); \
	AOCL_DTL_LOG_GEMM_INPUTS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(ch), *transa, *transb, *m, *n, *k, \
				(void*)alpha, *lda, *ldb, (void*)beta, *ldc); \
\
	/* Perform BLAS parameter checking. */ \
	PASTEBLACHK(blasname) \
	( \
	  MKSTR(ch), \
	  MKSTR(blasname), \
	  transa, \
	  transb, \
	  m, \
	  n, \
	  k, \
	  lda, \
	  ldb, \
	  ldc  \
	); \
\
	/* Quick return if possible. */ \
	if ( *m == 0 || *n == 0 || (( PASTEMAC(ch,eq0)( *alpha ) || *k == 0) \
	   && PASTEMAC(ch,eq1)( *beta ) )) \
	{ \
	  AOCL_DTL_LOG_GEMM_STATS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(ch), *m, *n, *k); \
	  AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1); \
	  /* Finalize BLIS. */ \
	  bli_finalize_auto(); \
	  return; \
	} \
\
	/* If alpha is zero scale C by beta and return early. */ \
	if( PASTEMAC(ch,eq0)( *alpha )) \
	{ \
	  bli_convert_blas_dim1(*m, m0); \
	  bli_convert_blas_dim1(*n, n0); \
	  const inc_t rs_c = 1; \
	  const inc_t cs_c = *ldc; \
\
	  PASTEMAC2(ch,scalm,_ex)( BLIS_NO_CONJUGATE, \
	              0, \
	              BLIS_NONUNIT_DIAG, \
	              BLIS_DENSE, \
	              m0, \
	              n0, \
	              (ftype*) beta, \
	              (ftype*) c, rs_c, cs_c, \
	              NULL, NULL \
	            ); \
\
	  AOCL_DTL_LOG_GEMM_STATS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(ch), *m, *n, *k); \
	  AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1); \
	  /* Finalize BLIS. */ \
	  bli_finalize_auto(); \
	  return; \
	} \
\
	/* Map BLAS chars to their corresponding BLIS enumerated type value. */ \
	bli_param_map_netlib_to_blis_trans( *transa, &blis_transa ); \
	bli_param_map_netlib_to_blis_trans( *transb, &blis_transb ); \
\
	/* Typecast BLAS integers to BLIS integers. */ \
	bli_convert_blas_dim1( *m, m0 ); \
	bli_convert_blas_dim1( *n, n0 ); \
	bli_convert_blas_dim1( *k, k0 ); \
\
	/* Set the row and column strides of the matrix operands. */ \
	rs_a = 1; \
	cs_a = *lda; \
	rs_b = 1; \
	cs_b = *ldb; \
	rs_c = 1; \
	cs_c = *ldc; \
\
	/* Call BLIS interface. */ \
	PASTEMAC2(ch,blisname,BLIS_TAPI_EX_SUF) \
	( \
	  blis_transa, \
	  blis_transb, \
	  m0, \
	  n0, \
	  k0, \
	  (ftype*)alpha, \
	  (ftype*)a, rs_a, cs_a, \
	  (ftype*)b, rs_b, cs_b, \
	  (ftype*)beta, \
	  (ftype*)c, rs_c, cs_c, \
	  NULL, \
	  NULL  \
	); \
\
	AOCL_DTL_LOG_GEMM_STATS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(ch), *m, *n, *k);\
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1); \
	/* Finalize BLIS. */				 \
	bli_finalize_auto(); \
} \
IF_BLIS_ENABLE_BLAS(\
void PASTEF77(ch,blasname) \
     ( \
       const f77_char* transa, \
       const f77_char* transb, \
       const f77_int*  m, \
       const f77_int*  n, \
       const f77_int*  k, \
       const ftype*    alpha, \
       const ftype*    a, const f77_int* lda, \
       const ftype*    b, const f77_int* ldb, \
       const ftype*    beta, \
	     ftype*    c, const f77_int* ldc  \
     ) \
{ \
	PASTEF77S(ch,blasname) ( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc ); \
} \
)

#else

#undef  GENTFUNC
#define GENTFUNC( ftype, ch, blasname, blisname ) \
\
void PASTEF77S(ch,blasname) \
     ( \
       const f77_char* transa, \
       const f77_char* transb, \
       const f77_int*  m, \
       const f77_int*  n, \
       const f77_int*  k, \
       const ftype*    alpha, \
       const ftype*    a, const f77_int* lda, \
       const ftype*    b, const f77_int* ldb, \
       const ftype*    beta, \
	     ftype*    c, const f77_int* ldc  \
     ) \
{ \
\
	trans_t blis_transa; \
	trans_t blis_transb; \
	dim_t   m0, n0, k0; \
\
	dim_t       m0_a, n0_a; \
	dim_t       m0_b, n0_b; \
\
	/* Initialize BLIS. */ \
	bli_init_auto(); \
\
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1); \
	AOCL_DTL_LOG_GEMM_INPUTS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(ch), *transa, *transb, *m, *n, *k, \
				(void*)alpha, *lda, *ldb, (void*)beta, *ldc); \
\
	/* Perform BLAS parameter checking. */ \
	PASTEBLACHK(blasname) \
	( \
	  MKSTR(ch), \
	  MKSTR(blasname), \
	  transa, \
	  transb, \
	  m, \
	  n, \
	  k, \
	  lda, \
	  ldb, \
	  ldc  \
	); \
\
	/* Quick return if possible. */ \
	if ( *m == 0 || *n == 0 || (( PASTEMAC(ch,eq0)( *alpha ) || *k == 0) \
	   && PASTEMAC(ch,eq1)( *beta ) )) \
	{ \
	  AOCL_DTL_LOG_GEMM_STATS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(ch), *m, *n, *k); \
	  AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1); \
	  /* Finalize BLIS. */ \
	  bli_finalize_auto(); \
	  return; \
	} \
\
	/* If alpha is zero scale C by beta and return early. */ \
	if( PASTEMAC(ch,eq0)( *alpha )) \
	{ \
	  bli_convert_blas_dim1(*m, m0); \
	  bli_convert_blas_dim1(*n, n0); \
	  const inc_t rs_c = 1; \
	  const inc_t cs_c = *ldc; \
\
	  PASTEMAC2(ch,scalm,_ex)( BLIS_NO_CONJUGATE, \
                   0, \
                   BLIS_NONUNIT_DIAG, \
                   BLIS_DENSE, \
                   m0, \
                   n0, \
                   (ftype*) beta, \
                   (ftype*) c, rs_c, cs_c, \
                   NULL, NULL \
                 ); \
\
	  AOCL_DTL_LOG_GEMM_STATS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(ch), *m, *n, *k); \
	  AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1); \
	  /* Finalize BLIS. */ \
	  bli_finalize_auto(); \
	  return; \
	} \
\
	/* Map BLAS chars to their corresponding BLIS enumerated type value. */ \
	bli_param_map_netlib_to_blis_trans( *transa, &blis_transa ); \
	bli_param_map_netlib_to_blis_trans( *transb, &blis_transb ); \
\
	/* Typecast BLAS integers to BLIS integers. */ \
	bli_convert_blas_dim1( *m, m0 ); \
	bli_convert_blas_dim1( *n, n0 ); \
	bli_convert_blas_dim1( *k, k0 ); \
\
	/* Set the row and column strides of the matrix operands. */ \
	const inc_t rs_a = 1; \
	const inc_t cs_a = *lda; \
	const inc_t rs_b = 1; \
	const inc_t cs_b = *ldb; \
	const inc_t rs_c = 1; \
	const inc_t cs_c = *ldc; \
\
	if( n0 == 1 ) \
	{ \
		if(bli_is_notrans(blis_transa)) \
		{ \
			PASTEMAC(ch,gemv_unf_var2)( \
					BLIS_NO_TRANSPOSE, \
					bli_extract_conj(blis_transb), \
					m0, k0, \
					(ftype*)alpha, \
					(ftype*)a, rs_a, cs_a,\
					(ftype*)b, bli_is_notrans(blis_transb)?rs_b:cs_b, \
					(ftype*) beta, \
					c, rs_c, \
					NULL \
					); \
		} \
		else \
		{ \
			PASTEMAC(ch,gemv_unf_var1)( \
					blis_transa, \
					bli_extract_conj(blis_transb), \
					k0, m0, \
					(ftype*)alpha, \
					(ftype*)a, rs_a, cs_a, \
					(ftype*)b, bli_is_notrans(blis_transb)?rs_b:cs_b, \
					(ftype*)beta, \
					c, rs_c, \
					NULL \
					); \
		} \
		AOCL_DTL_LOG_GEMM_STATS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(ch), *m, *n, *k); \
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1); \
		/* Finalize BLIS. */ \
  		bli_finalize_auto(); \
		return; \
	} \
	else if( m0 == 1 ) \
	{ \
		if(bli_is_notrans(blis_transb)) \
		{ \
			PASTEMAC(ch,gemv_unf_var1)( \
					blis_transb, \
					bli_extract_conj(blis_transa), \
					n0, k0, \
					(ftype*)alpha, \
					(ftype*)b, cs_b, rs_b, \
					(ftype*)a, bli_is_notrans(blis_transa)?cs_a:rs_a, \
					(ftype*)beta, \
					c, cs_c, \
					NULL \
					); \
		} \
		else \
		{ \
			PASTEMAC(ch,gemv_unf_var2)( \
					blis_transb, \
					bli_extract_conj(blis_transa), \
					k0, n0, \
					(ftype*)alpha, \
					(ftype*)b, cs_b, rs_b, \
					(ftype*)a, bli_is_notrans(blis_transa)?cs_a:rs_a, \
					(ftype*)beta, \
					c, cs_c, \
					NULL \
					); \
		} \
		AOCL_DTL_LOG_GEMM_STATS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(ch), *m, *n, *k); \
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1); \
		/* Finalize BLIS. */ \
  		bli_finalize_auto(); \
		return; \
	} \
\
	const num_t dt     = PASTEMAC(ch,type); \
\
	obj_t       alphao = BLIS_OBJECT_INITIALIZER_1X1; \
	obj_t       ao     = BLIS_OBJECT_INITIALIZER; \
	obj_t       bo     = BLIS_OBJECT_INITIALIZER; \
	obj_t       betao  = BLIS_OBJECT_INITIALIZER_1X1; \
	obj_t       co     = BLIS_OBJECT_INITIALIZER; \
\
	bli_set_dims_with_trans( blis_transa, m0, k0, &m0_a, &n0_a ); \
	bli_set_dims_with_trans( blis_transb, k0, n0, &m0_b, &n0_b ); \
\
	bli_obj_init_finish_1x1( dt, (ftype*)alpha, &alphao ); \
	bli_obj_init_finish_1x1( dt, (ftype*)beta,  &betao  ); \
\
	bli_obj_init_finish( dt, m0_a, n0_a, (ftype*)a, rs_a, cs_a, &ao ); \
	bli_obj_init_finish( dt, m0_b, n0_b, (ftype*)b, rs_b, cs_b, &bo ); \
	bli_obj_init_finish( dt, m0,   n0,   (ftype*)c, rs_c, cs_c, &co ); \
\
	bli_obj_set_conjtrans( blis_transa, &ao ); \
	bli_obj_set_conjtrans( blis_transb, &bo ); \
\
	PASTEMAC(blisname,BLIS_OAPI_EX_SUF) \
	( \
	  &alphao, \
	  &ao, \
	  &bo, \
	  &betao, \
	  &co, \
	  NULL, \
	  NULL  \
	); \
\
	AOCL_DTL_LOG_GEMM_STATS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(ch), *m, *n, *k); \
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1); \
	/* Finalize BLIS. */				 \
	bli_finalize_auto(); \
} \
IF_BLIS_ENABLE_BLAS(\
void PASTEF77(ch,blasname) \
     ( \
       const f77_char* transa, \
       const f77_char* transb, \
       const f77_int*  m, \
       const f77_int*  n, \
       const f77_int*  k, \
       const ftype*    alpha, \
       const ftype*    a, const f77_int* lda, \
       const ftype*    b, const f77_int* ldb, \
       const ftype*    beta, \
	     ftype*    c, const f77_int* ldc  \
     ) \
{ \
	PASTEF77S(ch,blasname) ( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc ); \
} \
)
#endif

INSERT_GENTFUNC_BLAS( gemm,gemm )

void dzgemm_blis_impl
     (
       const f77_char* transa,
       const f77_char* transb,
       const f77_int*  m,
       const f77_int*  n,
       const f77_int*  k,
       const dcomplex* alpha,
       const double*   a, const f77_int* lda,
       const dcomplex* b, const f77_int* ldb,
       const dcomplex* beta,
             dcomplex* c, const f77_int* ldc
     )
{

  trans_t blis_transa;
  trans_t blis_transb;
  dim_t   m0, n0, k0;

  /* Initialize BLIS. */
  bli_init_auto();

  AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1)
  AOCL_DTL_LOG_GEMM_INPUTS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(z), *transa, *transb, *m, *n, *k,
				(void*)alpha, *lda, *ldb, (void*)beta, *ldc);

  /* Perform BLAS parameter checking. */
	PASTEBLACHK(gemm)
	(
	  MKSTR(z),
	  MKSTR(gemm),
	  transa,
	  transb,
	  m,
	  n,
	  k,
	  lda,
	  ldb,
	  ldc
	);

	/* Quick return if possible. */
	if ( *m == 0 || *n == 0 || (( PASTEMAC(z,eq0)( *alpha ) || *k == 0)
	   && PASTEMAC(z,eq1)( *beta ) ))
	{
	  AOCL_DTL_LOG_GEMM_STATS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(z), *m, *n, *k);
	  AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
	  /* Finalize BLIS. */
	  bli_finalize_auto();
	  return;
	}

	/* If alpha is zero scale C by beta and return early. */
	if( PASTEMAC(z,eq0)( *alpha ))
	{
	  bli_convert_blas_dim1(*m, m0);
	  bli_convert_blas_dim1(*n, n0);
	  const inc_t rs_c = 1;
	  const inc_t cs_c = *ldc;

	  PASTEMAC2(z,scalm,_ex)( BLIS_NO_CONJUGATE,
	            0,
	            BLIS_NONUNIT_DIAG,
	            BLIS_DENSE,
	            m0,
	            n0,
	            (dcomplex*) beta,
	            (dcomplex*) c, rs_c, cs_c,
	            NULL, NULL
	  );

	  AOCL_DTL_LOG_GEMM_STATS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(z), *m, *n, *k);
	  AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
	  /* Finalize BLIS. */
	  bli_finalize_auto();
	  return;
	}

	/* Map BLAS chars to their corresponding BLIS enumerated type value. */
	bli_param_map_netlib_to_blis_trans( *transa, &blis_transa );
	bli_param_map_netlib_to_blis_trans( *transb, &blis_transb );

	/* Typecast BLAS integers to BLIS integers. */
	bli_convert_blas_dim1( *m, m0 );
	bli_convert_blas_dim1( *n, n0 );
	bli_convert_blas_dim1( *k, k0 );

	/* Set the row and column strides of the matrix operands. */
	const inc_t rs_a = 1;
	const inc_t cs_a = *lda;
	const inc_t rs_b = 1;
	const inc_t cs_b = *ldb;
	const inc_t rs_c = 1;
	const inc_t cs_c = *ldc;

	const num_t dt     = BLIS_DCOMPLEX;
	const num_t dt_a   = BLIS_DOUBLE;

	obj_t       alphao = BLIS_OBJECT_INITIALIZER_1X1;
	obj_t       ao     = BLIS_OBJECT_INITIALIZER;
	obj_t       bo     = BLIS_OBJECT_INITIALIZER;
	obj_t       betao  = BLIS_OBJECT_INITIALIZER_1X1;
	obj_t       co     = BLIS_OBJECT_INITIALIZER;

	dim_t       m0_a, n0_a;
	dim_t       m0_b, n0_b;

	bli_set_dims_with_trans( blis_transa, m0, k0, &m0_a, &n0_a );
	bli_set_dims_with_trans( blis_transb, k0, n0, &m0_b, &n0_b );

	bli_obj_init_finish_1x1( dt, (dcomplex*)alpha, &alphao );
	bli_obj_init_finish_1x1( dt, (dcomplex*)beta,  &betao  );

	bli_obj_init_finish( dt_a, m0_a, n0_a, (double*)a, rs_a, cs_a, &ao );
	bli_obj_init_finish( dt, m0_b, n0_b, (dcomplex*)b, rs_b, cs_b, &bo );
	bli_obj_init_finish( dt, m0,   n0,   (dcomplex*)c, rs_c, cs_c, &co );

	bli_obj_set_conjtrans( blis_transa, &ao );
	bli_obj_set_conjtrans( blis_transb, &bo );

	// fall back on native path when zgemm is not handled in sup path.
	//bli_gemmnat(&alphao, &ao, &bo, &betao, &co, NULL, NULL);

	/* Default to using native execution. */
	ind_t im = BLIS_NAT;

	/* Mix of real and complex matrix data types, so assuming
	   induced methods will not be available */

	/* Obtain a valid context from the gks using the induced
	   method id determined above. */
	cntx_t* cntx = bli_gks_query_ind_cntx( im, dt );

	rntm_t rntm_l;
	bli_rntm_init_from_global( &rntm_l );

	/* Invoke the operation's front-end and request the default control tree. */
	PASTEMAC(gemm,_front)( &alphao, &ao, &bo, &betao, &co, cntx, &rntm_l, NULL );

	AOCL_DTL_LOG_GEMM_STATS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(z), *m, *n, *k);
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
	/* Finalize BLIS. */
	bli_finalize_auto();
}// end of dzgemm_
#ifdef BLIS_ENABLE_BLAS
void dzgemm_
     (
       const f77_char* transa,
       const f77_char* transb,
       const f77_int*  m,
       const f77_int*  n,
       const f77_int*  k,
       const dcomplex* alpha,
       const double*   a, const f77_int* lda,
       const dcomplex* b, const f77_int* ldb,
       const dcomplex* beta,
             dcomplex* c, const f77_int* ldc
     )
{
    dzgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc );
}
#endif
