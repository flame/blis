/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2019 - 21, Advanced Micro Devices, Inc. All rights reserved.

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
#define ENABLE_INDUCED_METHOD 1
#ifdef BLIS_BLAS3_CALLS_TAPI

#undef  GENTFUNC
#define GENTFUNC( ftype, ch, blasname, blisname ) \
\
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
	trans_t blis_transa; \
	trans_t blis_transb; \
	dim_t   m0, n0, k0; \
	inc_t   rs_a, cs_a; \
	inc_t   rs_b, cs_b; \
	inc_t   rs_c, cs_c; \
\
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1); \
	/* Initialize BLIS. */ \
	bli_init_auto(); \
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
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1) \
	/* Finalize BLIS. */				 \
	bli_finalize_auto(); \
}

#else

#undef  GENTFUNC
#define GENTFUNC( ftype, ch, blasname, blisname ) \
\
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
	AOCL_DTL_LOG_GEMM_INPUTS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(ch), *transa, *transb, *m, *n, *k, (void*)alpha, *lda, *ldb, (void*)beta, *ldc); \
\
	trans_t blis_transa; \
	trans_t blis_transb; \
	dim_t   m0, n0, k0; \
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_INFO)	\
\
	dim_t       m0_a, n0_a; \
	dim_t       m0_b, n0_b; \
\
	/* Initialize BLIS. */ \
	bli_init_auto(); \
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
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO) \
	/* Finalize BLIS. */				 \
	bli_finalize_auto(); \
}
#endif

#ifdef BLIS_ENABLE_BLAS
#ifdef BLIS_CONFIG_EPYC
void dgemm_
(
	const f77_char* transa,
	const f77_char* transb,
	const f77_int* m,
	const f77_int* n,
	const f77_int* k,
	const double* alpha,
	const double* a, const f77_int* lda,
	const double* b, const f77_int* ldb,
	const double* beta,
	double* c, const f77_int* ldc
)
{
  AOCL_DTL_LOG_GEMM_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'D', *transa, *transb, *m, *n, *k, (void*)alpha, *lda, *ldb, (void*)beta, *ldc);

  trans_t blis_transa;
  trans_t blis_transb;
  dim_t   m0, n0, k0;
  AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_INFO)

    /* Initialize BLIS. */
    bli_init_auto();

  /* Perform BLAS parameter checking. */
  PASTEBLACHK(gemm)
    (
     MKSTR(d),
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

	/* Map BLAS chars to their corresponding BLIS enumerated type value. */
  bli_param_map_netlib_to_blis_trans(*transa, &blis_transa);
  bli_param_map_netlib_to_blis_trans(*transb, &blis_transb);

  /* Typecast BLAS integers to BLIS integers. */
  bli_convert_blas_dim1(*m, m0);
  bli_convert_blas_dim1(*n, n0);
  bli_convert_blas_dim1(*k, k0);


    /* Set the row and column strides of the matrix operands. */
    const inc_t rs_a = 1;
    const inc_t cs_a = *lda;
    const inc_t rs_b = 1;
    const inc_t cs_b = *ldb;
    const inc_t rs_c = 1;
    const inc_t cs_c = *ldc;

    if((k0 == 1) && bli_is_notrans(blis_transa) && bli_is_notrans(blis_transb))
    {
	bli_dgemm_ref_k1_nn( m0, n0, k0,
			  (double*)alpha,
			  (double*)a, *lda,
			  (double*)b, *ldb,
			  (double*)beta,
			  c, *ldc
			);
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
	/* Finalize BLIS */
	bli_finalize_auto();

	return;
    }

    if (n0 == 1)
    {
	if (bli_is_notrans(blis_transa))
	{
	    bli_dgemv_unf_var2(
		BLIS_NO_TRANSPOSE,
		bli_extract_conj(blis_transb),
		m0, k0,
		(double*)alpha,
		(double*)a, rs_a, cs_a,
		(double*)b, bli_is_notrans(blis_transb) ? rs_b : cs_b,
		(double*)beta,
		c, rs_c,
		((void*)0)
	    );
	}
	else
	{
	    bli_dgemv_unf_var1(
		blis_transa,
		bli_extract_conj(blis_transb),
		k0, m0,
		(double*)alpha,
		(double*)a, rs_a, cs_a,
		(double*)b, bli_is_notrans(blis_transb) ? rs_b : cs_b,
		(double*)beta,
		c, rs_c,
		((void*)0)
	    );
	}
	return;
    }
    else if (m0 == 1)
    {
	if (bli_is_notrans(blis_transb))
	{
	    bli_dgemv_unf_var1(
		blis_transb,
		bli_extract_conj(blis_transa),
		n0, k0,
		(double*)alpha,
		(double*)b, cs_b, rs_b,
		(double*)a, bli_is_notrans(blis_transa) ? cs_a : rs_a,
		(double*)beta,
		c, cs_c,
		((void*)0)
	    );
	}
	else
	{
	    bli_dgemv_unf_var2(
		blis_transb,
		bli_extract_conj(blis_transa),
		k0, n0,
		(double*)alpha,
		(double*)b, cs_b, rs_b,
		(double*)a, bli_is_notrans(blis_transa) ? cs_a : rs_a,
		(double*)beta,
		c, cs_c,
		((void*)0)
	    );
	}
	return;
    }

    const num_t dt = BLIS_DOUBLE;

    obj_t       alphao = BLIS_OBJECT_INITIALIZER_1X1;
    obj_t       ao = BLIS_OBJECT_INITIALIZER;
    obj_t       bo = BLIS_OBJECT_INITIALIZER;
    obj_t       betao = BLIS_OBJECT_INITIALIZER_1X1;
    obj_t       co = BLIS_OBJECT_INITIALIZER;

    dim_t       m0_a, n0_a;
    dim_t       m0_b, n0_b;

    bli_set_dims_with_trans(blis_transa, m0, k0, &m0_a, &n0_a);
    bli_set_dims_with_trans(blis_transb, k0, n0, &m0_b, &n0_b);

    bli_obj_init_finish_1x1(dt, (double*)alpha, &alphao);
    bli_obj_init_finish_1x1(dt, (double*)beta, &betao);

    bli_obj_init_finish(dt, m0_a, n0_a, (double*)a, rs_a, cs_a, &ao);
    bli_obj_init_finish(dt, m0_b, n0_b, (double*)b, rs_b, cs_b, &bo);
    bli_obj_init_finish(dt, m0, n0, (double*)c, rs_c, cs_c, &co);

    bli_obj_set_conjtrans(blis_transa, &ao);
    bli_obj_set_conjtrans(blis_transb, &bo);

    //cntx_t* cntx = bli_gks_query_cntx();
    //dim_t nt = bli_thread_get_num_threads(); // get number of threads
    bool nt = bli_thread_get_is_parallel(); // Check if parallel dgemm is invoked.

    // if m0 is large and (n0 & k0) < 10 - SMALL GEMM - ST is better
    //

#ifdef AOCL_DYNAMIC
    if (nt && ((n0 > 10 ) || (k0 > 10)) )
#else
      if (nt)
#endif
      {
	// Will call parallelized dgemm code - sup & native
	PASTEMAC(gemm, BLIS_OAPI_EX_SUF)
		(
			&alphao,
			&ao,
			&bo,
			&betao,
			&co,
			NULL,
			NULL
		 );

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
	/* Finalize BLIS. */
	bli_finalize_auto();
	return;
      }

    // The code below will be called when number of threads = 1.

#ifdef BLIS_ENABLE_SMALL_MATRIX

    //if( ((m0 + n0 -k0) < 2000) && ((m0 + k0-n0) < 2000) && ((n0 + k0-m0) < 2000) && (n0 > 2))
    if( ( ( (m0 + n0 -k0) < 2000) && ((m0 + k0-n0) < 2000) && ((n0 + k0-m0) < 2000) ) ||
	  ((n0 <= 10) && (k0 <=10)) )
      {
	err_t status;
	if (bli_is_notrans(blis_transa))
	  {
	    status =  bli_dgemm_small( &alphao,
					   &ao,
					   &bo,
					   &betao,
					   &co,
					   NULL, //cntx,
					   NULL
					   );
	  }
	else
	  {
	    status =  bli_dgemm_small_At ( &alphao,
					           &ao,
					           &bo,
					           &betao,
					           &co,
					           NULL, //cntx,
					           NULL
					         );
	  }

	if (status == BLIS_SUCCESS)
	  {
	    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
	    /* Finalize BLIS. */
	    bli_finalize_auto();

	    return;
	  }
      }

#endif //#ifdef BLIS_ENABLE_SMALL_MATRIX

    err_t status = bli_gemmsup(&alphao, &ao, &bo, &betao, &co, NULL, NULL);
	if (status == BLIS_SUCCESS)
	{
	    return;
	}

	// fall back on native path when dgemm is not handled in sup path.
	bli_gemmnat(&alphao, &ao, &bo, &betao, &co, NULL, NULL);


	/* PASTEMAC(gemm, BLIS_OAPI_EX_SUF) */
	/*	( */
	/*		&alphao, */
	/*		&ao, */
	/*		&bo, */
	/*		&betao, */
	/*		&co, */
	/*		NULL, */
	/*		NULL */
	/*	 ); */

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
	/* Finalize BLIS. */
	bli_finalize_auto();
} // end of dgemm_

void zgemm_
     (
       const f77_char* transa,
       const f77_char* transb,
       const f77_int*  m,
       const f77_int*  n,
       const f77_int*  k,
       const dcomplex*    alpha,
       const dcomplex*    a, const f77_int* lda,
       const dcomplex*    b, const f77_int* ldb,
       const dcomplex*    beta,
             dcomplex*    c, const f77_int* ldc
     )
{
  AOCL_DTL_LOG_GEMM_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'Z', *transa, *transb, *m, *n, *k, (void*)alpha, *lda, *ldb, (void*)beta, *ldc);

  trans_t blis_transa;
  trans_t blis_transb;
  dim_t   m0, n0, k0;
  AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_INFO)

    /* Initialize BLIS. */
    bli_init_auto();

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

	bli_obj_init_finish( dt, m0_a, n0_a, (dcomplex*)a, rs_a, cs_a, &ao );
	bli_obj_init_finish( dt, m0_b, n0_b, (dcomplex*)b, rs_b, cs_b, &bo );
	bli_obj_init_finish( dt, m0,   n0,   (dcomplex*)c, rs_c, cs_c, &co );

	bli_obj_set_conjtrans( blis_transa, &ao );
	bli_obj_set_conjtrans( blis_transb, &bo );

	// default instance peformance tuning is done in zgemm.
	// Single instance tuning is done based on env set.
	dim_t single_instance = bli_env_get_var( "BLIS_SINGLE_INSTANCE", -1 );

	//dim_t nt = bli_thread_get_num_threads(); // get number of threads
	bool nt = bli_thread_get_is_parallel(); // Check if parallel zgemm is invoked.
	if ( nt )
	{
		// Will call parallelized zgemm code - sup & native
		PASTEMAC(gemm, BLIS_OAPI_EX_SUF)
			(
			&alphao,
			&ao,
			&bo,
			&betao,
			&co,
			NULL,
			NULL
			);

		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
		/* Finalize BLIS. */
		bli_finalize_auto();
		return;
	}

    // The code below will be called when number of threads = 1.
#if ENABLE_INDUCED_METHOD
	/* 3m_sqp is optimal for certain matrix shapes.
	   Initial study that it works well for square sizes and sizes closer to square shape.

	   * Usage of 3m_sqp is restricted to sizes, where it is found efficient compared to native, sup and other induced method.
	   * Further investigation is necessary to make the usage choices more generic.  */
	bool sqp_on = false;
	if( (m0 == n0 ) && ( n0 == k0 ) && ( m0 == 128 ) )
	{
		sqp_on = true;
	}

	// current range of sizes used for 3m_sqp to be expaned after evaluation.
	if( ( m0 >= 4200) && ( m0 <= 4600 ) && ( ( n0 >= 326 ) || (n0 <= 1600 ) )
     && ( k0 == 1120 ) ) //to be tuned further.
	{
		sqp_on = true;
	}

	if( ( blis_transb == BLIS_NO_TRANSPOSE) && ( sqp_on == true ) )
	{
		//sqp algo is found better for n > 40
		if(bli_gemm_sqp(&alphao, &ao, &bo, &betao, &co, NULL, NULL)==BLIS_SUCCESS)
		{
			AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO)
			return;
		}
	}
#endif//ENABLE_INDUCED_METHOD

// native tuning resulted in better numbers compared to sup in constrained multi-instance
// sup has been enabled for single instance cases.
	if(single_instance==1)
	{
		err_t status = bli_gemmsup(&alphao, &ao, &bo, &betao, &co, NULL, NULL);
		if(status==BLIS_SUCCESS)
		{
			AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO)
			return;
		}
	}
	// fall back on native path when zgemm is not handled in sup path.
	bli_gemmnat(&alphao, &ao, &bo, &betao, &co, NULL, NULL);
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO)
	return;


	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO)
	/* Finalize BLIS. */
	bli_finalize_auto();
}// end of zgemm_


INSERT_GENTFUNC_BLAS_SC( gemm, gemm )
#else
INSERT_GENTFUNC_BLAS( gemm,gemm )
#endif

// Observed a regression in dgemm with this function addition.
// Disabling temporarily.
#if 0
void dzgemm_
     (
       const f77_char* transa,
       const f77_char* transb,
       const f77_int*  m,
       const f77_int*  n,
       const f77_int*  k,
       const dcomplex*    alpha,
       const double*    a, const f77_int* lda,
       const dcomplex*    b, const f77_int* ldb,
       const dcomplex*    beta,
             dcomplex*    c, const f77_int* ldc
     )
{

  AOCL_DTL_LOG_GEMM_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'Z', *transa, *transb, *m, *n, *k, (void*)alpha, *lda, *ldb, (void*)beta, *ldc);

  trans_t blis_transa;
  trans_t blis_transb;
  dim_t   m0, n0, k0;
  AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_INFO)

    /* Initialize BLIS. */
    bli_init_auto();

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

	bli_obj_init_finish( dt_a, m0_a, n0_a, (dcomplex*)a, rs_a, cs_a, &ao );
	bli_obj_init_finish( dt, m0_b, n0_b, (dcomplex*)b, rs_b, cs_b, &bo );
	bli_obj_init_finish( dt, m0,   n0,   (dcomplex*)c, rs_c, cs_c, &co );

	bli_obj_set_conjtrans( blis_transa, &ao );
	bli_obj_set_conjtrans( blis_transb, &bo );

		// fall back on native path when zgemm is not handled in sup path.
	bli_gemmnat(&alphao, &ao, &bo, &betao, &co, NULL, NULL);

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO)
	/* Finalize BLIS. */
	bli_finalize_auto();
}// end of dzgemm_
#endif
#endif
