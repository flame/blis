/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2019-2023, Advanced Micro Devices, Inc. All rights reserved.

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

#ifdef BLIS_BLAS3_CALLS_TAPI

#undef  GENTFUNC
#define GENTFUNC( ftype, ch, blasname, blisname ) \
\
void PASTEF77S(ch,blasname) \
     ( \
       const f77_char* side, \
       const f77_char* uploa, \
       const f77_char* transa, \
       const f77_char* diaga, \
       const f77_int*  m, \
       const f77_int*  n, \
       const ftype*    alpha, \
       const ftype*    a, const f77_int* lda, \
             ftype*    b, const f77_int* ldb  \
     ) \
{ \
        AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_INFO)    \
\
    side_t  blis_side; \
    uplo_t  blis_uploa; \
    trans_t blis_transa; \
    diag_t  blis_diaga; \
    dim_t   m0, n0; \
    inc_t   rs_a, cs_a; \
    inc_t   rs_b, cs_b; \
\
    /* Initialize BLIS. */ \
    bli_init_auto(); \
\
    /* Perform BLAS parameter checking. */ \
    PASTEBLACHK(blasname) \
    ( \
      MKSTR(ch), \
      MKSTR(blasname), \
      side, \
      uploa, \
      transa, \
      diaga, \
      m, \
      n, \
      lda, \
      ldb  \
    ); \
\
    /* Quick return if possible. */ \
    if ( *m == 0 || *n == 0 ) \
    { \
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO); \
        /* Finalize BLIS. */ \
        bli_finalize_auto(); \
        return; \
    } \
\
    /* Map BLAS chars to their corresponding BLIS enumerated type value. */ \
    bli_param_map_netlib_to_blis_side( *side,  &blis_side ); \
    bli_param_map_netlib_to_blis_uplo( *uploa, &blis_uploa ); \
    bli_param_map_netlib_to_blis_trans( *transa, &blis_transa ); \
    bli_param_map_netlib_to_blis_diag( *diaga, &blis_diaga ); \
\
    /* Typecast BLAS integers to BLIS integers. */ \
    bli_convert_blas_dim1( *m, m0 ); \
    bli_convert_blas_dim1( *n, n0 ); \
\
    /* Set the row and column strides of the matrix operands. */ \
    rs_a = 1; \
    cs_a = *lda; \
    rs_b = 1; \
    cs_b = *ldb; \
\
	/* If alpha is zero, set B to zero and return early */ \
	if( PASTEMAC(ch,eq0)( *alpha ) ) \
	{ \
		PASTEMAC2(ch,setm,_ex)( BLIS_NO_CONJUGATE, \
								0, \
								BLIS_NONUNIT_DIAG, \
								BLIS_DENSE, \
								m0, n0, \
								(ftype*) alpha, \
								(ftype*) b, rs_b, cs_b, \
								NULL, NULL \
							  ); \
		 AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1) \
		/* Finalize BLIS. */ \
		bli_finalize_auto(); \
		return; \
	} \
\
    /* Call BLIS interface. */ \
    PASTEMAC2(ch,blisname,BLIS_TAPI_EX_SUF) \
    ( \
      blis_side, \
      blis_uploa, \
      blis_transa, \
      blis_diaga, \
      m0, \
      n0, \
      (ftype*)alpha, \
      (ftype*)a, rs_a, cs_a, \
      (ftype*)b, rs_b, cs_b, \
      NULL, \
      NULL  \
    ); \
\
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO) \
    /* Finalize BLIS. */ \
    bli_finalize_auto(); \
} \
IF_BLIS_ENABLE_BLAS(\
void PASTEF77(ch,blasname) \
     ( \
       const f77_char* side, \
       const f77_char* uploa, \
       const f77_char* transa, \
       const f77_char* diaga, \
       const f77_int*  m, \
       const f77_int*  n, \
       const ftype*    alpha, \
       const ftype*    a, const f77_int* lda, \
             ftype*    b, const f77_int* ldb  \
     ) \
{ \
    PASTEF77S(ch,blasname) ( side, uploa, transa, diaga, m, n, alpha, a, lda, b, ldb ); \
} \
)
#else

#undef  GENTFUNC
#define GENTFUNC( ftype, ch, blasname, blisname ) \
\
void PASTEF77S(ch,blasname) \
     ( \
       const f77_char* side, \
       const f77_char* uploa, \
       const f77_char* transa, \
       const f77_char* diaga, \
       const f77_int*  m, \
       const f77_int*  n, \
       const ftype*    alpha, \
       const ftype*    a, const f77_int* lda, \
             ftype*    b, const f77_int* ldb  \
     ) \
{ \
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_INFO) \
    AOCL_DTL_LOG_TRSM_INPUTS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(ch), *side, *uploa, \
                 *transa, *diaga, *m, *n, (void*)alpha, *lda, *ldb); \
    side_t  blis_side; \
    uplo_t  blis_uploa; \
    trans_t blis_transa; \
    diag_t  blis_diaga; \
    dim_t   m0, n0; \
    ftype   a_conj; \
    conj_t  conja = BLIS_NO_CONJUGATE ; \
\
    /* Initialize BLIS. */ \
    bli_init_auto(); \
\
    /* Perform BLAS parameter checking. */ \
    PASTEBLACHK(blasname) \
    ( \
      MKSTR(ch), \
      MKSTR(blasname), \
      side, \
      uploa, \
      transa, \
      diaga, \
      m, \
      n, \
      lda, \
      ldb  \
    ); \
\
    /* Quick return if possible. */ \
    if ( *m == 0 || *n == 0 ) \
    { \
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO); \
        /* Finalize BLIS. */ \
        bli_finalize_auto(); \
        return; \
    } \
\
    /* Map BLAS chars to their corresponding BLIS enumerated type value. */ \
    bli_param_map_netlib_to_blis_side( *side,  &blis_side ); \
    bli_param_map_netlib_to_blis_uplo( *uploa, &blis_uploa ); \
    bli_param_map_netlib_to_blis_trans( *transa, &blis_transa ); \
    bli_param_map_netlib_to_blis_diag( *diaga, &blis_diaga ); \
\
    /* Typecast BLAS integers to BLIS integers. */ \
    bli_convert_blas_dim1( *m, m0 ); \
    bli_convert_blas_dim1( *n, n0 ); \
\
    /* Set the row and column strides of the matrix operands. */ \
    const inc_t rs_a = 1; \
    const inc_t cs_a = *lda; \
    const inc_t rs_b = 1; \
    const inc_t cs_b = *ldb; \
    const num_t dt = PASTEMAC(ch,type); \
\
	/* If alpha is zero, set B to zero and return early */ \
	if( PASTEMAC(ch,eq0)( *alpha ) ) \
	{ \
		PASTEMAC2(ch,setm,_ex)( BLIS_NO_CONJUGATE, \
								0, \
								BLIS_NONUNIT_DIAG, \
								BLIS_DENSE, \
								m0, n0, \
								(ftype*) alpha, \
								(ftype*) b, rs_b, cs_b, \
								NULL, NULL \
							  ); \
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1) \
		/* Finalize BLIS. */ \
		bli_finalize_auto(); \
		return; \
	} \
\
    /* ----------------------------------------------------------- */ \
    /*    TRSM API: AX = B, where X = B                            */ \
    /*    CALL TRSV when X & B are vector and when A is Matrix     */ \
    /*    Case 1: LEFT  : TRSM,  B(mxn) = A(mxm) * X(mxn)          */ \
    /*    Case 2: RIGHT : TRSM,  B(mxn) = X(mxn) * A(nxn)          */ \
    /* |--------|-------|-------|-------|------------------------| */ \
    /* |        |   A   |   X   |   B   |   Implementation       | */ \
    /* |--------|-------|-------|-------|------------------------| */ \
    /* | LEFT   |  mxm  |  mxn  |  mxn  |                        | */ \
    /* |--------|-------|-------|-------|------------------------| */ \
    /* | n = 1  |  mxm  |  mx1  |  mx1  |    TRSV                | */ \
    /* | m = 1  |  1x1  |  1xn  |  1xn  |    INVSCALS            | */ \
    /* |--------|-------|-------|-------|------------------------| */ \
    /* |--------|-------|-------|-------|------------------------| */ \
    /* |        |   X   |   A   |   B   |   Implementation       | */ \
    /* |--------|-------|-------|-------|------------------------| */ \
    /* | RIGHT  |  mxn  |  nxn  |  mxn  |                        | */ \
    /* |--------|-------|-------|-------|------------------------| */ \
    /* | n = 1  |  mx1  |  1x1  |  mx1  |  Transpose and INVSCALS| */ \
    /* | m = 1  |  1xn  |  nxn  |  1xn  |  Transpose and TRSV    | */ \
    /* |--------|-------|-------|-------|------------------------| */ \
    /*   If Transpose(A) uplo = lower then uplo = higher           */ \
    /*   If Transpose(A) uplo = higher then uplo = lower           */ \
    /* ----------------------------------------------------------- */ \
\
    if( n0 == 1 ) \
    { \
        if( blis_side == BLIS_LEFT ) \
        { \
            if(bli_is_notrans(blis_transa)) \
            { \
                PASTEMAC(ch, trsv_unf_var2) \
                ( \
                    blis_uploa, \
                    blis_transa, \
                    blis_diaga, \
                    m0, \
                    (ftype*)alpha, \
                    (ftype*)a, rs_a, cs_a, \
                    (ftype*)b, rs_b, \
                    NULL \
                ); \
                AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO)  \
                return; \
            } \
            else if(bli_is_trans(blis_transa)) \
            { \
                PASTEMAC(ch, trsv_unf_var1) \
                ( \
                    blis_uploa, \
                    blis_transa, \
                    blis_diaga, \
                    m0, \
                    (ftype*)alpha, \
                    (ftype*)a, rs_a, cs_a, \
                    (ftype*)b, rs_b, \
                    NULL \
                ); \
                AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO)  \
                return; \
            } \
        } \
        else if( ( blis_side == BLIS_RIGHT ) && ( m0 != 1 ) ) \
        { \
            /* b = alpha * b; */ \
            PASTEMAC2(ch,scalv,BLIS_TAPI_EX_SUF) \
            ( \
                conja, \
                m0, \
                (ftype*)alpha, \
                b, rs_b, \
                NULL, \
                NULL \
            ); \
            if(blis_diaga == BLIS_NONUNIT_DIAG) \
            { \
                conja = bli_extract_conj( blis_transa ); \
                PASTEMAC(ch,copycjs)( conja, *a, a_conj ); \
                for(int indx = 0; indx < m0; indx ++)  \
                { \
                    PASTEMAC(ch,invscals)( a_conj, b[indx] ); \
                } \
            }\
            AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO)  \
            return; \
        } \
    } \
    else if( m0 == 1 ) \
    { \
        if(blis_side == BLIS_RIGHT) \
        { \
            if(bli_is_notrans(blis_transa)) \
            { \
                if(blis_uploa == BLIS_UPPER) \
                    blis_uploa = BLIS_LOWER; \
                else \
                    blis_uploa = BLIS_UPPER; \
                PASTEMAC(ch, trsv_unf_var1)( \
                    blis_uploa, \
                    blis_transa, \
                    blis_diaga, \
                    n0, \
                    (ftype*)alpha, \
                    (ftype*)a, cs_a, rs_a, \
                    (ftype*)b, cs_b, \
                    NULL); \
                AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO)  \
                return; \
            } \
            else if(bli_is_trans(blis_transa)) \
            { \
                if(blis_uploa == BLIS_UPPER) \
                    blis_uploa = BLIS_LOWER; \
                else \
                    blis_uploa = BLIS_UPPER; \
                PASTEMAC(ch, trsv_unf_var2)( \
                    blis_uploa, \
                    blis_transa, \
                    blis_diaga, \
                    n0, \
                    (ftype*)alpha, \
                    (ftype*)a, cs_a, rs_a, \
                    (ftype*)b, cs_b, \
                    NULL); \
                AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO)  \
                return; \
            } \
        } \
        else if(( blis_side == BLIS_LEFT ) && ( n0 != 1 ))  \
        { \
            /* b = alpha * b; */ \
            PASTEMAC2(ch,scalv,BLIS_TAPI_EX_SUF) \
            ( \
                conja, \
                n0, \
                (ftype*)alpha,  \
                b, cs_b, \
                NULL, \
                NULL  \
            ); \
            if(blis_diaga == BLIS_NONUNIT_DIAG) \
            { \
                conja = bli_extract_conj( blis_transa ); \
                PASTEMAC(ch,copycjs)( conja, *a, a_conj ); \
                for(int indx = 0; indx < n0; indx ++) \
                { \
                    PASTEMAC(ch,invscals)( a_conj, b[indx*cs_b] ); \
                }\
            } \
            AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO)  \
            return; \
        } \
    } \
\
    const struc_t struca = BLIS_TRIANGULAR; \
\
    obj_t       alphao = BLIS_OBJECT_INITIALIZER_1X1; \
    obj_t       ao     = BLIS_OBJECT_INITIALIZER; \
    obj_t       bo     = BLIS_OBJECT_INITIALIZER; \
\
    dim_t       mn0_a; \
\
    bli_set_dim_with_side( blis_side, m0, n0, &mn0_a ); \
\
    bli_obj_init_finish_1x1( dt, (ftype*)alpha, &alphao ); \
\
    bli_obj_init_finish( dt, mn0_a, mn0_a, (ftype*)a, rs_a, cs_a, &ao ); \
    bli_obj_init_finish( dt, m0,    n0,    (ftype*)b, rs_b, cs_b, &bo ); \
\
    bli_obj_set_uplo( blis_uploa, &ao ); \
    bli_obj_set_diag( blis_diaga, &ao ); \
    bli_obj_set_conjtrans( blis_transa, &ao ); \
\
    bli_obj_set_struc( struca, &ao ); \
\
    PASTEMAC(blisname,BLIS_OAPI_EX_SUF) \
    ( \
      blis_side, \
      &alphao, \
      &ao, \
      &bo, \
      NULL, \
      NULL  \
    ); \
\
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO)  \
    /* Finalize BLIS. */ \
    bli_finalize_auto(); \
} \
void PASTEF77(ch,blasname) \
     ( \
       const f77_char* side, \
       const f77_char* uploa, \
       const f77_char* transa, \
       const f77_char* diaga, \
       const f77_int*  m, \
       const f77_int*  n, \
       const ftype*    alpha, \
       const ftype*    a, const f77_int* lda, \
             ftype*    b, const f77_int* ldb  \
     ) \
{ \
    PASTEF77S(ch,blasname) ( side, uploa, transa, diaga, m, n, alpha, a, lda, b, ldb ); \
} \

#endif


void strsm_blis_impl
(
    const f77_char* side,
    const f77_char* uploa,
    const f77_char* transa,
    const f77_char* diaga,
    const f77_int*  m,
    const f77_int*  n,
    const float*    alpha,
    const float*    a, const f77_int* lda,
    float*    b, const f77_int* ldb
)
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_INFO)
    AOCL_DTL_LOG_TRSM_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 's',
                             *side, *uploa,*transa, *diaga, *m, *n,
                            (void*)alpha,*lda, *ldb);

    side_t  blis_side;
    uplo_t  blis_uploa;
    trans_t blis_transa;
    diag_t  blis_diaga;
    dim_t   m0, n0;
    conj_t  conja = BLIS_NO_CONJUGATE ;

    /* Initialize BLIS. */
    bli_init_auto();

    /* Perform BLAS parameter checking. */
    PASTEBLACHK(trsm)
    (
      MKSTR(s),
      MKSTR(trsm),
      side,
      uploa,
      transa,
      diaga,
      m,
      n,
      lda,
      ldb
    );

    /* Quick return if possible. */
    if ( *m == 0 || *n == 0 )
    {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
        /* Finalize BLIS. */
        bli_finalize_auto();
        return;
    }

    /* Map BLAS chars to their corresponding BLIS enumerated type value. */
    bli_param_map_netlib_to_blis_side( *side,  &blis_side );
    bli_param_map_netlib_to_blis_uplo( *uploa, &blis_uploa );
    bli_param_map_netlib_to_blis_trans( *transa, &blis_transa );
    bli_param_map_netlib_to_blis_diag( *diaga, &blis_diaga );

    /* Typecast BLAS integers to BLIS integers. */
    bli_convert_blas_dim1( *m, m0 );
    bli_convert_blas_dim1( *n, n0 );

    /* Set the row and column strides of the matrix operands. */
    const inc_t rs_a = 1;
    const inc_t cs_a = *lda;
    const inc_t rs_b = 1;
    const inc_t cs_b = *ldb;
    const num_t dt = BLIS_FLOAT;

	/* If alpha is zero, set B to zero and return early */
	if( PASTEMAC(s,eq0)( *alpha ) )
	{
		PASTEMAC2(s,setm,_ex)( BLIS_NO_CONJUGATE,
								0,
								BLIS_NONUNIT_DIAG,
								BLIS_DENSE,
								m0, n0,
								(float*) alpha,
								(float*) b, rs_b, cs_b,
								NULL, NULL
							  );
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)
		/* Finalize BLIS. */
		bli_finalize_auto();
		return;
	}

    if( n0 == 1 )
    {
        if( blis_side == BLIS_LEFT )
        {
            if(bli_is_notrans(blis_transa))
            {
                bli_strsv_unf_var2
                (
                    blis_uploa,
                    blis_transa,
                    blis_diaga,
                    m0,
                    (float*)alpha,
                    (float*)a, rs_a, cs_a,
                    (float*)b, rs_b,
                    NULL
                );
                AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
                return;
            }
            else if(bli_is_trans(blis_transa))
            {
                bli_strsv_unf_var1
                (
                    blis_uploa,
                    blis_transa,
                    blis_diaga,
                    m0,
                    (float*)alpha,
                    (float*)a, rs_a, cs_a,
                    (float*)b, rs_b,
                    NULL
                );
                AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
                return;
            }
        }
        else if( ( blis_side == BLIS_RIGHT ) && ( m0 != 1 ) )
        {
            /* b = alpha * b; */
            bli_sscalv_ex
            (
                conja,
                m0,
                (float*)alpha,
                b, rs_b,
                NULL,
                NULL
            );
            if(blis_diaga == BLIS_NONUNIT_DIAG)
            {
                float inva = 1.0/ *a;
                for(dim_t indx = 0; indx < m0; indx ++)
                {
                    b[indx] = ( inva * b[indx] );
                }
            }
            AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
            return;
        }
    }
    else if( m0 == 1 )
    {
        if(blis_side == BLIS_RIGHT)
        {
            if(bli_is_notrans(blis_transa))
            {
                if(blis_uploa == BLIS_UPPER)
                    blis_uploa = BLIS_LOWER;
                else
                    blis_uploa = BLIS_UPPER;

                bli_strsv_unf_var1
                (
                    blis_uploa,
                    blis_transa,
                    blis_diaga,
                    n0,
                    (float*)alpha,
                    (float*)a, cs_a, rs_a,
                    (float*)b, cs_b,
                    NULL
                );
                AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
                return;
            }
            else if(bli_is_trans(blis_transa))
            {
                if(blis_uploa == BLIS_UPPER)
                    blis_uploa = BLIS_LOWER;
                else
                    blis_uploa = BLIS_UPPER;

                bli_strsv_unf_var2
                (
                    blis_uploa,
                    blis_transa,
                    blis_diaga,
                    n0,
                    (float*)alpha,
                    (float*)a, cs_a, rs_a,
                    (float*)b, cs_b,
                    NULL
                );
                AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
                return;
            }
        }
        else if(( blis_side == BLIS_LEFT ) && ( n0 != 1 ))
        {
            /* b = alpha * b; */
            bli_sscalv_ex
            (
                conja,
                n0,
                (float*)alpha,
                b, cs_b,
                NULL,
                NULL
            );
            if(blis_diaga == BLIS_NONUNIT_DIAG)
            {
                float inva = 1.0/ *a;
                for(dim_t indx = 0; indx < n0; indx ++)
                {
                    b[indx*cs_b] = (inva * b[indx*cs_b] );
                }
            }
            AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
            return;
        }
    }
    const struc_t struca = BLIS_TRIANGULAR;

    obj_t       alphao = BLIS_OBJECT_INITIALIZER_1X1;
    obj_t       ao     = BLIS_OBJECT_INITIALIZER;
    obj_t       bo     = BLIS_OBJECT_INITIALIZER;

    dim_t       mn0_a;

    bli_set_dim_with_side( blis_side, m0, n0, &mn0_a );

    bli_obj_init_finish_1x1( dt, (float*)alpha, &alphao );

    bli_obj_init_finish( dt, mn0_a, mn0_a, (float*)a, rs_a, cs_a, &ao );
    bli_obj_init_finish( dt, m0,    n0,    (float*)b, rs_b, cs_b, &bo );

    bli_obj_set_uplo( blis_uploa, &ao );
    bli_obj_set_diag( blis_diaga, &ao );
    bli_obj_set_conjtrans( blis_transa, &ao );

    bli_obj_set_struc( struca, &ao );

#ifdef BLIS_ENABLE_SMALL_MATRIX_TRSM
    // This function is invoked on all architectures including 'generic'.
    // Non-AVX2+FMA3 platforms will use the kernels derived from the context.
    if (bli_cpuid_is_avx2fma3_supported() == TRUE)
    {
	    /* bli_strsm_small is performing better existing native
	     * implementations for [m,n]<=1000 for single thread.
	     * In case of multithread when [m,n]<=128 single thread implementation
	     * is doing better than native multithread */
	    bool is_parallel = bli_thread_get_is_parallel();
	    if((!is_parallel && m0<=1000 && n0<=1000) ||
               (is_parallel && (m0+n0)<320))
	    {
		    err_t status;
		    status = bli_trsm_small
                             (
                               blis_side,
			       &alphao,
			       &ao,
			       &bo,
			       NULL,
			       NULL,
			       is_parallel
			     );
		    if (status == BLIS_SUCCESS)
		    {
			    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
			    /* Finalize BLIS. */
			    bli_finalize_auto();
			    return;
		    }
	    }
    }
#endif

    bli_trsmnat
    (
        blis_side,
        &alphao,
        &ao,
        &bo,
        NULL,
        NULL
    );

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO)
    /* Finalize BLIS. */
    bli_finalize_auto();
}
#ifdef BLIS_ENABLE_BLAS
void strsm_
(
    const f77_char* side,
    const f77_char* uploa,
    const f77_char* transa,
    const f77_char* diaga,
    const f77_int*  m,
    const f77_int*  n,
    const float*    alpha,
    const float*    a, const f77_int* lda,
    float*    b, const f77_int* ldb
)
{
    strsm_blis_impl ( side, uploa, transa, diaga, m, n, alpha, a, lda, b, ldb );
}
#endif
void dtrsm_blis_impl
(
    const f77_char* side,
    const f77_char* uploa,
    const f77_char* transa,
    const f77_char* diaga,
    const f77_int*  m,
    const f77_int*  n,
    const double*    alpha,
    const double*    a, const f77_int* lda,
    double*    b, const f77_int* ldb
)
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_INFO)
    AOCL_DTL_LOG_TRSM_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'd',
                             *side, *uploa,*transa, *diaga, *m, *n,
                            (void*)alpha,*lda, *ldb);

    side_t  blis_side;
    uplo_t  blis_uploa;
    trans_t blis_transa;
    diag_t  blis_diaga;
    dim_t   m0, n0;
    conj_t  conja = BLIS_NO_CONJUGATE ;

    /* Initialize BLIS. */
    bli_init_auto();

    /* Perform BLAS parameter checking. */
    PASTEBLACHK(trsm)
    (
      MKSTR(d),
      MKSTR(trsm),
      side,
      uploa,
      transa,
      diaga,
      m,
      n,
      lda,
      ldb
    );

    /* Quick return if possible. */
    if ( *m == 0 || *n == 0 )
    {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
        /* Finalize BLIS. */
        bli_finalize_auto();
        return;
    }

    /* Map BLAS chars to their corresponding BLIS enumerated type value. */
    bli_param_map_netlib_to_blis_side( *side,  &blis_side );
    bli_param_map_netlib_to_blis_uplo( *uploa, &blis_uploa );
    bli_param_map_netlib_to_blis_trans( *transa, &blis_transa );
    bli_param_map_netlib_to_blis_diag( *diaga, &blis_diaga );

    /* Typecast BLAS integers to BLIS integers. */
    bli_convert_blas_dim1( *m, m0 );
    bli_convert_blas_dim1( *n, n0 );

    /* Set the row and column strides of the matrix operands. */
    const inc_t rs_a = 1;
    const inc_t cs_a = *lda;
    const inc_t rs_b = 1;
    const inc_t cs_b = *ldb;
    const num_t dt = BLIS_DOUBLE;

	/* If alpha is zero, set B to zero and return early */
	if( PASTEMAC(d,eq0)( *alpha ) )
	{
		PASTEMAC2(d,setm,_ex)( BLIS_NO_CONJUGATE,
								0,
								BLIS_NONUNIT_DIAG,
								BLIS_DENSE,
								m0, n0,
								(double*) alpha,
								(double*) b, rs_b, cs_b,
								NULL, NULL
							  );
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)
		/* Finalize BLIS. */
		bli_finalize_auto();
		return;
	}

    if( n0 == 1 )
    {
        if( blis_side == BLIS_LEFT )
        {
            if(bli_is_notrans(blis_transa))
            {
                bli_dtrsv_unf_var2
                (
                    blis_uploa,
                    blis_transa,
                    blis_diaga,
                    m0,
                    (double*)alpha,
                    (double*)a, rs_a, cs_a,
                    (double*)b, rs_b,
                    NULL
                );
                AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
                return;
            }
            else if(bli_is_trans(blis_transa))
            {
                bli_dtrsv_unf_var1
                (
                    blis_uploa,
                    blis_transa,
                    blis_diaga,
                    m0,
                    (double*)alpha,
                    (double*)a, rs_a, cs_a,
                    (double*)b, rs_b,
                    NULL
                );
                AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
                return;
            }
        }
        else if( ( blis_side == BLIS_RIGHT ) && ( m0 != 1 ) )
        {
            /* b = alpha * b; */
            bli_dscalv_ex
            (
                conja,
                m0,
                (double*)alpha,
                b, rs_b,
                NULL,
                NULL
            );
            if(blis_diaga == BLIS_NONUNIT_DIAG)
            {
                double inva = 1.0/ *a;
                for(dim_t indx = 0; indx < m0; indx ++)
                {
                    b[indx] = ( inva * b[indx] );
                }
            }
            AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
            return;
        }
    }
    else if( m0 == 1 )
    {
        if(blis_side == BLIS_RIGHT)
        {
            if(bli_is_notrans(blis_transa))
            {
                if(blis_uploa == BLIS_UPPER)
                    blis_uploa = BLIS_LOWER;
                else
                    blis_uploa = BLIS_UPPER;

                bli_dtrsv_unf_var1
                (
                    blis_uploa,
                    blis_transa,
                    blis_diaga,
                    n0,
                    (double*)alpha,
                    (double*)a, cs_a, rs_a,
                    (double*)b, cs_b,
                    NULL
                );
                AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
                return;
            }
            else if(bli_is_trans(blis_transa))
            {
                if(blis_uploa == BLIS_UPPER)
                    blis_uploa = BLIS_LOWER;
                else
                    blis_uploa = BLIS_UPPER;

                bli_dtrsv_unf_var2
                (
                    blis_uploa,
                    blis_transa,
                    blis_diaga,
                    n0,
                    (double*)alpha,
                    (double*)a, cs_a, rs_a,
                    (double*)b, cs_b,
                    NULL
                );
                AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
                return;
            }
        }
        else if(( blis_side == BLIS_LEFT ) && ( n0 != 1 ))
        {
            /* b = alpha * b; */
            bli_dscalv_ex
            (
                conja,
                n0,
                (double*)alpha,
                b, cs_b,
                NULL,
                NULL
            );
            if(blis_diaga == BLIS_NONUNIT_DIAG)
            {
                double inva = 1.0/ *a;
                for(dim_t indx = 0; indx < n0; indx ++)
                {
                    b[indx*cs_b] = (inva * b[indx*cs_b] );
                }
            }
            AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
            return;
        }
    }

    const struc_t struca = BLIS_TRIANGULAR;

    obj_t       alphao = BLIS_OBJECT_INITIALIZER_1X1;
    obj_t       ao     = BLIS_OBJECT_INITIALIZER;
    obj_t       bo     = BLIS_OBJECT_INITIALIZER;

    dim_t       mn0_a;

    bli_set_dim_with_side( blis_side, m0, n0, &mn0_a );

    bli_obj_init_finish_1x1( dt, (double*)alpha, &alphao );

    bli_obj_init_finish( dt, mn0_a, mn0_a, (double*)a, rs_a, cs_a, &ao );
    bli_obj_init_finish( dt, m0,    n0,    (double*)b, rs_b, cs_b, &bo );

    bli_obj_set_uplo( blis_uploa, &ao );
    bli_obj_set_diag( blis_diaga, &ao );
    bli_obj_set_conjtrans( blis_transa, &ao );

    bli_obj_set_struc( struca, &ao );

#ifdef BLIS_ENABLE_SMALL_MATRIX_TRSM

    // This function is invoked on all architectures including 'generic'.
    // Non-AVX2+FMA3 platforms will use the kernels derived from the context.
    if (bli_cpuid_is_avx2fma3_supported() == TRUE)
    {
        // typedef for trsm small kernel function pointer
        typedef err_t (*dtrsm_small_ker_ft)
            (
              side_t   side,
              obj_t*   alpha,
              obj_t*   a,
              obj_t*   b,
              cntx_t*  cntx,
              cntl_t*  cntl,
              bool     is_parallel
            );
        err_t status = BLIS_NOT_YET_IMPLEMENTED;
        
        // trsm small kernel function pointer definition
        dtrsm_small_ker_ft ker_ft = NULL;

        // Query the architecture ID
        arch_t id = bli_arch_query_id();

        // dimensions of triangular matrix
        // for left variants, dim_a is m0,
        // for right variants, dim_a is n0
        dim_t dim_a = n0;
        if (blis_side == BLIS_LEFT)
            dim_a = m0;

        // size of output matrix(B)
        dim_t size_b = m0*n0;

        /* bli_dtrsm_small is performing better than existing native
         * implementations for dim_a<1500 and m0*n0<5e6 for single thread.
         * In case of multithread when [m+n]<320 single thread implementation
         * is doing better than small multithread and native multithread */
        bool is_parallel = bli_thread_get_is_parallel();
        if ((!is_parallel && ((dim_a < 1500) && (size_b < 5e6)) ) ||
            (is_parallel && (m0+n0)<320))
        {
            switch(id)
            {
                case BLIS_ARCH_ZEN4:
#if defined(BLIS_KERNELS_ZEN4)
                    /* For sizes where m and n < 50,avx2 kernels are performing better,
                     except for sizes where n is multiple of 8.*/
                    if (((n0 % 8 == 0) && (n0 < 50)) || ((m0 > 50) && (n0 > 50)))
                    {
                        ker_ft = bli_trsm_small_AVX512;
                    }
                    else
                    {
                        ker_ft = bli_trsm_small;
                    }
                    break;
#endif // BLIS_KERNELS_ZEN4
                case BLIS_ARCH_ZEN:
                case BLIS_ARCH_ZEN2:
                case BLIS_ARCH_ZEN3:
                default:
                    ker_ft = bli_trsm_small;
                    break;
            }
        }

#ifdef BLIS_ENABLE_OPENMP
        if( (ker_ft == NULL) && (is_parallel) &&
          ((dim_a < 2500) && (size_b < 5e6)) )
        {
            switch(id)
            {
                case BLIS_ARCH_ZEN4:
#if defined(BLIS_KERNELS_ZEN4)
                    ker_ft = bli_trsm_small_mt_AVX512;
                    break;
#endif// BLIS_KERNELS_ZEN4
                case BLIS_ARCH_ZEN:
                case BLIS_ARCH_ZEN2:
                case BLIS_ARCH_ZEN3:
                default:
                    ker_ft = bli_trsm_small_mt;
                    break;
            }
        }

#endif// BLIS_ENABLE_OPENMP
        if(ker_ft)
        {
            status = ker_ft(blis_side, &alphao, &ao, &bo, NULL, NULL, is_parallel);
        }
        if (status == BLIS_SUCCESS)
        {
            AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
            /* Finalize BLIS. */
            bli_finalize_auto();
            return;
        }
    } // bli_cpuid_is_avx2fma3_supported
#endif// END of BLIS_ENABLE_SMALL_MATRIX_TRSM

    bli_trsmnat
    (
        blis_side,
        &alphao,
        &ao,
        &bo,
        NULL,
        NULL
    );

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO)
    /* Finalize BLIS. */
    bli_finalize_auto();
}
#ifdef BLIS_ENABLE_BLAS
void dtrsm_
(
    const f77_char* side,
    const f77_char* uploa,
    const f77_char* transa,
    const f77_char* diaga,
    const f77_int*  m,
    const f77_int*  n,
    const double*    alpha,
    const double*    a, const f77_int* lda,
    double*    b, const f77_int* ldb
)
{
    dtrsm_blis_impl ( side, uploa, transa, diaga, m, n, alpha, a, lda, b, ldb );
}
#endif

void ztrsm_blis_impl
(
    const f77_char* side,
    const f77_char* uploa,
    const f77_char* transa,
    const f77_char* diaga,
    const f77_int*  m,
    const f77_int*  n,
    const dcomplex*    alpha,
    const dcomplex*    a, const f77_int* lda,
    dcomplex*    b, const f77_int* ldb
)
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_INFO)
    AOCL_DTL_LOG_TRSM_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'z',
                             *side, *uploa,*transa, *diaga, *m, *n,
                            (void*)alpha,*lda, *ldb);

    side_t  blis_side;
    uplo_t  blis_uploa;
    trans_t blis_transa;
    diag_t  blis_diaga;
    dim_t   m0, n0;
    conj_t  conja = BLIS_NO_CONJUGATE;

    /* Initialize BLIS. */
    bli_init_auto();

    /* Perform BLAS parameter checking. */
    PASTEBLACHK(trsm)
    (
      MKSTR(z),
      MKSTR(trsm),
      side,
      uploa,
      transa,
      diaga,
      m,
      n,
      lda,
      ldb
    );

    /* Quick return if possible. */
    if ( *m == 0 || *n == 0 )
    {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
        /* Finalize BLIS. */
        bli_finalize_auto();
        return;
    }

    /* Map BLAS chars to their corresponding BLIS enumerated type value. */
    bli_param_map_netlib_to_blis_side( *side,  &blis_side );
    bli_param_map_netlib_to_blis_uplo( *uploa, &blis_uploa );
    bli_param_map_netlib_to_blis_trans( *transa, &blis_transa );
    bli_param_map_netlib_to_blis_diag( *diaga, &blis_diaga );

    /* Typecast BLAS integers to BLIS integers. */
    bli_convert_blas_dim1( *m, m0 );
    bli_convert_blas_dim1( *n, n0 );

    /* Set the row and column strides of the matrix operands. */
    const inc_t rs_a = 1;
    const inc_t cs_a = *lda;
    const inc_t rs_b = 1;
    const inc_t cs_b = *ldb;
    const num_t dt = BLIS_DCOMPLEX;

	/* If alpha is zero, set B to zero and return early */
	if( PASTEMAC(z,eq0)( *alpha ) )
	{
		PASTEMAC2(z,setm,_ex)( BLIS_NO_CONJUGATE,
								0,
								BLIS_NONUNIT_DIAG,
								BLIS_DENSE,
								m0, n0,
								(dcomplex*) alpha,
								(dcomplex*) b, rs_b, cs_b,
								NULL, NULL
							  );
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)
		/* Finalize BLIS. */
		bli_finalize_auto();
		return;
	}

    if( n0 == 1 )
    {
        if( blis_side == BLIS_LEFT )
        {
            if(bli_is_notrans(blis_transa))
            {
                bli_ztrsv_unf_var2
                (
                    blis_uploa,
                    blis_transa,
                    blis_diaga,
                    m0,
                    (dcomplex*)alpha,
                    (dcomplex*)a, rs_a, cs_a,
                    (dcomplex*)b, rs_b,
                    NULL
                );
                AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
                return;
            }
            else if(bli_is_trans(blis_transa))
            {
                bli_ztrsv_unf_var1
                (
                    blis_uploa,
                    blis_transa,
                    blis_diaga,
                    m0,
                    (dcomplex*)alpha,
                    (dcomplex*)a, rs_a, cs_a,
                    (dcomplex*)b, rs_b,
                    NULL
                );
                AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
                return;
            }
        }
        else if( ( blis_side == BLIS_RIGHT ) && ( m0 != 1 ) )
        {
            bli_zscalv_ex
            (
                conja,
                m0,
                (dcomplex*)alpha,
                (dcomplex*)b, rs_b,
                NULL,
                NULL
            );
	    if(blis_diaga == BLIS_NONUNIT_DIAG)
	    {
		    dcomplex inva = {1.0, 0.0};
		    dcomplex a_dup;
		    /**
		     * For conjugate transpose and non-unit diagonal
		     * kernel, negating imaginary part of A.
		     * As the dimension of A is 1x1, there's going to
		     * be only one 1 element of A.
		     */
		    if(*transa == 'C' && *diaga == 'N')
		    {
			    a_dup.real = a->real;
			    a_dup.imag = a->imag * -1.0;
		    }
		    else
		    {
			    a_dup.real = a->real;
			    a_dup.imag = a->imag;
		    }

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
		    bli_zinvscals(a_dup, inva);
#else
		    inva.real = a_dup.real;
		    inva.imag = a_dup.imag;
#endif
		    for(dim_t indx = 0; indx < m0; indx ++)
		    {
#ifdef BLIS_ENABLE_TRSM_PREINVERSION
			    bli_zscals(inva, b[indx])
#else

			    bli_zinvscals(inva, b[indx])
#endif
		    }

	    }

	    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
	    return;
	}
    }
    else if( m0 == 1 )
    {
        if(blis_side == BLIS_RIGHT)
        {
            if(bli_is_notrans(blis_transa))
            {
                if(blis_uploa == BLIS_UPPER)
                    blis_uploa = BLIS_LOWER;
                else
                    blis_uploa = BLIS_UPPER;

                bli_ztrsv_unf_var1
                (
                    blis_uploa,
                    blis_transa,
                    blis_diaga,
                    n0,
                    (dcomplex*)alpha,
                    (dcomplex*)a, cs_a, rs_a,
                    (dcomplex*)b, cs_b,
                    NULL
                );
                AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
                return;
            }
            else if(bli_is_trans(blis_transa))
            {
                if(blis_uploa == BLIS_UPPER)
                    blis_uploa = BLIS_LOWER;
                else
                    blis_uploa = BLIS_UPPER;

                bli_ztrsv_unf_var2
                (
                    blis_uploa,
                    blis_transa,
                    blis_diaga,
                    n0,
                    (dcomplex*)alpha,
                    (dcomplex*)a, cs_a, rs_a,
                    (dcomplex*)b, cs_b,
                    NULL
                );
                AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
                return;
            }
        }
        else if(( blis_side == BLIS_LEFT ) && ( n0 != 1 ))
        {
            bli_zscalv_ex
            (
                conja,
                n0,
                (dcomplex*)alpha,
                (dcomplex*)b, cs_b,
                NULL,
                NULL
            );
            if(blis_diaga == BLIS_NONUNIT_DIAG)
            {
                dcomplex inva = {1.0, 0.0};
		dcomplex a_dup;
		/**
		 * For conjugate transpose and non-unit diagonal
		 * kernel, negating imaginary part of A.
		 * As the dimension of A is 1x1, there's going to
		 * be only one 1 element of A.
		 */
                if(*transa == 'C' && *diaga == 'N')
                {
                        a_dup.real = a->real;
                        a_dup.imag = a->imag * -1.0;
                }
                else
                {
			a_dup.real = a->real;
                        a_dup.imag = a->imag;
                }

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
		bli_zinvscals(a_dup, inva);
#else
		inva.real = a_dup.real;
		inva.imag = a_dup.imag;
#endif
                for(dim_t indx = 0; indx < n0; indx ++)
                {
#ifdef BLIS_ENABLE_TRSM_PREINVERSION
			bli_zscals(inva ,b[indx * cs_b])
#else

			bli_zinvscals(inva ,b[indx * cs_b])
#endif
                }
            }

	    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
	    return;

        }
    }

    const struc_t struca = BLIS_TRIANGULAR;

    obj_t       alphao = BLIS_OBJECT_INITIALIZER_1X1;
    obj_t       ao     = BLIS_OBJECT_INITIALIZER;
    obj_t       bo     = BLIS_OBJECT_INITIALIZER;

    dim_t       mn0_a;

    bli_set_dim_with_side( blis_side, m0, n0, &mn0_a );

    bli_obj_init_finish_1x1( dt, (dcomplex*)alpha, &alphao );

    bli_obj_init_finish( dt, mn0_a, mn0_a, (dcomplex*)a, rs_a, cs_a, &ao );
    bli_obj_init_finish( dt, m0,    n0,    (dcomplex*)b, rs_b, cs_b, &bo );

    bli_obj_set_uplo( blis_uploa, &ao );
    bli_obj_set_diag( blis_diaga, &ao );
    bli_obj_set_conjtrans( blis_transa, &ao );

    bli_obj_set_struc( struca, &ao );

#ifdef BLIS_ENABLE_SMALL_MATRIX_TRSM
    // This function is invoked on all architectures including 'generic'.
    // Non-AVX2+FMA3 platforms will use the kernels derived from the context.
    if (bli_cpuid_is_avx2fma3_supported() == TRUE)
    {
        /* bli_ztrsm_small is performing better existing native
        * implementations for [m,n]<=1000 for single thread.
        * In case of multithread when [m,n]<=128 single thread implementation
        * is doing better than native multithread */
        bool is_parallel = bli_thread_get_is_parallel();

        if((!is_parallel && m0<=500 && n0<=500) ||
           (is_parallel && (m0+n0)<128))
        {
            err_t status;
            status = bli_trsm_small
                    (
                        blis_side,
                        &alphao,
                        &ao,
                        &bo,
                        NULL,
                        NULL,
                        is_parallel
                    );
            if (status == BLIS_SUCCESS)
            {
                AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
                /* Finalize BLIS. */
                bli_finalize_auto();
                return;
            }
        }
    } // bli_cpuid_is_avx2fma3_supported
#endif// END of BLIS_ENABLE_SMALL_MATRIX_TRSM

    bli_trsmnat
    (
        blis_side,
        &alphao,
        &ao,
        &bo,
        NULL,
        NULL
    );

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO)
    /* Finalize BLIS. */
    bli_finalize_auto();
}
#ifdef BLIS_ENABLE_BLAS
void ztrsm_
(
    const f77_char* side,
    const f77_char* uploa,
    const f77_char* transa,
    const f77_char* diaga,
    const f77_int*  m,
    const f77_int*  n,
    const dcomplex*    alpha,
    const dcomplex*    a, const f77_int* lda,
    dcomplex*    b, const f77_int* ldb
)
{
    ztrsm_blis_impl ( side, uploa, transa, diaga, m, n, alpha, a, lda, b, ldb );
}
#endif

void ctrsm_blis_impl
(
    const f77_char* side,
    const f77_char* uploa,
    const f77_char* transa,
    const f77_char* diaga,
    const f77_int*  m,
    const f77_int*  n,
    const scomplex*    alpha,
    const scomplex*    a, const f77_int* lda,
    scomplex*    b, const f77_int* ldb
)
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_INFO)
    AOCL_DTL_LOG_TRSM_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'c',
                             *side, *uploa,*transa, *diaga, *m, *n,
                            (void*)alpha,*lda, *ldb);

    side_t  blis_side;
    uplo_t  blis_uploa;
    trans_t blis_transa;
    diag_t  blis_diaga;
    dim_t   m0, n0;
    conj_t  conja = BLIS_NO_CONJUGATE;

    /* Initialize BLIS. */
    bli_init_auto();

    /* Perform BLAS parameter checking. */
    PASTEBLACHK(trsm)
    (
      MKSTR(c),
      MKSTR(trsm),
      side,
      uploa,
      transa,
      diaga,
      m,
      n,
      lda,
      ldb
    );

    /* Quick return if possible. */
    if ( *m == 0 || *n == 0 )
    {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
        /* Finalize BLIS. */
        bli_finalize_auto();
        return;
    }

    /* Map BLAS chars to their corresponding BLIS enumerated type value. */
    bli_param_map_netlib_to_blis_side( *side,  &blis_side );
    bli_param_map_netlib_to_blis_uplo( *uploa, &blis_uploa );
    bli_param_map_netlib_to_blis_trans( *transa, &blis_transa );
    bli_param_map_netlib_to_blis_diag( *diaga, &blis_diaga );

    /* Typecast BLAS integers to BLIS integers. */
    bli_convert_blas_dim1( *m, m0 );
    bli_convert_blas_dim1( *n, n0 );

    /* Set the row and column strides of the matrix operands. */
    const inc_t rs_a = 1;
    const inc_t cs_a = *lda;
    const inc_t rs_b = 1;
    const inc_t cs_b = *ldb;
    const num_t dt = BLIS_SCOMPLEX;

	/* If alpha is zero, set B to zero and return early */
	if( PASTEMAC(c,eq0)( *alpha ) )
	{
		PASTEMAC2(c,setm,_ex)( BLIS_NO_CONJUGATE,
								0,
								BLIS_NONUNIT_DIAG,
								BLIS_DENSE,
								m0, n0,
								(scomplex*) alpha,
								(scomplex*) b, rs_b, cs_b,
								NULL, NULL
							  );
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)
		/* Finalize BLIS. */
		bli_finalize_auto();
		return;
	}

    if( n0 == 1 )
    {
        if( blis_side == BLIS_LEFT )
        {
            if(bli_is_notrans(blis_transa))
            {
                bli_ctrsv_unf_var2
                (
                    blis_uploa,
                    blis_transa,
                    blis_diaga,
                    m0,
                    (scomplex*)alpha,
                    (scomplex*)a, rs_a, cs_a,
                    (scomplex*)b, rs_b,
                    NULL
                );
                AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
                return;
            }
            else if(bli_is_trans(blis_transa))
            {
                bli_ctrsv_unf_var1
                (
                    blis_uploa,
                    blis_transa,
                    blis_diaga,
                    m0,
                    (scomplex*)alpha,
                    (scomplex*)a, rs_a, cs_a,
                    (scomplex*)b, rs_b,
                    NULL
                );
                AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
                return;
            }
        }
        else if( ( blis_side == BLIS_RIGHT ) && ( m0 != 1 ) )
        {
            bli_cscalv_ex
            (
                conja,
                m0,
                (scomplex*)alpha,
                (scomplex*)b, rs_b,
                NULL,
                NULL
            );
            if(blis_diaga == BLIS_NONUNIT_DIAG)
            {
                scomplex inva = {1.0, 0.0};
		scomplex a_dup;
		/**
		 * For conjugate transpose and non-unit diagonal
		 * kernel, negating imaginary part of A.
		 * As the dimension of A is 1x1, there's going to
		 * be only one 1 element of A.
		 */
                if(*transa == 'C' && *diaga == 'N')
                {
                        a_dup.real = a->real;
                        a_dup.imag = a->imag * -1.0;
                }
                else
                {
			a_dup.real = a->real;
                        a_dup.imag = a->imag;
                }

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
		bli_cinvscals(a_dup, inva);
#else
		inva.real = a_dup.real;
		inva.imag = a_dup.imag;
#endif

                for(dim_t indx = 0; indx < m0; indx ++)
                {
#ifdef BLIS_ENABLE_TRSM_PREINVERSION
			bli_cscals(inva ,b[indx])
#else
			bli_cinvscals(inva, b[indx])
#endif
                }
            }

	    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
	    return;

        }
    }
    else if( m0 == 1 )
    {
        if(blis_side == BLIS_RIGHT)
        {
            if(bli_is_notrans(blis_transa))
            {
                if(blis_uploa == BLIS_UPPER)
                    blis_uploa = BLIS_LOWER;
                else
                    blis_uploa = BLIS_UPPER;

                bli_ctrsv_unf_var1
                (
                    blis_uploa,
                    blis_transa,
                    blis_diaga,
                    n0,
                    (scomplex*)alpha,
                    (scomplex*)a, cs_a, rs_a,
                    (scomplex*)b, cs_b,
                    NULL
                );
                AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
                return;
            }
            else if(bli_is_trans(blis_transa))
            {
                if(blis_uploa == BLIS_UPPER)
                    blis_uploa = BLIS_LOWER;
                else
                    blis_uploa = BLIS_UPPER;

                bli_ctrsv_unf_var2
                (
                    blis_uploa,
                    blis_transa,
                    blis_diaga,
                    n0,
                    (scomplex*)alpha,
                    (scomplex*)a, cs_a, rs_a,
                    (scomplex*)b, cs_b,
                    NULL
                );
                AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
                return;
            }
        }
        else if(( blis_side == BLIS_LEFT ) && ( n0 != 1 ))
        {
            bli_cscalv_ex
            (
                conja,
                n0,
                (scomplex*)alpha,
                (scomplex*)b, cs_b,
                NULL,
                NULL
            );
            if(blis_diaga == BLIS_NONUNIT_DIAG)
            {
                scomplex inva = {1.0, 0.0};
		scomplex a_dup;
		/**
		 * For conjugate transpose and non-unit diagonal
		 * kernel, negating imaginary part of A.
		 * As the dimension of A is 1x1, there's going to
		 * be only one 1 element of A.
		 */
                if(*transa == 'C' && *diaga == 'N')
                {
                        a_dup.real = a->real;
                        a_dup.imag = a->imag * -1.0;
                }
                else
                {
			a_dup.real = a->real;
                        a_dup.imag = a->imag;
                }

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
		bli_cinvscals(a_dup, inva)
#else
		inva.real = a_dup.real;
		inva.imag = a_dup.imag;
#endif
                for(dim_t indx = 0; indx < n0; indx ++)
                {
#ifdef BLIS_ENABLE_TRSM_PREINVERSION
			bli_cscals(inva ,b[indx * cs_b])
#else
			bli_cinvscals(inva, b[indx * cs_b])
#endif

                }
            }

	    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
	    return;
        }
    }

    const struc_t struca = BLIS_TRIANGULAR;

    obj_t       alphao = BLIS_OBJECT_INITIALIZER_1X1;
    obj_t       ao     = BLIS_OBJECT_INITIALIZER;
    obj_t       bo     = BLIS_OBJECT_INITIALIZER;

    dim_t       mn0_a;

    bli_set_dim_with_side( blis_side, m0, n0, &mn0_a );

    bli_obj_init_finish_1x1( dt, (scomplex*)alpha, &alphao );

    bli_obj_init_finish( dt, mn0_a, mn0_a, (scomplex*)a, rs_a, cs_a, &ao );
    bli_obj_init_finish( dt, m0,    n0,    (scomplex*)b, rs_b, cs_b, &bo );

    bli_obj_set_uplo( blis_uploa, &ao );
    bli_obj_set_diag( blis_diaga, &ao );
    bli_obj_set_conjtrans( blis_transa, &ao );

    bli_obj_set_struc( struca, &ao );

#ifdef BLIS_ENABLE_SMALL_MATRIX_TRSM
    // This function is invoked on all architectures including 'generic'.
    // Non-AVX2+FMA3 platforms will use the kernels derived from the context.
    if (bli_cpuid_is_avx2fma3_supported() == TRUE)
    {
        /* bli_ztrsm_small is performing better existing native
        * implementations for [m,n]<=1000 for single thread.
        * In case of multithread when [m,n]<=128 single thread implementation
        * is doing better than native multithread */
        bool is_parallel = bli_thread_get_is_parallel();
        if((!is_parallel && m0<=1000 && n0<=1000) ||
           (is_parallel && (m0+n0)<320))
        {
            err_t status;
            status = bli_trsm_small
                    (
                        blis_side,
                        &alphao,
                        &ao,
                        &bo,
                        NULL,
                        NULL,
                        is_parallel
                    );
            if (status == BLIS_SUCCESS)
            {
                AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
                /* Finalize BLIS. */
                bli_finalize_auto();
                return;
            }
        }
    } // bli_cpuid_is_avx2fma3_supported
#endif

    bli_trsmnat
    (
        blis_side,
        &alphao,
        &ao,
        &bo,
        NULL,
        NULL
    );

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO)
    /* Finalize BLIS. */
    bli_finalize_auto();
}
#ifdef BLIS_ENABLE_BLAS
void ctrsm_
(
    const f77_char* side,
    const f77_char* uploa,
    const f77_char* transa,
    const f77_char* diaga,
    const f77_int*  m,
    const f77_int*  n,
    const scomplex*    alpha,
    const scomplex*    a, const f77_int* lda,
    scomplex*    b, const f77_int* ldb
)
{
    ctrsm_blis_impl ( side, uploa, transa, diaga, m, n, alpha, a, lda, b, ldb );
}

#endif
