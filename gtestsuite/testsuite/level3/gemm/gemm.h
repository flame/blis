#pragma once

#include "blis.h"
#include "common/testing_helpers.h"

/**
 * @brief Performs the operation:
 *        C := alpha*op( A )*op( B ) + beta*C,
 * where  op( A ) is one of
 *        op( A ) = A   or   op( A ) = A**T   or   op( A ) = A**H,
 * @param[in]     transa specifies the form of op( A ) to be used in
                         the matrix multiplication
 * @param[in]     transb specifies the form of op( B ) to be used in
                         the matrix multiplication
 * @param[in]     m      specifies  the number  of rows  of the  matrix
                         op( A )  and of the  matrix  C
 * @param[in]     n      specifies the number  of columns of the matrix
                         op( B ) and the number of columns of the matrix C
 * @param[in]     k      specifies  the number of columns of the matrix
                         op( A ) and the number of rows of the matrix op( B ).
 * @param[in]     alpha  specifies the scalar alpha.
 * @param[in]     ap     specifies pointer which points to the first element of ap
 * @param[in]     rsa    specifies row increment of ap.
 * @param[in]     csa    specifies column increment of ap.
 * @param[in]     bp     specifies pointer which points to the first element of bp
 * @param[in]     rsb    specifies row increment of bp.
 * @param[in]     csb    specifies column increment of bp.
 * @param[in]     beta   specifies the scalar beta.
 * @param[in,out] cp     specifies pointer which points to the first element of cp
 * @param[in]     rsc    specifies row increment of cp.
 * @param[in]     csc    specifies column increment of cp.
 */

template<typename T>
static void gemm_(char transa, char transb, gtint_t m, gtint_t n, gtint_t k, T* alpha,
                    T* ap, gtint_t lda,  T* bp, gtint_t ldb, T* beta, T* cp, gtint_t ldc )
{
    if constexpr (std::is_same<T, float>::value)
        sgemm_( &transa, &transb, &m, &n, &k, alpha, ap, &lda, bp, &ldb, beta, cp, &ldc );
    else if constexpr (std::is_same<T, double>::value)
        dgemm_( &transa, &transb, &m, &n, &k, alpha, ap, &lda, bp, &ldb, beta, cp, &ldc );
    else if constexpr (std::is_same<T, scomplex>::value)
        cgemm_( &transa, &transb, &m, &n, &k, alpha, ap, &lda, bp, &ldb, beta, cp, &ldc );
    else if constexpr (std::is_same<T, dcomplex>::value)
        zgemm_( &transa, &transb, &m, &n, &k, alpha, ap, &lda, bp, &ldb, beta, cp, &ldc );
    else
        throw std::runtime_error("Error in testsuite/level3/gemm.h: Invalid typename in gemm_().");
}

template<typename T>
static void cblas_gemm(char storage, char transa, char transb,
    gtint_t m, gtint_t n, gtint_t k, T* alpha, T* ap, gtint_t lda,
    T* bp, gtint_t ldb, T* beta, T* cp, gtint_t ldc)
{
    enum CBLAS_ORDER cblas_order;
    if( storage == 'c' || storage == 'C' )
        cblas_order = CblasColMajor;
    else
        cblas_order = CblasRowMajor;

    enum CBLAS_TRANSPOSE cblas_transa;
    if( transa == 't' )
        cblas_transa = CblasTrans;
    else if( transa == 'c' )
        cblas_transa = CblasConjTrans;
    else
        cblas_transa = CblasNoTrans;

    enum CBLAS_TRANSPOSE cblas_transb;
    if( transb == 't' )
        cblas_transb = CblasTrans;
    else if( transb == 'c' )
        cblas_transb = CblasConjTrans;
    else
        cblas_transb = CblasNoTrans;

    if constexpr (std::is_same<T, float>::value)
        cblas_sgemm( cblas_order, cblas_transa, cblas_transb, m, n, k, *alpha, ap, lda, bp, ldb, *beta, cp, ldc );
    else if constexpr (std::is_same<T, double>::value)
        cblas_dgemm( cblas_order, cblas_transa, cblas_transb, m, n, k, *alpha, ap, lda, bp, ldb, *beta, cp, ldc );
    else if constexpr (std::is_same<T, scomplex>::value)
        cblas_cgemm( cblas_order, cblas_transa, cblas_transb, m, n, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
    else if constexpr (std::is_same<T, dcomplex>::value)
        cblas_zgemm( cblas_order, cblas_transa, cblas_transb, m, n, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
    else
        throw std::runtime_error("Error in testsuite/level3/gemm.h: Invalid typename in cblas_gemm().");
}

template<typename T>
static void typed_gemm(char storage, char trnsa, char trnsb,
    gtint_t m, gtint_t n, gtint_t k, T* alpha, T* ap, gtint_t lda,
    T* bp, gtint_t ldb, T* beta, T* cp, gtint_t ldc)
{
    trans_t transa, transb;

    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_trans( trnsa, &transa );
    testinghelpers::char_to_blis_trans( trnsb, &transb );

    dim_t rsa,csa;
    dim_t rsb,csb;
    dim_t rsc,csc;

    rsa=rsb=rsc=1;
    csa=csb=csc=1;
    /* a = m x k       b = k x n       c = m x n    */
    if( (storage == 'c') || (storage == 'C') ) {
        csa = lda ;
        csb = ldb ;
        csc = ldc ;
    } else {
        rsa = lda ;
        rsb = ldb ;
        rsc = ldc ;
    }

    if constexpr (std::is_same<T, float>::value)
        bli_sgemm( transa, transb, m, n, k, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc );
    else if constexpr (std::is_same<T, double>::value)
        bli_dgemm( transa, transb, m, n, k, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_cgemm( transa, transb, m, n, k, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zgemm( transa, transb, m, n, k, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc );
    else
        throw std::runtime_error("Error in testsuite/level3/gemm.h: Invalid typename in typed_gemm().");
}

template<typename T>
static void gemm( char storage, char transa, char transb, gtint_t m, gtint_t n, gtint_t k,
    T* alpha, T* ap, gtint_t lda, T* bp, gtint_t ldb, T* beta, T* cp, gtint_t ldc )
{
#ifdef TEST_BLAS
    if( storage == 'c' || storage == 'C' )
        gemm_<T>( transa, transb, m, n, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
    else
        throw std::runtime_error("Error in testsuite/level3/gemm.h: BLAS interface cannot be tested for row-major order.");

#elif TEST_CBLAS
    cblas_gemm<T>( storage, transa, transb, m, n, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
#elif TEST_BLIS_TYPED
    typed_gemm<T>( storage, transa, transb, m, n, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
#else
    throw std::runtime_error("Error in testsuite/level3/gemm.h: No interfaces are set to be tested.");
#endif
}
