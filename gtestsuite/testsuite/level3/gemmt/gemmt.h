#pragma once

#include "blis.h"
#include "common/testing_helpers.h"

/**
 * @brief Performs the operation:
 *        C := alpha*op( A )*op( B ) + beta*C,
 * where  op( A ) is one of
 *        op( A ) = A   or   op( A ) = A**T   or   op( A ) = A**H.
 * Only accesses and updates the upper or the lower triangular part.
 * @param[in]     storage specifies storage format used for the matrices
 * @param[in]     side   specifies if the symmetric matrix A appears left or right in
                         the matrix multiplication
 * @param[in]     transa specifies the form of op( A ) to be used in
                         the matrix multiplication
 * @param[in]     transb specifies the form of op( B ) to be used in
                         the matrix multiplication
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
static void gemmt_(char uplo, char transa, char transb, gtint_t n, gtint_t k, T* alpha,
                    T* ap, gtint_t lda,  T* bp, gtint_t ldb, T* beta, T* cp, gtint_t ldc )
{
    if constexpr (std::is_same<T, float>::value)
        sgemmt_( &uplo, &transa, &transb, &n, &k, alpha, ap, &lda, bp, &ldb, beta, cp, &ldc );
    else if constexpr (std::is_same<T, double>::value)
        dgemmt_( &uplo, &transa, &transb, &n, &k, alpha, ap, &lda, bp, &ldb, beta, cp, &ldc );
    else if constexpr (std::is_same<T, scomplex>::value)
        cgemmt_( &uplo, &transa, &transb, &n, &k, alpha, ap, &lda, bp, &ldb, beta, cp, &ldc );
    else if constexpr (std::is_same<T, dcomplex>::value)
        zgemmt_( &uplo, &transa, &transb, &n, &k, alpha, ap, &lda, bp, &ldb, beta, cp, &ldc );
    else
        throw std::runtime_error("Error in testsuite/level3/gemmt.h: Invalid typename in gemmt_().");
}

template<typename T>
static void cblas_gemmt(char storage, char uplo, char transa, char transb,
    gtint_t n, gtint_t k, T* alpha, T* ap, gtint_t lda,
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

    enum CBLAS_UPLO cblas_uplo;
    if( (uplo == 'u') || (uplo == 'U') )
        cblas_uplo = CblasUpper;
    else
        cblas_uplo = CblasLower;

    if constexpr (std::is_same<T, float>::value)
        cblas_sgemmt( cblas_order, cblas_uplo, cblas_transa, cblas_transb, n, k, *alpha, ap, lda, bp, ldb, *beta, cp, ldc );
    else if constexpr (std::is_same<T, double>::value)
        cblas_dgemmt( cblas_order, cblas_uplo, cblas_transa, cblas_transb, n, k, *alpha, ap, lda, bp, ldb, *beta, cp, ldc );
    else if constexpr (std::is_same<T, scomplex>::value)
        cblas_cgemmt( cblas_order, cblas_uplo, cblas_transa, cblas_transb, n, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
    else if constexpr (std::is_same<T, dcomplex>::value)
        cblas_zgemmt( cblas_order, cblas_uplo, cblas_transa, cblas_transb, n, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
    else
        throw std::runtime_error("Error in testsuite/level3/gemmt.h: Invalid typename in cblas_gemmt().");
}

#ifdef TEST_BLIS_TYPED
template<typename T>
static void typed_gemmt(char storage, char uplo, char trnsa, char trnsb,
    gtint_t n, gtint_t k, T* alpha, T* ap, gtint_t lda,
    T* bp, gtint_t ldb, T* beta, T* cp, gtint_t ldc)
{
    trans_t transa, transb;
    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_trans( trnsa, &transa );
    testinghelpers::char_to_blis_trans( trnsb, &transb );

    uplo_t blis_uplo;
    testinghelpers::char_to_blis_uplo( uplo, &blis_uplo );

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
        bli_sgemmt( blis_uplo, transa, transb, n, k, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc );
    else if constexpr (std::is_same<T, double>::value)
        bli_dgemmt( blis_uplo, transa, transb, n, k, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_cgemmt( blis_uplo, transa, transb, n, k, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zgemmt( blis_uplo, transa, transb, n, k, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc );
    else
        throw std::runtime_error("Error in testsuite/level3/gemmt.h: Invalid typename in typed_gemmt().");
}
#endif
template<typename T>
static void gemmt( char storage, char uplo, char transa, char transb, gtint_t n, gtint_t k,
    T* alpha, T* ap, gtint_t lda, T* bp, gtint_t ldb, T* beta, T* cp, gtint_t ldc )
{
#ifdef TEST_BLAS
    if( storage == 'c' || storage == 'C' )
        gemmt_<T>( uplo, transa, transb, n, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
    else
        throw std::runtime_error("Error in testsuite/level3/gemmt.h: BLAS interface cannot be tested for row-major order.");

#elif TEST_CBLAS
    cblas_gemmt<T>( storage, uplo, transa, transb, n, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
#elif TEST_BLIS_TYPED
    //typed_gemmt<T>( storage, uplo, transa, transb, n, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
    throw std::runtime_error("Error in testsuite/level3/gemmt.h: BLIS-typed interface cannot be tested tested.");
#else
    throw std::runtime_error("Error in testsuite/level3/gemmt.h: No interfaces are set to be tested.");
#endif
}
