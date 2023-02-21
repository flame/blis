#pragma once

#include "blis.h"
#include "common/testing_helpers.h"

/**
 * @brief Performs the operation:
 *        C := alpha*A*B**T + alpha*B*A**T + beta*C
 *     or C := alpha*A**T*B + alpha*B**T*A + beta*C
 * @param[in]     storage specifies storage format used for the matrices
 * @param[in]     uplo   specifies if the upper or lower triangular part of A is used
 * @param[in]     transa specifies the form of op( A ) to be used in
                         the matrix multiplication
 * @param[in]     transb specifies the form of op( B ) to be used in
                         the matrix multiplication
 * @param[in]     m      specifies the number of rows and cols of the  matrix
                         op( A ) and rows of the matrix C and B
 * @param[in]     k      specifies the number of columns of the matrix
                         op( B ) and the number of columns of the matrix C
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

template<typename T, typename RT = typename testinghelpers::type_info<T>::real_type>
static void her2k_(char uplo, char transa, gtint_t m, gtint_t k, T* alpha,
                    T* ap, gtint_t lda, T* bp, gtint_t ldb, RT* beta, T* cp, gtint_t ldc )
{
    if constexpr (std::is_same<T, scomplex>::value)
        cher2k_( &uplo, &transa, &m, &k, alpha, ap, &lda, bp, &ldb, beta, cp, &ldc );
    else if constexpr (std::is_same<T, dcomplex>::value)
        zher2k_( &uplo, &transa, &m, &k, alpha, ap, &lda, bp, &ldb, beta, cp, &ldc );
    else
        throw std::runtime_error("Error in testsuite/level3/her2k.h: Invalid typename in her2k_().");
}

template<typename T, typename RT = typename testinghelpers::type_info<T>::real_type>
static void cblas_her2k(char storage, char uplo, char transa,
    gtint_t m, gtint_t k, T* alpha, T* ap, gtint_t lda,
    T* bp, gtint_t ldb, RT* beta, T* cp, gtint_t ldc)
{
    enum CBLAS_ORDER cblas_order;
    if( storage == 'c' || storage == 'C' )
        cblas_order = CblasColMajor;
    else
        cblas_order = CblasRowMajor;

    enum CBLAS_UPLO cblas_uplo;
    if( (uplo == 'u') || (uplo == 'U') )
        cblas_uplo = CblasUpper;
    else
        cblas_uplo = CblasLower;

    enum CBLAS_TRANSPOSE cblas_transa;
    if( transa == 't' )
        cblas_transa = CblasTrans;
    else if( transa == 'c' )
        cblas_transa = CblasConjTrans;
    else
        cblas_transa = CblasNoTrans;

    if constexpr (std::is_same<T, scomplex>::value)
        cblas_cher2k( cblas_order, cblas_uplo, cblas_transa, m, k, alpha, ap, lda, bp, ldb, *beta, cp, ldc );
    else if constexpr (std::is_same<T, dcomplex>::value)
        cblas_zher2k( cblas_order, cblas_uplo, cblas_transa, m, k, alpha, ap, lda, bp, ldb, *beta, cp, ldc );
    else
        throw std::runtime_error("Error in testsuite/level3/her2k.h: Invalid typename in cblas_her2k().");
}

template<typename T, typename RT = typename testinghelpers::type_info<T>::real_type>
static void typed_her2k(char storage, char uplo, char trnsa, char trnsb,
    gtint_t m, gtint_t k, T* alpha, T* ap, gtint_t lda,
    T* bp, gtint_t ldb, RT* beta, T* cp, gtint_t ldc)
{
    trans_t transa, transb;
    uplo_t blis_uplo;
    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_trans( trnsa, &transa );
    testinghelpers::char_to_blis_trans( trnsb, &transb );
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
        bli_sher2k( blis_uplo, transa, transb, m, k, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc );
    else if constexpr (std::is_same<T, double>::value)
        bli_dher2k( blis_uplo, transa, transb, m, k, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_cher2k( blis_uplo, transa, transb, m, k, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zher2k( blis_uplo, transa, transb, m, k, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc );
    else
        throw std::runtime_error("Error in testsuite/level3/her2k.h: Invalid typename in typed_her2k().");
}

template<typename T, typename RT = typename testinghelpers::type_info<T>::real_type>
static void her2k( char storage, char uplo, char transa, char transb, gtint_t m, gtint_t k,
    T* alpha, T* ap, gtint_t lda, T* bp, gtint_t ldb, RT* beta, T* cp, gtint_t ldc )
{
#ifdef TEST_BLAS
    if( storage == 'c' || storage == 'C' )
        her2k_<T>( uplo, transa, m, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
    else
        throw std::runtime_error("Error in testsuite/level3/her2k.h: BLAS interface cannot be tested for row-major order.");

#elif TEST_CBLAS
    cblas_her2k<T>( storage, uplo, transa, m, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
#elif TEST_BLIS_TYPED
    typed_her2k<T>( storage, uplo, transa, transb, m, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
#else
    throw std::runtime_error("Error in testsuite/level3/her2k.h: No interfaces are set to be tested.");
#endif
}
