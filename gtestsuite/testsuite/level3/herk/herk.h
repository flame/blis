#pragma once

#include "blis.h"
#include "common/testing_helpers.h"

/**
 * @brief Performs the operation:
 *        C := alpha*A*A**H + beta*C
 *     or C := alpha*A**H*A + beta*C
 * @param[in]     storage specifies storage format used for the matrices
 * @param[in]     uplo   specifies if the upper or lower triangular part of C is used
 * @param[in]     transa specifies the form of op( A ) to be used in
                         the matrix multiplication
 * @param[in]     n      specifies the number of rows and cols of C
 * @param[in]     k      specifies the number of rows of A, in case of transa = 'C',
 *                       and the columns of A otherwise.
 * @param[in]     alpha  specifies the scalar alpha.
 * @param[in]     ap     specifies pointer which points to the first element of ap
 * @param[in]     rsa    specifies row increment of ap.
 * @param[in]     csa    specifies column increment of ap.
 * @param[in]     beta   specifies the scalar beta.
 * @param[in,out] cp     specifies pointer which points to the first element of cp
 * @param[in]     rsc    specifies row increment of cp.
 * @param[in]     csc    specifies column increment of cp.
 */

template<typename T, typename RT = typename testinghelpers::type_info<T>::real_type>
static void herk_(char uplo, char transa, gtint_t m, gtint_t k, RT* alpha,
                    T* ap, gtint_t lda,  RT* beta, T* cp, gtint_t ldc )
{
    if constexpr (std::is_same<T, scomplex>::value)
        cherk_( &uplo, &transa, &m, &k, alpha, ap, &lda, beta, cp, &ldc );
    else if constexpr (std::is_same<T, dcomplex>::value)
        zherk_( &uplo, &transa, &m, &k, alpha, ap, &lda, beta, cp, &ldc );
    else
        throw std::runtime_error("Error in testsuite/level3/herk.h: Invalid typename in herk_().");
}

template<typename T, typename RT = typename testinghelpers::type_info<T>::real_type>
static void cblas_herk(char storage, char uplo, char trnsa,
    gtint_t m, gtint_t k, RT* alpha, T* ap, gtint_t lda,
    RT* beta, T* cp, gtint_t ldc)
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
    if( trnsa == 't' )
        cblas_transa = CblasTrans;
    else if( trnsa == 'c' )
        cblas_transa = CblasConjTrans;
    else
        cblas_transa = CblasNoTrans;

    if constexpr (std::is_same<T, scomplex>::value)
        cblas_cherk( cblas_order, cblas_uplo, cblas_transa, m, k, *alpha, ap, lda, *beta, cp, ldc );
    else if constexpr (std::is_same<T, dcomplex>::value)
        cblas_zherk( cblas_order, cblas_uplo, cblas_transa, m, k, *alpha, ap, lda, *beta, cp, ldc );
    else
        throw std::runtime_error("Error in testsuite/level3/herk.h: Invalid typename in cblas_herk().");
}

template<typename T, typename RT = typename testinghelpers::type_info<T>::real_type>
static void typed_herk(char storage, char uplo, char trnsa,
    gtint_t m, gtint_t k, RT* alpha, T* ap, gtint_t lda,
    RT* beta, T* cp, gtint_t ldc)
{
    trans_t transa;
    uplo_t blis_uplo;
    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_trans( trnsa, &transa );
    testinghelpers::char_to_blis_uplo( uplo, &blis_uplo );
    dim_t rsa,csa;
    dim_t rsc,csc;

    rsa=rsc=1;
    csa=csc=1;
    /* a = m x k   c = m x m    */
    if( (storage == 'c') || (storage == 'C') ) {
        csa = lda ;
        csc = ldc ;
    } else {
        rsa = lda ;
        rsc = ldc ;
    }

    if constexpr (std::is_same<T, float>::value)
        bli_sherk( blis_uplo, transa, m, k, alpha, ap, rsa, csa, beta, cp, rsc, csc );
    else if constexpr (std::is_same<T, double>::value)
        bli_dherk( blis_uplo, transa, m, k, alpha, ap, rsa, csa, beta, cp, rsc, csc );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_cherk( blis_uplo, transa, m, k, alpha, ap, rsa, csa, beta, cp, rsc, csc );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zherk( blis_uplo, transa, m, k, alpha, ap, rsa, csa, beta, cp, rsc, csc );
    else
        throw std::runtime_error("Error in testsuite/level3/herk.h: Invalid typename in typed_herk().");
}

template<typename T, typename RT = typename testinghelpers::type_info<T>::real_type>
static void herk( char storage, char uplo, char transa, gtint_t m, gtint_t k,
    RT* alpha, T* ap, gtint_t lda, RT* beta, T* cp, gtint_t ldc )
{
#ifdef TEST_BLAS
    if( storage == 'c' || storage == 'C' )
        herk_<T>( uplo, transa, m, k, alpha, ap, lda, beta, cp, ldc );
    else
        throw std::runtime_error("Error in testsuite/level3/herk.h: BLAS interface cannot be tested for row-major order.");
#elif TEST_CBLAS
    cblas_herk<T>( storage, uplo, transa, m, k, alpha, ap, lda, beta, cp, ldc );
#elif TEST_BLIS_TYPED
    typed_herk<T>( storage, uplo, transa, m, k, alpha, ap, lda, beta, cp, ldc );
#else
    throw std::runtime_error("Error in testsuite/level3/herk.h: No interfaces are set to be tested.");
#endif
}
