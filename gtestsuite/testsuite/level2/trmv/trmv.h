#pragma once

#include "blis.h"
#include "common/testing_helpers.h"

/**
 * @brief Performs the operation
  *    x := alpha * transa(A) * x
 * @param[in]     storage specifies the form of storage in the memory matrix A
 * @param[in]     uploa  specifies whether the upper or lower triangular part of the array A
 * @param[in]     transa specifies the form of op( A ) to be used in matrix multiplication
 * @param[in]     diaga  specifies whether the upper or lower triangular part of the array A
 * @param[in]     n      specifies the number  of rows  of the  matrix A
 * @param[in]     alpha  specifies the scalar alpha.
 * @param[in]     ap     specifies pointer which points to the first element of ap
 * @param[in]     lda    specifies leading dimension of the matrix.
 * @param[in,out] xp     specifies pointer which points to the first element of xp
 * @param[in]     incx   specifies storage spacing between elements of xp.

 */

template<typename T>
static void trmv_( char uploa, char transa, char diaga, gtint_t n,
                         T *ap, gtint_t lda, T *xp, gtint_t incx )
{
    if constexpr (std::is_same<T, float>::value)
        strmv_( &uploa, &transa, &diaga, &n, ap, &lda, xp, &incx );
    else if constexpr (std::is_same<T, double>::value)
        dtrmv_( &uploa, &transa, &diaga, &n, ap, &lda, xp, &incx );
    else if constexpr (std::is_same<T, scomplex>::value)
        ctrmv_( &uploa, &transa, &diaga, &n, ap, &lda, xp, &incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        ztrmv_( &uploa, &transa, &diaga, &n, ap, &lda, xp, &incx );
    else
        throw std::runtime_error("Error in testsuite/level2/trmv.h: Invalid typename in trmv_().");
}

template<typename T>
static void cblas_trmv( char storage, char uploa, char transa, char diaga,
                      gtint_t n, T *ap, gtint_t lda, T *xp, gtint_t incx )
{

    enum CBLAS_ORDER cblas_order;
    if( storage == 'c' || storage == 'C' )
        cblas_order = CblasColMajor;
    else
        cblas_order = CblasRowMajor;

    enum CBLAS_UPLO cblas_uploa;
    if( (uploa == 'u') || (uploa == 'U') )
        cblas_uploa = CblasUpper;
    else
        cblas_uploa = CblasLower;

    enum CBLAS_TRANSPOSE cblas_transa;
    if( transa == 't' )
        cblas_transa = CblasTrans;
    else if( transa == 'c' )
        cblas_transa = CblasConjTrans;
    else
        cblas_transa = CblasNoTrans;

    enum CBLAS_DIAG cblas_diaga;
    if( (diaga == 'u') || (diaga == 'U') )
        cblas_diaga = CblasUnit;
    else
        cblas_diaga = CblasNonUnit;

    if constexpr (std::is_same<T, float>::value)
        cblas_strmv( cblas_order, cblas_uploa, cblas_transa, cblas_diaga, n, ap, lda, xp, incx );
    else if constexpr (std::is_same<T, double>::value)
        cblas_dtrmv( cblas_order, cblas_uploa, cblas_transa, cblas_diaga, n, ap, lda, xp, incx );
    else if constexpr (std::is_same<T, scomplex>::value)
        cblas_ctrmv( cblas_order, cblas_uploa, cblas_transa, cblas_diaga, n, ap, lda, xp, incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        cblas_ztrmv( cblas_order, cblas_uploa, cblas_transa, cblas_diaga, n, ap, lda, xp, incx );
    else
        throw std::runtime_error("Error in testsuite/level2/trmv.h: Invalid typename in cblas_trmv().");
}

template<typename T>
static void typed_trmv( char storage, char uplo, char trans, char diag,
            gtint_t n, T *alpha, T *ap, gtint_t lda, T *xp, gtint_t incx )
{
    uplo_t  uploa;
    trans_t transa;
    diag_t  diaga;

    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_uplo ( uplo, &uploa );
    testinghelpers::char_to_blis_trans( trans, &transa );
    testinghelpers::char_to_blis_diag ( diag, &diaga );

    dim_t rsa,csa;
    rsa=csa=1;
    /* a = n x n   */
    if( (storage == 'c') || (storage == 'C') )
        csa = lda ;
    else
        rsa = lda ;

    if constexpr (std::is_same<T, float>::value)
        bli_strmv( uploa, transa, diaga, n, alpha, ap, rsa, csa, xp, incx );
    else if constexpr (std::is_same<T, double>::value)
        bli_dtrmv( uploa, transa, diaga, n, alpha, ap, rsa, csa, xp, incx );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_ctrmv( uploa, transa, diaga, n, alpha, ap, rsa, csa, xp, incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_ztrmv( uploa, transa, diaga, n, alpha, ap, rsa, csa, xp, incx );
    else

        throw std::runtime_error("Error in testsuite/level2/trmv.h: Invalid typename in typed_trmv().");
}

template<typename T>
static void trmv( char storage, char uploa, char transa, char diaga,
    gtint_t n, T *alpha, T *ap, gtint_t lda, T *xp, gtint_t incx )
{
#if (defined TEST_BLAS || defined  TEST_CBLAS)
    T one;
    testinghelpers::initone(one);
#endif

#ifdef TEST_BLAS
    if(( storage == 'c' || storage == 'C' ))
        if( *alpha == one )
            trmv_<T>( uploa, transa, diaga, n, ap, lda, xp, incx );
        else
            throw std::runtime_error("Error in testsuite/level2/trmv.h: BLAS interface cannot be tested for alpha != one.");
    else
        throw std::runtime_error("Error in testsuite/level2/trmv.h: BLAS interface cannot be tested for row-major order.");
#elif TEST_CBLAS
    if( *alpha == one )
        cblas_trmv<T>( storage, uploa, transa, diaga, n, ap, lda, xp, incx );
    else
      throw std::runtime_error("Error in testsuite/level2/trmv.h: CBLAS interface cannot be tested for alpha != one.");
#elif TEST_BLIS_TYPED
    typed_trmv<T>( storage, uploa, transa, diaga, n, alpha, ap, lda, xp, incx );
#else
    throw std::runtime_error("Error in testsuite/level2/trmv.h: No interfaces are set to be tested.");
#endif
}