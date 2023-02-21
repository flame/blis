#pragma once

#include "blis.h"
#include "common/testing_helpers.h"

/**
 * @brief Performs the operation:
 *   A := alpha*x*y**T + alpha*y*x**T + A,
 * @param[in]     storage specifies the form of storage in the memory matrix A
 * @param[in]     uploa  specifies whether the upper or lower triangular part of the array A
 * @param[in]     n      specifies the number  of rows  of the  matrix A
 * @param[in]     alpha  specifies the scalar alpha.
 * @param[in]     xp     specifies pointer which points to the first element of xp
 * @param[in]     incx   specifies storage spacing between elements of xp.
 * @param[in]     yp     specifies pointer which points to the first element of yp
 * @param[in]     incy   specifies storage spacing between elements of yp.
 * @param[in,out] ap     specifies pointer which points to the first element of ap
 * @param[in]     lda    specifies leading dimension of the matrix.
 */

template<typename T>
static void her2_( char uploa, gtint_t n, T* alpha, T* xp, gtint_t incx,
                              T* yp, gtint_t incy, T* ap, gtint_t lda )
{
    if constexpr (std::is_same<T, scomplex>::value)
        cher2_( &uploa, &n, alpha, xp, &incx, yp, &incy, ap, &lda );
    else if constexpr (std::is_same<T, dcomplex>::value)
        zher2_( &uploa, &n, alpha, xp, &incx, yp, &incy, ap, &lda );
    else
        throw std::runtime_error("Error in testsuite/level2/her2.h: Invalid typename in her2_().");
}

template<typename T>
static void cblas_her2( char storage, char uploa, gtint_t n, T* alpha,
       T* xp, gtint_t incx, T* yp, gtint_t incy, T* ap, gtint_t lda )
{
    enum CBLAS_ORDER cblas_order;
    if( storage == 'c' || storage == 'C' )
        cblas_order = CblasColMajor;
    else
        cblas_order = CblasRowMajor;

    enum CBLAS_UPLO cblas_uplo;
    if( (uploa == 'u') || (uploa == 'U') )
        cblas_uplo = CblasUpper;
    else
        cblas_uplo = CblasLower;

    if constexpr (std::is_same<T, scomplex>::value)
        cblas_cher2( cblas_order, cblas_uplo, n, alpha, xp, incx, yp, incy, ap, lda );
    else if constexpr (std::is_same<T, dcomplex>::value)
        cblas_zher2( cblas_order, cblas_uplo, n, alpha, xp, incx, yp, incy, ap, lda );
    else
        throw std::runtime_error("Error in testsuite/level2/her2.h: Invalid typename in cblas_her2().");
}

template<typename T>
static void typed_her2( char storage, char uplo, char conj_x, char conj_y,
    gtint_t n, T* alpha, T* x, gtint_t incx, T* y, gtint_t incy,
    T* a, gtint_t lda )
{
    uplo_t uploa;
    conj_t conjx;
    conj_t conjy;

    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_uplo ( uplo, &uploa );
    testinghelpers::char_to_blis_conj ( conj_x, &conjx );
    testinghelpers::char_to_blis_conj ( conj_y, &conjy );

    dim_t rsa,csa;
    rsa=csa=1;
    /* a = n x n   */
    if( (storage == 'c') || (storage == 'C') )
        csa = lda ;
    else
        rsa = lda ;

    if constexpr (std::is_same<T, scomplex>::value)
        bli_cher2( uploa, conjx, conjy, n, alpha, x, incx, y, incy, a, rsa, csa );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zher2( uploa, conjx, conjy, n, alpha, x, incx, y, incy, a, rsa, csa );
    else
        throw std::runtime_error("Error in testsuite/level2/her2.h: Invalid typename in typed_her2().");
}

template<typename T>
static void her2( char storage, char uploa, char conj_x, char conj_y, gtint_t n,
      T* alpha, T* xp, gtint_t incx, T* yp, gtint_t incy, T* ap, gtint_t lda )
{
#ifdef TEST_BLAS
    if( storage == 'c' || storage == 'C' )
        her2_<T>( uploa, n, alpha, xp, incx, yp, incy, ap, lda );
    else
        throw std::runtime_error("Error in testsuite/level2/her2.h: BLAS interface cannot be tested for row-major order.");
#elif TEST_CBLAS
    cblas_her2<T>( storage, uploa, n, alpha, xp, incx, yp, incy, ap, lda );
#elif TEST_BLIS_TYPED
    typed_her2<T>( storage, uploa, conj_x, conj_y, n, alpha, xp, incx, yp, incy, ap, lda );
#else
    throw std::runtime_error("Error in testsuite/level2/her2.h: No interfaces are set to be tested.");
#endif
}