#pragma once

#include "blis.h"
#include "common/testing_helpers.h"

/**
 * @brief Performs the operation:
 *             y := beta * y + alpha * x 
 *          or y := beta * y + alpha * conj(x) (BLIS_TYPED only)
 * @param[in] conjx denotes if x or conj(x) will be used for this operation (BLIS API specific)
 * @param[in] n vector length of x and y
 * @param[in] alpha scalar
 * @param[in] x pointer which points to the first element of x
 * @param[in] incx increment of x
 * @param[in] beta scalar
 * @param[in, out] y pointer which points to the first element of y
 * @param[in] incy increment of y
 */

template<typename T>
static void axpbyv_(gtint_t n, T alpha, T* x, gtint_t incx, T beta, T* y, gtint_t incy)
{
    if constexpr (std::is_same<T, float>::value)
        saxpby_( &n, &alpha, x, &incx, &beta, y, &incy );
    else if constexpr (std::is_same<T, double>::value)
        daxpby_( &n, &alpha, x, &incx, &beta, y, &incy );
    else if constexpr (std::is_same<T, scomplex>::value)
        caxpby_( &n, &alpha, x, &incx, &beta, y, &incy );
    else if constexpr (std::is_same<T, dcomplex>::value)
        zaxpby_( &n, &alpha, x, &incx, &beta, y, &incy );
    else
        throw std::runtime_error("Error in testsuite/level1/axpbyv.h: Invalid typename in axpbyv_().");
}

template<typename T>
static void cblas_axpbyv(gtint_t n, T alpha, T* x, gtint_t incx, T beta, T* y, gtint_t incy)
{
    if constexpr (std::is_same<T, float>::value)
        cblas_saxpby( n, alpha, x, incx, beta, y, incy );
    else if constexpr (std::is_same<T, double>::value)
        cblas_daxpby( n, alpha, x, incx, beta, y, incy );
    else if constexpr (std::is_same<T, scomplex>::value)
        cblas_caxpby( n, &alpha, x, incx, &beta, y, incy );
    else if constexpr (std::is_same<T, dcomplex>::value)
        cblas_zaxpby( n, &alpha, x, incx, &beta, y, incy );
    else
        throw std::runtime_error("Error in testsuite/level1/axpbyv.h: Invalid typename in cblas_axpbyv().");
}

template<typename T>
static void typed_axpbyv(char conj_x, gtint_t n, T alpha, T* x, gtint_t incx, T beta, T* y, gtint_t incy)
{
    conj_t conjx;
    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_conj( conj_x, &conjx );
    if constexpr (std::is_same<T, float>::value)
        bli_saxpbyv( conjx, n, &alpha, x, incx, &beta, y, incy );
    else if constexpr (std::is_same<T, double>::value)
        bli_daxpbyv( conjx, n, &alpha, x, incx, &beta, y, incy );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_caxpbyv( conjx, n, &alpha, x, incx, &beta, y, incy );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zaxpbyv( conjx, n, &alpha, x, incx, &beta, y, incy );
    else
        throw std::runtime_error("Error in testsuite/level1/axpbyv.h: Invalid typename in typed_axpbyv().");
}

template<typename T>
static void axpbyv(char conj_x, gtint_t n, T alpha, T* x, gtint_t incx, T beta, T* y, gtint_t incy)
{
#ifdef TEST_BLAS
    axpbyv_<T>( n, alpha, x, incx, beta, y, incy );
#elif TEST_CBLAS
    cblas_axpbyv<T>( n, alpha, x, incx, beta, y, incy );
#elif TEST_BLIS_TYPED
    typed_axpbyv<T>( conj_x, n, alpha, x, incx, beta, y, incy );
#else
    throw std::runtime_error("Error in testsuite/level1/axpbyv.h: No interfaces are set to be tested.");
#endif
}