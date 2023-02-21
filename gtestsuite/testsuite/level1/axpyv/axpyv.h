#pragma once

#include "blis.h"
#include "common/testing_helpers.h"

/**
 * @brief Performs the operation:
 *             y := y + alpha * x 
 *          or y := y + alpha * conj(x) BLIS_TYPED only
 * @param[in] conjx denotes if x or conj(x) will be used for this operation (BLIS API specific)
 * @param[in] n vector length of x and y
 * @param[in] alpha scalar
 * @param[in] x pointer which points to the first element of x
 * @param[in] incx increment of x
 * @param[in, out] y pointer which points to the first element of y
 * @param[in] incy increment of y
 */

template<typename T>
static void axpyv_(gtint_t n, T alpha, T* x, gtint_t incx, T* y, gtint_t incy)
{
    if constexpr (std::is_same<T, float>::value)
        saxpy_( &n, &alpha, x, &incx, y, &incy );
    else if constexpr (std::is_same<T, double>::value)
        daxpy_( &n, &alpha, x, &incx, y, &incy );
    else if constexpr (std::is_same<T, scomplex>::value)
        caxpy_( &n, &alpha, x, &incx, y, &incy );
    else if constexpr (std::is_same<T, dcomplex>::value)
        zaxpy_( &n, &alpha, x, &incx, y, &incy );
    else
        throw std::runtime_error("Error in testsuite/level1/axpyv.h: Invalid typename in axpyv_().");
}

template<typename T>
static void cblas_axpyv(gtint_t n, T alpha, T* x, gtint_t incx, T* y, gtint_t incy)
{
    if constexpr (std::is_same<T, float>::value)
        cblas_saxpy( n, alpha, x, incx, y, incy );
    else if constexpr (std::is_same<T, double>::value)
        cblas_daxpy( n, alpha, x, incx, y, incy );
    else if constexpr (std::is_same<T, scomplex>::value)
        cblas_caxpy( n, &alpha, x, incx, y, incy );
    else if constexpr (std::is_same<T, dcomplex>::value)
        cblas_zaxpy( n, &alpha, x, incx, y, incy );
    else
        throw std::runtime_error("Error in testsuite/level1/axpyv.h: Invalid typename in cblas_axpyv().");
}

template<typename T>
static void typed_axpyv(char conj_x, gtint_t n, T alpha, T* x, gtint_t incx, T* y, gtint_t incy)
{
    conj_t conjx;
    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_conj( conj_x, &conjx );
    if constexpr (std::is_same<T, float>::value)
        bli_saxpyv( conjx, n, &alpha, x, incx, y, incy );
    else if constexpr (std::is_same<T, double>::value)
        bli_daxpyv( conjx, n, &alpha, x, incx, y, incy );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_caxpyv( conjx, n, &alpha, x, incx, y, incy );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zaxpyv( conjx, n, &alpha, x, incx, y, incy );
    else
        throw std::runtime_error("Error in testsuite/level1/axpyv.h: Invalid typename in typed_axpyv().");
}

template<typename T>
static void axpyv(char conj_x, gtint_t n, T alpha, T* x, gtint_t incx, T* y, gtint_t incy)
{
#ifdef TEST_BLAS
    axpyv_<T>( n, alpha, x, incx, y, incy );
#elif TEST_CBLAS
    cblas_axpyv<T>( n, alpha, x, incx, y, incy );
#elif TEST_BLIS_TYPED
    typed_axpyv<T>( conj_x, n, alpha, x, incx, y, incy );
#else
    throw std::runtime_error("Error in testsuite/level1/axpyv.h: No interfaces are set to be tested.");
#endif
}