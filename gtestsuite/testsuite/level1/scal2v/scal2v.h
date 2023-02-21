#pragma once

#include "blis.h"
#include "common/testing_helpers.h"

/**
 * @brief Performs the operation:
 *             y := alpha * conj(x) (for BLIS interface only)
 * @param[in] conjx denotes if x or conj(x) will be used for this operation
 * @param[in] n vector length of x and y
 * @param[in] alpha scalar
 * @param[in] x pointer which points to the first element of x
 * @param[in] incx increment of x
 * @param[in,out] y pointer which points to the first element of y
 * @param[in] incy increment of y
 */

template<typename T>
static void typed_scal2v(char conj_x, gtint_t n, T alpha, T* x, gtint_t incx, T* y, gtint_t incy)
{
    conj_t conjx;
    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_conj( conj_x, &conjx );
    if constexpr (std::is_same<T, float>::value)
        bli_sscal2v( conjx, n, &alpha, x, incx, y, incy );
    else if constexpr (std::is_same<T, double>::value)
        bli_dscal2v( conjx, n, &alpha, x, incx, y, incy );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_cscal2v( conjx, n, &alpha, x, incx, y, incy );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zscal2v( conjx, n, &alpha, x, incx, y, incy );
    else
        throw std::runtime_error("Error in testsuite/level1/scal2v.h: Invalid typename in typed_scal2v().");
}


template<typename T>
static void scal2v(char conjx, gtint_t n, T alpha, T* x, gtint_t incx, T* y, gtint_t incy)
{
#ifdef TEST_BLAS
    throw std::runtime_error("Error in testsuite/level1/scal2v.h: BLAS interface is not available.");
#elif TEST_CBLAS
    throw std::runtime_error("Error in testsuite/level1/scal2v.h: BLAS interface is not available.");
#elif TEST_BLIS_TYPED
    typed_scal2v<T>( conjx, n, alpha, x, incx, y, incy );
#else
    throw std::runtime_error("Error in testsuite/level1/scal2v.h: No interfaces are set to be tested.");
#endif
}