#pragma once

#include "blis.h"
#include "common/testing_helpers.h"

/**
 * @brief Performs the operation:
 *             rho := beta * rho + alpha * conjx(x)^T * conjy(y), BLIS_TYPED only
 * @param[in] conjx denotes if x or conj(x) will be used for this operation
 * @param[in] conjy denotes if y or conj(y) will be used for this operation
 * @param[in] n vector length of x and y
 * @param[in] alpha scalar value
 * @param[in] x pointer which points to the first element of x
 * @param[in] incx increment of x
 * @param[in] y pointer which points to the first element of y
 * @param[in] incy increment of y
 * @param[in] beta scalar value
 * @param[in,out] rho are scalar
 */

template<typename T>
static void typed_dotxv( char conj_x, char conj_y, gtint_t n, T* alpha,
    T* x, gtint_t incx, T* y, gtint_t incy, T* beta, T* rho )
{
    conj_t conjx, conjy;
    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_conj( conj_x, &conjx );
    testinghelpers::char_to_blis_conj( conj_y, &conjy );
    if constexpr (std::is_same<T, float>::value)
        bli_sdotxv( conjx, conjy, n, alpha, x, incx, y, incy, beta, rho );
    else if constexpr (std::is_same<T, double>::value)
        bli_ddotxv( conjx, conjy, n, alpha, x, incx, y, incy, beta, rho );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_cdotxv( conjx, conjy, n, alpha, x, incx, y, incy, beta, rho );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zdotxv( conjx, conjy, n, alpha, x, incx, y, incy, beta, rho );
    else
        throw std::runtime_error("Error in testsuite/level1/addv.h: Invalid typename in typed_addv().");
}

template<typename T>
static void dotxv( char conjx, char conjy, gtint_t n, T* alpha,
    T* x, gtint_t incx, T* y, gtint_t incy, T* beta, T* rho )
{
#ifdef TEST_BLAS
    throw std::runtime_error("Error in testsuite/level1/dotxv.h: BLAS interface is not available.");
#elif TEST_CBLAS
    throw std::runtime_error("Error in testsuite/level1/dotxv.h: CBLAS interface is not available.");
#elif TEST_BLIS_TYPED
   typed_dotxv<T>( conjx, conjy, n, alpha, x, incx, y, incy, beta, rho );
#else
    throw std::runtime_error("Error in testsuite/level1/dotxv.h: No interfaces are set to be tested.");
#endif
}