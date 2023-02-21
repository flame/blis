#pragma once

#include "blis.h"
#include "common/testing_helpers.h"

/**
 * @brief Computes
 *             y := y + x or y := y + conj(x)
 *        This is a BLIS-specific API, not part of BLAS/CBLAS.
 * @param[in] conjx denotes if x or conj(x) will be used for this operation
 * @param[in] n vector length of x and y
 * @param[in] x pointer which points to the first element of x
 * @param[in] incx increment of x
 * @param[in, out] y pointer which points to the first element of y
 * @param[in] incy increment of y
 */

template<typename T>
static void typed_addv(char conj_x, gtint_t n, T* x, gtint_t incx, T* y, gtint_t incy)
{
    conj_t conjx;
    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_conj( conj_x, &conjx );
    if constexpr (std::is_same<T, float>::value)
        bli_saddv( conjx, n, x, incx, y, incy );
    else if constexpr (std::is_same<T, double>::value)
        bli_daddv( conjx, n, x, incx, y, incy );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_caddv( conjx, n, x, incx, y, incy );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zaddv( conjx, n, x, incx, y, incy );
    else
        throw std::runtime_error("Error in testsuite/level1/addv.h: Invalid typename in typed_addv().");
}

template<typename T>
static void addv(char conjx, gtint_t n, T* x, gtint_t incx, T* y, gtint_t incy)
{
#ifdef TEST_BLAS
    throw std::runtime_error("Error in testsuite/level1/addv.h: BLAS interface is not available.");
#elif TEST_CBLAS
    throw std::runtime_error("Error in testsuite/level1/addv.h: CBLAS interface is not available.");
#elif TEST_BLIS_TYPED
    typed_addv(conjx, n, x, incx, y, incy);
#else
    throw std::runtime_error("Error in testsuite/level1/addv.h: No interfaces are set to be tested.");
#endif
}