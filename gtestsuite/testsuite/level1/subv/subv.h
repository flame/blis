#pragma once

#include "blis.h"
#include "common/testing_helpers.h"

/**
 * @brief Performs the operation
 *             y := y - x or y := y - conj(x) (BLIS_TYPED only)
 * @param[in] conjx denotes if x or conj(x) will be used for this operation
 * @param[in] n vector length of x and y
 * @param[in] x pointer which points to the first element of x
 * @param[in] incx increment of x
 * @param[in, out] y pointer which points to the first element of y
 * @param[in] incy increment of y
 */

template<typename T>
static void typed_subv(char conj_x, gtint_t n, T* x, gtint_t incx, T* y, gtint_t incy)
{
    conj_t conjx;
    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_conj( conj_x, &conjx );
    if constexpr (std::is_same<T, float>::value)
        bli_ssubv( conjx, n, x, incx, y, incy );
    else if constexpr (std::is_same<T, double>::value)
        bli_dsubv( conjx, n, x, incx, y, incy );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_csubv( conjx, n, x, incx, y, incy );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zsubv( conjx, n, x, incx, y, incy );
    else
        throw std::runtime_error("Error in testsuite/level1/subv.h: Invalid typename in typed_subv().");
}

template<typename T>
static void subv(char conjx, gtint_t n, T* x, gtint_t incx, T* y, gtint_t incy)
{
#ifdef TEST_BLAS
    throw std::runtime_error("Error in testsuite/level1/subv.h: BLAS interface is not available.");
#elif TEST_CBLAS
    throw std::runtime_error("Error in testsuite/level1/subv.h: CBLAS interface is not available.");
#elif TEST_BLIS_TYPED
    typed_subv(conjx, n, x, incx, y, incy);
#else
    throw std::runtime_error("Error in testsuite/level1/subv.h: No interfaces are set to be tested.");
#endif
}