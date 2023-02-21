#pragma once

#include "blis.h"
#include "common/testing_helpers.h"

/**
 * @brief Performs the operation
 *             x := conjalpha(alpha) (BLIS_TYPED only)
 * @param[in] conjalpha denotes if alpha or conj(alpha) will be used for this operation
 * @param[in] n vector length of x
 * @param[in] alpha value to set in vector x.
 * @param[in,out] x pointer which points to the first element of x
 * @param[in] incx increment of x
 */

template<typename T>
static void typed_setv(char conjalpha, gtint_t n, T* alpha, T* x, gtint_t incx)
{
    conj_t conjx;
    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_conj( conjalpha, &conjx );
    if constexpr (std::is_same<T, float>::value)
        bli_ssetv( conjx, n, alpha, x, incx );
    else if constexpr (std::is_same<T, double>::value)
        bli_dsetv( conjx, n, alpha, x, incx );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_csetv( conjx, n, alpha, x, incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zsetv( conjx, n, alpha, x, incx );
    else
        throw std::runtime_error("Error in testsuite/level1/setv.h: Invalid typename in typed_setv().");
}

template<typename T>
static void setv(char conjalpha, gtint_t n, T* alpha, T* x, gtint_t incx)
{
#ifdef TEST_BLAS
    throw std::runtime_error("Error in testsuite/level1/setv.h: BLAS interface is not available.");
#elif TEST_CBLAS
    throw std::runtime_error("Error in testsuite/level1/setv.h: CBLAS interface is not available.");
#elif TEST_BLIS_TYPED
    typed_setv(conjalpha, n, alpha, x, incx);
#else
    throw std::runtime_error("Error in testsuite/level1/setv.h: No interfaces are set to be tested.");
#endif
}