#pragma once

#include "blis.h"
#include "common/testing_helpers.h"

/**
 * @brief Performs the operation:
 *             y := alpha * y 
 *          or y := conj(alpha) * y (for BLIS interface only)
 * @param[in] conjalpha denotes if alpha or conj(alpha) will be used for this operation
 * @param[in] n vector length of x and y
 * @param[in] alpha scalar
 * @param[in,out] x pointer which points to the first element of x
 * @param[in] incx increment of x
 */

template<typename T>
static void scalv_(gtint_t n, T alpha, T* x, gtint_t incx)
{
    if constexpr (std::is_same<T, float>::value)
        sscal_( &n, &alpha, x, &incx );
    else if constexpr (std::is_same<T, double>::value)
        dscal_( &n, &alpha, x, &incx );
    else if constexpr (std::is_same<T, scomplex>::value)
        cscal_( &n, &alpha, x, &incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        zscal_( &n, &alpha, x, &incx );
    else
        throw std::runtime_error("Error in testsuite/level1/scalv.h: Invalid typename in scalv_().");
}

template<typename T>
static void cblas_scalv(gtint_t n, T alpha, T* x, gtint_t incx)
{
    if constexpr (std::is_same<T, float>::value)
        cblas_sscal( n, alpha, x, incx );
    else if constexpr (std::is_same<T, double>::value)
        cblas_dscal( n, alpha, x, incx );
    else if constexpr (std::is_same<T, scomplex>::value)
        cblas_cscal( n, &alpha, x, incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        cblas_zscal( n, &alpha, x, incx );
    else
        throw std::runtime_error("Error in testsuite/level1/scalv.h: Invalid typename in cblas_scalv().");
}

template<typename T>
static void typed_scalv(char conj_alpha, gtint_t n, T alpha, T* x, gtint_t incx)
{
    conj_t conjalpha;
    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_conj( conj_alpha, &conjalpha );
    if constexpr (std::is_same<T, float>::value)
        bli_sscalv( conjalpha, n, &alpha, x, incx );
    else if constexpr (std::is_same<T, double>::value)
        bli_dscalv( conjalpha, n, &alpha, x, incx );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_cscalv( conjalpha, n, &alpha, x, incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zscalv( conjalpha, n, &alpha, x, incx );
    else
        throw std::runtime_error("Error in testsuite/level1/scalv.h: Invalid typename in typed_scalv().");
}


template<typename T>
static void scalv(char conj_alpha, gtint_t n, T alpha, T* x, gtint_t incx)
{
#ifdef TEST_BLAS
    scalv_<T>( n, alpha, x, incx );
#elif TEST_CBLAS
    cblas_scalv<T>( n, alpha, x, incx );
#elif TEST_BLIS_TYPED
    typed_scalv<T>( conj_alpha, n, alpha, x, incx );
#else
    throw std::runtime_error("Error in testsuite/level1/scalv.h: No interfaces are set to be tested.");
#endif
}