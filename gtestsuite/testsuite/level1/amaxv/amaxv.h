#pragma once

#include "blis.h"
#include "common/testing_helpers.h"

/**
 * @brief Finds the index of the first element that has the maximum absolute value.
 * @param[in] n vector length of x and y
 * @param[in] x pointer which points to the first element of x
 * @param[in] incx increment of x
 * 
 * If n < 1 or incx <= 0, return 0.
 */

template<typename T>
static gtint_t amaxv_(gtint_t n, T* x, gtint_t incx) {

    gtint_t idx;
    if constexpr (std::is_same<T, float>::value)
        idx = isamax_( &n, x, &incx );
    else if constexpr (std::is_same<T, double>::value)
        idx = idamax_( &n, x, &incx );
    else if constexpr (std::is_same<T, scomplex>::value)
        idx = icamax_( &n, x, &incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        idx = izamax_( &n, x, &incx );
    else
      throw std::runtime_error("Error in testsuite/level1/amaxv.h: Invalid typename in amaxv_().");

    // Since we are comparing against CBLAS which is 0-based and BLAS is 1-based, 
    // we need to use -1 here.
    return (idx-1);
}

template<typename T>
static gtint_t cblas_amaxv(gtint_t n, T* x, gtint_t incx) {

    gtint_t idx;
    if constexpr (std::is_same<T, float>::value)
      idx = cblas_isamax( n, x, incx );
    else if constexpr (std::is_same<T, double>::value)
      idx = cblas_idamax( n, x, incx );
    else if constexpr (std::is_same<T, scomplex>::value)
      idx = cblas_icamax( n, x, incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
      idx = cblas_izamax( n, x, incx );
    else
      throw std::runtime_error("Error in testsuite/level1/amaxv.h: Invalid typename in cblas_amaxv().");

    return idx;
}

template<typename T>
static gtint_t typed_amaxv(gtint_t n, T* x, gtint_t incx)
{
    gtint_t idx = 0;
    if constexpr (std::is_same<T, float>::value)
        bli_samaxv( n, x, incx, &idx );
    else if constexpr (std::is_same<T, double>::value)
        bli_damaxv( n, x, incx, &idx );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_camaxv( n, x, incx, &idx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zamaxv( n, x, incx, &idx );
    else
        throw std::runtime_error("Error in testsuite/level1/amaxddv.h: Invalid typename in typed_amaxv().");

    return idx;
}

template<typename T>
static gtint_t amaxv(gtint_t n, T* x, gtint_t incx)
{
#ifdef TEST_BLAS
    return amaxv_<T>(n, x, incx);
#elif TEST_CBLAS
    return cblas_amaxv<T>(n, x, incx);
#elif TEST_BLIS_TYPED
    return typed_amaxv(n, x, incx);
#else
    throw std::runtime_error("Error in testsuite/level1/amaxv.h: No interfaces are set to be tested.");
#endif
}