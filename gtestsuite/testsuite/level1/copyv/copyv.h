#pragma once

#include "blis.h"
#include "common/testing_helpers.h"

/**
 * @brief Performs the operation:
 *             y := x
 *          or y : = conj(x) BLIS_TYPED only
 * @param[in] conjx denotes if x or conj(x) will be used for this operation (BLIS API specific)
 * @param[in] n vector length of x and y
 * @param[in] x pointer which points to the first element of x
 * @param[in] incx increment of x
 * @param[out] y pointer which points to the first element of y
 * @param[in] incy increment of y
 */

template<typename T>
static void copyv_(gtint_t n, T* x, gtint_t incx, T* y, gtint_t incy) {

    if constexpr (std::is_same<T, float>::value)
        scopy_( &n, x, &incx, y, &incy );
    else if constexpr (std::is_same<T, double>::value)
        dcopy_( &n, x, &incx, y, &incy );
    else if constexpr (std::is_same<T, scomplex>::value)
        ccopy_( &n, x, &incx, y, &incy );
    else if constexpr (std::is_same<T, dcomplex>::value)
        zcopy_( &n, x, &incx, y, &incy );
    else
        throw std::runtime_error("Error in testsuite/level1/copyv.h: Invalid typename in copyv_().");
}

template<typename T>
static void cblas_copyv(gtint_t n, T* x, gtint_t incx, T* y, gtint_t incy) {

    if constexpr (std::is_same<T, float>::value)
      cblas_scopy( n, x, incx, y, incy );
    else if constexpr (std::is_same<T, double>::value)
      cblas_dcopy( n, x, incx, y, incy );
    else if constexpr (std::is_same<T, scomplex>::value)
      cblas_ccopy( n, x, incx, y, incy );
    else if constexpr (std::is_same<T, dcomplex>::value)
      cblas_zcopy( n, x, incx, y, incy );
    else
      throw std::runtime_error("Error in testsuite/level1/copyv.h: Invalid typename in cblas_copyv().");
}

template<typename T>
static void typed_copyv(char conjx, gtint_t n, T* x, gtint_t incx, T* y, gtint_t incy) {

    conj_t conj_x;
    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_conj( conjx, &conj_x );
    if constexpr (std::is_same<T, float>::value)
        bli_scopyv( conj_x, n, x, incx, y, incy );
    else if constexpr (std::is_same<T, double>::value)
        bli_dcopyv( conj_x, n, x, incx, y, incy );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_ccopyv( conj_x, n, x, incx, y, incy );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zcopyv( conj_x, n, x, incx, y, incy );
    else
      throw std::runtime_error("Error in testsuite/level1/copyv.h: Invalid typename in typed_copyv().");
}

template<typename T>
static void copyv(char conjx, gtint_t n, T* x, gtint_t incx, T* y, gtint_t incy)
{
#ifdef TEST_BLAS
    copyv_<T>(n, x, incx, y, incy);
#elif TEST_CBLAS
    cblas_copyv<T>(n, x, incx, y, incy);
#elif TEST_BLIS_TYPED
    typed_copyv<T>(conjx, n, x, incx, y, incy);
#else
    throw std::runtime_error("Error in testsuite/level1/copyv.h: No interfaces are set to be tested.");
#endif
}