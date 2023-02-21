#pragma once

#include "blis.h"
#include "common/testing_helpers.h"

/**
 * @brief Performs the operation:
 *             rho := conjx(x)^T * conjy(y)
 *          or rho := conjx(x)^T * conjy(y) (BLIS_TYPED only)
 * @param[in] conjx denotes if x or conj(x) will be used for this operation (BLIS API specific)
 * @param[in] conjy denotes if y or conj(y) will be used for this operation (BLIS API specific)
 * @param[in] n vector length of x and y
 * @param[in] x pointer which points to the first element of x
 * @param[in] incx increment of x
 * @param[in, out] y pointer which points to the first element of y
 * @param[in] incy increment of y
 * @param[in,out] rho is a scalar
 */

template<typename T>
static void dotv_(gtint_t n, T* x, gtint_t incx, T* y, gtint_t incy, T* rho) {

  if constexpr (std::is_same<T, float>::value)
    *rho = sdot_( &n, x, &incx, y, &incy );
  else if constexpr (std::is_same<T, double>::value)
    *rho = ddot_( &n, x, &incx, y, &incy );
  else if constexpr (std::is_same<T, scomplex>::value)
    *rho = cdotu_( &n, x, &incx, y, &incy );
  else if constexpr (std::is_same<T, dcomplex>::value)
    *rho = zdotu_( &n, x, &incx, y, &incy );
  else
    throw std::runtime_error("Error in testsuite/level1/dotv.h: Invalid typename in dotv_().");
}

template<typename T>
static void cblas_dotv(gtint_t n, T* x, gtint_t incx, T* y, gtint_t incy, T* rho) {

  if constexpr (std::is_same<T, float>::value)
    *rho = cblas_sdot( n, x, incx, y, incy );
  else if constexpr (std::is_same<T, double>::value)
    *rho = cblas_ddot( n, x, incx, y, incy );
  else if constexpr (std::is_same<T, scomplex>::value)
    cblas_cdotu_sub( n, x, incx, y, incy, rho );
  else if constexpr (std::is_same<T, dcomplex>::value)
    cblas_zdotu_sub( n, x, incx, y, incy, rho );
  else
    throw std::runtime_error("Error in testsuite/level1/dotv.h: Invalid typename in cblas_dotv().");
}

template<typename T>
static void typed_dotv(char conj_x, char conj_y, gtint_t n,
  T* x, gtint_t incx, T* y, gtint_t incy, T* rho) {

  conj_t conjx, conjy;
  // Map parameter characters to BLIS constants.
  testinghelpers::char_to_blis_conj( conj_x, &conjx );
  testinghelpers::char_to_blis_conj( conj_y, &conjy );
  if constexpr (std::is_same<T, float>::value)
    bli_sdotv( conjx, conjy, n, x, incx, y, incy, rho );
  else if constexpr (std::is_same<T, double>::value)
    bli_ddotv( conjx, conjy, n, x, incx, y, incy, rho );
  else if constexpr (std::is_same<T, scomplex>::value)
    bli_cdotv( conjx, conjy, n, x, incx, y, incy, rho );
  else if constexpr (std::is_same<T, dcomplex>::value)
    bli_zdotv( conjx, conjy, n, x, incx, y, incy, rho );
  else
    throw std::runtime_error("Error in testsuite/level1/dotv.h: Invalid typename in typed_dotv().");
}

template<typename T>
static void dotv(char conjx, char conjy, gtint_t n,
  T* x, gtint_t incx, T* y, gtint_t incy, T* rho)
{
#ifdef TEST_BLAS
    dotv_<T>(n, x, incx, y, incy, rho);
#elif TEST_CBLAS
    cblas_dotv<T>(n, x, incx, y, incy, rho);
#elif TEST_BLIS_TYPED
    typed_dotv<T>(conjx, conjy, n, x, incx, y, incy, rho);
#else
    throw std::runtime_error("Error in testsuite/level1/dotv.h: No interfaces are set to be tested.");
#endif
}