/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
	- Redistributions of source code must retain the above copyright
	  notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright
	  notice, this list of conditions and the following disclaimer in the
	  documentation and/or other materials provided with the distribution.
	- Neither the name(s) of the copyright holder(s) nor the names of its
	  contributors may be used to endorse or promote products derived
	  from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

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
