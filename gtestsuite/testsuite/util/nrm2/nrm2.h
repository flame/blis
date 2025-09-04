/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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
#include "inc/check_error.h"

/**
 * @brief Computes the Euclidean norm of x.
 *
 * Euclidean norm of a vector x is defined as nrm2 = sqrt(x'*x).
 * In case a vector element is NaN, nrm2 must be NaN.
 * In case a vector element is inf, and there is no element which is NaN, nrm2 must be inf.
 * If n <= 0, nrm2 returns zero.
 * If incx = 0, nrm2 returns sqrt(n*abs(x[0])**2).
 *
 * @param[in] n vector length
 * @param[in] x pointer which points to the first element of x
 * @param[in] incx increment of x
 * @return the Euclidean norm of x
 *
 *
 */

template<typename T, typename RT = typename testinghelpers::type_info<T>::real_type>
static RT nrm2_(gtint_t n, T* x, gtint_t incx){
    if constexpr (std::is_same<T, float>::value)
        return snrm2_( &n, x, &incx );
    else if constexpr (std::is_same<T, double>::value)
        return dnrm2_( &n, x, &incx );
    else if constexpr (std::is_same<T, scomplex>::value)
        return scnrm2_( &n, x, &incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        return dznrm2_( &n, x, &incx );
    else
      throw std::runtime_error("Error in testsuite/level1/nrm2.h: Invalid typename in nrm2_().");
}

template<typename T, typename RT = typename testinghelpers::type_info<T>::real_type>
static RT nrm2_blis_impl(gtint_t n, T* x, gtint_t incx){
    if constexpr (std::is_same<T, float>::value)
        return snrm2_blis_impl( &n, x, &incx );
    else if constexpr (std::is_same<T, double>::value)
        return dnrm2_blis_impl( &n, x, &incx );
    else if constexpr (std::is_same<T, scomplex>::value)
        return scnrm2_blis_impl( &n, x, &incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        return dznrm2_blis_impl( &n, x, &incx );
    else
      throw std::runtime_error("Error in testsuite/level1/nrm2.h: Invalid typename in nrm2_blis_impl().");
}

template<typename T, typename RT = typename testinghelpers::type_info<T>::real_type>
static RT cblas_nrm2(gtint_t n, T* x, gtint_t incx){
    if constexpr (std::is_same<T, float>::value)
        return cblas_snrm2( n, x, incx );
    else if constexpr (std::is_same<T, double>::value)
        return cblas_dnrm2( n, x, incx );
    else if constexpr (std::is_same<T, scomplex>::value)
        return cblas_scnrm2( n, x, incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        return cblas_dznrm2( n, x, incx );
    else
      throw std::runtime_error("Error in testsuite/level1/nrm2.h: Invalid typename in cblas_nrm2().");
}

template<typename T, typename RT = typename testinghelpers::type_info<T>::real_type>
static RT typed_nrm2(gtint_t n, T* x, gtint_t incx){
    RT nrm;
    if constexpr (std::is_same<T, float>::value)
        bli_snormfv(n, x, incx, &nrm);
    else if constexpr (std::is_same<T, double>::value)
        bli_dnormfv(n, x, incx, &nrm);
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_cnormfv(n, x, incx, &nrm);
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_znormfv(n, x, incx, &nrm);
    else
      throw std::runtime_error("Error in testsuite/level1/nrm2.h: Invalid typename in cblas_nrm2().");
    return nrm;
}

template<typename T, typename RT = typename testinghelpers::type_info<T>::real_type>
static RT nrm2(gtint_t n, T* x, gtint_t incx)
{

#ifdef TEST_INPUT_ARGS
    // Create copy of scalar input values so we can check that they are not altered.
    gtint_t n_cpy = n;
    gtint_t incx_cpy = incx;

    // Create copy of input arrays so we can check that they are not altered.
    T* x_cpy = nullptr;
    gtint_t size_x = testinghelpers::buff_dim( n, incx );
    if (x && size_x > 0)
    {
        x_cpy = new T[size_x];
        memcpy( x_cpy, x, size_x * sizeof( T ) );
    }
#endif

#ifdef TEST_BLAS
    return nrm2_<T>(n, x, incx);
#elif TEST_BLAS_BLIS_IMPL
    return nrm2_blis_impl<T>(n, x, incx);
#elif TEST_CBLAS
    return cblas_nrm2<T>(n, x, incx);
#elif TEST_BLIS_TYPED
    return typed_nrm2<T>(n, x, incx);
#else
    throw std::runtime_error("Error in testsuite/level1/axpyv.h: No interfaces are set to be tested.");
#endif

#ifdef TEST_INPUT_ARGS
    //----------------------------------------------------------
    // Check scalar inputs have not been modified.
    //----------------------------------------------------------

    computediff<gtint_t>( "n", n, n_cpy );
    computediff<gtint_t>( "incx", incx, incx_cpy );

    //----------------------------------------------------------
    // Bitwise-wise check array inputs have not been modified.
    //----------------------------------------------------------

    if (x && size_x > 0)
    {
        computediff<T>( "x", n, x, x_cpy, incx, true );
        delete[] x_cpy;
    }
#endif
}
