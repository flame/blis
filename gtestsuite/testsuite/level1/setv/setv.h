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

#ifdef TEST_UPPERCASE_ARGS
    conjalpha = static_cast<char>(std::toupper(static_cast<unsigned char>(conjalpha)));
#endif

#ifdef TEST_INPUT_ARGS
    // Create copy of scalar input values so we can check that they are not altered.
    char conjalpha_cpy = conjalpha;
    gtint_t n_cpy = n;
    T* alpha_cpy = alpha;
    gtint_t incx_cpy = incx;
#endif

#ifdef TEST_BLAS
    throw std::runtime_error("Error in testsuite/level1/setv.h: BLAS interface is not available.");
#elif TEST_BLAS_BLIS_IMPL
    throw std::runtime_error("Error in testsuite/level1/setv.h: BLAS_BLIS_IMPL interface is not available.");
#elif TEST_CBLAS
    throw std::runtime_error("Error in testsuite/level1/setv.h: CBLAS interface is not available.");
#elif TEST_BLIS_TYPED
    typed_setv(conjalpha, n, alpha, x, incx);
#else
    throw std::runtime_error("Error in testsuite/level1/setv.h: No interfaces are set to be tested.");
#endif

#ifdef TEST_INPUT_ARGS
    //----------------------------------------------------------
    // Check scalar inputs have not been modified.
    //----------------------------------------------------------

    computediff<char>( "conjalpha", conjalpha, conjalpha_cpy );
    computediff<gtint_t>( "n", n, n_cpy );
    if (alpha) computediff<T>( "alpha", *alpha, *alpha_cpy );
    computediff<gtint_t>( "incx", incx, incx_cpy );
#endif
}
