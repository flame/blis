/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

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
 *             B := alpha * op(A),
 *             where op(A) could be A, A(transpose), A(conjugate), A(conjugate-transpose)
 * @param[in] m number of rows in A, number of rows/columns in B
 * @param[in] n number of columns in A, number of columns/rows in B
 * @param[in] alpha scalar
 * @param[in] A pointer which points to the first element of A matrix
 * @param[in] lda leading dimension of A matrix
 * @param[in] stridea stride between two "continuous" elements in A
 * @param[in, out] B pointer which points to the first element of B matrix
 * @param[in] ldb leading dimension of B matrix
 * @param[in] strideb stride between two "continuous" elements in B
 */

template<typename T>
static void omatcopy2_( char trans, gtint_t m, gtint_t n, T alpha, T* A, gtint_t lda, gtint_t stridea, T* B, gtint_t ldb, gtint_t strideb )
{
    if constexpr (std::is_same<T, float>::value)
        somatcopy2_( &trans, &m, &n, (const float *)&alpha, A, &lda, &stridea, B, &ldb, &strideb );
    else if constexpr (std::is_same<T, double>::value)
        domatcopy2_( &trans, &m, &n, (const double *)&alpha, A, &lda, &stridea, B, &ldb, &strideb );
    else if constexpr (std::is_same<T, scomplex>::value)
        comatcopy2_( &trans, &m, &n, (const scomplex *)&alpha, A, &lda, &stridea, B, &ldb, &strideb );
    else if constexpr (std::is_same<T, dcomplex>::value)
        zomatcopy2_( &trans, &m, &n, (const dcomplex *)&alpha, A, &lda, &stridea, B, &ldb, &strideb );
    else
        throw std::runtime_error("Error in testsuite/extension/omatcopy2.h: Invalid typename in omatcopy2_().");
}

template<typename T>
static void omatcopy2( char trans, gtint_t m, gtint_t n, T alpha, T* A, gtint_t lda, gtint_t stridea, T* B, gtint_t ldb, gtint_t strideb )
{
#ifdef TEST_BLAS
    omatcopy2_<T>( trans, m, n, alpha, A, lda, stridea, B, ldb, strideb );
#else
    throw std::runtime_error("Error in testsuite/extension/omatcopy2.h: No interfaces are set to be tested.");
#endif
}

