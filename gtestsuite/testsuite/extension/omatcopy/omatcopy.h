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
 * @param[in, out] B pointer which points to the first element of B matrix
 * @param[in] ldb leading dimension of B matrix
 */

template<typename T>
static void omatcopy_( char trans, gtint_t m, gtint_t n, T alpha, T* A, gtint_t lda, T* B, gtint_t ldb )
{
    if constexpr (std::is_same<T, float>::value)
        somatcopy_( &trans, &m, &n, (const float *)&alpha, A, &lda, B, &ldb );
    else if constexpr (std::is_same<T, double>::value)
        domatcopy_( &trans, &m, &n, (const double *)&alpha, A, &lda, B, &ldb );
    else if constexpr (std::is_same<T, scomplex>::value)
        comatcopy_( &trans, &m, &n, (const scomplex *)&alpha, A, &lda, B, &ldb );
    else if constexpr (std::is_same<T, dcomplex>::value)
        zomatcopy_( &trans, &m, &n, (const dcomplex *)&alpha, A, &lda, B, &ldb );
    else
        throw std::runtime_error("Error in testsuite/level1/omatcopy.h: Invalid typename in omatcopy_().");
}

template<typename T>
static void omatcopy( char trans, gtint_t m, gtint_t n, T alpha, T* A, gtint_t lda, T* B, gtint_t ldb )
{
#ifdef TEST_BLAS
    omatcopy_<T>( trans, m, n, alpha, A, lda, B, ldb );
#else
    throw std::runtime_error("Error in testsuite/level1/omatcopy.h: No interfaces are set to be tested.");
#endif
}

