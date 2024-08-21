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
#include "inc/check_error.h"

/**
 * @brief Performs the operation:
 *             A := alpha * op(A),
 *             where op(A) could be A, A(transpose), A(conjugate), A(conjugate-transpose)
 * @param[in] m number of rows in A, number of rows/columns in B
 * @param[in] m number of columns in A, number of columns/rows in B
 * @param[in] alpha scalar
 * @param[in] A pointer which points to the first element of A matrix
 * @param[in] lda_in leading dimension of A(input) matrix
 * @param[in] lda_out leading dimension of A(output) matrix
 */

template<typename T>
static void imatcopy_( char trans, gtint_t m, gtint_t n, T alpha, T* A, gtint_t lda_in, gtint_t lda_out )
{
    if constexpr (std::is_same<T, float>::value)
        simatcopy_( &trans, &m, &n, (const float *)&alpha, A, &lda_in, &lda_out );
    else if constexpr (std::is_same<T, double>::value)
        dimatcopy_( &trans, &m, &n, (const double *)&alpha, A, &lda_in, &lda_out );
    else if constexpr (std::is_same<T, scomplex>::value)
        cimatcopy_( &trans, &m, &n, (const scomplex *)&alpha, A, &lda_in, &lda_out );
    else if constexpr (std::is_same<T, dcomplex>::value)
        zimatcopy_( &trans, &m, &n, (const dcomplex *)&alpha, A, &lda_in, &lda_out );
    else
        throw std::runtime_error("Error in testsuite/level1/imatcopy.h: Invalid typename in imatcopy_().");
}

template<typename T>
static void imatcopy( char trans, gtint_t m, gtint_t n, T alpha, T* A, gtint_t lda_in, gtint_t lda_out )
{
#ifdef TEST_UPPERCASE_ARGS
    trans = static_cast<char>(std::toupper(static_cast<unsigned char>(trans)));
#endif

#ifdef TEST_INPUT_ARGS
    // Create copy of scalar input values so we can check that they are not altered.
    char trans_cpy = trans;
    gtint_t m_cpy = m;
    gtint_t n_cpy = n;
    T alpha_cpy = alpha;
    gtint_t lda_in_cpy = lda_in;
    gtint_t lda_out_cpy = lda_out;
#endif

#ifdef TEST_BLAS_LIKE
    imatcopy_<T>( trans, m, n, alpha, A, lda_in, lda_out );
#else
    throw std::runtime_error("Error in testsuite/level1/imatcopy.h: No interfaces are set to be tested.");
#endif

#ifdef TEST_INPUT_ARGS
    //----------------------------------------------------------
    // Check scalar inputs have not been modified.
    //----------------------------------------------------------

    computediff<char>( "trans", trans, trans_cpy );
    computediff<gtint_t>( "m", m, m_cpy );
    computediff<gtint_t>( "n", n, n_cpy );
    computediff<T>( "alpha", alpha, alpha_cpy );
    computediff<gtint_t>( "lda_in", lda_in, lda_in_cpy );
    computediff<gtint_t>( "lda_out", lda_out, lda_out_cpy );
#endif
}
