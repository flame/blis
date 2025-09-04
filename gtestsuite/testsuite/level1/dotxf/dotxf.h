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

template<typename T>
static void typed_dotxf(
                char conj_a,
                char conj_x,
                gtint_t m,
                gtint_t b,
                T *alpha,
                T* A,
                gtint_t inca,
                gtint_t lda,
                T* x,
                gtint_t incx,
                T *beta,
                T* y,
                gtint_t incy)
{
    conj_t conja;
    conj_t conjx;
    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_conj( conj_a, &conja );
    testinghelpers::char_to_blis_conj( conj_x, &conjx );
    if constexpr (std::is_same<T, float>::value)
        bli_sdotxf(conja, conjx, m, b, alpha, A, inca, lda, x, incx, beta, y, incy);
    else if constexpr (std::is_same<T, double>::value)
        bli_ddotxf( conja, conjx, m, b, alpha, A, inca, lda, x, incx, beta, y, incy );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_cdotxf( conja, conjx, m, b, alpha, A, inca, lda, x, incx, beta, y, incy );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zdotxf( conja, conjx, m, b, alpha, A, inca, lda, x, incx, beta, y, incy );
    else
        throw std::runtime_error("Error in testsuite/level1/dotv.h: Invalid typename in typed_dotv().");
}

template<typename T>
static void dotxf(
                char conj_a,
                char conj_x,
                gtint_t m,
                gtint_t b,
                T *alpha,
                T* A,
                gtint_t inca,
                gtint_t lda,
                T* x,
                gtint_t incx,
                T *beta,
                T* y,
                gtint_t incy
)
{

#ifdef TEST_UPPERCASE_ARGS
    conj_a = static_cast<char>(std::toupper(static_cast<unsigned char>(conj_a)));
    conj_x = static_cast<char>(std::toupper(static_cast<unsigned char>(conj_x)));
#endif

#ifdef TEST_INPUT_ARGS
    // Create copy of scalar input values so we can check that they are not altered.
    char conj_a_cpy = conj_a;
    char conj_x_cpy = conj_x;
    gtint_t m_cpy = m;
    gtint_t b_cpy = b;
    T* alpha_cpy = alpha;
    gtint_t inca_cpy = inca;
    gtint_t lda_cpy = lda;
    gtint_t incx_cpy = incx;
    T* beta_cpy = beta;
    gtint_t incy_cpy = incy;

    // Create copy of input arrays so we can check that they are not altered.
    T* A_cpy = nullptr;
    gtint_t size_A = testinghelpers::matsize( 'c', 'n', m, b, lda );
    if (A && size_A > 0)
    {
        A_cpy = new T[size_A];
        memcpy( A_cpy, A, size_A * sizeof( T ) );
    }
    T* x_cpy = nullptr;
    gtint_t size_x = testinghelpers::buff_dim( m, incx );
    if (x && size_x > 0)
    {
        x_cpy = new T[size_x];
        memcpy( x_cpy, x, size_x * sizeof( T ) );
    }
#endif

/**
 * dotxf operation is defined as :
 * y := beta * y + alpha * conja(A) * conjx(x)
 * where A is an m x b matrix, and y and x are vectors.
 */
    typed_dotxf<T>(
               conj_a,
               conj_x,
               m,
               b,
               alpha,
               A,
               inca,
               lda,
               x,
               incx,
               beta,
               y,
               incy );

#ifdef TEST_INPUT_ARGS
    //----------------------------------------------------------
    // Check scalar inputs have not been modified.
    //----------------------------------------------------------

    computediff<char>( "conj_a", conj_a, conj_a_cpy );
    computediff<char>( "conj_x", conj_x, conj_x_cpy );
    computediff<gtint_t>( "m", m, m_cpy );
    computediff<gtint_t>( "b", b, b_cpy );
    if (alpha) computediff<T>( "alpha", *alpha, *alpha_cpy );
    computediff<gtint_t>( "inca", inca, inca_cpy );
    computediff<gtint_t>( "lda", lda, lda_cpy );
    computediff<gtint_t>( "incx", incx, incx_cpy );
    if (beta) computediff<T>( "beta", *beta, *beta_cpy );
    computediff<gtint_t>( "incy", incy, incy_cpy );

    //----------------------------------------------------------
    // Bitwise-wise check array inputs have not been modified.
    //----------------------------------------------------------

    if (A && size_A > 0)
    {
        computediff<T>( "A", 'c', m, b, A, A_cpy, lda, true );
        delete[] A_cpy;
    }

    if (x && size_x > 0)
    {
        computediff<T>( "x", m, x, x_cpy, incx, true );
        delete[] x_cpy;
    }
#endif
}
