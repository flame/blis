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

#include "axpyf.h"
#include "level1/ref_axpyf.h"
#include "inc/check_error.h"

/**
 * axpyf operation is defined as :
 * y := y + alpha * conja(A) * conjx(x)
 * where A is an m x b matrix, and y and x are vectors. 
 * Matrix should be represented as "A" instead of "a" to distinguish it from vector.
*/
template<typename T>
static void test_axpyf(
                conj_t conja,
                conj_t conjx,
                gint_t m,
                gint_t b,
                T *alpha,
                gint_t inca,
                gint_t lda_inc,
                gint_t incx,
                gint_t incy,
                double thresh
                )
{
    //----------------------------------------------------------
    //        Initialize vectors with random numbers.
    //----------------------------------------------------------

    // Compute the leading dimensions of A matrix.
    gtint_t lda = testinghelpers::get_leading_dimension( 'c', 'n', m, b, lda_inc );

    //----------------------------------------------------------
    //         Initialize matrics with random numbers
    //----------------------------------------------------------
    std::vector<T> A = testinghelpers::get_random_matrix<T>( -2, 8, 'c', 'n', m, b, lda );

    std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, m, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( -10, 10, m, incy );

    //----------------------------------------------------------
    //    Call reference implementation to get ref results.
    //----------------------------------------------------------
    // Create a copy of y so that we can check reference results.
    std::vector<T> y_ref(y);
    // conj_t, conj_t, long, long, double, double*, long, long, double*, long, double*, long)
    testinghelpers::ref_axpyf<T>( conja, conjx, m, b, alpha, A.data(), inca, lda, x.data(), incx, y_ref.data(), incy );

    //----------------------------------------------------------
    //                  Call BLIS function.
    //----------------------------------------------------------
    axpyf<T>( conja, conjx, m, b, alpha, A.data(), inca, lda, x.data(), incx, y.data(), incy );

    //---------------------------------------------------------
    //              Compute component-wise error.
    //----------------------------------------------------------
    computediff<T>( m, y.data(), y_ref.data(), incy, thresh );
}
