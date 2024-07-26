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
                char conj_a,
                char conj_x,
                gtint_t m,
                gtint_t b,
                T *alpha,
                gtint_t inca,
                gtint_t lda_inc,
                gtint_t incx,
                gtint_t incy,
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
    // char, char, long, long, double, double*, long, long, double*, long, double*, long)
    testinghelpers::ref_axpyf<T>( conj_a, conj_x, m, b, alpha, A.data(), inca, lda, x.data(), incx, y_ref.data(), incy );

    //----------------------------------------------------------
    //                  Call BLIS function.
    //----------------------------------------------------------
    axpyf<T>( conj_a, conj_x, m, b, alpha, A.data(), inca, lda, x.data(), incx, y.data(), incy );

    //---------------------------------------------------------
    //              Compute component-wise error.
    //----------------------------------------------------------
    computediff<T>( "y", m, y.data(), y_ref.data(), incy, thresh, true );
}


// Test-case logger : Used to print the test-case details
template <typename T>
class axpyfGenericPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char,
                                          char,
                                          gtint_t,
                                          gtint_t,
                                          T,
                                          gtint_t,
                                          gtint_t,
                                          gtint_t,
                                          gtint_t>> str) const {
        char conja    = std::get<0>(str.param);
        char conjx    = std::get<1>(str.param);
        gtint_t m     = std::get<2>(str.param);
        gtint_t b  = std::get<3>(str.param);
        T alpha  = std::get<4>(str.param);
        gtint_t incx     = std::get<7>(str.param);
        gtint_t incy  = std::get<8>(str.param);

        std::string str_name = "bli_";
        str_name += "_conja_" + std::string(&conja, 1);
        str_name += "_conjx_" + std::string(&conjx, 1);
        str_name += "_m_" + std::to_string(m);
        str_name += "_b_" + std::to_string(b);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name += "_incx_" + testinghelpers::get_value_string(incx);
        str_name += "_incy_" + testinghelpers::get_value_string(incy);
        return str_name;
    }
};
