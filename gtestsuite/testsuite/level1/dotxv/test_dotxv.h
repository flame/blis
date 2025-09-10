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

#include "dotxv.h"
#include "level1/ref_dotxv.h"
#include "inc/check_error.h"

/**
 * @brief Generic test body for dotxv operation.
 */

template<typename T>
static void test_dotxv( gtint_t n, char conjx, char conjy, T alpha,
                        gtint_t incx, gtint_t incy, T beta, double thresh )
{
    //----------------------------------------------------------
    //        Initialize vectors with random numbers.
    //----------------------------------------------------------
    std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, n, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( -10, 10, n, incy );

    //----------------------------------------------------------
    //    Call reference implementation to get ref results.
    //----------------------------------------------------------
    // Create a copy of y so that we can check reference results.
    std::vector<T> y_ref(y);
    T rho_ref;
    testinghelpers::initone<T>(rho_ref);
    testinghelpers::ref_dotxv<T>( conjx, conjy, n, alpha, x.data(), incx, y.data(), incy, beta, &rho_ref );

    //----------------------------------------------------------
    //                  Call BLIS function.
    //----------------------------------------------------------
    T rho;
    testinghelpers::initone<T>(rho);
    dotxv<T>( conjx, conjy, n, &alpha, x.data(), incx, y.data(), incy, &beta, &rho );

    //----------------------------------------------------------
    //              Compute error.
    //----------------------------------------------------------
    computediff<T>( "rho", rho, rho_ref, thresh );
}

// Test-case logger : Used to print the test-case details based on parameters
template <typename T>
class dotxvGenericPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<gtint_t,char,char,gtint_t,gtint_t,T,T>> str) const {
        gtint_t n      = std::get<0>(str.param);
        char conjx     = std::get<1>(str.param);
        char conjy     = std::get<2>(str.param);
        gtint_t incx   = std::get<3>(str.param);
        gtint_t incy   = std::get<4>(str.param);
        T alpha = std::get<5>(str.param);
        T beta  = std::get<6>(str.param);

        std::string str_name = API_PRINT;
        str_name += "_n_" + std::to_string(n);
        str_name += "_conjx_" + std::string(&conjx, 1);
        str_name += "_conjy_" + std::string(&conjy, 1);
        str_name += "_incx_" + testinghelpers::get_value_string(incx);
        str_name += "_incy_" + testinghelpers::get_value_string(incy);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name += "_beta_" + testinghelpers::get_value_string(beta);
        return str_name;
    }
};
