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

#include "amaxv.h"
#include "level1/ref_amaxv.h"
#include "inc/check_error.h"

/**
 * @brief Generic test body for amaxv operation.
 */

template<typename T>
static void test_amaxv( gtint_t n, gtint_t incx )
{
    //----------------------------------------------------------
    //        Initialize vectors with random numbers.
    //----------------------------------------------------------
    std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, n, incx );

    //----------------------------------------------------------
    //    Call reference implementation to get ref results.
    //----------------------------------------------------------
    gtint_t idx_ref = testinghelpers::ref_amaxv<T>( n, x.data(), incx );

    //----------------------------------------------------------
    //                  Call BLIS function.
    //----------------------------------------------------------
    gtint_t idx = amaxv<T>( n, x.data(), incx );

    //----------------------------------------------------------
    //              Compute component-wise error.
    //----------------------------------------------------------
    computediff<gtint_t>( "idx", idx, idx_ref );
}

/**
 * @brief Generic test body for amaxv operation with extreme values.
 */
template<typename T>
static void test_amaxv( gtint_t n, gtint_t incx, gtint_t xi, T xi_exval,
                      gtint_t xj, T xj_exval )
{
    //----------------------------------------------------------
    //        Initialize vectors with random numbers.
    //----------------------------------------------------------
    std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, n, incx );

    // Update the value at index xi to an extreme value, x_exval.
    if ( -1 < xi && xi < n ) x[xi * abs(incx)] = xi_exval;
    else                     return;

    // Update the value at index yi to an extreme value, y_exval.
    if ( -1 < xj && xj < n ) x[xj * abs(incx)] = xj_exval;
    else                     return;

    //----------------------------------------------------------
    //    Call reference implementation to get ref results.
    //----------------------------------------------------------
    gtint_t idx_ref = testinghelpers::ref_amaxv<T>( n, x.data(), incx );

    //----------------------------------------------------------
    //                  Call BLIS function.
    //----------------------------------------------------------
    gtint_t idx = amaxv<T>( n, x.data(), incx );

    //----------------------------------------------------------
    //              Compute component-wise error.
    //----------------------------------------------------------
    computediff<gtint_t>( "idx", idx, idx_ref );
}

// Test-case logger : Used to print the test-case details when vectors have exception value.
class amaxvGenericPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<gtint_t,gtint_t>> str) const {
        gtint_t n     = std::get<0>(str.param);
        gtint_t incx  = std::get<1>(str.param);

        std::string str_name = API_PRINT;
	      str_name += "_n_" + std::to_string(n);
        str_name += "_incx_" + testinghelpers::get_value_string(incx);
        return str_name;
    }
};

template<typename T>
class amaxvEVTPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<gtint_t,gtint_t,gtint_t,T,gtint_t,T>> str) const {
        gtint_t n     = std::get<0>(str.param);
        gtint_t incx  = std::get<1>(str.param);
        gtint_t xi    = std::get<2>(str.param);
        T xi_exval = std::get<3>(str.param);
        gtint_t xj    = std::get<4>(str.param);
        T xj_exval = std::get<5>(str.param);

        std::string str_name = API_PRINT;
        str_name += "_n_" + std::to_string(n);
        str_name += "_incx_" + testinghelpers::get_value_string(incx);
        str_name = str_name + "_X_" + std::to_string(xi) + "_" + testinghelpers::get_value_string(xi_exval);
        str_name = str_name + "_" + std::to_string(xj) + "_" + testinghelpers::get_value_string(xj_exval);
        return str_name;
    }
};
