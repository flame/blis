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

#include "asumv.h"
#include <limits>
#include "util/ref_asumv.h"
#include "inc/check_error.h"

/**
 * @brief Used for generic tests with random values in x.
 */
template<typename T>
void test_asumv( gtint_t n, gtint_t incx, double thresh )
{
    // Get real type from T.
    using RT = typename testinghelpers::type_info<T>::real_type;
    //----------------------------------------------------------
    //        Initialize vectors with random numbers.
    //----------------------------------------------------------
    std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, n, incx );

    //----------------------------------------------------------
    //    Call reference implementation to get ref results.
    //----------------------------------------------------------
    RT asum_ref = testinghelpers::ref_asumv<T>( n, x.data(), incx );

    //----------------------------------------------------------
    //                  Call BLIS function.
    //----------------------------------------------------------
    RT asum = asumv<T>(n, x.data(), incx);

    //----------------------------------------------------------
    //              Compute error.
    //----------------------------------------------------------
    computediff<RT>( "asum", asum, asum_ref, thresh );
}

/**
 * @brief Used to insert Exception Values in x vector.
 */
template<typename T>
void test_asumv( gtint_t n, gtint_t incx, gtint_t xi, T ix_exval,
                 gtint_t xj, T jx_exval, double thresh )
{
    // Get real type from T.
    using RT = typename testinghelpers::type_info<T>::real_type;
    //----------------------------------------------------------
    //        Initialize vectors with random numbers.
    //----------------------------------------------------------
    std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, n, incx );

    // Update the value at index xi to an extreme value, ix_exval.
    if ( -1 < xi && xi < n ) x[xi * incx] = ix_exval;
    else                     return;

    // Update the value at index xj to an extreme value, jx_exval.
    if ( -1 < xi && xi < n ) x[xj * incx] = jx_exval;
    else                     return;

    //----------------------------------------------------------
    //    Call reference implementation to get ref results.
    //----------------------------------------------------------
    RT asum_ref = testinghelpers::ref_asumv<T>( n, x.data(), incx );

    //----------------------------------------------------------
    //                  Call BLIS function.
    //----------------------------------------------------------
    RT asum = asumv<T>(n, x.data(), incx);

    //----------------------------------------------------------
    //              Compute error.
    //----------------------------------------------------------
    computediff<RT>( "asum", asum, asum_ref, thresh, true );
}


// Test-case logger : Used to print the test-case details based on parameters
class asumvGenericPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<gtint_t, gtint_t>> str) const {
        gtint_t n     = std::get<0>(str.param);
        gtint_t incx  = std::get<1>(str.param);

        std::string str_name = API_PRINT;
        str_name += "_n_" + std::to_string(n);
        str_name += "_incx_" + testinghelpers::get_value_string(incx);
        return str_name;
    }
};

template <typename T>
class asumvEVTPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<gtint_t, gtint_t, gtint_t, T, gtint_t, T>> str) const {
        gtint_t n        = std::get<0>(str.param);
        gtint_t incx     = std::get<1>(str.param);
        gtint_t xi       = std::get<2>(str.param);
        T  ix_exval = std::get<3>(str.param);
        gtint_t xj       = std::get<4>(str.param);
        T  jx_exval = std::get<5>(str.param);

        std::string str_name = API_PRINT;
        str_name += "_n_" + std::to_string(n);
        str_name += "_incx_" + testinghelpers::get_value_string(incx);
        str_name = str_name + "_X_" + std::to_string(xi);
        str_name = str_name + "_" + testinghelpers::get_value_string(ix_exval);
        str_name = str_name + "_X_" + std::to_string(xj);
        str_name = str_name + "_" + testinghelpers::get_value_string(jx_exval);
        return str_name;
    }
};
