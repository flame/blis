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

#include "nrm2.h"
#include <limits>
#include "util/ref_nrm2.h"
#include "inc/check_error.h"

// Used for generic tests with random values in x.
template<typename T>
void test_nrm2( gtint_t n, gtint_t incx, double thresh )
{
    // Get real type from T.
    using RT = typename testinghelpers::type_info<T>::real_type;
    //----------------------------------------------------------
    //        Initialize vectors with random numbers.
    //----------------------------------------------------------
    std::vector<T> x = testinghelpers::get_random_vector<T>( -10, -10, n, incx );

    //----------------------------------------------------------
    //    Call reference implementation to get ref results.
    //----------------------------------------------------------
    RT norm_ref = testinghelpers::ref_nrm2<T>( n, x.data(), incx );

    //----------------------------------------------------------
    //                  Call BLIS function.
    //----------------------------------------------------------
    RT norm = nrm2<T>(n, x.data(), incx);

    //----------------------------------------------------------
    //              Compute error.
    //----------------------------------------------------------
    computediff<RT>( "norm", norm, norm_ref, thresh );
}

// Test body used for extreme value testing, where we want to test
// cases where two extreme values are present.
// i is the index with corresponding extreme value iexval.
// j is the index with corresponding extreme value jexval.
template<typename T>
void test_nrm2( gtint_t n, gtint_t incx, gtint_t i, T iexval, gtint_t j = 0, T jexval = T{1.0})
{
    // Get real type from T.
    using RT = typename testinghelpers::type_info<T>::real_type;
    //----------------------------------------------------------
    //        Initialize vectors with random numbers.
    //----------------------------------------------------------
    std::vector<T> x = testinghelpers::get_random_vector<T>(-10, 10, n, incx);
    // Initialize ith element of vector x to iexval.
    x[i*std::abs(incx)] = iexval;
    // Initialize jth element of vector x to jexval.
    x[j*std::abs(incx)] = jexval;
    //----------------------------------------------------------
    //    Call reference implementation to get ref results.
    //----------------------------------------------------------
    RT norm_ref = testinghelpers::ref_nrm2<T>( n, x.data(), incx );

    //----------------------------------------------------------
    //                  Call BLIS function.
    //----------------------------------------------------------
    RT norm = nrm2<T>(n, x.data(), incx);

    //----------------------------------------------------------
    //              Compute error.
    //----------------------------------------------------------
    // Compare using NaN/Inf checks.
    computediff<RT>( "norm", norm, norm_ref, true );
}

// Test-case logger : Used to print the test-case details based on parameters
class nrm2GenericPrint {
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


// Test-case logger : Used to print the test-case details based on parameters
template <typename T>
class nrm2EVTPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<gtint_t, gtint_t, gtint_t, T, gtint_t, T>> str) const {
        // vector length:
        gtint_t n = std::get<0>(str.param);
        // stride size for x:
        gtint_t incx = std::get<1>(str.param);
        // index with extreme value iexval.
        gtint_t i = std::get<2>(str.param);
        T iexval = std::get<3>(str.param);
        // index with extreme value jexval.
        gtint_t j = std::get<4>(str.param);
        T jexval = std::get<5>(str.param);

        std::string str_name = API_PRINT;
        str_name += "_n_" + std::to_string(n);
        str_name += "_incx_" + testinghelpers::get_value_string(incx);
        str_name    = str_name + "_i" + std::to_string(i);
        std::string iexval_str = testinghelpers::get_value_string(iexval);
        str_name    = str_name + "_" + iexval_str;
        str_name    = str_name + "_j" + std::to_string(j);
        std::string jexval_str = testinghelpers::get_value_string(jexval);
        str_name    = str_name + "_" + jexval_str;
        return str_name;
    }
};
