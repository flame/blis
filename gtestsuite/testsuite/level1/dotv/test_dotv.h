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

#include "dotv.h"
#include "level1/ref_dotv.h"
#include "inc/check_error.h"

/**
 * @brief Generic test body for dotv operation.
 */

template<typename T>
static void test_dotv( char conjx, char conjy, gtint_t n, gtint_t incx,
                       gtint_t incy, double thresh )
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
    if constexpr (testinghelpers::type_info<T>::is_real)
        testinghelpers::ref_dotv<T>( n, x.data(), incx, y_ref.data(), incy, &rho_ref );
    else
        testinghelpers::ref_dotv<T>( conjx, conjy, n, x.data(), incx, y_ref.data(), incy, &rho_ref );

    //----------------------------------------------------------
    //                  Call BLIS function.
    //----------------------------------------------------------
    T rho;
    dotv<T>( conjx, conjy, n, x.data(), incx, y.data(), incy, &rho );

    //----------------------------------------------------------
    //              Compute error.
    //----------------------------------------------------------
    computediff<T>( "rho", rho, rho_ref, thresh );
}


/**
 * @brief Used to insert Exception Values in vectors x and y.
 */
template<typename T>
static void test_dotv( char conjx, char conjy, gtint_t n,
                       gtint_t incx, gtint_t xi, double x_exval,
                       gtint_t incy, gtint_t yi, double y_exval,
                       double thresh )
{
    //----------------------------------------------------------
    //        Initialize vectors with random numbers.
    //----------------------------------------------------------
    std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, n, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( -10, 10, n, incy );

    // Update the value at index xi to an extreme value, x_exval.
    if ( -1 < xi && xi < n ) x[xi * abs(incx)] = x_exval;
    else                     return;

    // Update the value at index yi to an extreme value, y_exval.
    if ( -1 < yi && yi < n ) y[yi * abs(incy)] = y_exval;
    else                     return;

    //----------------------------------------------------------
    //    Call reference implementation to get ref results.
    //----------------------------------------------------------
    // Create a copy of y so that we can check reference results.
    std::vector<T> y_ref(y);
    T rho_ref;
    if constexpr (testinghelpers::type_info<T>::is_real)
        testinghelpers::ref_dotv<T>( n, x.data(), incx, y_ref.data(), incy, &rho_ref );
    else
        testinghelpers::ref_dotv<T>( conjx, conjy, n, x.data(), incx, y_ref.data(), incy, &rho_ref );

    //----------------------------------------------------------
    //                  Call BLIS function.
    //----------------------------------------------------------
    T rho;
    dotv<T>( conjx, conjy, n, x.data(), incx, y.data(), incy, &rho );

    //----------------------------------------------------------
    //              Compute error.
    //----------------------------------------------------------
    computediff<T>( "rho", rho, rho_ref, thresh, true);
}


// Test-case logger : Used to print the test-case details based on parameters
class dotvGenericPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char,char,gtint_t,gtint_t,gtint_t>> str) const {
        char conjx    = std::get<0>(str.param);
        char conjy    = std::get<1>(str.param);
        gtint_t n     = std::get<2>(str.param);
        gtint_t incx  = std::get<3>(str.param);
        gtint_t incy  = std::get<4>(str.param);

        std::string str_name = API_PRINT;
        str_name += "_n_" + std::to_string(n);
        str_name += "_conjx_" + std::string(&conjx, 1);
        str_name += "_conjy_" + std::string(&conjy, 1);
        str_name += "_incx_" + testinghelpers::get_value_string(incx);
        str_name += "_incy_" + testinghelpers::get_value_string(incy);
        return str_name;
    }
};

template <typename T>
class dotvEVTPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char,char,gtint_t,gtint_t,gtint_t, T, gtint_t, gtint_t, T>> str) const {
        char    conjx   = std::get<0>(str.param);
        char    conjy   = std::get<1>(str.param);
        gtint_t n       = std::get<2>(str.param);
        gtint_t incx    = std::get<3>(str.param);
        gtint_t xi      = std::get<4>(str.param);
        T  x_exval = std::get<5>(str.param);
        gtint_t incy    = std::get<6>(str.param);
        gtint_t yi      = std::get<7>(str.param);
        T  y_exval = std::get<8>(str.param);

        std::string str_name = API_PRINT;
        str_name += "_n_" + std::to_string(n);
        str_name += "_conjx_" + std::string(&conjx, 1);
        str_name += "_conjy_" + std::string(&conjy, 1);
        str_name += "_incx_" + testinghelpers::get_value_string(incx);
        str_name = str_name + "_X_" + std::to_string(xi);
        str_name = str_name + "_" + testinghelpers::get_value_string(x_exval);
        str_name += "_incy_" + testinghelpers::get_value_string(incy);
        str_name = str_name + "_Y_" + std::to_string(yi);
        str_name = str_name + "_" + testinghelpers::get_value_string(y_exval);

        return str_name;
    }
};
