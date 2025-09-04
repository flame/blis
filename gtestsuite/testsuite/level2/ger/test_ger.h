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

#include "ger.h"
#include "level2/ref_ger.h"
#include "inc/check_error.h"
#include <stdexcept>
#include <algorithm>

template<typename T>
void test_ger( char storage, char conjx, char conjy, gtint_t m, gtint_t n,
    T alpha, gtint_t incx, gtint_t incy, gtint_t lda_inc, double thresh )
{
    // Compute the leading dimensions for matrix size calculation.
    gtint_t lda = testinghelpers::get_leading_dimension( storage, 'n', m, n, lda_inc );

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>( -2, 5, storage, 'n', m, n, lda );
    std::vector<T> x = testinghelpers::get_random_vector<T>( -3, 3, m, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( -3, 3, n, incy );

    // Create a copy of c so that we can check reference results.
    std::vector<T> a_ref(a);
    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    ger<T>( storage, conjx, conjy, m, n, &alpha, x.data(), incx,
                                              y.data(), incy, a.data(), lda );

    //----------------------------------------------------------
    //                  Call reference implementation.
    //----------------------------------------------------------
    testinghelpers::ref_ger<T>( storage, conjx, conjy, m, n, alpha,
                          x.data(), incx, y.data(), incy, a_ref.data(), lda );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( "a", storage, m, n, a.data(), a_ref.data(), lda, thresh );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

template<typename T>
void test_ger( char storage, char conjx, char conjy, gtint_t m, gtint_t n,
               T alpha, gtint_t incx, gtint_t incy, gtint_t lda_inc, gtint_t ai,
               gtint_t aj, T a_exval, gtint_t xi, T x_exval, gtint_t yi,
               T y_exval, double thresh )
{
    // Compute the leading dimensions for matrix size calculation.
    gtint_t lda = testinghelpers::get_leading_dimension( storage, 'n', m, n, lda_inc );

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>( -2, 5, storage, 'n', m, n, lda );
    std::vector<T> x = testinghelpers::get_random_vector<T>( -3, 3, m, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( -3, 3, n, incy );

    testinghelpers::set_ev_mat( storage, 'n', lda, ai, aj, a_exval, a.data() );
    // Update the value at index xi to an extreme value, x_exval.
    if ( -1 < xi && xi < n ) x[xi * abs(incx)] = x_exval;
    else                     return;

    // Update the value at index yi to an extreme value, y_exval.
    if ( -1 < yi && yi < n ) y[yi * abs(incy)] = y_exval;
    else                     return;

    // Create a copy of c so that we can check reference results.
    std::vector<T> a_ref(a);
    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    ger<T>( storage, conjx, conjy, m, n, &alpha, x.data(), incx,
                                              y.data(), incy, a.data(), lda );

    //----------------------------------------------------------
    //                  Call reference implementation.
    //----------------------------------------------------------
    testinghelpers::ref_ger<T>( storage, conjx, conjy, m, n, alpha,
                          x.data(), incx, y.data(), incy, a_ref.data(), lda );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( "A", storage, m, n, a.data(), a_ref.data(), lda, thresh, true );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// Test-case logger : Used to print the test-case details based on parameters
template <typename T>
class gerGenericPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char,char,char,gtint_t,gtint_t,T,gtint_t,gtint_t,gtint_t>> str) const {
        char storage    = std::get<0>(str.param);
        char conjx      = std::get<1>(str.param);
        char conjy      = std::get<2>(str.param);
        gtint_t m       = std::get<3>(str.param);
        gtint_t n       = std::get<4>(str.param);
        T alpha         = std::get<5>(str.param);
        gtint_t incx    = std::get<6>(str.param);
        gtint_t incy    = std::get<7>(str.param);
        gtint_t lda_inc = std::get<8>(str.param);

        std::string str_name = API_PRINT;
        str_name += "_stor_" + std::string(&storage, 1);
        str_name += "_conjx_" + std::string(&conjx, 1);
        str_name += "_conjy_" + std::string(&conjy, 1);
        str_name += "_m_" + std::to_string(m);
        str_name += "_n_" + std::to_string(n);
        str_name += "_incx_" + testinghelpers::get_value_string(incx);
        str_name += "_incy_" + testinghelpers::get_value_string(incy);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        gtint_t lda = testinghelpers::get_leading_dimension( storage, 'n', m, n, lda_inc );
        str_name += "_lda_i" + std::to_string(lda_inc) + "_" + std::to_string(lda);
        return str_name;
    }
};

template <typename T>
class gerEVTPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char,char,char,gtint_t,gtint_t,T,gtint_t,gtint_t,gtint_t,gtint_t,gtint_t,T,gtint_t,T,gtint_t,T>> str) const {
        char storage    = std::get<0>(str.param);
        char conjx      = std::get<1>(str.param);
        char conjy      = std::get<2>(str.param);
        gtint_t m       = std::get<3>(str.param);
        gtint_t n       = std::get<4>(str.param);
        T alpha         = std::get<5>(str.param);
        gtint_t incx    = std::get<6>(str.param);
        gtint_t incy    = std::get<7>(str.param);
        gtint_t lda_inc = std::get<8>(str.param);
        gtint_t ai      = std::get<9>(str.param);
        gtint_t aj      = std::get<10>(str.param);
        T a_exval       = std::get<11>(str.param);
        gtint_t xi      = std::get<12>(str.param);
        T x_exval       = std::get<13>(str.param);
        gtint_t yi      = std::get<14>(str.param);
        T y_exval       = std::get<15>(str.param);

        std::string str_name = API_PRINT;
        str_name += "_stor_" + std::string(&storage, 1);
        str_name += "_conjx_" + std::string(&conjx, 1);
        str_name += "_conjy_" + std::string(&conjy, 1);
        str_name += "_m_" + std::to_string(m);
        str_name += "_n_" + std::to_string(n);
        str_name += "_incx_" + testinghelpers::get_value_string(incx);
        str_name += "_incy_" + testinghelpers::get_value_string(incy);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        gtint_t lda = testinghelpers::get_leading_dimension( storage, 'n', m, n, lda_inc );
        str_name += "_lda_i" + std::to_string(lda_inc) + "_" + std::to_string(lda);
        str_name    = str_name + "_ai" + std::to_string(ai);
        str_name    = str_name + "_aj" + std::to_string(aj);
        str_name    = str_name + "_a_exval_" + testinghelpers::get_value_string(a_exval);
        str_name    = str_name + "_xi" + std::to_string(xi);
        str_name    = str_name + "_x_exval_" + testinghelpers::get_value_string(x_exval);
        str_name    = str_name + "_yi" + std::to_string(yi);
        str_name    = str_name + "_y_exval_" + testinghelpers::get_value_string(y_exval);

        return str_name;
    }
};
