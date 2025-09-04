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

#include <gtest/gtest.h>
#include "test_dotv.h"
#include "common/wrong_inputs_helpers.h"
#include "common/testing_helpers.h"
#include "inc/check_error.h"

template <typename T>
class dotv_IIT_ERS : public ::testing::Test {};
typedef ::testing::Types<float, double, scomplex, dcomplex> TypeParam;
TYPED_TEST_SUITE(dotv_IIT_ERS, TypeParam);

using namespace testinghelpers::IIT;

#if defined(TEST_BLAS_LIKE) || defined(TEST_CBLAS)

/*
    BLAS Early Return Scenarios(ERS):

    DOTV is expected to return early in the following cases:
    1. n <= 0
*/

// n < 0, with non-unit stride
TYPED_TEST(dotv_IIT_ERS, n_lt_zero_nonUnitStride)
{
    using T = TypeParam;
    gtint_t invalid_n = -1;
    gtint_t inc = 5;
    // Initialize rho (BLIS output) to garbage value.
    T rho = T{-7.3};
    // Initialize the expected output to zero.
    T rho_ref;
    testinghelpers::initzero<T>(rho_ref);

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    dotv<T>( CONJ, CONJ, invalid_n, nullptr, inc, nullptr, inc, &rho );
    // Computing the difference.
    computediff<T>( "rho", rho, rho_ref );

    // Test with all arguments correct except for the value we are choosing to test.
    // Initialize vectors with random numbers.
    std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, N, inc );
    std::vector<T> y = testinghelpers::get_random_vector<T>( -10, 10, N, inc );

    // Invoking DOTV with an invalid value of n.
    dotv<T>( CONJ, CONJ, invalid_n, x.data(), inc, y.data(), inc, &rho );

    // Computing the difference.
    computediff<T>( "rho", rho, rho_ref );
}

// n == 0, with non-unit stride
TYPED_TEST(dotv_IIT_ERS, n_eq_zero_nonUnitStride)
{
    using T = TypeParam;
    gtint_t invalid_n = 0;
    gtint_t inc = 5;
    // Initialize rho (BLIS output) to garbage value.
    T rho = T{-7.3};
    // Initialize the expected output to zero.
    T rho_ref;
    testinghelpers::initzero<T>(rho_ref);

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    dotv<T>( CONJ, CONJ, invalid_n, nullptr, inc, nullptr, inc, &rho );
    // Computing the difference.
    computediff<T>( "rho", rho, rho_ref );

    // Test with all arguments correct except for the value we are choosing to test.
    // Initialize vectors with random numbers.
    std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, N, inc );
    std::vector<T> y = testinghelpers::get_random_vector<T>( -10, 10, N, inc );

    // Invoking DOTV with an invalid value of n.
    dotv<T>( CONJ, CONJ, invalid_n, x.data(), inc, y.data(), inc, &rho );

    // Computing the difference.
    computediff<T>( "rho", rho, rho_ref );
}

// n < 0, with unit stride
TYPED_TEST(dotv_IIT_ERS, n_lt_zero_unitStride)
{
    using T = TypeParam;
    gtint_t invalid_n = -1;
    gtint_t unit_inc = 1;
    // Initialize rho (BLIS output) to garbage value.
    T rho = T{-7.3};
    // Initialize the expected output to zero.
    T rho_ref;
    testinghelpers::initzero<T>(rho_ref);

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    dotv<T>( CONJ, CONJ, invalid_n, nullptr, unit_inc, nullptr, unit_inc, &rho );
    // Computing the difference.
    computediff<T>( "rho", rho, rho_ref );

    // Test with all arguments correct except for the value we are choosing to test.
    // Initialize vectors with random numbers.
    std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, N, unit_inc );
    std::vector<T> y = testinghelpers::get_random_vector<T>( -10, 10, N, unit_inc );

    // Invoking DOTV with an invalid value of n.
    dotv<T>( CONJ, CONJ, invalid_n, x.data(), unit_inc, y.data(), unit_inc, &rho );

    // Computing the difference.
    computediff<T>( "rho", rho, rho_ref );
}

// n == 0, with unit stride
TYPED_TEST(dotv_IIT_ERS, n_eq_zero_unitStride)
{
    using T = TypeParam;
    gtint_t invalid_n = 0;
    gtint_t unit_inc = 1;
    // Initialize rho (BLIS output) to garbage value.
    T rho = T{-7.3};
    // Initialize the expected output to zero.
    T rho_ref;
    testinghelpers::initzero<T>(rho_ref);

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    dotv<T>( CONJ, CONJ, invalid_n, nullptr, unit_inc, nullptr, unit_inc, &rho );
    // Computing the difference.
    computediff<T>( "rho", rho, rho_ref );

    // Test with all arguments correct except for the value we are choosing to test.
    // Initialize vectors with random numbers.
    std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, N, unit_inc );
    std::vector<T> y = testinghelpers::get_random_vector<T>( -10, 10, N, unit_inc );

    // Invoking DOTV with an invalid value of n.
    dotv<T>( CONJ, CONJ, invalid_n, x.data(), unit_inc, y.data(), unit_inc, &rho );

    // Computing the difference.
    computediff<T>( "rho", rho, rho_ref );
}
#endif
