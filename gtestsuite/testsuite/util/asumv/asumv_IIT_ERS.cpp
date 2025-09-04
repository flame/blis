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
#include "test_asumv.h"
#include "common/wrong_inputs_helpers.h"
#include "common/testing_helpers.h"
#include "inc/check_error.h"

template <typename T>
class asumv_IIT_ERS : public ::testing::Test {};
typedef ::testing::Types<float, double, scomplex, dcomplex> TypeParam;
TYPED_TEST_SUITE(asumv_IIT_ERS, TypeParam);

using namespace testinghelpers::IIT;

#if defined(TEST_BLAS_LIKE) || defined(TEST_CBLAS)

/*
    BLAS Early Return Scenarios(ERS):

    ASUMV is expected to return early in the following cases:
    1. n <= 0
    2. inc <= 0
*/

// n < 0, with non-unit stride
TYPED_TEST(asumv_IIT_ERS, n_lt_zero_nonUnitStride)
{
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    gtint_t invalid_n = -1;
    gtint_t inc = 5;
    // Initialize asum (BLIS output) to garbage value.
    RT asum = RT{-7.3};
    // Initialize the expected output to zero.
    RT asum_ref;
    testinghelpers::initzero<RT>(asum_ref);

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    asum = asumv<T>( invalid_n, nullptr, inc );
    // Computing the difference.
    computediff<RT>( "asum", asum, asum_ref );

    // Test with all arguments correct except for the value we are choosing to test.
    // Initialize x vector with random numbers.
    std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, N, inc );

    // Invoking asumV with an invalid value of n.
    asum = asumv<T>( invalid_n, x.data(), inc );

    // Computing the difference.
    computediff<RT>( "asum", asum, asum_ref );
}

// n == 0, with non-unit stride
TYPED_TEST(asumv_IIT_ERS, n_eq_zero_nonUnitStride)
{
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    gtint_t invalid_n = 0;
    gtint_t inc = 5;
    // Initialize asum (BLIS output) to garbage value.
    RT asum = RT{-7.3};
    // Initialize the expected output to zero.
    RT asum_ref;
    testinghelpers::initzero<RT>(asum_ref);

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    asum = asumv<T>( invalid_n, nullptr, inc );
    // Computing the difference.
    computediff<RT>( "asum", asum, asum_ref );

    // Test with all arguments correct except for the value we are choosing to test.
    // Initialize x vector with random numbers.
    std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, N, inc );

    // Invoking asumV with an invalid value of n.
    asum = asumv<T>( invalid_n, x.data(), inc );

    // Computing the difference.
    computediff<RT>( "asum", asum, asum_ref );
}

// n < 0, with unit stride
TYPED_TEST(asumv_IIT_ERS, n_lt_zero_unitStride)
{
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    gtint_t invalid_n = -1;
    gtint_t unit_inc = 1;
    // Initialize asum (BLIS output) to garbage value.
    RT asum = RT{-7.3};
    // Initialize the expected output to zero.
    RT asum_ref;
    testinghelpers::initzero<RT>(asum_ref);

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    asum = asumv<T>( invalid_n, nullptr, unit_inc );
    // Computing the difference.
    computediff<RT>( "asum", asum, asum_ref );

    // Test with all arguments correct except for the value we are choosing to test.
    // Initialize x vector with random numbers.
    std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, N, unit_inc );

    // Invoking asumV with an invalid value of n.
    asum = asumv<T>( invalid_n, x.data(), unit_inc );

    // Computing the difference.
    computediff<RT>( "asum", asum, asum_ref );
}

// n == 0, with unit stride
TYPED_TEST(asumv_IIT_ERS, n_eq_zero_unitStride)
{
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    gtint_t invalid_n = 0;
    gtint_t unit_inc = 1;
    // Initialize asum (BLIS output) to garbage value.
    RT asum = RT{-7.3};
    // Initialize the expected output to zero.
    RT asum_ref;
    testinghelpers::initzero<RT>(asum_ref);

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    asum = asumv<T>( invalid_n, nullptr, unit_inc );
    // Computing the difference.
    computediff<RT>( "asum", asum, asum_ref );

    // Test with all arguments correct except for the value we are choosing to test.
    // Initialize x vector with random numbers.
    std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, N, unit_inc );

    // Invoking asumV with an invalid value of n.
    asum = asumv<T>( invalid_n, x.data(), unit_inc );

    // Computing the difference.
    computediff<RT>( "asum", asum, asum_ref );
}

// inc < 0
TYPED_TEST(asumv_IIT_ERS, inc_lt_0)
{
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    gtint_t invalid_inc = -1;
    // Initialize asum (BLIS output) to garbage value.
    RT asum = RT{-7.3};
    // Initialize the expected output to zero.
    RT asum_ref;
    testinghelpers::initzero<RT>(asum_ref);

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    asum = asumv<T>( N, nullptr, invalid_inc );
    // Computing the difference.
    computediff<RT>( "asum", asum, asum_ref );

    // Test with all arguments correct except for the value we are choosing to test.
    // Initialize x vector with random numbers.
    std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, N, INC );

    // Invoking asumV with an invalid value of n.
    asum = asumv<T>( N, x.data(), invalid_inc );

    // Computing the difference.
    computediff<RT>( "asum", asum, asum_ref );
}

// inc == 0
TYPED_TEST(asumv_IIT_ERS, inc_eq_0)
{
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    gtint_t invalid_inc = 0;
    // Initialize asum (BLIS output) to garbage value.
    RT asum = RT{-7.3};
    // Initialize the expected output to zero.
    RT asum_ref;
    testinghelpers::initzero<RT>(asum_ref);

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    asum = asumv<T>( N, nullptr, invalid_inc );
    // Computing the difference.
    computediff<RT>( "asum", asum, asum_ref );

    // Test with all arguments correct except for the value we are choosing to test.
    // Initialize x vector with random numbers.
    std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, N, INC );

    // Invoking asumV with an invalid value of n.
    asum = asumv<T>( N, x.data(), invalid_inc );

    // Computing the difference.
    computediff<RT>( "asum", asum, asum_ref );
}
#endif
