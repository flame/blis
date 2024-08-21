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
#include "test_scalv.h"
#include "common/wrong_inputs_helpers.h"
#include "common/testing_helpers.h"
#include "inc/check_error.h"

template <typename T>
class scalv_IIT_ERS : public ::testing::Test {};
typedef ::testing::Types<
                    // std::pair<type, real_type>
                    std::pair<   float,    float>,
                    std::pair<  double,   double>,
                    std::pair<scomplex, scomplex>,
                    std::pair<dcomplex, dcomplex>,
                    std::pair<scomplex,    float>,
                    std::pair<dcomplex,   double>
                    > TypeParam;
TYPED_TEST_SUITE(scalv_IIT_ERS, TypeParam);

using namespace testinghelpers::IIT;

#if defined(TEST_BLAS_LIKE) || defined(TEST_CBLAS)

/*
    BLAS Early Return Scenarios(ERS):

    SCALV is expected to return early in the following cases:
    1. n <= 0
    2. inc <= 0
    3. alpha == 1
*/

// n < 0, with non-unit stride
TYPED_TEST(scalv_IIT_ERS, n_lt_zero_nonUnitStride)
{
    using  T = typename TypeParam::first_type;
    using RT = typename TypeParam::second_type;
    gtint_t invalid_n = -1;
    gtint_t inc = 5;
    // Using alpha = 3 as a valid input since BLAS expects SCALV to return early
    // for alpha = 1.
    RT alpha = RT{3};

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    scalv<T, RT>( 'n', invalid_n, alpha, nullptr, inc );

    // Test with all arguments correct except for the value we are choosing to test.
    // Initialize x vector with random numbers.
    std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, N, inc );
    std::vector<T> x_ref(x);    // copy x to x_ref to verify elements of x are not modified.

    // Invoking SCALV with an invalid value of n.
    scalv<T, RT>( 'n', invalid_n, alpha, x.data(), inc );

    // Computing bitwise difference.
    computediff<T>( "x", N, x.data(), x_ref.data(), inc );
}

// n == 0, with non-unit stride
TYPED_TEST(scalv_IIT_ERS, n_eq_zero_nonUnitStride)
{
    using  T = typename TypeParam::first_type;
    using RT = typename TypeParam::second_type;
    gtint_t invalid_n = 0;
    gtint_t inc = 5;
    // Using alpha = 3 as a valid input since BLAS expects SCALV to return early
    // for alpha = 1.
    RT alpha = RT{3};

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    scalv<T, RT>( 'n', invalid_n, alpha, nullptr, inc );

    // Test with all arguments correct except for the value we are choosing to test.
    // Initialize x vector with random numbers.
    std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, N, inc );
    std::vector<T> x_ref(x);    // copy x to x_ref to verify elements of x are not modified.

    // Invoking SCALV with an invalid value of n.
    scalv<T, RT>( 'n', invalid_n, alpha, x.data(), inc );

    // Computing bitwise difference.
    computediff<T>( "x", N, x.data(), x_ref.data(), inc );
}

// n < 0, with unit stride
TYPED_TEST(scalv_IIT_ERS, n_lt_zero_unitStride)
{
    using  T = typename TypeParam::first_type;
    using RT = typename TypeParam::second_type;
    gtint_t invalid_n = -1;
    gtint_t unit_inc = 1;
    // Using alpha = 3 as a valid input since BLAS expects SCALV to return early
    // for alpha = 1.
    RT alpha = RT{3};

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    scalv<T, RT>( 'n', invalid_n, alpha, nullptr, unit_inc );

    // Test with all arguments correct except for the value we are choosing to test.
    // Initialize x vector with random numbers.
    std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, N, unit_inc );
    std::vector<T> x_ref(x);    // copy x to x_ref to verify elements of x are not modified.

    // Invoking SCALV with an invalid value of n.
    scalv<T, RT>( 'n', invalid_n, alpha, x.data(), unit_inc );

    // Computing bitwise difference.
    computediff<T>( "x", N, x.data(), x_ref.data(), unit_inc );
}

// n == 0, with unit stride
TYPED_TEST(scalv_IIT_ERS, n_eq_zero_unitStride)
{
    using  T = typename TypeParam::first_type;
    using RT = typename TypeParam::second_type;
    gtint_t invalid_n = 0;
    gtint_t unit_inc = 1;
    // Using alpha = 3 as a valid input since BLAS expects SCALV to return early
    // for alpha = 1.
    RT alpha = RT{3};

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    scalv<T, RT>( 'n', invalid_n, alpha, nullptr, unit_inc );

    // Test with all arguments correct except for the value we are choosing to test.
    // Initialize x vector with random numbers.
    std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, N, unit_inc );
    std::vector<T> x_ref(x);    // copy x to x_ref to verify elements of x are not modified.

    // Invoking SCALV with an invalid value of n.
    scalv<T, RT>( 'n', invalid_n, alpha, x.data(), unit_inc );

    // Computing bitwise difference.
    computediff<T>( "x", N, x.data(), x_ref.data(), unit_inc );
}

// inc < 0
TYPED_TEST(scalv_IIT_ERS, inc_lt_0)
{
    using  T = typename TypeParam::first_type;
    using RT = typename TypeParam::second_type;
    gtint_t invalid_inc = -1;
    // Using alpha = 3 as a valid input since BLAS expects SCALV to return early
    // for alpha = 1.
    RT alpha = RT{3};

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    scalv<T, RT>( 'n', N, alpha, nullptr, invalid_inc );

    // Test with all arguments correct except for the value we are choosing to test.
    // Initialize x vector with random numbers.
    std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, N, INC );
    std::vector<T> x_ref(x);    // copy x to x_ref to verify elements of x are not modified.

    // Invoking SCALV with an invalid value of n.
    scalv<T, RT>( 'n', N, alpha, x.data(), invalid_inc );

    // Computing bitwise difference.
    computediff<T>( "x", N, x.data(), x_ref.data(), INC );
}

// inc == 0
TYPED_TEST(scalv_IIT_ERS, inc_eq_0)
{
    using  T = typename TypeParam::first_type;
    using RT = typename TypeParam::second_type;
    gtint_t invalid_inc = 0;
    // Using alpha = 3 as a valid input since BLAS expects SCALV to return early
    // for alpha = 1.
    RT alpha = RT{3};

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    scalv<T, RT>( 'n', N, alpha, nullptr, invalid_inc );

    // Test with all arguments correct except for the value we are choosing to test.
    // Initialize x vector with random numbers.
    std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, N, INC );
    std::vector<T> x_ref(x);    // copy x to x_ref to verify elements of x are not modified.

    // Invoking SCALV with an invalid value of n.
    scalv<T, RT>( 'n', N, alpha, x.data(), invalid_inc );

    // Computing bitwise difference.
    computediff<T>( "x", N, x.data(), x_ref.data(), INC );
}

// alpha == 1, with non-unit stride
TYPED_TEST(scalv_IIT_ERS, alpha_eq_one_nonUnitStride)
{
    using  T = typename TypeParam::first_type;
    using RT = typename TypeParam::second_type;
    gtint_t inc = 5;
    RT invalid_alpha;
    testinghelpers::initone<RT>(invalid_alpha);

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    scalv<T, RT>( 'n', N, invalid_alpha, nullptr, inc );

    // Test with all arguments correct except for the value we are choosing to test.
    // Initialize x vector with random numbers.
    std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, N, inc );
    std::vector<T> x_ref(x);    // copy x to x_ref to verify elements of x are not modified.

    // Invoking SCALV with an invalid value of n.
    scalv<T, RT>( 'n', N, invalid_alpha, x.data(), inc );

    // Computing bitwise difference.
    computediff<T>( "x", N, x.data(), x_ref.data(), inc );
}

// alpha == 1, with unit stride
TYPED_TEST(scalv_IIT_ERS, alpha_eq_one_unitStride)
{
    using  T = typename TypeParam::first_type;
    using RT = typename TypeParam::second_type;
    gtint_t unit_inc = 1;
    RT invalid_alpha;
    testinghelpers::initone<RT>(invalid_alpha);

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    scalv<T, RT>( 'n', N, invalid_alpha, nullptr, unit_inc );

    // Test with all arguments correct except for the value we are choosing to test.
    // Initialize x vector with random numbers.
    std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, N, unit_inc );
    std::vector<T> x_ref(x);    // copy x to x_ref to verify elements of x are not modified.

    // Invoking SCALV with an invalid value of n.
    scalv<T, RT>( 'n', N, invalid_alpha, x.data(), unit_inc );

    // Computing bitwise difference.
    computediff<T>( "x", N, x.data(), x_ref.data(), unit_inc );
}
#endif
