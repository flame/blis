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
#include "test_ger.h"
#include "common/wrong_inputs_helpers.h"
#include "common/testing_helpers.h"
#include "inc/check_error.h"


template <typename T>
class ger_IIT_ERS_Test : public ::testing::Test {};
typedef ::testing::Types<float, double, scomplex, dcomplex> TypeParam;

TYPED_TEST_SUITE(ger_IIT_ERS_Test, TypeParam);

using namespace testinghelpers::IIT;

#if defined(TEST_BLAS) || defined(TEST_CBLAS)
/**
 * BLAS Early Return Scenarios(ERS):
 *
 * GER is expected to return early in the following cases:
 * 1. m == 0
 * 2. n == 0
 * 3. alpha == 0
 */
// m == 0, with unit stride
TYPED_TEST(ger_IIT_ERS_Test, m_eq_zero_unitStride)
{
    using T = TypeParam;
    gtint_t invalid_m = 0;
    gtint_t unit_inc = 1;

    std::vector<T> a = testinghelpers::get_random_matrix<T>( -2, 5, STORAGE, 'n', M, N, LDA );
    std::vector<T> x = testinghelpers::get_random_vector<T>( -3, 3, M, unit_inc );
    std::vector<T> y = testinghelpers::get_random_vector<T>( -3, 3, N, unit_inc );

    // Create a copy of a matrix so that we can check reference results.
    std::vector<T> a_ref(a);

    // Using a random non-zero value of alpha.
    T alpha = T{3};

    // Invoking GER with an invalid value of n.
    ger<T>( STORAGE, CONJ, CONJ, invalid_m, N, &alpha, x.data(), unit_inc,
            y.data(), unit_inc, a.data(), LDA );

    // Computing bitwise difference.
    computediff<T>( "A", STORAGE, M, N, a.data(), a_ref.data(), LDA );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// m == 0, with non-unit stride
TYPED_TEST(ger_IIT_ERS_Test, m_eq_zero_nonUnitStride)
{
    using T = TypeParam;
    gtint_t invalid_m = 0;
    gtint_t inc = 3;

    std::vector<T> a = testinghelpers::get_random_matrix<T>( -2, 5, STORAGE, 'n', M, N, LDA );
    std::vector<T> x = testinghelpers::get_random_vector<T>( -3, 3, M, inc );
    std::vector<T> y = testinghelpers::get_random_vector<T>( -3, 3, N, inc );

    // Create a copy of a matrix so that we can check reference results.
    std::vector<T> a_ref(a);

    // Using a random non-zero value of alpha.
    T alpha = T{3};

    // Invoking GER with an invalid value of n.
    ger<T>( STORAGE, CONJ, CONJ, invalid_m, N, &alpha, x.data(), inc,
            y.data(), inc, a.data(), LDA );

    // Computing bitwise difference.
    computediff<T>( "A", STORAGE, M, N, a.data(), a_ref.data(), LDA );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// n == 0, with unit stride
TYPED_TEST(ger_IIT_ERS_Test, n_eq_zero_unitStride)
{
    using T = TypeParam;
    gtint_t invalid_n = 0;
    gtint_t unit_inc = 1;

    std::vector<T> a = testinghelpers::get_random_matrix<T>( -2, 5, STORAGE, 'n', M, N, LDA );
    std::vector<T> x = testinghelpers::get_random_vector<T>( -3, 3, M, unit_inc );
    std::vector<T> y = testinghelpers::get_random_vector<T>( -3, 3, N, unit_inc );

    // Create a copy of a matrix so that we can check reference results.
    std::vector<T> a_ref(a);

    // Using a random non-zero value of alpha.
    T alpha = T{3};

    // Invoking GER with an invalid value of n.
    ger<T>( STORAGE, CONJ, CONJ, M, invalid_n, &alpha, x.data(), unit_inc,
            y.data(), unit_inc, a.data(), LDA );

    // Computing bitwise difference.
    computediff<T>( "A", STORAGE, M, N, a.data(), a_ref.data(), LDA );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// n == 0, with non-unit stride
TYPED_TEST(ger_IIT_ERS_Test, n_eq_zero_nonUnitStride)
{
    using T = TypeParam;
    gtint_t invalid_n = 0;
    gtint_t inc = 3;

    std::vector<T> a = testinghelpers::get_random_matrix<T>( -2, 5, STORAGE, 'n', M, N, LDA );
    std::vector<T> x = testinghelpers::get_random_vector<T>( -3, 3, M, inc );
    std::vector<T> y = testinghelpers::get_random_vector<T>( -3, 3, N, inc );

    // Create a copy of a matrix so that we can check reference results.
    std::vector<T> a_ref(a);

    // Using a random non-zero value of alpha.
    T alpha = T{3};

    // Invoking GER with an invalid value of n.
    ger<T>( STORAGE, CONJ, CONJ, M, invalid_n, &alpha, x.data(), inc,
            y.data(), inc, a.data(), LDA );

    // Computing bitwise difference.
    computediff<T>( "A", STORAGE, M, N, a.data(), a_ref.data(), LDA );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// alpha == 0, with unit stride
TYPED_TEST(ger_IIT_ERS_Test, alpha_eq_zero_unitStride)
{
    using T = TypeParam;
    gtint_t unit_inc = 1;

    std::vector<T> a = testinghelpers::get_random_matrix<T>( -2, 5, STORAGE, 'n', M, N, LDA );
    std::vector<T> x = testinghelpers::get_random_vector<T>( -3, 3, M, unit_inc );
    std::vector<T> y = testinghelpers::get_random_vector<T>( -3, 3, N, unit_inc );

    // Create a copy of a matrix so that we can check reference results.
    std::vector<T> a_ref(a);

    T zero_alpha = T{0};

    // Invoking GER with an invalid value of n.
    ger<T>( STORAGE, CONJ, CONJ, M, N, &zero_alpha, x.data(), unit_inc,
            y.data(), unit_inc, a.data(), LDA );

    // Computing bitwise difference.
    computediff<T>( "A", STORAGE, M, N, a.data(), a_ref.data(), LDA );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// alpha == 0, with non-unit stride
TYPED_TEST(ger_IIT_ERS_Test, alpha_eq_zero_nonUnitStride)
{
    using T = TypeParam;
    gtint_t inc = 3;

    std::vector<T> a = testinghelpers::get_random_matrix<T>( -2, 5, STORAGE, 'n', M, N, LDA );
    std::vector<T> x = testinghelpers::get_random_vector<T>( -3, 3, M, inc );
    std::vector<T> y = testinghelpers::get_random_vector<T>( -3, 3, N, inc );

    // Create a copy of a matrix so that we can check reference results.
    std::vector<T> a_ref(a);

    T zero_alpha = T{3};

    // Invoking GER with an invalid value of n.
    ger<T>( STORAGE, CONJ, CONJ, M, N, &zero_alpha, x.data(), inc,
            y.data(), inc, a.data(), LDA );

    // Computing bitwise difference.
    computediff<T>( "A", STORAGE, M, N, a.data(), a_ref.data(), LDA );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}


/**
 * BLAS Invalid Input Tests(IIT):
 *
 * Following conditions are considered as Invalid Inputs for GER:
 * 1. m < 0
 * 2. n < 0
 * 3. incx = 0
 * 4. incy = 0
 * 5. lda < max(1, m)
 */
// m < 0, with unit stride
TYPED_TEST(ger_IIT_ERS_Test, m_lt_zero_unitStride)
{
    using T = TypeParam;
    gtint_t invalid_m = -1;
    gtint_t unit_inc = 1;

    std::vector<T> a = testinghelpers::get_random_matrix<T>( -2, 5, STORAGE, 'n', M, N, LDA );
    std::vector<T> x = testinghelpers::get_random_vector<T>( -3, 3, M, unit_inc );
    std::vector<T> y = testinghelpers::get_random_vector<T>( -3, 3, N, unit_inc );

    // Create a copy of a matrix so that we can check reference results.
    std::vector<T> a_ref(a);

    // Using a random non-zero value of alpha.
    T alpha = T{3};

    // Invoking GER with an invalid value of n.
    ger<T>( STORAGE, CONJ, CONJ, invalid_m, N, &alpha, x.data(), unit_inc,
            y.data(), unit_inc, a.data(), LDA );

    // Computing bitwise difference.
    computediff<T>( "A", STORAGE, M, N, a.data(), a_ref.data(), LDA );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 1 );
#endif
}

// m < 0, with non-unit stride
TYPED_TEST(ger_IIT_ERS_Test, m_lt_zero_nonUnitStride)
{
    using T = TypeParam;
    gtint_t invalid_m = -1;
    gtint_t inc = 3;

    std::vector<T> a = testinghelpers::get_random_matrix<T>( -2, 5, STORAGE, 'n', M, N, LDA );
    std::vector<T> x = testinghelpers::get_random_vector<T>( -3, 3, M, inc );
    std::vector<T> y = testinghelpers::get_random_vector<T>( -3, 3, N, inc );

    // Create a copy of a matrix so that we can check reference results.
    std::vector<T> a_ref(a);

    // Using a random non-zero value of alpha.
    T alpha = T{3};

    // Invoking GER with an invalid value of n.
    ger<T>( STORAGE, CONJ, CONJ, invalid_m, N, &alpha, x.data(), inc,
            y.data(), inc, a.data(), LDA );

    // Computing bitwise difference.
    computediff<T>( "A", STORAGE, M, N, a.data(), a_ref.data(), LDA );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 1 );
#endif
}

// n < 0, with unit stride
TYPED_TEST(ger_IIT_ERS_Test, n_lt_zero_unitStride)
{
    using T = TypeParam;
    gtint_t invalid_n = -1;
    gtint_t unit_inc = 1;

    std::vector<T> a = testinghelpers::get_random_matrix<T>( -2, 5, STORAGE, 'n', M, N, LDA );
    std::vector<T> x = testinghelpers::get_random_vector<T>( -3, 3, M, unit_inc );
    std::vector<T> y = testinghelpers::get_random_vector<T>( -3, 3, N, unit_inc );

    // Create a copy of a matrix so that we can check reference results.
    std::vector<T> a_ref(a);

    // Using a random non-zero value of alpha.
    T alpha = T{3};

    // Invoking GER with an invalid value of n.
    ger<T>( STORAGE, CONJ, CONJ, M, invalid_n, &alpha, x.data(), unit_inc,
            y.data(), unit_inc, a.data(), LDA );

    // Computing bitwise difference.
    computediff<T>( "A", STORAGE, M, N, a.data(), a_ref.data(), LDA );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 2 );
#endif
}

// n < 0, with non-unit stride
TYPED_TEST(ger_IIT_ERS_Test, n_lt_zero_nonUnitStride)
{
    using T = TypeParam;
    gtint_t invalid_n = -1;
    gtint_t inc = 3;

    std::vector<T> a = testinghelpers::get_random_matrix<T>( -2, 5, STORAGE, 'n', M, N, LDA );
    std::vector<T> x = testinghelpers::get_random_vector<T>( -3, 3, M, inc );
    std::vector<T> y = testinghelpers::get_random_vector<T>( -3, 3, N, inc );

    // Create a copy of a matrix so that we can check reference results.
    std::vector<T> a_ref(a);

    // Using a random non-zero value of alpha.
    T alpha = T{3};

    // Invoking GER with an invalid value of n.
    ger<T>( STORAGE, CONJ, CONJ, M, invalid_n, &alpha, x.data(), inc,
            y.data(), inc, a.data(), LDA );

    // Computing bitwise difference.
    computediff<T>( "A", STORAGE, M, N, a.data(), a_ref.data(), LDA );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 2 );
#endif
}

// incx = 0, with unit incy
TYPED_TEST(ger_IIT_ERS_Test, incx_eq_zero_unitStride)
{
    using T = TypeParam;
    gtint_t invalid_incx = 0;
    gtint_t unit_inc = 1;

    std::vector<T> a = testinghelpers::get_random_matrix<T>( -2, 5, STORAGE, 'n', M, N, LDA );
    std::vector<T> x = testinghelpers::get_random_vector<T>( -3, 3, M, unit_inc );
    std::vector<T> y = testinghelpers::get_random_vector<T>( -3, 3, N, unit_inc );

    // Create a copy of a matrix so that we can check reference results.
    std::vector<T> a_ref(a);

    // Using a random non-zero value of alpha.
    T alpha = T{3};

    // Invoking GER with an invalid value of n.
    ger<T>( STORAGE, CONJ, CONJ, M, N, &alpha, x.data(), invalid_incx,
            y.data(), unit_inc, a.data(), LDA );

    // Computing bitwise difference.
    computediff<T>( "A", STORAGE, M, N, a.data(), a_ref.data(), LDA );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 5 );
#endif
}

// incx = 0, with non-unit incy
TYPED_TEST(ger_IIT_ERS_Test, incx_eq_zero_nonUnitStride)
{
    using T = TypeParam;
    gtint_t invalid_incx = 0;
    gtint_t inc = 3;

    std::vector<T> a = testinghelpers::get_random_matrix<T>( -2, 5, STORAGE, 'n', M, N, LDA );
    std::vector<T> x = testinghelpers::get_random_vector<T>( -3, 3, M, inc );
    std::vector<T> y = testinghelpers::get_random_vector<T>( -3, 3, N, inc );

    // Create a copy of a matrix so that we can check reference results.
    std::vector<T> a_ref(a);

    // Using a random non-zero value of alpha.
    T alpha = T{3};

    // Invoking GER with an invalid value of n.
    ger<T>( STORAGE, CONJ, CONJ, M, N, &alpha, x.data(), invalid_incx,
            y.data(), inc, a.data(), LDA );

    // Computing bitwise difference.
    computediff<T>( "A", STORAGE, M, N, a.data(), a_ref.data(), LDA );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 5 );
#endif
}

// incy = 0, with unit incy
TYPED_TEST(ger_IIT_ERS_Test, incy_eq_zero_unitStride)
{
    using T = TypeParam;
    gtint_t invalid_incy = 0;
    gtint_t unit_inc = 1;

    std::vector<T> a = testinghelpers::get_random_matrix<T>( -2, 5, STORAGE, 'n', M, N, LDA );
    std::vector<T> x = testinghelpers::get_random_vector<T>( -3, 3, M, unit_inc );
    std::vector<T> y = testinghelpers::get_random_vector<T>( -3, 3, N, unit_inc );

    // Create a copy of a matrix so that we can check reference results.
    std::vector<T> a_ref(a);

    // Using a random non-zero value of alpha.
    T alpha = T{3};

    // Invoking GER with an invalid value of n.
    ger<T>( STORAGE, CONJ, CONJ, M, N, &alpha, x.data(), unit_inc,
            y.data(), invalid_incy, a.data(), LDA );

    // Computing bitwise difference.
    computediff<T>( "A", STORAGE, M, N, a.data(), a_ref.data(), LDA );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 7 );
#endif
}

// incy = 0, with non-unit incy
TYPED_TEST(ger_IIT_ERS_Test, incy_eq_zero_nonUnitStride)
{
    using T = TypeParam;
    gtint_t invalid_incy = 0;
    gtint_t inc = 3;

    std::vector<T> a = testinghelpers::get_random_matrix<T>( -2, 5, STORAGE, 'n', M, N, LDA );
    std::vector<T> x = testinghelpers::get_random_vector<T>( -3, 3, M, inc );
    std::vector<T> y = testinghelpers::get_random_vector<T>( -3, 3, N, inc );

    // Create a copy of a matrix so that we can check reference results.
    std::vector<T> a_ref(a);

    // Using a random non-zero value of alpha.
    T alpha = T{3};

    // Invoking GER with an invalid value of n.
    ger<T>( STORAGE, CONJ, CONJ, M, N, &alpha, x.data(), inc,
            y.data(), invalid_incy, a.data(), LDA );

    // Computing bitwise difference.
    computediff<T>( "A", STORAGE, M, N, a.data(), a_ref.data(), LDA );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 7 );
#endif
}

// lda < max(1, M), with unit stride
TYPED_TEST(ger_IIT_ERS_Test, lda_lt_max_1_m_unitStride)
{
    using T = TypeParam;
    gtint_t invalid_lda = M - 1;
    gtint_t unit_inc = 1;

    std::vector<T> a = testinghelpers::get_random_matrix<T>( -2, 5, STORAGE, 'n', M, N, LDA );
    std::vector<T> x = testinghelpers::get_random_vector<T>( -3, 3, M, unit_inc );
    std::vector<T> y = testinghelpers::get_random_vector<T>( -3, 3, N, unit_inc );

    // Create a copy of a matrix so that we can check reference results.
    std::vector<T> a_ref(a);

    // Using a random non-zero value of alpha.
    T alpha = T{3};

    // Invoking GER with an invalid value of n.
    ger<T>( STORAGE, CONJ, CONJ, M, N, &alpha, x.data(), unit_inc,
            y.data(), unit_inc, a.data(), invalid_lda );

    // Computing bitwise difference.
    computediff<T>( "A", STORAGE, M, N, a.data(), a_ref.data(), LDA );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 9 );
#endif
}

// lda < max(1, M), with non-unit stride
TYPED_TEST(ger_IIT_ERS_Test, lda_lt_max_1_m_nonUnitStride)
{
    using T = TypeParam;
    gtint_t invalid_lda = LDA - 1;
    gtint_t inc = 3;

    std::vector<T> a = testinghelpers::get_random_matrix<T>( -2, 5, STORAGE, 'n', M, N, LDA );
    std::vector<T> x = testinghelpers::get_random_vector<T>( -3, 3, M, inc );
    std::vector<T> y = testinghelpers::get_random_vector<T>( -3, 3, N, inc );

    // Create a copy of a matrix so that we can check reference results.
    std::vector<T> a_ref(a);

    // Using a random non-zero value of alpha.
    T alpha = T{3};

    // Invoking GER with an invalid value of n.
    ger<T>( STORAGE, CONJ, CONJ, M, N, &alpha, x.data(), inc,
            y.data(), inc, a.data(), invalid_lda );

    // Computing bitwise difference.
    computediff<T>( "A", STORAGE, M, N, a.data(), a_ref.data(), LDA );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 9 );
#endif
}
#endif
