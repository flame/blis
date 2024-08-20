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

#include "level2/trmv/test_trmv.h"
#include "inc/check_error.h"
#include "common/testing_helpers.h"
#include "common/wrong_inputs_helpers.h"
#include <stdexcept>
#include <algorithm>
#include <gtest/gtest.h>

template <typename T>
class trmv_IIT_ERS : public ::testing::Test {};
typedef ::testing::Types<float, double, scomplex, dcomplex> TypeParam;
TYPED_TEST_SUITE(trmv_IIT_ERS, TypeParam);

// Adding namespace to get default parameters(valid case) from testinghelpers/common/wrong_input_helpers.h.
using namespace testinghelpers::IIT;

#if defined(TEST_CBLAS)
#define INFO_OFFSET 1
#else
#define INFO_OFFSET 0
#endif

#if defined(TEST_CBLAS)

/**
 * @brief Test trmv when STORAGE argument is incorrect
 *        when info == 1
 *
 */
TYPED_TEST(trmv_IIT_ERS, invalid_storage)
{
    using T = TypeParam;
    T alpha = T{1};

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    trmv<T>( 'x', UPLO, TRANS, DIAG, N, &alpha, nullptr, LDA, nullptr, INC);
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 1 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, TRANS, M, N, LDA);
    std::vector<T> x = testinghelpers::get_random_vector<T>(0, 1, N, INC);
    std::vector<T> x_ref(x);

    trmv<T>( 'x', UPLO, TRANS, DIAG, N, &alpha, a.data(), LDA, x.data(), INC);
    computediff<T>( "x", N, x.data(), x_ref.data(), INC );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 1 );
#endif
}

#endif

#if defined(TEST_BLAS_LIKE) || defined(TEST_CBLAS)

/*
    Incorrect Input Testing(IIT)

    BLAS exceptions get triggered in the following cases(for trmv):
    1. When UPLO  != 'L' || UPLO   != 'U'                  (info = 1)
    2. When TRANS != 'N' || TRANS  != 'T' || TRANS != 'C'  (info = 2)
    3. When DIAG  != 'U' || DIAG   != 'N'                  (info = 3)
    4. When n < 0                                          (info = 4)
    5. When lda < N                                        (info = 6)
    6. When incx == 0                                      (info = 8)
*/


/**
 * @brief Test trmv when UPLO argument is incorrect
 *        when info == 1
 *
 */
TYPED_TEST(trmv_IIT_ERS, invalid_UPLO)
{
    using T = TypeParam;
    T alpha = T{1};

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    trmv<T>( STORAGE, 'A', TRANS, DIAG, N, &alpha, nullptr, LDA, nullptr, INC);
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+1 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, TRANS, M, N, LDA);
    std::vector<T> x = testinghelpers::get_random_vector<T>(0, 1, N, INC);
    std::vector<T> x_ref(x);

    trmv<T>( STORAGE, 'A', TRANS, DIAG, N, &alpha, a.data(), LDA, x.data(), INC);
    computediff<T>( "x", N, x.data(), x_ref.data(), INC );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+1 );
#endif
}

/**
 * @brief Test trmv when TRANS argument is incorrect
 *        when info == 2
 *
 */
TYPED_TEST(trmv_IIT_ERS, invalid_TRANS)
{
    using T = TypeParam;
    T alpha = T{1};

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    trmv<T>( STORAGE, UPLO, 'A', DIAG, N, &alpha, nullptr, LDA, nullptr, INC);
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+2 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, TRANS, M, N, LDA);
    std::vector<T> x = testinghelpers::get_random_vector<T>(0, 1, N, INC);
    std::vector<T> x_ref(x);

    trmv<T>( STORAGE, UPLO, 'A', DIAG, N, &alpha, a.data(), LDA, x.data(), INC);
    computediff<T>( "x", N, x.data(), x_ref.data(), INC );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+2 );
#endif
}

/**
 * @brief Test trmv when DIAG argument is incorrect
 *        when info == 3
 */
TYPED_TEST(trmv_IIT_ERS, invalid_DIAG)
{
    using T = TypeParam;
    T alpha = T{1};

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    trmv<T>( STORAGE, UPLO, TRANS, 'A', N, &alpha, nullptr, LDA, nullptr, INC);
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+3 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, TRANS, M, N, LDA);
    std::vector<T> x = testinghelpers::get_random_vector<T>(0, 1, N, INC);
    std::vector<T> x_ref(x);

    trmv<T>( STORAGE, UPLO, TRANS, 'A', N, &alpha, a.data(), LDA, x.data(), INC);
    computediff<T>( "x", N, x.data(), x_ref.data(), INC );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+3 );
#endif
}

/**
 * @brief Test trmv when N is negative
 *        when info == 4
 */
TYPED_TEST(trmv_IIT_ERS, invalid_n)
{
    using T = TypeParam;
    T alpha = T{1};

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    trmv<T>( STORAGE, UPLO, TRANS, DIAG, -1, &alpha, nullptr, LDA, nullptr, INC);
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 4 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, TRANS, M, N, LDA);
    std::vector<T> x = testinghelpers::get_random_vector<T>(0, 1, N, INC);
    std::vector<T> x_ref(x);

    trmv<T>( STORAGE, UPLO, TRANS, DIAG, -1, &alpha, a.data(), LDA, x.data(), INC);
    computediff<T>( "x", N, x.data(), x_ref.data(), INC );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 4 );
#endif
}


/**
 * @brief Test trmv when lda < max(1, N)
 *        when info == 6
 */
TYPED_TEST(trmv_IIT_ERS, invalid_lda)
{
    using T = TypeParam;
    T alpha = T{1};

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    trmv<T>( STORAGE, UPLO, TRANS, DIAG, N, &alpha, nullptr, LDA - 1, nullptr, INC);
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 6 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, TRANS, M, N, LDA);
    std::vector<T> x = testinghelpers::get_random_vector<T>(0, 1, N, INC);
    std::vector<T> x_ref(x);

    trmv<T>( STORAGE, UPLO, TRANS, DIAG, N, &alpha, a.data(), LDA - 1, x.data(), INC);
    computediff<T>( "x", N, x.data(), x_ref.data(), INC );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 6 );
#endif
}

/**
 * @brief Test trmv when INCX == 0
 *        when info == 8
 */
TYPED_TEST(trmv_IIT_ERS, invalid_incx)
{
    using T = TypeParam;
    T alpha = T{1};

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    trmv<T>( STORAGE, UPLO, TRANS, DIAG, N, &alpha, nullptr, LDA, nullptr, 0);
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 8 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, TRANS, M, N, LDA);
    std::vector<T> x = testinghelpers::get_random_vector<T>(0, 1, N, INC);
    std::vector<T> x_ref(x);

    trmv<T>( STORAGE, UPLO, TRANS, DIAG, N, &alpha, a.data(), LDA, x.data(), 0);
    computediff<T>( "x", N, x.data(), x_ref.data(), INC );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 8 );
#endif
}


/*
    Early Return Scenarios(ERS) :

    The trmv API is expected to return early in the following cases:

    1. When n == 0.

*/

/**
 * @brief Test trmv when N is zero
 */
TYPED_TEST(trmv_IIT_ERS, n_eq_zero)
{
    using T = TypeParam;
    T alpha = T{1};

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    trmv<T>( STORAGE, UPLO, TRANS, DIAG, 0, &alpha, nullptr, LDA, nullptr, INC);
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, TRANS, M, N, LDA);
    std::vector<T> x = testinghelpers::get_random_vector<T>(0, 1, N, INC);
    std::vector<T> x_ref(x);

    trmv<T>( STORAGE, UPLO, TRANS, DIAG, 0, &alpha, a.data(), LDA, x.data(), INC);
    computediff<T>( "x", N, x.data(), x_ref.data(), INC );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

#endif
