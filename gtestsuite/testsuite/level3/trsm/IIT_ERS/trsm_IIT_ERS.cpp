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

#include "level3/trsm/trsm.h"
#include "inc/check_error.h"
#include "common/testing_helpers.h"
#include "common/wrong_inputs_helpers.h"
#include <stdexcept>
#include <algorithm>
#include <gtest/gtest.h>


template <typename T>
class trsm_IIT_ERS : public ::testing::Test {};
typedef ::testing::Types<float, double, scomplex, dcomplex> TypeParam;
TYPED_TEST_SUITE(trsm_IIT_ERS, TypeParam);

// Adding namespace to get default parameters(valid case) from testinghelpers/common/wrong_input_helpers.h.
using namespace testinghelpers::IIT;

#if defined(TEST_CBLAS)
#define INFO_OFFSET 1
#else
#define INFO_OFFSET 0
#endif

#if defined(TEST_CBLAS)

/**
 * @brief Test TRSM when storage argument is incorrect
 *        when info == 1
 */
TYPED_TEST(trsm_IIT_ERS, invalid_storage)
{
    using T = TypeParam;
    T ALPHA = T{2.3};

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    trsm<T>( 'x', SIDE, UPLO, TRANS, DIAG, M, N, &ALPHA, nullptr, LDA, nullptr, LDB);

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 1 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b_ref(b);

    trsm<T>( 'x', SIDE, UPLO, TRANS, DIAG, M, N, &ALPHA, a.data(), LDA, b.data(), LDB);
    computediff<T>( "B", STORAGE, M, N, b.data(), b_ref.data(), LDB );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 1 );
#endif
}

#endif

#if defined(TEST_BLAS_LIKE) || defined(TEST_CBLAS)

/**
 * @brief Test TRSM when side argument is incorrect
 *        when info == 1
 */
TYPED_TEST(trsm_IIT_ERS, invalid_side)
{
    using T = TypeParam;
    T ALPHA = T{2.3};

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    trsm<T>( STORAGE, 'a', UPLO, TRANS, DIAG, M, N, nullptr, nullptr, LDA, nullptr, LDB);
#else
    trsm<T>( STORAGE, 'a', UPLO, TRANS, DIAG, M, N, &ALPHA, nullptr, LDA, nullptr, LDB);
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+1 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b_ref(b);

    trsm<T>( STORAGE, 'a', UPLO, TRANS, DIAG, M, N, &ALPHA, nullptr, LDA, b.data(), LDB);
    computediff<T>( "B", STORAGE, M, N, b.data(), b_ref.data(), LDB );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+1 );
#endif
}

/**
 * @brief Test TRSM when UPLO argument is incorrect
 *        when info == 2
 *
 */
TYPED_TEST(trsm_IIT_ERS, invalid_UPLO)
{
    using T = TypeParam;
    T ALPHA = T{2.3};

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    trsm<T>( STORAGE, SIDE, 'a', TRANS, DIAG, M, N, nullptr, nullptr, LDA, nullptr, LDB);
#else
    trsm<T>( STORAGE, SIDE, 'a', TRANS, DIAG, M, N, &ALPHA, nullptr, LDA, nullptr, LDB);
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+2 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b_ref(b);

    trsm<T>( STORAGE, SIDE, 'a', TRANS, DIAG, M, N, &ALPHA, a.data(), LDA, b.data(), LDB);
    computediff<T>( "B", STORAGE, M, N, b.data(), b_ref.data(), LDB );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+2 );
#endif
}

/**
 * @brief Test TRSM when TRANS argument is incorrect
 *        when info == 3
 *
 */
TYPED_TEST(trsm_IIT_ERS, invalid_TRANS)
{
    using T = TypeParam;
    T ALPHA = T{2.3};

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    trsm<T>( STORAGE, SIDE, UPLO, 'a', DIAG, M, N, nullptr, nullptr, LDA, nullptr, LDB);
#else
    trsm<T>( STORAGE, SIDE, UPLO, 'a', DIAG, M, N, &ALPHA, nullptr, LDA, nullptr, LDB);
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+3 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b_ref(b);

    trsm<T>( STORAGE, SIDE, UPLO, 'a', DIAG, M, N, &ALPHA, a.data(), LDA, b.data(), LDB);
    computediff<T>( "B", STORAGE, M, N, b.data(), b_ref.data(), LDB );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+3 );
#endif
}

/**
 * @brief Test TRSM when DIAG argument is incorrect
 *        when info == 4
 */
TYPED_TEST(trsm_IIT_ERS, invalid_DIAG)
{
    using T = TypeParam;
    T ALPHA = T{2.3};

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    trsm<T>( STORAGE, SIDE, UPLO, TRANS, 'a', M, N, nullptr, nullptr, LDA, nullptr, LDB);
#else
    trsm<T>( STORAGE, SIDE, UPLO, TRANS, 'a', M, N, &ALPHA, nullptr, LDA, nullptr, LDB);
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+4 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b_ref(b);

    trsm<T>( STORAGE, SIDE, UPLO, TRANS, 'a', M, N, &ALPHA, a.data(), LDA, b.data(), LDB);
    computediff<T>( "B", STORAGE, M, N, b.data(), b_ref.data(), LDB );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+4 );
#endif
}

/**
 * @brief Test TRSM when m is negative
 *        when info == 5
 */
TYPED_TEST(trsm_IIT_ERS, invalid_m)
{
    using T = TypeParam;
    T ALPHA = T{2.3};

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    trsm<T>( STORAGE, SIDE, UPLO, TRANS, DIAG, -1, N, nullptr, nullptr, LDA, nullptr, LDB);
#else
    trsm<T>( STORAGE, SIDE, UPLO, TRANS, DIAG, -1, N, &ALPHA, nullptr, LDA, nullptr, LDB);
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 5 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b_ref(b);

    trsm<T>( STORAGE, SIDE, UPLO, TRANS, DIAG, -1, N, &ALPHA, a.data(), LDA, b.data(), LDB);
    computediff<T>( "B", STORAGE, M, N, b.data(), b_ref.data(), LDB );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 5 );
#endif
}

/**
 * @brief Test TRSM when n is negative
 *        when info == 6
 */
TYPED_TEST(trsm_IIT_ERS, invalid_n)
{
    using T = TypeParam;
    T ALPHA = T{2.3};

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    trsm<T>( STORAGE, SIDE, UPLO, TRANS, DIAG, M, -1, nullptr, nullptr, LDA, nullptr, LDB);
#else
    trsm<T>( STORAGE, SIDE, UPLO, TRANS, DIAG, M, -1, &ALPHA, nullptr, LDA, nullptr, LDB);
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 6 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b_ref(b);

    trsm<T>( STORAGE, SIDE, UPLO, TRANS, DIAG, M, -1, &ALPHA, a.data(), LDA, b.data(), LDB);
    computediff<T>( "B", STORAGE, M, N, b.data(), b_ref.data(), LDB );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 6 );
#endif
}

/**
 * @brief Test TRSM when lda is incorrect
 *        when info == 9
 */
TYPED_TEST(trsm_IIT_ERS, invalid_lda)
{
    using T = TypeParam;
    T ALPHA = T{2.3};

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    trsm<T>( STORAGE, SIDE, UPLO, TRANS, DIAG, M, N, nullptr, nullptr, LDA - 1, nullptr, LDB);
#else
    trsm<T>( STORAGE, SIDE, UPLO, TRANS, DIAG, M, N, &ALPHA, nullptr, LDA - 1, nullptr, LDB);
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 9 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b_ref(b);

    trsm<T>( STORAGE, SIDE, UPLO, TRANS, DIAG, M, N, &ALPHA, a.data(), LDA - 1, b.data(), LDB);
    computediff<T>( "B", STORAGE, M, N, b.data(), b_ref.data(), LDB );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 9 );
#endif
}

/**
 * @brief Test TRSM when ldb is incorrect
 *        when info == 11
 */
TYPED_TEST(trsm_IIT_ERS, invalid_ldb)
{
    using T = TypeParam;
    T ALPHA = T{2.3};

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    trsm<T>( STORAGE, SIDE, UPLO, TRANS, DIAG, M, N, nullptr, nullptr, LDA, nullptr, LDB - 1);
#else
    trsm<T>( STORAGE, SIDE, UPLO, TRANS, DIAG, M, N, &ALPHA, nullptr, LDA, nullptr, LDB - 1);
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 11 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b_ref(b);

    trsm<T>( STORAGE, SIDE, UPLO, TRANS, DIAG, M, N, &ALPHA, a.data(), LDA, b.data(), LDB - 1);
    computediff<T>( "B", STORAGE, M, N, b.data(), b_ref.data(), LDB );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 11 );
#endif
}


/*
    Early Return Scenarios(ERS) :

    The TRSM API is expected to return early in the following cases:

    1. When m == 0.
    2. When n == 0.
    3. When alpha == 0, set B to 0 only.

*/

/**
 * @brief Test TRSM when M is zero
 */
TYPED_TEST(trsm_IIT_ERS, m_eq_zero)
{
    using T = TypeParam;
    T ALPHA = T{2.3};

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    trsm<T>( STORAGE, SIDE, UPLO, TRANS, DIAG, 0, N, nullptr, nullptr, LDA, nullptr, LDB);
#else
    trsm<T>( STORAGE, SIDE, UPLO, TRANS, DIAG, 0, N, &ALPHA, nullptr, LDA, nullptr, LDB);
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b_ref(b);

    trsm<T>( STORAGE, SIDE, UPLO, TRANS, DIAG, 0, N, &ALPHA, a.data(), LDA, b.data(), LDB );
    computediff<T>( "B", STORAGE, M, N, b.data(), b_ref.data(), LDB );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

/**
 * @brief Test TRSM when N is zero
 */
TYPED_TEST(trsm_IIT_ERS, n_eq_zero)
{
    using T = TypeParam;
    T ALPHA = T{2.3};

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    trsm<T>( STORAGE, SIDE, UPLO, TRANS, DIAG, M, 0, nullptr, nullptr, LDA, nullptr, LDB);
#else
    trsm<T>( STORAGE, SIDE, UPLO, TRANS, DIAG, M, 0, &ALPHA, nullptr, LDA, nullptr, LDB);
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b_ref(b);

    trsm<T>( STORAGE, SIDE, UPLO, TRANS, DIAG, M, 0, &ALPHA, a.data(), LDA, b.data(), LDB );
    computediff<T>( "B", STORAGE, M, N, b.data(), b_ref.data(), LDB );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

/**
 * @brief Test TRSM when alpha is zero
 */
TYPED_TEST(trsm_IIT_ERS, alpha_eq_zero)
{
    using T = TypeParam;
    T ALPHA;
    testinghelpers::initzero<T>( ALPHA );

    std::vector<T> b = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b2(b);
    std::vector<T> zero_mat = testinghelpers::get_random_matrix<T>(0, 0, STORAGE, 'n', M, N, LDB);

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    trsm<T>( STORAGE, SIDE, UPLO, TRANS, DIAG, M, N, &ALPHA, nullptr, LDA, b2.data(), LDB);
    computediff<T>( "B", STORAGE, M, N, b2.data(), zero_mat.data(), LDB );
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);

    trsm<T>( STORAGE, SIDE, UPLO, TRANS, DIAG, M, N, &ALPHA, a.data(), LDA, b.data(), LDB );
    computediff<T>( "B", STORAGE, M, N, b.data(), zero_mat.data(), LDB );

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

#endif
