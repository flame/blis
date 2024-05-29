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


#ifdef TEST_BLAS

using namespace testinghelpers::IIT;

/**
 * @brief Test TRSM when side argument is incorrect
 *        when info == 1
 */
TYPED_TEST(trsm_IIT_ERS, invalid_side)
{
    using T = TypeParam;

    std::vector<T> b = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b_ref(b);

    trsm<T>( STORAGE, 'a', UPLO, TRANS, DIAG, M, N, nullptr, nullptr, LDA, b.data(), LDB);
    computediff<T>( "B", STORAGE, M, N, b.data(), b_ref.data(), LDB );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 1 );
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

    std::vector<T> b = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b_ref(b);

    trsm<T>( STORAGE, SIDE, 'a', TRANS, DIAG, M, N, nullptr, nullptr, LDA, b.data(), LDB);
    computediff<T>( "B", STORAGE, M, N, b.data(), b_ref.data(), LDB );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 2 );
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

    std::vector<T> b = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b_ref(b);

    trsm<T>( STORAGE, SIDE, UPLO, 'a', DIAG, M, N, nullptr, nullptr, LDA, b.data(), LDB);
    computediff<T>( "B", STORAGE, M, N, b.data(), b_ref.data(), LDB );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 3 );
#endif
}

/**
 * @brief Test TRSM when DIAG argument is incorrect
 *        when info == 4
 */
TYPED_TEST(trsm_IIT_ERS, invalid_DIAG)
{
    using T = TypeParam;

    std::vector<T> b = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b_ref(b);

    trsm<T>( STORAGE, SIDE, UPLO, TRANS, 'a', M, N, nullptr, nullptr, LDA, b.data(), LDB);
    computediff<T>( "B", STORAGE, M, N, b.data(), b_ref.data(), LDB );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 4 );
#endif
}

/**
 * @brief Test TRSM when m is negative
 *        when info == 5
 */
TYPED_TEST(trsm_IIT_ERS, invalid_m)
{
    using T = TypeParam;

    std::vector<T> b = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b_ref(b);

    trsm<T>( STORAGE, SIDE, UPLO, TRANS, DIAG, -2, N, nullptr, nullptr, LDA, b.data(), LDB);
    computediff<T>( "B", STORAGE, M, N, b.data(), b_ref.data(), LDB );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
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

    std::vector<T> b = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b_ref(b);

    trsm<T>( STORAGE, SIDE, UPLO, TRANS, DIAG, M, -2, nullptr, nullptr, LDA, b.data(), LDB);
    computediff<T>( "B", STORAGE, M, N, b.data(), b_ref.data(), LDB );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
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

    std::vector<T> b = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b_ref(b);

    trsm<T>( STORAGE, SIDE, UPLO, TRANS, DIAG, M, N, nullptr, nullptr, LDA - 1, b.data(), LDB);
    computediff<T>( "B", STORAGE, M, N, b.data(), b_ref.data(), LDB );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
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

    std::vector<T> b = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b_ref(b);

    trsm<T>( STORAGE, SIDE, UPLO, TRANS, DIAG, M, N, nullptr, nullptr, LDA, b.data(), LDB - 1);
    computediff<T>( "B", STORAGE, M, N, b.data(), b_ref.data(), LDB );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 11 );
#endif
}


/*
    Early Return Scenarios(ERS) :

    The TRSM API is expected to return early in the following cases:

    1. When m == 0.
    2. When n == 0.

*/

/**
 * @brief Test TRSM when M is zero
 */
TYPED_TEST(trsm_IIT_ERS, m_eq_zero)
{
    using T = TypeParam;

    std::vector<T> b = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b_ref(b);

    trsm<T>( STORAGE, SIDE, UPLO, TRANS, DIAG, 0, N, nullptr, nullptr, LDA, b.data(), LDB );
    computediff<T>( "B", STORAGE, M, N, b.data(), b_ref.data(), LDB );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

/**
 * @brief Test TRSM when N is zero
 */
TYPED_TEST(trsm_IIT_ERS, n_eq_zero)
{
    using T = TypeParam;

    std::vector<T> b = testinghelpers::get_random_matrix<T>(0, 1, STORAGE, 'n', M, N, LDB);
    std::vector<T> b_ref(b);

    trsm<T>( STORAGE, SIDE, UPLO, TRANS, DIAG, M, 0, nullptr, nullptr, LDA, b.data(), LDB );
    computediff<T>( "B", STORAGE, M, N, b.data(), b_ref.data(), LDB );

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

#endif
