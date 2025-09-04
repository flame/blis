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
#include "test_omatcopy.h"
#include "common/wrong_inputs_helpers.h"
#include "common/testing_helpers.h"
#include "inc/check_error.h"

template <typename T>
class omatcopy_IIT_ERS : public ::testing::Test {};
typedef ::testing::Types<float, double, scomplex, dcomplex> TypeParam;
TYPED_TEST_SUITE(omatcopy_IIT_ERS, TypeParam);

using namespace testinghelpers::IIT;

#if defined(TEST_BLAS_LIKE)

/*
    Incorrect Input Testing(IIT)

    The exceptions get triggered in the following cases:
    1. When TRANS != 'n' || TRANS != 't'  || TRANS != 'c' || TRANS != 'r'
    2. When m < 0
    3. When n < 0
    4. When lda < max(1, m).
    5. When ldb < max(1, thresh), thresh set based on TRANS value
*/

// When TRANS is invalid
TYPED_TEST(omatcopy_IIT_ERS, invalid_transa)
{
    using T = TypeParam;
    T alpha;
    testinghelpers::initone<T>( alpha );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    omatcopy<T>( 'Q', M, N, alpha, nullptr, LDA, nullptr, LDB);

    // Test with all arguments correct except for the value we are choosing to test.
    // Defining the A and B matrices with values for debugging purposes
    std::vector<T> A = testinghelpers::get_random_matrix<T>(-10, 10, 'c', 'n', M, N, LDA );
    std::vector<T> B = testinghelpers::get_random_matrix<T>(-10, 10, 'c', 'n', M, N, LDB );
    // Copy so that we check that the elements of B are not modified.
    std::vector<T> B_ref(B);

    // Call OMATCOPY with a invalid value for TRANS value for the operation.
    omatcopy<T>( 'Q', M, N, alpha, A.data(), LDA, B.data(), LDB);
    // Use bitwise comparison (no threshold).
    computediff<T>( "B", 'c', M, N, B.data(), B_ref.data(), LDB );
}

// When m < 0
TYPED_TEST(omatcopy_IIT_ERS, m_lt_zero)
{
    using T = TypeParam;
    T alpha;
    testinghelpers::initone<T>( alpha );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    omatcopy<T>( TRANS, -1, N, alpha, nullptr, LDA, nullptr, LDB);

    // Test with all arguments correct except for the value we are choosing to test.
    // Defining the A and B matrices with values for debugging purposes
    std::vector<T> A = testinghelpers::get_random_matrix<T>(-10, 10, 'c', 'n', M, N, LDA );
    std::vector<T> B = testinghelpers::get_random_matrix<T>(-10, 10, 'c', 'n', M, N, LDB );
    // Copy so that we check that the elements of B are not modified.
    std::vector<T> B_ref(B);

    // Call OMATCOPY with a invalid m for the operation.
    omatcopy<T>( TRANS, -1, N, alpha, A.data(), LDA, B.data(), LDB);
    // Use bitwise comparison (no threshold).
    computediff<T>( "B", 'c', M, N, B.data(), B_ref.data(), LDB );
}

// When n < 0
TYPED_TEST(omatcopy_IIT_ERS, n_lt_zero)
{
    using T = TypeParam;
    T alpha;
    testinghelpers::initone<T>( alpha );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    omatcopy<T>( TRANS, M, -1, alpha, nullptr, LDA, nullptr, LDB);

    // Test with all arguments correct except for the value we are choosing to test.
    // Defining the A and B matrices with values for debugging purposes
    std::vector<T> A = testinghelpers::get_random_matrix<T>(-10, 10, 'c', 'n', M, N, LDA );
    std::vector<T> B = testinghelpers::get_random_matrix<T>(-10, 10, 'c', 'n', M, N, LDB );
    // Copy so that we check that the elements of B are not modified.
    std::vector<T> B_ref(B);

    // Call OMATCOPY with a invalid n for the operation.
    omatcopy<T>( TRANS, M, -1, alpha, A.data(), LDA, B.data(), LDB);
    // Use bitwise comparison (no threshold).
    computediff<T>( "B", 'c', M, N, B.data(), B_ref.data(), LDB );
}

// When lda < m
TYPED_TEST(omatcopy_IIT_ERS, invalid_lda)
{
    using T = TypeParam;
    T alpha;
    testinghelpers::initone<T>( alpha );

    // Having different values for m and n
    gtint_t m = 5;
    gtint_t n = 10;

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    omatcopy<T>( 'n', m, n, alpha, nullptr, m - 1, nullptr, m);

    // Test with all arguments correct except for the value we are choosing to test.
    // Defining the A and B matrices with values for debugging purposes
    std::vector<T> A = testinghelpers::get_random_matrix<T>(-10, 10, 'c', 'n', m, n, m );
    std::vector<T> B = testinghelpers::get_random_matrix<T>(-10, 10, 'c', 'n', m, n, m );
    // Copy so that we check that the elements of B are not modified.
    std::vector<T> B_ref(B);

    // Call OMATCOPY with a invalid lda for the operation.
    omatcopy<T>( 'n', m, n, alpha, A.data(), m - 1, B.data(), m);
    // Use bitwise comparison (no threshold).
    computediff<T>( "B", 'c', m, n, B.data(), B_ref.data(), m );
}

// When ldb < m, with trans == 'n'
TYPED_TEST(omatcopy_IIT_ERS, invalid_ldb_no_transpose)
{
    using T = TypeParam;
    T alpha;
    testinghelpers::initone<T>( alpha );

    // Having different values for m and n
    gtint_t m = 5;
    gtint_t n = 10;
    char trans = 'n';

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    omatcopy<T>( trans, m, n, alpha, nullptr, m, nullptr, m - 1 );

    // Test with all arguments correct except for the value we are choosing to test.
    // Defining the A and B matrices with values for debugging purposes
    std::vector<T> A = testinghelpers::get_random_matrix<T>(-10, 10, 'c', 'n', m, n, m );
    std::vector<T> B = testinghelpers::get_random_matrix<T>(-10, 10, 'c', 'n', m, n, m );
    // Copy so that we check that the elements of B are not modified.
    std::vector<T> B_ref(B);

    // Call OMATCOPY with a invalid ldb for the operation.
    omatcopy<T>( trans, m, n, alpha, A.data(), m, B.data(), m - 1 );
    // Use bitwise comparison (no threshold).
    computediff<T>( "B", 'c', m, n, B.data(), B_ref.data(), m );
}

// When ldb < m, with trans == 'r'
TYPED_TEST(omatcopy_IIT_ERS, invalid_ldb_conjugate)
{
    using T = TypeParam;
    T alpha;
    testinghelpers::initone<T>( alpha );

    // Having different values for m and n
    gtint_t m = 5;
    gtint_t n = 10;
    char trans = 'r';

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    omatcopy<T>( trans, m, n, alpha, nullptr, m, nullptr, m - 1 );

    // Test with all arguments correct except for the value we are choosing to test.
    // Defining the A and B matrices with values for debugging purposes
    std::vector<T> A = testinghelpers::get_random_matrix<T>(-10, 10, 'c', 'n', m, n, m );
    std::vector<T> B = testinghelpers::get_random_matrix<T>(-10, 10, 'c', 'n', m, n, m );
    // Copy so that we check that the elements of B are not modified.
    std::vector<T> B_ref(B);

    // Call OMATCOPY with a invalid ldb for the operation.
    omatcopy<T>( trans, m, n, alpha, A.data(), m, B.data(), m - 1 );
    // Use bitwise comparison (no threshold).
    computediff<T>( "B", 'c', m, n, B.data(), B_ref.data(), m );
}

// When ldb < m, with trans == 't'
TYPED_TEST(omatcopy_IIT_ERS, invalid_ldb_transpose)
{
    using T = TypeParam;
    T alpha;
    testinghelpers::initone<T>( alpha );

    // Having different values for m and n
    gtint_t m = 5;
    gtint_t n = 10;
    char trans = 't';

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    omatcopy<T>( trans, m, n, alpha, nullptr, m, nullptr, n - 1 );

    // Test with all arguments correct except for the value we are choosing to test.
    // Defining the A and B matrices with values for debugging purposes
    std::vector<T> A = testinghelpers::get_random_matrix<T>(-10, 10, 'c', 'n', m, n, m );
    std::vector<T> B = testinghelpers::get_random_matrix<T>(-10, 10, 'c', 't', m, n, n );
    // Copy so that we check that the elements of B are not modified.
    std::vector<T> B_ref(B);

    // Call OMATCOPY with a invalid ldb for the operation.
    omatcopy<T>( trans, m, n, alpha, A.data(), m, B.data(), n - 1 );
    // Use bitwise comparison (no threshold).
    computediff<T>( "B", 'c', n, m, B.data(), B_ref.data(), n );
}

// When ldb < m, with trans == 'c'
TYPED_TEST(omatcopy_IIT_ERS, invalid_ldb_conjugate_transpose)
{
    using T = TypeParam;
    T alpha;
    testinghelpers::initone<T>( alpha );

    // Having different values for m and n
    gtint_t m = 5;
    gtint_t n = 10;
    char trans = 'c';

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    omatcopy<T>( trans, m, n, alpha, nullptr, m, nullptr, n - 1 );

    // Test with all arguments correct except for the value we are choosing to test.
    // Defining the A and B matrices with values for debugging purposes
    std::vector<T> A = testinghelpers::get_random_matrix<T>(-10, 10, 'c', 'n', m, n, m );
    std::vector<T> B = testinghelpers::get_random_matrix<T>(-10, 10, 'c', 't', m, n, n );
    // Copy so that we check that the elements of B are not modified.
    std::vector<T> B_ref(B);

    // Call OMATCOPY with a invalid ldb for the operation.
    omatcopy<T>( trans, m, n, alpha, A.data(), m, B.data(), n - 1 );
    // Use bitwise comparison (no threshold).
    computediff<T>( "B", 'c', n, m, B.data(), B_ref.data(), n );
}
#endif
