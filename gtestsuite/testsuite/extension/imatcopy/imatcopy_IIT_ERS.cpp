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
#include "test_imatcopy.h"
#include "common/wrong_inputs_helpers.h"
#include "common/testing_helpers.h"
#include "inc/check_error.h"

template <typename T>
class imatcopy_IIT_ERS : public ::testing::Test {};
typedef ::testing::Types<float, double, scomplex, dcomplex> TypeParam;
TYPED_TEST_SUITE(imatcopy_IIT_ERS, TypeParam);

using namespace testinghelpers::IIT;

#if defined(TEST_BLAS_LIKE)

/*
    Incorrect Input Testing(IIT)

    The exceptions get triggered in the following cases:
    1. When TRANS != 'n' || TRANS != 't'  || TRANS != 'c' || TRANS != 'r'
    2. When m < 0
    3. When n < 0
    4. When lda_in < max(1, m).
    5. When lda_out < max(1, thresh), thresh set based on TRANS value
*/

// When TRANS is invalid
TYPED_TEST(imatcopy_IIT_ERS, invalid_transa)
{
    using T = TypeParam;
    T alpha = T{2.3};

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    imatcopy<T>( 'Q', M, N, alpha, nullptr, LDA, LDA );

    // Test with all arguments correct except for the value we are choosing to test.
    // Defining the A matrix with values for debugging purposes
    std::vector<T> A = testinghelpers::get_random_matrix<T>(-10, 10, 'c', 'n', M, N, LDA );
    // Copy so that we check that the elements of A are not modified.
    std::vector<T> A_ref(A);

    // Call imatcopy with a invalid value for TRANS value for the operation.
    imatcopy<T>( 'Q', M, N, alpha, A.data(), LDA, LDA );
    // Use bitwise comparison (no threshold).
    computediff<T>( "A", 'c', M, N, A.data(), A_ref.data(), LDA );
}

// When m < 0
TYPED_TEST(imatcopy_IIT_ERS, m_lt_zero)
{
    using T = TypeParam;
    T alpha = T{2.3};

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    imatcopy<T>( TRANS, -1, N, alpha, nullptr, LDA, LDA );

    // Defining the A matrix with values for debugging purposes
    std::vector<T> A = testinghelpers::get_random_matrix<T>(-10, 10, 'c', 'n', M, N, LDA );
    // Copy so that we check that the elements of A are not modified.
    std::vector<T> A_ref(A);

    // Call imatcopy with a invalid m for the operation.
    imatcopy<T>( TRANS, -1, N, alpha, A.data(), LDA, LDA );
    // Use bitwise comparison (no threshold).
    computediff<T>( "A", 'c', M, N, A.data(), A_ref.data(), LDA );
}

// When n < 0
TYPED_TEST(imatcopy_IIT_ERS, n_lt_zero)
{
    using T = TypeParam;
    T alpha = T{2.3};

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    imatcopy<T>( TRANS, M, -1, alpha, nullptr, LDA, LDA );

    // Defining the A matrix with values for debugging purposes
    std::vector<T> A = testinghelpers::get_random_matrix<T>(-10, 10, 'c', 'n', M, N, LDA );
    // Copy so that we check that the elements of A are not modified.
    std::vector<T> A_ref(A);

    // Call imatcopy with a invalid n for the operation.
    imatcopy<T>( TRANS, M, -1, alpha, A.data(), LDA, LDA );
    // Use bitwise comparison (no threshold).
    computediff<T>( "A", 'c', M, N, A.data(), A_ref.data(), LDA );
}

// When lda < m
TYPED_TEST(imatcopy_IIT_ERS, invalid_lda_in)
{
    using T = TypeParam;
    T alpha = T{2.3};

    // Having different values for m and n
    gtint_t m = 10;
    gtint_t n = 5;

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    imatcopy<T>( TRANS, m, n, alpha, nullptr, m - 1, m );

    // Defining the A matrix with values for debugging purposes
    std::vector<T> A = testinghelpers::get_random_matrix<T>(-10, 10, 'c', 'n', m, n, m );
    // Copy so that we check that the elements of A are not modified.
    std::vector<T> A_ref(A);

    // Call imatcopy with a invalid lda for the operation.
    imatcopy<T>( 'n', m, n, alpha, A.data(), m - 1, m );
    // Use bitwise comparison (no threshold).
    computediff<T>( "A", 'c', m, n, A.data(), A_ref.data(), m );
}

// When lda_out < m, with trans == 'n'
TYPED_TEST(imatcopy_IIT_ERS, invalid_lda_out_no_transpose)
{
    using T = TypeParam;
    T alpha = T{2.3};

    // Having different values for m and n
    gtint_t m = 10;
    gtint_t n = 5;

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    imatcopy<T>( 'n', m, n, alpha, nullptr, m, m-1 );

    // Defining the A matrix with values for debugging purposes
    std::vector<T> A = testinghelpers::get_random_matrix<T>(-10, 10, 'c', 'n', m, n, m );
    // Copy so that we check that the elements of A are not modified.
    std::vector<T> A_ref(A);

    // Call imatcopy with a invalid lda for the operation.
    imatcopy<T>( 'n', m, n, alpha, A.data(), m, m-1 );
    // Use bitwise comparison (no threshold).
    computediff<T>( "A", 'c', m, n, A.data(), A_ref.data(), m );
}

// When lda_out < m, with trans == 'r'
TYPED_TEST(imatcopy_IIT_ERS, invalid_lda_out_conjugate)
{
    using T = TypeParam;
    T alpha = T{2.3};

    // Having different values for m and n
    gtint_t m = 10;
    gtint_t n = 5;

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    imatcopy<T>( 'r', m, n, alpha, nullptr, m, m-1 );

    // Defining the A matrix with values for debugging purposes
    std::vector<T> A = testinghelpers::get_random_matrix<T>(-10, 10, 'c', 'n', m, n, m );
    // Copy so that we check that the elements of A are not modified.
    std::vector<T> A_ref(A);

    // Call imatcopy with a invalid lda for the operation.
    imatcopy<T>( 'r', m, n, alpha, A.data(), m, m-1 );
    // Use bitwise comparison (no threshold).
    computediff<T>( "A", 'c', m, n, A.data(), A_ref.data(), m );
}

// When lda_out < m, with trans == 't'
TYPED_TEST(imatcopy_IIT_ERS, invalid_lda_out_transpose)
{
    using T = TypeParam;
    T alpha = T{2.3};

    // Having different values for m and n
    gtint_t m = 10;
    gtint_t n = 5;

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    imatcopy<T>( 't', m, n, alpha, nullptr, m, n-1 );

    // Defining the A matrix with values for debugging purposes
    std::vector<T> A = testinghelpers::get_random_matrix<T>(-10, 10, 'c', 'n', m, n, m );
    // Copy so that we check that the elements of A are not modified.
    std::vector<T> A_ref(A);

    // Call imatcopy with a invalid lda for the operation.
    imatcopy<T>( 't', m, n, alpha, A.data(), m, n-1 );
    // Use bitwise comparison (no threshold).
    computediff<T>( "A", 'c', m, n, A.data(), A_ref.data(), m );
}

// When lda_out < m, with trans == 'c'
TYPED_TEST(imatcopy_IIT_ERS, invalid_lda_out_conjugate_transpose)
{
    using T = TypeParam;
    T alpha = T{2.3};

    // Having different values for m and n
    gtint_t m = 10;
    gtint_t n = 5;

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    imatcopy<T>( 'c', m, n, alpha, nullptr, m, n-1 );

    // Defining the A matrix with values for debugging purposes
    std::vector<T> A = testinghelpers::get_random_matrix<T>(-10, 10, 'c', 'n', m, n, m );
    // Copy so that we check that the elements of A are not modified.
    std::vector<T> A_ref(A);

    // Call imatcopy with a invalid lda for the operation.
    imatcopy<T>( 'c', m, n, alpha, A.data(), m, n-1 );
    // Use bitwise comparison (no threshold).
    computediff<T>( "A", 'c', m, n, A.data(), A_ref.data(), m );
}
#endif
