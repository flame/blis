/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

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
#include "common/testing_helpers.h"
#include "level3/hemm/test_hemm.h"
#include "inc/check_error.h"
#include "common/wrong_inputs_helpers.h"

template <typename T>
class hemm_IIT_ERS : public ::testing::Test {};
typedef ::testing::Types<scomplex, dcomplex> TypeParam; // The supported datatypes from BLAS calls for hemm
TYPED_TEST_SUITE(hemm_IIT_ERS, TypeParam); // Defining individual testsuites based on the datatype support.

// Adding namespace to get default parameters(valid case) from testinghelpers/common/wrong_input_helpers.h.
using namespace testinghelpers::IIT;

#if defined(TEST_CBLAS)
#define INFO_OFFSET 1
#else
#define INFO_OFFSET 0
#endif

#if defined(TEST_CBLAS)

// When info == 1
TYPED_TEST(hemm_IIT_ERS, invalid_storage)
{
    using T = TypeParam;

    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    hemm<T>( 'x', SIDE, UPLO, CONJ, TRANS, M, N, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 1 );
#endif

    // Set the dimension for row/col of A, depending on the value of side.
    gtint_t K = ((SIDE == 'l')||(SIDE == 'L'))? M : N;
    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDB);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS hemm with a invalid value for TRANS value for A.
    hemm<T>( 'x', SIDE, UPLO, CONJ, TRANS, M, N, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, M, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 1 );
#endif
}

#endif

#if defined(TEST_BLAS_LIKE) || defined(TEST_CBLAS)

/*
    Incorrect Input Testing(IIT)

    BLAS exceptions get triggered in the following cases(for hemm):
    1. When SIDE != 'L' || SIDE != 'R' (info = 1)
    2. When UPLO != 'U' || UPLO != 'L' (info = 2)
    3. When m < 0 (info = 3)
    4. When n < 0 (info = 4)
    6. When lda < max(1, thresh) (info = 7), thresh set based on SIDE value
    7. When ldb < max(1, m) (info = 9)
    8. When ldc < max(1, m) (info = 12)

*/

// When info == 1
TYPED_TEST(hemm_IIT_ERS, invalid_side)
{
    using T = TypeParam;

    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    hemm<T>( STORAGE, 'p', UPLO, CONJ, TRANS, M, N, nullptr, nullptr, LDA, nullptr, LDB, nullptr, nullptr, LDC );
#else
    hemm<T>( STORAGE, 'p', UPLO, CONJ, TRANS, M, N, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+1 );
#endif

    // Set the dimension for row/col of A, depending on the value of side.
    gtint_t K = ((SIDE == 'l')||(SIDE == 'L'))? M : N;
    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDB);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS hemm with a invalid value for TRANS value for A.
    hemm<T>( STORAGE, 'p', UPLO, CONJ, TRANS, M, N, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, M, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+1 );
#endif
}

// When info == 2
TYPED_TEST(hemm_IIT_ERS, invalid_uplo)
{
    using T = TypeParam;

    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    hemm<T>( STORAGE, SIDE, 'p', CONJ, TRANS, M, N, nullptr, nullptr, LDA, nullptr, LDB, nullptr, nullptr, LDC );
#else
    hemm<T>( STORAGE, SIDE, 'p', CONJ, TRANS, M, N, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+2 );
#endif

    // Set the dimension for row/col of A, depending on the value of side.
    gtint_t K = ((SIDE == 'l')||(SIDE == 'L'))? M : N;
    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDB);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS hemm with a invalid value for TRANS value for B.
    hemm<T>( STORAGE, SIDE, 'p', CONJ, TRANS, M, N, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, M, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+2 );
#endif
}

// When info == 3
TYPED_TEST(hemm_IIT_ERS, m_lt_zero)
{
    using T = TypeParam;

    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, -1, N, nullptr, nullptr, LDA, nullptr, LDB, nullptr, nullptr, LDC );
#else
    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, -1, N, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 3 );
#endif

    // Set the dimension for row/col of A, depending on the value of side.
    gtint_t K = ((SIDE == 'l')||(SIDE == 'L'))? M : N;
    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDB);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS hemm with a invalid value for m.
    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, -1, N, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, M, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 3 );
#endif
}

// When info == 4
TYPED_TEST(hemm_IIT_ERS, n_lt_zero)
{
    using T = TypeParam;

    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, M, -1, nullptr, nullptr, LDA, nullptr, LDB, nullptr, nullptr, LDC );
#else
    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, M, -1, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 4 );
#endif

    // Set the dimension for row/col of A, depending on the value of side.
    gtint_t K = ((SIDE == 'l')||(SIDE == 'L'))? M : N;
    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDB);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS hemm with a invalid value for n.
    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, M, -1, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, M, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 4 );
#endif
}

// When info == 7
TYPED_TEST(hemm_IIT_ERS, invalid_lda_side_l)
{
    using T = TypeParam;

    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    hemm<T>( STORAGE, 'l', UPLO, CONJ, TRANS, M, N, nullptr, nullptr, M - 1, nullptr, LDB, nullptr, nullptr, LDC );
#else
    hemm<T>( STORAGE, 'l', UPLO, CONJ, TRANS, M, N, &alpha, nullptr, M - 1, nullptr, LDB, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 7 );
#endif

    // Set the dimension for row/col of A, depending on the value of side.
    gtint_t K = ((SIDE == 'l')||(SIDE == 'L'))? M : N;
    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDB);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS hemm with a invalid value for lda.
    hemm<T>( STORAGE, 'l', UPLO, CONJ, TRANS, M, N, &alpha, a.data(), M - 1, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, M, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 7 );
#endif
}

// When info == 7
TYPED_TEST(hemm_IIT_ERS, invalid_lda_side_r)
{
    using T = TypeParam;

    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, M, N, nullptr, nullptr, N - 1, nullptr, LDB, nullptr, nullptr, LDC );
#else
    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, M, N, &alpha, nullptr, N - 1, nullptr, LDB, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 7 );
#endif

    // Set the dimension for row/col of A, depending on the value of side.
    gtint_t K = ((SIDE == 'l')||(SIDE == 'L'))? M : N;
    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDB);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS hemm with a invalid value for lda.
    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, M, N, &alpha, a.data(), N - 1, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, M, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 7 );
#endif
}

// When info == 9
TYPED_TEST(hemm_IIT_ERS, invalid_ldb)
{
    using T = TypeParam;

    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, M, N, nullptr, nullptr, LDA, nullptr, M - 1, nullptr, nullptr, LDC );
#else
    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, M, N, &alpha, nullptr, LDA, nullptr, M - 1, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 9 );
#endif

    // Set the dimension for row/col of A, depending on the value of side.
    gtint_t K = ((SIDE == 'l')||(SIDE == 'L'))? M : N;
    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDB);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS hemm with a invalid value for ldb.
    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, M, N, &alpha, a.data(), LDA, b.data(), M - 1, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, M, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 9 );
#endif
}

// When info == 12
TYPED_TEST(hemm_IIT_ERS, invalid_ldc)
{
    using T = TypeParam;

    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, M, N, nullptr, nullptr, LDA, nullptr, LDB, nullptr, nullptr, M - 1 );
#else
    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, M, N, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, M - 1 );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 12 );
#endif

    // Set the dimension for row/col of A, depending on the value of side.
    gtint_t K = ((SIDE == 'l')||(SIDE == 'L'))? M : N;
    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDB);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS hemm with a invalid value for ldc.
    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, M, N, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), M - 1 );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, M, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 12 );
#endif
}

/*
    Early Return Scenarios(ERS) :

    The hemm API is expected to return early in the following cases:

    1. When m == 0.
    2. When n == 0.
    3. When (alpha == 0 or k == 0) and beta == 1.
    4. When alpha == 0 and beta == 0, set C = 0 only
    5. When alpha == 0 and beta /= 0 or 1, scale C by beta only

*/

// When m is 0
TYPED_TEST(hemm_IIT_ERS, m_eq_zero)
{
    using T = TypeParam;

    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, 0, N, nullptr, nullptr, LDA, nullptr, LDB, nullptr, nullptr, LDC );
#else
    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, 0, N, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif

    // Set the dimension for row/col of A, depending on the value of side.
    gtint_t K = ((SIDE == 'l')||(SIDE == 'L'))? M : N;
    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDB);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, 0, N, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, M, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// When n is 0
TYPED_TEST(hemm_IIT_ERS, n_eq_zero)
{
    using T = TypeParam;

    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, M, 0, nullptr, nullptr, LDA, nullptr, LDB, nullptr, nullptr, LDC );
#else
    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, M, 0, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif

    // Set the dimension for row/col of A, depending on the value of side.
    gtint_t K = ((SIDE == 'l')||(SIDE == 'L'))? M : N;
    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDB);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, M, 0, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, M, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// When alpha is 0 and beta is 1
TYPED_TEST(hemm_IIT_ERS, alpha_zero_beta_one)
{
    using T = TypeParam;

    T alpha, beta;
    testinghelpers::initzero<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, M, N, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#else
    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, M, N, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif

    // Set the dimension for row/col of A, depending on the value of side.
    gtint_t K = ((SIDE == 'l')||(SIDE == 'L'))? M : N;
    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDB);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, M, N, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, M, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// When k is 0 and beta is 1
TYPED_TEST(hemm_IIT_ERS, k_zero_beta_one)
{
    using T = TypeParam;

    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, M, 0, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#else
    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, M, 0, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif

    // Set the dimension for row/col of A, depending on the value of side.
    gtint_t K = ((SIDE == 'l')||(SIDE == 'L'))? M : N;
    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDB);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, M, 0, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, M, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// zero alpha and zero beta - set C to 0
TYPED_TEST(hemm_IIT_ERS, ZeroAlpha_ZeroBeta)
{
    using T = TypeParam;

    T alpha, beta;
    testinghelpers::initzero<T>( alpha );
    testinghelpers::initzero<T>( beta );

    // Matrix C should not be read, only set.
    std::vector<T> c( testinghelpers::matsize( STORAGE, 'N', M, N, LDC ) );
    testinghelpers::set_matrix( STORAGE, M, N, c.data(), 'N', LDC, testinghelpers::aocl_extreme<T>() );
    std::vector<T> c2(c);
    // Set up expected output matrix
    std::vector<T> zero_mat = testinghelpers::get_random_matrix<T>(0, 0, STORAGE, 'n', M, N, LDB);

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, M, N, &alpha, nullptr, LDA, nullptr, LDB, &beta, c2.data(), LDC );
    computediff<T>( "C", STORAGE, N, N, c2.data(), zero_mat.data(), LDC);
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif

    // Set the dimension for row/col of A, depending on the value of side.
    gtint_t K = ((SIDE == 'l')||(SIDE == 'L'))? M : N;
    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDB);
    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, M, N, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), zero_mat.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// zero alpha and non-zero/non-unit beta - scale C only
TYPED_TEST(hemm_IIT_ERS, ZeroAlpha_OtherBeta)
{
    using T = TypeParam;

    T alpha, beta;
    testinghelpers::initzero<T>( alpha );
    beta = T{2.0};
    double thresh = testinghelpers::getEpsilon<T>();

    // Set the dimension for row/col of A, depending on the value of side.
    gtint_t K = ((SIDE == 'l')||(SIDE == 'L'))? M : N;
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c2(c);
    std::vector<T> c_ref(c);

    testinghelpers::ref_hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, M, N, alpha,
               a.data(), LDA, b.data(), LDB, beta, c_ref.data(), LDC );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, M, N, &alpha, nullptr, LDA, nullptr, LDB, &beta, c2.data(), LDC );
    computediff<T>( "C", STORAGE, N, N, c2.data(), c_ref.data(), LDC, thresh);
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    hemm<T>( STORAGE, SIDE, UPLO, CONJ, TRANS, M, N, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC, thresh);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

#endif

