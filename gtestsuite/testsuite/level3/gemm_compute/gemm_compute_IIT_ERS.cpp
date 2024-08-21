/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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
#include "test_gemm_compute.h"
#include "common/wrong_inputs_helpers.h"
#include "common/testing_helpers.h"
#include "inc/check_error.h"

template <typename T>
class gemm_compute_IIT_ERS : public ::testing::Test {};
typedef ::testing::Types<float, double> TypeParam;
TYPED_TEST_SUITE(gemm_compute_IIT_ERS, TypeParam);

using namespace testinghelpers::IIT;

#if defined(TEST_CBLAS)
#define INFO_OFFSET 1
#else
#define INFO_OFFSET 0
#endif

#if defined(TEST_CBLAS)

// When info == 1
TYPED_TEST(gemm_compute_IIT_ERS, invalid_storage)
{
    using T = TypeParam;
    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    gemm_compute<T>( 'x', TRANS, TRANS, 'U', 'U', M, N, K, nullptr, nullptr, LDA, nullptr, LDB, nullptr, nullptr, LDC );
#else
    gemm_compute<T>( 'x', TRANS, TRANS, 'U', 'U', M, N, K, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 1 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, N, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);

    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS Gemm with a invalid value for storage.
    gemm_compute<T>( 'x', TRANS, TRANS, 'U', 'U', M, N, K, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 1 );
#endif
}

#endif

#if defined(TEST_BLAS_LIKE) || defined(TEST_CBLAS)

/*
    Incorrect Input Testing(IIT)

    BLAS exceptions get triggered in the following cases(for GEMM Compute):
    1. When TRANSA != 'N' || TRANSA != 'T' || TRANSA != 'C' || TRANSA != 'P' (info = 1)
    2. When TRANSB != 'N' || TRANSB != 'T' || TRANSB != 'C' || TRANSB != 'P' (info = 2)
    3. When m < 0 (info = 3)
    4. When n < 0 (info = 4)
    5. When k < 0 (info = 5)
    6. When lda < max(1, thresh) (info = 7), thresh set based on TRANSA value
    7. When ldb < max(1, thresh) (info = 9), thresh set based on TRANSB value
    8. When ldc < max(1, n) (info = 12)
*/

// When info == 1
TYPED_TEST(gemm_compute_IIT_ERS, invalid_transa)
{
    using T = TypeParam;
    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    gemm_compute<T>( STORAGE, 'x', TRANS, 'U', 'U', M, N, K, nullptr, nullptr, LDA, nullptr, LDB, nullptr, nullptr, LDC );
#else
    gemm_compute<T>( STORAGE, 'x', TRANS, 'U', 'U', M, N, K, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+1 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, N, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);

    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);
    // Call BLIS Gemm with a invalid value for TRANS value for A.
    gemm_compute<T>( STORAGE, 'x', TRANS, 'U', 'U', M, N, K, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+1 );
#endif
}

// When info == 2
TYPED_TEST(gemm_compute_IIT_ERS, invalid_transb)
{
    using T = TypeParam;
    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    gemm_compute<T>( STORAGE, TRANS, 'x', 'U', 'U', M, N, K, nullptr, nullptr, LDA, nullptr, LDB, nullptr, nullptr, LDC );
#else
    gemm_compute<T>( STORAGE, TRANS, 'x', 'U', 'U', M, N, K, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+2 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, N, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);

    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);
    // Call BLIS Gemm with a invalid value for TRANS value for B.
    gemm_compute<T>( STORAGE, TRANS, 'x', 'U', 'U', M, N, K, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+2 );
#endif
}

// When info == 3
TYPED_TEST(gemm_compute_IIT_ERS, m_lt_zero)
{
    using T = TypeParam;
    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', -1, N, K, nullptr, nullptr, LDA, nullptr, LDB, nullptr, nullptr, LDC );
#else
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', -1, N, K, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 3 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, N, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);

    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS Gemm with a invalid value for m.
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', -1, N, K, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 3 );
#endif
}

// When info == 4
TYPED_TEST(gemm_compute_IIT_ERS, n_lt_zero)
{
    using T = TypeParam;
    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, -1, K, nullptr, nullptr, LDA, nullptr, LDB, nullptr, nullptr, LDC );
#else
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, -1, K, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 4 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, N, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);

    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS Gemm with a invalid value for n.
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, -1, K, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 4 );
#endif
}

// When info == 5
TYPED_TEST(gemm_compute_IIT_ERS, k_lt_zero)
{
    using T = TypeParam;
    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, N, -1, nullptr, nullptr, LDA, nullptr, LDB, nullptr, nullptr, LDC );
#else
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, N, -1, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 5 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, N, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);

    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS Gemm with a invalid value for k.
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, N, -1, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 5 );
#endif
}

// When info == 7
TYPED_TEST(gemm_compute_IIT_ERS, invalid_lda)
{
    using T = TypeParam;
    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, N, K, nullptr, nullptr, LDA - 1, nullptr, LDB, nullptr, nullptr, LDC );
#else
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, N, K, &alpha, nullptr, LDA - 1, nullptr, LDB, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 7 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, N, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);

    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS Gemm with a invalid value for lda.
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, N, K, &alpha, nullptr, LDA - 1, nullptr, LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 7 );
#endif
}

// When info == 9
TYPED_TEST(gemm_compute_IIT_ERS, invalid_ldb)
{
    using T = TypeParam;
    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, N, K, nullptr, nullptr, LDA, nullptr, LDB - 1, nullptr, nullptr, LDC );
#else
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, N, K, &alpha, nullptr, LDA, nullptr, LDB - 1, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 9 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, N, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);

    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS Gemm with a invalid value for ldb.
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, N, K, &alpha, a.data(), LDA, b.data(), LDB - 1, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 9 );
#endif
}

// When info == 12
TYPED_TEST(gemm_compute_IIT_ERS, invalid_ldc_lt_zero)
{
    using T = TypeParam;
    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, N, K, nullptr, nullptr, LDA, nullptr, LDB, nullptr, nullptr, -1 );
#else
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, N, K, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, -1);
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 12 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, N, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);

    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS Gemm with a invalid value for ldc.
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, N, K, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), -1 );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 12 );
#endif
}

// When info == 12
TYPED_TEST(gemm_compute_IIT_ERS, invalid_ldc)
{
    using T = TypeParam;
    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, N, K, nullptr, nullptr, LDA, nullptr, LDB, nullptr, nullptr, LDC - 1);
#else
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, N, K, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC - 1);
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 12 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, N, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);

    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS Gemm with a invalid value for ldc.
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, N, K, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC - 1 );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 12 );
#endif
}

/*
    Early Return Scenarios(ERS) :

    The GEMM Compute API is expected to return early in the following cases:

    1. When m == 0.
    2. When n == 0.
*/

// When m = 0
TYPED_TEST(gemm_compute_IIT_ERS, m_eq_zero)
{
    using T = TypeParam;
    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', 0, N, K, nullptr, nullptr, LDA, nullptr, LDB, nullptr, nullptr, LDC );
#else
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', 0, N, K, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, N, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);

    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS Gemm with a invalid value for m.
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', 0, N, K, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// When n = 0
TYPED_TEST(gemm_compute_IIT_ERS, n_eq_zero)
{
    using T = TypeParam;
    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, 0, K, nullptr, nullptr, LDA, nullptr, LDB, nullptr, nullptr, LDC );
#else
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, 0, K, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, N, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);

    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS Gemm with a invalid value for m.
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, 0, K, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// When k is 0 and beta is 1
TYPED_TEST(gemm_compute_IIT_ERS, k_zero_beta_one)
{
    using T = TypeParam;
    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, N, 0, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, N, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);

    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS Gemm with a invalid value for m.
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, N, 0, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// zero alpha and zero beta - set C to 0
TYPED_TEST(gemm_compute_IIT_ERS, ZeroAlpha_ZeroBeta)
{
    using T = TypeParam;

    T alpha, beta;
    testinghelpers::initzero<T>( alpha );
    testinghelpers::initzero<T>( beta );

    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> zero_mat = testinghelpers::get_random_matrix<T>(0, 0, STORAGE, 'n', M, N, LDB);

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, N, LDB);

    // Enable packing of A matrix to accound for alpha = 0 scaling.
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'P', 'U', M, N, K, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), zero_mat.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// zero alpha and non-zero/non-unit beta - scale C only
TYPED_TEST(gemm_compute_IIT_ERS, ZeroAlpha_OtherBeta)
{
    using T = TypeParam;

    T alpha, beta;
    testinghelpers::initzero<T>( alpha );
    beta = T{2.0};
    double thresh = testinghelpers::getEpsilon<T>();

    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, N, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    testinghelpers::ref_gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, N, K, alpha,
               a.data(), LDA, b.data(), LDB, beta, c_ref.data(), LDC );

    // Test with all arguments correct except for the value we are choosing to test.
    gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, N, K, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC, thresh);

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

#endif
