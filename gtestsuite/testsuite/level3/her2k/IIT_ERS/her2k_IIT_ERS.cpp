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
#include "level3/her2k/test_her2k.h"
#include "inc/check_error.h"
#include "common/wrong_inputs_helpers.h"

template <typename T>
class her2k_IIT_ERS : public ::testing::Test {};
typedef ::testing::Types<scomplex, dcomplex> TypeParam; // The supported datatypes from BLAS calls for her2k
TYPED_TEST_SUITE(her2k_IIT_ERS, TypeParam); // Defining individual testsuites based on the datatype support.

// Adding namespace to get default parameters(valid case) from testinghelpers/common/wrong_input_helpers.h.
//using namespace testinghelpers::IIT;

#if defined(TEST_CBLAS)
#define INFO_OFFSET 1
#else
#define INFO_OFFSET 0
#endif

#if defined(TEST_CBLAS)

// When info == 1
TYPED_TEST(her2k_IIT_ERS, invalid_storage)
{
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    static const char STORAGE = 'c';
    static const char UPLO = 'u';
    static const char TRANS = 'n';
    static const gtint_t N = 4;
    static const gtint_t K = 4;
    // Set the dimension for row/col of A and B, depending on the value of trans.
    gtint_t LDA = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDB = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDC = N;
    T alpha;
    RT beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<RT>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    her2k<T>( 'x', UPLO, TRANS, TRANS, N, K, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 1 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS her2k with a invalid value for TRANS value for A.
    her2k<T>( 'x', UPLO, TRANS, TRANS, N, K, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
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

    BLAS exceptions get triggered in the following cases(for her2k):
    1. When SIDE != 'L' || SIDE != 'R' (info = 1)
    2. When UPLO != 'U' || UPLO != 'L' (info = 2)
    3. When m < 0 (info = 3)
    4. When n < 0 (info = 4)
    6. When lda < max(1, thresh) (info = 7), thresh set based on SIDE value
    7. When ldb < max(1, m) (info = 9)
    8. When ldc < max(1, m) (info = 12)

*/

// When info == 1
TYPED_TEST(her2k_IIT_ERS, invalid_uplo)
{
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    static const char STORAGE = 'c';
    static const char TRANS = 'n';
    static const gtint_t N = 4;
    static const gtint_t K = 4;
    // Set the dimension for row/col of A and B, depending on the value of trans.
    gtint_t LDA = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDB = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDC = N;
    T alpha;
    RT beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<RT>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    her2k<T>( STORAGE, 'p', TRANS, TRANS, N, K, nullptr, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#else
    her2k<T>( STORAGE, 'p', TRANS, TRANS, N, K, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+1 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS her2k with a invalid value for TRANS value for A.
    her2k<T>( STORAGE, 'p', TRANS, TRANS, N, K, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+1 );
#endif
}

// When info == 2
TYPED_TEST(her2k_IIT_ERS, invalid_trans)
{
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    static const char STORAGE = 'c';
    static const char UPLO = 'u';
    static const char TRANS = 'n';
    static const gtint_t N = 4;
    static const gtint_t K = 4;
    // Set the dimension for row/col of A and B, depending on the value of trans.
    gtint_t LDA = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDB = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDC = N;
    T alpha;
    RT beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<RT>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    her2k<T>( STORAGE, UPLO, 'p', TRANS, N, K, nullptr, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#else
    her2k<T>( STORAGE, UPLO, 'p', TRANS, N, K, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+2 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS her2k with a invalid value for TRANS value for B.
    her2k<T>( STORAGE, UPLO, 'p', TRANS, N, K, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+2 );
#endif
}

// When info == 3
TYPED_TEST(her2k_IIT_ERS, n_lt_zero)
{
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    static const char STORAGE = 'c';
    static const char UPLO = 'u';
    static const char TRANS = 'n';
    static const gtint_t N = 4;
    static const gtint_t K = 4;
    // Set the dimension for row/col of A and B, depending on the value of trans.
    gtint_t LDA = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDB = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDC = N;
    T alpha;
    RT beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<RT>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, -1, K, nullptr, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#else
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, -1, K, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 3 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS her2k with a invalid value for m.
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, -1, K, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 3 );
#endif
}

// When info == 4
TYPED_TEST(her2k_IIT_ERS, k_lt_zero)
{
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    static const char STORAGE = 'c';
    static const char UPLO = 'u';
    static const char TRANS = 'n';
    static const gtint_t N = 4;
    static const gtint_t K = 4;
    // Set the dimension for row/col of A and B, depending on the value of trans.
    gtint_t LDA = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDB = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDC = N;
    T alpha;
    RT beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<RT>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, -1, nullptr, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#else
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, -1, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 4 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS her2k with a invalid value for n.
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, -1, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 4 );
#endif
}

// When info == 7
TYPED_TEST(her2k_IIT_ERS, invalid_lda_trans_n)
{
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    static const char STORAGE = 'c';
    static const char UPLO = 'u';
    static const char TRANS = 'n';
    static const gtint_t N = 4;
    static const gtint_t K = 4;
    // Set the dimension for row/col of A and B, depending on the value of trans.
    gtint_t LDA = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDB = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDC = N;
    T alpha;
    RT beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<RT>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, K, nullptr, nullptr, LDA - 1, nullptr, LDB, &beta, nullptr, LDC );
#else
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, K, &alpha, nullptr, LDA - 1, nullptr, LDB, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 7 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS her2k with a invalid value for lda.
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, K, &alpha, a.data(), LDA - 1, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 7 );
#endif
}

// When info == 2 (invalid value of trans)
TYPED_TEST(her2k_IIT_ERS, invalid_lda_trans_t)
{
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    static const char STORAGE = 'c';
    static const char UPLO = 'u';
    static const char TRANS = 'T';
    static const gtint_t N = 4;
    static const gtint_t K = 4;
    // Set the dimension for row/col of A and B, depending on the value of trans.
    gtint_t LDA = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDB = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDC = N;
    T alpha;
    RT beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<RT>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, K, nullptr, nullptr, LDA - 1, nullptr, LDB, &beta, nullptr, LDC );
#else
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, K, &alpha, nullptr, LDA - 1, nullptr, LDB, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 2 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS her2k with a invalid value for lda.
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, K, &alpha, a.data(), LDA - 1, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 2 );
#endif
}

// When info == 7
TYPED_TEST(her2k_IIT_ERS, invalid_lda_trans_c)
{
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    static const char STORAGE = 'c';
    static const char UPLO = 'u';
    static const char TRANS = 'C';
    static const gtint_t N = 4;
    static const gtint_t K = 4;
    // Set the dimension for row/col of A and B, depending on the value of trans.
    gtint_t LDA = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDB = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDC = N;
    T alpha;
    RT beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<RT>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, K, nullptr, nullptr, LDA - 1, nullptr, LDB, &beta, nullptr, LDC );
#else
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, K, &alpha, nullptr, LDA - 1, nullptr, LDB, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 7 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS her2k with a invalid value for lda.
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, K, &alpha, a.data(), LDA - 1, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 7 );
#endif
}

// When info == 9
TYPED_TEST(her2k_IIT_ERS, invalid_ldb_trans_n)
{
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    static const char STORAGE = 'c';
    static const char UPLO = 'u';
    static const char TRANS = 'n';
    static const gtint_t N = 4;
    static const gtint_t K = 4;
    // Set the dimension for row/col of A and B, depending on the value of trans.
    gtint_t LDA = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDB = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDC = N;
    T alpha;
    RT beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<RT>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, K, nullptr, nullptr, LDA, nullptr, LDB - 1, &beta, nullptr, LDC );
#else
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, K, &alpha, nullptr, LDA, nullptr, LDB - 1, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 9 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS her2k with a invalid value for ldb.
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, K, &alpha, a.data(), LDA, b.data(), LDB - 1, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 9 );
#endif
}

// When info == 2 (invalid value of trans)
TYPED_TEST(her2k_IIT_ERS, invalid_ldb_trans_t)
{
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    static const char STORAGE = 'c';
    static const char UPLO = 'u';
    static const char TRANS = 'T';
    static const gtint_t N = 4;
    static const gtint_t K = 4;
    // Set the dimension for row/col of A and B, depending on the value of trans.
    gtint_t LDA = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDB = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDC = N;
    T alpha;
    RT beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<RT>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, K, nullptr, nullptr, LDA, nullptr, LDB - 1, &beta, nullptr, LDC );
#else
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, K, &alpha, nullptr, LDA, nullptr, LDB - 1, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 2 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS her2k with a invalid value for ldb.
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, K, &alpha, a.data(), LDA, b.data(), LDB - 1, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 2 );
#endif
}

// When info == 9 for real data. With complex data, info == 2.
TYPED_TEST(her2k_IIT_ERS, invalid_ldb_trans_c)
{
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    static const char STORAGE = 'c';
    static const char UPLO = 'u';
    static const char TRANS = 'C';
    static const gtint_t N = 4;
    static const gtint_t K = 4;
    // Set the dimension for row/col of A and B, depending on the value of trans.
    gtint_t LDA = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDB = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDC = N;
    T alpha;
    RT beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<RT>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, K, nullptr, nullptr, LDA, nullptr, LDB - 1, &beta, nullptr, LDC );
#else
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, K, &alpha, nullptr, LDA, nullptr, LDB - 1, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 9 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS her2k with a invalid value for ldb.
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, K, &alpha, a.data(), LDA, b.data(), LDB - 1, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 9 );
#endif
}

// When info == 12
TYPED_TEST(her2k_IIT_ERS, invalid_ldc)
{
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    static const char STORAGE = 'c';
    static const char UPLO = 'u';
    static const char TRANS = 'n';
    static const gtint_t N = 4;
    static const gtint_t K = 4;
    // Set the dimension for row/col of A and B, depending on the value of trans.
    gtint_t LDA = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDB = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDC = N;
    T alpha;
    RT beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<RT>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, K, nullptr, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC - 1 );
#else
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, K, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC - 1 );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 12 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    // Call BLIS her2k with a invalid value for ldc.
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, K, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC - 1 );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 12 );
#endif
}

/*
    Early Return Scenarios(ERS) :

    The her2k API is expected to return early in the following cases:

    1. When n == 0.
    2. When (alpha == 0 or k == 0) and beta == 1.
    3. When alpha == 0 and beta == 0, set C = 0 only
    4. When alpha == 0 and beta /= 0 or 1, scale C by beta only

*/

// When n is 0
TYPED_TEST(her2k_IIT_ERS, n_eq_zero)
{
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    static const char STORAGE = 'c';
    static const char UPLO = 'u';
    static const char TRANS = 'n';
    static const gtint_t N = 4;
    static const gtint_t K = 4;
    // Set the dimension for row/col of A and B, depending on the value of trans.
    gtint_t LDA = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDB = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDC = N;
    T alpha;
    RT beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<RT>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, 0, N, nullptr, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#else
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, 0, N, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    her2k<T>( STORAGE, UPLO, TRANS, TRANS, 0, N, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// When alpha is 0 and beta is 1
TYPED_TEST(her2k_IIT_ERS, alpha_zero_beta_one)
{
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    static const char STORAGE = 'c';
    static const char UPLO = 'u';
    static const char TRANS = 'n';
    static const gtint_t N = 4;
    static const gtint_t K = 4;
    // Set the dimension for row/col of A and B, depending on the value of trans.
    gtint_t LDA = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDB = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDC = N;
    T alpha;
    RT beta;
    testinghelpers::initzero<T>( alpha );
    testinghelpers::initone<RT>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, K, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#else
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, K, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, K, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// When k is 0 and beta is 1
TYPED_TEST(her2k_IIT_ERS, k_zero_beta_one)
{
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    static const char STORAGE = 'c';
    static const char UPLO = 'u';
    static const char TRANS = 'n';
    static const gtint_t N = 4;
    static const gtint_t K = 4;
    // Set the dimension for row/col of A and B, depending on the value of trans.
    gtint_t LDA = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDB = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDC = N;
    T alpha;
    RT beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<RT>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, 0, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#else
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, 0, &alpha, nullptr, LDA, nullptr, LDB, &beta, nullptr, LDC );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c_ref(c);

    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, 0, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// zero alpha and zero beta - set C to 0
TYPED_TEST(her2k_IIT_ERS, ZeroAlpha_ZeroBeta)
{
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    static const char STORAGE = 'c';
    static const char UPLO = 'u';
    static const char TRANS = 'n';
    static const gtint_t N = 4;
    static const gtint_t K = 4;
    // Set the dimension for row/col of A and B, depending on the value of trans.
    gtint_t LDA = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDB = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDC = N;
    T alpha, zero;
    RT beta;
    testinghelpers::initzero<T>( alpha );
    testinghelpers::initzero<RT>( beta );
    testinghelpers::initzero<T>( zero );

    // Matrix C should not be read, only set.
    std::vector<T> c( testinghelpers::matsize( STORAGE, 'N', N, N, LDC ) );
    testinghelpers::set_matrix( STORAGE, N, c.data(), UPLO, LDC, testinghelpers::aocl_extreme<T>() );
    std::vector<T> c2(c);
    // Set up expected output matrix
    std::vector<T> zero_mat( testinghelpers::matsize( STORAGE, 'N', N, N, LDC ) );
    testinghelpers::set_matrix( STORAGE, N, zero_mat.data(), UPLO, LDC, zero );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, K, &alpha, nullptr, LDA, nullptr, LDB, &beta, c2.data(), LDC );
    computediff<T>( "C", STORAGE, N, N, c2.data(), zero_mat.data(), LDC);
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDB);
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, K, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), zero_mat.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// zero alpha and non-zero/non-unit beta - scale C only
TYPED_TEST(her2k_IIT_ERS, ZeroAlpha_OtherBeta)
{
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    static const char STORAGE = 'c';
    static const char UPLO = 'u';
    static const char TRANS = 'n';
    static const gtint_t N = 4;
    static const gtint_t K = 4;
    // Set the dimension for row/col of A and B, depending on the value of trans.
    gtint_t LDA = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDB = ((TRANS == 'n')||(TRANS == 'N'))? N : K;
    gtint_t LDC = N;
    T alpha;
    RT beta;
    testinghelpers::initzero<T>( alpha );
    beta = RT{2.0};
    double thresh = testinghelpers::getEpsilon<T>();

    // Test with all arguments correct except for the value we are choosing to test.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDA);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, TRANS, N, K, LDB);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC);
    // Copy so that we check that the elements of C are not modified.
    std::vector<T> c2(c);
    std::vector<T> c_ref(c);

    testinghelpers::ref_her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, K, &alpha,
               a.data(), LDA, b.data(), LDB, beta, c_ref.data(), LDC );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, K, &alpha, nullptr, LDA, nullptr, LDB, &beta, c2.data(), LDC );
    computediff<T>( "C", STORAGE, N, N, c2.data(), c_ref.data(), LDC, thresh);
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.
    her2k<T>( STORAGE, UPLO, TRANS, TRANS, N, K, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
    // Use bitwise comparison (no threshold).
    computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC, thresh);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

#endif

