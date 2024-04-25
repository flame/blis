/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

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
#include "gemm.h"
#include "inc/check_error.h"
#include "common/wrong_inputs_helpers.h"

template <typename T>
class Gemm_IIT_ERS_Test : public ::testing::Test {};
typedef ::testing::Types<float, double, scomplex, dcomplex> TypeParam; // The supported datatypes from BLAS calls for GEMM
TYPED_TEST_SUITE(Gemm_IIT_ERS_Test, TypeParam); // Defining individual testsuites based on the datatype support.

// Adding namespace to get default parameters(valid case) from testinghelpers/common/wrong_input_helpers.h.
using namespace testinghelpers::IIT;

#if defined(TEST_BLAS) || defined(TEST_CBLAS)

/*
    Incorrect Input Testing(IIT)

    BLAS exceptions get triggered in the following cases(for GEMM):
    1. When TRANSA != 'N' || TRANSA != 'T'  || TRANSA != 'C' (info = 1)
    2. When TRANSB != 'N' || TRANSB != 'T'  || TRANSB != 'C' (info = 2)
    3. When m < 0 (info = 3)
    4. When n < 0 (info = 4)
    5. When k < 0 (info = 5)
    6. When lda < max(1, thresh) (info = 8), thresh set based on TRANSA value
    7. When ldb < max(1, thresh) (info = 10), thresh set based on TRANSB value
    8. When ldc < max(1, n) (info = 13)

*/

// When info == 1
TYPED_TEST(Gemm_IIT_ERS_Test, invalid_transa)
{
  using T = TypeParam;
  // Defining the C matrix with values for debugging purposes
  std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC);
  std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, K, LDA);
  std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, N, LDB);
  // Copy so that we check that the elements of C are not modified.
  std::vector<T> c_ref(c);
  T alpha, beta;

  testinghelpers::initone<T>( alpha );
  testinghelpers::initone<T>( beta );

  // Call BLIS Gemm with a invalid value for TRANS value for A.
  gemm<T>( STORAGE, 'p', TRANS, M, N, K, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
  // Use bitwise comparison (no threshold).
  computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 1 );
#endif
}

// When info == 2
TYPED_TEST(Gemm_IIT_ERS_Test, invalid_transb)
{
  using T = TypeParam;
  // Defining the C matrix with values for debugging purposes
  std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC);
  std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, K, LDA);
  std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, N, LDB);
  // Copy so that we check that the elements of C are not modified.
  std::vector<T> c_ref(c);
  T alpha, beta;

  testinghelpers::initone<T>( alpha );
  testinghelpers::initone<T>( beta );

  // Call BLIS Gemm with a invalid value for TRANS value for B.
  gemm<T>( STORAGE, TRANS, 'p', M, N, K, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
  // Use bitwise comparison (no threshold).
  computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 2 );
#endif
}

// When info == 3
TYPED_TEST(Gemm_IIT_ERS_Test, m_lt_zero)
{
  using T = TypeParam;
  // Defining the C matrix with values for debugging purposes
  std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC);
  std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, K, LDA);
  std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, N, LDB);
  // Copy so that we check that the elements of C are not modified.
  std::vector<T> c_ref(c);
  T alpha, beta;

  testinghelpers::initone<T>( alpha );
  testinghelpers::initone<T>( beta );
  // Call BLIS Gemm with a invalid value for m.
  gemm<T>( STORAGE, TRANS, TRANS, -1, N, K, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
  // Use bitwise comparison (no threshold).
  computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 3 );
#endif
}

// When info == 4
TYPED_TEST(Gemm_IIT_ERS_Test, n_lt_zero)
{
  using T = TypeParam;
  // Defining the C matrix with values for debugging purposes
  std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC);
  std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, K, LDA);
  std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, N, LDB);
  // Copy so that we check that the elements of C are not modified.
  std::vector<T> c_ref(c);
  T alpha, beta;

  testinghelpers::initone<T>( alpha );
  testinghelpers::initone<T>( beta );
  // Call BLIS Gemm with a invalid value for n.
  gemm<T>( STORAGE, TRANS, TRANS, M, -1, K, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
  // Use bitwise comparison (no threshold).
  computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 4 );
#endif
}

// When info == 5
TYPED_TEST(Gemm_IIT_ERS_Test, k_lt_zero)
{
  using T = TypeParam;
  // Defining the C matrix with values for debugging purposes
  std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC);
  std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, K, LDA);
  std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, N, LDB);
  // Copy so that we check that the elements of C are not modified.
  std::vector<T> c_ref(c);
  T alpha, beta;

  testinghelpers::initone<T>( alpha );
  testinghelpers::initone<T>( beta );
  // Call BLIS Gemm with a invalid value for k.
  gemm<T>( STORAGE, TRANS, TRANS, M, N, -1, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
  // Use bitwise comparison (no threshold).
  computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 5 );
#endif
}

// When info == 8
TYPED_TEST(Gemm_IIT_ERS_Test, invalid_lda)
{
  using T = TypeParam;
  // Defining the C matrix with values for debugging purposes
  std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC);
  std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, K, LDA);
  std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, N, LDB);
  // Copy so that we check that the elements of C are not modified.
  std::vector<T> c_ref(c);
  T alpha, beta;

  testinghelpers::initone<T>( alpha );
  testinghelpers::initone<T>( beta );
  // Call BLIS Gemm with a invalid value for lda.
  gemm<T>( STORAGE, TRANS, TRANS, M, N, K, &alpha, a.data(), LDA - 1, b.data(), LDB, &beta, c.data(), LDC );
  // Use bitwise comparison (no threshold).
  computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 8 );
#endif
}

// When info == 10
TYPED_TEST(Gemm_IIT_ERS_Test, invalid_ldb)
{
  using T = TypeParam;
  // Defining the C matrix with values for debugging purposes
  std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC);
  std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, K, LDA);
  std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, N, LDB);
  // Copy so that we check that the elements of C are not modified.
  std::vector<T> c_ref(c);
  T alpha, beta;

  testinghelpers::initone<T>( alpha );
  testinghelpers::initone<T>( beta );
  // Call BLIS Gemm with a invalid value for ldb.
  gemm<T>( STORAGE, TRANS, TRANS, M, N, K, &alpha, a.data(), LDA, b.data(), LDB - 1, &beta, c.data(), LDC );
  // Use bitwise comparison (no threshold).
  computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 10 );
#endif
}

// When info == 13
TYPED_TEST(Gemm_IIT_ERS_Test, invalid_ldc)
{
  using T = TypeParam;
  // Defining the C matrix with values for debugging purposes
  std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC);
  std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, K, LDA);
  std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, N, LDB);
  // Copy so that we check that the elements of C are not modified.
  std::vector<T> c_ref(c);
  T alpha, beta;

  testinghelpers::initone<T>( alpha );
  testinghelpers::initone<T>( beta );
  // Call BLIS Gemm with a invalid value for ldc.
  gemm<T>( STORAGE, TRANS, TRANS, M, N, K, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC - 1 );
  // Use bitwise comparison (no threshold).
  computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 13 );
#endif
}

/*
    Early Return Scenarios(ERS) :

    The GEMM API is expected to return early in the following cases:

    1. When m == 0.
    2. When n == 0.
    3. When (alpha == 0 or k == 0) and beta == 1.

*/

// When m is 0
TYPED_TEST(Gemm_IIT_ERS_Test, m_eq_zero)
{
  using T = TypeParam;
  // Defining the C matrix with values for debugging purposes
  std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC);
  std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, K, LDA);
  std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, N, LDB);
  // Copy so that we check that the elements of C are not modified.
  std::vector<T> c_ref(c);
  T alpha, beta;

  testinghelpers::initone<T>( alpha );
  testinghelpers::initone<T>( beta );
  gemm<T>( STORAGE, TRANS, TRANS, 0, N, K, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
  // Use bitwise comparison (no threshold).
  computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// When n is 0
TYPED_TEST(Gemm_IIT_ERS_Test, n_eq_zero)
{
  using T = TypeParam;
  // Defining the C matrix with values for debugging purposes
  std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC);
  std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, K, LDA);
  std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, N, LDB);
  // Copy so that we check that the elements of C are not modified.
  std::vector<T> c_ref(c);
  T alpha, beta;

  testinghelpers::initone<T>( alpha );
  testinghelpers::initone<T>( beta );
  gemm<T>( STORAGE, TRANS, TRANS, M, 0, K, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
  // Use bitwise comparison (no threshold).
  computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// When alpha is 0 and beta is 1
TYPED_TEST(Gemm_IIT_ERS_Test, alpha_zero_beta_one)
{
  using T = TypeParam;
  // Defining the C matrix with values for debugging purposes
  // Defining the C matrix with values for debugging purposes
  std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC);
  std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, K, LDA);
  std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, N, LDB);
  // Copy so that we check that the elements of C are not modified.
  std::vector<T> c_ref(c);
  T alpha, beta;

  testinghelpers::initzero<T>( alpha );
  testinghelpers::initone<T>( beta );

  gemm<T>( STORAGE, TRANS, TRANS, M, N, K, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
  // Use bitwise comparison (no threshold).
  computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// When k is 0 and beta is 1
TYPED_TEST(Gemm_IIT_ERS_Test, k_zero_beta_one)
{
  using T = TypeParam;
  // Defining the C matrix with values for debugging purposes
  // Defining the C matrix with values for debugging purposes
  std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC);
  std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, K, LDA);
  std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, N, LDB);
  // Copy so that we check that the elements of C are not modified.
  std::vector<T> c_ref(c);
  T alpha, beta;

  testinghelpers::initone<T>( alpha );
  testinghelpers::initone<T>( beta );

  gemm<T>( STORAGE, TRANS, TRANS, M, N, 0, &alpha, a.data(), LDA, b.data(), LDB, &beta, c.data(), LDC );
  // Use bitwise comparison (no threshold).
  computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

#if 0
/**
 * These testcases are disabled as blis aborts for null buffers.
 * Once respective blis framework changes are done to simply pass down
 * the error to the top level these testcases can be enabled.
*/
// When a matrix is null
TYPED_TEST(Gemm_IIT_ERS_Test, null_a_matrix)
{
  using T = TypeParam;
  // Defining the C matrix with values for debugging purposes
  std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);
  std::vector<T> b = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', K, N, LDB);
  // Copy so that we check that the elements of C are not modified.
  std::vector<T> c_ref(c);
  T alpha, beta;

  testinghelpers::initone<T>( alpha );
  testinghelpers::initone<T>( beta );

  gemm<T>( STORAGE, TRANS, TRANS, M, N, K, &alpha, nullptr, LDA, b.data(), LDB, &beta, c.data(), LDC );
  // Use bitwise comparison (no threshold).
  computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// When b matrix is null
TYPED_TEST(Gemm_IIT_ERS_Test, null_b_matrix)
{
  using T = TypeParam;
  // Defining the C matrix with values for debugging purposes
  std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, N, LDC);
  std::vector<T> a = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', M, K, LDA);
  // Copy so that we check that the elements of C are not modified.
  std::vector<T> c_ref(c);
  T alpha, beta;

  testinghelpers::initone<T>( alpha );
  testinghelpers::initone<T>( beta );

  gemm<T>( STORAGE, TRANS, TRANS, M, N, K, &alpha, a.data(), LDA, nullptr, LDB, &beta, c.data(), LDC );
  // Use bitwise comparison (no threshold).
  computediff<T>( "C", STORAGE, N, N, c.data(), c_ref.data(), LDC);

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}
#endif /* #IF 0 ENDS HERE */
#endif

