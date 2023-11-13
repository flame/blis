/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

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
class GEMM_Compute_IIT_ERS_Test : public ::testing::Test {};
typedef ::testing::Types<float, double> TypeParam;
TYPED_TEST_SUITE(GEMM_Compute_IIT_ERS_Test, TypeParam);

using namespace testinghelpers::IIT;

#ifdef TEST_BLAS

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
TYPED_TEST(GEMM_Compute_IIT_ERS_Test, invalid_transa)
{
  using T = TypeParam;
  // Defining the C matrix with values for debugging purposes
  std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC, 'f');

  // Copy so that we check that the elements of C are not modified.
  std::vector<T> c_ref(c);
  // Call BLIS Gemm with a invalid value for TRANS value for A.
  gemm_compute<T>( STORAGE, 'x', TRANS, 'U', 'U', M, N, K, nullptr, nullptr, LDA, nullptr, LDB, nullptr, nullptr, LDC );
  // Use bitwise comparison (no threshold).
  computediff<T>( STORAGE, N, N, c.data(), c_ref.data(), LDC);
}

// When info == 2
TYPED_TEST(GEMM_Compute_IIT_ERS_Test, invalid_transb)
{
  using T = TypeParam;
  // Defining the C matrix with values for debugging purposes
  std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC, 'f');

  // Copy so that we check that the elements of C are not modified.
  std::vector<T> c_ref(c);
  // Call BLIS Gemm with a invalid value for TRANS value for A.
  gemm_compute<T>( STORAGE, TRANS, 'x', 'U', 'U', M, N, K, nullptr, nullptr, LDA, nullptr, LDB, nullptr, nullptr, LDC );
  // Use bitwise comparison (no threshold).
  computediff<T>( STORAGE, N, N, c.data(), c_ref.data(), LDC);
}

// When info == 3
TYPED_TEST(GEMM_Compute_IIT_ERS_Test, m_lt_zero)
{
  using T = TypeParam;
  // Defining the C matrix with values for debugging purposes
  std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC, 'f');

  // Copy so that we check that the elements of C are not modified.
  std::vector<T> c_ref(c);
  // Call BLIS Gemm with a invalid value for m.
  gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', -1, N, K, nullptr, nullptr, LDA, nullptr, LDB, nullptr, nullptr, LDC );
  // Use bitwise comparison (no threshold).
  computediff<T>( STORAGE, N, N, c.data(), c_ref.data(), LDC);
}

// When info == 4
TYPED_TEST(GEMM_Compute_IIT_ERS_Test, n_lt_zero)
{
  using T = TypeParam;
  // Defining the C matrix with values for debugging purposes
  std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC, 'f');

  // Copy so that we check that the elements of C are not modified.
  std::vector<T> c_ref(c);
  // Call BLIS Gemm with a invalid value for m.
  gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, -1, K, nullptr, nullptr, LDA, nullptr, LDB, nullptr, nullptr, LDC );
  // Use bitwise comparison (no threshold).
  computediff<T>( STORAGE, N, N, c.data(), c_ref.data(), LDC);
}

// When info == 5
TYPED_TEST(GEMM_Compute_IIT_ERS_Test, k_lt_zero)
{
  using T = TypeParam;
  // Defining the C matrix with values for debugging purposes
  std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC, 'f');

  // Copy so that we check that the elements of C are not modified.
  std::vector<T> c_ref(c);
  // Call BLIS Gemm with a invalid value for m.
  gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, N, -1, nullptr, nullptr, LDA, nullptr, LDB, nullptr, nullptr, LDC );
  // Use bitwise comparison (no threshold).
  computediff<T>( STORAGE, N, N, c.data(), c_ref.data(), LDC);
}

// When info == 7
TYPED_TEST(GEMM_Compute_IIT_ERS_Test, invalid_lda)
{
  using T = TypeParam;
  // Defining the C matrix with values for debugging purposes
  std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC, 'f');

  // Copy so that we check that the elements of C are not modified.
  std::vector<T> c_ref(c);
  // Call BLIS Gemm with a invalid value for m.
  gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, N, K, nullptr, nullptr, LDA - 1, nullptr, LDB, nullptr, nullptr, LDC );
  // Use bitwise comparison (no threshold).
  computediff<T>( STORAGE, N, N, c.data(), c_ref.data(), LDC);
}

// When info == 9
TYPED_TEST(GEMM_Compute_IIT_ERS_Test, invalid_ldb)
{
  using T = TypeParam;
  // Defining the C matrix with values for debugging purposes
  std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC, 'f');

  // Copy so that we check that the elements of C are not modified.
  std::vector<T> c_ref(c);
  // Call BLIS Gemm with a invalid value for m.
  gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, N, K, nullptr, nullptr, LDA, nullptr, LDB - 1, nullptr, nullptr, LDC );
  // Use bitwise comparison (no threshold).
  computediff<T>( STORAGE, N, N, c.data(), c_ref.data(), LDC);
}

// When info == 12
TYPED_TEST(GEMM_Compute_IIT_ERS_Test, invalid_ldc_lt_zero)
{
  using T = TypeParam;
  // Defining the C matrix with values for debugging purposes
  std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC, 'f');

  // Copy so that we check that the elements of C are not modified.
  std::vector<T> c_ref(c);
  // Call BLIS Gemm with a invalid value for m.
  gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, N, K, nullptr, nullptr, LDA, nullptr, LDB, nullptr, nullptr, -1 );
  // Use bitwise comparison (no threshold).
  computediff<T>( STORAGE, N, N, c.data(), c_ref.data(), LDC);
}

// When info == 12
TYPED_TEST(GEMM_Compute_IIT_ERS_Test, invalid_ldc)
{
  using T = TypeParam;
  // Defining the C matrix with values for debugging purposes
  std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC, 'f');

  // Copy so that we check that the elements of C are not modified.
  std::vector<T> c_ref(c);
  // Call BLIS Gemm with a invalid value for m.
  gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, N, K, nullptr, nullptr, LDA, nullptr, LDB, nullptr, nullptr, LDC - 1 );
  // Use bitwise comparison (no threshold).
  computediff<T>( STORAGE, N, N, c.data(), c_ref.data(), LDC);
}

/*
    Early Return Scenarios(ERS) :

    The GEMM Compute API is expected to return early in the following cases:

    1. When m == 0.
    2. When n == 0.
*/

// When m = 0
TYPED_TEST(GEMM_Compute_IIT_ERS_Test, m_eq_zero)
{
  using T = TypeParam;
  // Defining the C matrix with values for debugging purposes
  std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC, 'f');

  // Copy so that we check that the elements of C are not modified.
  std::vector<T> c_ref(c);
  // Call BLIS Gemm with a invalid value for m.
  gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', 0, N, K, nullptr, nullptr, LDA, nullptr, LDB, nullptr, nullptr, LDC );
  // Use bitwise comparison (no threshold).
  computediff<T>( STORAGE, N, N, c.data(), c_ref.data(), LDC);
}

// When n = 0
TYPED_TEST(GEMM_Compute_IIT_ERS_Test, n_eq_zero)
{
  using T = TypeParam;
  // Defining the C matrix with values for debugging purposes
  std::vector<T> c = testinghelpers::get_random_matrix<T>(-10, 10, STORAGE, 'N', N, N, LDC, 'f');

  // Copy so that we check that the elements of C are not modified.
  std::vector<T> c_ref(c);
  // Call BLIS Gemm with a invalid value for m.
  gemm_compute<T>( STORAGE, TRANS, TRANS, 'U', 'U', M, 0, K, nullptr, nullptr, LDA, nullptr, LDB, nullptr, nullptr, LDC );
  // Use bitwise comparison (no threshold).
  computediff<T>( STORAGE, N, N, c.data(), c_ref.data(), LDC);
}
#endif
