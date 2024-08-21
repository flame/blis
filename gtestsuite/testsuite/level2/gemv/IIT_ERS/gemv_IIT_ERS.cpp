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
#include "level2/gemv/test_gemv.h"
#include "common/wrong_inputs_helpers.h"
#include "common/testing_helpers.h"
#include "inc/check_error.h"

template <typename T>
class gemv_IIT_ERS : public ::testing::Test {};
typedef ::testing::Types<float, double, scomplex, dcomplex> TypeParam;
TYPED_TEST_SUITE(gemv_IIT_ERS, TypeParam);

using namespace testinghelpers::IIT;

#if defined(TEST_CBLAS)
#define INFO_OFFSET 1
#else
#define INFO_OFFSET 0
#endif

#if defined(TEST_CBLAS)
TYPED_TEST(gemv_IIT_ERS, invalid_storage)
{
    using T = TypeParam;
    gtint_t incx = 3;
    gtint_t incy = 3;

    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    gemv<T>( 'x', TRANS, CONJ, M, N, &alpha, nullptr, LDA,
                         nullptr, incx, &beta, nullptr, incy );
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 1 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, TRANS, M, N, LDA);
    std::vector<T> x = testinghelpers::get_random_vector<T>( 1, 3, N, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( 1, 3, M, incy );

    // Create a copy of c so that we can check reference results.
    std::vector<T> y_ref(y);
    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    gemv<T>( 'x', TRANS, CONJ, M, N, &alpha, a.data(), LDA,
                         x.data(), incx, &beta, y.data(), incy );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( "y", N, y.data(), y_ref.data(), incy);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 1 );
#endif
}

#endif

#if defined(TEST_BLAS_LIKE) || defined(TEST_CBLAS)

/*
    Incorrect Input Testing(IIT)

    BLAS exceptions get triggered in the following cases(for GEMM):
    1. When TRANS != 'N' || TRANS != 'T'  || TRANS != 'C' (info = 1)
    2. When m < 0 (info = 2)
    3. When n < 0 (info = 3)
    4. When lda < m (info = 6)
    5. When incx = 0 (info = 8)
    6. When incy = 0 (info = 11)

*/

TYPED_TEST(gemv_IIT_ERS, invalid_trans)
{
    using T = TypeParam;
    gtint_t incx = 3;
    gtint_t incy = 3;

    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    gemv<T>( STORAGE, 'p', CONJ, M, N, nullptr, nullptr, LDA,
                         nullptr, incx, nullptr, nullptr, incy );
#else
    gemv<T>( STORAGE, 'p', CONJ, M, N, &alpha, nullptr, LDA,
                         nullptr, incx, &beta, nullptr, incy );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+1 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, TRANS, M, N, LDA);
    std::vector<T> x = testinghelpers::get_random_vector<T>( 1, 3, N, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( 1, 3, M, incy );

    // Create a copy of c so that we can check reference results.
    std::vector<T> y_ref(y);
    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    gemv<T>( STORAGE, 'p', CONJ, M, N, &alpha, a.data(), LDA,
                         x.data(), incx, &beta, y.data(), incy );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( "y", N, y.data(), y_ref.data(), incy);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, INFO_OFFSET+1 );
#endif
}

TYPED_TEST(gemv_IIT_ERS, m_lt_zero)
{
    using T = TypeParam;
    gtint_t invalid_m = -1;
    gtint_t incx = 3;
    gtint_t incy = 3;

    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    gemv<T>( STORAGE, TRANS, CONJ, invalid_m, N, nullptr, nullptr, LDA,
                         nullptr, incx, nullptr, nullptr, incy );
#else
    gemv<T>( STORAGE, TRANS, CONJ, invalid_m, N, &alpha, nullptr, LDA,
                         nullptr, incx, &beta, nullptr, incy );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 2 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, TRANS, M, N, LDA);
    std::vector<T> x = testinghelpers::get_random_vector<T>( 1, 3, N, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( 1, 3, M, incy );

    // Create a copy of c so that we can check reference results.
    std::vector<T> y_ref(y);
    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    gemv<T>( STORAGE, TRANS, CONJ, invalid_m, N, &alpha, a.data(), LDA,
                         x.data(), incx, &beta, y.data(), incy );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( "y", N, y.data(), y_ref.data(), incy);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 2 );
#endif
}

TYPED_TEST(gemv_IIT_ERS, n_lt_zero)
{
    using T = TypeParam;
    gtint_t invalid_n = -1;
    gtint_t incx = 3;
    gtint_t incy = 3;

    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    gemv<T>( STORAGE, TRANS, CONJ, M, invalid_n, nullptr, nullptr, LDA,
                         nullptr, incx, nullptr, nullptr, incy );
#else
    gemv<T>( STORAGE, TRANS, CONJ, M, invalid_n, &alpha, nullptr, LDA,
                         nullptr, incx, &beta, nullptr, incy );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 3 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, TRANS, M, N, LDA);
    std::vector<T> x = testinghelpers::get_random_vector<T>( 1, 3, N, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( 1, 3, M, incy );

    // Create a copy of y so that we can check reference results.
    std::vector<T> y_ref(y);

    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    gemv<T>( STORAGE, TRANS, CONJ, M, invalid_n, &alpha, a.data(), LDA,
                         x.data(), incx, &beta, y.data(), incy );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( "y", N, y.data(), y_ref.data(), incy);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 3 );
#endif
}

TYPED_TEST(gemv_IIT_ERS, invalid_lda)
{
    using T = TypeParam;
    gtint_t incx = 3;
    gtint_t incy = 3;

    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    gemv<T>( STORAGE, TRANS, CONJ, M, N, nullptr, nullptr, LDA - 1,
                         nullptr, incx, nullptr, nullptr, incy );
#else
    gemv<T>( STORAGE, TRANS, CONJ, M, N, &alpha, nullptr, LDA - 1,
                         nullptr, incx, &beta, nullptr, incy );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 6 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, TRANS, M, N, LDA);
    std::vector<T> x = testinghelpers::get_random_vector<T>( 1, 3, N, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( 1, 3, M, incy );

    // Create a copy of y so that we can check reference results.
    std::vector<T> y_ref(y);

    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    gemv<T>( STORAGE, TRANS, CONJ, M, N, &alpha, a.data(), LDA - 1,
                         x.data(), incx, &beta, y.data(), incy );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( "y", N, y.data(), y_ref.data(), incy);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 6 );
#endif
}

TYPED_TEST(gemv_IIT_ERS, incx_eq_zero)
{
    using T = TypeParam;
    gtint_t incx = 3;
    gtint_t incy = 3;

    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    gemv<T>( STORAGE, TRANS, CONJ, M, N, nullptr, nullptr, LDA,
                         nullptr, 0, nullptr, nullptr, incy );
#else
    gemv<T>( STORAGE, TRANS, CONJ, M, N, &alpha, nullptr, LDA,
                         nullptr, 0, &beta, nullptr, incy );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 8 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, TRANS, M, N, LDA);
    std::vector<T> x = testinghelpers::get_random_vector<T>( 1, 3, N, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( 1, 3, M, incy );

    // Create a copy of y so that we can check reference results.
    std::vector<T> y_ref(y);

    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    gemv<T>( STORAGE, TRANS, CONJ, M, N, &alpha, a.data(), LDA,
                         x.data(), 0, &beta, y.data(), incy );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( "y", N, y.data(), y_ref.data(), incy);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 8 );
#endif
}

TYPED_TEST(gemv_IIT_ERS, incy_eq_zero)
{
    using T = TypeParam;
    gtint_t incx = 3;
    gtint_t incy = 3;

    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    gemv<T>( STORAGE, TRANS, CONJ, M, N, nullptr, nullptr, LDA,
                         nullptr, incx, nullptr, nullptr, 0 );
#else
    gemv<T>( STORAGE, TRANS, CONJ, M, N, &alpha, nullptr, LDA,
                         nullptr, incx, &beta, nullptr, 0 );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 11 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, TRANS, M, N, LDA);
    std::vector<T> x = testinghelpers::get_random_vector<T>( 1, 3, N, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( 1, 3, M, incy );

    // Create a copy of y so that we can check reference results.
    std::vector<T> y_ref(y);

    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    gemv<T>( STORAGE, TRANS, CONJ, M, N, &alpha, a.data(), LDA,
                         x.data(), incx, &beta, y.data(), 0 );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( "y", N, y.data(), y_ref.data(), incy);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 11 );
#endif
}

/*
    BLAS Early Return Scenarios(ERS):

    GEMV is expected to return early in the following cases:
    1. m || n = 0
    2. alpha = 0 && beta = 1
*/

// m = 0
TYPED_TEST(gemv_IIT_ERS, m_eq_zero)
{
    using T = TypeParam;
    gtint_t invalid_m = 0;
    gtint_t incx = 2;
    gtint_t incy = 3;

    T alpha = T{1.3};
    T beta = T{0.7};

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    gemv<T>( STORAGE, TRANS, CONJ, invalid_m, N, nullptr, nullptr, LDA,
                         nullptr, incx, nullptr, nullptr, incy );
#else
    gemv<T>( STORAGE, TRANS, CONJ, invalid_m, N, &alpha, nullptr, LDA,
                         nullptr, incx, &beta, nullptr, incy );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, TRANS, M, N, LDA);
    std::vector<T> x = testinghelpers::get_random_vector<T>( 1, 3, N, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( 1, 3, M, incy );

    // Create a copy of c so that we can check reference results.
    std::vector<T> y_ref(y);
    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    gemv<T>( STORAGE, TRANS, CONJ, invalid_m, N, &alpha, a.data(), LDA,
                         x.data(), incx, &beta, y.data(), incy );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( "y", N, y.data(), y_ref.data(), incy);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// n = 0
TYPED_TEST(gemv_IIT_ERS, n_eq_zero)
{
    using T = TypeParam;
    gtint_t invalid_n = 0;
    gtint_t incx = 1;
    gtint_t incy = 1;

    T alpha = T{1.3};
    T beta = T{0.7};

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    gemv<T>( STORAGE, TRANS, CONJ, M, invalid_n, nullptr, nullptr, LDA,
                         nullptr, incx, nullptr, nullptr, incy );
#else
    gemv<T>( STORAGE, TRANS, CONJ, M, invalid_n, &alpha, nullptr, LDA,
                         nullptr, incx, &beta, nullptr, incy );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, TRANS, M, N, LDA);
    std::vector<T> x = testinghelpers::get_random_vector<T>( 1, 3, N, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( 1, 3, M, incy );

    // Create a copy of c so that we can check reference results.
    std::vector<T> y_ref(y);
    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    gemv<T>( STORAGE, TRANS, CONJ, M, invalid_n, &alpha, a.data(), LDA,
                         x.data(), incx, &beta, y.data(), incy );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( "y", N, y.data(), y_ref.data(), incy);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// m = 0, with unit alpha
TYPED_TEST(gemv_IIT_ERS, m_eq_zero_Unitbeta)
{
    using T = TypeParam;
    gtint_t invalid_m = 0;
    gtint_t incx = 2;
    gtint_t incy = 3;

    T alpha, beta;
    testinghelpers::initzero<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
#if defined(TEST_BLAS_LIKE)
    gemv<T>( STORAGE, TRANS, CONJ, invalid_m, N, &alpha, nullptr, LDA,
                         nullptr, incx, nullptr, nullptr, incy );
#else
    gemv<T>( STORAGE, TRANS, CONJ, invalid_m, N, &alpha, nullptr, LDA,
                         nullptr, incx, &beta, nullptr, incy );
#endif
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, TRANS, M, N, LDA);
    std::vector<T> x = testinghelpers::get_random_vector<T>( 1, 3, N, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( 1, 3, M, incy );

    // Create a copy of c so that we can check reference results.
    std::vector<T> y_ref(y);
    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    gemv<T>( STORAGE, TRANS, CONJ, invalid_m, N, &alpha, a.data(), LDA,
                         x.data(), incx, &beta, y.data(), incy );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( "y", N, y.data(), y_ref.data(), incy);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// n = 0, with unit alpha and beta
TYPED_TEST(gemv_IIT_ERS, n_eq_zero_UnitAlphaBeta)
{
    using T = TypeParam;
    gtint_t invalid_n = 0;
    gtint_t incx = 1;
    gtint_t incy = 1;

    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    gemv<T>( STORAGE, TRANS, CONJ, M, invalid_n, &alpha, nullptr, LDA,
                         nullptr, incx, &beta, nullptr, incy );
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, TRANS, M, N, LDA);
    std::vector<T> x = testinghelpers::get_random_vector<T>( 1, 3, N, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( 1, 3, M, incy );

    // Create a copy of c so that we can check reference results.
    std::vector<T> y_ref(y);
    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    gemv<T>( STORAGE, TRANS, CONJ, M, invalid_n, &alpha, a.data(), LDA,
                         x.data(), incx, &beta, y.data(), incy );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( "y", N, y.data(), y_ref.data(), incy);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// zero alpha and unit beta
TYPED_TEST(gemv_IIT_ERS, ZeroAlpha_UnitBeta)
{
    using T = TypeParam;
    gtint_t incx = 1;
    gtint_t incy = 1;

    T alpha, beta;
    testinghelpers::initzero<T>( alpha );
    testinghelpers::initone<T>( beta );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    gemv<T>( STORAGE, TRANS, CONJ, M, N, &alpha, nullptr, LDA,
                         nullptr, incx, &beta, nullptr, incy );
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, TRANS, M, N, LDA);
    std::vector<T> x = testinghelpers::get_random_vector<T>( 1, 3, N, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( 1, 3, M, incy );

    // Create a copy of c so that we can check reference results.
    std::vector<T> y_ref(y);
    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    gemv<T>( STORAGE, TRANS, CONJ, M, N, &alpha, a.data(), LDA,
                         x.data(), incx, &beta, y.data(), incy );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( "y", N, y.data(), y_ref.data(), incy);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// zero alpha and zero beta - set y to zero
TYPED_TEST(gemv_IIT_ERS, ZeroAlpha_ZeroBeta)
{
    using T = TypeParam;
    gtint_t incx = 3;
    gtint_t incy = 3;

    T alpha, beta;
    testinghelpers::initzero<T>( alpha );
    testinghelpers::initzero<T>( beta );

    std::vector<T> y = testinghelpers::get_random_vector<T>( 0, 1, N, incy );
    std::vector<T> y2(y);
    // Create a zero vector, since the output for alpha = beta = 0 should be a
    // zero vector.
    std::vector<T> zero_vec = testinghelpers::get_random_vector<T>( 0, 0, N, incy );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    gemv<T>( STORAGE, TRANS, CONJ, M, N, &alpha, nullptr, LDA,
                         nullptr, incx, &beta, y2.data(), incy );
    computediff<T>( "y", N, y2.data(), zero_vec.data(), incy);
#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, TRANS, M, N, LDA);
    std::vector<T> x = testinghelpers::get_random_vector<T>( 0, 1, M, incx );

    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    gemv<T>( STORAGE, TRANS, CONJ, M, N, &alpha, a.data(), LDA,
                         x.data(), incx, &beta, y.data(), incy );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( "y", N, y.data(), zero_vec.data(), incy);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// zero alpha and non-zero/non-unit beta - scale y only
TYPED_TEST(gemv_IIT_ERS, ZeroAlpha_OtherBeta)
{
    using T = TypeParam;
    gtint_t incx = 3;
    gtint_t incy = 3;

    T alpha, beta;
    testinghelpers::initzero<T>( alpha );
    beta = T{2.0};
    double thresh = testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, STORAGE, TRANS, M, N, LDA);
    std::vector<T> x = testinghelpers::get_random_vector<T>( 0, 1, M, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( 0, 1, N, incy );
    std::vector<T> y_ref(y);
    std::vector<T> y2(y);

    testinghelpers::ref_gemv<T>( STORAGE, TRANS, CONJ, M, N, alpha, a.data(), LDA,
                         x.data(), incx, beta, y_ref.data(), incy );

    // Test with nullptr for all suitable arguments that shouldn't be accessed.
    gemv<T>( STORAGE, TRANS, CONJ, M, N, &alpha, nullptr, LDA,
                         nullptr, incx, &beta, y2.data(), incy );

    computediff<T>( "y", N, y2.data(), y_ref.data(), incy, thresh);

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif

    // Test with all arguments correct except for the value we are choosing to test.

    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    gemv<T>( STORAGE, TRANS, CONJ, M, N, &alpha, a.data(), LDA,
                         x.data(), incx, &beta, y.data(), incy );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( "y", N, y.data(), y_ref.data(), incy, thresh);

#ifdef CAN_TEST_INFO_VALUE
    info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

#endif
