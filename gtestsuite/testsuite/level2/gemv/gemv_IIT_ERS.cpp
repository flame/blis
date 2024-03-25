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
#include "test_gemv.h"
#include "common/wrong_inputs_helpers.h"
#include "common/testing_helpers.h"
#include "inc/check_error.h"

template <typename T>
class gemv_IIT_ERS_Test : public ::testing::Test {};
typedef ::testing::Types<float, double, scomplex, dcomplex> TypeParam;
TYPED_TEST_SUITE(gemv_IIT_ERS_Test, TypeParam);

using namespace testinghelpers::IIT;

#if defined(TEST_BLAS) || defined(TEST_CBLAS)

/*
    BLAS Early Return Scenarios(ERS):

    GEMV is expected to return early in the following cases:
    1. m || n = 0
*/

// n = 0, with unit alpha
TYPED_TEST(gemv_IIT_ERS_Test, n_eq_zero_Unitalphabeta)
{
    using T = TypeParam;
    gtint_t invalid_n = 0;
    gtint_t incx = 1;
    gtint_t incy = 1;

    // Get correct vector lengths.
    // gtint_t lenx = ( testinghelpers::chknotrans( trnsa ) ) ? n : m ;
    // gtint_t leny = ( testinghelpers::chknotrans( trnsa ) ) ? m : n ;

    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> x = testinghelpers::get_random_vector<T>( 1, 3, M, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( 1, 3, N, incy );

    // Create a copy of c so that we can check reference results.
    std::vector<T> y_ref(y);
    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    gemv<T>( STORAGE, TRANS, CONJ, M, invalid_n, &alpha, nullptr, LDA,
                         x.data(), incx, &beta, y.data(), incy );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( N, y.data(), y_ref.data(), incy);
}

TYPED_TEST(gemv_IIT_ERS_Test, ZeroBeta_Unitalpha)
{
    using T = TypeParam;
    gtint_t incx = 1;
    gtint_t incy = 1;

    // Get correct vector lengths.
    // gtint_t lenx = ( testinghelpers::chknotrans( trnsa ) ) ? n : m ;
    // gtint_t leny = ( testinghelpers::chknotrans( trnsa ) ) ? m : n ;

    T alpha, beta;
    testinghelpers::initzero<T>( alpha );
    testinghelpers::initone<T>( beta );

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> x = testinghelpers::get_random_vector<T>( 1, 3, M, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( 1, 3, N, incy );

    // Create a copy of c so that we can check reference results.
    std::vector<T> y_ref(y);
    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    gemv<T>( STORAGE, TRANS, CONJ, M, N, &alpha, nullptr, LDA,
                         x.data(), incx, &beta, y.data(), incy );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( N, y.data(), y_ref.data(), incy);
}

TYPED_TEST(gemv_IIT_ERS_Test, m_eq_zero_Unitbeta)
{
    using T = TypeParam;
    gtint_t invalid_m = 0;
    gtint_t incx = 2;
    gtint_t incy = 3;

    // Get correct vector lengths.
    // gtint_t lenx = ( testinghelpers::chknotrans( trnsa ) ) ? n : m ;
    // gtint_t leny = ( testinghelpers::chknotrans( trnsa ) ) ? m : n ;

    T alpha, beta;
    testinghelpers::initzero<T>( alpha );
    testinghelpers::initone<T>( beta );

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    // std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, storage, 'n', m, n, LDA );
    std::vector<T> x = testinghelpers::get_random_vector<T>( 1, 3, M, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( 1, 3, N, incy );

    // Create a copy of c so that we can check reference results.
    std::vector<T> y_ref(y);
    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    gemv<T>( STORAGE, TRANS, CONJ, invalid_m, N, &alpha, nullptr, LDA,
                         x.data(), incx, &beta, y.data(), incy );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( N, y.data(), y_ref.data(), incy);
}

TYPED_TEST(gemv_IIT_ERS_Test, m_lt_zero_Unitscalar)
{
    using T = TypeParam;
    gtint_t invalid_m = -1;
    gtint_t incx = 3;
    gtint_t incy = 3;

    // Get correct vector lengths.
    // gtint_t lenx = ( testinghelpers::chknotrans( trnsa ) ) ? n : m ;
    // gtint_t leny = ( testinghelpers::chknotrans( trnsa ) ) ? m : n ;

    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> x = testinghelpers::get_random_vector<T>( 1, 3, M, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( 1, 3, N, incy );


    // Create a copy of c so that we can check reference results.
    std::vector<T> y_ref(y);
    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    gemv<T>( STORAGE, TRANS, CONJ, invalid_m, N, &alpha, nullptr, LDA,
                         x.data(), incx, &beta, y.data(), incy );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( N, y.data(), y_ref.data(), incy);
}

TYPED_TEST(gemv_IIT_ERS_Test, n_lt_zero_Unitscalar)
{
    using T = TypeParam;
    gtint_t invalid_n = -1;
    gtint_t incx = 3;
    gtint_t incy = 3;

    // Get correct vector lengths.
    // gtint_t lenx = ( testinghelpers::chknotrans( trnsa ) ) ? n : m ;
    // gtint_t leny = ( testinghelpers::chknotrans( trnsa ) ) ? m : n ;

    T alpha, beta;
    testinghelpers::initone<T>( alpha );
    testinghelpers::initone<T>( beta );

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> x = testinghelpers::get_random_vector<T>( 1, 3, M, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( 1, 3, N, incy );

    // Create a copy of y so that we can check reference results.
    std::vector<T> y_ref(y);

    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    gemv<T>( STORAGE, TRANS, CONJ, M, invalid_n, &alpha, nullptr, LDA,
                         x.data(), incx, &beta, y.data(), incy );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( N, y.data(), y_ref.data(), incy);
}

TYPED_TEST(gemv_IIT_ERS_Test, Zero_scalar)
{
    using T = TypeParam;
    gtint_t incx = 3;
    gtint_t incy = 3;

    // Get correct vector lengths.
    // gtint_t lenx = ( testinghelpers::chknotrans( trnsa ) ) ? n : m ;
    // gtint_t leny = ( testinghelpers::chknotrans( trnsa ) ) ? m : n ;

    T alpha, beta;
    testinghelpers::initzero<T>( alpha );
    testinghelpers::initzero<T>( beta );

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    // std::vector<T> a = testinghelpers::get_random_matrix<T>( 1, 5, storage, 'n', m, n, LDA );
    std::vector<T> x = testinghelpers::get_random_vector<T>( 0, 1, M, incx );
    std::vector<T> y = testinghelpers::get_random_vector<T>( 0, 1, N, incy );

    // Create a zero vector, since the output for alpha = beta = 0 should be a
    // zero vector.
    std::vector<T> zero_vec = testinghelpers::get_random_vector<T>( 0, 0, N, incy );;

    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    gemv<T>( STORAGE, TRANS, CONJ, M, N, &alpha, nullptr, LDA,
                         x.data(), incx, &beta, y.data(), incy );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( N, y.data(), zero_vec.data(), incy);
}
#endif
