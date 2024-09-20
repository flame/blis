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
#include "level3/gemm/test_gemm.h"

class dgemmUOT :
        public ::testing::TestWithParam<std::tuple<char,    // storage format
                                                   char,    // transa
                                                   char,    // transb
                                                   gtint_t, // over_under
                                                   gtint_t, // input_range
                                                   gtint_t, // m
                                                   gtint_t, // n
                                                   gtint_t, // k
                                                   double,  // alpha
                                                   double,  // beta
                                                   gtint_t, // lda_inc
                                                   gtint_t, // ldb_inc
                                                   gtint_t, // ldc_inc
                                                   gtint_t, // ai
                                                   gtint_t, // aj
                                                   gtint_t, // bi
                                                   gtint_t  // bj
                                                   >> {};

TEST_P( dgemmUOT, API )
{
     using T = double;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // matrix storage format(row major, column major)
    char storage = std::get<0>(GetParam());
    // denotes whether matrix a is n,t
    char transa = std::get<1>(GetParam());
    // denotes whether matrix b is n,t
    char transb = std::get<2>(GetParam());
    // over_under denotes whether overflow or underflow is to be tested
    gtint_t over_under = std::get<3>(GetParam());
    // input_range denotes the range of values that would be used to populate the matrices
    gtint_t input_range = std::get<4>(GetParam());
    // matrix size m
    gtint_t m  = std::get<5>(GetParam());
    // matrix size n
    gtint_t n  = std::get<6>(GetParam());
    // matrix size k
    gtint_t k  = std::get<7>(GetParam());
    // specifies alpha value
    T alpha = std::get<8>(GetParam());
    // specifies beta value
    T beta = std::get<9>(GetParam());
    // lda, ldb, ldc increments.
    // If increments are zero, then the array size matches the matrix size.
    // If increments are nonnegative, the array size is bigger than the matrix size.
    gtint_t lda_inc = std::get<10>(GetParam());
    gtint_t ldb_inc = std::get<11>(GetParam());
    gtint_t ldc_inc = std::get<12>(GetParam());

    // ai, aj, bi, bj are the indices where overflow/underflow values need to be inserted
    gtint_t ai = std::get<13>(GetParam());
    gtint_t aj = std::get<14>(GetParam());
    gtint_t bi = std::get<15>(GetParam());
    gtint_t bj = std::get<16>(GetParam());

    // Check gtestsuite gemm.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    if (m == 0 || n == 0)
        thresh = 0.0;
    else if ((alpha == testinghelpers::ZERO<T>() || k == 0) &&
             (beta == testinghelpers::ZERO<T>() || beta == testinghelpers::ONE<T>()))
        thresh = 0.0;
    else
        thresh = (3*k+1)*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_gemm<T>( storage, transa, transb, over_under, input_range, m, n, k, lda_inc, ldb_inc, ldc_inc, ai, aj, bi, bj, alpha, beta, thresh );

}

/*
    Tests for Overflow

    An Overflow condition occurs when the result of an operation or computation is larger than the
    maximum representable floating point value. For double precision floating points, the largest
    representable number is
            DBL_MAX = 1.7976931348623158e+308

    This test populates matrices with values close to DBL_MAX so that the subsequent operations lead
    to values larger than DBL_MAX and hence causes a floating point overflow.

    The argument over_under is used to indicate whether the test is an overflow or an underflow test.
    over_under = 0 indicates an overflow test

    The argument input_range is used to choose the range of values used to populate input matrices
    input_range = -1 for values < DBL_MAX
    input_range = 0 for values close to DBL_MAX
    input_range = 1 for values > DBL_MAX
*/

/* Overflow test for values much less than DBL_MAX */
INSTANTIATE_TEST_SUITE_P(
        overflow_within_limit,
        dgemmUOT,
        ::testing::Combine(
            // No condition based on storage scheme of matrices
            ::testing::Values('c'),                                       // storage format
            // No conditions based on trans of matrices
            ::testing::Values('n', 't'),                                  // transa
            ::testing::Values('n', 't'),                                  // transb

            ::testing::Values(0),                                         // over_under = 0 for overflow
            ::testing::Values(-1),                                        // input_range = -1 to test values less than DBL_MAX
            ::testing::Values(120, 256, 512),                             // m

            ::testing::Values(144, 237, 680),                             // n

            ::testing::Values(128, 557, 680),                             // k
            // No condition based on alpha
            ::testing::Values( -1.0),                                     // alpha
            // No condition based on beta
            ::testing::Values(-1.0),                                      // beta
            ::testing::Values(3),                                         // increment to the leading dim of a
            ::testing::Values(3),                                         // increment to the leading dim of b
            ::testing::Values(3),                                         // increment to the leading dim of c

            ::testing::Values(100),                                       // ai
            ::testing::Values(120),                                       // aj
            ::testing::Values(140),                                       // bi
            ::testing::Values(110)                                        // bj
        ),
        ::gemmOUTPrint<double>()
    );

/* Overflow test for values close to DBL_MAX */
INSTANTIATE_TEST_SUITE_P(
        overflow_close_to_limit,
        dgemmUOT,
        ::testing::Combine(
            // No condition based on storage scheme of matrices
            ::testing::Values('c'),                                       // storage format
            // No conditions based on trans of matrices
            ::testing::Values('n', 't'),                                  // transa
            ::testing::Values('n', 't'),                                  // transb

            ::testing::Values(0),                                         // over_under = 0 for overflow
            ::testing::Values(0),                                         // input_range = 0 to test values close to DBL_MAX
            ::testing::Values(120, 256, 512),                             // m

            ::testing::Values(144, 237, 680),                             // n

            ::testing::Values(128, 557, 680),                             // k
            // No condition based on alpha
            ::testing::Values( -1.0),                                     // alpha
            // No condition based on beta
            ::testing::Values(-1.0),                                      // beta
            ::testing::Values(0),                                         // increment to the leading dim of a
            ::testing::Values(0),                                         // increment to the leading dim of b
            ::testing::Values(0),                                         // increment to the leading dim of c

            ::testing::Values(110),                                       // ai
            ::testing::Values(130),                                       // aj
            ::testing::Values(140),                                       // bi
            ::testing::Values(120)                                        // bj
        ),
        ::gemmOUTPrint<double>()
    );


/* Overflow test for values close to DBL_MAX and aplha = 0*/
INSTANTIATE_TEST_SUITE_P(
        overflow_close_to_limit_alpha0,
        dgemmUOT,
        ::testing::Combine(
            // No condition based on storage scheme of matrices
            ::testing::Values('c'),                                       // storage format
            // No conditions based on trans of matrices
            ::testing::Values('n', 't'),                                  // transa
            ::testing::Values('n', 't'),                                  // transb

            ::testing::Values(0),                                         // over_under = 0 for overflow
            ::testing::Values(0),                                         // input_range = 0 to test values close to DBL_MAX
            ::testing::Values(120, 256, 512),                             // m

            ::testing::Values(144, 237, 680),                             // n

            ::testing::Values(128, 557, 680),                             // k
            // No condition based on alpha
            ::testing::Values(0),                                         // alpha
            // No condition based on beta
            ::testing::Values(-1.0),                                      // beta
            ::testing::Values(5),                                         // increment to the leading dim of a
            ::testing::Values(5),                                         // increment to the leading dim of b
            ::testing::Values(5),                                         // increment to the leading dim of c

            ::testing::Values(108),                                       // ai
            ::testing::Values(122),                                       // aj
            ::testing::Values(145),                                       // bi
            ::testing::Values(108)                                        // bj
        ),
        ::gemmOUTPrint<double>()
    );

/* Overflow test for values larger than DBL_MAX */
INSTANTIATE_TEST_SUITE_P(
        overflow_beyond_limit,
        dgemmUOT,
        ::testing::Combine(
            // No condition based on storage scheme of matrices
            ::testing::Values('c'),                                       // storage format
            // No conditions based on trans of matrices
            ::testing::Values('n', 't'),                                  // transa
            ::testing::Values('n', 't'),                                  // transb

            ::testing::Values(0),                                         // over_under = 0 for overflow
            ::testing::Values(1),                                         // input_range = 1 to test values larger than DBL_MAX
            ::testing::Values(120, 256, 512),                             // m

            ::testing::Values(144, 237, 680),                             // n

            ::testing::Values(128, 557, 680),                             // k
            // No condition based on alpha
            ::testing::Values( -1.0),                                     // alpha
            // No condition based on beta
            ::testing::Values(-1.0),                                      // beta
            ::testing::Values(0),                                         // increment to the leading dim of a
            ::testing::Values(0),                                         // increment to the leading dim of b
            ::testing::Values(0),                                         // increment to the leading dim of c

            ::testing::Values(110),                                       // ai
            ::testing::Values(140),                                       // aj
            ::testing::Values(130),                                       // bi
            ::testing::Values(100)                                        // bj
        ),
        ::gemmOUTPrint<double>()
    );


/*
    Tests for Underflow

    An underflow occurs when the result of an operation or a computation is smaller than the
    smallest representable floating point number. For double-precision floating points,
    the smallest representable number is
                DBL_MIN = 2.2250738585072014e-308

    This test populates matrices with values close to DBL_MIN so that the subsequent operations
    lead to values smaller than DBL_MIN and hence results in a floating point underflow.

    The argument over_under is used to indicate whether a test is an overflow or an underflow test.
    over_under=1 indicates an underflow test

    The argument input_range is used to choose the range of values used to populate input matrices
    input_range = -1 for values > DBL_MIN
    input_range = 0 for values close to DBL_MIN
    input_range = 1 for values < DBL_MIN

*/

/* Underflow test for values larger than DBL_MIN */
INSTANTIATE_TEST_SUITE_P(
        underflow_within_limit,
        dgemmUOT,
        ::testing::Combine(
            // No condition based on storage scheme of matrices
            ::testing::Values('c'),                                       // storage format
            // No conditions based on trans of matrices
            ::testing::Values('n', 't'),                                  // transa
            ::testing::Values('n', 't'),                                  // transb

            ::testing::Values(1),                                         // over_under = 1 for underflow
            ::testing::Values(-1),                                        // input_range = -1 to test values larger than DBL_MIN
            ::testing::Values(120, 256, 512),                             // m

            ::testing::Values(144, 237, 680),                             // n

            ::testing::Values(128, 557, 680),                             // k
            // No condition based on alpha
            ::testing::Values( -1.0),                                     // alpha
            // No condition based on beta
            ::testing::Values(-1.0),                                      // beta
            ::testing::Values(3),                                         // increment to the leading dim of a
            ::testing::Values(3),                                         // increment to the leading dim of b
            ::testing::Values(3),                                         // increment to the leading dim of c

            ::testing::Values(100),                                       // ai
            ::testing::Values(120),                                       // aj
            ::testing::Values(140),                                       // bi
            ::testing::Values(110)                                        // bj
        ),
        ::gemmOUTPrint<double>()
    );

/* Underflow test for values close to DBL_MIN */
INSTANTIATE_TEST_SUITE_P(
        underflow_close_to_limit,
        dgemmUOT,
        ::testing::Combine(
            // No condition based on storage scheme of matrices
            ::testing::Values('c'),                                       // storage format
            // No conditions based on trans of matrices
            ::testing::Values('n', 't'),                                  // transa
            ::testing::Values('n', 't'),                                  // transb

            ::testing::Values(1),                                         // over_under = 1 for underflow
            ::testing::Values(0),                                         // input_range = 0 to test values close to DBL_MIN
            ::testing::Values(120, 256, 512),                             // m

            ::testing::Values(144, 237, 680),                             // n

            ::testing::Values(128, 557, 680),                             // k
            // No condition based on alpha
            ::testing::Values( -1.0),                                     // alpha
            // No condition based on beta
            ::testing::Values(-1.0),                                      // beta
            ::testing::Values(5),                                         // increment to the leading dim of a
            ::testing::Values(5),                                         // increment to the leading dim of b
            ::testing::Values(5),                                         // increment to the leading dim of c

            ::testing::Values(101),                                       // ai
            ::testing::Values(118),                                       // aj
            ::testing::Values(132),                                       // bi
            ::testing::Values(110)                                        // bj
        ),
        ::gemmOUTPrint<double>()
    );

/* Underflow test for values close to DBL_MIN and alpha = 0 */
INSTANTIATE_TEST_SUITE_P(
        underflow_close_to_limit_alpha0,
        dgemmUOT,
        ::testing::Combine(
            // No condition based on storage scheme of matrices
            ::testing::Values('c'),                                       // storage format
            // No conditions based on trans of matrices
            ::testing::Values('n', 't'),                                  // transa
            ::testing::Values('n', 't'),                                  // transb

            ::testing::Values(1),                                         // over_under = 1 for underflow
            ::testing::Values(0),                                         // input_range = 0 to test values close to DBL_MIN
            ::testing::Values(120, 256, 512),                             // m

            ::testing::Values(144, 237, 680),                             // n

            ::testing::Values(128, 557, 680),                             // k
            // No condition based on alpha
            ::testing::Values(0),                                         // alpha
            // No condition based on beta
            ::testing::Values(-1.0),                                      // beta
            ::testing::Values(0),                                         // increment to the leading dim of a
            ::testing::Values(0),                                         // increment to the leading dim of b
            ::testing::Values(0),                                         // increment to the leading dim of c

            ::testing::Values(117),                                       // ai
            ::testing::Values(122),                                       // aj
            ::testing::Values(88),                                        // bi
            ::testing::Values(42)                                         // bj
        ),
        ::gemmOUTPrint<double>()
    );



/* Underflow test for values smaller than DBL_MIN */
INSTANTIATE_TEST_SUITE_P(
        underflow_beyond_limit,
        dgemmUOT,
        ::testing::Combine(
            // No condition based on storage scheme of matrices
            ::testing::Values('c'),                                       // storage format
            // No conditions based on trans of matrices
            ::testing::Values('n', 't'),                                  // transa
            ::testing::Values('n', 't'),                                  // transb

            ::testing::Values(1),                                         // over_under = 1 for underflow
            ::testing::Values(1),                                         // input_range = 1 to test values smaller than DBL_MIN
            ::testing::Values(120, 256, 512),                             // m

            ::testing::Values(144, 237, 680),                             // n

            ::testing::Values(128, 557, 680),                             // k
            // No condition based on alpha
            ::testing::Values(-1.0),                                      // alpha
            // No condition based on beta
            ::testing::Values(-1.0),                                      // beta
            ::testing::Values(3),                                         // increment to the leading dim of a
            ::testing::Values(3),                                         // increment to the leading dim of b
            ::testing::Values(3),                                         // increment to the leading dim of c

            ::testing::Values(44),                                        // ai
            ::testing::Values(135),                                       // aj
            ::testing::Values(100),                                       // bi
            ::testing::Values(105)                                        // bj
        ),
        ::gemmOUTPrint<double>()
    );
