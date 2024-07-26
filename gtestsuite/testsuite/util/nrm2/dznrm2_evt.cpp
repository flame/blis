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
#include "test_nrm2.h"

class dznrm2EVT :
        public ::testing::TestWithParam<std::tuple<gtint_t, gtint_t, gtint_t, dcomplex, gtint_t, dcomplex>>{};

TEST_P( dznrm2EVT, API )
{
    using T = dcomplex;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // vector length:
    gtint_t n = std::get<0>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<1>(GetParam());
    // index with extreme value iexval.
    gtint_t i = std::get<2>(GetParam());
    T iexval = std::get<3>(GetParam());
    // index with extreme value jexval.
    gtint_t j = std::get<4>(GetParam());
    T jexval = std::get<5>(GetParam());

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_nrm2<T>(n, incx, i, iexval, j, jexval);
}

static double NaN = std::numeric_limits<double>::quiet_NaN();
static double Inf = std::numeric_limits<double>::infinity();
/**
 * dznrm2 implementation is composed by two parts:
 * - vectorized path for n>2
 *      - for-loop for multiples of 4 (F4)
 *      - for-loop for multiples of 2 (F2)
 * - scalar path for n<=2 (S)
*/

// Test for scalar path.
// Testing for jexval=(1.0, 2.0), means that we test only one NaN/Inf value.
// for jexval also being an extreme value, we test all combinations
// of having first a NaN and then an Inf and so on.
INSTANTIATE_TEST_SUITE_P(
        scalar,
        dznrm2EVT,
        ::testing::Combine(
            // m size of vector
            ::testing::Values(gtint_t(2)),
            // stride size for x
            ::testing::Values(gtint_t(1)),
            // i : index of x that has value iexval
            ::testing::Values(0),
            // iexval
            ::testing::Values(dcomplex{NaN, 1.0}, dcomplex{Inf, 9.0}, dcomplex{-1.0, -Inf}, dcomplex{2.0, NaN}, dcomplex{NaN, Inf}, dcomplex{Inf, NaN}),
            ::testing::Values(1),
            ::testing::Values(dcomplex{1.0, 2.0}, dcomplex{NaN, 1.0}, dcomplex{Inf, 9.0}, dcomplex{-1.0, -Inf}, dcomplex{2.0, NaN})
        ),
        ::nrm2EVTPrint<dcomplex>()
    );

INSTANTIATE_TEST_SUITE_P(
        vector_F4,
        dznrm2EVT,
        ::testing::Combine(
            // m size of vector
            ::testing::Values(gtint_t(4)),
            // stride size for x
            ::testing::Values(gtint_t(1)),
            // i : index of x that has value iexval
            ::testing::Values(1),
            // iexval
            ::testing::Values(dcomplex{NaN, 1.0}, dcomplex{Inf, 9.0}, dcomplex{-1.0, -Inf}, dcomplex{2.0, NaN}, dcomplex{NaN, Inf}, dcomplex{Inf, NaN}),
            ::testing::Values(3),
            ::testing::Values(dcomplex{1.0, 2.0}, dcomplex{NaN, 1.0}, dcomplex{Inf, 9.0}, dcomplex{-1.0, -Inf}, dcomplex{2.0, NaN})
        ),
        ::nrm2EVTPrint<dcomplex>()
    );

// To test the second for-loop (F2), we use n = 6
// and ensure that the extreme values are on or after index 4.
INSTANTIATE_TEST_SUITE_P(
        vector_F2,
        dznrm2EVT,
        ::testing::Combine(
            // m size of vector
            ::testing::Values(gtint_t(6)),
            // stride size for x
            ::testing::Values(gtint_t(1)),
            // i : index of x that has value iexval
            ::testing::Values(4),
            // iexval
            ::testing::Values(dcomplex{NaN, 1.0}, dcomplex{Inf, 9.0}, dcomplex{-1.0, -Inf}, dcomplex{2.0, NaN}, dcomplex{NaN, Inf}, dcomplex{Inf, NaN}),
            ::testing::Values(5),
            ::testing::Values(dcomplex{1.0, 2.0}, dcomplex{NaN, 1.0}, dcomplex{Inf, 9.0}, dcomplex{-1.0, -Inf}, dcomplex{2.0, NaN})
        ),
        ::nrm2EVTPrint<dcomplex>()
    );

// Now let's check the combination of a vectorized path and
// the scalar path, by putting an extreme value in each
// to check that the checks are integrated correctly.
INSTANTIATE_TEST_SUITE_P(
        vector_scalar,
        dznrm2EVT,
        ::testing::Combine(
            // m size of vector
            ::testing::Values(gtint_t(7)),
            // stride size for x
            ::testing::Values(gtint_t(1)),
            // i : index of x that has value iexval
            ::testing::Values(2),
            // iexval
            ::testing::Values(dcomplex{NaN, 1.0}, dcomplex{Inf, 9.0}, dcomplex{-1.0, -Inf}, dcomplex{2.0, NaN}, dcomplex{NaN, Inf}, dcomplex{Inf, NaN}),
            ::testing::Values(6),
            ::testing::Values(dcomplex{NaN, 1.0}, dcomplex{Inf, 9.0}, dcomplex{-1.0, -Inf}, dcomplex{2.0, NaN})
        ),
        ::nrm2EVTPrint<dcomplex>()
    );

// Mutlthreading Unit Tester
/*
    The following instantiator has data points that would suffice
    the extreme value testing with 64 threads.

    Sizes 128 and 129 ensure that each thread gets size 2, with
    the first thread dealing with fringe case also, if required.

    Sizes 256, 257 and 259 ensure that each thread gets size 4, with
    the first two threads dealing wtih extra AVX and SSE cases also,
    if required.

    Sizes from 384 to 389 ensure that each thread gets size 6, with
    the first few threads dealing with extra AVX and SSE cases if needed.

    NOTE : Extreme values are induced at indices that are valid
           for all the listed sizes in the instantiator.

    Non-unit strides are also tested, since they might get packed
*/
INSTANTIATE_TEST_SUITE_P(
        EVT_MT_Unit_Tester,
        dznrm2EVT,
        ::testing::Combine(
            // m size of vector
            ::testing::Values(gtint_t(128),
                              gtint_t(129),
                              gtint_t(256),
                              gtint_t(257),
                              gtint_t(259),
                              gtint_t(384),
                              gtint_t(385),
                              gtint_t(386),
                              gtint_t(387),
                              gtint_t(388),
                              gtint_t(389)
            ),
            // stride size for x
            ::testing::Values(gtint_t(1), gtint_t(3)),
            // i : index of x that has value iexval
            ::testing::Values(2, 17, 65, 110),
            // iexval
            ::testing::Values(dcomplex{NaN, 1.0}, dcomplex{Inf, 9.0}, dcomplex{-1.0, -Inf}, dcomplex{2.0, NaN}, dcomplex{NaN, Inf}, dcomplex{Inf, NaN}),
            ::testing::Values(6, 25, 64, 127),
            ::testing::Values(dcomplex{NaN, 1.0}, dcomplex{Inf, 9.0}, dcomplex{-1.0, -Inf}, dcomplex{2.0, NaN})
        ),
        ::nrm2EVTPrint<dcomplex>()
    );

// Instantiator if AOCL_DYNAMIC is enabled
/*
  The instantiator here checks for correctness of
  the compute with sizes large enough to bypass
  the thread setting logic with AOCL_DYNAMIC enabled
*/
INSTANTIATE_TEST_SUITE_P(
        EVT_MT_AOCL_DYNAMIC,
        dznrm2EVT,
        ::testing::Combine(
            // m size of vector
            ::testing::Values(gtint_t(1530000),
                              gtint_t(1530001)
            ),
            // stride size for x
            ::testing::Values(gtint_t(1), gtint_t(5)),
            // i : index of x that has value iexval
            ::testing::Values(800000, 1000000),
            // iexval
            ::testing::Values(dcomplex{NaN, Inf}, dcomplex{-Inf, NaN}, dcomplex{Inf, 0.0}),
            ::testing::Values(1100000, 1500000),
            ::testing::Values(dcomplex{NaN, Inf}, dcomplex{-Inf, NaN}, dcomplex{Inf, 0.0})
        ),
        ::nrm2EVTPrint<dcomplex>()
    );
