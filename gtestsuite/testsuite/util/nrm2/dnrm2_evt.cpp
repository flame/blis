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

class dnrm2EVT :
        public ::testing::TestWithParam<std::tuple<gtint_t, gtint_t, gtint_t, double, gtint_t, double>> {};

TEST_P( dnrm2EVT, API )
{
    using T = double;
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
 * dnrm2 implementation is composed by two parts:
 * - vectorized path for n>4
 *      - for-loop for multiples of 8 (F8)
 *      - for-loop for multiples of 4 (F4)
 * - scalar path for n<=4 (S)
 */

// Test for scalar path.
// Testing for jexval=1.0, means that we test only one NaN/Inf value.
// for jexval also being an extreme value, we test all combinations
// of having first a NaN and then an Inf and so on.
INSTANTIATE_TEST_SUITE_P(
        scalar,
        dnrm2EVT,
        ::testing::Combine(
            // m size of vector
            ::testing::Values(gtint_t(3)),
            // stride size for x
            ::testing::Values(gtint_t(1)),
            // i : index of x that has value iexval
            ::testing::Values(0),
            // iexval
            ::testing::Values(NaN, Inf, -Inf),
            ::testing::Values(2),
            ::testing::Values(1.0, NaN, Inf, -Inf)
        ),
        ::nrm2EVTPrint<double>()
    );

INSTANTIATE_TEST_SUITE_P(
        vector_F8,
        dnrm2EVT,
        ::testing::Combine(
            // m size of vector
            ::testing::Values(gtint_t(8)),
            // stride size for x
            ::testing::Values(gtint_t(1)),
            // i : index of x that has value iexval
            ::testing::Values(3),
            // iexval
            ::testing::Values(NaN, Inf, -Inf),
            ::testing::Values(6),
            ::testing::Values(1.0, NaN, Inf, -Inf)
        ),
        ::nrm2EVTPrint<double>()
    );

// To test the second for-loop (F4), we use n = 12
// and ensure that the extreme values are on or after index 8.
INSTANTIATE_TEST_SUITE_P(
        vector_F4,
        dnrm2EVT,
        ::testing::Combine(
            // m size of vector
            ::testing::Values(gtint_t(12)),
            // stride size for x
            ::testing::Values(gtint_t(1)),
            // i : index of x that has value iexval
            ::testing::Values(9),
            // iexval
            ::testing::Values(NaN, Inf, -Inf),
            ::testing::Values(11),
            ::testing::Values(1.0, NaN, Inf, -Inf)
        ),
        ::nrm2EVTPrint<double>()
    );

// Now let's check the combination of a vectorized path and
// the scalar path, by putting an extreme value in each
// to check that the checks are integrated correctly.
INSTANTIATE_TEST_SUITE_P(
        vector_scalar,
        dnrm2EVT,
        ::testing::Combine(
            // m size of vector
            ::testing::Values(gtint_t(10)),
            // stride size for x
            ::testing::Values(gtint_t(1)),
            // i : index of x that has value iexval
            ::testing::Values(5),
            // iexval
            ::testing::Values(NaN, Inf, -Inf),
            ::testing::Values(8),
            ::testing::Values(1.0, NaN, Inf, -Inf)
        ),
        ::nrm2EVTPrint<double>()
    );

// Multithreading unit tester
/*
    The following instantiator has data points that would suffice
    the unit testing with <= 64 threads.

    Sizes from 256 to 259 ensure that each thread gets a minimum
    size of 4, with some sizes inducing fringe cases.

    Sizes from 512 to 515 ensure that each thread gets a minimum
    size of 8, with some sizes inducing fringe cases.

    Sizes from 768 to 771 ensure that each thread gets a minimum
    size of 12, with some sizes inducing fringe cases.

    NOTE : Extreme values are induced at indices that are valid
           for all the listed sizes in the instantiator.

    Non-unit strides are also tested, since they might get packed.
*/
INSTANTIATE_TEST_SUITE_P(
        EVT_MT_Unit_Tester,
        dnrm2EVT,
        ::testing::Combine(
            // m size of vector
            ::testing::Values(gtint_t(256),
                              gtint_t(257),
                              gtint_t(258),
                              gtint_t(259),
                              gtint_t(512),
                              gtint_t(513),
                              gtint_t(514),
                              gtint_t(515),
                              gtint_t(768),
                              gtint_t(769),
                              gtint_t(770),
                              gtint_t(771)),
            // stride size for x
            ::testing::Values(gtint_t(1), gtint_t(5)),
            // i : index of x that has value iexval
            ::testing::Values(0, 5, 100, 255),
            // iexval
            ::testing::Values(NaN, Inf, -Inf),
            ::testing::Values(4, 17, 125, 201),
            ::testing::Values(1.0, NaN, Inf, -Inf)
        ),
        ::nrm2EVTPrint<double>()
    );

// Instantiator if AOCL_DYNAMIC is enabled
/*
  The instantiator here checks for correctness of
  the compute with sizes large enough to bypass
  the thread setting logic with AOCL_DYNAMIC enabled
*/
INSTANTIATE_TEST_SUITE_P(
        EVT_MT_AOCLDynamic,
        dnrm2EVT,
        ::testing::Combine(
            // m size of vector
            ::testing::Values(gtint_t(2950000),
                              gtint_t(2950001),
                              gtint_t(2950002),
                              gtint_t(2950003)
            ),
            // stride size for x
            ::testing::Values(gtint_t(1), gtint_t(5)),
            // i : index of x that has value iexval
            ::testing::Values(1000000, 2000000),
            // iexval
            ::testing::Values(NaN, Inf),
            ::testing::Values(1500000, 2500000),
            ::testing::Values(-Inf, NaN)
        ),
        ::nrm2EVTPrint<double>()
    );
