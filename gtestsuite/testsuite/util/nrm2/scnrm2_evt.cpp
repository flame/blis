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

class scnrm2EVT :
        public ::testing::TestWithParam<std::tuple<gtint_t, gtint_t, gtint_t, scomplex, gtint_t, scomplex>>{};

TEST_P( scnrm2EVT, API )
{
    using T = scomplex;
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

static float NaN = std::numeric_limits<float>::quiet_NaN();
static float Inf = std::numeric_limits<float>::infinity();
/**
 * scnrm2 implementation is composed by two parts:
 * - vectorized path for n>=64
 *      - for-loop for multiples of 16 (F16)
 *      - for-loop for multiples of 12 (F12)
 *      - for-loop for multiples of 8  (F8)
 * - scalar path for n<64 (S)
*/

// Test for scalar path.
// Testing for jexval=(1.0, 2.0), means that we test only one NaN/Inf value.
// for jexval also being an extreme value, we test all combinations
// of having first a NaN and then an Inf and so on.
INSTANTIATE_TEST_SUITE_P(
        scalar,
        scnrm2EVT,
        ::testing::Combine(
            // m size of vector
            ::testing::Values(gtint_t(2)),
            // stride size for x
            ::testing::Values(gtint_t(1)),
            // i : index of x that has value iexval
            ::testing::Values(0),
            // iexval
            ::testing::Values(scomplex{NaN, 1.0}, scomplex{Inf, 9.0}, scomplex{-1.0, -Inf}, scomplex{2.0, NaN}, scomplex{NaN, Inf}, scomplex{Inf, NaN}),
            ::testing::Values(1),
            ::testing::Values(scomplex{1.0, 2.0}, scomplex{NaN, 1.0}, scomplex{Inf, 9.0}, scomplex{-1.0, -Inf}, scomplex{2.0, NaN})
        ),
        ::nrm2EVTPrint<scomplex>()
    );

INSTANTIATE_TEST_SUITE_P(
        vector_F16,
        scnrm2EVT,
        ::testing::Combine(
            // m size of vector
            ::testing::Values(gtint_t(64)),
            // stride size for x
            ::testing::Values(gtint_t(1)),
            // i : index of x that has value iexval
            ::testing::Values(10),
            // iexval
            ::testing::Values(scomplex{NaN, 1.0}, scomplex{Inf, 9.0}, scomplex{-1.0, -Inf}, scomplex{2.0, NaN}, scomplex{NaN, Inf}, scomplex{Inf, NaN}),
            ::testing::Values(30),
            ::testing::Values(scomplex{1.0, 2.0}, scomplex{NaN, 1.0}, scomplex{Inf, 9.0}, scomplex{-1.0, -Inf}, scomplex{2.0, NaN})
        ),
        ::nrm2EVTPrint<scomplex>()
    );

// To test the second for-loop (F12), we use n = 76 = 4*16+12
// and ensure that the extreme values are on or after index 64.
INSTANTIATE_TEST_SUITE_P(
        vector_F12,
        scnrm2EVT,
        ::testing::Combine(
            // m size of vector
            ::testing::Values(gtint_t(76)),
            // stride size for x
            ::testing::Values(gtint_t(1)),
            // i : index of x that has value iexval
            ::testing::Values(68),
            // iexval
            ::testing::Values(scomplex{NaN, 1.0}, scomplex{Inf, 9.0}, scomplex{-1.0, -Inf}, scomplex{2.0, NaN}, scomplex{NaN, Inf}, scomplex{Inf, NaN}),
            ::testing::Values(70),
            ::testing::Values(scomplex{1.0, 2.0}, scomplex{NaN, 1.0}, scomplex{Inf, 9.0}, scomplex{-1.0, -Inf}, scomplex{2.0, NaN})
        ),
        ::nrm2EVTPrint<scomplex>()
    );

// To test the second for-loop (F8), we use n = 72 = 4*16+8
// and ensure that the extreme values are on or after index 64.
INSTANTIATE_TEST_SUITE_P(
        vector_F8,
        scnrm2EVT,
        ::testing::Combine(
            // m size of vector
            ::testing::Values(gtint_t(72)),
            // stride size for x
            ::testing::Values(gtint_t(1)),
            // i : index of x that has value iexval
            ::testing::Values(66),
            // iexval
            ::testing::Values(scomplex{NaN, 1.0}, scomplex{Inf, 9.0}, scomplex{-1.0, -Inf}, scomplex{2.0, NaN}, scomplex{NaN, Inf}, scomplex{Inf, NaN}),
            ::testing::Values(70),
            ::testing::Values(scomplex{1.0, 2.0}, scomplex{NaN, 1.0}, scomplex{Inf, 9.0}, scomplex{-1.0, -Inf}, scomplex{2.0, NaN})
        ),
        ::nrm2EVTPrint<scomplex>()
    );

// Now let's check the combination of a vectorized path and
// the scalar path, by putting an extreme value in each
// to check that the checks are integrated correctly.
INSTANTIATE_TEST_SUITE_P(
        vector_scalar,
        scnrm2EVT,
        ::testing::Combine(
            // m size of vector
            ::testing::Values(gtint_t(79)),
            // stride size for x
            ::testing::Values(gtint_t(1)),
            // i : index of x that has value iexval
            ::testing::Values(25),
            // iexval
            ::testing::Values(scomplex{NaN, 1.0}, scomplex{Inf, 9.0}, scomplex{-1.0, -Inf}, scomplex{2.0, NaN}, scomplex{NaN, Inf}, scomplex{Inf, NaN}),
            ::testing::Values(68),
            ::testing::Values(scomplex{NaN, 1.0}, scomplex{Inf, 9.0}, scomplex{-1.0, -Inf}, scomplex{2.0, NaN})
        ),
        ::nrm2EVTPrint<scomplex>()
    );

