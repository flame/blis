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

class snrm2Generic :
        public ::testing::TestWithParam<std::tuple<gtint_t, gtint_t>> {};

TEST_P( snrm2Generic, API )
{
    using T = float;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // vector length:
    gtint_t n = std::get<0>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<1>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite asumv.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    if (n == 0)
        thresh = 0.0;
    else
        thresh = std::sqrt(n)*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_nrm2<T>( n, incx, thresh );
}

/**
 * Note: snrm2 scalar ONLY implementation is used, but we write the test
 * using values that worked for the vectorized path for the future.
 *
 * scnrm2 implementation is composed by two parts:
 * - vectorized path for n>=64
 *      - for-loop for multiples of 32 (F32)
 *      - for-loop for multiples of 24 (F24)
 *      - for-loop for multiples of 16 (F16)
 * - scalar path for n<64 (S)
*/
INSTANTIATE_TEST_SUITE_P(
        AT,
        snrm2Generic,
        ::testing::Combine(
            // m size of vector
            ::testing::Values(gtint_t(1),  // trivial case n=1
                              gtint_t(35), // will only go through S
                              gtint_t(64), // 2*32 - will only go through F32
                              gtint_t(76), // 2*32 + 12 - will go through F32 & S
                              gtint_t(80), // 2*32 + 16 - will go through F32 & F16
                              gtint_t(85), // 2*32 + 16 + 5 - will go through F32 & F16 & S
                              gtint_t(88), // 2*32 + 24 - will go through F32 & F24
                              gtint_t(91), // 2*32 + 24 + 3 - will go through F32 & F24 & S
                              gtint_t(124), // a few bigger numbers
                              gtint_t(167),
                              gtint_t(259)
            ),
            // stride size for x
            ::testing::Values(gtint_t(1), gtint_t(3)
#ifndef TEST_BLIS_TYPED
            , gtint_t(-1), gtint_t(-5)
#endif
        )                                                                    // stride size for x
        ),
        ::nrm2GenericPrint()
    );
