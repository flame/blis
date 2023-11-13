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
#include "test_nrm2.h"

class dznrm2Test :
        public ::testing::TestWithParam<std::tuple<gtint_t, gtint_t>> {};

TEST_P( dznrm2Test, RandomData )
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

    // Set the threshold for the errors:
    double thresh = 3*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_nrm2<T>(n, incx, thresh);
}

// Prints the test case combination
class dznrm2TestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<gtint_t, gtint_t>> str) const {
        gtint_t n     = std::get<0>(str.param);
        gtint_t incx  = std::get<1>(str.param);
#ifdef TEST_BLAS
        std::string str_name = "dznrm2_";
#elif TEST_CBLAS
        std::string str_name = "cblas_dznrm2";
#else  //#elif TEST_BLIS_TYPED
        std::string str_name = "bli_znormfv";
#endif
        str_name    = str_name + "_" + std::to_string(n);
        std::string incx_str = ( incx > 0) ? std::to_string(incx) : "m" + std::to_string(std::abs(incx));
        str_name    = str_name + "_" + incx_str;
        return str_name;
    }
};

/**
 * dznrm2 implementation is composed by two parts:
 * - vectorized path for n>2
 *      - for-loop for multiples of 4 (F4)
 *      - for-loop for multiples of 2 (F2)
 * - scalar path for n<=2 (S)
*/
INSTANTIATE_TEST_SUITE_P(
        AT_1T,
        dznrm2Test,
        ::testing::Combine(
            // m size of vector
            ::testing::Values(gtint_t(1),  // trivial case n=1
                              gtint_t(2),  // 1*2 - will only go through F2
                              gtint_t(4),  // 1*4 - will only go through F4
                              gtint_t(12), // 3*4 - will go through F4
                              gtint_t(17), // 4*4 + 1 - will go through F4 & S
                              gtint_t(22), // 5*4 + 2 - will go through F4 & F2
                              gtint_t(35), // 8*4 + 2 + 1 - will go through F4 & F2 & S
                              gtint_t(78), // a few bigger numbers
                              gtint_t(112),
                              gtint_t(187),
                              gtint_t(213)
            ),
            // stride size for x
            ::testing::Values(gtint_t(1), gtint_t(3)
#ifndef TEST_BLIS_TYPED
            , gtint_t(-1), gtint_t(-7)
#endif
        )
        ),
        ::dznrm2TestPrint()
    );

// Multithreading unit tester
/*
    The following instantiator has data points that would suffice
    the unit testing with 64 threads.

    Sizes 128 and 129 ensure that each thread gets a minimum
    size of 2, with some sizes inducing fringe cases.

    Sizes 256, 257 and 259 ensure that each thread gets a minimum
    size of 4, with some sizes inducing fringe cases.

    Sizes from 384 to 389 ensure that each thread gets a minimum
    size of 6( 4-block loop + 2-block loop), with some sizes inducing
    fringe cases.

    Non-unit strides are also tested, since they might get packed.
*/
INSTANTIATE_TEST_SUITE_P(
        AT_MT_Unit_Tester,
        dznrm2Test,
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
            ::testing::Values(gtint_t(1), gtint_t(3)
#ifndef TEST_BLIS_TYPED
            , gtint_t(-1), gtint_t(-7)
#endif
        )
        ),
        ::dznrm2TestPrint()
    );

// Instantiator if AOCL_DYNAMIC is enabled
/*
  The instantiator here checks for correctness of
  the compute with sizes large enough to bypass
  the thread setting logic with AOCL_DYNAMIC enabled
*/
INSTANTIATE_TEST_SUITE_P(
        AT_MT_AOCL_DYNAMIC,
        dznrm2Test,
        ::testing::Combine(
            // m size of vector
            ::testing::Values(gtint_t(1530000),
                              gtint_t(1530001)
            ),
            // stride size for x
            ::testing::Values(gtint_t(1), gtint_t(3)
#ifndef TEST_BLIS_TYPED
            , gtint_t(-1), gtint_t(-7)
#endif
        )
        ),
        ::dznrm2TestPrint()
    );
