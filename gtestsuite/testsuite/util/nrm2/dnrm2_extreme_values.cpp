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

class dnrm2_EVT :
        public ::testing::TestWithParam<std::tuple<gtint_t, gtint_t, gtint_t, double, gtint_t, double>> {};

TEST_P( dnrm2_EVT, EVT )
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

// Prints the test case combination
class dnrm2_TestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<gtint_t, gtint_t, gtint_t, double, gtint_t, double>> str) const {
        // vector length:
        gtint_t n = std::get<0>(str.param);
        // stride size for x:
        gtint_t incx = std::get<1>(str.param);
        // index with extreme value iexval.
        gtint_t i = std::get<2>(str.param);
        double iexval = std::get<3>(str.param);
        // index with extreme value jexval.
        gtint_t j = std::get<4>(str.param);
        double jexval = std::get<5>(str.param);
#ifdef TEST_BLAS
        std::string str_name = "dnrm2_";
#elif TEST_CBLAS
        std::string str_name = "cblas_dnrm2";
#else  //#elif TEST_BLIS_TYPED
        std::string str_name = "bli_dnormfv";
#endif
        str_name    = str_name + "_" + std::to_string(n);
        std::string incx_str = ( incx > 0) ? std::to_string(incx) : "m" + std::to_string(std::abs(incx));
        str_name    = str_name + "_" + incx_str;
        str_name    = str_name + "_i" + std::to_string(i);
        std::string iexval_str = testinghelpers::get_value_string(iexval);
        str_name    = str_name + "_" + iexval_str;
        str_name    = str_name + "_j" + std::to_string(j);
        std::string jexval_str = testinghelpers::get_value_string(jexval);
        str_name    = str_name + "_" + jexval_str;
        return str_name;
    }
};

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
        dnrm2_EVT,
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
        ::dnrm2_TestPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        vector_F8,
        dnrm2_EVT,
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
        ::dnrm2_TestPrint()
    );

// To test the second for-loop (F4), we use n = 12
// and ensure that the extreme values are on or after index 8.
INSTANTIATE_TEST_SUITE_P(
        vector_F4,
        dnrm2_EVT,
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
        ::dnrm2_TestPrint()
    );

// Now let's check the combination of a vectorized path and
// the scalar path, by putting an extreme value in each
// to check that the checks are integrated correctly.
INSTANTIATE_TEST_SUITE_P(
        vector_scalar,
        dnrm2_EVT,
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
        ::dnrm2_TestPrint()
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
        dnrm2_EVT,
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
        ::dnrm2_TestPrint()
    );

// Instantiator if AOCL_DYNAMIC is enabled
/*
  The instantiator here checks for correctness of
  the compute with sizes large enough to bypass
  the thread setting logic with AOCL_DYNAMIC enabled
*/
INSTANTIATE_TEST_SUITE_P(
        EVT_MT_AOCL_DYNAMIC,
        dnrm2_EVT,
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
        ::dnrm2_TestPrint()
    );
