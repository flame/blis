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

class snrm2_EVT :
        public ::testing::TestWithParam<std::tuple<gtint_t, gtint_t, gtint_t, float, gtint_t, float>> {};

TEST_P( snrm2_EVT, EVT )
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
class snrm2_TestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<gtint_t, gtint_t, gtint_t, float, gtint_t, float>> str) const {
        // vector length:
        gtint_t n = std::get<0>(str.param);
        // stride size for x:
        gtint_t incx = std::get<1>(str.param);
        // index with extreme value iexval.
        gtint_t i = std::get<2>(str.param);
        float iexval = std::get<3>(str.param);
        // index with extreme value jexval.
        gtint_t j = std::get<4>(str.param);
        float jexval = std::get<5>(str.param);
#ifdef TEST_BLAS
        std::string str_name = "snrm2_";
#elif TEST_CBLAS
        std::string str_name = "cblas_snrm2";
#else  //#elif TEST_BLIS_TYPED
        std::string str_name = "bli_snormfv";
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

static float NaN = std::numeric_limits<float>::quiet_NaN();
static float Inf = std::numeric_limits<float>::infinity();

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

// Test for scalar path.
// Testing for jexval=1.0, means that we test only one NaN/Inf value.
// for jexval also being an extreme value, we test all combinations
// of having first a NaN and then an Inf and so on.
INSTANTIATE_TEST_SUITE_P(
        scalar,
        snrm2_EVT,
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
        ::snrm2_TestPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        vector_F32,
        snrm2_EVT,
        ::testing::Combine(
            // m size of vector
            ::testing::Values(gtint_t(64)),
            // stride size for x
            ::testing::Values(gtint_t(1)),
            // i : index of x that has value iexval                   
            ::testing::Values(13),
            // iexval
            ::testing::Values(NaN, Inf, -Inf),
            ::testing::Values(26),
            ::testing::Values(1.0, NaN, Inf, -Inf)
        ),
        ::snrm2_TestPrint()
    );

// To test the second for-loop (F24), we use n = 88 = 2*32+24
// and ensure that the extreme values are on or after index 64.
INSTANTIATE_TEST_SUITE_P(
        vector_F24,
        snrm2_EVT,
        ::testing::Combine(
            // m size of vector
            ::testing::Values(gtint_t(88)),
            // stride size for x
            ::testing::Values(gtint_t(1)),
            // i : index of x that has value iexval                   
            ::testing::Values(70),
            // iexval
            ::testing::Values(NaN, Inf, -Inf),
            ::testing::Values(80),
            ::testing::Values(1.0, NaN, Inf, -Inf)
        ),
        ::snrm2_TestPrint()
    );

// To test the second for-loop (F16), we use n = 80 = 2*32+16
// and ensure that the extreme values are on or after index 64.
INSTANTIATE_TEST_SUITE_P(
        vector_F16,
        snrm2_EVT,
        ::testing::Combine(
            // m size of vector
            ::testing::Values(gtint_t(80)),
            // stride size for x
            ::testing::Values(gtint_t(1)),
            // i : index of x that has value iexval                   
            ::testing::Values(70),
            // iexval
            ::testing::Values(NaN, Inf, -Inf),
            ::testing::Values(75),
            ::testing::Values(1.0, NaN, Inf, -Inf)
        ),
        ::snrm2_TestPrint()
    );

// Now let's check the combination of a vectorized path and 
// the scalar path, by putting an extreme value in each
// to check that the checks are integrated correctly.
INSTANTIATE_TEST_SUITE_P(
        vector_scalar,
        snrm2_EVT,
        ::testing::Combine(
            // m size of vector
            ::testing::Values(gtint_t(68)),
            // stride size for x
            ::testing::Values(gtint_t(1)),
            // i : index of x that has value iexval                   
            ::testing::Values(5),
            // iexval
            ::testing::Values(NaN, Inf, -Inf),
            ::testing::Values(65),
            ::testing::Values(NaN, Inf, -Inf)
        ),
        ::snrm2_TestPrint()
    );

