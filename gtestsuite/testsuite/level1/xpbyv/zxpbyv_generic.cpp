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
#include "test_xpbyv.h"

class zxpbyvGenericTest :
        public ::testing::TestWithParam<std::tuple<char,
                                                   gtint_t,
                                                   gtint_t,
                                                   gtint_t,
                                                   dcomplex,
                                                   char>> {};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(zxpbyvGenericTest);

// Tests using random integers as vector elements.
TEST_P( zxpbyvGenericTest, RandomData )
{
    using T = dcomplex;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // denotes whether x or conj(x) will be added to y:
    char conj_x = std::get<0>(GetParam());
    // vector length:
    gtint_t n = std::get<1>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<2>(GetParam());
    // stride size for y:
    gtint_t incy = std::get<3>(GetParam());
    // beta
    T beta = std::get<4>(GetParam());
    // specifies the datatype for randomgenerators
    char datatype = std::get<5>(GetParam());

    // Set the threshold for the errors:
    double thresh = 2*testinghelpers::getEpsilon<T>();
    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_xpbyv<T>(conj_x, n, incx, incy, beta, thresh, datatype);
}

// Used to generate a test case with a sensible name.
// Beware that we cannot use fp numbers (e.g., 2.3) in the names,
// so we are only printing int(2.3). This should be enough for debugging purposes.
// If this poses an issue, please reach out.
class zxpbyvGenericTestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char,gtint_t,gtint_t,gtint_t,dcomplex,char>> str) const {
        char conj     = std::get<0>(str.param);
        gtint_t n     = std::get<1>(str.param);
        gtint_t incx  = std::get<2>(str.param);
        gtint_t incy  = std::get<3>(str.param);
        dcomplex beta = std::get<4>(str.param);
        char datatype = std::get<5>(str.param);
        std::string str_name = "bli_zxpbyv";
        str_name += "_" + std::to_string(n);
        str_name += "_" + std::string(&conj, 1);
        std::string incx_str = ( incx > 0) ? std::to_string(incx) : "m" + std::to_string(std::abs(incx));
        str_name += "_" + incx_str;
        std::string incy_str = ( incy > 0) ? std::to_string(incy) : "m" + std::to_string(std::abs(incy));
        str_name += "_" + incy_str;
        std::string beta_str = ( beta.real > 0) ? std::to_string(int(beta.real)) : ("m" + std::to_string(int(std::abs(beta.real))));
                  beta_str = beta_str + "pi" + (( beta.imag > 0) ? std::to_string(int(beta.imag)) : ("m" + std::to_string(int(std::abs(beta.imag)))));
        str_name = str_name + "_b" + beta_str;
        str_name = str_name + "_" + datatype;
        return str_name;
    }
};

#ifdef TEST_BLIS_TYPED
// Black box testing for generic and main use of zaxpby.
INSTANTIATE_TEST_SUITE_P(
        Blackbox,
        zxpbyvGenericTest,
        ::testing::Combine(
            ::testing::Values('n', 'c'),                                     // n: use x, c: use conj(x)
            ::testing::Range(gtint_t(10), gtint_t(101), 10),                 // m size of vector takes values from 10 to 100 with step size of 10.
            ::testing::Values(gtint_t(1)),                                   /*(gtint_t(-5), gtint_t(-17))*/  // stride size for x
            ::testing::Values(gtint_t(1)),                                   /*(gtint_t(-12), gtint_t(-4))*/  // stride size for y
            ::testing::Values(dcomplex{2.0, -1.0}, dcomplex{-2.0, 3.0}),     // beta
            ::testing::Values(ELEMENT_TYPE)                                  // i : integer, f : float  datatype type tested
        ),
        ::zxpbyvGenericTestPrint()
    );

// Test for non-unit increments.
// Only test very few cases as sanity check.
// We can modify the values using implementantion details.
INSTANTIATE_TEST_SUITE_P(
        NonUnitIncrements,
        zxpbyvGenericTest,
        ::testing::Combine(
            ::testing::Values('n', 'c'),                                     // n: use x, c: use conj(x)
            ::testing::Range(gtint_t(10), gtint_t(101), 10),                 // m size of vector takes values from 10 to 100 with step size of 10.
            ::testing::Values(gtint_t(2), gtint_t(11)),                      /*(gtint_t(-5), gtint_t(-17))*/  // stride size for x
            ::testing::Values(gtint_t(3), gtint_t(33)),                      /*(gtint_t(-12), gtint_t(-4))*/  // stride size for y
            ::testing::Values(dcomplex{4.0, 3.1}),                           // beta
            ::testing::Values(ELEMENT_TYPE)                                  // i : integer, f : float  datatype type tested
        ),
        ::zxpbyvGenericTestPrint()
    );
#endif
