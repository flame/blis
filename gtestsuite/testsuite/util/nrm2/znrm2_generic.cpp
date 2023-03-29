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

class znrm2Test :
        public ::testing::TestWithParam<std::tuple<gtint_t, gtint_t, char>> {};

TEST_P( znrm2Test, RandomData )
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
    // specifies the datatype for randomgenerators
    char datatype = std::get<2>(GetParam());

    // Set the threshold for the errors:
    double thresh = testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_nrm2<T>(n, incx, thresh, datatype);
}

// Prints the test case combination
class znrm2TestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<gtint_t, gtint_t, char>> str) const {
        gtint_t n     = std::get<0>(str.param);
        gtint_t incx  = std::get<1>(str.param);
        char datatype = std::get<2>(str.param);
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
        str_name    = str_name + "_" + datatype;
        return str_name;
    }
};

// Black box testing.
INSTANTIATE_TEST_SUITE_P(
        Blackbox,
        znrm2Test,
        ::testing::Combine(
            ::testing::Range(gtint_t(10), gtint_t(101), 10),                 // m size of vector takes values from 10 to 100 with step size of 10.
            ::testing::Values(gtint_t(1), gtint_t(2)
#ifndef TEST_BLIS_TYPED
            ,gtint_t(-1), gtint_t(-2)
#endif
        ),                                                                   // stride size for x
            ::testing::Values('i')                                           // i : integer, f : float  datatype type tested
        ),
        ::znrm2TestPrint()
    );
