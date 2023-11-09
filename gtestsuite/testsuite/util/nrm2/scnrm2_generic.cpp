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

class scnrm2Test :
        public ::testing::TestWithParam<std::tuple<gtint_t, gtint_t>> {};

TEST_P( scnrm2Test, RandomData )
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

    // Set the threshold for the errors:
    double thresh = std::sqrt(n)*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_nrm2<T>(n, incx, thresh);
}

// Prints the test case combination
class scnrm2TestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<gtint_t, gtint_t>> str) const {
        gtint_t n     = std::get<0>(str.param);
        gtint_t incx  = std::get<1>(str.param);
#ifdef TEST_BLAS
        std::string str_name = "scnrm2_";
#elif TEST_CBLAS
        std::string str_name = "cblas_scnrm2";
#else  //#elif TEST_BLIS_TYPED
        std::string str_name = "bli_cnormfv";
#endif
        str_name    = str_name + "_" + std::to_string(n);
        std::string incx_str = ( incx > 0) ? std::to_string(incx) : "m" + std::to_string(std::abs(incx));
        str_name    = str_name + "_" + incx_str;
        return str_name;
    }
};

/**
 * scnrm2 implementation is composed by two parts:
 * - vectorized path for n>=64
 *      - for-loop for multiples of 16 (F16)
 *      - for-loop for multiples of 12 (F12)
 *      - for-loop for multiples of 8  (F8)
 * - scalar path for n<64 (S)
*/
INSTANTIATE_TEST_SUITE_P(
        AT,
        scnrm2Test,
        ::testing::Combine(
            // m size of vector
            ::testing::Values(gtint_t(1),  // trivial case n=1
                              gtint_t(35), // will only go through S
                              gtint_t(64), // 4*16 - will only go through F16
                              gtint_t(67), // 4*16 + 3 - will go through F16 & S
                              gtint_t(72), // 4*16 + 8 - will go through F16 & F8
                              gtint_t(75), // 4*16 + 8 + 3 - will go through F16 & F8 & S
                              gtint_t(76), // 4*16 + 12 - will go through F16 & F12
                              gtint_t(78), // 4*16 + 12 + 2 - will go through F16 & F12 & S
                              gtint_t(112), // a few bigger numbers
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
        ::scnrm2TestPrint()
    );
