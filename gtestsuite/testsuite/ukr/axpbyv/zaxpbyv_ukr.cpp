/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

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
#include "test_axpbyv_ukr.h"

class zaxpbyvUkrTest :
        public ::testing::TestWithParam<std::tuple<zaxpbyv_ker_ft,  // Function pointer type for zaxpbyv kernels
                                                   char,            // conjx
                                                   gtint_t,         // n
                                                   gtint_t,         // incx
                                                   gtint_t,         // incy
                                                   dcomplex,          // alpha
                                                   dcomplex>> {};     // beta

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(zaxpbyvUkrTest);

// Tests using random integers as vector elements.
TEST_P( zaxpbyvUkrTest, AccuracyCheck )
{
    using T = dcomplex;

    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------

    // Assign the kernel address to the function pointer
    zaxpbyv_ker_ft ukr_fp = std::get<0>(GetParam());
    // denotes whether x or conj(x) will be added to y:
    char conj_x = std::get<1>(GetParam());
    // vector length:
    gtint_t n = std::get<2>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<3>(GetParam());
    // stride size for y:
    gtint_t incy = std::get<4>(GetParam());
    // alpha
    T alpha = std::get<5>(GetParam());
    // beta
    T beta = std::get<6>(GetParam());

    // Set the threshold for the errors:
    double thresh = 20 * testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_axpbyv_ukr<T, zaxpbyv_ker_ft>( ukr_fp, conj_x, n, incx, incy, alpha, beta, thresh );
}

// Test-case logger : Used to print the test-case details for unit testing the kernels.
// NOTE : The kernel name is the prefix in instantiator name, and thus is not printed
// with this logger.
class zaxpbyvUkrTestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<zaxpbyv_ker_ft,char,gtint_t,gtint_t,gtint_t,dcomplex,dcomplex>> str) const {
        char conj     = std::get<1>(str.param);
        gtint_t n     = std::get<2>(str.param);
        gtint_t incx  = std::get<3>(str.param);
        gtint_t incy  = std::get<4>(str.param);
        dcomplex alpha  = std::get<5>(str.param);
        dcomplex beta   = std::get<6>(str.param);

        std::string str_name = "zaxpbyv_ukr";
        str_name += "_n" + std::to_string(n);
        str_name += "_conjx" + std::string(&conj, 1);
        std::string incx_str = (incx > 0) ? std::to_string(incx) : "m" + std::to_string(std::abs(incx));
        str_name += "_incx" + incx_str;
        std::string incy_str = (incy > 0) ? std::to_string(incy) : "m" + std::to_string(std::abs(incy));
        str_name += "_incy" + incy_str;
        std::string alpha_str = (alpha.real > 0) ? std::to_string(int(alpha.real)) : ("m" + std::to_string(int(std::abs(alpha.real))));
        alpha_str = alpha_str + "pi" + ((alpha.imag > 0) ? std::to_string(int(alpha.imag)) : ("m" + std::to_string(int(std::abs(alpha.imag)))));
        std::string beta_str = (beta.real > 0) ? std::to_string(int(beta.real)) : ("m" + std::to_string(int(std::abs(beta.real))));
        beta_str = beta_str + "pi" + ((beta.imag > 0) ? std::to_string(int(beta.imag)) : ("m" + std::to_string(int(std::abs(beta.imag)))));
        str_name = str_name + "_a" + alpha_str;
        str_name = str_name + "_b" + beta_str;
        return str_name;
    }
};

#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)
// Unit testing with unit stride
INSTANTIATE_TEST_SUITE_P(
        bli_zaxpbyv_zen_int_unitStride,
        zaxpbyvUkrTest,
        ::testing::Combine(
            ::testing::Values(bli_zaxpbyv_zen_int),   // kernel address
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                              ,'c'                   // conjx parameter
#endif
                             ),
            ::testing::Values(gtint_t(32), gtint_t(45)), // size n
            ::testing::Values(gtint_t(1)),            // stride size for x
            ::testing::Values(gtint_t(1)),            // stride size for y
            ::testing::Values(dcomplex{2.2, -4.1}), // alpha
            ::testing::Values(dcomplex{2.2, -4.1})  // beta
        ),
        ::zaxpbyvUkrTestPrint()
    );
#endif