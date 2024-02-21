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
#include "test_scalv_ukr.h"

class zscalvUkrTest :
        public ::testing::TestWithParam<std::tuple<zscalv_ker_ft,   // Function pointer for zscalv kernels
                                                   char,            // conj_alpha
                                                   gtint_t,         // n
                                                   gtint_t,         // incx
                                                   dcomplex>> {};   // alpha
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(zscalvUkrTest);

// Tests using random integers as vector elements.
TEST_P( zscalvUkrTest, RandomData )
{
    using T = dcomplex;

    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------

    // denotes the kernel to be tested:
    zscalv_ker_ft ukr = std::get<0>(GetParam());
    // denotes whether alpha or conj(alpha) will be used:
    char conj_alpha = std::get<1>(GetParam());
    // vector length:
    gtint_t n = std::get<2>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<3>(GetParam());
    // alpha
    T alpha = std::get<4>(GetParam());

    // Set the threshold for the errors:
    double thresh = testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_scalv_ukr<T, T, zscalv_ker_ft>( ukr, conj_alpha, n, incx, alpha, thresh, true );
}

// Used to generate a test case with a sensible name.
// Beware that we cannot use fp numbers (e.g., 2.3) in the names,
// so we are only printing int(2.3). This should be enough for debugging purposes.
// If this poses an issue, please reach out.
class zscalvUkrTestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<zscalv_ker_ft, char, gtint_t, gtint_t, dcomplex>> str) const {
        char conjx = std::get<1>(str.param);
        gtint_t n = std::get<2>(str.param);
        gtint_t incx = std::get<3>(str.param);
        dcomplex alpha = std::get<4>(str.param);

        std::string str_name = "zscalvUkrTest";
        str_name += "_n" + std::to_string(n);
        str_name += (conjx == 'n') ? "_noconjx" : "_conjx";
        std::string incx_str = ( incx > 0) ? std::to_string(incx) : "m" + std::to_string(std::abs(incx));
        str_name += "_incx" + incx_str;
        std::string alpha_str = ( alpha.real > 0) ? std::to_string(int(alpha.real)) : ("m" + std::to_string(int(std::abs(alpha.real))));
                    alpha_str = alpha_str + "pi" + (( alpha.imag > 0) ? std::to_string(int(alpha.imag)) : ("m" + std::to_string(int(std::abs(alpha.imag)))));
        str_name = str_name + "_a" + alpha_str;

        return str_name;
    }
};


// ----------------------------------------------
// ----- Begin ZEN1/2/3 (AVX2) Kernel Tests -----
// ----------------------------------------------
#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)
// Tests for bli_zscalv_zen_int (AVX2) kernel.
/**
 * Loops:
 * L8      - Main loop, handles 8 elements
 * L4      - handles 4 elements
 * L2      - handles 2 elements
 * LScalar - leftover loop (also handles non-unit increments)
*/
INSTANTIATE_TEST_SUITE_P(
        bli_zscalv_zen_int_unitPositiveStride,
        zscalvUkrTest,
        ::testing::Combine(
            ::testing::Values(bli_zscalv_zen_int),
            // conj(alpha): uses n (no_conjugate) since it is real.
            ::testing::Values('n'),
            // m: size of vector.
            ::testing::Values(
                                gtint_t(16),       // L8 (executed twice)
                                gtint_t(15),       // L8 upto LScalar
                                gtint_t( 8),       // L8
                                gtint_t( 4),       // L4
                                gtint_t( 2),       // L2
                                gtint_t( 1)        // LScalar
            ),
            // incx: stride of x vector.
            ::testing::Values(
                                gtint_t(1)      // unit stride
            ),
            // alpha: value of scalar.
            ::testing::Values(
                                dcomplex{-5.1, -7.3},
                                dcomplex{ 0.0,  0.0},
                                dcomplex{ 7.3,  5.1}
            )
        ),
        ::zscalvUkrTestPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        bli_zscalv_zen_int_nonUnitPositiveStrides,
        zscalvUkrTest,
        ::testing::Combine(
            ::testing::Values(bli_zscalv_zen_int),
            // conj(alpha): uses n (no_conjugate) since it is real.
            ::testing::Values('n'),
            // m: size of vector.
            ::testing::Values(
                                gtint_t(3), gtint_t(30), gtint_t(112)
            ),
            // incx: stride of x vector.
            ::testing::Values(
                                gtint_t(3), gtint_t(7)       // few non-unit strides for sanity check
            ),
            // alpha: value of scalar.
            ::testing::Values(
                                dcomplex{-5.1, -7.3},
                                dcomplex{ 0.0,  0.0},
                                dcomplex{ 7.3,  5.1}
            )
        ),
        ::zscalvUkrTestPrint()
    );
#endif
// ----------------------------------------------
// -----  End ZEN1/2/3 (AVX2) Kernel Tests  -----
// ----------------------------------------------
