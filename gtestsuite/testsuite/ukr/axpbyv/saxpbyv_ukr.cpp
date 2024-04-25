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

class saxpbyvUkrTest :
        public ::testing::TestWithParam<std::tuple<saxpbyv_ker_ft,  // Function pointer type for saxpbyv kernels
                                                   char,            // conjx
                                                   gtint_t,         // n
                                                   gtint_t,         // incx
                                                   gtint_t,         // incy
                                                   float,          // alpha
                                                   float>> {};     // beta

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(saxpbyvUkrTest);

// Tests using random integers as vector elements.
TEST_P( saxpbyvUkrTest, AccuracyCheck )
{
    using T = float;

    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------

    // Assign the kernel address to the function pointer
    saxpbyv_ker_ft ukr_fp = std::get<0>(GetParam());
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
    // Check gtestsuite axpbyv.h (no netlib version) for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    if (n == 0)
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<T>())
        // Like SCALV
        if (beta == testinghelpers::ZERO<T>() || beta == testinghelpers::ONE<T>())
            thresh = 0.0;
        else
            thresh = testinghelpers::getEpsilon<T>();
    else if (beta == testinghelpers::ZERO<T>())
        // Like SCAL2V
        if (alpha == testinghelpers::ZERO<T>() || alpha == testinghelpers::ONE<T>())
            thresh = 0.0;
        else
            thresh = testinghelpers::getEpsilon<T>();
    else if (beta == testinghelpers::ONE<T>())
        // Like AXPYV
        if (alpha == testinghelpers::ZERO<T>())
            thresh = 0.0;
        else
            thresh = 2*testinghelpers::getEpsilon<T>();
    else if (alpha == testinghelpers::ONE<T>())
        thresh = 2*testinghelpers::getEpsilon<T>();
    else
        thresh = 3*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_axpbyv_ukr<T, saxpbyv_ker_ft>( ukr_fp, conj_x, n, incx, incy, alpha, beta, thresh );
}

// Test-case logger : Used to print the test-case details for unit testing the kernels.
// NOTE : The kernel name is the prefix in instantiator name, and thus is not printed
// with this logger.
class saxpbyvUkrTestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<saxpbyv_ker_ft,char,gtint_t,gtint_t,gtint_t,float,float>> str) const {
        char conjx     = std::get<1>(str.param);
        gtint_t n     = std::get<2>(str.param);
        gtint_t incx  = std::get<3>(str.param);
        gtint_t incy  = std::get<4>(str.param);
        float alpha  = std::get<5>(str.param);
        float beta   = std::get<6>(str.param);

        std::string str_name = "saxpbyv_ukr";
        str_name += "_n_" + std::to_string(n);
        str_name += ( conjx == 'n' )? "_noconjx" : "_conjx";
        str_name += "_incx_" + testinghelpers::get_value_string(incx);
        str_name += "_incy_" + testinghelpers::get_value_string(incy);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name += "_beta_" + testinghelpers::get_value_string(beta);
        return str_name;
    }
};

#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)
// Unit testing with unit stride
INSTANTIATE_TEST_SUITE_P(
        bli_saxpbyv_zen_int10_unitStride,
        saxpbyvUkrTest,
        ::testing::Combine(
            ::testing::Values(bli_saxpbyv_zen_int10), // kernel address
            ::testing::Values('n'),                   // use x, not conj(x) (since it is real)
            ::testing::Values(gtint_t(32), gtint_t(45)), // size n
            ::testing::Values(gtint_t(1)),            // stride size for x
            ::testing::Values(gtint_t(1)),            // stride size for y
            ::testing::Values(float(2.2)), // alpha
            ::testing::Values(float(-1.8))  // beta
        ),
        ::saxpbyvUkrTestPrint()
    );

// Unit testing with unit stride
INSTANTIATE_TEST_SUITE_P(
        bli_saxpbyv_zen_int_unitStride,
        saxpbyvUkrTest,
        ::testing::Combine(
            ::testing::Values(bli_saxpbyv_zen_int),   // kernel address
            ::testing::Values('n'),                   // use x, not conj(x) (since it is real)
            ::testing::Values(gtint_t(32), gtint_t(45)), // size n
            ::testing::Values(gtint_t(1)),            // stride size for x
            ::testing::Values(gtint_t(1)),            // stride size for y
            ::testing::Values(float(2.2)), // alpha
            ::testing::Values(float(-1.8))  // beta
        ),
        ::saxpbyvUkrTestPrint()
    );
#endif
