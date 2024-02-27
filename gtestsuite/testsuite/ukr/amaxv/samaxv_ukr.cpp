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
#include "test_amaxv_ukr.h"

class samaxvUkr :
        public ::testing::TestWithParam<std::tuple<samaxv_ker_ft,   // Function pointer type for samaxv kernels
                                                   gtint_t,         // n
                                                   gtint_t,         // incx
                                                   bool>> {};       // is_memory_test

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(samaxvUkr);

// Tests using random integers as vector elements.
TEST_P( samaxvUkr, AccuracyCheck )
{
    using T = float;

    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------

    // Assign the kernel address to the function pointer
    samaxv_ker_ft ukr_fp = std::get<0>(GetParam());
    // vector length:
    gtint_t n = std::get<1>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<2>(GetParam());
    // is_memory_test
    bool is_memory_test = std::get<3>(GetParam());

    // Set the threshold for the errors:
    double thresh = testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_amaxv_ukr<T, samaxv_ker_ft>( ukr_fp, n, incx, thresh, is_memory_test );
}

// Test-case logger : Used to print the test-case details for unit testing the kernels.
// NOTE : The kernel name is the prefix in instantiator name, and thus is not printed
// with this logger.
class samaxvUkrPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<samaxv_ker_ft,gtint_t,gtint_t,bool>> str) const {
        gtint_t n     = std::get<1>(str.param);
        gtint_t incx  = std::get<2>(str.param);
        bool is_memory_test = std::get<3>(str.param);

        std::string str_name = "n" + std::to_string(n);
        std::string incx_str = ( incx >= 0) ? std::to_string(incx) : "m" + std::to_string(std::abs(incx));
        str_name += "_incx" + incx_str;
        str_name += ( is_memory_test ) ? "_mem_test_enabled" : "_mem_test_disabled";
        return str_name;
    }
};

#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)
/*
    Unit testing for functionality of bli_samaxv_zen_int kernel.
    The code structure for bli_samaxv_zen_int( ... ) is as follows :

    For unit strides :
        Main loop    :  In blocks of 8 --> L8
        Fringe loops :  Element-wise loop --> LScalar

    For non-unit strides : A single loop, to process element wise.
*/
// Unit testing with unit strides, across all loops.
INSTANTIATE_TEST_SUITE_P(
        bli_samaxv_zen_int_unitStrides,
        samaxvUkr,
        ::testing::Combine(
            ::testing::Values(bli_samaxv_zen_int),   // kernel address
            ::testing::Values(gtint_t(8),            // for size n, L8
                              gtint_t(7),            // LScalar
                              gtint_t(40),           // 5*L8
                              gtint_t(47)),          // 5*L8 + LScalar
            ::testing::Values(gtint_t(1)),           // incx
            ::testing::Values(false, true)           // is_memory_test
        ),
        ::samaxvUkrPrint()
    );

// Unit testing with non-unit strides.
INSTANTIATE_TEST_SUITE_P(
        bli_samaxv_zen_int_nonUnitStrides,
        samaxvUkr,
        ::testing::Combine(
            ::testing::Values(bli_samaxv_zen_int),   // kernel address
            ::testing::Values(gtint_t(10),           // n, size of the vector
                              gtint_t(25)),
            ::testing::Values(gtint_t(5)),           // incx
            ::testing::Values(false, true)           // is_memory_test
        ),
        ::samaxvUkrPrint()
    );
#endif

#if defined(BLIS_KERNELS_ZEN4) && defined(GTEST_AVX512)
/*
    Unit testing for functionality of bli_samaxv_zen_int_avx512 kernel.
    The code structure for bli_samaxv_zen_int_avx512( ... ) is as follows :

    For unit strides :
        Main loop    :  In blocks of 80 --> L80
        Fringe loops :  In blocks of 16 --> L16
                        Element-wise loop --> LScalar

    For non-unit strides : A single loop, to process element wise.
*/
// Unit testing with unit strides, across all loops.
INSTANTIATE_TEST_SUITE_P(
        bli_samaxv_zen_int_avx512_unitStrides,
        samaxvUkr,
        ::testing::Combine(
            ::testing::Values(bli_samaxv_zen_int_avx512),   // kernel address
            ::testing::Values(gtint_t(80),                  // for size n, L80
                              gtint_t(48),                  // 3*L16 
                              gtint_t(16),                  // L16
                              gtint_t(11),                  // 11(LScalar)
                              gtint_t(317)),                // 3*L80 + 4*L16 + 13(LScalar)
            ::testing::Values(gtint_t(1)),                  // incx
            ::testing::Values(false, true)                  // is_memory_test
        ),
        ::samaxvUkrPrint()
    );

// Unit testing with non-unit strides.
INSTANTIATE_TEST_SUITE_P(
        bli_samaxv_zen_int_avx512_nonUnitStrides,
        samaxvUkr,
        ::testing::Combine(
            ::testing::Values(bli_samaxv_zen_int_avx512),   // kernel address
            ::testing::Values(gtint_t(10),                  // n, size of the vector
                              gtint_t(25)),
            ::testing::Values(gtint_t(5)),                  // incx
            ::testing::Values(false, true)                  // is_memory_test
        ),
        ::samaxvUkrPrint()
    );
#endif