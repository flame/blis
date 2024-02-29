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

class zaxpbyvUkr :
        public ::testing::TestWithParam<std::tuple<zaxpbyv_ker_ft,  // Function pointer type for zaxpbyv kernels
                                                   char,            // conjx
                                                   gtint_t,         // n
                                                   gtint_t,         // incx
                                                   gtint_t,         // incy
                                                   dcomplex,        // alpha
                                                   dcomplex,        // beta
                                                   bool>> {};       // is_memory_test

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(zaxpbyvUkr);

// Tests using random integers as vector elements.
TEST_P( zaxpbyvUkr, AccuracyCheck )
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
    // is_memory_test
    bool is_memory_test = std::get<7>(GetParam());

    // Set the threshold for the errors:
    double thresh = 3 * testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_axpbyv_ukr<T, zaxpbyv_ker_ft>( ukr_fp, conj_x, n, incx, incy, alpha, beta, thresh, is_memory_test );
}

// Test-case logger : Used to print the test-case details for unit testing the kernels.
// NOTE : The kernel name is the prefix in instantiator name, and thus is not printed
// with this logger.
class zaxpbyvUkrPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<zaxpbyv_ker_ft,char,gtint_t,gtint_t,gtint_t,dcomplex,dcomplex,bool>> str) const {
        char conjx     = std::get<1>(str.param);
        gtint_t n     = std::get<2>(str.param);
        gtint_t incx  = std::get<3>(str.param);
        gtint_t incy  = std::get<4>(str.param);
        dcomplex alpha  = std::get<5>(str.param);
        dcomplex beta   = std::get<6>(str.param);
        bool is_memory_test = std::get<7>(str.param);

        std::string str_name = "n" + std::to_string(n);
        str_name += ( conjx == 'n' )? "_noconj_x" : "_conj_x";
        std::string incx_str = (incx >= 0) ? std::to_string(incx) : "m" + std::to_string(std::abs(incx));
        str_name += "_incx" + incx_str;
        std::string incy_str = (incy >= 0) ? std::to_string(incy) : "m" + std::to_string(std::abs(incy));
        str_name += "_incy" + incy_str;
        std::string alpha_str = (alpha.real >= 0) ? std::to_string(int(alpha.real)) : ("m" + std::to_string(int(std::abs(alpha.real))));
        alpha_str = alpha_str + "pi" + ((alpha.imag >= 0) ? std::to_string(int(alpha.imag)) : ("m" + std::to_string(int(std::abs(alpha.imag)))));
        std::string beta_str = (beta.real >= 0) ? std::to_string(int(beta.real)) : ("m" + std::to_string(int(std::abs(beta.real))));
        beta_str = beta_str + "pi" + ((beta.imag >= 0) ? std::to_string(int(beta.imag)) : ("m" + std::to_string(int(std::abs(beta.imag)))));
        str_name = str_name + "_alpha" + alpha_str;
        str_name = str_name + "_beta" + beta_str;
        str_name += ( is_memory_test ) ? "_mem_test_enabled" : "_mem_test_disabled";
        return str_name;
    }
};

#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)
/*
    Unit testing for functionality of bli_zaxpbyv_zen_int kernel.
    The code structure for bli_zaxpbyv_zen_int( ... ) is as follows :
    For unit strides :
        Main loop    :  In blocks of 8 --> L8
        Fringe loops :  In blocks of 6 --> L6
                        In blocks of 4 --> L4
                        In blocks of 2 --> L2
                        Element-wise loop --> LScalar

    For non-unit strides : A single loop, to process element wise.
*/

INSTANTIATE_TEST_SUITE_P(
        bli_zaxpbyv_zen_int_unitStrides,
        zaxpbyvUkr,
        ::testing::Combine(
            ::testing::Values(bli_zaxpbyv_zen_int),                     // kernel address
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                            , 'c'                                       // conjx
#endif
            ),
            ::testing::Values(// Testing the loops standalone
                              gtint_t(8),                               // size n, for L8
                              gtint_t(6),                               // L6
                              gtint_t(4),                               // L4
                              gtint_t(2),                               // L2
                              gtint_t(1),                               // L1
                              gtint_t(56),                              // 7*L8
                              gtint_t(62),                              // 7*L8 + L6
                              gtint_t(60),                              // 7*L8 + L4
                              gtint_t(58),                              // 7*L8 + L2
                              gtint_t(57),                              // 7*L8 + 1(LScalar)
                              gtint_t(59),                              // 7*L8 + L2 + 1(LScalar)
                              gtint_t(61),                              // 7*L8 + L4 + 1(LScalar)
                              gtint_t(63)),                             // 7*L8 + L6 + 1(LScalar)
            ::testing::Values(gtint_t(1)),                              // stride size for x
            ::testing::Values(gtint_t(1)),                              // stride size for y
            ::testing::Values(dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{0.0, 1.0}, dcomplex{0.0, -1.0},
                              dcomplex{0.0, 0.0}, dcomplex{2.3, -3.7}), // alpha
            ::testing::Values(dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{0.0, 1.0}, dcomplex{0.0, -1.0},
                              dcomplex{0.0, 0.0}, dcomplex{2.3, -3.7}), // beta
            ::testing::Values(false, true)                              // is_memory_test
        ),
        ::zaxpbyvUkrPrint()

    );

INSTANTIATE_TEST_SUITE_P(
        bli_zaxpbyv_zen_int_nonUnitStrides,
        zaxpbyvUkr,
        ::testing::Combine(
            ::testing::Values(bli_zaxpbyv_zen_int),                     // kernel address
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                            , 'c'                                       // conjx
#endif
            ),
            ::testing::Values(gtint_t(10),                              // n, size of the vector
                              gtint_t(25)),
            ::testing::Values(gtint_t(5)),                              // stride size for x
            ::testing::Values(gtint_t(3)),                              // stride size for y
            ::testing::Values(dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{0.0, 1.0}, dcomplex{0.0, -1.0},
                              dcomplex{0.0, 0.0}, dcomplex{2.3, -3.7}), // alpha
            ::testing::Values(dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{0.0, 1.0}, dcomplex{0.0, -1.0},
                              dcomplex{0.0, 0.0}, dcomplex{2.3, -3.7}), // beta
            ::testing::Values(false, true)                              // is_memory_test
        ),
        ::zaxpbyvUkrPrint()
    );
#endif