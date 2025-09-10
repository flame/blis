/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
   Portions of this file consist of AI-generated content.

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
#include "test_addv_ukr.h"
#include "common/blis_version_defs.h"

class zaddvGeneric :
        public ::testing::TestWithParam<std::tuple<zaddv_ker_ft,    // Function pointer type for zaddv kernels
                                                   char,            // conj_x
                                                   gtint_t,         // n
                                                   gtint_t,         // incx
                                                   gtint_t,         // incy
                                                   bool>> {};       // is_memory_test

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(zaddvGeneric);

// Defining the testsuite to check the accuracy of zaddv micro-kernels
TEST_P( zaddvGeneric, UKR )
{
    using T = dcomplex;

    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------

    // Assign the kernel address to the function pointer
    zaddv_ker_ft ukr_fp = std::get<0>(GetParam());
    // denotes whether x or conj(x) will be added to y
    char conj_x = std::get<1>(GetParam());
    // vector length
    gtint_t n = std::get<2>(GetParam());
    // stride size for x
    gtint_t incx = std::get<3>(GetParam());
    // stride size for y
    gtint_t incy = std::get<4>(GetParam());
    // is_memory_test
    bool is_memory_test = std::get<5>(GetParam());

    // Set the threshold for the errors
    double threshold = 2 * testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_addv_ukr<T, zaddv_ker_ft>( ukr_fp, conj_x, n, incx, incy, threshold, is_memory_test );
}

// ----------------------------------------------
// ----- Begin ZEN1/2/3 (AVX2) Kernel Tests -----
// ----------------------------------------------
#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)
/*
    Unit testing for functionality of bli_zaddv_zen_int kernel.
    The code structure for bli_zaddv_zen_int( ... ) is as follows :
    For unit strides :
        Main loop    :  In blocks of 24  --> L24
        Fringe loops :  In blocks of 16  --> L16
                        In blocks of 8   --> L8
                        In blocks of 4   --> L4
                        In blocks of 2   --> L2
                        Element-wise loop --> LScalar

    For non-unit strides : A single loop, to process element wise.
*/
#ifdef K_bli_zaddv_zen_int
INSTANTIATE_TEST_SUITE_P(
        bli_zaddv_zen_int_unitStrides,
        zaddvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_zaddv_zen_int),      // kernel address
            ::testing::Values('n'                      // conjx
#ifdef TEST_BLIS_TYPED
                            , 'c'
#endif
                             ),
            ::testing::Values(// Testing the loops standalone
                              gtint_t(24),             // size n, for L24
                              gtint_t(16),             // L16
                              gtint_t(8),              // L8
                              gtint_t(4),              // L4
                              gtint_t(2),              // L2
                              gtint_t(1),              // LScalar
                              gtint_t(64),             // 2*L24 + L16
                              gtint_t(63)),            // 2*L24 + L8 + L4 + 3(LScalar)
            ::testing::Values(gtint_t(1)),             // stride size for x
            ::testing::Values(gtint_t(1)),             // stride size for y
            ::testing::Values(false, true)             // is_memory_test
        ),
        (::addvUKRPrint<dcomplex, zaddv_ker_ft>())
    );
#endif

#ifdef K_bli_zaddv_zen_int
INSTANTIATE_TEST_SUITE_P(
        bli_zaddv_zen_int_nonUnitStrides,
        zaddvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_zaddv_zen_int),      // kernel address
            ::testing::Values('n'                      // conjx
#ifdef TEST_BLIS_TYPED
                            , 'c'
#endif
                             ),
            ::testing::Values(// Testing the loops standalone
                              gtint_t(7),              // size n, for LScalar
                              gtint_t(15)),
            ::testing::Values(gtint_t(3), gtint_t(5)), // stride size for x
            ::testing::Values(gtint_t(2), gtint_t(4)), // stride size for y
            ::testing::Values(false, true)             // is_memory_test
        ),
        (::addvUKRPrint<dcomplex, zaddv_ker_ft>())
    );
#endif
#endif
// ----------------------------------------------
// -----  End ZEN1/2/3 (AVX2) Kernel Tests  -----
// ----------------------------------------------
