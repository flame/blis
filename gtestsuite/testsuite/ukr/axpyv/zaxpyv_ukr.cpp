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
#include "test_axpyv_ukr.h"

class zaxpyvGeneric :
        public ::testing::TestWithParam<std::tuple<zaxpyv_ker_ft,   // Function pointer type for zaxpyv kernels
                                                   char,            // conjx
                                                   gtint_t,         // n
                                                   gtint_t,         // incx
                                                   gtint_t,         // incy
                                                   dcomplex,        // alpha
                                                   bool>> {};       // is_memory_test

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(zaxpyvGeneric);

// Defining the testsuite to check the accuracy of zaxpyv micro-kernels
TEST_P( zaxpyvGeneric, UKR )
{
    using T = dcomplex;

    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------

    // Assign the kernel address to the function pointer
    zaxpyv_ker_ft ukr_fp = std::get<0>(GetParam());
    // denotes whether x or conj(x) will be added to y:
    char conj_x = std::get<1>(GetParam());
    // vector length
    gtint_t n = std::get<2>(GetParam());
    // stride size for x
    gtint_t incx = std::get<3>(GetParam());
    // stride size for y
    gtint_t incy = std::get<4>(GetParam());
    // alpha
    T alpha = std::get<5>(GetParam());
    // is_memory_test
    bool is_memory_test = std::get<6>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite axpbyv.h (no netlib version) for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    // No adjustment applied yet for complex data.
    double thresh;
    if (n == 0)
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<T>() || alpha == testinghelpers::ONE<T>())
        thresh = 0.0;
    else
        thresh = 2*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_axpyv_ukr<T, zaxpyv_ker_ft>( ukr_fp, conj_x, n, incx, incy, alpha, thresh, is_memory_test );
}

#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)
/*
    Unit testing for functionality of bli_zaxpyv_zen_int5 kernel.
    The code structure for bli_zaxpyv_zen_int10( ... ) is as follows :
    For unit strides :
        Main loop    :  In blocks of 14 --> L14
        Fringe loops :  In blocks of 10 --> L10
                        In blocks of 6  --> L6
                        In blocks of 4  --> L4
                        In blocks of 2  --> L2
                        Element-wise loop --> LScalar

    For non-unit strides : A single loop, to process element wise.
*/
// Unit testing with unit strides, across all loops.
INSTANTIATE_TEST_SUITE_P(
        bli_zaxpyv_zen_int5_unitStrides,
        zaxpyvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_zaxpyv_zen_int5),                     // kernel address
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                            , 'c'                                       // conjx
#endif
            ),
            ::testing::Values(// Testing the loops standalone
                              gtint_t(14),                              // size n, for L14
                              gtint_t(10),                              // L10
                              gtint_t(6),                               // L6
                              gtint_t(4),                               // L4
                              gtint_t(2),                               // L2
                              gtint_t(1),                               // LScalar
                              // Testing the loops with combination
                              gtint_t(42),                              // 3*L14
                              gtint_t(52),                              // 3*L14 + L10
                              gtint_t(48),                              // 3*L14 + L6
                              gtint_t(46),                              // 3*L14 + L4
                              gtint_t(45)),                             // 3*L14 + L2 + LScalar
            ::testing::Values(gtint_t(1)),                              // stride size for x
            ::testing::Values(gtint_t(1)),                              // stride size for y
            ::testing::Values(dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{0.0, 1.0}, dcomplex{0.0, -1.0},
                              dcomplex{0.0, -3.3}, dcomplex{4.3,-2.1},
                              dcomplex{0.0, 0.0}),                      // alpha
            ::testing::Values(false, true)                              // is_memory_test
        ),
        (::axpyvUKRPrint<dcomplex, zaxpyv_ker_ft>())
    );

// Unit testing for non unit strides
INSTANTIATE_TEST_SUITE_P(
        bli_zaxpyv_zen_int5_nonUnitStrides,
        zaxpyvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_zaxpyv_zen_int5),                     // kernel address
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                            , 'c'                                       // conjx
#endif
            ),
            ::testing::Values(gtint_t(2)),                              // n, size of the vector
            ::testing::Values(gtint_t(5)),                              // stride size for x
            ::testing::Values(gtint_t(3)),                              // stride size for y
            ::testing::Values(dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{0.0, 1.0}, dcomplex{0.0, -1.0},
                              dcomplex{0.0, -3.3}, dcomplex{4.3,-2.1},
                              dcomplex{0.0, 0.0}),                      // alpha
            ::testing::Values(false, true)                              // is_memory_test
        ),
        (::axpyvUKRPrint<dcomplex, zaxpyv_ker_ft>())
    );

#endif

#if defined(BLIS_KERNELS_ZEN4) && defined(GTEST_AVX512)
/*
    Unit testing for functionality of bli_zaxpyv_zen_int_avx512 kernel.
    The code structure for bli_zaxpyv_zen_int_avx512( ... ) is as follows :
    For unit strides :
        Main loop    :  In blocks of 32 --> L32
        Fringe loops :  In blocks of 16 --> L16
                        In blocks of 8  --> L8
                        In blocks of 4  --> L4
                        Masked loop     ---> LScalar

    For non-unit strides : A single loop, to process element wise.
*/
// Unit testing with unit strides, across all loops.
INSTANTIATE_TEST_SUITE_P(
        bli_zaxpyv_zen_int_avx512_unitStrides,
        zaxpyvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_zaxpyv_zen_int_avx512),               // kernel address
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                            , 'c'                                       // conjx
#endif
            ),
            ::testing::Values(// Testing the loops standalone
                              gtint_t(32),                              // size n, for L32
                              gtint_t(16),                              // L16
                              gtint_t(8),                               // L8
                              gtint_t(4),                               // L4
                              gtint_t(3),                               // LScalar
                              // Testing the loops with combination
                              gtint_t(96),                              // 3*L32
                              gtint_t(112),                             // 3*L32 + L116
                              gtint_t(120),                             // 3*L32 + L16 + L8
                              gtint_t(124),                             // 3*L32 + L16 + L8 + L4
                              gtint_t(127)),                            // 3*L32 + L16 + L8 + L4 + LScalar
            ::testing::Values(gtint_t(1)),                              // stride size for x
            ::testing::Values(gtint_t(1)),                              // stride size for y
            ::testing::Values(dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{0.0, 1.0}, dcomplex{0.0, -1.0},
                              dcomplex{0.0, -3.3}, dcomplex{4.3,-2.1},
                              dcomplex{0.0, 0.0}),                      // alpha
            ::testing::Values(false, true)                              // is_memory_test
        ),
        (::axpyvUKRPrint<dcomplex, zaxpyv_ker_ft>())
    );

// Unit testing for non unit strides
INSTANTIATE_TEST_SUITE_P(
        bli_zaxpyv_zen_int_avx512_nonUnitStrides,
        zaxpyvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_zaxpyv_zen_int_avx512),               // kernel address
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                            , 'c'                                       // conjx
#endif
            ),
            ::testing::Values(gtint_t(13)),                             // n, size of the vector
            ::testing::Values(gtint_t(5)),                              // stride size for x
            ::testing::Values(gtint_t(3)),                              // stride size for y
            ::testing::Values(dcomplex{1.0, 0.0}, dcomplex{-1.0, 0.0},
                              dcomplex{0.0, 1.0}, dcomplex{0.0, -1.0},
                              dcomplex{0.0, -3.3}, dcomplex{4.3,-2.1},
                              dcomplex{0.0, 0.0}),                      // alpha
            ::testing::Values(false, true)                              // is_memory_test
        ),
        (::axpyvUKRPrint<dcomplex, zaxpyv_ker_ft>())
    );

#endif
