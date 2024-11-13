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
#include "common/blis_version_defs.h"

class saxpyvGeneric :
        public ::testing::TestWithParam<std::tuple<saxpyv_ker_ft,   // Function pointer type for zaxpyv kernels
                                                   char,            // conj_x
                                                   gtint_t,         // n
                                                   gtint_t,         // incx
                                                   gtint_t,         // incy
                                                   float,           // alpha
                                                   bool>> {};       // is_memory_test

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(saxpyvGeneric);

// Defining the testsuite to check the accuracy of saxpyv micro-kernels
TEST_P( saxpyvGeneric, UKR )
{
    using T = float;

    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------

    // Assign the kernel address to the function pointer
    saxpyv_ker_ft ukr_fp = std::get<0>(GetParam());
    // denotes whether x or conj(x) will be added to y
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

    // Set the threshold for the errors
    double threshold = 2 * testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_axpyv_ukr<T, saxpyv_ker_ft>( ukr_fp, conj_x, n, incx, incy, alpha, threshold, is_memory_test );
}

#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)
/*
    Unit testing for functionality of bli_saxpyv_zen_int10 kernel.
    The code structure for bli_saxpyv_zen_int10( ... ) is as follows :
    For unit strides :
        Main loop    :  In blocks of 120 --> L120
        Fringe loops :  In blocks of 80  --> L80
                        In blocks of 40  --> L40
                        In blocks of 32  --> L32
                        In blocks of 16  --> L16
                        In blocks of 8   --> L8
                        Element-wise loop --> LScalar

    For non-unit strides : A single loop, to process element wise.
*/

#ifdef K_bli_saxpyv_zen_int10
INSTANTIATE_TEST_SUITE_P(
        bli_saxpyv_zen_int10_unitStrides,
        saxpyvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_saxpyv_zen_int10),   // kernel address
            ::testing::Values('n'),                    // use x, not conj(x) (since it is real)
            ::testing::Values(// Testing the loops standalone
                              gtint_t(120),            // size n, for L120
                              gtint_t(80),             // L80
                              gtint_t(40),             // L40
                              gtint_t(32),             // L32
                              gtint_t(16),             // L16
                              gtint_t(8),              // L8
                              gtint_t(7),              // LScalar
                              gtint_t(240),            // 2*L120
                              gtint_t(320),            // 2*L120 + L80
                              gtint_t(312),            // 2*L120 + L40 + L32
                              gtint_t(271)),           // 2*L120 + L16 + L8 + LScalar
            ::testing::Values(gtint_t(1)),             // stride size for x
            ::testing::Values(gtint_t(1)),             // stride size for y
            ::testing::Values(float(1.0), float(-1.0),
                              float(2.3), float(-4.5),
                              float(0.0)),             // alpha
            ::testing::Values(false, true)             // is_memory_test
        ),
        (::axpyvUKRPrint<float, saxpyv_ker_ft>())
    );
#endif

#ifdef K_bli_saxpyv_zen_int10
INSTANTIATE_TEST_SUITE_P(
        bli_saxpyv_zen_int10_nonUnitStrides,
        saxpyvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_saxpyv_zen_int10),    // kernel address
            ::testing::Values('n'),                     // use x, not conj(x) (since it is real)
            ::testing::Values(// Testing the loops standalone
                              gtint_t(7),               // size n, for LScalar
                              gtint_t(15)),
            ::testing::Values(gtint_t(3), gtint_t(5)),  // stride size for x
            ::testing::Values(gtint_t(3), gtint_t(5)),  // stride size for y
            ::testing::Values(float(1.0), float(-1.0),
                              float(2.3), float(-4.5),
                              float(0.0)),              // alpha
            ::testing::Values(false, true)              // is_memory_test
        ),
        (::axpyvUKRPrint<float, saxpyv_ker_ft>())
    );
#endif

/*
    Unit testing for functionality of bli_saxpyv_zen_int kernel.
    The code structure for bli_saxpyv_zen_int( ... ) is as follows :
    For unit strides :
        Main loop    :  In blocks of 32 --> L32
                        Element-wise loop --> LScalar

    For non-unit strides : A single loop, to process element wise.
*/

#ifdef K_bli_saxpyv_zen_int
INSTANTIATE_TEST_SUITE_P(
        bli_saxpyv_zen_int_unitStrides,
        saxpyvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_saxpyv_zen_int),     // kernel address
            ::testing::Values('n'),                    // use x, not conj(x) (since it is real)
            ::testing::Values(// Testing the loops standalone
                              gtint_t(32),             // size n, for L32
                              gtint_t(15),             // LScalar
                              gtint_t(79)),            // 2*L32 + LScalar
            ::testing::Values(gtint_t(1)),             // stride size for x
            ::testing::Values(gtint_t(1)),             // stride size for y
            ::testing::Values(float(1.0), float(-1.0),
                              float(2.3), float(-4.5),
                              float(0.0)),             // alpha
            ::testing::Values(false, true)             // is_memory_test
        ),
        (::axpyvUKRPrint<float, saxpyv_ker_ft>())
    );
#endif

#ifdef K_bli_saxpyv_zen_int
INSTANTIATE_TEST_SUITE_P(
        bli_saxpyv_zen_int_nonUnitStrides,
        saxpyvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_saxpyv_zen_int),      // kernel address
            ::testing::Values('n'),                     // use x, not conj(x) (since it is real)
            ::testing::Values(// Testing the loops standalone
                              gtint_t(7),               // size n, for LScalar
                              gtint_t(10)),
            ::testing::Values(gtint_t(3), gtint_t(5)),  // stride size for x
            ::testing::Values(gtint_t(3), gtint_t(5)),  // stride size for y
            ::testing::Values(float(1.0), float(-1.0),
                              float(2.3), float(-4.5),
                              float(0.0)),              // alpha
            ::testing::Values(false, true)              // is_memory_test
        ),
        (::axpyvUKRPrint<float, saxpyv_ker_ft>())
    );
#endif
#endif

#if defined(BLIS_KERNELS_ZEN4) && defined(GTEST_AVX512)
/*
    Unit testing for functionality of bli_saxpyv_zen_int_avx512 kernel.
    The code structure for bli_saxpyv_zen_int_avx512( ... ) is as follows :
    For unit strides :
        Main loop    :  In blocks of 128 --> L128
        Fringe loops :  In blocks of 64  --> L64
                        In blocks of 32  --> L32
                        In blocks of 16  --> L16
                        In blocks of 8   --> L8
                        Element-wise loop --> LScalar

    For non-unit strides : A single loop, to process element wise.
*/

#ifdef K_bli_saxpyv_zen_int_avx512
INSTANTIATE_TEST_SUITE_P(
        bli_saxpyv_zen_int_avx512_unitStrides,
        saxpyvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_saxpyv_zen_int_avx512), // kernel address
            ::testing::Values('n'),                        // use x, not conj(x) (since it is real)
            ::testing::Values(// Testing the loops standalone
                              gtint_t(128),                // size n, for L128
                              gtint_t(64),                 // L64
                              gtint_t(32),                 // L32
                              gtint_t(16),                 // L16
                              gtint_t(8),                  // L8
                              gtint_t(7),                  // LScalar
                              gtint_t(383)),               // 2*L128 + L64 + L32 + L16 + L8 + L7
            ::testing::Values(gtint_t(1)),                 // stride size for x
            ::testing::Values(gtint_t(1)),                 // stride size for y
            ::testing::Values(float(1.0), float(-1.0),
                              float(2.3), float(-4.5),
                              float(0.0)),                 // alpha
            ::testing::Values(false, true)                 // is_memory_test
        ),
        (::axpyvUKRPrint<float, saxpyv_ker_ft>())
    );
#endif

#ifdef K_bli_saxpyv_zen_int_avx512
INSTANTIATE_TEST_SUITE_P(
        bli_saxpyv_zen_int_avx512_nonUnitStrides,
        saxpyvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_saxpyv_zen_int_avx512),  // kernel address
            ::testing::Values('n'),                         // use x, not conj(x) (since it is real)
            ::testing::Values(// Testing the loops standalone
                              gtint_t(7),                   // size n, for LScalar
                              gtint_t(15)),
            ::testing::Values(gtint_t(3), gtint_t(5)),      // stride size for x
            ::testing::Values(gtint_t(3), gtint_t(5)),      // stride size for y
            ::testing::Values(float(1.0), float(-1.0),
                              float(2.3), float(-4.5),
                              float(0.0)),                  // alpha
            ::testing::Values(false, true)                  // is_memory_test
        ),
        (::axpyvUKRPrint<float, saxpyv_ker_ft>())
    );
#endif
#endif
