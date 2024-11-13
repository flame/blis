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
#include "common/blis_version_defs.h"

class daxpbyvGeneric :
        public ::testing::TestWithParam<std::tuple<daxpbyv_ker_ft,  // Function pointer type for daxpbyv kernels
                                                   char,            // conjx
                                                   gtint_t,         // n
                                                   gtint_t,         // incx
                                                   gtint_t,         // incy
                                                   double,          // alpha
                                                   double,          // beta
                                                   bool>> {};       // is_memory_test

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(daxpbyvGeneric);

// Tests using random integers as vector elements.
TEST_P( daxpbyvGeneric, UKR )
{
    using T = double;

    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------

    // Assign the kernel address to the function pointer
    daxpbyv_ker_ft ukr_fp = std::get<0>(GetParam());
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
    test_axpbyv_ukr<T, daxpbyv_ker_ft>( ukr_fp, conj_x, n, incx, incy, alpha, beta, thresh, is_memory_test );
}

#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)
/*
    Unit testing for functionality of bli_daxpbyv_zen_int10 kernel.
    The code structure for bli_daxpbyv_zen_int10( ... ) is as follows :
    For unit strides :
        Main loop    :  In blocks of 40 --> L40
        Fringe loops :  In blocks of 20 --> L20
                        In blocks of 8  --> L8
                        In blocks of 4  --> L4
                        Element-wise loop --> LScalar

    For non-unit strides : A single loop, to process element wise.
*/

// Unit testing with unit stride, across all loops.
#ifdef K_bli_daxpbyv_zen_int10
INSTANTIATE_TEST_SUITE_P(
        bli_daxpbyv_zen_int10_unitStrides,
        daxpbyvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_daxpbyv_zen_int10),      // kernel address
            ::testing::Values('n'),                        // use x, not conj(x) (since it is real)
            ::testing::Values(// Testing the loops standalone
                              gtint_t(40),                 // size n, for L40
                              gtint_t(20),                 // L20
                              gtint_t(8),                  // L8
                              gtint_t(4),                  // L4
                              gtint_t(2),                  // LScalar
                              // Testing the loops with combination
                              gtint_t(120),                // 3*L40
                              gtint_t(140),                // 3*L40 + L20
                              gtint_t(148),                // 3*L40 + L20 + L8
                              gtint_t(152),                // 3*L40 + L20 + L8 + L4
                              gtint_t(155)),               // 3*L40 + L20 + L8 + L4 + 3(LScalar)
            ::testing::Values(gtint_t(1)),                 // stride size for x
            ::testing::Values(gtint_t(1)),                 // stride size for y
            ::testing::Values(double(1.0), double(-1.0),
                              double(2.2), double(-4.1),
                              double(0.0)),                // alpha
            ::testing::Values(double(1.0), double(-1.0),
                              double(2.2), double(-4.1),
                              double(0.0)),                // beta
            ::testing::Values(false, true)                 // is_memory_test
        ),
        ((::axpbyvMemUKRPrint<double, daxpbyv_ker_ft>()))
    );
#endif

// Unit testing for non unit strides
#ifdef K_bli_daxpbyv_zen_int10
INSTANTIATE_TEST_SUITE_P(
        bli_daxpbyv_zen_int10_nonUnitStrides,
        daxpbyvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_daxpbyv_zen_int10),      // kernel address
            ::testing::Values('n'),                        // use x, not conj(x) (since it is real)
            ::testing::Values(gtint_t(10),                 // n, size of the vector
                              gtint_t(25)),
            ::testing::Values(gtint_t(5)),                 // stride size for x
            ::testing::Values(gtint_t(3)),                 // stride size for y
            ::testing::Values(double(1.0), double(-1.0),
                              double(2.2), double(-4.1),
                              double(0.0)),                // alpha
            ::testing::Values(double(1.0), double(-1.0),
                              double(2.2), double(-4.1),
                              double(0.0)),                // beta
            ::testing::Values(false, true)                 // is_memory_test
        ),
        (::axpbyvMemUKRPrint<double, daxpbyv_ker_ft>())
    );
#endif

/*
    Unit testing for functionality of bli_daxpbyv_zen_int kernel.
    The code structure for bli_daxpbyv_zen_int( ... ) is as follows :
    For unit strides :
        Main loop    :  In blocks of 16 --> L16
                        Element-wise loop --> LScalar

    For non-unit strides : A single loop, to process element wise.
*/
// Unit testing with Unit Strides, across all loops.
#ifdef K_bli_daxpbyv_zen_int
INSTANTIATE_TEST_SUITE_P(
        bli_daxpbyv_zen_int_unitStrides,
        daxpbyvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_daxpbyv_zen_int),        // kernel address
            ::testing::Values('n'),                        // use x, not conj(x) (since it is real)
            ::testing::Values(gtint_t(16),                 // size n, for L16
                              gtint_t(48),                 // 3*L16
                              gtint_t(57)),                // 3*L16 + 9(LScalar)
            ::testing::Values(gtint_t(1)),                 // stride size for x
            ::testing::Values(gtint_t(1)),                 // stride size for y
            ::testing::Values(double(1.0), double(-1.0),
                              double(2.2), double(-4.1),
                              double(0.0)),                // alpha
            ::testing::Values(double(1.0), double(-1.0),
                              double(2.2), double(-4.1),
                              double(0.0)),                // beta
            ::testing::Values(false, true)                 // is_memory_test
        ),
        (::axpbyvMemUKRPrint<double, daxpbyv_ker_ft>())
    );
#endif

// Unit testing for Non-Unit Stride
#ifdef K_bli_daxpbyv_zen_int
INSTANTIATE_TEST_SUITE_P(
        bli_daxpbyv_zen_int_nonUnitStrides,
        daxpbyvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_daxpbyv_zen_int),        // kernel address
            ::testing::Values('n'),                        // use x, not conj(x) (since it is real)
            ::testing::Values(gtint_t(10),                 // n, size of the vector
                              gtint_t(25)),
            ::testing::Values(gtint_t(5)),                 // stride size for x
            ::testing::Values(gtint_t(3)),                 // stride size for y
            ::testing::Values(double(1.0), double(-1.0),
                              double(2.2), double(-4.1),
                              double(0.0)),                // alpha
            ::testing::Values(double(1.0), double(-1.0),
                              double(2.2), double(-4.1),
                              double(0.0)),                // beta
            ::testing::Values(false, true)                 // is_memory_test
        ),
        (::axpbyvMemUKRPrint<double, daxpbyv_ker_ft>())
    );
#endif
#endif

#if defined(BLIS_KERNELS_ZEN4) && defined(GTEST_AVX512)
/*
    Unit testing for functionality of bli_daxpbyv_zen_int_avx512 kernel.
    The code structure for bli_daxpbyv_zen_int_avx512( ... ) is as follows :
    For unit strides :
        Main loop    :  In blocks of 64  --> L64
        Fringe loops :  In blocks of 32  --> L32
                        In blocks of 16  --> L16
                        In blocks of 8   --> L8
                        Element-wise loop --> LScalar

    For non-unit strides : A single loop, to process element wise.
*/

// Unit testing with unit stride, across all loops.
#ifdef K_bli_daxpbyv_zen_int_avx512
INSTANTIATE_TEST_SUITE_P(
        bli_daxpbyv_zen_int_avx512_unitStrides,
        daxpbyvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_daxpbyv_zen_int_avx512),      // kernel address
            ::testing::Values('n'),                        // use x, not conj(x) (since it is real)
            ::testing::Values(// Testing the loops standalone
                              gtint_t(64),                 // size n, for L64
                              gtint_t(32),                 // L32
                              gtint_t(16),                 // L16
                              gtint_t(8),                  // L8
                              gtint_t(7),                  // LScalar
                              gtint_t(191)),               // 2*L64 + L32 + L16 + L8 + 7(LScalar)
            ::testing::Values(gtint_t(1)),                 // stride size for x
            ::testing::Values(gtint_t(1)),                 // stride size for y
            ::testing::Values(double(1.0), double(-1.0),
                              double(2.2), double(-4.1),
                              double(0.0)),                // alpha
            ::testing::Values(double(1.0), double(-1.0),
                              double(2.2), double(-4.1),
                              double(0.0)),                // beta
            ::testing::Values(false, true)                 // is_memory_test
        ),
        ((::axpbyvMemUKRPrint<double, daxpbyv_ker_ft>()))
    );
#endif

// Unit testing for non unit strides
#ifdef K_bli_daxpbyv_zen_int_avx512
INSTANTIATE_TEST_SUITE_P(
        bli_daxpbyv_zen_int_avx512_nonUnitStrides,
        daxpbyvGeneric,
        ::testing::Combine(
            ::testing::Values(bli_daxpbyv_zen_int_avx512),      // kernel address
            ::testing::Values('n'),                        // use x, not conj(x) (since it is real)
            ::testing::Values(gtint_t(10),                 // n, size of the vector
                              gtint_t(25)),
            ::testing::Values(gtint_t(5)),                 // stride size for x
            ::testing::Values(gtint_t(3)),                 // stride size for y
            ::testing::Values(double(1.0), double(-1.0),
                              double(2.2), double(-4.1),
                              double(0.0)),                // alpha
            ::testing::Values(double(1.0), double(-1.0),
                              double(2.2), double(-4.1),
                              double(0.0)),                // beta
            ::testing::Values(false, true)                 // is_memory_test
        ),
        (::axpbyvMemUKRPrint<double, daxpbyv_ker_ft>())
    );
#endif
#endif
