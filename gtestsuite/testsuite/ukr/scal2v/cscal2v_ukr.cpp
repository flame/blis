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
#include "test_scal2v_ukr.h"
#include "common/blis_version_defs.h"

class cscal2vGeneric :
        public ::testing::TestWithParam<std::tuple<cscal2v_ker_ft,  // Function pointer for cscal2v kernels
                                                   char,            // conjx
                                                   gtint_t,         // n
                                                   gtint_t,         // incx
                                                   gtint_t,         // incy
                                                   scomplex,        // alpha
                                                   bool>> {};       // is_memory_test
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(cscal2vGeneric);

// Tests using random integers as vector elements.
TEST_P( cscal2vGeneric, UKR )
{
    using T = scomplex;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // denotes the kernel to be tested:
    cscal2v_ker_ft ukr = std::get<0>(GetParam());
    // denotes whether alpha or conjx will be used:
    char conjx = std::get<1>(GetParam());
    // vector length:
    gtint_t n = std::get<2>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<3>(GetParam());
    // stride size for y:
    gtint_t incy = std::get<4>(GetParam());
    // alpha:
    T alpha = std::get<5>(GetParam());
    // is_memory_test:
    bool is_memory_test = std::get<6>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite scal2v.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    if (n == 0)
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<T>() || alpha == testinghelpers::ONE<T>())
        thresh = 0.0;
    else
        thresh = testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_scal2v_ukr<T, T, cscal2v_ker_ft>( ukr, conjx, n, incx, incy, alpha, thresh, is_memory_test );
}

// ----------------------------------------------
// ----- Begin ZEN1/2/3 (AVX2) Kernel Tests -----
// ----------------------------------------------
#if defined(BLIS_KERNELS_ZEN) && defined(GTEST_AVX2FMA3)
/*
    Unit testing for functionality of bli_cscal2v_zen_int kernel.
    The code structure for bli_cscal2v_zen_int( ... ) is as follows :
    For unit strides :
        Main loop    :  In blocks of 16  --> L16
        Fringe loops :  In blocks of 8   --> L8
                        In blocks of 4   --> L4
                        Element-wise loop --> LScalar

    For non-unit strides : A single loop, to process element wise.
*/
#ifdef K_bli_cscal2v_zen_int
INSTANTIATE_TEST_SUITE_P(
        bli_cscal2v_zen_int_unitPositiveStride,
        cscal2vGeneric,
        ::testing::Combine(
            ::testing::Values(bli_cscal2v_zen_int),
            // conjx
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                            , 'c'
#endif
                             ),
            ::testing::Values(// Testing the loops standalone
                              gtint_t(16),                      // size n, for L16
                              gtint_t(8),                       // L8
                              gtint_t(4),                       // L4
                              gtint_t(3),                       // LScalar
                              gtint_t(79)),                     // 4*L16 + L8 + L4 + 3(LScalar)
            ::testing::Values(gtint_t(1)),                      // stride size for x
            ::testing::Values(gtint_t(1)),                      // stride size for y
            ::testing::Values(scomplex{1.0, 0.0}, scomplex{-1.0, 0.0},
                              scomplex{0.0, 1.0}, scomplex{0.0, -1.0},
                              scomplex{0.0, -3.3}, scomplex{4.3,-2.1},
                              scomplex{0.0, 0.0}),              // alpha
            ::testing::Values(false, true)                      // is_memory_test
        ),
        (::scal2vUKRPrint<scomplex,cscal2v_ker_ft>())
    );
#endif

#ifdef K_bli_cscal2v_zen_int
INSTANTIATE_TEST_SUITE_P(
        bli_cscal2v_zen_int_nonUnitPositiveStrides,
        cscal2vGeneric,
        ::testing::Combine(
            ::testing::Values(bli_cscal2v_zen_int),
            // conjx
            ::testing::Values('n'
#ifdef TEST_BLIS_TYPED
                            , 'c'
#endif
                             ),
            ::testing::Values(// Testing the loops standalone
                              gtint_t(7),                           // size n, for LScalar
                              gtint_t(15)),
            ::testing::Values(gtint_t(3), gtint_t(5)),              // stride size for x
            ::testing::Values(gtint_t(2), gtint_t(4)),              // stride size for y
            ::testing::Values(scomplex{1.0, 0.0}, scomplex{-1.0, 0.0},
                              scomplex{0.0, 1.0}, scomplex{0.0, -1.0},
                              scomplex{0.0, -3.3}, scomplex{4.3,-2.1},
                              scomplex{0.0, 0.0}),                  // alpha
            ::testing::Values(false, true)                          // is_memory_test
        ),
        (::scal2vUKRPrint<scomplex,cscal2v_ker_ft>())
    );
#endif
#endif
// ----------------------------------------------
// -----  End ZEN1/2/3 (AVX2) Kernel Tests  -----
// ----------------------------------------------
