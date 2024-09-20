/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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
#include "level1/axpbyv/test_axpbyv.h"

class zaxpbyvEVT :
        public ::testing::TestWithParam<std::tuple<char,          // transpose
                                                   gtint_t,       // n, size of the vector
                                                   gtint_t,       // incx
                                                   gtint_t,       // incy
                                                   gtint_t,       // xi, index for exval in x
                                                   dcomplex,      // xexval
                                                   gtint_t,       // yi, index for exval in y
                                                   dcomplex,      // yexval
                                                   dcomplex,      // alpha
                                                   dcomplex>> {}; // beta

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(zaxpbyvEVT);

// Tests using random integers as vector elements.
TEST_P( zaxpbyvEVT, API )
{
    using T = dcomplex;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // denotes whether x or conj(x) will be added to y:
    char conj_x = std::get<0>(GetParam());
    // vector length:
    gtint_t n = std::get<1>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<2>(GetParam());
    // stride size for y:
    gtint_t incy = std::get<3>(GetParam());
    // index for exval in x
    gtint_t xi = std::get<4>(GetParam());
    // exval for x
    T xexval = std::get<5>(GetParam());
    // index for exval in y
    gtint_t yj = std::get<6>(GetParam());
    // exval for x
    T yexval = std::get<7>(GetParam());
    // alpha
    T alpha = std::get<8>(GetParam());
    // beta
    T beta = std::get<9>(GetParam());

    // Set the threshold for the errors:
    // Check gtestsuite axpbyv.h (no netlib version) for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    // NOTE : Every mul for complex types involves 3 ops(2 muls + 1 add)
    double thresh;
    double adj = 3;
    if (n == 0)
        thresh = 0.0;
    else if (beta == testinghelpers::ZERO<T>())
    {
        // Like SETV or COPYV(no ops)
        if (alpha == testinghelpers::ZERO<T>() || alpha == testinghelpers::ONE<T>())
            thresh = 0.0;
        // Like SCAL2V(1 mul)
        else
            thresh = (1 * adj) * testinghelpers::getEpsilon<T>();
    }
    else if (beta == testinghelpers::ONE<T>())
    {
        // Like ERS(no ops)
        if (alpha == testinghelpers::ZERO<T>())
            thresh = 0.0;
        // Like ADDV(1 add)
        else if (alpha == testinghelpers::ONE<T>())
            thresh = testinghelpers::getEpsilon<T>();
        // Like AXPYV(1 mul and 1 add)
        else
            thresh = (1 * adj + 1) * testinghelpers::getEpsilon<T>();
    }
    else
    {
        // Like SCALV(1 mul)
        if (alpha == testinghelpers::ZERO<T>())
            thresh = (1 * adj) * testinghelpers::getEpsilon<T>();
        // Like AXPBYV(2 muls and 1 add)
        else
            thresh = (2 * adj + 1) * testinghelpers::getEpsilon<T>();
    }

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_axpbyv<T>(conj_x, n, incx, incy, alpha, beta, xi, xexval,
                   yj, yexval, thresh);
}

#if defined(REF_IS_NETLIB)
static double NaN = std::numeric_limits<double>::quiet_NaN();
static double Inf = std::numeric_limits<double>::infinity();

/*
    The code structure for bli_zaxpbyv_zen_int( ... ) is as follows :
    For unit strides :
        Main loop    :  In blocks of 8 --> L8
        Fringe loops :  In blocks of 6 --> L6
                        In blocks of 4 --> L4
                        In blocks of 2 --> L2

    For non-unit strides : A single loop, to process element wise.
    NOTE : Any size, requiring the fringe case of 1 with unit stride falls to
           the non-unit stride loop and executes it once for just the last element.

    The sizes chosen are as follows :
    59 - 7*L8 + L2 + 1(LScalar)
    60 - 7*L8 + L4
    62 - 7*L8 + L6

    For size 59  :  7*L8 + L2 + 1(LScalar)
    Indices are  :  0, 55 -> In L8
                    57    -> In L2
                    58    -> In LScalar

    For size 60  :  7*L8 + L4
    Indices are  :  0, 55 -> In L8
                    59    -> In L4

    For size 62  :  7*L8 + L6
    Indices are  :  0, 55 -> In L8
                    61    -> In L6

    The alpha and beta values are such that they check for compliance against possible
    optimizations that might have been done.

    P.S : Some test cases also check whether NaN has to be induced in the computation
          such as 0.0 * { {NaN, 0}, {+Inf, 0}, {-Inf, 0}, ... }, and a few more.
*/

// Exception value testing(on X vector alone) with unit strides
INSTANTIATE_TEST_SUITE_P(
    vecX_unitStrides,
    zaxpbyvEVT,
    ::testing::Combine(
        ::testing::Values('n' // n: use x, c: use conj(x)
#ifdef TEST_BLIS_TYPED
                          ,
                          'c' // this option is BLIS-api specific.
#endif
                          ),
        ::testing::Values(gtint_t(59), gtint_t(60), gtint_t(62)),       // n, size of vectors with unit-stride
        ::testing::Values(gtint_t(1)),                                  // stride size for x
        ::testing::Values(gtint_t(1)),                                  // stride size for y
        ::testing::Values(gtint_t(0), gtint_t(55), gtint_t(57),
                          gtint_t(58), gtint_t(59), gtint_t(61)),       // indices to set exception values on x
        ::testing::Values(dcomplex{NaN, 0.0}, dcomplex{-Inf, 0.0},
                          dcomplex{0.0, Inf}, dcomplex{-2.3, NaN},
                          dcomplex{4.5, -Inf}, dcomplex{NaN, Inf}),     // exception values to set on x
        ::testing::Values(gtint_t(0)),                                  // dummy index on y
        ::testing::Values(dcomplex{0.0, 0.0}),                          // dummy value on y
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0},
                          dcomplex{-1.0, 0.0}, dcomplex{0.0, 1.0},
                          dcomplex{0.0, -1.0}, dcomplex{-3.3, 1.7}),    // alpha
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0},
                          dcomplex{-1.0, 0.0}, dcomplex{0.0, 1.0},
                          dcomplex{0.0, -1.0}, dcomplex{-3.3, 1.7})     // beta
        ),
    ::axpbyvEVTPrint<dcomplex>());

// Exception value testing(on Y vector alone) with unit strides
INSTANTIATE_TEST_SUITE_P(
    vecY_unitStrides,
    zaxpbyvEVT,
    ::testing::Combine(
        ::testing::Values('n' // n: use x, c: use conj(x)
#ifdef TEST_BLIS_TYPED
                          ,
                          'c' // this option is BLIS-api specific.
#endif
                          ),
        ::testing::Values(gtint_t(59), gtint_t(60), gtint_t(62)),       // n, size of vectors with unit-stride
        ::testing::Values(gtint_t(1)),                                  // stride size for x
        ::testing::Values(gtint_t(1)),                                  // stride size for y
        ::testing::Values(gtint_t(0)),                                  // dummy index on x
        ::testing::Values(dcomplex{0.0, 0.0}),                          // dummy value on x
        ::testing::Values(gtint_t(0), gtint_t(55), gtint_t(57),
                          gtint_t(58), gtint_t(59), gtint_t(61)),       // indices to set exception values on y
        ::testing::Values(dcomplex{NaN, 0.0}, dcomplex{-Inf, 0.0},
                          dcomplex{0.0, Inf}, dcomplex{-2.3, NaN},
                          dcomplex{4.5, -Inf}, dcomplex{NaN, Inf}),     // exception values to set on y
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0},
                          dcomplex{-1.0, 0.0}, dcomplex{0.0, 1.0},
                          dcomplex{0.0, -1.0}, dcomplex{-3.3, 1.7}),    // alpha
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0},
                          dcomplex{-1.0, 0.0}, dcomplex{0.0, 1.0},
                          dcomplex{0.0, -1.0}, dcomplex{-3.3, 1.7})     // beta
        ),
    ::axpbyvEVTPrint<dcomplex>());

// Exception value testing(on X and Y vectors) with unit strides
INSTANTIATE_TEST_SUITE_P(
    vecXY_unitStrides,
    zaxpbyvEVT,
    ::testing::Combine(
        ::testing::Values('n' // n: use x, c: use conj(x)
#ifdef TEST_BLIS_TYPED
                          ,
                          'c' // this option is BLIS-api specific.
#endif
                          ),
        ::testing::Values(gtint_t(59), gtint_t(60), gtint_t(62)),       // n, size of vectors with unit-stride
        ::testing::Values(gtint_t(1)),                                  // stride size for x
        ::testing::Values(gtint_t(1)),                                  // stride size for y
        ::testing::Values(gtint_t(0), gtint_t(55), gtint_t(57),
                          gtint_t(58), gtint_t(59), gtint_t(61)),       // indices to set exception values on x
        ::testing::Values(dcomplex{NaN, 0.0}, dcomplex{-Inf, 0.0},
                          dcomplex{0.0, Inf}, dcomplex{-2.3, NaN},
                          dcomplex{4.5, -Inf}, dcomplex{NaN, Inf}),     // exception values to set on x
        ::testing::Values(gtint_t(0), gtint_t(55), gtint_t(57),
                          gtint_t(58), gtint_t(59), gtint_t(61)),       // indices to set exception values on y
        ::testing::Values(dcomplex{NaN, 0.0}, dcomplex{-Inf, 0.0},
                          dcomplex{0.0, Inf}, dcomplex{-2.3, NaN},
                          dcomplex{4.5, -Inf}, dcomplex{NaN, Inf}),     // exception values to set on y
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0},
                          dcomplex{-1.0, 0.0}, dcomplex{0.0, 1.0},
                          dcomplex{0.0, -1.0}, dcomplex{-3.3, 1.7}),    // alpha
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0},
                          dcomplex{-1.0, 0.0}, dcomplex{0.0, 1.0},
                          dcomplex{0.0, -1.0}, dcomplex{-3.3, 1.7})     // beta
        ),
    ::axpbyvEVTPrint<dcomplex>());

// Exception value testing(on vectors) with non-unit strides
// We have to test a single scalar loop. The indices are such
// that we cover _vecX_, _vecY_ and _vecXY_ cases together.
INSTANTIATE_TEST_SUITE_P(
    vecXY_nonUnitStrides,
    zaxpbyvEVT,
    ::testing::Combine(
        ::testing::Values('n' // n: use x, c: use conj(x)
#ifdef TEST_BLIS_TYPED
                          ,
                          'c' // this option is BLIS-api specific.
#endif
                          ),
        ::testing::Values(gtint_t(50)),                                 // n, size of vectors with non-unit strides
        ::testing::Values(gtint_t(3)),                                  // stride size for x
        ::testing::Values(gtint_t(5)),                                  // stride size for y
        ::testing::Values(gtint_t(1), gtint_t(27), gtint_t(49)),        // indices to set exception values on x
        ::testing::Values(dcomplex{NaN, 0.0}, dcomplex{-Inf, 0.0},
                          dcomplex{0.0, Inf}, dcomplex{-2.3, NaN},
                          dcomplex{4.5, -Inf}, dcomplex{NaN, Inf},
                          dcomplex{2.3, -3.5}),                         // exception values to set on x
        ::testing::Values(gtint_t(0), gtint_t(26), gtint_t(49)),        // indices to set exception values on y
        ::testing::Values(dcomplex{NaN, 0.0}, dcomplex{-Inf, 0.0},
                          dcomplex{0.0, Inf}, dcomplex{-2.3, NaN},
                          dcomplex{4.5, -Inf}, dcomplex{NaN, Inf},
                          dcomplex{2.3, -3.5}),                         // exception values to set on y
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0},
                          dcomplex{-1.0, 0.0}, dcomplex{0.0, 1.0},
                          dcomplex{0.0, -1.0}, dcomplex{-3.3, 1.7}),    // alpha
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0},
                          dcomplex{-1.0, 0.0}, dcomplex{0.0, 1.0},
                          dcomplex{0.0, -1.0}, dcomplex{-3.3, 1.7})     // beta
        ),
    ::axpbyvEVTPrint<dcomplex>());

/*
    Exception value testing on alpha and beta :
    Alpha values are set to Nan, +Inf or -Inf. A dummy
    value of 0.0 is induced in X and Y vectors, to further
    verify the propagation.
*/
INSTANTIATE_TEST_SUITE_P(
    alphaBeta_unitStrides,
    zaxpbyvEVT,
    ::testing::Combine(
        ::testing::Values('n' // n: use x, c: use conj(x)
#ifdef TEST_BLIS_TYPED
                          ,
                          'c' // this option is BLIS-api specific.
#endif
                          ),
        ::testing::Values(gtint_t(59), gtint_t(60), gtint_t(62)),       // n, size of vectors with unit-stride
        ::testing::Values(gtint_t(1)),                                  // stride size for x
        ::testing::Values(gtint_t(1)),                                  // stride size for y
        ::testing::Values(gtint_t(0)),                                  // indices to set zero on x
        ::testing::Values(dcomplex{0.0, 0.0}),
        ::testing::Values(gtint_t(0)),                                  // indices to set zero on y
        ::testing::Values(dcomplex{0.0, 0.0}),
        ::testing::Values(dcomplex{NaN, 0.0}, dcomplex{-Inf, 0.0},
                          dcomplex{0.0, Inf}, dcomplex{-2.3, NaN},
                          dcomplex{4.5, -Inf}, dcomplex{NaN, Inf},
                          dcomplex{2.3, -3.7}),                         // alpha
        ::testing::Values(dcomplex{NaN, 0.0}, dcomplex{-Inf, 0.0},
                          dcomplex{0.0, Inf}, dcomplex{-2.3, NaN},
                          dcomplex{4.5, -Inf}, dcomplex{NaN, Inf},
                          dcomplex{2.3, -3.7})                          // beta
        ),
    ::axpbyvEVTPrint<dcomplex>());

// Exception value testing(on alpha) with non-unit strided vectors
INSTANTIATE_TEST_SUITE_P(
    alphaBeta_nonUnitStrides,
    zaxpbyvEVT,
    ::testing::Combine(
        ::testing::Values('n' // n: use x, c: use conj(x)
#ifdef TEST_BLIS_TYPED
                          ,
                          'c' // this option is BLIS-api specific.
#endif
                          ),
        ::testing::Values(gtint_t(50)),                                 // n, size of vectors with non-unit strides
        ::testing::Values(gtint_t(3)),                                  // stride size for x
        ::testing::Values(gtint_t(5)),                                  // stride size for y
        ::testing::Values(gtint_t(0), gtint_t(25)),                     // indices to set zero on x
        ::testing::Values(dcomplex{0.0, 0.0}),
        ::testing::Values(gtint_t(0), gtint_t(40)),                     // indices to set zero on y
        ::testing::Values(dcomplex{0.0, 0.0}),
        ::testing::Values(dcomplex{NaN, 0.0}, dcomplex{-Inf, 0.0},
                          dcomplex{0.0, Inf}, dcomplex{-2.3, NaN},
                          dcomplex{4.5, -Inf}, dcomplex{NaN, Inf},
                          dcomplex{2.3, -3.7}),                         // alpha
        ::testing::Values(dcomplex{NaN, 0.0}, dcomplex{-Inf, 0.0},
                          dcomplex{0.0, Inf}, dcomplex{-2.3, NaN},
                          dcomplex{4.5, -Inf}, dcomplex{NaN, Inf},
                          dcomplex{2.3, -3.7})                          // beta
        ),
    ::axpbyvEVTPrint<dcomplex>());
#endif
