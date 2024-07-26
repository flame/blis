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
#include "test_axpbyv.h"

class saxpbyvEVT :
        public ::testing::TestWithParam<std::tuple<char,         // transpose
                                                   gtint_t,      // n, size of the vector
                                                   gtint_t,      // incx
                                                   gtint_t,      // incy
                                                   gtint_t,      // xi, index for exval in x
                                                   float,        // xexval
                                                   gtint_t,      // yi, index for exval in y
                                                   float,        // yexval
                                                   float,        // alpha
                                                   float>> {};   // beta

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(saxpbyvEVT);

// Tests using random values as vector elements,
// with exception values on the passed indices.
TEST_P( saxpbyvEVT, API )
{
    using T = float;
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
    double thresh;
    if (n == 0)
        thresh = 0.0;
    else if (beta == testinghelpers::ZERO<T>())
    {
        // Like SETV or COPYV(no ops)
        if (alpha == testinghelpers::ZERO<T>() || alpha == testinghelpers::ONE<T>())
            thresh = 0.0;
        // Like SCAL2V(1 mul)
        else
            thresh = testinghelpers::getEpsilon<T>();
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
            thresh = 2 * testinghelpers::getEpsilon<T>();
    }
    else
    {
        // Like SCALV(1 mul)
        if (alpha == testinghelpers::ZERO<T>())
            thresh = testinghelpers::getEpsilon<T>();
        // Like AXPBYV(2 muls and 1 add)
        else
            thresh = 3 * testinghelpers::getEpsilon<T>();
    }

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_axpbyv<T>(conj_x, n, incx, incy, alpha, beta, xi, xexval,
                   yj, yexval, thresh);
}

#if defined(REF_IS_NETLIB)
static float NaN = std::numeric_limits<float>::quiet_NaN();
static float Inf = std::numeric_limits<float>::infinity();

/*
    Exception value testing on vectors :
    DAXPBY currently uses the bli_saxpbyv_zen_int10( ... ) kernel for computation.
    The size and indices given in the instantiator are to ensure code coverage inside
    the kernel, and to verify the compliance accordingly.

    Kernel structure :
    Main loop    :  In blocks of 80 --> L00
    Fringe loops :  In blocks of 20 --> L40
                    In blocks of 32 --> L32
                    In blocks of 16 --> L16
                    In blocks of 8 --> L8
                    Element-wise loop --> LScalar

    For size 231 :  L80*2 + L40 + L16 + L8 + 7(LScalar)
    Indices are  :  0, 159 -> In L80
                    199    -> In L40
                    215    -> In L16
                    223    -> In L8
                    230    -> In LScalar

    For size 232 :  L80*2 + L40 + L32
    Indices are  :  0, 159 -> In L80
                    199    -> In L40
                    215    -> In L32r

    The alpha and beta values are such that they check for compliance against possible
    optimizations that might have been done.

    P.S : Some test cases also check whether NaN has to be induced in the computation
          as a result of 0.0 * { NaN, +Inf, -Inf }.
*/
// Exception value testing(on X vector alone) with unit strides
INSTANTIATE_TEST_SUITE_P(
    vecX_unitStrides,
    saxpbyvEVT,
    ::testing::Combine(
        ::testing::Values('n'),                                                     // use conjx as n for real types
        ::testing::Values(gtint_t(231), gtint_t(232)),                              // n, size of vectors with unit-stride
        ::testing::Values(gtint_t(1)),                                              // stride size for x
        ::testing::Values(gtint_t(1)),                                              // stride size for y
        ::testing::Values(gtint_t(0), gtint_t(159), gtint_t(199),
                          gtint_t(215), gtint_t(223), gtint_t(230)),                // indices to set exception values on x
        ::testing::Values(NaN, -Inf, Inf),                                          // exception values to set on x
        ::testing::Values(gtint_t(0)),                                              // dummy index on y
        ::testing::Values(float(0.0)),                                              // dummy value on y
        ::testing::Values(float(0.0), float(1.0), float(-1.0), float(-3.3)),        // alpha
        ::testing::Values(float(0.0), float(1.0), float(-1.0), float(4.5))          // beta
        ),
    ::axpbyvEVTPrint<float>());

// Exception value testing(on Y vector alone) with unit strides
INSTANTIATE_TEST_SUITE_P(
    vecY_unitStrides,
    saxpbyvEVT,
    ::testing::Combine(
        ::testing::Values('n'),                                                     // use conjx as n for real types
        ::testing::Values(gtint_t(231), gtint_t(232)),                              // n, size of vectors with unit-stride
        ::testing::Values(gtint_t(1)),                                              // stride size for x
        ::testing::Values(gtint_t(1)),                                              // stride size for y
        ::testing::Values(gtint_t(0)),                                              // dummy index on x
        ::testing::Values(float(0.0)),                                             // dummy value on x
        ::testing::Values(gtint_t(0), gtint_t(159), gtint_t(199),
                          gtint_t(215), gtint_t(223), gtint_t(230)),                // indices to set exception values on y
        ::testing::Values(NaN, -Inf, Inf),                                          // exception values to set on y
        ::testing::Values(float(0.0), float(1.0), float(-1.0), float(-3.3)),        // alpha
        ::testing::Values(float(0.0), float(1.0), float(-1.0), float(4.5))          // beta
        ),
    ::axpbyvEVTPrint<float>());

// Exception value testing(on X and Y vectors) with unit strides
INSTANTIATE_TEST_SUITE_P(
    vecXY_unitStrides,
    saxpbyvEVT,
    ::testing::Combine(
        ::testing::Values('n'),                                                     // use conjx as n for real types
        ::testing::Values(gtint_t(231), gtint_t(232)),                              // n, size of vectors with unit-stride
        ::testing::Values(gtint_t(1)),                                              // stride size for x
        ::testing::Values(gtint_t(1)),                                              // stride size for y
        ::testing::Values(gtint_t(0), gtint_t(159), gtint_t(199),
                          gtint_t(215), gtint_t(223), gtint_t(230)),                // indices to set exception values on x
        ::testing::Values(NaN, -Inf, Inf),                                          // exception values to set on x
        ::testing::Values(gtint_t(0), gtint_t(159), gtint_t(199),
                          gtint_t(215), gtint_t(223), gtint_t(230)),                // indices to set exception values on y
        ::testing::Values(NaN, -Inf, Inf),                                          // exception values to set on y
        ::testing::Values(float(0.0), float(1.0), float(-1.0), float(-3.3)),        // alpha
        ::testing::Values(float(0.0), float(1.0), float(-1.0), float(4.5))          // beta
        ),
    ::axpbyvEVTPrint<float>());

// Exception value testing(on vectors) with non-unit strides
// We have to test a single scalar loop. The indices are such
// that we cover _vecX_, _vecY_ and _vecXY_ cases together.
INSTANTIATE_TEST_SUITE_P(
    vec_nonUnitStrides,
    saxpbyvEVT,
    ::testing::Combine(
        ::testing::Values('n'),                                                     // use conjx as n for real types
        ::testing::Values(gtint_t(50)),                                             // n, size of vectors with non-unit strides
        ::testing::Values(gtint_t(3)),                                              // stride size for x
        ::testing::Values(gtint_t(5)),                                              // stride size for y
        ::testing::Values(gtint_t(1), gtint_t(27), gtint_t(49)),                    // indices to set exception values on x
        ::testing::Values(NaN, -Inf, Inf, 2.9),                                     // exception values to set on x
        ::testing::Values(gtint_t(0), gtint_t(26), gtint_t(49)),                    // indices to set exception values on y
        ::testing::Values(NaN, -Inf, Inf, -1.5),                                    // exception values to set on y
        ::testing::Values(float(0.0), float(1.0), float(-1.0), float(-3.3)),        // alpha
        ::testing::Values(float(0.0), float(1.0), float(-1.0), float(4.5))          // beta
        ),
    ::axpbyvEVTPrint<float>());

/*
    Exception value testing on alpha and/or beta :
    Alpha and/or beta values are set to Nan, +Inf or -Inf.
    Also, a normal value is given to alpha and beta to check
    for combinations where only X or Y involve scaling by an
    exception valued scalar. A dummy value of 0.0 is induced
    in X and Y vectors, to further verify the propagation.

    The size for the instantiators is chosen such that
    code coverage is ensured in the respective kernel.
*/
// Exception value testing(on alpha/beta) with unit strided vectors
INSTANTIATE_TEST_SUITE_P(
    alphaBeta_unitStrides,
    saxpbyvEVT,
    ::testing::Combine(
        ::testing::Values('n'),                         // use conjx as n for real types
        ::testing::Values(gtint_t(231), gtint_t(232)),  // n, size of vectors with unit-stride
        ::testing::Values(gtint_t(1)),                  // stride size for x
        ::testing::Values(gtint_t(1)),                  // stride size for y
        ::testing::Values(gtint_t(0)),                  // indices to set zero on x
        ::testing::Values(float(0.0)),
        ::testing::Values(gtint_t(0)),                  // indices to set zero on y
        ::testing::Values(float(0.0)),
        ::testing::Values(NaN, -Inf, Inf, 2.3),         // alpha
        ::testing::Values(NaN, -Inf, Inf, -1.9)         // beta
        ),
    ::axpbyvEVTPrint<float>());

// Exception value testing(on alpha/beta) with non-unit strided vectors
INSTANTIATE_TEST_SUITE_P(
    alphaBeta_nonUnitStrides,
    saxpbyvEVT,
    ::testing::Combine(
        ::testing::Values('n'),                         // use conjx as n for real types
        ::testing::Values(gtint_t(50)),                 // n, size of vector with non-unit strides
        ::testing::Values(gtint_t(3)),                  // stride size for x
        ::testing::Values(gtint_t(5)),                  // stride size for y
        ::testing::Values(gtint_t(1), gtint_t(25)),     // indices to set zero on x
        ::testing::Values(float(0.0)),
        ::testing::Values(gtint_t(0), gtint_t(40)),     // indices to set zero on y
        ::testing::Values(float(0.0)),
        ::testing::Values(NaN, -Inf, Inf, 2.3),         // alpha
        ::testing::Values(NaN, -Inf, Inf, -1.9)         // beta
        ),
    ::axpbyvEVTPrint<float>());
#endif
