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
#include "level1/axpyv/test_axpyv.h"

class zaxpyvEVT :
        public ::testing::TestWithParam<std::tuple<char,          // transpose
                                                   gtint_t,       // n, size of the vector
                                                   gtint_t,       // incx
                                                   gtint_t,       // incy
                                                   gtint_t,       // xi, index for exval in x
                                                   dcomplex,      // xexval
                                                   gtint_t,       // yi, index for exval in y
                                                   dcomplex,      // yexval
                                                   dcomplex>> {}; // alpha

// Tests using random values as vector elements,
// with exception values on the passed indices.
TEST_P( zaxpyvEVT, API )
{
    using T = dcomplex;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // denotes whether x or conj(x) will be added to y:
    char conj_x = std::get<0>(GetParam());
    // vector length
    gtint_t n = std::get<1>(GetParam());
    // stride size for x
    gtint_t incx = std::get<2>(GetParam());
    // stride size for y
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

    // Set the threshold for the errors:
    // Check gtestsuite subv.h (no netlib version) for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    if (n == 0)
        thresh = 0.0;
    else if (alpha == testinghelpers::ZERO<T>())
        thresh = 0.0;
    else if (alpha == testinghelpers::ONE<T>())
        thresh = testinghelpers::getEpsilon<T>();
    else
    {
        // Threshold adjustment
#ifdef BLIS_INT_ELEMENT_TYPE
        double adj = 1.0;
#else
        double adj = 1.5;
#endif
        thresh = adj*2*testinghelpers::getEpsilon<T>();
    }
    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_axpyv<T>(conj_x, n, incx, incy, alpha, xi, xexval,
                   yj, yexval, thresh);
}

static double NaN = std::numeric_limits<double>::quiet_NaN();
static double Inf = std::numeric_limits<double>::infinity();

/*
    Exception value testing on vectors :
    SAXPY currently uses the bli_zaxpyv_zen_int5( ... ) kernel for computation.
    The sizes and indices given in the instantiator are to ensure code coverage inside
    the kernel, and to verify the compliance accordingly.

    Kernel structure for bli_zaxpyv_zen_int5( ... ) :
    Main loop    :  In blocks of 14 --> L14
    Fringe loops :  In blocks of 10 --> L10
                    In blocks of 6  --> L6
                    In blocks of 4  --> L4
                    In blocks of 2  --> L2
                    Element-wise loop --> LScalar

    The sizes chosen are as follows :
    52 - 3*L14 + L10
    48 - 3*L14 + L6
    46 - 3*L14 + L4
    45 - 3*L14 + L2 + LScalar

    The following indices are sufficient to ensure code-coverage of loops
    in these sizes :
    0, 41 - In L14
    43    - In { L10, L6, L4, L2 }, based on the size
    44    - In { L10, L6, L4, LScalar }, based on the size

    The alpha values are such that they check for compliance against possible
    optimizations that might have been done.

    P.S : Some test cases also check whether NaN has to be induced in the computation
          such as 0.0 * { {NaN, 0}, {+Inf, 0}, {-Inf, 0}, ... }, and a few more.
*/

// Exception value testing(on X vector alone) with unit strides
INSTANTIATE_TEST_SUITE_P(
    vecX_unitStrides,
    zaxpyvEVT,
    ::testing::Combine(
        ::testing::Values('n' // n: use x, c: use conj(x)
#ifdef TEST_BLIS_TYPED
                          ,
                          'c' // this option is BLIS-api specific.
#endif
                          ),
        ::testing::Values(gtint_t(45), gtint_t(46),
                          gtint_t(48), gtint_t(52)),                    // n, size of vectors with unit-stride
        ::testing::Values(gtint_t(1)),                                  // stride size for x
        ::testing::Values(gtint_t(1)),                                  // stride size for y
        ::testing::Values(gtint_t(0), gtint_t(41),
                          gtint_t(43), gtint_t(44)),                    // indices to set exception values on x
        ::testing::Values(dcomplex{NaN, 0.0}, dcomplex{-Inf, 0.0},
                          dcomplex{0.0, Inf}, dcomplex{-2.3, NaN},
                          dcomplex{4.5, -Inf}, dcomplex{NaN, Inf}),     // exception values to set on x
        ::testing::Values(gtint_t(0)),                                  // dummy index on y
        ::testing::Values(dcomplex{0.0, 0.0}),                          // dummy value on y
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0},
                          dcomplex{-1.0, 0.0}, dcomplex{0.0, 1.0},
                          dcomplex{0.0, -1.0}, dcomplex{-3.3, 1.7})     // alpha
        ),
    ::axpyvEVTPrint<dcomplex>());

// Exception value testing(on Y vector alone) with unit strides
INSTANTIATE_TEST_SUITE_P(
    vecY_unitStrides,
    zaxpyvEVT,
    ::testing::Combine(
        ::testing::Values('n' // n: use x, c: use conj(x)
#ifdef TEST_BLIS_TYPED
                          ,
                          'c' // this option is BLIS-api specific.
#endif
                          ),
        ::testing::Values(gtint_t(45), gtint_t(46),
                          gtint_t(48), gtint_t(52)),                    // n, size of vectors with unit-stride
        ::testing::Values(gtint_t(1)),                                  // stride size for x
        ::testing::Values(gtint_t(1)),                                  // stride size for y
        ::testing::Values(gtint_t(0)),                                  // dummy index on x
        ::testing::Values(dcomplex{0.0, 0.0}),                          // dummy value on x
        ::testing::Values(gtint_t(0), gtint_t(41),
                          gtint_t(43), gtint_t(44)),                    // indices to set exception values on y
        ::testing::Values(dcomplex{NaN, 0.0}, dcomplex{-Inf, 0.0},
                          dcomplex{0.0, Inf}, dcomplex{-2.3, NaN},
                          dcomplex{4.5, -Inf}, dcomplex{NaN, Inf}),     // exception values to set on y
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0},
                          dcomplex{-1.0, 0.0}, dcomplex{0.0, 1.0},
                          dcomplex{0.0, -1.0}, dcomplex{-3.3, 1.7})     // alpha
        ),
    ::axpyvEVTPrint<dcomplex>());

// Exception value testing(on X and Y vectors) with unit strides
INSTANTIATE_TEST_SUITE_P(
    vecXY_unitStrides,
    zaxpyvEVT,
    ::testing::Combine(
        ::testing::Values('n' // n: use x, c: use conj(x)
#ifdef TEST_BLIS_TYPED
                          ,
                          'c' // this option is BLIS-api specific.
#endif
                          ),
        ::testing::Values(gtint_t(45), gtint_t(46),
                          gtint_t(48), gtint_t(52)),                    // n, size of vectors with unit-stride
        ::testing::Values(gtint_t(1)),                                  // stride size for x
        ::testing::Values(gtint_t(1)),                                  // stride size for y
        ::testing::Values(gtint_t(0), gtint_t(41),
                          gtint_t(43), gtint_t(44)),                    // indices to set exception values on x
        ::testing::Values(dcomplex{NaN, 0.0}, dcomplex{-Inf, 0.0},
                          dcomplex{0.0, Inf}, dcomplex{-2.3, NaN},
                          dcomplex{4.5, -Inf}, dcomplex{NaN, Inf}),     // exception values to set on x
        ::testing::Values(gtint_t(0), gtint_t(41),
                          gtint_t(43), gtint_t(44)),                    // indices to set exception values on y
        ::testing::Values(dcomplex{NaN, 0.0}, dcomplex{-Inf, 0.0},
                          dcomplex{0.0, Inf}, dcomplex{-2.3, NaN},
                          dcomplex{4.5, -Inf}, dcomplex{NaN, Inf}),     // exception values to set on y
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0},
                          dcomplex{-1.0, 0.0}, dcomplex{0.0, 1.0},
                          dcomplex{0.0, -1.0}, dcomplex{-3.3, 1.7})     // alpha
        ),
    ::axpyvEVTPrint<dcomplex>());

// Exception value testing(on vectors) with non-unit strides
// We have to test a single scalar loop. The indices are such
// that we cover _vecX_, _vecY_ and _vecXY_ cases together.
INSTANTIATE_TEST_SUITE_P(
    vecXY_nonUnitStrides,
    zaxpyvEVT,
    ::testing::Combine(
        ::testing::Values('n' // n: use x, c: use conj(x)
#ifdef TEST_BLIS_TYPED
                          ,
                          'c' // this option is BLIS-api specific.
#endif
                          ),
        ::testing::Values(gtint_t(50)),                                         // n, size of vectors with non-unit strides
        ::testing::Values(gtint_t(3)),                                          // stride size for x
        ::testing::Values(gtint_t(5)),                                          // stride size for y
        ::testing::Values(gtint_t(1), gtint_t(27), gtint_t(49)),                // indices to set exception values on x
        ::testing::Values(dcomplex{NaN, 0.0}, dcomplex{-Inf, 0.0},
                          dcomplex{0.0, Inf}, dcomplex{-2.3, NaN},
                          dcomplex{4.5, -Inf}, dcomplex{NaN, Inf},
                          dcomplex{2.3, -3.5}),                                 // exception values to set on x
        ::testing::Values(gtint_t(0), gtint_t(26), gtint_t(49)),                // indices to set exception values on y
        ::testing::Values(dcomplex{NaN, 0.0}, dcomplex{-Inf, 0.0},
                          dcomplex{0.0, Inf}, dcomplex{-2.3, NaN},
                          dcomplex{4.5, -Inf}, dcomplex{NaN, Inf},
                          dcomplex{2.3, -3.5}),                                 // exception values to set on y
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0},
                          dcomplex{-1.0, 0.0}, dcomplex{0.0, 1.0},
                          dcomplex{0.0, -1.0}, dcomplex{-3.3, 1.7})             // alpha
        ),
    ::axpyvEVTPrint<dcomplex>());

/*
    Exception value testing on alpha :
    Alpha values are set to Nan, +Inf or -Inf. A dummy
    value of 0.0 is induced in X and Y vectors, to further
    verify the propagation.

    The size(s) for _zen3 and _zen4 instantiators are chosen such
    that code coverage is ensured in the respective kernels.
*/
INSTANTIATE_TEST_SUITE_P(
    alpha_unitStrides,
    zaxpyvEVT,
    ::testing::Combine(
        ::testing::Values('n' // n: use x, c: use conj(x)
#ifdef TEST_BLIS_TYPED
                          ,
                          'c' // this option is BLIS-api specific.
#endif
                          ),
        ::testing::Values(gtint_t(45), gtint_t(46),
                          gtint_t(48), gtint_t(52)),                            // n, size of vectors with unit-stride
        ::testing::Values(gtint_t(1)),                                          // stride size for x
        ::testing::Values(gtint_t(1)),                                          // stride size for y
        ::testing::Values(gtint_t(0)),                                          // indices to set zero on x
        ::testing::Values(dcomplex{0.0, 0.0}),
        ::testing::Values(gtint_t(0)),                                          // indices to set zero on y
        ::testing::Values(dcomplex{0.0, 0.0}),
        ::testing::Values(dcomplex{NaN, 0.0}, dcomplex{-Inf, 0.0},
                          dcomplex{0.0, Inf}, dcomplex{-2.3, NaN},
                          dcomplex{4.5, -Inf}, dcomplex{NaN, Inf})              // alpha
        ),
    ::axpyvEVTPrint<dcomplex>());

// Exception value testing(on alpha) with non-unit strided vectors
INSTANTIATE_TEST_SUITE_P(
    alpha_nonUnitStrides,
    zaxpyvEVT,
    ::testing::Combine(
        ::testing::Values('n' // n: use x, c: use conj(x)
#ifdef TEST_BLIS_TYPED
                          ,
                          'c' // this option is BLIS-api specific.
#endif
                          ),
        ::testing::Values(gtint_t(50)),                                         // n, size of vectors with non-unit strides
        ::testing::Values(gtint_t(3)),                                          // stride size for x
        ::testing::Values(gtint_t(5)),                                          // stride size for y
        ::testing::Values(gtint_t(0), gtint_t(25)),                             // indices to set zero on x
        ::testing::Values(dcomplex{0.0, 0.0}),
        ::testing::Values(gtint_t(0), gtint_t(40)),                             // indices to set zero on y
        ::testing::Values(dcomplex{0.0, 0.0}),
        ::testing::Values(dcomplex{NaN, 0.0}, dcomplex{-Inf, 0.0},
                          dcomplex{0.0, Inf}, dcomplex{-2.3, NaN},
                          dcomplex{4.5, -Inf}, dcomplex{NaN, Inf})              // alpha
        ),
    ::axpyvEVTPrint<dcomplex>());
