/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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
#include "test_amaxv.h"

class damaxvEVT :
        public ::testing::TestWithParam<std::tuple<gtint_t,      // n
                                                   gtint_t,      // incx
                                                   gtint_t,      // xi, index for exval in x
                                                   double,       // xi_exval
                                                   gtint_t,      // xj, index for exval in x
                                                   double>> {};  // xj_exval

// Tests using random values as vector elements.
TEST_P( damaxvEVT, API )
{
    using T = double;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // vector length
    gtint_t n = std::get<0>(GetParam());
    // stride size for x
    gtint_t incx = std::get<1>(GetParam());
    // index for exval in x
    gtint_t xi = std::get<2>(GetParam());
    // exval for index xi
    T xi_exval = std::get<3>(GetParam());
    // index for exval in x
    gtint_t xj = std::get<4>(GetParam());
    // exval for index xj
    T xj_exval = std::get<5>(GetParam());

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_amaxv<T>( n, incx, xi, xi_exval, xj, xj_exval );
}

static double NaN = std::numeric_limits<double>::quiet_NaN();
static double Inf = std::numeric_limits<double>::infinity();

/*
    Exception value testing on vectors(Zen3) :
    DAMAXV currently uses the bli_damaxv_zen_int( ... ) kernel for computation on zen3
    machines.
    The sizes and indices given in the instantiator are to ensure code coverage inside
    the kernel.

    Kernel structure for bli_damaxv_zen_int( ... ) is as follows :
    bli_damaxv_zen_int() --> bli_vec_absmax_double() --> bli_vec_search_double()
    bli_vec_absmax_double() structure:
    For unit strides :
        Main loop    :  In blocks of 48 --> L48
        Fringe loops :  In blocks of 32 --> L32
                        In blocks of 16 --> L16
                        In blocks of 8  --> L8
                        In blocks of 4  --> L4
                        In blocks of 2  --> L2
                        Element-wise loop --> LScalar

    For non-unit strides : A single loop, to process element wise.

    bli_vec_search_double() structure:
    For unit strides :
        Main loop    :  In blocks of 4 --> L4
                        In blocks of 2 --> L2
                        Element-wise loop --> LScalar

    For non-unit strides : A single loop, to process element wise.

    The sizes chosen are as follows(in accordance to the structure in bli_vec_absmax_double()) :
    176 : 3*L48 + L32
    175 : 3*L48 + L16 + L8 + L4 + L2 + 1(LScalar)

    The following indices are sufficient to ensure code-coverage of loops :
    0 <= idx < 144    - In L48
    144 <= idx < 160  - In L32(for size 176), in L16(for size 175)
    160 <= idx < 168  - In L8
    168 <= idx < 172  - In L4
    172 <= idx < 174  - In L2
    174 <= idx < 175  - In LScalar

    These sizes and indices also ensure code coverage for bli_vec_search_double().
    The testsuite requires 2 indices(and 2 exception values) to be induced in the vector.
*/

// Exception value testing with unit strides
INSTANTIATE_TEST_SUITE_P(
    unitStrides_zen3,
    damaxvEVT,
    ::testing::Combine(
        ::testing::Values(gtint_t(175), gtint_t(176)),                          // n, size of vectors with unit-stride
        ::testing::Values(gtint_t(1)),                                          // stride size for x
        ::testing::Values(gtint_t(0), gtint_t(143), gtint_t(159),
                          gtint_t(167), gtint_t(171), gtint_t(173),
                          gtint_t(174)),                                        // xi, index for exval in xi_exval
        ::testing::Values(NaN, -Inf, Inf, double(2.3)),                         // xi_exval
        ::testing::Values(gtint_t(5), gtint_t(140), gtint_t(155),
                          gtint_t(163), gtint_t(170), gtint_t(172)),            // xj, index for exval in xj_exval
        ::testing::Values(NaN, -Inf, Inf, double(2.3))                          // xj_exval
        ),
        ::amaxvEVTPrint<double>()
    );

/*
    Exception value testing on vectors(Zen4) :
    damaxv currently uses the bli_damaxv_zen_int_avx512( ... ) kernel for computation
    on zen4 and zen5 machines.
    The sizes and indices given in the instantiator are to ensure code coverage inside
    the kernel.

    Kernel structure for bli_damaxv_zen_int( ... ) is as follows :
    For unit strides :
        Main loop    :  In blocks of 64 --> L64
        Fringe loops :  In blocks of 32 --> L32
                        In blocks of 16 --> L16
                        In blocks of 8  --> L8
                        Element-wise loop --> LScalar

    For non-unit strides : A single loop, to process element wise.

    The sizes chosen are as follows :
    383 - 5*L64 + L32 + L16 + L8 + 7(LScalar)

    The following indices are sufficient to ensure code-coverage of loops :
    0 <= idx < 320    - In L64
    320 <= idx < 352  - In L32
    352 <= idx < 368  - In L16
    368 <= idx < 376  - In L8
    376 <= idx < 383  - In LScalar

    The testsuite requires 2 indices(and 2 exception values) to be induced in the vector.
*/

// Exception value testing with unit strides
INSTANTIATE_TEST_SUITE_P(
    unitStrides_zen4,
    damaxvEVT,
    ::testing::Combine(
        ::testing::Values(gtint_t(383)),                                        // n, size of vectors with unit-stride
        ::testing::Values(gtint_t(1)),                                          // stride size for x
        ::testing::Values(gtint_t(0), gtint_t(300), gtint_t(325),
                          gtint_t(356), gtint_t(373), gtint_t(380)),            // xi, index for exval in xi_exval
        ::testing::Values(NaN, -Inf, Inf, double(2.3)),                         // xi_exval
        ::testing::Values(gtint_t(10), gtint_t(290), gtint_t(347),
                          gtint_t(361), gtint_t(364), gtint_t(376)),            // xj, index for exval in xj_exval
        ::testing::Values(NaN, -Inf, Inf, double(2.3))                          // xj_exval
        ),
        ::amaxvEVTPrint<double>()
    );


// Exception value testing with non-unit strides
INSTANTIATE_TEST_SUITE_P(
    nonUnitStrides,
    damaxvEVT,
    ::testing::Combine(
        ::testing::Values(gtint_t(10)),                                         // n, size of vectors with unit-stride
        ::testing::Values(gtint_t(3)),                                          // stride size for x
        ::testing::Values(gtint_t(0), gtint_t(5)),                              // xi, index for exval in xi_exval
        ::testing::Values(NaN, -Inf, Inf, double(2.3)),                         // xi_exval
        ::testing::Values(gtint_t(5), gtint_t(9)),                              // xj, index for exval in xj_exval
        ::testing::Values(NaN, -Inf, Inf, double(2.3))                          // xj_exval
        ),
        ::amaxvEVTPrint<double>()
    );
