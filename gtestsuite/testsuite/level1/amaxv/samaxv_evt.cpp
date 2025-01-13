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

class samaxvEVT :
        public ::testing::TestWithParam<std::tuple<gtint_t,      // n
                                                   gtint_t,      // incx
                                                   gtint_t,      // xi, index for exval in x
                                                   float,        // xi_exval
                                                   gtint_t,      // xj, index for exval in x
                                                   float>> {};   // xj_exval

// Tests using random values as vector elements.
TEST_P( samaxvEVT, API )
{
    
    using T = float;
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

static float NaN = std::numeric_limits<float>::quiet_NaN();
static float Inf = std::numeric_limits<float>::infinity();

/*
    Exception value testing on vectors(Zen3) :
    SAMAXV currently uses the bli_samaxv_zen_int( ... ) kernel for computation on zen3
    machines.
    The sizes and indices given in the instantiator are to ensure code coverage inside
    the kernel.

    Kernel structure for bli_samaxv_zen_int( ... ) is as follows :
    Main loop    :  In blocks of 64   --> L64
    Fringe loops :  In blocks of 32   --> L32
                    In blocks of 16   --> L16
                    In blocks of 8    --> L8
                    Element wise loop --> LScalar

    The sizes chosen are as follows :
    255 - 3*L64 + L32 + L16 + L8 + 7(LScalar)

    The following indices are sufficient to ensure code-coverage of loops :
    0 <= idx < 192   - In L64
    192 <= idx < 224 - In L32
    224 <= idx < 240 - In L16
    240 <= idx < 248 - In L8
    248 <= idx < 255 - In LScalar

    The testsuite requires 2 indices(and 2 exception values) to set exception values in the vector.
*/

// Exception value testing with unit strides
INSTANTIATE_TEST_SUITE_P(
    unitStrides_zen3,
    samaxvEVT,
    ::testing::Combine(
        ::testing::Values(gtint_t(255)),                                        // n, size of vectors with unit-stride
        ::testing::Values(gtint_t(1)),                                          // stride size for x
        ::testing::Values(gtint_t(0), gtint_t(150), gtint_t(215),
                          gtint_t(243), gtint_t(251)),                          // xi, index for exval in xi_exval
        ::testing::Values(NaN, -Inf, Inf, float(2.3)),                          // xi_exval
        ::testing::Values(gtint_t(10), gtint_t(64), gtint_t(237),
                          gtint_t(241), gtint_t(254)),                          // xj, index for exval in xj_exval
        ::testing::Values(NaN, -Inf, Inf, float(2.3))                           // xj_exval
        ),
        ::amaxvEVTPrint<float>()
    );

/*
    Exception value testing on vectors(Zen4) :
    damaxv currently uses the bli_damaxv_zen_int_avx512( ... ) kernel for computation
    on zen4 and zen5 machines.
    The sizes and indices given in the instantiator are to ensure code coverage inside
    the kernel.

    Kernel structure for bli_damaxv_zen_int( ... ) is as follows :
    For unit strides :
        Main loop    :  In blocks of 128 --> L128
        Fringe loops :  In blocks of 64  --> L64
                        In blocks of 32  --> L32
                        In blocks of 16  --> L16
                        In blocks of 8   --> L8
                        Element-wise loop --> LScalar

    For non-unit strides : A single loop, to process element wise.

    The sizes chosen are as follows :
    383 - 5*L128 + L64 + L32 + L16 + 15(LScalar)

    The following indices are sufficient to ensure code-coverage of loops :
    0 <= idx < 640    - In L128
    640 <= idx < 704  - In L64
    704 <= idx < 736  - In L32
    736 <= idx < 752  - In L16
    752 <= idx < 767  - In LScalar

    The testsuite requires 2 indices(and 2 exception values) to be induced in the vector.
*/
// Exception value testing with unit strides
INSTANTIATE_TEST_SUITE_P(
    unitStrides_zen4,
    samaxvEVT,
    ::testing::Combine(
        ::testing::Values(gtint_t(767)),                                        // n, size of vectors with unit-stride
        ::testing::Values(gtint_t(1)),                                          // stride size for x
        ::testing::Values(gtint_t(0), gtint_t(500), gtint_t(650),
                          gtint_t(705), gtint_t(749), gtint_t(761)),            // xi, index for exval in xi_exval
        ::testing::Values(NaN, -Inf, Inf, float(2.3)),                          // xi_exval
        ::testing::Values(gtint_t(100), gtint_t(621), gtint_t(690),
                          gtint_t(739), gtint_t(751), gtint_t(766)),            // xj, index for exval in xj_exval
        ::testing::Values(NaN, -Inf, Inf, float(2.3))                           // xj_exval
        ),
        ::amaxvEVTPrint<float>()
    );


// Exception value testing with non-unit strides
INSTANTIATE_TEST_SUITE_P(
    nonUnitStrides,
    samaxvEVT,
    ::testing::Combine(
        ::testing::Values(gtint_t(10)),                                         // n, size of vectors with unit-stride
        ::testing::Values(gtint_t(3)),                                          // stride size for x
        ::testing::Values(gtint_t(0), gtint_t(5)),                              // xi, index for exval in xi_exval
        ::testing::Values(NaN, Inf, -Inf, float(2.3)),                          // xi_exval
        ::testing::Values(gtint_t(1), gtint_t(9)),                              // xj, index for exval in xj_exval
        ::testing::Values(NaN, -Inf, Inf, float(2.3))                           // xj_exval
        ),
        ::amaxvEVTPrint<float>()
    );
