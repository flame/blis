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
#include "test_nrm2.h"
#include "common/wrong_inputs_helpers.h"

/*
    Early Return Scenarios(ERS) for BLAS/CBLAS compliance :

    The NRM2 API is expected to return early in the following cases:
    1. When n <= 0 (BLAS compliance).
*/

template <typename T>
class nrm2_IIT_ERS : public ::testing::Test {};
typedef ::testing::Types<float, double, scomplex, dcomplex> TypeParam;
TYPED_TEST_SUITE(nrm2_IIT_ERS, TypeParam);

// Adding namespace to get default parameters from testinghelpers/common/wrong_input_helpers.h.
using namespace testinghelpers::IIT;

// Early return n < 0.
TYPED_TEST(nrm2_IIT_ERS, negative_n) {
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    T x = T{-3.7};
    // initialize blis norm with garbage.
    RT blis_norm = -4.2;
    blis_norm = nrm2<T>(-2, &x, INC);

    computediff<RT>("norm", blis_norm, 0.0);
}

// Early return n = 0.
TYPED_TEST(nrm2_IIT_ERS, zero_n) {
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    gtint_t n = 0;
    gtint_t incx = 1;
    // initialize norm to ensure that it is set to zero from nrm2 and it does not simply return.
    RT blis_norm = 19.0;
    // using nullptr since x should not be accessed anyway.
    // If "x" is accessed before return then nrm2 would segfault.
    blis_norm = nrm2<T>(n, nullptr, incx);
    RT ref_norm = testinghelpers::ref_nrm2<T>(n, nullptr, incx);
    computediff<RT>("norm", blis_norm, ref_norm);
}
