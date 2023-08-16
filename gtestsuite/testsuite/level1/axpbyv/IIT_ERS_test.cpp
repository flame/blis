/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

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
#include "common/testing_helpers.h"
#include "axpbyv.h"
#include "inc/check_error.h"
#include "common/wrong_inputs_helpers.h"

template <typename T>
class Axpby_IIT_ERS_Test : public ::testing::Test {};
typedef ::testing::Types<float, double, scomplex, dcomplex> TypeParam; // The supported datatypes from BLAS calls for AXPBY
TYPED_TEST_SUITE(Axpby_IIT_ERS_Test, TypeParam); // Defining individual testsuites based on the datatype support.

// Adding namespace to get default parameters(valid case) from testinghelpers/common/wrong_input_helpers.h.
using namespace testinghelpers::IIT;

/*
    Early Return Scenarios(ERS) :

    The AXPBY API is expected to return early in the following cases:
    1. When n < 0.

*/

#ifdef TEST_BLAS

// When n < 0
TYPED_TEST(Axpby_IIT_ERS_Test, n_lt_zero)
{
  using T = TypeParam;
  // Defining the C matrix with values for debugging purposes
  std::vector<T> y = testinghelpers::get_random_vector<T>( -10, 10, N, INC );

  T alpha, beta;
  testinghelpers::initone<T>( alpha );
  testinghelpers::initzero<T>( beta );
  // Copy so that we check that the elements of C are not modified.
  std::vector<T> y_ref(y);

  axpbyv<T>( CONJ, -1, alpha, nullptr, INC, beta, y.data(), INC );
  // Use bitwise comparison (no threshold).
  computediff( N, y.data(), y_ref.data(), INC );
}

// When n = 0
TYPED_TEST(Axpby_IIT_ERS_Test, n_eq_zero)
{
  using T = TypeParam;
  // Defining the C matrix with values for debugging purposes
  std::vector<T> y = testinghelpers::get_random_vector<T>( -10, 10, N, INC );

  T alpha, beta;
  testinghelpers::initone<T>( alpha );
  testinghelpers::initzero<T>( beta );
  // Copy so that we check that the elements of C are not modified.
  std::vector<T> y_ref(y);

  axpbyv<T>( CONJ, 0, alpha, nullptr, INC, beta, y.data(), INC );
  // Use bitwise comparison (no threshold).
  computediff( N, y.data(), y_ref.data(), INC );
}

#endif

