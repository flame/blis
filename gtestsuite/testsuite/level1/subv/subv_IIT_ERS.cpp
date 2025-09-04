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
#include "test_subv.h"
#include "common/wrong_inputs_helpers.h"
#include "common/testing_helpers.h"
#include "inc/check_error.h"

template <typename T>
class subv_IIT_ERS : public ::testing::Test {};
typedef ::testing::Types<float, double, scomplex, dcomplex> TypeParam;
TYPED_TEST_SUITE(subv_IIT_ERS, TypeParam);

using namespace testinghelpers::IIT;

#if defined(TEST_BLIS_TYPED)

/*
    BLIS Early Return Scenarios(ERS):

    SUBV is expected to return early in the following cases:
    1. n <= 0
*/

// n < 0, with non-unit stride
TYPED_TEST(subv_IIT_ERS, n_lt_zero_nonUnitStride)
{
  using T = TypeParam;
  gtint_t invalid_n = -1;
  gtint_t inc = 5;

  // Test with all arguments correct except for the value we are choosing to test.
  // Defining the X & Y vectors with values for debugging purposes
  std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, N, inc );
  std::vector<T> y = testinghelpers::get_random_vector<T>( -10, 10, N, inc );

  // Copy so that we check that the elements of Y are not modified.
  std::vector<T> y_ref(y);

  // Call BLIS subv with a invalid value for n==-1 &  non-unit stride inc = 5.
  subv<T>( 'n', invalid_n,  x.data(), inc, y.data(), inc );

  // Use bitwise comparison (no threshold).
  computediff<T>( "y", N, y.data(), y_ref.data(), inc );
}

// n < 0, with unit stride
TYPED_TEST(subv_IIT_ERS, n_lt_zero_unitStride)
{
  using T = TypeParam;
  gtint_t invalid_n = -1;
  gtint_t inc = 1;

  // Test with all arguments correct except for the value we are choosing to test.
  // Defining the X & Y vectors with values for debugging purposes
  std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, N, inc );
  std::vector<T> y = testinghelpers::get_random_vector<T>( -10, 10, N, inc );

  // Copy so that we check that the elements of Y are not modified.
  std::vector<T> y_ref(y);

  // Call BLIS subv with a invalid value for n==-1 &  unit stride inc = 1.
  subv<T>( 'n', invalid_n, x.data(), inc, y.data(), inc );

  // Use bitwise comparison (no threshold).
  computediff<T>( "y", N, y.data(), y_ref.data(), inc );
}

// n == 0, with non-unit stride
TYPED_TEST(subv_IIT_ERS, n_eq_zero_nonUnitStride)
{
  using T = TypeParam;
  gtint_t invalid_n = 0;
  gtint_t inc = 2;

  // Test with all arguments correct except for the value we are choosing to test.
  // Defining the X & Y vectors with values for debugging purposes
  std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, N, inc );
  std::vector<T> y = testinghelpers::get_random_vector<T>( -10, 10, N, inc );

  // Copy so that we check that the elements of Y are not modified.
  std::vector<T> y_ref(y);

  // Call BLIS subv with a invalid value for n==0 &  non-unit stride inc = 2.
  subv<T>( 'n', invalid_n, x.data(), inc, y.data(), inc );

  // Use bitwise comparison (no threshold).
  computediff<T>( "y", N, y.data(), y_ref.data(), inc );
}

// n == 0, with unit stride
TYPED_TEST(subv_IIT_ERS, n_eq_zero_unitStride)
{
  using T = TypeParam;
  gtint_t invalid_n = 0;
  gtint_t inc = 1;

  // Test with all arguments correct except for the value we are choosing to test.
  // Defining the X & Y vectors with values for debugging purposes
  std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, N, inc );
  std::vector<T> y = testinghelpers::get_random_vector<T>( -10, 10, N, inc );

  // Copy so that we check that the elements of Y are not modified.
  std::vector<T> y_ref(y);

  // Call BLIS subv with a invalid value for n==0 &  unit stride inc = 1.
  subv<T>( 'n', invalid_n, x.data(), inc, y.data(), inc );

  // Use bitwise comparison (no threshold).
  computediff<T>( "y", N, y.data(), y_ref.data(), inc );
}
#endif
