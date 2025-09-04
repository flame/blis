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
#include "common/testing_helpers.h"
#include "level1/axpyv/axpyv.h"
#include "inc/check_error.h"
#include "common/wrong_inputs_helpers.h"

template <typename T>
class axpyv_IIT_ERS : public ::testing::Test {};
typedef ::testing::Types<float, double, scomplex, dcomplex> TypeParam; // The supported datatypes from BLAS/CBLAS calls for AXPY
TYPED_TEST_SUITE(axpyv_IIT_ERS, TypeParam); // Defining individual testsuites based on the datatype support.

// Adding namespace to get default parameters(valid case) from testinghelpers/common/wrong_input_helpers.h.
using namespace testinghelpers::IIT;

#if defined(TEST_BLAS_LIKE) || defined(TEST_CBLAS)
/*
    Early Return Scenarios(ERS) for BLAS/CBLAS compliance :

    The AXPY API is expected to return early in the following cases:
    1. When n <= 0 (BLAS compliance).
    2. When alpha = 0 (BLAS compliance).
*/

// Early return cases with non-unit strides on vectors
// When n < 0
TYPED_TEST(axpyv_IIT_ERS, n_lt_zero_nonUnitStrides)
{
  using T = TypeParam;
  T alpha;
  testinghelpers::initone<T>( alpha );

  // Test with nullptr for all suitable arguments that shouldn't be accessed.
  axpyv<T>( CONJ, -1, alpha, nullptr, 5, nullptr, 5 );

  // Test with all arguments correct except for the value we are choosing to test.
  // Defining the x vector
  std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, N, 5 );
  // Defining the y vector with values for debugging purposes
  std::vector<T> y = testinghelpers::get_random_vector<T>( -10, 10, N, 5 );

  // Copy so that we check that the elements of y are not modified.
  std::vector<T> y_ref(y);

  axpyv<T>( CONJ, -1, alpha, x.data(), 5, y.data(), 5 );
  // Use bitwise comparison (no threshold).
  computediff( "y", N, y.data(), y_ref.data(), 5 );
}

// When n = 0
TYPED_TEST(axpyv_IIT_ERS, n_eq_zero_nonUnitStrides)
{
  using T = TypeParam;
  T alpha;
  testinghelpers::initone<T>( alpha );

  // Test with nullptr for all suitable arguments that shouldn't be accessed.
  axpyv<T>( CONJ, 0, alpha, nullptr, 5, nullptr, 5 );

  // Test with all arguments correct except for the value we are choosing to test.
  // Defining the x vector
  std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, N, 5 );
  // Defining the y vector with values for debugging purposes
  std::vector<T> y = testinghelpers::get_random_vector<T>( -10, 10, N, 5 );

  // Copy so that we check that the elements of y are not modified.
  std::vector<T> y_ref(y);

  axpyv<T>( CONJ, 0, alpha, x.data(), 5, y.data(), 5 );
  // Use bitwise comparison (no threshold).
  computediff( "y", N, y.data(), y_ref.data(), 5 );
}

// When alpha = 0
TYPED_TEST(axpyv_IIT_ERS, alpha_eq_zero_nonUnitStrides)
{
  using T = TypeParam;
  T alpha;
  testinghelpers::initzero<T>( alpha );

  // Test with nullptr for all suitable arguments that shouldn't be accessed.
  axpyv<T>( CONJ, N, alpha, nullptr, 5, nullptr, 5 );

  // Test with all arguments correct except for the value we are choosing to test.
  // Defining the x vector
  std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, N, 5 );
  // Defining the y vector with values for debugging purposes
  std::vector<T> y = testinghelpers::get_random_vector<T>( -10, 10, N, 5 );

  // Copy so that we check that the elements of y are not modified.
  std::vector<T> y_ref(y);

  axpyv<T>( CONJ, N, alpha, x.data(), 5, y.data(), 5 );
  // Use bitwise comparison (no threshold).
  computediff( "y", N, y.data(), y_ref.data(), 5 );
}

// Early return cases with unit strides on vectors
// When n < 0
TYPED_TEST(axpyv_IIT_ERS, n_lt_zero_unitStrides)
{
  using T = TypeParam;
  T alpha;
  testinghelpers::initone<T>( alpha );

  // Test with nullptr for all suitable arguments that shouldn't be accessed.
  axpyv<T>( CONJ, -1, alpha, nullptr, 1, nullptr, 1 );

  // Test with all arguments correct except for the value we are choosing to test.
  // Defining the x vector
  std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, N, 1 );
  // Defining the y vector with values for debugging purposes
  std::vector<T> y = testinghelpers::get_random_vector<T>( -10, 10, N, 1 );

  // Copy so that we check that the elements of y are not modified.
  std::vector<T> y_ref(y);

  axpyv<T>( CONJ, -1, alpha, x.data(), 1, y.data(), 1 );
  // Use bitwise comparison (no threshold).
  computediff( "y", N, y.data(), y_ref.data(), 1 );
}

// When n = 0
TYPED_TEST(axpyv_IIT_ERS, n_eq_zero_unitStrides)
{
  using T = TypeParam;
  T alpha;
  testinghelpers::initone<T>( alpha );

  // Test with nullptr for all suitable arguments that shouldn't be accessed.
  axpyv<T>( CONJ, 0, alpha, nullptr, 1, nullptr, 1 );

  // Test with all arguments correct except for the value we are choosing to test.
  // Defining the x vector
  std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, N, 1 );
  // Defining the y vector with values for debugging purposes
  std::vector<T> y = testinghelpers::get_random_vector<T>( -10, 10, N, 1 );

  // Copy so that we check that the elements of y are not modified.
  std::vector<T> y_ref(y);

  axpyv<T>( CONJ, 0, alpha, x.data(), 1, y.data(), 1 );
  // Use bitwise comparison (no threshold).
  computediff( "y", N, y.data(), y_ref.data(), 1 );
}

// When alpha = 0
TYPED_TEST(axpyv_IIT_ERS, alpha_eq_zero_unitStrides)
{
  using T = TypeParam;
  T alpha;
  testinghelpers::initzero<T>( alpha );

  // Test with nullptr for all suitable arguments that shouldn't be accessed.
  axpyv<T>( CONJ, N, alpha, nullptr, 1, nullptr, 1 );

  // Test with all arguments correct except for the value we are choosing to test.
  // Defining the x vector
  std::vector<T> x = testinghelpers::get_random_vector<T>( -10, 10, N, 1 );
  // Defining the y vector with values for debugging purposes
  std::vector<T> y = testinghelpers::get_random_vector<T>( -10, 10, N, 1 );

  // Copy so that we check that the elements of y are not modified.
  std::vector<T> y_ref(y);

  axpyv<T>( CONJ, N, alpha, x.data(), 1, y.data(), 1 );
  // Use bitwise comparison (no threshold).
  computediff( "y", N, y.data(), y_ref.data(), 1 );
}
#endif

