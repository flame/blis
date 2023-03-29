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
#include "test_nrm2.h"

template <typename T>
class xnrm2 : public ::testing::Test {};
typedef ::testing::Types<float, double> TypeParam;
TYPED_TEST_SUITE(xnrm2, TypeParam);

TYPED_TEST(xnrm2, zeroFP) {
    using T = TypeParam;
    T x = T(0);

    T norm = nrm2<T,T>(1, &x, 1);
    EXPECT_EQ(0, norm);
}

TYPED_TEST(xnrm2, minFP) {
    using T = TypeParam;
    T x = std::numeric_limits<T>::min();

    T norm = nrm2<T,T>(1, &x, 1);
    EXPECT_EQ(x, norm);
}

TYPED_TEST(xnrm2, maxFP) {
    using T = TypeParam;
    T x = std::numeric_limits<T>::max();

    T norm = nrm2<T,T>(1, &x, 1);
    EXPECT_EQ(x, norm);
}

TEST(dnrm2, largeDouble) {
    using T = double;
    gtint_t n = 2;
    std::vector<T> x{3e300, 4e300}, y{-4e300, -3e300};

    T norm = nrm2<T,T>(n, x.data(), 1);
    EXPECT_EQ(5e300, norm);

    norm = nrm2<T,T>(n, y.data(), 1);
    EXPECT_EQ(5e300, norm);
}
