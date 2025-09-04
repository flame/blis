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

template <typename T>
class nrm2UOT : public ::testing::Test {};
typedef ::testing::Types<float, double, scomplex, dcomplex> TypeParam;
TYPED_TEST_SUITE(nrm2UOT, TypeParam);

// Testing for max representable number to see if overflow is handled correctly.
TYPED_TEST(nrm2UOT, maxFP_scalar) {
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;

    RT maxval = (std::numeric_limits<RT>::max)();
    T x = T{maxval};

    RT norm = nrm2<T>(1, &x, 1);
    computediff<RT>("norm", norm, maxval);
}
TYPED_TEST(nrm2UOT, maxFP_vectorized) {
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    gtint_t n = 64;
    std::vector<T> x(n, T{0});
    RT maxval = (std::numeric_limits<RT>::max)();
    x[17] = T{maxval};
    RT norm = nrm2<T>(n, x.data(), 1);
    computediff<RT>("norm", norm, maxval);
}

// Testing for min representable number to see if underflow is handled correctly.
TYPED_TEST(nrm2UOT, minFP_scalar) {
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;

    RT minval = (std::numeric_limits<RT>::min)();
    T x = T{minval};
    RT norm = nrm2<T>(1, &x, 1);
    computediff<RT>("norm", norm, minval);
}
TYPED_TEST(nrm2UOT, minFP_vectorized) {
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    gtint_t n = 64;
    std::vector<T> x(n, T{0});
    RT minval = (std::numeric_limits<RT>::min)();
    x[17] = T{minval};
    RT norm = nrm2<T>(n, x.data(), 1);
    computediff<RT>("norm", norm, minval);
}

// Since there are 2 different paths, vectorized and scalar,
// we break this into 2 tests, once for each case.
TYPED_TEST(nrm2UOT, zeroFP_scalar) {
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    T x = T{0};

    RT norm = nrm2<T>(1, &x, 1);
    computediff<RT>("norm", norm, 0);
}
TYPED_TEST(nrm2UOT, zeroFP_vectorized) {
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    gtint_t n = 64;
    std::vector<T> x(n, T{0});

    RT norm = nrm2<T>(n, x.data(), 1);
    computediff<RT>("norm", norm, 0);
}

/*
    Adding a type-parameterized test to check for
    overflow and underflow handling with multiple threads
    in case of dnrm2 and dznrm2. Can also be used if snrm2
    and scnrm2 are multithreaded.
*/

// Checking only for overflow, based on the threshold
TYPED_TEST( nrm2UOT, OFlow_MT ) {
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    gtint_t n = 2950000;
    std::vector<T> x(n, T{1.0}); // A normal value
    RT bigval;
    if constexpr ( std::is_same<RT, float>::value )
    {
        bigval = powf( ( float )FLT_RADIX, floorf( ( FLT_MAX_EXP - 23)  * 0.5f ) ) * ( 1.0f + FLT_EPSILON );
    }
    else
    {
        bigval = pow( ( double )FLT_RADIX, floor( ( DBL_MAX_EXP - 52)  * 0.5 ) ) * ( 1.0 + DBL_EPSILON );
    }

    // Set the threshold for the errors:
    double thresh = 2*testinghelpers::getEpsilon<T>();
    x[1000] = T{ bigval };
    x[50000] = T{ bigval };
    x[151001] = T{ bigval };
    x[2949999] = T{ bigval };

    RT norm = nrm2<T>( n, x.data(), 1 );
    RT ref_norm = testinghelpers::ref_nrm2<T>( n, x.data(), 1 );
    computediff<RT>( "norm", norm, ref_norm, thresh );
}

// Checking only for underflow, based on the threshold
TYPED_TEST( nrm2UOT, UFlow_MT ) {
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    gtint_t n = 2950000;
    std::vector<T> x(n, T{1.0}); // A normal value
    RT smlval;
    if constexpr ( std::is_same<RT, float>::value )
    {
        smlval = powf( ( float )FLT_RADIX, ceilf( ( FLT_MIN_EXP - 1 )  * 0.5f ) ) * ( 1.0f - FLT_EPSILON );
    }
    else
    {
        smlval = pow( ( double )FLT_RADIX, ceil( ( DBL_MIN_EXP - 1 )  * 0.5 ) ) * ( 1.0 - DBL_EPSILON );
    }

    // Set the threshold for the errors:
    double thresh = 2*testinghelpers::getEpsilon<T>();
    x[1000] = T{ smlval };
    x[50000] = T{ smlval };
    x[151001] = T{ smlval };
    x[2949999] = T{ smlval };

    RT norm = nrm2<T>( n, x.data(), 1 );
    RT ref_norm = testinghelpers::ref_nrm2<T>( n, x.data(), 1 );
    computediff<RT>( "norm", norm, ref_norm, thresh );
}

// Checking for both overflow and underflow, based on the thresholds
TYPED_TEST( nrm2UOT, OUFlow_MT ) {
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    gtint_t n = 2950000;
    std::vector<T> x(n, T{1.0}); // A normal value
    RT bigval, smlval;
    if constexpr ( std::is_same<RT, float>::value )
    {
        bigval = powf( ( float )FLT_RADIX, floorf( ( FLT_MAX_EXP - 23)  * 0.5f ) ) * ( 1.0f + FLT_EPSILON );
        smlval = powf( ( float )FLT_RADIX, ceilf( ( FLT_MIN_EXP - 1 )  * 0.5f ) ) * ( 1.0f - FLT_EPSILON );
    }
    else
    {
        bigval = pow( ( double )FLT_RADIX, floor( ( DBL_MAX_EXP - 52)  * 0.5 ) ) * ( 1.0 + DBL_EPSILON );
        smlval = pow( ( double )FLT_RADIX, ceil( ( DBL_MIN_EXP - 1 )  * 0.5 ) ) * ( 1.0 - DBL_EPSILON );
    }

    // Set the threshold for the errors:
    double thresh = 2*testinghelpers::getEpsilon<T>();
    x[1000] = T{ smlval };
    x[50000] = T{ bigval };
    x[151001] = T{ bigval };
    x[2949999] = T{ smlval };

    RT norm = nrm2<T>( n, x.data(), 1 );
    RT ref_norm = testinghelpers::ref_nrm2<T>( n, x.data(), 1 );
    computediff<RT>( "norm", norm, ref_norm, thresh );
}

// Specific test case used by an ISV.
// Checks for overflow.
TEST(dnrm2, largeDouble) {
    using T = double;
    gtint_t n = 2;
    std::vector<T> x{3e300, 4e300}, y{-4e300, -3e300};

    T norm = nrm2<T>(n, x.data(), 1);
    computediff<T>( "norm", norm, 5e300 );

    norm = nrm2<T>(n, y.data(), 1);
    computediff<T>( "norm", norm, 5e300 );
}
