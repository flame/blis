/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

class zamaxvGeneric :
        public ::testing::TestWithParam<std::tuple<gtint_t,		//n
                                                   gtint_t>> {};	//incx

// Tests using random values as vector elements.
TEST_P( zamaxvGeneric, API )
{
    using T = dcomplex;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // vector length
    gtint_t n = std::get<0>(GetParam());
    // stride size for x
    gtint_t incx = std::get<1>(GetParam());

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------

#ifdef OPENMP_NESTED_1diff
    #pragma omp parallel default(shared)
    {
	vary_num_threads();
        //std::cout << "Inside 1diff parallel regions\n";
        test_amaxv<T>( n, incx );
    }
#elif OPENMP_NESTED_2
    #pragma omp parallel default(shared)
    {
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 2 parallel regions\n";
        test_amaxv<T>( n, incx );
    }
    }
#elif OPENMP_NESTED_1
    #pragma omp parallel default(shared)
    {
        //std::cout << "Inside 1 parallel region\n";
        test_amaxv<T>( n, incx );
    }
#else
        //std::cout << "Not inside parallel region\n";
        test_amaxv<T>( n, incx );
#endif
}

//Black box testing extended for different range of values
INSTANTIATE_TEST_SUITE_P(
        Blackbox_Small_Sizes,
        zamaxvGeneric,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(11), 1),                    // n size of vector takes values from 1 to 11 with step size of 1.
            ::testing::Values(gtint_t(1))                                    // stride size for x
        ),
        ::amaxvGenericPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        Blackbox_Average_Sizes,
        zamaxvGeneric,
        ::testing::Combine(
            ::testing::Range(gtint_t(100), gtint_t(502), 50),                 // n size of vector takes values from 100 to 500 with step size of 50.
            ::testing::Values(gtint_t(1))                                     // stride size for x
        ),
        ::amaxvGenericPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        Blackbox_Max_Sizes,
        zamaxvGeneric,
        ::testing::Combine(
            ::testing::Range(gtint_t(1024), gtint_t(65535), 1023),           // n size of vector takes values from 2pow10 to 2pow16-1 with step size of 1023.
            ::testing::Values(gtint_t(1))                                    // stride size for x
        ),
        ::amaxvGenericPrint()
    );

//Non unit testing extended for different stride values
INSTANTIATE_TEST_SUITE_P(
        NonUnitIncrements_Stride,
        zamaxvGeneric,
        ::testing::Combine(
            ::testing::Values(gtint_t(123), gtint_t(111), gtint_t(20)),     // m size of vector
            ::testing::Values(gtint_t(4), gtint_t(8))                       // stride size for x
        ),
        ::amaxvGenericPrint()
    );

INSTANTIATE_TEST_SUITE_P(
        Blackbox_Stride_Greater,
        zamaxvGeneric,
        ::testing::Combine(
            ::testing::Range(gtint_t(1), gtint_t(10), 1),                    // n size of vector takes values from 1 to 10 with step size 1
            ::testing::Values(gtint_t(11))                                   // stride size for x
        ),
        ::amaxvGenericPrint()
    );
