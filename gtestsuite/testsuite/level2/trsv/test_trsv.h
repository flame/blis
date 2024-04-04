/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

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

#pragma once

#include "trsv.h"
#include "level2/ref_trsv.h"
#include "inc/check_error.h"
#include <stdexcept>
#include <algorithm>
#include "common/testing_helpers.h"

template<typename T>
void test_trsv(
                char storage,
                char uploa,
                char transa,
                char diaga,
                gtint_t n,
                T alpha,
                gtint_t lda_inc,
                gtint_t incx,
                double thresh,
                bool is_memory_test = false,
                bool is_evt_test = false,
                T evt_x = T{0},
                T evt_a = T{0}
             )
{
    // Compute the leading dimensions for matrix size calculation.
    gtint_t lda = testinghelpers::get_leading_dimension( storage, transa, n, n, lda_inc );

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------

    dim_t size_a = testinghelpers::matsize(storage, transa, n, n, lda) * sizeof(T);

    // Buffers for A matrix and X vector are always unaligned
    testinghelpers::ProtectedBuffer a(size_a, false, is_memory_test );
    testinghelpers::datagenerators::randomgenerators<T>( 1, 5, storage, n, n, (T*)(a.greenzone_1), transa, lda );

    dim_t size_x = testinghelpers::buff_dim(n, incx) * sizeof(T);
    testinghelpers::ProtectedBuffer x(size_x, false, is_memory_test );
    testinghelpers::datagenerators::randomgenerators<T>( 1, 3, n, incx, (T*)(x.greenzone_1) );

    T* a_ptr = (T*)(a.greenzone_1);
    T* x_ptr = (T*)(x.greenzone_1);

    // Make A matix diagonal dominant to make sure that algorithm doesn't diverge
    // This makes sure that the TRSV problem is solvable
    for ( dim_t a_dim = 0; a_dim < n; ++a_dim )
    {
        a_ptr[ a_dim + (a_dim* lda) ] = a_ptr[ a_dim + (a_dim* lda) ] * T{10};
    }

    // add extreme values to the X vector
    if ( is_evt_test )
    {
        x_ptr[ (rand() % n) * std::abs(incx) ] = evt_x;
    }

    // add extreme values to the A matrix
    if ( is_evt_test )
    {
        dim_t n_idx = rand() % n;
        dim_t m_idx = (std::max)((dim_t)0, n_idx - 1);
        a_ptr[ m_idx + (n_idx * lda) ] = evt_a;
        a_ptr[ m_idx + (m_idx *lda) ] = evt_a;
    }

    // skipped making A triangular
    // A matrix being a non triangular matrix could be a better test
    // because we are exepcted to read only from the upper or lower triangular
    // part of the data, contents of the rest of the matrix should not change the
    // result.
    // testinghelpers::make_triangular<T>( storage, uploa, n, a_ptr, lda );

    // Create a copy of x so that we can check reference results.
    std::vector<T> x_ref(testinghelpers::buff_dim(n, incx));
    memcpy(x_ref.data(), x_ptr, size_x);
    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    // add signal handler for segmentation fault
    testinghelpers::ProtectedBuffer::start_signal_handler();
    try
    {
        trsv<T>( storage, uploa, transa, diaga, n, &alpha, a_ptr, lda, x_ptr, incx );
        if (is_memory_test)
        {
            memcpy(a.greenzone_2, a.greenzone_1, size_a);
            memcpy(x.greenzone_2,  x_ref.data(), size_x);
            trsv<T>( storage, uploa, transa, diaga, n, &alpha, (T*)a.greenzone_2, lda, (T*)x.greenzone_2, incx );
        }
    }
    catch(const std::exception& e)
    {
        // reset to default signal handler
        testinghelpers::ProtectedBuffer::stop_signal_handler();

        // show failure in case seg fault was detected
        FAIL() << "Memory Test Failed";
    }
    // reset to default signal handler
    testinghelpers::ProtectedBuffer::stop_signal_handler();

    //----------------------------------------------------------
    //                  Call reference implementation.
    //----------------------------------------------------------
    testinghelpers::ref_trsv<T>( storage, uploa, transa, diaga, n, &alpha, a_ptr, lda, x_ref.data(), incx );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( "x", n, x_ptr, x_ref.data(), incx, thresh, is_evt_test );
}
