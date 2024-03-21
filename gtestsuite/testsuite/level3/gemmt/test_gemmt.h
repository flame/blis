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

#pragma once

#include "gemmt.h"
#include "level3/ref_gemmt.h"
#include "inc/check_error.h"
#include <stdexcept>
#include <algorithm>
#include "common/testing_helpers.h"

template<typename T>
void test_gemmt( char storage, char uplo, char trnsa, char trnsb, gtint_t n,
    gtint_t k, gtint_t lda_inc, gtint_t ldb_inc, gtint_t ldc_inc, T alpha,
    T beta, double thresh, bool is_mem_test=false, bool is_evt_test=false,
    T evt_a=T{0.0}, T evt_b=T{0.0}, T evt_c=T{0.0} )
{
    // Compute the leading dimensions of a, b, and c.
    gtint_t lda = testinghelpers::get_leading_dimension( storage, trnsa, n, k, lda_inc );
    gtint_t ldb = testinghelpers::get_leading_dimension( storage, trnsb, k, n, ldb_inc );
    gtint_t ldc = testinghelpers::get_leading_dimension( storage, 'n', n, n, ldc_inc );

    //----------------------------------------------------------
    //         Initialize matrics with random numbers
    //----------------------------------------------------------
    T *a_ptr, *b_ptr, *c_ptr;
    dim_t size_a = testinghelpers::matsize(storage, trnsa, n, k, lda) * sizeof(T);
    testinghelpers::ProtectedBuffer a(size_a, false, is_mem_test );
    a_ptr = (T*)a.greenzone_1;
    testinghelpers::datagenerators::randomgenerators<T>( -2, 8, storage, n, k, a_ptr, trnsa, lda);

    dim_t size_b = testinghelpers::matsize(storage, trnsb, k, n, ldb) * sizeof(T);
    testinghelpers::ProtectedBuffer b(size_b, false, is_mem_test );
    b_ptr = (T*)b.greenzone_1;
    testinghelpers::datagenerators::randomgenerators<T>( -5, 2, storage, k, n, b_ptr, trnsb, ldb);

    dim_t size_c = testinghelpers::matsize(storage, 'n', n, n, ldc) * sizeof(T);
    testinghelpers::ProtectedBuffer c(size_c, false, is_mem_test );
    c_ptr = (T*)c.greenzone_1;
    testinghelpers::datagenerators::randomgenerators<T>( -3, 5, storage, n, n, c_ptr, 'n', ldc);

    if ( is_evt_test )
    {
        dim_t n_rand = rand() % std::min(n, k);
        dim_t k_rand = rand() % std::min(n, k);
        a_ptr[n_rand + k_rand * lda] = evt_a;
    }

    if ( is_evt_test )
    {
        dim_t n_rand = rand() % std::min(n, k);
        dim_t k_rand = rand() % std::min(n, k);
        b_ptr[n_rand + k_rand * lda] = evt_a;
    }

    if ( is_evt_test )
    {
        dim_t n_rand = rand() % std::min(n, k);
        dim_t k_rand = rand() % std::min(n, k);
        b_ptr[n_rand + k_rand * lda] = evt_a;
    }

    // Create a copy of c so that we can check reference results.
    std::vector<T> c_ref(testinghelpers::matsize(storage, 'n', n, n, ldc));
    memcpy(c_ref.data(), c_ptr, size_c);

    // add signal handler for segmentation fault
    testinghelpers::ProtectedBuffer::start_signal_handler();
    try
    {
        //----------------------------------------------------------
        //                  Call BLIS function
        //----------------------------------------------------------
        gemmt<T>( storage, uplo, trnsa, trnsb, n, k, &alpha, a_ptr, lda,
                  b_ptr, ldb, &beta, c_ptr, ldc );
        if (is_mem_test)
        {
            memcpy(a.greenzone_2, a.greenzone_1, size_a);
            memcpy(b.greenzone_2, b.greenzone_1, size_b);
            memcpy(c.greenzone_2, c_ref.data(), size_c);

            gemmt<T>( storage, uplo, trnsa, trnsb, n, k, &alpha, (T*)a.greenzone_2, lda,
                      (T*)b.greenzone_2, ldb, &beta, (T*)c.greenzone_2, ldc );
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
    testinghelpers::ref_gemmt<T>( storage, uplo, trnsa, trnsb, n, k, alpha,
               a_ptr, lda, b_ptr, ldb, beta, c_ref.data(), ldc );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( storage, n, n, c_ptr, c_ref.data(), ldc, thresh, is_evt_test );
}
