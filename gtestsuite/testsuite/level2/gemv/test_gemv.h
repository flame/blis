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

#pragma once

#include "gemv.h"
#include "level2/ref_gemv.h"
#include "inc/check_error.h"
#include "common/testing_helpers.h"
#include <stdexcept>
#include <algorithm>

template<typename T>
void test_gemv( char storage, char transa, char conjx, gtint_t m, gtint_t n,
                T alpha, gtint_t lda_inc, gtint_t incx, T beta, gtint_t incy,
                double thresh, bool is_memory_test = false,
                bool is_evt_test = false, T a_exval = T{0}, T x_exval = T{0},
                T y_exval = T{0} )
{
    // Compute the leading dimensions for matrix size calculation.
    gtint_t lda = testinghelpers::get_leading_dimension( storage, 'n', m, n, lda_inc );

    dim_t size_a = testinghelpers::matsize( storage, 'n', m, n, lda ) * sizeof(T);
    testinghelpers::ProtectedBuffer a_buf(size_a, false, is_memory_test);
    testinghelpers::datagenerators::randomgenerators<T>( 1, 5, storage, m, n, (T*)(a_buf.greenzone_1), 'n', lda );

    // Get correct vector lengths.
    gtint_t lenx = ( testinghelpers::chknotrans( transa ) ) ? n : m ;
    gtint_t leny = ( testinghelpers::chknotrans( transa ) ) ? m : n ;

    dim_t size_x = testinghelpers::buff_dim(lenx, incx) * sizeof(T);
    dim_t size_y = testinghelpers::buff_dim(leny, incy) * sizeof(T);
    testinghelpers::ProtectedBuffer x_buf(size_x, false, is_memory_test);
    testinghelpers::ProtectedBuffer y_buf(size_y, false, is_memory_test);
    
    // For y_ref, we don't need different greenzones and any redzone.
    // Thus, we pass is_memory_test as false
    testinghelpers::ProtectedBuffer y_ref_buffer( size_y, false, false );

    testinghelpers::datagenerators::randomgenerators<T>( 1, 3, lenx, incx, (T*)(x_buf.greenzone_1) );
    testinghelpers::datagenerators::randomgenerators<T>( 1, 3, leny, incy, (T*)(y_buf.greenzone_1) );

    T* a = (T*)(a_buf.greenzone_1);
    T* x = (T*)(x_buf.greenzone_1);
    T* y = (T*)(y_buf.greenzone_1);
    T* y_ref = ( T* )y_ref_buffer.greenzone_1; // For y_ref, there is no greenzone_2

    // Copying the contents of y to y_ref
    memcpy( y_ref, y, size_y );

    if ( is_evt_test )
    {
        // Add extreme value to A matrix
        dim_t ai = rand() % m;
        dim_t aj = rand() % n;
        testinghelpers::set_ev_mat( storage, 'n', lda, ai, aj, a_exval, a );

        // Add extreme value to x vector
        x[ (rand() % lenx) * std::abs(incx) ] = x_exval;

        // Add extreme value to y vector
        y[ (rand() % leny) * std::abs(incy) ] = y_exval;
    }

    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    testinghelpers::ProtectedBuffer::start_signal_handler();
    try
    {
        gemv<T>( storage, transa, conjx, m, n, &alpha, a, lda, x, incx, &beta,
                 y, incy );
        
        if ( is_memory_test )
        {
            memcpy((a_buf.greenzone_2), (a_buf.greenzone_1), size_a);
            memcpy((x_buf.greenzone_2), (x_buf.greenzone_1), size_x);
            memcpy((y_buf.greenzone_2), y_ref, size_y);

            gemv<T>( storage, transa, conjx, m, n, &alpha,
                     (T*)(a_buf.greenzone_2), lda,
                     (T*)(x_buf.greenzone_2), incx,
                     &beta,
                     (T*)(y_buf.greenzone_2), incy );
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
    testinghelpers::ref_gemv<T>( storage, transa, conjx, m, n, alpha, a,
                                 lda, x, incx, beta, y_ref, incy );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( "y", leny, y, y_ref, incy, thresh );
}
