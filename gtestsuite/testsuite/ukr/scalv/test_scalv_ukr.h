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

#pragma once

#include <stdexcept>

#include "level1/scalv/scalv.h"
#include "level1/ref_scalv.h"
#include "inc/check_error.h"
#include "common/testing_helpers.h"

/**
 * @brief Microkernel test body for scalv operation.
 */
template<typename T, typename U, typename FT>
static void test_scalv_ukr( FT ukr, char conja_alpha, gtint_t n, gtint_t incx,
                            T alpha, double thresh, bool is_memory_test = false )
{
    // Obtain and allocate memory for vectors.
    T *x, *x_ref;

    gtint_t size_x = testinghelpers::buff_dim( n, incx ) * sizeof( T );

    testinghelpers::ProtectedBuffer x_buffer( size_x, false, is_memory_test );

    // is_memory_test = false for x_ref since we don't require different green
    // or red zones.
    testinghelpers::ProtectedBuffer x_ref_buffer( size_x, false, false );

    // Acquire the first set of greenzones for x.
    x = ( T* )x_buffer.greenzone_1;
    // There is no greenzone_2 for x_ref.
    x_ref = ( T* )x_ref_buffer.greenzone_1;

    // Initialize x with random data.
    testinghelpers::datagenerators::randomgenerators( -10, 10, n, incx, x );

    // Copying x to x_ref, for comparision after computation
    memcpy( x_ref, x, size_x );

    // Char conjx to BLIS conjx conversion
    conj_t blis_conjalpha;
    testinghelpers::char_to_blis_conj( conja_alpha, &blis_conjalpha );

    testinghelpers::ProtectedBuffer::start_signal_handler();
    try
    {
        // Invoking BLIS ukr.
        // This will check for out of bounds access within first redzone.
        ukr( blis_conjalpha, n, &alpha, x, incx, nullptr );

        if ( is_memory_test )
        {
            // Acquire the pointers near the second redzone.
            x = ( T* )x_buffer.greenzone_2;

            // Copy the data for x accordingly
            memcpy( x, x_ref, size_x );

            // Invoking BLIS ukr to check with the second redzone.
            ukr( blis_conjalpha, n, &alpha, x, incx, nullptr );
        }
    }
    catch(const std::exception& e)
    {
        // Reset to default signal handler
        testinghelpers::ProtectedBuffer::stop_signal_handler();

        // Show failure in case seg fault was detected
        FAIL() << "Memory Test Failed";
    }

    // Reset to default signal handler
    testinghelpers::ProtectedBuffer::stop_signal_handler();

    // Invoking the reference implementation to get reference results.
    if constexpr ( testinghelpers::type_info<T>::is_complex &&
                   testinghelpers::type_info<U>::is_real )
        testinghelpers::ref_scalv<T, U>( conja_alpha, n, alpha.real, x_ref, incx );
    else    // if constexpr ( std::is_same<T,U>::value )
        testinghelpers::ref_scalv<T, U>( conja_alpha, n, alpha, x_ref, incx );

    // Compute component-wise error.
    computediff<T>( n, x, x_ref, incx, thresh );
}
