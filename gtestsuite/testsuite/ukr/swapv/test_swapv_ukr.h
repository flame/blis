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

#include "level1/swapv/swapv.h"
#include "inc/check_error.h"

/**
 * @brief Microkernel test body for swapv operation.
 */
template<typename T, typename FT>
static void test_swapv_ukr( FT ukr, gtint_t n, gtint_t incx, gtint_t incy,
                            bool is_memory_test = false )
{
    // Obtain and allocate memory for vectors.
    T *x, *y, *x_ref, *y_ref;

    gtint_t size_x = testinghelpers::buff_dim( n, incx ) * sizeof( T );
    gtint_t size_y = testinghelpers::buff_dim( n, incy ) * sizeof( T );

    testinghelpers::ProtectedBuffer x_buffer( size_x, false, is_memory_test );
    testinghelpers::ProtectedBuffer y_buffer( size_y, false, is_memory_test );

    // is_memory_test = false for x_ref & y_ref since we don't require
    // different green or red zones.
    testinghelpers::ProtectedBuffer x_ref_buffer( size_x, false, false );
    testinghelpers::ProtectedBuffer y_ref_buffer( size_y, false, false );

    // Acquire the first set of greenzones for x.
    x = ( T* )x_buffer.greenzone_1;
    y = ( T* )y_buffer.greenzone_1;

    // There is no greenzone_2 for x_ref & y_ref
    x_ref = ( T* )x_ref_buffer.greenzone_1;
    y_ref = ( T* )y_ref_buffer.greenzone_1;

    // Initialize x with random data.
    testinghelpers::datagenerators::randomgenerators( -100, 100, n, incx, x );
    testinghelpers::datagenerators::randomgenerators( 110, 200, n, incy, y );

    // Copying x to x_ref & y to y_ref, for comparision after computation
    memcpy( x_ref, x, size_x );
    memcpy( y_ref, y, size_y );

    testinghelpers::ProtectedBuffer::start_signal_handler();
    try
    {
        // This will check for out of bounds access within first redzone.
        swapv<T>( n, x, incx, y, incy );

        if ( is_memory_test )
        {
            // Acquire the pointers near the second redzone.
            x = ( T* )x_buffer.greenzone_2;
            y = ( T* )y_buffer.greenzone_2;

            // Copy the data for x and y accordingly
            memcpy( x, x_ref, size_x );
            memcpy( y, y_ref, size_y );

            // Invoking  ukr to check with the second redzone.
            swapv<T>( n, x, incx, y, incy );
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

    //----------------------------------------------------------
    //              Compute binary comparison
    //----------------------------------------------------------
    computediff<T>( n, x, x_ref, y, y_ref, incx, incy, false );

}


// Test-case logger : Used to print the test-case details based on parameters
template <typename T>
class swapvUKRPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<T, gtint_t,gtint_t,gtint_t,bool>> str) const {
        gtint_t n      = std::get<1>(str.param);
        gtint_t incx   = std::get<2>(str.param);
        gtint_t incy   = std::get<3>(str.param);
        bool is_memory_test = std::get<4>(str.param);

        std::string str_name = "_n_" + std::to_string(n);
        str_name += "_incx_" + testinghelpers::get_value_string(incx);
        str_name += "_incy_" + testinghelpers::get_value_string(incy);
        str_name += ( is_memory_test ) ? "_mem_test_enabled" : "_mem_test_disabled";
        return str_name;
    }
};
