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

#include "level1/scal2v/scal2v.h"
#include "level1/ref_scal2v.h"
#include "inc/check_error.h"
#include "common/testing_helpers.h"

/**
 * @brief Microkernel test body for scal2v operation.
 */
template<typename T, typename U, typename FT>
static void test_scal2v_ukr( FT ukr, char conjx, gtint_t n, gtint_t incx, gtint_t incy,
                            T alpha, double thresh, bool is_memory_test = false )
{
    // Obtain and allocate memory for vectors.
    T *x, *y, *y_ref;

    // Sizes of x and y vectors
    gtint_t size_x = testinghelpers::buff_dim( n, incx ) * sizeof( T );
    gtint_t size_y = testinghelpers::buff_dim( n, incy ) * sizeof( T );

    // Create the object for the required operands
    // The kernel does not expect the memory to be aligned
    testinghelpers::ProtectedBuffer x_buffer( size_x, false, is_memory_test );
    testinghelpers::ProtectedBuffer y_buffer( size_y, false, is_memory_test );

    // For y_ref, we don't need different greenzones and any redzone.
    // Thus, we pass is_memory_test as false
    testinghelpers::ProtectedBuffer y_ref_buffer( size_y, false, false );

    // Acquire the first set of greenzones for x and y.
    x = ( T* )x_buffer.greenzone_1;
    y = ( T* )y_buffer.greenzone_1;

    // There is no greenzone_2 for y_ref.
    y_ref = ( T* )y_ref_buffer.greenzone_1;

    // Initialize x and y with random data.
    testinghelpers::datagenerators::randomgenerators( -10, 10, n, incx, x );
    testinghelpers::datagenerators::randomgenerators( -10, 10, n, incy, y );

    // Copying y to y_ref, for comparision after computation
    memcpy( y_ref, y, size_y );

    // Char conjx to BLIS conjx conversion
    conj_t blis_conjx;
    testinghelpers::char_to_blis_conj( conjx, &blis_conjx );

    testinghelpers::ProtectedBuffer::start_signal_handler();
    try
    {
        // Invoking BLIS ukr.
        // This will check for out of bounds access within first redzone.
        ukr( blis_conjx, n, &alpha, x, incx, y, incy, nullptr );

        if ( is_memory_test )
        {
            // Acquire the pointers near the second redzone.
            x = ( T* )x_buffer.greenzone_2;
            y = ( T* )y_buffer.greenzone_2;

            // Copy the data for x and y accordingly
            memcpy( x, x_buffer.greenzone_1, size_x );
            memcpy( y, y_ref, size_y );

            // Invoking BLIS ukr to check with the second redzone.
            ukr( blis_conjx, n, &alpha, x, incx, y, incy, nullptr );
        }
    }
    catch(const std::exception& e)
    {
        // Reset to default signal handler
        testinghelpers::ProtectedBuffer::stop_signal_handler();

        // Show failure in case seg fault was detected
        FAIL() << "Memory Test Failed";
    }

    //----------------------------------------------------------
    //    Call reference implementation to get ref results.
    //----------------------------------------------------------
    testinghelpers::ref_scal2v<T>( conjx, n, alpha, x, incx, y_ref, incy );

    //----------------------------------------------------------
    //              Compute component-wise error.
    //----------------------------------------------------------
    computediff<T>( "y", n, y, y_ref, incy, thresh );
}


// Test-case logger : Used to print the test-case details based on parameters
template <typename T, typename FT>
class scal2vUKRPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<FT,char,gtint_t,gtint_t,gtint_t,T,bool>> str) const {
        char conjx = std::get<1>(str.param);
        gtint_t n = std::get<2>(str.param);
        gtint_t incx = std::get<3>(str.param);
        gtint_t incy = std::get<4>(str.param);
        T alpha = std::get<5>(str.param);
        bool is_memory_test = std::get<6>(str.param);

        std::string str_name = "_n_" + std::to_string(n);
        str_name += "_conjx_" + std::string(&conjx, 1);
        str_name += "_incx_" + testinghelpers::get_value_string(incx);
        str_name += "_incy_" + testinghelpers::get_value_string(incy);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name += ( is_memory_test ) ? "_mem_test_enabled" : "_mem_test_disabled";

        return str_name;
    }
};
