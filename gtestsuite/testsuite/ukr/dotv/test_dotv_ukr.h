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
#include "level1/dotv/dotv.h"
#include "level1/ref_dotv.h"
#include "inc/check_error.h"
#include "common/testing_helpers.h"

/**
 * @brief Microkernel test body for dotv operation.
 */

template<typename T, typename FT>
static void test_dotv_ukr( FT ukr, char conjx, char conjy, gtint_t n, gtint_t incx,
                       gtint_t incy, double thresh, bool is_memory_test = false )
{
    // Obtain and allocate memory for vectors.
    T *x, *y, *y_ref;

    gtint_t size_x = testinghelpers::buff_dim( n, incx );
    gtint_t size_y = testinghelpers::buff_dim( n, incy );

    testinghelpers::ProtectedBuffer x_buf( size_x * sizeof( T ), false, is_memory_test );
    testinghelpers::ProtectedBuffer y_buf( size_y * sizeof( T ), false, is_memory_test );

    // No redzones are required for y_ref buffer thus, we pass is_memory_test = false.
    testinghelpers::ProtectedBuffer y_ref_buf( size_y * sizeof( T ), false, false );

    // Acquire the first set of greenzones for x and y
    x = ( T* )x_buf.greenzone_1;
    y = ( T* )y_buf.greenzone_1;
    y_ref = ( T* )y_ref_buf.greenzone_1; // For y_ref, there is no greenzone_2

    // Initialize the vectors with random data.
    testinghelpers::datagenerators::randomgenerators( -10, 10, n, incx, x );
    testinghelpers::datagenerators::randomgenerators( -10, 10, n, incy, y );

    // Copying the contents of y to y_ref, for comparision after computation.
    memcpy( y_ref, y, size_y * sizeof( T ) );

    T rho;
    // Create a copy of rho so that we can check reference results.
    T rho_ref;

    // conj? conversion to BLIS conjugate type.
    conj_t blis_conjx, blis_conjy;
    testinghelpers::char_to_blis_conj( conjx, &blis_conjx );
    testinghelpers::char_to_blis_conj( conjy, &blis_conjy );

    // Add signal handler for Segmentation Faults.
    testinghelpers::ProtectedBuffer::start_signal_handler();
    try
    {
        // Invoking BLIS ukr.
        // This will check for out of bounds access within first redzone.
        ukr( blis_conjx, blis_conjy, n, x, incx, y, incy, &rho, nullptr );

        if ( is_memory_test )
        {
            // Acquire the pointers near the second redzone.
            x = ( T* )x_buf.greenzone_2;
            y = ( T* )y_buf.greenzone_2;

            // Copy the data for x and y accordingly.
            memcpy( x, x_buf.greenzone_1, size_x * sizeof( T ) );
            memcpy( y, y_ref_buf.greenzone_1, size_y * sizeof( T ) );

            // Invoking BLIS ukr to check with the second redzone.
            ukr( blis_conjx, blis_conjy, n, x, incx, y, incy, &rho, nullptr );
        }
    }
    catch( const std::exception& e )
    {
        // Reset to default signal handler.
        testinghelpers::ProtectedBuffer::stop_signal_handler();

        // Show failure in case Segmentation Fault was detected.
        FAIL() << "Memory Test Failed";
    }

    // Reset to default signal handler.
    testinghelpers::ProtectedBuffer::stop_signal_handler();

    // Invoking the reference implementation to get reference results.
    if constexpr (testinghelpers::type_info<T>::is_real)
        testinghelpers::ref_dotv<T>( n, x, incx, y_ref, incy, &rho_ref );
    else
        testinghelpers::ref_dotv<T>( conjx, conjy, n, x, incx, y_ref, incy, &rho_ref );

    // Compute component-wise error.
    computediff<T>( "rho", rho, rho_ref, thresh );
}


// Test-case logger : Used to print the test-case details based on parameters
template <typename T>
class dotvUKRPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<T,char,char,gtint_t,gtint_t,gtint_t, bool>> str) const {
        char conjx    = std::get<1>(str.param);
        char conjy    = std::get<2>(str.param);
        gtint_t n     = std::get<3>(str.param);
        gtint_t incx  = std::get<4>(str.param);
        gtint_t incy  = std::get<5>(str.param);
        bool is_memory_test = std::get<6>(str.param);

        std::string str_name = "_n_" + std::to_string(n);
        str_name += "_conjx_" + std::string(&conjx, 1);
        str_name += "_conjy_" + std::string(&conjy, 1);
        str_name += "_incx_" + testinghelpers::get_value_string(incx);
        str_name += "_incy_" + testinghelpers::get_value_string(incy);
        str_name += ( is_memory_test ) ? "_mem_test_enabled" : "_mem_test_disabled";

        return str_name;
    }
};
