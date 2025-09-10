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
#include "level1/amaxv/amaxv.h"
#include "level1/ref_amaxv.h"
#include "inc/check_error.h"
#include "common/testing_helpers.h"

/**
 * @brief Test body for amaxv micro-kernels
 */

template<typename T, typename FT>
void test_amaxv_ukr( FT ukr_fp, gtint_t n, gtint_t incx, double thresh, bool is_memory_test = false )
{
    // Pointers to obtain the required memory.
    T *x, *x_copy;
    gtint_t size_x = testinghelpers::buff_dim( n, incx ) * sizeof( T );

    // Create the objects for the input operand
    // The kernel does not expect the memory to be aligned
    testinghelpers::ProtectedBuffer x_buffer( size_x, false, is_memory_test );

    // Creating x_copy, to save the contents of x(without any redzones)
    testinghelpers::ProtectedBuffer x_copy_buffer( size_x, false, false );

    // Acquire the first set of greenzones for x and y
    x = ( T* )x_buffer.greenzone_1;
    x_copy = ( T* )x_copy_buffer.greenzone_1; // For x_copy, there is no greenzone_2

    // Initialize the memory with random data
    testinghelpers::datagenerators::randomgenerators( -10, 10, n, incx, x );

    // Copying the contents of x to x_copy
    memcpy( x_copy, x, size_x );

    dim_t idx;

    // Add signal handler for segmentation fault
    testinghelpers::ProtectedBuffer::start_signal_handler();
    try
    {
        // Call the ukr function.
        // This call is made irrespective of is_memory_test.
        // This will check for out of bounds access with first redzone(if memory test is true)
        // Else, it will just call the ukr function.
        ukr_fp( n, x, incx, &idx, nullptr );

        if ( is_memory_test )
        {
            // Acquire the pointers near the second redzone
            x = ( T* )x_buffer.greenzone_2;

            // Copy the data for x and y accordingly
            memcpy( x, x_copy, size_x );

            // Call the ukr function, to check with the second redzone.
            ukr_fp( n, x, incx, &idx, nullptr );
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
    //    Call reference implementation to get ref results.
    //----------------------------------------------------------
    dim_t idx_ref = testinghelpers::ref_amaxv<T>( n, x, incx );

    //----------------------------------------------------------
    //              Compute component-wise error.
    //----------------------------------------------------------
    computediff<gtint_t>( "idx", idx, idx_ref );
}

// Test-case logger : Used to print the test-case details for unit testing the kernels.
// NOTE : The kernel name is the prefix in instantiator name, and thus is not printed
// with this logger.
template <typename T>
class amaxvUKRPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<T,gtint_t,gtint_t,bool>> str) const {
        gtint_t n     = std::get<1>(str.param);
        gtint_t incx  = std::get<2>(str.param);
        bool is_memory_test = std::get<3>(str.param);

        std::string str_name = "_n_" + std::to_string(n);
        str_name += "_incx_" + testinghelpers::get_value_string(incx);
        str_name += ( is_memory_test ) ? "_mem_test_enabled" : "_mem_test_disabled";
        return str_name;
    }
};
