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
#include "level1/axpyf/axpyf.h"
#include "level1/ref_axpyf.h"
#include "inc/check_error.h"
#include "common/testing_helpers.h"

/**
 * @brief Generic test body for axpby operation.
 */

// The function is templatized based on the datatype and function-pointer type to the kernel.
template<typename T, typename FT>
static void test_axpyf_ukr( FT ukr_fp, char conjA, char conjx, gtint_t m, gtint_t b_fuse,
                            T alpha, gtint_t inca, gtint_t lda_inc, gtint_t incx, gtint_t incy,
                            double thresh, bool is_memory_test = false )
{
    // Pointers to obtain the required memory.
    T *A, *x, *y, *y_ref;

    // Compute the leading dimensions of A matrix.
    gtint_t lda = testinghelpers::get_leading_dimension( 'c', 'n', m, b_fuse, lda_inc, inca );

    // Compute the sizes required to allocate memory for the operands
    gtint_t size_A = lda * b_fuse * sizeof( T );
    gtint_t size_x = testinghelpers::buff_dim( b_fuse, incx ) * sizeof( T );
    gtint_t size_y = testinghelpers::buff_dim( m, incy ) * sizeof( T );

    // Create the objects for the input and output operands
    // The kernel does not expect the memory to be aligned
    testinghelpers::ProtectedBuffer A_buffer( size_A, false, false );
    testinghelpers::ProtectedBuffer x_buffer( size_x, false, is_memory_test );
    testinghelpers::ProtectedBuffer y_buffer( size_y, false, is_memory_test );

    // For y_ref, we don't need different greenzones and any redzone.
    // Thus, we pass is_memory_test as false
    testinghelpers::ProtectedBuffer y_ref_buffer( size_y, false, false );

    // Acquire the first set of greenzones for A, x and y
    A = ( T* )A_buffer.greenzone_1;
    x = ( T* )x_buffer.greenzone_1;
    y = ( T* )y_buffer.greenzone_1;
    y_ref = ( T* )y_ref_buffer.greenzone_1;   // For y_ref, there is no greenzone_2

    // Initialize the memory with random data
    testinghelpers::datagenerators::randomgenerators( -2, 8, 'c', m, b_fuse, A, 'n', lda );
    testinghelpers::datagenerators::randomgenerators( -10, 10, b_fuse, incx, x );
    testinghelpers::datagenerators::randomgenerators( -10, 10, m, incy, y );

    // Copying the contents of y to y_ref
    memcpy( y_ref, y, size_y );

    // Char conjA and conjx to BLIS conjA and conjx conversion
    conj_t blis_conjA, blis_conjx;
    testinghelpers::char_to_blis_conj( conjA, &blis_conjA );
    testinghelpers::char_to_blis_conj( conjx, &blis_conjx );

    // Add signal handler for segmentation fault
    testinghelpers::ProtectedBuffer::start_signal_handler();
    try
    {
        // Call the ukr function.
        // This call is made irrespective of is_memory_test.
        // This will check for out of bounds access with first redzone(if memory test is true)
        // Else, it will just call the ukr function.
        ukr_fp
        (
          blis_conjA, blis_conjx,
          m, b_fuse, &alpha,
          A, inca, lda, x, incx,
          y, incy, nullptr
        );

        if ( is_memory_test )
        {
            // Acquire the pointers near the second redzone
            A = ( T* )A_buffer.greenzone_2;
            x = ( T* )x_buffer.greenzone_2;
            y = ( T* )y_buffer.greenzone_2;

            // Copy the data for A, x and y accordingly
            memcpy( A, A_buffer.greenzone_1, size_A );
            memcpy( x, x_buffer.greenzone_1, size_x );
            memcpy( y, y_ref, size_y );

            // Call the ukr function, to check with the second redzone.
            ukr_fp
            (
              blis_conjA, blis_conjx,
              m, b_fuse, &alpha,
              A, inca, lda, x, incx,
              y, incy, nullptr
            );
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
    testinghelpers::ref_axpyf<T>
                    (
                      conjA, conjx, m, b_fuse,
                      &alpha, A, inca, lda,
                      x, incx, y_ref, incy
                    );

    //----------------------------------------------------------
    //              Compute component-wise error.
    //----------------------------------------------------------
    computediff<T>( "y", m, y, y_ref, incy, thresh );
}

// Test-case logger : Used to print the test-case details for unit testing the kernels.
// NOTE : The kernel name is the prefix in instantiator name, and thus is not printed
// with this logger.
template<typename T, typename FT>
class axpyfUkrPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<FT,char,char,gtint_t,gtint_t,T,gtint_t,gtint_t,gtint_t,gtint_t,bool>> str) const {
        char conjA      = std::get<1>(str.param);
        char conjx      = std::get<2>(str.param);
        gtint_t m       = std::get<3>(str.param);
        gtint_t b_fuse  = std::get<4>(str.param);
        T alpha         = std::get<5>(str.param);
        gtint_t inca    = std::get<6>(str.param);
        gtint_t lda_inc = std::get<7>(str.param);
        gtint_t incx    = std::get<8>(str.param);
        gtint_t incy    = std::get<9>(str.param);
        bool is_memory_test = std::get<10>(str.param);

        std::string str_name = "";
        str_name += "_m_" + std::to_string(m);
        str_name += "_bf_" + std::to_string(b_fuse);
        str_name += "_conja_" + std::string(&conjA, 1);
        str_name += "_conjx_" + std::string(&conjx, 1);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name += "_inca_" + testinghelpers::get_value_string(inca);
        gtint_t lda = testinghelpers::get_leading_dimension( 'c', 'n', m, b_fuse, lda_inc, inca );
        str_name += "_lda_i" + testinghelpers::get_value_string(lda_inc) + "_" + std::to_string(lda);;
        str_name += "_incx_" + testinghelpers::get_value_string(incx);
        str_name += "_incy_" + testinghelpers::get_value_string(incy);
        str_name += ( is_memory_test ) ? "_mem_test_enabled" : "_mem_test_disabled";
        return str_name;
    }
};
