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

#include "imatcopy.h"
#include "extension/ref_imatcopy.h"
#include "inc/check_error.h"

/**
 * @brief Generic test body for imatcopy operation.
 */

template<typename T>
static void test_imatcopy( char storage, char trans, gtint_t m, gtint_t n, T alpha, gtint_t lda_in_inc, gtint_t lda_out_inc,
                          double thresh, bool is_memory_test = false, bool is_nan_inf_test = false, T exval = T{0.0} )
{
    // Set an alternative trans value that corresponds to only
    // whether the A matrix(output) should be mxn or nxm(only transposing)
    char A_out_trans;
    A_out_trans = ( ( trans == 'n' ) || ( trans == 'r' ) )? 'n' : 't';

    // Compute the leading dimensions of A(input) and A(output).
    gtint_t lda_in = testinghelpers::get_leading_dimension( storage, 'n', m, n, lda_in_inc );
    gtint_t lda_out = testinghelpers::get_leading_dimension( storage, A_out_trans, m, n, lda_out_inc );

    // Compute sizes of A(input) and A(output), in bytes
    gtint_t size_a_in = testinghelpers::matsize( storage, 'n', m, n, lda_in ) * sizeof( T );
    gtint_t size_a_out = testinghelpers::matsize( storage, A_out_trans, m, n, lda_out ) * sizeof( T );

    // A has to allocated the maximum of input and output sizes, for API compatibility
    gtint_t size_a = (std::max)( size_a_in, size_a_out );

    // Create the objects for the input and output operands
    // The API does not expect the memory to be aligned
    testinghelpers::ProtectedBuffer A_buf( size_a, false, is_memory_test );
    testinghelpers::ProtectedBuffer A_ref_buf( size_a, false, false );

    // Pointers to access the memory chunks
    T *A, *A_ref;

    // Acquire the first set of greenzones for A and A_ref
    A = ( T* )A_buf.greenzone_1;
    A_ref = ( T* )A_ref_buf.greenzone_1; // For A_ref, there is no greenzone_2

    // Initialize the memory with random data
    testinghelpers::datagenerators::randomgenerators( -10, 10, storage, m, n, A, 'n', lda_in );

    if( is_nan_inf_test )
    {
      gtint_t rand_m = rand() % m;
      gtint_t rand_n = rand() % n;
      gtint_t idx = ( storage == 'c' || storage == 'C' )? ( rand_m + rand_n * lda_in ) : ( rand_n + rand_m * lda_in );

      A[idx] = exval;
    }

    // Copying the contents of A to A_ref
    memcpy( A_ref, A, size_a );

    // Add signal handler for segmentation fault
    testinghelpers::ProtectedBuffer::start_signal_handler();
    try
    {
        // Call the API.
        // This call is made irrespective of is_memory_test.
        // This will check for out of bounds access with first redzone(if memory test is true)
        // Else, it will just call the ukr function.
        imatcopy<T>( trans, m, n, alpha, A, lda_in, lda_out );

        if ( is_memory_test )
        {
            // Acquire the pointers near the second redzone
            A = ( T* )A_buf.greenzone_2;

            // Copy the data for A accordingly
            // NOTE : The object for A will have acquired enough memory
            //        such that the greenzones in each do not overlap.
            memcpy( A, A_ref, size_a );

            // Call the API, to check with the second redzone.
            imatcopy<T>( trans, m, n, alpha, A, lda_in, lda_out );
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
    testinghelpers::ref_imatcopy<T>( storage, trans, m, n, alpha, A_ref, lda_in, lda_out );

    //----------------------------------------------------------
    //              Compute component-wise error.
    //----------------------------------------------------------

    if( A_out_trans == 'n' )
      computediff<T>( "A", storage, m, n, A, A_ref, lda_out, thresh, is_nan_inf_test );
    else
      computediff<T>( "A", storage, n, m, A, A_ref, lda_out, thresh, is_nan_inf_test );

}

// Test-case logger : Used to print the test-case details based on parameters
template <typename T>
class imatcopyGenericPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char,char,gtint_t,gtint_t,T,gtint_t,gtint_t,bool>> str) const {
        char storage   = std::get<0>(str.param);
        char trans     = std::get<1>(str.param);
        gtint_t m      = std::get<2>(str.param);
        gtint_t n      = std::get<3>(str.param);
        T alpha        = std::get<4>(str.param);
        gtint_t lda_inc = std::get<5>(str.param);
        gtint_t ldb_inc = std::get<6>(str.param);
        bool is_memory_test = std::get<7>(str.param);

        std::string str_name = API_PRINT;
        str_name += "_stor_" + std::string(&storage, 1);
        str_name += "_trans_" + std::string(&trans, 1);
        str_name += "_m_" + std::to_string(m);
        str_name += "_n_" + std::to_string(n);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        char mat_trans = ( ( trans == 'n' ) || ( trans == 'r' ) )? 'n' : 't';
        gtint_t lda_in = testinghelpers::get_leading_dimension( storage, 'n', m, n, lda_inc );
        gtint_t lda_out = testinghelpers::get_leading_dimension( storage, mat_trans, m, n, ldb_inc );
        str_name += "_lda_in_" + std::to_string(lda_in);
        str_name += "_lda_out_" + std::to_string(lda_out);
        str_name += ( is_memory_test ) ? "_mem_test_enabled" : "_mem_test_disabled";

        return str_name;
    }
};

template <typename T>
class imatcopyEVTPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char,char,gtint_t,gtint_t,T,gtint_t,gtint_t,T,bool>> str) const {
        char storage    = std::get<0>(str.param);
        char trans      = std::get<1>(str.param);
        gtint_t m       = std::get<2>(str.param);
        gtint_t n       = std::get<3>(str.param);
        T alpha  = std::get<4>(str.param);
        gtint_t lda_inc = std::get<5>(str.param);
        gtint_t ldb_inc = std::get<6>(str.param);
        T exval  = std::get<7>(str.param);

        std::string str_name = API_PRINT;
        str_name += "_stor_" + std::string(&storage, 1);
        str_name += "_trans_" + std::string(&trans, 1);
        str_name += "_m_" + std::to_string(m);
        str_name += "_n_" + std::to_string(n);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name = str_name + "_A_exval_" + testinghelpers::get_value_string(exval);
        gtint_t lda = testinghelpers::get_leading_dimension( storage, 'n', m, n, lda_inc );
        gtint_t ldb = testinghelpers::get_leading_dimension( storage, trans, m, n, ldb_inc );
        str_name += "_lda" + std::to_string(lda);
        str_name += "_ldb" + std::to_string(ldb);

        return str_name;
    }
};
