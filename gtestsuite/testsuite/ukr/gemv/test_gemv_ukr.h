/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

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
#include "blis.h"
#include "level2/gemv/gemv.h"
#include "level2/ref_gemv.h"
#include "inc/check_error.h"
#include "common/testing_helpers.h"
#include <stdexcept>
#include <algorithm>

template<typename T, typename FT>
static void test_gemv_ukr( FT ukr_fp, char storage, char transa, char conjx, gtint_t m, gtint_t n,
                T alpha, gtint_t lda_inc, gtint_t incx, T beta, gtint_t incy,
                double thresh, bool is_memory_test = false )
{
    // Compute the leading dimensions for matrix size calculation.
    gtint_t lda = testinghelpers::get_leading_dimension( storage, 'n', m, n, lda_inc );

    dim_t size_a = testinghelpers::matsize( storage, 'n', m, n, lda ) * sizeof(T);
    
    // The second parameter is false, as we don't expect the memory to be aligned.
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
    if (beta != testinghelpers::ZERO<T>())
        testinghelpers::datagenerators::randomgenerators<T>( 1, 3, leny, incy, (T*)(y_buf.greenzone_1) );
    else
    {
        // Vector Y should not be read, only set.
        testinghelpers::set_vector( leny, incy, (T*)(y_buf.greenzone_1), testinghelpers::aocl_extreme<T>() );
    }

    T* a = (T*)(a_buf.greenzone_1);
    T* x = (T*)(x_buf.greenzone_1);
    T* y = (T*)(y_buf.greenzone_1);
    T* y_ref = ( T* )y_ref_buffer.greenzone_1; // For y_ref, there is no greenzone_2

    // Char conjx to BLIS conjx conversion
    conj_t blis_conjx;
    testinghelpers::char_to_blis_conj( conjx, &blis_conjx );

    // Char transa to BLIS transa conversion
    trans_t blis_transa;
    testinghelpers::char_to_blis_trans( transa, &blis_transa );

    // Getting conja from blis_transa
    conj_t conja = bli_extract_conj(blis_transa);

    // Creating cntx
    cntx_t* cntx = NULL;

    // Copying the contents of y to y_ref
    memcpy( y_ref, y, size_y );

    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    testinghelpers::ProtectedBuffer::start_signal_handler();
    try
    {
        ukr_fp( conja, blis_conjx, m, n, &alpha, a, 1, lda, x, incx, &beta,
                 y, incy, cntx );

        if ( is_memory_test )
        {
            memcpy((a_buf.greenzone_2), (a_buf.greenzone_1), size_a);
            memcpy((x_buf.greenzone_2), (x_buf.greenzone_1), size_x);
            memcpy((y_buf.greenzone_2), y_ref, size_y);

            ukr_fp( conja, blis_conjx, m, n, &alpha,
                    (T*)(a_buf.greenzone_2), 1,lda,
                    (T*)(x_buf.greenzone_2), incx,
                    &beta,
                    (T*)(y_buf.greenzone_2), incy, cntx );
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

#ifdef CAN_TEST_INFO_VALUE
    gtint_t info = bli_info_get_info_value();
    computediff<gtint_t>( "info", info, 0 );
#endif
}

// Test-case logger : Used to print the test-case details based on parameters
template <typename T1, typename T2>
class gemvUKRPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<T2,char,char,char,gtint_t,gtint_t,T1,T1,gtint_t,gtint_t,gtint_t,bool>> str) const {
        char storage        = std::get<1>(str.param);
        char transa         = std::get<2>(str.param);
        char conjx          = std::get<3>(str.param);
        gtint_t m           = std::get<4>(str.param);
        gtint_t n           = std::get<5>(str.param);
        T1 alpha            = std::get<6>(str.param);
        T1 beta             = std::get<7>(str.param);
        gtint_t incx        = std::get<8>(str.param);
        gtint_t incy        = std::get<9>(str.param);
        gtint_t lda_inc     = std::get<10>(str.param);
        bool is_memory_test = std::get<11>(str.param);

        std::string str_name = API_PRINT;
        str_name += "_stor_" + std::string(&storage, 1);
        str_name += "_transa_" + std::string(&transa, 1);
        str_name += "_conjx_" + std::string(&conjx, 1);
        str_name += "_m_" + std::to_string(m);
        str_name += "_n_" + std::to_string(n);
        str_name += "_incx_" + testinghelpers::get_value_string(incx);
        str_name += "_incy_" + testinghelpers::get_value_string(incy);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name += "_beta_" + testinghelpers::get_value_string(beta);
        gtint_t lda = testinghelpers::get_leading_dimension( storage, 'n', m, n, lda_inc );
        str_name += "_lda_i" + std::to_string(lda_inc) + "_" + std::to_string(lda);
        str_name += ( is_memory_test ) ? "_mem_test_enabled" : "_mem_test_disabled";
        return str_name;
    }
};
