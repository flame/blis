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

// Function-pointer types for the GEMV micro-kernels/tiled callers under test.
// All GEMV kernels share the same BLIS-typed signature, taking trans_t as the
// first argument. These are declared here (rather than in blis.h) so the ukr
// test harnesses can parameterize over kernel pointers of the correct type.
typedef void (*sgemv_ker)( trans_t, conj_t, dim_t, dim_t, float*,    float*,    inc_t, inc_t, float*,    inc_t, float*,    float*,    inc_t, cntx_t* );
typedef void (*dgemv_ker)( trans_t, conj_t, dim_t, dim_t, double*,   double*,   inc_t, inc_t, double*,   inc_t, double*,   double*,   inc_t, cntx_t* );
typedef void (*cgemv_ker)( trans_t, conj_t, dim_t, dim_t, scomplex*, scomplex*, inc_t, inc_t, scomplex*, inc_t, scomplex*, scomplex*, inc_t, cntx_t* );
typedef void (*zgemv_ker)( trans_t, conj_t, dim_t, dim_t, dcomplex*, dcomplex*, inc_t, inc_t, dcomplex*, inc_t, dcomplex*, dcomplex*, inc_t, cntx_t* );

// The tiled single-thread callers and the multi-thread wrappers are compiled
// into the library but are not prototyped in the public blis.h. The ukr tests
// reference them directly, so forward-declare them here with C linkage. These
// are harmless if the corresponding kernel is absent from the build under test:
// instantiation is gated by the K_* macros, so an unreferenced declaration
// never triggers a link requirement.
#define GEMV_UKR_KER_PROT( ctype, fname ) \
    void fname( trans_t, conj_t, dim_t, dim_t, ctype*, ctype*, inc_t, inc_t, \
                ctype*, inc_t, ctype*, ctype*, inc_t, cntx_t* );
#ifdef __cplusplus
extern "C" {
#endif
// --- float (AVX2 / AVX-512) ---
GEMV_UKR_KER_PROT( float, bli_sgemv_t_zen_int_24x4 )
GEMV_UKR_KER_PROT( float, bli_sgemv_t_zen_int_24x4_mt )
GEMV_UKR_KER_PROT( float, bli_sgemv_n_zen_int_40x4 )
GEMV_UKR_KER_PROT( float, bli_sgemv_n_zen_int_40x4_mt )
GEMV_UKR_KER_PROT( float, bli_sgemv_m_zen_int_40x4 )
GEMV_UKR_KER_PROT( float, bli_sgemv_m_zen_int_40x4_mt_Mdiv )
GEMV_UKR_KER_PROT( float, bli_sgemv_m_zen_int_40x4_mt_Ndiv )
GEMV_UKR_KER_PROT( float, bli_sgemv_t_zen4_int_48x8 )
GEMV_UKR_KER_PROT( float, bli_sgemv_t_zen4_int_48x8_mt )
GEMV_UKR_KER_PROT( float, bli_sgemv_n_zen4_int_80x8 )
GEMV_UKR_KER_PROT( float, bli_sgemv_n_zen4_int_80x8_mt )
GEMV_UKR_KER_PROT( float, bli_sgemv_m_zen4_int_80x8 )
GEMV_UKR_KER_PROT( float, bli_sgemv_m_zen4_int_80x8_mt_Mdiv )
GEMV_UKR_KER_PROT( float, bli_sgemv_m_zen4_int_80x8_mt_Ndiv )
// --- double (AVX2 / AVX-512) ---
GEMV_UKR_KER_PROT( double, bli_dgemv_t_zen_int_16x4 )
GEMV_UKR_KER_PROT( double, bli_dgemv_t_zen_int_16x4_mt )
GEMV_UKR_KER_PROT( double, bli_dgemv_n_zen_int_20x4 )
GEMV_UKR_KER_PROT( double, bli_dgemv_n_zen_int_20x4_mt )
GEMV_UKR_KER_PROT( double, bli_dgemv_m_zen_int_20x4 )
GEMV_UKR_KER_PROT( double, bli_dgemv_t_zen4_int_32x8 )
GEMV_UKR_KER_PROT( double, bli_dgemv_t_zen4_int_32x8_mt )
GEMV_UKR_KER_PROT( double, bli_dgemv_n_zen4_int_40x8 )
GEMV_UKR_KER_PROT( double, bli_dgemv_n_zen4_int_40x8_mt )
// --- scomplex (AVX2 / AVX-512) ---
GEMV_UKR_KER_PROT( scomplex, bli_cgemv_t_zen_int_20x4 )
GEMV_UKR_KER_PROT( scomplex, bli_cgemv_t_zen_int_20x4_mt )
GEMV_UKR_KER_PROT( scomplex, bli_cgemv_n_zen_int_20x5 )
GEMV_UKR_KER_PROT( scomplex, bli_cgemv_n_zen_int_20x5_mt )
GEMV_UKR_KER_PROT( scomplex, bli_cgemv_t_zen4_int_40x8 )
GEMV_UKR_KER_PROT( scomplex, bli_cgemv_t_zen4_int_40x8_mt )
GEMV_UKR_KER_PROT( scomplex, bli_cgemv_n_zen4_int_40x10 )
GEMV_UKR_KER_PROT( scomplex, bli_cgemv_n_zen4_int_40x10_mt )
// --- dcomplex (AVX2 / AVX-512) ---
GEMV_UKR_KER_PROT( dcomplex, bli_zgemv_t_zen_int_10x4 )
GEMV_UKR_KER_PROT( dcomplex, bli_zgemv_t_zen_int_10x4_mt )
GEMV_UKR_KER_PROT( dcomplex, bli_zgemv_n_zen_int_10x5 )
GEMV_UKR_KER_PROT( dcomplex, bli_zgemv_n_zen_int_10x5_mt )
GEMV_UKR_KER_PROT( dcomplex, bli_zgemv_t_zen4_int_20x8 )
GEMV_UKR_KER_PROT( dcomplex, bli_zgemv_t_zen4_int_20x8_mt )
GEMV_UKR_KER_PROT( dcomplex, bli_zgemv_n_zen4_int_20x10 )
GEMV_UKR_KER_PROT( dcomplex, bli_zgemv_n_zen4_int_20x10_mt )
#ifdef __cplusplus
}
#endif
#undef GEMV_UKR_KER_PROT

template<typename T, typename FT>
static void test_gemv_ukr_conja( FT ukr_fp, char storage, char transa, char conjx, gtint_t m, gtint_t n,
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

    // Creating cntx. The GEMV entry-points look up L1V kernels (scalv/copyv)
    // from the context for the alpha==0 and conj-x/incy-buffering paths, so a
    // valid context is required (the framework always supplies one).
    cntx_t* cntx = bli_gks_query_cntx();

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


template<typename T, typename FT>
static void test_gemv_ukr_transa( FT ukr_fp, char storage, char transa, char conjx, gtint_t m, gtint_t n,
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

    // Creating cntx. The GEMV entry-points look up L1V kernels (scalv/copyv)
    // from the context for the alpha==0 and conj-x/incy-buffering paths, so a
    // valid context is required (the framework always supplies one).
    cntx_t* cntx = bli_gks_query_cntx();

    // Copying the contents of y to y_ref
    memcpy( y_ref, y, size_y );

    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    testinghelpers::ProtectedBuffer::start_signal_handler();
    try
    {
        ukr_fp( blis_transa, blis_conjx, m, n, &alpha, a, 1, lda, x, incx, &beta,
                 y, incy, cntx );

        if ( is_memory_test )
        {
            memcpy((a_buf.greenzone_2), (a_buf.greenzone_1), size_a);
            memcpy((x_buf.greenzone_2), (x_buf.greenzone_1), size_x);
            memcpy((y_buf.greenzone_2), y_ref, size_y);

            ukr_fp( blis_transa, blis_conjx, m, n, &alpha,
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
