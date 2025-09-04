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
#include "blis.h"
#include "level3/trsm/trsm.h"
#include "level3/ref_trsm.h"
#include "inc/check_error.h"
#include "common/testing_helpers.h"
#include "level3/trsm/test_trsm.h"


// function pointer for TRSM small kernels
typedef err_t (*trsm_small_ker_ft)
(
    side_t   side,
    obj_t*   alpha,
    obj_t*   a,
    obj_t*   b,
    cntx_t*  cntx,
    cntl_t*  cntl,
    bool     is_parallel
);

/*
* Function to test gemmtrsm ukr
*/
template<typename T, typename FT>
static void test_trsm_ukr( FT ukr_fp, char storage, char uploa, char diaga,
                          gtint_t m, gtint_t n, gtint_t k, T alpha,
                          gtint_t ldc_inc, double thresh, bool is_memory_test)
{
    gtint_t lda = m, ldb = n;
    gtint_t ldc = ldc_inc;


    // Allocate memory for A10(k*lda) and A11(m*lda)
    testinghelpers::ProtectedBuffer a10_buffer( (k+m) * lda * sizeof(T), false, is_memory_test );
    // Allocate aligned memory for B01(k*ldb) and B11(m*ldb)
    testinghelpers::ProtectedBuffer b01_buffer( (k+m) * ldb * sizeof(T), true , is_memory_test );


    T* a10 = (T*)a10_buffer.greenzone_1; // column major
    T* b01 = (T*)b01_buffer.greenzone_1; // row major

    // Initialize vectors with random numbers.
    random_generator_with_INF_NAN( a10, uploa, 'c', 'n', -0.1, 0.1, m, (k+m), lda);
    random_generator_with_INF_NAN( b01, uploa, 'r', 'n', -0.1, 0.1, (k+m), n, ldb);

    // Get A11(A10 + sizeof(A01)) and B11(B10 + sizeof(B10))
    T* a11  = a10 + (k*lda);
    T* b11  = b01 + (k*ldb);

    // make A11 triangular for trsm
    testinghelpers::make_triangular<T>( 'c', uploa, m, a11, lda );

    T* c, *c_ref, *b11_copy;
    gtint_t rs_c, cs_c, rs_c_ref, cs_c_ref;
    gtint_t size_c, size_c_ref;

    // allocate memory for C according to the storage scheme
    if (storage == 'r' || storage == 'R')
    {
        ldc += n;
        rs_c = ldc;
        cs_c = 1;
        rs_c_ref = rs_c;
        cs_c_ref = cs_c;
        size_c = ldc * m * sizeof(T);
        size_c_ref = size_c;
    }
    else if (storage == 'c' || storage == 'C')
    {
        ldc += m;
        rs_c = 1;
        cs_c = ldc;
        rs_c_ref = rs_c;
        cs_c_ref = cs_c;
        size_c = ldc * n * sizeof(T);
        size_c_ref = size_c;
    }
    else // general storage
    {
        ldc += m;

        // reference does not support general stride, therefore
        // reference is set as column major
        rs_c_ref = 1,
        cs_c_ref = ldc;

        // for general stride, rs_c and cs_c both are non unit stride
        // ldc is used to derieve both rs_c and cs_c
        rs_c = ldc;
        cs_c = ldc*ldc;
        size_c = ldc * n * ldc * sizeof(T);
        size_c_ref = ldc * n * 1   * sizeof(T);
    }

    // get memory for C and c_ref
    testinghelpers::ProtectedBuffer c_buffer(size_c, false, is_memory_test);
    c     = (T*)c_buffer.greenzone_1;
    c_ref = (T*)malloc( size_c_ref );

    // set c buffers to zero to ensure the unused region of C matrix (extra ldb) is zero
    memset(c,   0, size_c);
    memset(c_ref, 0, size_c_ref);

    // copy contents of B11 to C and C_ref
    for (gtint_t i = 0; i < m; ++i)
    {
        for (gtint_t j = 0; j < n; ++j)
        {
            c[j*cs_c + i*rs_c] = b11[i*ldb + j];
            c_ref[j*cs_c_ref + i*rs_c_ref] = b11[i*ldb + j];
        }
    }

    // Make A11 diagonal dominant in order to make sure that
    // input matrics are solvable
    // In case BLIS_ENABLE_TRSM_PREINVERSION is enabled,
    // diagonal elements of A11 have to be inverted twice,
    // once for making it diagonal dominant, and once for packing with
    // inversion, inverting it twice is equivalent to not inverting it at all.
    // Therefore, in case of BLIS_ENABLE_TRSM_PREINVERSION, diagonal elements
    // of A11 are not inverted.
#ifndef BLIS_ENABLE_TRSM_PREINVERSION
    for (gtint_t i =0;i< m; i++)
    {
        a11[i+i*lda] = T{1} / a11[i+i*lda];
    }
#endif

    // If A is unit diagonal, set diagonal elements of A11 to 1
    if (diaga == 'u' || diaga == 'U')
    {
        for (gtint_t i =0;i< m; i++)
        {
            a11[i+i*lda] = T{1};
        }
    }

    // add signal handler for segmentation fault
    testinghelpers::ProtectedBuffer::start_signal_handler();
    try
    {
        if ( is_memory_test )
        {
            // calling gemmtrsm ukr will modify b11 buffer
            // create a copy of B11 so that it can be restored
            // for the second call of gemmtrsm ukr
            b11_copy = (T*)malloc( m*ldb*sizeof(T) );
            memcpy(b11_copy, b11, m*ldb*sizeof(T));
        }

        // Call ukr
        ukr_fp
        (
            k,
            &alpha,
            a10, a11,
            b01, b11,
            c,
            rs_c, cs_c,
            nullptr, nullptr
        );
        if ( is_memory_test )
        {
            // set pointers to second buffer
            c =   (T*)c_buffer.greenzone_2;
            a10 = (T*)a10_buffer.greenzone_2;
            b01 = (T*)b01_buffer.greenzone_2;
            a11  = a10 + (k*lda);
            b11  = b01 + (k*ldb);

            // copy data from 1st buffer of A and B to second buffer
            memcpy(a10, a10_buffer.greenzone_1, (k+m) * lda * sizeof(T));
            memcpy(b01, b01_buffer.greenzone_1, k * ldb * sizeof(T));

            memset(c,     0, size_c);
            // restore B11 and copy contents of B11 to C
            for (gtint_t i = 0; i < m; ++i)
            {
                for (gtint_t j = 0; j < n; ++j)
                {
                    b11[i*ldb + j] = b11_copy[i*ldb + j];
                    c[j*cs_c + i*rs_c] = b11_copy[i*ldb + j];
                }
            }
            // free b11_copy
            free(b11_copy);

            // second call to ukr
            ukr_fp( k, &alpha, a10, a11, b01, b11, c, rs_c, cs_c, nullptr, nullptr );
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


#ifdef BLIS_ENABLE_TRSM_PREINVERSION
    // compensate for the trsm per-inversion
    for (gtint_t i =0;i< m; i++)
    {
        a11[i+i*lda] = T{1.0} / a11[i+i*lda];
    }
#endif

    // Call reference implementation to get ref results.
    if (storage == 'c' || storage == 'C')
    {
        testinghelpers::ref_gemm<T>( storage, 'n', 't', m, n, k, T{-1},
                                a10, lda, b01, ldb, alpha, c_ref, ldc);
        testinghelpers::ref_trsm<T>( storage, 'l', uploa, 'n', diaga, m, n, T{1}, a11,
                                lda, c_ref, ldc );
    }
    else if (storage == 'r' || storage == 'R')// row major
    {
        testinghelpers::ref_gemm<T>( storage, 't', 'n', m, n, k, T{-1},
                                a10, lda, b01, ldb, alpha, c_ref, ldc);

        // convert col major A11 to row Major for TRSM
        T temp = T{0};
        for(gtint_t i = 0; i < m; ++i)
        {
            for(gtint_t j = i; j< m; ++j)
            {
                temp = a11[i+j*lda];
                a11[i+j*lda] = a11[j+i*lda];
                a11[j+i*lda] = temp;
            }
        }

        testinghelpers::ref_trsm<T>( storage, 'l', uploa, 'n', diaga, m, n, T{1}, a11,
                                lda, c_ref, ldc );
    }
    else
    {
        testinghelpers::ref_gemm<T>( 'c', 'n', 't', m, n, k, T{-1},
                                a10, lda, b01, ldb, alpha, c_ref, ldc);
        testinghelpers::ref_trsm<T>( 'c', 'l', uploa, 'n', diaga, m, n, T{1}, a11,
                                lda, c_ref, ldc );

        // there is no equivalent blas call for gen storage,
        // in order to compare the gen stored C and column major stored
        // create a column major copy of C
        T* c_gs = (T*)malloc( ldc * n * 1   * sizeof(T) );
        memset(c_gs, 0, ldc * n * 1   * sizeof(T));

        for (gtint_t i = 0; i < m; ++i)
        {
            for (gtint_t j = 0; j < n; ++j)
            {
                c_gs[i*rs_c_ref + j*cs_c_ref] = c[i*rs_c + j*cs_c];
            }
        }

        c = c_gs;
    }

    // Compute component-wise error.
    computediff<T>( "C", storage, m, n, c, c_ref, ldc, thresh );

    if(storage != 'r' && storage != 'R' && storage != 'c' && storage != 'C')
    {
        // free c_gs in case of general stride
        free(c);
    }

    // free buffers
    free(c_ref);
}

template<typename T, typename FT>
static void test_trsm_small_ukr( FT ukr_fp, char side, char uploa, char diaga,
                        char transa, gtint_t m, gtint_t n, T alpha, gtint_t lda,
                        gtint_t ldb, double thresh, bool is_memory_test, num_t dt)
{
    // create blis objects
    obj_t ao = BLIS_OBJECT_INITIALIZER;
    obj_t bo = BLIS_OBJECT_INITIALIZER;
    obj_t alphao = BLIS_OBJECT_INITIALIZER_1X1;

    inc_t rs_a = 1;
    inc_t cs_a = lda;
    inc_t rs_b = 1;
    inc_t cs_b = ldb;

    side_t  blis_side;
    uplo_t  blis_uploa;
    trans_t blis_transa;
    diag_t  blis_diaga;
    dim_t   m0, n0;
    dim_t   mn0_a;
    bli_convert_blas_dim1( m, m0 );
    bli_convert_blas_dim1( n, n0 );

    bli_param_map_netlib_to_blis_side( side,  &blis_side );
    bli_param_map_netlib_to_blis_uplo( uploa, &blis_uploa );
    bli_param_map_netlib_to_blis_trans( transa, &blis_transa );
    bli_param_map_netlib_to_blis_diag( diaga, &blis_diaga );

    bli_set_dim_with_side( blis_side, m0, n0, &mn0_a );
    bli_obj_init_finish_1x1( dt, (T*)&alpha, &alphao );

    cs_a += mn0_a;
    cs_b += m;

    // Allocate memory for A (col major)
    testinghelpers::ProtectedBuffer a_buf( mn0_a * cs_a * sizeof(T), false, is_memory_test );
    // Allocate memory for B (col major)
    testinghelpers::ProtectedBuffer b_buf( n * cs_b * sizeof(T), false, is_memory_test );

    T* a = (T*)a_buf.greenzone_1;
    T* b = (T*)b_buf.greenzone_1;
    T* b_ref = (T*)malloc( n * cs_b * sizeof(T) ); // col major

    // Initialize buffers with random numbers.
    random_generator_with_INF_NAN( a, uploa, 'c', 'n', -0.1, 0.1, mn0_a, mn0_a, cs_a);
    random_generator_with_INF_NAN( b, uploa, 'c', 'n', -0.1, 0.1, m, n, cs_b);

    // copy contents of b to b_ref
    memcpy(b_ref, b, n * cs_b * sizeof(T));

    // make A triangular
    testinghelpers::make_triangular<T>( 'c', uploa, mn0_a, a, cs_a );

    // Make A11 diagonal dominant in order to make sure that
    // input matrics are solvable
    for (gtint_t i = 0; i < mn0_a; i++)
    {
        a[i+i*cs_a] = T{1} / a[i+i*cs_a];
    }

    bli_obj_init_finish( dt, mn0_a, mn0_a, (T*)a, rs_a, cs_a, &ao );
    bli_obj_init_finish( dt, m0,    n0,    (T*)b, rs_b, cs_b, &bo );

    const struc_t struca = BLIS_TRIANGULAR;

    bli_obj_set_uplo( blis_uploa, &ao );
    bli_obj_set_diag( blis_diaga, &ao );
    bli_obj_set_conjtrans( blis_transa, &ao );
    bli_obj_set_struc( struca, &ao );

    // add signal handler for segmentation fault
    testinghelpers::ProtectedBuffer::start_signal_handler();
    try
    {
        // call trsm small kernel
        ukr_fp(blis_side, &alphao, &ao, &bo, NULL, NULL, false);
        if ( is_memory_test )
        {
            // set A and B pointers to second buffer
            a = (T*)a_buf.greenzone_2;
            b = (T*)b_buf.greenzone_2;

            // copy data from first buffers of A and B to second buffer
            memcpy(b, b_ref, n * cs_b * sizeof(T));
            memcpy(a, (T*)a_buf.greenzone_1, mn0_a * cs_a * sizeof(T));
            bli_obj_init_finish( dt, m0, n0, (T*)b, rs_b, cs_b, &bo );
            bli_obj_init_finish( dt, mn0_a, mn0_a, (T*)a, rs_a, cs_a, &ao );

            // call trsm small kernel
            ukr_fp(blis_side, &alphao, &ao, &bo, NULL, NULL, false);
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

    // call to reference trsm
    testinghelpers::ref_trsm<T>( 'c', side, uploa, transa, diaga, m, n, alpha, a,
                                cs_a, b_ref, cs_b );

    computediff<T>( "B", 'c', m, n, b, b_ref, cs_b, thresh );

    // free memory
    free(b_ref);
}

// Test-case logger : Used to print the test-case details based on parameters
template <typename T1, typename T2>
class trsmSmallUKRPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<T2, char, char, char, char, gtint_t,
                                            gtint_t, T1, gtint_t, gtint_t, bool>> str) const{
        char side               = std::get<1>(str.param);
        char uploa              = std::get<2>(str.param);
        char diaga              = std::get<3>(str.param);
        char transa             = std::get<4>(str.param);
        gtint_t m               = std::get<5>(str.param);
        gtint_t n               = std::get<6>(str.param);
        T1  alpha               = std::get<7>(str.param);
        gtint_t lda_inc         = std::get<8>(str.param);
        gtint_t ldb_inc         = std::get<9>(str.param);
        bool is_memory_test     = std::get<10>(str.param);

        std::string str_name = "";
        str_name += "_side_" + std::string(&side, 1);
        str_name += "_uplo_" + std::string(&uploa, 1);
        str_name += "_transa_" + std::string(&transa, 1);
        str_name += "_diag_" + std::string(&diaga, 1);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name += "_m_" + std::to_string(m);
        str_name += "_n_" + std::to_string(n);
        gtint_t mn;
        testinghelpers::set_dim_with_side( side, m, n, &mn );
        gtint_t lda = lda_inc + mn;
        gtint_t ldb = ldb_inc + m;
        str_name += "_lda_i" + std::to_string(lda_inc) + "_" + std::to_string(lda);
        str_name += "_ldb_i" + std::to_string(ldb_inc) + "_" + std::to_string(ldb);
        str_name += ( is_memory_test ) ? "_mem_test_enabled" : "_mem_test_disabled";
        return str_name;
    }
};

template <typename T1, typename T2>
class trsmNatUKRPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<T2, char, char, char, gtint_t,
                                            gtint_t, gtint_t, T1, gtint_t, bool>> str) const{
        char storage            = std::get<1>(str.param);
        char uploa              = std::get<2>(str.param);
        char diaga              = std::get<3>(str.param);
        gtint_t m               = std::get<4>(str.param);
        gtint_t n               = std::get<5>(str.param);
        gtint_t k               = std::get<6>(str.param);
        T1  alpha               = std::get<7>(str.param);
        gtint_t ldc_inc         = std::get<8>(str.param);
        bool is_memory_test     = std::get<9>(str.param);

        std::string str_name = "";
        str_name += "_stor_" + std::string(&storage, 1);
        str_name += "_uplo_" + std::string(&uploa, 1);
        str_name += "_diag_" + std::string(&diaga, 1);
        str_name += "_alpha_" + testinghelpers::get_value_string(alpha);
        str_name += "_m_" + std::to_string(m);
        str_name += "_n_" + std::to_string(n);
        str_name += "_k_" + std::to_string(k);
        gtint_t ldc = testinghelpers::get_leading_dimension( storage, 'n', m, n, ldc_inc );
        str_name += "_ldc_i" + std::to_string(ldc_inc) + "_" + std::to_string(ldc);
        str_name += ( is_memory_test ) ? "_mem_test_enabled" : "_mem_test_disabled";
        return str_name;
    }
};
