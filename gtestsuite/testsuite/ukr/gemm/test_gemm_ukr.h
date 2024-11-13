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
#include "level3/ref_gemm.h"
#include "inc/check_error.h"
#include <stdexcept>
#include <algorithm>
#include "blis.h"
#include "common/blis_version_defs.h"

/**
 * @brief Generic test body for gemm operation.
 */

// The function is templatized based on the datatype and function-pointer type to the kernel.
template<typename T, typename FT>
static void test_gemmnat_ukr(
    char storage, gtint_t m, gtint_t n, gtint_t k, T alpha, T beta, FT ukr_fp, double thresh, bool is_memory_test = false )
{
    // In case of memory test:
    // Allocate packed buffer size for Matrix A, B native kernel works on packed buffer
    // Native kernel has preload or prebroadcase design
    // If we allocate size required by dimension then memtest fails
    obj_t a, b;
    obj_t ap, bp; // for packed buffers
    cntx_t* cntx;
    num_t dt = BLIS_DOUBLE;
    cntx = bli_gks_query_cntx();
    bli_obj_create(dt, m, k, 1, m, &a);
    bli_obj_create(dt, k, n, n, 1, &b);

    bli_obj_create(dt, m, k, 1, m, &ap);
    bli_obj_create(dt, k, n, n, 1, &bp);

#ifdef AOCL_42
    gtint_t sizea = bli_packm_init_pack( BLIS_NO_INVERT_DIAG, BLIS_GEMM, BLIS_PACKED_ROW_PANELS,
                        BLIS_PACK_FWD_IF_UPPER, BLIS_PACK_FWD_IF_LOWER,
                        BLIS_MR, BLIS_KR, &a, &ap, cntx) * sizeof(T);
    gtint_t sizeb = bli_packm_init_pack( BLIS_NO_INVERT_DIAG, BLIS_GEMM, BLIS_PACKED_COL_PANELS,
                             BLIS_PACK_FWD_IF_UPPER, BLIS_PACK_FWD_IF_LOWER,
                             BLIS_KR, BLIS_NR, &b, &bp, cntx ) * sizeof(T);
#else
    gtint_t sizea = bli_packm_init_pack( BLIS_NO_INVERT_DIAG, BLIS_PACKED_ROW_PANELS,
                        BLIS_PACK_FWD_IF_UPPER, BLIS_PACK_FWD_IF_LOWER,
                        BLIS_MR, BLIS_KR, &a, &ap, cntx) * sizeof(T);
    gtint_t sizeb = bli_packm_init_pack( BLIS_NO_INVERT_DIAG, BLIS_PACKED_COL_PANELS,
                             BLIS_PACK_FWD_IF_UPPER, BLIS_PACK_FWD_IF_LOWER,
                             BLIS_KR, BLIS_NR, &b, &bp, cntx ) * sizeof(T);
#endif

    // Create test operands
    // matrix A will be in col-storage
    // matrix B will be in row-storage
    // column * row = matrix -- rank-k update

    // Set matrix A dimensions
    gtint_t rs = 1;
    gtint_t cs = m;
    gtint_t lda = cs;
    //gtint_t sizea =  m * k * sizeof(T);

    // Set matrix B dimensions
    rs = n;
    cs = 1;
    gtint_t ldb = rs;
    //gtint_t sizeb =  k * n * sizeof(T);

    // Set matrix C dimensions
    gtint_t ldc  = m;
    if(storage == 'r' || storage == 'R')
    {
        rs = n;
        cs = 1;
        ldc = rs;
    }
    else
    {
        rs = 1;
        cs = m;
        ldc = cs;
    }
    gtint_t sizec =  m * n * sizeof(T);

    // Allocating aligned memory for A and B matrix as Native microkernel issues
    // VMOVAPD which expects memory to be accessed to be aligned.
    // Matrix C need not be aligned
    testinghelpers::ProtectedBuffer buf_a_ptrs( sizea, true, is_memory_test );
    testinghelpers::ProtectedBuffer buf_b_ptrs( sizeb, true, is_memory_test );
    testinghelpers::ProtectedBuffer buf_c_ptrs( sizec, false, is_memory_test );

    // Allocate memory for C Matrix used for reference computation
    testinghelpers::ProtectedBuffer buf_c_ref_ptrs( sizec, false , false );


    T* buf_a    = (T*)buf_a_ptrs.greenzone_1;
    T* buf_b    = (T*)buf_b_ptrs.greenzone_1;
    T* buf_c    = (T*)buf_c_ptrs.greenzone_1;
    T* buf_cref = (T*)buf_c_ref_ptrs.greenzone_1;

    /* Initialize Matrices with random numbers */
    testinghelpers::datagenerators::randomgenerators<T>( -2, 8, 'c', m, k, (T*)(buf_a), 'n', lda);
    testinghelpers::datagenerators::randomgenerators<T>( -5, 2, 'r', k, n, (T*)(buf_b), 'n', ldb);

    if (beta != testinghelpers::ZERO<T>())
        testinghelpers::datagenerators::randomgenerators<T>( -5, 2, storage , m, n, (T*)(buf_c), 'n', ldc);
    else
    {
        // Matrix C should not be read, only set.
        testinghelpers::set_matrix( storage, m, n, (T*)(buf_c), 'n', ldc, testinghelpers::aocl_extreme<T>() );
    }

    // Create a copy of c so that we can check reference results.
    memcpy(buf_cref, buf_c, sizec);

    /* Fill the auxinfo_t struct in case the micro-kernel uses it. */
    auxinfo_t data;
    bli_auxinfo_set_ps_a(0, &data);

    // add signal handler for segmentation fault
    testinghelpers::ProtectedBuffer::start_signal_handler();
    try
    {
        // call micro-kernel
        ukr_fp (
                k,
                &alpha,
                buf_a,
                buf_b,
                &beta,
                buf_c,
                rs,
                cs,
                &data,
                NULL
            );
        if ( is_memory_test )
        {
            // set pointers to second buffer
            buf_a    = (T*)buf_a_ptrs.greenzone_2;
            buf_b    = (T*)buf_b_ptrs.greenzone_2;
            buf_c    = (T*)buf_c_ptrs.greenzone_2;

            // copy data from 1st buffer of A and B to second buffer
            memcpy(buf_a, buf_a_ptrs.greenzone_1, sizea);
            memcpy(buf_b, buf_b_ptrs.greenzone_1, sizeb);

            //buf_c_ptrs.greenzone_1 has been updated with output from previous
            // gemm call, hence use buf_cref
            memcpy(buf_c, buf_cref, sizec);

            ukr_fp (
                    k,
                    &alpha,
                    buf_a,
                    buf_b,
                    &beta,
                    buf_c,
                    rs,
                    cs,
                    &data,
                    NULL
            );
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

    // In native micro-kernel
    // op(A) = No transpose & op(B) = transpose
    // for column-storage
    char transa = 'n';
    char transb = 't';

    // The objective here is to make storage of all matrices same
    // To do this we set transpose of A and B appropriately.
    if (storage == 'r' || storage == 'R')
    {
        // if row-storage
        transa = 't';
        transb = 'n';
        // because matrix A is created with col-storage
        // and matrix B is created with row-storage
        // Generally storage parameter in cblas signifies
        // storage of all matrices A, B and C.
        // since A is col-storage, A' will be row-storage
    }

    // call reference implementation
    testinghelpers::ref_gemm<T>( storage, transa, transb, m, n, k, alpha,
                                buf_a, lda, buf_b, ldb, beta, (T*)buf_cref, ldc);

    // Check component-wise error
    computediff<T>( "C", storage, m, n, (T*)buf_c, (T*)buf_cref, ldc, thresh );

}

// The function is templatized based on the datatype and function-pointer type to the kernel.
template<typename T, typename FT>
static void test_gemmk1_ukr( FT ukr_fp, gtint_t m, gtint_t n, gtint_t k, char storage, T alpha, T beta, double thresh, bool is_memory_test  = false )
{
    // Compute the leading dimensions of a, b, and c.
    //char storage = storageC;
    gtint_t lda = testinghelpers::get_leading_dimension( storage, 'n', m, k, 0 );
    gtint_t ldb = testinghelpers::get_leading_dimension( storage, 'n', k, n, 0 );
    gtint_t ldc = testinghelpers::get_leading_dimension( storage, 'n', m, n, 0 );

     //----------------------------------------------------------
    //         Initialize matrices with random numbers
    //----------------------------------------------------------
    gtint_t sizea =  testinghelpers::matsize( storage, 'n', m, k, lda ) * sizeof(T);
    gtint_t sizeb =  testinghelpers::matsize( storage, 'n', k, n, ldb ) * sizeof(T);
    gtint_t sizec =  testinghelpers::matsize( storage, 'n', m, n, ldc ) * sizeof(T);

    testinghelpers::ProtectedBuffer mat_a(sizea, false, is_memory_test);
    testinghelpers::ProtectedBuffer mat_b(sizeb, false, is_memory_test);
    testinghelpers::ProtectedBuffer mat_c(sizec, false, is_memory_test);
    testinghelpers::ProtectedBuffer mat_cref(sizec, false, false);

    T *buf_a = (T*)mat_a.greenzone_1;
    T *buf_b = (T*)mat_b.greenzone_1;
    T *buf_c = (T*)mat_c.greenzone_1;
    T* buf_cref = (T*)mat_cref.greenzone_1;

    // Check if the memory has been successfully allocated
    if ((buf_a == NULL) ||(buf_b == NULL) ||(buf_c == NULL) ||(buf_cref == NULL)) {
        printf("Memory not allocated for input and output Matrix.\n");
        return ;
    }
    testinghelpers::datagenerators::randomgenerators<T>( -2, 8, storage, m, k, (T*)(buf_a), 'n', lda);
    testinghelpers::datagenerators::randomgenerators<T>( -5, 2, storage, k, n, (T*)(buf_b), 'n', ldb);

    if (beta != testinghelpers::ZERO<T>())
        testinghelpers::datagenerators::randomgenerators<T>( -3, 5, storage , m, n, (T*)(buf_c), 'n', ldc);
    else
    {
        // Matrix C should not be read, only set.
        testinghelpers::set_matrix( storage, m, n, (T*)(buf_c), 'n', ldc, testinghelpers::aocl_extreme<T>() );
    }

    // Create a copy of c so that we can check reference results.
    memcpy(buf_cref, buf_c, sizec);

    // add signal handler for segmentation fault
    testinghelpers::ProtectedBuffer::start_signal_handler();
    try
    {
        // call micro-kernel
        ukr_fp (
            m,
            n,
            k,
            &alpha,
            buf_a,
            lda,
            buf_b,
            ldb,
            &beta,
            buf_c,
            ldc
            );

        if ( is_memory_test )
        {
            // set pointers to second buffer
            buf_a    = (T*)mat_a.greenzone_2;
            buf_b    = (T*)mat_b.greenzone_2;
            buf_c    = (T*)mat_c.greenzone_2;

            // Check if the memory has been successfully allocated
            if ((buf_a == NULL) || (buf_b == NULL) || (buf_c == NULL)) {
                printf("Memory not allocated for input or output Matrix for memory test.\n");
                return ;
            }

            // copy data from 1st buffer of A and B to second buffer
            memcpy(buf_a, mat_a.greenzone_1, sizea);
            memcpy(buf_b, mat_b.greenzone_1, sizeb);

            //buf_c_ptrs.greenzone_1 has been updated with output from previous
            // gemm call, hence use buf_cref
            memcpy(buf_c, buf_cref, sizec);

            // call micro-kernel
            ukr_fp (
                m,
                n,
                k,
                &alpha,
                buf_a,
                lda,
                buf_b,
                ldb,
                &beta,
                buf_c,
                ldc
                );
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

    // call reference implementation
    testinghelpers::ref_gemm<T>( storage, 'n', 'n', m, n, k, alpha,
                                 buf_a, lda, buf_b, ldb, beta, buf_cref, ldc);

    // Check component-wise error
    computediff<T>( "C", storage, m, n, buf_c, buf_cref, ldc, thresh );
}

template<typename T, typename FT>
static void test_gemmsup_ukr( FT ukr_fp, char trnsa, char trnsb, gtint_t m, gtint_t n, gtint_t k, T alpha, T beta,
                              char storageC, gtint_t MR, bool row_pref, double thresh, bool is_memory_test = false)
{
    // Compute the leading dimensions of a, b, and c.
    char storage = storageC;
    gtint_t lda = testinghelpers::get_leading_dimension( storage, trnsa, m, k, 0 );
    gtint_t ldb = testinghelpers::get_leading_dimension( storage, trnsb, k, n, 0 );
    gtint_t ldc = testinghelpers::get_leading_dimension( storage, 'n', m, n, 0 );

     //----------------------------------------------------------
    //         Initialize matrices with random numbers
    //----------------------------------------------------------
    gtint_t sizea =  testinghelpers::matsize( storage, trnsa, m, k, lda ) * sizeof(T);
    gtint_t sizeb =  testinghelpers::matsize( storage, trnsb, k, n, ldb ) * sizeof(T);
    gtint_t sizec =  testinghelpers::matsize( storage, 'n', m, n, ldc ) * sizeof(T);

    testinghelpers::ProtectedBuffer mat_a(sizea, false, is_memory_test);
    testinghelpers::ProtectedBuffer mat_b(sizeb, false, is_memory_test);
    testinghelpers::ProtectedBuffer mat_c(sizec, false, is_memory_test);
    testinghelpers::ProtectedBuffer mat_cref(sizec, false, false);

    T *buf_a = (T*)mat_a.greenzone_1;
    T *buf_b = (T*)mat_b.greenzone_1;
    T *buf_c = (T*)mat_c.greenzone_1;
    T *ref_c = (T*)mat_cref.greenzone_1;

    // Check if the memory has been successfully allocated
    if ((buf_a == NULL) ||(buf_b == NULL) ||(buf_c == NULL) ||(ref_c == NULL)) {
        printf("Memory not allocated for input and output Matrix.\n");
        return ;
    }
    testinghelpers::datagenerators::randomgenerators<T>( -2, 8, storage, m, k, (T*)(buf_a), trnsa, lda);
    testinghelpers::datagenerators::randomgenerators<T>( -5, 2, storage, k, n, (T*)(buf_b), trnsb, ldb);

    if (beta != testinghelpers::ZERO<T>())
        testinghelpers::datagenerators::randomgenerators<T>( -3, 5, storage , m, n, (T*)(buf_c), 'n', ldc);
    else
    {
        // Matrix C should not be read, only set.
        testinghelpers::set_matrix( storage, m, n, (T*)(buf_c), 'n', ldc, testinghelpers::aocl_extreme<T>() );
    }

    // Create a copy of c so that we can check reference results.
    memset(buf_c, 0, sizec);
    memset(ref_c, 0, sizec);
    inc_t str_id = 0;
    gtint_t rs_a = 1, cs_a = 1, rs_b = 1, cs_b = 1, rs_c = 1, cs_c = 1;
    gtint_t rs_a0 = 1, cs_a0 = 1, rs_b0 = 1, cs_b0 = 1;

    if(storage == 'r')
    {
        rs_a = lda;
        rs_b = ldb;
        rs_c = ldc;

        cs_a = 1;
        cs_b = 1;
        cs_c = 1;

        rs_a0 = lda;
        rs_b0 = ldb;

        cs_a0 = 1;
        cs_b0 = 1;
    }
    else
    {
        cs_a = lda;
        cs_b = ldb;
        cs_c = ldc;

        rs_a = 1;
        rs_b = 1;
        rs_c = 1;

        cs_a0 = lda;
        cs_b0 = ldb;

        rs_a0 = 1;
        rs_b0 = 1;
    }

    if(trnsb == 'n' || trnsb == 'N')
    {
        str_id = 1 * (rs_b == 1);     //1st bit
    }
    else if(trnsb == 't' || trnsb == 'T')
    {
        str_id = 1 * (cs_b == 1);     //1st bit
        rs_b = cs_b0;
        cs_b = rs_b0;
    }

    if(trnsa == 'n' || trnsa == 'N')
    {
        str_id |= ((1 * (rs_a == 1)) << 1); //2nd bit
    }
    else if(trnsa == 't' || trnsa == 'T')
    {
        str_id |= ((1 * (cs_a == 1)) << 1); //2nd bit
        rs_a = cs_a0;
        cs_a = rs_a0;
    }

    bool is_primary = false;

    str_id |= ((1 * (rs_c == 1)) << 2); //3rd bit

    if(str_id == 0 || str_id == 1 || str_id == 2 || str_id == 4)
    {
        is_primary = true;
    }

    auxinfo_t data;
    inc_t ps_a_use = (MR * rs_a);
    bli_auxinfo_set_ps_a( ps_a_use, &data );

    testinghelpers::ProtectedBuffer::start_signal_handler();
    try
    {
        if(is_primary == false && row_pref == true)
        {
            ukr_fp(
                BLIS_NO_CONJUGATE,
                BLIS_NO_CONJUGATE,
                n,
                m,
                k,
                &alpha,
                buf_b, cs_b, rs_b,
                buf_a, cs_a, rs_a,
                &beta,
                buf_c, cs_c, rs_c,
                &data,
                NULL
            );
        }
        else
        {
            ukr_fp(
                BLIS_NO_CONJUGATE,
                BLIS_NO_CONJUGATE,
                m,
                n,
                k,
                &alpha,
                buf_a, rs_a, cs_a,
                buf_b, rs_b, cs_b,
                &beta,
                buf_c, rs_c, cs_c,
                &data,
                NULL
            );
        }

        if ( is_memory_test )
        {
            // set pointers to second buffer
            buf_a    = (T*)mat_a.greenzone_2;
            buf_b    = (T*)mat_b.greenzone_2;
            buf_c    = (T*)mat_c.greenzone_2;

            // Check if the memory has been successfully allocated
            if ((buf_a == NULL) || (buf_b == NULL) || (buf_c == NULL)) {
                printf("Memory not allocated for input or output Matrix for memory test.\n");
                return ;
            }

            // copy data from 1st buffer of A and B to second buffer
            memcpy(buf_a, mat_a.greenzone_1, sizea);
            memcpy(buf_b, mat_b.greenzone_1, sizeb);

            //buf_c_ptrs.greenzone_1 has been updated with output from previous
            // gemm call, hence use buf_cref
            memcpy(buf_c, ref_c, sizec);

            if(is_primary == false && row_pref == true)
            {
                ukr_fp(
                    BLIS_NO_CONJUGATE,
                    BLIS_NO_CONJUGATE,
                    n,
                    m,
                    k,
                    &alpha,
                    buf_b, cs_b, rs_b,
                    buf_a, cs_a, rs_a,
                    &beta,
                    buf_c, cs_c, rs_c,
                    &data,
                    NULL
                );
            }
            else
            {
                ukr_fp(
                    BLIS_NO_CONJUGATE,
                    BLIS_NO_CONJUGATE,
                    m,
                    n,
                    k,
                    &alpha,
                    buf_a, rs_a, cs_a,
                    buf_b, rs_b, cs_b,
                    &beta,
                    buf_c, rs_c, cs_c,
                    &data,
                    NULL
                );
            }
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

    // call reference implementation
    testinghelpers::ref_gemm<T>( storage, trnsa, trnsb, m, n, k, alpha,
                                 buf_a, lda, buf_b, ldb, beta, ref_c, ldc);

    // Check component-wise error
    computediff<T>( "C", storage, m, n, buf_c, ref_c, ldc, thresh );
}
