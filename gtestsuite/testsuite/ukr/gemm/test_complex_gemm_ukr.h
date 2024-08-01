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
#include <signal.h>
#include "level3/ref_gemm.h"
#include "inc/check_error.h"
#include "blis.h"
#include "common/testing_helpers.h"

/**********************************************************************/
/************    Code path when memory test is disabled  **************/
/* 1. Compute Leading dimension of all matrix based on                */
/*    storage, size and trans parameters                              */
/* 2. Compute size of matrices for which memory needs to be allocated */
/* 3. Allocate memory for all matrices                                */
/* 4. Initialise matrices with random numbers                         */
/* 5. Copy blis output matrix content to reference output matrix      */
/* 6. Call blis micro kernel with output matrix                       */
/* 7. Call reference kernel with reference output matrix              */
/* 8. Compute difference of blis and reference output                 */
/*    based on threshold set                                          */
/**********************************************************************/
/************    Code path when memory test is enabled   **************/
/* 1. Compute Leading dimension of all matrix based on                */
/*    storage, size and trans parameters                              */
/* 2. Compute size of matrices for which memory needs to be allocated */
/* 3. Allocate 2 set of memories for A, B, C matrix                   */
/*    green_zone1: Memory near red_zone1                              */
/*    green_zone2: Memory near red_zone2                              */
/*    2 set of memory is required to check memory leaks               */
/*    before starting of buffer or after end of buffer                */
/* 4. Initialise matrices with random numbers                         */
/* 5. Call blis micro kernel with output matrix with green_zone1 ptr  */
/* 6. Call blis micro kernel again with green_zone2 ptr               */
/* 7. Failure is reported if there is out of bound read/write error   */
/* 8. Call reference kernel with reference output matrix to           */
/*    check for any accuracy failures                                 */
/* 9. Compute difference of blis and reference output                 */
/*    based on threshold set                                          */
/**********************************************************************/

template<typename T, typename FT>
static void test_complex_gemmsup_ukr( char storage, char trnsa, char trnsb, gtint_t m, gtint_t n, gtint_t k, T alpha, T beta, double thresh, FT ukr_fp, bool is_memory_test = false )
{
    // Compute the leading dimensions of a, b, and c.
    gtint_t lda = testinghelpers::get_leading_dimension( storage, trnsa, m, k, 0 );
    gtint_t ldb = testinghelpers::get_leading_dimension( storage, trnsb, k, n, 0 );
    gtint_t ldc = testinghelpers::get_leading_dimension( storage, 'n', m, n, 0 );

    //----------------------------------------------------------
    //         Compute size of Matrix: A, B, C
    //----------------------------------------------------------
    gtint_t sizea =  testinghelpers::matsize( storage, trnsa, m, k, lda ) * sizeof(T);
    gtint_t sizeb =  testinghelpers::matsize( storage, trnsb, k, n, ldb ) * sizeof(T);
    gtint_t sizec =  testinghelpers::matsize( storage, 'n', m, n, ldc ) * sizeof(T);

    // Allocate memory for Matrix: A, B, C, CRef
    testinghelpers::ProtectedBuffer buf_a_ptrs( sizea, false, is_memory_test );
    testinghelpers::ProtectedBuffer buf_b_ptrs( sizeb, false , is_memory_test );
    testinghelpers::ProtectedBuffer buf_c_ptrs( sizec, false , is_memory_test );

    /* No need to check for memory errors for reference code path,     */
    /* hence is_memory_test is set to false                            */
    testinghelpers::ProtectedBuffer buf_cref_ptrs( sizec, false , false );

    T* buf_a    = (T*)buf_a_ptrs.greenzone_1;
    T* buf_b    = (T*)buf_b_ptrs.greenzone_1;
    T* buf_c    = (T*)buf_c_ptrs.greenzone_1;
    T* buf_cref = (T*)buf_cref_ptrs.greenzone_1;

    testinghelpers::datagenerators::randomgenerators<T>( -2, 8, storage, m, k, (T*)(buf_a), trnsa, lda);
    testinghelpers::datagenerators::randomgenerators<T>( -5, 2, storage, k, n, (T*)(buf_b), trnsb, ldb);
    testinghelpers::datagenerators::randomgenerators<T>( -3, 5, storage, m, n, (T*)(buf_c), 'n', ldc);

    // Create a copy of c so that we can check reference results.
    memcpy(buf_cref, buf_c, sizec);

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

    if(trnsb == 't' || trnsb == 'T')
    {
        rs_b = cs_b0;
        cs_b = rs_b0;
    }

    if(trnsa == 't' || trnsa == 'T')
    {
        rs_a = cs_a0;
        cs_a = rs_a0;
    }
    // add signal handler for segmentation fault
    testinghelpers::ProtectedBuffer::start_signal_handler();
    try
    {
        auxinfo_t data;
        //Panel stride update is required only for zen4 sup kernels
        inc_t ps_a_use = (12 * rs_a); //12 = MR
        bli_auxinfo_set_ps_a( ps_a_use, &data );

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

            // second call to ukr
            auxinfo_t data;
            inc_t ps_a_use = (12 * rs_a); //12 = MR
            bli_auxinfo_set_ps_a( ps_a_use, &data );

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
                                 buf_a, lda, buf_b, ldb, beta, buf_cref, ldc);

    // Check component-wise error
    computediff<T>( "C", storage, m, n, buf_c, buf_cref, ldc, thresh );

}

// The function is templatized based on the datatype and function-pointer type to the kernel.
template<typename T, typename FT>
static void test_gemmnat_ukr( char storage, gtint_t m, gtint_t n, gtint_t k, T alpha, T beta, double thresh, FT ukr_fp, bool is_memory_test = false )
{

    /*************Memory requirement*****************************/
    /* General requirement of memory allocation:                */
    /*        Block                Microkernel                  */
    /*     A = MC * KC            A = MR * k                    */
    /*     B = NC * KC            B = NR * k                    */
    /*     C = MC * NC            C = MR * NR                   */
    /* Native kernel works on packed buffer for A and B matrix  */
    /* Memory requirement for input matrix for a block:         */
    /*     A = (MC + max(MR, NR)) * (KC + max(MR, NR))          */
    /*     B = (NC + max(MR, NR)) * (KC + max(MR, NR))          */
    /* Memory requirement for input matrix for a microkernel:   */
    /*     A = max(MR, NR) * (k + max(MR, NR))                  */
    /*     B = max(MR, NR) * (k + max(MR, NR))                  */
    /* MC, NC, KC - Cache block sizes                           */
    /* MR, NR - Micro kernel sizes                              */
    /* To support preloading feature inside microkernel,        */
    /* allocation of extra memory is must                       */
    /************************************************************/

    obj_t a, b;
    num_t dt = BLIS_DCOMPLEX;
    gtint_t maxmn = (std::max)(m,n);
    bli_obj_create(dt, m, k, 1, m, &a);
    bli_obj_create(dt, k, n, n, 1, &b);

    // Create test operands
    // matrix A will be in col-storage
    // matrix B will be in row-storage
    // column * row = matrix -- rank-k update

    // Set matrix A dimensions
    gtint_t rs = 1;
    gtint_t cs = m;
    gtint_t lda = cs;
    gtint_t sizea =  maxmn * (k+maxmn) * sizeof(T);

    // Set matrix B dimensions
    rs = n;
    cs = 1;
    gtint_t ldb = rs;
    gtint_t sizeb =  (k+maxmn) * maxmn * sizeof(T);

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
    testinghelpers::datagenerators::randomgenerators<T>( -5, 2, storage , m, n, (T*)(buf_c), 'n', ldc);

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
static void test_gemmk1_ukr( FT ukr_fp, gtint_t m, gtint_t n, gtint_t k, char storage, T alpha, T beta, bool memory_test  = false )
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

    testinghelpers::ProtectedBuffer mat_a(sizea, false, memory_test);
    testinghelpers::ProtectedBuffer mat_b(sizeb, false, memory_test);
    testinghelpers::ProtectedBuffer mat_c(sizec, false, memory_test);
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
    testinghelpers::datagenerators::randomgenerators<T>( -3, 5, storage, m, n, (T*)(buf_c), 'n', ldc);

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

        if(memory_test == true)
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

    // Set the threshold for the errors:
    // Check gtestsuite gemm.h or netlib source code for reminder of the
    // functionality from which we estimate operation count per element
    // of output, and hence the multipler for epsilon.
    double thresh;
    if (m == 0 || n == 0)
        thresh = 0.0;
    else if ((alpha == testinghelpers::ZERO<T>() || k == 0) && (beta == testinghelpers::ZERO<T>() ||
              beta == testinghelpers::ONE<T>()))
        thresh = 0.0;
    else
        thresh = (7*k+3)*testinghelpers::getEpsilon<T>();

    // call reference implementation
    testinghelpers::ref_gemm<T>( storage, 'n', 'n', m, n, k, alpha,
                                 buf_a, lda, buf_b, ldb, beta, buf_cref, ldc);

    // Check component-wise error
    computediff<T>( "C", storage, m, n, buf_c, buf_cref, ldc, thresh );
}
