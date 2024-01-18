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

/**
 * @brief Generic test body for gemm operation.
 */

// The function is templatized based on the datatype and function-pointer type to the kernel.
template<typename T, typename FT>
static void test_gemmnat_ukr( char storage, gtint_t m, gtint_t n, gtint_t k, T alpha, T beta, FT ukr_fp )
{
    gtint_t ldc  = m; // initialization

    // Create test operands
    // matrix A will be in col-storage
    // matrix B will be in row-storage
    // column * row = matrix -- rank-k update

    //Allocating aligned memory for A and B matrix as Native microkernel issues VMOVAPD which
    //expects memory to be accessed to be aligned.

    dim_t rs = 1;
    dim_t cs = 1;

    // create matrix A operand with col-storage
    rs = 1;
    cs = m;
    gtint_t lda = cs;
    gtint_t sizea =  m * k * sizeof(T);
    T *buf_a = (T*)aligned_alloc(BLIS_HEAP_STRIDE_ALIGN_SIZE, sizea);
    // Check if the memory has been successfully allocated
    if (buf_a == NULL) {
        printf("Matrix A: Memory not allocated.\n");
        return ;
    }
    testinghelpers::datagenerators::randomgenerators<T>( -2, 8, 'c', m, k, (T*)(buf_a), 'n', cs);

    // Create matrix B with row-storage
    rs = n;
    cs = 1;
    gtint_t ldb = rs;

    gtint_t sizeb =  k * n * sizeof(T);
    T *buf_b = (T*)aligned_alloc(BLIS_HEAP_STRIDE_ALIGN_SIZE, sizeb);
    // Check if the memory has been successfully allocated
    if (buf_b == NULL) {
        printf("Matrix B: Memory not allocated.\n");
        return ;
    }
    testinghelpers::datagenerators::randomgenerators<T>( -5, 2, 'r', k, n, (T*)(buf_b), 'n', rs);

    T *buf_c;
    T *buf_cref;
    gtint_t sizec;

    if(storage == 'r' || storage == 'R')
    {
        rs = n;
        cs = 1;
        ldc = rs;
        sizec =  m * n * sizeof(T);
        buf_c = (T*)malloc(sizec);
        testinghelpers::datagenerators::randomgenerators<T>( -5, 2, 'r', m, n, (T*)(buf_c), 'n', rs);
    }
    else
    {
        rs = 1;
        cs = m;
        ldc = cs;
        sizec =  m * n * sizeof(T);
        buf_c = (T*)malloc(sizec);
        testinghelpers::datagenerators::randomgenerators<T>( -5, 2, 'c', m, n, (T*)(buf_c), 'n', cs);
    }

    // Check if the memory has been successfully allocated
    if (buf_c == NULL) {
        printf("Matrix C: Memory not allocated.\n");
        return ;
    }
    buf_cref = (T*)malloc(sizec);
    // Check if the memory has been successfully allocated
    if (buf_cref == NULL) {
        printf("Matrix C Ref: Memory not allocated.\n");
        return ;
    }
    memcpy(buf_cref, buf_c, sizec);

    // Invoke micro-kernel
    auxinfo_t data;
    /* Fill the auxinfo_t struct in case the micro-kernel uses it. */
    bli_auxinfo_set_ps_a(0, &data);

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

    // Set the threshold for the errors:
    double thresh = 10 * (std::max(k,1)) * testinghelpers::getEpsilon<T>();

    // In native micro-kernel
    // op(A) = No transpose & op(B) = transpose
    // for column-storage
    char transa = 'n';
    char transb = 't';

    // The objective here is to make storage of all matrices same
    // To do this we set transpose of A and B appropriatley.
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
    computediff<T>( storage, m, n, (T*)buf_c, (T*)buf_cref, ldc, thresh );

    free(buf_a);
    free(buf_b);
    free(buf_c);
    free(buf_cref);
}


// The function is templatized based on the datatype and function-pointer type to the kernel.
template<typename T, typename FT>
static void test_gemmk1_ukr( FT ukr_fp, gtint_t m, gtint_t n, gtint_t k, char storage, T alpha, T beta )
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
    T *buf_a = (T*)malloc(sizea);
    T *buf_b = (T*)malloc(sizeb);
    T *buf_c = (T*)malloc(sizec);
    T *buf_cref = (T*)malloc(sizec);

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

    // Set the threshold for the errors:
    double thresh = 10 * std::max(n,std::max(k,m)) * testinghelpers::getEpsilon<T>();

    // call reference implementation
    testinghelpers::ref_gemm<T>( storage, 'n', 'n', m, n, k, alpha,
                                 buf_a, lda, buf_b, ldb, beta, buf_cref, ldc);

    // Check component-wise error
    computediff<T>( storage, m, n, buf_c, buf_cref, ldc, thresh );

    free(buf_a);
    free(buf_b);
    free(buf_c);
    free(buf_cref);
}




template<typename T, typename FT>
static void test_dgemmsup_ukr( FT ukr_fp, char trnsa, char trnsb, gtint_t m, gtint_t n, gtint_t k, T alpha, T beta, char storageC, gtint_t MR, bool row_pref)
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
    T *buf_a = (T*)malloc(sizea);
    T *buf_b = (T*)malloc(sizeb);
    T *buf_c = (T*)malloc(sizec);
    T *buf_cref = (T*)malloc(sizec);

    // Check if the memory has been successfully allocated
    if ((buf_a == NULL) ||(buf_b == NULL) ||(buf_c == NULL) ||(buf_cref == NULL)) {
        printf("Memory not allocated for input and output Matrix.\n");
        return ;
    }
    testinghelpers::datagenerators::randomgenerators<T>( -2, 8, storage, m, k, (T*)(buf_a), trnsa, lda);
    testinghelpers::datagenerators::randomgenerators<T>( -5, 2, storage, k, n, (T*)(buf_b), trnsb, ldb);
    testinghelpers::datagenerators::randomgenerators<T>( -3, 5, storage, m, n, (T*)(buf_c), 'n', ldc);

    // Create a copy of c so that we can check reference results.
    memcpy(buf_cref, buf_c, sizec);
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

    if(is_primary == false && row_pref == true)
    {
        auxinfo_t data;
        inc_t ps_a_use = (MR * rs_a);
        bli_auxinfo_set_ps_a( ps_a_use, &data );
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
        auxinfo_t data;
        inc_t ps_a_use = (MR * rs_a);
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

    // Set the threshold for the errors:
    double thresh = 10 * (std::max(k,1)) * testinghelpers::getEpsilon<T>();

    // call reference implementation
    testinghelpers::ref_gemm<T>( storage, trnsa, trnsb, m, n, k, alpha,
                                 buf_a, lda, buf_b, ldb, beta, buf_cref, ldc);

    // Check component-wise error
    computediff<T>( storage, m, n, buf_c, buf_cref, ldc, thresh );

    free(buf_a);
    free(buf_b);
    free(buf_c);
    free(buf_cref);
}

template<typename T, typename FT>
static void test_zgemmsup_ukr( char storage, char trnsa, char trnsb, gtint_t m, gtint_t n, gtint_t k, T alpha, T beta, FT ukr_fp)
{
    // Compute the leading dimensions of a, b, and c.
    gtint_t lda = testinghelpers::get_leading_dimension( storage, trnsa, m, k, 0 );
    gtint_t ldb = testinghelpers::get_leading_dimension( storage, trnsb, k, n, 0 );
    gtint_t ldc = testinghelpers::get_leading_dimension( storage, 'n', m, n, 0 );

    //----------------------------------------------------------
    //         Initialize matrices with random numbers
    //----------------------------------------------------------
    gtint_t sizea =  testinghelpers::matsize( storage, trnsa, m, k, lda ) * sizeof(T);
    gtint_t sizeb =  testinghelpers::matsize( storage, trnsb, k, n, ldb ) * sizeof(T);
    gtint_t sizec =  testinghelpers::matsize( storage, 'n', m, n, ldc ) * sizeof(T);
    T *buf_a = (T*)malloc(sizea);
    T *buf_b = (T*)malloc(sizeb);
    T *buf_c = (T*)malloc(sizec);
    T *buf_cref = (T*)malloc(sizec);

    // Check if the memory has been successfully allocated
    if ((buf_a == NULL) ||(buf_b == NULL) ||(buf_c == NULL) ||(buf_cref == NULL)) {
        printf("Memory not allocated for input and output Matrix.\n");
        return ;
    }

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

    //Panel stride update is required only for zen4 sup kernels
#if defined(BLIS_KERNELS_ZEN4) && defined(GTEST_AVX512)
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
#else
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
        NULL,
        NULL
        );
#endif

    // Set the threshold for the errors:
    double thresh = 20 * (std::max(k,1)) * testinghelpers::getEpsilon<T>();

    // call reference implementation
    testinghelpers::ref_gemm<T>( storage, trnsa, trnsb, m, n, k, alpha,
                                 buf_a, lda, buf_b, ldb, beta, buf_cref, ldc);

    // Check component-wise error
    computediff<T>( storage, m, n, buf_c, buf_cref, ldc, thresh );

    free(buf_a);
    free(buf_b);
    free(buf_c);
    free(buf_cref);

}
