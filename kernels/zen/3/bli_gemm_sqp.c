/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021, Advanced Micro Devices, Inc.

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
#include "blis.h"
#include "immintrin.h"

#define BLIS_LOADFIRST 0
#define BLIS_ENABLE_PREFETCH 1

#define MEM_ALLOC 1//malloc performs better than bli_malloc.
#define BLIS_MX8 8
#define BLIS_MX4 4
#define BLIS_MX1 1
#define DEBUG_3M_SQP 0

typedef struct  {
    siz_t data_size;
    siz_t size;
    void* alignedBuf;
    void* unalignedBuf;
}mem_block;

static err_t bli_zgemm_sqp_m8(gint_t m, gint_t n, gint_t k, double* a, guint_t lda, double* b, guint_t ldb, double* c, guint_t ldc, double alpha, double beta, bool isTransA, gint_t mx, gint_t* p_istart);
static err_t bli_dgemm_sqp_m8(gint_t m, gint_t n, gint_t k, double* a, guint_t lda, double* b, guint_t ldb, double* c, guint_t ldc, bool isTransA, double alpha, gint_t mx, gint_t* p_istart);

/*
* The bli_gemm_sqp (square packed) function performs dgemm and 3m zgemm.
* It focuses on square matrix sizes, where m=n=k. But supports non-square matrix sizes as well.
* Currently works for m multiple of 8 & column major storage and kernels. It has custom dgemm
* 8mxn block column preferred kernels with single load and store of C matrix to perform dgemm
* , which is also used as real kernel in 3m complex gemm computation.
*/
err_t bli_gemm_sqp
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       cntl_t* cntl
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);

    // if row major format return.
    if ((bli_obj_row_stride( a ) != 1) ||
        (bli_obj_row_stride( b ) != 1) ||
        (bli_obj_row_stride( c ) != 1))
    {
        return BLIS_INVALID_ROW_STRIDE;
    }

    if(bli_obj_has_conj(a) || bli_obj_has_conj(b))
    {
        return BLIS_NOT_YET_IMPLEMENTED;
    }

    if (bli_obj_has_trans( b ))
    {
        return BLIS_NOT_YET_IMPLEMENTED;
    }

    num_t dt = bli_obj_dt(c);
    gint_t m = bli_obj_length( c ); // number of rows of Matrix C
    gint_t n = bli_obj_width( c );  // number of columns of Matrix C
    gint_t k = bli_obj_length( b ); // number of rows of Matrix B

    guint_t lda = bli_obj_col_stride( a ); // column stride of matrix OP(A), where OP(A) is Transpose(A) if transA enabled.
    guint_t ldb = bli_obj_col_stride( b ); // column stride of matrix OP(B), where OP(B) is Transpose(B) if transB enabled.
    guint_t ldc = bli_obj_col_stride( c ); // column stride of matrix C

    if((m==0)||(n==0)||(k==0))
    {
        return BLIS_NOT_YET_IMPLEMENTED;
    }

    if((dt != BLIS_DCOMPLEX)&&(dt != BLIS_DOUBLE))
    {
        return BLIS_NOT_YET_IMPLEMENTED;
    }

    bool isTransA = false;
    if (bli_obj_has_trans( a ))
    {
        isTransA = true;
    }

    dim_t m8rem = m - ((m>>3)<<3);

    double* ap     = ( double* )bli_obj_buffer( a );
    double* bp     = ( double* )bli_obj_buffer( b );
    double* cp     = ( double* )bli_obj_buffer( c );
    gint_t istart = 0;
    gint_t* p_istart = &istart;
    *p_istart = 0;
    err_t status;
    if(dt==BLIS_DCOMPLEX)
    {
        dcomplex* alphap = ( dcomplex* )bli_obj_buffer( alpha );
        dcomplex* betap  = ( dcomplex* )bli_obj_buffer( beta );

        //alpha and beta both real are implemented. alpha and beta with imaginary component to be implemented.
        double alpha_real = alphap->real;
        double alpha_imag = alphap->imag;
        double beta_real  = betap->real;
        double beta_imag  = betap->imag;
        if( (alpha_imag!=0)||(beta_imag!=0) )
        {
            return BLIS_NOT_YET_IMPLEMENTED;
        }
        /* 3m zgemm implementation for C = AxB and C = AtxB */
#if 0
        return bli_zgemm_sqp_m8( m, n, k, ap, lda, bp, ldb, cp, ldc, alpha_real, beta_real, isTransA, 8,  p_istart);
#else
        status = bli_zgemm_sqp_m8( m, n, k, ap, lda, bp, ldb, cp, ldc, alpha_real, beta_real, isTransA, 8,  p_istart);
        if(m8rem==0)
        {
            return status;// No residue: done
        }
        else
        {
            //complete residue m blocks
            status = bli_zgemm_sqp_m8( m, n, k, ap, lda, bp, ldb, cp, ldc, alpha_real, beta_real, isTransA, 1, p_istart);
            return status;
        }
#endif
    }
    else if(dt == BLIS_DOUBLE)
    {
        double *alpha_cast, *beta_cast;
        alpha_cast = bli_obj_buffer_for_1x1(BLIS_DOUBLE, alpha);
        beta_cast = bli_obj_buffer_for_1x1(BLIS_DOUBLE, beta);

        if((*beta_cast)!=1.0)
        {
            return BLIS_NOT_YET_IMPLEMENTED;
        }
        if(((*alpha_cast)!=1.0)&&((*alpha_cast)!=-1.0))
        {
            return BLIS_NOT_YET_IMPLEMENTED;
        }
        /* dgemm implementation with 8mx5n major kernel and column preferred storage */
        status = bli_dgemm_sqp_m8( m, n, k, ap, lda, bp, ldb, cp, ldc, isTransA, (*alpha_cast), 8, p_istart);
        if(status==BLIS_SUCCESS)
        {
            if(m8rem==0)
            {
                return status;// No residue: done
            }
            else
            {
                //complete residue m blocks
                status = bli_dgemm_sqp_m8( m, n, k, ap, lda, bp, ldb, cp, ldc, isTransA, (*alpha_cast), 1, p_istart);
                return status;
            }
        }

    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
    return BLIS_NOT_YET_IMPLEMENTED;
};

/************************************************************************************************************/
/************************** dgemm kernels (8mxn) column preffered  ******************************************/
/************************************************************************************************************/

/*  Main dgemm kernel 8mx6n with single load and store of C matrix block
    alpha = +/-1 and beta = +/-1,0 handled while packing.*/
inc_t bli_kernel_8mx6n(gint_t n, gint_t k, gint_t j, double* aPacked, guint_t lda, double* b, guint_t ldb, double* c, guint_t ldc)
{
    gint_t p;

    __m256d av0, av1;
    __m256d bv0, bv1;
    __m256d cv0, cv1, cv2, cv3, cv4, cv5;
    __m256d cx0, cx1, cx2, cx3, cx4, cx5;
    double* pb, * pc;

    pb = b;
    pc = c;
    inc_t ldc6 = ldc * 6; inc_t ldb6 = ldb * 6;

    for (j = 0; j <= (n - 6); j += 6) {
        double* pcldc = pc + ldc;
        double* pcldc2 = pcldc + ldc;
        double* pcldc3 = pcldc2 + ldc;
        double* pcldc4 = pcldc3 + ldc;
        double* pcldc5 = pcldc4 + ldc;

        double* pbldb = pb + ldb;
        double* pbldb2 = pbldb + ldb;
        double* pbldb3 = pbldb2 + ldb;
        double* pbldb4 = pbldb3 + ldb;
        double* pbldb5 = pbldb4 + ldb;

#if BLIS_ENABLE_PREFETCH
        _mm_prefetch((char*)(pc), _MM_HINT_T0);
        _mm_prefetch((char*)(pcldc), _MM_HINT_T0);
        _mm_prefetch((char*)(pcldc2), _MM_HINT_T0);
        _mm_prefetch((char*)(pcldc3), _MM_HINT_T0);
        _mm_prefetch((char*)(pcldc4), _MM_HINT_T0);
        _mm_prefetch((char*)(pcldc5), _MM_HINT_T0);

        _mm_prefetch((char*)(aPacked), _MM_HINT_T0);

        _mm_prefetch((char*)(pb), _MM_HINT_T0);
        _mm_prefetch((char*)(pbldb), _MM_HINT_T0);
        _mm_prefetch((char*)(pbldb2), _MM_HINT_T0);
        _mm_prefetch((char*)(pbldb3), _MM_HINT_T0);
        _mm_prefetch((char*)(pbldb4), _MM_HINT_T0);
        _mm_prefetch((char*)(pbldb5), _MM_HINT_T0);
#endif
        /* C matrix column major load */
#if BLIS_LOADFIRST
        cv0 = _mm256_loadu_pd(pc);     cx0 = _mm256_loadu_pd(pc + 4);
        cv1 = _mm256_loadu_pd(pcldc);  cx1 = _mm256_loadu_pd(pcldc + 4);
        cv2 = _mm256_loadu_pd(pcldc2); cx2 = _mm256_loadu_pd(pcldc2 + 4);
        cv3 = _mm256_loadu_pd(pcldc3); cx3 = _mm256_loadu_pd(pcldc3 + 4);
        cv4 = _mm256_loadu_pd(pcldc4); cx4 = _mm256_loadu_pd(pcldc4 + 4);
        cv5 = _mm256_loadu_pd(pcldc5); cx5 = _mm256_loadu_pd(pcldc5 + 4);
#else
        cv0 = _mm256_setzero_pd();  cx0 = _mm256_setzero_pd();
        cv1 = _mm256_setzero_pd();  cx1 = _mm256_setzero_pd();
        cv2 = _mm256_setzero_pd();  cx2 = _mm256_setzero_pd();
        cv3 = _mm256_setzero_pd();  cx3 = _mm256_setzero_pd();
        cv4 = _mm256_setzero_pd();  cx4 = _mm256_setzero_pd();
        cv5 = _mm256_setzero_pd();  cx5 = _mm256_setzero_pd();
#endif
        double* x = aPacked;
        double* pb0 = pb;
        for (p = 0; p < k; p += 1) {
            av0 = _mm256_loadu_pd(x); x += 4;          av1 = _mm256_loadu_pd(x); x += 4;
            bv0 = _mm256_broadcast_sd  (pb0); pb0++;
            bv1 = _mm256_broadcast_sd(pbldb); pbldb++;
            cv0 = _mm256_fmadd_pd(av0, bv0, cv0);
            cx0 = _mm256_fmadd_pd(av1, bv0, cx0);
            cv1 = _mm256_fmadd_pd(av0, bv1, cv1);
            cx1 = _mm256_fmadd_pd(av1, bv1, cx1);

            bv0 = _mm256_broadcast_sd(pbldb2);pbldb2++;
            bv1 = _mm256_broadcast_sd(pbldb3);pbldb3++;
            cv2 = _mm256_fmadd_pd(av0, bv0, cv2);
            cx2 = _mm256_fmadd_pd(av1, bv0, cx2);
            cv3 = _mm256_fmadd_pd(av0, bv1, cv3);
            cx3 = _mm256_fmadd_pd(av1, bv1, cx3);

            bv0 = _mm256_broadcast_sd(pbldb4);pbldb4++;
            bv1 = _mm256_broadcast_sd(pbldb5);pbldb5++;
            cv4 = _mm256_fmadd_pd(av0, bv0, cv4);
            cx4 = _mm256_fmadd_pd(av1, bv0, cx4);
            cv5 = _mm256_fmadd_pd(av0, bv1, cv5);
            cx5 = _mm256_fmadd_pd(av1, bv1, cx5);
        }
#if BLIS_LOADFIRST
#else
        bv0 = _mm256_loadu_pd(pc);     bv1 = _mm256_loadu_pd(pc + 4);
        cv0 = _mm256_add_pd(cv0, bv0); cx0 = _mm256_add_pd(cx0, bv1);

        av0 = _mm256_loadu_pd(pcldc);  av1 = _mm256_loadu_pd(pcldc + 4);
        cv1 = _mm256_add_pd(cv1, av0); cx1 = _mm256_add_pd(cx1, av1);

        bv0 = _mm256_loadu_pd(pcldc2); bv1 = _mm256_loadu_pd(pcldc2 + 4);
        cv2 = _mm256_add_pd(cv2, bv0); cx2 = _mm256_add_pd(cx2, bv1);

        av0 = _mm256_loadu_pd(pcldc3); av1 = _mm256_loadu_pd(pcldc3 + 4);
        cv3 = _mm256_add_pd(cv3, av0); cx3 = _mm256_add_pd(cx3, av1);

        bv0 = _mm256_loadu_pd(pcldc4); bv1 = _mm256_loadu_pd(pcldc4 + 4);
        cv4 = _mm256_add_pd(cv4, bv0); cx4 = _mm256_add_pd(cx4, bv1);

        av0 = _mm256_loadu_pd(pcldc5); av1 = _mm256_loadu_pd(pcldc5 + 4);
        cv5 = _mm256_add_pd(cv5, av0); cx5 = _mm256_add_pd(cx5, av1);
#endif
        /* C matrix column major store */
        _mm256_storeu_pd(pc, cv0);
        _mm256_storeu_pd(pc + 4, cx0);

        _mm256_storeu_pd(pcldc, cv1);
        _mm256_storeu_pd(pcldc + 4, cx1);

        _mm256_storeu_pd(pcldc2, cv2);
        _mm256_storeu_pd(pcldc2 + 4, cx2);

        _mm256_storeu_pd(pcldc3, cv3);
        _mm256_storeu_pd(pcldc3 + 4, cx3);

        _mm256_storeu_pd(pcldc4, cv4);
        _mm256_storeu_pd(pcldc4 + 4, cx4);

        _mm256_storeu_pd(pcldc5, cv5);
        _mm256_storeu_pd(pcldc5 + 4, cx5);

        pc += ldc6;pb += ldb6;
    }
    //printf(" 8x6:j:%d ", j);
    return j;
}

/*  alternative Main dgemm kernel 8mx5n with single load and store of C matrix block
    alpha = +/-1 and beta = +/-1,0 handled while packing.*/
inc_t bli_kernel_8mx5n(gint_t n, gint_t k, gint_t j, double* aPacked, guint_t lda, double* b, guint_t ldb, double* c, guint_t ldc)
{
    gint_t p;
    __m256d av0;
    __m256d bv0, bv1, bv2, bv3;
    __m256d cv0, cv1, cv2, cv3;
    __m256d cx0, cx1, cx2, cx3;
    __m256d bv4, cv4, cx4;
    double* pb, * pc;

    pb = b;
    pc = c;
    inc_t ldc5 = ldc * 5; inc_t ldb5 = ldb * 5;

    for (; j <= (n - 5); j += 5) {

        double* pcldc = pc + ldc;
        double* pcldc2 = pcldc + ldc;
        double* pcldc3 = pcldc2 + ldc;
        double* pcldc4 = pcldc3 + ldc;

        double* pbldb = pb + ldb;
        double* pbldb2 = pbldb + ldb;
        double* pbldb3 = pbldb2 + ldb;
        double* pbldb4 = pbldb3 + ldb;

#if BLIS_ENABLE_PREFETCH
        _mm_prefetch((char*)(pc), _MM_HINT_T0);
        _mm_prefetch((char*)(pcldc), _MM_HINT_T0);
        _mm_prefetch((char*)(pcldc2), _MM_HINT_T0);
        _mm_prefetch((char*)(pcldc3), _MM_HINT_T0);
        _mm_prefetch((char*)(pcldc4), _MM_HINT_T0);

        _mm_prefetch((char*)(aPacked), _MM_HINT_T0);

        _mm_prefetch((char*)(pb), _MM_HINT_T0);
        _mm_prefetch((char*)(pbldb), _MM_HINT_T0);
        _mm_prefetch((char*)(pbldb2), _MM_HINT_T0);
        _mm_prefetch((char*)(pbldb3), _MM_HINT_T0);
        _mm_prefetch((char*)(pbldb4), _MM_HINT_T0);
#endif
        /* C matrix column major load */
#if BLIS_LOADFIRST
        cv0 = _mm256_loadu_pd(pc);     cx0 = _mm256_loadu_pd(pc + 4);
        cv1 = _mm256_loadu_pd(pcldc);  cx1 = _mm256_loadu_pd(pcldc + 4);
        cv2 = _mm256_loadu_pd(pcldc2); cx2 = _mm256_loadu_pd(pcldc2 + 4);
        cv3 = _mm256_loadu_pd(pcldc3); cx3 = _mm256_loadu_pd(pcldc3 + 4);
        cv4 = _mm256_loadu_pd(pcldc4); cx4 = _mm256_loadu_pd(pcldc4 + 4);
#else
        cv0 = _mm256_setzero_pd();  cx0 = _mm256_setzero_pd();
        cv1 = _mm256_setzero_pd();  cx1 = _mm256_setzero_pd();
        cv2 = _mm256_setzero_pd();  cx2 = _mm256_setzero_pd();
        cv3 = _mm256_setzero_pd();  cx3 = _mm256_setzero_pd();
        cv4 = _mm256_setzero_pd();  cx4 = _mm256_setzero_pd();
#endif
        double* x = aPacked;
        double* pb0 = pb;
        for (p = 0; p < k; p += 1) {
            bv0 = _mm256_broadcast_sd(pb0); pb0++;
            bv1 = _mm256_broadcast_sd(pbldb); pbldb++;
            bv2 = _mm256_broadcast_sd(pbldb2); pbldb2++;
            bv3 = _mm256_broadcast_sd(pbldb3);pbldb3++;
            bv4 = _mm256_broadcast_sd(pbldb4);pbldb4++;

            av0 = _mm256_loadu_pd(x); x += 4;
            cv0 = _mm256_fmadd_pd(av0, bv0, cv0);
            cv1 = _mm256_fmadd_pd(av0, bv1, cv1);
            cv2 = _mm256_fmadd_pd(av0, bv2, cv2);
            cv3 = _mm256_fmadd_pd(av0, bv3, cv3);
            cv4 = _mm256_fmadd_pd(av0, bv4, cv4);

            av0 = _mm256_loadu_pd(x); x += 4;
            cx0 = _mm256_fmadd_pd(av0, bv0, cx0);
            cx1 = _mm256_fmadd_pd(av0, bv1, cx1);
            cx2 = _mm256_fmadd_pd(av0, bv2, cx2);
            cx3 = _mm256_fmadd_pd(av0, bv3, cx3);
            cx4 = _mm256_fmadd_pd(av0, bv4, cx4);
        }
#if BLIS_LOADFIRST
#else
        bv0 = _mm256_loadu_pd(pc);     bv1 = _mm256_loadu_pd(pc + 4);
        cv0 = _mm256_add_pd(cv0, bv0); cx0 = _mm256_add_pd(cx0, bv1);

        bv2 = _mm256_loadu_pd(pcldc);  bv3 = _mm256_loadu_pd(pcldc + 4);
        cv1 = _mm256_add_pd(cv1, bv2); cx1 = _mm256_add_pd(cx1, bv3);

        bv0 = _mm256_loadu_pd(pcldc2); bv1 = _mm256_loadu_pd(pcldc2 + 4);
        cv2 = _mm256_add_pd(cv2, bv0); cx2 = _mm256_add_pd(cx2, bv1);

        bv2 = _mm256_loadu_pd(pcldc3); bv3 = _mm256_loadu_pd(pcldc3 + 4);
        cv3 = _mm256_add_pd(cv3, bv2); cx3 = _mm256_add_pd(cx3, bv3);

        bv0 = _mm256_loadu_pd(pcldc4); bv1 = _mm256_loadu_pd(pcldc4 + 4);
        cv4 = _mm256_add_pd(cv4, bv0); cx4 = _mm256_add_pd(cx4, bv1);
#endif
        /* C matrix column major store */
        _mm256_storeu_pd(pc, cv0);
        _mm256_storeu_pd(pc + 4, cx0);

        _mm256_storeu_pd(pcldc, cv1);
        _mm256_storeu_pd(pcldc + 4, cx1);

        _mm256_storeu_pd(pcldc2, cv2);
        _mm256_storeu_pd(pcldc2 + 4, cx2);

        _mm256_storeu_pd(pcldc3, cv3);
        _mm256_storeu_pd(pcldc3 + 4, cx3);

        _mm256_storeu_pd(pcldc4, cv4);
        _mm256_storeu_pd(pcldc4 + 4, cx4);

        pc += ldc5;pb += ldb5;
    }
    //printf(" 8x5:j:%d ", j);
    return j;
}

/* residue dgemm kernel 8mx4n with single load and store of C matrix block
   Code could be optimized further, complete ymm register set is not used.
   Being residue kernel, its of lesser priority.
*/
inc_t bli_kernel_8mx4n(gint_t n, gint_t k, gint_t j, double* aPacked, guint_t lda, double* b, guint_t ldb, double* c, guint_t ldc)
{
    gint_t p;
    __m256d av0;
    __m256d bv0, bv1, bv2, bv3;
    __m256d cv0, cv1, cv2, cv3;
    __m256d cx0, cx1, cx2, cx3;
    double* pb, * pc;

    pb = b;
    pc = c;
    inc_t ldc4 = ldc * 4; inc_t ldb4 = ldb * 4;

    for (; j <= (n - 4); j += 4) {

        double* pcldc = pc + ldc; double* pcldc2 = pcldc + ldc; double* pcldc3 = pcldc2 + ldc;
        double* pbldb = pb + ldb; double* pbldb2 = pbldb + ldb; double* pbldb3 = pbldb2 + ldb;

        cv0 = _mm256_loadu_pd(pc);     cx0 = _mm256_loadu_pd(pc + 4);
        cv1 = _mm256_loadu_pd(pcldc);  cx1 = _mm256_loadu_pd(pcldc + 4);
        cv2 = _mm256_loadu_pd(pcldc2); cx2 = _mm256_loadu_pd(pcldc2 + 4);
        cv3 = _mm256_loadu_pd(pcldc3); cx3 = _mm256_loadu_pd(pcldc3 + 4);
        {
            double* x = aPacked;
            double* pb0 = pb;
            for (p = 0; p < k; p += 1) {
                // better kernel to be written since more register are available.
                bv0 = _mm256_broadcast_sd(pb0);    pb0++;
                bv1 = _mm256_broadcast_sd(pbldb);  pbldb++;
                bv2 = _mm256_broadcast_sd(pbldb2); pbldb2++;
                bv3 = _mm256_broadcast_sd(pbldb3); pbldb3++;

                av0 = _mm256_loadu_pd(x); x += 4;
                cv0 = _mm256_fmadd_pd(av0, bv0, cv0);
                cv1 = _mm256_fmadd_pd(av0, bv1, cv1);
                cv2 = _mm256_fmadd_pd(av0, bv2, cv2);
                cv3 = _mm256_fmadd_pd(av0, bv3, cv3);

                av0 = _mm256_loadu_pd(x); x += 4;
                cx0 = _mm256_fmadd_pd(av0, bv0, cx0);
                cx1 = _mm256_fmadd_pd(av0, bv1, cx1);
                cx2 = _mm256_fmadd_pd(av0, bv2, cx2);
                cx3 = _mm256_fmadd_pd(av0, bv3, cx3);
            }
        }
        _mm256_storeu_pd(pc, cv0);
        _mm256_storeu_pd(pc + 4, cx0);
        _mm256_storeu_pd(pcldc, cv1);
        _mm256_storeu_pd(pcldc + 4, cx1);
        _mm256_storeu_pd(pcldc2, cv2);
        _mm256_storeu_pd(pcldc2 + 4, cx2);
        _mm256_storeu_pd(pcldc3, cv3);
        _mm256_storeu_pd(pcldc3 + 4, cx3);

        pc += ldc4;pb += ldb4;
    }// j loop 4 multiple
    //printf(" 8x4:j:%d ", j);
    return j;
}

/* residue dgemm kernel 8mx3n with single load and store of C matrix block
   Code could be optimized further, complete ymm register set is not used.
   Being residue kernel, its of lesser priority.
*/
inc_t bli_kernel_8mx3n(gint_t n, gint_t k, gint_t j, double* aPacked, guint_t lda, double* b, guint_t ldb, double* c, guint_t ldc)
{
    gint_t p;
    __m256d av0;
    __m256d bv0, bv1, bv2;
    __m256d cv0, cv1, cv2;
    __m256d cx0, cx1, cx2;
    double* pb, * pc;

    pb = b;
    pc = c;

    inc_t ldc3 = ldc * 3; inc_t ldb3 = ldb * 3;

    for (; j <= (n - 3); j += 3) {

        double* pcldc = pc + ldc; double* pcldc2 = pcldc + ldc;
        double* pbldb = pb + ldb; double* pbldb2 = pbldb + ldb;

        cv0 = _mm256_loadu_pd(pc);     cx0 = _mm256_loadu_pd(pc + 4);
        cv1 = _mm256_loadu_pd(pcldc);  cx1 = _mm256_loadu_pd(pcldc + 4);
        cv2 = _mm256_loadu_pd(pcldc2); cx2 = _mm256_loadu_pd(pcldc2 + 4);
        {
            double* x = aPacked;
            double* pb0 = pb;
            for (p = 0; p < k; p += 1) {
                bv0 = _mm256_broadcast_sd(pb0); pb0++;
                bv1 = _mm256_broadcast_sd(pbldb); pbldb++;
                bv2 = _mm256_broadcast_sd(pbldb2); pbldb2++;

                av0 = _mm256_loadu_pd(x); x += 4;
                cv0 = _mm256_fmadd_pd(av0, bv0, cv0);
                cv1 = _mm256_fmadd_pd(av0, bv1, cv1);
                cv2 = _mm256_fmadd_pd(av0, bv2, cv2);

                av0 = _mm256_loadu_pd(x); x += 4;
                cx0 = _mm256_fmadd_pd(av0, bv0, cx0);
                cx1 = _mm256_fmadd_pd(av0, bv1, cx1);
                cx2 = _mm256_fmadd_pd(av0, bv2, cx2);
            }
        }

        _mm256_storeu_pd(pc, cv0);
        _mm256_storeu_pd(pc + 4, cx0);
        _mm256_storeu_pd(pcldc, cv1);
        _mm256_storeu_pd(pcldc + 4, cx1);
        _mm256_storeu_pd(pcldc2, cv2);
        _mm256_storeu_pd(pcldc2 + 4, cx2);

        pc += ldc3;pb += ldb3;
    }// j loop 3 multiple
    //printf(" 8x3:j:%d ", j);
    return j;
}

/* residue dgemm kernel 8mx2n with single load and store of C matrix block
   Code could be optimized further, complete ymm register set is not used.
   Being residue kernel, its of lesser priority.
*/
inc_t bli_kernel_8mx2n(gint_t n, gint_t k, gint_t j, double* aPacked, guint_t lda, double* b, guint_t ldb, double* c, guint_t ldc)
{
    gint_t p;
    __m256d av0;
    __m256d bv0, bv1;
    __m256d cv0, cv1;
    __m256d cx0, cx1;
    double* pb, * pc;

    pb = b;
    pc = c;
    inc_t ldc2 = ldc * 2; inc_t ldb2 = ldb * 2;

    for (; j <= (n - 2); j += 2) {
        double* pcldc = pc + ldc;
        double* pbldb = pb + ldb;

        cv0 = _mm256_loadu_pd(pc);     cx0 = _mm256_loadu_pd(pc + 4);
        cv1 = _mm256_loadu_pd(pcldc);  cx1 = _mm256_loadu_pd(pcldc + 4);
        {
            double* x = aPacked;
            double* pb0 = pb;
            for (p = 0; p < k; p += 1) {
                bv0 = _mm256_broadcast_sd(pb0); pb0++;
                bv1 = _mm256_broadcast_sd(pbldb); pbldb++;

                av0 = _mm256_loadu_pd(x); x += 4;
                cv0 = _mm256_fmadd_pd(av0, bv0, cv0);
                cv1 = _mm256_fmadd_pd(av0, bv1, cv1);

                av0 = _mm256_loadu_pd(x); x += 4;
                cx0 = _mm256_fmadd_pd(av0, bv0, cx0);
                cx1 = _mm256_fmadd_pd(av0, bv1, cx1);
            }
        }
        _mm256_storeu_pd(pc, cv0);
        _mm256_storeu_pd(pc + 4, cx0);
        _mm256_storeu_pd(pcldc, cv1);
        _mm256_storeu_pd(pcldc + 4, cx1);

        pc += ldc2;pb += ldb2;
    }// j loop 2 multiple
    //printf(" 8x2:j:%d ", j);
    return j;
}

/* residue dgemm kernel 8mx1n with single load and store of C matrix block
   Code could be optimized further, complete ymm register set is not used.
   Being residue kernel, its of lesser priority.
*/
inc_t bli_kernel_8mx1n(gint_t n, gint_t k, gint_t j, double* aPacked, guint_t lda, double* b, guint_t ldb, double* c, guint_t ldc)
{
    gint_t p;
    __m256d av0;
    __m256d bv0;
    __m256d cv0;
    __m256d cx0;
    double* pb, * pc;

    pb = b;
    pc = c;

    for (; j <= (n - 1); j += 1) {
        cv0 = _mm256_loadu_pd(pc);     cx0 = _mm256_loadu_pd(pc + 4);
        double* x = aPacked;
        double* pb0 = pb;
        for (p = 0; p < k; p += 1) {
            bv0 = _mm256_broadcast_sd(pb0); pb0++;

            av0 = _mm256_loadu_pd(x); x += 4;
            cv0 = _mm256_fmadd_pd(av0, bv0, cv0);

            av0 = _mm256_loadu_pd(x); x += 4;
            cx0 = _mm256_fmadd_pd(av0, bv0, cx0);
        }
        _mm256_storeu_pd(pc, cv0);
        _mm256_storeu_pd(pc + 4, cx0);
        pc += ldc;pb += ldb;
    }// j loop 1 multiple
    //printf(" 8x1:j:%d ", j);
    return j;
}

#if 0
/************************************************************************************************************/
/************************** dgemm kernels (4mxn) column preffered  ******************************************/
/************************************************************************************************************/
/*  Residue dgemm kernel 4mx10n with single load and store of C matrix block
    alpha = +/-1 and beta = +/-1,0 handled while packing.*/
inc_t bli_kernel_4mx10n(gint_t n, gint_t k, gint_t j, double* aPacked, guint_t lda, double* b, guint_t ldb, double* c, guint_t ldc)
{
    gint_t p;
    /*            incomplete */
    __m256d av0;
    __m256d bv0, bv1, bv2, bv3;
    __m256d cv0, cv1, cv2, cv3;
    __m256d cx0, cx1, cx2, cx3;
    __m256d bv4, cv4, cx4;
    double* pb, * pc;

    pb = b;
    pc = c;
    inc_t ldc10 = ldc * 10; inc_t ldb10 = ldb * 10;

    for (j = 0; j <= (n - 10); j += 10) {

        double* pcldc = pc + ldc; double* pcldc2 = pcldc + ldc; double* pcldc3 = pcldc2 + ldc; double* pcldc4 = pcldc3 + ldc;
        double* pbldb = pb + ldb; double* pbldb2 = pbldb + ldb; double* pbldb3 = pbldb2 + ldb; double* pbldb4 = pbldb3 + ldb;

#if BLIS_ENABLE_PREFETCH
        _mm_prefetch((char*)(pc), _MM_HINT_T0);
        _mm_prefetch((char*)(pcldc), _MM_HINT_T0);
        _mm_prefetch((char*)(pcldc2), _MM_HINT_T0);
        _mm_prefetch((char*)(pcldc3), _MM_HINT_T0);
        _mm_prefetch((char*)(pcldc4), _MM_HINT_T0);

        _mm_prefetch((char*)(aPacked), _MM_HINT_T0);

        _mm_prefetch((char*)(pb), _MM_HINT_T0);
        _mm_prefetch((char*)(pbldb), _MM_HINT_T0);
        _mm_prefetch((char*)(pbldb2), _MM_HINT_T0);
        _mm_prefetch((char*)(pbldb3), _MM_HINT_T0);
        _mm_prefetch((char*)(pbldb4), _MM_HINT_T0);
#endif
        /* C matrix column major load */
#if BLIS_LOADFIRST
        cv0 = _mm256_loadu_pd(pc);
        cv1 = _mm256_loadu_pd(pcldc);
        cv2 = _mm256_loadu_pd(pcldc2);
        cv3 = _mm256_loadu_pd(pcldc3);
        cv4 = _mm256_loadu_pd(pcldc4);
#else
        cv0 = _mm256_setzero_pd();
        cv1 = _mm256_setzero_pd();
        cv2 = _mm256_setzero_pd();
        cv3 = _mm256_setzero_pd();
        cv4 = _mm256_setzero_pd();
#endif
        double* x = aPacked;
        double* pb0 = pb;
        for (p = 0; p < k; p += 1) {
            bv0 = _mm256_broadcast_sd(pb0); pb0++;
            bv1 = _mm256_broadcast_sd(pbldb); pbldb++;
            bv2 = _mm256_broadcast_sd(pbldb2); pbldb2++;
            bv3 = _mm256_broadcast_sd(pbldb3);pbldb3++;
            bv4 = _mm256_broadcast_sd(pbldb4);pbldb4++;

            av0 = _mm256_loadu_pd(x); x += 4;
            cv0 = _mm256_fmadd_pd(av0, bv0, cv0);
            cv1 = _mm256_fmadd_pd(av0, bv1, cv1);
            cv2 = _mm256_fmadd_pd(av0, bv2, cv2);
            cv3 = _mm256_fmadd_pd(av0, bv3, cv3);
            cv4 = _mm256_fmadd_pd(av0, bv4, cv4);

        }
#if BLIS_LOADFIRST
#else
        bv0 = _mm256_loadu_pd(pc);
        cv0 = _mm256_add_pd(cv0, bv0);

        bv2 = _mm256_loadu_pd(pcldc);
        cv1 = _mm256_add_pd(cv1, bv2);

        bv0 = _mm256_loadu_pd(pcldc2);
        cv2 = _mm256_add_pd(cv2, bv0);

        bv2 = _mm256_loadu_pd(pcldc3);
        cv3 = _mm256_add_pd(cv3, bv2);

        bv0 = _mm256_loadu_pd(pcldc4);
        cv4 = _mm256_add_pd(cv4, bv0);
#endif
        /* C matrix column major store */
        _mm256_storeu_pd(pc, cv0);
        _mm256_storeu_pd(pcldc, cv1);
        _mm256_storeu_pd(pcldc2, cv2);
        _mm256_storeu_pd(pcldc3, cv3);
        _mm256_storeu_pd(pcldc4, cv4);


        pc += ldc10;pb += ldb10;
    }

    return j;
}

/* residue dgemm kernel 4mx1n with single load and store of C matrix block
   Code could be optimized further, complete ymm register set is not used.
   Being residue kernel, its of lesser priority.
*/
inc_t bli_kernel_4mx1n(gint_t n, gint_t k, gint_t j, double* aPacked, guint_t lda, double* b, guint_t ldb, double* c, guint_t ldc)
{
    gint_t p;
    __m256d av0;
    __m256d bv0;
    __m256d cv0;
    double* pb, * pc;

    pb = b;
    pc = c;

    for (; j <= (n - 1); j += 1) {
        cv0 = _mm256_loadu_pd(pc);
        double* x = aPacked;
        double* pb0 = pb;
        for (p = 0; p < k; p += 1) {
            bv0 = _mm256_broadcast_sd(pb0); pb0++;
            av0 = _mm256_loadu_pd(x); x += 4;
            cv0 = _mm256_fmadd_pd(av0, bv0, cv0);
        }
        _mm256_storeu_pd(pc, cv0);
        pc += ldc;pb += ldb;
    }// j loop 1 multiple
    return j;
}

#endif
/************************************************************************************************************/
/************************** dgemm kernels (1mxn) column preffered  ******************************************/
/************************************************************************************************************/

/* residue dgemm kernel 1mx1n with single load and store of C matrix block
   Code could be optimized further, complete ymm register set is not used.
   Being residue kernel, its of lesser priority.
*/
inc_t bli_kernel_1mx1n(gint_t n, gint_t k, gint_t j, double* aPacked, guint_t lda, double* b, guint_t ldb, double* c, guint_t ldc)
{
    gint_t p;
    double a0;
    double b0;
    double c0;
    double* pb, * pc;

    pb = b;
    pc = c;

    for (; j <= (n - 1); j += 1) {
        c0 = *pc;
        double* x = aPacked;
        double* pb0 = pb;
        for (p = 0; p < k; p += 1) {
            b0 = *pb0; pb0++;
            a0 = *x;   x++;
            c0 += (a0 * b0);
        }
        *pc = c0;
        pc += ldc;pb += ldb;
    }// j loop 1 multiple
    //printf(" 1x1:j:%d ", j);
    return j;
}

/* Ax8 packing subroutine */
void bli_prepackA_8(double* pa, double* aPacked, gint_t k, guint_t lda, bool isTransA, double alpha)
{
    __m256d av0, av1, ymm0;
    if(isTransA==false)
    {
        if(alpha==1.0)
        {
            for (gint_t p = 0; p < k; p += 1) {
                av0 = _mm256_loadu_pd(pa);       av1 = _mm256_loadu_pd(pa + 4); pa += lda;
                _mm256_storeu_pd(aPacked, av0);  _mm256_storeu_pd(aPacked + 4, av1);
                aPacked += BLIS_MX8;
            }
        }
        else if(alpha==-1.0)
        {
            ymm0 = _mm256_setzero_pd();//set zero
            for (gint_t p = 0; p < k; p += 1) {
                av0 = _mm256_loadu_pd(pa);       av1 = _mm256_loadu_pd(pa + 4); pa += lda;
                av0 = _mm256_sub_pd(ymm0,av0);   av1 = _mm256_sub_pd(ymm0,av1); // a = 0 - a;
                _mm256_storeu_pd(aPacked, av0);  _mm256_storeu_pd(aPacked + 4, av1);
                aPacked += BLIS_MX8;
            }
        }
    }
    else
    {
        if(alpha==1.0)
        {
            //A Transpose case:
            for (gint_t i = 0; i < BLIS_MX8 ; i++)
            {
                gint_t idx = i * lda;
                for (gint_t p = 0; p < k; p ++)
                {
                    double ar_ = *(pa+idx+p);
                    gint_t sidx = p * BLIS_MX8;
                    *(aPacked + sidx + i) = ar_;
                }
            }
        }
        else if(alpha==-1.0)
        {
            //A Transpose case:
            for (gint_t i = 0; i < BLIS_MX8 ; i++)
            {
                gint_t idx = i * lda;
                for (gint_t p = 0; p < k; p ++)
                {
                    double ar_ = *(pa+idx+p);
                    gint_t sidx = p * BLIS_MX8;
                    *(aPacked + sidx + i) = -ar_;
                }
            }
        }
    }
}

/* Ax4 packing subroutine */
void bli_prepackA_4(double* pa, double* aPacked, gint_t k, guint_t lda, bool isTransA, double alpha)
{
    __m256d av0, ymm0;
    if(isTransA==false)
    {
        if(alpha==1.0)
        {
            for (gint_t p = 0; p < k; p += 1) {
                av0 = _mm256_loadu_pd(pa);       pa += lda;
                _mm256_storeu_pd(aPacked, av0);
                aPacked += BLIS_MX4;
            }
        }
        else if(alpha==-1.0)
        {
            ymm0 = _mm256_setzero_pd();//set zero
            for (gint_t p = 0; p < k; p += 1) {
                av0 = _mm256_loadu_pd(pa);       pa += lda;
                av0 = _mm256_sub_pd(ymm0,av0);   // a = 0 - a;
                _mm256_storeu_pd(aPacked, av0);
                aPacked += BLIS_MX4;
            }
        }
    }
    else
    {
        if(alpha==1.0)
        {
            //A Transpose case:
            for (gint_t i = 0; i < BLIS_MX4 ; i++)
            {
                gint_t idx = i * lda;
                for (gint_t p = 0; p < k; p ++)
                {
                    double ar_ = *(pa+idx+p);
                    gint_t sidx = p * BLIS_MX4;
                    *(aPacked + sidx + i) = ar_;
                }
            }
        }
        else if(alpha==-1.0)
        {
            //A Transpose case:
            for (gint_t i = 0; i < BLIS_MX4 ; i++)
            {
                gint_t idx = i * lda;
                for (gint_t p = 0; p < k; p ++)
                {
                    double ar_ = *(pa+idx+p);
                    gint_t sidx = p * BLIS_MX4;
                    *(aPacked + sidx + i) = -ar_;
                }
            }
        }
    }
}

/* Ax1 packing subroutine */
void bli_prepackA_1(double* pa, double* aPacked, gint_t k, guint_t lda, bool isTransA, double alpha)
{
    if(isTransA==false)
    {
        if(alpha==1.0)
        {
            for (gint_t p = 0; p < k; p += 1) {
                *aPacked = *pa;
                pa += lda;
                aPacked++;
            }
        }
        else if(alpha==-1.0)
        {
            for (gint_t p = 0; p < k; p += 1) {
                *aPacked = -(*pa);
                pa += lda;
                aPacked++;
            }
        }
    }
    else
    {
        if(alpha==1.0)
        {
            //A Transpose case:
            for (gint_t p = 0; p < k; p ++)
            {
                double ar_ = *(pa+p);
                *(aPacked + p) = ar_;
            }
        }
        else if(alpha==-1.0)
        {
            //A Transpose case:
            for (gint_t p = 0; p < k; p ++)
            {
                double ar_ = *(pa+p);
                *(aPacked + p) = -ar_;
            }
        }
    }
}

/************************************************************************************************************/
/***************************************** dgemm_sqp implementation******************************************/
/************************************************************************************************************/
/* dgemm_sqp implementation packs A matrix based on lda and m size. dgemm_sqp focuses mainly on square matrixes
   but also supports non-square matrix. Current support is limiteed to m multiple of 8 and column storage.
   C = AxB and C = AtxB is handled in the design. AtxB case is done by transposing A matrix while packing A.
   In majority of use-case, alpha are +/-1, so instead of explicitly multiplying alpha its done
   during packing itself by changing sign.
*/
static err_t bli_dgemm_sqp_m8(gint_t m, gint_t n, gint_t k, double* a, guint_t lda, double* b, guint_t ldb, double* c, guint_t ldc, bool isTransA, double alpha, gint_t mx, gint_t* p_istart)
{
    double* aPacked;
    double* aligned = NULL;
    gint_t i;

    bool pack_on = false;
    if((m!=mx)||(m!=lda)||isTransA)
    {
        pack_on = true;
    }

    if(pack_on==true)
    {
        aligned = (double*)bli_malloc_user(sizeof(double) * k * mx);
        if(aligned==NULL)
        {
            return BLIS_MALLOC_RETURNED_NULL;
        }
    }
    for (i = (*p_istart); i <= (m-mx); i += mx) //this loop can be threaded. no of workitems = m/8
    {
        inc_t j = 0;
        double* ci = c + i;
        if(pack_on==true)
        {
            aPacked = aligned;
            double *pa = a + i;
            if(isTransA==true)
            {
                pa = a + (i*lda);
            }
            /* should be changed to func pointer */
            if(mx==8)
            {
                bli_prepackA_8(pa, aPacked, k, lda, isTransA, alpha);
            }
#if 0//mx=4, kernels not yet implemented.
            else if(mx==4)
            {
                bli_prepackA_4(pa, aPacked, k, lda, isTransA, alpha);
            }
#endif//0
            else if(mx==1)
            {
                bli_prepackA_1(pa, aPacked, k, lda, isTransA, alpha);
            }
        }
        else
        {
            aPacked = a+i;
        }
        if(mx==8)
        {
            //printf("\n mx8i:%3ld ", i);
            j = bli_kernel_8mx6n(n, k, j, aPacked, lda, b, ldb, ci, ldc);
            if (j <= (n - 5))
            {
                j = bli_kernel_8mx5n(n, k, j, aPacked, lda, b + (j * ldb), ldb, ci + (j * ldc), ldc);
            }
            if (j <= (n - 4))
            {
                j = bli_kernel_8mx4n(n, k, j, aPacked, lda, b + (j * ldb), ldb, ci + (j * ldc), ldc);
            }
            if (j <= (n - 3))
            {
                j = bli_kernel_8mx3n(n, k, j, aPacked, lda, b + (j * ldb), ldb, ci + (j * ldc), ldc);
            }
            if (j <= (n - 2))
            {
                j = bli_kernel_8mx2n(n, k, j, aPacked, lda, b + (j * ldb), ldb, ci + (j * ldc), ldc);
            }
            if (j <= (n - 1))
            {
                j = bli_kernel_8mx1n(n, k, j, aPacked, lda, b + (j * ldb), ldb, ci + (j * ldc), ldc);
            }
        }
        /* mx==4 to be implemented */
        else if(mx==1)
        {
            //printf("\n mx1i:%3ld ", i);
            j = bli_kernel_1mx1n(n, k, j, aPacked, lda, b, ldb, ci, ldc);
        }
        *p_istart = i + mx;
    }

    if(pack_on==true)
    {
        bli_free_user(aligned);
    }

    return BLIS_SUCCESS;
}

gint_t bli_getaligned(mem_block* mem_req)
{

    guint_t memSize = mem_req->data_size * mem_req->size;
    if (memSize == 0)
    {
        return -1;
    }
    memSize += 128;// extra 128 bytes added for alignment. Could be minimized to 64.
#if MEM_ALLOC
#ifdef BLIS_ENABLE_MEM_TRACING
    printf( "malloc(): size %ld\n",( long )memSize;
    fflush( stdout );
#endif
    mem_req->unalignedBuf = (double*)malloc(memSize);
    if (mem_req->unalignedBuf == NULL)
    {
        return -1;
    }

    int64_t address = (int64_t)mem_req->unalignedBuf;
    address += (-address) & 63; //64 bytes alignment done.
    mem_req->alignedBuf = (double*)address;
#else
    mem_req->alignedBuf = bli_malloc_user( memSize );
    if (mem_req->alignedBuf == NULL)
    {
        return -1;
    }
#endif
    return 0;
}

gint_t bli_allocateWorkspace(gint_t n, gint_t k, mem_block *mxr, mem_block *mxi, mem_block *msx)
{
    //allocate workspace
    mxr->data_size = mxi->data_size = msx->data_size = sizeof(double);
    mxr->size = mxi->size = n * k;
    msx->size = n * k;
    mxr->alignedBuf = mxi->alignedBuf = msx->alignedBuf = NULL;

    if (!((bli_getaligned(mxr) == 0) && (bli_getaligned(mxi) == 0) && (bli_getaligned(msx) == 0)))
    {
#if MEM_ALLOC
        if(mxr->unalignedBuf)
        {
            free(mxr->unalignedBuf);
        }
        if(mxi->unalignedBuf)
        {
            free(mxi->unalignedBuf);
        }
        if(msx->unalignedBuf)
        {
            free(msx->unalignedBuf);
        }
#else
        bli_free_user(mxr->alignedBuf);
        bli_free_user(mxi->alignedBuf);
        bli_free_user(msx->alignedBuf);
#endif
        return -1;
    }
    return 0;
}

void bli_add_m(gint_t m,gint_t n,double* w,double* c)
{
    double* pc = c;
    double* pw = w;
    gint_t count = m*n;
    gint_t i = 0;
    __m256d cv0, wv0;

    for (; i <= (count-4); i+=4)
    {
        cv0 = _mm256_loadu_pd(pc);
        wv0 = _mm256_loadu_pd(pw); pw += 4;
        cv0 = _mm256_add_pd(cv0,wv0);
        _mm256_storeu_pd(pc, cv0); pc += 4;
    }
    for (; i < count; i++)
    {
        *pc = *pc + *pw;
        pc++; pw++;
    }


}

void bli_sub_m(gint_t m, gint_t n, double* w, double* c)
{
    double* pc = c;
    double* pw = w;
    gint_t count = m*n;
    gint_t i = 0;
    __m256d cv0, wv0;

    for (; i <= (count-4); i+=4)
    {
        cv0 = _mm256_loadu_pd(pc);
        wv0 = _mm256_loadu_pd(pw); pw += 4;
        cv0 = _mm256_sub_pd(cv0,wv0);
        _mm256_storeu_pd(pc, cv0); pc += 4;
    }
    for (; i < count; i++)
    {
        *pc = *pc - *pw;
        pc++; pw++;
    }
}

/* Pack real and imaginary parts in separate buffers and also multipy with multiplication factor */
void bli_packX_real_imag(double* pb, guint_t n, guint_t k, guint_t ldb, double* pbr, double* pbi, double mul, gint_t mx)
{
    gint_t j, p;
    __m256d av0, av1, zerov;
    __m256d tv0, tv1;
    gint_t max_k = (k*2)-8;

    if((mul ==1.0)||(mul==-1.0))
    {
        if(mul ==1.0) /* handles alpha or beta = 1.0 */
        {
            for (j = 0; j < n; j++)
            {
                for (p = 0; p <= max_k; p += 8)
                {
                    double* pbp = pb + p;
                    av0 = _mm256_loadu_pd(pbp);   //ai1, ar1, ai0, ar0
                    av1 = _mm256_loadu_pd(pbp+4); //ai3, ar3, ai2, ar2
                    //
                    tv0 = _mm256_permute2f128_pd(av0, av1, 0x20);//ai2, ar2, ai0, ar0
                    tv1 = _mm256_permute2f128_pd(av0, av1, 0x31);//ai3, ar3, ai1, ar1
                    av0 = _mm256_unpacklo_pd(tv0, tv1);//ar3, ar2, ar1, ar0
                    av1 = _mm256_unpackhi_pd(tv0, tv1);//ai3, ai2, ai1, ai0

                    _mm256_storeu_pd(pbr, av0); pbr += 4;
                    _mm256_storeu_pd(pbi, av1); pbi += 4;
                }

                for (; p < (k*2); p += 2)// (real + imag)*k
                {
                    double br = *(pb + p) ;
                    double bi = *(pb + p + 1);
                    *pbr = br;
                    *pbi = bi;
                    pbr++; pbi++;
                }
                pb = pb + ldb;
            }
        }
        else /* handles alpha or beta = - 1.0 */
        {
            zerov = _mm256_setzero_pd();
            for (j = 0; j < n; j++)
            {
                for (p = 0; p <= max_k; p += 8)
                {
                    double* pbp = pb + p;
                    av0 = _mm256_loadu_pd(pbp); //ai1, ar1, ai0, ar0
                    av1 = _mm256_loadu_pd(pbp+4);//ai3, ar3, ai2, ar2

                    tv0 = _mm256_permute2f128_pd(av0, av1, 0x20);//ai2, ar2, ai0, ar0
                    tv1 = _mm256_permute2f128_pd(av0, av1, 0x31);//ai3, ar3, ai1, ar1
                    av0 = _mm256_unpacklo_pd(tv0, tv1);//ar3, ar2, ar1, ar0
                    av1 = _mm256_unpackhi_pd(tv0, tv1);//ai3, ai2, ai1, ai0

                    //negate
                    av0 = _mm256_sub_pd(zerov,av0);
                    av1 = _mm256_sub_pd(zerov,av1);

                    _mm256_storeu_pd(pbr, av0); pbr += 4;
                    _mm256_storeu_pd(pbi, av1); pbi += 4;
                }

                for (; p < (k*2); p += 2)// (real + imag)*k
                {
                    double br = -*(pb + p) ;
                    double bi = -*(pb + p + 1);
                    *pbr = br;
                    *pbi = bi;
                    pbr++; pbi++;
                }
                pb = pb + ldb;
            }
        }
    }
    else /* handles alpha or beta is not equal +/- 1.0 */
    {
        for (j = 0; j < n; j++)
        {
            for (p = 0; p < (k*2); p += 2)// (real + imag)*k
            {
                double br_ = mul * (*(pb + p));
                double bi_ = mul * (*(pb + p + 1));
                *pbr = br_;
                *pbi = bi_;
                pbr++; pbi++;
            }
            pb = pb + ldb;
        }
    }
}

/* Pack real and imaginary parts in separate buffers and compute sum of real and imaginary part */
void bli_packX_real_imag_sum(double* pb, guint_t n, guint_t k, guint_t ldb, double* pbr, double* pbi, double* pbs, double mul, gint_t mx)
{
    gint_t j, p;
    __m256d av0, av1, zerov;
    __m256d tv0, tv1, sum;
    gint_t max_k = (k*2) - 8;
    if((mul ==1.0)||(mul==-1.0))
    {
        if(mul ==1.0)
        {
            for (j = 0; j < n; j++)
            {
                for (p=0; p <= max_k; p += 8)
                {
                    double* pbp = pb + p;
                    av0 = _mm256_loadu_pd(pbp);//ai1, ar1, ai0, ar0
                    av1 = _mm256_loadu_pd(pbp+4);//ai3, ar3, ai2, ar2

                    tv0 = _mm256_permute2f128_pd(av0, av1, 0x20);//ai2, ar2, ai0, ar0
                    tv1 = _mm256_permute2f128_pd(av0, av1, 0x31);//ai3, ar3, ai1, ar1
                    av0 = _mm256_unpacklo_pd(tv0, tv1);//ar3, ar2, ar1, ar0
                    av1 = _mm256_unpackhi_pd(tv0, tv1);//ai3, ai2, ai1, ai0
                    sum = _mm256_add_pd(av0, av1);
                    _mm256_storeu_pd(pbr, av0); pbr += 4;
                    _mm256_storeu_pd(pbi, av1); pbi += 4;
                    _mm256_storeu_pd(pbs, sum); pbs += 4;
                }

                for (; p < (k*2); p += 2)// (real + imag)*k
                {
                    double br = *(pb + p) ;
                    double bi = *(pb + p + 1);
                    *pbr = br;
                    *pbi = bi;
                    *pbs = br + bi;

                    pbr++; pbi++; pbs++;
                }
                pb = pb + ldb;
            }
        }
        else
        {
            zerov = _mm256_setzero_pd();
            for (j = 0; j < n; j++)
            {
                for (p = 0; p <= max_k; p += 8)
                {
                    double* pbp = pb + p;
                    av0 = _mm256_loadu_pd(pbp);//ai1, ar1, ai0, ar0
                    av1 = _mm256_loadu_pd(pbp+4);//ai3, ar3, ai2, ar2

                    tv0 = _mm256_permute2f128_pd(av0, av1, 0x20);//ai2, ar2, ai0, ar0
                    tv1 = _mm256_permute2f128_pd(av0, av1, 0x31);//ai3, ar3, ai1, ar1
                    av0 = _mm256_unpacklo_pd(tv0, tv1);//ar3, ar2, ar1, ar0
                    av1 = _mm256_unpackhi_pd(tv0, tv1);//ai3, ai2, ai1, ai0

                    //negate
                    av0 = _mm256_sub_pd(zerov,av0);
                    av1 = _mm256_sub_pd(zerov,av1);

                    sum = _mm256_add_pd(av0, av1);
                    _mm256_storeu_pd(pbr, av0); pbr += 4;
                    _mm256_storeu_pd(pbi, av1); pbi += 4;
                    _mm256_storeu_pd(pbs, sum); pbs += 4;
                }

                for (; p < (k*2); p += 2)// (real + imag)*k
                {
                    double br = -*(pb + p) ;
                    double bi = -*(pb + p + 1);
                    *pbr = br;
                    *pbi = bi;
                    *pbs = br + bi;

                    pbr++; pbi++; pbs++;
                }
                pb = pb + ldb;
            }
        }
    }
    else
    {
        for (j = 0; j < n; j++)
        {
            for (p = 0; p < (k*2); p += 2)// (real + imag)*k
            {
                double br_ = mul * (*(pb + p));
                double bi_ = mul * (*(pb + p + 1));
                *pbr = br_;
                *pbi = bi_;
                *pbs = br_ + bi_;

                pbr++; pbi++; pbs++;
            }
            pb = pb + ldb;
        }
    }
}

/* Pack real and imaginary parts of A matrix in separate buffers and compute sum of real and imaginary part */
void bli_packA_real_imag_sum(double *pa, gint_t i, guint_t k, guint_t lda, double *par, double *pai, double *pas, bool isTransA, gint_t mx)
{
    __m256d av0, av1, av2, av3;
    __m256d tv0, tv1, sum;
    gint_t p;

    if(mx==8)
    {
        if(isTransA==false)
        {
            pa = pa +i;
            for (p = 0; p < k; p += 1)
            {
                //for (int ii = 0; ii < MX8 * 2; ii += 2) //real + imag : Rkernel needs 8 elements each.
                av0 = _mm256_loadu_pd(pa);
                av1 = _mm256_loadu_pd(pa+4);
                av2 = _mm256_loadu_pd(pa+8);
                av3 = _mm256_loadu_pd(pa+12);

                tv0 = _mm256_permute2f128_pd(av0, av1, 0x20);
                tv1 = _mm256_permute2f128_pd(av0, av1, 0x31);
                av0 = _mm256_unpacklo_pd(tv0, tv1);
                av1 = _mm256_unpackhi_pd(tv0, tv1);
                sum = _mm256_add_pd(av0, av1);
                _mm256_storeu_pd(par, av0); par += 4;
                _mm256_storeu_pd(pai, av1); pai += 4;
                _mm256_storeu_pd(pas, sum); pas += 4;

                tv0 = _mm256_permute2f128_pd(av2, av3, 0x20);
                tv1 = _mm256_permute2f128_pd(av2, av3, 0x31);
                av2 = _mm256_unpacklo_pd(tv0, tv1);
                av3 = _mm256_unpackhi_pd(tv0, tv1);
                sum = _mm256_add_pd(av2, av3);
                _mm256_storeu_pd(par, av2); par += 4;
                _mm256_storeu_pd(pai, av3); pai += 4;
                _mm256_storeu_pd(pas, sum); pas += 4;

                pa = pa + lda;
            }
        }
        else
        {
            gint_t idx = (i/2) * lda;
            pa = pa + idx;
    #if 0
            for (int p = 0; p <= ((2*k)-8); p += 8)
            {
                //for (int ii = 0; ii < MX8 * 2; ii += 2) //real + imag : Rkernel needs 8 elements each.
                av0 = _mm256_loadu_pd(pa);
                av1 = _mm256_loadu_pd(pa+4);
                av2 = _mm256_loadu_pd(pa+8);
                av3 = _mm256_loadu_pd(pa+12);

                //transpose 4x4
                tv0 = _mm256_unpacklo_pd(av0, av1);
                tv1 = _mm256_unpackhi_pd(av0, av1);
                tv2 = _mm256_unpacklo_pd(av2, av3);
                tv3 = _mm256_unpackhi_pd(av2, av3);

                av0 = _mm256_permute2f128_pd(tv0, tv2, 0x20);
                av1 = _mm256_permute2f128_pd(tv1, tv3, 0x20);
                av2 = _mm256_permute2f128_pd(tv0, tv2, 0x31);
                av3 = _mm256_permute2f128_pd(tv1, tv3, 0x31);

                //get real, imag and sum
                tv0 = _mm256_permute2f128_pd(av0, av1, 0x20);
                tv1 = _mm256_permute2f128_pd(av0, av1, 0x31);
                av0 = _mm256_unpacklo_pd(tv0, tv1);
                av1 = _mm256_unpackhi_pd(tv0, tv1);
                sum = _mm256_add_pd(av0, av1);
                _mm256_storeu_pd(par, av0); par += 4;
                _mm256_storeu_pd(pai, av1); pai += 4;
                _mm256_storeu_pd(pas, sum); pas += 4;

                tv0 = _mm256_permute2f128_pd(av2, av3, 0x20);
                tv1 = _mm256_permute2f128_pd(av2, av3, 0x31);
                av2 = _mm256_unpacklo_pd(tv0, tv1);
                av3 = _mm256_unpackhi_pd(tv0, tv1);
                sum = _mm256_add_pd(av2, av3);
                _mm256_storeu_pd(par, av2); par += 4;
                _mm256_storeu_pd(pai, av3); pai += 4;
                _mm256_storeu_pd(pas, sum); pas += 4;

                pa = pa + lda;
            }
    #endif
            //A Transpose case:
            for (gint_t ii = 0; ii < BLIS_MX8 ; ii++)
            {
                gint_t idx = ii * lda;
                gint_t sidx;
                gint_t max_k = (k*2) - 8;
                for (p = 0; p <= max_k; p += 8)
                {
                    double ar0_ = *(pa + idx + p);
                    double ai0_ = *(pa + idx + p + 1);

                    double ar1_ = *(pa + idx + p + 2);
                    double ai1_ = *(pa + idx + p + 3);

                    double ar2_ = *(pa + idx + p + 4);
                    double ai2_ = *(pa + idx + p + 5);

                    double ar3_ = *(pa + idx + p + 6);
                    double ai3_ = *(pa + idx + p + 7);

                    sidx = (p/2) * BLIS_MX8;
                    *(par + sidx + ii) = ar0_;
                    *(pai + sidx + ii) = ai0_;
                    *(pas + sidx + ii) = ar0_ + ai0_;

                    sidx = ((p+2)/2) * BLIS_MX8;
                    *(par + sidx + ii) = ar1_;
                    *(pai + sidx + ii) = ai1_;
                    *(pas + sidx + ii) = ar1_ + ai1_;

                    sidx = ((p+4)/2) * BLIS_MX8;
                    *(par + sidx + ii) = ar2_;
                    *(pai + sidx + ii) = ai2_;
                    *(pas + sidx + ii) = ar2_ + ai2_;

                    sidx = ((p+6)/2) * BLIS_MX8;
                    *(par + sidx + ii) = ar3_;
                    *(pai + sidx + ii) = ai3_;
                    *(pas + sidx + ii) = ar3_ + ai3_;

                }

                for (; p < (k*2); p += 2)
                {
                    double ar_ = *(pa + idx + p);
                    double ai_ = *(pa + idx + p + 1);
                    gint_t sidx = (p/2) * BLIS_MX8;
                    *(par + sidx + ii) = ar_;
                    *(pai + sidx + ii) = ai_;
                    *(pas + sidx + ii) = ar_ + ai_;
                }
            }
        }
    }   //mx==8
    else//mx==1
    {
        if(isTransA==false)
        {
            pa = pa +i;
            //A No transpose case:
            for (gint_t p = 0; p < k; p += 1)
            {
                gint_t idx = p * lda;
                for (gint_t ii = 0; ii < (mx*2) ; ii += 2)
                { //real + imag : Rkernel needs 8 elements each.
                    double ar_ = *(pa + idx + ii);
                    double ai_ = *(pa + idx + ii + 1);
                    *par = ar_;
                    *pai = ai_;
                    *pas = ar_ + ai_;
                    par++; pai++; pas++;
                }
            }
        }
        else
        {
            gint_t idx = (i/2) * lda;
            pa = pa + idx;

            //A Transpose case:
            for (gint_t ii = 0; ii < mx ; ii++)
            {
                gint_t idx = ii * lda;
                gint_t sidx;
                for (p = 0; p < (k*2); p += 2)
                {
                    double ar0_ = *(pa + idx + p);
                    double ai0_ = *(pa + idx + p + 1);

                    sidx = (p/2) * mx;
                    *(par + sidx + ii) = ar0_;
                    *(pai + sidx + ii) = ai0_;
                    *(pas + sidx + ii) = ar0_ + ai0_;

                }
            }
        }
    }//mx==1
}

/************************************************************************************************************/
/***************************************** 3m_sqp implementation   ******************************************/
/************************************************************************************************************/
/* 3m_sqp implementation packs A, B and C matrix and uses dgemm_sqp real kernel implementation.
   3m_sqp focuses mainly on square matrixes but also supports non-square matrix. Current support is limiteed to
   m multiple of 8 and column storage.
*/
static err_t bli_zgemm_sqp_m8(gint_t m, gint_t n, gint_t k, double* a, guint_t lda, double* b, guint_t ldb, double* c, guint_t ldc, double alpha, double beta, bool isTransA, gint_t mx, gint_t* p_istart)
{
    inc_t m2 = m<<1;
    inc_t mxmul2 = mx<<1;
    if((*p_istart) > (m2-mxmul2))
    {
        return BLIS_SUCCESS;
    }
    /* B matrix */
    double* br, * bi, * bs;
    mem_block mbr, mbi, mbs;
    if(bli_allocateWorkspace(n, k, &mbr, &mbi, &mbs)!=0)
    {
        return BLIS_FAILURE;
    }
    br = (double*)mbr.alignedBuf;
    bi = (double*)mbi.alignedBuf;
    bs = (double*)mbs.alignedBuf;

    //multiply lda, ldb and ldc by 2 to account for real and imaginary components per dcomplex.
    lda = lda * 2;
    ldb = ldb * 2;
    ldc = ldc * 2;

    /* Split    b  (br, bi) and
       compute  bs = br + bi    */
    double* pbr = br;
    double* pbi = bi;
    double* pbs = bs;

    gint_t j;

    /* b matrix real and imag packing and compute. */
    bli_packX_real_imag_sum(b, n, k, ldb, pbr, pbi, pbs, alpha, mx);
#if 0//bug in above api to be fixed for mx = 1
    if((alpha ==1.0)||(alpha==-1.0))
    {
        if(alpha ==1.0)
        {
            for (j = 0; j < n; j++)
            {
                for (p = 0; p < (k*2); p += 2)// (real + imag)*k
                {
                    double br_ = b[(j * ldb) + p];
                    double bi_ = b[(j * ldb) + p + 1];
                    *pbr = br_;
                    *pbi = bi_;
                    *pbs = br_ + bi_;

                    pbr++; pbi++; pbs++;
                }
            }
        }
        else
        {
            for (j = 0; j < n; j++)
            {
                for (p = 0; p < (k*2); p += 2)// (real + imag)*k
                {
                    double br_ = -b[(j * ldb) + p];
                    double bi_ = -b[(j * ldb) + p + 1];
                    *pbr = br_;
                    *pbi = bi_;
                    *pbs = br_ + bi_;

                    pbr++; pbi++; pbs++;
                }
            }
        }
    }
    else
    {
        for (j = 0; j < n; j++)
        {
            for (p = 0; p < (k*2); p += 2)// (real + imag)*k
            {
                double br_ = alpha * b[(j * ldb) + p];
                double bi_ = alpha * b[(j * ldb) + p + 1];
                *pbr = br_;
                *pbi = bi_;
                *pbs = br_ + bi_;

                pbr++; pbi++; pbs++;
            }
        }
    }
#endif
    /* Workspace memory allocation currently done dynamically
    This needs to be taken from already allocated memory pool in application for better performance */
    /* A matrix */
    double* ar, * ai, * as;
    mem_block mar, mai, mas;
    if(bli_allocateWorkspace(mx, k, &mar, &mai, &mas) !=0)
    {
        return BLIS_FAILURE;
    }
    ar = (double*)mar.alignedBuf;
    ai = (double*)mai.alignedBuf;
    as = (double*)mas.alignedBuf;


    /* w matrix */
    double* w;
    mem_block mw;
    mw.data_size = sizeof(double);
    mw.size = mx * n;
    if (bli_getaligned(&mw) != 0)
    {
        return BLIS_FAILURE;
    }
    w = (double*)mw.alignedBuf;

    /* cr matrix */
    double* cr;
    mem_block mcr;
    mcr.data_size = sizeof(double);
    mcr.size = mx * n;
    if (bli_getaligned(&mcr) != 0)
    {
        return BLIS_FAILURE;
    }
    cr = (double*)mcr.alignedBuf;


    /* ci matrix */
    double* ci;
    mem_block mci;
    mci.data_size = sizeof(double);
    mci.size = mx * n;
    if (bli_getaligned(&mci) != 0)
    {
        return BLIS_FAILURE;
    }
    ci = (double*)mci.alignedBuf;
    inc_t i;
    gint_t max_m = (m2-mxmul2);
    for (i = (*p_istart); i <= max_m; i += mxmul2) //this loop can be threaded.
    {
        ////////////// operation 1 /////////////////

        /* Split    a  (ar, ai) and
           compute  as = ar + ai   */
        double* par = ar;
        double* pai = ai;
        double* pas = as;

        /* a matrix real and imag packing and compute. */
        bli_packA_real_imag_sum(a, i, k, lda, par, pai, pas, isTransA, mx);

        double* pcr = cr;
        double* pci = ci;

        //Split Cr and Ci and beta multiplication done.
        double* pc = c + i;
        bli_packX_real_imag(pc, n, mx, ldc, pcr, pci, beta, mx);
#if 0   //bug in above api to be fixed for mx = 1
        if((beta ==1.0)||(beta==-1.0))
        {
            if(beta ==1.0)
            {
                for (j = 0; j < n; j++)
                {
                    for (gint_t ii = 0; ii < mxmul2; ii += 2)
                    {
                        double cr_ = c[(j * ldc) + i + ii];
                        double ci_ = c[(j * ldc) + i + ii + 1];
                        *pcr = cr_;
                        *pci = ci_;
                        pcr++; pci++;
                    }
                }
            }
            else
            {
                //beta = -1.0
                for (j = 0; j < n; j++)
                {
                    for (gint_t ii = 0; ii < mxmul2; ii += 2)
                    {
                        double cr_ = -c[(j * ldc) + i + ii];
                        double ci_ = -c[(j * ldc) + i + ii + 1];
                        *pcr = cr_;
                        *pci = ci_;
                        pcr++; pci++;
                    }
                }
            }
        }
        else
        {
            for (j = 0; j < n; j++)
            {
                for (gint_t ii = 0; ii < mxmul2; ii += 2)
                {
                    double cr_ = beta*c[(j * ldc) + i + ii];
                    double ci_ = beta*c[(j * ldc) + i + ii + 1];
                    *pcr = cr_;
                    *pci = ci_;
                    pcr++; pci++;
                }
            }
        }
#endif
        //Ci := rgemm( SA, SB, Ci )
        gint_t istart = 0;
        gint_t* p_is = &istart;
        *p_is = 0;
        bli_dgemm_sqp_m8(mx, n, k, as, mx, bs, k, ci, mx, false, 1.0, mx, p_is);



        ////////////// operation 2 /////////////////
        //Wr: = dgemm_sqp(Ar, Br, 0)  // Wr output 8xn
        double* wr = w;
        for (j = 0; j < n; j++) {
            for (gint_t ii = 0; ii < mx; ii += 1) {
                *wr = 0;
                wr++;
            }
        }
        wr = w;

        *p_is = 0;
        bli_dgemm_sqp_m8(mx, n, k, ar, mx, br, k, wr, mx, false, 1.0, mx, p_is);
        //Cr : = addm(Wr, Cr)
        bli_add_m(mx, n, wr, cr);
        //Ci : = subm(Wr, Ci)
        bli_sub_m(mx, n, wr, ci);




        ////////////// operation 3 /////////////////
        //Wi : = dgemm_sqp(Ai, Bi, 0)  // Wi output 8xn
        double* wi = w;
        for (j = 0; j < n; j++) {
            for (gint_t ii = 0; ii < mx; ii += 1) {
                *wi = 0;
                wi++;
            }
        }
        wi = w;

        *p_is = 0;
        bli_dgemm_sqp_m8(mx, n, k, ai, mx, bi, k, wi, mx, false, 1.0, mx, p_is);
        //Cr : = subm(Wi, Cr)
        bli_sub_m(mx, n, wi, cr);
        //Ci : = subm(Wi, Ci)
        bli_sub_m(mx, n, wi, ci);


        pcr = cr;
        pci = ci;

        for (j = 0; j < n; j++)
        {
            for (gint_t ii = 0; ii < mxmul2; ii += 2)
            {
                c[(j * ldc) + i + ii]     = *pcr;
                c[(j * ldc) + i + ii + 1] = *pci;
                pcr++; pci++;
            }
        }
        *p_istart = i + mxmul2;
    }

#if MEM_ALLOC
    if(mar.unalignedBuf)
    {
        free(mar.unalignedBuf);
    }
    if(mai.unalignedBuf)
    {
        free(mai.unalignedBuf);
    }
    if(mas.unalignedBuf)
    {
        free(mas.unalignedBuf);
    }
    if(mw.unalignedBuf)
    {
        free(mw.unalignedBuf);
    }
    if(mcr.unalignedBuf)
    {
        free(mcr.unalignedBuf);
    }

    if(mci.unalignedBuf)
    {
        free(mci.unalignedBuf);
    }
    if(mbr.unalignedBuf)
    {
        free(mbr.unalignedBuf);
    }

    if(mbi.unalignedBuf)
    {
        free(mbi.unalignedBuf);
    }

    if(mbs.unalignedBuf)
    {
        free(mbs.unalignedBuf);
    }
#else
    /* free workspace buffers */
    bli_free_user(mbr.alignedBuf);
    bli_free_user(mbi.alignedBuf);
    bli_free_user(mbs.alignedBuf);
    bli_free_user(mar.alignedBuf);
    bli_free_user(mai.alignedBuf);
    bli_free_user(mas.alignedBuf);
    bli_free_user(mw.alignedBuf);
    bli_free_user(mcr.alignedBuf);
    bli_free_user(mci.alignedBuf);
#endif
    return BLIS_SUCCESS;
}