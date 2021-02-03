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
#define ENABLE_PREFETCH 1

#define MX8 8
#define DEBUG_3M_SQP 0

typedef struct  {
    siz_t data_size;
    siz_t size;
    void* alignedBuf;
}mem_block;

static err_t bli_zgemm_sqp_m8(gint_t m, gint_t n, gint_t k, double* a, guint_t lda, double* b, guint_t ldb, double* c, guint_t ldc, double alpha, double beta, bool isTransA);
static err_t bli_dgemm_m8(gint_t m, gint_t n, gint_t k, double* a, guint_t lda, double* b, guint_t ldb, double* c, guint_t ldc, bool isTransA, double alpha);

/*
* The bli_gemm_sqp (square packed) function would focus of square matrix sizes, where m=n=k.
* Custom 8mxn block kernels with single load and store of C matrix, to perform gemm computation.
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
    if(m8rem!=0)
    {
        /* Residue kernel m4 and m1 to be implemented */
        return BLIS_NOT_YET_IMPLEMENTED;
    }

    double* ap     = ( double* )bli_obj_buffer( a );
    double* bp     = ( double* )bli_obj_buffer( b );
    double* cp     = ( double* )bli_obj_buffer( c );
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
        return bli_zgemm_sqp_m8( m, n, k, ap, lda, bp, ldb, cp, ldc, alpha_real, beta_real, isTransA);
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
        return bli_dgemm_m8( m, n, k, ap, lda, bp, ldb, cp, ldc, isTransA, (*alpha_cast));
    }

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
    return BLIS_NOT_YET_IMPLEMENTED;
};

/*  core dgemm kernel 8mx5n with single load and store of C matrix block
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

    for (j = 0; j <= (n - 5); j += 5) {

        double* pcldc = pc + ldc; double* pcldc2 = pcldc + ldc; double* pcldc3 = pcldc2 + ldc; double* pcldc4 = pcldc3 + ldc;
        double* pbldb = pb + ldb; double* pbldb2 = pbldb + ldb; double* pbldb3 = pbldb2 + ldb; double* pbldb4 = pbldb3 + ldb;

#if ENABLE_PREFETCH
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
                aPacked += MX8;
            }
        }
        else if(alpha==-1.0)
        {
            ymm0 = _mm256_setzero_pd();//set zero
            for (gint_t p = 0; p < k; p += 1) {
                av0 = _mm256_loadu_pd(pa);       av1 = _mm256_loadu_pd(pa + 4); pa += lda;
                av0 = _mm256_sub_pd(ymm0,av0);   av1 = _mm256_sub_pd(ymm0,av1); // a = 0 - a;
                _mm256_storeu_pd(aPacked, av0);  _mm256_storeu_pd(aPacked + 4, av1);
                aPacked += MX8;
            }
        }
    }
    else
    {
        if(alpha==1.0)
        {
            //A Transpose case:
            for (gint_t i = 0; i < MX8 ; i++)
            {
                gint_t idx = i * lda;
                for (gint_t p = 0; p < k; p ++)
                {
                    double ar_ = *(pa+idx+p);
                    gint_t sidx = p * MX8;
                    *(aPacked + sidx + i) = ar_;
                }
            }
        }
        else if(alpha==-1.0)
        {
            //A Transpose case:
            for (gint_t i = 0; i < MX8 ; i++)
            {
                gint_t idx = i * lda;
                for (gint_t p = 0; p < k; p ++)
                {
                    double ar_ = *(pa+idx+p);
                    gint_t sidx = p * MX8;
                    *(aPacked + sidx + i) = -ar_;
                }
            }
        }
    }
}

/* A8x4 packing subroutine */
void bli_prepackA_8x4(double* pa, double* aPacked, gint_t k, guint_t lda)
{
    __m256d av00, av10;
    __m256d av01, av11;
    __m256d av02, av12;
    __m256d av03, av13;

    for (gint_t p = 0; p < k; p += 4) {
        av00 = _mm256_loadu_pd(pa);      av10 = _mm256_loadu_pd(pa + 4); pa += lda;
        av01 = _mm256_loadu_pd(pa);      av11 = _mm256_loadu_pd(pa + 4); pa += lda;
        av02 = _mm256_loadu_pd(pa);      av12 = _mm256_loadu_pd(pa + 4); pa += lda;
        av03 = _mm256_loadu_pd(pa);      av13 = _mm256_loadu_pd(pa + 4); pa += lda;

        _mm256_storeu_pd(aPacked, av00);      _mm256_storeu_pd(aPacked + 4, av10);
        _mm256_storeu_pd(aPacked + 8, av01);  _mm256_storeu_pd(aPacked + 12, av11);
        _mm256_storeu_pd(aPacked + 16, av02); _mm256_storeu_pd(aPacked + 20, av12);
        _mm256_storeu_pd(aPacked + 24, av03); _mm256_storeu_pd(aPacked + 28, av13);

        aPacked += 32;
    }
}

/* dgemm real kernel, which handles m multiple of 8.
m multiple of 4 and 1 to be implemented later */
static err_t bli_dgemm_m8(gint_t m, gint_t n, gint_t k, double* a, guint_t lda, double* b, guint_t ldb, double* c, guint_t ldc, bool isTransA, double alpha)
{
    double* aPacked;
    double* aligned = NULL;

	bool pack_on = false;
	if((m!=MX8)||(m!=lda)||isTransA)
	{
		pack_on = true;
	}

	if(pack_on==true)
	{
		aligned = (double*)bli_malloc_user(sizeof(double) * k * MX8);
	}

    for (gint_t i = 0; i < m; i += MX8) //this loop can be threaded. no of workitems = m/8
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
			bli_prepackA_8(pa, aPacked, k, lda, isTransA, alpha);
			//bli_prepackA_8x4(a + i, aPacked, k, lda);
		}
		else
        {
			aPacked = a+i;
		}

        j = bli_kernel_8mx5n(n, k, j, aPacked, lda, b, ldb, ci, ldc);
        if (j <= n - 4)
        {
            j = bli_kernel_8mx4n(n, k, j, aPacked, lda, b + (j * ldb), ldb, ci + (j * ldc), ldc);
        }
        if (j <= n - 3)
        {
            j = bli_kernel_8mx3n(n, k, j, aPacked, lda, b + (j * ldb), ldb, ci + (j * ldc), ldc);
        }
        if (j <= n - 2)
        {
            j = bli_kernel_8mx2n(n, k, j, aPacked, lda, b + (j * ldb), ldb, ci + (j * ldc), ldc);
        }
        if (j <= n - 1)
        {
            j = bli_kernel_8mx1n(n, k, j, aPacked, lda, b + (j * ldb), ldb, ci + (j * ldc), ldc);
        }
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
    memSize += 128;

    mem_req->alignedBuf = bli_malloc_user( memSize );
    if (mem_req->alignedBuf == NULL)
    {
        return -1;
    }
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
        bli_free_user(mxr->alignedBuf);
        bli_free_user(mxi->alignedBuf);
        bli_free_user(msx->alignedBuf);
        return -1;
    }
    return 0;
}

void bli_add_m(gint_t m,gint_t n,double* w,double* c)
{
    double* pc = c;
    double* pw = w;
    for (gint_t i = 0; i < m*n; i++)
    {
        *pc = *pc + *pw;
        pc++; pw++;
    }
}

void bli_sub_m(gint_t m, gint_t n, double* w, double* c)
{
    double* pc = c;
    double* pw = w;
    for (gint_t i = 0; i < m * n; i++)
    {
        *pc = *pc - *pw;
        pc++; pw++;
    }
}

/****************************************************************/
/* mmm_sqp implementation, which calls dgemm_sqp as real kernel */
/****************************************************************/


static err_t bli_zgemm_sqp_m8(gint_t m, gint_t n, gint_t k, double* a, guint_t lda, double* b, guint_t ldb, double* c, guint_t ldc, double alpha, double beta, bool isTransA)
{
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

//debug to be removed.
#if DEBUG_3M_SQP
    double ax[8][16] = { {10,-10,20,-20,30,-30,40,-40,50,-50,60,-60,70,-70,80,-80},
                     {1.1,-1.1,2.1,-2.1,3.1,-3.1,4.1,-4.1,5.1,-5.1,6.1,-6.1,7.1,-7.1,8.1,-8.1},
                     {1.2,-1.2,2.2,-2.2,3.2,-3.2,4.2,-4.2,5.2,-5.2,6.2,-6.2,7.2,-7.2,8.2,-8.2},
                     {1.3,-1.3,2.3,-2.3,3.3,-3.3,4.3,-4.3,5.3,-5.3,6.3,-6.3,7.3,-7.3,8.3,-8.3},

                     {1,-1,2,-2,3,-3,4,-4,5,-5,6,-6,7,-7,8,-8},
                     {1,-1,2,-2,3,-3,4,-4,5,-5,6,-6,7,-7,8,-8},
                     {1,-1,2,-2,3,-3,4,-4,5,-5,6,-6,7,-7,8,-8},
                     {1,-1,2,-2,3,-3,4,-4,5,-5,6,-6,7,-7,8,-8} };


    double bx[6][16] = { {10,-10,20,-20,30,-30,40,-40,50,-50,60,-60,70,-70,80,-80},
                         {1.1,-1.1,2.1,-2.1,3.1,-3.1,4.1,-4.1,5.1,-5.1,6.1,-6.1,7.1,-7.1,8.1,-8.1},
                         {1.2,-1.2,2.2,-2.2,3.2,-3.2,4.2,-4.2,5.2,-5.2,6.2,-6.2,7.2,-7.2,8.2,-8.2},

                         {1,-1,2,-2,3,-3,4,-4,5,-5,6,-6,7,-7,8,-8},
                         {1,-1,2,-2,3,-3,4,-4,5,-5,6,-6,7,-7,8,-8},
                         {1,-1,2,-2,3,-3,4,-4,5,-5,6,-6,7,-7,8,-8} };

    double cx[8][12] = { {10,-10,20,-20,30,-30,40,-40,50,-50,60,-60},
                 {1.1,-1.1,2.1,-2.1,3.1,-3.1,4.1,-4.1,5.1,-5.1,6.1,-6.1},
                 {1.2,-1.2,2.2,-2.2,3.2,-3.2,4.2,-4.2,5.2,-5.2,6.2,-6.2},
                 {1.3,-1.3,2.3,-2.3,3.3,-3.3,4.3,-4.3,5.3,-5.3,6.3,-6.3},

                 {1,-1,2,-2,3,-3,4,-4,5,-5,6,-6},
                 {1,-1,2,-2,3,-3,4,-4,5,-5,6,-6},
                 {1,-1,2,-2,3,-3,4,-4,5,-5,6,-6},
                 {1,-1,2,-2,3,-3,4,-4,5,-5,6,-6} };

    b = &bx[0][0];
    a = &ax[0][0];
    c = &cx[0][0];
#endif

	/* Split    b  (br, bi) and
       compute  bs = br + bi    */
    double* pbr = br;
    double* pbi = bi;
    double* pbs = bs;

    gint_t j, p;

    /* b matrix real and imag packing and compute to be vectorized. */
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

    /* Workspace memory allocation currently done dynamically
    This needs to be taken from already allocated memory pool in application for better performance */
    /* A matrix */
    double* ar, * ai, * as;
    mem_block mar, mai, mas;
    if(bli_allocateWorkspace(8, k, &mar, &mai, &mas) !=0)
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
    mw.size = 8 * n;
    if (bli_getaligned(&mw) != 0)
    {
        return BLIS_FAILURE;
    }
    w = (double*)mw.alignedBuf;

    /* cr matrix */
    double* cr;
    mem_block mcr;
    mcr.data_size = sizeof(double);
    mcr.size = 8 * n;
    if (bli_getaligned(&mcr) != 0)
    {
        return BLIS_FAILURE;
    }
    cr = (double*)mcr.alignedBuf;


    /* ci matrix */
    double* ci;
    mem_block mci;
    mci.data_size = sizeof(double);
    mci.size = 8 * n;
    if (bli_getaligned(&mci) != 0)
    {
        return BLIS_FAILURE;
    }
    ci = (double*)mci.alignedBuf;

    for (inc_t i = 0; i < (2*m); i += (2*MX8)) //this loop can be threaded.
    {
        ////////////// operation 1 /////////////////

        /* Split    a  (ar, ai) and
           compute  as = ar + ai   */
        double* par = ar;
        double* pai = ai;
        double* pas = as;

        /* a matrix real and imag packing and compute to be vectorized. */
		if(isTransA==false)
		{
			//A No transpose case:
			for (gint_t p = 0; p < k; p += 1) {
				for (gint_t ii = 0; ii < (2*MX8) ; ii += 2) { //real + imag : Rkernel needs 8 elements each.
					double ar_ = a[(p * lda) + i + ii];
					double ai_ = a[(p * lda) + i + ii+1];
					*par = ar_;
					*pai = ai_;
					*pas = ar_ + ai_;
					par++; pai++; pas++;
				}
			}
		}
		else
		{
            //A Transpose case:
			for (gint_t ii = 0; ii < MX8 ; ii++)
			{
				gint_t idx = ((i/2) + ii) * lda;
				for (gint_t s = 0; s < (k*2); s += 2)
				{
					double ar_ = a[ idx + s];
					double ai_ = a[ idx + s + 1];
					gint_t sidx = s * (MX8/2);
					*(par + sidx + ii) = ar_;
					*(pai + sidx + ii) = ai_;
					*(pas + sidx + ii) = ar_ + ai_;
				}
			}
		}

        double* pcr = cr;
        double* pci = ci;

        //Split Cr and Ci and beta multiplication done.
		if((beta ==1.0)||(beta==-1.0))
		{
			if(beta ==1.0)
			{
				for (j = 0; j < n; j++)
                {
					for (gint_t ii = 0; ii < (2*MX8); ii += 2)
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
					for (gint_t ii = 0; ii < (2*MX8); ii += 2)
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
				for (gint_t ii = 0; ii < (2*MX8); ii += 2)
                {
					double cr_ = beta*c[(j * ldc) + i + ii];
					double ci_ = beta*c[(j * ldc) + i + ii + 1];
					*pcr = cr_;
					*pci = ci_;
					pcr++; pci++;
				}
			}
		}

        //Ci := rgemm( SA, SB, Ci )
        bli_dgemm_m8(MX8, n, k, as, MX8, bs, k, ci, MX8, false, 1.0);



        ////////////// operation 2 /////////////////
        //Wr: = dgemm_sqp(Ar, Br, 0)  // Wr output 8xn
        double* wr = w;
        for (j = 0; j < n; j++) {
            for (gint_t ii = 0; ii < MX8; ii += 1) {
                *wr = 0;
                wr++;
            }
        }
        wr = w;

        bli_dgemm_m8(MX8, n, k, ar, MX8, br, k, wr, MX8, false, 1.0);
        //Cr : = addm(Wr, Cr)
        bli_add_m(MX8, n, wr, cr);
        //Ci : = subm(Wr, Ci)
        bli_sub_m(MX8, n, wr, ci);




        ////////////// operation 3 /////////////////
        //Wi : = dgemm_sqp(Ai, Bi, 0)  // Wi output 8xn
        double* wi = w;
        for (j = 0; j < n; j++) {
            for (gint_t ii = 0; ii < MX8; ii += 1) {
                *wi = 0;
                wi++;
            }
        }
        wi = w;

        bli_dgemm_m8(MX8, n, k, ai, MX8, bi, k, wi, MX8, false, 1.0);
        //Cr : = subm(Wi, Cr)
        bli_sub_m(MX8, n, wi, cr);
        //Ci : = subm(Wi, Ci)
        bli_sub_m(MX8, n, wi, ci);


        pcr = cr;
        pci = ci;

        for (j = 0; j < n; j++)
        {
            for (gint_t ii = 0; ii < (2*MX8); ii += 2)
            {
                c[(j * ldc) + i + ii]     = *pcr;
                c[(j * ldc) + i + ii + 1] = *pci;
                pcr++; pci++;
            }
        }

    }

//debug to be removed.
#if DEBUG_3M_SQP
    for (gint_t jj = 0; jj < n;jj++)
    {
        for (gint_t ii = 0; ii < m;ii++)
        {
            printf("( %4.2lf %4.2lf) ", *cr, *ci);
            cr++;ci++;
        }
        printf("\n");
    }
#endif

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

    return BLIS_SUCCESS;
}