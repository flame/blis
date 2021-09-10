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
#include "bli_gemm_sqp_kernels.h"

#define BLIS_LOADFIRST 0
#define BLIS_ENABLE_PREFETCH 1

#define BLIS_MX8 8
#define BLIS_MX4 4
#define BLIS_MX1 1

/****************************************************************************/
/*************** dgemm kernels (8mxn) column preffered  *********************/
/****************************************************************************/

/*  Main dgemm kernel 8mx6n with single load and store of C matrix block
    alpha = +/-1 and beta = +/-1,0 handled while packing.*/
inc_t bli_sqp_dgemm_kernel_8mx6n(gint_t n,
                                gint_t k,
                                gint_t j,
                                double* aPacked,
                                guint_t lda,
                                double* b,
                                guint_t ldb,
                                double* c,
                                guint_t ldc)
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
inc_t bli_sqp_dgemm_kernel_8mx5n(   gint_t n,
                                    gint_t k,
                                    gint_t j,
                                    double* aPacked,
                                    guint_t lda,
                                    double* b,
                                    guint_t ldb,
                                    double* c,
                                    guint_t ldc)
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
inc_t bli_sqp_dgemm_kernel_8mx4n(   gint_t n,
                                    gint_t k,
                                    gint_t j,
                                    double* aPacked,
                                    guint_t lda,
                                    double* b,
                                    guint_t ldb,
                                    double* c,
                                    guint_t ldc)
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
inc_t bli_sqp_dgemm_kernel_8mx3n(   gint_t n,
                                    gint_t k,
                                    gint_t j,
                                    double* aPacked,
                                    guint_t lda,
                                    double* b,
                                    guint_t ldb,
                                    double* c,
                                    guint_t ldc)
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
inc_t bli_sqp_dgemm_kernel_8mx2n(   gint_t n,
                                    gint_t k,
                                    gint_t j,
                                    double* aPacked,
                                    guint_t lda,
                                    double* b,
                                    guint_t ldb,
                                    double* c,
                                    guint_t ldc)
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
inc_t bli_sqp_dgemm_kernel_8mx1n(   gint_t n,
                                    gint_t k,
                                    gint_t j,
                                    double* aPacked,
                                    guint_t lda,
                                    double* b,
                                    guint_t ldb,
                                    double* c,
                                    guint_t ldc)
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
inc_t bli_sqp_dgemm_kernel_4mx10n(  gint_t n,
                                    gint_t k,
                                    gint_t j,
                                    double* aPacked,
                                    guint_t lda,
                                    double* b,
                                    guint_t ldb,
                                    double* c,
                                    guint_t ldc)
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
inc_t bli_sqp_dgemm_kernel_4mx1n(   gint_t n,
                                    gint_t k,
                                    gint_t j,
                                    double* aPacked,
                                    guint_t lda,
                                    double* b,
                                    guint_t ldb,
                                    double* c,
                                    guint_t ldc)
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
inc_t bli_sqp_dgemm_kernel_1mx1n(   gint_t n,
                                    gint_t k,
                                    gint_t j,
                                    double* aPacked,
                                    guint_t lda,
                                    double* b,
                                    guint_t ldb,
                                    double* c,
                                    guint_t ldc)
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

inc_t bli_sqp_dgemm_kernel_mxn( gint_t n,
                                gint_t k,
                                gint_t j,
                                double* aPacked,
                                guint_t lda,
                                double* b,
                                guint_t ldb,
                                double* c,
                                guint_t ldc,
                                gint_t mx)
{
    gint_t p;
    double cx[7];

    double* pb, * pc;

    pb = b;
    pc = c;

    for (; j <= (n - 1); j += 1) {
        //cv0 = _mm256_loadu_pd(pc);
        for (int i = 0; i < mx; i++)
        {
            cx[i] = *(pc + i);
        }

        double* x = aPacked;
        double* pb0 = pb;
        for (p = 0; p < k; p += 1) {
            //bv0 = _mm256_broadcast_sd(pb0);
            double b0 = *pb0;
            pb0++;
            for (int i = 0; i < mx; i++)
            {
                cx[i] += (*(x + i)) * b0;//cv0 = _mm256_fmadd_pd(av0, bv0, cv0);
            }
            //av0 = _mm256_loadu_pd(x);
            x += mx;
        }
        //_mm256_storeu_pd(pc, cv0);
        for (int i = 0; i < mx; i++)
        {
            *(pc + i) = cx[i];
        }
        pc += ldc;pb += ldb;
    }// j loop 1 multiple
    //printf(" mx1:j:%d ", j);
    return j;
}

void bli_sqp_prepackA(  double* pa,
                        double* aPacked,
                        gint_t k,
                        guint_t lda,
                        bool isTransA,
                        double alpha,
                        gint_t mx)
{
    //printf(" pmx:%d ",mx);
    if(mx==8)
    {
        bli_prepackA_8(pa,aPacked,k, lda,isTransA, alpha);
    }
    else if(mx==4)
    {
        bli_prepackA_4(pa,aPacked,k, lda,isTransA, alpha);
    }
    else if(mx>4)
    {
        bli_prepackA_G4(pa,aPacked,k, lda,isTransA, alpha, mx);
    }
    else
    {
        bli_prepackA_L4(pa,aPacked,k, lda,isTransA, alpha, mx);
    }
}

/* Ax8 packing subroutine */
void bli_prepackA_8(double* pa,
                    double* aPacked,
                    gint_t k,
                    guint_t lda,
                    bool isTransA,
                    double alpha)
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
    else //subroutine below to be optimized
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
void bli_prepackA_4(double* pa,
                    double* aPacked,
                    gint_t k,
                    guint_t lda,
                    bool isTransA,
                    double alpha)
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
    else //subroutine below to be optimized
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

/* A packing m>4 subroutine */
void bli_prepackA_G4(   double* pa,
                        double* aPacked,
                        gint_t k,
                        guint_t lda,
                        bool isTransA,
                        double alpha,
                        gint_t mx)
{
    __m256d av0, ymm0;
    gint_t mrem = mx - 4;

    if(isTransA==false)
    {
        if(alpha==1.0)
        {
            for (gint_t p = 0; p < k; p += 1) {
                av0 = _mm256_loadu_pd(pa);
                _mm256_storeu_pd(aPacked, av0);
                for (gint_t i = 0; i < mrem; i += 1) {
                    *(aPacked+4+i) = *(pa+4+i);
                }
                aPacked += mx;pa += lda;
            }
        }
        else if(alpha==-1.0)
        {
            ymm0 = _mm256_setzero_pd();//set zero
            for (gint_t p = 0; p < k; p += 1) {
                av0 = _mm256_loadu_pd(pa);
                av0 = _mm256_sub_pd(ymm0,av0);   // a = 0 - a;
                _mm256_storeu_pd(aPacked, av0);
                for (gint_t i = 0; i < mrem; i += 1) {
                    *(aPacked+4+i) = -*(pa+4+i);
                }
                aPacked += mx;pa += lda;
            }
        }
    }
    else //subroutine below to be optimized
    {
        if(alpha==1.0)
        {
            //A Transpose case:
            for (gint_t i = 0; i < mx ; i++)
            {
                gint_t idx = i * lda;
                for (gint_t p = 0; p < k; p ++)
                {
                    double ar_ = *(pa+idx+p);
                    gint_t sidx = p * mx;
                    *(aPacked + sidx + i) = ar_;
                }
            }
        }
        else if(alpha==-1.0)
        {
            //A Transpose case:
            for (gint_t i = 0; i < mx ; i++)
            {
                gint_t idx = i * lda;
                for (gint_t p = 0; p < k; p ++)
                {
                    double ar_ = *(pa+idx+p);
                    gint_t sidx = p * mx;
                    *(aPacked + sidx + i) = -ar_;
                }
            }
        }
    }

}

/* A packing m<4 subroutine */
void bli_prepackA_L4(   double* pa,
                        double* aPacked,
                        gint_t k,
                        guint_t lda,
                        bool isTransA,
                        double alpha,
                        gint_t mx)
{
    if(isTransA==false)
    {
        if(alpha==1.0)
        {
            for (gint_t p = 0; p < k; p += 1)
            {
                for (gint_t i = 0; i < mx; i += 1)
                {
                    *(aPacked+i) = *(pa+i);
                }
                aPacked += mx;pa += lda;
            }
        }
        else if(alpha==-1.0)
        {
            for (gint_t p = 0; p < k; p += 1)
            {
                for (gint_t i = 0; i < mx; i += 1)
                {
                    *(aPacked+i) = -*(pa+i);
                }
                aPacked += mx;pa += lda;
            }
        }
    }
    else
    {
        if(alpha==1.0)
        {
            //A Transpose case:
            for (gint_t i = 0; i < mx ; i++)
            {
                gint_t idx = i * lda;
                for (gint_t p = 0; p < k; p ++)
                {
                    double ar_ = *(pa+idx+p);
                    gint_t sidx = p * mx;
                    *(aPacked + sidx + i) = ar_;
                }
            }
        }
        else if(alpha==-1.0)
        {
            //A Transpose case:
            for (gint_t i = 0; i < mx ; i++)
            {
                gint_t idx = i * lda;
                for (gint_t p = 0; p < k; p ++)
                {
                    double ar_ = *(pa+idx+p);
                    gint_t sidx = p * mx;
                    *(aPacked + sidx + i) = -ar_;
                }
            }
        }
    }


}

/* Ax1 packing subroutine */
void bli_prepackA_1(double* pa,
                    double* aPacked,
                    gint_t k,
                    guint_t lda,
                    bool isTransA,
                    double alpha)
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


void bli_add_m( gint_t m,
                gint_t n,
                double* w,
                double* c)
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

void bli_sub_m( gint_t m,
                gint_t n,
                double* w,
                double* c)
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
void bli_3m_sqp_packC_real_imag(double* pc,
                                guint_t n,
                                guint_t m,
                                guint_t ldc,
                                double* pcr,
                                double* pci,
                                double mul,
                                gint_t mx)
{
    gint_t j, p;
    __m256d av0, av1, zerov;
    __m256d tv0, tv1;
    gint_t max_m = (m*2)-8;

    if((mul ==1.0)||(mul==-1.0))
    {
        if(mul ==1.0) /* handles alpha or beta = 1.0 */
        {
            for (j = 0; j < n; j++)
            {
                for (p = 0; p <= max_m; p += 8)
                {
                    double* pbp = pc + p;
                    av0 = _mm256_loadu_pd(pbp);   //ai1, ar1, ai0, ar0
                    av1 = _mm256_loadu_pd(pbp+4); //ai3, ar3, ai2, ar2

                    tv0 = _mm256_permute2f128_pd(av0, av1, 0x20);//ai2, ar2, ai0, ar0
                    tv1 = _mm256_permute2f128_pd(av0, av1, 0x31);//ai3, ar3, ai1, ar1
                    av0 = _mm256_unpacklo_pd(tv0, tv1);//ar3, ar2, ar1, ar0
                    av1 = _mm256_unpackhi_pd(tv0, tv1);//ai3, ai2, ai1, ai0

                    _mm256_storeu_pd(pcr, av0); pcr += 4;
                    _mm256_storeu_pd(pci, av1); pci += 4;
                }

                for (; p < (m*2); p += 2)// (real + imag)*m
                {
                    double br = *(pc + p) ;
                    double bi = *(pc + p + 1);
                    *pcr = br;
                    *pci = bi;
                    pcr++; pci++;
                }
                pc = pc + ldc;
            }
        }
        else /* handles alpha or beta = - 1.0 */
        {
            zerov = _mm256_setzero_pd();
            for (j = 0; j < n; j++)
            {
                for (p = 0; p <= max_m; p += 8)
                {
                    double* pbp = pc + p;
                    av0 = _mm256_loadu_pd(pbp); //ai1, ar1, ai0, ar0
                    av1 = _mm256_loadu_pd(pbp+4);//ai3, ar3, ai2, ar2

                    tv0 = _mm256_permute2f128_pd(av0, av1, 0x20);//ai2, ar2, ai0, ar0
                    tv1 = _mm256_permute2f128_pd(av0, av1, 0x31);//ai3, ar3, ai1, ar1
                    av0 = _mm256_unpacklo_pd(tv0, tv1);//ar3, ar2, ar1, ar0
                    av1 = _mm256_unpackhi_pd(tv0, tv1);//ai3, ai2, ai1, ai0

                    //negate
                    av0 = _mm256_sub_pd(zerov,av0);
                    av1 = _mm256_sub_pd(zerov,av1);

                    _mm256_storeu_pd(pcr, av0); pcr += 4;
                    _mm256_storeu_pd(pci, av1); pci += 4;
                }

                for (; p < (m*2); p += 2)// (real + imag)*m
                {
                    double br = -*(pc + p) ;
                    double bi = -*(pc + p + 1);
                    *pcr = br;
                    *pci = bi;
                    pcr++; pci++;
                }
                pc = pc + ldc;
            }
        }
    }
    else if(mul==0) /* handles alpha or beta is equal to zero */
    {
        double br_ = 0;
        double bi_ = 0;
        for (j = 0; j < n; j++)
        {
            for (p = 0; p < (m*2); p += 2)// (real + imag)*m
            {
                *pcr = br_;
                *pci = bi_;
                pcr++; pci++;
            }
            pc = pc + ldc;
        }
    }
    else /* handles alpha or beta is not equal +/- 1.0 and zero */
    {
        for (j = 0; j < n; j++)
        {
            for (p = 0; p < (m*2); p += 2)// (real + imag)*m
            {
                double br_ = mul * (*(pc + p));
                double bi_ = mul * (*(pc + p + 1));
                *pcr = br_;
                *pci = bi_;
                pcr++; pci++;
            }
            pc = pc + ldc;
        }
    }
}

/* Pack real and imaginary parts in separate buffers and compute sum of real and imaginary part */
void bli_3m_sqp_packB_real_imag_sum(double* pb,
                                    guint_t n,
                                    guint_t k,
                                    guint_t ldb,
                                    double* pbr,
                                    double* pbi,
                                    double* pbs,
                                    double mul,
                                    gint_t mx)
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
void bli_3m_sqp_packA_real_imag_sum(double *pa,
                                    gint_t i,
                                    guint_t k,
                                    guint_t lda,
                                    double *par,
                                    double *pai,
                                    double *pas,
                                    trans_t transa,
                                    gint_t mx,
                                    gint_t p)
{
    __m256d av0, av1, av2, av3;
    __m256d tv0, tv1, sum, zerov;
    gint_t poffset = p;
#if KLP
#endif
    if(mx==8)
    {
        if(transa == BLIS_NO_TRANSPOSE)
        {
            pa = pa +i;
#if KLP
            pa = pa + (p*lda);
#else
            p = 0;
#endif
            for (; p < k; p += 1)
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
        else if(transa == BLIS_CONJ_NO_TRANSPOSE)
        {
            zerov = _mm256_setzero_pd();
            pa = pa +i;
#if KLP
            pa = pa + (p*lda);
#else
            p = 0;
#endif
            for (; p < k; p += 1)
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
                av1 = _mm256_sub_pd(zerov,av1);//negate imaginary component
                sum = _mm256_add_pd(av0, av1);
                _mm256_storeu_pd(par, av0); par += 4;
                _mm256_storeu_pd(pai, av1); pai += 4;
                _mm256_storeu_pd(pas, sum); pas += 4;

                tv0 = _mm256_permute2f128_pd(av2, av3, 0x20);
                tv1 = _mm256_permute2f128_pd(av2, av3, 0x31);
                av2 = _mm256_unpacklo_pd(tv0, tv1);
                av3 = _mm256_unpackhi_pd(tv0, tv1);
                av3 = _mm256_sub_pd(zerov,av3);//negate imaginary component
                sum = _mm256_add_pd(av2, av3);
                _mm256_storeu_pd(par, av2); par += 4;
                _mm256_storeu_pd(pai, av3); pai += 4;
                _mm256_storeu_pd(pas, sum); pas += 4;

                pa = pa + lda;
            }
        }
        else if(transa == BLIS_TRANSPOSE)
        {
            gint_t idx = (i/2) * lda;
            pa = pa + idx;
#if KLP
#else
            p = 0;
#endif
            //A Transpose case:
            for (gint_t ii = 0; ii < BLIS_MX8 ; ii++)
            {
                gint_t idx = ii * lda;
                gint_t sidx;
                gint_t pidx = 0;
                gint_t max_k = (k*2) - 8;
                for (p = poffset; p <= max_k; p += 8)
                {
                    double ar0_ = *(pa + idx + p);
                    double ai0_ = *(pa + idx + p + 1);

                    double ar1_ = *(pa + idx + p + 2);
                    double ai1_ = *(pa + idx + p + 3);

                    double ar2_ = *(pa + idx + p + 4);
                    double ai2_ = *(pa + idx + p + 5);

                    double ar3_ = *(pa + idx + p + 6);
                    double ai3_ = *(pa + idx + p + 7);

                    sidx = (pidx/2) * BLIS_MX8;
                    *(par + sidx + ii) = ar0_;
                    *(pai + sidx + ii) = ai0_;
                    *(pas + sidx + ii) = ar0_ + ai0_;

                    sidx = ((pidx+2)/2) * BLIS_MX8;
                    *(par + sidx + ii) = ar1_;
                    *(pai + sidx + ii) = ai1_;
                    *(pas + sidx + ii) = ar1_ + ai1_;

                    sidx = ((pidx+4)/2) * BLIS_MX8;
                    *(par + sidx + ii) = ar2_;
                    *(pai + sidx + ii) = ai2_;
                    *(pas + sidx + ii) = ar2_ + ai2_;

                    sidx = ((pidx+6)/2) * BLIS_MX8;
                    *(par + sidx + ii) = ar3_;
                    *(pai + sidx + ii) = ai3_;
                    *(pas + sidx + ii) = ar3_ + ai3_;
                    pidx += 8;

                }

                for (; p < (k*2); p += 2)
                {
                    double ar_ = *(pa + idx + p);
                    double ai_ = *(pa + idx + p + 1);
                    gint_t sidx = (pidx/2) * BLIS_MX8;
                    *(par + sidx + ii) = ar_;
                    *(pai + sidx + ii) = ai_;
                    *(pas + sidx + ii) = ar_ + ai_;
                    pidx += 2;
                }
            }
        }
        else if(transa == BLIS_CONJ_TRANSPOSE)
        {
            gint_t idx = (i/2) * lda;
            pa = pa + idx;
#if KLP
#else
            p = 0;
#endif
            //A conjugate Transpose case:
            for (gint_t ii = 0; ii < BLIS_MX8 ; ii++)
            {
                gint_t idx = ii * lda;
                gint_t sidx;
                gint_t pidx = 0;
                gint_t max_k = (k*2) - 8;
                for (p = poffset; p <= max_k; p += 8)
                {
                    double ar0_ = *(pa + idx + p);
                    double ai0_ = -(*(pa + idx + p + 1));

                    double ar1_ = *(pa + idx + p + 2);
                    double ai1_ = -(*(pa + idx + p + 3));

                    double ar2_ = *(pa + idx + p + 4);
                    double ai2_ = -(*(pa + idx + p + 5));

                    double ar3_ = *(pa + idx + p + 6);
                    double ai3_ = -(*(pa + idx + p + 7));

                    sidx = (pidx/2) * BLIS_MX8;
                    *(par + sidx + ii) = ar0_;
                    *(pai + sidx + ii) = ai0_;
                    *(pas + sidx + ii) = ar0_ + ai0_;

                    sidx = ((pidx+2)/2) * BLIS_MX8;
                    *(par + sidx + ii) = ar1_;
                    *(pai + sidx + ii) = ai1_;
                    *(pas + sidx + ii) = ar1_ + ai1_;

                    sidx = ((pidx+4)/2) * BLIS_MX8;
                    *(par + sidx + ii) = ar2_;
                    *(pai + sidx + ii) = ai2_;
                    *(pas + sidx + ii) = ar2_ + ai2_;

                    sidx = ((pidx+6)/2) * BLIS_MX8;
                    *(par + sidx + ii) = ar3_;
                    *(pai + sidx + ii) = ai3_;
                    *(pas + sidx + ii) = ar3_ + ai3_;
                    pidx += 8;
                }

                for (; p < (k*2); p += 2)
                {
                    double ar_ = *(pa + idx + p);
                    double ai_ = -(*(pa + idx + p + 1));
                    gint_t sidx = (pidx/2) * BLIS_MX8;
                    *(par + sidx + ii) = ar_;
                    *(pai + sidx + ii) = ai_;
                    *(pas + sidx + ii) = ar_ + ai_;
                    pidx += 2;
                }
            }
        }
    }   //mx==8
    else//mx==1
    {
        if(transa == BLIS_NO_TRANSPOSE)
        {
            pa = pa + i;
#if KLP
#else
            p = 0;
#endif
            //A No transpose case:
            for (; p < k; p += 1)
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
        else if(transa == BLIS_CONJ_NO_TRANSPOSE)
        {
            pa = pa + i;
#if KLP
#else
            p = 0;
#endif
            //A conjuate No transpose case:
            for (; p < k; p += 1)
            {
                gint_t idx = p * lda;
                for (gint_t ii = 0; ii < (mx*2) ; ii += 2)
                { //real + imag : Rkernel needs 8 elements each.
                    double ar_ = *(pa + idx + ii);
                    double ai_ = -(*(pa + idx + ii + 1));// conjugate: negate imaginary component
                    *par = ar_;
                    *pai = ai_;
                    *pas = ar_ + ai_;
                    par++; pai++; pas++;
                }
            }
        }
        else if(transa == BLIS_TRANSPOSE)
        {
            gint_t idx = (i/2) * lda;
            pa = pa + idx;
#if KLP
#else
            p = 0;
#endif
            //A Transpose case:
            for (gint_t ii = 0; ii < mx ; ii++)
            {
                gint_t idx = ii * lda;
                gint_t sidx;
                gint_t pidx = 0;
                for (p = poffset;p < (k*2); p += 2)
                {
                    double ar0_ = *(pa + idx + p);
                    double ai0_ = *(pa + idx + p + 1);

                    sidx = (pidx/2) * mx;
                    *(par + sidx + ii) = ar0_;
                    *(pai + sidx + ii) = ai0_;
                    *(pas + sidx + ii) = ar0_ + ai0_;
                    pidx += 2;

                }
            }
        }
        else if(transa == BLIS_CONJ_TRANSPOSE)
        {
            gint_t idx = (i/2) * lda;
            pa = pa + idx;
#if KLP
#else
            p = 0;
#endif
            //A Transpose case:
            for (gint_t ii = 0; ii < mx ; ii++)
            {
                gint_t idx = ii * lda;
                gint_t sidx;
                gint_t pidx = 0;
                for (p = poffset;p < (k*2); p += 2)
                {
                    double ar0_ = *(pa + idx + p);
                    double ai0_ = -(*(pa + idx + p + 1));

                    sidx = (pidx/2) * mx;
                    *(par + sidx + ii) = ar0_;
                    *(pai + sidx + ii) = ai0_;
                    *(pas + sidx + ii) = ar0_ + ai0_;
                    pidx += 2;

                }
            }
        }
    }//mx==1
}

