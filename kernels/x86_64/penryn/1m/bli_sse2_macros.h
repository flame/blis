#ifndef BLIS_SSE2_MACROS_H
#define BLIS_SSE2_MACROS_H

#include "blis.h"

#include "emmintrin.h"

static void load_sse2_d2x2(const double *A, inc_t lda, __m128d rows[2])
{
    rows[0] = _mm_loadu_pd(A + 0*lda);
    rows[1] = _mm_loadu_pd(A + 1*lda);
}

static void store_sse2_d2x2(double *B, inc_t ldb, __m128d rows[2])
{
    _mm_storeu_pd(B + 0*ldb, rows[0]);
    _mm_storeu_pd(B + 1*ldb, rows[1]);
}

static void trans_sse2_d2x2(__m128d rows[2])
{
    __m128d t1, t2;
    // rows[0] = (a00,a01)
    // rows[1] = (a10,a11)
    t1 = _mm_shuffle_pd(rows[0], rows[1], 0x0);
    t2 = _mm_shuffle_pd(rows[0], rows[1], 0x3);
    // t1 = (a00,a10)
    // t2 = (a01,a11)
    rows[0] = t1;
    rows[1] = t2;
}

static void copy_sse2_d2x1(const double *A, double *B)
{
    __m128d row = _mm_loadu_pd(A);
    _mm_storeu_pd(B, row);
}

static void copy_scale_sse2_d2x1(double alpha, const double *A, double *B)
{
    __m128d alpha_bcast = _mm_set1_pd(alpha);
    __m128d row = _mm_loadu_pd(A);
    row = _mm_mul_pd(row, alpha_bcast);
    _mm_storeu_pd(B, row);
}

static void copy_sse2_d2x2(const double *A, inc_t lda, double *B, inc_t ldb)
{
    __m128d rows[2];
    load_sse2_d2x2(A, lda, rows);
    store_sse2_d2x2(B, ldb, rows);
}

static void copy_scale_sse2_d2x2(double alpha, const double *A, inc_t lda, double *B, inc_t ldb)
{
    __m128d alpha_bcast = _mm_set1_pd(alpha);
    __m128d rows[2];
    load_sse2_d2x2(A, lda, rows);
    rows[0] = _mm_mul_pd(rows[0], alpha_bcast);
    rows[1] = _mm_mul_pd(rows[1], alpha_bcast);
    store_sse2_d2x2(B, ldb, rows);
}

static void copy_trans_sse2_d2x2(const double *A, inc_t lda, double *B, inc_t ldb)
{
    __m128d rows[2];
    load_sse2_d2x2(A, lda, rows);
    trans_sse2_d2x2(rows);
    store_sse2_d2x2(B, ldb, rows);
}

static void copy_scale_trans_sse2_d2x2(double alpha, const double *A, inc_t lda, double *B, inc_t ldb)
{
    __m128d alpha_bcast = _mm_set1_pd(alpha);
    __m128d rows[2];
    load_sse2_d2x2(A, lda, rows);
    rows[0] = _mm_mul_pd(rows[0], alpha_bcast);
    rows[1] = _mm_mul_pd(rows[1], alpha_bcast);
    trans_sse2_d2x2(rows);
    store_sse2_d2x2(B, ldb, rows);
}

#endif
