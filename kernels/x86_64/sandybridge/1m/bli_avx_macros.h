#ifndef BLIS_AVX_MACROS_H
#define BLIS_AVX_MACROS_H

#include "blis.h"

#include "immintrin.h"

static void load_avx_d4x4(const double *A, inc_t lda, __m256d rows[4])
{
    rows[0] = _mm256_loadu_pd(A + 0*lda);
    rows[1] = _mm256_loadu_pd(A + 1*lda);
    rows[2] = _mm256_loadu_pd(A + 2*lda);
    rows[3] = _mm256_loadu_pd(A + 3*lda);
}

static void load_avx_d4x2(const double *A, inc_t lda, __m256d rows[2])
{
    rows[0] = _mm256_loadu_pd(A + 0*lda);
    rows[1] = _mm256_loadu_pd(A + 1*lda);
}

static void load_avx_d2x4(const double *A, inc_t lda, __m128d rows[4])
{
    rows[0] = _mm_loadu_pd(A + 0*lda);
    rows[1] = _mm_loadu_pd(A + 1*lda);
    rows[2] = _mm_loadu_pd(A + 2*lda);
    rows[3] = _mm_loadu_pd(A + 3*lda);
}

static void store_avx_d4x4(double *B, inc_t ldb, __m256d rows[4])
{
    _mm256_storeu_pd(B + 0*ldb, rows[0]);
    _mm256_storeu_pd(B + 1*ldb, rows[1]);
    _mm256_storeu_pd(B + 2*ldb, rows[2]);
    _mm256_storeu_pd(B + 3*ldb, rows[3]);
}

static void store_avx_d2x4(double *B, inc_t ldb, __m128d rows[4])
{
    _mm_storeu_pd(B + 0*ldb, rows[0]);
    _mm_storeu_pd(B + 1*ldb, rows[1]);
    _mm_storeu_pd(B + 2*ldb, rows[2]);
    _mm_storeu_pd(B + 3*ldb, rows[3]);
}

static void trans_avx_d4x4(__m256d rows[4])
{
    __m256d t1, t2, t3, t4;
    // rows[0] = (a00,a01,a02,a03)
    // rows[1] = (a10,a11,a12,a13)
    // rows[2] = (a20,a21,a22,a23)
    // rows[3] = (a30,a31,a32,a33)
    t1 = _mm256_shuffle_pd(rows[0], rows[1], 0x0);
    t2 = _mm256_shuffle_pd(rows[0], rows[1], 0xf);
    t3 = _mm256_shuffle_pd(rows[2], rows[3], 0x0);
    t4 = _mm256_shuffle_pd(rows[2], rows[3], 0xf);
    // t1 = (a00,a10,a02,a12)
    // t2 = (a01,a11,a03,a13)
    // t3 = (a20,a30,a22,a32)
    // t4 = (a21,a31,a23,a33)
    rows[0] = _mm256_permute2f128_pd(t1, t3, 0x20);
    rows[1] = _mm256_permute2f128_pd(t2, t4, 0x20);
    rows[2] = _mm256_permute2f128_pd(t1, t3, 0x31);
    rows[3] = _mm256_permute2f128_pd(t2, t4, 0x31);
    // rows[0] = (a00,a10,a20,a30)
    // rows[1] = (a01,a11,a21,a31)
    // rows[2] = (a02,a12,a22,a32)
    // rows[3] = (a03,a13,a23,a33)
}

static void trans_avx_d4x2(__m256d rows1[2], __m128d rows2[4])
{
    __m256d t1, t2;
    // rows1[0] = (a00,a01,a02,a03)
    // rows1[1] = (a10,a11,a12,a13)
    t1 = _mm256_shuffle_pd(rows1[0], rows1[1], 0x0);
    t2 = _mm256_shuffle_pd(rows1[0], rows1[1], 0xf);
    // t1 = (a00,a10,a02,a12)
    // t2 = (a01,a11,a03,a13)
    rows2[0] = _mm256_extractf128_pd(t1, 0x0);
    rows2[1] = _mm256_extractf128_pd(t2, 0x0);
    rows2[2] = _mm256_extractf128_pd(t1, 0x1);
    rows2[3] = _mm256_extractf128_pd(t2, 0x1);
    // rows2[0] = (a00,a10)
    // rows2[1] = (a01,a11)
    // rows2[2] = (a02,a12)
    // rows2[3] = (a03,a13)
}

static void copy_avx_d4x1(const double *A, double *B)
{
    __m256d row = _mm256_loadu_pd(A);
    _mm256_storeu_pd(B, row);
}

static void copy_scale_avx_d4x1(double alpha, const double *A, double *B)
{
    __m256d alpha_bcast = _mm256_set1_pd(alpha);
    __m256d row = _mm256_loadu_pd(A);
    row = _mm256_mul_pd(row, alpha_bcast);
    _mm256_storeu_pd(B, row);
}

static void copy_avx_d2x1(const double *A, double *B)
{
    __m128d row = _mm_loadu_pd(A);
    _mm_storeu_pd(B, row);
}

static void copy_scale_avx_d2x1(double alpha, const double *A, double *B)
{
    __m128d alpha_bcast = _mm_set1_pd(alpha);
    __m128d row = _mm_loadu_pd(A);
    row = _mm_mul_pd(row, alpha_bcast);
    _mm_storeu_pd(B, row);
}

static void copy_avx_d4x4(const double *A, inc_t lda, double *B, inc_t ldb)
{
    __m256d rows[4];
    load_avx_d4x4(A, lda, rows);
    store_avx_d4x4(B, ldb, rows);
}

static void copy_scale_avx_d4x4(double alpha, const double *A, inc_t lda, double *B, inc_t ldb)
{
    __m256d alpha_bcast = _mm256_set1_pd(alpha);
    __m256d rows[4];
    load_avx_d4x4(A, lda, rows);
    rows[0] = _mm256_mul_pd(rows[0], alpha_bcast);
    rows[1] = _mm256_mul_pd(rows[1], alpha_bcast);
    rows[2] = _mm256_mul_pd(rows[2], alpha_bcast);
    rows[3] = _mm256_mul_pd(rows[3], alpha_bcast);
    store_avx_d4x4(B, ldb, rows);
}

static void copy_avx_d2x4(const double *A, inc_t lda, double *B, inc_t ldb)
{
    __m128d rows[4];
    load_avx_d2x4(A, lda, rows);
    store_avx_d2x4(B, ldb, rows);
}

static void copy_scale_avx_d2x4(double alpha, const double *A, inc_t lda, double *B, inc_t ldb)
{
    __m128d alpha_bcast = _mm_set1_pd(alpha);
    __m128d rows[4];
    load_avx_d2x4(A, lda, rows);
    rows[0] = _mm_mul_pd(rows[0], alpha_bcast);
    rows[1] = _mm_mul_pd(rows[1], alpha_bcast);
    rows[2] = _mm_mul_pd(rows[2], alpha_bcast);
    rows[3] = _mm_mul_pd(rows[3], alpha_bcast);
    store_avx_d2x4(B, ldb, rows);
}

static void copy_trans_avx_d4x4(const double *A, inc_t lda, double *B, inc_t ldb)
{
    __m256d rows[4];
    load_avx_d4x4(A, lda, rows);
    trans_avx_d4x4(rows);
    store_avx_d4x4(B, ldb, rows);
}

static void copy_scale_trans_avx_d4x4(double alpha, const double *A, inc_t lda, double *B, inc_t ldb)
{
    __m256d alpha_bcast = _mm256_set1_pd(alpha);
    __m256d rows[4];
    load_avx_d4x4(A, lda, rows);
    rows[0] = _mm256_mul_pd(rows[0], alpha_bcast);
    rows[1] = _mm256_mul_pd(rows[1], alpha_bcast);
    rows[2] = _mm256_mul_pd(rows[2], alpha_bcast);
    rows[3] = _mm256_mul_pd(rows[3], alpha_bcast);
    trans_avx_d4x4(rows);
    store_avx_d4x4(B, ldb, rows);
}

static void copy_trans_avx_d4x2(const double *A, inc_t lda, double *B, inc_t ldb)
{
    __m256d rows1[2];
    __m128d rows2[4];
    load_avx_d4x2(A, lda, rows1);
    trans_avx_d4x2(rows1, rows2);
    store_avx_d2x4(B, ldb, rows2);
}

static void copy_scale_trans_avx_d4x2(double alpha, const double *A, inc_t lda, double *B, inc_t ldb)
{
    __m256d alpha_bcast = _mm256_set1_pd(alpha);
    __m256d rows1[2];
    __m128d rows2[4];
    load_avx_d4x2(A, lda, rows1);
    rows1[0] = _mm256_mul_pd(rows1[0], alpha_bcast);
    rows1[1] = _mm256_mul_pd(rows1[1], alpha_bcast);
    trans_avx_d4x2(rows1, rows2);
    store_avx_d2x4(B, ldb, rows2);
}

#endif
