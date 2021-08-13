/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2018-2021, Advanced Micro Devices, Inc. All rights reserved.

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
#ifdef BLIS_ENABLE_SMALL_MATRIX_TRSM
#include "immintrin.h"

#define BLIS_ENABLE_PREFETCH_IN_TRSM_SMALL

/*
   declaration of trsm small kernels function pointer
*/
typedef err_t (*trsmsmall_ker_ft)
(
    obj_t* AlphaObj,
    obj_t* a,
    obj_t* b,
    cntx_t* cntx,
    cntl_t* cntl
);

//AX = B; A is lower triangular; No transpose;
//double precision; non-unit diagonal
//A.'X = B;  A is upper triangular;
//A has to be transposed; double precision

BLIS_INLINE err_t bli_dtrsm_small_AutXB_AlXB
(
    obj_t* AlphaObj,
    obj_t* a,
    obj_t* b,
    cntx_t* cntx,
    cntl_t* cntl
);

/*  TRSM for the case AX = alpha * B, Double precision
 *  A is upper-triangular, non-transpose, non-unit diagonal
 *  dimensions A: mxm X: mxn B: mxn
*/
//AX = B;  A is lower triangular; transpose; double precision

BLIS_INLINE err_t bli_dtrsm_small_AltXB_AuXB
(
    obj_t* AlphaObj,
    obj_t* a,
    obj_t* b,
    cntx_t* cntx,
    cntl_t* cntl
);

//XA = B; A is upper-triangular; A is transposed;
//double precision; non-unit diagonal
// XA = B; A is lower-traingular; No transpose;
//double precision; non-unit diagonal

BLIS_INLINE  err_t bli_dtrsm_small_XAutB_XAlB
(
    obj_t* AlphaObj,
    obj_t* a,
    obj_t* b,
    cntx_t* cntx,
    cntl_t* cntl
);

// XA = B; A is upper triangular; No transpose;
//double presicion; non-unit diagonal
//XA = B; A is lower-triangular; A is transposed;
// double precision; non-unit-diagonal

BLIS_INLINE  err_t bli_dtrsm_small_XAltB_XAuB
(
    obj_t* AlphaObj,
    obj_t* a,
    obj_t* b,
    cntx_t* cntx,
    cntl_t* cntl
);

//AX = B; A is lower triangular; transpose;
//double precision; non-unit diagonal
BLIS_INLINE err_t dtrsm_AltXB_ref
(
    double *A,
    double *B,
    dim_t M,
    dim_t N,
    dim_t lda,
    dim_t ldb,
    bool is_unitdiag
);

/*
 * The preinversion of diagonal elements are enabled/disabled
 * based on configuration.
 */
#ifdef BLIS_ENABLE_TRSM_PREINVERSION
#define DIAG_ELE_INV_OPS(a,b)  (a / b)
#define DIAG_ELE_EVAL_OPS(a,b) (a * b)
#endif

#ifdef BLIS_DISABLE_TRSM_PREINVERSION
#define DIAG_ELE_INV_OPS(a,b)  (a * b)
#define DIAG_ELE_EVAL_OPS(a,b) (a / b)
#endif

/*
 * Reference implementations
 * ToDo: We can combine all these reference implementation
         into a macro
*/
//A'X = B;  A is upper triangular; transpose;
//non-unitDiagonal double precision
BLIS_INLINE err_t dtrsm_AutXB_ref
(
    double *A,
    double *B,
    dim_t M,
    dim_t N,
    dim_t lda,
    dim_t ldb,
    bool unitDiagonal
)
{
    dim_t i, j, k;
    for (k = 0; k < M; k++)
    {
        double lkk_inv = 1.0;
        if(!unitDiagonal) lkk_inv = DIAG_ELE_INV_OPS(lkk_inv,A[k+k*lda]);
        for (j = 0; j < N; j++)
        {
            B[k + j*ldb] = DIAG_ELE_EVAL_OPS(B[k + j*ldb] , lkk_inv);
            for (i = k+1; i < M; i++)
            {
                B[i + j*ldb] -= A[i*lda + k] * B[k + j*ldb];
            }
        }
    }// k -loop
    return BLIS_SUCCESS;
}// end of function

/* TRSM scalar code for the case AX = alpha * B
 * A is upper-triangular, non-unit-diagonal
 * Dimensions:  A: mxm   X: mxn B:mxn
 */
BLIS_INLINE err_t dtrsm_AuXB_ref
(
    double *A,
    double *B,
    dim_t M,
    dim_t N,
    dim_t lda,
    dim_t ldb,
    bool is_unitdiag
)
{
    dim_t i, j, k;
    for (k = M-1; k >= 0; k--)
    {
        double lkk_inv = 1.0;
        if(!is_unitdiag) lkk_inv = DIAG_ELE_INV_OPS(lkk_inv,A[k+k*lda]);
        for (j = N -1; j >= 0; j--)
        {
            B[k + j*ldb] = DIAG_ELE_EVAL_OPS(B[k + j*ldb],lkk_inv);
            for (i = k-1; i >=0; i--)
            {
                B[i + j*ldb] -= A[i + k*lda] * B[k + j*ldb];
            }
        }
    }// k -loop
    return BLIS_SUCCESS;
}// end of function

/* TRSM scalar code for the case AX = alpha * B
 * A is lower-triangular, non-unit-diagonal, no transpose
 * Dimensions:  A: mxm   X: mxn B:mxn
 */
BLIS_INLINE err_t dtrsm_AlXB_ref
(
    double *A,
    double *B,
    dim_t M,
    dim_t N,
    dim_t lda,
    dim_t ldb,
    bool is_unitdiag
)
{
    dim_t i, j, k;
    for (k = 0; k < M; k++)
    {
        double lkk_inv = 1.0;
        if(!is_unitdiag) lkk_inv = DIAG_ELE_INV_OPS(lkk_inv,A[k+k*lda]);
        for (j = 0; j < N; j++)
        {
            B[k + j*ldb] = DIAG_ELE_EVAL_OPS(B[k + j*ldb],lkk_inv);
            for (i = k+1; i < M; i++)
            {
                B[i + j*ldb] -= A[i + k*lda] * B[k + j*ldb];
            }
        }
    }// k -loop
    return BLIS_SUCCESS;
}// end of function

/* TRSM scalar code for the case AX = alpha * B
 * A is lower-triangular, non-unit-diagonal, transpose
 * Dimensions:  A: mxm   X: mxn B:mxn
 */
BLIS_INLINE err_t dtrsm_AltXB_ref
(
    double *A,
    double *B,
    dim_t M,
    dim_t N,
    dim_t lda,
    dim_t ldb,
    bool is_unitdiag
)
{
    dim_t i, j, k;
    for (k = M-1; k >= 0; k--)
    {
        double lkk_inv = 1.0;
        if(!is_unitdiag) lkk_inv = DIAG_ELE_INV_OPS(lkk_inv,A[k+k*lda]);
        for (j = N -1; j >= 0; j--)
        {
            B[k + j*ldb] = DIAG_ELE_EVAL_OPS(B[k + j*ldb],lkk_inv);
            for (i = k-1; i >=0; i--)
            {
                B[i + j*ldb] -= A[i*lda + k] * B[k + j*ldb];
            }
        }
    }// k -loop
    return BLIS_SUCCESS;
}// end of function

/* TRSM scalar code for the case XA = alpha * B
 * A is upper-triangular, non-unit/unit diagonal no transpose
 * Dimensions: X:mxn A:nxn B:mxn
 */
BLIS_INLINE err_t dtrsm_XAuB_ref
(
    double *A,
    double *B,
    dim_t M,
    dim_t N,
    dim_t lda,
    dim_t ldb,
    bool is_unitdiag
)
{
    dim_t i, j, k;
    for(k = 0; k < N; k++)
    {
       double lkk_inv = 1.0;
       if(!is_unitdiag) lkk_inv = DIAG_ELE_INV_OPS(lkk_inv,A[k+k*lda]);
       for(i = 0; i < M; i++)
       {
           B[i+k*ldb] = DIAG_ELE_EVAL_OPS(B[i + k*ldb],lkk_inv);
           for(j = k+1; j < N; j++)
           {
               B[i+j*ldb] -= B[i+k*ldb] * A[k+j*lda];
           }
       }

    }
    return BLIS_SUCCESS;
}

/* TRSM scalar code for the case XA = alpha * B
 * A is lower-triangular, non-unit/unit triangular, no transpose
 * Dimensions: X:mxn A:nxn B:mxn
 */
BLIS_INLINE err_t dtrsm_XAlB_ref
(
    double *A,
    double *B,
    dim_t M,
    dim_t N,
    dim_t lda,
    dim_t ldb,
    bool is_unitdiag
)
{
    dim_t i, j, k;

    for(k = N;k--;)
    {
        double lkk_inv = 1.0;
        if(!is_unitdiag) lkk_inv = DIAG_ELE_INV_OPS(lkk_inv,A[k+k*lda]);
        for(i = M;i--;)
        {
            B[i+k*ldb] = DIAG_ELE_EVAL_OPS(B[i + k*ldb],lkk_inv);
            for(j = k;j--;)
            {
                B[i+j*ldb] -= B[i+k*ldb] * A[k+j*lda];
            }
        }
    }
    return BLIS_SUCCESS;
}


/* TRSM scalar code for the case XA = alpha * B
 *A is upper-triangular, non-unit/unit diagonal, A is transposed
 * Dimensions: X:mxn A:nxn B:mxn
 */
BLIS_INLINE err_t dtrsm_XAutB_ref
(
    double *A,
    double *B,
    dim_t M,
    dim_t N,
    dim_t lda,
    dim_t ldb,
    bool is_unitdiag
)
{
    dim_t i, j, k;

    for(k = N; k--;)
    {
        double lkk_inv = 1.0;
        if(!is_unitdiag) lkk_inv = DIAG_ELE_INV_OPS(lkk_inv,A[k+k*lda]);
        for(i = M; i--;)
        {
            B[i+k*ldb] = DIAG_ELE_EVAL_OPS(B[i+k*ldb],lkk_inv);
            for(j = k; j--;)
            {
                B[i+j*ldb] -= B[i+k*ldb] * A[j+k*lda];
            }
        }
    }
    return BLIS_SUCCESS;
}


/* TRSM scalar code for the case XA = alpha * B
 * A is lower-triangular, non-unit/unit diagonal, A has to be transposed
 * Dimensions: X:mxn A:nxn B:mxn
 */
BLIS_INLINE err_t dtrsm_XAltB_ref
(
    double *A,
    double *B,
    dim_t M,
    dim_t N,
    dim_t lda,
    dim_t ldb,
    bool is_unitdiag
)
{
    dim_t i, j, k;
    for(k = 0; k < N; k++)
    {
        double lkk_inv = 1.0;
        if(!is_unitdiag) lkk_inv = DIAG_ELE_INV_OPS(lkk_inv,A[k+k*lda]);
        for(i = 0; i < M; i++)
        {
            B[i+k*ldb] = DIAG_ELE_EVAL_OPS(B[i+k*ldb],lkk_inv);
            for(j = k+1; j < N; j++)
            {
                B[i+j*ldb] -= B[i+k*ldb] * A[j+k*lda];
            }
        }
    }
    return BLIS_SUCCESS;
}

#ifdef BLIS_DISABLE_TRSM_PREINVERSION
#define DTRSM_SMALL_DIV_OR_SCALE _mm256_div_pd
#endif

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
#define DTRSM_SMALL_DIV_OR_SCALE _mm256_mul_pd
#endif

/*Initialize */
#define BLIS_SET_YMM_REG_ZEROS \
      ymm3 = _mm256_setzero_pd(); \
      ymm4 = _mm256_setzero_pd(); \
      ymm5 = _mm256_setzero_pd(); \
      ymm6 = _mm256_setzero_pd(); \
      ymm7 = _mm256_setzero_pd(); \
      ymm8 = _mm256_setzero_pd(); \
      ymm9 = _mm256_setzero_pd(); \
      ymm10 = _mm256_setzero_pd(); \
      ymm11 = _mm256_setzero_pd(); \
      ymm12 = _mm256_setzero_pd(); \
      ymm13 = _mm256_setzero_pd(); \
      ymm14 = _mm256_setzero_pd(); \
      ymm15 = _mm256_setzero_pd();

/*GEMM block used in trsm small right cases*/
#define BLIS_DTRSM_SMALL_GEMM_6nx8m(a01,b10,cs_b,p_lda,k_iter) \
    for(k = 0; k < k_iter; k++) \
    {\
        /*load 8x1 block of B10*/ \
        ymm0 = _mm256_loadu_pd((double const *)b10); \
        ymm1 = _mm256_loadu_pd((double const *)(b10 + 4)); \
        \
        /*broadcast 1st row of A01*/ \
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 0)); \
        ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3); \
        ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4); \
        \
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 1)); \
        ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5); \
        ymm6 = _mm256_fmadd_pd(ymm2, ymm1, ymm6); \
        \
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 2)); \
        ymm7 = _mm256_fmadd_pd(ymm2, ymm0, ymm7); \
        ymm8 = _mm256_fmadd_pd(ymm2, ymm1, ymm8); \
        \
        /*Prefetch the next micro panel*/ \
        _mm_prefetch((char*)( b10 + 8*cs_b), _MM_HINT_T0); \
        \
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 3)); \
        ymm9 = _mm256_fmadd_pd(ymm2, ymm0, ymm9); \
        ymm10 = _mm256_fmadd_pd(ymm2, ymm1, ymm10); \
        \
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 4)); \
        ymm11 = _mm256_fmadd_pd(ymm2, ymm0, ymm11); \
        ymm12 = _mm256_fmadd_pd(ymm2, ymm1, ymm12); \
        \
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 5)); \
        ymm13 = _mm256_fmadd_pd(ymm2, ymm0, ymm13); \
        ymm14 = _mm256_fmadd_pd(ymm2, ymm1, ymm14); \
        a01 += 1;\
        b10 += cs_b; \
    }

#define BLIS_DTRSM_SMALL_GEMM_6nx4m(a01,b10,cs_b,p_lda,k_iter) \
    for(k = 0; k < k_iter; k++)      /*loop for number of GEMM operations*/\
    {\
        /*load 8x1 block of B10*/\
        ymm0 = _mm256_loadu_pd((double const *)b10); /*B10[0][0] B10[1][0] B10[2][0] B10[3][0]*/\
\
        /*broadcast 1st row of A01*/\
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 0));  /*A01[0][0]*/\
        ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 1));  /*A01[0][1]*/\
        ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 2)); /*A01[0][2]*/\
        ymm7 = _mm256_fmadd_pd(ymm2, ymm0, ymm7);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 3)); /*A01[0][3]*/\
        ymm9 = _mm256_fmadd_pd(ymm2, ymm0, ymm9);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 4)); /*A01[0][4]*/\
        ymm11 = _mm256_fmadd_pd(ymm2, ymm0, ymm11);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 5)); /*A01[0][5]*/\
        ymm13 = _mm256_fmadd_pd(ymm2, ymm0, ymm13);\
\
        a01 += 1;  /*move to next row*/\
        b10 += cs_b;\
    }

#define BLIS_DTRSM_SMALL_GEMM_4nx8m(a01,b10,cs_b,p_lda,k_iter) \
    for(k = 0; k < k_iter; k++)      /*loop for number of GEMM operations*/\
    {\
        /*load 8x1 block of B10*/\
        ymm0 = _mm256_loadu_pd((double const *)b10);\
        ymm1 = _mm256_loadu_pd((double const *)(b10 + 4));\
\
        /*broadcast 1st row of A01*/\
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 0));  /*A01[0][0]*/\
        ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3);\
        ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 1));  /*A01[0][1]*/\
        ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5);\
        ymm6 = _mm256_fmadd_pd(ymm2, ymm1, ymm6);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 2)); /*A01[0][2]*/\
        ymm7 = _mm256_fmadd_pd(ymm2, ymm0, ymm7);\
        ymm8 = _mm256_fmadd_pd(ymm2, ymm1, ymm8);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 3)); /*A01[0][3]*/\
        ymm9 = _mm256_fmadd_pd(ymm2, ymm0, ymm9);\
        ymm10 = _mm256_fmadd_pd(ymm2, ymm1, ymm10);\
\
        a01 += 1;  /*move to next row*/\
        b10 += cs_b;\
    }

#define BLIS_DTRSM_SMALL_GEMM_3nx8m(a01,b10,cs_b,p_lda,k_iter)\
    for(k = 0; k < k_iter; k++)      /*loop for number of GEMM operations*/\
    {\
        /*load 8x1 block of B10*/\
        ymm0 = _mm256_loadu_pd((double const *)b10);\
        ymm1 = _mm256_loadu_pd((double const *)(b10 + 4));\
\
        /*broadcast 1st row of A01*/\
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 0));  /*A01[0][0]*/\
        ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3);\
        ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 1));  /*A01[0][1]*/\
        ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5);\
        ymm6 = _mm256_fmadd_pd(ymm2, ymm1, ymm6);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 2)); /*A01[0][2]*/\
        ymm7 = _mm256_fmadd_pd(ymm2, ymm0, ymm7);\
        ymm8 = _mm256_fmadd_pd(ymm2, ymm1, ymm8);\
\
        a01 += 1;  /*move to next row*/\
        b10 += cs_b;\
    }

#define BLIS_DTRSM_SMALL_GEMM_2nx8m(a01,b10,cs_b,p_lda,k_iter)\
    for(k = 0; k < k_iter; k++)      /*loop for number of GEMM operations*/\
    {\
        /*load 8x1 block of B10*/\
        ymm0 = _mm256_loadu_pd((double const *)b10);/*B10[0][0] B10[1][0] B10[2][0] B10[3][0]*/\
        ymm1 = _mm256_loadu_pd((double const *)(b10 + 4));/*B10[4][0] B10[5][0] B10[6][0] B10[7][0]*/\
\
        /*broadcast 1st row of A01*/\
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 0));  /*A01[0][0]*/\
        ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3);\
        ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 1));  /*A01[0][1]*/\
        ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5);\
        ymm6 = _mm256_fmadd_pd(ymm2, ymm1, ymm6);\
\
        a01 += 1;  /*move to next row*/\
        b10 += cs_b;\
    }

#define BLIS_DTRSM_SMALL_GEMM_1nx8m(a01,b10,cs_b,p_lda,k_iter)\
    for(k = 0; k < k_iter; k++)      /*loop for number of GEMM operations*/\
    {\
        /*load 8x1 block of B10*/\
        ymm0 = _mm256_loadu_pd((double const *)b10);/*B10[0][0] B10[1][0] B10[2][0] B10[3][0]*/\
        ymm1 = _mm256_loadu_pd((double const *)(b10 + 4));/*B10[4][0] B10[5][0] B10[6][0] B10[7][0]*/\
\
        /*broadcast 1st row of A01*/\
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 0));  /*A01[0][0]*/\
        ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3);\
        ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4);\
\
        a01 += 1;  /*move to next row*/\
        b10 += cs_b;\
    }

#define BLIS_DTRSM_SMALL_GEMM_4nx4m(a01,b10,cs_b,p_lda,k_iter) \
    for(k = 0; k < k_iter; k++)      /*loop for number of GEMM operations*/\
    {\
        /*load 8x1 block of B10*/\
        ymm0 = _mm256_loadu_pd((double const *)b10);/*B10[0][0] B10[1][0] B10[2][0] B10[3][0]*/\
\
        /*broadcast 1st row of A01*/\
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 0));  /*A01[0][0]*/\
        ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 1));  /*A01[0][1]*/\
        ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 2)); /*A01[0][2]*/\
        ymm7 = _mm256_fmadd_pd(ymm2, ymm0, ymm7);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 3)); /*A01[0][3]*/\
        ymm9 = _mm256_fmadd_pd(ymm2, ymm0, ymm9);\
\
        a01 += 1;  /*move to next row*/\
        b10 += cs_b;\
    }

#define BLIS_DTRSM_SMALL_GEMM_3nx4m(a01,b10,cs_b,p_lda,k_iter) \
    for(k = 0; k < k_iter; k++)      /*loop for number of GEMM operations*/\
    {\
        /*load 8x1 block of B10*/\
        ymm0 = _mm256_loadu_pd((double const *)b10);/*B10[0][0] B10[1][0] B10[2][0] B10[3][0]*/\
\
        /*broadcast 1st row of A01*/\
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 0));  /*A01[0][0]*/\
        ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 1));  /*A01[0][1]*/\
        ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 2)); /*A01[0][2]*/\
        ymm7 = _mm256_fmadd_pd(ymm2, ymm0, ymm7);\
\
        a01 += 1;  /*move to next row*/\
        b10 += cs_b;\
    }

#define BLIS_DTRSM_SMALL_GEMM_2nx4m(a01,b10,cs_b,p_lda,k_iter) \
    for(k = 0; k < k_iter; k++)      /*loop for number of GEMM operations*/\
    {\
        /*load 8x1 block of B10*/\
        ymm0 = _mm256_loadu_pd((double const *)b10);/*B10[0][0] B10[1][0] B10[2][0] B10[3][0]*/\
\
        /*broadcast 1st row of A01*/\
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 0));  /*A01[0][0]*/\
        ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 1));  /*A01[0][1]*/\
        ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5);\
\
        a01 += 1;  /*move to next row*/\
        b10 += cs_b;\
    }

#define BLIS_DTRSM_SMALL_GEMM_1nx4m(a01,b10,cs_b,p_lda,k_iter) \
    for(k = 0; k < k_iter; k++)      /*loop for number of GEMM operations*/\
    {\
        /*load 8x1 block of B10*/\
        ymm0 = _mm256_loadu_pd((double const *)b10);/*B10[0][0] B10[1][0] B10[2][0] B10[3][0]*/\
\
        /*broadcast 1st row of A01*/\
        ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 0));  /*A01[0][0]*/\
        ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3);\
\
        a01 += 1;  /*move to next row*/\
        b10 += cs_b;\
    }

/*GEMM block used in dtrsm small left cases*/
#define BLIS_DTRSM_SMALL_GEMM_8mx6n(a10,b01,cs_b,p_lda,k_iter) \
    double *b01_prefetch = b01 + 8;      \
    for(k = 0; k< k_iter; k++) \
    { \
        ymm0 = _mm256_loadu_pd((double const *)(a10)); \
        ymm1 = _mm256_loadu_pd((double const *)(a10 + 4)); \
        _mm_prefetch((char*)( a10 + 64), _MM_HINT_T0); \
        /*Calculate the next micro pannel address to prefetch*/ \
        if(k & 0x7) b01_prefetch += cs_b; \
        else b01_prefetch = b01+ 8; \
        ymm2 = _mm256_broadcast_sd((double const *)(b01)); \
        ymm8 = _mm256_fmadd_pd(ymm2, ymm0, ymm8); \
        ymm12 = _mm256_fmadd_pd(ymm2, ymm1, ymm12); \
        \
        ymm2 = _mm256_broadcast_sd((double const *)(b01 + cs_b * 1)); \
        ymm9 = _mm256_fmadd_pd(ymm2, ymm0, ymm9); \
        ymm13 = _mm256_fmadd_pd(ymm2, ymm1, ymm13); \
        \
        ymm2 = _mm256_broadcast_sd((double const *)(b01 + cs_b * 2)); \
        ymm10 = _mm256_fmadd_pd(ymm2, ymm0, ymm10); \
        ymm14 = _mm256_fmadd_pd(ymm2, ymm1, ymm14); \
        \
        /*Prefetch the next 6x8 micro panelof B */ \
        _mm_prefetch((char*)( b01_prefetch), _MM_HINT_T0); \
        \
        ymm2 = _mm256_broadcast_sd((double const *)(b01 + cs_b * 3)); \
        ymm11 = _mm256_fmadd_pd(ymm2, ymm0, ymm11); \
        ymm15 = _mm256_fmadd_pd(ymm2, ymm1, ymm15); \
        \
        ymm2 = _mm256_broadcast_sd((double const *)(b01 + cs_b * 4)); \
        ymm4 = _mm256_fmadd_pd(ymm2, ymm0, ymm4); \
        ymm6 = _mm256_fmadd_pd(ymm2, ymm1, ymm6); \
        \
        ymm2 = _mm256_broadcast_sd((double const *)(b01 + cs_b * 5)); \
        ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5); \
        ymm7 = _mm256_fmadd_pd(ymm2, ymm1, ymm7); \
        \
        b01 += 1;  \
        a10 += p_lda; \
    }

#define BLIS_DTRSM_SMALL_GEMM_8mx4n(a10,b01,cs_b,p_lda,k_iter) \
    for(k = 0; k< k_iter; k++)   /*loop for number of GEMM operations*/\
    {\
        ymm0 = _mm256_loadu_pd((double const *)(a10));\
        ymm1 = _mm256_loadu_pd((double const *)(a10 + 4));\
\
        ymm2 = _mm256_broadcast_sd((double const *)(b01));\
        ymm8 = _mm256_fmadd_pd(ymm2, ymm0, ymm8);\
        ymm12 = _mm256_fmadd_pd(ymm2, ymm1, ymm12);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(b01 + cs_b * 1));\
        ymm9 = _mm256_fmadd_pd(ymm2, ymm0, ymm9);\
        ymm13 = _mm256_fmadd_pd(ymm2, ymm1, ymm13);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(b01 + cs_b * 2));\
        ymm10 = _mm256_fmadd_pd(ymm2, ymm0, ymm10);\
        ymm14 = _mm256_fmadd_pd(ymm2, ymm1, ymm14);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(b01 + cs_b * 3));\
        ymm11 = _mm256_fmadd_pd(ymm2, ymm0, ymm11);\
        ymm15 = _mm256_fmadd_pd(ymm2, ymm1, ymm15);\
\
        b01 += 1;           /*move to  next row of B*/\
        a10 += p_lda;        /*pointer math to calculate next block of A for GEMM*/\
    }

#define BLIS_DTRSM_SMALL_GEMM_8mx3n(a10,b01,cs_b,p_lda,k_iter) \
    for(k = 0; k< k_iter; k++)   /*loop for number of GEMM operations*/\
    {\
        ymm0 = _mm256_loadu_pd((double const *)(a10));\
        ymm1 = _mm256_loadu_pd((double const *)(a10 + 4));\
\
        ymm2 = _mm256_broadcast_sd((double const *)(b01 + cs_b * 0));\
        ymm8 = _mm256_fmadd_pd(ymm2, ymm0, ymm8);\
        ymm12 = _mm256_fmadd_pd(ymm2, ymm1, ymm12);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(b01 + cs_b * 1));\
        ymm9 = _mm256_fmadd_pd(ymm2, ymm0, ymm9);\
        ymm13 = _mm256_fmadd_pd(ymm2, ymm1, ymm13);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(b01 + cs_b * 2));\
        ymm10 = _mm256_fmadd_pd(ymm2, ymm0, ymm10);\
        ymm14 = _mm256_fmadd_pd(ymm2, ymm1, ymm14);\
\
        b01 += 1;   /*move to  next row of B*/\
        a10 += p_lda;/*pointer math to calculate next block of A for GEMM*/\
    }

#define BLIS_DTRSM_SMALL_GEMM_8mx2n(a10,b01,cs_b,p_lda,k_iter) \
    for(k = 0; k< k_iter; k++)   /*loop for number of GEMM operations*/\
    {\
        ymm0 = _mm256_loadu_pd((double const *)(a10));\
        ymm1 = _mm256_loadu_pd((double const *)(a10 + 4));\
\
        ymm2 = _mm256_broadcast_sd((double const *)(b01 + cs_b * 0));\
        ymm8 = _mm256_fmadd_pd(ymm2, ymm0, ymm8);\
        ymm12 = _mm256_fmadd_pd(ymm2, ymm1, ymm12);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(b01 + cs_b * 1));\
        ymm9 = _mm256_fmadd_pd(ymm2, ymm0, ymm9);\
        ymm13 = _mm256_fmadd_pd(ymm2, ymm1, ymm13);\
\
        b01 += 1;   /*move to  next row of B*/\
        a10 += p_lda;/*pointer math to calculate next block of A for GEMM*/\
    }

#define BLIS_DTRSM_SMALL_GEMM_8mx1n(a10,b01,cs_b,p_lda,k_iter) \
    for(k = 0; k< k_iter; k++)   /*loop for number of GEMM operations*/\
    {\
        ymm0 = _mm256_loadu_pd((double const *)(a10));\
        ymm1 = _mm256_loadu_pd((double const *)(a10 + 4));\
\
        ymm2 = _mm256_broadcast_sd((double const *)(b01 + cs_b * 0));\
        ymm8 = _mm256_fmadd_pd(ymm2, ymm0, ymm8);\
        ymm12 = _mm256_fmadd_pd(ymm2, ymm1, ymm12);\
        b01 += 1;   /*move to  next row of B*/\
        a10 += p_lda;/*pointer math to calculate next block of A for GEMM*/\
    }

#define BLIS_DTRSM_SMALL_GEMM_4mx6n(a10,b01,cs_b,p_lda,k_iter) \
    for(k = 0; k< k_iter; k++)   /*loop for number of GEMM operations*/\
    {\
        ymm0 = _mm256_loadu_pd((double const *)(a10));\
\
        ymm2 = _mm256_broadcast_sd((double const *)(b01 + cs_b * 0));\
        ymm8 = _mm256_fmadd_pd(ymm2, ymm0, ymm8);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(b01 + cs_b * 1));\
        ymm9 = _mm256_fmadd_pd(ymm2, ymm0, ymm9);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(b01 + cs_b * 2));\
        ymm10 = _mm256_fmadd_pd(ymm2, ymm0, ymm10);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(b01 + cs_b * 3));\
        ymm11 = _mm256_fmadd_pd(ymm2, ymm0, ymm11);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(b01 + cs_b * 4));\
        ymm4 = _mm256_fmadd_pd(ymm2, ymm0, ymm4);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(b01 + cs_b * 5));\
        ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5);\
\
        b01 += 1;   /*move to  next row of B*/\
        a10 += p_lda;/*pointer math to calculate next block of A for GEMM*/\
    }

#define BLIS_DTRSM_SMALL_GEMM_4mx4n(a10,b01,cs_b,p_lda,k_iter) \
    for(k = 0; k< k_iter; k++)   /*loop for number of GEMM operations*/\
    {\
        ymm0 = _mm256_loadu_pd((double const *)(a10));\
\
        ymm2 = _mm256_broadcast_sd((double const *)(b01 + cs_b * 0));\
        ymm8 = _mm256_fmadd_pd(ymm2, ymm0, ymm8);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(b01 + cs_b * 1));\
        ymm9 = _mm256_fmadd_pd(ymm2, ymm0, ymm9);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(b01 + cs_b * 2));\
        ymm10 = _mm256_fmadd_pd(ymm2, ymm0, ymm10);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(b01 + cs_b * 3));\
        ymm11 = _mm256_fmadd_pd(ymm2, ymm0, ymm11);\
\
        b01 += 1;   /*move to  next row of B*/\
        a10 += p_lda;/*pointer math to calculate next block of A for GEMM*/\
    }

#define BLIS_DTRSM_SMALL_GEMM_4mx3n(a10,b01,cs_b,p_lda,k_iter) \
    for(k = 0; k< k_iter; k++)   /*loop for number of GEMM operations*/\
    {\
        ymm0 = _mm256_loadu_pd((double const *)(a10));\
\
        ymm2 = _mm256_broadcast_sd((double const *)(b01 + cs_b * 0));\
        ymm8 = _mm256_fmadd_pd(ymm2, ymm0, ymm8);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(b01 + cs_b * 1));\
        ymm9 = _mm256_fmadd_pd(ymm2, ymm0, ymm9);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(b01 + cs_b * 2));\
        ymm10 = _mm256_fmadd_pd(ymm2, ymm0, ymm10);\
\
        b01 += 1;   /*move to  next row of B*/\
        a10 += p_lda;/*pointer math to calculate next block of A for GEMM*/\
    }

#define BLIS_DTRSM_SMALL_GEMM_4mx2n(a10,b01,cs_b,p_lda,k_iter) \
    for(k = 0; k< k_iter; k++)   /*loop for number of GEMM operations*/\
    {\
        ymm0 = _mm256_loadu_pd((double const *)(a10));\
\
        ymm2 = _mm256_broadcast_sd((double const *)(b01 + cs_b * 0));\
        ymm8 = _mm256_fmadd_pd(ymm2, ymm0, ymm8);\
\
        ymm2 = _mm256_broadcast_sd((double const *)(b01 + cs_b * 1));\
        ymm9 = _mm256_fmadd_pd(ymm2, ymm0, ymm9);\
\
        b01 += 1;   /*move to  next row of B*/\
        a10 += p_lda;/*pointer math to calculate next block of A for GEMM*/\
    }

#define BLIS_DTRSM_SMALL_GEMM_4mx1n(a10,b01,cs_b,p_lda,k_iter) \
    for(k = 0; k< k_iter; k++)   /*loop for number of GEMM operations*/\
    {\
        ymm0 = _mm256_loadu_pd((double const *)(a10));\
\
        ymm2 = _mm256_broadcast_sd((double const *)(b01 + cs_b * 0));\
        ymm8 = _mm256_fmadd_pd(ymm2, ymm0, ymm8);\
\
        b01 += 1;   /*move to  next row of B*/\
        a10 += p_lda;/*pointer math to calculate next block of A for GEMM*/\
    }

/*
   Load b11 of size 6x8 and multiply with alpha
   Add the GEMM output and perform inregister transose of b11
   to peform DTRSM operation for left cases.
*/
#define BLIS_DTRSM_SMALL_NREG_TRANSPOSE_6x8(b11,cs_b,AlphaVal) \
        ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));\
\
        ymm0 = _mm256_loadu_pd((double const *)(b11));\
        ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *1));\
        ymm2 = _mm256_loadu_pd((double const *)(b11 + cs_b *2));\
        ymm3 = _mm256_loadu_pd((double const *)(b11 + cs_b *3));\
        ymm0 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);\
        ymm1 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);\
        ymm2 = _mm256_fmsub_pd(ymm2, ymm16, ymm10);\
        ymm3 = _mm256_fmsub_pd(ymm3, ymm16, ymm11);\
\
        ymm9 = _mm256_unpacklo_pd(ymm0, ymm1);          \
        ymm11 = _mm256_unpacklo_pd(ymm2, ymm3);         \
        ymm8 = _mm256_permute2f128_pd(ymm9,ymm11,0x20); \
        ymm10 = _mm256_permute2f128_pd(ymm9,ymm11,0x31);\
        ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);          \
        ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);          \
        ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);  \
        ymm11 = _mm256_permute2f128_pd(ymm0,ymm1,0x31); \
\
        ymm0 = _mm256_loadu_pd((double const *)(b11 + 4));\
        ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *1 + 4));\
        ymm2 = _mm256_loadu_pd((double const *)(b11 + cs_b *2 + 4));\
        ymm3 = _mm256_loadu_pd((double const *)(b11 + cs_b *3 + 4));\
        ymm0 = _mm256_fmsub_pd(ymm0, ymm16, ymm12);\
        ymm1 = _mm256_fmsub_pd(ymm1, ymm16, ymm13);\
        ymm2 = _mm256_fmsub_pd(ymm2, ymm16, ymm14);\
        ymm3 = _mm256_fmsub_pd(ymm3, ymm16, ymm15);\
\
        ymm13 = _mm256_unpacklo_pd(ymm0, ymm1);\
        ymm15 = _mm256_unpacklo_pd(ymm2, ymm3);\
        ymm12 = _mm256_permute2f128_pd(ymm13,ymm15,0x20);\
        ymm14 = _mm256_permute2f128_pd(ymm13,ymm15,0x31);\
        ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);\
        ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);\
\
        ymm13 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);\
        ymm15 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);\
        ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *4));\
        ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *5));\
        ymm0 = _mm256_fmsub_pd(ymm0, ymm16, ymm4);\
        ymm1 = _mm256_fmsub_pd(ymm1, ymm16, ymm5);\
        ymm2 = _mm256_loadu_pd((double const *)(b11 + cs_b *4 + 4));\
        ymm3 = _mm256_loadu_pd((double const *)(b11 + cs_b *5 + 4));\
        ymm2 = _mm256_fmsub_pd(ymm2, ymm16, ymm6);\
        ymm3 = _mm256_fmsub_pd(ymm3, ymm16, ymm7);\
\
        ymm16 = _mm256_broadcast_sd((double const *)(&ones));\
        ymm7 = _mm256_unpacklo_pd(ymm0, ymm1);\
        ymm4 = _mm256_permute2f128_pd(ymm7,ymm16,0x20);\
        ymm6 = _mm256_permute2f128_pd(ymm7,ymm16,0x31);\
\
        ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);\
        ymm5 = _mm256_permute2f128_pd(ymm0,ymm16,0x20);\
        ymm7 = _mm256_permute2f128_pd(ymm0,ymm16,0x31);\
        ymm18 = _mm256_unpacklo_pd(ymm2, ymm3);\
        ymm17 = _mm256_permute2f128_pd(ymm18,ymm16,0x20);\
        ymm19 = _mm256_permute2f128_pd(ymm18,ymm16,0x31);\
\
        /*unpackhigh*/\
        ymm20 = _mm256_unpackhi_pd(ymm2, ymm3);\
\
        /*rearrange high elements*/\
        ymm18 = _mm256_permute2f128_pd(ymm20,ymm16,0x20);\
        ymm20 = _mm256_permute2f128_pd(ymm20,ymm16,0x31);

#define BLIS_DTRSM_SMALL_NREG_TRANSPOSE_8x6_AND_STORE(b11,cs_b)\
        ymm1 = _mm256_unpacklo_pd(ymm8, ymm9);\
        ymm3 = _mm256_unpacklo_pd(ymm10, ymm11);\
\
        /*rearrange low elements*/\
        ymm0 = _mm256_permute2f128_pd(ymm1, ymm3, 0x20);\
        ymm2 = _mm256_permute2f128_pd(ymm1, ymm3, 0x31);\
\
        /*unpack high*/\
        ymm8 = _mm256_unpackhi_pd(ymm8, ymm9);\
        ymm9 = _mm256_unpackhi_pd(ymm10, ymm11);\
\
        /*rearrange high elements*/\
        ymm1 = _mm256_permute2f128_pd(ymm8, ymm9, 0x20);\
        ymm3 = _mm256_permute2f128_pd(ymm8, ymm9, 0x31);\
\
        _mm256_storeu_pd((double *)(b11), ymm0);\
        _mm256_storeu_pd((double *)(b11 + cs_b * 1), ymm1);\
        _mm256_storeu_pd((double *)(b11 + cs_b * 2), ymm2);\
        _mm256_storeu_pd((double *)(b11 + cs_b * 3), ymm3);\
\
        /*unpacklow*/\
        ymm1 = _mm256_unpacklo_pd(ymm12, ymm13);\
        ymm3 = _mm256_unpacklo_pd(ymm14, ymm15);\
\
        /*rearrange low elements*/\
        ymm0 = _mm256_permute2f128_pd(ymm1, ymm3, 0x20);\
        ymm2 = _mm256_permute2f128_pd(ymm1, ymm3, 0x31);\
\
        /*unpack high*/\
        ymm12 = _mm256_unpackhi_pd(ymm12, ymm13);\
        ymm13 = _mm256_unpackhi_pd(ymm14, ymm15);\
\
        /*rearrange high elements*/\
        ymm1 = _mm256_permute2f128_pd(ymm12, ymm13, 0x20);\
        ymm3 = _mm256_permute2f128_pd(ymm12, ymm13, 0x31);\
\
        _mm256_storeu_pd((double *)(b11 + 4), ymm0);\
        _mm256_storeu_pd((double *)(b11 + cs_b * 1 + 4), ymm1);\
        _mm256_storeu_pd((double *)(b11 + cs_b * 2 + 4), ymm2);\
        _mm256_storeu_pd((double *)(b11 + cs_b * 3 + 4), ymm3);\
\
        /*unpacklow*/\
        ymm1 = _mm256_unpacklo_pd(ymm4, ymm5);\
        ymm3 = _mm256_unpacklo_pd(ymm6, ymm7);\
\
        /*rearrange low elements*/\
        ymm0 = _mm256_permute2f128_pd(ymm1, ymm3, 0x20);\
\
        /*unpack high*/\
        ymm4 = _mm256_unpackhi_pd(ymm4, ymm5);\
        ymm5 = _mm256_unpackhi_pd(ymm6, ymm7);\
\
        /*rearrange high elements*/\
        ymm1 = _mm256_permute2f128_pd(ymm4, ymm5, 0x20);\
\
        _mm256_storeu_pd((double *)(b11 + cs_b * 4), ymm0);\
        _mm256_storeu_pd((double *)(b11 + cs_b * 5), ymm1);\
\
        /*unpacklow*/\
        ymm1 = _mm256_unpacklo_pd(ymm17, ymm18);\
        ymm3 = _mm256_unpacklo_pd(ymm19, ymm20);\
\
        /*rearrange low elements*/\
        ymm0 = _mm256_permute2f128_pd(ymm1, ymm3, 0x20);\
\
        /*unpack high*/\
        ymm17 = _mm256_unpackhi_pd(ymm17, ymm18);\
        ymm18 = _mm256_unpackhi_pd(ymm19, ymm20);\
\
        /*rearrange high elements*/\
        ymm1 = _mm256_permute2f128_pd(ymm17, ymm18, 0x20);\
\
        _mm256_storeu_pd((double *)(b11 + cs_b * 4 + 4), ymm0);\
        _mm256_storeu_pd((double *)(b11 + cs_b * 5 + 4), ymm1);

#define BLIS_PRE_DTRSM_SMALL_3M_3N(AlphaVal,b11,cs_b)\
    ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     /*register to hold alpha*/\
\
    ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));\
    ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *1));\
    xmm5 = _mm_loadu_pd((double const*)(b11 + cs_b * 2));\
    ymm2 = _mm256_broadcast_sd((double const*)(b11 + cs_b * 2 + 2));\
    ymm2 = _mm256_insertf128_pd(ymm2, xmm5, 0);\
\
    ymm8 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);\
    ymm9 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);\
    ymm10 = _mm256_fmsub_pd(ymm2, ymm16, ymm10);\
\
    ymm0 = _mm256_blend_pd(ymm8, ymm0, 0x08);\
    ymm1 = _mm256_blend_pd(ymm9, ymm1, 0x08);\
    ymm2 = _mm256_blend_pd(ymm10, ymm2, 0x08);\
\
    _mm256_storeu_pd((double *)(b11), ymm0);                   /*store(B11[0-3][0])*/\
    _mm256_storeu_pd((double *)(b11 + cs_b * 1), ymm1);        /*store(B11[0-3][1])*/\
    xmm5 = _mm256_extractf128_pd(ymm2, 0);\
    _mm_storeu_pd((double *)(b11 + cs_b * 2), xmm5);\
    _mm_storel_pd((b11 + cs_b * 2 + 2), _mm256_extractf128_pd(ymm2, 1));

#define BLIS_PRE_DTRSM_SMALL_3M_2N(AlphaVal,b11,cs_b)\
    ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     /*register to hold alpha*/\
\
    ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));\
    xmm5 = _mm_loadu_pd((double const*)(b11 + cs_b * 1));\
    ymm1 = _mm256_broadcast_sd((double const*)(b11 + cs_b * 1 + 2));\
    ymm1 = _mm256_insertf128_pd(ymm1, xmm5, 0);\
\
    ymm8 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);\
    ymm9 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);\
\
    ymm0 = _mm256_blend_pd(ymm8, ymm0, 0x08);\
    ymm1 = _mm256_blend_pd(ymm9, ymm1, 0x08);\
\
    _mm256_storeu_pd((double *)(b11), ymm0);                /*store(B11[0-3][0])*/\
    xmm5 = _mm256_extractf128_pd(ymm1, 0);\
    _mm_storeu_pd((double *)(b11 + cs_b * 1), xmm5);\
    _mm_storel_pd((b11 + cs_b * 1 + 2), _mm256_extractf128_pd(ymm1, 1));

#define BLIS_PRE_DTRSM_SMALL_3M_1N(AlphaVal,b11,cs_b)\
    ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     /*register to hold alpha*/\
\
    xmm5 = _mm_loadu_pd((double const*)(b11 + cs_b * 0));\
    ymm0 = _mm256_broadcast_sd((double const*)(b11 + cs_b * 0 + 2));\
    ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);\
    ymm8 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);\
    ymm0 = _mm256_blend_pd(ymm8, ymm0, 0x08);\
\
    xmm5 = _mm256_extractf128_pd(ymm0, 0);\
    _mm_storeu_pd((double *)(b11), xmm5);\
    _mm_storel_pd((b11 + 2), _mm256_extractf128_pd(ymm0, 1));


#define BLIS_PRE_DTRSM_SMALL_2M_3N(AlphaVal,b11,cs_b)\
    ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     /*register to hold alpha*/\
\
    ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));\
    ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *1));\
    xmm5 = _mm_loadu_pd((double const*)(b11 + cs_b * 2));\
    ymm2 = _mm256_insertf128_pd(ymm2, xmm5, 0);\
\
    ymm8 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);\
    ymm9 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);\
    ymm10 = _mm256_fmsub_pd(ymm2, ymm16, ymm10);\
\
    ymm0 = _mm256_blend_pd(ymm8, ymm0, 0x0C);\
    ymm1 = _mm256_blend_pd(ymm9, ymm1, 0x0C);\
    ymm2 = _mm256_blend_pd(ymm10, ymm2, 0x0C);\
\
    _mm256_storeu_pd((double *)(b11), ymm0);                   /*store(B11[0-3][0])*/\
    _mm256_storeu_pd((double *)(b11 + cs_b * 1), ymm1);        /*store(B11[0-3][1])*/\
    xmm5 = _mm256_extractf128_pd(ymm2, 0);\
    _mm_storeu_pd((double *)(b11 + cs_b * 2), xmm5);

#define BLIS_PRE_DTRSM_SMALL_2M_2N(AlphaVal,b11,cs_b)\
    ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     /*register to hold alpha*/\
\
    ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));\
    xmm5 = _mm_loadu_pd((double const*)(b11 + cs_b * 1));\
    ymm1 = _mm256_insertf128_pd(ymm1, xmm5, 0);\
\
    ymm8 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);\
    ymm9 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);\
\
    ymm0 = _mm256_blend_pd(ymm8, ymm0, 0x0C);\
    ymm1 = _mm256_blend_pd(ymm9, ymm1, 0x0C);\
\
    _mm256_storeu_pd((double *)(b11), ymm0);                /*store(B11[0-3][0])*/\
    xmm5 = _mm256_extractf128_pd(ymm1, 0);\
    _mm_storeu_pd((double *)(b11 + cs_b * 1), xmm5);

#define BLIS_PRE_DTRSM_SMALL_2M_1N(AlphaVal,b11,cs_b)\
    ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     /*register to hold alpha*/\
\
    xmm5 = _mm_loadu_pd((double const*)(b11 + cs_b * 0));\
    ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);\
\
    ymm8 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);\
    ymm0 = _mm256_blend_pd(ymm8, ymm0, 0x0C);\
\
    xmm5 = _mm256_extractf128_pd(ymm0, 0);\
    _mm_storeu_pd((double *)(b11 + cs_b * 0), xmm5);

#define BLIS_PRE_DTRSM_SMALL_1M_3N(AlphaVal,b11,cs_b)\
    ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     /*register to hold alpha*/\
\
    ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b *0));\
    ymm1 = _mm256_broadcast_sd((double const *)(b11 + cs_b *1));\
    ymm2 = _mm256_broadcast_sd((double const *)(b11 + cs_b *2));\
\
    ymm8 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);\
    ymm9 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);\
    ymm10 = _mm256_fmsub_pd(ymm2, ymm16, ymm10);\
\
    ymm0 = _mm256_blend_pd(ymm8, ymm0, 0x0E);\
    ymm1 = _mm256_blend_pd(ymm9, ymm1, 0x0E);\
    ymm2 = _mm256_blend_pd(ymm10, ymm2, 0x0E);\
\
    _mm_storel_pd((b11 + cs_b * 0), _mm256_extractf128_pd(ymm0, 0));\
    _mm_storel_pd((b11 + cs_b * 1), _mm256_extractf128_pd(ymm1, 0));\
    _mm_storel_pd((b11 + cs_b * 2), _mm256_extractf128_pd(ymm2, 0));

#define BLIS_PRE_DTRSM_SMALL_1M_2N(AlphaVal,b11,cs_b)\
    ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     /*register to hold alpha*/\
\
    ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b *0));\
    ymm1 = _mm256_broadcast_sd((double const *)(b11 + cs_b *1));\
\
    ymm8 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);\
    ymm9 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);\
\
    ymm0 = _mm256_blend_pd(ymm8, ymm0, 0x0E);\
    ymm1 = _mm256_blend_pd(ymm9, ymm1, 0x0E);\
\
    _mm_storel_pd((b11 + cs_b * 0), _mm256_extractf128_pd(ymm0, 0));\
    _mm_storel_pd((b11 + cs_b * 1), _mm256_extractf128_pd(ymm1, 0));

#define BLIS_PRE_DTRSM_SMALL_1M_1N(AlphaVal,b11,cs_b)\
    ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     /*register to hold alpha*/\
\
    ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b *0));\
    ymm8 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);\
\
    ymm0 = _mm256_blend_pd(ymm8, ymm0, 0x0E);\
\
    _mm_storel_pd((b11 + cs_b * 0), _mm256_extractf128_pd(ymm0, 0));

/* pre & post TRSM for Right remainder cases*/
#define BLIS_PRE_DTRSM_SMALL_3N_3M(AlphaVal,b11,cs_b)\
    ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);         /*register to hold alpha*/\
\
    ymm0 = _mm256_loadu_pd((double const *)b11);\
    ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);\
\
    ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));\
    ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);\
\
    xmm5 = _mm_loadu_pd((double const*)(b11 + cs_b * 2));\
    ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b*2 + 2));\
    ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);\
    ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);

#define BLIS_POST_DTRSM_SMALL_3N_3M(b11,cs_b)\
    ymm0 = _mm256_loadu_pd((double const *)b11);\
    ymm3 = _mm256_blend_pd(ymm0, ymm3, 0x07);\
    ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));\
    ymm5 = _mm256_blend_pd(ymm0, ymm5, 0x07);\
    xmm5 = _mm_loadu_pd((double const*)(b11 + cs_b * 2));\
    ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b*2 + 2));\
    ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);\
    ymm7 = _mm256_blend_pd(ymm0, ymm7, 0x07);\
\
    _mm256_storeu_pd((double *)b11, ymm3);\
    _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);\
    xmm5 = _mm256_extractf128_pd(ymm7, 0);\
    _mm_storeu_pd((double *)(b11 + cs_b * 2),xmm5);\
    _mm_storel_pd((b11 + cs_b * 2 + 2), _mm256_extractf128_pd(ymm7, 1));

#define BLIS_PRE_DTRSM_SMALL_3N_2M(AlphaVal,b11,cs_b)\
    ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);         /*register to hold alpha*/\
\
    ymm0 = _mm256_loadu_pd((double const *)b11);\
    ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);\
\
    ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));\
    ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);\
\
    xmm5 = _mm_loadu_pd((double const*)(b11 + cs_b * 2));\
    ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);\
    ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);

#define BLIS_POST_DTRSM_SMALL_3N_2M(b11,cs_b)\
    ymm0 = _mm256_loadu_pd((double const *)b11);\
    ymm3 = _mm256_blend_pd(ymm0, ymm3, 0x03);\
    ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));\
    ymm5 = _mm256_blend_pd(ymm0, ymm5, 0x03);\
    xmm5 = _mm_loadu_pd((double const*)(b11 + cs_b * 2));\
    ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);\
    ymm7 = _mm256_blend_pd(ymm0, ymm7, 0x03);\
\
    _mm256_storeu_pd((double *)b11, ymm3);\
    _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);\
    xmm5 = _mm256_extractf128_pd(ymm7, 0);\
    _mm_storeu_pd((double *)(b11 + cs_b * 2),xmm5);

#define BLIS_PRE_DTRSM_SMALL_3N_1M(AlphaVal,b11,cs_b)\
    ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);         /*register to hold alpha*/\
\
    ymm0 = _mm256_broadcast_sd((double const *)b11);\
    ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);\
\
    ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b));\
    ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);\
\
    ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b*2));\
    ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);

#define BLIS_POST_DTRSM_SMALL_3N_1M(b11,cs_b)\
    ymm0 = _mm256_broadcast_sd((double const *)b11);\
    ymm3 = _mm256_blend_pd(ymm0, ymm3, 0x01);\
    ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b));\
    ymm5 = _mm256_blend_pd(ymm0, ymm5, 0x01);\
    ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b*2));\
    ymm7 = _mm256_blend_pd(ymm0, ymm7, 0x01);\
\
    _mm_storel_pd((b11 + cs_b * 0), _mm256_extractf128_pd(ymm3, 0));\
    _mm_storel_pd((b11 + cs_b * 1), _mm256_extractf128_pd(ymm5, 0));\
    _mm_storel_pd((b11 + cs_b * 2), _mm256_extractf128_pd(ymm7, 0));

#define BLIS_PRE_DTRSM_SMALL_2N_3M(AlphaVal,b11,cs_b)\
    ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);         /*register to hold alpha*/\
\
    ymm0 = _mm256_loadu_pd((double const *)b11);\
    ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);\
\
    xmm5 = _mm_loadu_pd((double const*)(b11 + cs_b * 1));\
    ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b*1 + 2));\
    ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);\
    ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);

#define BLIS_POST_DTRSM_SMALL_2N_3M(b11,cs_b)\
    ymm0 = _mm256_loadu_pd((double const *)b11);\
    ymm3 = _mm256_blend_pd(ymm0, ymm3, 0x07);\
    xmm5 = _mm_loadu_pd((double const*)(b11 + cs_b * 1));\
    ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b*1 + 2));\
    ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);\
    ymm5 = _mm256_blend_pd(ymm0, ymm5, 0x07);\
\
    _mm256_storeu_pd((double *)b11, ymm3);\
    xmm5 = _mm256_extractf128_pd(ymm5, 0);\
    _mm_storeu_pd((double *)(b11 + cs_b*1), xmm5);\
    _mm_storel_pd((b11 + cs_b * 1 + 2), _mm256_extractf128_pd(ymm5, 1));

#define BLIS_PRE_DTRSM_SMALL_2N_2M(AlphaVal,b11,cs_b)\
    ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);         /*register to hold alpha*/\
\
    ymm0 = _mm256_loadu_pd((double const *)b11);\
    ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);\
\
    xmm5 = _mm_loadu_pd((double const*)(b11 + cs_b * 1));\
    ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);\
    ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);

#define BLIS_POST_DTRSM_SMALL_2N_2M(b11,cs_b)\
    ymm0 = _mm256_loadu_pd((double const *)b11);\
    ymm3 = _mm256_blend_pd(ymm0, ymm3, 0x03);\
    xmm5 = _mm_loadu_pd((double const*)(b11 + cs_b * 1));\
    ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);\
    ymm5 = _mm256_blend_pd(ymm0, ymm5, 0x03);\
\
    _mm256_storeu_pd((double *)b11, ymm3);\
    xmm5 = _mm256_extractf128_pd(ymm5, 0);\
    _mm_storeu_pd((double *)(b11 + cs_b*1), xmm5);

#define BLIS_PRE_DTRSM_SMALL_2N_1M(AlphaVal,b11,cs_b)\
    ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);         /*register to hold alpha*/\
\
    ymm0 = _mm256_broadcast_sd((double const *)b11);\
    ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);\
\
    ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b));\
    ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);

#define BLIS_POST_DTRSM_SMALL_2N_1M(b11,cs_b)\
    ymm0 = _mm256_broadcast_sd((double const *)b11);\
    ymm3 = _mm256_blend_pd(ymm0, ymm3, 0x01);\
    ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b));\
    ymm5 = _mm256_blend_pd(ymm0, ymm5, 0x01);\
\
    _mm_storel_pd(b11 , _mm256_extractf128_pd(ymm3, 0));\
    _mm_storel_pd((b11 + cs_b * 1), _mm256_extractf128_pd(ymm5, 0));

#define BLIS_PRE_DTRSM_SMALL_1N_3M(AlphaVal,b11,cs_b)\
    ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);         /*register to hold alpha*/\
\
    xmm5 = _mm_loadu_pd((double const*)(b11));\
    ymm0 = _mm256_broadcast_sd((double const *)(b11+ 2));\
    ymm6 = _mm256_insertf128_pd(ymm0, xmm5, 0);\
    ymm3 = _mm256_fmsub_pd(ymm6, ymm15, ymm3);

#define BLIS_POST_DTRSM_SMALL_1N_3M(b11,cs_b)\
    xmm5 = _mm256_extractf128_pd(ymm3, 0);\
    _mm_storeu_pd((double *)(b11), xmm5);\
    _mm_storel_pd((b11 + 2), _mm256_extractf128_pd(ymm3, 1));

#define BLIS_PRE_DTRSM_SMALL_1N_2M(AlphaVal,b11,cs_b)\
    ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);         /*register to hold alpha*/\
\
    xmm5 = _mm_loadu_pd((double const*)(b11));\
    ymm6 = _mm256_insertf128_pd(ymm0, xmm5, 0);\
    ymm3 = _mm256_fmsub_pd(ymm6, ymm15, ymm3);

#define BLIS_POST_DTRSM_SMALL_1N_2M(b11,cs_b)\
    ymm0 = _mm256_loadu_pd((double const *)b11);\
    ymm3 = _mm256_blend_pd(ymm6, ymm3, 0x03);\
\
    xmm5 = _mm256_extractf128_pd(ymm3, 0);\
    _mm_storeu_pd((double *)(b11), xmm5);

#define BLIS_PRE_DTRSM_SMALL_1N_1M(AlphaVal,b11,cs_b)\
    ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);         /*register to hold alpha*/\
\
    ymm6 = _mm256_broadcast_sd((double const *)b11);\
    ymm3 = _mm256_fmsub_pd(ymm6, ymm15, ymm3);

#define BLIS_POST_DTRSM_SMALL_1N_1M(b11,cs_b)\
    ymm3 = _mm256_blend_pd(ymm6, ymm3, 0x01);\
\
    _mm_storel_pd(b11, _mm256_extractf128_pd(ymm3, 0));

/* multiply with Alpha pre TRSM for 6*8 kernel*/
#define BLIS_PRE_DTRSM_SMALL_6x8(AlphaVal,b11,cs_b)\
    ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);\
\
    ymm0 = _mm256_loadu_pd((double const *)b11);\
    ymm1 = _mm256_loadu_pd((double const *)(b11 + 4));\
\
    ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);\
    ymm4 = _mm256_fmsub_pd(ymm1, ymm15, ymm4);\
\
    ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));\
    ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b + 4));\
\
    ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);\
    ymm6 = _mm256_fmsub_pd(ymm1, ymm15, ymm6);\
\
    ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*2));\
    ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b*2 + 4));\
\
    ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);\
    ymm8 = _mm256_fmsub_pd(ymm1, ymm15, ymm8);\
\
    ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*3));\
    ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b*3 + 4));\
\
    ymm9 = _mm256_fmsub_pd(ymm0, ymm15, ymm9);\
    ymm10 = _mm256_fmsub_pd(ymm1, ymm15, ymm10);\
\
    ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*4));\
    ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b*4 + 4));\
\
    ymm11 = _mm256_fmsub_pd(ymm0, ymm15, ymm11);\
    ymm12 = _mm256_fmsub_pd(ymm1, ymm15, ymm12);\
\
    ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*5));\
    ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b*5 + 4));\
\
    ymm13 = _mm256_fmsub_pd(ymm0, ymm15, ymm13);\
    ymm14 = _mm256_fmsub_pd(ymm1, ymm15, ymm14);

#define BLIS_PRE_DTRSM_SMALL_4x8(AlphaVal,b11,cs_b)\
    ymm15 = _mm256_broadcast_sd((double const *)(&AlphaVal));\
\
    ymm0 = _mm256_loadu_pd((double const *)b11);\
    ymm1 = _mm256_loadu_pd((double const *)(b11 + 4));\
\
    ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);\
    ymm4 = _mm256_fmsub_pd(ymm1, ymm15, ymm4);\
\
    ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));\
    ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b + 4));\
\
    ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);\
    ymm6 = _mm256_fmsub_pd(ymm1, ymm15, ymm6);\
\
    ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*2));\
    ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b*2 + 4));\
\
    ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);\
    ymm8 = _mm256_fmsub_pd(ymm1, ymm15, ymm8);\
\
    ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*3));\
    ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b*3 + 4));\
\
    ymm9 = _mm256_fmsub_pd(ymm0, ymm15, ymm9);\
    ymm10 = _mm256_fmsub_pd(ymm1, ymm15, ymm10);

#define BLIS_PRE_DTRSM_SMALL_6x4(AlphaVal,b11,cs_b)\
    ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);         /*register to hold alpha*/\
\
    ymm0 = _mm256_loadu_pd((double const *)b11);\
    ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);\
\
    ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));\
    ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);\
\
    ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*2));\
    ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);\
\
    ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*3));\
    ymm9 = _mm256_fmsub_pd(ymm0, ymm15, ymm9);\
\
    ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*4));\
    ymm11 = _mm256_fmsub_pd(ymm0, ymm15, ymm11);\
\
    ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*5));\
    ymm13 = _mm256_fmsub_pd(ymm0, ymm15, ymm13);

/*
    Pack a block of 8xk or 6xk from input buffer into packed buffer
    directly or after transpose based on input params
*/
BLIS_INLINE void bli_dtrsm_small_pack
(
    char side,
    dim_t size,
    bool trans,
    double *inbuf,
    dim_t cs_a,
    double *pbuff,
    dim_t p_lda,
    dim_t mr
)
{
    //scratch registers
    __m256d ymm0, ymm1, ymm2, ymm3;
    __m256d ymm4, ymm5, ymm6, ymm7;
    __m256d ymm8, ymm9, ymm10, ymm11;
    __m256d ymm12, ymm13;
    __m128d xmm0,xmm1,xmm2,xmm3;
    double zero = 0.0;

    if(side=='L'||side=='l')
    {
        /*Left case is 8xk*/
        if(trans)
        {
              /*
                -------------      -------------
                |           |      |     |     |
                |    4x8    |      |     |     |
                -------------  ==> | 8x4 | 8x4 |
                |    4x8    |      |     |     |
                |           |      |     |     |
                -------------      -------------
            */
            for(dim_t x = 0; x < size; x += mr)
            {
                ymm0 = _mm256_loadu_pd((double const *)(inbuf));
                ymm10 = _mm256_loadu_pd((double const *)(inbuf + 4));
                ymm1 = _mm256_loadu_pd((double const *)(inbuf + cs_a));
                ymm11 = _mm256_loadu_pd((double const *)(inbuf + 4 + cs_a));
                ymm2 = _mm256_loadu_pd((double const *)(inbuf + cs_a * 2));
                ymm12 = _mm256_loadu_pd((double const *)(inbuf + 4 + cs_a * 2));
                ymm3 = _mm256_loadu_pd((double const *)(inbuf + cs_a * 3));
                ymm13 = _mm256_loadu_pd((double const *)(inbuf + 4 + cs_a * 3));

                ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
                ymm5 = _mm256_unpacklo_pd(ymm2, ymm3);
                ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);
                ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
                ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);
                ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

                _mm256_storeu_pd((double *)(pbuff), ymm6);
                _mm256_storeu_pd((double *)(pbuff + p_lda), ymm7);
                _mm256_storeu_pd((double *)(pbuff + p_lda*2), ymm8);
                _mm256_storeu_pd((double *)(pbuff + p_lda*3), ymm9);

                ymm4 = _mm256_unpacklo_pd(ymm10, ymm11);
                ymm5 = _mm256_unpacklo_pd(ymm12, ymm13);

                ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);

                ymm0 = _mm256_unpackhi_pd(ymm10, ymm11);
                ymm1 = _mm256_unpackhi_pd(ymm12, ymm13);

                ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

                _mm256_storeu_pd((double *)(pbuff + p_lda * 4), ymm6);
                _mm256_storeu_pd((double *)(pbuff + p_lda * 5), ymm7);
                _mm256_storeu_pd((double *)(pbuff + p_lda * 6), ymm8);
                _mm256_storeu_pd((double *)(pbuff + p_lda * 7), ymm9);

                ymm0 = _mm256_loadu_pd((double const *)(inbuf + cs_a * 4));
                ymm10 = _mm256_loadu_pd((double const *)(inbuf + cs_a * 4 + 4));
                ymm1 = _mm256_loadu_pd((double const *)(inbuf + cs_a * 5));
                ymm11 = _mm256_loadu_pd((double const *)(inbuf + cs_a * 5 + 4));
                ymm2 = _mm256_loadu_pd((double const *)(inbuf + cs_a * 6));
                ymm12 = _mm256_loadu_pd((double const *)(inbuf + cs_a * 6 + 4));
                ymm3 = _mm256_loadu_pd((double const *)(inbuf + cs_a * 7));
                ymm13 = _mm256_loadu_pd((double const *)(inbuf + cs_a * 7 + 4));

                ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
                ymm5 = _mm256_unpacklo_pd(ymm2, ymm3);
                ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);
                ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
                ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);
                ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

                _mm256_storeu_pd((double *)(pbuff + 4), ymm6);
                _mm256_storeu_pd((double *)(pbuff + 4 + p_lda), ymm7);
                _mm256_storeu_pd((double *)(pbuff + 4 + p_lda*2), ymm8);
                _mm256_storeu_pd((double *)(pbuff + 4 + p_lda*3), ymm9);

                ymm4 = _mm256_unpacklo_pd(ymm10, ymm11);
                ymm5 = _mm256_unpacklo_pd(ymm12, ymm13);
                ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);
                ymm0 = _mm256_unpackhi_pd(ymm10, ymm11);
                ymm1 = _mm256_unpackhi_pd(ymm12, ymm13);
                ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

                _mm256_storeu_pd((double *)(pbuff + 4 + p_lda * 4), ymm6);
                _mm256_storeu_pd((double *)(pbuff + 4 + p_lda * 5), ymm7);
                _mm256_storeu_pd((double *)(pbuff + 4 + p_lda * 6), ymm8);
                _mm256_storeu_pd((double *)(pbuff + 4 + p_lda * 7), ymm9);

                inbuf += mr;
                pbuff += mr*mr;
            }
        }else
        {
            //Expected multiples of 4
            p_lda = 8;
            for(dim_t x = 0; x < size; x++)
            {
                ymm0 = _mm256_loadu_pd((double const *)(inbuf));
                _mm256_storeu_pd((double *)(pbuff), ymm0);
                ymm1 = _mm256_loadu_pd((double const *)(inbuf + 4));
                _mm256_storeu_pd((double *)(pbuff + 4), ymm1);
                inbuf+=cs_a;
                pbuff+=p_lda;
            }
        }
    }else if(side=='R'||side=='r')
    {

        if(trans)
        {
             /*
                 ------------------        ----------
                 |     |     |             |     |    |
                 | 4x4 | 4x4 |             | 4x4 |4x2 |
                 -------------  ==>        -------------
                 |     |     |             |     |    |
                 | 2x4 | 2x4 |             | 2x4 |2x2 |
                 -------------------       -------------
             */
            for(dim_t x=0; x<p_lda; x += mr)
            {
                ymm0 = _mm256_loadu_pd((double const *)(inbuf));
                ymm1 = _mm256_loadu_pd((double const *)(inbuf + cs_a));
                ymm2 = _mm256_loadu_pd((double const *)(inbuf + cs_a * 2));
                ymm3 = _mm256_loadu_pd((double const *)(inbuf + cs_a * 3));

                ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
                ymm5 = _mm256_unpacklo_pd(ymm2, ymm3);
                ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);

                ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
                ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);
                ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

                _mm256_storeu_pd((double *)(pbuff), ymm6);
                _mm256_storeu_pd((double *)(pbuff + p_lda), ymm7);
                _mm256_storeu_pd((double *)(pbuff + p_lda*2), ymm8);
                _mm256_storeu_pd((double *)(pbuff + p_lda*3), ymm9);

                ymm10 = _mm256_loadu_pd((double const *)(inbuf + 4));
                ymm11 = _mm256_loadu_pd((double const *)(inbuf + 4 + cs_a));
                ymm12 = _mm256_loadu_pd((double const *)(inbuf + 4 + cs_a * 2));
                ymm13 = _mm256_loadu_pd((double const *)(inbuf + 4 + cs_a * 3));

                ymm4 = _mm256_unpacklo_pd(ymm10, ymm11);
                ymm5 = _mm256_unpacklo_pd(ymm12, ymm13);
                ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                ymm8 = _mm256_broadcast_sd((double const *)&zero);

                ymm0 = _mm256_unpackhi_pd(ymm10, ymm11);
                ymm1 = _mm256_unpackhi_pd(ymm12, ymm13);
                ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                ymm9 = _mm256_broadcast_sd((double const *)&zero);

                _mm256_storeu_pd((double *)(pbuff + p_lda * 4), ymm6);
                _mm256_storeu_pd((double *)(pbuff + p_lda * 5), ymm7);

                ymm0 = _mm256_loadu_pd((double const *)(inbuf + cs_a * 4));
                ymm10 = _mm256_loadu_pd((double const *)(inbuf + cs_a * 4 + 4));
                ymm1 = _mm256_loadu_pd((double const *)(inbuf + cs_a * 5));
                ymm11 = _mm256_loadu_pd((double const *)(inbuf + cs_a * 5 + 4));

                ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
                ymm5 = _mm256_broadcast_sd((double const *)&zero);
                ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);

                ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
                ymm1 = _mm256_broadcast_sd((double const *)&zero);
                ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

                _mm_storeu_pd((double *)(pbuff + 4), _mm256_extractf128_pd(ymm6,0));
                _mm_storeu_pd((double *)(pbuff + 4 + p_lda), _mm256_extractf128_pd(ymm7,0));
                _mm_storeu_pd((double *)(pbuff + 4 + p_lda*2), _mm256_extractf128_pd(ymm8,0));
                _mm_storeu_pd((double *)(pbuff + 4 + p_lda*3), _mm256_extractf128_pd(ymm9,0));

                ymm4 = _mm256_unpacklo_pd(ymm10, ymm11);
                ymm5 = _mm256_broadcast_sd((double const *)&zero);

                ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                ymm8 = _mm256_broadcast_sd((double const *)&zero);
                ymm0 = _mm256_unpackhi_pd(ymm10, ymm11);
                ymm1 = _mm256_broadcast_sd((double const *)&zero);
                ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                ymm9 = _mm256_broadcast_sd((double const *)&zero);

                _mm_storeu_pd((double *)(pbuff + p_lda * 4 + 4), _mm256_extractf128_pd(ymm6,0));
                _mm_storeu_pd((double *)(pbuff + p_lda * 5 + 4), _mm256_extractf128_pd(ymm7,0));
                inbuf += mr*cs_a;
                pbuff += mr;
            }
        }else{
            for(dim_t i=0; i<(size>>2); i++)
            {
                ymm0 = _mm256_loadu_pd((double const *)(inbuf + cs_a * 0 ));
                _mm256_storeu_pd((double *)(pbuff + p_lda * 0), ymm0);
                ymm1 = _mm256_loadu_pd((double const *)(inbuf + cs_a * 1 ));
                _mm256_storeu_pd((double *)(pbuff + p_lda * 1), ymm1);
                ymm2 = _mm256_loadu_pd((double const *)(inbuf + cs_a * 2));
                _mm256_storeu_pd((double *)(pbuff + p_lda * 2), ymm2);
                ymm3 = _mm256_loadu_pd((double const *)(inbuf + cs_a * 3 ));
                _mm256_storeu_pd((double *)(pbuff + p_lda * 3), ymm3);
                ymm0 = _mm256_loadu_pd((double const *)(inbuf + cs_a * 4 ));
                _mm256_storeu_pd((double *)(pbuff + p_lda * 4), ymm0);
                ymm1 = _mm256_loadu_pd((double const *)(inbuf + cs_a * 5));
                _mm256_storeu_pd((double *)(pbuff + p_lda * 5), ymm1);
                inbuf += 4;
                pbuff += 4;
            }

            if(size & 0x3)
            {
                xmm0 = _mm_loadu_pd((double const *)(inbuf + cs_a * 0));
                _mm_storeu_pd((double *)(pbuff + p_lda * 0 ), xmm0);
                xmm1 = _mm_loadu_pd((double const *)(inbuf + cs_a * 1));
                _mm_storeu_pd((double *)(pbuff + p_lda * 1), xmm1);
                xmm2 = _mm_loadu_pd((double const *)(inbuf + cs_a * 2));
                _mm_storeu_pd((double *)(pbuff + p_lda * 2), xmm2);
                xmm3 = _mm_loadu_pd((double const *)(inbuf + cs_a * 3));
                _mm_storeu_pd((double *)(pbuff + p_lda * 3), xmm3);
                xmm0 = _mm_loadu_pd((double const *)(inbuf + cs_a * 4));
                _mm_storeu_pd((double *)(pbuff + p_lda * 4), xmm0);
                xmm1 = _mm_loadu_pd((double const *)(inbuf + cs_a * 5));
                _mm_storeu_pd((double *)(pbuff + p_lda * 5), xmm1);
            }
        }
    }
}
/*
    Pack diagonal elements of A block (8 or 6) into an array
    a. This helps in utilze cache line efficiently in TRSM operation
    b. store ones when input is unit diagonal
*/
BLIS_INLINE void dtrsm_small_pack_diag_element
(
    bool is_unitdiag,
    double *a11,
    dim_t cs_a,
    double *d11_pack,
    dim_t size
)
{
    __m256d ymm0, ymm1, ymm2, ymm3;
    __m256d ymm4, ymm5;
    double ones = 1.0;
    bool is_eight = (size==8) ? 1 : 0;
    ymm4 = ymm5 = _mm256_broadcast_sd((double const *)&ones);
    if(!is_unitdiag)
    {
        //broadcast diagonal elements of A11
        ymm0 = _mm256_broadcast_sd((double const *)(a11));
        ymm1 = _mm256_broadcast_sd((double const *)(a11+ cs_a +1));
        ymm2 = _mm256_broadcast_sd((double const *)(a11+ cs_a*2 + 2));
        ymm3 = _mm256_broadcast_sd((double const *)(a11+ cs_a*3 + 3));

        //Pick one element each column and create a 4 element vector and store
        ymm0 = _mm256_unpacklo_pd(ymm0, ymm1);
        ymm1 = _mm256_unpacklo_pd(ymm2, ymm3);
        ymm1 = _mm256_blend_pd(ymm0, ymm1, 0x0C);

        #ifdef BLIS_DISABLE_TRSM_PREINVERSION
        ymm4 = ymm1;
        #endif
        #ifdef BLIS_ENABLE_TRSM_PREINVERSION
        ymm4 = _mm256_div_pd(ymm4, ymm1);
        #endif

        //broadcast diagonal elements of A11
        ymm0 = _mm256_broadcast_sd((double const *)(a11 + 4 + cs_a*4));
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5 + cs_a*5));
        //Pick one element each column and create a 4 element vector and store
        ymm0 = _mm256_unpacklo_pd(ymm0, ymm1);
        if(is_eight) {
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 6 + cs_a*6));
            ymm3 = _mm256_broadcast_sd((double const *)(a11 + 7 + cs_a*7));
            ymm1 = _mm256_unpacklo_pd(ymm2, ymm3);
        }
        ymm1 = _mm256_blend_pd(ymm0, ymm1, 0x0C);

        #ifdef BLIS_DISABLE_TRSM_PREINVERSION
        ymm5 = ymm1;
        #endif

        #ifdef BLIS_ENABLE_TRSM_PREINVERSION
        ymm5 = _mm256_div_pd(ymm5, ymm1);
        #endif
    }
    _mm256_store_pd((double *)(d11_pack), ymm4);
    if(is_eight){
        _mm256_store_pd((double *)(d11_pack + 4), ymm5);
    }else{
        _mm_storeu_pd((double *)(d11_pack + 4), _mm256_extractf128_pd(ymm5,0));
    }
}

/*
 * Kernels Table
*/
trsmsmall_ker_ft ker_fps[8] =
{
    bli_dtrsm_small_AutXB_AlXB,
    bli_dtrsm_small_AltXB_AuXB,
    bli_dtrsm_small_AltXB_AuXB,
    bli_dtrsm_small_AutXB_AlXB,
    bli_dtrsm_small_XAutB_XAlB,
    bli_dtrsm_small_XAltB_XAuB,
    bli_dtrsm_small_XAltB_XAuB,
    bli_dtrsm_small_XAutB_XAlB
};

/*
* The bli_trsm_small implements a version of TRSM where A is packed and reused
*
* Input:  A: MxM (triangular matrix)
*         B: MxN matrix
* Output: X: MxN matrix such that
             AX = alpha*B or XA = alpha*B or A'X = alpha*B or XA' = alpha*B
* Here the output X is stored in B
*
* Note: Currently only dtrsm is supported when A & B are column-major
*/
err_t bli_trsm_small
(
    side_t  side,
    obj_t*  alpha,
    obj_t*  a,
    obj_t*  b,
    cntx_t* cntx,
    cntl_t* cntl
)
{
    err_t err;
    dim_t m = bli_obj_length(b);
    dim_t n = bli_obj_width(b);

    if(!(m && n)) {
        return BLIS_SUCCESS;
    }

    bool uplo = bli_obj_is_upper(a);
    bool transa = bli_obj_has_trans(a);

    /* ToDo: Temporary threshold condition for trsm single thread.
    *  It will be updated with arch based threshold function which reads
    *  tunned thresholds for all 64 (datatype,side,uplo,transa,unit,) trsm
       combinations. We arrived to this condition based on performance
       comparsion with only available native path
    */
    if(m > 1000 || n > 1000) {
        return BLIS_NOT_YET_IMPLEMENTED;
    }

    /* If alpha is zero, B matrix will become zero after scaling
       hence solution is also zero matrix */
    if (bli_obj_equals(alpha, &BLIS_ZERO)) {
        return BLIS_NOT_YET_IMPLEMENTED; // scale B by alpha
    }

    // Return if inputs are row major as currently
    // we are supporing col major only
    if ((bli_obj_row_stride(a) != 1) ||
        (bli_obj_row_stride(b) != 1)) {
        return BLIS_INVALID_ROW_STRIDE;
    }

    //Curretnly optimized for double data type only
    num_t dt = bli_obj_dt(a);
    if (dt != BLIS_DOUBLE) {
        return BLIS_NOT_YET_IMPLEMENTED;
    }

    // A is expected to be triangular in trsm
    if (!bli_obj_is_upper_or_lower (a)) {
        return BLIS_EXPECTED_TRIANGULAR_OBJECT;
    }

    /*
     *  Compose kernel index based on inputs
    */

    dim_t keridx = ( (( side & 0x1) << 2) |
                     (( uplo & 0x1) << 1) |
                     ( transa & 0x1) );


    trsmsmall_ker_ft ker_fp = ker_fps[ keridx ];

    /*Call the kernel*/
    err = ker_fp
          (
              alpha,
              a,
              b,
              cntx,
              cntl
          );

    return err;
};

/*implements TRSM for the case XA = alpha * B
 *A is lower triangular, non-unit diagonal/unit diagonal, transpose
 *dimensions: X:mxn     A:nxn       B: mxn
 *
 *   b11--->         a01 ---->
    *****************   ***********
    *b01*b11*   *   *   * *    *  *
b11 *   *   *   *   *    **a01 *  * a11
 |  *****************     *********  |
 |  *   *   *   *   *      *a11*  *  |
 |  *   *   *   *   *       *  *  *  |
 v  *****************        ******  v
    *   *   *   *   *         *   *
    *   *   *   *   *          *  *
    *****************           * *
                                  *
 *implements TRSM for the case XA = alpha * B
 *A is upper triangular, non-unit diagonal/unit diagonal, no transpose
 *dimensions: X:mxn     A:nxn       B: mxn
 *
 *   b11--->         a01 ---->
    *****************   ***********
    *b01*b11*   *   *   * *    *  *
b11 *   *   *   *   *    **a01 *  * a11
 |  *****************     *********  |
 |  *   *   *   *   *      *a11*  *  |
 |  *   *   *   *   *       *  *  *  |
 v  *****************        ******  v
    *   *   *   *   *         *   *
    *   *   *   *   *          *  *
    *****************           * *
                                  *

*/

BLIS_INLINE  err_t bli_dtrsm_small_XAltB_XAuB
(
    obj_t* AlphaObj,
    obj_t* a,
    obj_t* b,
    cntx_t* cntx,
    cntl_t* cntl
)
{
    dim_t m = bli_obj_length(b);  //number of rows
    dim_t n = bli_obj_width(b);   //number of columns
    dim_t d_mr = 8,d_nr = 6;

    bool transa = bli_obj_has_trans(a);
    dim_t cs_a, rs_a;

    // Swap rs_a & cs_a in case of non-tranpose.
    if(transa)
    {
        cs_a = bli_obj_col_stride(a); // column stride of A
        rs_a = bli_obj_row_stride(a); // row stride of A
    }
    else
    {
        cs_a = bli_obj_row_stride(a); // row stride of A
        rs_a = bli_obj_col_stride(a); // column stride of A
    }
    dim_t cs_b = bli_obj_col_stride(b); //column stride of matrix B

    dim_t i, j, k;        //loop variablse
    dim_t k_iter;         //determines the number of GEMM operations to be done

    double ones = 1.0;
    double zero = 0.0;
    bool is_unitdiag = bli_obj_has_unit_diag(a);

    double AlphaVal = *(double *)AlphaObj->buffer;    //value of Alpha
    double* restrict L = a->buffer;      //pointer to matrix A
    double* restrict B = b->buffer;      //pointer to matrix B

    double *a01, *a11, *b10, *b11;   //pointers for GEMM and TRSM blocks

    gint_t required_packing_A = 1;
    mem_t local_mem_buf_A_s = {0};
    double *D_A_pack = NULL;
    double d11_pack[d_mr] __attribute__((aligned(64)));
    rntm_t rntm;

    bli_rntm_init_from_global( &rntm );
    bli_rntm_set_num_threads_only( 1, &rntm );
    bli_membrk_rntm_set_membrk( &rntm );

    siz_t buffer_size = bli_pool_block_size(
                          bli_membrk_pool(
                            bli_packbuf_index(BLIS_BITVAL_BUFFER_FOR_A_BLOCK),
                            bli_rntm_membrk(&rntm)));

    if( (d_nr * n * sizeof(double)) > buffer_size)
        return BLIS_NOT_YET_IMPLEMENTED;

    if (required_packing_A == 1)
    {
        // Get the buffer from the pool.
        bli_membrk_acquire_m(&rntm,
                             buffer_size,
                             BLIS_BITVAL_BUFFER_FOR_A_BLOCK,
                             &local_mem_buf_A_s);
        if(FALSE==bli_mem_is_alloc(&local_mem_buf_A_s)) return BLIS_NULL_POINTER;
        D_A_pack = bli_mem_buffer(&local_mem_buf_A_s);
        if(NULL==D_A_pack) return BLIS_NULL_POINTER;
    }

    //ymm scratch reginsters
    __m256d ymm0, ymm1, ymm2, ymm3;
    __m256d ymm4, ymm5, ymm6, ymm7;
    __m256d ymm8, ymm9, ymm10, ymm11;
    __m256d ymm12, ymm13, ymm14, ymm15;

    __m128d xmm5;

    /*
    Performs solving TRSM for 6 rows at a time from  0 to n/6 in steps of d_nr
    a. Load and pack A (a01 block), the size of packing 6x6 to 6x (n-6)
       First there will be no GEMM and no packing of a01 because it is only TRSM
    b. Using packed a01 block and b10 block perform GEMM operation
    c. Use GEMM outputs, perform TRSM operation using a11, b11 and update B
    d. Repeat b for m cols of B in steps of d_mr
    */

    for(j = 0; (j+d_nr-1) < n; j += d_nr)     //loop along 'N' direction
    {
        a01 = L + j*rs_a;                     //pointer to block of A to be used in GEMM
        a11 = L + j*cs_a + j*rs_a;                 //pointer to block of A to be used for TRSM

        //double *ptr_a10_dup = D_A_pack;

        dim_t p_lda = j; // packed leading dimension
        // perform copy of A to packed buffer D_A_pack

        if(transa)
        {
            /*
            Pack current A block (a01) into packed buffer memory D_A_pack
            a. This a10 block is used in GEMM portion only and this
               a01 block size will be increasing by d_nr for every next iteration
               until it reaches 6x(n-6) which is the maximum GEMM alone block size in A
            b. This packed buffer is reused to calculate all m cols of B matrix
            */
            bli_dtrsm_small_pack('R', j, 1, a01, cs_a, D_A_pack, p_lda,d_nr);

            /*
               Pack 6 diagonal elements of A block into an array
               a. This helps in utilze cache line efficiently in TRSM operation
               b. store ones when input is unit diagonal
            */

            dtrsm_small_pack_diag_element(is_unitdiag,a11,cs_a,d11_pack,d_nr);
        }
        else
        {
            bli_dtrsm_small_pack('R', j, 0, a01, rs_a, D_A_pack, p_lda,d_nr);
            dtrsm_small_pack_diag_element(is_unitdiag,a11,rs_a,d11_pack,d_nr);
        }

        /*
        a. Perform GEMM using a01, b10.
        b. Perform TRSM on a11, b11
        c. This loop GEMM+TRSM loops operates with 8x6 block size
           along m dimension for every d_mr columns of B10 where
           packed A buffer is reused in computing all m cols of B.
        d. Same approach is used in remaining fringe cases.
        */
        for(i = 0; (i+d_mr-1) < m; i += d_mr)     //loop along 'M' direction
        {
            a01 = D_A_pack;
            a11 = L + j*cs_a + j*rs_a;                 //pointer to block of A to be used for TRSM
            b10 = B + i;                          //pointer to block of B to be used in GEMM
            b11 = B + i + j*cs_b;                 //pointer to block of B to be used for TRSM

            k_iter = j;                    //number of GEMM operations to be done(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            /*
            Peform GEMM between a01 and b10 blocks
            For first itteration there will be no GEMM operation
            where k_iter are zero
            */
            BLIS_DTRSM_SMALL_GEMM_6nx8m(a01,b10,cs_b,p_lda,k_iter)

            /*
            Load b11 of size 8x6 and multiply with alpha
            Add the GEMM output to b11
            and peform TRSM operation.
            */

            BLIS_PRE_DTRSM_SMALL_6x8(AlphaVal,b11,cs_b)

            ///implement TRSM///

            /*
            Compute 6x8 TRSM block by using GEMM block output in register
            a. The 6x8 input (gemm outputs) are stored in combinations of ymm registers
                1. ymm3, ymm4 2. ymm5, ymm6 3. ymm7, ymm8, 4. ymm9, ymm10
                5. ymm11, ymm12 6. ymm13,ymm14
            b. Towards the end TRSM output will be stored back into b11
            */

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));

            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);
            ymm4 = DTRSM_SMALL_DIV_OR_SCALE(ymm4, ymm0);

            //extract a11
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(row 1):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 1*rs_a));

            ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);
            ymm6 = _mm256_fnmadd_pd(ymm1, ymm4, ymm6);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));

            ymm7 = _mm256_fnmadd_pd(ymm1, ymm3, ymm7);
            ymm8 = _mm256_fnmadd_pd(ymm1, ymm4, ymm8);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));

            ymm9 = _mm256_fnmadd_pd(ymm1, ymm3, ymm9);
            ymm10 = _mm256_fnmadd_pd(ymm1, ymm4, ymm10);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*rs_a));

            ymm11 = _mm256_fnmadd_pd(ymm1, ymm3, ymm11);
            ymm12 = _mm256_fnmadd_pd(ymm1, ymm4, ymm12);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*rs_a));

            ymm13 = _mm256_fnmadd_pd(ymm1, ymm3, ymm13);
            ymm14 = _mm256_fnmadd_pd(ymm1, ymm4, ymm14);

            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);
            ymm6 = DTRSM_SMALL_DIV_OR_SCALE(ymm6, ymm0);

            a11 += cs_a;

            //extract a22
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            //(row 2):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));

            ymm7 = _mm256_fnmadd_pd(ymm1, ymm5, ymm7);
            ymm8 = _mm256_fnmadd_pd(ymm1, ymm6, ymm8);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));

            ymm9 = _mm256_fnmadd_pd(ymm1, ymm5, ymm9);
            ymm10 = _mm256_fnmadd_pd(ymm1, ymm6, ymm10);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*rs_a));

            ymm11 = _mm256_fnmadd_pd(ymm1, ymm5, ymm11);
            ymm12 = _mm256_fnmadd_pd(ymm1, ymm6, ymm12);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*rs_a));

            ymm13 = _mm256_fnmadd_pd(ymm1, ymm5, ymm13);
            ymm14 = _mm256_fnmadd_pd(ymm1, ymm6, ymm14);

            ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);
            ymm8 = DTRSM_SMALL_DIV_OR_SCALE(ymm8, ymm0);

            a11 += cs_a;

            //extract a33
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

            //(Row 3): FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));

            ymm9 = _mm256_fnmadd_pd(ymm1, ymm7, ymm9);
            ymm10 = _mm256_fnmadd_pd(ymm1, ymm8, ymm10);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*rs_a));

            ymm11 = _mm256_fnmadd_pd(ymm1, ymm7, ymm11);
            ymm12 = _mm256_fnmadd_pd(ymm1, ymm8, ymm12);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*rs_a));

            ymm13 = _mm256_fnmadd_pd(ymm1, ymm7, ymm13);
            ymm14 = _mm256_fnmadd_pd(ymm1, ymm8, ymm14);

            ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);
            ymm10 = DTRSM_SMALL_DIV_OR_SCALE(ymm10, ymm0);

            a11 += cs_a;

            //extract a44
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 4));

            //(row 4):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*rs_a));

            ymm11 = _mm256_fnmadd_pd(ymm1, ymm9, ymm11);
            ymm12 = _mm256_fnmadd_pd(ymm1, ymm10, ymm12);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*rs_a));

            ymm13 = _mm256_fnmadd_pd(ymm1, ymm9, ymm13);
            ymm14 = _mm256_fnmadd_pd(ymm1, ymm10, ymm14);

            ymm11 = DTRSM_SMALL_DIV_OR_SCALE(ymm11, ymm0);
            ymm12 = DTRSM_SMALL_DIV_OR_SCALE(ymm12, ymm0);

            a11 += cs_a;

            //extract a55
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 5));

            //(Row 5): FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*rs_a));

            ymm13 = _mm256_fnmadd_pd(ymm1, ymm11, ymm13);
            ymm14 = _mm256_fnmadd_pd(ymm1, ymm12, ymm14);

            ymm13 = DTRSM_SMALL_DIV_OR_SCALE(ymm13, ymm0);
            ymm14 = DTRSM_SMALL_DIV_OR_SCALE(ymm14, ymm0);

            _mm256_storeu_pd((double *)b11, ymm3);
            _mm256_storeu_pd((double *)(b11 + 4), ymm4);
            _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
            _mm256_storeu_pd((double *)(b11 + cs_b + 4), ymm6);
            _mm256_storeu_pd((double *)(b11 + cs_b*2), ymm7);
            _mm256_storeu_pd((double *)(b11 + cs_b*2 + 4), ymm8);
            _mm256_storeu_pd((double *)(b11 + cs_b*3), ymm9);
            _mm256_storeu_pd((double *)(b11 + cs_b*3 + 4), ymm10);
            _mm256_storeu_pd((double *)(b11 + cs_b*4), ymm11);
            _mm256_storeu_pd((double *)(b11 + cs_b*4 + 4), ymm12);
            _mm256_storeu_pd((double *)(b11 + cs_b*5), ymm13);
            _mm256_storeu_pd((double *)(b11 + cs_b*5 + 4), ymm14);
        }

        dim_t m_remainder = m - i;
        if(m_remainder >= 4)
        {
            a01 = D_A_pack;
            a11 = L + j*cs_a + j*rs_a;                 //pointer to block of A to be used for TRSM
            b10 = B + i;                          //pointer to block of B to be used in GEMM
            b11 = B + i + j*cs_b;                 //pointer to block of B to be used for TRSM

            k_iter = j;                    //number of GEMM operations to be done(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_6nx4m(a01,b10,cs_b,p_lda,k_iter)

            // Load b11 of size 4x6 and multiply with alpha
            BLIS_PRE_DTRSM_SMALL_6x4(AlphaVal,b11,cs_b)

            ///implement TRSM///

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

            //extract a11
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(row 1):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 1*rs_a));
            ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));
            ymm7 = _mm256_fnmadd_pd(ymm1, ymm3, ymm7);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm1, ymm3, ymm9);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*rs_a));
            ymm11 = _mm256_fnmadd_pd(ymm1, ymm3, ymm11);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*rs_a));
            ymm13 = _mm256_fnmadd_pd(ymm1, ymm3, ymm13);

            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

            a11 += cs_a;

            //extract a22
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            //(row 2):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));
            ymm7 = _mm256_fnmadd_pd(ymm1, ymm5, ymm7);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm1, ymm5, ymm9);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*rs_a));
            ymm11 = _mm256_fnmadd_pd(ymm1, ymm5, ymm11);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*rs_a));
            ymm13 = _mm256_fnmadd_pd(ymm1, ymm5, ymm13);

            ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

            a11 += cs_a;

            //extract a33
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

            //(Row 3): FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm1, ymm7, ymm9);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*rs_a));
            ymm11 = _mm256_fnmadd_pd(ymm1, ymm7, ymm11);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*rs_a));
            ymm13 = _mm256_fnmadd_pd(ymm1, ymm7, ymm13);

            ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);

            a11 += cs_a;

            //extract a44
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 4));

            //(row 4):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*rs_a));
            ymm11 = _mm256_fnmadd_pd(ymm1, ymm9, ymm11);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*rs_a));
            ymm13 = _mm256_fnmadd_pd(ymm1, ymm9, ymm13);

            ymm11 = DTRSM_SMALL_DIV_OR_SCALE(ymm11, ymm0);

            a11 += cs_a;

            //extract a55
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 5));

            //(Row 5): FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*rs_a));
            ymm13 = _mm256_fnmadd_pd(ymm1, ymm11, ymm13);

            ymm13 = DTRSM_SMALL_DIV_OR_SCALE(ymm13, ymm0);

            _mm256_storeu_pd((double *)b11, ymm3);
            _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
            _mm256_storeu_pd((double *)(b11 + cs_b*2), ymm7);
            _mm256_storeu_pd((double *)(b11 + cs_b*3), ymm9);
            _mm256_storeu_pd((double *)(b11 + cs_b*4), ymm11);
            _mm256_storeu_pd((double *)(b11 + cs_b*5), ymm13);

            m_remainder -= 4;
            i += 4;
        }

        if(m_remainder == 3)
        {
            a01 = D_A_pack;
            a11 = L + j*cs_a + j*rs_a;                 //pointer to block of A to be used for TRSM
            b10 = B + i;                          //pointer to block of B to be used in GEMM
            b11 = B + i + j*cs_b;                 //pointer to block of B to be used for TRSM

            k_iter = j;                    //number of GEMM operations to be done(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_6nx4m(a01,b10,cs_b,p_lda,k_iter)

            // Load b11 of size 4x6 and multiply with alpha
            BLIS_PRE_DTRSM_SMALL_6x4(AlphaVal,b11,cs_b)

            ///implement TRSM///

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

            //extract a11
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(row 1):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 1*rs_a));
            ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));
            ymm7 = _mm256_fnmadd_pd(ymm1, ymm3, ymm7);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm1, ymm3, ymm9);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*rs_a));
            ymm11 = _mm256_fnmadd_pd(ymm1, ymm3, ymm11);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*rs_a));
            ymm13 = _mm256_fnmadd_pd(ymm1, ymm3, ymm13);

            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

            a11 += cs_a;

            //extract a22
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            //(row 2):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));
            ymm7 = _mm256_fnmadd_pd(ymm1, ymm5, ymm7);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm1, ymm5, ymm9);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*rs_a));
            ymm11 = _mm256_fnmadd_pd(ymm1, ymm5, ymm11);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*rs_a));
            ymm13 = _mm256_fnmadd_pd(ymm1, ymm5, ymm13);

            ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

            a11 += cs_a;

            //extract a33
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

            //(Row 3): FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm1, ymm7, ymm9);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*rs_a));
            ymm11 = _mm256_fnmadd_pd(ymm1, ymm7, ymm11);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*rs_a));
            ymm13 = _mm256_fnmadd_pd(ymm1, ymm7, ymm13);

            ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);

            a11 += cs_a;

            //extract a44
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 4));

            //(row 4):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*rs_a));
            ymm11 = _mm256_fnmadd_pd(ymm1, ymm9, ymm11);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*rs_a));
            ymm13 = _mm256_fnmadd_pd(ymm1, ymm9, ymm13);

            ymm11 = DTRSM_SMALL_DIV_OR_SCALE(ymm11, ymm0);

            a11 += cs_a;

            //extract a55
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 5));

            //(Row 5): FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*rs_a));
            ymm13 = _mm256_fnmadd_pd(ymm1, ymm11, ymm13);

            ymm13 = DTRSM_SMALL_DIV_OR_SCALE(ymm13, ymm0);

            ymm0 = _mm256_loadu_pd((double const *)b11);
            ymm3 = _mm256_blend_pd(ymm0, ymm3, 0x07);
            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));      //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm5 = _mm256_blend_pd(ymm0, ymm5, 0x07);
            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*2));    //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm7 = _mm256_blend_pd(ymm0, ymm7, 0x07);
            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*3));    //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm9 = _mm256_blend_pd(ymm0, ymm9, 0x07);
            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*4));    //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm11 = _mm256_blend_pd(ymm0, ymm11, 0x07);
            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*5));    //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm13 = _mm256_blend_pd(ymm0, ymm13, 0x07);

            _mm256_storeu_pd((double *)b11, ymm3);
            _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
            _mm256_storeu_pd((double *)(b11 + cs_b*2), ymm7);
            _mm256_storeu_pd((double *)(b11 + cs_b*3), ymm9);
            _mm256_storeu_pd((double *)(b11 + cs_b*4), ymm11);
            _mm256_storeu_pd((double *)(b11 + cs_b*5), ymm13);

            m_remainder -= 3;
            i += 3;
        }
        else if(m_remainder == 2)
        {
            a01 = D_A_pack;
            a11 = L + j*cs_a + j*rs_a;          //pointer to block of A to be used for TRSM
            b10 = B + i;                   //pointer to block of B to be used in GEMM
            b11 = B + i + j*cs_b;          //pointer to block of B to be used for TRSM

            k_iter = j;                    //number of GEMM operations to be done(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_6nx4m(a01,b10,cs_b,p_lda,k_iter)

            // Load b11 of size 4x6 and multiply with alpha
            BLIS_PRE_DTRSM_SMALL_6x4(AlphaVal,b11,cs_b)

            ///implement TRSM///

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

            //extract a11
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(row 1):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 1*rs_a));
            ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));
            ymm7 = _mm256_fnmadd_pd(ymm1, ymm3, ymm7);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm1, ymm3, ymm9);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*rs_a));
            ymm11 = _mm256_fnmadd_pd(ymm1, ymm3, ymm11);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*rs_a));
            ymm13 = _mm256_fnmadd_pd(ymm1, ymm3, ymm13);

            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

            a11 += cs_a;

            //extract a22
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            //(row 2):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));
            ymm7 = _mm256_fnmadd_pd(ymm1, ymm5, ymm7);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm1, ymm5, ymm9);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*rs_a));
            ymm11 = _mm256_fnmadd_pd(ymm1, ymm5, ymm11);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*rs_a));
            ymm13 = _mm256_fnmadd_pd(ymm1, ymm5, ymm13);

            ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

            a11 += cs_a;

            //extract a33
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

            //(Row 3): FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm1, ymm7, ymm9);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*rs_a));
            ymm11 = _mm256_fnmadd_pd(ymm1, ymm7, ymm11);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*rs_a));
            ymm13 = _mm256_fnmadd_pd(ymm1, ymm7, ymm13);

            ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);

            a11 += cs_a;

            //extract a44
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 4));

            //(row 4):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*rs_a));
            ymm11 = _mm256_fnmadd_pd(ymm1, ymm9, ymm11);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*rs_a));
            ymm13 = _mm256_fnmadd_pd(ymm1, ymm9, ymm13);

            ymm11 = DTRSM_SMALL_DIV_OR_SCALE(ymm11, ymm0);

            a11 += cs_a;

            //extract a55
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 5));

            //(Row 5): FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*rs_a));
            ymm13 = _mm256_fnmadd_pd(ymm1, ymm11, ymm13);

            ymm13 = DTRSM_SMALL_DIV_OR_SCALE(ymm13, ymm0);

            ymm0 = _mm256_loadu_pd((double const *)b11);
            ymm3 = _mm256_blend_pd(ymm0, ymm3, 0x03);
            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm5 = _mm256_blend_pd(ymm0, ymm5, 0x03);
            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*2));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm7 = _mm256_blend_pd(ymm0, ymm7, 0x03);
            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*3));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm9 = _mm256_blend_pd(ymm0, ymm9, 0x03);
            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*4));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm11 = _mm256_blend_pd(ymm0, ymm11, 0x03);
            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*5));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm13 = _mm256_blend_pd(ymm0, ymm13, 0x03);

            _mm256_storeu_pd((double *)b11, ymm3);
            _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
            _mm256_storeu_pd((double *)(b11 + cs_b*2), ymm7);
            _mm256_storeu_pd((double *)(b11 + cs_b*3), ymm9);
            _mm256_storeu_pd((double *)(b11 + cs_b*4), ymm11);
            _mm256_storeu_pd((double *)(b11 + cs_b*5), ymm13);

            m_remainder -= 2;
            i += 2;
        }
        else if(m_remainder == 1)
        {
            a01 = D_A_pack;
            a11 = L + j*cs_a + j*rs_a;                 //pointer to block of A to be used for TRSM
            b10 = B + i;                          //pointer to block of B to be used in GEMM
            b11 = B + i + j*cs_b;                 //pointer to block of B to be used for TRSM

            k_iter = j;                    //number of GEMM operations to be done(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_6nx4m(a01,b10,cs_b,p_lda,k_iter)

            // Load b11 of size 4x6 and multiply with alpha
            BLIS_PRE_DTRSM_SMALL_6x4(AlphaVal,b11,cs_b)

            ///implement TRSM///

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

            //extract a11
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(row 1):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 1*rs_a));
            ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));
            ymm7 = _mm256_fnmadd_pd(ymm1, ymm3, ymm7);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm1, ymm3, ymm9);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*rs_a));
            ymm11 = _mm256_fnmadd_pd(ymm1, ymm3, ymm11);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*rs_a));
            ymm13 = _mm256_fnmadd_pd(ymm1, ymm3, ymm13);

            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

            a11 += cs_a;

            //extract a22
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            //(row 2):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));
            ymm7 = _mm256_fnmadd_pd(ymm1, ymm5, ymm7);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm1, ymm5, ymm9);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*rs_a));
            ymm11 = _mm256_fnmadd_pd(ymm1, ymm5, ymm11);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*rs_a));
            ymm13 = _mm256_fnmadd_pd(ymm1, ymm5, ymm13);

            ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

            a11 += cs_a;

            //extract a33
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

            //(Row 3): FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm1, ymm7, ymm9);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*rs_a));
            ymm11 = _mm256_fnmadd_pd(ymm1, ymm7, ymm11);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*rs_a));
            ymm13 = _mm256_fnmadd_pd(ymm1, ymm7, ymm13);

            ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);

            a11 += cs_a;

            //extract a44
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 4));

            //(row 4):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*rs_a));
            ymm11 = _mm256_fnmadd_pd(ymm1, ymm9, ymm11);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*rs_a));
            ymm13 = _mm256_fnmadd_pd(ymm1, ymm9, ymm13);

            ymm11 = DTRSM_SMALL_DIV_OR_SCALE(ymm11, ymm0);

            a11 += cs_a;

            //extract a55
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 5));

            //(Row 5): FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*rs_a));
            ymm13 = _mm256_fnmadd_pd(ymm1, ymm11, ymm13);

            ymm13 = DTRSM_SMALL_DIV_OR_SCALE(ymm13, ymm0);

            ymm0 = _mm256_loadu_pd((double const *)b11);
            ymm3 = _mm256_blend_pd(ymm0, ymm3, 0x01);
            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm5 = _mm256_blend_pd(ymm0, ymm5, 0x01);
            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*2));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm7 = _mm256_blend_pd(ymm0, ymm7, 0x01);
            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*3));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm9 = _mm256_blend_pd(ymm0, ymm9, 0x01);
            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*4));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm11 = _mm256_blend_pd(ymm0, ymm11, 0x01);
            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*5));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm13 = _mm256_blend_pd(ymm0, ymm13, 0x01);

            _mm256_storeu_pd((double *)b11, ymm3);
            _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
            _mm256_storeu_pd((double *)(b11 + cs_b*2), ymm7);
            _mm256_storeu_pd((double *)(b11 + cs_b*3), ymm9);
            _mm256_storeu_pd((double *)(b11 + cs_b*4), ymm11);
            _mm256_storeu_pd((double *)(b11 + cs_b*5), ymm13);

            m_remainder -= 1;
            i += 1;
        }
    }

    dim_t n_remainder = n - j;

    /*
    Reminder cases starts here:
    a. Similar logic and code flow used in computing full block (6x8)
       above holds for reminder cases too.
    */

    if(n_remainder >= 4)
    {
        a01 = L + j*rs_a;                     //pointer to block of A to be used in GEMM
        a11 = L + j*cs_a + j*rs_a;                 //pointer to block of A to be used for TRSM

        double *ptr_a10_dup = D_A_pack;

        dim_t p_lda = j; // packed leading dimension
        // perform copy of A to packed buffer D_A_pack

        if(transa)
        {
            for(dim_t x =0;x < p_lda;x+=d_nr)
            {
                ymm0 = _mm256_loadu_pd((double const *)(a01));
                ymm1 = _mm256_loadu_pd((double const *)(a01 + cs_a));
                ymm2 = _mm256_loadu_pd((double const *)(a01 + cs_a * 2));
                ymm3 = _mm256_loadu_pd((double const *)(a01 + cs_a * 3));

                ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
                ymm5 = _mm256_unpacklo_pd(ymm2, ymm3);

                ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);

                ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
                ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);

                ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

                _mm256_storeu_pd((double *)(ptr_a10_dup), ymm6);
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda), ymm7);
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*2), ymm8);
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*3), ymm9);

                ymm0 = _mm256_loadu_pd((double const *)(a01 + cs_a * 4));
                ymm1 = _mm256_loadu_pd((double const *)(a01 + cs_a * 5));

                ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
                ymm5 = _mm256_broadcast_sd((double const *)&zero);

                ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);

                ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
                ymm1 = _mm256_broadcast_sd((double const *)&zero);

                ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

                _mm_storeu_pd((double *)(ptr_a10_dup + 4), _mm256_extractf128_pd(ymm6,0));
                _mm_storeu_pd((double *)(ptr_a10_dup + 4 + p_lda), _mm256_extractf128_pd(ymm7,0));
                _mm_storeu_pd((double *)(ptr_a10_dup + 4 + p_lda*2), _mm256_extractf128_pd(ymm8,0));
                _mm_storeu_pd((double *)(ptr_a10_dup + 4 + p_lda*3), _mm256_extractf128_pd(ymm9,0));

                a01 += d_nr*cs_a;
                ptr_a10_dup += d_nr;
            }
        }
        else
        {
            dim_t loop_count = p_lda/4;

            for(dim_t x =0;x < loop_count;x++)
            {
                ymm15 = _mm256_loadu_pd((double const *)(a01 + rs_a * 0 + x*4));
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda * 0 + x*4), ymm15);
                ymm15 = _mm256_loadu_pd((double const *)(a01 + rs_a * 1 + x*4));
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda * 1 + x*4), ymm15);
                ymm15 = _mm256_loadu_pd((double const *)(a01 + rs_a * 2 + x*4));
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda * 2 + x*4), ymm15);
                ymm15 = _mm256_loadu_pd((double const *)(a01 + rs_a * 3 + x*4));
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda * 3 + x*4), ymm15);
            }

            dim_t remainder_loop_count = p_lda - loop_count*4;

            __m128d xmm0;
            if(remainder_loop_count != 0)
            {
                xmm0 = _mm_loadu_pd((double const *)(a01 + rs_a * 0 + loop_count*4));
                _mm_storeu_pd((double *)(ptr_a10_dup + p_lda * 0 + loop_count*4), xmm0);
                xmm0 = _mm_loadu_pd((double const *)(a01 + rs_a * 1 + loop_count*4));
                _mm_storeu_pd((double *)(ptr_a10_dup + p_lda * 1 + loop_count*4), xmm0);
                xmm0 = _mm_loadu_pd((double const *)(a01 + rs_a * 2 + loop_count*4));
                _mm_storeu_pd((double *)(ptr_a10_dup + p_lda * 2 + loop_count*4), xmm0);
                xmm0 = _mm_loadu_pd((double const *)(a01 + rs_a * 3 + loop_count*4));
                _mm_storeu_pd((double *)(ptr_a10_dup + p_lda * 3 + loop_count*4), xmm0);
            }
        }

        ymm4 = _mm256_broadcast_sd((double const *)&ones);
        if(!is_unitdiag)
        {
            if(transa)
            {
                //broadcast diagonal elements of A11
                ymm0 = _mm256_broadcast_sd((double const *)(a11));
                ymm1 = _mm256_broadcast_sd((double const *)(a11+ cs_a*1 + 1));
                ymm2 = _mm256_broadcast_sd((double const *)(a11+ cs_a*2 + 2));
                ymm3 = _mm256_broadcast_sd((double const *)(a11+ cs_a*3 + 3));
            }
            else
            {
                //broadcast diagonal elements of A11
                ymm0 = _mm256_broadcast_sd((double const *)(a11));
                ymm1 = _mm256_broadcast_sd((double const *)(a11+ rs_a*1 + 1));
                ymm2 = _mm256_broadcast_sd((double const *)(a11+ rs_a*2 + 2));
                ymm3 = _mm256_broadcast_sd((double const *)(a11+ rs_a*3 + 3));
            }

            ymm0 = _mm256_unpacklo_pd(ymm0, ymm1);
            ymm1 = _mm256_unpacklo_pd(ymm2, ymm3);

            ymm1 = _mm256_blend_pd(ymm0, ymm1, 0x0C);
            #ifdef BLIS_DISABLE_TRSM_PREINVERSION
            ymm4 = ymm1;
            #endif
            #ifdef BLIS_ENABLE_TRSM_PREINVERSION
            ymm4 = _mm256_div_pd(ymm4, ymm1);
            #endif
        }
        _mm256_storeu_pd((double *)(d11_pack), ymm4);

        for(i = 0; (i+d_mr-1) < m; i += d_mr)     //loop along 'M' direction
        {
            a01 = D_A_pack;
            a11 = L + j*cs_a + j*rs_a;                 //pointer to block of A to be used for TRSM
            b10 = B + i;                          //pointer to block of B to be used in GEMM
            b11 = B + i + j*cs_b;                 //pointer to block of B to be used for TRSM

            k_iter = j;                    //number of GEMM operations to be done(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_4nx8m(a01,b10,cs_b,p_lda,k_iter)

            BLIS_PRE_DTRSM_SMALL_4x8(AlphaVal,b11,cs_b)

            ///implement TRSM///

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));

            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);
            ymm4 = DTRSM_SMALL_DIV_OR_SCALE(ymm4, ymm0);

            //extract a11
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(row 1):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 1*rs_a));

            ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);
            ymm6 = _mm256_fnmadd_pd(ymm1, ymm4, ymm6);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));

            ymm7 = _mm256_fnmadd_pd(ymm1, ymm3, ymm7);
            ymm8 = _mm256_fnmadd_pd(ymm1, ymm4, ymm8);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));

            ymm9 = _mm256_fnmadd_pd(ymm1, ymm3, ymm9);
            ymm10 = _mm256_fnmadd_pd(ymm1, ymm4, ymm10);

            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);
            ymm6 = DTRSM_SMALL_DIV_OR_SCALE(ymm6, ymm0);

            a11 += cs_a;

            //extract a22
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            //(row 2):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));

            ymm7 = _mm256_fnmadd_pd(ymm1, ymm5, ymm7);
            ymm8 = _mm256_fnmadd_pd(ymm1, ymm6, ymm8);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));

            ymm9 = _mm256_fnmadd_pd(ymm1, ymm5, ymm9);
            ymm10 = _mm256_fnmadd_pd(ymm1, ymm6, ymm10);

            ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);
            ymm8 = DTRSM_SMALL_DIV_OR_SCALE(ymm8, ymm0);

            a11 += cs_a;

            //extract a33
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

            //(Row 3): FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));

            ymm9 = _mm256_fnmadd_pd(ymm1, ymm7, ymm9);
            ymm10 = _mm256_fnmadd_pd(ymm1, ymm8, ymm10);

            ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);
            ymm10 = DTRSM_SMALL_DIV_OR_SCALE(ymm10, ymm0);

            _mm256_storeu_pd((double *)b11, ymm3);
            _mm256_storeu_pd((double *)(b11 + 4), ymm4);
            _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
            _mm256_storeu_pd((double *)(b11 + cs_b + 4), ymm6);
            _mm256_storeu_pd((double *)(b11 + cs_b*2), ymm7);
            _mm256_storeu_pd((double *)(b11 + cs_b*2 + 4), ymm8);
            _mm256_storeu_pd((double *)(b11 + cs_b*3), ymm9);
            _mm256_storeu_pd((double *)(b11 + cs_b*3 + 4), ymm10);
        }

        dim_t m_remainder = m - i;
        if(m_remainder >= 4)
        {
            a01 = D_A_pack;
            a11 = L + j*cs_a + j*rs_a;                 //pointer to block of A to be used for TRSM
            b10 = B + i;                          //pointer to block of B to be used in GEMM
            b11 = B + i + j*cs_b;                 //pointer to block of B to be used for TRSM

            k_iter = j;                    //number of GEMM operations to be done(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_4nx4m(a01,b10,cs_b,p_lda,k_iter)

            ymm15 = _mm256_broadcast_sd((double const *)(&AlphaVal));         //register to hold alpha

            ymm0 = _mm256_loadu_pd((double const *)b11);    //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
            ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);      //B11[0-3][0] * alpha -= ymm0

            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));      //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);    //B11[0-3][1] * alpha-= ymm2

            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*2));    //B11[0][2] B11[1][2] B11[2][2] B11[3][2]
            ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);    //B11[0-3][2] * alpha -= ymm4

            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*3));    //B11[0][3] B11[1][3] B11[2][3] B11[3][3]
            ymm9 = _mm256_fmsub_pd(ymm0, ymm15, ymm9);    //B11[0-3][3] * alpha -= ymm6

            ///implement TRSM///

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

            //extract a11
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(row 1):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 1*rs_a));
            ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));
            ymm7 = _mm256_fnmadd_pd(ymm1, ymm3, ymm7);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm1, ymm3, ymm9);

            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

            a11 += cs_a;

            //extract a22
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            //(row 2):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));
            ymm7 = _mm256_fnmadd_pd(ymm1, ymm5, ymm7);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm1, ymm5, ymm9);

            ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

            a11 += cs_a;

            //extract a33
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

            //(Row 3): FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm1, ymm7, ymm9);

            ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);

            _mm256_storeu_pd((double *)b11, ymm3);
            _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
            _mm256_storeu_pd((double *)(b11 + cs_b*2), ymm7);
            _mm256_storeu_pd((double *)(b11 + cs_b*3), ymm9);

            m_remainder -= 4;
            i += 4;
        }

        if(m_remainder == 3)
        {
            a01 = D_A_pack;
            a11 = L + j*cs_a + j*rs_a;          //pointer to block of A to be used for TRSM
            b10 = B + i;                   //pointer to block of B to be used in GEMM
            b11 = B + i + j*cs_b;          //pointer to block of B to be used for TRSM

            k_iter = j;                    //number of GEMM operations to be done(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_4nx4m(a01,b10,cs_b,p_lda,k_iter)

            ymm15 = _mm256_broadcast_sd((double const *)(&AlphaVal));         //register to hold alpha

            ymm0 = _mm256_loadu_pd((double const *)b11);                            //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
            ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);      //B11[0-3][0] * alpha -= ymm0

            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);    //B11[0-3][1] * alpha-= ymm2

            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*2));        //B11[0][2] B11[1][2] B11[2][2] B11[3][2]
            ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);    //B11[0-3][2] * alpha -= ymm4

            xmm5 = _mm_loadu_pd((double const*)(b11 + cs_b * 3));
            ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b*3 + 2));
            ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);
            ymm9 = _mm256_fmsub_pd(ymm0, ymm15, ymm9);    //B11[0-3][3] * alpha -= ymm6

            ///implement TRSM///

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

            //extract a11
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(row 1):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 1*rs_a));
            ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));
            ymm7 = _mm256_fnmadd_pd(ymm1, ymm3, ymm7);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm1, ymm3, ymm9);

            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

            a11 += cs_a;

            //extract a22
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            //(row 2):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));
            ymm7 = _mm256_fnmadd_pd(ymm1, ymm5, ymm7);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm1, ymm5, ymm9);

            ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

            a11 += cs_a;

            //extract a33
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

            //(Row 3): FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm1, ymm7, ymm9);

            ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);

            ymm0 = _mm256_loadu_pd((double const *)b11);
            ymm3 = _mm256_blend_pd(ymm0, ymm3, 0x07);
            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm5 = _mm256_blend_pd(ymm0, ymm5, 0x07);
            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*2));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm7 = _mm256_blend_pd(ymm0, ymm7, 0x07);
            xmm5 = _mm_loadu_pd((double const*)(b11 + cs_b * 3));
            ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b*3 + 2));
            ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);                      //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm9 = _mm256_blend_pd(ymm0, ymm9, 0x07);

            _mm256_storeu_pd((double *)b11, ymm3);
            _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
            _mm256_storeu_pd((double *)(b11 + cs_b*2), ymm7);
            xmm5 = _mm256_extractf128_pd(ymm9, 0);
            _mm_storeu_pd((double *)(b11 + cs_b * 3),xmm5);
            _mm_storel_pd((b11 + cs_b * 3 + 2), _mm256_extractf128_pd(ymm9, 1));

            m_remainder -= 3;
            i += 3;
        }
        else if(m_remainder == 2)
        {
            a01 = D_A_pack;
            a11 = L + j*cs_a + j*rs_a;                 //pointer to block of A to be used for TRSM
            b10 = B + i;                          //pointer to block of B to be used in GEMM
            b11 = B + i + j*cs_b;                 //pointer to block of B to be used for TRSM

            k_iter = j;                    //number of GEMM operations to be done(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_4nx4m(a01,b10,cs_b,p_lda,k_iter)

            ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);         //register to hold alpha

            ymm0 = _mm256_loadu_pd((double const *)b11);                            //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
            ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);      //B11[0-3][0] * alpha -= ymm0

            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);    //B11[0-3][1] * alpha-= ymm2

            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*2));        //B11[0][2] B11[1][2] B11[2][2] B11[3][2]
            ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);    //B11[0-3][2] * alpha -= ymm4

            xmm5 = _mm_loadu_pd((double const*)(b11 + cs_b * 3));
            ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);
            ymm9 = _mm256_fmsub_pd(ymm0, ymm15, ymm9);    //B11[0-3][3] * alpha -= ymm6

            ///implement TRSM///

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

            //extract a11
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(row 1):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 1*rs_a));
            ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));
            ymm7 = _mm256_fnmadd_pd(ymm1, ymm3, ymm7);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm1, ymm3, ymm9);

            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

            a11 += cs_a;

            //extract a22
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            //(row 2):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));
            ymm7 = _mm256_fnmadd_pd(ymm1, ymm5, ymm7);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm1, ymm5, ymm9);

            ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

            a11 += cs_a;

            //extract a33
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

            //(Row 3): FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm1, ymm7, ymm9);

            ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);

            ymm0 = _mm256_loadu_pd((double const *)b11);
            ymm3 = _mm256_blend_pd(ymm0, ymm3, 0x03);
            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm5 = _mm256_blend_pd(ymm0, ymm5, 0x03);
            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*2));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm7 = _mm256_blend_pd(ymm0, ymm7, 0x03);
            xmm5 = _mm_loadu_pd((double const*)(b11 + cs_b * 3));
            ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);
            ymm9 = _mm256_blend_pd(ymm0, ymm9, 0x03);

            _mm256_storeu_pd((double *)b11, ymm3);
            _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
            _mm256_storeu_pd((double *)(b11 + cs_b*2), ymm7);
            xmm5 = _mm256_extractf128_pd(ymm9, 0);
            _mm_storeu_pd((double *)(b11 + cs_b * 3),xmm5);

            m_remainder -= 2;
            i += 2;
        }
        else if(m_remainder == 1)
        {
            a01 = D_A_pack;
            a11 = L + j*cs_a + j*rs_a;                 //pointer to block of A to be used for TRSM
            b10 = B + i;                          //pointer to block of B to be used in GEMM
            b11 = B + i + j*cs_b;                 //pointer to block of B to be used for TRSM

            k_iter = j;                    //number of GEMM operations to be done(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_4nx4m(a01,b10,cs_b,p_lda,k_iter)

            ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);         //register to hold alpha

            ymm0 = _mm256_broadcast_sd((double const *)b11);                            //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
            ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);      //B11[0-3][0] * alpha -= ymm0

            ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);    //B11[0-3][1] * alpha-= ymm2

            ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b*2));        //B11[0][2] B11[1][2] B11[2][2] B11[3][2]
            ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);    //B11[0-3][2] * alpha -= ymm4

            ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b*3));        //B11[0][3] B11[1][3] B11[2][3] B11[3][3]
            ymm9 = _mm256_fmsub_pd(ymm0, ymm15, ymm9);    //B11[0-3][3] * alpha -= ymm6

            ///implement TRSM///

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

            //extract a11
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(row 1):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 1*rs_a));
            ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));
            ymm7 = _mm256_fnmadd_pd(ymm1, ymm3, ymm7);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm1, ymm3, ymm9);

            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

            a11 += cs_a;

            //extract a22
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            //(row 2):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));
            ymm7 = _mm256_fnmadd_pd(ymm1, ymm5, ymm7);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm1, ymm5, ymm9);

            ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

            a11 += cs_a;

            //extract a33
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

            //(Row 3): FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm1, ymm7, ymm9);

            ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);

            ymm0 = _mm256_loadu_pd((double const *)b11);
            ymm3 = _mm256_blend_pd(ymm0, ymm3, 0x01);
            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm5 = _mm256_blend_pd(ymm0, ymm5, 0x01);
            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*2));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm7 = _mm256_blend_pd(ymm0, ymm7, 0x01);
            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*3));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm9 = _mm256_blend_pd(ymm0, ymm9, 0x01);

            _mm_storel_pd((b11 + cs_b * 0), _mm256_extractf128_pd(ymm3, 0));
            _mm_storel_pd((b11 + cs_b * 1), _mm256_extractf128_pd(ymm5, 0));
            _mm_storel_pd((b11 + cs_b * 2), _mm256_extractf128_pd(ymm7, 0));
            _mm_storel_pd((b11 + cs_b * 3), _mm256_extractf128_pd(ymm9, 0));

            m_remainder -= 1;
            i += 1;
        }
        j += 4;
        n_remainder -= 4;
    }

    if(n_remainder == 3)
    {
        a01 = L + j*rs_a;                     //pointer to block of A to be used in GEMM
        a11 = L + j*cs_a + j*rs_a;                 //pointer to block of A to be used for TRSM

        double *ptr_a10_dup = D_A_pack;

        dim_t p_lda = j; // packed leading dimension
        // perform copy of A to packed buffer D_A_pack

        if(transa)
        {
            for(dim_t x =0;x < p_lda;x+=d_nr)
            {
                ymm0 = _mm256_loadu_pd((double const *)(a01));
                ymm1 = _mm256_loadu_pd((double const *)(a01 + cs_a));
                ymm2 = _mm256_loadu_pd((double const *)(a01 + cs_a * 2));
                ymm3 = _mm256_loadu_pd((double const *)(a01 + cs_a * 3));

                ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
                ymm5 = _mm256_unpacklo_pd(ymm2, ymm3);

                ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);

                ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
                ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);

                ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

                _mm256_storeu_pd((double *)(ptr_a10_dup), ymm6);
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda), ymm7);
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*2), ymm8);
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*3), ymm9);

                ymm0 = _mm256_loadu_pd((double const *)(a01 + cs_a * 4));
                ymm1 = _mm256_loadu_pd((double const *)(a01 + cs_a * 5));

                ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
                ymm5 = _mm256_broadcast_sd((double const *)&zero);

                ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);

                ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
                ymm1 = _mm256_broadcast_sd((double const *)&zero);

                ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

                _mm_storeu_pd((double *)(ptr_a10_dup + 4), _mm256_extractf128_pd(ymm6,0));
                _mm_storeu_pd((double *)(ptr_a10_dup + 4 + p_lda), _mm256_extractf128_pd(ymm7,0));
                _mm_storeu_pd((double *)(ptr_a10_dup + 4 + p_lda*2), _mm256_extractf128_pd(ymm8,0));
                _mm_storeu_pd((double *)(ptr_a10_dup + 4 + p_lda*3), _mm256_extractf128_pd(ymm9,0));

                a01 += d_nr*cs_a;
                ptr_a10_dup += d_nr;
            }
        }
        else
        {
            dim_t loop_count = p_lda/4;

            for(dim_t x =0;x < loop_count;x++)
            {
                ymm15 = _mm256_loadu_pd((double const *)(a01 + rs_a * 0 + x*4));
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda * 0 + x*4), ymm15);
                ymm15 = _mm256_loadu_pd((double const *)(a01 + rs_a * 1 + x*4));
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda * 1 + x*4), ymm15);
                ymm15 = _mm256_loadu_pd((double const *)(a01 + rs_a * 2 + x*4));
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda * 2 + x*4), ymm15);
            }

            dim_t remainder_loop_count = p_lda - loop_count*4;

            __m128d xmm0;
            if(remainder_loop_count != 0)
            {
                xmm0 = _mm_loadu_pd((double const *)(a01 + rs_a * 0 + loop_count*4));
                _mm_storeu_pd((double *)(ptr_a10_dup + p_lda * 0 + loop_count*4), xmm0);
                xmm0 = _mm_loadu_pd((double const *)(a01 + rs_a * 1 + loop_count*4));
                _mm_storeu_pd((double *)(ptr_a10_dup + p_lda * 1 + loop_count*4), xmm0);
                xmm0 = _mm_loadu_pd((double const *)(a01 + rs_a * 2 + loop_count*4));
                _mm_storeu_pd((double *)(ptr_a10_dup + p_lda * 2 + loop_count*4), xmm0);
            }
        }

        ymm4 = _mm256_broadcast_sd((double const *)&ones);
        if(!is_unitdiag)
        {
            if(transa)
            {
                //broadcast diagonal elements of A11
                ymm0 = _mm256_broadcast_sd((double const *)(a11));
                ymm1 = _mm256_broadcast_sd((double const *)(a11+ cs_a*1 + 1));
                ymm2 = _mm256_broadcast_sd((double const *)(a11+ cs_a*2 + 2));
            }
            else
            {
                //broadcast diagonal elements of A11
                ymm0 = _mm256_broadcast_sd((double const *)(a11));
                ymm1 = _mm256_broadcast_sd((double const *)(a11+ rs_a*1 + 1));
                ymm2 = _mm256_broadcast_sd((double const *)(a11+ rs_a*2 + 2));
            }
            ymm3 = _mm256_broadcast_sd((double const *)&ones);

            ymm0 = _mm256_unpacklo_pd(ymm0, ymm1);
            ymm1 = _mm256_unpacklo_pd(ymm2, ymm3);

            ymm1 = _mm256_blend_pd(ymm0, ymm1, 0x0C);
            #ifdef BLIS_DISABLE_TRSM_PREINVERSION
            ymm4 = ymm1;
            #endif
            #ifdef BLIS_ENABLE_TRSM_PREINVERSION
            ymm4 = _mm256_div_pd(ymm4, ymm1);
            #endif
        }
        _mm256_storeu_pd((double *)(d11_pack), ymm4);

        for(i = 0; (i+d_mr-1) < m; i += d_mr)     //loop along 'M' direction
        {
            a01 = D_A_pack;
            a11 = L + j*cs_a + j*rs_a;                 //pointer to block of A to be used for TRSM
            b10 = B + i;                          //pointer to block of B to be used in GEMM
            b11 = B + i + j*cs_b;                 //pointer to block of B to be used for TRSM

            k_iter = j;                    //number of GEMM operations to be done(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_3nx8m(a01,b10,cs_b,p_lda,k_iter)

            ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);

            ymm0 = _mm256_loadu_pd((double const *)b11);                            //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
            ymm1 = _mm256_loadu_pd((double const *)(b11 + 4));                   //B11[4][0] B11[5][0] B11[6][0] B11[7][0]

            ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);      //B11[0-3][0] * alpha -= ymm0
            ymm4 = _mm256_fmsub_pd(ymm1, ymm15, ymm4);      //B11[4-7][0] * alpha-= ymm1

            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b + 4));           //B11[4][1] B11[5][1] B11[6][1] B11[7][1]

            ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);    //B11[0-3][1] * alpha-= ymm2
            ymm6 = _mm256_fmsub_pd(ymm1, ymm15, ymm6);    //B11[4-7][1] * alpha -= ymm3

            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*2));        //B11[0][2] B11[1][2] B11[2][2] B11[3][2]
            ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b*2 + 4)); //B11[4][2] B11[5][2] B11[6][2] B11[7][2]

            ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);    //B11[0-3][2] * alpha -= ymm4
            ymm8 = _mm256_fmsub_pd(ymm1, ymm15, ymm8);    //B11[4-7][2] * alpha -= ymm5

            ///implement TRSM///

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));

            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);
            ymm4 = DTRSM_SMALL_DIV_OR_SCALE(ymm4, ymm0);

            //extract a11
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(row 1):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 1*rs_a));

            ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);
            ymm6 = _mm256_fnmadd_pd(ymm1, ymm4, ymm6);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));

            ymm7 = _mm256_fnmadd_pd(ymm1, ymm3, ymm7);
            ymm8 = _mm256_fnmadd_pd(ymm1, ymm4, ymm8);

            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);
            ymm6 = DTRSM_SMALL_DIV_OR_SCALE(ymm6, ymm0);

            a11 += cs_a;

            //extract a22
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            //(row 2):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));

            ymm7 = _mm256_fnmadd_pd(ymm1, ymm5, ymm7);
            ymm8 = _mm256_fnmadd_pd(ymm1, ymm6, ymm8);

            ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);
            ymm8 = DTRSM_SMALL_DIV_OR_SCALE(ymm8, ymm0);

            _mm256_storeu_pd((double *)b11, ymm3);
            _mm256_storeu_pd((double *)(b11 + 4), ymm4);
            _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
            _mm256_storeu_pd((double *)(b11 + cs_b + 4), ymm6);
            _mm256_storeu_pd((double *)(b11 + cs_b*2), ymm7);
            _mm256_storeu_pd((double *)(b11 + cs_b*2 + 4), ymm8);
        }

        dim_t m_remainder = m - i;
        if(m_remainder >= 4)
        {
            a01 = D_A_pack;
            a11 = L + j*cs_a + j*rs_a;  //pointer to block of A to be used for TRSM
            b10 = B + i;           //pointer to block of B to be used in GEMM
            b11 = B + i + j*cs_b;  //pointer to block of B to be used for TRSM

            k_iter = j;   //number of GEMM operations to be done(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_3nx4m(a01,b10,cs_b,p_lda,k_iter)

            ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);         //register to hold alpha

            ymm0 = _mm256_loadu_pd((double const *)b11);     //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
            ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);      //B11[0-3][0] * alpha -= ymm0

            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));  //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);    //B11[0-3][1] * alpha-= ymm2

            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*2)); //B11[0][2] B11[1][2] B11[2][2] B11[3][2]
            ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);    //B11[0-3][2] * alpha -= ymm4

            ///implement TRSM///
            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

            //extract a11
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(row 1):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 1*rs_a));
            ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));
            ymm7 = _mm256_fnmadd_pd(ymm1, ymm3, ymm7);

            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

            a11 += cs_a;

            //extract a22
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            //(row 2):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));
            ymm7 = _mm256_fnmadd_pd(ymm1, ymm5, ymm7);

            ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

            _mm256_storeu_pd((double *)b11, ymm3);
            _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
            _mm256_storeu_pd((double *)(b11 + cs_b*2), ymm7);

            m_remainder -= 4;
            i += 4;
        }

        if(m_remainder == 3)
        {
            a01 = D_A_pack;
            a11 = L + j*cs_a + j*rs_a;  //pointer to block of A to be used for TRSM
            b10 = B + i;           //pointer to block of B to be used in GEMM
            b11 = B + i + j*cs_b;  //pointer to block of B to be used for TRSM

            k_iter = j;  //number of GEMM operations to be done(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_3nx4m(a01,b10,cs_b,p_lda,k_iter)

            BLIS_PRE_DTRSM_SMALL_3N_3M(AlphaVal,b11,cs_b)

            ///implement TRSM///

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

            //extract a11
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(row 1):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 1*rs_a));
            ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));
            ymm7 = _mm256_fnmadd_pd(ymm1, ymm3, ymm7);

            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

            a11 += cs_a;

            //extract a22
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            //(row 2):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));
            ymm7 = _mm256_fnmadd_pd(ymm1, ymm5, ymm7);

            ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

            BLIS_POST_DTRSM_SMALL_3N_3M(b11,cs_b)

            m_remainder -= 3;
            i += 3;
        }
        else if(m_remainder == 2)
        {
            a01 = D_A_pack;
            a11 = L + j*cs_a + j*rs_a;                 //pointer to block of A to be used for TRSM
            b10 = B + i;                          //pointer to block of B to be used in GEMM
            b11 = B + i + j*cs_b;                 //pointer to block of B to be used for TRSM

            k_iter = j;                    //number of GEMM operations to be done(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_3nx4m(a01,b10,cs_b,p_lda,k_iter)

            BLIS_PRE_DTRSM_SMALL_3N_2M(AlphaVal,b11,cs_b)

            ///implement TRSM///

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

            //extract a11
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(row 1):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 1*rs_a));
            ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));
            ymm7 = _mm256_fnmadd_pd(ymm1, ymm3, ymm7);

            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

            a11 += cs_a;

            //extract a22
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            //(row 2):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));
            ymm7 = _mm256_fnmadd_pd(ymm1, ymm5, ymm7);

            ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

            BLIS_POST_DTRSM_SMALL_3N_2M(b11,cs_b)

            m_remainder -= 2;
            i += 2;
        }
        else if(m_remainder == 1)
        {
            a01 = D_A_pack;
            a11 = L + j*cs_a + j*rs_a;                 //pointer to block of A to be used for TRSM
            b10 = B + i;                          //pointer to block of B to be used in GEMM
            b11 = B + i + j*cs_b;                 //pointer to block of B to be used for TRSM

            k_iter = j;                    //number of GEMM operations to be done(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_3nx4m(a01,b10,cs_b,p_lda,k_iter)

            BLIS_PRE_DTRSM_SMALL_3N_1M(AlphaVal,b11,cs_b)

            ///implement TRSM///

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

            //extract a11
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(row 1):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 1*rs_a));
            ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));
            ymm7 = _mm256_fnmadd_pd(ymm1, ymm3, ymm7);

            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

            a11 += cs_a;

            //extract a22
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            //(row 2):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));
            ymm7 = _mm256_fnmadd_pd(ymm1, ymm5, ymm7);

            ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

            BLIS_POST_DTRSM_SMALL_3N_1M(b11,cs_b)

            m_remainder -= 1;
            i += 1;
        }
        j += 3;
        n_remainder -= 3;
    }
    else if(n_remainder == 2)
    {
        a01 = L + j*rs_a;                     //pointer to block of A to be used in GEMM
        a11 = L + j*cs_a + j*rs_a;                 //pointer to block of A to be used for TRSM

        double *ptr_a10_dup = D_A_pack;

        dim_t p_lda = j; // packed leading dimension
        // perform copy of A to packed buffer D_A_pack

        if(transa)
        {
            for(dim_t x =0;x < p_lda;x+=d_nr)
            {
                ymm0 = _mm256_loadu_pd((double const *)(a01));
                ymm1 = _mm256_loadu_pd((double const *)(a01 + cs_a));
                ymm2 = _mm256_loadu_pd((double const *)(a01 + cs_a * 2));
                ymm3 = _mm256_loadu_pd((double const *)(a01 + cs_a * 3));

                ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
                ymm5 = _mm256_unpacklo_pd(ymm2, ymm3);

                ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);

                ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
                ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);

                ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

                _mm256_storeu_pd((double *)(ptr_a10_dup), ymm6);
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda), ymm7);
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*2), ymm8);
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*3), ymm9);

                ymm0 = _mm256_loadu_pd((double const *)(a01 + cs_a * 4));
                ymm1 = _mm256_loadu_pd((double const *)(a01 + cs_a * 5));

                ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
                ymm5 = _mm256_broadcast_sd((double const *)&zero);

                ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);

                ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
                ymm1 = _mm256_broadcast_sd((double const *)&zero);

                ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

                _mm_storeu_pd((double *)(ptr_a10_dup + 4), _mm256_extractf128_pd(ymm6,0));
                _mm_storeu_pd((double *)(ptr_a10_dup + 4 + p_lda), _mm256_extractf128_pd(ymm7,0));
                _mm_storeu_pd((double *)(ptr_a10_dup + 4 + p_lda*2), _mm256_extractf128_pd(ymm8,0));
                _mm_storeu_pd((double *)(ptr_a10_dup + 4 + p_lda*3), _mm256_extractf128_pd(ymm9,0));

                a01 += d_nr*cs_a;
                ptr_a10_dup += d_nr;
            }
        }
        else
        {
            dim_t loop_count = p_lda/4;

            for(dim_t x =0;x < loop_count;x++)
            {
                ymm15 = _mm256_loadu_pd((double const *)(a01 + rs_a * 0 + x*4));
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda * 0 + x*4), ymm15);
                ymm15 = _mm256_loadu_pd((double const *)(a01 + rs_a * 1 + x*4));
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda * 1 + x*4), ymm15);
            }

            dim_t remainder_loop_count = p_lda - loop_count*4;

            __m128d xmm0;
            if(remainder_loop_count != 0)
            {
                xmm0 = _mm_loadu_pd((double const *)(a01 + rs_a * 0 + loop_count*4));
                _mm_storeu_pd((double *)(ptr_a10_dup + p_lda * 0 + loop_count*4), xmm0);
                xmm0 = _mm_loadu_pd((double const *)(a01 + rs_a * 1 + loop_count*4));
                _mm_storeu_pd((double *)(ptr_a10_dup + p_lda * 1 + loop_count*4), xmm0);
            }
        }

        ymm4 = _mm256_broadcast_sd((double const *)&ones);
        if(!is_unitdiag)
        {
            if(transa)
            {
                //broadcast diagonal elements of A11
                ymm0 = _mm256_broadcast_sd((double const *)(a11));
                ymm1 = _mm256_broadcast_sd((double const *)(a11+cs_a*1 + 1));
            }
            else
            {
                //broadcast diagonal elements of A11
                ymm0 = _mm256_broadcast_sd((double const *)(a11));
                ymm1 = _mm256_broadcast_sd((double const *)(a11+rs_a*1 + 1));
            }
            ymm2 = _mm256_broadcast_sd((double const *)&ones);
            ymm3 = _mm256_broadcast_sd((double const *)&ones);

            ymm0 = _mm256_unpacklo_pd(ymm0, ymm1);
            ymm1 = _mm256_unpacklo_pd(ymm2, ymm3);

            ymm1 = _mm256_blend_pd(ymm0, ymm1, 0x0C);
            #ifdef BLIS_DISABLE_TRSM_PREINVERSION
            ymm4 = ymm1;
            #endif
            #ifdef BLIS_ENABLE_TRSM_PREINVERSION
            ymm4 = _mm256_div_pd(ymm4, ymm1);
            #endif
        }
        _mm256_storeu_pd((double *)(d11_pack), ymm4);

        for(i = 0; (i+d_mr-1) < m; i += d_mr)     //loop along 'M' direction
        {
            a01 = D_A_pack;
            a11 = L + j*cs_a + j*rs_a;                 //pointer to block of A to be used for TRSM
            b10 = B + i;                          //pointer to block of B to be used in GEMM
            b11 = B + i + j*cs_b;                 //pointer to block of B to be used for TRSM

            k_iter = j;                    //number of GEMM operations to be done(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_2nx8m(a01,b10,cs_b,p_lda,k_iter)

            ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);

            ymm0 = _mm256_loadu_pd((double const *)b11);                            //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
            ymm1 = _mm256_loadu_pd((double const *)(b11 + 4));                   //B11[4][0] B11[5][0] B11[6][0] B11[7][0]

            ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);      //B11[0-3][0] * alpha -= ymm0
            ymm4 = _mm256_fmsub_pd(ymm1, ymm15, ymm4);      //B11[4-7][0] * alpha-= ymm1

            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b + 4));           //B11[4][1] B11[5][1] B11[6][1] B11[7][1]

            ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);    //B11[0-3][1] * alpha-= ymm2
            ymm6 = _mm256_fmsub_pd(ymm1, ymm15, ymm6);    //B11[4-7][1] * alpha -= ymm3

            ///implement TRSM///

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));

            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);
            ymm4 = DTRSM_SMALL_DIV_OR_SCALE(ymm4, ymm0);

            //extract a11
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(row 1):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 1*rs_a));

            ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);
            ymm6 = _mm256_fnmadd_pd(ymm1, ymm4, ymm6);

            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);
            ymm6 = DTRSM_SMALL_DIV_OR_SCALE(ymm6, ymm0);

            _mm256_storeu_pd((double *)b11, ymm3);
            _mm256_storeu_pd((double *)(b11 + 4), ymm4);
            _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
            _mm256_storeu_pd((double *)(b11 + cs_b + 4), ymm6);
        }

        dim_t m_remainder = m - i;
        if(m_remainder >= 4)
        {
            a01 = D_A_pack;
            a11 = L + j*cs_a + j*rs_a;                 //pointer to block of A to be used for TRSM
            b10 = B + i;                          //pointer to block of B to be used in GEMM
            b11 = B + i + j*cs_b;                 //pointer to block of B to be used for TRSM

            k_iter = j;                    //number of GEMM operations to be done(in blocks of 4x4)

            ymm3 = _mm256_setzero_pd();
            ymm5 = _mm256_setzero_pd();

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_2nx4m(a01,b10,cs_b,p_lda,k_iter)

            ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);         //register to hold alpha

            ymm0 = _mm256_loadu_pd((double const *)b11);                            //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
            ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);      //B11[0-3][0] * alpha -= ymm0

            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);    //B11[0-3][1] * alpha-= ymm2

            ///implement TRSM///

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

            //extract a11
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(row 1):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 1*rs_a));
            ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);

            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

            _mm256_storeu_pd((double *)b11, ymm3);
            _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);

            m_remainder -= 4;
            i += 4;
        }

        if(m_remainder == 3)
        {
            a01 = D_A_pack;
            a11 = L + j*cs_a + j*rs_a;                 //pointer to block of A to be used for TRSM
            b10 = B + i;                          //pointer to block of B to be used in GEMM
            b11 = B + i + j*cs_b;                 //pointer to block of B to be used for TRSM

            k_iter = j;                    //number of GEMM operations to be done(in blocks of 4x4)

            ymm3 = _mm256_setzero_pd();
            ymm5 = _mm256_setzero_pd();

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_2nx4m(a01,b10,cs_b,p_lda,k_iter)

            BLIS_PRE_DTRSM_SMALL_2N_3M(AlphaVal,b11,cs_b)

            ///implement TRSM///

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

            //extract a11
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(row 1):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 1*rs_a));
            ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);

            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

            BLIS_POST_DTRSM_SMALL_2N_3M(b11,cs_b)

            m_remainder -= 3;
            i += 3;
        }
        else if(m_remainder == 2)
        {
            a01 = D_A_pack;
            a11 = L + j*cs_a + j*rs_a;                 //pointer to block of A to be used for TRSM
            b10 = B + i;                          //pointer to block of B to be used in GEMM
            b11 = B + i + j*cs_b;                 //pointer to block of B to be used for TRSM

            k_iter = j;                    //number of GEMM operations to be done(in blocks of 4x4)

            ymm3 = _mm256_setzero_pd();
            ymm5 = _mm256_setzero_pd();

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_2nx4m(a01,b10,cs_b,p_lda,k_iter)

            BLIS_PRE_DTRSM_SMALL_2N_2M(AlphaVal,b11,cs_b)

            ///implement TRSM///
            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

            //extract a11
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(row 1):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 1*rs_a));
            ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);

            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

            BLIS_POST_DTRSM_SMALL_2N_2M(b11,cs_b)

            m_remainder -= 2;
            i += 2;
        }
        else if(m_remainder == 1)
        {
            a01 = D_A_pack;
            a11 = L + j*cs_a + j*rs_a;                 //pointer to block of A to be used for TRSM
            b10 = B + i;                          //pointer to block of B to be used in GEMM
            b11 = B + i + j*cs_b;                 //pointer to block of B to be used for TRSM

            k_iter = j;                    //number of GEMM operations to be done(in blocks of 4x4)

            ymm3 = _mm256_setzero_pd();
            ymm5 = _mm256_setzero_pd();

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_2nx4m(a01,b10,cs_b,p_lda,k_iter)

            BLIS_PRE_DTRSM_SMALL_2N_1M(AlphaVal,b11,cs_b)

            ///implement TRSM///

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

            //extract a11
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(row 1):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 1*rs_a));
            ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);

            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

            BLIS_POST_DTRSM_SMALL_2N_1M(b11,cs_b)

            m_remainder -= 1;
            i += 1;
        }
        j += 2;
        n_remainder -= 2;
    }
    else if(n_remainder == 1)
    {
        a01 = L + j*rs_a;                     //pointer to block of A to be used in GEMM
        a11 = L + j*cs_a + j*rs_a;                 //pointer to block of A to be used for TRSM

        double *ptr_a10_dup = D_A_pack;

        dim_t p_lda = j; // packed leading dimension
        // perform copy of A to packed buffer D_A_pack

        if(transa)
        {
            for(dim_t x =0;x < p_lda;x+=d_nr)
            {
                ymm0 = _mm256_loadu_pd((double const *)(a01));
                ymm1 = _mm256_loadu_pd((double const *)(a01 + cs_a));
                ymm2 = _mm256_loadu_pd((double const *)(a01 + cs_a * 2));
                ymm3 = _mm256_loadu_pd((double const *)(a01 + cs_a * 3));

                ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
                ymm5 = _mm256_unpacklo_pd(ymm2, ymm3);

                ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);

                ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
                ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);

                ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

                _mm256_storeu_pd((double *)(ptr_a10_dup), ymm6);
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda), ymm7);
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*2), ymm8);
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*3), ymm9);

                ymm0 = _mm256_loadu_pd((double const *)(a01 + cs_a * 4));
                ymm1 = _mm256_loadu_pd((double const *)(a01 + cs_a * 5));

                ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
                ymm5 = _mm256_broadcast_sd((double const *)&zero);

                ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);

                ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
                ymm1 = _mm256_broadcast_sd((double const *)&zero);

                ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

                _mm_storeu_pd((double *)(ptr_a10_dup + 4), _mm256_extractf128_pd(ymm6,0));
                _mm_storeu_pd((double *)(ptr_a10_dup + 4 + p_lda), _mm256_extractf128_pd(ymm7,0));
                _mm_storeu_pd((double *)(ptr_a10_dup + 4 + p_lda*2), _mm256_extractf128_pd(ymm8,0));
                _mm_storeu_pd((double *)(ptr_a10_dup + 4 + p_lda*3), _mm256_extractf128_pd(ymm9,0));

                a01 += d_nr*cs_a;
                ptr_a10_dup += d_nr;
            }
        }
        else
        {
            dim_t loop_count = p_lda/4;

            for(dim_t x =0;x < loop_count;x++)
            {
                ymm15 = _mm256_loadu_pd((double const *)(a01 + rs_a * 0 + x*4));
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda * 0 + x*4), ymm15);
            }

            dim_t remainder_loop_count = p_lda - loop_count*4;

            __m128d xmm0;
            if(remainder_loop_count != 0)
            {
                xmm0 = _mm_loadu_pd((double const *)(a01 + rs_a * 0 + loop_count*4));
                _mm_storeu_pd((double *)(ptr_a10_dup + p_lda * 0 + loop_count*4), xmm0);
            }
        }

        ymm4 = _mm256_broadcast_sd((double const *)&ones);
        if(!is_unitdiag)
        {
            //broadcast diagonal elements of A11
            ymm0 = _mm256_broadcast_sd((double const *)(a11));
            ymm1 = _mm256_broadcast_sd((double const *)&ones);
            ymm2 = _mm256_broadcast_sd((double const *)&ones);
            ymm3 = _mm256_broadcast_sd((double const *)&ones);

            ymm0 = _mm256_unpacklo_pd(ymm0, ymm1);
            ymm1 = _mm256_unpacklo_pd(ymm2, ymm3);

            ymm1 = _mm256_blend_pd(ymm0, ymm1, 0x0C);
            #ifdef BLIS_DISABLE_TRSM_PREINVERSION
            ymm4 = ymm1;
            #endif
            #ifdef BLIS_ENABLE_TRSM_PREINVERSION
            ymm4 = _mm256_div_pd(ymm4, ymm1);
            #endif
        }
        _mm256_storeu_pd((double *)(d11_pack), ymm4);

        for(i = 0; (i+d_mr-1) < m; i += d_mr)     //loop along 'M' direction
        {
            a01 = D_A_pack;
            a11 = L + j*cs_a + j*rs_a;                 //pointer to block of A to be used for TRSM
            b10 = B + i;                          //pointer to block of B to be used in GEMM
            b11 = B + i + j*cs_b;                 //pointer to block of B to be used for TRSM

            k_iter = j;                    //number of GEMM operations to be done(in blocks of 4x4)

            ymm3 = _mm256_setzero_pd();
            ymm4 = _mm256_setzero_pd();
            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_1nx8m(a01,b10,cs_b,p_lda,k_iter)

            ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);

            ymm0 = _mm256_loadu_pd((double const *)b11);                            //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
            ymm1 = _mm256_loadu_pd((double const *)(b11 + 4));                   //B11[4][0] B11[5][0] B11[6][0] B11[7][0]

            ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);      //B11[0-3][0] * alpha -= ymm0
            ymm4 = _mm256_fmsub_pd(ymm1, ymm15, ymm4);      //B11[4-7][0] * alpha-= ymm1

            ///implement TRSM///

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));

            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);
            ymm4 = DTRSM_SMALL_DIV_OR_SCALE(ymm4, ymm0);

            _mm256_storeu_pd((double *)b11, ymm3);
            _mm256_storeu_pd((double *)(b11 + 4), ymm4);
        }

        dim_t m_remainder = m - i;
        if(m_remainder >= 4)
        {
            a01 = D_A_pack;
            a11 = L + j*cs_a + j*rs_a;                 //pointer to block of A to be used for TRSM
            b10 = B + i;                          //pointer to block of B to be used in GEMM
            b11 = B + i + j*cs_b;                 //pointer to block of B to be used for TRSM

            k_iter = j;                    //number of GEMM operations to be done(in blocks of 4x4)

            ymm3 = _mm256_setzero_pd();

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_1nx4m(a01,b10,cs_b,p_lda,k_iter)

            ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);         //register to hold alpha

            ymm0 = _mm256_loadu_pd((double const *)b11);                            //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
            ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);      //B11[0-3][0] * alpha -= ymm0

            ///implement TRSM///

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

            _mm256_storeu_pd((double *)b11, ymm3);

            m_remainder -= 4;
            i += 4;
        }

        if(m_remainder == 3)
        {
            a01 = D_A_pack;
            a11 = L + j*cs_a + j*rs_a;                 //pointer to block of A to be used for TRSM
            b10 = B + i;                          //pointer to block of B to be used in GEMM
            b11 = B + i + j*cs_b;                 //pointer to block of B to be used for TRSM

            k_iter = j;                    //number of GEMM operations to be done(in blocks of 4x4)

            ymm3 = _mm256_setzero_pd();

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_1nx4m(a01,b10,cs_b,p_lda,k_iter)

            BLIS_PRE_DTRSM_SMALL_1N_3M(AlphaVal,b11,cs_b)

            ///implement TRSM///

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

            BLIS_POST_DTRSM_SMALL_1N_3M(b11,cs_b)

            m_remainder -= 3;
            i += 3;
        }
        else if(m_remainder == 2)
        {
            a01 = D_A_pack;
            a11 = L + j*cs_a + j*rs_a;                 //pointer to block of A to be used for TRSM
            b10 = B + i;                          //pointer to block of B to be used in GEMM
            b11 = B + i + j*cs_b;                 //pointer to block of B to be used for TRSM

            k_iter = j;                    //number of GEMM operations to be done(in blocks of 4x4)

            ymm3 = _mm256_setzero_pd();

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_1nx4m(a01,b10,cs_b,p_lda,k_iter)

            BLIS_PRE_DTRSM_SMALL_1N_2M(AlphaVal,b11,cs_b)

            ///implement TRSM///

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

            BLIS_POST_DTRSM_SMALL_1N_2M(b11,cs_b)

            m_remainder -= 2;
            i += 2;
        }
        else if(m_remainder == 1)
        {
            a01 = D_A_pack;
            a11 = L + j*cs_a + j*rs_a;                 //pointer to block of A to be used for TRSM
            b10 = B + i;                          //pointer to block of B to be used in GEMM
            b11 = B + i + j*cs_b;                 //pointer to block of B to be used for TRSM

            k_iter = j;                    //number of GEMM operations to be done(in blocks of 4x4)

            ymm3 = _mm256_setzero_pd();

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_1nx4m(a01,b10,cs_b,p_lda,k_iter)

            BLIS_PRE_DTRSM_SMALL_1N_1M(AlphaVal,b11,cs_b)

            ///implement TRSM///

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

            BLIS_POST_DTRSM_SMALL_1N_1M(b11,cs_b)

            m_remainder -= 1;
            i += 1;
        }
        j += 1;
        n_remainder -= 1;
    }

    if ((required_packing_A == 1) && bli_mem_is_alloc( &local_mem_buf_A_s ))
    {
      bli_membrk_release(&rntm,
                         &local_mem_buf_A_s);
    }

    return BLIS_SUCCESS;
}

/*implements TRSM for the case XA = alpha * B
 *A is upper triangular, non-unit diagonal/unit diagonal, transpose
 *dimensions: X:mxn     A:nxn       B: mxn
 *
 *  <---b11            <---a11
    *****************      *
    *b01*b11*   *   *      * *
 ^  *   *   *   *   *    ^ *   *
 |  *****************    | *******
 |  *   *   *   *   *    | *     * *
 |  *   *   *   *   *   a01*     *   *
b10 *****************      *************
    *   *   *   *   *      *     *     * *
    *   *   *   *   *      *     *     *   *
    *****************      *******************

 *implements TRSM for the case XA = alpha * B
 *A is lower triangular, non-unit diagonal/unit diagonal, no transpose
 *dimensions: X:mxn     A:nxn       B: mxn
 *
 *   <---b11            <---a11
    *****************      *
    *b01*b11*   *   *      * *
 ^  *   *   *   *   *    ^ *   *
 |  *****************    | *******
 |  *   *   *   *   *    | *     * *
 |  *   *   *   *   *   a01*     *   *
b10 *****************      *************
    *   *   *   *   *      *     *     * *
    *   *   *   *   *      *     *     *   *
    *****************      *******************

*/
BLIS_INLINE  err_t bli_dtrsm_small_XAutB_XAlB
(
    obj_t* AlphaObj,
    obj_t* a,
    obj_t* b,
    cntx_t* cntx,
    cntl_t* cntl
)
{
    dim_t m = bli_obj_length(b);  //number of rows
    dim_t n = bli_obj_width(b);   //number of columns

    bool transa = bli_obj_has_trans(a);
    dim_t cs_a, rs_a;
    dim_t d_mr = 8,d_nr = 6;

    // Swap rs_a & cs_a in case of non-tranpose.
    if(transa)
    {
        cs_a = bli_obj_col_stride(a); // column stride of A
        rs_a = bli_obj_row_stride(a); // row stride of A
    }
    else
    {
        cs_a = bli_obj_row_stride(a); // row stride of A
        rs_a = bli_obj_col_stride(a); // column stride of A
    }
    dim_t cs_b = bli_obj_col_stride(b); //column stride of matrix B

    dim_t i, j, k;        //loop variablse
    dim_t k_iter;         //determines the number of GEMM operations to be done

    double ones = 1.0;
    double zero = 0.0;
    bool is_unitdiag = bli_obj_has_unit_diag(a);

    double AlphaVal = *(double *)AlphaObj->buffer;    //value of Alpha
    double* restrict L = a->buffer;      //pointer to matrix A
    double* restrict B = b->buffer;      //pointer to matrix B

    double *a01, *a11, *b10, *b11;   //pointers for GEMM and TRSM blocks

    gint_t required_packing_A = 1;
    mem_t local_mem_buf_A_s = {0};
    double *D_A_pack = NULL;
    double d11_pack[d_mr] __attribute__((aligned(64)));
    rntm_t rntm;

    bli_rntm_init_from_global( &rntm );
    bli_rntm_set_num_threads_only( 1, &rntm );
    bli_membrk_rntm_set_membrk( &rntm );

    siz_t buffer_size = bli_pool_block_size(
                          bli_membrk_pool(
                            bli_packbuf_index(BLIS_BITVAL_BUFFER_FOR_A_BLOCK),
                            bli_rntm_membrk(&rntm)));

    if( (d_nr * n * sizeof(double)) > buffer_size)
        return BLIS_NOT_YET_IMPLEMENTED;

    if (required_packing_A == 1)
    {
      // Get the buffer from the pool.
      bli_membrk_acquire_m(&rntm,
                           buffer_size,
                           BLIS_BITVAL_BUFFER_FOR_A_BLOCK,
                           &local_mem_buf_A_s);
      if(FALSE==bli_mem_is_alloc(&local_mem_buf_A_s)) return BLIS_NULL_POINTER;
      D_A_pack = bli_mem_buffer(&local_mem_buf_A_s);
      if(NULL==D_A_pack) return BLIS_NULL_POINTER;
    }

    //ymm scratch reginsters
    __m256d ymm0, ymm1, ymm2, ymm3;
    __m256d ymm4, ymm5, ymm6, ymm7;
    __m256d ymm8, ymm9, ymm10, ymm11;
    __m256d ymm12, ymm13, ymm14, ymm15;

    __m128d xmm5;

    /*
    Performs solving TRSM for 6 rows at a time from  0 to n/6 in steps of d_nr
    a. Load and pack A (a01 block), the size of packing 6x6 to 6x (n-6)
       First there will be no GEMM and no packing of a01 because it is only TRSM
    b. Using packed a01 block and b10 block perform GEMM operation
    c. Use GEMM outputs, perform TRSM operation using a11, b11 and update B
    d. Repeat b for m cols of B in steps of d_mr
    */

    for(j = (n-d_nr); (j+1) > 0; j -= d_nr)     //loop along 'N' direction
    {
        a01 = L + (j*rs_a) + (j+d_nr)*cs_a;                     //pointer to block of A to be used in GEMM
        a11 = L + (j*cs_a) + (j*rs_a);                 //pointer to block of A to be used for TRSM

        //double *ptr_a10_dup = D_A_pack;

        dim_t p_lda = (n-j-d_nr); // packed leading dimension
        // perform copy of A to packed buffer D_A_pack

        if(transa)
        {
            /*
            Pack current A block (a01) into packed buffer memory D_A_pack
            a. This a10 block is used in GEMM portion only and this
               a01 block size will be increasing by d_nr for every next iteration
               until it reaches 6x(n-6) which is the maximum GEMM alone block size in A
            b. This packed buffer is reused to calculate all m cols of B matrix
            */
            bli_dtrsm_small_pack('R', p_lda, 1, a01, cs_a, D_A_pack, p_lda,d_nr);

            /*
               Pack 6 diagonal elements of A block into an array
               a. This helps in utilze cache line efficiently in TRSM operation
               b. store ones when input is unit diagonal
            */
            dtrsm_small_pack_diag_element(is_unitdiag,a11,cs_a,d11_pack,d_nr);
        }
        else
        {
            bli_dtrsm_small_pack('R', p_lda, 0, a01, rs_a, D_A_pack, p_lda,d_nr);
            dtrsm_small_pack_diag_element(is_unitdiag,a11,rs_a,d11_pack,d_nr);
        }

        /*
        a. Perform GEMM using a01, b10.
        b. Perform TRSM on a11, b11
        c. This loop GEMM+TRSM loops operates with 8x6 block size
           along m dimension for every d_mr columns of B10 where
           packed A buffer is reused in computing all m cols of B.
        d. Same approach is used in remaining fringe cases.
        */
        for(i = (m-d_mr); (i+1) > 0; i -= d_mr)     //loop along 'M' direction
        {
            a01 = D_A_pack;
            a11 = L + j*cs_a + j*rs_a;                 //pointer to block of A to be used for TRSM
            b10 = B + i + (j+d_nr)*cs_b;                          //pointer to block of B to be used in GEMM
            b11 = B + (i) + (j)*cs_b;                 //pointer to block of B to be used for TRSM

            k_iter = (n-j-d_nr);                    //number of GEMM operations to be done(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            /*
            Peform GEMM between a01 and b10 blocks
            For first itteration there will be no GEMM operation
            where k_iter are zero
            */

            BLIS_DTRSM_SMALL_GEMM_6nx8m(a01,b10,cs_b,p_lda,k_iter)

            /*
            Load b11 of size 8x6 and multiply with alpha
            Add the GEMM output to b11
            and peform TRSM operation.
            */

            BLIS_PRE_DTRSM_SMALL_6x8(AlphaVal,b11,cs_b)

            ///implement TRSM///

            /*
            Compute 6x8 TRSM block by using GEMM block output in register
            a. The 6x8 input (gemm outputs) are stored in combinations of ymm registers
                1. ymm3, ymm4 2. ymm5, ymm6 3. ymm7, ymm8, 4. ymm9, ymm10
                5. ymm11, ymm12 6. ymm13,ymm14
            b. Towards the end TRSM output will be stored back into b11
            */

            //extract a55
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 5));

            ymm13 = DTRSM_SMALL_DIV_OR_SCALE(ymm13, ymm0);
            ymm14 = DTRSM_SMALL_DIV_OR_SCALE(ymm14, ymm0);

            //extract a44
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 4));

            //(row 5):FMA operations
            //ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a + 4*rs_a));

            ymm11 = _mm256_fnmadd_pd(ymm1, ymm13, ymm11);
            ymm12 = _mm256_fnmadd_pd(ymm1, ymm14, ymm12);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a + 3*rs_a));

            ymm9 = _mm256_fnmadd_pd(ymm1, ymm13, ymm9);
            ymm10 = _mm256_fnmadd_pd(ymm1, ymm14, ymm10);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a + 2*rs_a));

            ymm7 = _mm256_fnmadd_pd(ymm1, ymm13, ymm7);
            ymm8 = _mm256_fnmadd_pd(ymm1, ymm14, ymm8);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a + 1*rs_a));

            ymm5 = _mm256_fnmadd_pd(ymm1, ymm13, ymm5);
            ymm6 = _mm256_fnmadd_pd(ymm1, ymm14, ymm6);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a));

            ymm3 = _mm256_fnmadd_pd(ymm1, ymm13, ymm3);
            ymm4 = _mm256_fnmadd_pd(ymm1, ymm14, ymm4);

            ymm11 = DTRSM_SMALL_DIV_OR_SCALE(ymm11, ymm0);
            ymm12 = DTRSM_SMALL_DIV_OR_SCALE(ymm12, ymm0);

            //extract a33
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

            //(row 4):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*cs_a + 3*rs_a));

            ymm9 = _mm256_fnmadd_pd(ymm1, ymm11, ymm9);
            ymm10 = _mm256_fnmadd_pd(ymm1, ymm12, ymm10);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*cs_a + 2*rs_a));

            ymm7 = _mm256_fnmadd_pd(ymm1, ymm11, ymm7);
            ymm8 = _mm256_fnmadd_pd(ymm1, ymm12, ymm8);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*cs_a + 1*rs_a));

            ymm5 = _mm256_fnmadd_pd(ymm1, ymm11, ymm5);
            ymm6 = _mm256_fnmadd_pd(ymm1, ymm12, ymm6);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*cs_a));

            ymm3 = _mm256_fnmadd_pd(ymm1, ymm11, ymm3);
            ymm4 = _mm256_fnmadd_pd(ymm1, ymm12, ymm4);

            ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);
            ymm10 = DTRSM_SMALL_DIV_OR_SCALE(ymm10, ymm0);

            //extract a22
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            //(Row 3): FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 2*rs_a));

            ymm7 = _mm256_fnmadd_pd(ymm1, ymm9, ymm7);
            ymm8 = _mm256_fnmadd_pd(ymm1, ymm10, ymm8);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 1*rs_a));

            ymm5 = _mm256_fnmadd_pd(ymm1, ymm9, ymm5);
            ymm6 = _mm256_fnmadd_pd(ymm1, ymm10, ymm6);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a));

            ymm3 = _mm256_fnmadd_pd(ymm1, ymm9, ymm3);
            ymm4 = _mm256_fnmadd_pd(ymm1, ymm10, ymm4);

            ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);
            ymm8 = DTRSM_SMALL_DIV_OR_SCALE(ymm8, ymm0);

            //extract a11
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(row 2):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 1*rs_a));

            ymm5 = _mm256_fnmadd_pd(ymm1, ymm7, ymm5);
            ymm6 = _mm256_fnmadd_pd(ymm1, ymm8, ymm6);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a));

            ymm3 = _mm256_fnmadd_pd(ymm1, ymm7, ymm3);
            ymm4 = _mm256_fnmadd_pd(ymm1, ymm8, ymm4);

            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);
            ymm6 = DTRSM_SMALL_DIV_OR_SCALE(ymm6, ymm0);

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack ));

            //(Row 1): FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));

            ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);
            ymm4 = _mm256_fnmadd_pd(ymm1, ymm6, ymm4);

            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);
            ymm4 = DTRSM_SMALL_DIV_OR_SCALE(ymm4, ymm0);

            _mm256_storeu_pd((double *)b11, ymm3);
            _mm256_storeu_pd((double *)(b11 + 4), ymm4);
            _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
            _mm256_storeu_pd((double *)(b11 + cs_b + 4), ymm6);
            _mm256_storeu_pd((double *)(b11 + cs_b*2), ymm7);
            _mm256_storeu_pd((double *)(b11 + cs_b*2 + 4), ymm8);
            _mm256_storeu_pd((double *)(b11 + cs_b*3), ymm9);
            _mm256_storeu_pd((double *)(b11 + cs_b*3 + 4), ymm10);
            _mm256_storeu_pd((double *)(b11 + cs_b*4), ymm11);
            _mm256_storeu_pd((double *)(b11 + cs_b*4 + 4), ymm12);
            _mm256_storeu_pd((double *)(b11 + cs_b*5), ymm13);
            _mm256_storeu_pd((double *)(b11 + cs_b*5 + 4), ymm14);
        }

        dim_t m_remainder = i + d_mr;
        if(m_remainder >= 4)
        {
            a01 = D_A_pack;
            a11 = L + (j*cs_a) + (j*rs_a);
            b10 = B + (m_remainder - 4) + (j+d_nr)*cs_b;                          //pointer to block of B to be used in GEMM
            b11 = B + (m_remainder - 4) + (j*cs_b);

            k_iter = (n-j-d_nr);                    //number of GEMM operations to be done(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_6nx4m(a01,b10,cs_b,p_lda,k_iter)

            // Load b11 of size 4x6 and multiply with alpha
            BLIS_PRE_DTRSM_SMALL_6x4(AlphaVal,b11,cs_b)

            ///implement TRSM///

            //extract a55
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 5));
            ymm13 = DTRSM_SMALL_DIV_OR_SCALE(ymm13, ymm0);

            //extract a44
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 4));

            //(row 5):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a + 4*rs_a));
            ymm11 = _mm256_fnmadd_pd(ymm1, ymm13, ymm11);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a + 3*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm1, ymm13, ymm9);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a + 2*rs_a));
            ymm7 = _mm256_fnmadd_pd(ymm1, ymm13, ymm7);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a + 1*rs_a));
            ymm5 = _mm256_fnmadd_pd(ymm1, ymm13, ymm5);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a));
            ymm3 = _mm256_fnmadd_pd(ymm1, ymm13, ymm3);

            ymm11 = DTRSM_SMALL_DIV_OR_SCALE(ymm11, ymm0);

            //extract a33
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

            //(row 4):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*cs_a + 3*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm1, ymm11, ymm9);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*cs_a + 2*rs_a));
            ymm7 = _mm256_fnmadd_pd(ymm1, ymm11, ymm7);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*cs_a + 1*rs_a));
            ymm5 = _mm256_fnmadd_pd(ymm1, ymm11, ymm5);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*cs_a));
            ymm3 = _mm256_fnmadd_pd(ymm1, ymm11, ymm3);

            ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);

            //extract a22
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            //(Row 3): FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 2*rs_a));
            ymm7 = _mm256_fnmadd_pd(ymm1, ymm9, ymm7);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 1*rs_a));
            ymm5 = _mm256_fnmadd_pd(ymm1, ymm9, ymm5);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a));
            ymm3 = _mm256_fnmadd_pd(ymm1, ymm9, ymm3);

            ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

            //extract a11
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(row 2):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 1*rs_a));
            ymm5 = _mm256_fnmadd_pd(ymm1, ymm7, ymm5);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a));
            ymm3 = _mm256_fnmadd_pd(ymm1, ymm7, ymm3);

            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack ));

            //(Row 1): FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));
            ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);

            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

            _mm256_storeu_pd((double *)b11, ymm3);
            _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
            _mm256_storeu_pd((double *)(b11 + cs_b*2), ymm7);
            _mm256_storeu_pd((double *)(b11 + cs_b*3), ymm9);
            _mm256_storeu_pd((double *)(b11 + cs_b*4), ymm11);
            _mm256_storeu_pd((double *)(b11 + cs_b*5), ymm13);

            m_remainder -=4;
        }

        if(m_remainder)
        {
            if(3 == m_remainder)
            {
                a01 = D_A_pack;
                a11 = L + (j*cs_a) + (j*rs_a);
                b10 = B + (j+d_nr)*cs_b + (m_remainder - 3);                          //pointer to block of B to be used in GEMM
                b11 = B + (m_remainder - 3) + (j*cs_b);

                k_iter = (n-j-d_nr);                    //number of GEMM operations to be done(in blocks of 4x4)

                /*Fill zeros into ymm registers used in gemm accumulations */
                BLIS_SET_YMM_REG_ZEROS

                ///GEMM implementation starts///
                BLIS_DTRSM_SMALL_GEMM_6nx4m(a01,b10,cs_b,p_lda,k_iter)

                // Load b11 of size 4x6 and multiply with alpha
                BLIS_PRE_DTRSM_SMALL_6x4(AlphaVal,b11,cs_b)

                ///implement TRSM///

                //extract a55
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 5));
                ymm13 = DTRSM_SMALL_DIV_OR_SCALE(ymm13, ymm0);

                //extract a44
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 4));

                //(row 5):FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a + 4*rs_a));
                ymm11 = _mm256_fnmadd_pd(ymm1, ymm13, ymm11);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a + 3*rs_a));
                ymm9 = _mm256_fnmadd_pd(ymm1, ymm13, ymm9);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a + 2*rs_a));
                ymm7 = _mm256_fnmadd_pd(ymm1, ymm13, ymm7);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a + 1*rs_a));
                ymm5 = _mm256_fnmadd_pd(ymm1, ymm13, ymm5);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm13, ymm3);

                ymm11 = DTRSM_SMALL_DIV_OR_SCALE(ymm11, ymm0);

                //extract a33
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

                //(row 4):FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*cs_a + 3*rs_a));
                ymm9 = _mm256_fnmadd_pd(ymm1, ymm11, ymm9);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*cs_a + 2*rs_a));
                ymm7 = _mm256_fnmadd_pd(ymm1, ymm11, ymm7);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*cs_a + 1*rs_a));
                ymm5 = _mm256_fnmadd_pd(ymm1, ymm11, ymm5);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm11, ymm3);

                ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);

                //extract a22
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

                //(Row 3): FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 2*rs_a));
                ymm7 = _mm256_fnmadd_pd(ymm1, ymm9, ymm7);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 1*rs_a));
                ymm5 = _mm256_fnmadd_pd(ymm1, ymm9, ymm5);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm9, ymm3);

                ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

                //extract a11
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

                //(row 2):FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 1*rs_a));
                ymm5 = _mm256_fnmadd_pd(ymm1, ymm7, ymm5);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm7, ymm3);

                ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

                //extract a00
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack ));

                //(Row 1): FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);

                ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

                ymm0 = _mm256_loadu_pd((double const *)b11);
                ymm3 = _mm256_blend_pd(ymm0, ymm3, 0x07);
                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
                ymm5 = _mm256_blend_pd(ymm0, ymm5, 0x07);
                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*2));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
                ymm7 = _mm256_blend_pd(ymm0, ymm7, 0x07);
                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*3));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
                ymm9 = _mm256_blend_pd(ymm0, ymm9, 0x07);
                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*4));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
                ymm11 = _mm256_blend_pd(ymm0, ymm11, 0x07);
                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*5));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
                ymm13 = _mm256_blend_pd(ymm0, ymm13, 0x07);

                _mm256_storeu_pd((double *)b11, ymm3);
                _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
                _mm256_storeu_pd((double *)(b11 + cs_b*2), ymm7);
                _mm256_storeu_pd((double *)(b11 + cs_b*3), ymm9);
                _mm256_storeu_pd((double *)(b11 + cs_b*4), ymm11);
                _mm256_storeu_pd((double *)(b11 + cs_b*5), ymm13);

                m_remainder -=3;
            }
            else if(2 == m_remainder)
            {
                a01 = D_A_pack;
                a11 = L + (j*cs_a) + (j*rs_a);
                b10 = B + (j+d_nr)*cs_b + (m_remainder - 2);                          //pointer to block of B to be used in GEMM
                b11 = B + (m_remainder - 2) + (j*cs_b);

                k_iter = (n-j-d_nr);                    //number of GEMM operations to be done(in blocks of 4x4)

                /*Fill zeros into ymm registers used in gemm accumulations */
                BLIS_SET_YMM_REG_ZEROS

                ///GEMM implementation starts///
                BLIS_DTRSM_SMALL_GEMM_6nx4m(a01,b10,cs_b,p_lda,k_iter)

                // Load b11 of size 4x6 and multiply with alpha
                BLIS_PRE_DTRSM_SMALL_6x4(AlphaVal,b11,cs_b)

                ///implement TRSM///

                //extract a55
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 5));
                ymm13 = DTRSM_SMALL_DIV_OR_SCALE(ymm13, ymm0);

                //extract a44
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 4));

                //(row 5):FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a + 4*rs_a));
                ymm11 = _mm256_fnmadd_pd(ymm1, ymm13, ymm11);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a + 3*rs_a));
                ymm9 = _mm256_fnmadd_pd(ymm1, ymm13, ymm9);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a + 2*rs_a));
                ymm7 = _mm256_fnmadd_pd(ymm1, ymm13, ymm7);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a + 1*rs_a));
                ymm5 = _mm256_fnmadd_pd(ymm1, ymm13, ymm5);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm13, ymm3);

                ymm11 = DTRSM_SMALL_DIV_OR_SCALE(ymm11, ymm0);

                //extract a33
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

                //(row 4):FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*cs_a + 3*rs_a));
                ymm9 = _mm256_fnmadd_pd(ymm1, ymm11, ymm9);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*cs_a + 2*rs_a));
                ymm7 = _mm256_fnmadd_pd(ymm1, ymm11, ymm7);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*cs_a + 1*rs_a));
                ymm5 = _mm256_fnmadd_pd(ymm1, ymm11, ymm5);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm11, ymm3);

                ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);

                //extract a22
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

                //(Row 3): FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 2*rs_a));
                ymm7 = _mm256_fnmadd_pd(ymm1, ymm9, ymm7);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 1*rs_a));
                ymm5 = _mm256_fnmadd_pd(ymm1, ymm9, ymm5);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm9, ymm3);

                ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

                //extract a11
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

                //(row 2):FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 1*rs_a));
                ymm5 = _mm256_fnmadd_pd(ymm1, ymm7, ymm5);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm7, ymm3);

                ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

                //extract a00
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack ));

                //(Row 1): FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);

                ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

                ymm0 = _mm256_loadu_pd((double const *)b11);
                ymm3 = _mm256_blend_pd(ymm0, ymm3, 0x03);
                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
                ymm5 = _mm256_blend_pd(ymm0, ymm5, 0x03);
                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*2));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
                ymm7 = _mm256_blend_pd(ymm0, ymm7, 0x03);
                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*3));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
                ymm9 = _mm256_blend_pd(ymm0, ymm9, 0x03);
                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*4));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
                ymm11 = _mm256_blend_pd(ymm0, ymm11, 0x03);
                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*5));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
                ymm13 = _mm256_blend_pd(ymm0, ymm13, 0x03);

                _mm256_storeu_pd((double *)b11, ymm3);
                _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
                _mm256_storeu_pd((double *)(b11 + cs_b*2), ymm7);
                _mm256_storeu_pd((double *)(b11 + cs_b*3), ymm9);
                _mm256_storeu_pd((double *)(b11 + cs_b*4), ymm11);
                _mm256_storeu_pd((double *)(b11 + cs_b*5), ymm13);

                m_remainder -=2;
            }
            else if (1 == m_remainder)
            {
                a01 = D_A_pack;
                a11 = L + (j*cs_a) + (j*rs_a);
                b10 = B + (j+d_nr)*cs_b + (m_remainder - 1);                          //pointer to block of B to be used in GEMM
                b11 = B + (m_remainder - 1) + (j*cs_b);

                k_iter = (n-j-d_nr);                    //number of GEMM operations to be done(in blocks of 4x4)

                /*Fill zeros into ymm registers used in gemm accumulations */
                BLIS_SET_YMM_REG_ZEROS

                ///GEMM implementation starts///
                BLIS_DTRSM_SMALL_GEMM_6nx4m(a01,b10,cs_b,p_lda,k_iter)

                // Load b11 of size 4x6 and multiply with alpha
                BLIS_PRE_DTRSM_SMALL_6x4(AlphaVal,b11,cs_b)

                ///implement TRSM///

                //extract a55
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 5));
                ymm13 = DTRSM_SMALL_DIV_OR_SCALE(ymm13, ymm0);

                //extract a44
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 4));

                //(row 5):FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a + 4*rs_a));
                ymm11 = _mm256_fnmadd_pd(ymm1, ymm13, ymm11);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a + 3*rs_a));
                ymm9 = _mm256_fnmadd_pd(ymm1, ymm13, ymm9);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a + 2*rs_a));
                ymm7 = _mm256_fnmadd_pd(ymm1, ymm13, ymm7);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a + 1*rs_a));
                ymm5 = _mm256_fnmadd_pd(ymm1, ymm13, ymm5);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm13, ymm3);

                ymm11 = DTRSM_SMALL_DIV_OR_SCALE(ymm11, ymm0);

                //extract a33
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

                //(row 4):FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*cs_a + 3*rs_a));
                ymm9 = _mm256_fnmadd_pd(ymm1, ymm11, ymm9);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*cs_a + 2*rs_a));
                ymm7 = _mm256_fnmadd_pd(ymm1, ymm11, ymm7);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*cs_a + 1*rs_a));
                ymm5 = _mm256_fnmadd_pd(ymm1, ymm11, ymm5);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 4*cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm11, ymm3);

                ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);

                //extract a22
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

                //(Row 3): FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 2*rs_a));
                ymm7 = _mm256_fnmadd_pd(ymm1, ymm9, ymm7);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 1*rs_a));
                ymm5 = _mm256_fnmadd_pd(ymm1, ymm9, ymm5);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm9, ymm3);

                ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

                //extract a11
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

                //(row 2):FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 1*rs_a));
                ymm5 = _mm256_fnmadd_pd(ymm1, ymm7, ymm5);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm7, ymm3);

                ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

                //extract a00
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack ));

                //(Row 1): FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);

                ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

                ymm0 = _mm256_loadu_pd((double const *)b11);
                ymm3 = _mm256_blend_pd(ymm0, ymm3, 0x01);
                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
                ymm5 = _mm256_blend_pd(ymm0, ymm5, 0x01);
                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*2));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
                ymm7 = _mm256_blend_pd(ymm0, ymm7, 0x01);
                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*3));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
                ymm9 = _mm256_blend_pd(ymm0, ymm9, 0x01);
                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*4));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
                ymm11 = _mm256_blend_pd(ymm0, ymm11, 0x01);
                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*5));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
                ymm13 = _mm256_blend_pd(ymm0, ymm13, 0x01);

                _mm256_storeu_pd((double *)b11, ymm3);
                _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
                _mm256_storeu_pd((double *)(b11 + cs_b*2), ymm7);
                _mm256_storeu_pd((double *)(b11 + cs_b*3), ymm9);
                _mm256_storeu_pd((double *)(b11 + cs_b*4), ymm11);
                _mm256_storeu_pd((double *)(b11 + cs_b*5), ymm13);

                m_remainder -=1;
            }
        }
    }

    dim_t n_remainder = j + d_nr;

    /*
    Reminder cases starts here:
    a. Similar logic and code flow used in computing full block (6x8)
       above holds for reminder cases too.
    */

    if(n_remainder >= 4)
    {
        a01 = L + (n_remainder - 4)*rs_a + n_remainder*cs_a;                     //pointer to block of A to be used in GEMM
        a11 = L + (n_remainder - 4)*cs_a + (n_remainder - 4)*rs_a;                 //pointer to block of A to be used for TRSM

        double *ptr_a10_dup = D_A_pack;

        dim_t p_lda = (n-n_remainder); // packed leading dimension
        // perform copy of A to packed buffer D_A_pack

        if(transa)
        {
            for(dim_t x =0;x < p_lda;x+=d_nr)
            {
                ymm0 = _mm256_loadu_pd((double const *)(a01));
                ymm1 = _mm256_loadu_pd((double const *)(a01 + cs_a));
                ymm2 = _mm256_loadu_pd((double const *)(a01 + cs_a * 2));
                ymm3 = _mm256_loadu_pd((double const *)(a01 + cs_a * 3));

                ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
                ymm5 = _mm256_unpacklo_pd(ymm2, ymm3);

                ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);

                ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
                ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);

                ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

                _mm256_storeu_pd((double *)(ptr_a10_dup), ymm6);
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda), ymm7);
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*2), ymm8);
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*3), ymm9);

                ymm0 = _mm256_loadu_pd((double const *)(a01 + cs_a * 4));
                ymm1 = _mm256_loadu_pd((double const *)(a01 + cs_a * 5));

                ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
                ymm5 = _mm256_broadcast_sd((double const *)&zero);

                ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);

                ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
                ymm1 = _mm256_broadcast_sd((double const *)&zero);

                ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

                _mm_storeu_pd((double *)(ptr_a10_dup + 4), _mm256_extractf128_pd(ymm6,0));
                _mm_storeu_pd((double *)(ptr_a10_dup + 4 + p_lda), _mm256_extractf128_pd(ymm7,0));
                _mm_storeu_pd((double *)(ptr_a10_dup + 4 + p_lda*2), _mm256_extractf128_pd(ymm8,0));
                _mm_storeu_pd((double *)(ptr_a10_dup + 4 + p_lda*3), _mm256_extractf128_pd(ymm9,0));

                a01 += d_nr*cs_a;
                ptr_a10_dup += d_nr;
            }
        }
        else
        {
            dim_t loop_count = (n-n_remainder)/4;

            for(dim_t x =0;x < loop_count;x++)
            {
                ymm15 = _mm256_loadu_pd((double const *)(a01 + rs_a * 0 + x*4));
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda * 0 + x*4), ymm15);
                ymm15 = _mm256_loadu_pd((double const *)(a01 + rs_a * 1 + x*4));
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda * 1 + x*4), ymm15);
                ymm15 = _mm256_loadu_pd((double const *)(a01 + rs_a * 2 + x*4));
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda * 2 + x*4), ymm15);
                ymm15 = _mm256_loadu_pd((double const *)(a01 + rs_a * 3 + x*4));
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda * 3 + x*4), ymm15);
            }

            dim_t remainder_loop_count = p_lda - loop_count*4;

            __m128d xmm0;
            if(remainder_loop_count != 0)
            {
                xmm0 = _mm_loadu_pd((double const *)(a01 + rs_a * 0 + loop_count*4));
                _mm_storeu_pd((double *)(ptr_a10_dup + p_lda * 0 + loop_count*4), xmm0);
                xmm0 = _mm_loadu_pd((double const *)(a01 + rs_a * 1 + loop_count*4));
                _mm_storeu_pd((double *)(ptr_a10_dup + p_lda * 1 + loop_count*4), xmm0);
                xmm0 = _mm_loadu_pd((double const *)(a01 + rs_a * 2 + loop_count*4));
                _mm_storeu_pd((double *)(ptr_a10_dup + p_lda * 2 + loop_count*4), xmm0);
                xmm0 = _mm_loadu_pd((double const *)(a01 + rs_a * 3 + loop_count*4));
                _mm_storeu_pd((double *)(ptr_a10_dup + p_lda * 3 + loop_count*4), xmm0);
            }
        }

        ymm4 = _mm256_broadcast_sd((double const *)&ones);
        if(!is_unitdiag)
        {
            if(transa)
            {
                //broadcast diagonal elements of A11
                ymm0 = _mm256_broadcast_sd((double const *)(a11));
                ymm1 = _mm256_broadcast_sd((double const *)(a11+ cs_a*1 + 1));
                ymm2 = _mm256_broadcast_sd((double const *)(a11+ cs_a*2 + 2));
                ymm3 = _mm256_broadcast_sd((double const *)(a11+ cs_a*3 + 3));
            }
            else
            {
                //broadcast diagonal elements of A11
                ymm0 = _mm256_broadcast_sd((double const *)(a11));
                ymm1 = _mm256_broadcast_sd((double const *)(a11+ rs_a*1 + 1));
                ymm2 = _mm256_broadcast_sd((double const *)(a11+ rs_a*2 + 2));
                ymm3 = _mm256_broadcast_sd((double const *)(a11+ rs_a*3 + 3));
            }

            ymm0 = _mm256_unpacklo_pd(ymm0, ymm1);
            ymm1 = _mm256_unpacklo_pd(ymm2, ymm3);

            ymm1 = _mm256_blend_pd(ymm0, ymm1, 0x0C);
            #ifdef BLIS_DISABLE_TRSM_PREINVERSION
            ymm4 = ymm1;
            #endif
            #ifdef BLIS_ENABLE_TRSM_PREINVERSION
            ymm4 = _mm256_div_pd(ymm4, ymm1);
            #endif
        }
        _mm256_storeu_pd((double *)(d11_pack), ymm4);

        for(i = (m-d_mr); (i+1) > 0; i -= d_mr)     //loop along 'M' direction
        {
            a01 = D_A_pack;
            a11 = L + (n_remainder - 4)*cs_a + (n_remainder - 4)*rs_a;                 //pointer to block of A to be used for TRSM
            b10 = B + i + (n_remainder)*cs_b;                          //pointer to block of B to be used in GEMM
            b11 = B + (i) + (n_remainder - 4)*cs_b;                 //pointer to block of B to be used for TRSM

            k_iter = (n-n_remainder);                    //number of GEMM operations to be done(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_4nx8m(a01,b10,cs_b,p_lda,k_iter)

            BLIS_PRE_DTRSM_SMALL_4x8(AlphaVal,b11,cs_b)

            ///implement TRSM///

            //extract a33
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

            ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);
            ymm10 = DTRSM_SMALL_DIV_OR_SCALE(ymm10, ymm0);

            //extract a22
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            //(Row 3): FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 2*rs_a));

            ymm7 = _mm256_fnmadd_pd(ymm1, ymm9, ymm7);
            ymm8 = _mm256_fnmadd_pd(ymm1, ymm10, ymm8);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 1*rs_a));

            ymm5 = _mm256_fnmadd_pd(ymm1, ymm9, ymm5);
            ymm6 = _mm256_fnmadd_pd(ymm1, ymm10, ymm6);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a));

            ymm3 = _mm256_fnmadd_pd(ymm1, ymm9, ymm3);
            ymm4 = _mm256_fnmadd_pd(ymm1, ymm10, ymm4);

            ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);
            ymm8 = DTRSM_SMALL_DIV_OR_SCALE(ymm8, ymm0);

            //extract a11
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(row 2):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 1*rs_a));

            ymm5 = _mm256_fnmadd_pd(ymm1, ymm7, ymm5);
            ymm6 = _mm256_fnmadd_pd(ymm1, ymm8, ymm6);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a));

            ymm3 = _mm256_fnmadd_pd(ymm1, ymm7, ymm3);
            ymm4 = _mm256_fnmadd_pd(ymm1, ymm8, ymm4);

            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);
            ymm6 = DTRSM_SMALL_DIV_OR_SCALE(ymm6, ymm0);

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack ));

            //(Row 1): FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));

            ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);
            ymm4 = _mm256_fnmadd_pd(ymm1, ymm6, ymm4);

            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);
            ymm4 = DTRSM_SMALL_DIV_OR_SCALE(ymm4, ymm0);

            _mm256_storeu_pd((double *)b11, ymm3);
            _mm256_storeu_pd((double *)(b11 + 4), ymm4);
            _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
            _mm256_storeu_pd((double *)(b11 + cs_b + 4), ymm6);
            _mm256_storeu_pd((double *)(b11 + cs_b*2), ymm7);
            _mm256_storeu_pd((double *)(b11 + cs_b*2 + 4), ymm8);
            _mm256_storeu_pd((double *)(b11 + cs_b*3), ymm9);
            _mm256_storeu_pd((double *)(b11 + cs_b*3 + 4), ymm10);
        }

        dim_t m_remainder = i + d_mr;
        if(m_remainder >= 4)
        {
            a01 = D_A_pack;
            a11 = L + (n_remainder - 4)*cs_a + (n_remainder - 4)*rs_a;                 //pointer to block of A to be used for TRSM
            b10 = B + (m_remainder - 4) + (n_remainder)*cs_b;                          //pointer to block of B to be used in GEMM
            b11 = B + (m_remainder - 4) + (n_remainder - 4)*cs_b;                 //pointer to block of B to be used for TRSM

            k_iter = (n-n_remainder);                    //number of GEMM operations to be done(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_4nx4m(a01,b10,cs_b,p_lda,k_iter)

            ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);         //register to hold alpha

            ymm0 = _mm256_loadu_pd((double const *)b11);                            //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
            ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);      //B11[0-3][0] * alpha -= ymm0

            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);    //B11[0-3][1] * alpha-= ymm2

            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*2));        //B11[0][2] B11[1][2] B11[2][2] B11[3][2]
            ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);    //B11[0-3][2] * alpha -= ymm4

            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*3));        //B11[0][3] B11[1][3] B11[2][3] B11[3][3]
            ymm9 = _mm256_fmsub_pd(ymm0, ymm15, ymm9);    //B11[0-3][3] * alpha -= ymm6

            ///implement TRSM///

            //extract a33
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 3));
            ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);

            //extract a22
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            //(Row 3): FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 2*rs_a));
            ymm7 = _mm256_fnmadd_pd(ymm1, ymm9, ymm7);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 1*rs_a));
            ymm5 = _mm256_fnmadd_pd(ymm1, ymm9, ymm5);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a));
            ymm3 = _mm256_fnmadd_pd(ymm1, ymm9, ymm3);

            ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

            //extract a11
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(row 2):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 1*rs_a));
            ymm5 = _mm256_fnmadd_pd(ymm1, ymm7, ymm5);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a));
            ymm3 = _mm256_fnmadd_pd(ymm1, ymm7, ymm3);

            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack ));

            //(Row 1): FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));
            ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);

            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

            _mm256_storeu_pd((double *)b11, ymm3);
            _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
            _mm256_storeu_pd((double *)(b11 + cs_b*2), ymm7);
            _mm256_storeu_pd((double *)(b11 + cs_b*3), ymm9);

            m_remainder -=4;
        }

        if(m_remainder)
        {
            if(3 == m_remainder)
            {
                a01 = D_A_pack;
                a11 = L + (n_remainder - 4)*cs_a + (n_remainder - 4)*rs_a;                 //pointer to block of A to be used for TRSM
                b10 = B + (m_remainder - 3) + (n_remainder)*cs_b;                          //pointer to block of B to be used in GEMM
                b11 = B + (m_remainder - 3) + (n_remainder - 4)*cs_b;                 //pointer to block of B to be used for TRSM

                k_iter = (n-n_remainder);                    //number of GEMM operations to be done(in blocks of 4x4)

                /*Fill zeros into ymm registers used in gemm accumulations */
                BLIS_SET_YMM_REG_ZEROS

                ///GEMM implementation starts///
                BLIS_DTRSM_SMALL_GEMM_4nx4m(a01,b10,cs_b,p_lda,k_iter)

                ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);         //register to hold alpha

                ymm0 = _mm256_loadu_pd((double const *)b11);                            //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
                ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);      //B11[0-3][0] * alpha -= ymm0

                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
                ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);    //B11[0-3][1] * alpha-= ymm2

                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*2));        //B11[0][2] B11[1][2] B11[2][2] B11[3][2]
                ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);    //B11[0-3][2] * alpha -= ymm4

                xmm5 = _mm_loadu_pd((double const*)(b11 + cs_b * 3));
                ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b*3 + 2));
                ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);
                ymm9 = _mm256_fmsub_pd(ymm0, ymm15, ymm9);    //B11[0-3][3] * alpha -= ymm6

                ///implement TRSM///

                //extract a33
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 3));
                ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);

                //extract a22
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

                //(Row 3): FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 2*rs_a));
                ymm7 = _mm256_fnmadd_pd(ymm1, ymm9, ymm7);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 1*rs_a));
                ymm5 = _mm256_fnmadd_pd(ymm1, ymm9, ymm5);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm9, ymm3);

                ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

                //extract a11
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

                //(row 2):FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 1*rs_a));
                ymm5 = _mm256_fnmadd_pd(ymm1, ymm7, ymm5);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm7, ymm3);

                ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

                //extract a00
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack ));

                //(Row 1): FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);

                ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

                ymm0 = _mm256_loadu_pd((double const *)b11);
                ymm3 = _mm256_blend_pd(ymm0, ymm3, 0x07);
                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
                ymm5 = _mm256_blend_pd(ymm0, ymm5, 0x07);
                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*2));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
                ymm7 = _mm256_blend_pd(ymm0, ymm7, 0x07);
                xmm5 = _mm_loadu_pd((double const*)(b11 + cs_b * 3));
                ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b*3 + 2));
                ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);                      //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
                ymm9 = _mm256_blend_pd(ymm0, ymm9, 0x07);

                _mm256_storeu_pd((double *)b11, ymm3);
                _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
                _mm256_storeu_pd((double *)(b11 + cs_b*2), ymm7);
                xmm5 = _mm256_extractf128_pd(ymm9, 0);
                _mm_storeu_pd((double *)(b11 + cs_b * 3),xmm5);
                _mm_storel_pd((b11 + cs_b * 3 + 2), _mm256_extractf128_pd(ymm9, 1));

                m_remainder -=3;
            }
            else if(2 == m_remainder)
            {
                a01 = D_A_pack;
                a11 = L + (n_remainder - 4)*cs_a + (n_remainder - 4)*rs_a;                 //pointer to block of A to be used for TRSM
                b10 = B + (m_remainder - 2) + (n_remainder)*cs_b;                          //pointer to block of B to be used in GEMM
                b11 = B + (m_remainder - 2) + (n_remainder - 4)*cs_b;                 //pointer to block of B to be used for TRSM

                k_iter = (n-n_remainder);                    //number of GEMM operations to be done(in blocks of 4x4)

                /*Fill zeros into ymm registers used in gemm accumulations */
                BLIS_SET_YMM_REG_ZEROS

                ///GEMM implementation starts///
                BLIS_DTRSM_SMALL_GEMM_4nx4m(a01,b10,cs_b,p_lda,k_iter)

                ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);         //register to hold alpha

                ymm0 = _mm256_loadu_pd((double const *)b11);                            //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
                ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);      //B11[0-3][0] * alpha -= ymm0

                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
                ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);    //B11[0-3][1] * alpha-= ymm2

                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*2));        //B11[0][2] B11[1][2] B11[2][2] B11[3][2]
                ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);    //B11[0-3][2] * alpha -= ymm4

                xmm5 = _mm_loadu_pd((double const*)(b11 + cs_b * 3));
                ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);
                ymm9 = _mm256_fmsub_pd(ymm0, ymm15, ymm9);    //B11[0-3][3] * alpha -= ymm6

                ///implement TRSM///

                //extract a33
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 3));
                ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);

                //extract a22
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

                //(Row 3): FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 2*rs_a));
                ymm7 = _mm256_fnmadd_pd(ymm1, ymm9, ymm7);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 1*rs_a));
                ymm5 = _mm256_fnmadd_pd(ymm1, ymm9, ymm5);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm9, ymm3);

                ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

                //extract a11
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

                //(row 2):FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 1*rs_a));
                ymm5 = _mm256_fnmadd_pd(ymm1, ymm7, ymm5);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm7, ymm3);

                ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

                //extract a00
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack ));

                //(Row 1): FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);

                ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

                ymm0 = _mm256_loadu_pd((double const *)b11);
                ymm3 = _mm256_blend_pd(ymm0, ymm3, 0x03);
                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
                ymm5 = _mm256_blend_pd(ymm0, ymm5, 0x03);
                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*2));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
                ymm7 = _mm256_blend_pd(ymm0, ymm7, 0x03);
                xmm5 = _mm_loadu_pd((double const*)(b11 + cs_b * 3));
                ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);
                ymm9 = _mm256_blend_pd(ymm0, ymm9, 0x03);

                _mm256_storeu_pd((double *)b11, ymm3);
                _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
                _mm256_storeu_pd((double *)(b11 + cs_b*2), ymm7);
                xmm5 = _mm256_extractf128_pd(ymm9, 0);
                _mm_storeu_pd((double *)(b11 + cs_b * 3),xmm5);

                m_remainder -=2;
            }
            else if (1 == m_remainder)
            {
                a01 = D_A_pack;
                a11 = L + (n_remainder - 4)*cs_a + (n_remainder - 4)*rs_a;                 //pointer to block of A to be used for TRSM
                b10 = B + (m_remainder - 1) + (n_remainder)*cs_b;                          //pointer to block of B to be used in GEMM
                b11 = B + (m_remainder - 1) + (n_remainder - 4)*cs_b;                 //pointer to block of B to be used for TRSM

                k_iter = (n-n_remainder);                    //number of GEMM operations to be done(in blocks of 4x4)

                /*Fill zeros into ymm registers used in gemm accumulations */
                BLIS_SET_YMM_REG_ZEROS

                ///GEMM implementation starts///
                BLIS_DTRSM_SMALL_GEMM_4nx4m(a01,b10,cs_b,p_lda,k_iter)

                ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);         //register to hold alpha

                ymm0 = _mm256_broadcast_sd((double const *)b11);                            //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
                ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);      //B11[0-3][0] * alpha -= ymm0

                ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
                ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);    //B11[0-3][1] * alpha-= ymm2

                ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b*2));        //B11[0][2] B11[1][2] B11[2][2] B11[3][2]
                ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);    //B11[0-3][2] * alpha -= ymm4

                ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b*3));        //B11[0][3] B11[1][3] B11[2][3] B11[3][3]
                ymm9 = _mm256_fmsub_pd(ymm0, ymm15, ymm9);    //B11[0-3][3] * alpha -= ymm6

                ///implement TRSM///

                //extract a33
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 3));
                ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);

                //extract a22
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

                //(Row 3): FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 2*rs_a));
                ymm7 = _mm256_fnmadd_pd(ymm1, ymm9, ymm7);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 1*rs_a));
                ymm5 = _mm256_fnmadd_pd(ymm1, ymm9, ymm5);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm9, ymm3);

                ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

                //extract a11
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

                //(row 2):FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 1*rs_a));
                ymm5 = _mm256_fnmadd_pd(ymm1, ymm7, ymm5);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm7, ymm3);

                ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

                //extract a00
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack ));

                //(Row 1): FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);

                ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);
                ymm0 = _mm256_broadcast_sd((double const *)b11);
                ymm3 = _mm256_blend_pd(ymm0, ymm3, 0x01);
                ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
                ymm5 = _mm256_blend_pd(ymm0, ymm5, 0x01);
                ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b*2));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
                ymm7 = _mm256_blend_pd(ymm0, ymm7, 0x01);
                ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b*3));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
                ymm9 = _mm256_blend_pd(ymm0, ymm9, 0x01);

                _mm_storel_pd((b11 + cs_b * 0), _mm256_extractf128_pd(ymm3, 0));
                _mm_storel_pd((b11 + cs_b * 1), _mm256_extractf128_pd(ymm5, 0));
                _mm_storel_pd((b11 + cs_b * 2), _mm256_extractf128_pd(ymm7, 0));
                _mm_storel_pd((b11 + cs_b * 3), _mm256_extractf128_pd(ymm9, 0));

                m_remainder -=1;
            }
        }
        n_remainder -= 4;
    }

    if(n_remainder == 3)
    {
        a01 = L + (n_remainder - 3)*rs_a + n_remainder*cs_a;                     //pointer to block of A to be used in GEMM
        a11 = L + (n_remainder - 3)*cs_a + (n_remainder - 3)*rs_a;                 //pointer to block of A to be used for TRSM

        double *ptr_a10_dup = D_A_pack;

        dim_t p_lda = (n-n_remainder); // packed leading dimension
        // perform copy of A to packed buffer D_A_pack

        if(transa)
        {
            for(dim_t x =0;x < p_lda;x+=d_nr)
            {
                ymm0 = _mm256_loadu_pd((double const *)(a01));
                ymm1 = _mm256_loadu_pd((double const *)(a01 + cs_a));
                ymm2 = _mm256_loadu_pd((double const *)(a01 + cs_a * 2));
                ymm3 = _mm256_loadu_pd((double const *)(a01 + cs_a * 3));

                ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
                ymm5 = _mm256_unpacklo_pd(ymm2, ymm3);

                ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);

                ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
                ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);

                ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

                _mm256_storeu_pd((double *)(ptr_a10_dup), ymm6);
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda), ymm7);
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*2), ymm8);
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*3), ymm9);

                ymm0 = _mm256_loadu_pd((double const *)(a01 + cs_a * 4));
                ymm1 = _mm256_loadu_pd((double const *)(a01 + cs_a * 5));

                ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
                ymm5 = _mm256_broadcast_sd((double const *)&zero);

                ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);

                ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
                ymm1 = _mm256_broadcast_sd((double const *)&zero);

                ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

                _mm_storeu_pd((double *)(ptr_a10_dup + 4), _mm256_extractf128_pd(ymm6,0));
                _mm_storeu_pd((double *)(ptr_a10_dup + 4 + p_lda), _mm256_extractf128_pd(ymm7,0));
                _mm_storeu_pd((double *)(ptr_a10_dup + 4 + p_lda*2), _mm256_extractf128_pd(ymm8,0));
                _mm_storeu_pd((double *)(ptr_a10_dup + 4 + p_lda*3), _mm256_extractf128_pd(ymm9,0));

                a01 += d_nr*cs_a;
                ptr_a10_dup += d_nr;
            }
        }
        else
        {
            dim_t loop_count = (n-n_remainder)/4;

            for(dim_t x =0;x < loop_count;x++)
            {
                ymm15 = _mm256_loadu_pd((double const *)(a01 + rs_a * 0 + x*4));
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda * 0 + x*4), ymm15);
                ymm15 = _mm256_loadu_pd((double const *)(a01 + rs_a * 1 + x*4));
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda * 1 + x*4), ymm15);
                ymm15 = _mm256_loadu_pd((double const *)(a01 + rs_a * 2 + x*4));
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda * 2 + x*4), ymm15);
            }

            dim_t remainder_loop_count = p_lda - loop_count*4;

            __m128d xmm0;
            if(remainder_loop_count != 0)
            {
                xmm0 = _mm_loadu_pd((double const *)(a01 + rs_a * 0 + loop_count*4));
                _mm_storeu_pd((double *)(ptr_a10_dup + p_lda * 0 + loop_count*4), xmm0);
                xmm0 = _mm_loadu_pd((double const *)(a01 + rs_a * 1 + loop_count*4));
                _mm_storeu_pd((double *)(ptr_a10_dup + p_lda * 1 + loop_count*4), xmm0);
                xmm0 = _mm_loadu_pd((double const *)(a01 + rs_a * 2 + loop_count*4));
                _mm_storeu_pd((double *)(ptr_a10_dup + p_lda * 2 + loop_count*4), xmm0);
            }
        }

        ymm4 = _mm256_broadcast_sd((double const *)&ones);
        if(!is_unitdiag)
        {
            if(transa)
            {
                //broadcast diagonal elements of A11
                ymm0 = _mm256_broadcast_sd((double const *)(a11));
                ymm1 = _mm256_broadcast_sd((double const *)(a11+ cs_a*1 + 1));
                ymm2 = _mm256_broadcast_sd((double const *)(a11+ cs_a*2 + 2));
                ymm3 = _mm256_broadcast_sd((double const *)&ones);
            }
            else
            {
                ymm0 = _mm256_broadcast_sd((double const *)(a11));
                ymm1 = _mm256_broadcast_sd((double const *)(a11+ rs_a*1 + 1));
                ymm2 = _mm256_broadcast_sd((double const *)(a11+ rs_a*2 + 2));
                ymm3 = _mm256_broadcast_sd((double const *)&ones);
            }

            ymm0 = _mm256_unpacklo_pd(ymm0, ymm1);
            ymm1 = _mm256_unpacklo_pd(ymm2, ymm3);

            ymm1 = _mm256_blend_pd(ymm0, ymm1, 0x0C);
            #ifdef BLIS_DISABLE_TRSM_PREINVERSION
            ymm4 = ymm1;
            #endif
            #ifdef BLIS_ENABLE_TRSM_PREINVERSION
            ymm4 = _mm256_div_pd(ymm4, ymm1);
            #endif
        }
        _mm256_storeu_pd((double *)(d11_pack), ymm4);

        for(i = (m-d_mr); (i+1) > 0; i -= d_mr)     //loop along 'M' direction
        {
            a01 = D_A_pack;
            a11 = L + (n_remainder - 3)*cs_a + (n_remainder - 3)*rs_a;                 //pointer to block of A to be used for TRSM
            b10 = B + i + (n_remainder)*cs_b;                          //pointer to block of B to be used in GEMM
            b11 = B + (i) + (n_remainder - 3)*cs_b;                 //pointer to block of B to be used for TRSM

            k_iter = (n-n_remainder);                    //number of GEMM operations to be done(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_3nx8m(a01,b10,cs_b,p_lda,k_iter)

            ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);

            ymm0 = _mm256_loadu_pd((double const *)b11);    //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
            ymm1 = _mm256_loadu_pd((double const *)(b11 + 4)); //B11[4][0] B11[5][0] B11[6][0] B11[7][0]

            ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);  //B11[0-3][0] * alpha -= ymm0
            ymm4 = _mm256_fmsub_pd(ymm1, ymm15, ymm4);  //B11[4-7][0] * alpha-= ymm1

            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b + 4));           //B11[4][1] B11[5][1] B11[6][1] B11[7][1]

            ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);    //B11[0-3][1] * alpha-= ymm2
            ymm6 = _mm256_fmsub_pd(ymm1, ymm15, ymm6);    //B11[4-7][1] * alpha -= ymm3

            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*2));        //B11[0][2] B11[1][2] B11[2][2] B11[3][2]
            ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b*2 + 4)); //B11[4][2] B11[5][2] B11[6][2] B11[7][2]

            ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);    //B11[0-3][2] * alpha -= ymm4
            ymm8 = _mm256_fmsub_pd(ymm1, ymm15, ymm8);    //B11[4-7][2] * alpha -= ymm5

            ///implement TRSM///

            //extract a22
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);
            ymm8 = DTRSM_SMALL_DIV_OR_SCALE(ymm8, ymm0);

            //extract a11
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(row 2):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 1*rs_a));

            ymm5 = _mm256_fnmadd_pd(ymm1, ymm7, ymm5);
            ymm6 = _mm256_fnmadd_pd(ymm1, ymm8, ymm6);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a));

            ymm3 = _mm256_fnmadd_pd(ymm1, ymm7, ymm3);
            ymm4 = _mm256_fnmadd_pd(ymm1, ymm8, ymm4);

            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);
            ymm6 = DTRSM_SMALL_DIV_OR_SCALE(ymm6, ymm0);

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack ));

            //(Row 1): FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));

            ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);
            ymm4 = _mm256_fnmadd_pd(ymm1, ymm6, ymm4);

            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);
            ymm4 = DTRSM_SMALL_DIV_OR_SCALE(ymm4, ymm0);

            _mm256_storeu_pd((double *)b11, ymm3);
            _mm256_storeu_pd((double *)(b11 + 4), ymm4);
            _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
            _mm256_storeu_pd((double *)(b11 + cs_b + 4), ymm6);
            _mm256_storeu_pd((double *)(b11 + cs_b*2), ymm7);
            _mm256_storeu_pd((double *)(b11 + cs_b*2 + 4), ymm8);
        }

        dim_t m_remainder = i + d_mr;
        if(m_remainder >= 4)
        {
            a01 = D_A_pack;
            a11 = L + (n_remainder - 3)*cs_a + (n_remainder - 3)*rs_a;                 //pointer to block of A to be used for TRSM
            b10 = B + (m_remainder - 4) + (n_remainder)*cs_b;                          //pointer to block of B to be used in GEMM
            b11 = B + (m_remainder - 4) + (n_remainder - 3)*cs_b;                 //pointer to block of B to be used for TRSM

            k_iter = (n-n_remainder);                    //number of GEMM operations to be done(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_3nx4m(a01,b10,cs_b,p_lda,k_iter)

            ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);         //register to hold alpha

            ymm0 = _mm256_loadu_pd((double const *)b11);                            //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
            ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);      //B11[0-3][0] * alpha -= ymm0

            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);    //B11[0-3][1] * alpha-= ymm2

            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b*2));        //B11[0][2] B11[1][2] B11[2][2] B11[3][2]
            ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);    //B11[0-3][2] * alpha -= ymm4

            ///implement TRSM///

            //extract a22
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));
            ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

            //extract a11
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(row 2):FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 1*rs_a));
            ymm5 = _mm256_fnmadd_pd(ymm1, ymm7, ymm5);

            ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a));
            ymm3 = _mm256_fnmadd_pd(ymm1, ymm7, ymm3);

            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack ));

            //(Row 1): FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));
            ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);

            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

            _mm256_storeu_pd((double *)b11, ymm3);
            _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
            _mm256_storeu_pd((double *)(b11 + cs_b*2), ymm7);

            m_remainder -=4;
        }

        if(m_remainder)
        {
            if(3 == m_remainder)
            {
                a01 = D_A_pack;
                a11 = L + (n_remainder - 3)*cs_a + (n_remainder - 3)*rs_a;                 //pointer to block of A to be used for TRSM
                b10 = B + (m_remainder - 3) + (n_remainder)*cs_b;                          //pointer to block of B to be used in GEMM
                b11 = B + (m_remainder - 3) + (n_remainder - 3)*cs_b;                 //pointer to block of B to be used for TRSM

                k_iter = (n-n_remainder);                    //number of GEMM operations to be done(in blocks of 4x4)

                /*Fill zeros into ymm registers used in gemm accumulations */
                BLIS_SET_YMM_REG_ZEROS

                ///GEMM implementation starts///
                BLIS_DTRSM_SMALL_GEMM_3nx4m(a01,b10,cs_b,p_lda,k_iter)

                BLIS_PRE_DTRSM_SMALL_3N_3M(AlphaVal,b11,cs_b)

                ///implement TRSM///
                //extract a22
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));
                ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

                //extract a11
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

                //(row 2):FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 1*rs_a));
                ymm5 = _mm256_fnmadd_pd(ymm1, ymm7, ymm5);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm7, ymm3);

                ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

                //extract a00
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack ));

                //(Row 1): FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);

                ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

                BLIS_POST_DTRSM_SMALL_3N_3M(b11,cs_b)

                m_remainder -=3;
            }
            else if(2 == m_remainder)
            {
                a01 = D_A_pack;
                a11 = L + (n_remainder - 3)*cs_a + (n_remainder - 3)*rs_a;                 //pointer to block of A to be used for TRSM
                b10 = B + (m_remainder - 2) + (n_remainder)*cs_b;                          //pointer to block of B to be used in GEMM
                b11 = B + (m_remainder - 2) + (n_remainder - 3)*cs_b;                 //pointer to block of B to be used for TRSM

                k_iter = (n-n_remainder);                    //number of GEMM operations to be done(in blocks of 4x4)

                /*Fill zeros into ymm registers used in gemm accumulations */
                BLIS_SET_YMM_REG_ZEROS

                ///GEMM implementation starts///
                BLIS_DTRSM_SMALL_GEMM_3nx4m(a01,b10,cs_b,p_lda,k_iter)

                BLIS_PRE_DTRSM_SMALL_3N_2M(AlphaVal,b11,cs_b)

                ///implement TRSM///

                //extract a22
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));
                ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

                //extract a11
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

                //(row 2):FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 1*rs_a));
                ymm5 = _mm256_fnmadd_pd(ymm1, ymm7, ymm5);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm7, ymm3);

                ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

                //extract a00
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack ));

                //(Row 1): FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);

                ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

                BLIS_POST_DTRSM_SMALL_3N_2M(b11,cs_b)

                m_remainder -=2;
            }
            else if (1 == m_remainder)
            {
                a01 = D_A_pack;
                a11 = L + (n_remainder - 3)*cs_a + (n_remainder - 3)*rs_a;                 //pointer to block of A to be used for TRSM
                b10 = B + (m_remainder - 1) + (n_remainder)*cs_b;                          //pointer to block of B to be used in GEMM
                b11 = B + (m_remainder - 1) + (n_remainder - 3)*cs_b;                 //pointer to block of B to be used for TRSM

                k_iter = (n-n_remainder);                    //number of GEMM operations to be done(in blocks of 4x4)

                /*Fill zeros into ymm registers used in gemm accumulations */
                BLIS_SET_YMM_REG_ZEROS

                ///GEMM implementation starts///
                BLIS_DTRSM_SMALL_GEMM_3nx4m(a01,b10,cs_b,p_lda,k_iter)

                BLIS_PRE_DTRSM_SMALL_3N_1M(AlphaVal,b11,cs_b)

                ///implement TRSM///

                //extract a22
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));
                ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

                //extract a11
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

                //(row 2):FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 1*rs_a));
                ymm5 = _mm256_fnmadd_pd(ymm1, ymm7, ymm5);

                ymm1 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm7, ymm3);

                ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

                //extract a00
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack ));

                //(Row 1): FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);

                ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

                BLIS_POST_DTRSM_SMALL_3N_1M(b11,cs_b)

                m_remainder -=1;
            }
        }
        n_remainder -= 3;
    }
    else if(n_remainder == 2)
    {
        a01 = L + (n_remainder - 2)*rs_a + n_remainder*cs_a;                     //pointer to block of A to be used in GEMM
        a11 = L + (n_remainder - 2)*cs_a + (n_remainder - 2)*rs_a;                 //pointer to block of A to be used for TRSM

        double *ptr_a10_dup = D_A_pack;

        dim_t p_lda = (n-n_remainder); // packed leading dimension
        // perform copy of A to packed buffer D_A_pack

        if(transa)
        {
            for(dim_t x =0;x < p_lda;x+=d_nr)
            {
                ymm0 = _mm256_loadu_pd((double const *)(a01));
                ymm1 = _mm256_loadu_pd((double const *)(a01 + cs_a));
                ymm2 = _mm256_loadu_pd((double const *)(a01 + cs_a * 2));
                ymm3 = _mm256_loadu_pd((double const *)(a01 + cs_a * 3));

                ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
                ymm5 = _mm256_unpacklo_pd(ymm2, ymm3);

                ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);

                ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
                ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);

                ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

                _mm256_storeu_pd((double *)(ptr_a10_dup), ymm6);
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda), ymm7);
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*2), ymm8);
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*3), ymm9);

                ymm0 = _mm256_loadu_pd((double const *)(a01 + cs_a * 4));
                ymm1 = _mm256_loadu_pd((double const *)(a01 + cs_a * 5));

                ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
                ymm5 = _mm256_broadcast_sd((double const *)&zero);

                ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);

                ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
                ymm1 = _mm256_broadcast_sd((double const *)&zero);

                ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

                _mm_storeu_pd((double *)(ptr_a10_dup + 4), _mm256_extractf128_pd(ymm6,0));
                _mm_storeu_pd((double *)(ptr_a10_dup + 4 + p_lda), _mm256_extractf128_pd(ymm7,0));
                _mm_storeu_pd((double *)(ptr_a10_dup + 4 + p_lda*2), _mm256_extractf128_pd(ymm8,0));
                _mm_storeu_pd((double *)(ptr_a10_dup + 4 + p_lda*3), _mm256_extractf128_pd(ymm9,0));

                a01 += d_nr*cs_a;
                ptr_a10_dup += d_nr;
            }
        }
        else
        {
            dim_t loop_count = (n-n_remainder)/4;

            for(dim_t x =0;x < loop_count;x++)
            {
                ymm15 = _mm256_loadu_pd((double const *)(a01 + rs_a * 0 + x*4));
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda * 0 + x*4), ymm15);
                ymm15 = _mm256_loadu_pd((double const *)(a01 + rs_a * 1 + x*4));
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda * 1 + x*4), ymm15);
            }

            dim_t remainder_loop_count = p_lda - loop_count*4;

            __m128d xmm0;
            if(remainder_loop_count != 0)
            {
                xmm0 = _mm_loadu_pd((double const *)(a01 + rs_a * 0 + loop_count*4));
                _mm_storeu_pd((double *)(ptr_a10_dup + p_lda * 0 + loop_count*4), xmm0);
                xmm0 = _mm_loadu_pd((double const *)(a01 + rs_a * 1 + loop_count*4));
                _mm_storeu_pd((double *)(ptr_a10_dup + p_lda * 1 + loop_count*4), xmm0);
            }
        }

        ymm4 = _mm256_broadcast_sd((double const *)&ones);
        if(!is_unitdiag)
        {
            if(transa)
            {
                //broadcast diagonal elements of A11
                ymm0 = _mm256_broadcast_sd((double const *)(a11));
                ymm1 = _mm256_broadcast_sd((double const *)(a11+cs_a*1 + 1));
            }
            else
            {
                //broadcast diagonal elements of A11
                ymm0 = _mm256_broadcast_sd((double const *)(a11));
                ymm1 = _mm256_broadcast_sd((double const *)(a11+rs_a*1 + 1));
            }
            ymm2 = _mm256_broadcast_sd((double const *)&ones);
            ymm3 = _mm256_broadcast_sd((double const *)&ones);

            ymm0 = _mm256_unpacklo_pd(ymm0, ymm1);
            ymm1 = _mm256_unpacklo_pd(ymm2, ymm3);

            ymm1 = _mm256_blend_pd(ymm0, ymm1, 0x0C);
            #ifdef BLIS_DISABLE_TRSM_PREINVERSION
            ymm4 = ymm1;
            #endif
            #ifdef BLIS_ENABLE_TRSM_PREINVERSION
            ymm4 = _mm256_div_pd(ymm4, ymm1);
            #endif
        }
        _mm256_storeu_pd((double *)(d11_pack), ymm4);

        for(i = (m-d_mr); (i+1) > 0; i -= d_mr)     //loop along 'M' direction
        {
            a01 = D_A_pack;
            a11 = L + (n_remainder - 2)*cs_a + (n_remainder - 2)*rs_a;                 //pointer to block of A to be used for TRSM
            b10 = B + i + (n_remainder)*cs_b;                          //pointer to block of B to be used in GEMM
            b11 = B + (i) + (n_remainder - 2)*cs_b;                 //pointer to block of B to be used for TRSM

            k_iter = (n-n_remainder);                    //number of GEMM operations to be done(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_2nx8m(a01,b10,cs_b,p_lda,k_iter)

            ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);

            ymm0 = _mm256_loadu_pd((double const *)b11);                            //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
            ymm1 = _mm256_loadu_pd((double const *)(b11 + 4));                   //B11[4][0] B11[5][0] B11[6][0] B11[7][0]

            ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);      //B11[0-3][0] * alpha -= ymm0
            ymm4 = _mm256_fmsub_pd(ymm1, ymm15, ymm4);      //B11[4-7][0] * alpha-= ymm1

            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b + 4));           //B11[4][1] B11[5][1] B11[6][1] B11[7][1]

            ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);    //B11[0-3][1] * alpha-= ymm2
            ymm6 = _mm256_fmsub_pd(ymm1, ymm15, ymm6);    //B11[4-7][1] * alpha -= ymm3

            ///implement TRSM///

            //extract a11
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);
            ymm6 = DTRSM_SMALL_DIV_OR_SCALE(ymm6, ymm0);

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack ));

            //(Row 1): FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));

            ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);
            ymm4 = _mm256_fnmadd_pd(ymm1, ymm6, ymm4);

            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);
            ymm4 = DTRSM_SMALL_DIV_OR_SCALE(ymm4, ymm0);

            _mm256_storeu_pd((double *)b11, ymm3);
            _mm256_storeu_pd((double *)(b11 + 4), ymm4);
            _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
            _mm256_storeu_pd((double *)(b11 + cs_b + 4), ymm6);
        }

        dim_t m_remainder = i + d_mr;
        if(m_remainder >= 4)
        {
            a01 = D_A_pack;
            a11 = L + (n_remainder - 2)*cs_a + (n_remainder - 2)*rs_a;                 //pointer to block of A to be used for TRSM
            b10 = B + (m_remainder - 4) + (n_remainder)*cs_b;                          //pointer to block of B to be used in GEMM
            b11 = B + (m_remainder - 4) + (n_remainder - 2)*cs_b;                 //pointer to block of B to be used for TRSM

            k_iter = (n-n_remainder);                    //number of GEMM operations to be done(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_2nx4m(a01,b10,cs_b,p_lda,k_iter)

            ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);         //register to hold alpha

            ymm0 = _mm256_loadu_pd((double const *)b11);                            //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
            ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);      //B11[0-3][0] * alpha -= ymm0

            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));                   //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);    //B11[0-3][1] * alpha-= ymm2

            ///implement TRSM///

            //extract a11
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));
            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack ));

            //(Row 1): FMA operations
            ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));
            ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);

            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

            _mm256_storeu_pd((double *)b11, ymm3);
            _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);

            m_remainder -=4;
        }

        if(m_remainder)
        {
            if(3 == m_remainder)
            {
                a01 = D_A_pack;
                a11 = L + (n_remainder - 2)*cs_a + (n_remainder - 2)*rs_a;                 //pointer to block of A to be used for TRSM
                b10 = B + (m_remainder - 3) + (n_remainder)*cs_b;                          //pointer to block of B to be used in GEMM
                b11 = B + (m_remainder - 3) + (n_remainder - 2)*cs_b;                 //pointer to block of B to be used for TRSM

                k_iter = (n-n_remainder);                    //number of GEMM operations to be done(in blocks of 4x4)

                /*Fill zeros into ymm registers used in gemm accumulations */
                BLIS_SET_YMM_REG_ZEROS

                ///GEMM implementation starts///
                BLIS_DTRSM_SMALL_GEMM_2nx4m(a01,b10,cs_b,p_lda,k_iter)

                BLIS_PRE_DTRSM_SMALL_2N_3M(AlphaVal,b11,cs_b)

                ///implement TRSM///

                //extract a11
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));
                ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

                //extract a00
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack ));

                //(Row 1): FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);

                ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

                BLIS_POST_DTRSM_SMALL_2N_3M(b11,cs_b)

                m_remainder -=3;
            }
            else if(2 == m_remainder)
            {
                a01 = D_A_pack;
                a11 = L + (n_remainder - 2)*cs_a + (n_remainder - 2)*rs_a;                 //pointer to block of A to be used for TRSM
                b10 = B + (m_remainder - 2) + (n_remainder)*cs_b;                          //pointer to block of B to be used in GEMM
                b11 = B + (m_remainder - 2) + (n_remainder - 2)*cs_b;                 //pointer to block of B to be used for TRSM

                k_iter = (n-n_remainder);                    //number of GEMM operations to be done(in blocks of 4x4)

                /*Fill zeros into ymm registers used in gemm accumulations */
                BLIS_SET_YMM_REG_ZEROS

                ///GEMM implementation starts///
                BLIS_DTRSM_SMALL_GEMM_2nx4m(a01,b10,cs_b,p_lda,k_iter)

                BLIS_PRE_DTRSM_SMALL_2N_2M(AlphaVal,b11,cs_b)
                ///implement TRSM///

                //extract a11
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));
                ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

                //extract a00
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack ));

                //(Row 1): FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);

                ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

                BLIS_POST_DTRSM_SMALL_2N_2M(b11,cs_b)

                m_remainder -=2;
            }
            else if (1 == m_remainder)
            {
                a01 = D_A_pack;
                a11 = L + (n_remainder - 2)*cs_a + (n_remainder - 2)*rs_a;                 //pointer to block of A to be used for TRSM
                b10 = B + (m_remainder - 1) + (n_remainder)*cs_b;                          //pointer to block of B to be used in GEMM
                b11 = B + (m_remainder - 1) + (n_remainder - 2)*cs_b;                 //pointer to block of B to be used for TRSM

                k_iter = (n-n_remainder);                    //number of GEMM operations to be done(in blocks of 4x4)

                /*Fill zeros into ymm registers used in gemm accumulations */
                BLIS_SET_YMM_REG_ZEROS

                ///GEMM implementation starts///
                BLIS_DTRSM_SMALL_GEMM_2nx4m(a01,b10,cs_b,p_lda,k_iter)

                BLIS_PRE_DTRSM_SMALL_2N_1M(AlphaVal,b11,cs_b)
                ///implement TRSM///

                //extract a11
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));
                ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

                //extract a00
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack ));

                //(Row 1): FMA operations
                ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));
                ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);

                ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

                BLIS_POST_DTRSM_SMALL_2N_1M(b11,cs_b)

                m_remainder -=1;
            }
        }
        n_remainder -= 2;
    }
    else if(n_remainder == 1)
    {
        a01 = L + (n_remainder - 1)*rs_a + n_remainder*cs_a;                     //pointer to block of A to be used in GEMM
        a11 = L + (n_remainder - 1)*cs_a + (n_remainder - 1)*rs_a;                 //pointer to block of A to be used for TRSM

        double *ptr_a10_dup = D_A_pack;

        dim_t p_lda = (n-n_remainder); // packed leading dimension
        // perform copy of A to packed buffer D_A_pack

        if(transa)
        {
            for(dim_t x =0;x < p_lda;x+=d_nr)
            {
                ymm0 = _mm256_loadu_pd((double const *)(a01));
                ymm1 = _mm256_loadu_pd((double const *)(a01 + cs_a));
                ymm2 = _mm256_loadu_pd((double const *)(a01 + cs_a * 2));
                ymm3 = _mm256_loadu_pd((double const *)(a01 + cs_a * 3));

                ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
                ymm5 = _mm256_unpacklo_pd(ymm2, ymm3);

                ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);

                ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
                ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);

                ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

                _mm256_storeu_pd((double *)(ptr_a10_dup), ymm6);
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda), ymm7);
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*2), ymm8);
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*3), ymm9);

                ymm0 = _mm256_loadu_pd((double const *)(a01 + cs_a * 4));
                ymm1 = _mm256_loadu_pd((double const *)(a01 + cs_a * 5));

                ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
                ymm5 = _mm256_broadcast_sd((double const *)&zero);

                ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);

                ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
                ymm1 = _mm256_broadcast_sd((double const *)&zero);

                ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

                _mm_storeu_pd((double *)(ptr_a10_dup + 4), _mm256_extractf128_pd(ymm6,0));
                _mm_storeu_pd((double *)(ptr_a10_dup + 4 + p_lda), _mm256_extractf128_pd(ymm7,0));
                _mm_storeu_pd((double *)(ptr_a10_dup + 4 + p_lda*2), _mm256_extractf128_pd(ymm8,0));
                _mm_storeu_pd((double *)(ptr_a10_dup + 4 + p_lda*3), _mm256_extractf128_pd(ymm9,0));

                a01 += d_nr*cs_a;
                ptr_a10_dup += d_nr;
            }
        }
        else
        {
            dim_t loop_count = (n-n_remainder)/4;

            for(dim_t x =0;x < loop_count;x++)
            {
                ymm15 = _mm256_loadu_pd((double const *)(a01 + rs_a * 0 + x*4));
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda * 0 + x*4), ymm15);
            }

            dim_t remainder_loop_count = p_lda - loop_count*4;

            __m128d xmm0;
            if(remainder_loop_count != 0)
            {
                xmm0 = _mm_loadu_pd((double const *)(a01 + rs_a * 0 + loop_count*4));
                _mm_storeu_pd((double *)(ptr_a10_dup + p_lda * 0 + loop_count*4), xmm0);
            }
        }

        ymm4 = _mm256_broadcast_sd((double const *)&ones);
        if(!is_unitdiag)
        {
            //broadcast diagonal elements of A11
            ymm0 = _mm256_broadcast_sd((double const *)(a11));
            ymm1 = _mm256_broadcast_sd((double const *)&ones);
            ymm2 = _mm256_broadcast_sd((double const *)&ones);
            ymm3 = _mm256_broadcast_sd((double const *)&ones);

            ymm0 = _mm256_unpacklo_pd(ymm0, ymm1);
            ymm1 = _mm256_unpacklo_pd(ymm2, ymm3);

            ymm1 = _mm256_blend_pd(ymm0, ymm1, 0x0C);
            #ifdef BLIS_DISABLE_TRSM_PREINVERSION
            ymm4 = ymm1;
            #endif
            #ifdef BLIS_ENABLE_TRSM_PREINVERSION
            ymm4 = _mm256_div_pd(ymm4, ymm1);
            #endif
        }
        _mm256_storeu_pd((double *)(d11_pack), ymm4);

        for(i = (m-d_mr); (i+1) > 0; i -= d_mr)     //loop along 'M' direction
        {
            a01 = D_A_pack;
            a11 = L + (n_remainder - 1)*cs_a + (n_remainder - 1)*rs_a;                 //pointer to block of A to be used for TRSM
            b10 = B + i + (n_remainder)*cs_b;                          //pointer to block of B to be used in GEMM
            b11 = B + (i) + (n_remainder - 1)*cs_b;                 //pointer to block of B to be used for TRSM

            k_iter = (n-n_remainder);                    //number of GEMM operations to be done(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_1nx8m(a01,b10,cs_b,p_lda,k_iter)

            ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);

            ymm0 = _mm256_loadu_pd((double const *)b11);                            //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
            ymm1 = _mm256_loadu_pd((double const *)(b11 + 4));                   //B11[4][0] B11[5][0] B11[6][0] B11[7][0]

            ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);      //B11[0-3][0] * alpha -= ymm0
            ymm4 = _mm256_fmsub_pd(ymm1, ymm15, ymm4);      //B11[4-7][0] * alpha-= ymm1

            ///implement TRSM///
            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack ));
            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);
            ymm4 = DTRSM_SMALL_DIV_OR_SCALE(ymm4, ymm0);

            _mm256_storeu_pd((double *)b11, ymm3);
            _mm256_storeu_pd((double *)(b11 + 4), ymm4);
        }

        dim_t m_remainder = i + d_mr;
        if(m_remainder >= 4)
        {
            a01 = D_A_pack;
            a11 = L + (n_remainder - 1)*cs_a + (n_remainder - 1)*rs_a;                 //pointer to block of A to be used for TRSM
            b10 = B + (m_remainder - 4) + (n_remainder)*cs_b;                          //pointer to block of B to be used in GEMM
            b11 = B + (m_remainder - 4) + (n_remainder - 1)*cs_b;                 //pointer to block of B to be used for TRSM

            k_iter = (n-n_remainder);                    //number of GEMM operations to be done(in blocks of 4x4)

            ymm3 = _mm256_setzero_pd();

            ///GEMM implementation starts///
            BLIS_DTRSM_SMALL_GEMM_1nx4m(a01,b10,cs_b,p_lda,k_iter)

            ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);         //register to hold alpha

            ymm0 = _mm256_loadu_pd((double const *)b11);                            //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
            ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);      //B11[0-3][0] * alpha -= ymm0

            ///implement TRSM///
            //extract a00
            ymm0 = _mm256_broadcast_sd((double const *)(d11_pack ));
            ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

            _mm256_storeu_pd((double *)b11, ymm3);

            m_remainder -=4;
        }

        if(m_remainder)
        {
            if(3 == m_remainder)
            {
                a01 = D_A_pack;
                a11 = L + (n_remainder - 1)*cs_a + (n_remainder - 1)*rs_a;                 //pointer to block of A to be used for TRSM
                b10 = B + (m_remainder - 3) + (n_remainder)*cs_b;                          //pointer to block of B to be used in GEMM
                b11 = B + (m_remainder - 3) + (n_remainder - 1)*cs_b;                 //pointer to block of B to be used for TRSM

                k_iter = (n-n_remainder);                    //number of GEMM operations to be done(in blocks of 4x4)

                ymm3 = _mm256_setzero_pd();

                ///GEMM implementation starts///
                BLIS_DTRSM_SMALL_GEMM_1nx4m(a01,b10,cs_b,p_lda,k_iter)

                BLIS_PRE_DTRSM_SMALL_1N_3M(AlphaVal,b11,cs_b)

                ///implement TRSM///
                //extract a00
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack ));
                ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

                ymm0 = _mm256_loadu_pd((double const *)b11);
                ymm3 = _mm256_blend_pd(ymm6, ymm3, 0x07);

                BLIS_POST_DTRSM_SMALL_1N_3M(b11,cs_b)

                m_remainder -=3;
            }
            else if(2 == m_remainder)
            {
                a01 = D_A_pack;
                a11 = L + (n_remainder - 1)*cs_a + (n_remainder - 1)*rs_a;                 //pointer to block of A to be used for TRSM
                b10 = B + (m_remainder - 2) + (n_remainder)*cs_b;                          //pointer to block of B to be used in GEMM
                b11 = B + (m_remainder - 2) + (n_remainder - 1)*cs_b;                 //pointer to block of B to be used for TRSM

                k_iter = (n-n_remainder);                    //number of GEMM operations to be done(in blocks of 4x4)

                ymm3 = _mm256_setzero_pd();

                ///GEMM implementation starts///
                BLIS_DTRSM_SMALL_GEMM_1nx4m(a01,b10,cs_b,p_lda,k_iter)

                BLIS_PRE_DTRSM_SMALL_1N_2M(AlphaVal,b11,cs_b)

                ///implement TRSM///
                //extract a00
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack ));
                ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

                BLIS_POST_DTRSM_SMALL_1N_2M(b11,cs_b)

                m_remainder -=2;
            }
            else if (1 == m_remainder)
            {
                a01 = D_A_pack;
                a11 = L + (n_remainder - 1)*cs_a + (n_remainder - 1)*rs_a;   //pointer to block of A to be used for TRSM
                b10 = B + (m_remainder - 1) + (n_remainder)*cs_b;   //pointer to block of B to be used in GEMM
                b11 = B + (m_remainder - 1) + (n_remainder - 1)*cs_b;  //pointer to block of B to be used for TRSM

                k_iter = (n-n_remainder);                    //number of GEMM operations to be done(in blocks of 4x4)

                ymm3 = _mm256_setzero_pd();

                ///GEMM implementation starts///
                BLIS_DTRSM_SMALL_GEMM_1nx4m(a01,b10,cs_b,p_lda,k_iter)

                BLIS_PRE_DTRSM_SMALL_1N_1M(AlphaVal,b11,cs_b)

                ///implement TRSM///
                //extract a00
                ymm0 = _mm256_broadcast_sd((double const *)(d11_pack ));
                ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

                BLIS_POST_DTRSM_SMALL_1N_1M(b11,cs_b)

                m_remainder -=1;
            }
        }
        n_remainder -= 1;
    }

    if ((required_packing_A == 1) && bli_mem_is_alloc( &local_mem_buf_A_s ))
    {
      bli_membrk_release(&rntm,
                         &local_mem_buf_A_s);
    }
    return BLIS_SUCCESS;
}

/*  TRSM for the case AX = alpha * B, Double precision
 *  A is lower-triangular, transpose, non-unit diagonal
 *  dimensions A: mxm X: mxn B: mxn
*/
BLIS_INLINE err_t bli_dtrsm_small_AltXB_AuXB
(
    obj_t* AlphaObj,
    obj_t* a,
    obj_t* b,
    cntx_t* cntx,
    cntl_t* cntl
)
{
    dim_t m = bli_obj_length(b);          // number of rows of matrix B
    dim_t n = bli_obj_width(b);           // number of columns of matrix B

    bool transa = bli_obj_has_trans(a);
    dim_t cs_a, rs_a;
    dim_t d_mr = 8,d_nr = 6;

    // Swap rs_a & cs_a in case of non-tranpose.
    if(transa)
    {
        cs_a = bli_obj_col_stride(a); // column stride of A
        rs_a = bli_obj_row_stride(a); // row stride of A
    }
    else
    {
        cs_a = bli_obj_row_stride(a); // row stride of A
        rs_a = bli_obj_col_stride(a); // column stride of A
    }
    dim_t cs_b = bli_obj_col_stride(b); // column stride of B

    dim_t i, j, k;                        //loop variables
    dim_t k_iter;                         //number of times GEMM to be performed

    double AlphaVal = *(double *)AlphaObj->buffer;    //value of alpha
    double *L =  a->buffer;               //pointer to  matrix A
    double *B =  b->buffer;               //pointer to matrix B

    //pointers that point to blocks for GEMM and TRSM
    double *a10, *a11, *b01, *b11;

    double ones = 1.0;
    bool is_unitdiag = bli_obj_has_unit_diag(a);

    //scratch registers
    __m256d ymm0, ymm1, ymm2, ymm3;
    __m256d ymm4, ymm5, ymm6, ymm7;
    __m256d ymm8, ymm9, ymm10, ymm11;
    __m256d ymm12, ymm13, ymm14, ymm15;
    __m256d ymm16, ymm17, ymm18, ymm19;
    __m256d ymm20;

    __m128d xmm5;

    gint_t required_packing_A = 1;
    mem_t local_mem_buf_A_s = {0};
    double *D_A_pack = NULL;
    double d11_pack[d_mr] __attribute__((aligned(64)));
    rntm_t rntm;

    bli_rntm_init_from_global( &rntm );
    bli_rntm_set_num_threads_only( 1, &rntm );
    bli_membrk_rntm_set_membrk( &rntm );

    siz_t buffer_size = bli_pool_block_size(
                            bli_membrk_pool(
                            bli_packbuf_index(BLIS_BITVAL_BUFFER_FOR_A_BLOCK),
                            bli_rntm_membrk(&rntm)));

    if((d_mr * m * sizeof(double)) > buffer_size)
        return BLIS_NOT_YET_IMPLEMENTED;

    if(required_packing_A == 1)
    {
        // Get the buffer from the pool.
        bli_membrk_acquire_m(&rntm,
                             buffer_size,
                             BLIS_BITVAL_BUFFER_FOR_A_BLOCK,
                             &local_mem_buf_A_s);
        if(FALSE==bli_mem_is_alloc(&local_mem_buf_A_s)) return BLIS_NULL_POINTER;
        D_A_pack = bli_mem_buffer(&local_mem_buf_A_s);
        if(NULL==D_A_pack) return BLIS_NULL_POINTER;
    }

    /*
        Performs solving TRSM for 8 colmns at a time from  0 to m/d_mr in steps of d_mr
        a. Load, transpose, Pack A (a10 block), the size of packing 8x6 to 8x (m-d_mr)
           First there will be no GEMM and no packing of a10 because it is only TRSM
        b. Using packed a10 block and b01 block perform GEMM operation
        c. Use GEMM outputs, perform TRSM operaton using a11, b11 and update B
        d. Repeat b,c for n rows of B in steps of d_nr
    */
    for(i = (m - d_mr); (i + 1) > 0; i -= d_mr)
    {
        a10 = L + (i*cs_a) + (i + d_mr)*rs_a;  //pointer to block of A to be used for GEMM
        a11 = L + (i*cs_a) + (i*rs_a);     //pointer to block of A to be used for TRSM

        // Do transpose for a10 & store in D_A_pack
        //ptr_a10_dup = D_A_pack;

        dim_t p_lda = d_mr; // packed leading dimension

        if(transa)
        {
            /*
              Load, transpose and pack current A block (a10) into packed buffer memory D_A_pack
              a. This a10 block is used in GEMM portion only and this
                 a10 block size will be increasing by d_mr for every next itteration
                 untill it reaches 8x(m-8) which is the maximum GEMM alone block size in A
              b. This packed buffer is reused to calculate all n rows of B matrix
            */
            bli_dtrsm_small_pack('L', (m-i-d_mr), 1, a10, cs_a, D_A_pack,p_lda,d_mr);

               /*
               Pack 8 diagonal elements of A block into an array
               a. This helps in utilze cache line efficiently in TRSM operation
               b. store ones when input is unit diagonal
            */
            dtrsm_small_pack_diag_element(is_unitdiag,a11,cs_a,d11_pack,d_mr);
        }
        else
        {
            bli_dtrsm_small_pack('L', (m-i-d_mr), 0, a10, rs_a, D_A_pack,p_lda,d_mr);
            dtrsm_small_pack_diag_element(is_unitdiag,a11,rs_a,d11_pack,d_mr);
        }

        /*
            a. Perform GEMM using a10, b01.
            b. Perform TRSM on a11, b11
            c. This loop GEMM+TRSM loops operates with 8x6 block size
               along n dimension for every d_nr rows of b01 where
               packed A buffer is reused in computing all n rows of B.
            d. Same approch is used in remaining fringe cases.
        */
        for(j = (n - d_nr); (j + 1) > 0; j -= d_nr)
        {
            a10 = D_A_pack;
            b01 = B + (j * cs_b) + i + d_mr;            //pointer to block of B to be used for GEMM
            b11 = B + (j * cs_b) + i;                  //pointer to block of B to be used for TRSM

            k_iter = (m - i - d_mr);

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            /*
                Peform GEMM between a10 and b01 blocks
                For first itteration there will be no GEMM operation
                where k_iter are zero
            */
            BLIS_DTRSM_SMALL_GEMM_8mx6n(a10,b01,cs_b,p_lda,k_iter)

            /*
               Load b11 of size 6x8 and multiply with alpha
               Add the GEMM output and perform inregister transose of b11
               to peform TRSM operation.
            */
            BLIS_DTRSM_SMALL_NREG_TRANSPOSE_6x8(b11,cs_b,AlphaVal)

            /*
                Compute 8x6 TRSM block by using GEMM block output in register
                a. The 8x6 input (gemm outputs) are stored in combinations of ymm registers
                    1. ymm15, ymm20 2. ymm14, ymm19 3. ymm13, ymm18 , 4. ymm12, ymm17
                    5. ymm11, ymm7 6. ymm10, ymm6, 7.ymm9, ymm5   8. ymm8, ymm4
                    where ymm15-ymm8 holds 8x4 data and reaming 8x2 will be hold by
                    other registers
                b. Towards the end do in regiser transpose of TRSM output and store in b11
            */
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 7));

            //perform mul operation
            ymm15 = DTRSM_SMALL_DIV_OR_SCALE(ymm15, ymm1);
            ymm20 = DTRSM_SMALL_DIV_OR_SCALE(ymm20, ymm1);

            //extract a22
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 6));

            //(ROw7): FMA operations
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 6*cs_a + 7*rs_a));
            ymm14 = _mm256_fnmadd_pd(ymm2, ymm15, ymm14);
            ymm19 = _mm256_fnmadd_pd(ymm2, ymm20, ymm19);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a + 7*rs_a));
            ymm13 = _mm256_fnmadd_pd(ymm2, ymm15, ymm13);
            ymm18 = _mm256_fnmadd_pd(ymm2, ymm20, ymm18);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 4*cs_a + 7*rs_a));
            ymm12 = _mm256_fnmadd_pd(ymm2, ymm15, ymm12);
            ymm17 = _mm256_fnmadd_pd(ymm2, ymm20, ymm17);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 7*rs_a));
            ymm11 = _mm256_fnmadd_pd(ymm2, ymm15, ymm11);
            ymm7 = _mm256_fnmadd_pd(ymm2, ymm20, ymm7);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 7*rs_a));
            ymm10 = _mm256_fnmadd_pd(ymm2, ymm15, ymm10);
            ymm6 = _mm256_fnmadd_pd(ymm2, ymm20, ymm6);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a + 7*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm2, ymm15, ymm9);
            ymm5 = _mm256_fnmadd_pd(ymm2, ymm20, ymm5);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 7*rs_a));
            ymm8 = _mm256_fnmadd_pd(ymm2, ymm15, ymm8);
            ymm4 = _mm256_fnmadd_pd(ymm2, ymm20, ymm4);

            //perform mul operation
            ymm14 = DTRSM_SMALL_DIV_OR_SCALE(ymm14, ymm1);
            ymm19 = DTRSM_SMALL_DIV_OR_SCALE(ymm19, ymm1);

            //extract a11
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 5));

            //(ROw6): FMA operations
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a + 6*rs_a));
            ymm13 = _mm256_fnmadd_pd(ymm2, ymm14, ymm13);
            ymm18 = _mm256_fnmadd_pd(ymm2, ymm19, ymm18);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 4*cs_a + 6*rs_a));
            ymm12 = _mm256_fnmadd_pd(ymm2, ymm14, ymm12);
            ymm17 = _mm256_fnmadd_pd(ymm2, ymm19, ymm17);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 6*rs_a));
            ymm11 = _mm256_fnmadd_pd(ymm2, ymm14, ymm11);
            ymm7 = _mm256_fnmadd_pd(ymm2, ymm19, ymm7);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 6*rs_a));
            ymm10 = _mm256_fnmadd_pd(ymm2, ymm14, ymm10);
            ymm6 = _mm256_fnmadd_pd(ymm2, ymm19, ymm6);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a + 6*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm2, ymm14, ymm9);
            ymm5 = _mm256_fnmadd_pd(ymm2, ymm19, ymm5);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 6*rs_a));
            ymm8 = _mm256_fnmadd_pd(ymm2, ymm14, ymm8);
            ymm4 = _mm256_fnmadd_pd(ymm2, ymm19, ymm4);

            //perform mul operation
            ymm13 = DTRSM_SMALL_DIV_OR_SCALE(ymm13, ymm1);
            ymm18 = DTRSM_SMALL_DIV_OR_SCALE(ymm18, ymm1);

            //extract a00
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 4));

            //(ROw5): FMA operations
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 4*cs_a + 5*rs_a));
            ymm12 = _mm256_fnmadd_pd(ymm2, ymm13, ymm12);
            ymm17 = _mm256_fnmadd_pd(ymm2, ymm18, ymm17);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 5*rs_a));
            ymm11 = _mm256_fnmadd_pd(ymm2, ymm13, ymm11);
            ymm7 = _mm256_fnmadd_pd(ymm2, ymm18, ymm7);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 5*rs_a));
            ymm10 = _mm256_fnmadd_pd(ymm2, ymm13, ymm10);
            ymm6 = _mm256_fnmadd_pd(ymm2, ymm18, ymm6);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a + 5*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm2, ymm13, ymm9);
            ymm5 = _mm256_fnmadd_pd(ymm2, ymm18, ymm5);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 5*rs_a));
            ymm8 = _mm256_fnmadd_pd(ymm2, ymm13, ymm8);
            ymm4 = _mm256_fnmadd_pd(ymm2, ymm18, ymm4);

            //perform mul operation
            ymm12 = DTRSM_SMALL_DIV_OR_SCALE(ymm12, ymm1);
            ymm17 = DTRSM_SMALL_DIV_OR_SCALE(ymm17, ymm1);

            //extract a33
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

            //(ROw4): FMA operations
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 4*rs_a));
            ymm11 = _mm256_fnmadd_pd(ymm2, ymm12, ymm11);
            ymm7 = _mm256_fnmadd_pd(ymm2, ymm17, ymm7);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 4*rs_a));
            ymm10 = _mm256_fnmadd_pd(ymm2, ymm12, ymm10);
            ymm6 = _mm256_fnmadd_pd(ymm2, ymm17, ymm6);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a + 4*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm2, ymm12, ymm9);
            ymm5 = _mm256_fnmadd_pd(ymm2, ymm17, ymm5);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 4*rs_a));
            ymm8 = _mm256_fnmadd_pd(ymm2, ymm12, ymm8);
            ymm4 = _mm256_fnmadd_pd(ymm2, ymm17, ymm4);

            //perform mul operation
            ymm11 = DTRSM_SMALL_DIV_OR_SCALE(ymm11, ymm1);
            ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm1);

            //extract a22
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            //(ROw3): FMA operations
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 3*rs_a));
            ymm10 = _mm256_fnmadd_pd(ymm2, ymm11, ymm10);
            ymm6 = _mm256_fnmadd_pd(ymm2, ymm7, ymm6);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a + 3*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm2, ymm11, ymm9);
            ymm5 = _mm256_fnmadd_pd(ymm2, ymm7, ymm5);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));
            ymm8 = _mm256_fnmadd_pd(ymm2, ymm11, ymm8);
            ymm4 = _mm256_fnmadd_pd(ymm2, ymm7, ymm4);

            //perform mul operation
            ymm10 = DTRSM_SMALL_DIV_OR_SCALE(ymm10, ymm1);
            ymm6 = DTRSM_SMALL_DIV_OR_SCALE(ymm6, ymm1);

            //extract a11
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(ROw2): FMA operations
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a + 2*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm2, ymm10, ymm9);
            ymm5 = _mm256_fnmadd_pd(ymm2, ymm6, ymm5);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));
            ymm8 = _mm256_fnmadd_pd(ymm2, ymm10, ymm8);
            ymm4 = _mm256_fnmadd_pd(ymm2, ymm6, ymm4);

            //perform mul operation
            ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm1);
            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm1);

            //extract a00
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack));

            //(ROw2): FMA operations
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 1*rs_a));
            ymm8 = _mm256_fnmadd_pd(ymm2, ymm9, ymm8);
            ymm4 = _mm256_fnmadd_pd(ymm2, ymm5, ymm4);

            //perform mul operation
            ymm8 = DTRSM_SMALL_DIV_OR_SCALE(ymm8, ymm1);
            ymm4 = DTRSM_SMALL_DIV_OR_SCALE(ymm4, ymm1);

            BLIS_DTRSM_SMALL_NREG_TRANSPOSE_8x6_AND_STORE(b11,cs_b)
        }

        dim_t n_remainder = j + d_nr;
        if(n_remainder >= 4)
        {
            a10 = D_A_pack;
            a11 = L + (i*cs_a) + (i*rs_a);
            b01 = B + ((n_remainder - 4)* cs_b) + i + d_mr;
            b11 = B + ((n_remainder - 4)* cs_b) + i;

            k_iter = (m - i - d_mr);

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            ///GEMM code begins///
            BLIS_DTRSM_SMALL_GEMM_8mx4n(a10,b01,cs_b,p_lda,k_iter)

            ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     //register to hold alpha

            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));    //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
            ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *1));    //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm2 = _mm256_loadu_pd((double const *)(b11 + cs_b *2));    //B11[0][2] B11[1][2] B11[2][2] B11[3][2]
            ymm3 = _mm256_loadu_pd((double const *)(b11 + cs_b *3));    //B11[0][3] B11[1][3] B11[2][3] B11[3][3]

            ymm4 = _mm256_loadu_pd((double const *)(b11 + cs_b *0 + 4));    //B11[0][4] B11[1][4] B11[2][4] B11[3][4]
            ymm5 = _mm256_loadu_pd((double const *)(b11 + cs_b *1 + 4));    //B11[0][5] B11[1][5] B11[2][5] B11[3][5]
            ymm6 = _mm256_loadu_pd((double const *)(b11 + cs_b *2 + 4));    //B11[0][6] B11[1][6] B11[2][6] B11[3][6]
            ymm7 = _mm256_loadu_pd((double const *)(b11 + cs_b *3 + 4));    //B11[0][7] B11[1][7] B11[2][7] B11[3][7]

            ymm0 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);  //B11[0-3][0] * alpha -= B01[0-3][0]
            ymm1 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);  //B11[0-3][1] * alpha -= B01[0-3][1]
            ymm2 = _mm256_fmsub_pd(ymm2, ymm16, ymm10); //B11[0-3][2] * alpha -= B01[0-3][2]
            ymm3 = _mm256_fmsub_pd(ymm3, ymm16, ymm11); //B11[0-3][3] * alpha -= B01[0-3][3]
            ymm4 = _mm256_fmsub_pd(ymm4, ymm16, ymm12); //B11[0-3][4] * alpha -= B01[0-3][4]
            ymm5 = _mm256_fmsub_pd(ymm5, ymm16, ymm13); //B11[0-3][5] * alpha -= B01[0-3][5]
            ymm6 = _mm256_fmsub_pd(ymm6, ymm16, ymm14); //B11[0-3][6] * alpha -= B01[0-3][6]
            ymm7 = _mm256_fmsub_pd(ymm7, ymm16, ymm15); //B11[0-3][7] * alpha -= B01[0-3][7]

            ///implement TRSM///

            ///transpose of B11//
            ///unpacklow///
            ymm9 = _mm256_unpacklo_pd(ymm0, ymm1);  //B11[0][0] B11[0][1] B11[2][0] B11[2][1]
            ymm11 = _mm256_unpacklo_pd(ymm2, ymm3); //B11[0][2] B11[0][3] B11[2][2] B11[2][3]

            ymm13 = _mm256_unpacklo_pd(ymm4, ymm5); //B11[0][4] B11[0][5] B11[2][4] B11[2][5]
            ymm15 = _mm256_unpacklo_pd(ymm6, ymm7); //B11[0][6] B11[0][7] B11[2][6] B11[2][7]

            //rearrange low elements
            ymm8 = _mm256_permute2f128_pd(ymm9,ymm11,0x20);     //B11[0][0] B11[0][1] B11[0][2] B11[0][3]
            ymm10 = _mm256_permute2f128_pd(ymm9,ymm11,0x31);    //B11[2][0] B11[2][1] B11[2][2] B11[2][3]

            ymm12 = _mm256_permute2f128_pd(ymm13,ymm15,0x20);   //B11[4][0] B11[4][1] B11[4][2] B11[4][3]
            ymm14 = _mm256_permute2f128_pd(ymm13,ymm15,0x31);   //B11[6][0] B11[6][1] B11[6][2] B11[6][3]

            ////unpackhigh////
            ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);  //B11[1][0] B11[1][1] B11[3][0] B11[3][1]
            ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);  //B11[1][2] B11[1][3] B11[3][2] B11[3][3]

            ymm4 = _mm256_unpackhi_pd(ymm4, ymm5);  //B11[1][4] B11[1][5] B11[3][4] B11[3][5]
            ymm5 = _mm256_unpackhi_pd(ymm6, ymm7);  //B11[1][6] B11[1][7] B11[3][6] B11[3][7]

            //rearrange high elements
            ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);  //B11[1][0] B11[1][1] B11[1][2] B11[1][3]
            ymm11 = _mm256_permute2f128_pd(ymm0,ymm1,0x31); //B11[3][0] B11[3][1] B11[3][2] B11[3][3]

            ymm13 = _mm256_permute2f128_pd(ymm4,ymm5,0x20); //B11[5][0] B11[5][1] B11[5][2] B11[5][3]
            ymm15 = _mm256_permute2f128_pd(ymm4,ymm5,0x31); //B11[7][0] B11[7][1] B11[7][2] B11[7][3]

            //extract a33
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 7));

            //perform mul operation
            ymm15 = DTRSM_SMALL_DIV_OR_SCALE(ymm15, ymm1);

            //extract a22
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 6));

            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 6*cs_a + 7*rs_a));
            ymm3 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a + 7*rs_a));
            ymm4 = _mm256_broadcast_sd((double const *)(a11 + 4*cs_a + 7*rs_a));
            ymm5 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 7*rs_a));
            ymm6 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 7*rs_a));
            ymm7 = _mm256_broadcast_sd((double const *)(a11 + cs_a + 7*rs_a));
            ymm16 = _mm256_broadcast_sd((double const *)(a11 + 7*rs_a));

            //(ROw7): FMA operations
            ymm14 = _mm256_fnmadd_pd(ymm2, ymm15, ymm14);
            ymm13 = _mm256_fnmadd_pd(ymm3, ymm15, ymm13);
            ymm12 = _mm256_fnmadd_pd(ymm4, ymm15, ymm12);
            ymm11 = _mm256_fnmadd_pd(ymm5, ymm15, ymm11);
            ymm10 = _mm256_fnmadd_pd(ymm6, ymm15, ymm10);
            ymm9 = _mm256_fnmadd_pd(ymm7, ymm15, ymm9);
            ymm8 = _mm256_fnmadd_pd(ymm16, ymm15, ymm8);

            //perform mul operation
            ymm14 = DTRSM_SMALL_DIV_OR_SCALE(ymm14, ymm1);

            //extract a11
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 5));

            ymm3 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a + 6*rs_a));
            ymm4 = _mm256_broadcast_sd((double const *)(a11 + 4*cs_a + 6*rs_a));
            ymm5 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 6*rs_a));
            ymm6 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 6*rs_a));
            ymm7 = _mm256_broadcast_sd((double const *)(a11 + cs_a + 6*rs_a));
            ymm16 = _mm256_broadcast_sd((double const *)(a11 + 6*rs_a));

            //(ROw6): FMA operations
            ymm13 = _mm256_fnmadd_pd(ymm3, ymm14, ymm13);
            ymm12 = _mm256_fnmadd_pd(ymm4, ymm14, ymm12);
            ymm11 = _mm256_fnmadd_pd(ymm5, ymm14, ymm11);
            ymm10 = _mm256_fnmadd_pd(ymm6, ymm14, ymm10);
            ymm9 = _mm256_fnmadd_pd(ymm7, ymm14, ymm9);
            ymm8 = _mm256_fnmadd_pd(ymm16, ymm14, ymm8);

            //perform mul operation
            ymm13 = DTRSM_SMALL_DIV_OR_SCALE(ymm13, ymm1);

            //extract a00
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 4));

            ymm4 = _mm256_broadcast_sd((double const *)(a11 + 4*cs_a + 5*rs_a));
            ymm5 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 5*rs_a));
            ymm6 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 5*rs_a));
            ymm7 = _mm256_broadcast_sd((double const *)(a11 + cs_a + 5*rs_a));
            ymm16 = _mm256_broadcast_sd((double const *)(a11 + 5*rs_a));

            //(ROw5): FMA operations
            ymm12 = _mm256_fnmadd_pd(ymm4, ymm13, ymm12);
            ymm11 = _mm256_fnmadd_pd(ymm5, ymm13, ymm11);
            ymm10 = _mm256_fnmadd_pd(ymm6, ymm13, ymm10);
            ymm9 = _mm256_fnmadd_pd(ymm7, ymm13, ymm9);
            ymm8 = _mm256_fnmadd_pd(ymm16, ymm13, ymm8);

            //perform mul operation
            ymm12 = DTRSM_SMALL_DIV_OR_SCALE(ymm12, ymm1);

            //extract a33
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

            ymm5 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 4*rs_a));
            ymm6 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 4*rs_a));
            ymm7 = _mm256_broadcast_sd((double const *)(a11 + cs_a + 4*rs_a));
            ymm16 = _mm256_broadcast_sd((double const *)(a11 + 4*rs_a));

            //(ROw4): FMA operations
            ymm11 = _mm256_fnmadd_pd(ymm5, ymm12, ymm11);
            ymm10 = _mm256_fnmadd_pd(ymm6, ymm12, ymm10);
            ymm9 = _mm256_fnmadd_pd(ymm7, ymm12, ymm9);
            ymm8 = _mm256_fnmadd_pd(ymm16, ymm12, ymm8);

            //perform mul operation
            ymm11 = DTRSM_SMALL_DIV_OR_SCALE(ymm11, ymm1);

            //extract a22
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            ymm6 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 3*rs_a));
            ymm7 = _mm256_broadcast_sd((double const *)(a11 + cs_a + 3*rs_a));
            ymm16 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));

            //(ROw3): FMA operations
            ymm10 = _mm256_fnmadd_pd(ymm6, ymm11, ymm10);
            ymm9 = _mm256_fnmadd_pd(ymm7, ymm11, ymm9);
            ymm8 = _mm256_fnmadd_pd(ymm16, ymm11, ymm8);

            //perform mul operation
            ymm10 = DTRSM_SMALL_DIV_OR_SCALE(ymm10, ymm1);

            //extract a11
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            ymm7 = _mm256_broadcast_sd((double const *)(a11 + cs_a + 2*rs_a));
            ymm16 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));

            //(ROw2): FMA operations
            ymm9 = _mm256_fnmadd_pd(ymm7, ymm10, ymm9);
            ymm8 = _mm256_fnmadd_pd(ymm16, ymm10, ymm8);

            //perform mul operation
            ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm1);

            //extract a00
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack));

            ymm16 = _mm256_broadcast_sd((double const *)(a11 + 1*rs_a));

            //(ROw2): FMA operations
            ymm8 = _mm256_fnmadd_pd(ymm16, ymm9, ymm8);

            //perform mul operation
            ymm8 = DTRSM_SMALL_DIV_OR_SCALE(ymm8, ymm1);

            //unpacklow//
            ymm1 = _mm256_unpacklo_pd(ymm8, ymm9);              //B11[0][0] B11[1][0] B11[0][2] B11[1][2]
            ymm3 = _mm256_unpacklo_pd(ymm10, ymm11);            //B11[2][0] B11[3][0] B11[2][2] B11[3][2]

            ymm5 = _mm256_unpacklo_pd(ymm12, ymm13);            //B11[4][0] B11[5][0] B11[4][2] B11[5][2]
            ymm7 = _mm256_unpacklo_pd(ymm14, ymm15);            //B11[6][0] B11[7][0] B11[6][2] B11[7][2]

            //rearrange low elements
            ymm0 = _mm256_permute2f128_pd(ymm1, ymm3, 0x20);    //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
            ymm2 = _mm256_permute2f128_pd(ymm1, ymm3, 0x31);    //B11[0][2] B11[1][2] B11[2][2] B11[3][2]

            ymm4 = _mm256_permute2f128_pd(ymm5, ymm7, 0x20);    //B11[4][0] B11[5][0] B11[6][0] B11[7][0]
            ymm6 = _mm256_permute2f128_pd(ymm5, ymm7, 0x31);    //B11[4][2] B11[5][2] B11[6][2] B11[7][2]

            ///unpack high///
            ymm8 = _mm256_unpackhi_pd(ymm8, ymm9);              //B11[0][1] B11[1][1] B11[0][3] B11[1][3]
            ymm9 = _mm256_unpackhi_pd(ymm10, ymm11);            //B11[2][1] B11[3][1] B11[2][3] B11[3][3]

            ymm12 = _mm256_unpackhi_pd(ymm12, ymm13);           //B11[4][1] B11[5][1] B11[4][3] B11[5][3]
            ymm13 = _mm256_unpackhi_pd(ymm14, ymm15);           //B11[6][1] B11[7][1] B11[6][3] B11[7][3]

            //rearrange high elements
            ymm1 = _mm256_permute2f128_pd(ymm8, ymm9, 0x20);    //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm3 = _mm256_permute2f128_pd(ymm8, ymm9, 0x31);    //B11[0][3] B11[1][3] B11[2][3] B11[3][3]

            ymm5 = _mm256_permute2f128_pd(ymm12, ymm13, 0x20);   //B11[4][1] B11[5][1] B11[6][1] B11[7][1]
            ymm7 = _mm256_permute2f128_pd(ymm12, ymm13, 0x31);   //B11[4][3] B11[5][3] B11[6][3] B11[7][3]

            _mm256_storeu_pd((double *)(b11 + cs_b * 0), ymm0);        //store B11[0][0-3]
            _mm256_storeu_pd((double *)(b11 + cs_b * 1), ymm1);        //store B11[1][0-3]
            _mm256_storeu_pd((double *)(b11 + cs_b * 2), ymm2);        //store B11[2][0-3]
            _mm256_storeu_pd((double *)(b11 + cs_b * 3), ymm3);        //store B11[3][0-3]
            _mm256_storeu_pd((double *)(b11 + cs_b * 0 + 4), ymm4);    //store B11[4][0-3]
            _mm256_storeu_pd((double *)(b11 + cs_b * 1 + 4), ymm5);    //store B11[5][0-3]
            _mm256_storeu_pd((double *)(b11 + cs_b * 2 + 4), ymm6);    //store B11[6][0-3]
            _mm256_storeu_pd((double *)(b11 + cs_b * 3 + 4), ymm7);    //store B11[7][0-3]
            n_remainder -=4;
        }

        if(n_remainder)   //implementation fo remaining columns(when 'N' is not a multiple of d_nr)() n = 3
        {
            a10 = D_A_pack;
            a11 = L + (i*cs_a) + (i*rs_a);
            b01 = B + i + d_mr;
            b11 = B + i;

            k_iter = (m - i - d_mr) ;

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            if(3 == n_remainder)
            {
                ///GEMM code begins///
                BLIS_DTRSM_SMALL_GEMM_8mx3n(a10,b01,cs_b,p_lda,k_iter)

                ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     //register to hold alpha

                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));     //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
                ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *1));     //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
                ymm2 = _mm256_loadu_pd((double const *)(b11 + cs_b *2));     //B11[0][2] B11[1][2] B11[2][2] B11[3][2]

                ymm4 = _mm256_loadu_pd((double const *)(b11 + cs_b *0 + 4)); //B11[0][4] B11[1][4] B11[2][4] B11[3][4]
                ymm5 = _mm256_loadu_pd((double const *)(b11 + cs_b *1 + 4)); //B11[0][5] B11[1][5] B11[2][5] B11[3][5]
                ymm6 = _mm256_loadu_pd((double const *)(b11 + cs_b *2 + 4)); //B11[0][6] B11[1][6] B11[2][6] B11[3][6]

                ymm0 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);                   //B11[0-3][0] * alpha -= B01[0-3][0]
                ymm1 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);                   //B11[0-3][1] * alpha -= B01[0-3][1]
                ymm2 = _mm256_fmsub_pd(ymm2, ymm16, ymm10);                  //B11[0-3][2] * alpha -= B01[0-3][2]
                ymm3 = _mm256_broadcast_sd((double const *)(&ones));

                ymm4 = _mm256_fmsub_pd(ymm4, ymm16, ymm12);                  //B11[0-3][4] * alpha -= B01[0-3][4]
                ymm5 = _mm256_fmsub_pd(ymm5, ymm16, ymm13);                  //B11[0-3][5] * alpha -= B01[0-3][5]
                ymm6 = _mm256_fmsub_pd(ymm6, ymm16, ymm14);                  //B11[0-3][6] * alpha -= B01[0-3][6]
                ymm7 = _mm256_broadcast_sd((double const *)(&ones));
            }
            else if(2 == n_remainder)
            {
                ///GEMM code begins///
                BLIS_DTRSM_SMALL_GEMM_8mx2n(a10,b01,cs_b,p_lda,k_iter)

                ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));      //register to hold alpha

                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));     //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
                ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *1));     //B11[0][1] B11[1][1] B11[2][1] B11[3][1]

                ymm4 = _mm256_loadu_pd((double const *)(b11 + cs_b *0 + 4)); //B11[0][4] B11[1][4] B11[2][4] B11[3][4]
                ymm5 = _mm256_loadu_pd((double const *)(b11 + cs_b *1 + 4)); //B11[0][5] B11[1][5] B11[2][5] B11[3][5]

                ymm0 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);                   //B11[0-3][0] * alpha -= B01[0-3][0]
                ymm1 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);                   //B11[0-3][1] * alpha -= B01[0-3][1]
                ymm2 = _mm256_broadcast_sd((double const *)(&ones));
                ymm3 = _mm256_broadcast_sd((double const *)(&ones));

                ymm4 = _mm256_fmsub_pd(ymm4, ymm16, ymm12);                  //B11[0-3][4] * alpha -= B01[0-3][4]
                ymm5 = _mm256_fmsub_pd(ymm5, ymm16, ymm13);                  //B11[0-3][5] * alpha -= B01[0-3][5]
                ymm6 = _mm256_broadcast_sd((double const *)(&ones));
                ymm7 = _mm256_broadcast_sd((double const *)(&ones));
            }
            else if(1 == n_remainder)
            {
                ///GEMM code begins///
                BLIS_DTRSM_SMALL_GEMM_8mx1n(a10,b01,cs_b,p_lda,k_iter)

                ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));      //register to hold alpha

                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));     //B11[0][0] B11[1][0] B11[2][0] B11[3][0]

                ymm4 = _mm256_loadu_pd((double const *)(b11 + cs_b *0 + 4)); //B11[0][4] B11[1][4] B11[2][4] B11[3][4]

                ymm0 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);                   //B11[0-3][0] * alpha -= B01[0-3][0]
                ymm1 = _mm256_broadcast_sd((double const *)(&ones));
                ymm2 = _mm256_broadcast_sd((double const *)(&ones));
                ymm3 = _mm256_broadcast_sd((double const *)(&ones));

                ymm4 = _mm256_fmsub_pd(ymm4, ymm16, ymm12);                  //B11[0-3][4] * alpha -= B01[0-3][4]
                ymm5 = _mm256_broadcast_sd((double const *)(&ones));
                ymm6 = _mm256_broadcast_sd((double const *)(&ones));
                ymm7 = _mm256_broadcast_sd((double const *)(&ones));
            }
            ///implement TRSM///

            ///transpose of B11//
            ///unpacklow///
            ymm9 = _mm256_unpacklo_pd(ymm0, ymm1);                      //B11[0][0] B11[0][1] B11[2][0] B11[2][1]
            ymm11 = _mm256_unpacklo_pd(ymm2, ymm3);                     //B11[0][2] B11[0][3] B11[2][2] B11[2][3]

            ymm13 = _mm256_unpacklo_pd(ymm4, ymm5);                     //B11[0][4] B11[0][5] B11[2][4] B11[2][5]
            ymm15 = _mm256_unpacklo_pd(ymm6, ymm7);                     //B11[0][6] B11[0][7] B11[2][6] B11[2][7]

            //rearrange low elements
            ymm8 = _mm256_permute2f128_pd(ymm9,ymm11,0x20);             //B11[0][0] B11[0][1] B11[0][2] B11[0][3]
            ymm10 = _mm256_permute2f128_pd(ymm9,ymm11,0x31);            //B11[2][0] B11[2][1] B11[2][2] B11[2][3]

            ymm12 = _mm256_permute2f128_pd(ymm13,ymm15,0x20);           //B11[4][0] B11[4][1] B11[4][2] B11[4][3]
            ymm14 = _mm256_permute2f128_pd(ymm13,ymm15,0x31);           //B11[6][0] B11[6][1] B11[6][2] B11[6][3]

            ////unpackhigh////
            ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);                      //B11[1][0] B11[1][1] B11[3][0] B11[3][1]
            ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);                      //B11[1][2] B11[1][3] B11[3][2] B11[3][3]

            ymm4 = _mm256_unpackhi_pd(ymm4, ymm5);                      //B11[1][4] B11[1][5] B11[3][4] B11[3][5]
            ymm5 = _mm256_unpackhi_pd(ymm6, ymm7);                      //B11[1][6] B11[1][7] B11[3][6] B11[3][7]

            //rearrange high elements
            ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);              //B11[1][0] B11[1][1] B11[1][2] B11[1][3]
            ymm11 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);             //B11[3][0] B11[3][1] B11[3][2] B11[3][3]

            ymm13 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);             //B11[5][0] B11[5][1] B11[5][2] B11[5][3]
            ymm15 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);             //B11[7][0] B11[7][1] B11[7][2] B11[7][3]

            //extract a33
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 7));

            //perform mul operation
            ymm15 = DTRSM_SMALL_DIV_OR_SCALE(ymm15, ymm1);

            //extract a22
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 6));

            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 6*cs_a + 7*rs_a));
            ymm3 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a + 7*rs_a));
            ymm4 = _mm256_broadcast_sd((double const *)(a11 + 4*cs_a + 7*rs_a));
            ymm5 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 7*rs_a));
            ymm6 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 7*rs_a));
            ymm7 = _mm256_broadcast_sd((double const *)(a11 + cs_a + 7*rs_a));
            ymm16 = _mm256_broadcast_sd((double const *)(a11 + 7*rs_a));

            //(ROw7): FMA operations
            ymm14 = _mm256_fnmadd_pd(ymm2, ymm15, ymm14);
            ymm13 = _mm256_fnmadd_pd(ymm3, ymm15, ymm13);
            ymm12 = _mm256_fnmadd_pd(ymm4, ymm15, ymm12);
            ymm11 = _mm256_fnmadd_pd(ymm5, ymm15, ymm11);
            ymm10 = _mm256_fnmadd_pd(ymm6, ymm15, ymm10);
            ymm9 = _mm256_fnmadd_pd(ymm7, ymm15, ymm9);
            ymm8 = _mm256_fnmadd_pd(ymm16, ymm15, ymm8);

            //perform mul operation
            ymm14 = DTRSM_SMALL_DIV_OR_SCALE(ymm14, ymm1);

            //extract a11
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 5));

            ymm3 = _mm256_broadcast_sd((double const *)(a11 + 5*cs_a + 6*rs_a));
            ymm4 = _mm256_broadcast_sd((double const *)(a11 + 4*cs_a + 6*rs_a));
            ymm5 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 6*rs_a));
            ymm6 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 6*rs_a));
            ymm7 = _mm256_broadcast_sd((double const *)(a11 + cs_a + 6*rs_a));
            ymm16 = _mm256_broadcast_sd((double const *)(a11 + 6*rs_a));

            //(ROw6): FMA operations
            ymm13 = _mm256_fnmadd_pd(ymm3, ymm14, ymm13);
            ymm12 = _mm256_fnmadd_pd(ymm4, ymm14, ymm12);
            ymm11 = _mm256_fnmadd_pd(ymm5, ymm14, ymm11);
            ymm10 = _mm256_fnmadd_pd(ymm6, ymm14, ymm10);
            ymm9 = _mm256_fnmadd_pd(ymm7, ymm14, ymm9);
            ymm8 = _mm256_fnmadd_pd(ymm16, ymm14, ymm8);

            //perform mul operation
            ymm13 = DTRSM_SMALL_DIV_OR_SCALE(ymm13, ymm1);

            //extract a00
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 4));

            ymm4 = _mm256_broadcast_sd((double const *)(a11 + 4*cs_a + 5*rs_a));
            ymm5 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 5*rs_a));
            ymm6 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 5*rs_a));
            ymm7 = _mm256_broadcast_sd((double const *)(a11 + cs_a + 5*rs_a));
            ymm16 = _mm256_broadcast_sd((double const *)(a11 + 5*rs_a));

            //(ROw5): FMA operations
            ymm12 = _mm256_fnmadd_pd(ymm4, ymm13, ymm12);
            ymm11 = _mm256_fnmadd_pd(ymm5, ymm13, ymm11);
            ymm10 = _mm256_fnmadd_pd(ymm6, ymm13, ymm10);
            ymm9 = _mm256_fnmadd_pd(ymm7, ymm13, ymm9);
            ymm8 = _mm256_fnmadd_pd(ymm16, ymm13, ymm8);

            //perform mul operation
            ymm12 = DTRSM_SMALL_DIV_OR_SCALE(ymm12, ymm1);

            //extract a33
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

            ymm5 = _mm256_broadcast_sd((double const *)(a11 + 3*cs_a + 4*rs_a));
            ymm6 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 4*rs_a));
            ymm7 = _mm256_broadcast_sd((double const *)(a11 + cs_a + 4*rs_a));
            ymm16 = _mm256_broadcast_sd((double const *)(a11 + 4*rs_a));

            //(ROw4): FMA operations
            ymm11 = _mm256_fnmadd_pd(ymm5, ymm12, ymm11);
            ymm10 = _mm256_fnmadd_pd(ymm6, ymm12, ymm10);
            ymm9 = _mm256_fnmadd_pd(ymm7, ymm12, ymm9);
            ymm8 = _mm256_fnmadd_pd(ymm16, ymm12, ymm8);

            //perform mul operation
            ymm11 = DTRSM_SMALL_DIV_OR_SCALE(ymm11, ymm1);

            //extract a22
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            ymm6 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 3*rs_a));
            ymm7 = _mm256_broadcast_sd((double const *)(a11 + cs_a + 3*rs_a));
            ymm16 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));

            //(ROw3): FMA operations
            ymm10 = _mm256_fnmadd_pd(ymm6, ymm11, ymm10);
            ymm9 = _mm256_fnmadd_pd(ymm7, ymm11, ymm9);
            ymm8 = _mm256_fnmadd_pd(ymm16, ymm11, ymm8);

            //perform mul operation
            ymm10 = DTRSM_SMALL_DIV_OR_SCALE(ymm10, ymm1);

            //extract a11
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            ymm7 = _mm256_broadcast_sd((double const *)(a11 + cs_a + 2*rs_a));
            ymm16 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));

            //(ROw2): FMA operations
            ymm9 = _mm256_fnmadd_pd(ymm7, ymm10, ymm9);
            ymm8 = _mm256_fnmadd_pd(ymm16, ymm10, ymm8);

            //perform mul operation
            ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm1);

            //extract a00
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack));

            ymm16 = _mm256_broadcast_sd((double const *)(a11 + 1*rs_a));

            //(ROw2): FMA operations
            ymm8 = _mm256_fnmadd_pd(ymm16, ymm9, ymm8);

            //perform mul operation
            ymm8 = DTRSM_SMALL_DIV_OR_SCALE(ymm8, ymm1);

            //unpacklow//
            ymm1 = _mm256_unpacklo_pd(ymm8, ymm9);              //B11[0][0] B11[1][0] B11[0][2] B11[1][2]
            ymm3 = _mm256_unpacklo_pd(ymm10, ymm11);            //B11[2][0] B11[3][0] B11[2][2] B11[3][2]

            ymm5 = _mm256_unpacklo_pd(ymm12, ymm13);            //B11[4][0] B11[5][0] B11[4][2] B11[5][2]
            ymm7 = _mm256_unpacklo_pd(ymm14, ymm15);            //B11[6][0] B11[7][0] B11[6][2] B11[7][2]

            //rearrange low elements
            ymm0 = _mm256_permute2f128_pd(ymm1, ymm3, 0x20);    //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
            ymm2 = _mm256_permute2f128_pd(ymm1, ymm3, 0x31);    //B11[0][2] B11[1][2] B11[2][2] B11[3][2]

            ymm4 = _mm256_permute2f128_pd(ymm5, ymm7, 0x20);    //B11[4][0] B11[5][0] B11[6][0] B11[7][0]
            ymm6 = _mm256_permute2f128_pd(ymm5, ymm7, 0x31);    //B11[4][2] B11[5][2] B11[6][2] B11[7][2]

            ///unpack high///
            ymm8 = _mm256_unpackhi_pd(ymm8, ymm9);              //B11[0][1] B11[1][1] B11[0][3] B11[1][3]
            ymm9 = _mm256_unpackhi_pd(ymm10, ymm11);            //B11[2][1] B11[3][1] B11[2][3] B11[3][3]

            ymm12 = _mm256_unpackhi_pd(ymm12, ymm13);           //B11[4][1] B11[5][1] B11[4][3] B11[5][3]
            ymm13 = _mm256_unpackhi_pd(ymm14, ymm15);           //B11[6][1] B11[7][1] B11[6][3] B11[7][3]

            //rearrange high elements
            ymm1 = _mm256_permute2f128_pd(ymm8, ymm9, 0x20);    //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm3 = _mm256_permute2f128_pd(ymm8, ymm9, 0x31);    //B11[0][3] B11[1][3] B11[2][3] B11[3][3]

            ymm5 = _mm256_permute2f128_pd(ymm12, ymm13, 0x20);  //B11[4][1] B11[5][1] B11[6][1] B11[7][1]
            ymm7 = _mm256_permute2f128_pd(ymm12, ymm13, 0x31);  //B11[4][3] B11[5][3] B11[6][3] B11[7][3]

            if(3 == n_remainder)
            {
                _mm256_storeu_pd((double *)(b11 + cs_b * 0), ymm0);        //store B11[0][0-3]
                _mm256_storeu_pd((double *)(b11 + cs_b * 1), ymm1);        //store B11[1][0-3]
                _mm256_storeu_pd((double *)(b11 + cs_b * 2), ymm2);        //store B11[2][0-3]
                _mm256_storeu_pd((double *)(b11 + cs_b * 0 + 4), ymm4);    //store B11[4][0-3]
                _mm256_storeu_pd((double *)(b11 + cs_b * 1 + 4), ymm5);    //store B11[5][0-3]
                _mm256_storeu_pd((double *)(b11 + cs_b * 2 + 4), ymm6);    //store B11[6][0-3]
            }
            else if(2 == n_remainder)
            {
                _mm256_storeu_pd((double *)(b11 + cs_b * 0), ymm0);        //store B11[0][0-3]
                _mm256_storeu_pd((double *)(b11 + cs_b * 1), ymm1);        //store B11[1][0-3]
                _mm256_storeu_pd((double *)(b11 + cs_b * 0 + 4), ymm4);    //store B11[4][0-3]
                _mm256_storeu_pd((double *)(b11 + cs_b * 1 + 4), ymm5);    //store B11[5][0-3]
            }
            else if(1 == n_remainder)
            {
                _mm256_storeu_pd((double *)(b11 + cs_b * 0), ymm0);        //store B11[0][0-3]
                _mm256_storeu_pd((double *)(b11 + cs_b * 0 + 4), ymm4);    //store B11[4][0-3]
            }
        }
    }// End of multiples of d_mr blocks in m-dimension

    // Repetative A blocks will be 4*4
    dim_t m_remainder = i + d_mr;
    if(m_remainder >= 4)
    {
        i = m_remainder - 4;
        a10 = L + (i*cs_a) + (i + 4)*rs_a;           //pointer to block of A to be used for GEMM
        a11 = L + (i*cs_a) + (i*rs_a);                    //pointer to block of A to be used for TRSM

        // Do transpose for a10 & store in D_A_pack
        double *ptr_a10_dup = D_A_pack;
        dim_t p_lda = 4; // packed leading dimension
        if(transa)
        {
            for(dim_t x =0;x < m-i+4;x+=p_lda)
            {
                ymm0 = _mm256_loadu_pd((double const *)(a10));
                ymm1 = _mm256_loadu_pd((double const *)(a10 + cs_a));
                ymm2 = _mm256_loadu_pd((double const *)(a10 + cs_a * 2));
                ymm3 = _mm256_loadu_pd((double const *)(a10 + cs_a * 3));

                ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
                ymm5 = _mm256_unpacklo_pd(ymm2, ymm3);

                ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);

                ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
                ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);

                ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

                _mm256_storeu_pd((double *)(ptr_a10_dup), ymm6);
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda), ymm7);
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*2), ymm8);
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*3), ymm9);

                a10 += p_lda;
                ptr_a10_dup += p_lda*p_lda;
            }
        }
        else
        {
            for(dim_t x =0;x < m-i-4;x++)
            {
                ymm0 = _mm256_loadu_pd((double const *)(a10 + x*rs_a));
                _mm256_storeu_pd((double *)(ptr_a10_dup + x*p_lda), ymm0);
            }
        }

        ymm4 = _mm256_broadcast_sd((double const *)&ones);
        if(!is_unitdiag)
        {
            if(transa)
            {
                //broadcast diagonal elements of A11
                ymm0 = _mm256_broadcast_sd((double const *)(a11));
                ymm1 = _mm256_broadcast_sd((double const *)(a11+cs_a*1 + 1));
                ymm2 = _mm256_broadcast_sd((double const *)(a11+cs_a*2 + 2));
                ymm3 = _mm256_broadcast_sd((double const *)(a11+cs_a*3 + 3));
            }
            else
            {
                //broadcast diagonal elements of A11
                ymm0 = _mm256_broadcast_sd((double const *)(a11));
                ymm1 = _mm256_broadcast_sd((double const *)(a11+rs_a*1 + 1));
                ymm2 = _mm256_broadcast_sd((double const *)(a11+rs_a*2 + 2));
                ymm3 = _mm256_broadcast_sd((double const *)(a11+rs_a*3 + 3));
            }

            ymm0 = _mm256_unpacklo_pd(ymm0, ymm1);
            ymm1 = _mm256_unpacklo_pd(ymm2, ymm3);
            ymm1 = _mm256_blend_pd(ymm0, ymm1, 0x0C);
            #ifdef BLIS_DISABLE_TRSM_PREINVERSION
            ymm4 = ymm1;
            #endif
            #ifdef BLIS_ENABLE_TRSM_PREINVERSION
            ymm4 = _mm256_div_pd(ymm4, ymm1);
            #endif
        }
        _mm256_storeu_pd((double *)(d11_pack), ymm4);

        //cols
        for(j = (n - d_nr); (j + 1) > 0; j -= d_nr)    //loop along 'N' dimension
        {
            a10 = D_A_pack;
            a11 = L + (i*cs_a) + (i*rs_a);                    //pointer to block of A to be used for TRSM
            b01 = B + (j*cs_b) + i + 4;                //pointer to block of B to be used for GEMM
            b11 = B + (j* cs_b) + i;                   //pointer to block of B to be used for TRSM

            k_iter = (m - i - 4);                  //number of times GEMM to be performed(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            ///GEMM code begins///
            BLIS_DTRSM_SMALL_GEMM_4mx6n(a10,b01,cs_b,p_lda,k_iter)

            ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     //register to hold alpha

            ///implement TRSM///

            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));
            ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *1));
            ymm2 = _mm256_loadu_pd((double const *)(b11 + cs_b *2));
            ymm3 = _mm256_loadu_pd((double const *)(b11 + cs_b *3));
            ymm0 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);
            ymm1 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);
            ymm2 = _mm256_fmsub_pd(ymm2, ymm16, ymm10);
            ymm3 = _mm256_fmsub_pd(ymm3, ymm16, ymm11);

            ///transpose of B11//
            ///unpacklow///
            ymm9 = _mm256_unpacklo_pd(ymm0, ymm1);              //B11[0][0] B11[0][1] B11[2][0] B11[2][1]
            ymm11 = _mm256_unpacklo_pd(ymm2, ymm3);             //B11[0][2] B11[0][3] B11[2][2] B11[2][3]

            //rearrange low elements
            ymm8 = _mm256_permute2f128_pd(ymm9,ymm11,0x20);     //B11[0][0] B11[0][1] B11[0][2] B11[0][3]
            ymm10 = _mm256_permute2f128_pd(ymm9,ymm11,0x31);    //B11[2][0] B11[2][1] B11[2][2] B11[2][3]

            ////unpackhigh////
            ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);              //B11[1][0] B11[1][1] B11[3][0] B11[3][1]
            ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);              //B11[1][2] B11[1][3] B11[3][2] B11[3][3]

            //rearrange high elements
            ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);      //B11[1][0] B11[1][1] B11[1][2] B11[1][3]
            ymm11 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);     //B11[3][0] B11[3][1] B11[3][2] B11[3][3]

            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *4));
            ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *5));
            ymm0 = _mm256_fmsub_pd(ymm0, ymm16, ymm4);
            ymm1 = _mm256_fmsub_pd(ymm1, ymm16, ymm5);

            ymm16 = _mm256_broadcast_sd((double const *)(&ones));

            ////unpacklow////
            ymm7 = _mm256_unpacklo_pd(ymm0, ymm1);              //B11[0][0] B11[0][1] B11[2][0] B11[2][1]

            //rearrange low elements
            ymm4 = _mm256_permute2f128_pd(ymm7,ymm16,0x20);    //B11[0][0] B11[0][1] B11[0][2] B11[0][3]
            ymm6 = _mm256_permute2f128_pd(ymm7,ymm16,0x31);    //B11[2][0] B11[2][1] B11[2][2] B11[2][3]

            ////unpackhigh////
            ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);             //B11[1][0] B11[1][1] B11[3][0] B11[3][1]

            //rearrange high elements
            ymm5 = _mm256_permute2f128_pd(ymm0,ymm16,0x20);    //B11[1][0] B11[1][1] B11[1][2] B11[1][3]
            ymm7 = _mm256_permute2f128_pd(ymm0,ymm16,0x31);    //B11[3][0] B11[3][1] B11[3][2] B11[3][3]


            //extract a33
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

            //perform mul operation
            ymm11 = DTRSM_SMALL_DIV_OR_SCALE(ymm11, ymm1);
            ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm1);

            //extract a22
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            //(ROw3): FMA operations
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 3*rs_a));
            ymm10 = _mm256_fnmadd_pd(ymm2, ymm11, ymm10);
            ymm6 = _mm256_fnmadd_pd(ymm2, ymm7, ymm6);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a + 3*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm2, ymm11, ymm9);
            ymm5 = _mm256_fnmadd_pd(ymm2, ymm7, ymm5);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));
            ymm8 = _mm256_fnmadd_pd(ymm2, ymm11, ymm8);
            ymm4 = _mm256_fnmadd_pd(ymm2, ymm7, ymm4);

            //perform mul operation
            ymm10 = DTRSM_SMALL_DIV_OR_SCALE(ymm10, ymm1);
            ymm6 = DTRSM_SMALL_DIV_OR_SCALE(ymm6, ymm1);

            //extract a11
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(ROw2): FMA operations
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a + 2*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm2, ymm10, ymm9);
            ymm5 = _mm256_fnmadd_pd(ymm2, ymm6, ymm5);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));
            ymm8 = _mm256_fnmadd_pd(ymm2, ymm10, ymm8);
            ymm4 = _mm256_fnmadd_pd(ymm2, ymm6, ymm4);

            //perform mul operation
            ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm1);
            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm1);

            //extract a00
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack));

            //(ROw2): FMA operations
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 1*rs_a));
            ymm8 = _mm256_fnmadd_pd(ymm2, ymm9, ymm8);
            ymm4 = _mm256_fnmadd_pd(ymm2, ymm5, ymm4);

            //perform mul operation
            ymm8 = DTRSM_SMALL_DIV_OR_SCALE(ymm8, ymm1);
            ymm4 = DTRSM_SMALL_DIV_OR_SCALE(ymm4, ymm1);

            //unpacklow//
            ymm1 = _mm256_unpacklo_pd(ymm8, ymm9);              //B11[0][0] B11[1][0] B11[0][2] B11[1][2]
            ymm3 = _mm256_unpacklo_pd(ymm10, ymm11);            //B11[2][0] B11[3][0] B11[2][2] B11[3][2]

            //rearrange low elements
            ymm0 = _mm256_permute2f128_pd(ymm1, ymm3, 0x20);    //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
            ymm2 = _mm256_permute2f128_pd(ymm1, ymm3, 0x31);    //B11[0][2] B11[1][2] B11[2][2] B11[3][2]

            ///unpack high///
            ymm8 = _mm256_unpackhi_pd(ymm8, ymm9);              //B11[0][1] B11[1][1] B11[0][3] B11[1][3]
            ymm9 = _mm256_unpackhi_pd(ymm10, ymm11);            //B11[2][1] B11[3][1] B11[2][3] B11[3][3]

            //rearrange high elements
            ymm1 = _mm256_permute2f128_pd(ymm8, ymm9, 0x20);    //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm3 = _mm256_permute2f128_pd(ymm8, ymm9, 0x31);    //B11[0][3] B11[1][3] B11[2][3] B11[3][3]

            _mm256_storeu_pd((double *)(b11 + cs_b * 0), ymm0);    //store B11[0][0-3]
            _mm256_storeu_pd((double *)(b11 + cs_b * 1), ymm1);    //store B11[1][0-3]
            _mm256_storeu_pd((double *)(b11 + cs_b * 2), ymm2);    //store B11[2][0-3]
            _mm256_storeu_pd((double *)(b11 + cs_b * 3), ymm3);    //store B11[3][0-3]

            //unpacklow//
            ymm1 = _mm256_unpacklo_pd(ymm4, ymm5);              //B11[0][0] B11[1][0] B11[0][2] B11[1][2]
            ymm3 = _mm256_unpacklo_pd(ymm6, ymm7);              //B11[2][0] B11[3][0] B11[2][2] B11[3][2]

            //rearrange low elements
            ymm0 = _mm256_permute2f128_pd(ymm1, ymm3, 0x20);    //B11[0][0] B11[1][0] B11[2][0] B11[3][0]

            ///unpack high///
            ymm4 = _mm256_unpackhi_pd(ymm4, ymm5);              //B11[0][1] B11[1][1] B11[0][3] B11[1][3]
            ymm5 = _mm256_unpackhi_pd(ymm6, ymm7);              //B11[2][1] B11[3][1] B11[2][3] B11[3][3]

            //rearrange high elements
            ymm1 = _mm256_permute2f128_pd(ymm4, ymm5, 0x20);    //B11[0][1] B11[1][1] B11[2][1] B11[3][1]

            _mm256_storeu_pd((double *)(b11 + cs_b * 4), ymm0); //store B11[0][0-3]
            _mm256_storeu_pd((double *)(b11 + cs_b * 5), ymm1); //store B11[1][0-3]
        }
        dim_t n_remainder = j + d_nr;
        if((n_remainder >= 4))
        {
            a10 = D_A_pack;
            a11 = L + (i*cs_a) + (i*rs_a);                             //pointer to block of A to be used for TRSM
            b01 = B + ((n_remainder - 4)* cs_b) + i + 4;        //pointer to block of B to be used for GEMM
            b11 = B + ((n_remainder - 4)* cs_b) + i;            //pointer to block of B to be used for TRSM

            k_iter = (m - i - 4);                           //number of times GEMM to be performed(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            ///GEMM code begins///
            BLIS_DTRSM_SMALL_GEMM_4mx4n(a10,b01,cs_b,p_lda,k_iter)

            ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     //register to hold alpha

            ///implement TRSM///

            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));
            ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *1));
            ymm2 = _mm256_loadu_pd((double const *)(b11 + cs_b *2));
            ymm3 = _mm256_loadu_pd((double const *)(b11 + cs_b *3));
            ymm0 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);
            ymm1 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);
            ymm2 = _mm256_fmsub_pd(ymm2, ymm16, ymm10);
            ymm3 = _mm256_fmsub_pd(ymm3, ymm16, ymm11);

            ///transpose of B11//
            ///unpacklow///
            ymm9 = _mm256_unpacklo_pd(ymm0, ymm1);              //B11[0][0] B11[0][1] B11[2][0] B11[2][1]
            ymm11 = _mm256_unpacklo_pd(ymm2, ymm3);             //B11[0][2] B11[0][3] B11[2][2] B11[2][3]

            //rearrange low elements
            ymm8 = _mm256_permute2f128_pd(ymm9,ymm11,0x20);     //B11[0][0] B11[0][1] B11[0][2] B11[0][3]
            ymm10 = _mm256_permute2f128_pd(ymm9,ymm11,0x31);    //B11[2][0] B11[2][1] B11[2][2] B11[2][3]

            ////unpackhigh////
            ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);              //B11[1][0] B11[1][1] B11[3][0] B11[3][1]
            ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);              //B11[1][2] B11[1][3] B11[3][2] B11[3][3]

            //rearrange high elements
            ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);      //B11[1][0] B11[1][1] B11[1][2] B11[1][3]
            ymm11 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);     //B11[3][0] B11[3][1] B11[3][2] B11[3][3]

            //extract a33
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

            //perform mul operation
            ymm11 = DTRSM_SMALL_DIV_OR_SCALE(ymm11, ymm1);

            //extract a22
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            //(ROw3): FMA operations
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 3*rs_a));
            ymm10 = _mm256_fnmadd_pd(ymm2, ymm11, ymm10);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a + 3*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm2, ymm11, ymm9);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));
            ymm8 = _mm256_fnmadd_pd(ymm2, ymm11, ymm8);

            //perform mul operation
            ymm10 = DTRSM_SMALL_DIV_OR_SCALE(ymm10, ymm1);

            //extract a11
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(ROw2): FMA operations
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a + 2*rs_a));
            ymm9 = _mm256_fnmadd_pd(ymm2, ymm10, ymm9);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));
            ymm8 = _mm256_fnmadd_pd(ymm2, ymm10, ymm8);

            //perform mul operation
            ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm1);

            //extract a00
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack));

            //(ROw2): FMA operations
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + 1*rs_a));
            ymm8 = _mm256_fnmadd_pd(ymm2, ymm9, ymm8);

            //perform mul operation
            ymm8 = DTRSM_SMALL_DIV_OR_SCALE(ymm8, ymm1);

            //unpacklow//
            ymm1 = _mm256_unpacklo_pd(ymm8, ymm9);              //B11[0][0] B11[1][0] B11[0][2] B11[1][2]
            ymm3 = _mm256_unpacklo_pd(ymm10, ymm11);            //B11[2][0] B11[3][0] B11[2][2] B11[3][2]

            //rearrange low elements
            ymm0 = _mm256_permute2f128_pd(ymm1, ymm3, 0x20);    //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
            ymm2 = _mm256_permute2f128_pd(ymm1, ymm3, 0x31);    //B11[0][2] B11[1][2] B11[2][2] B11[3][2]

            ///unpack high///
            ymm8 = _mm256_unpackhi_pd(ymm8, ymm9);              //B11[0][1] B11[1][1] B11[0][3] B11[1][3]
            ymm9 = _mm256_unpackhi_pd(ymm10, ymm11);            //B11[2][1] B11[3][1] B11[2][3] B11[3][3]

            //rearrange high elements
            ymm1 = _mm256_permute2f128_pd(ymm8, ymm9, 0x20);    //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm3 = _mm256_permute2f128_pd(ymm8, ymm9, 0x31);    //B11[0][3] B11[1][3] B11[2][3] B11[3][3]

            _mm256_storeu_pd((double *)(b11 + cs_b * 0), ymm0);    //store B11[0][0-3]
            _mm256_storeu_pd((double *)(b11 + cs_b * 1), ymm1);    //store B11[1][0-3]
            _mm256_storeu_pd((double *)(b11 + cs_b * 2), ymm2);    //store B11[2][0-3]
            _mm256_storeu_pd((double *)(b11 + cs_b * 3), ymm3);    //store B11[3][0-3]
            n_remainder = n_remainder - 4;
        }

        if(n_remainder)   //implementation fo remaining columns(when 'N' is not a multiple of d_nr)() n = 3
        {
            a10 = D_A_pack;
            a11 = L + (i*cs_a) + (i*rs_a);
            b01 = B + i + 4;
            b11 = B + i;

            k_iter = (m - i - 4);

            ymm8 = _mm256_setzero_pd();
            ymm9 = _mm256_setzero_pd();
            ymm10 = _mm256_setzero_pd();

            if(3 == n_remainder)
            {
                BLIS_DTRSM_SMALL_GEMM_4mx3n(a10,b01,cs_b,p_lda,k_iter)

                ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     //register to hold alpha

                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));    //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
                ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *1));    //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
                ymm2 = _mm256_loadu_pd((double const *)(b11 + cs_b *2));    //B11[0][2] B11[1][2] B11[2][2] B11[3][2]

                ymm0 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);                  //B11[0-3][0] * alpha -= B01[0-3][0]
                ymm1 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);                  //B11[0-3][1] * alpha -= B01[0-3][1]
                ymm2 = _mm256_fmsub_pd(ymm2, ymm16, ymm10);                 //B11[0-3][2] * alpha -= B01[0-3][2]
                ymm3 = _mm256_broadcast_sd((double const *)(&ones));
            }
            else if(2 == n_remainder)
            {
                BLIS_DTRSM_SMALL_GEMM_4mx2n(a10,b01,cs_b,p_lda,k_iter)

                ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     //register to hold alpha

                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));    //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
                ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *1));    //B11[0][1] B11[1][1] B11[2][1] B11[3][1]

                ymm0 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);                  //B11[0-3][0] * alpha -= B01[0-3][0]
                ymm1 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);                  //B11[0-3][1] * alpha -= B01[0-3][1]
                ymm2 = _mm256_broadcast_sd((double const *)(&ones));
                ymm3 = _mm256_broadcast_sd((double const *)(&ones));
            }
            else if(1 == n_remainder)
            {
                BLIS_DTRSM_SMALL_GEMM_4mx1n(a10,b01,cs_b,p_lda,k_iter)

                ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     //register to hold alpha

                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));    //B11[0][0] B11[1][0] B11[2][0] B11[3][0]

                ymm0 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);                  //B11[0-3][0] * alpha -= B01[0-3][0]
                ymm1 = _mm256_broadcast_sd((double const *)(&ones));
                ymm2 = _mm256_broadcast_sd((double const *)(&ones));
                ymm3 = _mm256_broadcast_sd((double const *)(&ones));
            }

            ///implement TRSM///

            ///transpose of B11//
            ///unpacklow///
            ymm9 = _mm256_unpacklo_pd(ymm0, ymm1);                      //B11[0][0] B11[0][1] B11[2][0] B11[2][1]
            ymm11 = _mm256_unpacklo_pd(ymm2, ymm3);                     //B11[0][2] B11[0][3] B11[2][2] B11[2][3]

            //rearrange low elements
            ymm8 = _mm256_permute2f128_pd(ymm9,ymm11,0x20);             //B11[0][0] B11[0][1] B11[0][2] B11[0][3]
            ymm10 = _mm256_permute2f128_pd(ymm9,ymm11,0x31);            //B11[2][0] B11[2][1] B11[2][2] B11[2][3]

            ////unpackhigh////
            ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);                      //B11[1][0] B11[1][1] B11[3][0] B11[3][1]
            ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);                      //B11[1][2] B11[1][3] B11[3][2] B11[3][3]

            //rearrange high elements
            ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);              //B11[1][0] B11[1][1] B11[1][2] B11[1][3]
            ymm11 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);             //B11[3][0] B11[3][1] B11[3][2] B11[3][3]

            //extract a33
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

            //perform mul operation
            ymm11 = DTRSM_SMALL_DIV_OR_SCALE(ymm11, ymm1);

            //extract a22
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            ymm6 = _mm256_broadcast_sd((double const *)(a11 + 2*cs_a + 3*rs_a));
            ymm7 = _mm256_broadcast_sd((double const *)(a11 + cs_a + 3*rs_a));
            ymm16 = _mm256_broadcast_sd((double const *)(a11 + 3*rs_a));

            //(ROw3): FMA operations
            ymm10 = _mm256_fnmadd_pd(ymm6, ymm11, ymm10);
            ymm9 = _mm256_fnmadd_pd(ymm7, ymm11, ymm9);
            ymm8 = _mm256_fnmadd_pd(ymm16, ymm11, ymm8);

            //perform mul operation
            ymm10 = DTRSM_SMALL_DIV_OR_SCALE(ymm10, ymm1);

            //extract a11
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            ymm7 = _mm256_broadcast_sd((double const *)(a11 + cs_a + 2*rs_a));
            ymm16 = _mm256_broadcast_sd((double const *)(a11 + 2*rs_a));

            //(ROw2): FMA operations
            ymm9 = _mm256_fnmadd_pd(ymm7, ymm10, ymm9);
            ymm8 = _mm256_fnmadd_pd(ymm16, ymm10, ymm8);

            //perform mul operation
            ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm1);

            //extract a00
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack));

            ymm16 = _mm256_broadcast_sd((double const *)(a11 + 1*rs_a));

            //(ROw2): FMA operations
            ymm8 = _mm256_fnmadd_pd(ymm16, ymm9, ymm8);

            //perform mul operation
            ymm8 = DTRSM_SMALL_DIV_OR_SCALE(ymm8, ymm1);

            //unpacklow//
            ymm1 = _mm256_unpacklo_pd(ymm8, ymm9);              //B11[0][0] B11[1][0] B11[0][2] B11[1][2]
            ymm3 = _mm256_unpacklo_pd(ymm10, ymm11);            //B11[2][0] B11[3][0] B11[2][2] B11[3][2]

            //rearrange low elements
            ymm0 = _mm256_permute2f128_pd(ymm1, ymm3, 0x20);    //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
            ymm2 = _mm256_permute2f128_pd(ymm1, ymm3, 0x31);    //B11[0][2] B11[1][2] B11[2][2] B11[3][2]

            ///unpack high///
            ymm8 = _mm256_unpackhi_pd(ymm8, ymm9);              //B11[0][1] B11[1][1] B11[0][3] B11[1][3]
            ymm9 = _mm256_unpackhi_pd(ymm10, ymm11);            //B11[2][1] B11[3][1] B11[2][3] B11[3][3]

            //rearrange high elements
            ymm1 = _mm256_permute2f128_pd(ymm8, ymm9, 0x20);    //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm3 = _mm256_permute2f128_pd(ymm8, ymm9, 0x31);    //B11[0][3] B11[1][3] B11[2][3] B11[3][3]

            if(3 == n_remainder)
            {
                _mm256_storeu_pd((double *)(b11 + cs_b * 0), ymm0);    //store B11[0][0-3]
                _mm256_storeu_pd((double *)(b11 + cs_b * 1), ymm1);    //store B11[1][0-3]
                _mm256_storeu_pd((double *)(b11 + cs_b * 2), ymm2);    //store B11[2][0-3]
            }
            else if(2 == n_remainder)
            {
                _mm256_storeu_pd((double *)(b11 + cs_b * 0), ymm0);    //store B11[0][0-3]
                _mm256_storeu_pd((double *)(b11 + cs_b * 1), ymm1);    //store B11[1][0-3]
            }
            else if(1 == n_remainder)
            {
                _mm256_storeu_pd((double *)(b11 + cs_b * 0), ymm0);    //store B11[0][0-3]
            }
        }
        m_remainder -= 4;
    }

        a10 = L + m_remainder*rs_a;

        // Do transpose for a10 & store in D_A_pack
        double *ptr_a10_dup = D_A_pack;
        if(3 == m_remainder) // Repetative A blocks will be 3*3
        {
            dim_t p_lda = 4; // packed leading dimension
            if(transa)
            {
                for(dim_t x =0;x < m-m_remainder;x+=p_lda)
                {
                    ymm0 = _mm256_loadu_pd((double const *)(a10));
                    ymm1 = _mm256_loadu_pd((double const *)(a10 + cs_a));
                    ymm2 = _mm256_loadu_pd((double const *)(a10 + cs_a * 2));
                    ymm3 = _mm256_broadcast_sd((double const *)&ones);

                    ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
                    ymm5 = _mm256_unpacklo_pd(ymm2, ymm3);

                    ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                    ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);

                    ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
                    ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);

                    ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                    ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

                    _mm256_storeu_pd((double *)(ptr_a10_dup), ymm6);
                    _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda), ymm7);
                    _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*2), ymm8);
                    _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*3), ymm9);

                    a10 += p_lda;
                    ptr_a10_dup += p_lda*p_lda;
                }
            }
            else
            {
                for(dim_t x =0;x < m-m_remainder;x++)
                {
                    ymm0 = _mm256_loadu_pd((double const *)(a10 + x*rs_a));
                    _mm256_storeu_pd((double *)(ptr_a10_dup + x*p_lda), ymm0);
                }
            }

            //cols
            for(j = (n - d_nr); (j + 1) > 0; j -= d_nr)    //loop along 'N' dimension
            {
                a10 = D_A_pack;
                a11 = L;                                   //pointer to block of A to be used for TRSM
                b01 = B + (j* cs_b) + m_remainder;         //pointer to block of B to be used for GEMM
                b11 = B + (j* cs_b);                       //pointer to block of B to be used for TRSM

                k_iter = (m - m_remainder);            //number of times GEMM to be performed(in blocks of 4x4)

                /*Fill zeros into ymm registers used in gemm accumulations */
                BLIS_SET_YMM_REG_ZEROS

                ///GEMM code begins///
                BLIS_DTRSM_SMALL_GEMM_4mx6n(a10,b01,cs_b,p_lda,k_iter)

                ///GEMM code ends///
                ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));    //register to store alpha value
                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));
                ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *1));
                ymm2 = _mm256_loadu_pd((double const *)(b11 + cs_b *2));
                ymm3 = _mm256_loadu_pd((double const *)(b11 + cs_b *3));
                ymm8 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);
                ymm9 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);
                ymm10 = _mm256_fmsub_pd(ymm2, ymm16, ymm10);
                ymm11 = _mm256_fmsub_pd(ymm3, ymm16, ymm11);

                ymm0 = _mm256_blend_pd(ymm8, ymm0, 0x08);
                ymm1 = _mm256_blend_pd(ymm9, ymm1, 0x08);
                ymm2 = _mm256_blend_pd(ymm10, ymm2, 0x08);
                ymm3 = _mm256_blend_pd(ymm11, ymm3, 0x08);

                _mm256_storeu_pd((double *)(b11), ymm0);                   //store(B11[0-3][0])
                _mm256_storeu_pd((double *)(b11 + cs_b * 1), ymm1);        //store(B11[0-3][1])
                _mm256_storeu_pd((double *)(b11 + cs_b * 2), ymm2);        //store(B11[0-3][2])
                _mm256_storeu_pd((double *)(b11 + cs_b * 3), ymm3);        //store(B11[0-3][3])

                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *4));
                ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *5));
                ymm8 = _mm256_fmsub_pd(ymm0, ymm16, ymm4);
                ymm9 = _mm256_fmsub_pd(ymm1, ymm16, ymm5);

                ymm0 = _mm256_blend_pd(ymm8, ymm0, 0x08);
                ymm1 = _mm256_blend_pd(ymm9, ymm1, 0x08);

                _mm256_storeu_pd((double *)(b11 + cs_b * 4), ymm0);        //store(B11[0-3][2])
                _mm256_storeu_pd((double *)(b11 + cs_b * 5), ymm1);        //store(B11[0-3][3])

                if(transa)
                dtrsm_AltXB_ref(a11, b11, m_remainder, 6, cs_a, cs_b, is_unitdiag);
                else
                dtrsm_AuXB_ref(a11, b11, m_remainder, 6, rs_a, cs_b, is_unitdiag);
            }

            dim_t n_remainder = j + d_nr;
            if((n_remainder >= 4))
            {
                a10 = D_A_pack;
                a11 = L;                                                   //pointer to block of A to be used for TRSM
                b01 = B + ((n_remainder - 4)* cs_b) + m_remainder;         //pointer to block of B to be used for GEMM
                b11 = B + ((n_remainder - 4)* cs_b);                       //pointer to block of B to be used for TRSM

                k_iter = (m - m_remainder);                            //number of times GEMM to be performed(in blocks of 4x4)

                /*Fill zeros into ymm registers used in gemm accumulations */
                BLIS_SET_YMM_REG_ZEROS

                ///GEMM code begins///
                BLIS_DTRSM_SMALL_GEMM_4mx4n(a10,b01,cs_b,p_lda,k_iter)

                ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     //register to hold alpha

                ///implement TRSM///

                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));
                ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *1));
                ymm2 = _mm256_loadu_pd((double const *)(b11 + cs_b *2));
                xmm5 = _mm_loadu_pd((double const*)(b11 + cs_b * 3));
                ymm3 = _mm256_broadcast_sd((double const *)(b11 + cs_b*3 + 2));
                ymm3 = _mm256_insertf128_pd(ymm3, xmm5, 0);

                ymm8 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);
                ymm9 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);
                ymm10 = _mm256_fmsub_pd(ymm2, ymm16, ymm10);
                ymm11 = _mm256_fmsub_pd(ymm3, ymm16, ymm11);

                ymm0 = _mm256_blend_pd(ymm8, ymm0, 0x08);
                ymm1 = _mm256_blend_pd(ymm9, ymm1, 0x08);
                ymm2 = _mm256_blend_pd(ymm10, ymm2, 0x08);
                ymm3 = _mm256_blend_pd(ymm11, ymm3, 0x08);

                _mm256_storeu_pd((double *)(b11), ymm0);                   //store(B11[0-3][0])
                _mm256_storeu_pd((double *)(b11 + cs_b * 1), ymm1);        //store(B11[0-3][1])
                _mm256_storeu_pd((double *)(b11 + cs_b * 2), ymm2);        //store(B11[0-3][2])
                xmm5 = _mm256_extractf128_pd(ymm3, 0);
                _mm_storeu_pd((double *)(b11 + cs_b * 3),xmm5);
                _mm_storel_pd((b11 + cs_b * 3 + 2), _mm256_extractf128_pd(ymm3, 1));

                if(transa)
                dtrsm_AltXB_ref(a11, b11, m_remainder, 4, cs_a, cs_b, is_unitdiag);
                else
                dtrsm_AuXB_ref(a11, b11, m_remainder, 4, rs_a, cs_b, is_unitdiag);
                n_remainder -= 4;
            }

            if(n_remainder)
            {
                a10 = D_A_pack;
                a11 = L;                         //pointer to block of A to be used for TRSM
                b01 = B + m_remainder;           //pointer to block of B to be used for GEMM
                b11 = B;                         //pointer to block of B to be used for TRSM

                k_iter = (m - m_remainder);  //number of times GEMM to be performed(in blocks of 4x4)

                /*Fill zeros into ymm registers used in gemm accumulations */
                BLIS_SET_YMM_REG_ZEROS

                if(3 == n_remainder)
                {
                    ///GEMM code begins///
                    BLIS_DTRSM_SMALL_GEMM_4mx3n(a10,b01,cs_b,p_lda,k_iter)

                    BLIS_PRE_DTRSM_SMALL_3M_3N(AlphaVal,b11,cs_b)

                    if(transa)
                    dtrsm_AltXB_ref(a11, b11, m_remainder, 3, cs_a, cs_b, is_unitdiag);
                    else
                    dtrsm_AuXB_ref(a11, b11, m_remainder, 3, rs_a, cs_b, is_unitdiag);
                }
                else if(2 == n_remainder)
                {
                    ///GEMM code begins///
                    BLIS_DTRSM_SMALL_GEMM_4mx2n(a10,b01,cs_b,p_lda,k_iter)

                    BLIS_PRE_DTRSM_SMALL_3M_2N(AlphaVal,b11,cs_b)

                    if(transa)
                    dtrsm_AltXB_ref(a11, b11, m_remainder, 2, cs_a, cs_b, is_unitdiag);
                    else
                    dtrsm_AuXB_ref(a11, b11, m_remainder, 2, rs_a, cs_b, is_unitdiag);
                }
                else if(1 == n_remainder)
                {
                    ///GEMM code begins///
                    BLIS_DTRSM_SMALL_GEMM_4mx1n(a10,b01,cs_b,p_lda,k_iter)

                    BLIS_PRE_DTRSM_SMALL_3M_1N(AlphaVal,b11,cs_b)

                    if(transa)
                    dtrsm_AltXB_ref(a11, b11, m_remainder, 1, cs_a, cs_b, is_unitdiag);
                    else
                    dtrsm_AuXB_ref(a11, b11, m_remainder, 1, rs_a, cs_b, is_unitdiag);
                }
            }
        }
        else if(2 == m_remainder) // Repetative A blocks will be 2*2
        {
            dim_t p_lda = 4; // packed leading dimension
            if(transa)
            {
                for(dim_t x =0;x < m-m_remainder;x+=p_lda)
                {
                    ymm0 = _mm256_loadu_pd((double const *)(a10));
                    ymm1 = _mm256_loadu_pd((double const *)(a10 + cs_a));
                    ymm2 = _mm256_broadcast_sd((double const *)&ones);
                    ymm3 = _mm256_broadcast_sd((double const *)&ones);

                    ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
                    ymm5 = _mm256_unpacklo_pd(ymm2, ymm3);

                    ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                    ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);

                    ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
                    ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);

                    ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                    ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

                    _mm256_storeu_pd((double *)(ptr_a10_dup), ymm6);
                    _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda), ymm7);
                    _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*2), ymm8);
                    _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*3), ymm9);

                    a10 += p_lda;
                    ptr_a10_dup += p_lda*p_lda;
                }
            }
            else
            {
                for(dim_t x =0;x < m-m_remainder;x++)
                {
                    ymm0 = _mm256_loadu_pd((double const *)(a10 + x*rs_a));
                    _mm256_storeu_pd((double *)(ptr_a10_dup + x*p_lda), ymm0);
                }
            }
            //cols
            for(j = (n - d_nr); (j + 1) > 0; j -= d_nr)    //loop along 'N' dimension
            {
                a10 = D_A_pack;
                a11 = L;                                   //pointer to block of A to be used for TRSM
                b01 = B + (j* cs_b) + m_remainder;         //pointer to block of B to be used for GEMM
                b11 = B + (j* cs_b);                       //pointer to block of B to be used for TRSM

                k_iter = (m - m_remainder);            //number of times GEMM to be performed(in blocks of 4x4)

                /*Fill zeros into ymm registers used in gemm accumulations */
                BLIS_SET_YMM_REG_ZEROS

                ///GEMM code begins///
                BLIS_DTRSM_SMALL_GEMM_4mx6n(a10,b01,cs_b,p_lda,k_iter)

                ///GEMM code ends///
                ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));    //register to store alpha value
                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));
                ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *1));
                ymm2 = _mm256_loadu_pd((double const *)(b11 + cs_b *2));
                ymm3 = _mm256_loadu_pd((double const *)(b11 + cs_b *3));
                ymm8 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);
                ymm9 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);
                ymm10 = _mm256_fmsub_pd(ymm2, ymm16, ymm10);
                ymm11 = _mm256_fmsub_pd(ymm3, ymm16, ymm11);

                ymm0 = _mm256_blend_pd(ymm8, ymm0, 0x0C);
                ymm1 = _mm256_blend_pd(ymm9, ymm1, 0x0C);
                ymm2 = _mm256_blend_pd(ymm10, ymm2, 0x0C);
                ymm3 = _mm256_blend_pd(ymm11, ymm3, 0x0C);

                _mm256_storeu_pd((double *)(b11), ymm0);                   //store(B11[0-3][0])
                _mm256_storeu_pd((double *)(b11 + cs_b * 1), ymm1);        //store(B11[0-3][1])
                _mm256_storeu_pd((double *)(b11 + cs_b * 2), ymm2);        //store(B11[0-3][2])
                _mm256_storeu_pd((double *)(b11 + cs_b * 3), ymm3);        //store(B11[0-3][3])

                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *4));
                ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *5));
                ymm8 = _mm256_fmsub_pd(ymm0, ymm16, ymm4);
                ymm9 = _mm256_fmsub_pd(ymm1, ymm16, ymm5);

                ymm0 = _mm256_blend_pd(ymm8, ymm0, 0x0C);
                ymm1 = _mm256_blend_pd(ymm9, ymm1, 0x0C);

                _mm256_storeu_pd((double *)(b11 + cs_b * 4), ymm0);        //store(B11[0-3][2])
                _mm256_storeu_pd((double *)(b11 + cs_b * 5), ymm1);        //store(B11[0-3][3])

                if(transa)
                dtrsm_AltXB_ref(a11, b11, m_remainder, 6, cs_a, cs_b, is_unitdiag);
                else
                dtrsm_AuXB_ref(a11, b11, m_remainder, 6, rs_a, cs_b, is_unitdiag);
            }
            dim_t n_remainder = j + d_nr;
            if((n_remainder >= 4))
            {
                a10 = D_A_pack;
                a11 = L;                                                  //pointer to block of A to be used for TRSM
                b01 = B + ((n_remainder - 4)* cs_b) + m_remainder;        //pointer to block of B to be used for GEMM
                b11 = B + ((n_remainder - 4)* cs_b);                      //pointer to block of B to be used for TRSM

                k_iter = (m - m_remainder);                           //number of times GEMM to be performed(in blocks of 4x4)

                /*Fill zeros into ymm registers used in gemm accumulations */
                BLIS_SET_YMM_REG_ZEROS

                ///GEMM code begins///
                BLIS_DTRSM_SMALL_GEMM_4mx4n(a10,b01,cs_b,p_lda,k_iter)

                ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     //register to hold alpha

                ///implement TRSM///

                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));
                ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *1));
                ymm2 = _mm256_loadu_pd((double const *)(b11 + cs_b *2));
                xmm5 = _mm_loadu_pd((double const*)(b11 + cs_b * 3));
                ymm3 = _mm256_insertf128_pd(ymm3, xmm5, 0);

                ymm8 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);
                ymm9 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);
                ymm10 = _mm256_fmsub_pd(ymm2, ymm16, ymm10);
                ymm11 = _mm256_fmsub_pd(ymm3, ymm16, ymm11);

                ymm0 = _mm256_blend_pd(ymm8, ymm0, 0x0C);
                ymm1 = _mm256_blend_pd(ymm9, ymm1, 0x0C);
                ymm2 = _mm256_blend_pd(ymm10, ymm2, 0x0C);
                ymm3 = _mm256_blend_pd(ymm11, ymm3, 0x0C);

                _mm256_storeu_pd((double *)(b11), ymm0);                   //store(B11[0-3][0])
                _mm256_storeu_pd((double *)(b11 + cs_b * 1), ymm1);        //store(B11[0-3][1])
                _mm256_storeu_pd((double *)(b11 + cs_b * 2), ymm2);        //store(B11[0-3][2])
                xmm5 = _mm256_extractf128_pd(ymm3, 0);
                _mm_storeu_pd((double *)(b11 + cs_b * 3), xmm5);

                if(transa)
                dtrsm_AltXB_ref(a11, b11, m_remainder, 4, cs_a, cs_b, is_unitdiag);
                else
                dtrsm_AuXB_ref(a11, b11, m_remainder, 4, rs_a, cs_b, is_unitdiag);
                n_remainder -= 4;
            }
            if(n_remainder)
            {
                a10 = D_A_pack;
                a11 = L;                         //pointer to block of A to be used for TRSM
                b01 = B + m_remainder;           //pointer to block of B to be used for GEMM
                b11 = B;                         //pointer to block of B to be used for TRSM

                k_iter = (m - m_remainder);  //number of times GEMM to be performed(in blocks of 4x4)

                /*Fill zeros into ymm registers used in gemm accumulations */
                BLIS_SET_YMM_REG_ZEROS

                if(3 == n_remainder)
                {
                    ///GEMM code begins///
                    BLIS_DTRSM_SMALL_GEMM_4mx3n(a10,b01,cs_b,p_lda,k_iter)

                    BLIS_PRE_DTRSM_SMALL_2M_3N(AlphaVal,b11,cs_b)

                    if(transa)
                    dtrsm_AltXB_ref(a11, b11, m_remainder, 3, cs_a, cs_b, is_unitdiag);
                    else
                    dtrsm_AuXB_ref(a11, b11, m_remainder, 3, rs_a, cs_b, is_unitdiag);
                }
                else if(2 == n_remainder)
                {
                    ///GEMM code begins///
                    BLIS_DTRSM_SMALL_GEMM_4mx2n(a10,b01,cs_b,p_lda,k_iter)

                    BLIS_PRE_DTRSM_SMALL_2M_2N(AlphaVal,b11,cs_b)

                    if(transa)
                    dtrsm_AltXB_ref(a11, b11, m_remainder, 2, cs_a, cs_b, is_unitdiag);
                    else
                    dtrsm_AuXB_ref(a11, b11, m_remainder, 2, rs_a, cs_b, is_unitdiag);
                }
                else if(1 == n_remainder)
                {
                    ///GEMM code begins///
                    BLIS_DTRSM_SMALL_GEMM_4mx1n(a10,b01,cs_b,p_lda,k_iter)

                    BLIS_PRE_DTRSM_SMALL_2M_1N(AlphaVal,b11,cs_b)
                    if(transa)
                    dtrsm_AltXB_ref(a11, b11, m_remainder, 1, cs_a, cs_b, is_unitdiag);
                    else
                    dtrsm_AuXB_ref(a11, b11, m_remainder, 1, rs_a, cs_b, is_unitdiag);
                }
            }

        }
        else if(1 == m_remainder) // Repetative A blocks will be 1*1
        {
            dim_t p_lda = 4; // packed leading dimension
            if(transa)
            {
                for(dim_t x =0;x < m-m_remainder;x+=p_lda)
                {
                    ymm0 = _mm256_loadu_pd((double const *)(a10));
                    ymm1 = _mm256_broadcast_sd((double const *)&ones);
                    ymm2 = _mm256_broadcast_sd((double const *)&ones);
                    ymm3 = _mm256_broadcast_sd((double const *)&ones);

                    ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
                    ymm5 = _mm256_unpacklo_pd(ymm2, ymm3);

                    ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                    ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);

                    ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
                    ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);

                    ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                    ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

                    _mm256_storeu_pd((double *)(ptr_a10_dup), ymm6);
                    _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda), ymm7);
                    _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*2), ymm8);
                    _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*3), ymm9);

                    a10 += p_lda;
                    ptr_a10_dup += p_lda*p_lda;
                }
            }
            else
            {
                for(dim_t x =0;x < m-m_remainder;x++)
                {
                    ymm0 = _mm256_loadu_pd((double const *)(a10 + x*rs_a));
                    _mm256_storeu_pd((double *)(ptr_a10_dup + x*p_lda), ymm0);
                }
            }
            //cols
            for(j = (n - d_nr); (j + 1) > 0; j -= d_nr)      //loop along 'N' dimension
            {
                a10 = D_A_pack;
                a11 = L;                                     //pointer to block of A to be used for TRSM
                b01 = B + (j* cs_b) + m_remainder;           //pointer to block of B to be used for GEMM
                b11 = B + (j* cs_b);                         //pointer to block of B to be used for TRSM

                k_iter = (m - m_remainder);              //number of times GEMM to be performed(in blocks of 4x4)

                /*Fill zeros into ymm registers used in gemm accumulations */
                BLIS_SET_YMM_REG_ZEROS

                ///GEMM code begins///
                BLIS_DTRSM_SMALL_GEMM_4mx6n(a10,b01,cs_b,p_lda,k_iter)

                ///GEMM code ends///
                ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));    //register to store alpha value
                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));
                ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *1));
                ymm2 = _mm256_loadu_pd((double const *)(b11 + cs_b *2));
                ymm3 = _mm256_loadu_pd((double const *)(b11 + cs_b *3));
                ymm8 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);
                ymm9 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);
                ymm10 = _mm256_fmsub_pd(ymm2, ymm16, ymm10);
                ymm11 = _mm256_fmsub_pd(ymm3, ymm16, ymm11);

                ymm0 = _mm256_blend_pd(ymm8, ymm0, 0x0E);
                ymm1 = _mm256_blend_pd(ymm9, ymm1, 0x0E);
                ymm2 = _mm256_blend_pd(ymm10, ymm2, 0x0E);
                ymm3 = _mm256_blend_pd(ymm11, ymm3, 0x0E);

                _mm256_storeu_pd((double *)(b11), ymm0);                   //store(B11[0-3][0])
                _mm256_storeu_pd((double *)(b11 + cs_b * 1), ymm1);        //store(B11[0-3][1])
                _mm256_storeu_pd((double *)(b11 + cs_b * 2), ymm2);        //store(B11[0-3][2])
                _mm256_storeu_pd((double *)(b11 + cs_b * 3), ymm3);        //store(B11[0-3][3])

                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *4));
                ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *5));
                ymm8 = _mm256_fmsub_pd(ymm0, ymm16, ymm4);
                ymm9 = _mm256_fmsub_pd(ymm1, ymm16, ymm5);

                ymm0 = _mm256_blend_pd(ymm8, ymm0, 0x0E);
                ymm1 = _mm256_blend_pd(ymm9, ymm1, 0x0E);

                _mm256_storeu_pd((double *)(b11 + cs_b * 4), ymm0);        //store(B11[0-3][2])
                _mm256_storeu_pd((double *)(b11 + cs_b * 5), ymm1);        //store(B11[0-3][3])

                if(transa)
                dtrsm_AltXB_ref(a11, b11, m_remainder, 6, cs_a, cs_b, is_unitdiag);
                else
                dtrsm_AuXB_ref(a11, b11, m_remainder, 6, rs_a, cs_b, is_unitdiag);
            }
            dim_t n_remainder = j + d_nr;
            if((n_remainder >= 4))
            {
                a10 = D_A_pack;
                a11 = L;                                               //pointer to block of A to be used for TRSM
                b01 = B + ((n_remainder - 4)* cs_b) + m_remainder;     //pointer to block of B to be used for GEMM
                b11 = B + ((n_remainder - 4)* cs_b);                   //pointer to block of B to be used for TRSM

                k_iter = (m - m_remainder);                        //number of times GEMM to be performed(in blocks of 4x4)

                /*Fill zeros into ymm registers used in gemm accumulations */
                BLIS_SET_YMM_REG_ZEROS

                ///GEMM code begins///
                BLIS_DTRSM_SMALL_GEMM_4mx4n(a10,b01,cs_b,p_lda,k_iter)

                ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     //register to hold alpha

                ///implement TRSM///

                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));
                ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *1));
                ymm2 = _mm256_loadu_pd((double const *)(b11 + cs_b *2));
                ymm3 = _mm256_loadu_pd((double const *)(b11 + cs_b *3));
                ymm8 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);
                ymm9 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);
                ymm10 = _mm256_fmsub_pd(ymm2, ymm16, ymm10);
                ymm11 = _mm256_fmsub_pd(ymm3, ymm16, ymm11);

                ymm0 = _mm256_blend_pd(ymm8, ymm0, 0x0E);
                ymm1 = _mm256_blend_pd(ymm9, ymm1, 0x0E);
                ymm2 = _mm256_blend_pd(ymm10, ymm2, 0x0E);
                ymm3 = _mm256_blend_pd(ymm11, ymm3, 0x0E);

                _mm256_storeu_pd((double *)(b11), ymm0);                   //store(B11[0-3][0])
                _mm256_storeu_pd((double *)(b11 + cs_b * 1), ymm1);        //store(B11[0-3][1])
                _mm256_storeu_pd((double *)(b11 + cs_b * 2), ymm2);        //store(B11[0-3][2])
                _mm256_storeu_pd((double *)(b11 + cs_b * 3), ymm3);        //store(B11[0-3][3])

                if(transa)
                dtrsm_AltXB_ref(a11, b11, m_remainder, 4, cs_a, cs_b, is_unitdiag);
                else
                dtrsm_AuXB_ref(a11, b11, m_remainder, 4, rs_a, cs_b, is_unitdiag);
                n_remainder  -= 4;
            }
            if(n_remainder)
            {
                a10 = D_A_pack;
                a11 = L;                  //pointer to block of A to be used for TRSM
                b01 = B + m_remainder;    //pointer to block of B to be used for GEMM
                b11 = B;                  //pointer to block of B to be used for TRSM

                k_iter = (m - m_remainder);          //number of times GEMM to be performed(in blocks of 4x4)

                /*Fill zeros into ymm registers used in gemm accumulations */
                BLIS_SET_YMM_REG_ZEROS

                if(3 == n_remainder)
                {
                    ///GEMM code begins///
                    BLIS_DTRSM_SMALL_GEMM_4mx3n(a10,b01,cs_b,p_lda,k_iter)

                    BLIS_PRE_DTRSM_SMALL_1M_3N(AlphaVal,b11,cs_b)

                    if(transa)
                    dtrsm_AltXB_ref(a11, b11, m_remainder, 3, cs_a, cs_b, is_unitdiag);
                    else
                    dtrsm_AuXB_ref(a11, b11, m_remainder, 3, rs_a, cs_b, is_unitdiag);
                }
                else if(2 == n_remainder)
                {
                    ///GEMM code begins///
                    BLIS_DTRSM_SMALL_GEMM_4mx2n(a10,b01,cs_b,p_lda,k_iter)

                    BLIS_PRE_DTRSM_SMALL_1M_2N(AlphaVal,b11,cs_b)

                    if(transa)
                    dtrsm_AltXB_ref(a11, b11, m_remainder, 2, cs_a, cs_b, is_unitdiag);
                    else
                    dtrsm_AuXB_ref(a11, b11, m_remainder, 2, rs_a, cs_b, is_unitdiag);
                }
                else if(1 == n_remainder)
                {
                    ///GEMM code begins///
                    BLIS_DTRSM_SMALL_GEMM_4mx1n(a10,b01,cs_b,p_lda,k_iter)

                    BLIS_PRE_DTRSM_SMALL_1M_1N(AlphaVal,b11,cs_b)

                    if(transa)
                        dtrsm_AltXB_ref(a11, b11, m_remainder, 1, cs_a, cs_b, is_unitdiag);
                    else
                        dtrsm_AuXB_ref(a11, b11, m_remainder, 1, rs_a, cs_b, is_unitdiag);
                }
            }
        }

    if ((required_packing_A == 1) &&
         bli_mem_is_alloc( &local_mem_buf_A_s ))
    {
        bli_membrk_release(&rntm,&local_mem_buf_A_s);
    }
  return BLIS_SUCCESS;
}

/*  TRSM for the Left Upper case AX = alpha * B, Double precision
 *  A is Left side, upper-triangular, transpose, non-unit/unit diagonal
 *  dimensions A: mxm X: mxn B: mxn
    a10 ---->              b11--->
  ***********          *****************
  * *    *  *          *b01*b11*   *   *
   **a10 *  * a11  b11 *   *   *   *   *
    *********  |    |  *****************
     *a11*  *  |    |  *   *   *   *   *
      *  *  *  |    |  *   *   *   *   *
       ******  v    v  *****************
        *   *          *   *   *   *   *
         *  *          *   *   *   *   *
          * *          *****************
            *
    a11--->

 *  TRSM for the case AX = alpha * B, Double precision
 *  A is Left side, lower-triangular, no-transpose, non-unit/unit diagonal
 *  dimensions A: mxm X: mxn B: mxn

                            b01--->
    *                   *****************
    **                  *   *   *   *   *
    * *                 *   *   *   *   *
    *  *                *b01*   *   *   *
    *   *               *   *   *   *   *
a10 ******          b11 *****************
 |  *   * *          |  *   *   *   *   *
 |  *   *  *         |  *   *   *   *   *
 |  *a10*a11*        |  *b11*   *   *   *
 v  *   *    *       v  *   *   *   *   *
    ***********         *****************
    *   *    * *        *   *   *   *   *
    *   *    *  *       *   *   *   *   *
    *   *    *   *      *   *   *   *   *
    *   *    *    *     *   *   *   *   *
    ****************    *****************
        a11--->
*/
BLIS_INLINE err_t bli_dtrsm_small_AutXB_AlXB
(
    obj_t* AlphaObj,
    obj_t* a,
    obj_t* b,
    cntx_t* cntx,
    cntl_t* cntl
)
{
    dim_t m = bli_obj_length(b); // number of rows of matrix B
    dim_t n = bli_obj_width(b);  // number of columns of matrix B

    bool transa = bli_obj_has_trans(a);
    dim_t cs_a, rs_a;
    dim_t d_mr = 8,d_nr = 6;

    // Swap rs_a & cs_a in case of non-tranpose.
    if(transa)
    {
        cs_a = bli_obj_col_stride(a); // column stride of A
        rs_a = bli_obj_row_stride(a); // row stride of A
    }
    else
    {
        cs_a = bli_obj_row_stride(a); // row stride of A
        rs_a = bli_obj_col_stride(a); // column stride of A
    }
    dim_t cs_b = bli_obj_col_stride(b); // column stride of B

    dim_t i, j, k;    //loop variables
    dim_t k_iter;     //number of times GEMM to be performed

    double AlphaVal = *(double *)AlphaObj->buffer;    //value of alpha
    double *L =  a->buffer;       //pointer to  matrix A
    double *B =  b->buffer;       //pointer to matrix B

    double *a10, *a11, *b01, *b11;    //pointers that point to blocks for GEMM and TRSM

    double ones = 1.0;
    bool is_unitdiag = bli_obj_has_unit_diag(a);

    //scratch registers
    __m256d ymm0, ymm1, ymm2, ymm3;
    __m256d ymm4, ymm5, ymm6, ymm7;
    __m256d ymm8, ymm9, ymm10, ymm11;
    __m256d ymm12, ymm13, ymm14, ymm15;
    __m256d ymm16, ymm17, ymm18, ymm19;
    __m256d ymm20;

    __m128d xmm5;

    gint_t required_packing_A = 1;
    mem_t local_mem_buf_A_s = {0};
    double *D_A_pack = NULL;
    double d11_pack[d_mr] __attribute__((aligned(64)));
    rntm_t rntm;

    bli_rntm_init_from_global( &rntm );
    bli_rntm_set_num_threads_only( 1, &rntm );
    bli_membrk_rntm_set_membrk( &rntm );

    siz_t buffer_size = bli_pool_block_size(
                          bli_membrk_pool(
                            bli_packbuf_index(BLIS_BITVAL_BUFFER_FOR_A_BLOCK),
                            bli_rntm_membrk(&rntm)));

    if ( (d_mr * m * sizeof(double)) > buffer_size)
        return BLIS_NOT_YET_IMPLEMENTED;

    if (required_packing_A == 1)
    {
        // Get the buffer from the pool.
        bli_membrk_acquire_m(&rntm,
                             buffer_size,
                             BLIS_BITVAL_BUFFER_FOR_A_BLOCK,
                             &local_mem_buf_A_s);
        if(FALSE==bli_mem_is_alloc(&local_mem_buf_A_s)) return BLIS_NULL_POINTER;
        D_A_pack = bli_mem_buffer(&local_mem_buf_A_s);
        if(NULL==D_A_pack) return BLIS_NULL_POINTER;
    }

    /*
        Performs solving TRSM for 8 colmns at a time from  0 to m/8 in steps of d_mr
        a. Load, transpose, Pack A (a10 block), the size of packing 8x6 to 8x (m-8)
           First there will be no GEMM and no packing of a10 because it is only TRSM
        b. Using packed a10 block and b01 block perform GEMM operation
        c. Use GEMM outputs, perform TRSM operaton using a11, b11 and update B
        d. Repeat b,c for n rows of B in steps of d_nr
    */
    for(i = 0;(i+d_mr-1) < m; i += d_mr)  //loop along 'M' dimension
    {
        a10 = L + (i*cs_a);                 //pointer to block of A to be used for GEMM
        a11 = L + (i*rs_a) + (i*cs_a);
        dim_t p_lda = d_mr; // packed leading dimension

        if(transa)
        {
            /*
              Load, tranpose and pack current A block (a10) into packed buffer memory D_A_pack
              a. This a10 block is used in GEMM portion only and this
                 a10 block size will be increasing by d_mr for every next itteration
                 untill it reaches 8x(m-8) which is the maximum GEMM alone block size in A
              b. This packed buffer is reused to calculate all n rows of B matrix
            */
            bli_dtrsm_small_pack('L', i, 1, a10, cs_a, D_A_pack, p_lda,d_mr);

            /*
               Pack 8 diagonal elements of A block into an array
               a. This helps in utilze cache line efficiently in TRSM operation
               b. store ones when input is unit diagonal
            */
            dtrsm_small_pack_diag_element(is_unitdiag,a11,cs_a,d11_pack,d_mr);
        }
        else
        {
            bli_dtrsm_small_pack('L', i, 0, a10, rs_a, D_A_pack, p_lda,d_mr);
            dtrsm_small_pack_diag_element(is_unitdiag,a11,rs_a,d11_pack,d_mr);
        }

        /*
            a. Perform GEMM using a10, b01.
            b. Perform TRSM on a11, b11
            c. This loop GEMM+TRSM loops operates with 8x6 block size
               along n dimension for every d_nr rows of b01 where
               packed A buffer is reused in computing all n rows of B.
            d. Same approch is used in remaining fringe cases.
        */
        dim_t temp = n - d_nr + 1;
        for(j = 0; j < temp; j += d_nr)   //loop along 'N' dimension
        {
            a10 = D_A_pack;
            a11 = L + (i*rs_a) + (i*cs_a);     //pointer to block of A to be used for TRSM
            b01 = B + j*cs_b;           //pointer to block of B to be used for GEMM
            b11 = B + i + j* cs_b;      //pointer to block of B to be used for TRSM

            k_iter = i;

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            /*
              Peform GEMM between a10 and b01 blocks
              For first itteration there will be no GEMM operation
              where k_iter are zero
            */
            BLIS_DTRSM_SMALL_GEMM_8mx6n(a10,b01,cs_b,p_lda,k_iter)

            /*
               Load b11 of size 6x8 and multiply with alpha
               Add the GEMM output and perform inregister transose of b11
               to peform TRSM operation.
            */
            BLIS_DTRSM_SMALL_NREG_TRANSPOSE_6x8(b11,cs_b,AlphaVal)

            /*
                Compute 8x6 TRSM block by using GEMM block output in register
                a. The 8x6 input (gemm outputs) are stored in combinations of ymm registers
                    1. ymm8, ymm4 2. ymm9, ymm5 3. ymm10, ymm6, 4. ymm11, ymm7
                    5. ymm12, ymm17 6. ymm13,ymm18, 7. ymm14,ymm19 8. ymm15, ymm20
                    where ymm8-ymm15 holds 8x4 data and reaming 8x2 will be hold by
                    other registers
                b. Towards the end do in regiser transpose of TRSM output and store in b11
            */
            ////extract a00
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack));

            //perform mul operation
            ymm8 = DTRSM_SMALL_DIV_OR_SCALE(ymm8, ymm1);
            ymm4 = DTRSM_SMALL_DIV_OR_SCALE(ymm4, ymm1);

            //extract a11
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(ROw1): FMA operations
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*1));
            ymm9 = _mm256_fnmadd_pd(ymm2, ymm8, ymm9);
            ymm5 = _mm256_fnmadd_pd(ymm2, ymm4, ymm5);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*2));
            ymm10 = _mm256_fnmadd_pd(ymm2, ymm8, ymm10);
            ymm6 = _mm256_fnmadd_pd(ymm2, ymm4, ymm6);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*3));
            ymm11 = _mm256_fnmadd_pd(ymm2, ymm8, ymm11);
            ymm7 = _mm256_fnmadd_pd(ymm2, ymm4, ymm7);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*4));
            ymm12 = _mm256_fnmadd_pd(ymm2, ymm8, ymm12);
            ymm17 = _mm256_fnmadd_pd(ymm2, ymm4, ymm17);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*5));
            ymm13 = _mm256_fnmadd_pd(ymm2, ymm8, ymm13);
            ymm18 = _mm256_fnmadd_pd(ymm2, ymm4, ymm18);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*6));
            ymm14 = _mm256_fnmadd_pd(ymm2, ymm8, ymm14);
            ymm19 = _mm256_fnmadd_pd(ymm2, ymm4, ymm19);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*7));
            ymm15 = _mm256_fnmadd_pd(ymm2, ymm8, ymm15);
            ymm20 = _mm256_fnmadd_pd(ymm2, ymm4, ymm20);


            //perform mul operation
            ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm1);
            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm1);

            a11 += rs_a;

            //extract a22
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            //(ROw2): FMA operations
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*2));
            ymm10 = _mm256_fnmadd_pd(ymm2, ymm9, ymm10);
            ymm6 = _mm256_fnmadd_pd(ymm2, ymm5, ymm6);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*3));
            ymm11 = _mm256_fnmadd_pd(ymm2, ymm9, ymm11);
            ymm7 = _mm256_fnmadd_pd(ymm2, ymm5, ymm7);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*4));
            ymm12 = _mm256_fnmadd_pd(ymm2, ymm9, ymm12);
            ymm17 = _mm256_fnmadd_pd(ymm2, ymm5, ymm17);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*5));
            ymm13 = _mm256_fnmadd_pd(ymm2, ymm9, ymm13);
            ymm18 = _mm256_fnmadd_pd(ymm2, ymm5, ymm18);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*6));
            ymm14 = _mm256_fnmadd_pd(ymm2, ymm9, ymm14);
            ymm19 = _mm256_fnmadd_pd(ymm2, ymm5, ymm19);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*7));
            ymm15 = _mm256_fnmadd_pd(ymm2, ymm9, ymm15);
            ymm20 = _mm256_fnmadd_pd(ymm2, ymm5, ymm20);

            //perform mul operation
            ymm10 = DTRSM_SMALL_DIV_OR_SCALE(ymm10, ymm1);
            ymm6 = DTRSM_SMALL_DIV_OR_SCALE(ymm6, ymm1);

            a11 += rs_a;

            //extract a33
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

            //(ROw5): FMA operations
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*3));
            ymm11 = _mm256_fnmadd_pd(ymm2, ymm10, ymm11);
            ymm7 = _mm256_fnmadd_pd(ymm2, ymm6, ymm7);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*4));
            ymm12 = _mm256_fnmadd_pd(ymm2, ymm10, ymm12);
            ymm17 = _mm256_fnmadd_pd(ymm2, ymm6, ymm17);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*5));
            ymm13 = _mm256_fnmadd_pd(ymm2, ymm10, ymm13);
            ymm18 = _mm256_fnmadd_pd(ymm2, ymm6, ymm18);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*6));
            ymm14 = _mm256_fnmadd_pd(ymm2, ymm10, ymm14);
            ymm19 = _mm256_fnmadd_pd(ymm2, ymm6, ymm19);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*7));
            ymm15 = _mm256_fnmadd_pd(ymm2, ymm10, ymm15);
            ymm20 = _mm256_fnmadd_pd(ymm2, ymm6, ymm20);

            //perform mul operation
            ymm11 = DTRSM_SMALL_DIV_OR_SCALE(ymm11, ymm1);
            ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm1);

            a11 += rs_a;

            //extract a44
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 4));
            //(ROw4): FMA operations
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*4));
            ymm12 = _mm256_fnmadd_pd(ymm2, ymm11, ymm12);
            ymm17 = _mm256_fnmadd_pd(ymm2, ymm7, ymm17);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*5));
            ymm13 = _mm256_fnmadd_pd(ymm2, ymm11, ymm13);
            ymm18 = _mm256_fnmadd_pd(ymm2, ymm7, ymm18);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*6));
            ymm14 = _mm256_fnmadd_pd(ymm2, ymm11, ymm14);
            ymm19 = _mm256_fnmadd_pd(ymm2, ymm7, ymm19);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*7));
            ymm15 = _mm256_fnmadd_pd(ymm2, ymm11, ymm15);
            ymm20 = _mm256_fnmadd_pd(ymm2, ymm7, ymm20);

            //perform mul operation
            ymm12 = DTRSM_SMALL_DIV_OR_SCALE(ymm12, ymm1);
            ymm17 = DTRSM_SMALL_DIV_OR_SCALE(ymm17, ymm1);

            a11 += rs_a;

            //extract a55
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 5));

            //(ROw5): FMA operations
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*5));
            ymm13 = _mm256_fnmadd_pd(ymm2, ymm12, ymm13);
            ymm18 = _mm256_fnmadd_pd(ymm2, ymm17, ymm18);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*6));
            ymm14 = _mm256_fnmadd_pd(ymm2, ymm12, ymm14);
            ymm19 = _mm256_fnmadd_pd(ymm2, ymm17, ymm19);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*7));
            ymm15 = _mm256_fnmadd_pd(ymm2, ymm12, ymm15);
            ymm20 = _mm256_fnmadd_pd(ymm2, ymm17, ymm20);

            //perform mul operation
            ymm13 = DTRSM_SMALL_DIV_OR_SCALE(ymm13, ymm1);
            ymm18 = DTRSM_SMALL_DIV_OR_SCALE(ymm18, ymm1);

            a11 += rs_a;

            //extract a66
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 6));

            //(ROw6): FMA operations
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*6));
            ymm14 = _mm256_fnmadd_pd(ymm2, ymm13, ymm14);
            ymm19 = _mm256_fnmadd_pd(ymm2, ymm18, ymm19);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*7));
            ymm15 = _mm256_fnmadd_pd(ymm2, ymm13, ymm15);
            ymm20 = _mm256_fnmadd_pd(ymm2, ymm18, ymm20);

            //perform mul operation
            ymm14 = DTRSM_SMALL_DIV_OR_SCALE(ymm14, ymm1);
            ymm19 = DTRSM_SMALL_DIV_OR_SCALE(ymm19, ymm1);

            a11 += rs_a;

            //extract a77
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 7));

            //(ROw7): FMA operations
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*7));
            ymm15 = _mm256_fnmadd_pd(ymm2, ymm14, ymm15);
            ymm20 = _mm256_fnmadd_pd(ymm2, ymm19, ymm20);

            //perform mul operation
            ymm15 = DTRSM_SMALL_DIV_OR_SCALE(ymm15, ymm1);
            ymm20 = DTRSM_SMALL_DIV_OR_SCALE(ymm20, ymm1);

            a11 += rs_a;

            BLIS_DTRSM_SMALL_NREG_TRANSPOSE_8x6_AND_STORE(b11,cs_b)
        }

        dim_t n_rem = n-j;
        if(n_rem >= 4)
        {
            a10 = D_A_pack;
            a11 = L + (i*rs_a) + (i*cs_a);     //pointer to block of A to be used for TRSM
            b01 = B + j*cs_b;           //pointer to block of B to be used for GEMM
            b11 = B + i + j* cs_b;      //pointer to block of B to be used for TRSM

            k_iter = i ;      //number of times GEMM to be performed(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            ///GEMM code begins///
            BLIS_DTRSM_SMALL_GEMM_8mx4n(a10,b01,cs_b,p_lda,k_iter)

            ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     //register to hold alpha

            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));    //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
            ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *1));    //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm2 = _mm256_loadu_pd((double const *)(b11 + cs_b *2));    //B11[0][2] B11[1][2] B11[2][2] B11[3][2]
            ymm3 = _mm256_loadu_pd((double const *)(b11 + cs_b *3));    //B11[0][3] B11[1][3] B11[2][3] B11[3][3]
            ymm4 = _mm256_loadu_pd((double const *)(b11 + cs_b *0 + 4));    //B11[0][4] B11[1][4] B11[2][4] B11[3][4]
            ymm5 = _mm256_loadu_pd((double const *)(b11 + cs_b *1 + 4));    //B11[0][5] B11[1][5] B11[2][5] B11[3][5]
            ymm6 = _mm256_loadu_pd((double const *)(b11 + cs_b *2 + 4));    //B11[0][6] B11[1][6] B11[2][6] B11[3][6]
            ymm7 = _mm256_loadu_pd((double const *)(b11 + cs_b *3 + 4));    //B11[0][7] B11[1][7] B11[2][7] B11[3][7]

            ymm0 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);  //B11[0-3][0] * alpha -= B01[0-3][0]
            ymm1 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);  //B11[0-3][1] * alpha -= B01[0-3][1]
            ymm2 = _mm256_fmsub_pd(ymm2, ymm16, ymm10); //B11[0-3][2] * alpha -= B01[0-3][2]
            ymm3 = _mm256_fmsub_pd(ymm3, ymm16, ymm11); //B11[0-3][3] * alpha -= B01[0-3][3]
            ymm4 = _mm256_fmsub_pd(ymm4, ymm16, ymm12); //B11[0-3][4] * alpha -= B01[0-3][4]
            ymm5 = _mm256_fmsub_pd(ymm5, ymm16, ymm13); //B11[0-3][5] * alpha -= B01[0-3][5]
            ymm6 = _mm256_fmsub_pd(ymm6, ymm16, ymm14); //B11[0-3][6] * alpha -= B01[0-3][6]
            ymm7 = _mm256_fmsub_pd(ymm7, ymm16, ymm15); //B11[0-3][7] * alpha -= B01[0-3][7]

            ///implement TRSM///

            ///transpose of B11//
            ///unpacklow///
            ymm9 = _mm256_unpacklo_pd(ymm0, ymm1);  //B11[0][0] B11[0][1] B11[2][0] B11[2][1]
            ymm11 = _mm256_unpacklo_pd(ymm2, ymm3); //B11[0][2] B11[0][3] B11[2][2] B11[2][3]

            ymm13 = _mm256_unpacklo_pd(ymm4, ymm5); //B11[0][4] B11[0][5] B11[2][4] B11[2][5]
            ymm15 = _mm256_unpacklo_pd(ymm6, ymm7); //B11[0][6] B11[0][7] B11[2][6] B11[2][7]

            //rearrange low elements
            ymm8 = _mm256_permute2f128_pd(ymm9,ymm11,0x20);     //B11[0][0] B11[0][1] B11[0][2] B11[0][3]
            ymm10 = _mm256_permute2f128_pd(ymm9,ymm11,0x31);    //B11[2][0] B11[2][1] B11[2][2] B11[2][3]

            ymm12 = _mm256_permute2f128_pd(ymm13,ymm15,0x20);   //B11[4][0] B11[4][1] B11[4][2] B11[4][3]
            ymm14 = _mm256_permute2f128_pd(ymm13,ymm15,0x31);   //B11[6][0] B11[6][1] B11[6][2] B11[6][3]

            ////unpackhigh////
            ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);  //B11[1][0] B11[1][1] B11[3][0] B11[3][1]
            ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);  //B11[1][2] B11[1][3] B11[3][2] B11[3][3]

            ymm4 = _mm256_unpackhi_pd(ymm4, ymm5);  //B11[1][4] B11[1][5] B11[3][4] B11[3][5]
            ymm5 = _mm256_unpackhi_pd(ymm6, ymm7);  //B11[1][6] B11[1][7] B11[3][6] B11[3][7]

            //rearrange high elements
            ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);  //B11[1][0] B11[1][1] B11[1][2] B11[1][3]
            ymm11 = _mm256_permute2f128_pd(ymm0,ymm1,0x31); //B11[3][0] B11[3][1] B11[3][2] B11[3][3]

            ymm13 = _mm256_permute2f128_pd(ymm4,ymm5,0x20); //B11[5][0] B11[5][1] B11[5][2] B11[5][3]
            ymm15 = _mm256_permute2f128_pd(ymm4,ymm5,0x31); //B11[7][0] B11[7][1] B11[7][2] B11[7][3]

            ymm0 = _mm256_broadcast_sd((double const *)&ones);

            //extract a00
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack));

            //perform mul operation
            ymm8 = DTRSM_SMALL_DIV_OR_SCALE(ymm8, ymm1);

            //extract a11
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*1));
            ymm3 = _mm256_broadcast_sd((double const *)(a11 + cs_a*2));
            ymm4 = _mm256_broadcast_sd((double const *)(a11 + cs_a*3));
            ymm5 = _mm256_broadcast_sd((double const *)(a11 + cs_a*4));
            ymm6 = _mm256_broadcast_sd((double const *)(a11 + cs_a*5));
            ymm7 = _mm256_broadcast_sd((double const *)(a11 + cs_a*6));
            ymm16 = _mm256_broadcast_sd((double const *)(a11 + cs_a*7));

            a11 += rs_a;

            //(ROw1): FMA operations
            ymm9 = _mm256_fnmadd_pd(ymm2, ymm8, ymm9);
            ymm10 = _mm256_fnmadd_pd(ymm3, ymm8, ymm10);
            ymm11 = _mm256_fnmadd_pd(ymm4, ymm8, ymm11);
            ymm12 = _mm256_fnmadd_pd(ymm5, ymm8, ymm12);
            ymm13 = _mm256_fnmadd_pd(ymm6, ymm8, ymm13);
            ymm14 = _mm256_fnmadd_pd(ymm7, ymm8, ymm14);
            ymm15 = _mm256_fnmadd_pd(ymm16, ymm8, ymm15);

            //perform mul operation
            ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm1);

            ymm3 = _mm256_broadcast_sd((double const *)(a11 + cs_a*2));
            ymm4 = _mm256_broadcast_sd((double const *)(a11 + cs_a*3));
            ymm5 = _mm256_broadcast_sd((double const *)(a11 + cs_a*4));
            ymm6 = _mm256_broadcast_sd((double const *)(a11 + cs_a*5));
            ymm7 = _mm256_broadcast_sd((double const *)(a11 + cs_a*6));
            ymm16 = _mm256_broadcast_sd((double const *)(a11 + cs_a*7));

            a11 += rs_a;

            //extract a22
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            //(ROw2): FMA operations
            ymm10 = _mm256_fnmadd_pd(ymm3, ymm9, ymm10);
            ymm11 = _mm256_fnmadd_pd(ymm4, ymm9, ymm11);
            ymm12 = _mm256_fnmadd_pd(ymm5, ymm9, ymm12);
            ymm13 = _mm256_fnmadd_pd(ymm6, ymm9, ymm13);
            ymm14 = _mm256_fnmadd_pd(ymm7, ymm9, ymm14);
            ymm15 = _mm256_fnmadd_pd(ymm16, ymm9, ymm15);

            //perform mul operation
            ymm10 = DTRSM_SMALL_DIV_OR_SCALE(ymm10, ymm1);

            ymm4 = _mm256_broadcast_sd((double const *)(a11 + cs_a*3));
            ymm5 = _mm256_broadcast_sd((double const *)(a11 + cs_a*4));
            ymm6 = _mm256_broadcast_sd((double const *)(a11 + cs_a*5));
            ymm7 = _mm256_broadcast_sd((double const *)(a11 + cs_a*6));
            ymm16 = _mm256_broadcast_sd((double const *)(a11 + cs_a*7));

            a11 += rs_a;

            //extract a33
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

            //(ROw5): FMA operations
            ymm11 = _mm256_fnmadd_pd(ymm4, ymm10, ymm11);
            ymm12 = _mm256_fnmadd_pd(ymm5, ymm10, ymm12);
            ymm13 = _mm256_fnmadd_pd(ymm6, ymm10, ymm13);
            ymm14 = _mm256_fnmadd_pd(ymm7, ymm10, ymm14);
            ymm15 = _mm256_fnmadd_pd(ymm16, ymm10, ymm15);

            //perform mul operation
            ymm11 = DTRSM_SMALL_DIV_OR_SCALE(ymm11, ymm1);

            ymm0 = _mm256_broadcast_sd((double const *)&ones);

            //extract a44
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 4));

            ymm5 = _mm256_broadcast_sd((double const *)(a11 + cs_a*4));
            ymm6 = _mm256_broadcast_sd((double const *)(a11 + cs_a*5));
            ymm7 = _mm256_broadcast_sd((double const *)(a11 + cs_a*6));
            ymm16 = _mm256_broadcast_sd((double const *)(a11 + cs_a*7));

            a11 += rs_a;

            //(ROw4): FMA operations
            ymm12 = _mm256_fnmadd_pd(ymm5, ymm11, ymm12);
            ymm13 = _mm256_fnmadd_pd(ymm6, ymm11, ymm13);
            ymm14 = _mm256_fnmadd_pd(ymm7, ymm11, ymm14);
            ymm15 = _mm256_fnmadd_pd(ymm16, ymm11, ymm15);

            //perform mul operation
            ymm12 = DTRSM_SMALL_DIV_OR_SCALE(ymm12, ymm1);

            ymm6 = _mm256_broadcast_sd((double const *)(a11 + cs_a*5));
            ymm7 = _mm256_broadcast_sd((double const *)(a11 + cs_a*6));
            ymm16 = _mm256_broadcast_sd((double const *)(a11 + cs_a*7));

            a11 += rs_a;

            //extract a55
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 5));

            //(ROw5): FMA operations
            ymm13 = _mm256_fnmadd_pd(ymm6, ymm12, ymm13);
            ymm14 = _mm256_fnmadd_pd(ymm7, ymm12, ymm14);
            ymm15 = _mm256_fnmadd_pd(ymm16, ymm12, ymm15);

            //perform mul operation
            ymm13 = DTRSM_SMALL_DIV_OR_SCALE(ymm13, ymm1);

            ymm7 = _mm256_broadcast_sd((double const *)(a11 + cs_a*6));
            ymm16 = _mm256_broadcast_sd((double const *)(a11 +cs_a*7));

            a11 += rs_a;

            //extract a66
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 6));

            //(ROw6): FMA operations
            ymm14 = _mm256_fnmadd_pd(ymm7, ymm13, ymm14);
            ymm15 = _mm256_fnmadd_pd(ymm16, ymm13, ymm15);

            //perform mul operation
            ymm14 = DTRSM_SMALL_DIV_OR_SCALE(ymm14, ymm1);

            //extract a77
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 7));

            ymm16 = _mm256_broadcast_sd((double const *)(a11 + cs_a*7));

            a11 += rs_a;
            //(ROw7): FMA operations
            ymm15 = _mm256_fnmadd_pd(ymm16, ymm14, ymm15);

            //perform mul operation
            ymm15 = DTRSM_SMALL_DIV_OR_SCALE(ymm15, ymm1);

            //unpacklow//
            ymm1 = _mm256_unpacklo_pd(ymm8, ymm9);      //B11[0][0] B11[1][0] B11[0][2] B11[1][2]
            ymm3 = _mm256_unpacklo_pd(ymm10, ymm11);    //B11[2][0] B11[3][0] B11[2][2] B11[3][2]

            ymm5 = _mm256_unpacklo_pd(ymm12, ymm13);    //B11[4][0] B11[5][0] B11[4][2] B11[5][2]
            ymm7 = _mm256_unpacklo_pd(ymm14, ymm15);    //B11[6][0] B11[7][0] B11[6][2] B11[7][2]

            //rearrange low elements
            ymm0 = _mm256_permute2f128_pd(ymm1, ymm3, 0x20);    //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
            ymm2 = _mm256_permute2f128_pd(ymm1, ymm3, 0x31);    //B11[0][2] B11[1][2] B11[2][2] B11[3][2]

            ymm4 = _mm256_permute2f128_pd(ymm5, ymm7, 0x20);    //B11[4][0] B11[5][0] B11[6][0] B11[7][0]
            ymm6 = _mm256_permute2f128_pd(ymm5, ymm7, 0x31);    //B11[4][2] B11[5][2] B11[6][2] B11[7][2]

            ///unpack high///
            ymm8 = _mm256_unpackhi_pd(ymm8, ymm9);    //B11[0][1] B11[1][1] B11[0][3] B11[1][3]
            ymm9 = _mm256_unpackhi_pd(ymm10, ymm11);    //B11[2][1] B11[3][1] B11[2][3] B11[3][3]

            ymm12 = _mm256_unpackhi_pd(ymm12, ymm13);    //B11[4][1] B11[5][1] B11[4][3] B11[5][3]
            ymm13 = _mm256_unpackhi_pd(ymm14, ymm15);    //B11[6][1] B11[7][1] B11[6][3] B11[7][3]

            //rearrange high elements
            ymm1 = _mm256_permute2f128_pd(ymm8, ymm9, 0x20);    //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm3 = _mm256_permute2f128_pd(ymm8, ymm9, 0x31);    //B11[0][3] B11[1][3] B11[2][3] B11[3][3]

            ymm5 = _mm256_permute2f128_pd(ymm12, ymm13, 0x20);    //B11[4][1] B11[5][1] B11[6][1] B11[7][1]
            ymm7 = _mm256_permute2f128_pd(ymm12, ymm13, 0x31);    //B11[4][3] B11[5][3] B11[6][3] B11[7][3]

            _mm256_storeu_pd((double *)(b11 + cs_b * 0), ymm0);    //store B11[0][0-3]
            _mm256_storeu_pd((double *)(b11 + cs_b * 1), ymm1);    //store B11[1][0-3]
            _mm256_storeu_pd((double *)(b11 + cs_b * 2), ymm2);    //store B11[2][0-3]
            _mm256_storeu_pd((double *)(b11 + cs_b * 3), ymm3);    //store B11[3][0-3]
            _mm256_storeu_pd((double *)(b11 + cs_b * 0 + 4), ymm4);    //store B11[4][0-3]
            _mm256_storeu_pd((double *)(b11 + cs_b * 1 + 4), ymm5);    //store B11[5][0-3]
            _mm256_storeu_pd((double *)(b11 + cs_b * 2 + 4), ymm6);    //store B11[6][0-3]
            _mm256_storeu_pd((double *)(b11 + cs_b * 3 + 4), ymm7);    //store B11[7][0-3]

            n_rem -=4;
            j +=4;
        }

        if(n_rem)
        {
            a10 = D_A_pack;
            a11 = L + (i*rs_a) + (i*cs_a);     //pointer to block of A to be used for TRSM
            b01 = B + j*cs_b;           //pointer to block of B to be used for GEMM
            b11 = B + i + j* cs_b;      //pointer to block of B to be used for TRSM

            k_iter = i;      //number of times GEMM to be performed(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            if(3 == n_rem)
            {
                ///GEMM code begins///
                BLIS_DTRSM_SMALL_GEMM_8mx3n(a10,b01,cs_b,p_lda,k_iter)

                ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     //register to hold alpha

                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));    //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
                ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *1));    //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
                ymm2 = _mm256_loadu_pd((double const *)(b11 + cs_b *2));    //B11[0][2] B11[1][2] B11[2][2] B11[3][2]

                ymm4 = _mm256_loadu_pd((double const *)(b11 + cs_b *0 + 4));    //B11[0][4] B11[1][4] B11[2][4] B11[3][4]
                ymm5 = _mm256_loadu_pd((double const *)(b11 + cs_b *1 + 4));    //B11[0][5] B11[1][5] B11[2][5] B11[3][5]
                ymm6 = _mm256_loadu_pd((double const *)(b11 + cs_b *2 + 4));    //B11[0][6] B11[1][6] B11[2][6] B11[3][6]

                ymm0 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);  //B11[0-3][0] * alpha -= B01[0-3][0]
                ymm1 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);  //B11[0-3][1] * alpha -= B01[0-3][1]
                ymm2 = _mm256_fmsub_pd(ymm2, ymm16, ymm10); //B11[0-3][2] * alpha -= B01[0-3][2]
                ymm3 = _mm256_broadcast_sd((double const *)(&ones));

                ymm4 = _mm256_fmsub_pd(ymm4, ymm16, ymm12); //B11[0-3][4] * alpha -= B01[0-3][4]
                ymm5 = _mm256_fmsub_pd(ymm5, ymm16, ymm13); //B11[0-3][5] * alpha -= B01[0-3][5]
                ymm6 = _mm256_fmsub_pd(ymm6, ymm16, ymm14); //B11[0-3][6] * alpha -= B01[0-3][6]
                ymm7 = _mm256_broadcast_sd((double const *)(&ones));
            }
            else if(2 == n_rem)
            {
                ///GEMM code begins///
                BLIS_DTRSM_SMALL_GEMM_8mx2n(a10,b01,cs_b,p_lda,k_iter)

                ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     //register to hold alpha

                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));    //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
                ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *1));    //B11[0][1] B11[1][1] B11[2][1] B11[3][1]

                ymm4 = _mm256_loadu_pd((double const *)(b11 + cs_b *0 + 4));    //B11[0][4] B11[1][4] B11[2][4] B11[3][4]
                ymm5 = _mm256_loadu_pd((double const *)(b11 + cs_b *1 + 4));    //B11[0][5] B11[1][5] B11[2][5] B11[3][5]

                ymm0 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);  //B11[0-3][0] * alpha -= B01[0-3][0]
                ymm1 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);  //B11[0-3][1] * alpha -= B01[0-3][1]
                ymm2 = _mm256_broadcast_sd((double const *)(&ones));
                ymm3 = _mm256_broadcast_sd((double const *)(&ones));

                ymm4 = _mm256_fmsub_pd(ymm4, ymm16, ymm12); //B11[0-3][4] * alpha -= B01[0-3][4]
                ymm5 = _mm256_fmsub_pd(ymm5, ymm16, ymm13); //B11[0-3][5] * alpha -= B01[0-3][5]
                ymm6 = _mm256_broadcast_sd((double const *)(&ones));
                ymm7 = _mm256_broadcast_sd((double const *)(&ones));
            }
            else if(1 == n_rem)
            {
                ///GEMM code begins///
                BLIS_DTRSM_SMALL_GEMM_8mx1n(a10,b01,cs_b,p_lda,k_iter)

                ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     //register to hold alpha

                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));    //B11[0][0] B11[1][0] B11[2][0] B11[3][0]

                ymm4 = _mm256_loadu_pd((double const *)(b11 + cs_b *0 + 4));    //B11[0][4] B11[1][4] B11[2][4] B11[3][4]

                ymm0 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);  //B11[0-3][0] * alpha -= B01[0-3][0]
                ymm1 = _mm256_broadcast_sd((double const *)(&ones));
                ymm2 = _mm256_broadcast_sd((double const *)(&ones));
                ymm3 = _mm256_broadcast_sd((double const *)(&ones));

                ymm4 = _mm256_fmsub_pd(ymm4, ymm16, ymm12); //B11[0-3][4] * alpha -= B01[0-3][4]
                ymm5 = _mm256_broadcast_sd((double const *)(&ones));
                ymm6 = _mm256_broadcast_sd((double const *)(&ones));
                ymm7 = _mm256_broadcast_sd((double const *)(&ones));
            }
            ///implement TRSM///

            ///transpose of B11//
            ///unpacklow///
            ymm9 = _mm256_unpacklo_pd(ymm0, ymm1);  //B11[0][0] B11[0][1] B11[2][0] B11[2][1]
            ymm11 = _mm256_unpacklo_pd(ymm2, ymm3); //B11[0][2] B11[0][3] B11[2][2] B11[2][3]

            ymm13 = _mm256_unpacklo_pd(ymm4, ymm5); //B11[0][4] B11[0][5] B11[2][4] B11[2][5]
            ymm15 = _mm256_unpacklo_pd(ymm6, ymm7); //B11[0][6] B11[0][7] B11[2][6] B11[2][7]

            //rearrange low elements
            ymm8 = _mm256_permute2f128_pd(ymm9,ymm11,0x20);     //B11[0][0] B11[0][1] B11[0][2] B11[0][3]
            ymm10 = _mm256_permute2f128_pd(ymm9,ymm11,0x31);    //B11[2][0] B11[2][1] B11[2][2] B11[2][3]

            ymm12 = _mm256_permute2f128_pd(ymm13,ymm15,0x20);   //B11[4][0] B11[4][1] B11[4][2] B11[4][3]
            ymm14 = _mm256_permute2f128_pd(ymm13,ymm15,0x31);   //B11[6][0] B11[6][1] B11[6][2] B11[6][3]

            ////unpackhigh////
            ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);  //B11[1][0] B11[1][1] B11[3][0] B11[3][1]
            ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);  //B11[1][2] B11[1][3] B11[3][2] B11[3][3]

            ymm4 = _mm256_unpackhi_pd(ymm4, ymm5);  //B11[1][4] B11[1][5] B11[3][4] B11[3][5]
            ymm5 = _mm256_unpackhi_pd(ymm6, ymm7);  //B11[1][6] B11[1][7] B11[3][6] B11[3][7]

            //rearrange high elements
            ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);  //B11[1][0] B11[1][1] B11[1][2] B11[1][3]
            ymm11 = _mm256_permute2f128_pd(ymm0,ymm1,0x31); //B11[3][0] B11[3][1] B11[3][2] B11[3][3]

            ymm13 = _mm256_permute2f128_pd(ymm4,ymm5,0x20); //B11[5][0] B11[5][1] B11[5][2] B11[5][3]
            ymm15 = _mm256_permute2f128_pd(ymm4,ymm5,0x31); //B11[7][0] B11[7][1] B11[7][2] B11[7][3]

            ymm0 = _mm256_broadcast_sd((double const *)&ones);

            //extract a00
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack));

            //perform mul operation
            ymm8 = DTRSM_SMALL_DIV_OR_SCALE(ymm8, ymm1);

            //extract a11
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*1));
            ymm3 = _mm256_broadcast_sd((double const *)(a11 + cs_a*2));
            ymm4 = _mm256_broadcast_sd((double const *)(a11 + cs_a*3));
            ymm5 = _mm256_broadcast_sd((double const *)(a11 + cs_a*4));
            ymm6 = _mm256_broadcast_sd((double const *)(a11 + cs_a*5));
            ymm7 = _mm256_broadcast_sd((double const *)(a11 + cs_a*6));
            ymm16 = _mm256_broadcast_sd((double const *)(a11 + cs_a*7));

            a11 += rs_a;

            //(ROw1): FMA operations
            ymm9 = _mm256_fnmadd_pd(ymm2, ymm8, ymm9);
            ymm10 = _mm256_fnmadd_pd(ymm3, ymm8, ymm10);
            ymm11 = _mm256_fnmadd_pd(ymm4, ymm8, ymm11);
            ymm12 = _mm256_fnmadd_pd(ymm5, ymm8, ymm12);
            ymm13 = _mm256_fnmadd_pd(ymm6, ymm8, ymm13);
            ymm14 = _mm256_fnmadd_pd(ymm7, ymm8, ymm14);
            ymm15 = _mm256_fnmadd_pd(ymm16, ymm8, ymm15);

            //perform mul operation
            ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm1);

            ymm3 = _mm256_broadcast_sd((double const *)(a11 + cs_a*2));
            ymm4 = _mm256_broadcast_sd((double const *)(a11 + cs_a*3));
            ymm5 = _mm256_broadcast_sd((double const *)(a11 + cs_a*4));
            ymm6 = _mm256_broadcast_sd((double const *)(a11 + cs_a*5));
            ymm7 = _mm256_broadcast_sd((double const *)(a11 + cs_a*6));
            ymm16 = _mm256_broadcast_sd((double const *)(a11 + cs_a*7));

            a11 += rs_a;

            //extract a22
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            //(ROw2): FMA operations
            ymm10 = _mm256_fnmadd_pd(ymm3, ymm9, ymm10);
            ymm11 = _mm256_fnmadd_pd(ymm4, ymm9, ymm11);
            ymm12 = _mm256_fnmadd_pd(ymm5, ymm9, ymm12);
            ymm13 = _mm256_fnmadd_pd(ymm6, ymm9, ymm13);
            ymm14 = _mm256_fnmadd_pd(ymm7, ymm9, ymm14);
            ymm15 = _mm256_fnmadd_pd(ymm16, ymm9, ymm15);

            //perform mul operation
            ymm10 = DTRSM_SMALL_DIV_OR_SCALE(ymm10, ymm1);

            ymm4 = _mm256_broadcast_sd((double const *)(a11 + cs_a*3));
            ymm5 = _mm256_broadcast_sd((double const *)(a11 + cs_a*4));
            ymm6 = _mm256_broadcast_sd((double const *)(a11 + cs_a*5));
            ymm7 = _mm256_broadcast_sd((double const *)(a11 + cs_a*6));
            ymm16 = _mm256_broadcast_sd((double const *)(a11 + cs_a*7));

            a11 += rs_a;

            //extract a33
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

            //(ROw5): FMA operations
            ymm11 = _mm256_fnmadd_pd(ymm4, ymm10, ymm11);
            ymm12 = _mm256_fnmadd_pd(ymm5, ymm10, ymm12);
            ymm13 = _mm256_fnmadd_pd(ymm6, ymm10, ymm13);
            ymm14 = _mm256_fnmadd_pd(ymm7, ymm10, ymm14);
            ymm15 = _mm256_fnmadd_pd(ymm16, ymm10, ymm15);

            //perform mul operation
            ymm11 = DTRSM_SMALL_DIV_OR_SCALE(ymm11, ymm1);

            ymm0 = _mm256_broadcast_sd((double const *)&ones);

            //extract a44
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 4));

            ymm5 = _mm256_broadcast_sd((double const *)(a11 + cs_a*4));
            ymm6 = _mm256_broadcast_sd((double const *)(a11 + cs_a*5));
            ymm7 = _mm256_broadcast_sd((double const *)(a11 + cs_a*6));
            ymm16 = _mm256_broadcast_sd((double const *)(a11 + cs_a*7));

            a11 += rs_a;

            //(ROw4): FMA operations
            ymm12 = _mm256_fnmadd_pd(ymm5, ymm11, ymm12);
            ymm13 = _mm256_fnmadd_pd(ymm6, ymm11, ymm13);
            ymm14 = _mm256_fnmadd_pd(ymm7, ymm11, ymm14);
            ymm15 = _mm256_fnmadd_pd(ymm16, ymm11, ymm15);

            //perform mul operation
            ymm12 = DTRSM_SMALL_DIV_OR_SCALE(ymm12, ymm1);

            ymm6 = _mm256_broadcast_sd((double const *)(a11 + cs_a*5));
            ymm7 = _mm256_broadcast_sd((double const *)(a11 + cs_a*6));
            ymm16 = _mm256_broadcast_sd((double const *)(a11 + cs_a*7));

            a11 += rs_a;

            //extract a55
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 5));


            //(ROw5): FMA operations
            ymm13 = _mm256_fnmadd_pd(ymm6, ymm12, ymm13);
            ymm14 = _mm256_fnmadd_pd(ymm7, ymm12, ymm14);
            ymm15 = _mm256_fnmadd_pd(ymm16, ymm12, ymm15);

            //perform mul operation
            ymm13 = DTRSM_SMALL_DIV_OR_SCALE(ymm13, ymm1);

            ymm7 = _mm256_broadcast_sd((double const *)(a11 + cs_a*6));
            ymm16 = _mm256_broadcast_sd((double const *)(a11 +cs_a*7));

            a11 += rs_a;

            //extract a66
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 6));


            //(ROw6): FMA operations
            ymm14 = _mm256_fnmadd_pd(ymm7, ymm13, ymm14);
            ymm15 = _mm256_fnmadd_pd(ymm16, ymm13, ymm15);

            //perform mul operation
            ymm14 = DTRSM_SMALL_DIV_OR_SCALE(ymm14, ymm1);

            //extract a77
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 7));

            ymm16 = _mm256_broadcast_sd((double const *)(a11 + cs_a*7));

            a11 += rs_a;
            //(ROw7): FMA operations
            ymm15 = _mm256_fnmadd_pd(ymm16, ymm14, ymm15);

            //perform mul operation
            ymm15 = DTRSM_SMALL_DIV_OR_SCALE(ymm15, ymm1);

            //unpacklow//
            ymm1 = _mm256_unpacklo_pd(ymm8, ymm9);      //B11[0][0] B11[1][0] B11[0][2] B11[1][2]
            ymm3 = _mm256_unpacklo_pd(ymm10, ymm11);    //B11[2][0] B11[3][0] B11[2][2] B11[3][2]

            ymm5 = _mm256_unpacklo_pd(ymm12, ymm13);    //B11[4][0] B11[5][0] B11[4][2] B11[5][2]
            ymm7 = _mm256_unpacklo_pd(ymm14, ymm15);    //B11[6][0] B11[7][0] B11[6][2] B11[7][2]

            //rearrange low elements
            ymm0 = _mm256_permute2f128_pd(ymm1, ymm3, 0x20);    //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
            ymm2 = _mm256_permute2f128_pd(ymm1, ymm3, 0x31);    //B11[0][2] B11[1][2] B11[2][2] B11[3][2]

            ymm4 = _mm256_permute2f128_pd(ymm5, ymm7, 0x20);    //B11[4][0] B11[5][0] B11[6][0] B11[7][0]
            ymm6 = _mm256_permute2f128_pd(ymm5, ymm7, 0x31);    //B11[4][2] B11[5][2] B11[6][2] B11[7][2]

            ///unpack high///
            ymm8 = _mm256_unpackhi_pd(ymm8, ymm9);    //B11[0][1] B11[1][1] B11[0][3] B11[1][3]
            ymm9 = _mm256_unpackhi_pd(ymm10, ymm11);    //B11[2][1] B11[3][1] B11[2][3] B11[3][3]

            ymm12 = _mm256_unpackhi_pd(ymm12, ymm13);    //B11[4][1] B11[5][1] B11[4][3] B11[5][3]
            ymm13 = _mm256_unpackhi_pd(ymm14, ymm15);    //B11[6][1] B11[7][1] B11[6][3] B11[7][3]

            //rearrange high elements
            ymm1 = _mm256_permute2f128_pd(ymm8, ymm9, 0x20);    //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm3 = _mm256_permute2f128_pd(ymm8, ymm9, 0x31);    //B11[0][3] B11[1][3] B11[2][3] B11[3][3]

            ymm5 = _mm256_permute2f128_pd(ymm12, ymm13, 0x20);    //B11[4][1] B11[5][1] B11[6][1] B11[7][1]
            ymm7 = _mm256_permute2f128_pd(ymm12, ymm13, 0x31);    //B11[4][3] B11[5][3] B11[6][3] B11[7][3]

            if(3 == n_rem)
            {
                _mm256_storeu_pd((double *)(b11 + cs_b * 0), ymm0);    //store B11[0][0-3]
                _mm256_storeu_pd((double *)(b11 + cs_b * 1), ymm1);    //store B11[1][0-3]
                _mm256_storeu_pd((double *)(b11 + cs_b * 2), ymm2);    //store B11[2][0-3]
                _mm256_storeu_pd((double *)(b11 + cs_b * 0 + 4), ymm4);    //store B11[4][0-3]
                _mm256_storeu_pd((double *)(b11 + cs_b * 1 + 4), ymm5);    //store B11[5][0-3]
                _mm256_storeu_pd((double *)(b11 + cs_b * 2 + 4), ymm6);    //store B11[6][0-3]
            }
            else if(2 == n_rem)
            {
                _mm256_storeu_pd((double *)(b11 + cs_b * 0), ymm0);    //store B11[0][0-3]
                _mm256_storeu_pd((double *)(b11 + cs_b * 1), ymm1);    //store B11[1][0-3]
                _mm256_storeu_pd((double *)(b11 + cs_b * 0 + 4), ymm4);    //store B11[4][0-3]
                _mm256_storeu_pd((double *)(b11 + cs_b * 1 + 4), ymm5);    //store B11[5][0-3]
            }
            else if(1 == n_rem)
            {
                _mm256_storeu_pd((double *)(b11 + cs_b * 0), ymm0);    //store B11[0][0-3]
                _mm256_storeu_pd((double *)(b11 + cs_b * 0 + 4), ymm4);    //store B11[4][0-3]
            }
        }
    }

    //======================M remainder cases================================
    dim_t m_rem = m-i;
    if(m_rem>=4)                      //implementation for reamainder rows(when 'M' is not a multiple of d_mr)
    {
        a10 = L + (i*cs_a);             //pointer to block of A to be used for GEMM
        a11 = L + (i*rs_a) + (i*cs_a);
        double *ptr_a10_dup = D_A_pack;
        dim_t p_lda = 4; // packed leading dimension

        if(transa)
        {
            for(dim_t x =0;x < i;x+=p_lda)
            {
                ymm0 = _mm256_loadu_pd((double const *)(a10));
                ymm1 = _mm256_loadu_pd((double const *)(a10 + cs_a));
                ymm2 = _mm256_loadu_pd((double const *)(a10 + cs_a * 2));
                ymm3 = _mm256_loadu_pd((double const *)(a10 + cs_a * 3));

                ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
                ymm5 = _mm256_unpacklo_pd(ymm2, ymm3);

                ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);

                ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
                ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);

                ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

                _mm256_storeu_pd((double *)(ptr_a10_dup), ymm6);
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda), ymm7);
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*2), ymm8);
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*3), ymm9);

                a10 += p_lda;
                ptr_a10_dup += p_lda*p_lda;
            }
        }
        else
        {
            for(dim_t x =0;x < i;x++)
            {
                ymm0 = _mm256_loadu_pd((double const *)(a10 + rs_a * x));
                _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda * x), ymm0);
            }
        }

        ymm4 = _mm256_broadcast_sd((double const *)&ones);
        if(!is_unitdiag)
        {
            if(transa)
            {
                //broadcast diagonal elements of A11
                ymm0 = _mm256_broadcast_sd((double const *)(a11));
                ymm1 = _mm256_broadcast_sd((double const *)(a11+cs_a*1 + 1));
                ymm2 = _mm256_broadcast_sd((double const *)(a11+cs_a*2 + 2));
                ymm3 = _mm256_broadcast_sd((double const *)(a11+cs_a*3 + 3));
            }
            else
            {
                //broadcast diagonal elements of A11
                ymm0 = _mm256_broadcast_sd((double const *)(a11));
                ymm1 = _mm256_broadcast_sd((double const *)(a11+rs_a*1 + 1));
                ymm2 = _mm256_broadcast_sd((double const *)(a11+rs_a*2 + 2));
                ymm3 = _mm256_broadcast_sd((double const *)(a11+rs_a*3 + 3));
            }

            ymm0 = _mm256_unpacklo_pd(ymm0, ymm1);
            ymm1 = _mm256_unpacklo_pd(ymm2, ymm3);
            ymm1 = _mm256_blend_pd(ymm0, ymm1, 0x0C);
            #ifdef BLIS_DISABLE_TRSM_PREINVERSION
            ymm4 = ymm1;
            #endif
            #ifdef BLIS_ENABLE_TRSM_PREINVERSION
            ymm4 = _mm256_div_pd(ymm4, ymm1);
            #endif
        }
        _mm256_storeu_pd((double *)(d11_pack), ymm4);

        for(j = 0; (j+d_nr-1) < n; j += d_nr)   //loop along 'N' dimension
        {
            a10 = D_A_pack;            //pointer to block of A to be used for GEMM
            a11 = L + (i*rs_a) + (i*cs_a);        //pointer to block of A to be used for TRSM
            b01 = B + (j*cs_b);        //pointer to block of B to be used for GEMM
            b11 = B + i + (j* cs_b);        //pointer to block of B to be used for TRSM

            k_iter = i;            //number of times GEMM operation to be done(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            ///GEMM code begins///
            BLIS_DTRSM_SMALL_GEMM_4mx6n(a10,b01,cs_b,p_lda,k_iter)

            ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     //register to hold alpha

            ///implement TRSM///
            ymm0 = _mm256_loadu_pd((double const *)(b11));    //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
            ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *1));    //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm2 = _mm256_loadu_pd((double const *)(b11 + cs_b *2));    //B11[0][2] B11[1][2] B11[2][2] B11[3][2]
            ymm3 = _mm256_loadu_pd((double const *)(b11 + cs_b *3));    //B11[0][3] B11[1][3] B11[2][3] B11[3][3]
            ymm0 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);  //B11[0-3][0] * alpha -= B01[0-3][0]
            ymm1 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);  //B11[0-3][1] * alpha -= B01[0-3][1]
            ymm2 = _mm256_fmsub_pd(ymm2, ymm16, ymm10); //B11[0-3][2] * alpha -= B01[0-3][2]
            ymm3 = _mm256_fmsub_pd(ymm3, ymm16, ymm11); //B11[0-3][3] * alpha -= B01[0-3][3]

            ///transpose of B11//
            ///unpacklow///
            ymm9 = _mm256_unpacklo_pd(ymm0, ymm1);  //B11[0][0] B11[0][1] B11[2][0] B11[2][1]
            ymm11 = _mm256_unpacklo_pd(ymm2, ymm3); //B11[0][2] B11[0][3] B11[2][2] B11[2][3]

            //rearrange low elements
            ymm8 = _mm256_permute2f128_pd(ymm9,ymm11,0x20);     //B11[0][0] B11[0][1] B11[0][2] B11[0][3]
            ymm10 = _mm256_permute2f128_pd(ymm9,ymm11,0x31);    //B11[2][0] B11[2][1] B11[2][2] B11[2][3]

            ////unpackhigh////
            ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);  //B11[1][0] B11[1][1] B11[3][0] B11[3][1]
            ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);  //B11[1][2] B11[1][3] B11[3][2] B11[3][3]

            //rearrange high elements
            ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);  //B11[1][0] B11[1][1] B11[1][2] B11[1][3]
            ymm11 = _mm256_permute2f128_pd(ymm0,ymm1,0x31); //B11[3][0] B11[3][1] B11[3][2] B11[3][3]


            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *4));
            ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *5));
            ymm0 = _mm256_fmsub_pd(ymm0, ymm16, ymm4);
            ymm1 = _mm256_fmsub_pd(ymm1, ymm16, ymm5);


            ymm16 = _mm256_broadcast_sd((double const *)(&ones));

            ////unpacklow////
            ymm7 = _mm256_unpacklo_pd(ymm0, ymm1);        //B11[0][0] B11[0][1] B11[2][0] B11[2][1]
            //ymm16;

            //rearrange low elements
            ymm4 = _mm256_permute2f128_pd(ymm7,ymm16,0x20);    //B11[0][0] B11[0][1] B11[0][2] B11[0][3]
            ymm6 = _mm256_permute2f128_pd(ymm7,ymm16,0x31);//B11[2][0] B11[2][1] B11[2][2] B11[2][3]

            ////unpackhigh////
            ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);    //B11[1][0] B11[1][1] B11[3][0] B11[3][1]
            //ymm16;

            //rearrange high elements
            ymm5 = _mm256_permute2f128_pd(ymm0,ymm16,0x20);    //B11[1][0] B11[1][1] B11[1][2] B11[1][3]
            ymm7 = _mm256_permute2f128_pd(ymm0,ymm16,0x31);    //B11[3][0] B11[3][1] B11[3][2] B11[3][3]
            //b11 transpose end

            ////extract a00
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack));

            //perform mul operation
            ymm8 = DTRSM_SMALL_DIV_OR_SCALE(ymm8, ymm1);
            ymm4 = DTRSM_SMALL_DIV_OR_SCALE(ymm4, ymm1);

            //extract a11
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            //(ROw1): FMA operations
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*1));
            ymm9 = _mm256_fnmadd_pd(ymm2, ymm8, ymm9);
            ymm5 = _mm256_fnmadd_pd(ymm2, ymm4, ymm5);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*2));
            ymm10 = _mm256_fnmadd_pd(ymm2, ymm8, ymm10);
            ymm6 = _mm256_fnmadd_pd(ymm2, ymm4, ymm6);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*3));
            ymm11 = _mm256_fnmadd_pd(ymm2, ymm8, ymm11);
            ymm7 = _mm256_fnmadd_pd(ymm2, ymm4, ymm7);


            //perform mul operation
            ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm1);
            ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm1);

            a11 += rs_a;

            //extract a22
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            //(ROw2): FMA operations
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*2));
            ymm10 = _mm256_fnmadd_pd(ymm2, ymm9, ymm10);
            ymm6 = _mm256_fnmadd_pd(ymm2, ymm5, ymm6);
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*3));
            ymm11 = _mm256_fnmadd_pd(ymm2, ymm9, ymm11);
            ymm7 = _mm256_fnmadd_pd(ymm2, ymm5, ymm7);

            //perform mul operation
            ymm10 = DTRSM_SMALL_DIV_OR_SCALE(ymm10, ymm1);
            ymm6 = DTRSM_SMALL_DIV_OR_SCALE(ymm6, ymm1);

            a11 += rs_a;

            //extract a33
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

            //(ROw5): FMA operations
            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*3));
            ymm11 = _mm256_fnmadd_pd(ymm2, ymm10, ymm11);
            ymm7 = _mm256_fnmadd_pd(ymm2, ymm6, ymm7);

            //perform mul operation
            ymm11 = DTRSM_SMALL_DIV_OR_SCALE(ymm11, ymm1);
            ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm1);

            a11 += rs_a;

            //unpacklow//
            ymm1 = _mm256_unpacklo_pd(ymm8, ymm9);      //B11[0][0] B11[1][0] B11[0][2] B11[1][2]
            ymm3 = _mm256_unpacklo_pd(ymm10, ymm11);    //B11[2][0] B11[3][0] B11[2][2] B11[3][2]

            //rearrange low elements
            ymm0 = _mm256_permute2f128_pd(ymm1, ymm3, 0x20);    //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
            ymm2 = _mm256_permute2f128_pd(ymm1, ymm3, 0x31);    //B11[0][2] B11[1][2] B11[2][2] B11[3][2]

            ///unpack high///
            ymm8 = _mm256_unpackhi_pd(ymm8, ymm9);    //B11[0][1] B11[1][1] B11[0][3] B11[1][3]
            ymm9 = _mm256_unpackhi_pd(ymm10, ymm11);    //B11[2][1] B11[3][1] B11[2][3] B11[3][3]

            //rearrange high elements
            ymm1 = _mm256_permute2f128_pd(ymm8, ymm9, 0x20);    //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm3 = _mm256_permute2f128_pd(ymm8, ymm9, 0x31);    //B11[0][3] B11[1][3] B11[2][3] B11[3][3]

            _mm256_storeu_pd((double *)(b11 + cs_b * 0), ymm0);    //store B11[0][0-3]
            _mm256_storeu_pd((double *)(b11 + cs_b * 1), ymm1);    //store B11[1][0-3]
            _mm256_storeu_pd((double *)(b11 + cs_b * 2), ymm2);    //store B11[2][0-3]
            _mm256_storeu_pd((double *)(b11 + cs_b * 3), ymm3);    //store B11[3][0-3]

            //unpacklow//
            ymm1 = _mm256_unpacklo_pd(ymm4, ymm5);      //B11[0][0] B11[1][0] B11[0][2] B11[1][2]
            ymm3 = _mm256_unpacklo_pd(ymm6, ymm7);    //B11[2][0] B11[3][0] B11[2][2] B11[3][2]

            //rearrange low elements
            ymm0 = _mm256_permute2f128_pd(ymm1, ymm3, 0x20);    //B11[0][0] B11[1][0] B11[2][0] B11[3][0]

            ///unpack high///
            ymm4 = _mm256_unpackhi_pd(ymm4, ymm5);    //B11[0][1] B11[1][1] B11[0][3] B11[1][3]
            ymm5 = _mm256_unpackhi_pd(ymm6, ymm7);    //B11[2][1] B11[3][1] B11[2][3] B11[3][3]

            //rearrange high elements
            ymm1 = _mm256_permute2f128_pd(ymm4, ymm5, 0x20);    //B11[0][1] B11[1][1] B11[2][1] B11[3][1]

            _mm256_storeu_pd((double *)(b11 + cs_b * 4), ymm0);    //store B11[0][0-3]
            _mm256_storeu_pd((double *)(b11 + cs_b * 5), ymm1);    //store B11[1][0-3]
        }

        dim_t n_rem = n-j;
        if(n_rem >= 4)
        {
            a10 = D_A_pack;
            a11 = L + (i*rs_a) + (i*cs_a);     //pointer to block of A to be used for TRSM
            b01 = B + j*cs_b;           //pointer to block of B to be used for GEMM
            b11 = B + i + j* cs_b;      //pointer to block of B to be used for TRSM

            k_iter = i;      //number of times GEMM to be performed(in blocks of 4x4)

            /*Fill zeros into ymm registers used in gemm accumulations */
            BLIS_SET_YMM_REG_ZEROS

            BLIS_DTRSM_SMALL_GEMM_4mx4n(a10,b01,cs_b,p_lda,k_iter)

            ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     //register to hold alpha

            ///implement TRSM///

            ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));
            ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *1));
            ymm2 = _mm256_loadu_pd((double const *)(b11 + cs_b *2));
            ymm3 = _mm256_loadu_pd((double const *)(b11 + cs_b *3));
            ymm0 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);
            ymm1 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);
            ymm2 = _mm256_fmsub_pd(ymm2, ymm16, ymm10);
            ymm3 = _mm256_fmsub_pd(ymm3, ymm16, ymm11);

            ///transpose of B11//
            ///unpacklow///
            ymm9 = _mm256_unpacklo_pd(ymm0, ymm1);  //B11[0][0] B11[0][1] B11[2][0] B11[2][1]
            ymm11 = _mm256_unpacklo_pd(ymm2, ymm3); //B11[0][2] B11[0][3] B11[2][2] B11[2][3]

            //rearrange low elements
            ymm8 = _mm256_permute2f128_pd(ymm9,ymm11,0x20);     //B11[0][0] B11[0][1] B11[0][2] B11[0][3]
            ymm10 = _mm256_permute2f128_pd(ymm9,ymm11,0x31);    //B11[2][0] B11[2][1] B11[2][2] B11[2][3]

            ////unpackhigh////
            ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);  //B11[1][0] B11[1][1] B11[3][0] B11[3][1]
            ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);  //B11[1][2] B11[1][3] B11[3][2] B11[3][3]

            //rearrange high elements
            ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);  //B11[1][0] B11[1][1] B11[1][2] B11[1][3]
            ymm11 = _mm256_permute2f128_pd(ymm0,ymm1,0x31); //B11[3][0] B11[3][1] B11[3][2] B11[3][3]

            ymm0 = _mm256_broadcast_sd((double const *)&ones);

            //extract a00
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack));

            //perform mul operation
            ymm8 = DTRSM_SMALL_DIV_OR_SCALE(ymm8, ymm1);

            //extract a11
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*1));
            ymm3 = _mm256_broadcast_sd((double const *)(a11 + cs_a*2));
            ymm4 = _mm256_broadcast_sd((double const *)(a11 + cs_a*3));

            a11 += rs_a;

            //(ROw1): FMA operations
            ymm9 = _mm256_fnmadd_pd(ymm2, ymm8, ymm9);
            ymm10 = _mm256_fnmadd_pd(ymm3, ymm8, ymm10);
            ymm11 = _mm256_fnmadd_pd(ymm4, ymm8, ymm11);

            //perform mul operation
            ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm1);

            ymm3 = _mm256_broadcast_sd((double const *)(a11 + cs_a*2));
            ymm4 = _mm256_broadcast_sd((double const *)(a11 + cs_a*3));

            a11 += rs_a;

            //extract a22
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            //(ROw2): FMA operations
            ymm10 = _mm256_fnmadd_pd(ymm3, ymm9, ymm10);
            ymm11 = _mm256_fnmadd_pd(ymm4, ymm9, ymm11);

            //perform mul operation
            ymm10 = DTRSM_SMALL_DIV_OR_SCALE(ymm10, ymm1);

            ymm4 = _mm256_broadcast_sd((double const *)(a11 + cs_a*3));

            a11 += rs_a;

            //extract a33
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

            //(ROw5): FMA operations
            ymm11 = _mm256_fnmadd_pd(ymm4, ymm10, ymm11);

            //perform mul operation
            ymm11 = DTRSM_SMALL_DIV_OR_SCALE(ymm11, ymm1);

            //unpacklow//
            ymm1 = _mm256_unpacklo_pd(ymm8, ymm9);      //B11[0][0] B11[1][0] B11[0][2] B11[1][2]
            ymm3 = _mm256_unpacklo_pd(ymm10, ymm11);    //B11[2][0] B11[3][0] B11[2][2] B11[3][2]

            //rearrange low elements
            ymm0 = _mm256_permute2f128_pd(ymm1, ymm3, 0x20);    //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
            ymm2 = _mm256_permute2f128_pd(ymm1, ymm3, 0x31);    //B11[0][2] B11[1][2] B11[2][2] B11[3][2]

            ///unpack high///
            ymm8 = _mm256_unpackhi_pd(ymm8, ymm9);    //B11[0][1] B11[1][1] B11[0][3] B11[1][3]
            ymm9 = _mm256_unpackhi_pd(ymm10, ymm11);    //B11[2][1] B11[3][1] B11[2][3] B11[3][3]

            //rearrange high elements
            ymm1 = _mm256_permute2f128_pd(ymm8, ymm9, 0x20);    //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm3 = _mm256_permute2f128_pd(ymm8, ymm9, 0x31);    //B11[0][3] B11[1][3] B11[2][3] B11[3][3]

            _mm256_storeu_pd((double *)(b11 + cs_b * 0), ymm0);    //store B11[0][0-3]
            _mm256_storeu_pd((double *)(b11 + cs_b * 1), ymm1);    //store B11[1][0-3]
            _mm256_storeu_pd((double *)(b11 + cs_b * 2), ymm2);    //store B11[2][0-3]
            _mm256_storeu_pd((double *)(b11 + cs_b * 3), ymm3);    //store B11[3][0-3]

            n_rem -= 4;
            j += 4;
        }
        if(n_rem)
        {
            a10 = D_A_pack;
            a11 = L + (i*rs_a) + (i*cs_a);     //pointer to block of A to be used for TRSM
            b01 = B + j*cs_b;           //pointer to block of B to be used for GEMM
            b11 = B + i + j* cs_b;      //pointer to block of B to be used for TRSM

            k_iter = i;             //number of times GEMM to be performed(in blocks of 4x4)

            ymm8 = _mm256_setzero_pd();
            ymm9 = _mm256_setzero_pd();
            ymm10 = _mm256_setzero_pd();

            if(3 == n_rem)
            {
                ///GEMM code begins///
                BLIS_DTRSM_SMALL_GEMM_4mx3n(a10,b01,cs_b,p_lda,k_iter)

                ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     //register to hold alpha

                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));    //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
                ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *1));    //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
                ymm2 = _mm256_loadu_pd((double const *)(b11 + cs_b *2));    //B11[0][2] B11[1][2] B11[2][2] B11[3][2]

                ymm0 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);  //B11[0-3][0] * alpha -= B01[0-3][0]
                ymm1 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);  //B11[0-3][1] * alpha -= B01[0-3][1]
                ymm2 = _mm256_fmsub_pd(ymm2, ymm16, ymm10); //B11[0-3][2] * alpha -= B01[0-3][2]
                ymm3 = _mm256_broadcast_sd((double const *)(&ones));
            }
            else if(2 == n_rem)
            {
                ///GEMM code begins///
                BLIS_DTRSM_SMALL_GEMM_4mx2n(a10,b01,cs_b,p_lda,k_iter)

                ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     //register to hold alpha

                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));    //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
                ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *1));    //B11[0][1] B11[1][1] B11[2][1] B11[3][1]

                ymm0 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);  //B11[0-3][0] * alpha -= B01[0-3][0]
                ymm1 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);  //B11[0-3][1] * alpha -= B01[0-3][1]
                ymm2 = _mm256_broadcast_sd((double const *)(&ones));
                ymm3 = _mm256_broadcast_sd((double const *)(&ones));
            }
            else if(1 == n_rem)
            {
                ///GEMM code begins///
                BLIS_DTRSM_SMALL_GEMM_4mx1n(a10,b01,cs_b,p_lda,k_iter)

                ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     //register to hold alpha

                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));    //B11[0][0] B11[1][0] B11[2][0] B11[3][0]

                ymm0 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);  //B11[0-3][0] * alpha -= B01[0-3][0]
                ymm1 = _mm256_broadcast_sd((double const *)(&ones));
                ymm2 = _mm256_broadcast_sd((double const *)(&ones));
                ymm3 = _mm256_broadcast_sd((double const *)(&ones));
            }

            ///transpose of B11//
            ///unpacklow///
            ymm9 = _mm256_unpacklo_pd(ymm0, ymm1);  //B11[0][0] B11[0][1] B11[2][0] B11[2][1]
            ymm11 = _mm256_unpacklo_pd(ymm2, ymm3); //B11[0][2] B11[0][3] B11[2][2] B11[2][3]

            //rearrange low elements
            ymm8 = _mm256_permute2f128_pd(ymm9,ymm11,0x20);     //B11[0][0] B11[0][1] B11[0][2] B11[0][3]
            ymm10 = _mm256_permute2f128_pd(ymm9,ymm11,0x31);    //B11[2][0] B11[2][1] B11[2][2] B11[2][3]

            ////unpackhigh////
            ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);  //B11[1][0] B11[1][1] B11[3][0] B11[3][1]
            ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);  //B11[1][2] B11[1][3] B11[3][2] B11[3][3]

            //rearrange high elements
            ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);  //B11[1][0] B11[1][1] B11[1][2] B11[1][3]
            ymm11 = _mm256_permute2f128_pd(ymm0,ymm1,0x31); //B11[3][0] B11[3][1] B11[3][2] B11[3][3]

            ymm0 = _mm256_broadcast_sd((double const *)&ones);

            ////extract a00
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack));

            //perform mul operation
            ymm8 = DTRSM_SMALL_DIV_OR_SCALE(ymm8, ymm1);

            //extract a11
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

            ymm2 = _mm256_broadcast_sd((double const *)(a11 + cs_a*1));
            ymm3 = _mm256_broadcast_sd((double const *)(a11 + cs_a*2));
            ymm4 = _mm256_broadcast_sd((double const *)(a11 + cs_a*3));

            a11 += rs_a;

            //(ROw1): FMA operations
            ymm9 = _mm256_fnmadd_pd(ymm2, ymm8, ymm9);
            ymm10 = _mm256_fnmadd_pd(ymm3, ymm8, ymm10);
            ymm11 = _mm256_fnmadd_pd(ymm4, ymm8, ymm11);

            //perform mul operation
            ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm1);

            ymm3 = _mm256_broadcast_sd((double const *)(a11 + cs_a*2));
            ymm4 = _mm256_broadcast_sd((double const *)(a11 + cs_a*3));

            a11 += rs_a;

            //extract a22
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

            //(ROw2): FMA operations
            ymm10 = _mm256_fnmadd_pd(ymm3, ymm9, ymm10);
            ymm11 = _mm256_fnmadd_pd(ymm4, ymm9, ymm11);

            //perform mul operation
            ymm10 = DTRSM_SMALL_DIV_OR_SCALE(ymm10, ymm1);

            ymm4 = _mm256_broadcast_sd((double const *)(a11 + cs_a*3));

            a11 += rs_a;

            //extract a33
            ymm1 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

            //(ROw5): FMA operations
            ymm11 = _mm256_fnmadd_pd(ymm4, ymm10, ymm11);

            //perform mul operation
            ymm11 = DTRSM_SMALL_DIV_OR_SCALE(ymm11, ymm1);

            //unpacklow//
            ymm1 = _mm256_unpacklo_pd(ymm8, ymm9);      //B11[0][0] B11[1][0] B11[0][2] B11[1][2]
            ymm3 = _mm256_unpacklo_pd(ymm10, ymm11);    //B11[2][0] B11[3][0] B11[2][2] B11[3][2]

            //rearrange low elements
            ymm0 = _mm256_permute2f128_pd(ymm1, ymm3, 0x20);    //B11[0][0] B11[1][0] B11[2][0] B11[3][0]
            ymm2 = _mm256_permute2f128_pd(ymm1, ymm3, 0x31);    //B11[0][2] B11[1][2] B11[2][2] B11[3][2]

            ///unpack high///
            ymm8 = _mm256_unpackhi_pd(ymm8, ymm9);    //B11[0][1] B11[1][1] B11[0][3] B11[1][3]
            ymm9 = _mm256_unpackhi_pd(ymm10, ymm11);    //B11[2][1] B11[3][1] B11[2][3] B11[3][3]

            //rearrange high elements
            ymm1 = _mm256_permute2f128_pd(ymm8, ymm9, 0x20);    //B11[0][1] B11[1][1] B11[2][1] B11[3][1]
            ymm3 = _mm256_permute2f128_pd(ymm8, ymm9, 0x31);    //B11[0][3] B11[1][3] B11[2][3] B11[3][3]

            if(3 == n_rem)
            {
                _mm256_storeu_pd((double *)(b11 + cs_b * 0), ymm0);    //store B11[0][0-3]
                _mm256_storeu_pd((double *)(b11 + cs_b * 1), ymm1);    //store B11[1][0-3]
                _mm256_storeu_pd((double *)(b11 + cs_b * 2), ymm2);    //store B11[2][0-3]
            }
            else if(2 == n_rem)
            {
                _mm256_storeu_pd((double *)(b11 + cs_b * 0), ymm0);    //store B11[0][0-3]
                _mm256_storeu_pd((double *)(b11 + cs_b * 1), ymm1);    //store B11[1][0-3]
            }
            else if(1 == n_rem)
            {
                _mm256_storeu_pd((double *)(b11 + cs_b * 0), ymm0);    //store B11[0][0-3]
            }
        }
        m_rem -=4;
        i +=4;
    }

    if(m_rem)
    {
        a10 = L + (i*cs_a);               //pointer to block of A to be used for GEMM
        // Do transpose for a10 & store in D_A_pack
        double *ptr_a10_dup = D_A_pack;
        if(3 == m_rem) // Repetative A blocks will be 3*3
        {
            dim_t p_lda = 4; // packed leading dimension
            if(transa)
            {
                for(dim_t x=0;x<i;x+=p_lda)
                {
                    ymm0 = _mm256_loadu_pd((double const *)(a10));
                    ymm1 = _mm256_loadu_pd((double const *)(a10 + cs_a));
                    ymm2 = _mm256_loadu_pd((double const *)(a10 + cs_a * 2));
                    ymm3 = _mm256_broadcast_sd((double const *)&ones);

                    ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
                    ymm5 = _mm256_unpacklo_pd(ymm2, ymm3);

                    ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                    ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);

                    ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
                    ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);

                    ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                    ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

                    _mm256_storeu_pd((double *)(ptr_a10_dup), ymm6);
                    _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda), ymm7);
                    _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*2), ymm8);
                    _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*3), ymm9);

                    a10 += p_lda;
                    ptr_a10_dup += p_lda*p_lda;
                }
            }
            else
            {
                for(dim_t x=0;x<i;x++)
                {
                    ymm0 = _mm256_loadu_pd((double const *)(a10 + rs_a * x));
                    _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda * x), ymm0);
                }
            }

            //cols
            for(j = 0; (j+d_nr-1) < n; j += d_nr)     //loop along 'N' dimension
            {
                a10 = D_A_pack;           //pointer to block of A to be used for GEMM
                a11 = L + (i*rs_a) + (i*cs_a);             //pointer to block of A to be used for TRSM
                b01 = B + (j*cs_b);        //pointer to block of B to be used for GEMM
                b11 = B + i + (j* cs_b);        //pointer to block of B to be used for TRSM

                k_iter = i;          //number of times GEMM to be performed(in blocks of 4x4)

                /*Fill zeros into ymm registers used in gemm accumulations */
                BLIS_SET_YMM_REG_ZEROS

                ///GEMM code begins///
                BLIS_DTRSM_SMALL_GEMM_4mx6n(a10,b01,cs_b,p_lda,k_iter)

                ///GEMM code ends///
                ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));    //register to store alpha value

                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));
                ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *1));
                ymm2 = _mm256_loadu_pd((double const *)(b11 + cs_b *2));
                ymm3 = _mm256_loadu_pd((double const *)(b11 + cs_b *3));
                ymm8 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);
                ymm9 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);
                ymm10 = _mm256_fmsub_pd(ymm2, ymm16, ymm10);
                ymm11 = _mm256_fmsub_pd(ymm3, ymm16, ymm11);

                ymm0 = _mm256_blend_pd(ymm8, ymm0, 0x08);
                ymm1 = _mm256_blend_pd(ymm9, ymm1, 0x08);
                ymm2 = _mm256_blend_pd(ymm10, ymm2, 0x08);
                ymm3 = _mm256_blend_pd(ymm11, ymm3, 0x08);

                _mm256_storeu_pd((double *)(b11), ymm0);                //store(B11[0-3][0])
                _mm256_storeu_pd((double *)(b11 + cs_b * 1), ymm1);        //store(B11[0-3][1])
                _mm256_storeu_pd((double *)(b11 + cs_b * 2), ymm2);        //store(B11[0-3][2])
                _mm256_storeu_pd((double *)(b11 + cs_b * 3), ymm3);        //store(B11[0-3][3])

                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *4));
                ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *5));
                ymm8 = _mm256_fmsub_pd(ymm0, ymm16, ymm4);
                ymm9 = _mm256_fmsub_pd(ymm1, ymm16, ymm5);

                ymm0 = _mm256_blend_pd(ymm8, ymm0, 0x08);
                ymm1 = _mm256_blend_pd(ymm9, ymm1, 0x08);

                _mm256_storeu_pd((double *)(b11 + cs_b * 4), ymm0);        //store(B11[0-3][2])
                _mm256_storeu_pd((double *)(b11 + cs_b * 5), ymm1);        //store(B11[0-3][3])

                if(transa)
                dtrsm_AutXB_ref(a11, b11, m_rem, 6, cs_a, cs_b,is_unitdiag);
                else
                dtrsm_AlXB_ref(a11, b11, m_rem, 6, rs_a, cs_b, is_unitdiag);
            }

            dim_t n_rem = n-j;
            if((n_rem >= 4))
            {
                a10 = D_A_pack;            //pointer to block of A to be used for GEMM
                a11 = L + (i*rs_a) + (i*cs_a);        //pointer to block of A to be used for TRSM
                b01 = B + (j*cs_b);        //pointer to block of B to be used for GEMM
                b11 = B + i + (j* cs_b);        //pointer to block of B to be used for TRSM

                k_iter = i;          //number of times GEMM to be performed(in blocks of 4x4)

                /*Fill zeros into ymm registers used in gemm accumulations */
                BLIS_SET_YMM_REG_ZEROS

                ///GEMM code begins///
                BLIS_DTRSM_SMALL_GEMM_4mx4n(a10,b01,cs_b,p_lda,k_iter)

                ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     //register to hold alpha

                ///implement TRSM///

                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));
                ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *1));
                ymm2 = _mm256_loadu_pd((double const *)(b11 + cs_b *2));
                xmm5 = _mm_loadu_pd((double const*)(b11 + cs_b * 3));
                ymm3 = _mm256_broadcast_sd((double const *)(b11 + cs_b*3 + 2));
                ymm3 = _mm256_insertf128_pd(ymm3, xmm5, 0);
                ymm8 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);
                ymm9 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);
                ymm10 = _mm256_fmsub_pd(ymm2, ymm16, ymm10);
                ymm11 = _mm256_fmsub_pd(ymm3, ymm16, ymm11);

                ymm0 = _mm256_blend_pd(ymm8, ymm0, 0x08);
                ymm1 = _mm256_blend_pd(ymm9, ymm1, 0x08);
                ymm2 = _mm256_blend_pd(ymm10, ymm2, 0x08);
                ymm3 = _mm256_blend_pd(ymm11, ymm3, 0x08);

                _mm256_storeu_pd((double *)(b11), ymm0);                //store(B11[0-3][0])
                _mm256_storeu_pd((double *)(b11 + cs_b * 1), ymm1);        //store(B11[0-3][1])
                _mm256_storeu_pd((double *)(b11 + cs_b * 2), ymm2);        //store(B11[0-3][2])
                xmm5 = _mm256_extractf128_pd(ymm3, 0);
                _mm_storeu_pd((double *)(b11 + cs_b * 3),xmm5);
                _mm_storel_pd((b11 + cs_b * 3 + 2), _mm256_extractf128_pd(ymm3, 1));

                if(transa)
                dtrsm_AutXB_ref(a11, b11, m_rem, 4, cs_a, cs_b,is_unitdiag);
                else
                dtrsm_AlXB_ref(a11, b11, m_rem, 4, rs_a, cs_b, is_unitdiag);
                n_rem -= 4;
                j +=4;
            }

            if(n_rem)
            {
                a10 = D_A_pack;            //pointer to block of A to be used for GEMM
                a11 = L + (i*rs_a) + (i*cs_a);        //pointer to block of A to be used for TRSM
                b01 = B + (j*cs_b);        //pointer to block of B to be used for GEMM
                b11 = B + i + (j* cs_b);        //pointer to block of B to be used for TRSM

                k_iter = i;          //number of times GEMM to be performed(in blocks of 4x4)

                /*Fill zeros into ymm registers used in gemm accumulations */
                BLIS_SET_YMM_REG_ZEROS

                if(3 == n_rem)
                {
                    ///GEMM code begins///
                    BLIS_DTRSM_SMALL_GEMM_4mx3n(a10,b01,cs_b,p_lda,k_iter)

                    BLIS_PRE_DTRSM_SMALL_3M_3N(AlphaVal,b11,cs_b)

                    if(transa)
                    dtrsm_AutXB_ref(a11, b11, m_rem, 3, cs_a, cs_b,is_unitdiag);
                    else
                    dtrsm_AlXB_ref(a11, b11, m_rem, 3, rs_a, cs_b, is_unitdiag);
                }
                else if(2 == n_rem)
                {
                    ///GEMM code begins///
                    BLIS_DTRSM_SMALL_GEMM_4mx2n(a10,b01,cs_b,p_lda,k_iter)

                    BLIS_PRE_DTRSM_SMALL_3M_2N(AlphaVal,b11,cs_b)

                    if(transa)
                    dtrsm_AutXB_ref(a11, b11, m_rem, 2, cs_a, cs_b,is_unitdiag);
                    else
                    dtrsm_AlXB_ref(a11, b11, m_rem, 2, rs_a, cs_b, is_unitdiag);
                }
                else if(1 == n_rem)
                {
                    ///GEMM code begins///
                    BLIS_DTRSM_SMALL_GEMM_4mx1n(a10,b01,cs_b,p_lda,k_iter)

                    BLIS_PRE_DTRSM_SMALL_3M_1N(AlphaVal,b11,cs_b)

                    if(transa)
                    dtrsm_AutXB_ref(a11, b11, m_rem, 1, cs_a, cs_b, is_unitdiag);
                    else
                    dtrsm_AlXB_ref(a11, b11, m_rem, 1, rs_a, cs_b, is_unitdiag);
                }
            }
        }
        else if(2 == m_rem) // Repetative A blocks will be 2*2
        {
            dim_t p_lda = 4; // packed leading dimension
            if(transa)
            {
                for(dim_t x=0;x<i;x+=p_lda)
                {
                    ymm0 = _mm256_loadu_pd((double const *)(a10));
                    ymm1 = _mm256_loadu_pd((double const *)(a10 + cs_a));
                    ymm2 = _mm256_broadcast_sd((double const *)&ones);
                    ymm3 = _mm256_broadcast_sd((double const *)&ones);

                    ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
                    ymm5 = _mm256_unpacklo_pd(ymm2, ymm3);

                    ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                    ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);

                    ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
                    ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);

                    ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                    ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

                    _mm256_storeu_pd((double *)(ptr_a10_dup), ymm6);
                    _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda), ymm7);
                    _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*2), ymm8);
                    _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*3), ymm9);

                    a10 += p_lda;
                    ptr_a10_dup += p_lda*p_lda;
                }
            }
            else
            {
                for(dim_t x=0;x<i;x++)
                {
                    ymm0 = _mm256_loadu_pd((double const *)(a10 + rs_a * x));
                    _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda * x), ymm0);
                }
            }
            //cols
            for(j = 0; (j+d_nr-1) < n; j += d_nr)   //loop along 'N' dimension
            {
                a10 = D_A_pack;           //pointer to block of A to be used for GEMM
                a11 = L + (i*rs_a) + (i*cs_a);             //pointer to block of A to be used for TRSM
                b01 = B + (j*cs_b);        //pointer to block of B to be used for GEMM
                b11 = B + i + (j* cs_b);        //pointer to block of B to be used for TRSM

                k_iter = i;          //number of times GEMM to be performed(in blocks of 4x4)

                /*Fill zeros into ymm registers used in gemm accumulations */
                BLIS_SET_YMM_REG_ZEROS

                ///GEMM code begins///
                BLIS_DTRSM_SMALL_GEMM_4mx6n(a10,b01,cs_b,p_lda,k_iter)

                ///GEMM code ends///
                ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));    //register to store alpha value

                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));
                ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *1));
                ymm2 = _mm256_loadu_pd((double const *)(b11 + cs_b *2));
                ymm3 = _mm256_loadu_pd((double const *)(b11 + cs_b *3));
                ymm8 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);
                ymm9 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);
                ymm10 = _mm256_fmsub_pd(ymm2, ymm16, ymm10);
                ymm11 = _mm256_fmsub_pd(ymm3, ymm16, ymm11);

                ymm0 = _mm256_blend_pd(ymm8, ymm0, 0x0C);
                ymm1 = _mm256_blend_pd(ymm9, ymm1, 0x0C);
                ymm2 = _mm256_blend_pd(ymm10, ymm2, 0x0C);
                ymm3 = _mm256_blend_pd(ymm11, ymm3, 0x0C);

                _mm256_storeu_pd((double *)(b11), ymm0);                //store(B11[0-3][0])
                _mm256_storeu_pd((double *)(b11 + cs_b * 1), ymm1);        //store(B11[0-3][1])
                _mm256_storeu_pd((double *)(b11 + cs_b * 2), ymm2);        //store(B11[0-3][2])
                _mm256_storeu_pd((double *)(b11 + cs_b * 3), ymm3);        //store(B11[0-3][3])

                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *4));
                ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *5));
                ymm8 = _mm256_fmsub_pd(ymm0, ymm16, ymm4);
                ymm9 = _mm256_fmsub_pd(ymm1, ymm16, ymm5);

                ymm0 = _mm256_blend_pd(ymm8, ymm0, 0x0C);
                ymm1 = _mm256_blend_pd(ymm9, ymm1, 0x0C);

                _mm256_storeu_pd((double *)(b11 + cs_b * 4), ymm0);        //store(B11[0-3][2])
                _mm256_storeu_pd((double *)(b11 + cs_b * 5), ymm1);        //store(B11[0-3][3])

                if(transa)
                dtrsm_AutXB_ref(a11, b11, m_rem, 6, cs_a, cs_b, is_unitdiag);
                else
                dtrsm_AlXB_ref(a11, b11, m_rem, 6, rs_a, cs_b, is_unitdiag);
            }

            dim_t n_rem = n-j;
            if((n_rem >= 4))
            {
                a10 = D_A_pack;            //pointer to block of A to be used for GEMM
                a11 = L + (i*rs_a) + (i*cs_a);        //pointer to block of A to be used for TRSM
                b01 = B + (j*cs_b);        //pointer to block of B to be used for GEMM
                b11 = B + i + (j* cs_b);        //pointer to block of B to be used for TRSM

                k_iter = i;          //number of times GEMM to be performed(in blocks of 4x4)

                /*Fill zeros into ymm registers used in gemm accumulations */
                BLIS_SET_YMM_REG_ZEROS

                ///GEMM code begins///
                BLIS_DTRSM_SMALL_GEMM_4mx4n(a10,b01,cs_b,p_lda,k_iter)

                ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     //register to hold alpha

                ///implement TRSM///

                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));
                ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *1));
                ymm2 = _mm256_loadu_pd((double const *)(b11 + cs_b *2));
                xmm5 = _mm_loadu_pd((double const*)(b11 + cs_b * 3));
                ymm3 = _mm256_insertf128_pd(ymm3, xmm5, 0);

                ymm8 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);
                ymm9 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);
                ymm10 = _mm256_fmsub_pd(ymm2, ymm16, ymm10);
                ymm11 = _mm256_fmsub_pd(ymm3, ymm16, ymm11);

                ymm0 = _mm256_blend_pd(ymm8, ymm0, 0x0C);
                ymm1 = _mm256_blend_pd(ymm9, ymm1, 0x0C);
                ymm2 = _mm256_blend_pd(ymm10, ymm2, 0x0C);
                ymm3 = _mm256_blend_pd(ymm11, ymm3, 0x0C);

                _mm256_storeu_pd((double *)(b11), ymm0);                //store(B11[0-3][0])
                _mm256_storeu_pd((double *)(b11 + cs_b * 1), ymm1);        //store(B11[0-3][1])
                _mm256_storeu_pd((double *)(b11 + cs_b * 2), ymm2);        //store(B11[0-3][2])
                 xmm5 = _mm256_extractf128_pd(ymm3, 0);
                _mm_storeu_pd((double *)(b11 + cs_b * 3), xmm5);

                if(transa)
                dtrsm_AutXB_ref(a11, b11, m_rem, 4, cs_a, cs_b, is_unitdiag);
                else
                dtrsm_AlXB_ref(a11, b11, m_rem, 4, rs_a, cs_b, is_unitdiag);
                n_rem -= 4;
                j +=4;
            }
            if(n_rem)
            {
                a10 = D_A_pack;            //pointer to block of A to be used for GEMM
                a11 = L + (i*rs_a) + (i*cs_a);        //pointer to block of A to be used for TRSM
                b01 = B + (j*cs_b);        //pointer to block of B to be used for GEMM
                b11 = B + i + (j* cs_b);        //pointer to block of B to be used for TRSM

                k_iter = i;          //number of times GEMM to be performed(in blocks of 4x4)

                /*Fill zeros into ymm registers used in gemm accumulations */
                BLIS_SET_YMM_REG_ZEROS

                if(3 == n_rem)
                {
                    ///GEMM code begins///
                    BLIS_DTRSM_SMALL_GEMM_4mx3n(a10,b01,cs_b,p_lda,k_iter)

                    BLIS_PRE_DTRSM_SMALL_2M_3N(AlphaVal,b11,cs_b)

                    if(transa)
                    dtrsm_AutXB_ref(a11, b11, m_rem, 3, cs_a, cs_b, is_unitdiag);
                    else
                    dtrsm_AlXB_ref(a11, b11, m_rem, 3, rs_a, cs_b, is_unitdiag);
                }
                else if(2 == n_rem)
                {
                    ///GEMM code begins///
                    BLIS_DTRSM_SMALL_GEMM_4mx2n(a10,b01,cs_b,p_lda,k_iter)

                    BLIS_PRE_DTRSM_SMALL_2M_2N(AlphaVal,b11,cs_b)

                    if(transa)
                    dtrsm_AutXB_ref(a11, b11, m_rem, 2, cs_a, cs_b, is_unitdiag);
                    else
                    dtrsm_AlXB_ref(a11, b11, m_rem, 2, rs_a, cs_b, is_unitdiag);
                }
                else if(1 == n_rem)
                {
                    ///GEMM code begins///
                    BLIS_DTRSM_SMALL_GEMM_4mx1n(a10,b01,cs_b,p_lda,k_iter)

                    BLIS_PRE_DTRSM_SMALL_2M_1N(AlphaVal,b11,cs_b)

                    if(transa)
                    dtrsm_AutXB_ref(a11, b11, m_rem, 1, cs_a, cs_b, is_unitdiag);
                    else
                    dtrsm_AlXB_ref(a11, b11, m_rem, 1, rs_a, cs_b, is_unitdiag);
                }
            }
            m_rem -=2;
            i+=2;
        }
        else if(1 == m_rem) // Repetative A blocks will be 1*1
        {
            dim_t p_lda = 4; // packed leading dimension
            if(transa)
            {
                for(dim_t x=0;x<i;x+=p_lda)
                {
                    ymm0 = _mm256_loadu_pd((double const *)(a10));
                    ymm1 = _mm256_broadcast_sd((double const *)&ones);
                    ymm2 = _mm256_broadcast_sd((double const *)&ones);
                    ymm3 = _mm256_broadcast_sd((double const *)&ones);

                    ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
                    ymm5 = _mm256_unpacklo_pd(ymm2, ymm3);

                    ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
                    ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);

                    ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
                    ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);

                    ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
                    ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

                    _mm256_storeu_pd((double *)(ptr_a10_dup), ymm6);
                    _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda), ymm7);
                    _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*2), ymm8);
                    _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda*3), ymm9);

                    a10 += p_lda;
                    ptr_a10_dup += p_lda*p_lda;
                }
            }
            else
            {
                for(dim_t x=0;x<i;x++)
                {
                    ymm0 = _mm256_loadu_pd((double const *)(a10 + rs_a * x));
                    _mm256_storeu_pd((double *)(ptr_a10_dup + p_lda * x), ymm0);
                }
            }
            //cols
            for(j = 0; (j+d_nr-1) < n; j += d_nr)   //loop along 'N' dimension
            {
                a10 = D_A_pack;           //pointer to block of A to be used for GEMM
                a11 = L + (i*rs_a) + (i*cs_a);             //pointer to block of A to be used for TRSM
                b01 = B + (j*cs_b);        //pointer to block of B to be used for GEMM
                b11 = B + i + (j* cs_b);        //pointer to block of B to be used for TRSM

                k_iter = i;          //number of times GEMM to be performed(in blocks of 4x4)

                /*Fill zeros into ymm registers used in gemm accumulations */
                BLIS_SET_YMM_REG_ZEROS

                ///GEMM code begins///
                BLIS_DTRSM_SMALL_GEMM_4mx6n(a10,b01,cs_b,p_lda,k_iter)

                ///GEMM code ends///
                ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));    //register to store alpha value

                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *0));
                ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *1));
                ymm2 = _mm256_loadu_pd((double const *)(b11 + cs_b *2));
                ymm3 = _mm256_loadu_pd((double const *)(b11 + cs_b *3));
                ymm8 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);
                ymm9 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);
                ymm10 = _mm256_fmsub_pd(ymm2, ymm16, ymm10);
                ymm11 = _mm256_fmsub_pd(ymm3, ymm16, ymm11);

                ymm0 = _mm256_blend_pd(ymm8, ymm0, 0x0E);
                ymm1 = _mm256_blend_pd(ymm9, ymm1, 0x0E);
                ymm2 = _mm256_blend_pd(ymm10, ymm2, 0x0E);
                ymm3 = _mm256_blend_pd(ymm11, ymm3, 0x0E);

                _mm256_storeu_pd((double *)(b11), ymm0);                //store(B11[0-3][0])
                _mm256_storeu_pd((double *)(b11 + cs_b * 1), ymm1);        //store(B11[0-3][1])
                _mm256_storeu_pd((double *)(b11 + cs_b * 2), ymm2);        //store(B11[0-3][2])
                _mm256_storeu_pd((double *)(b11 + cs_b * 3), ymm3);        //store(B11[0-3][3])

                ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b *4));
                ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b *5));
                ymm8 = _mm256_fmsub_pd(ymm0, ymm16, ymm4);
                ymm9 = _mm256_fmsub_pd(ymm1, ymm16, ymm5);

                ymm0 = _mm256_blend_pd(ymm8, ymm0, 0x0E);
                ymm1 = _mm256_blend_pd(ymm9, ymm1, 0x0E);

                _mm256_storeu_pd((double *)(b11 + cs_b * 4), ymm0);        //store(B11[0-3][2])
                _mm256_storeu_pd((double *)(b11 + cs_b * 5), ymm1);        //store(B11[0-3][3])

                if(transa)
                dtrsm_AutXB_ref(a11, b11, m_rem, 6, cs_a, cs_b, is_unitdiag);
                else
                dtrsm_AlXB_ref(a11, b11, m_rem, 6, rs_a, cs_b, is_unitdiag);
            }
            dim_t n_rem = n-j;
            if((n_rem >= 4))
            {
                a10 = D_A_pack;           //pointer to block of A to be used for GEMM
                a11 = L + (i*rs_a) + (i*cs_a);             //pointer to block of A to be used for TRSM
                b01 = B + (j*cs_b);        //pointer to block of B to be used for GEMM
                b11 = B + i + (j* cs_b);        //pointer to block of B to be used for TRSM

                k_iter = i;          //number of times GEMM to be performed(in blocks of 4x4)

                /*Fill zeros into ymm registers used in gemm accumulations */
                BLIS_SET_YMM_REG_ZEROS

                ///GEMM code begins///
                BLIS_DTRSM_SMALL_GEMM_4mx4n(a10,b01,cs_b,p_lda,k_iter)

                ymm16 = _mm256_broadcast_sd((double const *)(&AlphaVal));     //register to hold alpha

                ///implement TRSM///
                ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b *0));
                ymm1 = _mm256_broadcast_sd((double const *)(b11 + cs_b *1));
                ymm2 = _mm256_broadcast_sd((double const *)(b11 + cs_b *2));
                ymm3 = _mm256_broadcast_sd((double const *)(b11 + cs_b *3));

                ymm8 = _mm256_fmsub_pd(ymm0, ymm16, ymm8);
                ymm9 = _mm256_fmsub_pd(ymm1, ymm16, ymm9);
                ymm10 = _mm256_fmsub_pd(ymm2, ymm16, ymm10);
                ymm11 = _mm256_fmsub_pd(ymm3, ymm16, ymm11);

                ymm0 = _mm256_blend_pd(ymm8, ymm0, 0x0E);
                ymm1 = _mm256_blend_pd(ymm9, ymm1, 0x0E);
                ymm2 = _mm256_blend_pd(ymm10, ymm2, 0x0E);
                ymm3 = _mm256_blend_pd(ymm11, ymm3, 0x0E);

                _mm_storel_pd((b11 + cs_b * 0), _mm256_extractf128_pd(ymm0, 0));
                _mm_storel_pd((b11 + cs_b * 1), _mm256_extractf128_pd(ymm1, 0));
                _mm_storel_pd((b11 + cs_b * 2), _mm256_extractf128_pd(ymm2, 0));
                _mm_storel_pd((b11 + cs_b * 3), _mm256_extractf128_pd(ymm3, 0));

                if(transa)
                dtrsm_AutXB_ref(a11, b11, m_rem, 4, cs_a, cs_b, is_unitdiag);
                else
                dtrsm_AlXB_ref(a11, b11, m_rem, 4, rs_a, cs_b, is_unitdiag);
                n_rem -= 4;
                j+=4;
            }

            if(n_rem)
            {
                a10 = D_A_pack;           //pointer to block of A to be used for GEMM
                a11 = L + (i*rs_a) + (i*cs_a);             //pointer to block of A to be used for TRSM
                b01 = B + (j*cs_b);        //pointer to block of B to be used for GEMM
                b11 = B + i + (j* cs_b);        //pointer to block of B to be used for TRSM

                k_iter = i;          //number of times GEMM to be performed(in blocks of 4x4)

                /*Fill zeros into ymm registers used in gemm accumulations */
                BLIS_SET_YMM_REG_ZEROS

                if(3 == n_rem)
                {
                    ///GEMM code begins///
                    BLIS_DTRSM_SMALL_GEMM_4mx3n(a10,b01,cs_b,p_lda,k_iter)

                    BLIS_PRE_DTRSM_SMALL_1M_3N(AlphaVal,b11,cs_b)

                    if(transa)
                    dtrsm_AutXB_ref(a11, b11, m_rem, 3, cs_a, cs_b, is_unitdiag);
                    else
                    dtrsm_AlXB_ref(a11, b11, m_rem, 3, rs_a, cs_b, is_unitdiag);
                }
                else if(2 == n_rem)
                {
                    ///GEMM code begins///
                    BLIS_DTRSM_SMALL_GEMM_4mx2n(a10,b01,cs_b,p_lda,k_iter)

                    BLIS_PRE_DTRSM_SMALL_1M_2N(AlphaVal,b11,cs_b)

                    if(transa)
                    dtrsm_AutXB_ref(a11, b11, m_rem, 2, cs_a, cs_b, is_unitdiag);
                    else
                    dtrsm_AlXB_ref(a11, b11, m_rem, 2, rs_a, cs_b, is_unitdiag);
                }
                else if(1 == n_rem)
                {
                    ///GEMM code begins///
                    BLIS_DTRSM_SMALL_GEMM_4mx1n(a10,b01,cs_b,p_lda,k_iter)

                    BLIS_PRE_DTRSM_SMALL_1M_1N(AlphaVal,b11,cs_b)

                    if(transa)
                    dtrsm_AutXB_ref(a11, b11, m_rem, 1, cs_a, cs_b, is_unitdiag);
                    else
                    dtrsm_AutXB_ref(a11, b11, m_rem, 1, rs_a, cs_b, is_unitdiag);
                }
            }
            m_rem -=1;
            i+=1;
        }
    }

    if ((required_packing_A == 1) &&
        bli_mem_is_alloc( &local_mem_buf_A_s ))
    {
        bli_membrk_release(&rntm, &local_mem_buf_A_s);
    }
  return BLIS_SUCCESS;
}
#endif //BLIS_ENABLE_SMALL_MATRIX_TRSM