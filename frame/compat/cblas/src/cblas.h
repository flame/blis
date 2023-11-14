/*

   Copyright (C) 2020-2023, Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef CBLAS_H
#define CBLAS_H
#include <stddef.h>

// We need to #include "bli_type_defs.h" in order to pull in the
// definition of f77_int. But in order to #include that header, we
// also need to pull in the headers that precede it in blis.h.
#include "bli_system.h"
#include "bli_lang_defs.h"

#include "bli_config.h"
#include "bli_config_macro_defs.h"
#include "bli_type_defs.h"

/*
 * Enumerated and derived types
 */
enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
enum CBLAS_UPLO {CblasUpper=121, CblasLower=122};
enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132};
enum CBLAS_SIDE {CblasLeft=141, CblasRight=142};
enum CBLAS_STORAGE {CblasPacked=151};
enum CBLAS_IDENTIFIER {CblasAMatrix=161, CblasBMatrix=162};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS functions (complex are recast as routines)
 * ===========================================================================
 */
BLIS_EXPORT_BLAS float  cblas_sdsdot(f77_int N, float alpha, const float *X,
                    f77_int incX, const float *Y, f77_int incY);
BLIS_EXPORT_BLAS double cblas_dsdot(f77_int N, const float *X, f77_int incX, const float *Y,
                   f77_int incY);
BLIS_EXPORT_BLAS float  cblas_sdot(f77_int N, const float  *X, f77_int incX,
                  const float  *Y, f77_int incY);
BLIS_EXPORT_BLAS double cblas_ddot(f77_int N, const double *X, f77_int incX,
                  const double *Y, f77_int incY);

/*
 * Functions having prefixes Z and C only
 */
BLIS_EXPORT_BLAS void   cblas_cdotu_sub(f77_int N, const void *X, f77_int incX,
                       const void *Y, f77_int incY, void *dotu);
BLIS_EXPORT_BLAS void   cblas_cdotc_sub(f77_int N, const void *X, f77_int incX,
                       const void *Y, f77_int incY, void *dotc);

BLIS_EXPORT_BLAS void   cblas_zdotu_sub(f77_int N, const void *X, f77_int incX,
                       const void *Y, f77_int incY, void *dotu);
BLIS_EXPORT_BLAS void   cblas_zdotc_sub(f77_int N, const void *X, f77_int incX,
                       const void *Y, f77_int incY, void *dotc);


/*
 * Functions having prefixes S D SC DZ
 */
BLIS_EXPORT_BLAS float  cblas_snrm2(f77_int N, const float *X, f77_int incX);
BLIS_EXPORT_BLAS float  cblas_sasum(f77_int N, const float *X, f77_int incX);

BLIS_EXPORT_BLAS double cblas_dnrm2(f77_int N, const double *X, f77_int incX);
BLIS_EXPORT_BLAS double cblas_dasum(f77_int N, const double *X, f77_int incX);

BLIS_EXPORT_BLAS float  cblas_scnrm2(f77_int N, const void *X, f77_int incX);
BLIS_EXPORT_BLAS float  cblas_scasum(f77_int N, const void *X, f77_int incX);

BLIS_EXPORT_BLAS double cblas_dznrm2(f77_int N, const void *X, f77_int incX);
BLIS_EXPORT_BLAS double cblas_dzasum(f77_int N, const void *X, f77_int incX);


/*
 * Functions having standard 4 prefixes (S D C Z)
 */
BLIS_EXPORT_BLAS f77_int cblas_isamax(f77_int N, const float  *X, f77_int incX);
BLIS_EXPORT_BLAS f77_int cblas_idamax(f77_int N, const double *X, f77_int incX);
BLIS_EXPORT_BLAS f77_int cblas_icamax(f77_int N, const void   *X, f77_int incX);
BLIS_EXPORT_BLAS f77_int cblas_izamax(f77_int N, const void   *X, f77_int incX);

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS routines
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (s, d, c, z)
 */
void BLIS_EXPORT_BLAS cblas_sswap(f77_int N, float *X, f77_int incX,
                 float *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_scopy(f77_int N, const float *X, f77_int incX,
                 float *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_saxpy(f77_int N, float alpha, const float *X,
                 f77_int incX, float *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_saxpby(f77_int N, float alpha, const float *X,
                 f77_int incX, float beta, float *Y, f77_int incY);

void BLIS_EXPORT_BLAS cblas_dswap(f77_int N, double *X, f77_int incX,
                 double *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_dcopy(f77_int N, const double *X, f77_int incX,
                 double *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_daxpy(f77_int N, double alpha, const double *X,
                 f77_int incX, double *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_daxpby(f77_int N, double alpha, const double *X,
                 f77_int incX, double beta, double *Y, f77_int incY);

void BLIS_EXPORT_BLAS cblas_cswap(f77_int N, void *X, f77_int incX,
                 void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_ccopy(f77_int N, const void *X, f77_int incX,
                 void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_caxpy(f77_int N, const void *alpha, const void *X,
                 f77_int incX, void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_caxpby(f77_int N, const void *alpha,
                const void *X, f77_int incX, const void* beta,
                void *Y, f77_int incY);

void BLIS_EXPORT_BLAS cblas_zswap(f77_int N, void *X, f77_int incX,
                 void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_zcopy(f77_int N, const void *X, f77_int incX,
                 void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_zaxpy(f77_int N, const void *alpha, const void *X,
                 f77_int incX, void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_zaxpby(f77_int N, const void *alpha,
                const void *X, f77_int incX, const void *beta,
                void *Y, f77_int incY);


/*
 * Routines with S and D prefix only
 */
void BLIS_EXPORT_BLAS cblas_srotg(float *a, float *b, float *c, float *s);
void BLIS_EXPORT_BLAS cblas_srotmg(float *d1, float *d2, float *b1, const float b2, float *P);
void BLIS_EXPORT_BLAS cblas_srot(f77_int N, float *X, f77_int incX,
                float *Y, f77_int incY, const float c, const float s);
void BLIS_EXPORT_BLAS cblas_srotm(f77_int N, float *X, f77_int incX,
                float *Y, f77_int incY, const float *P);

void BLIS_EXPORT_BLAS cblas_drotg(double *a, double *b, double *c, double *s);
void BLIS_EXPORT_BLAS cblas_drotmg(double *d1, double *d2, double *b1, const double b2, double *P);
void BLIS_EXPORT_BLAS cblas_drot(f77_int N, double *X, f77_int incX,
                double *Y, f77_int incY, const double c, const double  s);
void BLIS_EXPORT_BLAS cblas_drotm(f77_int N, double *X, f77_int incX,
                double *Y, f77_int incY, const double *P);


/*
 * Routines with S D C Z CS and ZD prefixes
 */
void BLIS_EXPORT_BLAS cblas_sscal(f77_int N, float alpha, float *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_dscal(f77_int N, double alpha, double *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_cscal(f77_int N, const void *alpha, void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_zscal(f77_int N, const void *alpha, void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_csscal(f77_int N, float alpha, void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_zdscal(f77_int N, double alpha, void *X, f77_int incX);

/*
 * ===========================================================================
 * Prototypes for level 2 BLAS
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (S, D, C, Z)
 */
void BLIS_EXPORT_BLAS cblas_sgemv(enum CBLAS_ORDER order,
                 enum CBLAS_TRANSPOSE TransA, f77_int M, f77_int N,
                 float alpha, const float *A, f77_int lda,
                 const float *X, f77_int incX, float beta,
                 float *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_sgbmv(enum CBLAS_ORDER order,
                 enum CBLAS_TRANSPOSE TransA, f77_int M, f77_int N,
                 f77_int KL, f77_int KU, float alpha,
                 const float *A, f77_int lda, const float *X,
                 f77_int incX, float beta, float *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_strmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const float *A, f77_int lda,
                 float *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_stbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, f77_int K, const float *A, f77_int lda,
                 float *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_stpmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const float *Ap, float *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_strsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const float *A, f77_int lda, float *X,
                 f77_int incX);
void BLIS_EXPORT_BLAS cblas_stbsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, f77_int K, const float *A, f77_int lda,
                 float *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_stpsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const float *Ap, float *X, f77_int incX);

void BLIS_EXPORT_BLAS cblas_dgemv(enum CBLAS_ORDER order,
                 enum CBLAS_TRANSPOSE TransA, f77_int M, f77_int N,
                 double alpha, const double *A, f77_int lda,
                 const double *X, f77_int incX, double beta,
                 double *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_dgbmv(enum CBLAS_ORDER order,
                 enum CBLAS_TRANSPOSE TransA, f77_int M, f77_int N,
                 f77_int KL, f77_int KU, double alpha,
                 const double *A, f77_int lda, const double *X,
                 f77_int incX, double beta, double *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_dtrmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const double *A, f77_int lda,
                 double *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_dtbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, f77_int K, const double *A, f77_int lda,
                 double *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_dtpmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const double *Ap, double *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_dtrsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const double *A, f77_int lda, double *X,
                 f77_int incX);
void BLIS_EXPORT_BLAS cblas_dtbsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, f77_int K, const double *A, f77_int lda,
                 double *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_dtpsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const double *Ap, double *X, f77_int incX);

void BLIS_EXPORT_BLAS cblas_cgemv(enum CBLAS_ORDER order,
                 enum CBLAS_TRANSPOSE TransA, f77_int M, f77_int N,
                 const void *alpha, const void *A, f77_int lda,
                 const void *X, f77_int incX, const void *beta,
                 void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_cgbmv(enum CBLAS_ORDER order,
                 enum CBLAS_TRANSPOSE TransA, f77_int M, f77_int N,
                 f77_int KL, f77_int KU, const void *alpha,
                 const void *A, f77_int lda, const void *X,
                 f77_int incX, const void *beta, void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_ctrmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const void *A, f77_int lda,
                 void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_ctbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, f77_int K, const void *A, f77_int lda,
                 void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_ctpmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const void *Ap, void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_ctrsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const void *A, f77_int lda, void *X,
                 f77_int incX);
void BLIS_EXPORT_BLAS cblas_ctbsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, f77_int K, const void *A, f77_int lda,
                 void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_ctpsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const void *Ap, void *X, f77_int incX);

void BLIS_EXPORT_BLAS cblas_zgemv(enum CBLAS_ORDER order,
                 enum CBLAS_TRANSPOSE TransA, f77_int M, f77_int N,
                 const void *alpha, const void *A, f77_int lda,
                 const void *X, f77_int incX, const void *beta,
                 void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_zgbmv(enum CBLAS_ORDER order,
                 enum CBLAS_TRANSPOSE TransA, f77_int M, f77_int N,
                 f77_int KL, f77_int KU, const void *alpha,
                 const void *A, f77_int lda, const void *X,
                 f77_int incX, const void *beta, void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_ztrmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const void *A, f77_int lda,
                 void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_ztbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, f77_int K, const void *A, f77_int lda,
                 void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_ztpmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const void *Ap, void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_ztrsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const void *A, f77_int lda, void *X,
                 f77_int incX);
void BLIS_EXPORT_BLAS cblas_ztbsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, f77_int K, const void *A, f77_int lda,
                 void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_ztpsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const void *Ap, void *X, f77_int incX);


/*
 * Routines with S and D prefixes only
 */
void BLIS_EXPORT_BLAS cblas_ssymv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, float alpha, const float *A,
                 f77_int lda, const float *X, f77_int incX,
                 float beta, float *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_ssbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, f77_int K, float alpha, const float *A,
                 f77_int lda, const float *X, f77_int incX,
                 float beta, float *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_sspmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, float alpha, const float *Ap,
                 const float *X, f77_int incX,
                 float beta, float *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_sger(enum CBLAS_ORDER order, f77_int M, f77_int N,
                float alpha, const float *X, f77_int incX,
                const float *Y, f77_int incY, float *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_ssyr(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, float alpha, const float *X,
                f77_int incX, float *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_sspr(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, float alpha, const float *X,
                f77_int incX, float *Ap);
void BLIS_EXPORT_BLAS cblas_ssyr2(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, float alpha, const float *X,
                f77_int incX, const float *Y, f77_int incY, float *A,
                f77_int lda);
void BLIS_EXPORT_BLAS cblas_sspr2(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, float alpha, const float *X,
                f77_int incX, const float *Y, f77_int incY, float *A);

void BLIS_EXPORT_BLAS cblas_dsymv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, double alpha, const double *A,
                 f77_int lda, const double *X, f77_int incX,
                 double beta, double *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_dsbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, f77_int K, double alpha, const double *A,
                 f77_int lda, const double *X, f77_int incX,
                 double beta, double *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_dspmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, double alpha, const double *Ap,
                 const double *X, f77_int incX,
                 double beta, double *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_dger(enum CBLAS_ORDER order, f77_int M, f77_int N,
                double alpha, const double *X, f77_int incX,
                const double *Y, f77_int incY, double *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_dsyr(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, double alpha, const double *X,
                f77_int incX, double *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_dspr(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, double alpha, const double *X,
                f77_int incX, double *Ap);
void BLIS_EXPORT_BLAS cblas_dsyr2(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, double alpha, const double *X,
                f77_int incX, const double *Y, f77_int incY, double *A,
                f77_int lda);
void BLIS_EXPORT_BLAS cblas_dspr2(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, double alpha, const double *X,
                f77_int incX, const double *Y, f77_int incY, double *A);


/*
 * Routines with C and Z prefixes only
 */
void BLIS_EXPORT_BLAS cblas_chemv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, const void *alpha, const void *A,
                 f77_int lda, const void *X, f77_int incX,
                 const void *beta, void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_chbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, f77_int K, const void *alpha, const void *A,
                 f77_int lda, const void *X, f77_int incX,
                 const void *beta, void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_chpmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, const void *alpha, const void *Ap,
                 const void *X, f77_int incX,
                 const void *beta, void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_cgeru(enum CBLAS_ORDER order, f77_int M, f77_int N,
                 const void *alpha, const void *X, f77_int incX,
                 const void *Y, f77_int incY, void *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_cgerc(enum CBLAS_ORDER order, f77_int M, f77_int N,
                 const void *alpha, const void *X, f77_int incX,
                 const void *Y, f77_int incY, void *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_cher(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, float alpha, const void *X, f77_int incX,
                void *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_chpr(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, float alpha, const void *X,
                f77_int incX, void *A);
void BLIS_EXPORT_BLAS cblas_cher2(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo, f77_int N,
                const void *alpha, const void *X, f77_int incX,
                const void *Y, f77_int incY, void *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_chpr2(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo, f77_int N,
                const void *alpha, const void *X, f77_int incX,
                const void *Y, f77_int incY, void *Ap);

void BLIS_EXPORT_BLAS cblas_zhemv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, const void *alpha, const void *A,
                 f77_int lda, const void *X, f77_int incX,
                 const void *beta, void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_zhbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, f77_int K, const void *alpha, const void *A,
                 f77_int lda, const void *X, f77_int incX,
                 const void *beta, void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_zhpmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, const void *alpha, const void *Ap,
                 const void *X, f77_int incX,
                 const void *beta, void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_zgeru(enum CBLAS_ORDER order, f77_int M, f77_int N,
                 const void *alpha, const void *X, f77_int incX,
                 const void *Y, f77_int incY, void *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_zgerc(enum CBLAS_ORDER order, f77_int M, f77_int N,
                 const void *alpha, const void *X, f77_int incX,
                 const void *Y, f77_int incY, void *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_zher(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, double alpha, const void *X, f77_int incX,
                void *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_zhpr(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, double alpha, const void *X,
                f77_int incX, void *A);
void BLIS_EXPORT_BLAS cblas_zher2(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo, f77_int N,
                const void *alpha, const void *X, f77_int incX,
                const void *Y, f77_int incY, void *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_zhpr2(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo, f77_int N,
                const void *alpha, const void *X, f77_int incX,
                const void *Y, f77_int incY, void *Ap);

/*
 * ===========================================================================
 * Prototypes for level 3 BLAS
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (S, D, C, Z)
 */
void BLIS_EXPORT_BLAS cblas_sgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, f77_int M, f77_int N,
                 f77_int K, float alpha, const float *A,
                 f77_int lda, const float *B, f77_int ldb,
                 float beta, float *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_ssymm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, f77_int M, f77_int N,
                 float alpha, const float *A, f77_int lda,
                 const float *B, f77_int ldb, float beta,
                 float *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_ssyrk(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                 float alpha, const float *A, f77_int lda,
                 float beta, float *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_ssyr2k(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                  enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                  float alpha, const float *A, f77_int lda,
                  const float *B, f77_int ldb, float beta,
                  float *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_strmm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, f77_int M, f77_int N,
                 float alpha, const float *A, f77_int lda,
                 float *B, f77_int ldb);
void BLIS_EXPORT_BLAS cblas_strsm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, f77_int M, f77_int N,
                 float alpha, const float *A, f77_int lda,
                 float *B, f77_int ldb);
/** \addtogroup APIS BLIS Extension API
 *  @{
 */

/** \addtogroup INTERFACE CBLAS INTERFACE
 * \ingroup BLIS Extension API
 *  @{
 */


/**
* sgemmt computes scalar-matrix-matrix product with general matrices. It adds the result to the upper or lower part of scalar-matrix product.
* It accesses and updates a triangular part of the square result matrix.
* The operation is defined as
* C := alpha*Mat(A) * Mat(B) + beta*C,
* where:
* Mat(X) is one of Mat(X) = X, or Mat(X) = \f$X^T\f$, or Mat(X) = \f$X^H\f$,
* alpha and beta are scalars,
* A, B and C are matrices:
* Mat(A) is an nxk matrix,
* Mat(B) is a kxn matrix,
* C is an nxn upper or lower triangular matrix.
*
* @param[in] Order Storage scheme of matrices. CblasRowMajor or CblasColMajor
* @param[in] Uplo Specifies whether the upper or lower triangular part of the array c is used. CblasUpper or CblasLower
* @param[in] TransA Specifies the form of Mat(A) used in the matrix multiplication:
* if transa = CblasNoTrans, then Mat(A) = A;
* if transa = CblasTrans, then Mat(A) =\f$A^T\f$;
* if transa = CblasConjTrans, then Mat(A) = \f$A^H\f$.
* @param[in] TransB Specifies the form of Mat(B) used in the matrix multiplication:
* if transb = CblasNoTrans, then Mat(B) = B;
* if transb = CblasTrans, then Mat(B) = \f$B^T\f$;
* if transb = CblasConjTrans, then Mat(B) = \f$B^H\f$.
* @param[in] N Specifies the order of the matrix C.
* @param[in] K Specifies the number of columns of the matrix Mat(A) and the number of rows of the matrix Mat(B).
* @param[in] alpha Specifies the scalar alpha.
* @param[in] A  The array is float matrix A.
* @param[in] lda Specifies the leading dimension of a
* @param[in] B The array is float matrix B.
* @param[in] ldb Specifies the leading dimension of b
* @param[in] beta Specifies the scalar beta.
* @param[in,out] C The array is float matrix C.
* @param[in] ldc Specifies the leading dimension of c
* @return None
*/
void BLIS_EXPORT_BLAS cblas_sgemmt(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
         enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
         f77_int N, f77_int K, float alpha, const float *A,
                 f77_int lda, const float *B, f77_int ldb,
                 float beta, float *C, f77_int ldc);
/** @}*/
void BLIS_EXPORT_BLAS cblas_dgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, f77_int M, f77_int N,
                 f77_int K, double alpha, const double *A,
                 f77_int lda, const double *B, f77_int ldb,
                 double beta, double *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_dsymm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, f77_int M, f77_int N,
                 double alpha, const double *A, f77_int lda,
                 const double *B, f77_int ldb, double beta,
                 double *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_dsyrk(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                 double alpha, const double *A, f77_int lda,
                 double beta, double *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_dsyr2k(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                  enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                  double alpha, const double *A, f77_int lda,
                  const double *B, f77_int ldb, double beta,
                  double *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_dtrmm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, f77_int M, f77_int N,
                 double alpha, const double *A, f77_int lda,
                 double *B, f77_int ldb);
void BLIS_EXPORT_BLAS cblas_dtrsm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, f77_int M, f77_int N,
                 double alpha, const double *A, f77_int lda,
                 double *B, f77_int ldb);
/** \addtogroup INTERFACE CBLAS INTERFACE
 *  @{
 */

/**
* dgemmt computes scalar-matrix-matrix product with general matrices. It adds the result to the upper or lower part of scalar-matrix product.
* It accesses and updates a triangular part of the square result matrix.
* The operation is defined as
* C := alpha*Mat(A) * Mat(B) + beta*C,
* where:
* Mat(X) is one of Mat(X) = X, or Mat(X) = \f$X^T\f$, or Mat(X) = \f$X^H\f$,
* alpha and beta are scalars,
* A, B and C are matrices:
* Mat(A) is an nxk matrix,
* Mat(B) is a kxn matrix,
* C is an nxn upper or lower triangular matrix.
*
* @param[in] Order Storage scheme of matrices. CblasRowMajor or CblasColMajor
* @param[in] Uplo Specifies whether the upper or lower triangular part of the array c is used. CblasUpper or CblasLower
* @param[in] TransA Specifies the form of Mat(A) used in the matrix multiplication:
* if transa = CblasNoTrans, then Mat(A) = A;
* if transa = CblasTrans, then Mat(A) =\f$A^T\f$;
* if transa = CblasConjTrans, then Mat(A) = \f$A^H\f$.
* @param[in] TransB Specifies the form of Mat(B) used in the matrix multiplication:
* if transb = CblasNoTrans, then Mat(B) = B;
* if transb = CblasTrans, then Mat(B) = \f$B^T\f$;
* if transb = CblasConjTrans, then Mat(B) = \f$B^H\f$.
* @param[in] N Specifies the order of the matrix C.
* @param[in] K Specifies the number of columns of the matrix Mat(A) and the number of rows of the matrix Mat(B).
* @param[in] alpha Specifies the scalar alpha.
* @param[in] A  The array is float matrix A.
* @param[in] lda Specifies the leading dimension of a
* @param[in] B The array is float matrix B.
* @param[in] ldb Specifies the leading dimension of b
* @param[in] beta Specifies the scalar beta.
* @param[in,out] C The array is float matrix C.
* @param[in] ldc Specifies the leading dimension of c
* @return None
*/
void BLIS_EXPORT_BLAS cblas_dgemmt(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
         enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
         f77_int N, f77_int K, double alpha, const double *A,
                 f77_int lda, const double *B, f77_int ldb,
                 double beta, double *C, f77_int ldc);
/** @}*/
void BLIS_EXPORT_BLAS cblas_cgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, f77_int M, f77_int N,
                 f77_int K, const void *alpha, const void *A,
                 f77_int lda, const void *B, f77_int ldb,
                 const void *beta, void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_csymm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, f77_int M, f77_int N,
                 const void *alpha, const void *A, f77_int lda,
                 const void *B, f77_int ldb, const void *beta,
                 void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_csyrk(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                 const void *alpha, const void *A, f77_int lda,
                 const void *beta, void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_csyr2k(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                  enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                  const void *alpha, const void *A, f77_int lda,
                  const void *B, f77_int ldb, const void *beta,
                  void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_ctrmm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, f77_int M, f77_int N,
                 const void *alpha, const void *A, f77_int lda,
                 void *B, f77_int ldb);
void BLIS_EXPORT_BLAS cblas_ctrsm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, f77_int M, f77_int N,
                 const void *alpha, const void *A, f77_int lda,
                 void *B, f77_int ldb);
/** \addtogroup INTERFACE CBLAS INTERFACE
 *  @{
 */

/**
* cgemmt computes scalar-matrix-matrix product with general matrices. It adds the result to the upper or lower part of scalar-matrix product.
* It accesses and updates a triangular part of the square result matrix.
* The operation is defined as
* C := alpha*Mat(A) * Mat(B) + beta*C,
* where:
* Mat(X) is one of Mat(X) = X, or Mat(X) = \f$X^T\f$, or Mat(X) = \f$X^H\f$,
* alpha and beta are scalars,
* A, B and C are matrices:
* Mat(A) is an nxk matrix,
* Mat(B) is a kxn matrix,
* C is an nxn upper or lower triangular matrix.
*
* @param[in] Order Storage scheme of matrices. CblasRowMajor or CblasColMajor
* @param[in] Uplo Specifies whether the upper or lower triangular part of the array c is used. CblasUpper or CblasLower
* @param[in] TransA Specifies the form of Mat(A) used in the matrix multiplication:
* if transa = CblasNoTrans, then Mat(A) = A;
* if transa = CblasTrans, then Mat(A) =\f$A^T\f$;
* if transa = CblasConjTrans, then Mat(A) = \f$A^H\f$.
* @param[in] TransB Specifies the form of Mat(B) used in the matrix multiplication:
* if transb = CblasNoTrans, then Mat(B) = B;
* if transb = CblasTrans, then Mat(B) = \f$B^T\f$;
* if transb = CblasConjTrans, then Mat(B) = \f$B^H\f$.
* @param[in] N Specifies the order of the matrix C.
* @param[in] K Specifies the number of columns of the matrix Mat(A) and the number of rows of the matrix Mat(B).
* @param[in] alpha Specifies the scalar alpha.
* @param[in] A  The array is float matrix A.
* @param[in] lda Specifies the leading dimension of a
* @param[in] B The array is float matrix B.
* @param[in] ldb Specifies the leading dimension of b
* @param[in] beta Specifies the scalar beta.
* @param[in,out] C The array is float matrix C.
* @param[in] ldc Specifies the leading dimension of c
* @return None
*/
void BLIS_EXPORT_BLAS cblas_cgemmt(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
         enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
         f77_int N, f77_int K, const void *alpha, const void *A,
                 f77_int lda, const void *B, f77_int ldb,
                 const void *beta, void *C, f77_int ldc);
/** @}*/
void BLIS_EXPORT_BLAS cblas_zgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, f77_int M, f77_int N,
                 f77_int K, const void *alpha, const void *A,
                 f77_int lda, const void *B, f77_int ldb,
                 const void *beta, void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_zsymm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, f77_int M, f77_int N,
                 const void *alpha, const void *A, f77_int lda,
                 const void *B, f77_int ldb, const void *beta,
                 void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_zsyrk(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                 const void *alpha, const void *A, f77_int lda,
                 const void *beta, void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_zsyr2k(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                  enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                  const void *alpha, const void *A, f77_int lda,
                  const void *B, f77_int ldb, const void *beta,
                  void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_ztrmm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, f77_int M, f77_int N,
                 const void *alpha, const void *A, f77_int lda,
                 void *B, f77_int ldb);
void BLIS_EXPORT_BLAS cblas_ztrsm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, f77_int M, f77_int N,
                 const void *alpha, const void *A, f77_int lda,
                 void *B, f77_int ldb);
/** \addtogroup INTERFACE CBLAS INTERFACE
 *  @{
 */

/**
* zgemmt computes scalar-matrix-matrix product with general matrices. It adds the result to the upper or lower part of scalar-matrix product.
* It accesses and updates a triangular part of the square result matrix.
* The operation is defined as
* C := alpha*Mat(A) * Mat(B) + beta*C,
* where:
* Mat(X) is one of Mat(X) = X, or Mat(X) = \f$X^T\f$, or Mat(X) = \f$X^H\f$,
* alpha and beta are scalars,
* A, B and C are matrices:
* Mat(A) is an nxk matrix,
* Mat(B) is a kxn matrix,
* C is an nxn upper or lower triangular matrix.
*
* @param[in] Order Storage scheme of matrices. CblasRowMajor or CblasColMajor
* @param[in] Uplo Specifies whether the upper or lower triangular part of the array c is used. CblasUpper or CblasLower
* @param[in] TransA Specifies the form of Mat(A) used in the matrix multiplication:
* if transa = CblasNoTrans, then Mat(A) = A;
* if transa = CblasTrans, then Mat(A) =\f$A^T\f$;
* if transa = CblasConjTrans, then Mat(A) = \f$A^H\f$.
* @param[in] TransB Specifies the form of Mat(B) used in the matrix multiplication:
* if transb = CblasNoTrans, then Mat(B) = B;
* if transb = CblasTrans, then Mat(B) = \f$B^T\f$;
* if transb = CblasConjTrans, then Mat(B) = \f$B^H\f$.
* @param[in] N Specifies the order of the matrix C.
* @param[in] K Specifies the number of columns of the matrix Mat(A) and the number of rows of the matrix Mat(B).
* @param[in] alpha Specifies the scalar alpha.
* @param[in] A  The array is float matrix A.
* @param[in] lda Specifies the leading dimension of a
* @param[in] B The array is float matrix B.
* @param[in] ldb Specifies the leading dimension of b
* @param[in] beta Specifies the scalar beta.
* @param[in,out] C The array is float matrix C.
* @param[in] ldc Specifies the leading dimension of c
* @return None
*/
void BLIS_EXPORT_BLAS cblas_zgemmt(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
         enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
         f77_int N, f77_int K, const void *alpha, const void *A,
                 f77_int lda, const void *B, f77_int ldb,
                 const void *beta, void *C, f77_int ldc);
/** @}*/

/*
 * Routines with prefixes C and Z only
 */
void BLIS_EXPORT_BLAS cblas_chemm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, f77_int M, f77_int N,
                 const void *alpha, const void *A, f77_int lda,
                 const void *B, f77_int ldb, const void *beta,
                 void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_cherk(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                 float alpha, const void *A, f77_int lda,
                 float beta, void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_cher2k(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                  enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                  const void *alpha, const void *A, f77_int lda,
                  const void *B, f77_int ldb, float beta,
                  void *C, f77_int ldc);

void BLIS_EXPORT_BLAS cblas_zhemm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, f77_int M, f77_int N,
                 const void *alpha, const void *A, f77_int lda,
                 const void *B, f77_int ldb, const void *beta,
                 void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_zherk(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                 double alpha, const void *A, f77_int lda,
                 double beta, void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_zher2k(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                  enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                  const void *alpha, const void *A, f77_int lda,
                  const void *B, f77_int ldb, double beta,
                  void *C, f77_int ldc);

void BLIS_EXPORT_BLAS cblas_xerbla(f77_int p, const char *rout, const char *form, ...);

/*
 * ===========================================================================
 * Prototypes for extension BLAS routines
 * ===========================================================================
 */

BLIS_EXPORT_BLAS float  cblas_scabs1( const void *z);
BLIS_EXPORT_BLAS double  cblas_dcabs1( const void *z);


/*
 * ===========================================================================
 * BLAS Extension prototypes
 * ===========================================================================
 */

// -- Batch APIs -------
/** \addtogroup INTERFACE CBLAS INTERFACE
 *  @{
 */

/**
 * cblas_sgemm_batch interface resembles the GEMM interface.
 * Arguments are arrays of pointers to matrices and parameters.
 * It batches multiple independent small GEMM operations of fixed or variable sizes into a group
 * and then spawn multiple threads for different GEMM instances within the group.
 *
 * @param[in] Order Storage scheme of matrices. CblasRowMajor or CblasColMajor
 * @param[in] TransA_array Array of pointers, dimension (group_count), specifies the form of Mat( A ) to be used in the matrix multiplication as follows:
 *                     Mat( A ) = A
 *                     Mat( A ) = \f$A^T\f$
 *                     Mat( A ) = \f$A^H\f$
 * @param[in] TransB_array Array of pointers, dimension (group_count), specifies the form of Mat( B ) to be used in the matrix multiplication as follows:
 *                     Mat( B ) = B
 *                     Mat( B ) = \f$B^T\f$
 *                     Mat( B ) = \f$B^H\f$
 * @param[in] M_array Array of pointers, dimension (group_count), each is a number of rows of matrices A and of matrices C.
 * @param[in] N_array Array of pointers, dimension (group_count), each is a number of columns of matrices B and of matrices C.
 * @param[in] K_array Array of pointers, dimension (group_count), each is a number of columns of matrices A and number of rows of matrices B.
 * @param[in] alpha_array Array of pointers, dimension (group_count) each is a scalar alpha for each GEMM.
 * @param[in] A Array of pointers, dimension (group_count), Each is a matrix A of float datatype.
 * @param[in] lda_array Array of pointers, dimension (group_count), each f77_int lda_array specifies the first dimension of matrix A.
 * @param[in] B Array of pointers, dimension (group_count), Each is a matrix B of float datatype.
 * @param[in] ldb_array Array of pointers, dimension (group_count), each f77_int ldb_array specifies the first dimension of matrix B.
 * @param[in] beta_array Array of pointers, dimension (group_count) each is a scalar beta for each GEMM.
 * @param[in,out] C Array of pointers, dimension (group_count), Each is a matrix C of float datatype.
 * @param[in] ldc_array Array of pointers, dimension (group_count), each f77_int ldc_array specifies the first dimension of matrix C.
 * @param[in] group_count group_count specifies total number of groups. Usually it is used for having batch of variable size GEMM. Where each group batches GEMMs of some fixed size.
 * @param[in] group_size Array of pointer, each is number of GEMM to be performed per group(batch).
 * @return None
 */
void BLIS_EXPORT_BLAS cblas_sgemm_batch(enum CBLAS_ORDER Order,
                 enum CBLAS_TRANSPOSE *TransA_array,
                 enum CBLAS_TRANSPOSE *TransB_array,
                 f77_int *M_array, f77_int *N_array,
                 f77_int *K_array, const float *alpha_array, const float **A,
                 f77_int *lda_array, const float **B, f77_int *ldb_array,
                 const float *beta_array, float **C, f77_int *ldc_array,
                 f77_int group_count, f77_int *group_size);

/**
 * cblas_dgemm_batch interface resembles the GEMM interface.
 * Arguments are arrays of pointers to matrices and parameters.
 * It batches multiple independent small GEMM operations of fixed or variable sizes into a group
 * and then spawn multiple threads for different GEMM instances within the group.
 *
 * @param[in] Order Storage scheme of matrices. CblasRowMajor or CblasColMajor
 * @param[in] TransA_array Array of pointers, dimension (group_count), specifies the form of Mat( A ) to be used in the matrix multiplication as follows:
 *                     Mat( A ) = A
 *                     Mat( A ) = \f$A^T\f$
 *                     Mat( A ) = \f$A^H\f$
 * @param[in] TransB_array Array of pointers, dimension (group_count), specifies the form of Mat( B ) to be used in the matrix multiplication as follows:
 *                     Mat( B ) = B
 *                     Mat( B ) = \f$B^T\f$
 *                     Mat( B ) = \f$B^H\f$
 * @param[in] M_array Array of pointers, dimension (group_count), each is a number of rows of matrices A and of matrices C.
 * @param[in] N_array Array of pointers, dimension (group_count), each is a number of columns of matrices B and of matrices C.
 * @param[in] K_array Array of pointers, dimension (group_count), each is a number of columns of matrices A and number of rows of matrices B.
 * @param[in] alpha_array Array of pointers, dimension (group_count) each is a scalar alpha for each GEMM.
 * @param[in] A Array of pointers, dimension (group_count), Each is a matrix A of double datatype.
 * @param[in] lda_array Array of pointers, dimension (group_count), each f77_int lda_array specifies the first dimension of matrix A.
 * @param[in] B Array of pointers, dimension (group_count), Each is a matrix B of double datatype.
 * @param[in] ldb_array Array of pointers, dimension (group_count), each f77_int ldb_array specifies the first dimension of matrix B.
 * @param[in] beta_array Array of pointers, dimension (group_count) each is a scalar beta for each GEMM.
 * @param[in,out] C Array of pointers, dimension (group_count), Each is a matrix C of double datatype.
 * @param[in] ldc_array Array of pointers, dimension (group_count), each f77_int ldc_array specifies the first dimension of matrix C.
 * @param[in] group_count group_count specifies total number of groups. Usually it is used for having batch of variable size GEMM. Where each group batches GEMMs of some fixed size.
 * @param[in] group_size Array of pointer, each is number of GEMM to be performed per group(batch).
 * @return None
 */
void BLIS_EXPORT_BLAS cblas_dgemm_batch(enum CBLAS_ORDER Order,
                 enum CBLAS_TRANSPOSE *TransA_array,
                 enum CBLAS_TRANSPOSE *TransB_array,
                 f77_int *M_array, f77_int *N_array,
                 f77_int *K_array, const double *alpha_array,
                 const double **A,f77_int *lda_array,
                 const double **B, f77_int *ldb_array,
                 const double *beta_array, double **C, f77_int *ldc_array,
                 f77_int group_count, f77_int *group_size);

/**
 * cblas_cgemm_batch interface resembles the GEMM interface.
 * Arguments are arrays of pointers to matrices and parameters.
 * It batches multiple independent small GEMM operations of fixed or variable sizes into a group
 * and then spawn multiple threads for different GEMM instances within the group.
 *
 * @param[in] Order Storage scheme of matrices. CblasRowMajor or CblasColMajor
 * @param[in] TransA_array Array of pointers, dimension (group_count), specifies the form of Mat( A ) to be used in the matrix multiplication as follows:
 *                     Mat( A ) = A
 *                     Mat( A ) = \f$A^T\f$
 *                     Mat( A ) = \f$A^H\f$
 * @param[in] TransB_array Array of pointers, dimension (group_count), specifies the form of Mat( B ) to be used in the matrix multiplication as follows:
 *                     Mat( B ) = B
 *                     Mat( B ) = \f$B^T\f$
 *                     Mat( B ) = \f$B^H\f$
 * @param[in] M_array Array of pointers, dimension (group_count), each is a number of rows of matrices A and of matrices C.
 * @param[in] N_array Array of pointers, dimension (group_count), each is a number of columns of matrices B and of matrices C.
 * @param[in] K_array Array of pointers, dimension (group_count), each is a number of columns of matrices A and number of rows of matrices B.
 * @param[in] alpha_array Array of pointers, dimension (group_count) each is a scalar alpha for each GEMM.
 * @param[in] A Array of pointers, dimension (group_count), Each is a matrix A of scomplex datatype.
 * @param[in] lda_array Array of pointers, dimension (group_count), each f77_int lda_array specifies the first dimension of matrix A.
 * @param[in] B Array of pointers, dimension (group_count), Each is a matrix B of scomplex datatype.
 * @param[in] ldb_array Array of pointers, dimension (group_count), each f77_int ldb_array specifies the first dimension of matrix B.
 * @param[in] beta_array Array of pointers, dimension (group_count) each is a scalar beta for each GEMM.
 * @param[in,out] C Array of pointers, dimension (group_count), Each is a matrix C of scomplex datatype.
 * @param[in] ldc_array Array of pointers, dimension (group_count), each f77_int ldc_array specifies the first dimension of matrix C.
 * @param[in] group_count group_count specifies total number of groups. Usually it is used for having batch of variable size GEMM. Where each group batches GEMMs of some fixed size.
 * @param[in] group_size Array of pointer, each is number of GEMM to be performed per group(batch).
 * @return None
 */

void BLIS_EXPORT_BLAS cblas_cgemm_batch(enum CBLAS_ORDER Order,
                 enum CBLAS_TRANSPOSE *TransA_array,
                 enum CBLAS_TRANSPOSE *TransB_array,
                 f77_int *M_array, f77_int *N_array,
                 f77_int *K_array, const void *alpha_array, const void **A,
                 f77_int *lda_array, const void **B, f77_int *ldb_array,
                 const void *beta_array, void **C, f77_int *ldc_array,
                 f77_int group_count, f77_int *group_size);

 /**
 * cblas_zgemm_batch interface resembles the GEMM interface.
 * Arguments are arrays of pointers to matrices and parameters.
 * It batches multiple independent small GEMM operations of fixed or variable sizes into a group
 * and then spawn multiple threads for different GEMM instances within the group.
 *
 * @param[in] Order Storage scheme of matrices. CblasRowMajor or CblasColMajor
 * @param[in] TransA_array Array of pointers, dimension (group_count), specifies the form of Mat( A ) to be used in the matrix multiplication as follows:
 *                     Mat( A ) = A
 *                     Mat( A ) = \f$A^T\f$
 *                     Mat( A ) = \f$A^H\f$
 * @param[in] TransB_array Array of pointers, dimension (group_count), specifies the form of Mat( B ) to be used in the matrix multiplication as follows:
 *                     Mat( B ) = B
 *                     Mat( B ) = \f$B^T\f$
 *                     Mat( B ) = \f$B^H\f$
 * @param[in] M_array Array of pointers, dimension (group_count), each is a number of rows of matrices A and of matrices C.
 * @param[in] N_array Array of pointers, dimension (group_count), each is a number of columns of matrices B and of matrices C.
 * @param[in] K_array Array of pointers, dimension (group_count), each is a number of columns of matrices A and number of rows of matrices B.
 * @param[in] alpha_array Array of pointers, dimension (group_count) each is a scalar alpha for each GEMM.
 * @param[in] A Array of pointers, dimension (group_count), Each is a matrix A of dcomplex datatype.
 * @param[in] lda_array Array of pointers, dimension (group_count), each f77_int lda_array specifies the first dimension of matrix A.
 * @param[in] B Array of pointers, dimension (group_count), Each is a matrix B of dcomplex datatype.
 * @param[in] ldb_array Array of pointers, dimension (group_count), each f77_int ldb_array specifies the first dimension of matrix B.
 * @param[in] beta_array Array of pointers, dimension (group_count) each is a scalar beta for each GEMM.
 * @param[in,out] C Array of pointers, dimension (group_count), Each is a matrix C of dcomplex datatype.
 * @param[in] ldc_array Array of pointers, dimension (group_count), each f77_int ldc_array specifies the first dimension of matrix C.
 * @param[in] group_count group_count specifies total number of groups. Usually it is used for having batch of variable size GEMM. Where each group batches GEMMs of some fixed size.
 * @param[in] group_size Array of pointer, each is number of GEMM to be performed per group(batch).
 * @return None
 */
void BLIS_EXPORT_BLAS cblas_zgemm_batch(enum CBLAS_ORDER Order,
                 enum CBLAS_TRANSPOSE *TransA_array,
                 enum CBLAS_TRANSPOSE *TransB_array,
                 f77_int *M_array, f77_int *N_array,
                 f77_int *K_array, const void *alpha_array, const void **A,
                 f77_int *lda_array, const void **B, f77_int *ldb_array,
                 const void *beta_array, void **C, f77_int *ldc_array,
                 f77_int group_count, f77_int *group_size);
/** @}*/
void BLIS_EXPORT_BLAS cblas_cgemm3m(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, f77_int M, f77_int N,
                 f77_int K, const void *alpha, const void *A,
                 f77_int lda, const void *B, f77_int ldb,
                 const void *beta, void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_zgemm3m(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, f77_int M, f77_int N,
                 f77_int K, const void *alpha, const void *A,
                 f77_int lda, const void *B, f77_int ldb,
                 const void *beta, void *C, f77_int ldc);

// -- AMIN APIs -------
BLIS_EXPORT_BLAS f77_int cblas_isamin(f77_int N, const float  *X, f77_int incX);
BLIS_EXPORT_BLAS f77_int cblas_idamin(f77_int N, const double *X, f77_int incX);
BLIS_EXPORT_BLAS f77_int cblas_icamin(f77_int N, const void   *X, f77_int incX);
BLIS_EXPORT_BLAS f77_int cblas_izamin(f77_int N, const void   *X, f77_int incX);


// -- PACK COMPUTE APIs --
/** \addtogroup INTERFACE CBLAS INTERFACE
 *  @{
 */

/**
* cblas_sgemm_pack_get_size calculates and returns the number of bytes necessary
* to store the specified matrix after packing.
*
* @param[in] Identifier Specifies the matrix to be packed. CblasAMatrix or CblasBMatrix.
* @param[in] M Specifies the number of rows of the matrix Mat(A) and the number of columns of the matrix Mat(B).
* @param[in] N Specifies the order of the matrix C.
* @param[in] K Specifies the number of columns of the matrix Mat(A) and the number of rows of the matrix Mat(B).
* @return The size in bytes required to store the specified matrix after packing.
*/
BLIS_EXPORT_BLAS f77_int cblas_sgemm_pack_get_size(enum CBLAS_IDENTIFIER Identifier,
                const f77_int M, const f77_int N, const f77_int K);

/**
* cblas_dgemm_pack_get_size calculates and returns the number of bytes necessary
* to store the specified matrix after packing.
*
* @param[in] Identifier Specifies the matrix to be packed. CblasAMatrix or CblasBMatrix.
* @param[in] M Specifies the number of rows of the matrix Mat(A) and the number of columns of the matrix Mat(B).
* @param[in] N Specifies the order of the matrix C.
* @param[in] K Specifies the number of columns of the matrix Mat(A) and the number of rows of the matrix Mat(B).
* @return The size in bytes required to store the specified matrix after packing.
*/
BLIS_EXPORT_BLAS f77_int cblas_dgemm_pack_get_size(enum CBLAS_IDENTIFIER Identifier,
                const f77_int M, const f77_int N, const f77_int K);

/**
* cblas_sgemm_pack scales by alpha and packs the specified matrix into the
* allocated buffer. It is imperative to allocate a buffer of type float and size
* as returned by the cblas_sgemm_pack_get_size() before invoking this routine.
*
* @note If both the matrices are to be packed, the user must ensure that only
* one matrix is packed with the scalar alpha and the other with a unit-scalar.
*
* @param[in] Order Storage scheme of matrices. CblasRowMajor or CblasColMajor.
* @param[in] Identifier Specifies the matrix to be packed. CblasAMatrix or CblasBMatrix.
* @param[in] Trans Specifies the form of Mat(X) used in the matrix multiplication:
* if trans = CblasNoTrans, then Mat(X) = X;
* if trans = CblasTrans, then Mat(X) = \f$X^T\f$;
* if trans = CblasConjTrans, then Mat(X) = \f$X^H\f$.
* @param[in] M Specifies the number of rows of the matrix Mat(A) and the number of columns of the matrix Mat(B).
* @param[in] N Specifies the order of the matrix C.
* @param[in] K Specifies the number of columns of the matrix Mat(A) and the number of rows of the matrix Mat(B).
* @param[in] alpha Specifies the scalar alpha.
* @param[in] src The matrix to be packed.
* @param[in] ld Specifies the leading dimension of the matrix to be packed.
* @param[out] dest The buffer to store the scaled and packed matrix.
* @return None
*/
BLIS_EXPORT_BLAS void cblas_sgemm_pack(enum CBLAS_ORDER Order,
                 enum CBLAS_IDENTIFIER Identifier, enum CBLAS_TRANSPOSE Trans,
                 const f77_int M, const f77_int N, const f77_int K,
                 const float alpha, const float *src, const f77_int ld,
                 float* dest );

/**
* cblas_dgemm_pack scales by alpha and packs the specified matrix into the
* allocated buffer. It is imperative to allocate a buffer of type double and
* size as returned by the cblas_dgemm_pack_get_size() before invoking this
* routine.
*
* @note If both the matrices are to be packed, the user must ensure that only
* one matrix is packed with the scalar alpha and the other with a unit-scalar.
*
* @param[in] Order Storage scheme of matrices. CblasRowMajor or CblasColMajor.
* @param[in] Identifier Specifies the matrix to be packed. CblasAMatrix or CblasBMatrix.
* @param[in] Trans Specifies the form of Mat(X) used in the matrix multiplication:
* if trans = CblasNoTrans, then Mat(X) = X;
* if trans = CblasTrans, then Mat(X) = \f$X^T\f$;
* if trans = CblasConjTrans, then Mat(X) = \f$X^H\f$.
* @param[in] M Specifies the number of rows of the matrix Mat(A) and the number of columns of the matrix Mat(B).
* @param[in] N Specifies the order of the matrix C.
* @param[in] K Specifies the number of columns of the matrix Mat(A) and the number of rows of the matrix Mat(B).
* @param[in] alpha Specifies the scalar alpha.
* @param[in] src The matrix to be packed.
* @param[in] ld Specifies the leading dimension of the matrix to be packed.
* @param[out] dest The buffer to store the scaled and packed matrix.
* @return None
*/
BLIS_EXPORT_BLAS void cblas_dgemm_pack(enum CBLAS_ORDER Order,
                 enum CBLAS_IDENTIFIER Identifier, enum CBLAS_TRANSPOSE Trans,
                 const f77_int M, const f77_int N, const f77_int K,
                 const double alpha, const double *src, const f77_int ld,
                 double* dest );

/**
* cblas_sgemm_compute computes the matrix-matrix product where one or both the
* input matrices are packed and adds this to the scalar-matrix product. This
* operation is defined as:
* C := Mat(A) * Mat(B) + beta*C,
* where,
* Mat(X) is one of Mat(X) = X, or Mat(X) = \f$X^T\f$, or Mat(X) = \f$X^H\f$,
* beta is a scalar,
* A, B and C are matrices:
* Mat(A) is an nxk matrix, or a packed matrix buffer,
* Mat(B) is a kxn matrix, or a packed matrix buffer,
* C is an mxn matrix.
*
* @note In case both the matrices are to be packed, the user must ensure that
* only one matrix is packed with alpha scalar and the other with a unit-scalar,
* during the packing process
*
* @param[in] Order Storage scheme of matrices. CblasRowMajor or CblasColMajor.
* @param[in] TransA Specifies the form of Mat(A) used in the matrix multiplication:
* if transa = CblasNoTrans, then Mat(A) = A;
* if transa = CblasTrans, then Mat(A) = \f$A^T\f$;
* if transa = CblasConjTrans, then Mat(A) = \f$A^H\f$;
* if transa = CblasPacked, then A matrix is packed and lda is ignored.
* @param[in] TransB Specifies the form of Mat(B) used in the matrix multiplication:
* if transb = CblasNoTrans, then Mat(B) = B;
* if transb = CblasTrans, then Mat(B) = \f$B^T\f$;
* if transb = CblasConjTrans, then Mat(B) = \f$B^H\f$;
* if transb = CblasPacked, then B matrix is packed and ldb is ignored.
* @param[in] M Specifies the number of rows of the matrix Mat(A) and the number of columns of the matrix Mat(B).
* @param[in] N Specifies the order of the matrix C.
* @param[in] K Specifies the number of columns of the matrix Mat(A) and the number of rows of the matrix Mat(B).
* @param[in] A  The array is float matrix A or a buffer with packed matrix A.
* @param[in] lda Specifies the leading dimension of A.
* @param[in] B The array is float matrix B or a buffer with packed matrix B.
* @param[in] ldb Specifies the leading dimension of B.
* @param[in] beta Specifies the scalar beta.
* @param[in,out] C The array is float matrix C.
* @param[in] ldc Specifies the leading dimension of C.
* @return None
*/
BLIS_EXPORT_BLAS void cblas_sgemm_compute(enum CBLAS_ORDER Order,
                 f77_int TransA, f77_int TransB,
                 const f77_int M, const f77_int N, const f77_int K,
                 const float* A, f77_int lda, const float* B, f77_int ldb,
                 float beta, float* C, f77_int ldc);

/**
* cblas_dgemm_compute computes the matrix-matrix product where one or both the
* input matrices are packed and adds this to the scalar-matrix product. This
* operation is defined as:
* C := Mat(A) * Mat(B) + beta*C,
* where,
* Mat(X) is one of Mat(X) = X, or Mat(X) = \f$X^T\f$, or Mat(X) = \f$X^H\f$,
* beta is a scalar,
* A, B and C are matrices:
* Mat(A) is an nxk matrix, or a packed matrix buffer,
* Mat(B) is a kxn matrix, or a packed matrix buffer,
* C is an mxn matrix.
*
* @note In case both the matrices are to be packed, the user must ensure that
* only one matrix is packed with alpha scalar and the other with a unit-scalar,
* during the packing process
*
* @param[in] Order Storage scheme of matrices. CblasRowMajor or CblasColMajor.
* @param[in] TransA Specifies the form of Mat(A) used in the matrix multiplication:
* if transa = CblasNoTrans, then Mat(A) = A;
* if transa = CblasTrans, then Mat(A) = \f$A^T\f$;
* if transa = CblasConjTrans, then Mat(A) = \f$A^H\f$;
* if transa = CblasPacked, then A matrix is packed and lda is ignored.
* @param[in] TransB Specifies the form of Mat(B) used in the matrix multiplication:
* if transb = CblasNoTrans, then Mat(B) = B;
* if transb = CblasTrans, then Mat(B) = \f$B^T\f$;
* if transb = CblasConjTrans, then Mat(B) = \f$B^H\f$;
* if transb = CblasPacked, then B matrix is packed and ldb is ignored.
* @param[in] M Specifies the number of rows of the matrix Mat(A) and the number of columns of the matrix Mat(B).
* @param[in] N Specifies the order of the matrix C.
* @param[in] K Specifies the number of columns of the matrix Mat(A) and the number of rows of the matrix Mat(B).
* @param[in] A  The array is double matrix A or a buffer with packed matrix A.
* @param[in] lda Specifies the leading dimension of A.
* @param[in] B The array is double matrix B or a buffer with packed matrix B.
* @param[in] ldb Specifies the leading dimension of B.
* @param[in] beta Specifies the scalar beta.
* @param[in,out] C The array is double matrix C.
* @param[in] ldc Specifies the leading dimension of C.
* @return None
*/
BLIS_EXPORT_BLAS void cblas_dgemm_compute(enum CBLAS_ORDER Order,
                 f77_int TransA, f77_int TransB,
                 const f77_int M, const f77_int N, const f77_int K,
                 const double* A, f77_int lda, const double* B, f77_int ldb,
                 double beta, double* C, f77_int ldc);
/** @}*/

#ifdef __cplusplus
}
#endif
#endif
