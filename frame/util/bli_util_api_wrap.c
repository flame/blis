/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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

// file define different formats of BLAS APIs- uppercase with
// and without underscore, lowercase without underscore.

#include "blis.h"
#include "bli_util_api_wrap.h"

#ifdef BLIS_ENABLE_BLAS

// wrapper functions to support additional symbols
#ifndef BLIS_ENABLE_NO_UNDERSCORE_API
#ifndef BLIS_ENABLE_UPPERCASE_API
void CAXPY(const f77_int *n,const scomplex  *ca,const scomplex  *cx,const f77_int *incx,scomplex  *cy,const f77_int *incy)
{
    caxpy_blis_impl( n, ca, cx, incx, cy, incy);
}

void caxpy(const f77_int *n,const scomplex  *ca,const scomplex  *cx,const f77_int *incx,scomplex  *cy,const f77_int *incy)
{
    caxpy_blis_impl( n, ca, cx, incx, cy, incy);
}

void CAXPY_(const f77_int *n,const scomplex  *ca,const scomplex  *cx,const f77_int *incx,scomplex  *cy,const f77_int *incy)
{
    caxpy_blis_impl( n, ca, cx, incx, cy, incy);
}

void CCOPY(const f77_int *n,const scomplex  *cx,const f77_int *incx,scomplex  *cy,const f77_int *incy)
{
    ccopy_blis_impl( n, cx, incx, cy, incy);
}

void ccopy(const f77_int *n,const scomplex  *cx,const f77_int *incx,scomplex  *cy,const f77_int *incy)
{
    ccopy_blis_impl( n, cx, incx, cy, incy);
}

void CCOPY_(const f77_int *n,const scomplex  *cx,const f77_int *incx,scomplex  *cy,const f77_int *incy)
{
    ccopy_blis_impl( n, cx, incx, cy, incy);
}

#ifdef BLIS_DISABLE_COMPLEX_RETURN_INTEL
scomplex CDOTC(const f77_int* n,const scomplex*   x, const f77_int* incx,const scomplex*   y, const f77_int* incy)
{
    return cdotc_blis_impl ( n, x, incx, y, incy);
}

scomplex cdotc(const f77_int* n,const scomplex*   x, const f77_int* incx,const scomplex*   y, const f77_int* incy)
{
    return cdotc_blis_impl ( n, x, incx, y, incy);
}

scomplex CDOTC_(const f77_int* n,const scomplex*   x, const f77_int* incx,const scomplex*   y, const f77_int* incy)
{
    return cdotc_blis_impl ( n, x, incx, y, incy);
}

scomplex CDOTU(const f77_int* n,const scomplex*   x, const f77_int* incx,const scomplex*   y, const f77_int* incy)
{
    return cdotu_blis_impl ( n, x, incx, y, incy);
}

scomplex cdotu(const f77_int* n,const scomplex*   x, const f77_int* incx,const scomplex*   y, const f77_int* incy)
{
    return cdotu_blis_impl ( n, x, incx, y, incy);
}

scomplex CDOTU_(const f77_int* n,const scomplex*   x, const f77_int* incx,const scomplex*   y, const f77_int* incy)
{
    return cdotu_blis_impl ( n, x, incx, y, incy);
}

dcomplex ZDOTC(const f77_int* n, const dcomplex*   x, const f77_int* incx, const dcomplex*   y, const f77_int* incy)
{
    return zdotc_blis_impl ( n, x, incx, y, incy);
}

dcomplex zdotc(const f77_int* n, const dcomplex*   x, const f77_int* incx, const dcomplex*   y, const f77_int* incy)
{
    return zdotc_blis_impl ( n, x, incx, y, incy);
}

dcomplex ZDOTC_(const f77_int* n, const dcomplex*   x, const f77_int* incx, const dcomplex*   y, const f77_int* incy)
{
    return zdotc_blis_impl ( n, x, incx, y, incy);
}

dcomplex ZDOTU (const f77_int* n, const dcomplex*   x, const f77_int* incx, const dcomplex*   y, const f77_int* incy)
{
    return zdotu_blis_impl ( n, x, incx, y, incy);
}

dcomplex zdotu (const f77_int* n, const dcomplex*   x, const f77_int* incx, const dcomplex*   y, const f77_int* incy)
{
    return zdotu_blis_impl ( n, x, incx, y, incy);
}

dcomplex ZDOTU_(const f77_int* n, const dcomplex*   x, const f77_int* incx, const dcomplex*   y, const f77_int* incy)
{
    return zdotu_blis_impl ( n, x, incx, y, incy);
}
#else
void CDOTC(scomplex* retval,const f77_int *n, const scomplex  *cx, const f77_int *incx, const scomplex  *cy, const f77_int *incy)
{
    cdotc_blis_impl( retval, n, cx, incx, cy, incy);
}

void cdotc(scomplex* retval,const f77_int *n, const scomplex  *cx, const f77_int *incx, const scomplex  *cy, const f77_int *incy)
{
    cdotc_blis_impl( retval, n, cx, incx, cy, incy);
}

void CDOTC_(scomplex* retval,const f77_int *n, const scomplex  *cx, const f77_int *incx, const scomplex  *cy, const f77_int *incy)
{
    cdotc_blis_impl( retval, n, cx, incx, cy, incy);
}

void CDOTU(scomplex* retval,const f77_int *n, const scomplex  *cx, const f77_int *incx, const scomplex  *cy, const f77_int *incy)
{
    cdotu_blis_impl( retval, n, cx, incx, cy, incy);
}

void cdotu(scomplex* retval,const f77_int *n, const scomplex  *cx, const f77_int *incx, const scomplex  *cy, const f77_int *incy)
{
    cdotu_blis_impl( retval, n, cx, incx, cy, incy);
}

void CDOTU_(scomplex* retval,const f77_int *n, const scomplex  *cx, const f77_int *incx, const scomplex  *cy, const f77_int *incy)
{
    cdotu_blis_impl( retval, n, cx, incx, cy, incy);
}

void ZDOTC(dcomplex* retval,const f77_int *n, const dcomplex *zx, const f77_int *incx, const dcomplex *zy, const f77_int *incy)
{
    zdotc_blis_impl( retval, n, zx, incx, zy, incy);
}

void zdotc(dcomplex* retval,const f77_int *n, const dcomplex *zx, const f77_int *incx, const dcomplex *zy, const f77_int *incy)
{
    zdotc_blis_impl( retval, n, zx, incx, zy, incy);
}

void ZDOTC_(dcomplex* retval,const f77_int *n, const dcomplex *zx, const f77_int *incx, const dcomplex *zy, const f77_int *incy)
{
    zdotc_blis_impl( retval, n, zx, incx, zy, incy);
}

void ZDOTU(dcomplex* retval,const f77_int *n, const dcomplex *zx, const f77_int *incx, const dcomplex *zy, const f77_int *incy)
{
    zdotu_blis_impl( retval, n, zx, incx, zy, incy);
}

void zdotu(dcomplex* retval,const f77_int *n, const dcomplex *zx, const f77_int *incx, const dcomplex *zy, const f77_int *incy)
{
    zdotu_blis_impl( retval, n, zx, incx, zy, incy);
}

void ZDOTU_(dcomplex* retval,const f77_int *n, const dcomplex *zx, const f77_int *incx, const dcomplex *zy, const f77_int *incy)
{
    zdotu_blis_impl( retval, n, zx, incx, zy, incy);
}
#endif

void CGBMV(const char   *trans,const f77_int *m,const f77_int *n,const f77_int *kl,const f77_int *ku,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *x,const f77_int *incx,const scomplex  *beta,scomplex  *y,const f77_int *incy)
{
    cgbmv_blis_impl( trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

void cgbmv(const char   *trans,const f77_int *m,const f77_int *n,const f77_int *kl,const f77_int *ku,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *x,const f77_int *incx,const scomplex  *beta,scomplex  *y,const f77_int *incy)
{
    cgbmv_blis_impl( trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

void CGBMV_(const char   *trans,const f77_int *m,const f77_int *n,const f77_int *kl,const f77_int *ku,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *x,const f77_int *incx,const scomplex  *beta,scomplex  *y,const f77_int *incy)
{
    cgbmv_blis_impl( trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

void CGEMM(const char   *transa,const char   *transb,const f77_int *m,const f77_int *n,const f77_int *k,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *b,const f77_int *ldb,const scomplex  *beta,scomplex  *c,const f77_int *ldc)
{
    cgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void cgemm(const char   *transa,const char   *transb,const f77_int *m,const f77_int *n,const f77_int *k,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *b,const f77_int *ldb,const scomplex  *beta,scomplex  *c,const f77_int *ldc)
{
    cgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void CGEMM_(const char   *transa,const char   *transb,const f77_int *m,const f77_int *n,const f77_int *k,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *b,const f77_int *ldb,const scomplex  *beta,scomplex  *c,const f77_int *ldc)
{
    cgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void CGEMV(const char   *trans,const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *x,const f77_int *incx,const scomplex  *beta,scomplex  *y,const f77_int *incy)
{
    cgemv_blis_impl( trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void cgemv(const char   *trans,const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *x,const f77_int *incx,const scomplex  *beta,scomplex  *y,const f77_int *incy)
{
    cgemv_blis_impl( trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void CGEMV_(const char   *trans,const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *x,const f77_int *incx,const scomplex  *beta,scomplex  *y,const f77_int *incy)
{
    cgemv_blis_impl( trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void CGERC(const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *x,const f77_int *incx,const scomplex  *y,const f77_int *incy,scomplex  *a,const f77_int *lda)
{
    cgerc_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void cgerc(const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *x,const f77_int *incx,const scomplex  *y,const f77_int *incy,scomplex  *a,const f77_int *lda)
{
    cgerc_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void CGERC_(const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *x,const f77_int *incx,const scomplex  *y,const f77_int *incy,scomplex  *a,const f77_int *lda)
{
    cgerc_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void CGERU(const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *x,const f77_int *incx,const scomplex  *y,const f77_int *incy,scomplex  *a,const f77_int *lda)
{
    cgeru_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void cgeru(const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *x,const f77_int *incx,const scomplex  *y,const f77_int *incy,scomplex  *a,const f77_int *lda)
{
    cgeru_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void CGERU_(const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *x,const f77_int *incx,const scomplex  *y,const f77_int *incy,scomplex  *a,const f77_int *lda)
{
    cgeru_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void CHBMV(const char   *uplo,const f77_int *n,const f77_int *k,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *x,const f77_int *incx,const scomplex  *beta,scomplex  *y,const f77_int *incy)
{
    chbmv_blis_impl( uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void chbmv(const char   *uplo,const f77_int *n,const f77_int *k,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *x,const f77_int *incx,const scomplex  *beta,scomplex  *y,const f77_int *incy)
{
    chbmv_blis_impl( uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void CHBMV_(const char   *uplo,const f77_int *n,const f77_int *k,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *x,const f77_int *incx,const scomplex  *beta,scomplex  *y,const f77_int *incy)
{
    chbmv_blis_impl( uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void CHEMM(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *b,const f77_int *ldb,const scomplex  *beta,scomplex  *c,const f77_int *ldc)
{
    chemm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void chemm(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *b,const f77_int *ldb,const scomplex  *beta,scomplex  *c,const f77_int *ldc)
{
    chemm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void CHEMM_(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *b,const f77_int *ldb,const scomplex  *beta,scomplex  *c,const f77_int *ldc)
{
    chemm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void CHEMV(const char   *uplo,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *x,const f77_int *incx,const scomplex  *beta,scomplex  *y,const f77_int *incy)
{
    chemv_blis_impl( uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

void chemv(const char   *uplo,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *x,const f77_int *incx,const scomplex  *beta,scomplex  *y,const f77_int *incy)
{
    chemv_blis_impl( uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

void CHEMV_(const char   *uplo,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *x,const f77_int *incx,const scomplex  *beta,scomplex  *y,const f77_int *incy)
{
    chemv_blis_impl( uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

void CHER(const char   *uplo,const f77_int *n,const float  *alpha,const scomplex  *x,const f77_int *incx,scomplex  *a,const f77_int *lda)
{
    cher_blis_impl( uplo, n, alpha, x, incx, a, lda);
}

void cher(const char   *uplo,const f77_int *n,const float  *alpha,const scomplex  *x,const f77_int *incx,scomplex  *a,const f77_int *lda)
{
    cher_blis_impl( uplo, n, alpha, x, incx, a, lda);
}

void CHER_(const char   *uplo,const f77_int *n,const float  *alpha,const scomplex  *x,const f77_int *incx,scomplex  *a,const f77_int *lda)
{
    cher_blis_impl( uplo, n, alpha, x, incx, a, lda);
}

void CHER2(const char   *uplo,const f77_int *n,const scomplex  *alpha,const scomplex  *x,const f77_int *incx,const scomplex  *y,const f77_int *incy,scomplex  *a,const f77_int *lda)
{
    cher2_blis_impl( uplo, n, alpha, x, incx, y, incy, a, lda);
}

void cher2(const char   *uplo,const f77_int *n,const scomplex  *alpha,const scomplex  *x,const f77_int *incx,const scomplex  *y,const f77_int *incy,scomplex  *a,const f77_int *lda)
{
    cher2_blis_impl( uplo, n, alpha, x, incx, y, incy, a, lda);
}

void CHER2_(const char   *uplo,const f77_int *n,const scomplex  *alpha,const scomplex  *x,const f77_int *incx,const scomplex  *y,const f77_int *incy,scomplex  *a,const f77_int *lda)
{
    cher2_blis_impl( uplo, n, alpha, x, incx, y, incy, a, lda);
}

void CHER2K(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *b,const f77_int *ldb,const float  *beta,scomplex  *c,const f77_int *ldc)
{
    cher2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void cher2k(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *b,const f77_int *ldb,const float  *beta,scomplex  *c,const f77_int *ldc)
{
    cher2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void CHER2K_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *b,const f77_int *ldb,const float  *beta,scomplex  *c,const f77_int *ldc)
{
    cher2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void CHERK(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const float  *alpha,const scomplex  *a,const f77_int *lda,const float  *beta,scomplex  *c,const f77_int *ldc)
{
    cherk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void cherk(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const float  *alpha,const scomplex  *a,const f77_int *lda,const float  *beta,scomplex  *c,const f77_int *ldc)
{
    cherk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void CHERK_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const float  *alpha,const scomplex  *a,const f77_int *lda,const float  *beta,scomplex  *c,const f77_int *ldc)
{
    cherk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void CHPMV(const char   *uplo,const f77_int *n,const scomplex  *alpha,const scomplex  *ap,const scomplex  *x,const f77_int *incx,const scomplex  *beta,scomplex  *y,const f77_int *incy)
{
    chpmv_blis_impl( uplo, n, alpha, ap, x, incx, beta, y, incy);
}

void chpmv(const char   *uplo,const f77_int *n,const scomplex  *alpha,const scomplex  *ap,const scomplex  *x,const f77_int *incx,const scomplex  *beta,scomplex  *y,const f77_int *incy)
{
    chpmv_blis_impl( uplo, n, alpha, ap, x, incx, beta, y, incy);
}

void CHPMV_(const char   *uplo,const f77_int *n,const scomplex  *alpha,const scomplex  *ap,const scomplex  *x,const f77_int *incx,const scomplex  *beta,scomplex  *y,const f77_int *incy)
{
    chpmv_blis_impl( uplo, n, alpha, ap, x, incx, beta, y, incy);
}

void CHPR(const char   *uplo,const f77_int *n,const float  *alpha,const scomplex  *x,const f77_int *incx,scomplex  *ap)
{
    chpr_blis_impl( uplo, n, alpha, x, incx, ap);
}

void chpr(const char   *uplo,const f77_int *n,const float  *alpha,const scomplex  *x,const f77_int *incx,scomplex  *ap)
{
    chpr_blis_impl( uplo, n, alpha, x, incx, ap);
}

void CHPR_(const char   *uplo,const f77_int *n,const float  *alpha,const scomplex  *x,const f77_int *incx,scomplex  *ap)
{
    chpr_blis_impl( uplo, n, alpha, x, incx, ap);
}

void CHPR2(const char   *uplo,const f77_int *n,const scomplex  *alpha,const scomplex  *x,const f77_int *incx,const scomplex  *y,const f77_int *incy,scomplex  *ap)
{
    chpr2_blis_impl( uplo, n, alpha, x, incx, y, incy, ap);
}

void chpr2(const char   *uplo,const f77_int *n,const scomplex  *alpha,const scomplex  *x,const f77_int *incx,const scomplex  *y,const f77_int *incy,scomplex  *ap)
{
    chpr2_blis_impl( uplo, n, alpha, x, incx, y, incy, ap);
}

void CHPR2_(const char   *uplo,const f77_int *n,const scomplex  *alpha,const scomplex  *x,const f77_int *incx,const scomplex  *y,const f77_int *incy,scomplex  *ap)
{
    chpr2_blis_impl( uplo, n, alpha, x, incx, y, incy, ap);
}

void CROTG(scomplex  *ca, bla_scomplex  *cb, bla_real  *c,scomplex  *s)
{
    crotg_blis_impl( ca, cb, c, s);
}

void crotg(scomplex  *ca, bla_scomplex  *cb, bla_real  *c,scomplex  *s)
{
    crotg_blis_impl( ca, cb, c, s);
}

void CROTG_(scomplex  *ca, bla_scomplex  *cb, bla_real  *c,scomplex  *s)
{
    crotg_blis_impl( ca, cb, c, s);
}

void CSCAL(const f77_int *n,const scomplex  *ca,scomplex  *cx,const f77_int *incx)
{
    cscal_blis_impl( n, ca, cx, incx);
}

void cscal(const f77_int *n,const scomplex  *ca,scomplex  *cx,const f77_int *incx)
{
    cscal_blis_impl( n, ca, cx, incx);
}

void CSCAL_(const f77_int *n,const scomplex  *ca,scomplex  *cx,const f77_int *incx)
{
    cscal_blis_impl( n, ca, cx, incx);
}

void CSROT(const f77_int *n,scomplex  *cx,const f77_int *incx,scomplex  *cy,const f77_int *incy,const float  *c,const float  *s)
{
    csrot_blis_impl( n, cx, incx, cy, incy, c, s);
}

void csrot(const f77_int *n,scomplex  *cx,const f77_int *incx,scomplex  *cy,const f77_int *incy,const float  *c,const float  *s)
{
    csrot_blis_impl( n, cx, incx, cy, incy, c, s);
}

void CSROT_(const f77_int *n,scomplex  *cx,const f77_int *incx,scomplex  *cy,const f77_int *incy,const float  *c,const float  *s)
{
    csrot_blis_impl( n, cx, incx, cy, incy, c, s);
}

void CSSCAL(const f77_int *n,const float  *sa,scomplex  *cx,const f77_int *incx)
{
    csscal_blis_impl( n, sa, cx, incx);
}

void csscal(const f77_int *n,const float  *sa,scomplex  *cx,const f77_int *incx)
{
    csscal_blis_impl( n, sa, cx, incx);
}

void CSSCAL_(const f77_int *n,const float  *sa,scomplex  *cx,const f77_int *incx)
{
    csscal_blis_impl( n, sa, cx, incx);
}

void CSWAP(const f77_int *n,scomplex  *cx,const f77_int *incx,scomplex  *cy,const f77_int *incy)
{
    cswap_blis_impl( n, cx, incx, cy, incy);
}

void cswap(const f77_int *n,scomplex  *cx,const f77_int *incx,scomplex  *cy,const f77_int *incy)
{
    cswap_blis_impl( n, cx, incx, cy, incy);
}

void CSWAP_(const f77_int *n,scomplex  *cx,const f77_int *incx,scomplex  *cy,const f77_int *incy)
{
    cswap_blis_impl( n, cx, incx, cy, incy);
}

void CSYMM(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *b,const f77_int *ldb,const scomplex  *beta,scomplex  *c,const f77_int *ldc)
{
    csymm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void csymm(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *b,const f77_int *ldb,const scomplex  *beta,scomplex  *c,const f77_int *ldc)
{
    csymm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void CSYMM_(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *b,const f77_int *ldb,const scomplex  *beta,scomplex  *c,const f77_int *ldc)
{
    csymm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void CSYR2K(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *b,const f77_int *ldb,const scomplex  *beta,scomplex  *c,const f77_int *ldc)
{
    csyr2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void csyr2k(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *b,const f77_int *ldb,const scomplex  *beta,scomplex  *c,const f77_int *ldc)
{
    csyr2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void CSYR2K_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *b,const f77_int *ldb,const scomplex  *beta,scomplex  *c,const f77_int *ldc)
{
    csyr2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void CSYRK(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *beta,scomplex  *c,const f77_int *ldc)
{
    csyrk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void csyrk(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *beta,scomplex  *c,const f77_int *ldc)
{
    csyrk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void CSYRK_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *beta,scomplex  *c,const f77_int *ldc)
{
    csyrk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void CTBMV(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const scomplex  *a,const f77_int *lda,scomplex  *x,const f77_int *incx)
{
    ctbmv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void ctbmv(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const scomplex  *a,const f77_int *lda,scomplex  *x,const f77_int *incx)
{
    ctbmv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void CTBMV_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const scomplex  *a,const f77_int *lda,scomplex  *x,const f77_int *incx)
{
    ctbmv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void CTBSV(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const scomplex  *a,const f77_int *lda,scomplex  *x,const f77_int *incx)
{
    ctbsv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void ctbsv(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const scomplex  *a,const f77_int *lda,scomplex  *x,const f77_int *incx)
{
    ctbsv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void CTBSV_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const scomplex  *a,const f77_int *lda,scomplex  *x,const f77_int *incx)
{
    ctbsv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void CTPMV(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const scomplex  *ap,scomplex  *x,const f77_int *incx)
{
    ctpmv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void ctpmv(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const scomplex  *ap,scomplex  *x,const f77_int *incx)
{
    ctpmv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void CTPMV_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const scomplex  *ap,scomplex  *x,const f77_int *incx)
{
    ctpmv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void CTPSV(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const scomplex  *ap,scomplex  *x,const f77_int *incx)
{
    ctpsv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void ctpsv(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const scomplex  *ap,scomplex  *x,const f77_int *incx)
{
    ctpsv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void CTPSV_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const scomplex  *ap,scomplex  *x,const f77_int *incx)
{
    ctpsv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void CTRMM(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,scomplex  *b,const f77_int *ldb)
{
    ctrmm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void ctrmm(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,scomplex  *b,const f77_int *ldb)
{
    ctrmm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void CTRMM_(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,scomplex  *b,const f77_int *ldb)
{
    ctrmm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void CTRMV(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const scomplex  *a,const f77_int *lda,scomplex  *x,const f77_int *incx)
{
    ctrmv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void ctrmv(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const scomplex  *a,const f77_int *lda,scomplex  *x,const f77_int *incx)
{
    ctrmv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void CTRMV_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const scomplex  *a,const f77_int *lda,scomplex  *x,const f77_int *incx)
{
    ctrmv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void CTRSM(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,scomplex  *b,const f77_int *ldb)
{
    ctrsm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void ctrsm(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,scomplex  *b,const f77_int *ldb)
{
    ctrsm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void CTRSM_(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,scomplex  *b,const f77_int *ldb)
{
    ctrsm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void CTRSV(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const scomplex  *a,const f77_int *lda,scomplex *x,const f77_int *incx)
{
    ctrsv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void ctrsv(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const scomplex *a,const f77_int *lda,scomplex *x,const f77_int *incx)
{
    ctrsv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void CTRSV_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const scomplex *a,const f77_int *lda,scomplex *x,const f77_int *incx)
{
    ctrsv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

double DASUM(const f77_int *n,const double *dx,const f77_int *incx)
{
    return dasum_blis_impl( n, dx, incx);
}

double dasum(const f77_int *n,const double *dx,const f77_int *incx)
{
    return dasum_blis_impl( n, dx, incx);
}

double DASUM_(const f77_int *n,const double *dx,const f77_int *incx)
{
    return dasum_blis_impl( n, dx, incx);
}

void DAXPY(const f77_int *n,const double *da,const double *dx,const f77_int *incx,double *dy,const f77_int *incy)
{
    daxpy_blis_impl( n, da, dx, incx, dy, incy);
}

void daxpy(const f77_int *n,const double *da,const double *dx,const f77_int *incx,double *dy,const f77_int *incy)
{
    daxpy_blis_impl( n, da, dx, incx, dy, incy);
}

void DAXPY_(const f77_int *n,const double *da,const double *dx,const f77_int *incx,double *dy,const f77_int *incy)
{
    daxpy_blis_impl( n, da, dx, incx, dy, incy);
}

double DCABS1(bla_dcomplex *z)
{
    return dcabs1_blis_impl( z);
}

double dcabs1(bla_dcomplex *z)
{
    return dcabs1_blis_impl( z);
}

double DCABS1_(bla_dcomplex *z)
{
    return dcabs1_blis_impl( z);
}

void DCOPY(const f77_int *n,const double *dx,const f77_int *incx,double *dy,const f77_int *incy)
{
    dcopy_blis_impl( n, dx, incx, dy, incy);
}

void dcopy(const f77_int *n,const double *dx,const f77_int *incx,double *dy,const f77_int *incy)
{
    dcopy_blis_impl( n, dx, incx, dy, incy);
}

void DCOPY_(const f77_int *n,const double *dx,const f77_int *incx,double *dy,const f77_int *incy)
{
    dcopy_blis_impl( n, dx, incx, dy, incy);
}

double DDOT(const f77_int *n,const double *dx,const f77_int *incx,const double *dy,const f77_int *incy)
{
    return ddot_blis_impl( n, dx, incx, dy, incy);
}

double ddot(const f77_int *n,const double *dx,const f77_int *incx,const double *dy,const f77_int *incy)
{
    return ddot_blis_impl( n, dx, incx, dy, incy);
}

double DDOT_(const f77_int *n,const double *dx,const f77_int *incx,const double *dy,const f77_int *incy)
{
    return ddot_blis_impl( n, dx, incx, dy, incy);
}

void DGBMV(const char   *trans,const f77_int *m,const f77_int *n,const f77_int *kl,const f77_int *ku,const double *alpha,const double *a,const f77_int *lda,const double *x,const f77_int *incx,const double *beta,double *y,const f77_int *incy)
{
    dgbmv_blis_impl( trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

void dgbmv(const char   *trans,const f77_int *m,const f77_int *n,const f77_int *kl,const f77_int *ku,const double *alpha,const double *a,const f77_int *lda,const double *x,const f77_int *incx,const double *beta,double *y,const f77_int *incy)
{
    dgbmv_blis_impl( trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

void DGBMV_(const char   *trans,const f77_int *m,const f77_int *n,const f77_int *kl,const f77_int *ku,const double *alpha,const double *a,const f77_int *lda,const double *x,const f77_int *incx,const double *beta,double *y,const f77_int *incy)
{
    dgbmv_blis_impl( trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

void DGEMM(const char   *transa,const char   *transb,const f77_int *m,const f77_int *n,const f77_int *k,const double *alpha,const double *a,const f77_int *lda,const double *b,const f77_int *ldb,const double *beta,double *c,const f77_int *ldc)
{
    dgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void dgemm(const char   *transa,const char   *transb,const f77_int *m,const f77_int *n,const f77_int *k,const double *alpha,const double *a,const f77_int *lda,const double *b,const f77_int *ldb,const double *beta,double *c,const f77_int *ldc)
{
    dgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void DGEMM_(const char   *transa,const char   *transb,const f77_int *m,const f77_int *n,const f77_int *k,const double *alpha,const double *a,const f77_int *lda,const double *b,const f77_int *ldb,const double *beta,double *c,const f77_int *ldc)
{
    dgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void DZGEMM( const f77_char *transa, const f77_char *transb, const f77_int *m, const f77_int *n, const f77_int *k, const dcomplex *alpha, const double *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc )
{
    dzgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void dzgemm( const f77_char *transa, const f77_char *transb, const f77_int *m, const f77_int *n, const f77_int *k, const dcomplex *alpha, const double *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc )
{
    dzgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void DZGEMM_( const f77_char *transa, const f77_char *transb, const f77_int *m, const f77_int *n, const f77_int *k, const dcomplex *alpha, const double *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc )
{
    dzgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
void DGEMV(const char   *trans,const f77_int *m,const f77_int *n,const double *alpha,const double *a,const f77_int *lda,const double *x,const f77_int *incx,const double *beta,double *y,const f77_int *incy)
{
    dgemv_blis_impl( trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void dgemv(const char   *trans,const f77_int *m,const f77_int *n,const double *alpha,const double *a,const f77_int *lda,const double *x,const f77_int *incx,const double *beta,double *y,const f77_int *incy)
{
    dgemv_blis_impl( trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void DGEMV_(const char   *trans,const f77_int *m,const f77_int *n,const double *alpha,const double *a,const f77_int *lda,const double *x,const f77_int *incx,const double *beta,double *y,const f77_int *incy)
{
    dgemv_blis_impl( trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void DGER(const f77_int *m,const f77_int *n,const double *alpha,const double *x,const f77_int *incx,const double *y,const f77_int *incy,double *a,const f77_int *lda)
{
    dger_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void dger(const f77_int *m,const f77_int *n,const double *alpha,const double *x,const f77_int *incx,const double *y,const f77_int *incy,double *a,const f77_int *lda)
{
    dger_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void DGER_(const f77_int *m,const f77_int *n,const double *alpha,const double *x,const f77_int *incx,const double *y,const f77_int *incy,double *a,const f77_int *lda)
{
    dger_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

double DNRM2(const f77_int *n,const double *x,const f77_int *incx)
{
    return dnrm2_blis_impl( n, x, incx);
}

double dnrm2(const f77_int *n,const double *x,const f77_int *incx)
{
    return dnrm2_blis_impl( n, x, incx);
}

double DNRM2_(const f77_int *n,const double *x,const f77_int *incx)
{
    return dnrm2_blis_impl( n, x, incx);
}

void DROT(const f77_int *n,double *dx,const f77_int *incx,double *dy,const f77_int *incy,const double *c,const double *s)
{
    drot_blis_impl( n, dx, incx, dy, incy, c, s);
}

void drot(const f77_int *n,double *dx,const f77_int *incx,double *dy,const f77_int *incy,const double *c,const double *s)
{
    drot_blis_impl( n, dx, incx, dy, incy, c, s);
}

void DROT_(const f77_int *n,double *dx,const f77_int *incx,double *dy,const f77_int *incy,const double *c,const double *s)
{
    drot_blis_impl( n, dx, incx, dy, incy, c, s);
}

void DROTG(double *da,double *db,double *c,double *s)
{
    drotg_blis_impl( da, db, c, s);
}

void drotg(double *da,double *db,double *c,double *s)
{
    drotg_blis_impl( da, db, c, s);
}

void DROTG_(double *da,double *db,double *c,double *s)
{
    drotg_blis_impl( da, db, c, s);
}

void DROTM(const f77_int *n,double *dx,const f77_int *incx,double *dy,const f77_int *incy,const double *dparam)
{
    drotm_blis_impl( n, dx, incx, dy, incy, dparam);
}

void drotm(const f77_int *n,double *dx,const f77_int *incx,double *dy,const f77_int *incy,const double *dparam)
{
    drotm_blis_impl( n, dx, incx, dy, incy, dparam);
}

void DROTM_(const f77_int *n,double *dx,const f77_int *incx,double *dy,const f77_int *incy,const double *dparam)
{
    drotm_blis_impl( n, dx, incx, dy, incy, dparam);
}

void DROTMG(double *dd1,double *dd2,double *dx1,const double *dy1,double *dparam)
{
    drotmg_blis_impl( dd1, dd2, dx1, dy1, dparam);
}

void drotmg(double *dd1,double *dd2,double *dx1,const double *dy1,double *dparam)
{
    drotmg_blis_impl( dd1, dd2, dx1, dy1, dparam);
}

void DROTMG_(double *dd1,double *dd2,double *dx1,const double *dy1,double *dparam)
{
    drotmg_blis_impl( dd1, dd2, dx1, dy1, dparam);
}

void DSBMV(const char   *uplo,const f77_int *n,const f77_int *k,const double *alpha,const double *a,const f77_int *lda,const double *x,const f77_int *incx,const double *beta,double *y,const f77_int *incy)
{
    dsbmv_blis_impl( uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void dsbmv(const char   *uplo,const f77_int *n,const f77_int *k,const double *alpha,const double *a,const f77_int *lda,const double *x,const f77_int *incx,const double *beta,double *y,const f77_int *incy)
{
    dsbmv_blis_impl( uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void DSBMV_(const char   *uplo,const f77_int *n,const f77_int *k,const double *alpha,const double *a,const f77_int *lda,const double *x,const f77_int *incx,const double *beta,double *y,const f77_int *incy)
{
    dsbmv_blis_impl( uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void DSCAL(const f77_int *n,const double *da,double *dx,const f77_int *incx)
{
    dscal_blis_impl( n, da, dx, incx);
}

void dscal(const f77_int *n,const double *da,double *dx,const f77_int *incx)
{
    dscal_blis_impl( n, da, dx, incx);
}

void DSCAL_(const f77_int *n,const double *da,double *dx,const f77_int *incx)
{
    dscal_blis_impl( n, da, dx, incx);
}

double DSDOT(const f77_int *n,const float  *sx,const f77_int *incx,const float  *sy,const f77_int *incy)
{
    return dsdot_blis_impl( n, sx, incx, sy, incy);
}

double dsdot(const f77_int *n,const float  *sx,const f77_int *incx,const float  *sy,const f77_int *incy)
{
    return dsdot_blis_impl( n, sx, incx, sy, incy);
}

double DSDOT_(const f77_int *n,const float  *sx,const f77_int *incx,const float  *sy,const f77_int *incy)
{
    return dsdot_blis_impl( n, sx, incx, sy, incy);
}

void DSPMV(const char   *uplo,const f77_int *n,const double *alpha,const double *ap,const double *x,const f77_int *incx,const double *beta,double *y,const f77_int *incy)
{
    dspmv_blis_impl( uplo, n, alpha, ap, x, incx, beta, y, incy);
}

void dspmv(const char   *uplo,const f77_int *n,const double *alpha,const double *ap,const double *x,const f77_int *incx,const double *beta,double *y,const f77_int *incy)
{
    dspmv_blis_impl( uplo, n, alpha, ap, x, incx, beta, y, incy);
}

void DSPMV_(const char   *uplo,const f77_int *n,const double *alpha,const double *ap,const double *x,const f77_int *incx,const double *beta,double *y,const f77_int *incy)
{
    dspmv_blis_impl( uplo, n, alpha, ap, x, incx, beta, y, incy);
}

void DSPR(const char   *uplo,const f77_int *n,const double *alpha,const double *x,const f77_int *incx,double *ap)
{
    dspr_blis_impl( uplo, n, alpha, x, incx, ap);
}

void dspr(const char   *uplo,const f77_int *n,const double *alpha,const double *x,const f77_int *incx,double *ap)
{
    dspr_blis_impl( uplo, n, alpha, x, incx, ap);
}

void DSPR_(const char   *uplo,const f77_int *n,const double *alpha,const double *x,const f77_int *incx,double *ap)
{
    dspr_blis_impl( uplo, n, alpha, x, incx, ap);
}

void DSPR2(const char   *uplo,const f77_int *n,const double *alpha,const double *x,const f77_int *incx,const double *y,const f77_int *incy,double *ap)
{
    dspr2_blis_impl( uplo, n, alpha, x, incx, y, incy, ap);
}

void dspr2(const char   *uplo,const f77_int *n,const double *alpha,const double *x,const f77_int *incx,const double *y,const f77_int *incy,double *ap)
{
    dspr2_blis_impl( uplo, n, alpha, x, incx, y, incy, ap);
}

void DSPR2_(const char   *uplo,const f77_int *n,const double *alpha,const double *x,const f77_int *incx,const double *y,const f77_int *incy,double *ap)
{
    dspr2_blis_impl( uplo, n, alpha, x, incx, y, incy, ap);
}

void DSWAP(const f77_int *n,double *dx,const f77_int *incx,double *dy,const f77_int *incy)
{
    dswap_blis_impl( n, dx, incx, dy, incy);
}

void dswap(const f77_int *n,double *dx,const f77_int *incx,double *dy,const f77_int *incy)
{
    dswap_blis_impl( n, dx, incx, dy, incy);
}

void DSWAP_(const f77_int *n,double *dx,const f77_int *incx,double *dy,const f77_int *incy)
{
    dswap_blis_impl( n, dx, incx, dy, incy);
}

void DSYMM(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const double *alpha,const double *a,const f77_int *lda,const double *b,const f77_int *ldb,const double *beta,double *c,const f77_int *ldc)
{
    dsymm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void dsymm(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const double *alpha,const double *a,const f77_int *lda,const double *b,const f77_int *ldb,const double *beta,double *c,const f77_int *ldc)
{
    dsymm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void DSYMM_(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const double *alpha,const double *a,const f77_int *lda,const double *b,const f77_int *ldb,const double *beta,double *c,const f77_int *ldc)
{
    dsymm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void DSYMV(const char   *uplo,const f77_int *n,const double *alpha,const double *a,const f77_int *lda,const double *x,const f77_int *incx,const double *beta,double *y,const f77_int *incy)
{
    dsymv_blis_impl( uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

void dsymv(const char   *uplo,const f77_int *n,const double *alpha,const double *a,const f77_int *lda,const double *x,const f77_int *incx,const double *beta,double *y,const f77_int *incy)
{
    dsymv_blis_impl( uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

void DSYMV_(const char   *uplo,const f77_int *n,const double *alpha,const double *a,const f77_int *lda,const double *x,const f77_int *incx,const double *beta,double *y,const f77_int *incy)
{
    dsymv_blis_impl( uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

void DSYR(const char   *uplo,const f77_int *n,const double *alpha,const double *x,const f77_int *incx,double *a,const f77_int *lda)
{
    dsyr_blis_impl( uplo, n, alpha, x, incx, a, lda);
}

void dsyr(const char   *uplo,const f77_int *n,const double *alpha,const double *x,const f77_int *incx,double *a,const f77_int *lda)
{
    dsyr_blis_impl( uplo, n, alpha, x, incx, a, lda);
}

void DSYR_(const char   *uplo,const f77_int *n,const double *alpha,const double *x,const f77_int *incx,double *a,const f77_int *lda)
{
    dsyr_blis_impl( uplo, n, alpha, x, incx, a, lda);
}

void DSYR2(const char   *uplo,const f77_int *n,const double *alpha,const double *x,const f77_int *incx,const double *y,const f77_int *incy,double *a,const f77_int *lda)
{
    dsyr2_blis_impl( uplo, n, alpha, x, incx, y, incy, a, lda);
}

void dsyr2(const char   *uplo,const f77_int *n,const double *alpha,const double *x,const f77_int *incx,const double *y,const f77_int *incy,double *a,const f77_int *lda)
{
    dsyr2_blis_impl( uplo, n, alpha, x, incx, y, incy, a, lda);
}

void DSYR2_(const char   *uplo,const f77_int *n,const double *alpha,const double *x,const f77_int *incx,const double *y,const f77_int *incy,double *a,const f77_int *lda)
{
    dsyr2_blis_impl( uplo, n, alpha, x, incx, y, incy, a, lda);
}

void DSYR2K(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const double *alpha,const double *a,const f77_int *lda,const double *b,const f77_int *ldb,const double *beta,double *c,const f77_int *ldc)
{
    dsyr2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void dsyr2k(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const double *alpha,const double *a,const f77_int *lda,const double *b,const f77_int *ldb,const double *beta,double *c,const f77_int *ldc)
{
    dsyr2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void DSYR2K_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const double *alpha,const double *a,const f77_int *lda,const double *b,const f77_int *ldb,const double *beta,double *c,const f77_int *ldc)
{
    dsyr2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void DSYRK(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const double *alpha,const double *a,const f77_int *lda,const double *beta,double *c,const f77_int *ldc)
{
    dsyrk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void dsyrk(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const double *alpha,const double *a,const f77_int *lda,const double *beta,double *c,const f77_int *ldc)
{
    dsyrk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void DSYRK_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const double *alpha,const double *a,const f77_int *lda,const double *beta,double *c,const f77_int *ldc)
{
    dsyrk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void DTBMV(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const double *a,const f77_int *lda,double *x,const f77_int *incx)
{
    dtbmv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void dtbmv(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const double *a,const f77_int *lda,double *x,const f77_int *incx)
{
    dtbmv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void DTBMV_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const double *a,const f77_int *lda,double *x,const f77_int *incx)
{
    dtbmv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void DTBSV(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const double *a,const f77_int *lda,double *x,const f77_int *incx)
{
    dtbsv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void dtbsv(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const double *a,const f77_int *lda,double *x,const f77_int *incx)
{
    dtbsv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void DTBSV_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const double *a,const f77_int *lda,double *x,const f77_int *incx)
{
    dtbsv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void DTPMV(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const double *ap,double *x,const f77_int *incx)
{
    dtpmv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void dtpmv(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const double *ap,double *x,const f77_int *incx)
{
    dtpmv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void DTPMV_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const double *ap,double *x,const f77_int *incx)
{
    dtpmv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void DTPSV(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const double *ap,double *x,const f77_int *incx)
{
    dtpsv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void dtpsv(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const double *ap,double *x,const f77_int *incx)
{
    dtpsv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void DTPSV_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const double *ap,double *x,const f77_int *incx)
{
    dtpsv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void DTRMM(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const double *alpha,const double *a,const f77_int *lda,double *b,const f77_int *ldb)
{
    dtrmm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void dtrmm(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const double *alpha,const double *a,const f77_int *lda,double *b,const f77_int *ldb)
{
    dtrmm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void DTRMM_(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const double *alpha,const double *a,const f77_int *lda,double *b,const f77_int *ldb)
{
    dtrmm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void DTRMV(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const double *a,const f77_int *lda,double *x,const f77_int *incx)
{
    dtrmv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void dtrmv(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const double *a,const f77_int *lda,double *x,const f77_int *incx)
{
    dtrmv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void DTRMV_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const double *a,const f77_int *lda,double *x,const f77_int *incx)
{
    dtrmv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void DTRSM(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const double *alpha,const double *a,const f77_int *lda,double *b,const f77_int *ldb)
{
    dtrsm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void dtrsm(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const double *alpha,const double *a,const f77_int *lda,double *b,const f77_int *ldb)
{
    dtrsm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void DTRSM_(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const double *alpha,const double *a,const f77_int *lda,double *b,const f77_int *ldb)
{
    dtrsm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void DTRSV(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const double *a,const f77_int *lda,double *x,const f77_int *incx)
{
    dtrsv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void dtrsv(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const double *a,const f77_int *lda,double *x,const f77_int *incx)
{
    dtrsv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void DTRSV_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const double *a,const f77_int *lda,double *x,const f77_int *incx)
{
    dtrsv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

double DZASUM(const f77_int *n,const dcomplex *zx,const f77_int *incx)
{
    return dzasum_blis_impl( n, zx, incx);
}

double dzasum(const f77_int *n,const dcomplex *zx,const f77_int *incx)
{
    return dzasum_blis_impl( n, zx, incx);
}

double DZASUM_(const f77_int *n,const dcomplex *zx,const f77_int *incx)
{
    return dzasum_blis_impl( n, zx, incx);
}

double DZNRM2(const f77_int *n,const dcomplex *x,const f77_int *incx)
{
    return dznrm2_blis_impl( n, x, incx);
}

double dznrm2(const f77_int *n,const dcomplex *x,const f77_int *incx)
{
    return dznrm2_blis_impl( n, x, incx);
}

double DZNRM2_(const f77_int *n,const dcomplex *x,const f77_int *incx)
{
    return dznrm2_blis_impl( n, x, incx);
}

f77_int ICAMAX(const f77_int *n,const scomplex  *cx,const f77_int *incx)
{
    return icamax_blis_impl( n, cx, incx);
}

f77_int icamax(const f77_int *n,const scomplex  *cx,const f77_int *incx)
{
    return icamax_blis_impl( n, cx, incx);
}

f77_int ICAMAX_(const f77_int *n,const scomplex  *cx,const f77_int *incx)
{
    return icamax_blis_impl( n, cx, incx);
}

f77_int IDAMAX(const f77_int *n,const double *dx,const f77_int *incx)
{
    return idamax_blis_impl( n, dx, incx);
}

f77_int idamax(const f77_int *n,const double *dx,const f77_int *incx)
{
    return idamax_blis_impl( n, dx, incx);
}

f77_int IDAMAX_(const f77_int *n,const double *dx,const f77_int *incx)
{
    return idamax_blis_impl( n, dx, incx);
}

f77_int ISAMAX(const f77_int *n,const float  *sx,const f77_int *incx)
{
    return isamax_blis_impl( n, sx, incx);
}

f77_int isamax(const f77_int *n,const float  *sx,const f77_int *incx)
{
    return isamax_blis_impl( n, sx, incx);
}

f77_int ISAMAX_(const f77_int *n,const float  *sx,const f77_int *incx)
{
    return isamax_blis_impl( n, sx, incx);
}

f77_int IZAMAX(const f77_int *n,const dcomplex *zx,const f77_int *incx)
{
    return izamax_blis_impl( n, zx, incx);
}

f77_int izamax(const f77_int *n,const dcomplex *zx,const f77_int *incx)
{
    return izamax_blis_impl( n, zx, incx);
}

f77_int IZAMAX_(const f77_int *n,const dcomplex *zx,const f77_int *incx)
{
    return izamax_blis_impl( n, zx, incx);
}

f77_int LSAME(const char   *ca,const char   *cb,const f77_int a,const f77_int b)
{
    return lsame_blis_impl( ca, cb, a, b);
}

f77_int LSAME_(const char   *ca,const char   *cb,const f77_int a,const f77_int b)
{
    return lsame_blis_impl( ca, cb, a, b);
}

f77_int lsame(const char   *ca,const char   *cb,const f77_int a,const f77_int b)
{
    return lsame_blis_impl( ca, cb, a, b);
}

float SASUM(const f77_int *n,const float  *sx, const f77_int *incx)
{
    return sasum_blis_impl( n, sx, incx);
}

float sasum(const f77_int *n,const float  *sx, const f77_int *incx)
{
    return sasum_blis_impl( n, sx, incx);
}

float SASUM_(const f77_int *n,const float  *sx, const f77_int *incx)
{
    return sasum_blis_impl( n, sx, incx);
}

void SAXPY(const f77_int *n,const float  *sa,const float  *sx,const f77_int *incx,float  *sy,const f77_int *incy)
{
    saxpy_blis_impl( n, sa, sx, incx, sy, incy);
}

void saxpy(const f77_int *n,const float  *sa,const float  *sx,const f77_int *incx,float  *sy,const f77_int *incy)
{
    saxpy_blis_impl( n, sa, sx, incx, sy, incy);
}

void SAXPY_(const f77_int *n,const float  *sa,const float  *sx,const f77_int *incx,float  *sy,const f77_int *incy)
{
    saxpy_blis_impl( n, sa, sx, incx, sy, incy);
}


float SCASUM(const f77_int *n,const scomplex  *cx, const f77_int *incx)
{
    return scasum_blis_impl( n, cx, incx);
}

float scasum(const f77_int *n,const scomplex  *cx, const f77_int *incx)
{
    return scasum_blis_impl( n, cx, incx);
}

float SCASUM_(const f77_int *n,const scomplex  *cx, const f77_int *incx)
{
    return scasum_blis_impl( n, cx, incx);
}



float SCNRM2(const f77_int *n,const scomplex  *x, const f77_int *incx)
{
    return scnrm2_blis_impl( n, x, incx);
}

float scnrm2(const f77_int *n,const scomplex  *x, const f77_int *incx)
{
    return scnrm2_blis_impl( n, x, incx);
}

float SCNRM2_(const f77_int *n,const scomplex  *x, const f77_int *incx)
{
    return scnrm2_blis_impl( n, x, incx);
}


void SCOPY(const f77_int *n,const float  *sx,const f77_int *incx,float  *sy,const f77_int *incy)
{
    scopy_blis_impl( n, sx, incx, sy, incy);
}

void scopy(const f77_int *n,const float  *sx,const f77_int *incx,float  *sy,const f77_int *incy)
{
    scopy_blis_impl( n, sx, incx, sy, incy);
}

void SCOPY_(const f77_int *n,const float  *sx,const f77_int *incx,float  *sy,const f77_int *incy)
{
    scopy_blis_impl( n, sx, incx, sy, incy);
}


float SDOT(const f77_int *n,const float  *sx, const f77_int *incx, const float  *sy, const f77_int *incy)
{
    return sdot_blis_impl( n, sx, incx, sy, incy);
}

float sdot(const f77_int *n,const float  *sx, const f77_int *incx, const float  *sy, const f77_int *incy)
{
    return sdot_blis_impl( n, sx, incx, sy, incy);
}

float SDOT_(const f77_int *n,const float  *sx, const f77_int *incx, const float  *sy, const f77_int *incy)
{
    return sdot_blis_impl( n, sx, incx, sy, incy);
}


float SDSDOT(const f77_int *n,const float  *sb, const float  *sx, const f77_int *incx, const float  *sy, const f77_int *incy)
{
    return sdsdot_blis_impl( n, sb, sx, incx, sy, incy);
}

float sdsdot(const f77_int *n,const float  *sb, const float  *sx, const f77_int *incx, const float  *sy, const f77_int *incy)
{
    return sdsdot_blis_impl( n, sb, sx, incx, sy, incy);
}

float SDSDOT_(const f77_int *n,const float  *sb, const float  *sx, const f77_int *incx, const float  *sy, const f77_int *incy)
{
    return sdsdot_blis_impl( n, sb, sx, incx, sy, incy);
}


void SGBMV(const char   *trans,const f77_int *m,const f77_int *n,const f77_int *kl,const f77_int *ku,const float  *alpha,const float  *a,const f77_int *lda,const float  *x,const f77_int *incx,const float  *beta,float  *y,const f77_int *incy)
{
    sgbmv_blis_impl( trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

void sgbmv(const char   *trans,const f77_int *m,const f77_int *n,const f77_int *kl,const f77_int *ku,const float  *alpha,const float  *a,const f77_int *lda,const float  *x,const f77_int *incx,const float  *beta,float  *y,const f77_int *incy)
{
    sgbmv_blis_impl( trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

void SGBMV_(const char   *trans,const f77_int *m,const f77_int *n,const f77_int *kl,const f77_int *ku,const float  *alpha,const float  *a,const f77_int *lda,const float  *x,const f77_int *incx,const float  *beta,float  *y,const f77_int *incy)
{
    sgbmv_blis_impl( trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

void SGEMM(const char   *transa,const char   *transb,const f77_int *m,const f77_int *n,const f77_int *k,const float  *alpha,const float  *a,const f77_int *lda,const float  *b,const f77_int *ldb,const float  *beta,float  *c,const f77_int *ldc)
{
    sgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void sgemm(const char   *transa,const char   *transb,const f77_int *m,const f77_int *n,const f77_int *k,const float  *alpha,const float  *a,const f77_int *lda,const float  *b,const f77_int *ldb,const float  *beta,float  *c,const f77_int *ldc)
{
    sgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void SGEMM_(const char   *transa,const char   *transb,const f77_int *m,const f77_int *n,const f77_int *k,const float  *alpha,const float  *a,const f77_int *lda,const float  *b,const f77_int *ldb,const float  *beta,float  *c,const f77_int *ldc)
{
    sgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void SGEMV(const char   *trans,const f77_int *m,const f77_int *n,const float  *alpha,const float  *a,const f77_int *lda,const float  *x,const f77_int *incx,const float  *beta,float  *y,const f77_int *incy)
{
    sgemv_blis_impl( trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void sgemv(const char   *trans,const f77_int *m,const f77_int *n,const float  *alpha,const float  *a,const f77_int *lda,const float  *x,const f77_int *incx,const float  *beta,float  *y,const f77_int *incy)
{
    sgemv_blis_impl( trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void SGEMV_(const char   *trans,const f77_int *m,const f77_int *n,const float  *alpha,const float  *a,const f77_int *lda,const float  *x,const f77_int *incx,const float  *beta,float  *y,const f77_int *incy)
{
    sgemv_blis_impl( trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void SGER(const f77_int *m,const f77_int *n,const float  *alpha,const float  *x,const f77_int *incx,const float  *y,const f77_int *incy,float  *a,const f77_int *lda)
{
    sger_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void sger(const f77_int *m,const f77_int *n,const float  *alpha,const float  *x,const f77_int *incx,const float  *y,const f77_int *incy,float  *a,const f77_int *lda)
{
    sger_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void SGER_(const f77_int *m,const f77_int *n,const float  *alpha,const float  *x,const f77_int *incx,const float  *y,const f77_int *incy,float  *a,const f77_int *lda)
{
    sger_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}


float SNRM2(const f77_int *n,const float  *x, const f77_int *incx)
{
    return snrm2_blis_impl( n, x, incx);
}

float snrm2(const f77_int *n,const float  *x, const f77_int *incx)
{
    return snrm2_blis_impl( n, x, incx);
}

float SNRM2_(const f77_int *n,const float  *x, const f77_int *incx)
{
    return snrm2_blis_impl( n, x, incx);
}


void SROT(const f77_int *n,float  *sx,const f77_int *incx,float  *sy,const f77_int *incy,const float  *c,const float  *s)
{
    srot_blis_impl( n, sx, incx, sy, incy, c, s);
}

void srot(const f77_int *n,float  *sx,const f77_int *incx,float  *sy,const f77_int *incy,const float  *c,const float  *s)
{
    srot_blis_impl( n, sx, incx, sy, incy, c, s);
}

void SROT_(const f77_int *n,float  *sx,const f77_int *incx,float  *sy,const f77_int *incy,const float  *c,const float  *s)
{
    srot_blis_impl( n, sx, incx, sy, incy, c, s);
}

void SROTG(float  *sa,float  *sb,float  *c,float  *s)
{
    srotg_blis_impl( sa, sb, c, s);
}

void srotg(float  *sa,float  *sb,float  *c,float  *s)
{
    srotg_blis_impl( sa, sb, c, s);
}

void SROTG_(float  *sa,float  *sb,float  *c,float  *s)
{
    srotg_blis_impl( sa, sb, c, s);
}

void SROTM(const f77_int *n,float  *sx,const f77_int *incx,float  *sy,const f77_int *incy,const float  *sparam)
{
    srotm_blis_impl( n, sx, incx, sy, incy, sparam);
}

void srotm(const f77_int *n,float  *sx,const f77_int *incx,float  *sy,const f77_int *incy,const float  *sparam)
{
    srotm_blis_impl( n, sx, incx, sy, incy, sparam);
}

void SROTM_(const f77_int *n,float  *sx,const f77_int *incx,float  *sy,const f77_int *incy,const float  *sparam)
{
    srotm_blis_impl( n, sx, incx, sy, incy, sparam);
}

void SROTMG(float  *sd1,float  *sd2,float  *sx1,const float  *sy1,float  *sparam)
{
    srotmg_blis_impl( sd1, sd2, sx1, sy1, sparam);
}

void srotmg(float  *sd1,float  *sd2,float  *sx1,const float  *sy1,float  *sparam)
{
    srotmg_blis_impl( sd1, sd2, sx1, sy1, sparam);
}

void SROTMG_(float  *sd1,float  *sd2,float  *sx1,const float  *sy1,float  *sparam)
{
    srotmg_blis_impl( sd1, sd2, sx1, sy1, sparam);
}

void SSBMV(const char   *uplo,const f77_int *n,const f77_int *k,const float  *alpha,const float  *a,const f77_int *lda,const float  *x,const f77_int *incx,const float  *beta,float  *y,const f77_int *incy)
{
    ssbmv_blis_impl( uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void ssbmv(const char   *uplo,const f77_int *n,const f77_int *k,const float  *alpha,const float  *a,const f77_int *lda,const float  *x,const f77_int *incx,const float  *beta,float  *y,const f77_int *incy)
{
    ssbmv_blis_impl( uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void SSBMV_(const char   *uplo,const f77_int *n,const f77_int *k,const float  *alpha,const float  *a,const f77_int *lda,const float  *x,const f77_int *incx,const float  *beta,float  *y,const f77_int *incy)
{
    ssbmv_blis_impl( uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void SSCAL(const f77_int *n,const float  *sa,float  *sx,const f77_int *incx)
{
    sscal_blis_impl( n, sa, sx, incx);
}

void sscal(const f77_int *n,const float  *sa,float  *sx,const f77_int *incx)
{
    sscal_blis_impl( n, sa, sx, incx);
}

void SSCAL_(const f77_int *n,const float  *sa,float  *sx,const f77_int *incx)
{
    sscal_blis_impl( n, sa, sx, incx);
}

void SSPMV(const char   *uplo,const f77_int *n,const float  *alpha,const float  *ap,const float  *x,const f77_int *incx,const float  *beta,float  *y,const f77_int *incy)
{
    sspmv_blis_impl( uplo, n, alpha, ap, x, incx, beta, y, incy);
}

void sspmv(const char   *uplo,const f77_int *n,const float  *alpha,const float  *ap,const float  *x,const f77_int *incx,const float  *beta,float  *y,const f77_int *incy)
{
    sspmv_blis_impl( uplo, n, alpha, ap, x, incx, beta, y, incy);
}

void SSPMV_(const char   *uplo,const f77_int *n,const float  *alpha,const float  *ap,const float  *x,const f77_int *incx,const float  *beta,float  *y,const f77_int *incy)
{
    sspmv_blis_impl( uplo, n, alpha, ap, x, incx, beta, y, incy);
}

void SSPR(const char   *uplo,const f77_int *n,const float  *alpha,const float  *x,const f77_int *incx,float  *ap)
{
    sspr_blis_impl( uplo, n, alpha, x, incx, ap);
}

void sspr(const char   *uplo,const f77_int *n,const float  *alpha,const float  *x,const f77_int *incx,float  *ap)
{
    sspr_blis_impl( uplo, n, alpha, x, incx, ap);
}

void SSPR_(const char   *uplo,const f77_int *n,const float  *alpha,const float  *x,const f77_int *incx,float  *ap)
{
    sspr_blis_impl( uplo, n, alpha, x, incx, ap);
}

void SSPR2(const char   *uplo,const f77_int *n,const float  *alpha,const float  *x,const f77_int *incx,const float  *y,const f77_int *incy,float  *ap)
{
    sspr2_blis_impl( uplo, n, alpha, x, incx, y, incy, ap);
}

void sspr2(const char   *uplo,const f77_int *n,const float  *alpha,const float  *x,const f77_int *incx,const float  *y,const f77_int *incy,float  *ap)
{
    sspr2_blis_impl( uplo, n, alpha, x, incx, y, incy, ap);
}

void SSPR2_(const char   *uplo,const f77_int *n,const float  *alpha,const float  *x,const f77_int *incx,const float  *y,const f77_int *incy,float  *ap)
{
    sspr2_blis_impl( uplo, n, alpha, x, incx, y, incy, ap);
}

void SSWAP(const f77_int *n,float  *sx,const f77_int *incx,float  *sy,const f77_int *incy)
{
    sswap_blis_impl( n, sx, incx, sy, incy);
}

void sswap(const f77_int *n,float  *sx,const f77_int *incx,float  *sy,const f77_int *incy)
{
    sswap_blis_impl( n, sx, incx, sy, incy);
}

void SSWAP_(const f77_int *n,float  *sx,const f77_int *incx,float  *sy,const f77_int *incy)
{
    sswap_blis_impl( n, sx, incx, sy, incy);
}

void SSYMM(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const float  *alpha,const float  *a,const f77_int *lda,const float  *b,const f77_int *ldb,const float  *beta,float  *c,const f77_int *ldc)
{
    ssymm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ssymm(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const float  *alpha,const float  *a,const f77_int *lda,const float  *b,const f77_int *ldb,const float  *beta,float  *c,const f77_int *ldc)
{
    ssymm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void SSYMM_(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const float  *alpha,const float  *a,const f77_int *lda,const float  *b,const f77_int *ldb,const float  *beta,float  *c,const f77_int *ldc)
{
    ssymm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void SSYMV(const char   *uplo,const f77_int *n,const float  *alpha,const float  *a,const f77_int *lda,const float  *x,const f77_int *incx,const float  *beta,float  *y,const f77_int *incy)
{
    ssymv_blis_impl( uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

void ssymv(const char   *uplo,const f77_int *n,const float  *alpha,const float  *a,const f77_int *lda,const float  *x,const f77_int *incx,const float  *beta,float  *y,const f77_int *incy)
{
    ssymv_blis_impl( uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

void SSYMV_(const char   *uplo,const f77_int *n,const float  *alpha,const float  *a,const f77_int *lda,const float  *x,const f77_int *incx,const float  *beta,float  *y,const f77_int *incy)
{
    ssymv_blis_impl( uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

void SSYR(const char   *uplo,const f77_int *n,const float  *alpha,const float  *x,const f77_int *incx,float  *a,const f77_int *lda)
{
    ssyr_blis_impl( uplo, n, alpha, x, incx, a, lda);
}

void ssyr(const char   *uplo,const f77_int *n,const float  *alpha,const float  *x,const f77_int *incx,float  *a,const f77_int *lda)
{
    ssyr_blis_impl( uplo, n, alpha, x, incx, a, lda);
}

void SSYR_(const char   *uplo,const f77_int *n,const float  *alpha,const float  *x,const f77_int *incx,float  *a,const f77_int *lda)
{
    ssyr_blis_impl( uplo, n, alpha, x, incx, a, lda);
}

void SSYR2(const char   *uplo,const f77_int *n,const float  *alpha,const float  *x,const f77_int *incx,const float  *y,const f77_int *incy,float  *a,const f77_int *lda)
{
    ssyr2_blis_impl( uplo, n, alpha, x, incx, y, incy, a, lda);
}

void ssyr2(const char   *uplo,const f77_int *n,const float  *alpha,const float  *x,const f77_int *incx,const float  *y,const f77_int *incy,float  *a,const f77_int *lda)
{
    ssyr2_blis_impl( uplo, n, alpha, x, incx, y, incy, a, lda);
}

void SSYR2_(const char   *uplo,const f77_int *n,const float  *alpha,const float  *x,const f77_int *incx,const float  *y,const f77_int *incy,float  *a,const f77_int *lda)
{
    ssyr2_blis_impl( uplo, n, alpha, x, incx, y, incy, a, lda);
}

void SSYR2K(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const float  *alpha,const float  *a,const f77_int *lda,const float  *b,const f77_int *ldb,const float  *beta,float  *c,const f77_int *ldc)
{
    ssyr2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ssyr2k(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const float  *alpha,const float  *a,const f77_int *lda,const float  *b,const f77_int *ldb,const float  *beta,float  *c,const f77_int *ldc)
{
    ssyr2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void SSYR2K_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const float  *alpha,const float  *a,const f77_int *lda,const float  *b,const f77_int *ldb,const float  *beta,float  *c,const f77_int *ldc)
{
    ssyr2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void SSYRK(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const float  *alpha,const float  *a,const f77_int *lda,const float  *beta,float  *c,const f77_int *ldc)
{
    ssyrk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void ssyrk(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const float  *alpha,const float  *a,const f77_int *lda,const float  *beta,float  *c,const f77_int *ldc)
{
    ssyrk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void SSYRK_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const float  *alpha,const float  *a,const f77_int *lda,const float  *beta,float  *c,const f77_int *ldc)
{
    ssyrk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void STBMV(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const float  *a,const f77_int *lda,float  *x,const f77_int *incx)
{
    stbmv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void stbmv(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const float  *a,const f77_int *lda,float  *x,const f77_int *incx)
{
    stbmv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void STBMV_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const float  *a,const f77_int *lda,float  *x,const f77_int *incx)
{
    stbmv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void STBSV(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const float  *a,const f77_int *lda,float  *x,const f77_int *incx)
{
    stbsv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void stbsv(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const float  *a,const f77_int *lda,float  *x,const f77_int *incx)
{
    stbsv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void STBSV_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const float  *a,const f77_int *lda,float  *x,const f77_int *incx)
{
    stbsv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void STPMV(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const float  *ap,float  *x,const f77_int *incx)
{
    stpmv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void stpmv(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const float  *ap,float  *x,const f77_int *incx)
{
    stpmv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void STPMV_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const float  *ap,float  *x,const f77_int *incx)
{
    stpmv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void STPSV(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const float  *ap,float  *x,const f77_int *incx)
{
    stpsv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void stpsv(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const float  *ap,float  *x,const f77_int *incx)
{
    stpsv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void STPSV_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const float  *ap,float  *x,const f77_int *incx)
{
    stpsv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void STRMM(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const float  *alpha,const float  *a,const f77_int *lda,float  *b,const f77_int *ldb)
{
    strmm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void strmm(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const float  *alpha,const float  *a,const f77_int *lda,float  *b,const f77_int *ldb)
{
    strmm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void STRMM_(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const float  *alpha,const float  *a,const f77_int *lda,float  *b,const f77_int *ldb)
{
    strmm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void STRMV(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const float  *a,const f77_int *lda,float  *x,const f77_int *incx)
{
    strmv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void strmv(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const float  *a,const f77_int *lda,float  *x,const f77_int *incx)
{
    strmv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void STRMV_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const float  *a,const f77_int *lda,float  *x,const f77_int *incx)
{
    strmv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void STRSM(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const float  *alpha,const float  *a,const f77_int *lda,float  *b,const f77_int *ldb)
{
    strsm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void strsm(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const float  *alpha,const float  *a,const f77_int *lda,float  *b,const f77_int *ldb)
{
    strsm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void STRSM_(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const float  *alpha,const float  *a,const f77_int *lda,float  *b,const f77_int *ldb)
{
    strsm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void STRSV(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const float  *a,const f77_int *lda,float  *x,const f77_int *incx)
{
    strsv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void strsv(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const float  *a,const f77_int *lda,float  *x,const f77_int *incx)
{
    strsv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void STRSV_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const float  *a,const f77_int *lda,float  *x,const f77_int *incx)
{
    strsv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

int XERBLA(const char   *srname,const f77_int *info, ftnlen n)
{
    return xerbla_blis_impl( srname, info, n);
}

int XERBLA_(const char   *srname,const f77_int *info, ftnlen n)
{
    return xerbla_blis_impl( srname, info, n);
}

int xerbla(const char   *srname,const f77_int *info, ftnlen n)
{
    return xerbla_blis_impl( srname, info, n);
}

void ZAXPY(const f77_int *n,const dcomplex *za,const dcomplex *zx,const f77_int *incx,dcomplex *zy,const f77_int *incy)
{
    zaxpy_blis_impl( n, za, zx, incx, zy, incy);
}

void zaxpy(const f77_int *n,const dcomplex *za,const dcomplex *zx,const f77_int *incx,dcomplex *zy,const f77_int *incy)
{
    zaxpy_blis_impl( n, za, zx, incx, zy, incy);
}

void ZAXPY_(const f77_int *n,const dcomplex *za,const dcomplex *zx,const f77_int *incx,dcomplex *zy,const f77_int *incy)
{
    zaxpy_blis_impl( n, za, zx, incx, zy, incy);
}

void ZCOPY(const f77_int *n,const dcomplex *zx,const f77_int *incx,dcomplex *zy,const f77_int *incy)
{
    zcopy_blis_impl( n, zx, incx, zy, incy);
}

void zcopy(const f77_int *n,const dcomplex *zx,const f77_int *incx,dcomplex *zy,const f77_int *incy)
{
    zcopy_blis_impl( n, zx, incx, zy, incy);
}

void ZCOPY_(const f77_int *n,const dcomplex *zx,const f77_int *incx,dcomplex *zy,const f77_int *incy)
{
    zcopy_blis_impl( n, zx, incx, zy, incy);
}

void ZDROT(const f77_int *n,dcomplex *cx,const f77_int *incx,dcomplex *cy,const f77_int *incy,const double *c,const double *s)
{
    zdrot_blis_impl( n, cx, incx, cy, incy, c, s);
}

void zdrot(const f77_int *n,dcomplex *cx,const f77_int *incx,dcomplex *cy,const f77_int *incy,const double *c,const double *s)
{
    zdrot_blis_impl( n, cx, incx, cy, incy, c, s);
}

void ZDROT_(const f77_int *n,dcomplex *cx,const f77_int *incx,dcomplex *cy,const f77_int *incy,const double *c,const double *s)
{
    zdrot_blis_impl( n, cx, incx, cy, incy, c, s);
}

void ZDSCAL(const f77_int *n,const double *da,dcomplex *zx,const f77_int *incx)
{
    zdscal_blis_impl( n, da, zx, incx);
}

void zdscal(const f77_int *n,const double *da,dcomplex *zx,const f77_int *incx)
{
    zdscal_blis_impl( n, da, zx, incx);
}

void ZDSCAL_(const f77_int *n,const double *da,dcomplex *zx,const f77_int *incx)
{
    zdscal_blis_impl( n, da, zx, incx);
}

void ZGBMV(const char   *trans,const f77_int *m,const f77_int *n,const f77_int *kl,const f77_int *ku,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *x,const f77_int *incx,const dcomplex *beta,dcomplex *y,const f77_int *incy)
{
    zgbmv_blis_impl( trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

void zgbmv(const char   *trans,const f77_int *m,const f77_int *n,const f77_int *kl,const f77_int *ku,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *x,const f77_int *incx,const dcomplex *beta,dcomplex *y,const f77_int *incy)
{
    zgbmv_blis_impl( trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

void ZGBMV_(const char   *trans,const f77_int *m,const f77_int *n,const f77_int *kl,const f77_int *ku,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *x,const f77_int *incx,const dcomplex *beta,dcomplex *y,const f77_int *incy)
{
    zgbmv_blis_impl( trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

void ZGEMM(const char   *transa,const char   *transb,const f77_int *m,const f77_int *n,const f77_int *k,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *b,const f77_int *ldb,const dcomplex *beta,dcomplex *c,const f77_int *ldc)
{
    zgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void zgemm(const char   *transa,const char   *transb,const f77_int *m,const f77_int *n,const f77_int *k,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *b,const f77_int *ldb,const dcomplex *beta,dcomplex *c,const f77_int *ldc)
{
    zgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ZGEMM_(const char   *transa,const char   *transb,const f77_int *m,const f77_int *n,const f77_int *k,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *b,const f77_int *ldb,const dcomplex *beta,dcomplex *c,const f77_int *ldc)
{
    zgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ZGEMV(const char   *trans,const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *x,const f77_int *incx,const dcomplex *beta,dcomplex *y,const f77_int *incy)
{
    zgemv_blis_impl( trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void zgemv(const char   *trans,const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *x,const f77_int *incx,const dcomplex *beta,dcomplex *y,const f77_int *incy)
{
    zgemv_blis_impl( trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void ZGEMV_(const char   *trans,const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *x,const f77_int *incx,const dcomplex *beta,dcomplex *y,const f77_int *incy)
{
    zgemv_blis_impl( trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void ZGERC(const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *x,const f77_int *incx,const dcomplex *y,const f77_int *incy,dcomplex *a,const f77_int *lda)
{
    zgerc_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void zgerc(const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *x,const f77_int *incx,const dcomplex *y,const f77_int *incy,dcomplex *a,const f77_int *lda)
{
    zgerc_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void ZGERC_(const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *x,const f77_int *incx,const dcomplex *y,const f77_int *incy,dcomplex *a,const f77_int *lda)
{
    zgerc_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void ZGERU(const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *x,const f77_int *incx,const dcomplex *y,const f77_int *incy,dcomplex *a,const f77_int *lda)
{
    zgeru_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void zgeru(const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *x,const f77_int *incx,const dcomplex *y,const f77_int *incy,dcomplex *a,const f77_int *lda)
{
    zgeru_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void ZGERU_(const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *x,const f77_int *incx,const dcomplex *y,const f77_int *incy,dcomplex *a,const f77_int *lda)
{
    zgeru_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void ZHBMV(const char   *uplo,const f77_int *n,const f77_int *k,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *x,const f77_int *incx,const dcomplex *beta,dcomplex *y,const f77_int *incy)
{
    zhbmv_blis_impl( uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void zhbmv(const char   *uplo,const f77_int *n,const f77_int *k,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *x,const f77_int *incx,const dcomplex *beta,dcomplex *y,const f77_int *incy)
{
    zhbmv_blis_impl( uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void ZHBMV_(const char   *uplo,const f77_int *n,const f77_int *k,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *x,const f77_int *incx,const dcomplex *beta,dcomplex *y,const f77_int *incy)
{
    zhbmv_blis_impl( uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void ZHEMM(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *b,const f77_int *ldb,const dcomplex *beta,dcomplex *c,const f77_int *ldc)
{
    zhemm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void zhemm(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *b,const f77_int *ldb,const dcomplex *beta,dcomplex *c,const f77_int *ldc)
{
    zhemm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ZHEMM_(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *b,const f77_int *ldb,const dcomplex *beta,dcomplex *c,const f77_int *ldc)
{
    zhemm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ZHEMV(const char   *uplo,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *x,const f77_int *incx,const dcomplex *beta,dcomplex *y,const f77_int *incy)
{
    zhemv_blis_impl( uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

void zhemv(const char   *uplo,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *x,const f77_int *incx,const dcomplex *beta,dcomplex *y,const f77_int *incy)
{
    zhemv_blis_impl( uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

void ZHEMV_(const char   *uplo,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *x,const f77_int *incx,const dcomplex *beta,dcomplex *y,const f77_int *incy)
{
    zhemv_blis_impl( uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

void ZHER(const char   *uplo,const f77_int *n,const double *alpha,const dcomplex *x,const f77_int *incx,dcomplex *a,const f77_int *lda)
{
    zher_blis_impl( uplo, n, alpha, x, incx, a, lda);
}

void zher(const char   *uplo,const f77_int *n,const double *alpha,const dcomplex *x,const f77_int *incx,dcomplex *a,const f77_int *lda)
{
    zher_blis_impl( uplo, n, alpha, x, incx, a, lda);
}

void ZHER_(const char   *uplo,const f77_int *n,const double *alpha,const dcomplex *x,const f77_int *incx,dcomplex *a,const f77_int *lda)
{
    zher_blis_impl( uplo, n, alpha, x, incx, a, lda);
}

void ZHER2(const char   *uplo,const f77_int *n,const dcomplex *alpha,const dcomplex *x,const f77_int *incx,const dcomplex *y,const f77_int *incy,dcomplex *a,const f77_int *lda)
{
    zher2_blis_impl( uplo, n, alpha, x, incx, y, incy, a, lda);
}

void zher2(const char   *uplo,const f77_int *n,const dcomplex *alpha,const dcomplex *x,const f77_int *incx,const dcomplex *y,const f77_int *incy,dcomplex *a,const f77_int *lda)
{
    zher2_blis_impl( uplo, n, alpha, x, incx, y, incy, a, lda);
}

void ZHER2_(const char   *uplo,const f77_int *n,const dcomplex *alpha,const dcomplex *x,const f77_int *incx,const dcomplex *y,const f77_int *incy,dcomplex *a,const f77_int *lda)
{
    zher2_blis_impl( uplo, n, alpha, x, incx, y, incy, a, lda);
}

void ZHER2K(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *b,const f77_int *ldb,const double *beta,dcomplex *c,const f77_int *ldc)
{
    zher2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void zher2k(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *b,const f77_int *ldb,const double *beta,dcomplex *c,const f77_int *ldc)
{
    zher2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ZHER2K_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *b,const f77_int *ldb,const double *beta,dcomplex *c,const f77_int *ldc)
{
    zher2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ZHERK(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const double *alpha,const dcomplex *a,const f77_int *lda,const double *beta,dcomplex *c,const f77_int *ldc)
{
    zherk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void zherk(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const double *alpha,const dcomplex *a,const f77_int *lda,const double *beta,dcomplex *c,const f77_int *ldc)
{
    zherk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void ZHERK_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const double *alpha,const dcomplex *a,const f77_int *lda,const double *beta,dcomplex *c,const f77_int *ldc)
{
    zherk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void ZHPMV(const char   *uplo,const f77_int *n,const dcomplex *alpha,const dcomplex *ap,const dcomplex *x,const f77_int *incx,const dcomplex *beta,dcomplex *y,const f77_int *incy)
{
    zhpmv_blis_impl( uplo, n, alpha, ap, x, incx, beta, y, incy);
}

void zhpmv(const char   *uplo,const f77_int *n,const dcomplex *alpha,const dcomplex *ap,const dcomplex *x,const f77_int *incx,const dcomplex *beta,dcomplex *y,const f77_int *incy)
{
    zhpmv_blis_impl( uplo, n, alpha, ap, x, incx, beta, y, incy);
}

void ZHPMV_(const char   *uplo,const f77_int *n,const dcomplex *alpha,const dcomplex *ap,const dcomplex *x,const f77_int *incx,const dcomplex *beta,dcomplex *y,const f77_int *incy)
{
    zhpmv_blis_impl( uplo, n, alpha, ap, x, incx, beta, y, incy);
}

void ZHPR(const char   *uplo,const f77_int *n,const bla_double *alpha,const dcomplex *x,const f77_int *incx,dcomplex *ap)
{
    zhpr_blis_impl( uplo, n, alpha, x, incx, ap);
}

void zhpr(const char   *uplo,const f77_int *n,const bla_double *alpha,const dcomplex *x,const f77_int *incx,dcomplex *ap)
{
    zhpr_blis_impl( uplo, n, alpha, x, incx, ap);
}

void ZHPR_(const char   *uplo,const f77_int *n,const bla_double *alpha,const dcomplex *x,const f77_int *incx,dcomplex *ap)
{
    zhpr_blis_impl( uplo, n, alpha, x, incx, ap);
}

void ZHPR2(const char   *uplo,const f77_int *n,const dcomplex *alpha,const dcomplex *x,const f77_int *incx,const dcomplex *y,const f77_int *incy,dcomplex *ap)
{
    zhpr2_blis_impl( uplo, n, alpha, x, incx, y, incy, ap);
}

void zhpr2(const char   *uplo,const f77_int *n,const dcomplex *alpha,const dcomplex *x,const f77_int *incx,const dcomplex *y,const f77_int *incy,dcomplex *ap)
{
    zhpr2_blis_impl( uplo, n, alpha, x, incx, y, incy, ap);
}

void ZHPR2_(const char   *uplo,const f77_int *n,const dcomplex *alpha,const dcomplex *x,const f77_int *incx,const dcomplex *y,const f77_int *incy,dcomplex *ap)
{
    zhpr2_blis_impl( uplo, n, alpha, x, incx, y, incy, ap);
}

void ZROTG(dcomplex *ca,bla_dcomplex *cb,bla_double *c,dcomplex *s)
{
    zrotg_blis_impl( ca, cb, c, s);
}

void zrotg(dcomplex *ca,bla_dcomplex *cb,bla_double *c,dcomplex *s)
{
    zrotg_blis_impl( ca, cb, c, s);
}

void ZROTG_(dcomplex *ca,bla_dcomplex *cb,bla_double *c,dcomplex *s)
{
    zrotg_blis_impl( ca, cb, c, s);
}

void ZSCAL(const f77_int *n,const dcomplex *za,dcomplex *zx,const f77_int *incx)
{
    zscal_blis_impl( n, za, zx, incx);
}

void zscal(const f77_int *n,const dcomplex *za,dcomplex *zx,const f77_int *incx)
{
    zscal_blis_impl( n, za, zx, incx);
}

void ZSCAL_(const f77_int *n,const dcomplex *za,dcomplex *zx,const f77_int *incx)
{
    zscal_blis_impl( n, za, zx, incx);
}

void ZSWAP(const f77_int *n,dcomplex *zx,const f77_int *incx,dcomplex *zy,const f77_int *incy)
{
    zswap_blis_impl( n, zx, incx, zy, incy);
}

void zswap(const f77_int *n,dcomplex *zx,const f77_int *incx,dcomplex *zy,const f77_int *incy)
{
    zswap_blis_impl( n, zx, incx, zy, incy);
}

void ZSWAP_(const f77_int *n,dcomplex *zx,const f77_int *incx,dcomplex *zy,const f77_int *incy)
{
    zswap_blis_impl( n, zx, incx, zy, incy);
}

void ZSYMM(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *b,const f77_int *ldb,const dcomplex *beta,dcomplex *c,const f77_int *ldc)
{
    zsymm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void zsymm(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *b,const f77_int *ldb,const dcomplex *beta,dcomplex *c,const f77_int *ldc)
{
    zsymm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ZSYMM_(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *b,const f77_int *ldb,const dcomplex *beta,dcomplex *c,const f77_int *ldc)
{
    zsymm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ZSYR2K(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *b,const f77_int *ldb,const dcomplex *beta,dcomplex *c,const f77_int *ldc)
{
    zsyr2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void zsyr2k(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *b,const f77_int *ldb,const dcomplex *beta,dcomplex *c,const f77_int *ldc)
{
    zsyr2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ZSYR2K_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *b,const f77_int *ldb,const dcomplex *beta,dcomplex *c,const f77_int *ldc)
{
    zsyr2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ZSYRK(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *beta,dcomplex *c,const f77_int *ldc)
{
    zsyrk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void zsyrk(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *beta,dcomplex *c,const f77_int *ldc)
{
    zsyrk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void ZSYRK_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *beta,dcomplex *c,const f77_int *ldc)
{
    zsyrk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void ZTBMV(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const dcomplex *a,const f77_int *lda,dcomplex *x,const f77_int *incx)
{
    ztbmv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void ztbmv(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const dcomplex *a,const f77_int *lda,dcomplex *x,const f77_int *incx)
{
    ztbmv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void ZTBMV_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const dcomplex *a,const f77_int *lda,dcomplex *x,const f77_int *incx)
{
    ztbmv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void ZTBSV(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const dcomplex *a,const f77_int *lda,dcomplex *x,const f77_int *incx)
{
    ztbsv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void ztbsv(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const dcomplex *a,const f77_int *lda,dcomplex *x,const f77_int *incx)
{
    ztbsv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void ZTBSV_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const dcomplex *a,const f77_int *lda,dcomplex *x,const f77_int *incx)
{
    ztbsv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void ZTPMV(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const dcomplex *ap,dcomplex *x,const f77_int *incx)
{
    ztpmv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void ztpmv(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const dcomplex *ap,dcomplex *x,const f77_int *incx)
{
    ztpmv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void ZTPMV_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const dcomplex *ap,dcomplex *x,const f77_int *incx)
{
    ztpmv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void ZTPSV(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const dcomplex *ap,dcomplex *x,const f77_int *incx)
{
    ztpsv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void ztpsv(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const dcomplex *ap,dcomplex *x,const f77_int *incx)
{
    ztpsv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void ZTPSV_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const dcomplex *ap,dcomplex *x,const f77_int *incx)
{
    ztpsv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void ZTRMM(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,dcomplex *b,const f77_int *ldb)
{
    ztrmm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void ztrmm(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,dcomplex *b,const f77_int *ldb)
{
    ztrmm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void ZTRMM_(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,dcomplex *b,const f77_int *ldb)
{
    ztrmm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void ZTRMV(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const dcomplex *a,const f77_int *lda,dcomplex *x,const f77_int *incx)
{
    ztrmv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void ztrmv(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const dcomplex *a,const f77_int *lda,dcomplex *x,const f77_int *incx)
{
    ztrmv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void ZTRMV_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const dcomplex *a,const f77_int *lda,dcomplex *x,const f77_int *incx)
{
    ztrmv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void ZTRSM(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,dcomplex *b,const f77_int *ldb)
{
    ztrsm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void ztrsm(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,dcomplex *b,const f77_int *ldb)
{
    ztrsm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void ZTRSM_(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,dcomplex *b,const f77_int *ldb)
{
    ztrsm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void ZTRSV(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const dcomplex *a,const f77_int *lda,dcomplex *x,const f77_int *incx)
{
    ztrsv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void ztrsv(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const dcomplex *a,const f77_int *lda,dcomplex *x,const f77_int *incx)
{
    ztrsv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void ZTRSV_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const dcomplex *a,const f77_int *lda,dcomplex *x,const f77_int *incx)
{
    ztrsv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

#ifdef BLIS_ENABLE_CBLAS

void CDOTCSUB( const f77_int* n, const scomplex* x,const f77_int* incx, const scomplex* y, const f77_int* incy, scomplex* rval)
{
    cdotcsub_blis_impl( n, x, incx, y, incy, rval);
}

void cdotcsub( const f77_int* n, const scomplex* x,const f77_int* incx, const scomplex* y, const f77_int* incy, scomplex* rval)
{
    cdotcsub_blis_impl( n, x, incx, y, incy, rval);
}

void CDOTCSUB_( const f77_int* n, const scomplex* x,const f77_int* incx, const scomplex* y, const f77_int* incy, scomplex* rval)
{
    cdotcsub_blis_impl( n, x, incx, y, incy, rval);
}

void CDOTUSUB( const f77_int* n, const scomplex* x,const f77_int* incxy, const scomplex* y, const f77_int* incy, scomplex* rval)
{
    cdotusub_blis_impl( n, x, incxy, y, incy, rval);
}

void cdotusub( const f77_int* n, const scomplex* x,const f77_int* incxy, const scomplex* y, const f77_int* incy, scomplex* rval)
{
    cdotusub_blis_impl( n, x, incxy, y, incy, rval);
}

void CDOTUSUB_( const f77_int* n, const scomplex* x,const f77_int* incxy, const scomplex* y, const f77_int* incy, scomplex* rval)
{
    cdotusub_blis_impl( n, x, incxy, y, incy, rval);
}

#endif // BLIS_ENABLE_CBLAS

void CGEMM3M( const f77_char* transa, const f77_char* transb, const f77_int* m, const f77_int* n, const f77_int* k, const scomplex* alpha, const scomplex* a, const f77_int* lda, const scomplex* b, const f77_int* ldb, const scomplex* beta, scomplex* c, const f77_int* ldc)
{
    cgemm3m_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void cgemm3m( const f77_char* transa, const f77_char* transb, const f77_int* m, const f77_int* n, const f77_int* k, const scomplex* alpha, const scomplex* a, const f77_int* lda, const scomplex* b, const f77_int* ldb, const scomplex* beta, scomplex* c, const f77_int* ldc)
{
    cgemm3m_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void CGEMM3M_( const f77_char* transa, const f77_char* transb, const f77_int* m, const f77_int* n, const f77_int* k, const scomplex* alpha, const scomplex* a, const f77_int* lda, const scomplex* b, const f77_int* ldb, const scomplex* beta, scomplex* c, const f77_int* ldc)
{
    cgemm3m_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void CGEMM_BATCH( const f77_char* transa_array, const f77_char* transb_array,const f77_int *m_array, const f77_int *n_array, const f77_int *k_array,const scomplex* alpha_array, const scomplex** a_array, const  f77_int *lda_array, const scomplex** b_array, const f77_int *ldb_array, const scomplex* beta_array, scomplex** c_array, const f77_int *ldc_array, const f77_int* group_count, const f77_int *group_size)
{
    cgemm_batch_blis_impl( transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

void cgemm_batch( const f77_char* transa_array, const f77_char* transb_array,const f77_int *m_array, const f77_int *n_array, const f77_int *k_array,const scomplex* alpha_array, const scomplex** a_array, const  f77_int *lda_array, const scomplex** b_array, const f77_int *ldb_array, const scomplex* beta_array, scomplex** c_array, const f77_int *ldc_array, const f77_int* group_count, const f77_int *group_size)
{
    cgemm_batch_blis_impl( transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

void CGEMM_BATCH_( const f77_char* transa_array, const f77_char* transb_array,const f77_int *m_array, const f77_int *n_array, const f77_int *k_array,const scomplex* alpha_array, const scomplex** a_array, const  f77_int *lda_array, const scomplex** b_array, const f77_int *ldb_array, const scomplex* beta_array, scomplex** c_array, const f77_int *ldc_array, const f77_int* group_count, const f77_int *group_size)
{
    cgemm_batch_blis_impl( transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

void CGEMMT( const f77_char* uploc, const f77_char* transa, const f77_char* transb, const f77_int* n, const f77_int* k, const  scomplex* alpha, const scomplex* a, const f77_int* lda, const scomplex* b, const f77_int* ldb, const scomplex* beta, scomplex* c, const f77_int* ldc)
{
    cgemmt_blis_impl( uploc, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void cgemmt( const f77_char* uploc, const f77_char* transa, const f77_char* transb, const f77_int* n, const f77_int* k, const  scomplex* alpha, const scomplex* a, const f77_int* lda, const scomplex* b, const f77_int* ldb, const scomplex* beta, scomplex* c, const f77_int* ldc)
{
    cgemmt_blis_impl( uploc, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void CGEMMT_( const f77_char* uploc, const f77_char* transa, const f77_char* transb, const f77_int* n, const f77_int* k, const  scomplex* alpha, const scomplex* a, const f77_int* lda, const scomplex* b, const f77_int* ldb, const scomplex* beta, scomplex* c, const f77_int* ldc)
{
    cgemmt_blis_impl( uploc, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

//#ifdef BLIS_ENABLE_CBLAS

void CIMATCOPY(f77_char* trans, f77_int* rows, f77_int* cols, const scomplex* alpha,scomplex* aptr, f77_int* lda, f77_int* ldb)
{
    cimatcopy_( trans, rows, cols, alpha, aptr, lda, ldb);
}

void cimatcopy(f77_char* trans, f77_int* rows, f77_int* cols, const scomplex* alpha,scomplex* aptr, f77_int* lda, f77_int* ldb)
{
    cimatcopy_( trans, rows, cols, alpha, aptr, lda, ldb);
}

void CIMATCOPY_(f77_char* trans, f77_int* rows, f77_int* cols, const scomplex* alpha,scomplex* aptr, f77_int* lda, f77_int* ldb)
{
    cimatcopy_( trans, rows, cols, alpha, aptr, lda, ldb);
}

void COMATADD(f77_char* transa,f77_char* transb, f77_int* m, f77_int* n, const scomplex* alpha, const scomplex* A, f77_int* lda,const scomplex* beta, scomplex* B, f77_int* ldb, scomplex* C, f77_int* ldc)
{
    comatadd_( transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

void comatadd(f77_char* transa,f77_char* transb, f77_int* m, f77_int* n, const scomplex* alpha, const scomplex* A, f77_int* lda,const scomplex* beta, scomplex* B, f77_int* ldb, scomplex* C, f77_int* ldc)
{
    comatadd_( transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

void COMATADD_(f77_char* transa,f77_char* transb, f77_int* m, f77_int* n, const scomplex* alpha, const scomplex* A, f77_int* lda,const scomplex* beta, scomplex* B, f77_int* ldb, scomplex* C, f77_int* ldc)
{
    comatadd_( transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

void COMATCOPY2(f77_char* trans, f77_int* rows, f77_int* cols, const scomplex* alpha, const scomplex* aptr, f77_int* lda,f77_int* stridea, scomplex* bptr, f77_int* ldb,f77_int* strideb)
{
    comatcopy2_( trans, rows, cols, alpha, aptr, lda, stridea, bptr, ldb, strideb);
}

void comatcopy2(f77_char* trans, f77_int* rows, f77_int* cols, const scomplex* alpha, const scomplex* aptr, f77_int* lda,f77_int* stridea, scomplex* bptr, f77_int* ldb,f77_int* strideb)
{
    comatcopy2_( trans, rows, cols, alpha, aptr, lda, stridea, bptr, ldb, strideb);
}

void COMATCOPY2_(f77_char* trans, f77_int* rows, f77_int* cols, const scomplex* alpha, const scomplex* aptr, f77_int* lda,f77_int* stridea, scomplex* bptr, f77_int* ldb,f77_int* strideb)
{
    comatcopy2_( trans, rows, cols, alpha, aptr, lda, stridea, bptr, ldb, strideb);
}

void COMATCOPY(f77_char* trans, f77_int* rows, f77_int* cols, const scomplex* alpha, const scomplex* aptr, f77_int* lda, scomplex* bptr, f77_int* ldb)
{
    comatcopy_( trans, rows, cols, alpha, aptr, lda, bptr, ldb);
}

void comatcopy(f77_char* trans, f77_int* rows, f77_int* cols, const scomplex* alpha, const scomplex* aptr, f77_int* lda, scomplex* bptr, f77_int* ldb)
{
    comatcopy_( trans, rows, cols, alpha, aptr, lda, bptr, ldb);
}

void COMATCOPY_(f77_char* trans, f77_int* rows, f77_int* cols, const scomplex* alpha, const scomplex* aptr, f77_int* lda, scomplex* bptr, f77_int* ldb)
{
    comatcopy_( trans, rows, cols, alpha, aptr, lda, bptr, ldb);
}

//#endif // BLIS_ENABLE_CBLAS

#ifdef BLIS_ENABLE_CBLAS

void DASUMSUB(const f77_int* n, const double* x, const f77_int* incx, double* rval)
{
    dasumsub_blis_impl( n, x, incx, rval);
}

void dasumsub(const f77_int* n, const double* x, const f77_int* incx, double* rval)
{
    dasumsub_blis_impl( n, x, incx, rval);
}

void DASUMSUB_(const f77_int* n, const double* x, const f77_int* incx, double* rval)
{
    dasumsub_blis_impl( n, x, incx, rval);
}

#endif // BLIS_ENABLE_CBLAS

void DAXPBY(const f77_int* n, const double* alpha, const double *x, const f77_int* incx, const double* beta, double *y, const f77_int* incy)
{
    daxpby_blis_impl( n, alpha, x, incx, beta, y, incy);
}

void daxpby(const f77_int* n, const double* alpha, const double *x, const f77_int* incx, const double* beta, double *y, const f77_int* incy)
{
    daxpby_blis_impl( n, alpha, x, incx, beta, y, incy);
}

void DAXPBY_(const f77_int* n, const double* alpha, const double *x, const f77_int* incx, const double* beta, double *y, const f77_int* incy)
{
    daxpby_blis_impl( n, alpha, x, incx, beta, y, incy);
}

#ifdef BLIS_ENABLE_CBLAS

void DDOTSUB(const f77_int* n, const double* x, const f77_int* incx, const double* y, const f77_int* incy, double* rval)
{
    ddotsub_blis_impl( n, x, incx, y, incy, rval);
}

void ddotsub(const f77_int* n, const double* x, const f77_int* incx, const double* y, const f77_int* incy, double* rval)
{
    ddotsub_blis_impl( n, x, incx, y, incy, rval);
}

void DDOTSUB_(const f77_int* n, const double* x, const f77_int* incx, const double* y, const f77_int* incy, double* rval)
{
    ddotsub_blis_impl( n, x, incx, y, incy, rval);
}

#endif // BLIS_ENABLE_CBLAS

void DGEMM_BATCH( const f77_char* transa_array, const f77_char* transb_array,const f77_int *m_array, const f77_int *n_array, const f77_int *k_array,const double* alpha_array, const double** a_array, const  f77_int *lda_array, const double** b_array, const f77_int *ldb_array, const double* beta_array, double** c_array, const f77_int *ldc_array, const f77_int* group_count, const f77_int *group_size)
{
    dgemm_batch_blis_impl( transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

void dgemm_batch( const f77_char* transa_array, const f77_char* transb_array,const f77_int *m_array, const f77_int *n_array, const f77_int *k_array,const double* alpha_array, const double** a_array, const  f77_int *lda_array, const double** b_array, const f77_int *ldb_array, const double* beta_array, double** c_array, const f77_int *ldc_array, const f77_int* group_count, const f77_int *group_size)
{
    dgemm_batch_blis_impl( transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

void DGEMM_BATCH_( const f77_char* transa_array, const f77_char* transb_array,const f77_int *m_array, const f77_int *n_array, const f77_int *k_array,const double* alpha_array, const double** a_array, const  f77_int *lda_array, const double** b_array, const f77_int *ldb_array, const double* beta_array, double** c_array, const f77_int *ldc_array, const f77_int* group_count, const f77_int *group_size)
{
    dgemm_batch_blis_impl( transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

f77_int DGEMM_PACK_GET_SIZE(const f77_char* identifier, const f77_int* pm, const f77_int* pn, const f77_int* pk)
{
    return dgemm_pack_get_size_blis_impl( identifier, pm, pn, pk );
}

f77_int dgemm_pack_get_size(const f77_char* identifier, const f77_int* pm, const f77_int* pn, const f77_int* pk)
{
    return dgemm_pack_get_size_blis_impl( identifier, pm, pn, pk );
}

f77_int DGEMM_PACK_GET_SIZE_(const f77_char* identifier, const f77_int* pm, const f77_int* pn, const f77_int* pk)
{
    return dgemm_pack_get_size_blis_impl( identifier, pm, pn, pk );
}

void DGEMM_PACK( const f77_char* identifier, const f77_char* trans, const f77_int* mm, const f77_int* nn, const f77_int* kk, const double* alpha, const double* src, const f77_int* pld, double* dest )
{
    dgemm_pack_blis_impl( identifier, trans, mm, nn, kk, alpha, src, pld, dest );
}

void dgemm_pack( const f77_char* identifier, const f77_char* trans, const f77_int* mm, const f77_int* nn, const f77_int* kk, const double* alpha, const double* src, const f77_int* pld, double* dest )
{
    dgemm_pack_blis_impl( identifier, trans, mm, nn, kk, alpha, src, pld, dest );
}

void DGEMM_PACK_( const f77_char* identifier, const f77_char* trans, const f77_int* mm, const f77_int* nn, const f77_int* kk, const double* alpha, const double* src, const f77_int* pld, double* dest )
{
    dgemm_pack_blis_impl( identifier, trans, mm, nn, kk, alpha, src, pld, dest );
}

void DGEMM_COMPUTE( const f77_char* transa, const f77_char* transb, const f77_int* m, const f77_int* n, const f77_int* k, const double* a, const f77_int* lda, const double* b, const f77_int* ldb, const double* beta, double* c, const f77_int* ldc )
{
    f77_int rs_a = 1;
    f77_int rs_b = 1;
    f77_int rs_c = 1;
    dgemm_compute_blis_impl( transa, transb, m, n, k, a, &rs_a, lda, b, &rs_b, ldb, beta, c, &rs_c, ldc );
}

void dgemm_compute( const f77_char* transa, const f77_char* transb, const f77_int* m, const f77_int* n, const f77_int* k, const double* a, const f77_int* lda, const double* b, const f77_int* ldb, const double* beta, double* c, const f77_int* ldc )
{
    f77_int rs_a = 1;
    f77_int rs_b = 1;
    f77_int rs_c = 1;
    dgemm_compute_blis_impl( transa, transb, m, n, k, a, &rs_a, lda, b, &rs_b, ldb, beta, c, &rs_c, ldc );
}

void DGEMM_COMPUTE_( const f77_char* transa, const f77_char* transb, const f77_int* m, const f77_int* n, const f77_int* k, const double* a, const f77_int* lda, const double* b, const f77_int* ldb, const double* beta, double* c, const f77_int* ldc )
{
    f77_int rs_a = 1;
    f77_int rs_b = 1;
    f77_int rs_c = 1;
    dgemm_compute_blis_impl( transa, transb, m, n, k, a, &rs_a, lda, b, &rs_b, ldb, beta, c, &rs_c, ldc );
}

void DGEMMT( const f77_char* uploc, const f77_char* transa, const f77_char* transb, const f77_int* n, const f77_int* k, const  double* alpha, const double* a, const f77_int* lda, const double* b, const f77_int* ldb, const double* beta, double* c, const f77_int* ldc)
{
    dgemmt_blis_impl( uploc, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void dgemmt( const f77_char* uploc, const f77_char* transa, const f77_char* transb, const f77_int* n, const f77_int* k, const  double* alpha, const double* a, const f77_int* lda, const double* b, const f77_int* ldb, const double* beta, double* c, const f77_int* ldc)
{
    dgemmt_blis_impl( uploc, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void DGEMMT_( const f77_char* uploc, const f77_char* transa, const f77_char* transb, const f77_int* n, const f77_int* k, const  double* alpha, const double* a, const f77_int* lda, const double* b, const f77_int* ldb, const double* beta, double* c, const f77_int* ldc)
{
    dgemmt_blis_impl( uploc, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#ifdef BLIS_ENABLE_CBLAS

void DNRM2SUB(const f77_int* n, const double* x, const f77_int* incx, double *rval)
{
    dnrm2sub_blis_impl( n, x, incx, rval);
}

void dnrm2sub(const f77_int* n, const double* x, const f77_int* incx, double *rval)
{
    dnrm2sub_blis_impl( n, x, incx, rval);
}

void DNRM2SUB_(const f77_int* n, const double* x, const f77_int* incx, double *rval)
{
    dnrm2sub_blis_impl( n, x, incx, rval);
}

#endif // BLIS_ENABLE_CBLAS

//#ifdef BLIS_ENABLE_CBLAS

void DOMATADD(f77_char* transa,f77_char* transb, f77_int* m, f77_int* n, const double* alpha, const double* A, f77_int* lda, const double* beta, const double* B, f77_int* ldb, double* C, f77_int* ldc)
{
    domatadd_( transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

void domatadd(f77_char* transa,f77_char* transb, f77_int* m, f77_int* n, const double* alpha, const double* A, f77_int* lda, const double* beta, const double* B, f77_int* ldb, double* C, f77_int* ldc)
{
    domatadd_( transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

void DOMATADD_(f77_char* transa,f77_char* transb, f77_int* m, f77_int* n, const double* alpha, const double* A, f77_int* lda, const double* beta, const double* B, f77_int* ldb, double* C, f77_int* ldc)
{
    domatadd_( transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

void DOMATCOPY2(f77_char* trans, f77_int* rows, f77_int* cols, const double* alpha, const double* aptr, f77_int* lda,f77_int* stridea, double* bptr, f77_int* ldb,f77_int* strideb)
{
    domatcopy2_( trans, rows, cols, alpha, aptr, lda, stridea, bptr, ldb, strideb);
}

void domatcopy2(f77_char* trans, f77_int* rows, f77_int* cols, const double* alpha, const double* aptr, f77_int* lda,f77_int* stridea, double* bptr, f77_int* ldb,f77_int* strideb)
{
    domatcopy2_( trans, rows, cols, alpha, aptr, lda, stridea, bptr, ldb, strideb);
}

void DOMATCOPY2_(f77_char* trans, f77_int* rows, f77_int* cols, const double* alpha, const double* aptr, f77_int* lda,f77_int* stridea, double* bptr, f77_int* ldb,f77_int* strideb)
{
    domatcopy2_( trans, rows, cols, alpha, aptr, lda, stridea, bptr, ldb, strideb);
}

void DOMATCOPY(f77_char* trans, f77_int* rows, f77_int* cols, const double* alpha, const double* aptr, f77_int* lda, double* bptr, f77_int* ldb)
{
    domatcopy_( trans, rows, cols, alpha, aptr, lda, bptr, ldb);
}

void domatcopy(f77_char* trans, f77_int* rows, f77_int* cols, const double* alpha, const double* aptr, f77_int* lda, double* bptr, f77_int* ldb)
{
    domatcopy_( trans, rows, cols, alpha, aptr, lda, bptr, ldb);
}

void DOMATCOPY_(f77_char* trans, f77_int* rows, f77_int* cols, const double* alpha, const double* aptr, f77_int* lda, double* bptr, f77_int* ldb)
{
    domatcopy_( trans, rows, cols, alpha, aptr, lda, bptr, ldb);
}

//#endif // BLIS_ENABLE_CBLAS

#ifdef BLIS_ENABLE_CBLAS

void DZASUMSUB(const f77_int* n, const dcomplex* x, const f77_int* incx, double* rval)
{
    dzasumsub_blis_impl( n, x, incx, rval);
}

void dzasumsub(const f77_int* n, const dcomplex* x, const f77_int* incx, double* rval)
{
    dzasumsub_blis_impl( n, x, incx, rval);
}

void DZASUMSUB_(const f77_int* n, const dcomplex* x, const f77_int* incx, double* rval)
{
    dzasumsub_blis_impl( n, x, incx, rval);
}

void DZNRM2SUB(const f77_int* n, const dcomplex* x, const f77_int* incx, double* rval)
{
    dznrm2sub_blis_impl( n, x, incx, rval);
}

void dznrm2sub(const f77_int* n, const dcomplex* x, const f77_int* incx, double* rval)
{
    dznrm2sub_blis_impl( n, x, incx, rval);
}

void DZNRM2SUB_(const f77_int* n, const dcomplex* x, const f77_int* incx, double* rval)
{
    dznrm2sub_blis_impl( n, x, incx, rval);
}

void ICAMAXSUB(const f77_int* n, const scomplex* x, const f77_int* incx, f77_int* rval)
{
    icamaxsub_blis_impl( n, x, incx, rval);
}

void icamaxsub(const f77_int* n, const scomplex* x, const f77_int* incx, f77_int* rval)
{
    icamaxsub_blis_impl( n, x, incx, rval);
}

void ICAMAXSUB_(const f77_int* n, const scomplex* x, const f77_int* incx, f77_int* rval)
{
    icamaxsub_blis_impl( n, x, incx, rval);
}

#endif // BLIS_ENABLE_CBLAS

f77_int ICAMIN( const f77_int* n, const scomplex* x, const f77_int* incx)
{
    return icamin_blis_impl( n, x, incx);
}

f77_int icamin( const f77_int* n, const scomplex* x, const f77_int* incx)
{
    return icamin_blis_impl( n, x, incx);
}

f77_int ICAMIN_( const f77_int* n, const scomplex* x, const f77_int* incx)
{
    return icamin_blis_impl( n, x, incx);
}

#ifdef BLIS_ENABLE_CBLAS

void ICAMINSUB( const f77_int* n, const scomplex* x, const f77_int* incx, f77_int* rval)
{
    icaminsub_blis_impl( n, x, incx, rval);
}

void icaminsub( const f77_int* n, const scomplex* x, const f77_int* incx, f77_int* rval)
{
    icaminsub_blis_impl( n, x, incx, rval);
}

void ICAMINSUB_( const f77_int* n, const scomplex* x, const f77_int* incx, f77_int* rval)
{
    icaminsub_blis_impl( n, x, incx, rval);
}

void IDAMAXSUB( const f77_int* n, const double* x, const f77_int* incx, f77_int* rval)
{
    idamaxsub_blis_impl( n, x, incx, rval);
}

void idamaxsub( const f77_int* n, const double* x, const f77_int* incx, f77_int* rval)
{
    idamaxsub_blis_impl( n, x, incx, rval);
}

void IDAMAXSUB_( const f77_int* n, const double* x, const f77_int* incx, f77_int* rval)
{
    idamaxsub_blis_impl( n, x, incx, rval);
}

#endif // BLIS_ENABLE_CBLAS

f77_int IDAMIN( const f77_int* n, const double* x, const f77_int* incx)
{
    return idamin_blis_impl( n, x, incx);
}

f77_int idamin( const f77_int* n, const double* x, const f77_int* incx)
{
    return idamin_blis_impl( n, x, incx);
}

f77_int IDAMIN_( const f77_int* n, const double* x, const f77_int* incx)
{
    return idamin_blis_impl( n, x, incx);
}

#ifdef BLIS_ENABLE_CBLAS

void IDAMINSUB(const f77_int* n, const double* x, const f77_int* incx, f77_int* rval)
{
    idaminsub_blis_impl( n, x, incx, rval);
}

void idaminsub(const f77_int* n, const double* x, const f77_int* incx, f77_int* rval)
{
    idaminsub_blis_impl( n, x, incx, rval);
}

void IDAMINSUB_(const f77_int* n, const double* x, const f77_int* incx, f77_int* rval)
{
    idaminsub_blis_impl( n, x, incx, rval);
}

void ISAMAXSUB( const f77_int* n, const float* x, const f77_int* incx, f77_int* rval)
{
    isamaxsub_blis_impl( n, x, incx, rval);
}

void isamaxsub( const f77_int* n, const float* x, const f77_int* incx, f77_int* rval)
{
    isamaxsub_blis_impl( n, x, incx, rval);
}

void ISAMAXSUB_( const f77_int* n, const float* x, const f77_int* incx, f77_int* rval)
{
    isamaxsub_blis_impl( n, x, incx, rval);
}

#endif // BLIS_ENABLE_CBLAS

f77_int ISAMIN( const f77_int* n, const float* x, const f77_int* incx)
{
    return isamin_blis_impl( n, x, incx);
}

f77_int isamin( const f77_int* n, const float* x, const f77_int* incx)
{
    return isamin_blis_impl( n, x, incx);
}

f77_int ISAMIN_( const f77_int* n, const float* x, const f77_int* incx)
{
    return isamin_blis_impl( n, x, incx);
}

#ifdef BLIS_ENABLE_CBLAS

void ISAMINSUB( const f77_int* n, const float* x, const f77_int* incx, f77_int* rval)
{
    isaminsub_blis_impl( n, x, incx, rval);
}

void isaminsub( const f77_int* n, const float* x, const f77_int* incx, f77_int* rval)
{
    isaminsub_blis_impl( n, x, incx, rval);
}

void ISAMINSUB_( const f77_int* n, const float* x, const f77_int* incx, f77_int* rval)
{
    isaminsub_blis_impl( n, x, incx, rval);
}

void IZAMAXSUB( const f77_int* n, const dcomplex* x, const f77_int* incx, f77_int* rval)
{
    izamaxsub_blis_impl( n, x, incx, rval);
}

void izamaxsub( const f77_int* n, const dcomplex* x, const f77_int* incx, f77_int* rval)
{
    izamaxsub_blis_impl( n, x, incx, rval);
}

void IZAMAXSUB_( const f77_int* n, const dcomplex* x, const f77_int* incx, f77_int* rval)
{
    izamaxsub_blis_impl( n, x, incx, rval);
}

#endif // BLIS_ENABLE_CBLAS

f77_int IZAMIN( const f77_int* n, const dcomplex* x, const f77_int* incx)
{
    return izamin_blis_impl( n, x, incx);
}

f77_int izamin( const f77_int* n, const dcomplex* x, const f77_int* incx)
{
    return izamin_blis_impl( n, x, incx);
}

f77_int IZAMIN_( const f77_int* n, const dcomplex* x, const f77_int* incx)
{
    return izamin_blis_impl( n, x, incx);
}

#ifdef BLIS_ENABLE_CBLAS

void IZAMINSUB( const f77_int* n, const dcomplex* x, const f77_int* incx, f77_int* rval)
{
    izaminsub_blis_impl( n, x, incx, rval);
}

void izaminsub( const f77_int* n, const dcomplex* x, const f77_int* incx, f77_int* rval)
{
    izaminsub_blis_impl( n, x, incx, rval);
}

void IZAMINSUB_( const f77_int* n, const dcomplex* x, const f77_int* incx, f77_int* rval)
{
    izaminsub_blis_impl( n, x, incx, rval);
}

void SASUMSUB( const f77_int* n, const float* x, const f77_int* incx, float* rval)
{
    sasumsub_blis_impl( n, x, incx, rval);
}

void sasumsub( const f77_int* n, const float* x, const f77_int* incx, float* rval)
{
    sasumsub_blis_impl( n, x, incx, rval);
}

void SASUMSUB_( const f77_int* n, const float* x, const f77_int* incx, float* rval)
{
    sasumsub_blis_impl( n, x, incx, rval);
}

#endif // BLIS_ENABLE_CBLAS

void SAXPBY( const f77_int* n, const float* alpha, const float *x, const f77_int* incx, const float* beta, float *y, const f77_int* incy)
{
    saxpby_blis_impl( n, alpha, x, incx, beta, y, incy);
}

void saxpby( const f77_int* n, const float* alpha, const float *x, const f77_int* incx, const float* beta, float *y, const f77_int* incy)
{
    saxpby_blis_impl( n, alpha, x, incx, beta, y, incy);
}

void SAXPBY_( const f77_int* n, const float* alpha, const float *x, const f77_int* incx, const float* beta, float *y, const f77_int* incy)
{
    saxpby_blis_impl( n, alpha, x, incx, beta, y, incy);
}

#ifdef BLIS_ENABLE_CBLAS

void SCASUMSUB( const f77_int* n, const scomplex* x, const f77_int* incx, float* rval)
{
    scasumsub_blis_impl( n, x, incx, rval);
}

void scasumsub( const f77_int* n, const scomplex* x, const f77_int* incx, float* rval)
{
    scasumsub_blis_impl( n, x, incx, rval);
}

void SCASUMSUB_( const f77_int* n, const scomplex* x, const f77_int* incx, float* rval)
{
    scasumsub_blis_impl( n, x, incx, rval);
}

void SCNRM2SUB( const f77_int* n, const scomplex* x, const f77_int* incx, float* rval)
{
    scnrm2sub_blis_impl( n, x, incx, rval);
}

void scnrm2sub( const f77_int* n, const scomplex* x, const f77_int* incx, float* rval)
{
    scnrm2sub_blis_impl( n, x, incx, rval);
}

void SCNRM2SUB_( const f77_int* n, const scomplex* x, const f77_int* incx, float* rval)
{
    scnrm2sub_blis_impl( n, x, incx, rval);
}

void SDOTSUB( const f77_int* n, const float* x, const f77_int* incx, const float* y, const f77_int* incy, float* rval)
{
    sdotsub_blis_impl( n, x, incx, y, incy, rval);
}

void sdotsub( const f77_int* n, const float* x, const f77_int* incx, const float* y, const f77_int* incy, float* rval)
{
    sdotsub_blis_impl( n, x, incx, y, incy, rval);
}

void SDOTSUB_( const f77_int* n, const float* x, const f77_int* incx, const float* y, const f77_int* incy, float* rval)
{
    sdotsub_blis_impl( n, x, incx, y, incy, rval);
}

#endif // BLIS_ENABLE_CBLAS

void SGEMM_BATCH(const f77_char* transa_array, const f77_char* transb_array,const f77_int *m_array, const f77_int *n_array, const f77_int *k_array,const float* alpha_array, const float** a_array, const  f77_int *lda_array, const float** b_array, const f77_int *ldb_array, const float* beta_array, float** c_array, const f77_int *ldc_array, const f77_int* group_count, const f77_int *group_size)
{
    sgemm_batch_blis_impl( transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

void sgemm_batch(const f77_char* transa_array, const f77_char* transb_array,const f77_int *m_array, const f77_int *n_array, const f77_int *k_array,const float* alpha_array, const float** a_array, const  f77_int *lda_array, const float** b_array, const f77_int *ldb_array, const float* beta_array, float** c_array, const f77_int *ldc_array, const f77_int* group_count, const f77_int *group_size)
{
    sgemm_batch_blis_impl( transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

void SGEMM_BATCH_(const f77_char* transa_array, const f77_char* transb_array,const f77_int *m_array, const f77_int *n_array, const f77_int *k_array,const float* alpha_array, const float** a_array, const  f77_int *lda_array, const float** b_array, const f77_int *ldb_array, const float* beta_array, float** c_array, const f77_int *ldc_array, const f77_int* group_count, const f77_int *group_size)
{
    sgemm_batch_blis_impl( transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

f77_int SGEMM_PACK_GET_SIZE(const f77_char* identifier, const f77_int* pm, const f77_int* pn, const f77_int* pk)
{
    return sgemm_pack_get_size_blis_impl( identifier, pm, pn, pk );
}

f77_int sgemm_pack_get_size(const f77_char* identifier, const f77_int* pm, const f77_int* pn, const f77_int* pk)
{
    return sgemm_pack_get_size_blis_impl( identifier, pm, pn, pk );
}

f77_int SGEMM_PACK_GET_SIZE_(const f77_char* identifier, const f77_int* pm, const f77_int* pn, const f77_int* pk)
{
    return sgemm_pack_get_size_blis_impl( identifier, pm, pn, pk );
}

void SGEMM_PACK( const f77_char* identifier, const f77_char* trans, const f77_int* mm, const f77_int* nn, const f77_int* kk, const float* alpha, const float* src, const f77_int* pld, float* dest )
{
    sgemm_pack_blis_impl( identifier, trans, mm, nn, kk, alpha, src, pld, dest );
}

void sgemm_pack( const f77_char* identifier, const f77_char* trans, const f77_int* mm, const f77_int* nn, const f77_int* kk, const float* alpha, const float* src, const f77_int* pld, float* dest )
{
    sgemm_pack_blis_impl( identifier, trans, mm, nn, kk, alpha, src, pld, dest );
}

void SGEMM_PACK_( const f77_char* identifier, const f77_char* trans, const f77_int* mm, const f77_int* nn, const f77_int* kk, const float* alpha, const float* src, const f77_int* pld, float* dest )
{
    sgemm_pack_blis_impl( identifier, trans, mm, nn, kk, alpha, src, pld, dest );
}

void SGEMM_COMPUTE( const f77_char* transa, const f77_char* transb, const f77_int* m, const f77_int* n, const f77_int* k, const float* a, const f77_int* lda, const float* b, const f77_int* ldb, const float* beta, float* c, const f77_int* ldc )
{
    f77_int rs_a = 1;
    f77_int rs_b = 1;
    f77_int rs_c = 1;
    sgemm_compute_blis_impl( transa, transb, m, n, k, a, &rs_a, lda, b, &rs_b, ldb, beta, c, &rs_c, ldc );
}

void sgemm_compute( const f77_char* transa, const f77_char* transb, const f77_int* m, const f77_int* n, const f77_int* k, const float* a, const f77_int* lda, const float* b, const f77_int* ldb, const float* beta, float* c, const f77_int* ldc )
{
    f77_int rs_a = 1;
    f77_int rs_b = 1;
    f77_int rs_c = 1;
    sgemm_compute_blis_impl( transa, transb, m, n, k, a, &rs_a, lda, b, &rs_b, ldb, beta, c, &rs_c, ldc );
}

void SGEMM_COMPUTE_( const f77_char* transa, const f77_char* transb, const f77_int* m, const f77_int* n, const f77_int* k, const float* a, const f77_int* lda, const float* b, const f77_int* ldb, const float* beta, float* c, const f77_int* ldc )
{
    f77_int rs_a = 1;
    f77_int rs_b = 1;
    f77_int rs_c = 1;
    sgemm_compute_blis_impl( transa, transb, m, n, k, a, &rs_a, lda, b, &rs_b, ldb, beta, c, &rs_c, ldc );
}

void SGEMMT( const f77_char* uploc, const f77_char* transa, const f77_char* transb, const f77_int* n, const f77_int* k, const  float* alpha, const float* a, const f77_int* lda, const float* b, const f77_int* ldb, const float* beta, float* c, const f77_int* ldc)
{
    sgemmt_blis_impl( uploc, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void sgemmt( const f77_char* uploc, const f77_char* transa, const f77_char* transb, const f77_int* n, const f77_int* k, const  float* alpha, const float* a, const f77_int* lda, const float* b, const f77_int* ldb, const float* beta, float* c, const f77_int* ldc)
{
    sgemmt_blis_impl( uploc, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void SGEMMT_( const f77_char* uploc, const f77_char* transa, const f77_char* transb, const f77_int* n, const f77_int* k, const  float* alpha, const float* a, const f77_int* lda, const float* b, const f77_int* ldb, const float* beta, float* c, const f77_int* ldc)
{
    sgemmt_blis_impl( uploc, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

//#ifdef BLIS_ENABLE_CBLAS

void SIMATCOPY( f77_char* trans, f77_int* rows, f77_int* cols, const float* alpha,float* aptr, f77_int* lda, f77_int* ldb)
{
    simatcopy_( trans, rows, cols, alpha, aptr, lda, ldb);
}

void simatcopy( f77_char* trans, f77_int* rows, f77_int* cols, const float* alpha,float* aptr, f77_int* lda, f77_int* ldb)
{
    simatcopy_( trans, rows, cols, alpha, aptr, lda, ldb);
}

void SIMATCOPY_( f77_char* trans, f77_int* rows, f77_int* cols, const float* alpha,float* aptr, f77_int* lda, f77_int* ldb)
{
    simatcopy_( trans, rows, cols, alpha, aptr, lda, ldb);
}

//#endif // BLIS_ENABLE_CBLAS

#ifdef BLIS_ENABLE_CBLAS

void SNRM2SUB( const f77_int* n, const float* x, const f77_int* incx, float *rval)
{
    snrm2sub_blis_impl( n, x, incx, rval);
}

void snrm2sub( const f77_int* n, const float* x, const f77_int* incx, float *rval)
{
    snrm2sub_blis_impl( n, x, incx, rval);
}

void SNRM2SUB_( const f77_int* n, const float* x, const f77_int* incx, float *rval)
{
    snrm2sub_blis_impl( n, x, incx, rval);
}

#endif // BLIS_ENABLE_CBLAS

//#ifdef BLIS_ENABLE_CBLAS

void SOMATADD( f77_char* transa,f77_char* transb, f77_int* m, f77_int* n, const float* alpha, const float* A, f77_int* lda, const float* beta, const float* B, f77_int* ldb, float* C, f77_int* ldc)
{
    somatadd_( transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

void somatadd( f77_char* transa,f77_char* transb, f77_int* m, f77_int* n, const float* alpha, const float* A, f77_int* lda, const float* beta, const float* B, f77_int* ldb, float* C, f77_int* ldc)
{
    somatadd_( transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

void SOMATADD_( f77_char* transa,f77_char* transb, f77_int* m, f77_int* n, const float* alpha, const float* A, f77_int* lda, const float* beta, const float* B, f77_int* ldb, float* C, f77_int* ldc)
{
    somatadd_( transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

void SOMATCOPY2( f77_char* trans, f77_int* rows, f77_int* cols, const float* alpha, const float* aptr, f77_int* lda,f77_int* stridea, float* bptr, f77_int* ldb,f77_int* strideb)
{
    somatcopy2_( trans, rows, cols, alpha, aptr, lda, stridea, bptr, ldb, strideb);
}

void somatcopy2( f77_char* trans, f77_int* rows, f77_int* cols, const float* alpha, const float* aptr, f77_int* lda,f77_int* stridea, float* bptr, f77_int* ldb,f77_int* strideb)
{
    somatcopy2_( trans, rows, cols, alpha, aptr, lda, stridea, bptr, ldb, strideb);
}

void SOMATCOPY2_( f77_char* trans, f77_int* rows, f77_int* cols, const float* alpha, const float* aptr, f77_int* lda,f77_int* stridea, float* bptr, f77_int* ldb,f77_int* strideb)
{
    somatcopy2_( trans, rows, cols, alpha, aptr, lda, stridea, bptr, ldb, strideb);
}

void SOMATCOPY( f77_char* trans, f77_int* rows, f77_int* cols, const float* alpha, const float* aptr, f77_int* lda, float* bptr, f77_int* ldb)
{
    somatcopy_( trans, rows, cols, alpha, aptr, lda, bptr, ldb);
}

void somatcopy( f77_char* trans, f77_int* rows, f77_int* cols, const float* alpha, const float* aptr, f77_int* lda, float* bptr, f77_int* ldb)
{
    somatcopy_( trans, rows, cols, alpha, aptr, lda, bptr, ldb);
}

void SOMATCOPY_( f77_char* trans, f77_int* rows, f77_int* cols, const float* alpha, const float* aptr, f77_int* lda, float* bptr, f77_int* ldb)
{
    somatcopy_( trans, rows, cols, alpha, aptr, lda, bptr, ldb);
}

//#endif // BLIS_ENABLE_CBLAS

void ZAXPBY( const f77_int* n, const dcomplex* alpha, const dcomplex *x, const f77_int* incx, const dcomplex* beta, dcomplex *y, const f77_int* incy)
{
    zaxpby_blis_impl( n, alpha, x, incx, beta, y, incy);
}

void zaxpby( const f77_int* n, const dcomplex* alpha, const dcomplex *x, const f77_int* incx, const dcomplex* beta, dcomplex *y, const f77_int* incy)
{
    zaxpby_blis_impl( n, alpha, x, incx, beta, y, incy);
}

void ZAXPBY_( const f77_int* n, const dcomplex* alpha, const dcomplex *x, const f77_int* incx, const dcomplex* beta, dcomplex *y, const f77_int* incy)
{
    zaxpby_blis_impl( n, alpha, x, incx, beta, y, incy);
}

#ifdef BLIS_ENABLE_CBLAS

void ZDOTCSUB( const f77_int* n, const dcomplex* x, const f77_int* incx, const dcomplex* y, const f77_int* incy, dcomplex* rval)
{
    zdotcsub_blis_impl( n, x, incx, y, incy, rval);
}

void zdotcsub( const f77_int* n, const dcomplex* x, const f77_int* incx, const dcomplex* y, const f77_int* incy, dcomplex* rval)
{
    zdotcsub_blis_impl( n, x, incx, y, incy, rval);
}

void ZDOTCSUB_( const f77_int* n, const dcomplex* x, const f77_int* incx, const dcomplex* y, const f77_int* incy, dcomplex* rval)
{
    zdotcsub_blis_impl( n, x, incx, y, incy, rval);
}

void ZDOTUSUB( const f77_int* n, const dcomplex* x, const f77_int* incx,const dcomplex* y, const f77_int* incy, dcomplex* rval)
{
    zdotusub_blis_impl( n, x, incx, y, incy, rval);
}

void zdotusub( const f77_int* n, const dcomplex* x, const f77_int* incx,const dcomplex* y, const f77_int* incy, dcomplex* rval)
{
    zdotusub_blis_impl( n, x, incx, y, incy, rval);
}

void ZDOTUSUB_( const f77_int* n, const dcomplex* x, const f77_int* incx,const dcomplex* y, const f77_int* incy, dcomplex* rval)
{
    zdotusub_blis_impl( n, x, incx, y, incy, rval);
}

#endif // BLIS_ENABLE_CBLAS

void ZGEMM3M( const f77_char* transa, const f77_char* transb, const f77_int* m, const f77_int* n, const f77_int* k, const dcomplex* alpha, const dcomplex* a, const f77_int* lda, const dcomplex* b, const f77_int* ldb, const dcomplex* beta, dcomplex* c, const f77_int* ldc)
{
    zgemm3m_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void zgemm3m( const f77_char* transa, const f77_char* transb, const f77_int* m, const f77_int* n, const f77_int* k, const dcomplex* alpha, const dcomplex* a, const f77_int* lda, const dcomplex* b, const f77_int* ldb, const dcomplex* beta, dcomplex* c, const f77_int* ldc)
{
    zgemm3m_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ZGEMM3M_( const f77_char* transa, const f77_char* transb, const f77_int* m, const f77_int* n, const f77_int* k, const dcomplex* alpha, const dcomplex* a, const f77_int* lda, const dcomplex* b, const f77_int* ldb, const dcomplex* beta, dcomplex* c, const f77_int* ldc)
{
    zgemm3m_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ZGEMM_BATCH(  const f77_char* transa_array, const f77_char* transb_array,const f77_int *m_array, const f77_int *n_array, const f77_int *k_array,const dcomplex* alpha_array, const dcomplex** a_array, const  f77_int *lda_array, const dcomplex** b_array, const f77_int *ldb_array, const dcomplex* beta_array, dcomplex** c_array, const f77_int *ldc_array, const f77_int* group_count, const f77_int *group_size)
{
    zgemm_batch_blis_impl( transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

void zgemm_batch(  const f77_char* transa_array, const f77_char* transb_array,const f77_int *m_array, const f77_int *n_array, const f77_int *k_array,const dcomplex* alpha_array, const dcomplex** a_array, const  f77_int *lda_array, const dcomplex** b_array, const f77_int *ldb_array, const dcomplex* beta_array, dcomplex** c_array, const f77_int *ldc_array, const f77_int* group_count, const f77_int *group_size)
{
    zgemm_batch_blis_impl( transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

void ZGEMM_BATCH_(  const f77_char* transa_array, const f77_char* transb_array,const f77_int *m_array, const f77_int *n_array, const f77_int *k_array,const dcomplex* alpha_array, const dcomplex** a_array, const  f77_int *lda_array, const dcomplex** b_array, const f77_int *ldb_array, const dcomplex* beta_array, dcomplex** c_array, const f77_int *ldc_array, const f77_int* group_count, const f77_int *group_size)
{
    zgemm_batch_blis_impl( transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

void ZGEMMT( const f77_char* uploc, const f77_char* transa, const f77_char* transb, const f77_int* n, const f77_int* k, const  dcomplex* alpha, const dcomplex* a, const f77_int* lda, const dcomplex* b, const f77_int* ldb, const dcomplex* beta, dcomplex* c, const f77_int* ldc)
{
    zgemmt_blis_impl( uploc, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void zgemmt( const f77_char* uploc, const f77_char* transa, const f77_char* transb, const f77_int* n, const f77_int* k, const  dcomplex* alpha, const dcomplex* a, const f77_int* lda, const dcomplex* b, const f77_int* ldb, const dcomplex* beta, dcomplex* c, const f77_int* ldc)
{
    zgemmt_blis_impl( uploc, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ZGEMMT_( const f77_char* uploc, const f77_char* transa, const f77_char* transb, const f77_int* n, const f77_int* k, const  dcomplex* alpha, const dcomplex* a, const f77_int* lda, const dcomplex* b, const f77_int* ldb, const dcomplex* beta, dcomplex* c, const f77_int* ldc)
{
    zgemmt_blis_impl( uploc, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

//#ifdef BLIS_ENABLE_CBLAS

void ZIMATCOPY(f77_char* trans, f77_int* rows, f77_int* cols, const dcomplex* alpha,dcomplex* aptr, f77_int* lda, f77_int* ldb)
{
    zimatcopy_( trans, rows, cols, alpha, aptr, lda, ldb);
}

void zimatcopy(f77_char* trans, f77_int* rows, f77_int* cols, const dcomplex* alpha,dcomplex* aptr, f77_int* lda, f77_int* ldb)
{
    zimatcopy_( trans, rows, cols, alpha, aptr, lda, ldb);
}

void ZIMATCOPY_(f77_char* trans, f77_int* rows, f77_int* cols, const dcomplex* alpha,dcomplex* aptr, f77_int* lda, f77_int* ldb)
{
    zimatcopy_( trans, rows, cols, alpha, aptr, lda, ldb);
}

void ZOMATADD(f77_char* transa,f77_char* transb, f77_int* m, f77_int* n, const dcomplex* alpha, const dcomplex* A, f77_int* lda,const dcomplex* beta, dcomplex* B, f77_int* ldb, dcomplex* C, f77_int* ldc)
{
    zomatadd_( transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

void zomatadd(f77_char* transa,f77_char* transb, f77_int* m, f77_int* n, const dcomplex* alpha, const dcomplex* A, f77_int* lda,const dcomplex* beta, dcomplex* B, f77_int* ldb, dcomplex* C, f77_int* ldc)
{
    zomatadd_( transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

void ZOMATADD_(f77_char* transa,f77_char* transb, f77_int* m, f77_int* n, const dcomplex* alpha, const dcomplex* A, f77_int* lda,const dcomplex* beta, dcomplex* B, f77_int* ldb, dcomplex* C, f77_int* ldc)
{
    zomatadd_( transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

void ZOMATCOPY2(f77_char* trans, f77_int* rows, f77_int* cols, const dcomplex* alpha, const dcomplex* aptr, f77_int* lda,f77_int* stridea, dcomplex* bptr, f77_int* ldb,f77_int* strideb)
{
    zomatcopy2_( trans, rows, cols, alpha, aptr, lda, stridea, bptr, ldb, strideb);
}

void zomatcopy2(f77_char* trans, f77_int* rows, f77_int* cols, const dcomplex* alpha, const dcomplex* aptr, f77_int* lda,f77_int* stridea, dcomplex* bptr, f77_int* ldb,f77_int* strideb)
{
    zomatcopy2_( trans, rows, cols, alpha, aptr, lda, stridea, bptr, ldb, strideb);
}

void ZOMATCOPY2_(f77_char* trans, f77_int* rows, f77_int* cols, const dcomplex* alpha, const dcomplex* aptr, f77_int* lda,f77_int* stridea, dcomplex* bptr, f77_int* ldb,f77_int* strideb)
{
    zomatcopy2_( trans, rows, cols, alpha, aptr, lda, stridea, bptr, ldb, strideb);
}

void ZOMATCOPY(f77_char* trans, f77_int* rows, f77_int* cols, const dcomplex* alpha, const dcomplex* aptr, f77_int* lda, dcomplex* bptr, f77_int* ldb)
{
    zomatcopy_( trans, rows, cols, alpha, aptr, lda, bptr, ldb);
}

void zomatcopy(f77_char* trans, f77_int* rows, f77_int* cols, const dcomplex* alpha, const dcomplex* aptr, f77_int* lda, dcomplex* bptr, f77_int* ldb)
{
    zomatcopy_( trans, rows, cols, alpha, aptr, lda, bptr, ldb);
}

void ZOMATCOPY_(f77_char* trans, f77_int* rows, f77_int* cols, const dcomplex* alpha, const dcomplex* aptr, f77_int* lda, dcomplex* bptr, f77_int* ldb)
{
    zomatcopy_( trans, rows, cols, alpha, aptr, lda, bptr, ldb);
}

//#endif // BLIS_ENABLE_CBLAS

float SCABS1(bla_scomplex* z)
{
    return scabs1_blis_impl( z);
}

float scabs1(bla_scomplex* z)
{
    return scabs1_blis_impl( z);
}

float SCABS1_(bla_scomplex* z)
{
    return scabs1_blis_impl( z);

}

#ifdef BLIS_ENABLE_CBLAS

void SDSDOTSUB( const f77_int* n, float* sb, const float* x, const f77_int* incx, const float* y, const f77_int* incy, float* dot)
{
    sdsdotsub_blis_impl( n, sb, x, incx, y, incy, dot);
}

void sdsdotsub( const f77_int* n, float* sb, const float* x, const f77_int* incx, const float* y, const f77_int* incy, float* dot)
{
    sdsdotsub_blis_impl( n, sb, x, incx, y, incy, dot);
}

void SDSDOTSUB_( const f77_int* n, float* sb, const float* x, const f77_int* incx, const float* y, const f77_int* incy, float* dot)
{
    sdsdotsub_blis_impl( n, sb, x, incx, y, incy, dot);
}

void DSDOTSUB( const f77_int* n, const float* x, const f77_int* incx, const float* y, const f77_int* incy, double* dot)
{
    dsdotsub_blis_impl( n, x, incx, y, incy, dot);
}

void dsdotsub( const f77_int* n, const float* x, const f77_int* incx, const float* y, const f77_int* incy, double* dot)
{
    dsdotsub_blis_impl( n, x, incx, y, incy, dot);
}

void DSDOTSUB_( const f77_int* n, const float* x, const f77_int* incx, const float* y, const f77_int* incy, double* dot)
{
    dsdotsub_blis_impl( n, x, incx, y, incy, dot);
}

#endif // BLIS_ENABLE_CBLAS

void CAXPBY( const f77_int* n,  const scomplex* alpha,  const scomplex *x,  const f77_int* incx,  const scomplex* beta,  scomplex *y,  const f77_int* incy)
{
    caxpby_blis_impl(n, alpha, x, incx, beta, y, incy);
}

void caxpby( const f77_int* n,  const scomplex* alpha,  const scomplex *x,  const f77_int* incx,  const scomplex* beta,  scomplex *y,  const f77_int* incy)
{
    caxpby_blis_impl(n, alpha, x, incx, beta, y, incy);
}

void CAXPBY_( const f77_int* n,  const scomplex* alpha,  const scomplex *x,  const f77_int* incx,  const scomplex* beta,  scomplex *y,  const f77_int* incy)
{
    caxpby_blis_impl(n, alpha, x, incx, beta, y, incy);
}

#endif
#endif

#endif // BLIS_ENABLE_BLAS
