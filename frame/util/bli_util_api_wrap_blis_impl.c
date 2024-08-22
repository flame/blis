/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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

// wrapper functions to support additional symbols
#ifndef BLIS_ENABLE_NO_UNDERSCORE_API
#ifndef BLIS_ENABLE_UPPERCASE_API
void CAXPY_BLIS_IMPL(const f77_int *n,const scomplex  *ca,const scomplex  *cx,const f77_int *incx,scomplex  *cy,const f77_int *incy)
{
    caxpy_blis_impl( n, ca, cx, incx, cy, incy);
}

void caxpy_blis_impl_(const f77_int *n,const scomplex  *ca,const scomplex  *cx,const f77_int *incx,scomplex  *cy,const f77_int *incy)
{
    caxpy_blis_impl( n, ca, cx, incx, cy, incy);
}

void CAXPY_BLIS_IMPL_(const f77_int *n,const scomplex  *ca,const scomplex  *cx,const f77_int *incx,scomplex  *cy,const f77_int *incy)
{
    caxpy_blis_impl( n, ca, cx, incx, cy, incy);
}

void CCOPY_BLIS_IMPL(const f77_int *n,const scomplex  *cx,const f77_int *incx,scomplex  *cy,const f77_int *incy)
{
    ccopy_blis_impl( n, cx, incx, cy, incy);
}

void ccopy_blis_impl_(const f77_int *n,const scomplex  *cx,const f77_int *incx,scomplex  *cy,const f77_int *incy)
{
    ccopy_blis_impl( n, cx, incx, cy, incy);
}

void CCOPY_BLIS_IMPL_(const f77_int *n,const scomplex  *cx,const f77_int *incx,scomplex  *cy,const f77_int *incy)
{
    ccopy_blis_impl( n, cx, incx, cy, incy);
}

#ifdef BLIS_DISABLE_COMPLEX_RETURN_INTEL
scomplex CDOTC_BLIS_IMPL(const f77_int* n,const scomplex*   x, const f77_int* incx,const scomplex*   y, const f77_int* incy)
{
    return cdotc_blis_impl( n, x, incx, y, incy);
}

scomplex cdotc_blis_impl_(const f77_int* n,const scomplex*   x, const f77_int* incx,const scomplex*   y, const f77_int* incy)
{
    return cdotc_blis_impl( n, x, incx, y, incy);
}

scomplex CDOTC_BLIS_IMPL_(const f77_int* n,const scomplex*   x, const f77_int* incx,const scomplex*   y, const f77_int* incy)
{
    return cdotc_blis_impl( n, x, incx, y, incy);
}

scomplex CDOTU_BLIS_IMPL(const f77_int* n,const scomplex*   x, const f77_int* incx,const scomplex*   y, const f77_int* incy)
{
    return cdotu_blis_impl( n, x, incx, y, incy);
}

scomplex cdotu_blis_impl_(const f77_int* n,const scomplex*   x, const f77_int* incx,const scomplex*   y, const f77_int* incy)
{
    return cdotu_blis_impl( n, x, incx, y, incy);
}

scomplex CDOTU_BLIS_IMPL_(const f77_int* n,const scomplex*   x, const f77_int* incx,const scomplex*   y, const f77_int* incy)
{
    return cdotu_blis_impl( n, x, incx, y, incy);
}

dcomplex ZDOTC_BLIS_IMPL(const f77_int* n, const dcomplex*   x, const f77_int* incx, const dcomplex*   y, const f77_int* incy)
{
    return zdotc_blis_impl( n, x, incx, y, incy);
}

dcomplex zdotc_blis_impl_(const f77_int* n, const dcomplex*   x, const f77_int* incx, const dcomplex*   y, const f77_int* incy)
{
    return zdotc_blis_impl( n, x, incx, y, incy);
}

dcomplex ZDOTC_BLIS_IMPL_(const f77_int* n, const dcomplex*   x, const f77_int* incx, const dcomplex*   y, const f77_int* incy)
{
    return zdotc_blis_impl( n, x, incx, y, incy);
}

dcomplex ZDOTU_BLIS_IMPL(const f77_int* n, const dcomplex*   x, const f77_int* incx, const dcomplex*   y, const f77_int* incy)
{
    return zdotu_blis_impl( n, x, incx, y, incy);
}

dcomplex zdotu_blis_impl_(const f77_int* n, const dcomplex*   x, const f77_int* incx, const dcomplex*   y, const f77_int* incy)
{
    return zdotu_blis_impl( n, x, incx, y, incy);
}

dcomplex ZDOTU_BLIS_IMPL_(const f77_int* n, const dcomplex*   x, const f77_int* incx, const dcomplex*   y, const f77_int* incy)
{
    return zdotu_blis_impl( n, x, incx, y, incy);
}
#else
void CDOTC_BLIS_IMPL(scomplex* retval,const f77_int *n, const scomplex  *cx, const f77_int *incx, const scomplex  *cy, const f77_int *incy)
{
    cdotc_blis_impl( retval, n, cx, incx, cy, incy);
}

void cdotc_blis_impl_(scomplex* retval,const f77_int *n, const scomplex  *cx, const f77_int *incx, const scomplex  *cy, const f77_int *incy)
{
    cdotc_blis_impl( retval, n, cx, incx, cy, incy);
}

void CDOTC_BLIS_IMPL_(scomplex* retval,const f77_int *n, const scomplex  *cx, const f77_int *incx, const scomplex  *cy, const f77_int *incy)
{
    cdotc_blis_impl( retval, n, cx, incx, cy, incy);
}

void CDOTU_BLIS_IMPL(scomplex* retval,const f77_int *n, const scomplex  *cx, const f77_int *incx, const scomplex  *cy, const f77_int *incy)
{
    cdotu_blis_impl( retval, n, cx, incx, cy, incy);
}

void cdotu_blis_impl_(scomplex* retval,const f77_int *n, const scomplex  *cx, const f77_int *incx, const scomplex  *cy, const f77_int *incy)
{
    cdotu_blis_impl( retval, n, cx, incx, cy, incy);
}

void CDOTU_BLIS_IMPL_(scomplex* retval,const f77_int *n, const scomplex  *cx, const f77_int *incx, const scomplex  *cy, const f77_int *incy)
{
    cdotu_blis_impl( retval, n, cx, incx, cy, incy);
}

void ZDOTC_BLIS_IMPL(dcomplex* retval,const f77_int *n, const dcomplex *zx, const f77_int *incx, const dcomplex *zy, const f77_int *incy)
{
    zdotc_blis_impl( retval, n, zx, incx, zy, incy);
}

void zdotc_blis_impl_(dcomplex* retval,const f77_int *n, const dcomplex *zx, const f77_int *incx, const dcomplex *zy, const f77_int *incy)
{
    zdotc_blis_impl( retval, n, zx, incx, zy, incy);
}

void ZDOTC_BLIS_IMPL_(dcomplex* retval,const f77_int *n, const dcomplex *zx, const f77_int *incx, const dcomplex *zy, const f77_int *incy)
{
    zdotc_blis_impl( retval, n, zx, incx, zy, incy);
}

void ZDOTU_BLIS_IMPL(dcomplex* retval,const f77_int *n, const dcomplex *zx, const f77_int *incx, const dcomplex *zy, const f77_int *incy)
{
    zdotu_blis_impl( retval, n, zx, incx, zy, incy);
}

void zdotu_blis_impl_(dcomplex* retval,const f77_int *n, const dcomplex *zx, const f77_int *incx, const dcomplex *zy, const f77_int *incy)
{
    zdotu_blis_impl( retval, n, zx, incx, zy, incy);
}

void ZDOTU_BLIS_IMPL_(dcomplex* retval,const f77_int *n, const dcomplex *zx, const f77_int *incx, const dcomplex *zy, const f77_int *incy)
{
    zdotu_blis_impl( retval, n, zx, incx, zy, incy);
}
#endif

void CGBMV_BLIS_IMPL(const char   *trans,const f77_int *m,const f77_int *n,const f77_int *kl,const f77_int *ku,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *x,const f77_int *incx,const scomplex  *beta,scomplex  *y,const f77_int *incy)
{
    cgbmv_blis_impl( trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

void cgbmv_blis_impl_(const char   *trans,const f77_int *m,const f77_int *n,const f77_int *kl,const f77_int *ku,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *x,const f77_int *incx,const scomplex  *beta,scomplex  *y,const f77_int *incy)
{
    cgbmv_blis_impl( trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

void CGBMV_BLIS_IMPL_(const char   *trans,const f77_int *m,const f77_int *n,const f77_int *kl,const f77_int *ku,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *x,const f77_int *incx,const scomplex  *beta,scomplex  *y,const f77_int *incy)
{
    cgbmv_blis_impl( trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

void CGEMM_BLIS_IMPL(const char   *transa,const char   *transb,const f77_int *m,const f77_int *n,const f77_int *k,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *b,const f77_int *ldb,const scomplex  *beta,scomplex  *c,const f77_int *ldc)
{
    cgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void cgemm_blis_impl_(const char   *transa,const char   *transb,const f77_int *m,const f77_int *n,const f77_int *k,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *b,const f77_int *ldb,const scomplex  *beta,scomplex  *c,const f77_int *ldc)
{
    cgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void CGEMM_BLIS_IMPL_(const char   *transa,const char   *transb,const f77_int *m,const f77_int *n,const f77_int *k,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *b,const f77_int *ldb,const scomplex  *beta,scomplex  *c,const f77_int *ldc)
{
    cgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void CGEMV_BLIS_IMPL(const char   *trans,const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *x,const f77_int *incx,const scomplex  *beta,scomplex  *y,const f77_int *incy)
{
    cgemv_blis_impl( trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void cgemv_blis_impl_(const char   *trans,const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *x,const f77_int *incx,const scomplex  *beta,scomplex  *y,const f77_int *incy)
{
    cgemv_blis_impl( trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void CGEMV_BLIS_IMPL_(const char   *trans,const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *x,const f77_int *incx,const scomplex  *beta,scomplex  *y,const f77_int *incy)
{
    cgemv_blis_impl( trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void CGERC_BLIS_IMPL(const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *x,const f77_int *incx,const scomplex  *y,const f77_int *incy,scomplex  *a,const f77_int *lda)
{
    cgerc_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void cgerc_blis_impl_(const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *x,const f77_int *incx,const scomplex  *y,const f77_int *incy,scomplex  *a,const f77_int *lda)
{
    cgerc_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void CGERC_BLIS_IMPL_(const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *x,const f77_int *incx,const scomplex  *y,const f77_int *incy,scomplex  *a,const f77_int *lda)
{
    cgerc_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void CGERU_BLIS_IMPL(const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *x,const f77_int *incx,const scomplex  *y,const f77_int *incy,scomplex  *a,const f77_int *lda)
{
    cgeru_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void cgeru_blis_impl_(const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *x,const f77_int *incx,const scomplex  *y,const f77_int *incy,scomplex  *a,const f77_int *lda)
{
    cgeru_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void CGERU_BLIS_IMPL_(const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *x,const f77_int *incx,const scomplex  *y,const f77_int *incy,scomplex  *a,const f77_int *lda)
{
    cgeru_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void CHBMV_BLIS_IMPL(const char   *uplo,const f77_int *n,const f77_int *k,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *x,const f77_int *incx,const scomplex  *beta,scomplex  *y,const f77_int *incy)
{
    chbmv_blis_impl( uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void chbmv_blis_impl_(const char   *uplo,const f77_int *n,const f77_int *k,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *x,const f77_int *incx,const scomplex  *beta,scomplex  *y,const f77_int *incy)
{
    chbmv_blis_impl( uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void CHBMV_BLIS_IMPL_(const char   *uplo,const f77_int *n,const f77_int *k,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *x,const f77_int *incx,const scomplex  *beta,scomplex  *y,const f77_int *incy)
{
    chbmv_blis_impl( uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void CHEMM_BLIS_IMPL(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *b,const f77_int *ldb,const scomplex  *beta,scomplex  *c,const f77_int *ldc)
{
    chemm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void chemm_blis_impl_(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *b,const f77_int *ldb,const scomplex  *beta,scomplex  *c,const f77_int *ldc)
{
    chemm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void CHEMM_BLIS_IMPL_(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *b,const f77_int *ldb,const scomplex  *beta,scomplex  *c,const f77_int *ldc)
{
    chemm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void CHEMV_BLIS_IMPL(const char   *uplo,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *x,const f77_int *incx,const scomplex  *beta,scomplex  *y,const f77_int *incy)
{
    chemv_blis_impl( uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

void chemv_blis_impl_(const char   *uplo,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *x,const f77_int *incx,const scomplex  *beta,scomplex  *y,const f77_int *incy)
{
    chemv_blis_impl( uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

void CHEMV_BLIS_IMPL_(const char   *uplo,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *x,const f77_int *incx,const scomplex  *beta,scomplex  *y,const f77_int *incy)
{
    chemv_blis_impl( uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

void CHER_BLIS_IMPL(const char   *uplo,const f77_int *n,const float  *alpha,const scomplex  *x,const f77_int *incx,scomplex  *a,const f77_int *lda)
{
    cher_blis_impl( uplo, n, alpha, x, incx, a, lda);
}

void cher_blis_impl_(const char   *uplo,const f77_int *n,const float  *alpha,const scomplex  *x,const f77_int *incx,scomplex  *a,const f77_int *lda)
{
    cher_blis_impl( uplo, n, alpha, x, incx, a, lda);
}

void CHER_BLIS_IMPL_(const char   *uplo,const f77_int *n,const float  *alpha,const scomplex  *x,const f77_int *incx,scomplex  *a,const f77_int *lda)
{
    cher_blis_impl( uplo, n, alpha, x, incx, a, lda);
}

void CHER2_BLIS_IMPL(const char   *uplo,const f77_int *n,const scomplex  *alpha,const scomplex  *x,const f77_int *incx,const scomplex  *y,const f77_int *incy,scomplex  *a,const f77_int *lda)
{
    cher2_blis_impl( uplo, n, alpha, x, incx, y, incy, a, lda);
}

void cher2_blis_impl_(const char   *uplo,const f77_int *n,const scomplex  *alpha,const scomplex  *x,const f77_int *incx,const scomplex  *y,const f77_int *incy,scomplex  *a,const f77_int *lda)
{
    cher2_blis_impl( uplo, n, alpha, x, incx, y, incy, a, lda);
}

void CHER2_BLIS_IMPL_(const char   *uplo,const f77_int *n,const scomplex  *alpha,const scomplex  *x,const f77_int *incx,const scomplex  *y,const f77_int *incy,scomplex  *a,const f77_int *lda)
{
    cher2_blis_impl( uplo, n, alpha, x, incx, y, incy, a, lda);
}

void CHER2K_BLIS_IMPL(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *b,const f77_int *ldb,const float  *beta,scomplex  *c,const f77_int *ldc)
{
    cher2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void cher2k_blis_impl_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *b,const f77_int *ldb,const float  *beta,scomplex  *c,const f77_int *ldc)
{
    cher2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void CHER2K_BLIS_IMPL_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *b,const f77_int *ldb,const float  *beta,scomplex  *c,const f77_int *ldc)
{
    cher2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void CHERK_BLIS_IMPL(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const float  *alpha,const scomplex  *a,const f77_int *lda,const float  *beta,scomplex  *c,const f77_int *ldc)
{
    cherk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void cherk_blis_impl_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const float  *alpha,const scomplex  *a,const f77_int *lda,const float  *beta,scomplex  *c,const f77_int *ldc)
{
    cherk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void CHERK_BLIS_IMPL_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const float  *alpha,const scomplex  *a,const f77_int *lda,const float  *beta,scomplex  *c,const f77_int *ldc)
{
    cherk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void CHPMV_BLIS_IMPL(const char   *uplo,const f77_int *n,const scomplex  *alpha,const scomplex  *ap,const scomplex  *x,const f77_int *incx,const scomplex  *beta,scomplex  *y,const f77_int *incy)
{
    chpmv_blis_impl( uplo, n, alpha, ap, x, incx, beta, y, incy);
}

void chpmv_blis_impl_(const char   *uplo,const f77_int *n,const scomplex  *alpha,const scomplex  *ap,const scomplex  *x,const f77_int *incx,const scomplex  *beta,scomplex  *y,const f77_int *incy)
{
    chpmv_blis_impl( uplo, n, alpha, ap, x, incx, beta, y, incy);
}

void CHPMV_BLIS_IMPL_(const char   *uplo,const f77_int *n,const scomplex  *alpha,const scomplex  *ap,const scomplex  *x,const f77_int *incx,const scomplex  *beta,scomplex  *y,const f77_int *incy)
{
    chpmv_blis_impl( uplo, n, alpha, ap, x, incx, beta, y, incy);
}

void CHPR_BLIS_IMPL(const char   *uplo,const f77_int *n,const float  *alpha,const scomplex  *x,const f77_int *incx,scomplex  *ap)
{
    chpr_blis_impl( uplo, n, alpha, x, incx, ap);
}

void chpr_blis_impl_(const char   *uplo,const f77_int *n,const float  *alpha,const scomplex  *x,const f77_int *incx,scomplex  *ap)
{
    chpr_blis_impl( uplo, n, alpha, x, incx, ap);
}

void CHPR_BLIS_IMPL_(const char   *uplo,const f77_int *n,const float  *alpha,const scomplex  *x,const f77_int *incx,scomplex  *ap)
{
    chpr_blis_impl( uplo, n, alpha, x, incx, ap);
}

void CHPR2_BLIS_IMPL(const char   *uplo,const f77_int *n,const scomplex  *alpha,const scomplex  *x,const f77_int *incx,const scomplex  *y,const f77_int *incy,scomplex  *ap)
{
    chpr2_blis_impl( uplo, n, alpha, x, incx, y, incy, ap);
}

void chpr2_blis_impl_(const char   *uplo,const f77_int *n,const scomplex  *alpha,const scomplex  *x,const f77_int *incx,const scomplex  *y,const f77_int *incy,scomplex  *ap)
{
    chpr2_blis_impl( uplo, n, alpha, x, incx, y, incy, ap);
}

void CHPR2_BLIS_IMPL_(const char   *uplo,const f77_int *n,const scomplex  *alpha,const scomplex  *x,const f77_int *incx,const scomplex  *y,const f77_int *incy,scomplex  *ap)
{
    chpr2_blis_impl( uplo, n, alpha, x, incx, y, incy, ap);
}

void CROTG_BLIS_IMPL(scomplex  *ca, bla_scomplex  *cb, bla_real  *c,scomplex  *s)
{
    crotg_blis_impl( ca, cb, c, s);
}

void crotg_blis_impl_(scomplex  *ca, bla_scomplex  *cb, bla_real  *c,scomplex  *s)
{
    crotg_blis_impl( ca, cb, c, s);
}

void CROTG_BLIS_IMPL_(scomplex  *ca, bla_scomplex  *cb, bla_real  *c,scomplex  *s)
{
    crotg_blis_impl( ca, cb, c, s);
}

void CSCAL_BLIS_IMPL(const f77_int *n,const scomplex  *ca,scomplex  *cx,const f77_int *incx)
{
    cscal_blis_impl( n, ca, cx, incx);
}

void cscal_blis_impl_(const f77_int *n,const scomplex  *ca,scomplex  *cx,const f77_int *incx)
{
    cscal_blis_impl( n, ca, cx, incx);
}

void CSCAL_BLIS_IMPL_(const f77_int *n,const scomplex  *ca,scomplex  *cx,const f77_int *incx)
{
    cscal_blis_impl( n, ca, cx, incx);
}

void CSROT_BLIS_IMPL(const f77_int *n,scomplex  *cx,const f77_int *incx,scomplex  *cy,const f77_int *incy,const float  *c,const float  *s)
{
    csrot_blis_impl( n, cx, incx, cy, incy, c, s);
}

void csrot_blis_impl_(const f77_int *n,scomplex  *cx,const f77_int *incx,scomplex  *cy,const f77_int *incy,const float  *c,const float  *s)
{
    csrot_blis_impl( n, cx, incx, cy, incy, c, s);
}

void CSROT_BLIS_IMPL_(const f77_int *n,scomplex  *cx,const f77_int *incx,scomplex  *cy,const f77_int *incy,const float  *c,const float  *s)
{
    csrot_blis_impl( n, cx, incx, cy, incy, c, s);
}

void CSSCAL_BLIS_IMPL(const f77_int *n,const float  *sa,scomplex  *cx,const f77_int *incx)
{
    csscal_blis_impl( n, sa, cx, incx);
}

void csscal_blis_impl_(const f77_int *n,const float  *sa,scomplex  *cx,const f77_int *incx)
{
    csscal_blis_impl( n, sa, cx, incx);
}

void CSSCAL_BLIS_IMPL_(const f77_int *n,const float  *sa,scomplex  *cx,const f77_int *incx)
{
    csscal_blis_impl( n, sa, cx, incx);
}

void CSWAP_BLIS_IMPL(const f77_int *n,scomplex  *cx,const f77_int *incx,scomplex  *cy,const f77_int *incy)
{
    cswap_blis_impl( n, cx, incx, cy, incy);
}

void cswap_blis_impl_(const f77_int *n,scomplex  *cx,const f77_int *incx,scomplex  *cy,const f77_int *incy)
{
    cswap_blis_impl( n, cx, incx, cy, incy);
}

void CSWAP_BLIS_IMPL_(const f77_int *n,scomplex  *cx,const f77_int *incx,scomplex  *cy,const f77_int *incy)
{
    cswap_blis_impl( n, cx, incx, cy, incy);
}

void CSYMM_BLIS_IMPL(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *b,const f77_int *ldb,const scomplex  *beta,scomplex  *c,const f77_int *ldc)
{
    csymm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void csymm_blis_impl_(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *b,const f77_int *ldb,const scomplex  *beta,scomplex  *c,const f77_int *ldc)
{
    csymm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void CSYMM_BLIS_IMPL_(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *b,const f77_int *ldb,const scomplex  *beta,scomplex  *c,const f77_int *ldc)
{
    csymm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void CSYR2K_BLIS_IMPL(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *b,const f77_int *ldb,const scomplex  *beta,scomplex  *c,const f77_int *ldc)
{
    csyr2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void csyr2k_blis_impl_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *b,const f77_int *ldb,const scomplex  *beta,scomplex  *c,const f77_int *ldc)
{
    csyr2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void CSYR2K_BLIS_IMPL_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *b,const f77_int *ldb,const scomplex  *beta,scomplex  *c,const f77_int *ldc)
{
    csyr2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void CSYRK_BLIS_IMPL(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *beta,scomplex  *c,const f77_int *ldc)
{
    csyrk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void csyrk_blis_impl_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *beta,scomplex  *c,const f77_int *ldc)
{
    csyrk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void CSYRK_BLIS_IMPL_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,const scomplex  *beta,scomplex  *c,const f77_int *ldc)
{
    csyrk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void CTBMV_BLIS_IMPL(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const scomplex  *a,const f77_int *lda,scomplex  *x,const f77_int *incx)
{
    ctbmv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void ctbmv_blis_impl_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const scomplex  *a,const f77_int *lda,scomplex  *x,const f77_int *incx)
{
    ctbmv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void CTBMV_BLIS_IMPL_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const scomplex  *a,const f77_int *lda,scomplex  *x,const f77_int *incx)
{
    ctbmv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void CTBSV_BLIS_IMPL(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const scomplex  *a,const f77_int *lda,scomplex  *x,const f77_int *incx)
{
    ctbsv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void ctbsv_blis_impl_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const scomplex  *a,const f77_int *lda,scomplex  *x,const f77_int *incx)
{
    ctbsv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void CTBSV_BLIS_IMPL_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const scomplex  *a,const f77_int *lda,scomplex  *x,const f77_int *incx)
{
    ctbsv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void CTPMV_BLIS_IMPL(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const scomplex  *ap,scomplex  *x,const f77_int *incx)
{
    ctpmv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void ctpmv_blis_impl_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const scomplex  *ap,scomplex  *x,const f77_int *incx)
{
    ctpmv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void CTPMV_BLIS_IMPL_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const scomplex  *ap,scomplex  *x,const f77_int *incx)
{
    ctpmv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void CTPSV_BLIS_IMPL(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const scomplex  *ap,scomplex  *x,const f77_int *incx)
{
    ctpsv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void ctpsv_blis_impl_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const scomplex  *ap,scomplex  *x,const f77_int *incx)
{
    ctpsv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void CTPSV_BLIS_IMPL_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const scomplex  *ap,scomplex  *x,const f77_int *incx)
{
    ctpsv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void CTRMM_BLIS_IMPL(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,scomplex  *b,const f77_int *ldb)
{
    ctrmm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void ctrmm_blis_impl_(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,scomplex  *b,const f77_int *ldb)
{
    ctrmm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void CTRMM_BLIS_IMPL_(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,scomplex  *b,const f77_int *ldb)
{
    ctrmm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void CTRMV_BLIS_IMPL(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const scomplex  *a,const f77_int *lda,scomplex  *x,const f77_int *incx)
{
    ctrmv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void ctrmv_blis_impl_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const scomplex  *a,const f77_int *lda,scomplex  *x,const f77_int *incx)
{
    ctrmv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void CTRMV_BLIS_IMPL_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const scomplex  *a,const f77_int *lda,scomplex  *x,const f77_int *incx)
{
    ctrmv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void CTRSM_BLIS_IMPL(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,scomplex  *b,const f77_int *ldb)
{
    ctrsm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void ctrsm_blis_impl_(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,scomplex  *b,const f77_int *ldb)
{
    ctrsm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void CTRSM_BLIS_IMPL_(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const scomplex  *alpha,const scomplex  *a,const f77_int *lda,scomplex  *b,const f77_int *ldb)
{
    ctrsm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void CTRSV_BLIS_IMPL(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const scomplex  *a,const f77_int *lda,scomplex *x,const f77_int *incx)
{
    ctrsv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void ctrsv_blis_impl_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const scomplex *a,const f77_int *lda,scomplex *x,const f77_int *incx)
{
    ctrsv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void CTRSV_BLIS_IMPL_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const scomplex *a,const f77_int *lda,scomplex *x,const f77_int *incx)
{
    ctrsv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

double DASUM_BLIS_IMPL(const f77_int *n,const double *dx,const f77_int *incx)
{
    return dasum_blis_impl( n, dx, incx);
}

double dasum_blis_impl_(const f77_int *n,const double *dx,const f77_int *incx)
{
    return dasum_blis_impl( n, dx, incx);
}

double DASUM_BLIS_IMPL_(const f77_int *n,const double *dx,const f77_int *incx)
{
    return dasum_blis_impl( n, dx, incx);
}

void DAXPY_BLIS_IMPL(const f77_int *n,const double *da,const double *dx,const f77_int *incx,double *dy,const f77_int *incy)
{
    daxpy_blis_impl( n, da, dx, incx, dy, incy);
}

void daxpy_blis_impl_(const f77_int *n,const double *da,const double *dx,const f77_int *incx,double *dy,const f77_int *incy)
{
    daxpy_blis_impl( n, da, dx, incx, dy, incy);
}

void DAXPY_BLIS_IMPL_(const f77_int *n,const double *da,const double *dx,const f77_int *incx,double *dy,const f77_int *incy)
{
    daxpy_blis_impl( n, da, dx, incx, dy, incy);
}

double DCABS1_BLIS_IMPL(bla_dcomplex *z)
{
    return dcabs1_blis_impl( z);
}

double dcabs1_blis_impl_(bla_dcomplex *z)
{
    return dcabs1_blis_impl( z);
}

double DCABS1_BLIS_IMPL_(bla_dcomplex *z)
{
    return dcabs1_blis_impl( z);
}

void DCOPY_BLIS_IMPL(const f77_int *n,const double *dx,const f77_int *incx,double *dy,const f77_int *incy)
{
    dcopy_blis_impl( n, dx, incx, dy, incy);
}

void dcopy_blis_impl_(const f77_int *n,const double *dx,const f77_int *incx,double *dy,const f77_int *incy)
{
    dcopy_blis_impl( n, dx, incx, dy, incy);
}

void DCOPY_BLIS_IMPL_(const f77_int *n,const double *dx,const f77_int *incx,double *dy,const f77_int *incy)
{
    dcopy_blis_impl( n, dx, incx, dy, incy);
}

double DDOT_BLIS_IMPL(const f77_int *n,const double *dx,const f77_int *incx,const double *dy,const f77_int *incy)
{
    return ddot_blis_impl( n, dx, incx, dy, incy);
}

double ddot_blis_impl_(const f77_int *n,const double *dx,const f77_int *incx,const double *dy,const f77_int *incy)
{
    return ddot_blis_impl( n, dx, incx, dy, incy);
}

double DDOT_BLIS_IMPL_(const f77_int *n,const double *dx,const f77_int *incx,const double *dy,const f77_int *incy)
{
    return ddot_blis_impl( n, dx, incx, dy, incy);
}

void DGBMV_BLIS_IMPL(const char   *trans,const f77_int *m,const f77_int *n,const f77_int *kl,const f77_int *ku,const double *alpha,const double *a,const f77_int *lda,const double *x,const f77_int *incx,const double *beta,double *y,const f77_int *incy)
{
    dgbmv_blis_impl( trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

void dgbmv_blis_impl_(const char   *trans,const f77_int *m,const f77_int *n,const f77_int *kl,const f77_int *ku,const double *alpha,const double *a,const f77_int *lda,const double *x,const f77_int *incx,const double *beta,double *y,const f77_int *incy)
{
    dgbmv_blis_impl( trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

void DGBMV_BLIS_IMPL_(const char   *trans,const f77_int *m,const f77_int *n,const f77_int *kl,const f77_int *ku,const double *alpha,const double *a,const f77_int *lda,const double *x,const f77_int *incx,const double *beta,double *y,const f77_int *incy)
{
    dgbmv_blis_impl( trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

void DGEMM_BLIS_IMPL(const char   *transa,const char   *transb,const f77_int *m,const f77_int *n,const f77_int *k,const double *alpha,const double *a,const f77_int *lda,const double *b,const f77_int *ldb,const double *beta,double *c,const f77_int *ldc)
{
    dgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void dgemm_blis_impl_(const char   *transa,const char   *transb,const f77_int *m,const f77_int *n,const f77_int *k,const double *alpha,const double *a,const f77_int *lda,const double *b,const f77_int *ldb,const double *beta,double *c,const f77_int *ldc)
{
    dgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void DGEMM_BLIS_IMPL_(const char   *transa,const char   *transb,const f77_int *m,const f77_int *n,const f77_int *k,const double *alpha,const double *a,const f77_int *lda,const double *b,const f77_int *ldb,const double *beta,double *c,const f77_int *ldc)
{
    dgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void DZGEMM_BLIS_IMPL( const f77_char *transa, const f77_char *transb, const f77_int *m, const f77_int *n, const f77_int *k, const dcomplex *alpha, const double *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc )
{
    dzgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void dzgemm_blis_impl_( const f77_char *transa, const f77_char *transb, const f77_int *m, const f77_int *n, const f77_int *k, const dcomplex *alpha, const double *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc )
{
    dzgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void DZGEMM_BLIS_IMPL_( const f77_char *transa, const f77_char *transb, const f77_int *m, const f77_int *n, const f77_int *k, const dcomplex *alpha, const double *a, const f77_int *lda, const dcomplex *b, const f77_int *ldb, const dcomplex *beta, dcomplex *c, const f77_int *ldc )
{
    dzgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void DGEMV_BLIS_IMPL(const char   *trans,const f77_int *m,const f77_int *n,const double *alpha,const double *a,const f77_int *lda,const double *x,const f77_int *incx,const double *beta,double *y,const f77_int *incy)
{
    dgemv_blis_impl( trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void dgemv_blis_impl_(const char   *trans,const f77_int *m,const f77_int *n,const double *alpha,const double *a,const f77_int *lda,const double *x,const f77_int *incx,const double *beta,double *y,const f77_int *incy)
{
    dgemv_blis_impl( trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void DGEMV_BLIS_IMPL_(const char   *trans,const f77_int *m,const f77_int *n,const double *alpha,const double *a,const f77_int *lda,const double *x,const f77_int *incx,const double *beta,double *y,const f77_int *incy)
{
    dgemv_blis_impl( trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void DGER_BLIS_IMPL(const f77_int *m,const f77_int *n,const double *alpha,const double *x,const f77_int *incx,const double *y,const f77_int *incy,double *a,const f77_int *lda)
{
    dger_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void dger_blis_impl_(const f77_int *m,const f77_int *n,const double *alpha,const double *x,const f77_int *incx,const double *y,const f77_int *incy,double *a,const f77_int *lda)
{
    dger_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void DGER_BLIS_IMPL_(const f77_int *m,const f77_int *n,const double *alpha,const double *x,const f77_int *incx,const double *y,const f77_int *incy,double *a,const f77_int *lda)
{
    dger_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

double DNRM2_BLIS_IMPL(const f77_int *n,const double *x,const f77_int *incx)
{
    return dnrm2_blis_impl( n, x, incx);
}

double dnrm2_blis_impl_(const f77_int *n,const double *x,const f77_int *incx)
{
    return dnrm2_blis_impl( n, x, incx);
}

double DNRM2_BLIS_IMPL_(const f77_int *n,const double *x,const f77_int *incx)
{
    return dnrm2_blis_impl( n, x, incx);
}

void DROT_BLIS_IMPL(const f77_int *n,double *dx,const f77_int *incx,double *dy,const f77_int *incy,const double *c,const double *s)
{
    drot_blis_impl( n, dx, incx, dy, incy, c, s);
}

void drot_blis_impl_(const f77_int *n,double *dx,const f77_int *incx,double *dy,const f77_int *incy,const double *c,const double *s)
{
    drot_blis_impl( n, dx, incx, dy, incy, c, s);
}

void DROT_BLIS_IMPL_(const f77_int *n,double *dx,const f77_int *incx,double *dy,const f77_int *incy,const double *c,const double *s)
{
    drot_blis_impl( n, dx, incx, dy, incy, c, s);
}

void DROTG_BLIS_IMPL(double *da,double *db,double *c,double *s)
{
    drotg_blis_impl( da, db, c, s);
}

void drotg_blis_impl_(double *da,double *db,double *c,double *s)
{
    drotg_blis_impl( da, db, c, s);
}

void DROTG_BLIS_IMPL_(double *da,double *db,double *c,double *s)
{
    drotg_blis_impl( da, db, c, s);
}

void DROTM_BLIS_IMPL(const f77_int *n,double *dx,const f77_int *incx,double *dy,const f77_int *incy,const double *dparam)
{
    drotm_blis_impl( n, dx, incx, dy, incy, dparam);
}

void drotm_blis_impl_(const f77_int *n,double *dx,const f77_int *incx,double *dy,const f77_int *incy,const double *dparam)
{
    drotm_blis_impl( n, dx, incx, dy, incy, dparam);
}

void DROTM_BLIS_IMPL_(const f77_int *n,double *dx,const f77_int *incx,double *dy,const f77_int *incy,const double *dparam)
{
    drotm_blis_impl( n, dx, incx, dy, incy, dparam);
}

void DROTMG_BLIS_IMPL(double *dd1,double *dd2,double *dx1,const double *dy1,double *dparam)
{
    drotmg_blis_impl( dd1, dd2, dx1, dy1, dparam);
}

void drotmg_blis_impl_(double *dd1,double *dd2,double *dx1,const double *dy1,double *dparam)
{
    drotmg_blis_impl( dd1, dd2, dx1, dy1, dparam);
}

void DROTMG_BLIS_IMPL_(double *dd1,double *dd2,double *dx1,const double *dy1,double *dparam)
{
    drotmg_blis_impl( dd1, dd2, dx1, dy1, dparam);
}

void DSBMV_BLIS_IMPL(const char   *uplo,const f77_int *n,const f77_int *k,const double *alpha,const double *a,const f77_int *lda,const double *x,const f77_int *incx,const double *beta,double *y,const f77_int *incy)
{
    dsbmv_blis_impl( uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void dsbmv_blis_impl_(const char   *uplo,const f77_int *n,const f77_int *k,const double *alpha,const double *a,const f77_int *lda,const double *x,const f77_int *incx,const double *beta,double *y,const f77_int *incy)
{
    dsbmv_blis_impl( uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void DSBMV_BLIS_IMPL_(const char   *uplo,const f77_int *n,const f77_int *k,const double *alpha,const double *a,const f77_int *lda,const double *x,const f77_int *incx,const double *beta,double *y,const f77_int *incy)
{
    dsbmv_blis_impl( uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void DSCAL_BLIS_IMPL(const f77_int *n,const double *da,double *dx,const f77_int *incx)
{
    dscal_blis_impl( n, da, dx, incx);
}

void dscal_blis_impl_(const f77_int *n,const double *da,double *dx,const f77_int *incx)
{
    dscal_blis_impl( n, da, dx, incx);
}

void DSCAL_BLIS_IMPL_(const f77_int *n,const double *da,double *dx,const f77_int *incx)
{
    dscal_blis_impl( n, da, dx, incx);
}

double DSDOT_BLIS_IMPL(const f77_int *n,const float  *sx,const f77_int *incx,const float  *sy,const f77_int *incy)
{
    return dsdot_blis_impl( n, sx, incx, sy, incy);
}

double dsdot_blis_impl_(const f77_int *n,const float  *sx,const f77_int *incx,const float  *sy,const f77_int *incy)
{
    return dsdot_blis_impl( n, sx, incx, sy, incy);
}

double DSDOT_BLIS_IMPL_(const f77_int *n,const float  *sx,const f77_int *incx,const float  *sy,const f77_int *incy)
{
    return dsdot_blis_impl( n, sx, incx, sy, incy);
}

void DSPMV_BLIS_IMPL(const char   *uplo,const f77_int *n,const double *alpha,const double *ap,const double *x,const f77_int *incx,const double *beta,double *y,const f77_int *incy)
{
    dspmv_blis_impl( uplo, n, alpha, ap, x, incx, beta, y, incy);
}

void dspmv_blis_impl_(const char   *uplo,const f77_int *n,const double *alpha,const double *ap,const double *x,const f77_int *incx,const double *beta,double *y,const f77_int *incy)
{
    dspmv_blis_impl( uplo, n, alpha, ap, x, incx, beta, y, incy);
}

void DSPMV_BLIS_IMPL_(const char   *uplo,const f77_int *n,const double *alpha,const double *ap,const double *x,const f77_int *incx,const double *beta,double *y,const f77_int *incy)
{
    dspmv_blis_impl( uplo, n, alpha, ap, x, incx, beta, y, incy);
}

void DSPR_BLIS_IMPL(const char   *uplo,const f77_int *n,const double *alpha,const double *x,const f77_int *incx,double *ap)
{
    dspr_blis_impl( uplo, n, alpha, x, incx, ap);
}

void dspr_blis_impl_(const char   *uplo,const f77_int *n,const double *alpha,const double *x,const f77_int *incx,double *ap)
{
    dspr_blis_impl( uplo, n, alpha, x, incx, ap);
}

void DSPR_BLIS_IMPL_(const char   *uplo,const f77_int *n,const double *alpha,const double *x,const f77_int *incx,double *ap)
{
    dspr_blis_impl( uplo, n, alpha, x, incx, ap);
}

void DSPR2_BLIS_IMPL(const char   *uplo,const f77_int *n,const double *alpha,const double *x,const f77_int *incx,const double *y,const f77_int *incy,double *ap)
{
    dspr2_blis_impl( uplo, n, alpha, x, incx, y, incy, ap);
}

void dspr2_blis_impl_(const char   *uplo,const f77_int *n,const double *alpha,const double *x,const f77_int *incx,const double *y,const f77_int *incy,double *ap)
{
    dspr2_blis_impl( uplo, n, alpha, x, incx, y, incy, ap);
}

void DSPR2_BLIS_IMPL_(const char   *uplo,const f77_int *n,const double *alpha,const double *x,const f77_int *incx,const double *y,const f77_int *incy,double *ap)
{
    dspr2_blis_impl( uplo, n, alpha, x, incx, y, incy, ap);
}

void DSWAP_BLIS_IMPL(const f77_int *n,double *dx,const f77_int *incx,double *dy,const f77_int *incy)
{
    dswap_blis_impl( n, dx, incx, dy, incy);
}

void dswap_blis_impl_(const f77_int *n,double *dx,const f77_int *incx,double *dy,const f77_int *incy)
{
    dswap_blis_impl( n, dx, incx, dy, incy);
}

void DSWAP_BLIS_IMPL_(const f77_int *n,double *dx,const f77_int *incx,double *dy,const f77_int *incy)
{
    dswap_blis_impl( n, dx, incx, dy, incy);
}

void DSYMM_BLIS_IMPL(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const double *alpha,const double *a,const f77_int *lda,const double *b,const f77_int *ldb,const double *beta,double *c,const f77_int *ldc)
{
    dsymm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void dsymm_blis_impl_(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const double *alpha,const double *a,const f77_int *lda,const double *b,const f77_int *ldb,const double *beta,double *c,const f77_int *ldc)
{
    dsymm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void DSYMM_BLIS_IMPL_(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const double *alpha,const double *a,const f77_int *lda,const double *b,const f77_int *ldb,const double *beta,double *c,const f77_int *ldc)
{
    dsymm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void DSYMV_BLIS_IMPL(const char   *uplo,const f77_int *n,const double *alpha,const double *a,const f77_int *lda,const double *x,const f77_int *incx,const double *beta,double *y,const f77_int *incy)
{
    dsymv_blis_impl( uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

void dsymv_blis_impl_(const char   *uplo,const f77_int *n,const double *alpha,const double *a,const f77_int *lda,const double *x,const f77_int *incx,const double *beta,double *y,const f77_int *incy)
{
    dsymv_blis_impl( uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

void DSYMV_BLIS_IMPL_(const char   *uplo,const f77_int *n,const double *alpha,const double *a,const f77_int *lda,const double *x,const f77_int *incx,const double *beta,double *y,const f77_int *incy)
{
    dsymv_blis_impl( uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

void DSYR_BLIS_IMPL(const char   *uplo,const f77_int *n,const double *alpha,const double *x,const f77_int *incx,double *a,const f77_int *lda)
{
    dsyr_blis_impl( uplo, n, alpha, x, incx, a, lda);
}

void dsyr_blis_impl_(const char   *uplo,const f77_int *n,const double *alpha,const double *x,const f77_int *incx,double *a,const f77_int *lda)
{
    dsyr_blis_impl( uplo, n, alpha, x, incx, a, lda);
}

void DSYR_BLIS_IMPL_(const char   *uplo,const f77_int *n,const double *alpha,const double *x,const f77_int *incx,double *a,const f77_int *lda)
{
    dsyr_blis_impl( uplo, n, alpha, x, incx, a, lda);
}

void DSYR2_BLIS_IMPL(const char   *uplo,const f77_int *n,const double *alpha,const double *x,const f77_int *incx,const double *y,const f77_int *incy,double *a,const f77_int *lda)
{
    dsyr2_blis_impl( uplo, n, alpha, x, incx, y, incy, a, lda);
}

void dsyr2_blis_impl_(const char   *uplo,const f77_int *n,const double *alpha,const double *x,const f77_int *incx,const double *y,const f77_int *incy,double *a,const f77_int *lda)
{
    dsyr2_blis_impl( uplo, n, alpha, x, incx, y, incy, a, lda);
}

void DSYR2_BLIS_IMPL_(const char   *uplo,const f77_int *n,const double *alpha,const double *x,const f77_int *incx,const double *y,const f77_int *incy,double *a,const f77_int *lda)
{
    dsyr2_blis_impl( uplo, n, alpha, x, incx, y, incy, a, lda);
}

void DSYR2K_BLIS_IMPL(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const double *alpha,const double *a,const f77_int *lda,const double *b,const f77_int *ldb,const double *beta,double *c,const f77_int *ldc)
{
    dsyr2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void dsyr2k_blis_impl_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const double *alpha,const double *a,const f77_int *lda,const double *b,const f77_int *ldb,const double *beta,double *c,const f77_int *ldc)
{
    dsyr2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void DSYR2K_BLIS_IMPL_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const double *alpha,const double *a,const f77_int *lda,const double *b,const f77_int *ldb,const double *beta,double *c,const f77_int *ldc)
{
    dsyr2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void DSYRK_BLIS_IMPL(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const double *alpha,const double *a,const f77_int *lda,const double *beta,double *c,const f77_int *ldc)
{
    dsyrk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void dsyrk_blis_impl_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const double *alpha,const double *a,const f77_int *lda,const double *beta,double *c,const f77_int *ldc)
{
    dsyrk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void DSYRK_BLIS_IMPL_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const double *alpha,const double *a,const f77_int *lda,const double *beta,double *c,const f77_int *ldc)
{
    dsyrk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void DTBMV_BLIS_IMPL(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const double *a,const f77_int *lda,double *x,const f77_int *incx)
{
    dtbmv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void dtbmv_blis_impl_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const double *a,const f77_int *lda,double *x,const f77_int *incx)
{
    dtbmv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void DTBMV_BLIS_IMPL_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const double *a,const f77_int *lda,double *x,const f77_int *incx)
{
    dtbmv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void DTBSV_BLIS_IMPL(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const double *a,const f77_int *lda,double *x,const f77_int *incx)
{
    dtbsv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void dtbsv_blis_impl_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const double *a,const f77_int *lda,double *x,const f77_int *incx)
{
    dtbsv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void DTBSV_BLIS_IMPL_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const double *a,const f77_int *lda,double *x,const f77_int *incx)
{
    dtbsv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void DTPMV_BLIS_IMPL(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const double *ap,double *x,const f77_int *incx)
{
    dtpmv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void dtpmv_blis_impl_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const double *ap,double *x,const f77_int *incx)
{
    dtpmv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void DTPMV_BLIS_IMPL_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const double *ap,double *x,const f77_int *incx)
{
    dtpmv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void DTPSV_BLIS_IMPL(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const double *ap,double *x,const f77_int *incx)
{
    dtpsv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void dtpsv_blis_impl_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const double *ap,double *x,const f77_int *incx)
{
    dtpsv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void DTPSV_BLIS_IMPL_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const double *ap,double *x,const f77_int *incx)
{
    dtpsv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void DTRMM_BLIS_IMPL(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const double *alpha,const double *a,const f77_int *lda,double *b,const f77_int *ldb)
{
    dtrmm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void dtrmm_blis_impl_(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const double *alpha,const double *a,const f77_int *lda,double *b,const f77_int *ldb)
{
    dtrmm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void DTRMM_BLIS_IMPL_(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const double *alpha,const double *a,const f77_int *lda,double *b,const f77_int *ldb)
{
    dtrmm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void DTRMV_BLIS_IMPL(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const double *a,const f77_int *lda,double *x,const f77_int *incx)
{
    dtrmv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void dtrmv_blis_impl_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const double *a,const f77_int *lda,double *x,const f77_int *incx)
{
    dtrmv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void DTRMV_BLIS_IMPL_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const double *a,const f77_int *lda,double *x,const f77_int *incx)
{
    dtrmv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void DTRSM_BLIS_IMPL(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const double *alpha,const double *a,const f77_int *lda,double *b,const f77_int *ldb)
{
    dtrsm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void dtrsm_blis_impl_(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const double *alpha,const double *a,const f77_int *lda,double *b,const f77_int *ldb)
{
    dtrsm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void DTRSM_BLIS_IMPL_(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const double *alpha,const double *a,const f77_int *lda,double *b,const f77_int *ldb)
{
    dtrsm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void DTRSV_BLIS_IMPL(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const double *a,const f77_int *lda,double *x,const f77_int *incx)
{
    dtrsv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void dtrsv_blis_impl_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const double *a,const f77_int *lda,double *x,const f77_int *incx)
{
    dtrsv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void DTRSV_BLIS_IMPL_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const double *a,const f77_int *lda,double *x,const f77_int *incx)
{
    dtrsv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

double DZASUM_BLIS_IMPL(const f77_int *n,const dcomplex *zx,const f77_int *incx)
{
    return dzasum_blis_impl( n, zx, incx);
}

double dzasum_blis_impl_(const f77_int *n,const dcomplex *zx,const f77_int *incx)
{
    return dzasum_blis_impl( n, zx, incx);
}

double DZASUM_BLIS_IMPL_(const f77_int *n,const dcomplex *zx,const f77_int *incx)
{
    return dzasum_blis_impl( n, zx, incx);
}

double DZNRM2_BLIS_IMPL(const f77_int *n,const dcomplex *x,const f77_int *incx)
{
    return dznrm2_blis_impl( n, x, incx);
}

double dznrm2_blis_impl_(const f77_int *n,const dcomplex *x,const f77_int *incx)
{
    return dznrm2_blis_impl( n, x, incx);
}

double DZNRM2_BLIS_IMPL_(const f77_int *n,const dcomplex *x,const f77_int *incx)
{
    return dznrm2_blis_impl( n, x, incx);
}

f77_int ICAMAX_BLIS_IMPL(const f77_int *n,const scomplex  *cx,const f77_int *incx)
{
    return icamax_blis_impl( n, cx, incx);
}

f77_int icamax_blis_impl_(const f77_int *n,const scomplex  *cx,const f77_int *incx)
{
    return icamax_blis_impl( n, cx, incx);
}

f77_int ICAMAX_BLIS_IMPL_(const f77_int *n,const scomplex  *cx,const f77_int *incx)
{
    return icamax_blis_impl( n, cx, incx);
}

f77_int IDAMAX_BLIS_IMPL(const f77_int *n,const double *dx,const f77_int *incx)
{
    return idamax_blis_impl( n, dx, incx);
}

f77_int idamax_blis_impl_(const f77_int *n,const double *dx,const f77_int *incx)
{
    return idamax_blis_impl( n, dx, incx);
}

f77_int IDAMAX_BLIS_IMPL_(const f77_int *n,const double *dx,const f77_int *incx)
{
    return idamax_blis_impl( n, dx, incx);
}

f77_int ISAMAX_BLIS_IMPL(const f77_int *n,const float  *sx,const f77_int *incx)
{
    return isamax_blis_impl( n, sx, incx);
}

f77_int isamax_blis_impl_(const f77_int *n,const float  *sx,const f77_int *incx)
{
    return isamax_blis_impl( n, sx, incx);
}

f77_int ISAMAX_BLIS_IMPL_(const f77_int *n,const float  *sx,const f77_int *incx)
{
    return isamax_blis_impl( n, sx, incx);
}

f77_int IZAMAX_BLIS_IMPL(const f77_int *n,const dcomplex *zx,const f77_int *incx)
{
    return izamax_blis_impl( n, zx, incx);
}

f77_int izamax_blis_impl_(const f77_int *n,const dcomplex *zx,const f77_int *incx)
{
    return izamax_blis_impl( n, zx, incx);
}

f77_int IZAMAX_BLIS_IMPL_(const f77_int *n,const dcomplex *zx,const f77_int *incx)
{
    return izamax_blis_impl( n, zx, incx);
}

f77_int LSAME_BLIS_IMPL(const char   *ca,const char   *cb,const f77_int a,const f77_int b)
{
    return lsame_blis_impl( ca, cb, a, b);
}

f77_int LSAME_BLIS_IMPL_(const char   *ca,const char   *cb,const f77_int a,const f77_int b)
{
    return lsame_blis_impl( ca, cb, a, b);
}

f77_int lsame_blis_impl_(const char   *ca,const char   *cb,const f77_int a,const f77_int b)
{
    return lsame_blis_impl( ca, cb, a, b);
}

float SASUM_BLIS_IMPL(const f77_int *n,const float  *sx, const f77_int *incx)
{
    return sasum_blis_impl( n, sx, incx);
}

float sasum_blis_impl_(const f77_int *n,const float  *sx, const f77_int *incx)
{
    return sasum_blis_impl( n, sx, incx);
}

float SASUM_BLIS_IMPL_(const f77_int *n,const float  *sx, const f77_int *incx)
{
    return sasum_blis_impl( n, sx, incx);
}

void SAXPY_BLIS_IMPL(const f77_int *n,const float  *sa,const float  *sx,const f77_int *incx,float  *sy,const f77_int *incy)
{
    saxpy_blis_impl( n, sa, sx, incx, sy, incy);
}

void saxpy_blis_impl_(const f77_int *n,const float  *sa,const float  *sx,const f77_int *incx,float  *sy,const f77_int *incy)
{
    saxpy_blis_impl( n, sa, sx, incx, sy, incy);
}

void SAXPY_BLIS_IMPL_(const f77_int *n,const float  *sa,const float  *sx,const f77_int *incx,float  *sy,const f77_int *incy)
{
    saxpy_blis_impl( n, sa, sx, incx, sy, incy);
}


float SCASUM_BLIS_IMPL(const f77_int *n,const scomplex  *cx, const f77_int *incx)
{
    return scasum_blis_impl( n, cx, incx);
}

float scasum_blis_impl_(const f77_int *n,const scomplex  *cx, const f77_int *incx)
{
    return scasum_blis_impl( n, cx, incx);
}

float SCASUM_BLIS_IMPL_(const f77_int *n,const scomplex  *cx, const f77_int *incx)
{
    return scasum_blis_impl( n, cx, incx);
}



float SCNRM2_BLIS_IMPL(const f77_int *n,const scomplex  *x, const f77_int *incx)
{
    return scnrm2_blis_impl( n, x, incx);
}

float scnrm2_blis_impl_(const f77_int *n,const scomplex  *x, const f77_int *incx)
{
    return scnrm2_blis_impl( n, x, incx);
}

float SCNRM2_BLIS_IMPL_(const f77_int *n,const scomplex  *x, const f77_int *incx)
{
    return scnrm2_blis_impl( n, x, incx);
}


void SCOPY_BLIS_IMPL(const f77_int *n,const float  *sx,const f77_int *incx,float  *sy,const f77_int *incy)
{
    scopy_blis_impl( n, sx, incx, sy, incy);
}

void scopy_blis_impl_(const f77_int *n,const float  *sx,const f77_int *incx,float  *sy,const f77_int *incy)
{
    scopy_blis_impl( n, sx, incx, sy, incy);
}

void SCOPY_BLIS_IMPL_(const f77_int *n,const float  *sx,const f77_int *incx,float  *sy,const f77_int *incy)
{
    scopy_blis_impl( n, sx, incx, sy, incy);
}


float SDOT_BLIS_IMPL(const f77_int *n,const float  *sx, const f77_int *incx, const float  *sy, const f77_int *incy)
{
    return sdot_blis_impl( n, sx, incx, sy, incy);
}

float sdot_blis_impl_(const f77_int *n,const float  *sx, const f77_int *incx, const float  *sy, const f77_int *incy)
{
    return sdot_blis_impl( n, sx, incx, sy, incy);
}

float SDOT_BLIS_IMPL_(const f77_int *n,const float  *sx, const f77_int *incx, const float  *sy, const f77_int *incy)
{
    return sdot_blis_impl( n, sx, incx, sy, incy);
}


float SDSDOT_BLIS_IMPL(const f77_int *n,const float  *sb, const float  *sx, const f77_int *incx, const float  *sy, const f77_int *incy)
{
    return sdsdot_blis_impl( n, sb, sx, incx, sy, incy);
}

float sdsdot_blis_impl_(const f77_int *n,const float  *sb, const float  *sx, const f77_int *incx, const float  *sy, const f77_int *incy)
{
    return sdsdot_blis_impl( n, sb, sx, incx, sy, incy);
}

float SDSDOT_BLIS_IMPL_(const f77_int *n,const float  *sb, const float  *sx, const f77_int *incx, const float  *sy, const f77_int *incy)
{
    return sdsdot_blis_impl( n, sb, sx, incx, sy, incy);
}


void SGBMV_BLIS_IMPL(const char   *trans,const f77_int *m,const f77_int *n,const f77_int *kl,const f77_int *ku,const float  *alpha,const float  *a,const f77_int *lda,const float  *x,const f77_int *incx,const float  *beta,float  *y,const f77_int *incy)
{
    sgbmv_blis_impl( trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

void sgbmv_blis_impl_(const char   *trans,const f77_int *m,const f77_int *n,const f77_int *kl,const f77_int *ku,const float  *alpha,const float  *a,const f77_int *lda,const float  *x,const f77_int *incx,const float  *beta,float  *y,const f77_int *incy)
{
    sgbmv_blis_impl( trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

void SGBMV_BLIS_IMPL_(const char   *trans,const f77_int *m,const f77_int *n,const f77_int *kl,const f77_int *ku,const float  *alpha,const float  *a,const f77_int *lda,const float  *x,const f77_int *incx,const float  *beta,float  *y,const f77_int *incy)
{
    sgbmv_blis_impl( trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

void SGEMM_BLIS_IMPL(const char   *transa,const char   *transb,const f77_int *m,const f77_int *n,const f77_int *k,const float  *alpha,const float  *a,const f77_int *lda,const float  *b,const f77_int *ldb,const float  *beta,float  *c,const f77_int *ldc)
{
    sgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void sgemm_blis_impl_(const char   *transa,const char   *transb,const f77_int *m,const f77_int *n,const f77_int *k,const float  *alpha,const float  *a,const f77_int *lda,const float  *b,const f77_int *ldb,const float  *beta,float  *c,const f77_int *ldc)
{
    sgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void SGEMM_BLIS_IMPL_(const char   *transa,const char   *transb,const f77_int *m,const f77_int *n,const f77_int *k,const float  *alpha,const float  *a,const f77_int *lda,const float  *b,const f77_int *ldb,const float  *beta,float  *c,const f77_int *ldc)
{
    sgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void SGEMV_BLIS_IMPL(const char   *trans,const f77_int *m,const f77_int *n,const float  *alpha,const float  *a,const f77_int *lda,const float  *x,const f77_int *incx,const float  *beta,float  *y,const f77_int *incy)
{
    sgemv_blis_impl( trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void sgemv_blis_impl_(const char   *trans,const f77_int *m,const f77_int *n,const float  *alpha,const float  *a,const f77_int *lda,const float  *x,const f77_int *incx,const float  *beta,float  *y,const f77_int *incy)
{
    sgemv_blis_impl( trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void SGEMV_BLIS_IMPL_(const char   *trans,const f77_int *m,const f77_int *n,const float  *alpha,const float  *a,const f77_int *lda,const float  *x,const f77_int *incx,const float  *beta,float  *y,const f77_int *incy)
{
    sgemv_blis_impl( trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void SGER_BLIS_IMPL(const f77_int *m,const f77_int *n,const float  *alpha,const float  *x,const f77_int *incx,const float  *y,const f77_int *incy,float  *a,const f77_int *lda)
{
    sger_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void sger_blis_impl_(const f77_int *m,const f77_int *n,const float  *alpha,const float  *x,const f77_int *incx,const float  *y,const f77_int *incy,float  *a,const f77_int *lda)
{
    sger_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void SGER_BLIS_IMPL_(const f77_int *m,const f77_int *n,const float  *alpha,const float  *x,const f77_int *incx,const float  *y,const f77_int *incy,float  *a,const f77_int *lda)
{
    sger_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}


float SNRM2_BLIS_IMPL(const f77_int *n,const float  *x, const f77_int *incx)
{
    return snrm2_blis_impl( n, x, incx);
}

float snrm2_blis_impl_(const f77_int *n,const float  *x, const f77_int *incx)
{
    return snrm2_blis_impl( n, x, incx);
}

float SNRM2_BLIS_IMPL_(const f77_int *n,const float  *x, const f77_int *incx)
{
    return snrm2_blis_impl( n, x, incx);
}


void SROT_BLIS_IMPL(const f77_int *n,float  *sx,const f77_int *incx,float  *sy,const f77_int *incy,const float  *c,const float  *s)
{
    srot_blis_impl( n, sx, incx, sy, incy, c, s);
}

void srot_blis_impl_(const f77_int *n,float  *sx,const f77_int *incx,float  *sy,const f77_int *incy,const float  *c,const float  *s)
{
    srot_blis_impl( n, sx, incx, sy, incy, c, s);
}

void SROT_BLIS_IMPL_(const f77_int *n,float  *sx,const f77_int *incx,float  *sy,const f77_int *incy,const float  *c,const float  *s)
{
    srot_blis_impl( n, sx, incx, sy, incy, c, s);
}

void SROTG_BLIS_IMPL(float  *sa,float  *sb,float  *c,float  *s)
{
    srotg_blis_impl( sa, sb, c, s);
}

void srotg_blis_impl_(float  *sa,float  *sb,float  *c,float  *s)
{
    srotg_blis_impl( sa, sb, c, s);
}

void SROTG_BLIS_IMPL_(float  *sa,float  *sb,float  *c,float  *s)
{
    srotg_blis_impl( sa, sb, c, s);
}

void SROTM_BLIS_IMPL(const f77_int *n,float  *sx,const f77_int *incx,float  *sy,const f77_int *incy,const float  *sparam)
{
    srotm_blis_impl( n, sx, incx, sy, incy, sparam);
}

void srotm_blis_impl_(const f77_int *n,float  *sx,const f77_int *incx,float  *sy,const f77_int *incy,const float  *sparam)
{
    srotm_blis_impl( n, sx, incx, sy, incy, sparam);
}

void SROTM_BLIS_IMPL_(const f77_int *n,float  *sx,const f77_int *incx,float  *sy,const f77_int *incy,const float  *sparam)
{
    srotm_blis_impl( n, sx, incx, sy, incy, sparam);
}

void SROTMG_BLIS_IMPL(float  *sd1,float  *sd2,float  *sx1,const float  *sy1,float  *sparam)
{
    srotmg_blis_impl( sd1, sd2, sx1, sy1, sparam);
}

void srotmg_blis_impl_(float  *sd1,float  *sd2,float  *sx1,const float  *sy1,float  *sparam)
{
    srotmg_blis_impl( sd1, sd2, sx1, sy1, sparam);
}

void SROTMG_BLIS_IMPL_(float  *sd1,float  *sd2,float  *sx1,const float  *sy1,float  *sparam)
{
    srotmg_blis_impl( sd1, sd2, sx1, sy1, sparam);
}

void SSBMV_BLIS_IMPL(const char   *uplo,const f77_int *n,const f77_int *k,const float  *alpha,const float  *a,const f77_int *lda,const float  *x,const f77_int *incx,const float  *beta,float  *y,const f77_int *incy)
{
    ssbmv_blis_impl( uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void ssbmv_blis_impl_(const char   *uplo,const f77_int *n,const f77_int *k,const float  *alpha,const float  *a,const f77_int *lda,const float  *x,const f77_int *incx,const float  *beta,float  *y,const f77_int *incy)
{
    ssbmv_blis_impl( uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void SSBMV_BLIS_IMPL_(const char   *uplo,const f77_int *n,const f77_int *k,const float  *alpha,const float  *a,const f77_int *lda,const float  *x,const f77_int *incx,const float  *beta,float  *y,const f77_int *incy)
{
    ssbmv_blis_impl( uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void SSCAL_BLIS_IMPL(const f77_int *n,const float  *sa,float  *sx,const f77_int *incx)
{
    sscal_blis_impl( n, sa, sx, incx);
}

void sscal_blis_impl_(const f77_int *n,const float  *sa,float  *sx,const f77_int *incx)
{
    sscal_blis_impl( n, sa, sx, incx);
}

void SSCAL_BLIS_IMPL_(const f77_int *n,const float  *sa,float  *sx,const f77_int *incx)
{
    sscal_blis_impl( n, sa, sx, incx);
}

void SSPMV_BLIS_IMPL(const char   *uplo,const f77_int *n,const float  *alpha,const float  *ap,const float  *x,const f77_int *incx,const float  *beta,float  *y,const f77_int *incy)
{
    sspmv_blis_impl( uplo, n, alpha, ap, x, incx, beta, y, incy);
}

void sspmv_blis_impl_(const char   *uplo,const f77_int *n,const float  *alpha,const float  *ap,const float  *x,const f77_int *incx,const float  *beta,float  *y,const f77_int *incy)
{
    sspmv_blis_impl( uplo, n, alpha, ap, x, incx, beta, y, incy);
}

void SSPMV_BLIS_IMPL_(const char   *uplo,const f77_int *n,const float  *alpha,const float  *ap,const float  *x,const f77_int *incx,const float  *beta,float  *y,const f77_int *incy)
{
    sspmv_blis_impl( uplo, n, alpha, ap, x, incx, beta, y, incy);
}

void SSPR_BLIS_IMPL(const char   *uplo,const f77_int *n,const float  *alpha,const float  *x,const f77_int *incx,float  *ap)
{
    sspr_blis_impl( uplo, n, alpha, x, incx, ap);
}

void sspr_blis_impl_(const char   *uplo,const f77_int *n,const float  *alpha,const float  *x,const f77_int *incx,float  *ap)
{
    sspr_blis_impl( uplo, n, alpha, x, incx, ap);
}

void SSPR_BLIS_IMPL_(const char   *uplo,const f77_int *n,const float  *alpha,const float  *x,const f77_int *incx,float  *ap)
{
    sspr_blis_impl( uplo, n, alpha, x, incx, ap);
}

void SSPR2_BLIS_IMPL(const char   *uplo,const f77_int *n,const float  *alpha,const float  *x,const f77_int *incx,const float  *y,const f77_int *incy,float  *ap)
{
    sspr2_blis_impl( uplo, n, alpha, x, incx, y, incy, ap);
}

void sspr2_blis_impl_(const char   *uplo,const f77_int *n,const float  *alpha,const float  *x,const f77_int *incx,const float  *y,const f77_int *incy,float  *ap)
{
    sspr2_blis_impl( uplo, n, alpha, x, incx, y, incy, ap);
}

void SSPR2_BLIS_IMPL_(const char   *uplo,const f77_int *n,const float  *alpha,const float  *x,const f77_int *incx,const float  *y,const f77_int *incy,float  *ap)
{
    sspr2_blis_impl( uplo, n, alpha, x, incx, y, incy, ap);
}

void SSWAP_BLIS_IMPL(const f77_int *n,float  *sx,const f77_int *incx,float  *sy,const f77_int *incy)
{
    sswap_blis_impl( n, sx, incx, sy, incy);
}

void sswap_blis_impl_(const f77_int *n,float  *sx,const f77_int *incx,float  *sy,const f77_int *incy)
{
    sswap_blis_impl( n, sx, incx, sy, incy);
}

void SSWAP_BLIS_IMPL_(const f77_int *n,float  *sx,const f77_int *incx,float  *sy,const f77_int *incy)
{
    sswap_blis_impl( n, sx, incx, sy, incy);
}

void SSYMM_BLIS_IMPL(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const float  *alpha,const float  *a,const f77_int *lda,const float  *b,const f77_int *ldb,const float  *beta,float  *c,const f77_int *ldc)
{
    ssymm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ssymm_blis_impl_(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const float  *alpha,const float  *a,const f77_int *lda,const float  *b,const f77_int *ldb,const float  *beta,float  *c,const f77_int *ldc)
{
    ssymm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void SSYMM_BLIS_IMPL_(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const float  *alpha,const float  *a,const f77_int *lda,const float  *b,const f77_int *ldb,const float  *beta,float  *c,const f77_int *ldc)
{
    ssymm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void SSYMV_BLIS_IMPL(const char   *uplo,const f77_int *n,const float  *alpha,const float  *a,const f77_int *lda,const float  *x,const f77_int *incx,const float  *beta,float  *y,const f77_int *incy)
{
    ssymv_blis_impl( uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

void ssymv_blis_impl_(const char   *uplo,const f77_int *n,const float  *alpha,const float  *a,const f77_int *lda,const float  *x,const f77_int *incx,const float  *beta,float  *y,const f77_int *incy)
{
    ssymv_blis_impl( uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

void SSYMV_BLIS_IMPL_(const char   *uplo,const f77_int *n,const float  *alpha,const float  *a,const f77_int *lda,const float  *x,const f77_int *incx,const float  *beta,float  *y,const f77_int *incy)
{
    ssymv_blis_impl( uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

void SSYR_BLIS_IMPL(const char   *uplo,const f77_int *n,const float  *alpha,const float  *x,const f77_int *incx,float  *a,const f77_int *lda)
{
    ssyr_blis_impl( uplo, n, alpha, x, incx, a, lda);
}

void ssyr_blis_impl_(const char   *uplo,const f77_int *n,const float  *alpha,const float  *x,const f77_int *incx,float  *a,const f77_int *lda)
{
    ssyr_blis_impl( uplo, n, alpha, x, incx, a, lda);
}

void SSYR_BLIS_IMPL_(const char   *uplo,const f77_int *n,const float  *alpha,const float  *x,const f77_int *incx,float  *a,const f77_int *lda)
{
    ssyr_blis_impl( uplo, n, alpha, x, incx, a, lda);
}

void SSYR2_BLIS_IMPL(const char   *uplo,const f77_int *n,const float  *alpha,const float  *x,const f77_int *incx,const float  *y,const f77_int *incy,float  *a,const f77_int *lda)
{
    ssyr2_blis_impl( uplo, n, alpha, x, incx, y, incy, a, lda);
}

void ssyr2_blis_impl_(const char   *uplo,const f77_int *n,const float  *alpha,const float  *x,const f77_int *incx,const float  *y,const f77_int *incy,float  *a,const f77_int *lda)
{
    ssyr2_blis_impl( uplo, n, alpha, x, incx, y, incy, a, lda);
}

void SSYR2_BLIS_IMPL_(const char   *uplo,const f77_int *n,const float  *alpha,const float  *x,const f77_int *incx,const float  *y,const f77_int *incy,float  *a,const f77_int *lda)
{
    ssyr2_blis_impl( uplo, n, alpha, x, incx, y, incy, a, lda);
}

void SSYR2K_BLIS_IMPL(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const float  *alpha,const float  *a,const f77_int *lda,const float  *b,const f77_int *ldb,const float  *beta,float  *c,const f77_int *ldc)
{
    ssyr2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ssyr2k_blis_impl_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const float  *alpha,const float  *a,const f77_int *lda,const float  *b,const f77_int *ldb,const float  *beta,float  *c,const f77_int *ldc)
{
    ssyr2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void SSYR2K_BLIS_IMPL_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const float  *alpha,const float  *a,const f77_int *lda,const float  *b,const f77_int *ldb,const float  *beta,float  *c,const f77_int *ldc)
{
    ssyr2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void SSYRK_BLIS_IMPL(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const float  *alpha,const float  *a,const f77_int *lda,const float  *beta,float  *c,const f77_int *ldc)
{
    ssyrk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void ssyrk_blis_impl_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const float  *alpha,const float  *a,const f77_int *lda,const float  *beta,float  *c,const f77_int *ldc)
{
    ssyrk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void SSYRK_BLIS_IMPL_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const float  *alpha,const float  *a,const f77_int *lda,const float  *beta,float  *c,const f77_int *ldc)
{
    ssyrk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void STBMV_BLIS_IMPL(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const float  *a,const f77_int *lda,float  *x,const f77_int *incx)
{
    stbmv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void stbmv_blis_impl_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const float  *a,const f77_int *lda,float  *x,const f77_int *incx)
{
    stbmv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void STBMV_BLIS_IMPL_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const float  *a,const f77_int *lda,float  *x,const f77_int *incx)
{
    stbmv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void STBSV_BLIS_IMPL(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const float  *a,const f77_int *lda,float  *x,const f77_int *incx)
{
    stbsv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void stbsv_blis_impl_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const float  *a,const f77_int *lda,float  *x,const f77_int *incx)
{
    stbsv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void STBSV_BLIS_IMPL_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const float  *a,const f77_int *lda,float  *x,const f77_int *incx)
{
    stbsv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void STPMV_BLIS_IMPL(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const float  *ap,float  *x,const f77_int *incx)
{
    stpmv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void stpmv_blis_impl_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const float  *ap,float  *x,const f77_int *incx)
{
    stpmv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void STPMV_BLIS_IMPL_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const float  *ap,float  *x,const f77_int *incx)
{
    stpmv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void STPSV_BLIS_IMPL(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const float  *ap,float  *x,const f77_int *incx)
{
    stpsv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void stpsv_blis_impl_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const float  *ap,float  *x,const f77_int *incx)
{
    stpsv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void STPSV_BLIS_IMPL_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const float  *ap,float  *x,const f77_int *incx)
{
    stpsv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void STRMM_BLIS_IMPL(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const float  *alpha,const float  *a,const f77_int *lda,float  *b,const f77_int *ldb)
{
    strmm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void strmm_blis_impl_(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const float  *alpha,const float  *a,const f77_int *lda,float  *b,const f77_int *ldb)
{
    strmm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void STRMM_BLIS_IMPL_(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const float  *alpha,const float  *a,const f77_int *lda,float  *b,const f77_int *ldb)
{
    strmm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void STRMV_BLIS_IMPL(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const float  *a,const f77_int *lda,float  *x,const f77_int *incx)
{
    strmv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void strmv_blis_impl_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const float  *a,const f77_int *lda,float  *x,const f77_int *incx)
{
    strmv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void STRMV_BLIS_IMPL_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const float  *a,const f77_int *lda,float  *x,const f77_int *incx)
{
    strmv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void STRSM_BLIS_IMPL(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const float  *alpha,const float  *a,const f77_int *lda,float  *b,const f77_int *ldb)
{
    strsm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void strsm_blis_impl_(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const float  *alpha,const float  *a,const f77_int *lda,float  *b,const f77_int *ldb)
{
    strsm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void STRSM_BLIS_IMPL_(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const float  *alpha,const float  *a,const f77_int *lda,float  *b,const f77_int *ldb)
{
    strsm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void STRSV_BLIS_IMPL(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const float  *a,const f77_int *lda,float  *x,const f77_int *incx)
{
    strsv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void strsv_blis_impl_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const float  *a,const f77_int *lda,float  *x,const f77_int *incx)
{
    strsv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void STRSV_BLIS_IMPL_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const float  *a,const f77_int *lda,float  *x,const f77_int *incx)
{
    strsv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void XERBLA_BLIS_IMPL(const char   *srname,const f77_int *info, ftnlen n)
{
    xerbla_blis_impl( srname, info, n);
}

void XERBLA_BLIS_IMPL_(const char   *srname,const f77_int *info, ftnlen n)
{
    xerbla_blis_impl( srname, info, n);
}

void xerbla_blis_impl_(const char   *srname,const f77_int *info, ftnlen n)
{
    xerbla_blis_impl( srname, info, n);
}

void ZAXPY_BLIS_IMPL(const f77_int *n,const dcomplex *za,const dcomplex *zx,const f77_int *incx,dcomplex *zy,const f77_int *incy)
{
    zaxpy_blis_impl( n, za, zx, incx, zy, incy);
}

void zaxpy_blis_impl_(const f77_int *n,const dcomplex *za,const dcomplex *zx,const f77_int *incx,dcomplex *zy,const f77_int *incy)
{
    zaxpy_blis_impl( n, za, zx, incx, zy, incy);
}

void ZAXPY_BLIS_IMPL_(const f77_int *n,const dcomplex *za,const dcomplex *zx,const f77_int *incx,dcomplex *zy,const f77_int *incy)
{
    zaxpy_blis_impl( n, za, zx, incx, zy, incy);
}

void ZCOPY_BLIS_IMPL(const f77_int *n,const dcomplex *zx,const f77_int *incx,dcomplex *zy,const f77_int *incy)
{
    zcopy_blis_impl( n, zx, incx, zy, incy);
}

void zcopy_blis_impl_(const f77_int *n,const dcomplex *zx,const f77_int *incx,dcomplex *zy,const f77_int *incy)
{
    zcopy_blis_impl( n, zx, incx, zy, incy);
}

void ZCOPY_BLIS_IMPL_(const f77_int *n,const dcomplex *zx,const f77_int *incx,dcomplex *zy,const f77_int *incy)
{
    zcopy_blis_impl( n, zx, incx, zy, incy);
}

void ZDROT_BLIS_IMPL(const f77_int *n,dcomplex *cx,const f77_int *incx,dcomplex *cy,const f77_int *incy,const double *c,const double *s)
{
    zdrot_blis_impl( n, cx, incx, cy, incy, c, s);
}

void zdrot_blis_impl_(const f77_int *n,dcomplex *cx,const f77_int *incx,dcomplex *cy,const f77_int *incy,const double *c,const double *s)
{
    zdrot_blis_impl( n, cx, incx, cy, incy, c, s);
}

void ZDROT_BLIS_IMPL_(const f77_int *n,dcomplex *cx,const f77_int *incx,dcomplex *cy,const f77_int *incy,const double *c,const double *s)
{
    zdrot_blis_impl( n, cx, incx, cy, incy, c, s);
}

void ZDSCAL_BLIS_IMPL(const f77_int *n,const double *da,dcomplex *zx,const f77_int *incx)
{
    zdscal_blis_impl( n, da, zx, incx);
}

void zdscal_blis_impl_(const f77_int *n,const double *da,dcomplex *zx,const f77_int *incx)
{
    zdscal_blis_impl( n, da, zx, incx);
}

void ZDSCAL_BLIS_IMPL_(const f77_int *n,const double *da,dcomplex *zx,const f77_int *incx)
{
    zdscal_blis_impl( n, da, zx, incx);
}

void ZGBMV_BLIS_IMPL(const char   *trans,const f77_int *m,const f77_int *n,const f77_int *kl,const f77_int *ku,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *x,const f77_int *incx,const dcomplex *beta,dcomplex *y,const f77_int *incy)
{
    zgbmv_blis_impl( trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

void zgbmv_blis_impl_(const char   *trans,const f77_int *m,const f77_int *n,const f77_int *kl,const f77_int *ku,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *x,const f77_int *incx,const dcomplex *beta,dcomplex *y,const f77_int *incy)
{
    zgbmv_blis_impl( trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

void ZGBMV_BLIS_IMPL_(const char   *trans,const f77_int *m,const f77_int *n,const f77_int *kl,const f77_int *ku,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *x,const f77_int *incx,const dcomplex *beta,dcomplex *y,const f77_int *incy)
{
    zgbmv_blis_impl( trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

void ZGEMM_BLIS_IMPL(const char   *transa,const char   *transb,const f77_int *m,const f77_int *n,const f77_int *k,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *b,const f77_int *ldb,const dcomplex *beta,dcomplex *c,const f77_int *ldc)
{
    zgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void zgemm_blis_impl_(const char   *transa,const char   *transb,const f77_int *m,const f77_int *n,const f77_int *k,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *b,const f77_int *ldb,const dcomplex *beta,dcomplex *c,const f77_int *ldc)
{
    zgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ZGEMM_BLIS_IMPL_(const char   *transa,const char   *transb,const f77_int *m,const f77_int *n,const f77_int *k,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *b,const f77_int *ldb,const dcomplex *beta,dcomplex *c,const f77_int *ldc)
{
    zgemm_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ZGEMV_BLIS_IMPL(const char   *trans,const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *x,const f77_int *incx,const dcomplex *beta,dcomplex *y,const f77_int *incy)
{
    zgemv_blis_impl( trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void zgemv_blis_impl_(const char   *trans,const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *x,const f77_int *incx,const dcomplex *beta,dcomplex *y,const f77_int *incy)
{
    zgemv_blis_impl( trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void ZGEMV_BLIS_IMPL_(const char   *trans,const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *x,const f77_int *incx,const dcomplex *beta,dcomplex *y,const f77_int *incy)
{
    zgemv_blis_impl( trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

void ZGERC_BLIS_IMPL(const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *x,const f77_int *incx,const dcomplex *y,const f77_int *incy,dcomplex *a,const f77_int *lda)
{
    zgerc_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void zgerc_blis_impl_(const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *x,const f77_int *incx,const dcomplex *y,const f77_int *incy,dcomplex *a,const f77_int *lda)
{
    zgerc_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void ZGERC_BLIS_IMPL_(const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *x,const f77_int *incx,const dcomplex *y,const f77_int *incy,dcomplex *a,const f77_int *lda)
{
    zgerc_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void ZGERU_BLIS_IMPL(const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *x,const f77_int *incx,const dcomplex *y,const f77_int *incy,dcomplex *a,const f77_int *lda)
{
    zgeru_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void zgeru_blis_impl_(const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *x,const f77_int *incx,const dcomplex *y,const f77_int *incy,dcomplex *a,const f77_int *lda)
{
    zgeru_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void ZGERU_BLIS_IMPL_(const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *x,const f77_int *incx,const dcomplex *y,const f77_int *incy,dcomplex *a,const f77_int *lda)
{
    zgeru_blis_impl( m, n, alpha, x, incx, y, incy, a, lda);
}

void ZHBMV_BLIS_IMPL(const char   *uplo,const f77_int *n,const f77_int *k,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *x,const f77_int *incx,const dcomplex *beta,dcomplex *y,const f77_int *incy)
{
    zhbmv_blis_impl( uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void zhbmv_blis_impl_(const char   *uplo,const f77_int *n,const f77_int *k,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *x,const f77_int *incx,const dcomplex *beta,dcomplex *y,const f77_int *incy)
{
    zhbmv_blis_impl( uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void ZHBMV_BLIS_IMPL_(const char   *uplo,const f77_int *n,const f77_int *k,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *x,const f77_int *incx,const dcomplex *beta,dcomplex *y,const f77_int *incy)
{
    zhbmv_blis_impl( uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void ZHEMM_BLIS_IMPL(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *b,const f77_int *ldb,const dcomplex *beta,dcomplex *c,const f77_int *ldc)
{
    zhemm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void zhemm_blis_impl_(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *b,const f77_int *ldb,const dcomplex *beta,dcomplex *c,const f77_int *ldc)
{
    zhemm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ZHEMM_BLIS_IMPL_(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *b,const f77_int *ldb,const dcomplex *beta,dcomplex *c,const f77_int *ldc)
{
    zhemm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ZHEMV_BLIS_IMPL(const char   *uplo,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *x,const f77_int *incx,const dcomplex *beta,dcomplex *y,const f77_int *incy)
{
    zhemv_blis_impl( uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

void zhemv_blis_impl_(const char   *uplo,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *x,const f77_int *incx,const dcomplex *beta,dcomplex *y,const f77_int *incy)
{
    zhemv_blis_impl( uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

void ZHEMV_BLIS_IMPL_(const char   *uplo,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *x,const f77_int *incx,const dcomplex *beta,dcomplex *y,const f77_int *incy)
{
    zhemv_blis_impl( uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

void ZHER_BLIS_IMPL(const char   *uplo,const f77_int *n,const double *alpha,const dcomplex *x,const f77_int *incx,dcomplex *a,const f77_int *lda)
{
    zher_blis_impl( uplo, n, alpha, x, incx, a, lda);
}

void zher_blis_impl_(const char   *uplo,const f77_int *n,const double *alpha,const dcomplex *x,const f77_int *incx,dcomplex *a,const f77_int *lda)
{
    zher_blis_impl( uplo, n, alpha, x, incx, a, lda);
}

void ZHER_BLIS_IMPL_(const char   *uplo,const f77_int *n,const double *alpha,const dcomplex *x,const f77_int *incx,dcomplex *a,const f77_int *lda)
{
    zher_blis_impl( uplo, n, alpha, x, incx, a, lda);
}

void ZHER2_BLIS_IMPL(const char   *uplo,const f77_int *n,const dcomplex *alpha,const dcomplex *x,const f77_int *incx,const dcomplex *y,const f77_int *incy,dcomplex *a,const f77_int *lda)
{
    zher2_blis_impl( uplo, n, alpha, x, incx, y, incy, a, lda);
}

void zher2_blis_impl_(const char   *uplo,const f77_int *n,const dcomplex *alpha,const dcomplex *x,const f77_int *incx,const dcomplex *y,const f77_int *incy,dcomplex *a,const f77_int *lda)
{
    zher2_blis_impl( uplo, n, alpha, x, incx, y, incy, a, lda);
}

void ZHER2_BLIS_IMPL_(const char   *uplo,const f77_int *n,const dcomplex *alpha,const dcomplex *x,const f77_int *incx,const dcomplex *y,const f77_int *incy,dcomplex *a,const f77_int *lda)
{
    zher2_blis_impl( uplo, n, alpha, x, incx, y, incy, a, lda);
}

void ZHER2K_BLIS_IMPL(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *b,const f77_int *ldb,const double *beta,dcomplex *c,const f77_int *ldc)
{
    zher2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void zher2k_blis_impl_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *b,const f77_int *ldb,const double *beta,dcomplex *c,const f77_int *ldc)
{
    zher2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ZHER2K_BLIS_IMPL_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *b,const f77_int *ldb,const double *beta,dcomplex *c,const f77_int *ldc)
{
    zher2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ZHERK_BLIS_IMPL(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const double *alpha,const dcomplex *a,const f77_int *lda,const double *beta,dcomplex *c,const f77_int *ldc)
{
    zherk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void zherk_blis_impl_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const double *alpha,const dcomplex *a,const f77_int *lda,const double *beta,dcomplex *c,const f77_int *ldc)
{
    zherk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void ZHERK_BLIS_IMPL_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const double *alpha,const dcomplex *a,const f77_int *lda,const double *beta,dcomplex *c,const f77_int *ldc)
{
    zherk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void ZHPMV_BLIS_IMPL(const char   *uplo,const f77_int *n,const dcomplex *alpha,const dcomplex *ap,const dcomplex *x,const f77_int *incx,const dcomplex *beta,dcomplex *y,const f77_int *incy)
{
    zhpmv_blis_impl( uplo, n, alpha, ap, x, incx, beta, y, incy);
}

void zhpmv_blis_impl_(const char   *uplo,const f77_int *n,const dcomplex *alpha,const dcomplex *ap,const dcomplex *x,const f77_int *incx,const dcomplex *beta,dcomplex *y,const f77_int *incy)
{
    zhpmv_blis_impl( uplo, n, alpha, ap, x, incx, beta, y, incy);
}

void ZHPMV_BLIS_IMPL_(const char   *uplo,const f77_int *n,const dcomplex *alpha,const dcomplex *ap,const dcomplex *x,const f77_int *incx,const dcomplex *beta,dcomplex *y,const f77_int *incy)
{
    zhpmv_blis_impl( uplo, n, alpha, ap, x, incx, beta, y, incy);
}

void ZHPR_BLIS_IMPL(const char   *uplo,const f77_int *n,const bla_double *alpha,const dcomplex *x,const f77_int *incx,dcomplex *ap)
{
    zhpr_blis_impl( uplo, n, alpha, x, incx, ap);
}

void zhpr_blis_impl_(const char   *uplo,const f77_int *n,const bla_double *alpha,const dcomplex *x,const f77_int *incx,dcomplex *ap)
{
    zhpr_blis_impl( uplo, n, alpha, x, incx, ap);
}

void ZHPR_BLIS_IMPL_(const char   *uplo,const f77_int *n,const bla_double *alpha,const dcomplex *x,const f77_int *incx,dcomplex *ap)
{
    zhpr_blis_impl( uplo, n, alpha, x, incx, ap);
}

void ZHPR2_BLIS_IMPL(const char   *uplo,const f77_int *n,const dcomplex *alpha,const dcomplex *x,const f77_int *incx,const dcomplex *y,const f77_int *incy,dcomplex *ap)
{
    zhpr2_blis_impl( uplo, n, alpha, x, incx, y, incy, ap);
}

void zhpr2_blis_impl_(const char   *uplo,const f77_int *n,const dcomplex *alpha,const dcomplex *x,const f77_int *incx,const dcomplex *y,const f77_int *incy,dcomplex *ap)
{
    zhpr2_blis_impl( uplo, n, alpha, x, incx, y, incy, ap);
}

void ZHPR2_BLIS_IMPL_(const char   *uplo,const f77_int *n,const dcomplex *alpha,const dcomplex *x,const f77_int *incx,const dcomplex *y,const f77_int *incy,dcomplex *ap)
{
    zhpr2_blis_impl( uplo, n, alpha, x, incx, y, incy, ap);
}

void ZROTG_BLIS_IMPL(dcomplex *ca,bla_dcomplex *cb,bla_double *c,dcomplex *s)
{
    zrotg_blis_impl( ca, cb, c, s);
}

void zrotg_blis_impl_(dcomplex *ca,bla_dcomplex *cb,bla_double *c,dcomplex *s)
{
    zrotg_blis_impl( ca, cb, c, s);
}

void ZROTG_BLIS_IMPL_(dcomplex *ca,bla_dcomplex *cb,bla_double *c,dcomplex *s)
{
    zrotg_blis_impl( ca, cb, c, s);
}

void ZSCAL_BLIS_IMPL(const f77_int *n,const dcomplex *za,dcomplex *zx,const f77_int *incx)
{
    zscal_blis_impl( n, za, zx, incx);
}

void zscal_blis_impl_(const f77_int *n,const dcomplex *za,dcomplex *zx,const f77_int *incx)
{
    zscal_blis_impl( n, za, zx, incx);
}

void ZSCAL_BLIS_IMPL_(const f77_int *n,const dcomplex *za,dcomplex *zx,const f77_int *incx)
{
    zscal_blis_impl( n, za, zx, incx);
}

void ZSWAP_BLIS_IMPL(const f77_int *n,dcomplex *zx,const f77_int *incx,dcomplex *zy,const f77_int *incy)
{
    zswap_blis_impl( n, zx, incx, zy, incy);
}

void zswap_blis_impl_(const f77_int *n,dcomplex *zx,const f77_int *incx,dcomplex *zy,const f77_int *incy)
{
    zswap_blis_impl( n, zx, incx, zy, incy);
}

void ZSWAP_BLIS_IMPL_(const f77_int *n,dcomplex *zx,const f77_int *incx,dcomplex *zy,const f77_int *incy)
{
    zswap_blis_impl( n, zx, incx, zy, incy);
}

void ZSYMM_BLIS_IMPL(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *b,const f77_int *ldb,const dcomplex *beta,dcomplex *c,const f77_int *ldc)
{
    zsymm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void zsymm_blis_impl_(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *b,const f77_int *ldb,const dcomplex *beta,dcomplex *c,const f77_int *ldc)
{
    zsymm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ZSYMM_BLIS_IMPL_(const char   *side,const char   *uplo,const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *b,const f77_int *ldb,const dcomplex *beta,dcomplex *c,const f77_int *ldc)
{
    zsymm_blis_impl( side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ZSYR2K_BLIS_IMPL(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *b,const f77_int *ldb,const dcomplex *beta,dcomplex *c,const f77_int *ldc)
{
    zsyr2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void zsyr2k_blis_impl_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *b,const f77_int *ldb,const dcomplex *beta,dcomplex *c,const f77_int *ldc)
{
    zsyr2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ZSYR2K_BLIS_IMPL_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *b,const f77_int *ldb,const dcomplex *beta,dcomplex *c,const f77_int *ldc)
{
    zsyr2k_blis_impl( uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ZSYRK_BLIS_IMPL(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *beta,dcomplex *c,const f77_int *ldc)
{
    zsyrk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void zsyrk_blis_impl_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *beta,dcomplex *c,const f77_int *ldc)
{
    zsyrk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void ZSYRK_BLIS_IMPL_(const char   *uplo,const char   *trans,const f77_int *n,const f77_int *k,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,const dcomplex *beta,dcomplex *c,const f77_int *ldc)
{
    zsyrk_blis_impl( uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

void ZTBMV_BLIS_IMPL(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const dcomplex *a,const f77_int *lda,dcomplex *x,const f77_int *incx)
{
    ztbmv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void ztbmv_blis_impl_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const dcomplex *a,const f77_int *lda,dcomplex *x,const f77_int *incx)
{
    ztbmv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void ZTBMV_BLIS_IMPL_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const dcomplex *a,const f77_int *lda,dcomplex *x,const f77_int *incx)
{
    ztbmv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void ZTBSV_BLIS_IMPL(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const dcomplex *a,const f77_int *lda,dcomplex *x,const f77_int *incx)
{
    ztbsv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void ztbsv_blis_impl_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const dcomplex *a,const f77_int *lda,dcomplex *x,const f77_int *incx)
{
    ztbsv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void ZTBSV_BLIS_IMPL_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const f77_int *k,const dcomplex *a,const f77_int *lda,dcomplex *x,const f77_int *incx)
{
    ztbsv_blis_impl( uplo, trans, diag, n, k, a, lda, x, incx);
}

void ZTPMV_BLIS_IMPL(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const dcomplex *ap,dcomplex *x,const f77_int *incx)
{
    ztpmv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void ztpmv_blis_impl_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const dcomplex *ap,dcomplex *x,const f77_int *incx)
{
    ztpmv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void ZTPMV_BLIS_IMPL_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const dcomplex *ap,dcomplex *x,const f77_int *incx)
{
    ztpmv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void ZTPSV_BLIS_IMPL(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const dcomplex *ap,dcomplex *x,const f77_int *incx)
{
    ztpsv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void ztpsv_blis_impl_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const dcomplex *ap,dcomplex *x,const f77_int *incx)
{
    ztpsv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void ZTPSV_BLIS_IMPL_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const dcomplex *ap,dcomplex *x,const f77_int *incx)
{
    ztpsv_blis_impl( uplo, trans, diag, n, ap, x, incx);
}

void ZTRMM_BLIS_IMPL(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,dcomplex *b,const f77_int *ldb)
{
    ztrmm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void ztrmm_blis_impl_(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,dcomplex *b,const f77_int *ldb)
{
    ztrmm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void ZTRMM_BLIS_IMPL_(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,dcomplex *b,const f77_int *ldb)
{
    ztrmm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void ZTRMV_BLIS_IMPL(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const dcomplex *a,const f77_int *lda,dcomplex *x,const f77_int *incx)
{
    ztrmv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void ztrmv_blis_impl_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const dcomplex *a,const f77_int *lda,dcomplex *x,const f77_int *incx)
{
    ztrmv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void ZTRMV_BLIS_IMPL_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const dcomplex *a,const f77_int *lda,dcomplex *x,const f77_int *incx)
{
    ztrmv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void ZTRSM_BLIS_IMPL(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,dcomplex *b,const f77_int *ldb)
{
    ztrsm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void ztrsm_blis_impl_(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,dcomplex *b,const f77_int *ldb)
{
    ztrsm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void ZTRSM_BLIS_IMPL_(const char   *side,const char   *uplo,const char   *transa,const char   *diag,const f77_int *m,const f77_int *n,const dcomplex *alpha,const dcomplex *a,const f77_int *lda,dcomplex *b,const f77_int *ldb)
{
    ztrsm_blis_impl( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

void ZTRSV_BLIS_IMPL(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const dcomplex *a,const f77_int *lda,dcomplex *x,const f77_int *incx)
{
    ztrsv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void ztrsv_blis_impl_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const dcomplex *a,const f77_int *lda,dcomplex *x,const f77_int *incx)
{
    ztrsv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

void ZTRSV_BLIS_IMPL_(const char   *uplo,const char   *trans,const char   *diag,const f77_int *n,const dcomplex *a,const f77_int *lda,dcomplex *x,const f77_int *incx)
{
    ztrsv_blis_impl( uplo, trans, diag, n, a, lda, x, incx);
}

#ifdef BLIS_ENABLE_CBLAS

void CDOTCSUB_BLIS_IMPL( const f77_int* n, const scomplex* x,const f77_int* incx, const scomplex* y, const f77_int* incy, scomplex* rval)
{
    cdotcsub_blis_impl( n, x, incx, y, incy, rval);
}

void cdotcsub_blis_impl_( const f77_int* n, const scomplex* x,const f77_int* incx, const scomplex* y, const f77_int* incy, scomplex* rval)
{
    cdotcsub_blis_impl( n, x, incx, y, incy, rval);
}

void CDOTCSUB_BLIS_IMPL_( const f77_int* n, const scomplex* x,const f77_int* incx, const scomplex* y, const f77_int* incy, scomplex* rval)
{
    cdotcsub_blis_impl( n, x, incx, y, incy, rval);
}

void CDOTUSUB_BLIS_IMPL( const f77_int* n, const scomplex* x,const f77_int* incxy, const scomplex* y, const f77_int* incy, scomplex* rval)
{
    cdotusub_blis_impl( n, x, incxy, y, incy, rval);
}

void cdotusub_blis_impl_( const f77_int* n, const scomplex* x,const f77_int* incxy, const scomplex* y, const f77_int* incy, scomplex* rval)
{
    cdotusub_blis_impl( n, x, incxy, y, incy, rval);
}

void CDOTUSUB_BLIS_IMPL_( const f77_int* n, const scomplex* x,const f77_int* incxy, const scomplex* y, const f77_int* incy, scomplex* rval)
{
    cdotusub_blis_impl( n, x, incxy, y, incy, rval);
}

#endif // BLIS_ENABLE_CBLAS

void CGEMM3M_BLIS_IMPL( const f77_char* transa, const f77_char* transb, const f77_int* m, const f77_int* n, const f77_int* k, const scomplex* alpha, const scomplex* a, const f77_int* lda, const scomplex* b, const f77_int* ldb, const scomplex* beta, scomplex* c, const f77_int* ldc)
{
    cgemm3m_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void cgemm3m_blis_impl_( const f77_char* transa, const f77_char* transb, const f77_int* m, const f77_int* n, const f77_int* k, const scomplex* alpha, const scomplex* a, const f77_int* lda, const scomplex* b, const f77_int* ldb, const scomplex* beta, scomplex* c, const f77_int* ldc)
{
    cgemm3m_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void CGEMM3M_BLIS_IMPL_( const f77_char* transa, const f77_char* transb, const f77_int* m, const f77_int* n, const f77_int* k, const scomplex* alpha, const scomplex* a, const f77_int* lda, const scomplex* b, const f77_int* ldb, const scomplex* beta, scomplex* c, const f77_int* ldc)
{
    cgemm3m_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void CGEMM_BATCH_BLIS_IMPL( const f77_char* transa_array, const f77_char* transb_array,const f77_int *m_array, const f77_int *n_array, const f77_int *k_array,const scomplex* alpha_array, const scomplex** a_array, const  f77_int *lda_array, const scomplex** b_array, const f77_int *ldb_array, const scomplex* beta_array, scomplex** c_array, const f77_int *ldc_array, const f77_int* group_count, const f77_int *group_size)
{
    cgemm_batch_blis_impl( transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

void cgemm_batch_blis_impl_( const f77_char* transa_array, const f77_char* transb_array,const f77_int *m_array, const f77_int *n_array, const f77_int *k_array,const scomplex* alpha_array, const scomplex** a_array, const  f77_int *lda_array, const scomplex** b_array, const f77_int *ldb_array, const scomplex* beta_array, scomplex** c_array, const f77_int *ldc_array, const f77_int* group_count, const f77_int *group_size)
{
    cgemm_batch_blis_impl( transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

void CGEMM_BATCH_BLIS_IMPL_( const f77_char* transa_array, const f77_char* transb_array,const f77_int *m_array, const f77_int *n_array, const f77_int *k_array,const scomplex* alpha_array, const scomplex** a_array, const  f77_int *lda_array, const scomplex** b_array, const f77_int *ldb_array, const scomplex* beta_array, scomplex** c_array, const f77_int *ldc_array, const f77_int* group_count, const f77_int *group_size)
{
    cgemm_batch_blis_impl( transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

void CGEMMT_BLIS_IMPL( const f77_char* uploc, const f77_char* transa, const f77_char* transb, const f77_int* n, const f77_int* k, const  scomplex* alpha, const scomplex* a, const f77_int* lda, const scomplex* b, const f77_int* ldb, const scomplex* beta, scomplex* c, const f77_int* ldc)
{
    cgemmt_blis_impl( uploc, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void cgemmt_blis_impl_( const f77_char* uploc, const f77_char* transa, const f77_char* transb, const f77_int* n, const f77_int* k, const  scomplex* alpha, const scomplex* a, const f77_int* lda, const scomplex* b, const f77_int* ldb, const scomplex* beta, scomplex* c, const f77_int* ldc)
{
    cgemmt_blis_impl( uploc, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void CGEMMT_BLIS_IMPL_( const f77_char* uploc, const f77_char* transa, const f77_char* transb, const f77_int* n, const f77_int* k, const  scomplex* alpha, const scomplex* a, const f77_int* lda, const scomplex* b, const f77_int* ldb, const scomplex* beta, scomplex* c, const f77_int* ldc)
{
    cgemmt_blis_impl( uploc, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#ifdef BLIS_ENABLE_CBLAS

void DASUMSUB_BLIS_IMPL(const f77_int* n, const double* x, const f77_int* incx, double* rval)
{
    dasumsub_blis_impl( n, x, incx, rval);
}

void dasumsub_blis_impl_(const f77_int* n, const double* x, const f77_int* incx, double* rval)
{
    dasumsub_blis_impl( n, x, incx, rval);
}

void DASUMSUB_BLIS_IMPL_(const f77_int* n, const double* x, const f77_int* incx, double* rval)
{
    dasumsub_blis_impl( n, x, incx, rval);
}

#endif // BLIS_ENABLE_CBLAS

void DAXPBY_BLIS_IMPL(const f77_int* n, const double* alpha, const double *x, const f77_int* incx, const double* beta, double *y, const f77_int* incy)
{
    daxpby_blis_impl( n, alpha, x, incx, beta, y, incy);
}

void daxpby_blis_impl_(const f77_int* n, const double* alpha, const double *x, const f77_int* incx, const double* beta, double *y, const f77_int* incy)
{
    daxpby_blis_impl( n, alpha, x, incx, beta, y, incy);
}

void DAXPBY_BLIS_IMPL_(const f77_int* n, const double* alpha, const double *x, const f77_int* incx, const double* beta, double *y, const f77_int* incy)
{
    daxpby_blis_impl( n, alpha, x, incx, beta, y, incy);
}

#ifdef BLIS_ENABLE_CBLAS

void DDOTSUB_BLIS_IMPL(const f77_int* n, const double* x, const f77_int* incx, const double* y, const f77_int* incy, double* rval)
{
    ddotsub_blis_impl( n, x, incx, y, incy, rval);
}

void ddotsub_blis_impl_(const f77_int* n, const double* x, const f77_int* incx, const double* y, const f77_int* incy, double* rval)
{
    ddotsub_blis_impl( n, x, incx, y, incy, rval);
}

void DDOTSUB_BLIS_IMPL_(const f77_int* n, const double* x, const f77_int* incx, const double* y, const f77_int* incy, double* rval)
{
    ddotsub_blis_impl( n, x, incx, y, incy, rval);
}

#endif // BLIS_ENABLE_CBLAS

void DGEMM_BATCH_BLIS_IMPL( const f77_char* transa_array, const f77_char* transb_array,const f77_int *m_array, const f77_int *n_array, const f77_int *k_array,const double* alpha_array, const double** a_array, const  f77_int *lda_array, const double** b_array, const f77_int *ldb_array, const double* beta_array, double** c_array, const f77_int *ldc_array, const f77_int* group_count, const f77_int *group_size)
{
    dgemm_batch_blis_impl( transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

void dgemm_batch_blis_impl_( const f77_char* transa_array, const f77_char* transb_array,const f77_int *m_array, const f77_int *n_array, const f77_int *k_array,const double* alpha_array, const double** a_array, const  f77_int *lda_array, const double** b_array, const f77_int *ldb_array, const double* beta_array, double** c_array, const f77_int *ldc_array, const f77_int* group_count, const f77_int *group_size)
{
    dgemm_batch_blis_impl( transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

void DGEMM_BATCH_BLIS_IMPL_( const f77_char* transa_array, const f77_char* transb_array,const f77_int *m_array, const f77_int *n_array, const f77_int *k_array,const double* alpha_array, const double** a_array, const  f77_int *lda_array, const double** b_array, const f77_int *ldb_array, const double* beta_array, double** c_array, const f77_int *ldc_array, const f77_int* group_count, const f77_int *group_size)
{
    dgemm_batch_blis_impl( transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

f77_int DGEMM_PACK_GET_SIZE_BLIS_IMPL(const f77_char* identifier, const f77_int* pm, const f77_int* pn, const f77_int* pk)
{
    return dgemm_pack_get_size_blis_impl( identifier, pm, pn, pk );
}

f77_int dgemm_pack_get_size_blis_impl_(const f77_char* identifier, const f77_int* pm, const f77_int* pn, const f77_int* pk)
{
    return dgemm_pack_get_size_blis_impl( identifier, pm, pn, pk );
}

f77_int DGEMM_PACK_GET_SIZE_BLIS_IMPL_(const f77_char* identifier, const f77_int* pm, const f77_int* pn, const f77_int* pk)
{
    return dgemm_pack_get_size_blis_impl( identifier, pm, pn, pk );
}

void DGEMM_PACK_BLIS_IMPL( const f77_char* identifier, const f77_char* trans, const f77_int* mm, const f77_int* nn, const f77_int* kk, const double* alpha, const double* src, const f77_int* pld, double* dest )
{
    dgemm_pack_blis_impl( identifier, trans, mm, nn, kk, alpha, src, pld, dest );
}

void dgemm_pack_blis_impl_( const f77_char* identifier, const f77_char* trans, const f77_int* mm, const f77_int* nn, const f77_int* kk, const double* alpha, const double* src, const f77_int* pld, double* dest )
{
    dgemm_pack_blis_impl( identifier, trans, mm, nn, kk, alpha, src, pld, dest );
}

void DGEMM_PACK_BLIS_IMPL_( const f77_char* identifier, const f77_char* trans, const f77_int* mm, const f77_int* nn, const f77_int* kk, const double* alpha, const double* src, const f77_int* pld, double* dest )
{
    dgemm_pack_blis_impl( identifier, trans, mm, nn, kk, alpha, src, pld, dest );
}

void DGEMM_COMPUTE_BLIS_IMPL( const f77_char* transa, const f77_char* transb, const f77_int* m, const f77_int* n, const f77_int* k, const double* a, const f77_int* lda, const double* b, const f77_int* ldb, const double* beta, double* c, const f77_int* ldc )
{
    f77_int rs_a = 1;
    f77_int rs_b = 1;
    f77_int rs_c = 1;
    dgemm_compute_blis_impl( transa, transb, m, n, k, a, &rs_a, lda, b, &rs_b, ldb, beta, c, &rs_c, ldc );
}

void dgemm_compute_blis_impl_( const f77_char* transa, const f77_char* transb, const f77_int* m, const f77_int* n, const f77_int* k, const double* a, const f77_int* lda, const double* b, const f77_int* ldb, const double* beta, double* c, const f77_int* ldc )
{
    f77_int rs_a = 1;
    f77_int rs_b = 1;
    f77_int rs_c = 1;
    dgemm_compute_blis_impl( transa, transb, m, n, k, a, &rs_a, lda, b, &rs_b, ldb, beta, c, &rs_c, ldc );
}

void DGEMM_COMPUTE_BLIS_IMPL_( const f77_char* transa, const f77_char* transb, const f77_int* m, const f77_int* n, const f77_int* k, const double* a, const f77_int* lda, const double* b, const f77_int* ldb, const double* beta, double* c, const f77_int* ldc )
{
    f77_int rs_a = 1;
    f77_int rs_b = 1;
    f77_int rs_c = 1;
    dgemm_compute_blis_impl( transa, transb, m, n, k, a, &rs_a, lda, b, &rs_b, ldb, beta, c, &rs_c, ldc );
}

void DGEMMT_BLIS_IMPL( const f77_char* uploc, const f77_char* transa, const f77_char* transb, const f77_int* n, const f77_int* k, const  double* alpha, const double* a, const f77_int* lda, const double* b, const f77_int* ldb, const double* beta, double* c, const f77_int* ldc)
{
    dgemmt_blis_impl( uploc, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void dgemmt_blis_impl_( const f77_char* uploc, const f77_char* transa, const f77_char* transb, const f77_int* n, const f77_int* k, const  double* alpha, const double* a, const f77_int* lda, const double* b, const f77_int* ldb, const double* beta, double* c, const f77_int* ldc)
{
    dgemmt_blis_impl( uploc, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void DGEMMT_BLIS_IMPL_( const f77_char* uploc, const f77_char* transa, const f77_char* transb, const f77_int* n, const f77_int* k, const  double* alpha, const double* a, const f77_int* lda, const double* b, const f77_int* ldb, const double* beta, double* c, const f77_int* ldc)
{
    dgemmt_blis_impl( uploc, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#ifdef BLIS_ENABLE_CBLAS

void DNRM2SUB_BLIS_IMPL(const f77_int* n, const double* x, const f77_int* incx, double *rval)
{
    dnrm2sub_blis_impl( n, x, incx, rval);
}

void dnrm2sub_blis_impl_(const f77_int* n, const double* x, const f77_int* incx, double *rval)
{
    dnrm2sub_blis_impl( n, x, incx, rval);
}

void DNRM2SUB_BLIS_IMPL_(const f77_int* n, const double* x, const f77_int* incx, double *rval)
{
    dnrm2sub_blis_impl( n, x, incx, rval);
}

#endif // BLIS_ENABLE_CBLAS

#ifdef BLIS_ENABLE_CBLAS

void DZASUMSUB_BLIS_IMPL(const f77_int* n, const dcomplex* x, const f77_int* incx, double* rval)
{
    dzasumsub_blis_impl( n, x, incx, rval);
}

void dzasumsub_blis_impl_(const f77_int* n, const dcomplex* x, const f77_int* incx, double* rval)
{
    dzasumsub_blis_impl( n, x, incx, rval);
}

void DZASUMSUB_BLIS_IMPL_(const f77_int* n, const dcomplex* x, const f77_int* incx, double* rval)
{
    dzasumsub_blis_impl( n, x, incx, rval);
}

void DZNRM2SUB_BLIS_IMPL(const f77_int* n, const dcomplex* x, const f77_int* incx, double* rval)
{
    dznrm2sub_blis_impl( n, x, incx, rval);
}

void dznrm2sub_blis_impl_(const f77_int* n, const dcomplex* x, const f77_int* incx, double* rval)
{
    dznrm2sub_blis_impl( n, x, incx, rval);
}

void DZNRM2SUB_BLIS_IMPL_(const f77_int* n, const dcomplex* x, const f77_int* incx, double* rval)
{
    dznrm2sub_blis_impl( n, x, incx, rval);
}

void ICAMAXSUB_BLIS_IMPL(const f77_int* n, const scomplex* x, const f77_int* incx, f77_int* rval)
{
    icamaxsub_blis_impl( n, x, incx, rval);
}

void icamaxsub_blis_impl_(const f77_int* n, const scomplex* x, const f77_int* incx, f77_int* rval)
{
    icamaxsub_blis_impl( n, x, incx, rval);
}

void ICAMAXSUB_BLIS_IMPL_(const f77_int* n, const scomplex* x, const f77_int* incx, f77_int* rval)
{
    icamaxsub_blis_impl( n, x, incx, rval);
}

#endif // BLIS_ENABLE_CBLAS

f77_int ICAMIN_BLIS_IMPL( const f77_int* n, const scomplex* x, const f77_int* incx)
{
    return icamin_blis_impl( n, x, incx);
}

f77_int icamin_blis_impl_( const f77_int* n, const scomplex* x, const f77_int* incx)
{
    return icamin_blis_impl( n, x, incx);
}

f77_int ICAMIN_BLIS_IMPL_( const f77_int* n, const scomplex* x, const f77_int* incx)
{
    return icamin_blis_impl( n, x, incx);
}

#ifdef BLIS_ENABLE_CBLAS

void ICAMINSUB_BLIS_IMPL( const f77_int* n, const scomplex* x, const f77_int* incx, f77_int* rval)
{
    icaminsub_blis_impl( n, x, incx, rval);
}

void icaminsub_blis_impl_( const f77_int* n, const scomplex* x, const f77_int* incx, f77_int* rval)
{
    icaminsub_blis_impl( n, x, incx, rval);
}

void ICAMINSUB_BLIS_IMPL_( const f77_int* n, const scomplex* x, const f77_int* incx, f77_int* rval)
{
    icaminsub_blis_impl( n, x, incx, rval);
}

void IDAMAXSUB_BLIS_IMPL( const f77_int* n, const double* x, const f77_int* incx, f77_int* rval)
{
    idamaxsub_blis_impl( n, x, incx, rval);
}

void idamaxsub_blis_impl_( const f77_int* n, const double* x, const f77_int* incx, f77_int* rval)
{
    idamaxsub_blis_impl( n, x, incx, rval);
}

void IDAMAXSUB_BLIS_IMPL_( const f77_int* n, const double* x, const f77_int* incx, f77_int* rval)
{
    idamaxsub_blis_impl( n, x, incx, rval);
}

#endif // BLIS_ENABLE_CBLAS

f77_int IDAMIN_BLIS_IMPL( const f77_int* n, const double* x, const f77_int* incx)
{
    return idamin_blis_impl( n, x, incx);
}

f77_int idamin_blis_impl_( const f77_int* n, const double* x, const f77_int* incx)
{
    return idamin_blis_impl( n, x, incx);
}

f77_int IDAMIN_BLIS_IMPL_( const f77_int* n, const double* x, const f77_int* incx)
{
    return idamin_blis_impl( n, x, incx);
}

#ifdef BLIS_ENABLE_CBLAS

void IDAMINSUB_BLIS_IMPL(const f77_int* n, const double* x, const f77_int* incx, f77_int* rval)
{
    idaminsub_blis_impl( n, x, incx, rval);
}

void idaminsub_blis_impl_(const f77_int* n, const double* x, const f77_int* incx, f77_int* rval)
{
    idaminsub_blis_impl( n, x, incx, rval);
}

void IDAMINSUB_BLIS_IMPL_(const f77_int* n, const double* x, const f77_int* incx, f77_int* rval)
{
    idaminsub_blis_impl( n, x, incx, rval);
}

void ISAMAXSUB_BLIS_IMPL( const f77_int* n, const float* x, const f77_int* incx, f77_int* rval)
{
    isamaxsub_blis_impl( n, x, incx, rval);
}

void isamaxsub_blis_impl_( const f77_int* n, const float* x, const f77_int* incx, f77_int* rval)
{
    isamaxsub_blis_impl( n, x, incx, rval);
}

void ISAMAXSUB_BLIS_IMPL_( const f77_int* n, const float* x, const f77_int* incx, f77_int* rval)
{
    isamaxsub_blis_impl( n, x, incx, rval);
}

#endif // BLIS_ENABLE_CBLAS

f77_int ISAMIN_BLIS_IMPL( const f77_int* n, const float* x, const f77_int* incx)
{
    return isamin_blis_impl( n, x, incx);
}

f77_int isamin_blis_impl_( const f77_int* n, const float* x, const f77_int* incx)
{
    return isamin_blis_impl( n, x, incx);
}

f77_int ISAMIN_BLIS_IMPL_( const f77_int* n, const float* x, const f77_int* incx)
{
    return isamin_blis_impl( n, x, incx);
}

#ifdef BLIS_ENABLE_CBLAS

void ISAMINSUB_BLIS_IMPL( const f77_int* n, const float* x, const f77_int* incx, f77_int* rval)
{
    isaminsub_blis_impl( n, x, incx, rval);
}

void isaminsub_blis_impl_( const f77_int* n, const float* x, const f77_int* incx, f77_int* rval)
{
    isaminsub_blis_impl( n, x, incx, rval);
}

void ISAMINSUB_BLIS_IMPL_( const f77_int* n, const float* x, const f77_int* incx, f77_int* rval)
{
    isaminsub_blis_impl( n, x, incx, rval);
}

void IZAMAXSUB_BLIS_IMPL( const f77_int* n, const dcomplex* x, const f77_int* incx, f77_int* rval)
{
    izamaxsub_blis_impl( n, x, incx, rval);
}

void izamaxsub_blis_impl_( const f77_int* n, const dcomplex* x, const f77_int* incx, f77_int* rval)
{
    izamaxsub_blis_impl( n, x, incx, rval);
}

void IZAMAXSUB_BLIS_IMPL_( const f77_int* n, const dcomplex* x, const f77_int* incx, f77_int* rval)
{
    izamaxsub_blis_impl( n, x, incx, rval);
}

#endif // BLIS_ENABLE_CBLAS

f77_int IZAMIN_BLIS_IMPL( const f77_int* n, const dcomplex* x, const f77_int* incx)
{
    return izamin_blis_impl( n, x, incx);
}

f77_int izamin_blis_impl_( const f77_int* n, const dcomplex* x, const f77_int* incx)
{
    return izamin_blis_impl( n, x, incx);
}

f77_int IZAMIN_BLIS_IMPL_( const f77_int* n, const dcomplex* x, const f77_int* incx)
{
    return izamin_blis_impl( n, x, incx);
}

#ifdef BLIS_ENABLE_CBLAS

void IZAMINSUB_BLIS_IMPL( const f77_int* n, const dcomplex* x, const f77_int* incx, f77_int* rval)
{
    izaminsub_blis_impl( n, x, incx, rval);
}

void izaminsub_blis_impl_( const f77_int* n, const dcomplex* x, const f77_int* incx, f77_int* rval)
{
    izaminsub_blis_impl( n, x, incx, rval);
}

void IZAMINSUB_BLIS_IMPL_( const f77_int* n, const dcomplex* x, const f77_int* incx, f77_int* rval)
{
    izaminsub_blis_impl( n, x, incx, rval);
}

void SASUMSUB_BLIS_IMPL( const f77_int* n, const float* x, const f77_int* incx, float* rval)
{
    sasumsub_blis_impl( n, x, incx, rval);
}

void sasumsub_blis_impl_( const f77_int* n, const float* x, const f77_int* incx, float* rval)
{
    sasumsub_blis_impl( n, x, incx, rval);
}

void SASUMSUB_BLIS_IMPL_( const f77_int* n, const float* x, const f77_int* incx, float* rval)
{
    sasumsub_blis_impl( n, x, incx, rval);
}

#endif // BLIS_ENABLE_CBLAS

void SAXPBY_BLIS_IMPL( const f77_int* n, const float* alpha, const float *x, const f77_int* incx, const float* beta, float *y, const f77_int* incy)
{
    saxpby_blis_impl( n, alpha, x, incx, beta, y, incy);
}

void saxpby_blis_impl_( const f77_int* n, const float* alpha, const float *x, const f77_int* incx, const float* beta, float *y, const f77_int* incy)
{
    saxpby_blis_impl( n, alpha, x, incx, beta, y, incy);
}

void SAXPBY_BLIS_IMPL_( const f77_int* n, const float* alpha, const float *x, const f77_int* incx, const float* beta, float *y, const f77_int* incy)
{
    saxpby_blis_impl( n, alpha, x, incx, beta, y, incy);
}

#ifdef BLIS_ENABLE_CBLAS

void SCASUMSUB_BLIS_IMPL( const f77_int* n, const scomplex* x, const f77_int* incx, float* rval)
{
    scasumsub_blis_impl( n, x, incx, rval);
}

void scasumsub_blis_impl_( const f77_int* n, const scomplex* x, const f77_int* incx, float* rval)
{
    scasumsub_blis_impl( n, x, incx, rval);
}

void SCASUMSUB_BLIS_IMPL_( const f77_int* n, const scomplex* x, const f77_int* incx, float* rval)
{
    scasumsub_blis_impl( n, x, incx, rval);
}

void SCNRM2SUB_BLIS_IMPL( const f77_int* n, const scomplex* x, const f77_int* incx, float* rval)
{
    scnrm2sub_blis_impl( n, x, incx, rval);
}

void scnrm2sub_blis_impl_( const f77_int* n, const scomplex* x, const f77_int* incx, float* rval)
{
    scnrm2sub_blis_impl( n, x, incx, rval);
}

void SCNRM2SUB_BLIS_IMPL_( const f77_int* n, const scomplex* x, const f77_int* incx, float* rval)
{
    scnrm2sub_blis_impl( n, x, incx, rval);
}

void SDOTSUB_BLIS_IMPL( const f77_int* n, const float* x, const f77_int* incx, const float* y, const f77_int* incy, float* rval)
{
    sdotsub_blis_impl( n, x, incx, y, incy, rval);
}

void sdotsub_blis_impl_( const f77_int* n, const float* x, const f77_int* incx, const float* y, const f77_int* incy, float* rval)
{
    sdotsub_blis_impl( n, x, incx, y, incy, rval);
}

void SDOTSUB_BLIS_IMPL_( const f77_int* n, const float* x, const f77_int* incx, const float* y, const f77_int* incy, float* rval)
{
    sdotsub_blis_impl( n, x, incx, y, incy, rval);
}

#endif // BLIS_ENABLE_CBLAS

void SGEMM_BATCH_BLIS_IMPL(const f77_char* transa_array, const f77_char* transb_array,const f77_int *m_array, const f77_int *n_array, const f77_int *k_array,const float* alpha_array, const float** a_array, const  f77_int *lda_array, const float** b_array, const f77_int *ldb_array, const float* beta_array, float** c_array, const f77_int *ldc_array, const f77_int* group_count, const f77_int *group_size)
{
    sgemm_batch_blis_impl( transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

void sgemm_batch_blis_impl_(const f77_char* transa_array, const f77_char* transb_array,const f77_int *m_array, const f77_int *n_array, const f77_int *k_array,const float* alpha_array, const float** a_array, const  f77_int *lda_array, const float** b_array, const f77_int *ldb_array, const float* beta_array, float** c_array, const f77_int *ldc_array, const f77_int* group_count, const f77_int *group_size)
{
    sgemm_batch_blis_impl( transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

void SGEMM_BATCH_BLIS_IMPL_(const f77_char* transa_array, const f77_char* transb_array,const f77_int *m_array, const f77_int *n_array, const f77_int *k_array,const float* alpha_array, const float** a_array, const  f77_int *lda_array, const float** b_array, const f77_int *ldb_array, const float* beta_array, float** c_array, const f77_int *ldc_array, const f77_int* group_count, const f77_int *group_size)
{
    sgemm_batch_blis_impl( transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

f77_int SGEMM_PACK_GET_SIZE_BLIS_IMPL(const f77_char* identifier, const f77_int* pm, const f77_int* pn, const f77_int* pk)
{
    return sgemm_pack_get_size_blis_impl( identifier, pm, pn, pk );
}

f77_int sgemm_pack_get_size_blis_impl_(const f77_char* identifier, const f77_int* pm, const f77_int* pn, const f77_int* pk)
{
    return sgemm_pack_get_size_blis_impl( identifier, pm, pn, pk );
}

f77_int SGEMM_PACK_GET_SIZE_BLIS_IMPL_(const f77_char* identifier, const f77_int* pm, const f77_int* pn, const f77_int* pk)
{
    return sgemm_pack_get_size_blis_impl( identifier, pm, pn, pk );
}

void SGEMM_PACK_BLIS_IMPL( const f77_char* identifier, const f77_char* trans, const f77_int* mm, const f77_int* nn, const f77_int* kk, const float* alpha, const float* src, const f77_int* pld, float* dest )
{
    sgemm_pack_blis_impl( identifier, trans, mm, nn, kk, alpha, src, pld, dest );
}

void sgemm_pack_blis_impl_( const f77_char* identifier, const f77_char* trans, const f77_int* mm, const f77_int* nn, const f77_int* kk, const float* alpha, const float* src, const f77_int* pld, float* dest )
{
    sgemm_pack_blis_impl( identifier, trans, mm, nn, kk, alpha, src, pld, dest );
}

void SGEMM_PACK_BLIS_IMPL_( const f77_char* identifier, const f77_char* trans, const f77_int* mm, const f77_int* nn, const f77_int* kk, const float* alpha, const float* src, const f77_int* pld, float* dest )
{
    sgemm_pack_blis_impl( identifier, trans, mm, nn, kk, alpha, src, pld, dest );
}

void SGEMM_COMPUTE_BLIS_IMPL( const f77_char* transa, const f77_char* transb, const f77_int* m, const f77_int* n, const f77_int* k, const float* a, const f77_int* lda, const float* b, const f77_int* ldb, const float* beta, float* c, const f77_int* ldc )
{
    f77_int rs_a = 1;
    f77_int rs_b = 1;
    f77_int rs_c = 1;
    sgemm_compute_blis_impl( transa, transb, m, n, k, a, &rs_a, lda, b, &rs_b, ldb, beta, c, &rs_c, ldc );
}

void sgemm_compute_blis_impl_( const f77_char* transa, const f77_char* transb, const f77_int* m, const f77_int* n, const f77_int* k, const float* a, const f77_int* lda, const float* b, const f77_int* ldb, const float* beta, float* c, const f77_int* ldc )
{
    f77_int rs_a = 1;
    f77_int rs_b = 1;
    f77_int rs_c = 1;
    sgemm_compute_blis_impl( transa, transb, m, n, k, a, &rs_a, lda, b, &rs_b, ldb, beta, c, &rs_c, ldc );
}

void SGEMM_COMPUTE_BLIS_IMPL_( const f77_char* transa, const f77_char* transb, const f77_int* m, const f77_int* n, const f77_int* k, const float* a, const f77_int* lda, const float* b, const f77_int* ldb, const float* beta, float* c, const f77_int* ldc )
{
    f77_int rs_a = 1;
    f77_int rs_b = 1;
    f77_int rs_c = 1;
    sgemm_compute_blis_impl( transa, transb, m, n, k, a, &rs_a, lda, b, &rs_b, ldb, beta, c, &rs_c, ldc );
}

void SGEMMT_BLIS_IMPL( const f77_char* uploc, const f77_char* transa, const f77_char* transb, const f77_int* n, const f77_int* k, const  float* alpha, const float* a, const f77_int* lda, const float* b, const f77_int* ldb, const float* beta, float* c, const f77_int* ldc)
{
    sgemmt_blis_impl( uploc, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void sgemmt_blis_impl_( const f77_char* uploc, const f77_char* transa, const f77_char* transb, const f77_int* n, const f77_int* k, const  float* alpha, const float* a, const f77_int* lda, const float* b, const f77_int* ldb, const float* beta, float* c, const f77_int* ldc)
{
    sgemmt_blis_impl( uploc, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void SGEMMT_BLIS_IMPL_( const f77_char* uploc, const f77_char* transa, const f77_char* transb, const f77_int* n, const f77_int* k, const  float* alpha, const float* a, const f77_int* lda, const float* b, const f77_int* ldb, const float* beta, float* c, const f77_int* ldc)
{
    sgemmt_blis_impl( uploc, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#ifdef BLIS_ENABLE_CBLAS

void SNRM2SUB_BLIS_IMPL( const f77_int* n, const float* x, const f77_int* incx, float *rval)
{
    snrm2sub_blis_impl( n, x, incx, rval);
}

void snrm2sub_blis_impl_( const f77_int* n, const float* x, const f77_int* incx, float *rval)
{
    snrm2sub_blis_impl( n, x, incx, rval);
}

void SNRM2SUB_BLIS_IMPL_( const f77_int* n, const float* x, const f77_int* incx, float *rval)
{
    snrm2sub_blis_impl( n, x, incx, rval);
}

#endif // BLIS_ENABLE_CBLAS

void ZAXPBY_BLIS_IMPL( const f77_int* n, const dcomplex* alpha, const dcomplex *x, const f77_int* incx, const dcomplex* beta, dcomplex *y, const f77_int* incy)
{
    zaxpby_blis_impl( n, alpha, x, incx, beta, y, incy);
}

void zaxpby_blis_impl_( const f77_int* n, const dcomplex* alpha, const dcomplex *x, const f77_int* incx, const dcomplex* beta, dcomplex *y, const f77_int* incy)
{
    zaxpby_blis_impl( n, alpha, x, incx, beta, y, incy);
}

void ZAXPBY_BLIS_IMPL_( const f77_int* n, const dcomplex* alpha, const dcomplex *x, const f77_int* incx, const dcomplex* beta, dcomplex *y, const f77_int* incy)
{
    zaxpby_blis_impl( n, alpha, x, incx, beta, y, incy);
}

#ifdef BLIS_ENABLE_CBLAS

void ZDOTCSUB_BLIS_IMPL( const f77_int* n, const dcomplex* x, const f77_int* incx, const dcomplex* y, const f77_int* incy, dcomplex* rval)
{
    zdotcsub_blis_impl( n, x, incx, y, incy, rval);
}

void zdotcsub_blis_impl_( const f77_int* n, const dcomplex* x, const f77_int* incx, const dcomplex* y, const f77_int* incy, dcomplex* rval)
{
    zdotcsub_blis_impl( n, x, incx, y, incy, rval);
}

void ZDOTCSUB_BLIS_IMPL_( const f77_int* n, const dcomplex* x, const f77_int* incx, const dcomplex* y, const f77_int* incy, dcomplex* rval)
{
    zdotcsub_blis_impl( n, x, incx, y, incy, rval);
}

void ZDOTUSUB_BLIS_IMPL( const f77_int* n, const dcomplex* x, const f77_int* incx,const dcomplex* y, const f77_int* incy, dcomplex* rval)
{
    zdotusub_blis_impl( n, x, incx, y, incy, rval);
}

void zdotusub_blis_impl_( const f77_int* n, const dcomplex* x, const f77_int* incx,const dcomplex* y, const f77_int* incy, dcomplex* rval)
{
    zdotusub_blis_impl( n, x, incx, y, incy, rval);
}

void ZDOTUSUB_BLIS_IMPL_( const f77_int* n, const dcomplex* x, const f77_int* incx,const dcomplex* y, const f77_int* incy, dcomplex* rval)
{
    zdotusub_blis_impl( n, x, incx, y, incy, rval);
}

#endif // BLIS_ENABLE_CBLAS

void ZGEMM3M_BLIS_IMPL( const f77_char* transa, const f77_char* transb, const f77_int* m, const f77_int* n, const f77_int* k, const dcomplex* alpha, const dcomplex* a, const f77_int* lda, const dcomplex* b, const f77_int* ldb, const dcomplex* beta, dcomplex* c, const f77_int* ldc)
{
    zgemm3m_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void zgemm3m_blis_impl_( const f77_char* transa, const f77_char* transb, const f77_int* m, const f77_int* n, const f77_int* k, const dcomplex* alpha, const dcomplex* a, const f77_int* lda, const dcomplex* b, const f77_int* ldb, const dcomplex* beta, dcomplex* c, const f77_int* ldc)
{
    zgemm3m_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ZGEMM3M_BLIS_IMPL_( const f77_char* transa, const f77_char* transb, const f77_int* m, const f77_int* n, const f77_int* k, const dcomplex* alpha, const dcomplex* a, const f77_int* lda, const dcomplex* b, const f77_int* ldb, const dcomplex* beta, dcomplex* c, const f77_int* ldc)
{
    zgemm3m_blis_impl( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ZGEMM_BATCH_BLIS_IMPL(  const f77_char* transa_array, const f77_char* transb_array,const f77_int *m_array, const f77_int *n_array, const f77_int *k_array,const dcomplex* alpha_array, const dcomplex** a_array, const  f77_int *lda_array, const dcomplex** b_array, const f77_int *ldb_array, const dcomplex* beta_array, dcomplex** c_array, const f77_int *ldc_array, const f77_int* group_count, const f77_int *group_size)
{
    zgemm_batch_blis_impl( transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

void zgemm_batch_blis_impl_(  const f77_char* transa_array, const f77_char* transb_array,const f77_int *m_array, const f77_int *n_array, const f77_int *k_array,const dcomplex* alpha_array, const dcomplex** a_array, const  f77_int *lda_array, const dcomplex** b_array, const f77_int *ldb_array, const dcomplex* beta_array, dcomplex** c_array, const f77_int *ldc_array, const f77_int* group_count, const f77_int *group_size)
{
    zgemm_batch_blis_impl( transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

void ZGEMM_BATCH_BLIS_IMPL_(  const f77_char* transa_array, const f77_char* transb_array,const f77_int *m_array, const f77_int *n_array, const f77_int *k_array,const dcomplex* alpha_array, const dcomplex** a_array, const  f77_int *lda_array, const dcomplex** b_array, const f77_int *ldb_array, const dcomplex* beta_array, dcomplex** c_array, const f77_int *ldc_array, const f77_int* group_count, const f77_int *group_size)
{
    zgemm_batch_blis_impl( transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

void ZGEMMT_BLIS_IMPL( const f77_char* uploc, const f77_char* transa, const f77_char* transb, const f77_int* n, const f77_int* k, const  dcomplex* alpha, const dcomplex* a, const f77_int* lda, const dcomplex* b, const f77_int* ldb, const dcomplex* beta, dcomplex* c, const f77_int* ldc)
{
    zgemmt_blis_impl( uploc, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void zgemmt_blis_impl_( const f77_char* uploc, const f77_char* transa, const f77_char* transb, const f77_int* n, const f77_int* k, const  dcomplex* alpha, const dcomplex* a, const f77_int* lda, const dcomplex* b, const f77_int* ldb, const dcomplex* beta, dcomplex* c, const f77_int* ldc)
{
    zgemmt_blis_impl( uploc, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void ZGEMMT_BLIS_IMPL_( const f77_char* uploc, const f77_char* transa, const f77_char* transb, const f77_int* n, const f77_int* k, const  dcomplex* alpha, const dcomplex* a, const f77_int* lda, const dcomplex* b, const f77_int* ldb, const dcomplex* beta, dcomplex* c, const f77_int* ldc)
{
    zgemmt_blis_impl( uploc, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

float SCABS1_BLIS_IMPL(bla_scomplex* z)
{
    return scabs1_blis_impl( z);
}

float scabs1_blis_impl_(bla_scomplex* z)
{
    return scabs1_blis_impl( z);
}

float SCABS1_BLIS_IMPL_(bla_scomplex* z)
{
    return scabs1_blis_impl( z);

}

#ifdef BLIS_ENABLE_CBLAS

void SDSDOTSUB_BLIS_IMPL( const f77_int* n, float* sb, const float* x, const f77_int* incx, const float* y, const f77_int* incy, float* dot)
{
    sdsdotsub_blis_impl( n, sb, x, incx, y, incy, dot);
}

void sdsdotsub_blis_impl_( const f77_int* n, float* sb, const float* x, const f77_int* incx, const float* y, const f77_int* incy, float* dot)
{
    sdsdotsub_blis_impl( n, sb, x, incx, y, incy, dot);
}

void SDSDOTSUB_BLIS_IMPL_( const f77_int* n, float* sb, const float* x, const f77_int* incx, const float* y, const f77_int* incy, float* dot)
{
    sdsdotsub_blis_impl( n, sb, x, incx, y, incy, dot);
}

void DSDOTSUB_BLIS_IMPL( const f77_int* n, const float* x, const f77_int* incx, const float* y, const f77_int* incy, double* dot)
{
    dsdotsub_blis_impl( n, x, incx, y, incy, dot);
}

void dsdotsub_blis_impl_( const f77_int* n, const float* x, const f77_int* incx, const float* y, const f77_int* incy, double* dot)
{
    dsdotsub_blis_impl( n, x, incx, y, incy, dot);
}

void DSDOTSUB_BLIS_IMPL_( const f77_int* n, const float* x, const f77_int* incx, const float* y, const f77_int* incy, double* dot)
{
    dsdotsub_blis_impl( n, x, incx, y, incy, dot);
}

#endif // BLIS_ENABLE_CBLAS

void CAXPBY_BLIS_IMPL( const f77_int* n,  const scomplex* alpha,  const scomplex *x,  const f77_int* incx,  const scomplex* beta,  scomplex *y,  const f77_int* incy)
{
    caxpby_blis_impl(n, alpha, x, incx, beta, y, incy);
}

void caxpby_blis_impl_( const f77_int* n,  const scomplex* alpha,  const scomplex *x,  const f77_int* incx,  const scomplex* beta,  scomplex *y,  const f77_int* incy)
{
    caxpby_blis_impl(n, alpha, x, incx, beta, y, incy);
}

void CAXPBY_BLIS_IMPL_( const f77_int* n,  const scomplex* alpha,  const scomplex *x,  const f77_int* incx,  const scomplex* beta,  scomplex *y,  const f77_int* incy)
{
    caxpby_blis_impl(n, alpha, x, incx, beta, y, incy);
}

#endif
#endif
