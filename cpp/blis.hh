/******************************************************************************
* Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

/*! @file blis.hh
 *  blis.hh defines all the BLAS CPP templated public interfaces
 *  */
#ifndef BLIS_HH
#define BLIS_HH

#include "cblas.hh"
#include "blis_util.hh"

namespace blis {

template< typename T >
void rotg(
    T *a,
    T *b,
    T *c,
    T *s )
{
    cblas_rotg(a, b, c, s);
}

template< typename T >
void rotmg(
    T *d1,
    T *d2,
    T *a,
    T  b,
    T  param[5] )
{
    cblas_rotmg(d1, d2, a, b, param );
}

template< typename T >
void rot(
    int64_t n,
    T *x, int64_t incx,
    T *y, int64_t incy,
    T c,
    T s )
{
    cblas_rot( n, x, incx, y, incy, c, s );
}

template< typename T >
void rotm(
    int64_t n,
    T *x, int64_t incx,
    T *y, int64_t incy,
    const T *P)
{
    cblas_rotm( n, x, incx, y, incy, P );
}

template< typename T >
void swap(
    int64_t n,
    T *x, int64_t incx,
    T *y, int64_t incy )
{
    cblas_swap( n, x, incx, y, incy );
}

template< typename T >
void scal(
    int64_t n,
    T alpha,
    T* x, int64_t incx )
{
    cblas_scal( n, alpha, x, incx );
}

template< typename T >
void copy(
    int64_t n,
    T const *x, int64_t incx,
    T       *y, int64_t incy )
{
    cblas_copy( n, x, incx, y, incy );
}

template< typename T >
void axpy(
    int64_t n,
    T alpha,
    T const *x, int64_t incx,
    T       *y, int64_t incy )
{
    cblas_axpy( n, alpha, x, incx, y, incy );
}

template< typename TX, typename TY >
TY dot(
    int64_t n,
    TX const *x, int64_t incx,
    TX const *y, int64_t incy )
{
    return cblas_dot( n, x, incx, y, incy );
}

template< typename T >
T dotu(
    int64_t n,
    T const *x, int64_t incx,
    T const *y, int64_t incy )
{
    return cblas_dotu( n, x, incx, y, incy );
}

template< typename T >
T dotc(
    int64_t n,
    T const *x, int64_t incx,
    T const *y, int64_t incy )
{
    return cblas_dotc( n, x, incx, y, incy );
}

template< typename T >
T sdsdot(
    int64_t n,
    T alpha,
    T const *x, int64_t incx,
    T const *y, int64_t incy )
{
    return cblas_sdsdot( n, alpha, x, incx, y, incy );
}

template< typename T >
real_type<T>
nrm2(
    int64_t n,
    T const * x, int64_t incx )
{
    return cblas_nrm2( n, x, incx );
}

template< typename T >
real_type<T>
asum(
    int64_t n,
    T const *x, int64_t incx )
{
    return cblas_asum( n, x, incx );
}

template< typename T >
int64_t iamax(
    int64_t n,
    T const *x, int64_t incx )
{
    return cblas_iamax( n, x, incx );
}
template< typename T >
void gemv(
    CBLAS_ORDER layout,
    CBLAS_TRANSPOSE trans,
    int64_t m, int64_t n,
    T alpha,
    T const *A, int64_t lda,
    T const *x, int64_t incx,
    T beta,
    T *y, int64_t incy )
{
    cblas_gemv(layout, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template< typename T >
void gbmv(
    CBLAS_ORDER layout,
    CBLAS_TRANSPOSE trans,
    int64_t m, int64_t n,
    int64_t kl, int64_t ku,
    T alpha,
    T const *A, int64_t lda,
    T const *x, int64_t incx,
    T beta,
    T *y, int64_t incy )
{
    cblas_gbmv(layout, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}

template< typename T >
void hemv(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    int64_t n,
    T alpha,
    T const *A, int64_t lda,
    T const *x, int64_t incx,
    T beta,
    T *y, int64_t incy )
{
    cblas_hemv(layout, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

template< typename T >
void hbmv(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    int64_t n, int64_t k,
    T alpha,
    T const *A, int64_t lda,
    T const *x, int64_t incx,
    T beta,
    T *y, int64_t incy )
{
    cblas_hbmv(layout, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

template< typename T >
void hpmv(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    int64_t n,
    T alpha,
    T const *Ap,
    T const *x, int64_t incx,
    T beta,
    T *y, int64_t incy )
{
    cblas_hpmv(layout, uplo, n, alpha, Ap, x, incx, beta, y, incy);
}

template< typename T >
void symv(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    int64_t n,
    T alpha,
    T const *A, int64_t lda,
    T const *x, int64_t incx,
    T beta,
    T *y, int64_t incy )
{
    cblas_symv(layout, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

template< typename T >
void sbmv(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    int64_t n, int64_t k,
    T alpha,
    T const *A, int64_t lda,
    T const *x, int64_t incx,
    T beta,
    T *y, int64_t incy )
{
    cblas_sbmv(layout, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

template< typename T >
void spmv(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    int64_t n,
    T alpha,
    T const *Ap,
    T const *x, int64_t incx,
    T beta,
    T *y, int64_t incy )
{
    cblas_spmv(layout, uplo, n, alpha, Ap, x, incx, beta, y, incy);
}

template< typename T >
void trmv(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    int64_t n,
    T const *A, int64_t lda,
    T       *x, int64_t incx )
{
    cblas_trmv(layout, uplo, trans, diag, n, A, lda, x, incx);
}

template< typename T >
void tbmv(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    int64_t n, int64_t k,
    T const *A, int64_t lda,
    T       *x, int64_t incx )
{
    cblas_tbmv(layout, uplo, trans, diag, n, k, A, lda, x, incx);
}


template< typename T >
void tpmv(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    int64_t n,
    T const *Ap,
    T       *x, int64_t incx )
{
    cblas_tpmv(layout, uplo, trans, diag, n, Ap, x, incx);
}

template< typename T >
void trsv(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    int64_t n,
    T const *A, int64_t lda,
    T       *x, int64_t incx )
{
    cblas_trsv(layout, uplo, trans, diag, n, A, lda, x, incx);
}

template< typename T >
void tbsv(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    int64_t n, int64_t k,
    T const *A, int64_t lda,
    T       *x, int64_t incx )
{
    cblas_tbsv(layout, uplo, trans, diag, n, k, A, lda, x, incx);
}


template< typename T >
void tpsv(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    int64_t n,
    T const *Ap,
    T       *x, int64_t incx )
{
    cblas_tpsv(layout, uplo, trans, diag, n, Ap, x, incx);
}

template< typename T >
void ger(
    CBLAS_ORDER layout,
    int64_t m, int64_t n,
    T alpha,
    T const *x, int64_t incx,
    T const *y, int64_t incy,
    T *A, int64_t lda )
{
    cblas_ger(layout, m, n, alpha, x, incx, y, incy, A, lda);
}

template< typename T >
void geru(
    CBLAS_ORDER layout,
    int64_t m, int64_t n,
    T alpha,
    T const *x, int64_t incx,
    T const *y, int64_t incy,
    T *A, int64_t lda )
{
    cblas_geru(layout, m, n, alpha, x, incx, y, incy, A, lda);
}

template< typename T >
void gerc(
    CBLAS_ORDER layout,
    int64_t m, int64_t n,
    T alpha,
    T const *x, int64_t incx,
    T const *y, int64_t incy,
    T *A, int64_t lda )
{
    cblas_gerc(layout, m, n, alpha, x, incx, y, incy, A, lda);
}

template< typename T >
void her(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    int64_t n,
    real_type<T> alpha,  // zher takes double alpha; use real
    T const *x, int64_t incx,
    T       *A, int64_t lda )
{
    cblas_her(layout, uplo, n, alpha, x, incx, A, lda);
}

template< typename T >
void hpr(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    int64_t n,
    real_type<T> alpha,  // zher takes double alpha; use real
    T const *x, int64_t incx,
    T       *Ap )
{
    cblas_hpr(layout, uplo, n, alpha, x, incx, Ap);
}
template< typename T >
void her2(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    int64_t n,
    T alpha,  
    T const *x, int64_t incx,
    T const *y, int64_t incy,
    T       *A, int64_t lda )
{
    cblas_her2(layout, uplo, n, alpha, x, incx, y, incy, A, lda);
}

template< typename T >
void hpr2(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    int64_t n,
    T alpha,  
    T const *x, int64_t incx,
    T const *y, int64_t incy,
    T       *Ap )
{
    cblas_hpr2(layout, uplo, n, alpha, x, incx, y, incy, Ap);
}
template< typename T >
void syr(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    int64_t n,
    T alpha,  
    T const *x, int64_t incx,
    T       *A, int64_t lda )
{
    cblas_syr(layout, uplo, n, alpha, x, incx, A, lda);
}

template< typename T >
void spr(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    int64_t n,
    T alpha,  
    T const *x, int64_t incx,
    T       *Ap )
{
    cblas_spr(layout, uplo, n, alpha, x, incx, Ap);
}
template< typename T >
void syr2(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    int64_t n,
    T alpha,
    T const *x, int64_t incx,
    T const *y, int64_t incy,
    T       *A, int64_t lda )
{
    cblas_syr2(layout, uplo, n, alpha, x, incx, y, incy, A, lda);
}

template< typename T >
void spr2(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    int64_t n,
    T alpha,
    T const *x, int64_t incx,
    T const *y, int64_t incy,
    T       *Ap )
{
    cblas_spr2(layout, uplo, n, alpha, x, incx, y, incy, Ap);
}

/*! \b Purpose:	

  GEMM  performs general matrix-matrix multiply for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

  C := alpha*op( A )*op( B ) + beta*C,

  where  op( X ) is one of

  op( X ) = X   or   op( X ) = X**T   or   op( X ) = X**H,

  alpha and beta are scalars, and A, B and C are matrices, with op( A )
  an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.

  \param[in] layout
  layout is enum CBLAS_ORDER
  layout specifies Matrix storage as follows:
  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.

  \param[in] transA
  transA is CBLAS_TRANSPOSE
  On entry, transA specifies the form of op( A ) to be used in
  the matrix multiplication as follows:
  transA = CBLAS_TRANSPOSE::CblasNoTrans,  op( A ) = A.
  transA = CBLAS_TRANSPOSE::CblasTrans,  op( A ) = A**T.
  transA = CBLAS_TRANSPOSE::CblasConjTrans,  op( A ) = A**H.

  \param[in] transB
  transB is CBLAS_TRANSPOSE
  On entry, transB specifies the form of op( B ) to be used in
  the matrix multiplication as follows:
  transB = CBLAS_TRANSPOSE::CblasNoTrans,  op( B ) = B.
  transB = CBLAS_TRANSPOSE::CblasTrans,  op( B ) = B**T.
  transB = CBLAS_TRANSPOSE::CblasConjTrans,  op( B ) = B**H.

  \param[in] m
  m is INTEGER
  On entry,  m  specifies  the number  of rows  of the  matrix
  op( A )  and of the  matrix  C.  m  must  be at least  zero.

  \param[in] n
  n is INTEGER
  On entry,  n  specifies the number  of columns of the matrix
  op( B ) and the number of columns of the matrix C. n must be
  at least zero.

  \param[in] k
  k is INTEGER
  On entry,  k  specifies  the number of columns of the matrix
  op( A ) and the number of rows of the matrix op( B ). k must
  be at least  zero.

  \param[in] alpha
  alpha is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, alpha specifies the scalar alpha.

  \param[in] A
  A is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If transA = CblasNoTrans:
  m-by-k , stored in an lda-by-k array [RowMajor: m-by-lda].
  Otherwise:
  k-by-m , stored in an lda-by-m array [RowMajor: k-by-lda].

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  If transA = CblasNoTrans: lda >= max(1, m) [RowMajor: lda >= max(1, k)].
  Otherwise:                lda >= max(1, k) [RowMajor: lda >= max(1, m)].

  \param[in] B
  B is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If transA = CblasNoTrans:
  k-by-n , stored in an ldb-by-n array [RowMajor: k-by-ldb].
  Otherwise:
  n-by-k , stored in an ldb-by-k array [RowMajor: n-by-ldb].

  \param[in] ldb
  ldb is INTEGER
  On entry, ldb specifies the Leading dimension of B
  If transA = CblasNoTrans: ldb >= max(1, k) [RowMajor: ldb >= max(1, n)].
  Otherwise:                ldb >= max(1, n) [RowMajor: ldb >= max(1, k)].

  \param[in] beta
  beta is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, beta specifies the scalar alpha.When  beta  is
  supplied as zero then C need not be set on input.

  \param[in,out] C
  C is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array, dimension :
  m-by-n stored in an ldc-by-n array [RowMajor: m-by-ldc].
  Before entry, the leading  m by n  part of the array  C must
  contain the matrix  C,  except when  beta  is zero, in which
  case C need not be set on entry.
  On exit, the array  C  is overwritten by the  m by n  matrix
  ( alpha*op( A )*op( B ) + beta*C ).

  \param[in] ldc
  ldc is INTEGER
  On entry, ldc specifies the first dimension of C
  ldc >= max(1, m) [RowMajor: ldc >= max(1, n)].
  */	
template< typename T >
void gemm(
    CBLAS_ORDER layout,
    CBLAS_TRANSPOSE transA,
    CBLAS_TRANSPOSE transB,
    int64_t m, int64_t n, int64_t k,
    T alpha,
    T const *A, int64_t lda,
    T const *B, int64_t ldb,
    T beta,
    T       *C, int64_t ldc )
{
    cblas_gemm(layout, transA, transB, m, n, k, alpha, A,lda, B, ldb, beta, C, ldc);

}

/*! \b Purpose:	

  TRSM  performs solves one of the matrix equations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

  op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,

  where alpha is a scalar, X and B are m by n matrices, A is a unit, or
  non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
  where  op( X ) is one of
      
  op( A ) = A   or   op( A ) = A**T   or   op( A ) = A**H.

  The matrix X is overwritten on B.

  \param[in] layout
  layout is enum CBLAS_ORDER
  layout specifies Matrix storage as follows:
  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.

  \param[in] side
  side is enum CBLAS_SIDE
  side specifies specifies whether op( A ) appears on the left
  or right of X as follows:
  side = CBLAS_SIDE::CblasLeft   op( A )*X = alpha*B.
  side = CBLAS_SIDE::CblasRight   op( A )*X = alpha*B.

  \param[in] uplo
  uplo is enum CBLAS_UPLO
  uplo specifies specifies whether the matrix A is an upper or
  lower triangular matrix as follows:
  uplo = CBLAS_UPLO::CblasUpper   A is an upper triangular matrix.
  uplo = CBLAS_UPLO::CblasLower   A is a lower triangular matrix.
  
  \param[in] trans
  trans is CBLAS_TRANSPOSE
  On entry, trans specifies the form of op( A ) to be used in
  the matrix multiplication as follows:
  trans = CBLAS_TRANSPOSE::CblasNoTrans,  op( A ) = A.
  trans = CBLAS_TRANSPOSE::CblasTrans,  op( A ) = A**T.
  trans = CBLAS_TRANSPOSE::CblasConjTrans,  op( A ) = A**H.
  
  \param[in] diag
  diag is enum CBLAS_DIAG
  diag specifies specifies whether or not A is unit triangular
  as follows:
  diag = CBLAS_DIAG::CblasUnit   A is assumed to be unit triangular.
  diag = CBLAS_DIAG::CblasNonUnit   A is not assumed to be unit
                                 triangular.

  \param[in] m
  m is INTEGER
  On entry,  m  specifies  the number  of rows  of the  matrix
  B.  m  must  be at least  zero.

  \param[in] n
  n is INTEGER
  On entry,  n  specifies the number  of columns of the matrix
  B. n must be at least zero.

  \param[in] alpha
  alpha is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, alpha specifies the scalar alpha.

  \param[in] A
  A is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If side = CblasLeft:
  the m-by-m matrix A, stored in an lda-by-m array [RowMajor: m-by-lda].
  If side = CblasRight:
  the n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  If side = CblasLeft: lda >= max(1, m) .
  If side = CblasRight:lda >= max(1, k) .

  \param[in,out] B
  B is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  m-by-n , stored in an ldb-by-n array [RowMajor: m-by-ldb]. 
  on exit  is overwritten by the solution matrix  X.

  \param[in] ldb
  ldb is INTEGER
  On entry, ldb specifies the Leading dimension of B
  ldb >= max(1, m) [RowMajor: ldb >= max(1, n)].
  */
template< typename T >
void trsm(
    CBLAS_ORDER layout,
    CBLAS_SIDE side,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    int64_t m,
    int64_t n,
    T alpha,
    T const *A, int64_t lda,
    T       *B, int64_t ldb )
{
    cblas_trsm( layout, side, uplo, trans, diag, m, n,  alpha, A, lda, B, ldb);
}
/*! \b Purpose:	

  TRMM  performs solves one of the matrix equations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

  B := alpha*op( A )*B,   or   B := alpha*B*op( A ),

  where alpha is a scalar, B is an m by n matrices, A is a unit, or
  non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
  op( A ) = A   or   op( A ) = A**T.

  \param[in] layout
  layout is enum CBLAS_ORDER
  layout specifies Matrix storage as follows:
  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.

  \param[in] side
  side is enum CBLAS_SIDE
  side specifies whether op( A ) multiplies B from left or right of X
  as follows:
  side = CBLAS_SIDE::CblasLeft   B := alpha*op( A )*B.
  side = CBLAS_SIDE::CblasRight  B := alpha*B*op( A ).

  \param[in] uplo
  uplo is enum CBLAS_UPLO
  uplo specifies whether the matrix A is an upper or lower triangular
  matrix as follows:
  uplo = CBLAS_UPLO::CblasUpper   A is an upper triangular matrix.
  uplo = CBLAS_UPLO::CblasLower   A is a lower triangular matrix.

  \param[in] trans
  trans is CBLAS_TRANSPOSE
  On entry, trans specifies the form of op( A ) to be used in
  the matrix multiplication as follows:
  trans = CBLAS_TRANSPOSE::CblasNoTrans,  op( A ) = A.
  trans = CBLAS_TRANSPOSE::CblasTrans,  op( A ) = A**T.
  trans = CBLAS_TRANSPOSE::CblasConjTrans,  op( A ) = A**T.

  \param[in] diag
  diag is enum CBLAS_DIAG
  diag specifies specifies whether or not A is unit triangular
  as follows:
  diag = CBLAS_DIAG::CblasUnit   A is assumed to be unit triangular.
  diag = CBLAS_DIAG::CblasNonUnit   A is not assumed to be unit
                                 triangular.

  \param[in] m
  m is INTEGER
  On entry,  m  specifies  the number  of rows  of the  matrix
  B.  m  must  be at least  zero.

  \param[in] n
  n is INTEGER
  On entry,  n  specifies the number  of columns of the matrix
  B. n must be at least zero.

  \param[in] alpha
  alpha is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, alpha specifies the scalar alpha.When  alpha is
  zero then  A is not referenced and  B need not be set before
  entry.

  \param[in] A
  A is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If side = CblasLeft:
  the m-by-m matrix A, stored in an lda-by-m array [RowMajor: m-by-lda].
  If side = CblasRight:
  the n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  If side = CblasLeft: lda >= max(1, m) .
  If side = CblasRight:lda >= max(1, n) .

  \param[in,out] B
  B is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  m-by-n , stored in an ldb-by-n array [RowMajor: m-by-ldb].

  \param[in] ldb
  ldb is INTEGER
  On entry, ldb specifies the Leading dimension of B
  ldb >= max(1, m) [RowMajor: ldb >= max(1, n)].
  */
template< typename T >
void trmm(
    CBLAS_ORDER layout,
    CBLAS_SIDE side,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    int64_t m,
    int64_t n,
    T alpha,
    T const *A, int64_t lda,
    T       *B, int64_t ldb )
{
    cblas_trmm( layout, side, uplo, trans, diag, m, n,  alpha, A, lda, B, ldb);
}

/*! \b Purpose:	

  HEMM  performs solves one of the matrix-matrix operations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

  C := alpha*A*B + beta*C

  or

  C := alpha*B*A + beta*C,

  where alpha is a scalar,  A is an hermitian matrix
  C and B are m by n matrices

  \param[in] layout
  layout is enum CBLAS_ORDER
  layout specifies Matrix storage as follows:
  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.

  \param[in] side
  side is enum CBLAS_SIDE
  side specifies specifies whether the  hermitian matrix  A
  appears on the  left or right  in the  operation as follows:
  side = CBLAS_SIDE::CblasLeft   C := alpha*A*B + beta*C,
  side = CBLAS_SIDE::CblasRight   C := alpha*B*A + beta*C

  \param[in] uplo
  uplo is enum CBLAS_UPLO
  uplo specifies specifies whether  the  upper  or  lower
  triangular  part  of  the  hermitian  matrix   A  is  to  be
  referenced as follows:
  uplo = CBLAS_UPLO::CblasUpper   Only the upper triangular part of the
                                  hermitian matrix is to be referenced.
  uplo = CBLAS_UPLO::CblasLower   Only the lower triangular part of the
                                  hermitian matrix is to be referenced.

  \param[in] m
  m is INTEGER
  On entry,  m  specifies  the number  of rows  of the  matrix
  C.  m  must  be at least  zero.

  \param[in] n
  n is INTEGER
  On entry,  n  specifies the number  of columns of the matrix
  C. n must be at least zero.

  \param[in] alpha
  alpha is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, alpha specifies the scalar alpha.

  \param[in] A
  A is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If side = CblasLeft:
  the m-by-m matrix A, stored in an lda-by-m array [RowMajor: m-by-lda].
  If side = CblasRight:
  the n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  If side = CblasLeft: lda >= max(1, m) .
  If side = CblasRight:lda >= max(1, k) .

  \param[in] B
  B is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  m-by-n , stored in an ldb-by-n array [RowMajor: m-by-ldb].

  \param[in] ldb
  ldb is INTEGER
  On entry, ldb specifies the Leading dimension of B
  ldb >= max(1, m) [RowMajor: ldb >= max(1, n)].

  \param[in] beta
  beta is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, beta specifies the scalar beta.
  If beta is zero, C need not be set on input

  \param[in,out] C
  C is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  m-by-n , stored in an ldc-by-n array [RowMajor: m-by-ldc].

  \param[in] ldc
  ldc is INTEGER
  On entry, ldc specifies the Leading dimension of C
  ldc >= max(1, m) [RowMajor: ldc >= max(1, n)].
  */
template< typename T >
void hemm(
    CBLAS_ORDER layout,
    CBLAS_SIDE side,
    CBLAS_UPLO uplo,
    int64_t m, int64_t n,
    T alpha,
    T const *A, int64_t lda,
    T const *B, int64_t ldb,
    T beta,
    T       *C, int64_t ldc )
{
    cblas_hemm( layout, side, uplo, m, n,  alpha, A, lda, B, ldb, beta, C, ldc);
}

/*! \b Purpose:	
  
  SYMM  performs solves one of the matrix-matrix operations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

  C := alpha*A*B + beta*C

  or

  C := alpha*B*A + beta*C,

  where alpha is a scalar,  A is an symmetric matrix
  C and B are m by n matrices

  \param[in] layout
  layout is enum CBLAS_ORDER
  layout specifies Matrix storage as follows:
  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.

  \param[in] side
  side is enum CBLAS_SIDE
  side specifies specifies whether the  symmetric matrix  A
  appears on the  left or right  in the  operation as follows:
  side = CBLAS_SIDE::CblasLeft   C := alpha*A*B + beta*C,
  side = CBLAS_SIDE::CblasRight   C := alpha*B*A + beta*C

  \param[in] uplo
  uplo is enum CBLAS_UPLO
  uplo specifies specifies whether  the  upper  or  lower
  triangular  part  of  the  symmetric  matrix   A  is  to  be
  referenced as follows:
  uplo = CBLAS_UPLO::CblasUpper   Only the upper triangular part of the
                                  symmetric matrix is to be referenced.
  uplo = CBLAS_UPLO::CblasLower   Only the lower triangular part of the
                                  symmetric matrix is to be referenced.

  \param[in] m
  m is INTEGER
  On entry,  m  specifies  the number  of rows  of the  matrix
  C.  m  must  be at least  zero.

  \param[in] n
  n is INTEGER
  On entry,  n  specifies the number  of columns of the matrix
  C. n must be at least zero.

  \param[in] alpha
  alpha is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, alpha specifies the scalar alpha.

  \param[in] A
  A is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If side = CblasLeft:
  the m-by-m matrix A, stored in an lda-by-m array [RowMajor: m-by-lda].
  If side = CblasRight:
  the n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  If side = CblasLeft: lda >= max(1, m) .
  If side = CblasRight:lda >= max(1, k) .

  \param[in] B
  B is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  m-by-n , stored in an ldb-by-n array [RowMajor: m-by-ldb].

  \param[in] ldb
  ldb is INTEGER
  On entry, ldb specifies the Leading dimension of B
  ldb >= max(1, m) [RowMajor: ldb >= max(1, n)].

  \param[in] beta
  beta is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, beta specifies the scalar beta.
  If beta is zero, C need not be set on input

  \param[in, out] C
  C is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  m-by-n , stored in an ldc-by-n array [RowMajor: m-by-ldc].

  \param[in] ldc
  ldc is INTEGER
  On entry, ldc specifies the Leading dimension of C
  ldc >= max(1, m) [RowMajor: ldc >= max(1, n)].
  */
template< typename T >
void symm(
    CBLAS_ORDER layout,
    CBLAS_SIDE side,
    CBLAS_UPLO uplo,
    int64_t m, int64_t n,
    T alpha,
    T const *A, int64_t lda,
    T const *B, int64_t ldb,
    T beta,
    T       *C, int64_t ldc )
{
    cblas_symm( layout, side, uplo, m, n,  alpha, A, lda, B, ldb, beta, C, ldc);
}

/*! \b Purpose:	

  SYRK  performs one of the symmetric rank k operations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

  C := alpha*A*A**T + beta*C,

  or

  C := alpha*A**T*A + beta*C,

  where  alpha and beta  are scalars,  C is an  n by n symmetric matrix
  and  A  is an  n by k  matrix in the first case and a  k by n  matrix
  in the second case.

  \param[in] layout
  layout is enum CBLAS_LAYOUT
  layout specifies Matrix storage as follows:
  layout = CBLAS_LAYOUT::CblasRowMajor or Layout::CblasColMajor.

  \param[in] uplo
  uplo is enum CBLAS_UPLO
  uplo specifies specifies whether  the  upper  or  lower
  triangular  part  of  the  array C  is  to  be  referenced
  as follows:
  uplo = CBLAS_UPLO::CblasUpper   Only the upper triangular part of C
                                  is to be referenced.
  uplo = CBLAS_UPLO::CblasLower   Only the lower triangular part of C
                                  is to be referenced.

  \param[in] trans
  trans is CBLAS_TRANSPOSE
  On entry, trans specifies the operation to be used as follows:
  trans = CBLAS_TRANSPOSE::CblasNoTrans,C := alpha*A*A**T + beta*C.
  trans = CBLAS_TRANSPOSE::CblasTrans,C := alpha*A**T*A + beta*C.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the order of the matrix C.  n must be
  at least zero.

  \param[in] k
  k is INTEGER
  If trans = CblasNoTrans: k is number of columns of the matrix A.
  Otherwise:               k is number of rows    of the matrix A.
  k must be at least  zero.

  \param[in] alpha
  alpha is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, alpha specifies the scalar alpha.

  \param[in] A
  A is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If transA = CblasNoTrans:
  n-by-k , stored in an lda-by-k array [RowMajor: n-by-lda].
  Otherwise:
  k-by-n , stored in an lda-by-n array [RowMajor: k-by-lda].

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  If transA = CblasNoTrans: lda >= max(1, n) [RowMajor: lda >= max(1, k)].
  Otherwise:                lda >= max(1, k) [RowMajor: lda >= max(1, n)].

  \param[in] beta
  beta is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, beta specifies the scalar alpha.When  beta  is
  supplied as zero then C need not be set on input.

  \param[in,out] C
  C is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array, dimension :
  The n-by-n symmetric matrix C,
  stored in an ldc-by-n array [RowMajor: n-by-ldc].
  On exit, the array  C  is overwritten by the  lower/upper
  triangular part of the updated matrix.

  \param[in] ldc
  ldc is INTEGER
  On entry, ldc specifies the first dimension of C
  ldc >= max(1, n)
  */
template< typename T >
void syrk(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    int64_t n, int64_t k,
    T alpha,
    T const *A, int64_t lda,
    T beta,
    T       *C, int64_t ldc )
{
    cblas_syrk( layout, uplo, trans,  n, k,  alpha, A, lda, beta, C, ldc);
}

/*! \b Purpose:	

  SYR2K  performs one of the symmetric rank 2k operations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

  C := alpha*A*B**T + alpha*B*A**T + beta*C,

  or

  C := alpha*A**T*B + alpha*B**T*A + beta*C,

  where  alpha and beta  are scalars,  C is an  n by n symmetric matrix
  and  A  and B are n by k  matrices in the first case and k by n  matrices
  in the second case.

  \param[in] layout
  layout is enum CBLAS_LAYOUT
  layout specifies Matrix storage as follows:
  layout = CBLAS_LAYOUT::CblasRowMajor or Layout::CblasColMajor.

  \param[in] uplo
  uplo is enum CBLAS_UPLO
  uplo specifies specifies whether  the  upper  or  lower
  triangular  part  of  the  array C  is  to  be  referenced
  as follows:
  uplo = CBLAS_UPLO::CblasUpper   Only the upper triangular part of C
                                  is to be referenced.
  uplo = CBLAS_UPLO::CblasLower   Only the lower triangular part of C
                                  is to be referenced.

  \param[in] trans
  trans is CBLAS_TRANSPOSE
  On entry, trans specifies the operation to be used as follows:
  trans = CBLAS_TRANSPOSE::CblasNoTrans,C := alpha*A*B**T + alpha*B*A**T + beta*C.
  trans = CBLAS_TRANSPOSE::CblasTrans,  C := alpha*A**T*B + alpha*B**T*A + beta*C.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the order of the matrix C.  n must be
  at least zero.

  \param[in] k
  k is INTEGER
  If trans = CblasNoTrans: k is number of columns of the matrices A & B.
  Otherwise:               k is number of rows    of the matrices A & B.
  k must be at least  zero.

  \param[in] alpha
  alpha is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, alpha specifies the scalar alpha.

  \param[in] A
  A is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If trans = CblasNoTrans:
  n-by-k , stored in an lda-by-k array [RowMajor: n-by-lda].
  Otherwise:
  k-by-n , stored in an lda-by-n array [RowMajor: k-by-lda].

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  If trans = CblasNoTrans: lda >= max(1, n) [RowMajor: lda >= max(1, k)].
  Otherwise:                lda >= max(1, k) [RowMajor: lda >= max(1, n)].

  \param[in] B
  B is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If trans = CblasNoTrans:
  n-by-k , stored in an ldb-by-k array [RowMajor: n-by-ldb].
  Otherwise:
  k-by-n , stored in an ldb-by-n array [RowMajor: k-by-ldb]

  \param[in] ldb
  ldb is INTEGER
  On entry, ldb specifies the Leading dimension of B
  If trans = CblasNoTrans: ldb >= max(1, n) [RowMajor: ldb >= max(1, k)].
  Otherwise:               ldb >= max(1, k) [RowMajor: ldb >= max(1, n)].

  \param[in] beta
  beta is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, beta specifies the scalar alpha.When  beta  is
  supplied as zero then C need not be set on input.

  \param[in,out] C
  C is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array, dimension :
  The n-by-n symmetric matrix C,
  stored in an ldc-by-n array [RowMajor: n-by-ldc].
  On exit, the array  C  is overwritten by the  lower/upper
  triangular part of the updated matrix.

  \param[in] ldc
  ldc is INTEGER
  On entry, ldc specifies the first dimension of C
  ldc >= max(1, n)
  */
template< typename T >
void syr2k(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    int64_t n, int64_t k,
    T alpha,
    T const *A, int64_t lda,
    T const *B, int64_t ldb,
    T beta,
    T       *C, int64_t ldc )
{
    cblas_syr2k( layout, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc );	
}

/*! \b Purpose:	

  HERK  performs one of the hermitian rank k operations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

    C := alpha*A*B**H + conjg( alpha )*B*A**H + beta*C,
  
  or
  
    C := alpha*A**H*B + conjg( alpha )*B**H*A + beta*C,

  where  alpha and beta  are real scalars,  C is an  n by n  hermitian 
  matrix and  A  is an n by k  matrix in the first case and 
  k by n  matrix in the second case.

  \param[in] layout
  layout is enum CBLAS_LAYOUT
  layout specifies Matrix storage as follows:
  layout = CBLAS_LAYOUT::CblasRowMajor or Layout::CblasColMajor.

  \param[in] uplo
  uplo is enum CBLAS_UPLO
  uplo specifies specifies whether  the  upper  or  lower
  triangular  part  of  the  array C  is  to  be  referenced 
  as follows:
  uplo = CBLAS_UPLO::CblasUpper   Only the upper triangular part of C
                                  is to be referenced.
  uplo = CBLAS_UPLO::CblasLower   Only the lower triangular part of C
                                  is to be referenced.

  \param[in] trans
  trans is CBLAS_TRANSPOSE
  On entry, trans specifies the operation to be used as follows:
  trans = CBLAS_TRANSPOSE::CblasNoTrans,  C := alpha*A*A**H + beta*C.
  trans = CBLAS_TRANSPOSE::CblasConjTrans,C := alpha*A**H*A + beta*C.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the order of the matrix C.  n must be
  at least zero.

  \param[in] k
  k is INTEGER
  If trans = CblasNoTrans: k is number of columns of the matrix   A. 
  Otherwise:               k is number of rows    of the matrix   A.
  k must be at least  zero.

  \param[in] alpha
  alpha is REAL/DOUBLE PRECISION
  On entry, alpha specifies the scalar alpha.

  \param[in] A
  A is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If trans = CblasNoTrans:
  n-by-k , stored in an lda-by-k array [RowMajor: n-by-lda].
  Otherwise:
  k-by-n , stored in an lda-by-n array [RowMajor: k-by-lda].

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  If trans = CblasNoTrans: lda >= max(1, n) [RowMajor: lda >= max(1, k)].
  Otherwise:                lda >= max(1, k) [RowMajor: lda >= max(1, n)].
  
  \param[in] beta
  beta is REAL/DOUBLE PRECISION
  On entry, beta specifies the scalar alpha.When  beta  is
  supplied as zero then C need not be set on input.

  \param[in,out] C
  C is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array, dimension :
  The n-by-n Hermitian matrix C,
  stored in an ldc-by-n array [RowMajor: n-by-ldc].
  On exit, the array  C  is overwritten by the  lower/upper 
  triangular part of the updated matrix.

  \param[in] ldc
  ldc is INTEGER
  On entry, ldc specifies the first dimension of C
  ldc >= max(1, n) 
  */
template< typename T >
void herk(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    int64_t n, int64_t k,
    real_type<T> alpha,
    T const *A, int64_t lda,
    real_type<T> beta,
    T       *C, int64_t ldc )
{
    cblas_herk( layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc );	
}

/*! \b Purpose:	

  HER2K  performs one of the hermitian rank 2k operations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

    C := alpha*A*B**H + conjg( alpha )*B*A**H + beta*C,
  
  or
  
    C := alpha*A**H*B + conjg( alpha )*B**H*A + beta*C,

  where  alpha and beta  are scalars with  beta  real,  C is an  n by n 
  hermitian matrix and  A  and B are n by k  matrices in the first case 
  and k by n  matrices in the second case.

  \param[in] layout
  layout is enum CBLAS_LAYOUT
  layout specifies Matrix storage as follows:
  layout = CBLAS_LAYOUT::CblasRowMajor or Layout::CblasColMajor.

  \param[in] uplo
  uplo is enum CBLAS_UPLO
  uplo specifies specifies whether  the  upper  or  lower
  triangular  part  of  the  array C  is  to  be  referenced 
  as follows:
  uplo = CBLAS_UPLO::CblasUpper   Only the upper triangular part of C
                                  is to be referenced.
  uplo = CBLAS_UPLO::CblasLower   Only the lower triangular part of C
                                  is to be referenced.

  \param[in] trans
  trans is CBLAS_TRANSPOSE
  On entry, trans specifies the operation to be used as follows:
  trans = CBLAS_TRANSPOSE::CblasNoTrans,  C := alpha*A*B**H + conjg( alpha )*B*A**H + beta*C.
  trans = CBLAS_TRANSPOSE::CblasConjTrans,C := alpha*A**H*B + conjg( alpha )*B**H*A + beta*C.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the order of the matrix C.  n must be
  at least zero.

  \param[in] k
  k is INTEGER
  If trans = CblasNoTrans: k is number of columns of the matrices A & B. 
  Otherwise:               k is number of rows    of the matrices A & B.
  k must be at least  zero.

  \param[in] alpha
  alpha is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, alpha specifies the scalar alpha.

  \param[in] A
  A is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If trans = CblasNoTrans:
  n-by-k , stored in an lda-by-k array [RowMajor: n-by-lda].
  Otherwise:
  k-by-n , stored in an lda-by-n array [RowMajor: k-by-lda].

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  If trans = CblasNoTrans: lda >= max(1, n) [RowMajor: lda >= max(1, k)].
  Otherwise:                lda >= max(1, k) [RowMajor: lda >= max(1, n)].

  \param[in] B
  B is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If trans = CblasNoTrans:
  n-by-k , stored in an ldb-by-k array [RowMajor: n-by-ldb].
  Otherwise:
  k-by-n , stored in an ldb-by-n array [RowMajor: k-by-ldb]

  \param[in] ldb
  ldb is INTEGER
  On entry, ldb specifies the Leading dimension of B
  If trans = CblasNoTrans: ldb >= max(1, n) [RowMajor: ldb >= max(1, k)].
  Otherwise:               ldb >= max(1, k) [RowMajor: ldb >= max(1, n)].
  
  \param[in] beta
  beta is REAL/DOUBLE PRECISION
  On entry, beta specifies the scalar alpha.When  beta  is
  supplied as zero then C need not be set on input.

  \param[in,out] C
  C is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array, dimension :
  The n-by-n Hermitian matrix C,
  stored in an ldc-by-n array [RowMajor: n-by-ldc].
  On exit, the array  C  is overwritten by the  lower/upper 
  triangular part of the updated matrix.

  \param[in] ldc
  ldc is INTEGER
  On entry, ldc specifies the first dimension of C
  ldc >= max(1, n) 
  */
template< typename T >
void her2k(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    int64_t n, int64_t k,
    T alpha,
    T const *A, int64_t lda,
    T const *B, int64_t ldb,
    real_type<T> beta,
    T       *C, int64_t ldc )
{
    cblas_her2k( layout, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc );	
}

}  // namespace blis
#endif        //  #ifndef BLIS_HH
