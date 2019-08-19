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
 *  blis.hh defines all the CPP templated public interfaces
 *  */
#ifndef BLIS_HH
#define BLIS_HH

#include "cblas.hh"
#include "blis_util.hh"
#include <limits>

namespace blis {
/*! @brief \b GEMM

  \verbatim

  GEMM  performs general matrix-matrix multiply for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

  C := alpha*op( A )*op( B ) + beta*C,

  where  op( X ) is one of

  op( X ) = X   or   op( X ) = X**T   or   op( X ) = X**H,

  alpha and beta are scalars, and A, B and C are matrices, with op( A )
  an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
  \endverbatim

  \param[in] layout
  \verbatim
  layout is enum CBLAS_ORDER

  layout specifies Matrix storage as follows:

  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.
  \endverbatim

  \param[in] transA
  \verbatim
  
  transA is CBLAS_TRANSPOSE
  On entry, transA specifies the form of op( A ) to be used in
  the matrix multiplication as follows:

  transA = CBLAS_TRANSPOSE::CblasNoTrans,  op( A ) = A.

  transA = CBLAS_TRANSPOSE::CblasTrans,  op( A ) = A**T.

  transA = CBLAS_TRANSPOSE::CblasConjTrans,  op( A ) = A**H.
  \endverbatim

  \param[in] transB
  \verbatim
  transB is CBLAS_TRANSPOSE
  On entry, transB specifies the form of op( B ) to be used in
  the matrix multiplication as follows:

  transB = CBLAS_TRANSPOSE::CblasNoTrans,  op( B ) = B.

  transB = CBLAS_TRANSPOSE::CblasTrans,  op( B ) = B**T.

  transB = CBLAS_TRANSPOSE::CblasConjTrans,  op( B ) = B**H.
  \endverbatim

  \param[in] m
  \verbatim
  m is INTEGER
  On entry,  m  specifies  the number  of rows  of the  matrix
  op( A )  and of the  matrix  C.  m  must  be at least  zero.
  \endverbatim

  \param[in] n
  \verbatim
  n is INTEGER
  On entry,  n  specifies the number  of columns of the matrix
  op( B ) and the number of columns of the matrix C. n must be
  at least zero.
  \endverbatim

  \param[in] k
  \verbatim
  k is INTEGER
  On entry,  k  specifies  the number of columns of the matrix
  op( A ) and the number of rows of the matrix op( B ). k must
  be at least  zero.
  \endverbatim

  \param[in] alpha
  \verbatim
  alpha is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, alpha specifies the scalar alpha.
  \endverbatim

  \param[in] A
  \verbatim
  A is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If transA = CblasNoTrans:
  m-by-k , stored in an lda-by-k array [RowMajor: m-by-lda].
  Otherwise:
  k-by-m , stored in an lda-by-m array [RowMajor: k-by-lda].
  \endverbatim

  \param[in] lda
  \verbatim
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  If transA = CblasNoTrans: lda >= max(1, m) [RowMajor: lda >= max(1, k)].
  Otherwise:                lda >= max(1, k) [RowMajor: lda >= max(1, m)].
  \endverbatim

  \param[in] B
  \verbatim
  B is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If transA = CblasNoTrans:
  k-by-n , stored in an ldb-by-n array [RowMajor: k-by-ldb].
  Otherwise:
  n-by-k , stored in an ldb-by-k array [RowMajor: n-by-ldb].
  \endverbatim

  \param[in] ldb
  \verbatim
  ldb is INTEGER
  On entry, ldb specifies the Leading dimension of B
  If transA = CblasNoTrans: ldb >= max(1, k) [RowMajor: ldb >= max(1, n)].
  Otherwise:                ldb >= max(1, n) [RowMajor: ldb >= max(1, k)].
  \endverbatim

  \param[in] beta
  \verbatim
  beta is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, beta specifies the scalar alpha.When  beta  is
  supplied as zero then C need not be set on input.
  \endverbatim

  \param[in,out] C
  \verbatim
  C is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array, dimension :
  m-by-n stored in an ldc-by-n array [RowMajor: m-by-ldc].
  Before entry, the leading  m by n  part of the array  C must
  contain the matrix  C,  except when  beta  is zero, in which
  case C need not be set on entry.
  On exit, the array  C  is overwritten by the  m by n  matrix
  ( alpha*op( A )*op( B ) + beta*C ).
  \endverbatim

  \param[in] ldc
  \verbatim
  ldc is INTEGER
  On entry, ldc specifies the first dimension of C
  ldc >= max(1, m) [RowMajor: ldc >= max(1, n)].
  \endverbatim

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

/*! @brief \b TRSM

  \verbatim

  TRSM  performs solves one of the matrix equations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

      op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,

  where alpha is a scalar, X and B are m by n matrices, A is a unit, or
  non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
  where  op( X ) is one of
      
	  op( A ) = A   or   op( A ) = A**T   or   op( A ) = A**H.

  The matrix X is overwritten on B.
  \endverbatim

  \param[in] layout
  \verbatim
  layout is enum CBLAS_ORDER

  layout specifies Matrix storage as follows:

  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.
  \endverbatim

  \param[in] side
  \verbatim
  side is enum CBLAS_SIDE

  side specifies specifies whether op( A ) appears on the left
  or right of X as follows:
  
  side = CBLAS_SIDE::CblasLeft   op( A )*X = alpha*B.
	  
  side = CBLAS_SIDE::CblasRight   op( A )*X = alpha*B.
  \endverbatim

  \param[in] uplo
  \verbatim
  uplo is enum CBLAS_UPLO

  uplo specifies specifies whether the matrix A is an upper or
  lower triangular matrix as follows:
  
  uplo = CBLAS_UPLO::CblasUpper   A is an upper triangular matrix.
	  
  uplo = CBLAS_UPLO::CblasLower   A is a lower triangular matrix.
  \endverbatim
  
  \param[in] trans
  \verbatim

  trans is CBLAS_TRANSPOSE
  On entry, trans specifies the form of op( A ) to be used in
  the matrix multiplication as follows:

  trans = CBLAS_TRANSPOSE::CblasNoTrans,  op( A ) = A.

  trans = CBLAS_TRANSPOSE::CblasTrans,  op( A ) = A**T.

  trans = CBLAS_TRANSPOSE::CblasConjTrans,  op( A ) = A**H.
  \endverbatim
  
  \param[in] diag
  \verbatim
  diag is enum CBLAS_DIAG

  diag specifies specifies whether or not A is unit triangular
  as follows:
  
  diag = CBLAS_DIAG::CblasUnit   A is assumed to be unit triangular.
	  
  diag = CBLAS_DIAG::CblasNonUnit   A is not assumed to be unit
                                 triangular.
  \endverbatim

  \param[in] m
  \verbatim
  m is INTEGER
  On entry,  m  specifies  the number  of rows  of the  matrix
  B.  m  must  be at least  zero.
  \endverbatim

  \param[in] n
  \verbatim
  n is INTEGER
  On entry,  n  specifies the number  of columns of the matrix
  B. n must be at least zero.
  \endverbatim

  \param[in] alpha
  \verbatim
  alpha is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, alpha specifies the scalar alpha.
  \endverbatim

  \param[in] A
  \verbatim
  A is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If side = CblasLeft:
  the m-by-m matrix A, stored in an lda-by-m array [RowMajor: m-by-lda].
  If side = CblasRight:
  the n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].
  \endverbatim

  \param[in] lda
  \verbatim
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  If side = CblasLeft: lda >= max(1, m) .
  If side = CblasRight:lda >= max(1, k) .
  \endverbatim

  \param[in] B
  \verbatim
  B is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  m-by-n , stored in an ldb-by-n array [RowMajor: m-by-ldb]. 
  on exit  is overwritten by the solution matrix  X.
  \endverbatim

  \param[in] ldb
  \verbatim
  ldb is INTEGER
  On entry, ldb specifies the Leading dimension of B
  ldb >= max(1, m) [RowMajor: ldb >= max(1, n)].
  \endverbatim

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

/*! @brief \b HEMM

  \verbatim

  HEMM  performs solves one of the matrix-matrix operations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

      C := alpha*A*B + beta*C
  or
      C := alpha*B*A + beta*C,

  where alpha is a scalar,  A is an hermitian matrix
  C and B are m by n matrices
  \endverbatim

  \param[in] layout
  \verbatim
  layout is enum CBLAS_ORDER

  layout specifies Matrix storage as follows:

  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.
  \endverbatim

  \param[in] side
  \verbatim
  side is enum CBLAS_SIDE

  side specifies specifies whether the  hermitian matrix  A
  appears on the  left or right  in the  operation as follows:

  side = CBLAS_SIDE::CblasLeft   C := alpha*A*B + beta*C,

  side = CBLAS_SIDE::CblasRight   C := alpha*B*A + beta*C
  \endverbatim

  \param[in] uplo
  \verbatim
  uplo is enum CBLAS_UPLO

  uplo specifies specifies whether  the  upper  or  lower
  triangular  part  of  the  hermitian  matrix   A  is  to  be
  referenced as follows:

  uplo = CBLAS_UPLO::CblasUpper   Only the upper triangular part of the
                                  hermitian matrix is to be referenced.

  uplo = CBLAS_UPLO::CblasLower   Only the lower triangular part of the
                                  hermitian matrix is to be referenced.
  \endverbatim

  \param[in] m
  \verbatim
  m is INTEGER
  On entry,  m  specifies  the number  of rows  of the  matrix
  C.  m  must  be at least  zero.
  \endverbatim

  \param[in] n
  \verbatim
  n is INTEGER
  On entry,  n  specifies the number  of columns of the matrix
  C. n must be at least zero.
  \endverbatim

  \param[in] alpha
  \verbatim
  alpha is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, alpha specifies the scalar alpha.
  \endverbatim

  \param[in] A
  \verbatim
  A is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If side = CblasLeft:
  the m-by-m matrix A, stored in an lda-by-m array [RowMajor: m-by-lda].
  If side = CblasRight:
  the n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].
  \endverbatim

  \param[in] lda
  \verbatim
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  If side = CblasLeft: lda >= max(1, m) .
  If side = CblasRight:lda >= max(1, k) .
  \endverbatim

  \param[in] B
  \verbatim
  B is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  m-by-n , stored in an ldb-by-n array [RowMajor: m-by-ldb].
  \endverbatim

  \param[in] ldb
  \verbatim
  ldb is INTEGER
  On entry, ldb specifies the Leading dimension of B
  ldb >= max(1, m) [RowMajor: ldb >= max(1, n)].
  \endverbatim

  \param[in] beta
  \verbatim
  beta is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, beta specifies the scalar beta.
  If beta is zero, C need not be set on input
  \endverbatim

  \param[in] C
  \verbatim
  C is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  m-by-n , stored in an ldc-by-n array [RowMajor: m-by-ldc].
  \endverbatim

  \param[in] ldc
  \verbatim
  ldc is INTEGER
  On entry, ldc specifies the Leading dimension of C
  ldc >= max(1, m) [RowMajor: ldc >= max(1, n)].
  \endverbatim

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

/*! @brief \b SYMM

  \verbatim

  SYMM  performs solves one of the matrix-matrix operations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

      C := alpha*A*B + beta*C
  or
      C := alpha*B*A + beta*C,

  where alpha is a scalar,  A is an symmetric matrix
  C and B are m by n matrices
  \endverbatim

  \param[in] layout
  \verbatim
  layout is enum CBLAS_ORDER

  layout specifies Matrix storage as follows:

  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.
  \endverbatim

  \param[in] side
  \verbatim
  side is enum CBLAS_SIDE

  side specifies specifies whether the  symmetric matrix  A
  appears on the  left or right  in the  operation as follows:

  side = CBLAS_SIDE::CblasLeft   C := alpha*A*B + beta*C,

  side = CBLAS_SIDE::CblasRight   C := alpha*B*A + beta*C
  \endverbatim

  \param[in] uplo
  \verbatim
  uplo is enum CBLAS_UPLO

  uplo specifies specifies whether  the  upper  or  lower
  triangular  part  of  the  symmetric  matrix   A  is  to  be
  referenced as follows:

  uplo = CBLAS_UPLO::CblasUpper   Only the upper triangular part of the
                                  symmetric matrix is to be referenced.

  uplo = CBLAS_UPLO::CblasLower   Only the lower triangular part of the
                                  symmetric matrix is to be referenced.
  \endverbatim

  \param[in] m
  \verbatim
  m is INTEGER
  On entry,  m  specifies  the number  of rows  of the  matrix
  C.  m  must  be at least  zero.
  \endverbatim

  \param[in] n
  \verbatim
  n is INTEGER
  On entry,  n  specifies the number  of columns of the matrix
  C. n must be at least zero.
  \endverbatim

  \param[in] alpha
  \verbatim
  alpha is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, alpha specifies the scalar alpha.
  \endverbatim

  \param[in] A
  \verbatim
  A is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If side = CblasLeft:
  the m-by-m matrix A, stored in an lda-by-m array [RowMajor: m-by-lda].
  If side = CblasRight:
  the n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].
  \endverbatim

  \param[in] lda
  \verbatim
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  If side = CblasLeft: lda >= max(1, m) .
  If side = CblasRight:lda >= max(1, k) .
  \endverbatim

  \param[in] B
  \verbatim
  B is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  m-by-n , stored in an ldb-by-n array [RowMajor: m-by-ldb].
  \endverbatim

  \param[in] ldb
  \verbatim
  ldb is INTEGER
  On entry, ldb specifies the Leading dimension of B
  ldb >= max(1, m) [RowMajor: ldb >= max(1, n)].
  \endverbatim

  \param[in] beta
  \verbatim
  beta is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, beta specifies the scalar beta.
  If beta is zero, C need not be set on input
  \endverbatim

  \param[in] C
  \verbatim
  C is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  m-by-n , stored in an ldc-by-n array [RowMajor: m-by-ldc].
  \endverbatim

  \param[in] ldc
  \verbatim
  ldc is INTEGER
  On entry, ldc specifies the Leading dimension of C
  ldc >= max(1, m) [RowMajor: ldc >= max(1, n)].
  \endverbatim

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

/*! @brief \b SYRK

  \verbatim

  SYRK  performs one of the symmetric rank k operations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

  C := alpha*A*A**T + beta*C,

  or

  C := alpha*A**T*A + beta*C,

  where  alpha and beta  are scalars,  C is an  n by n symmetric matrix
  and  A  is an  n by k  matrix in the first case and a  k by n  matrix
  in the second case.
  \endverbatim

  \param[in] layout
  \verbatim
  layout is enum CBLAS_LAYOUT

  layout specifies Matrix storage as follows:

  layout = CBLAS_LAYOUT::CblasRowMajor or Layout::CblasColMajor.
  \endverbatim

  \param[in] uplo
  \verbatim
  uplo is enum CBLAS_UPLO

  uplo specifies specifies whether  the  upper  or  lower
  triangular  part  of  the  array C  is  to  be  referenced
  as follows:

  uplo = CBLAS_UPLO::CblasUpper   Only the upper triangular part of C
                                  is to be referenced.

  uplo = CBLAS_UPLO::CblasLower   Only the lower triangular part of C
                                  is to be referenced.
  \endverbatim

  \param[in] trans
  \verbatim

  trans is CBLAS_TRANSPOSE
  On entry, trans specifies the operation to be used as follows:

  trans = CBLAS_TRANSPOSE::CblasNoTrans,C := alpha*A*A**T + beta*C.

  trans = CBLAS_TRANSPOSE::CblasTrans,C := alpha*A**T*A + beta*C.
  \endverbatim

  \param[in] n
  \verbatim
  n is INTEGER
  On entry,  n specifies the order of the matrix C.  n must be
  at least zero.
  \endverbatim

  \param[in] k
  \verbatim
  k is INTEGER
  If trans = CblasNoTrans: k is number of columns of the matrix A.
  Otherwise:               k is number of rows    of the matrix A.
  k must be at least  zero.
  \endverbatim

  \param[in] alpha
  \verbatim
  alpha is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, alpha specifies the scalar alpha.
  \endverbatim

  \param[in] A
  \verbatim
  A is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If transA = CblasNoTrans:
  n-by-k , stored in an lda-by-k array [RowMajor: n-by-lda].
  Otherwise:
  k-by-n , stored in an lda-by-n array [RowMajor: k-by-lda].
  \endverbatim

  \param[in] lda
  \verbatim
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  If transA = CblasNoTrans: lda >= max(1, n) [RowMajor: lda >= max(1, k)].
  Otherwise:                lda >= max(1, k) [RowMajor: lda >= max(1, n)].
  \endverbatim

  \param[in] beta
  \verbatim
  beta is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, beta specifies the scalar alpha.When  beta  is
  supplied as zero then C need not be set on input.
  \endverbatim

  \param[in,out] C
  \verbatim
  C is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array, dimension :
  The n-by-n symmetric matrix C,
  stored in an ldc-by-n array [RowMajor: n-by-ldc].
  On exit, the array  C  is overwritten by the  lower/upper
  triangular part of the updated matrix.
  \endverbatim

  \param[in] ldc
  \verbatim
  ldc is INTEGER
  On entry, ldc specifies the first dimension of C
  ldc >= max(1, n)
  \endverbatim

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

/*! @brief \b SYR2K

  \verbatim

  SYR2K  performs one of the symmetric rank 2k operations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

  C := alpha*A*B**T + alpha*B*A**T + beta*C,

  or

  C := alpha*A**T*B + alpha*B**T*A + beta*C,

  where  alpha and beta  are scalars,  C is an  n by n symmetric matrix
  and  A  and B are n by k  matrices in the first case and k by n  matrices
  in the second case.
  \endverbatim

  \param[in] layout
  \verbatim
  layout is enum CBLAS_LAYOUT

  layout specifies Matrix storage as follows:

  layout = CBLAS_LAYOUT::CblasRowMajor or Layout::CblasColMajor.
  \endverbatim

  \param[in] uplo
  \verbatim
  uplo is enum CBLAS_UPLO

  uplo specifies specifies whether  the  upper  or  lower
  triangular  part  of  the  array C  is  to  be  referenced
  as follows:

  uplo = CBLAS_UPLO::CblasUpper   Only the upper triangular part of C
                                  is to be referenced.

  uplo = CBLAS_UPLO::CblasLower   Only the lower triangular part of C
                                  is to be referenced.
  \endverbatim

  \param[in] trans
  \verbatim

  trans is CBLAS_TRANSPOSE
  On entry, trans specifies the operation to be used as follows:

  trans = CBLAS_TRANSPOSE::CblasNoTrans,C := alpha*A*B**T + alpha*B*A**T
                                             + beta*C.

  trans = CBLAS_TRANSPOSE::CblasTrans,  C := alpha*A**T*B + alpha*B**T*A
                                             + beta*C.
  \endverbatim

  \param[in] n
  \verbatim
  n is INTEGER
  On entry,  n specifies the order of the matrix C.  n must be
  at least zero.
  \endverbatim

  \param[in] k
  \verbatim
  k is INTEGER
  If trans = CblasNoTrans: k is number of columns of the matrices A & B.
  Otherwise:               k is number of rows    of the matrices A & B.
  k must be at least  zero.
  \endverbatim

  \param[in] alpha
  \verbatim
  alpha is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, alpha specifies the scalar alpha.
  \endverbatim

  \param[in] A
  \verbatim
  A is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If trans = CblasNoTrans:
  n-by-k , stored in an lda-by-k array [RowMajor: n-by-lda].
  Otherwise:
  k-by-n , stored in an lda-by-n array [RowMajor: k-by-lda].
  \endverbatim

  \param[in] lda
  \verbatim
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  If trans = CblasNoTrans: lda >= max(1, n) [RowMajor: lda >= max(1, k)].
  Otherwise:                lda >= max(1, k) [RowMajor: lda >= max(1, n)].
  \endverbatim

  \param[in] B
  \verbatim
  B is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If trans = CblasNoTrans:
  n-by-k , stored in an ldb-by-k array [RowMajor: n-by-ldb].
  Otherwise:
  k-by-n , stored in an ldb-by-n array [RowMajor: k-by-ldb]
  \endverbatim

  \param[in] ldb
  \verbatim
  ldb is INTEGER
  On entry, ldb specifies the Leading dimension of B
  If trans = CblasNoTrans: ldb >= max(1, n) [RowMajor: ldb >= max(1, k)].
  Otherwise:               ldb >= max(1, k) [RowMajor: ldb >= max(1, n)].
  \endverbatim

  \param[in] beta
  \verbatim
  beta is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, beta specifies the scalar alpha.When  beta  is
  supplied as zero then C need not be set on input.
  \endverbatim

  \param[in,out] C
  \verbatim
  C is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array, dimension :
  The n-by-n symmetric matrix C,
  stored in an ldc-by-n array [RowMajor: n-by-ldc].
  On exit, the array  C  is overwritten by the  lower/upper
  triangular part of the updated matrix.
  \endverbatim

  \param[in] ldc
  \verbatim
  ldc is INTEGER
  On entry, ldc specifies the first dimension of C
  ldc >= max(1, n)
  \endverbatim

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

/*! @brief \b HERK

  \verbatim

  HERK  performs one of the hermitian rank k operations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

    C := alpha*A*B**H + conjg( alpha )*B*A**H + beta*C,
  
  or
  
    C := alpha*A**H*B + conjg( alpha )*B**H*A + beta*C,

  where  alpha and beta  are real scalars,  C is an  n by n  hermitian 
  matrix and  A  is an n by k  matrix in the first case and 
  k by n  matrix in the second case.
  \endverbatim

  \param[in] layout
  \verbatim
  layout is enum CBLAS_LAYOUT

  layout specifies Matrix storage as follows:

  layout = CBLAS_LAYOUT::CblasRowMajor or Layout::CblasColMajor.
  \endverbatim

  \param[in] uplo
  \verbatim
  uplo is enum CBLAS_UPLO

  uplo specifies specifies whether  the  upper  or  lower
  triangular  part  of  the  array C  is  to  be  referenced 
  as follows:

  uplo = CBLAS_UPLO::CblasUpper   Only the upper triangular part of C
                                  is to be referenced.

  uplo = CBLAS_UPLO::CblasLower   Only the lower triangular part of C
                                  is to be referenced.
  \endverbatim

  \param[in] trans
  \verbatim

  trans is CBLAS_TRANSPOSE
  On entry, trans specifies the operation to be used as follows:

  trans = CBLAS_TRANSPOSE::CblasNoTrans,  C := alpha*A*A**H + beta*C.

  trans = CBLAS_TRANSPOSE::CblasConjTrans,C := alpha*A**H*A + beta*C.
  \endverbatim

  \param[in] n
  \verbatim
  n is INTEGER
  On entry,  n specifies the order of the matrix C.  n must be
  at least zero.
  \endverbatim

  \param[in] k
  \verbatim
  k is INTEGER
  If trans = CblasNoTrans: k is number of columns of the matrix   A. 
  Otherwise:               k is number of rows    of the matrix   A.
  k must be at least  zero.
  \endverbatim

  \param[in] alpha
  \verbatim
  alpha is REAL/DOUBLE PRECISION
  On entry, alpha specifies the scalar alpha.
  \endverbatim

  \param[in] A
  \verbatim
  A is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If trans = CblasNoTrans:
  n-by-k , stored in an lda-by-k array [RowMajor: n-by-lda].
  Otherwise:
  k-by-n , stored in an lda-by-n array [RowMajor: k-by-lda].
  \endverbatim

  \param[in] lda
  \verbatim
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  If trans = CblasNoTrans: lda >= max(1, n) [RowMajor: lda >= max(1, k)].
  Otherwise:                lda >= max(1, k) [RowMajor: lda >= max(1, n)].
  \endverbatim
  
  \param[in] beta
  \verbatim
  beta is REAL/DOUBLE PRECISION
  On entry, beta specifies the scalar alpha.When  beta  is
  supplied as zero then C need not be set on input.
  \endverbatim

  \param[in,out] C
  \verbatim
  C is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array, dimension :
  The n-by-n Hermitian matrix C,
  stored in an ldc-by-n array [RowMajor: n-by-ldc].
  On exit, the array  C  is overwritten by the  lower/upper 
  triangular part of the updated matrix.
  \endverbatim

  \param[in] ldc
  \verbatim
  ldc is INTEGER
  On entry, ldc specifies the first dimension of C
  ldc >= max(1, n) 
  \endverbatim
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

/*! @brief \b HER2K

  \verbatim

  HER2K  performs one of the hermitian rank 2k operations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

    C := alpha*A*B**H + conjg( alpha )*B*A**H + beta*C,
  
  or
  
    C := alpha*A**H*B + conjg( alpha )*B**H*A + beta*C,

  where  alpha and beta  are scalars with  beta  real,  C is an  n by n 
  hermitian matrix and  A  and B are n by k  matrices in the first case 
  and k by n  matrices in the second case.
  \endverbatim

  \param[in] layout
  \verbatim
  layout is enum CBLAS_LAYOUT

  layout specifies Matrix storage as follows:

  layout = CBLAS_LAYOUT::CblasRowMajor or Layout::CblasColMajor.
  \endverbatim

  \param[in] uplo
  \verbatim
  uplo is enum CBLAS_UPLO

  uplo specifies specifies whether  the  upper  or  lower
  triangular  part  of  the  array C  is  to  be  referenced 
  as follows:

  uplo = CBLAS_UPLO::CblasUpper   Only the upper triangular part of C
                                  is to be referenced.

  uplo = CBLAS_UPLO::CblasLower   Only the lower triangular part of C
                                  is to be referenced.
  \endverbatim

  \param[in] trans
  \verbatim

  trans is CBLAS_TRANSPOSE
  On entry, trans specifies the operation to be used as follows:

  trans = CBLAS_TRANSPOSE::CblasNoTrans,  C := alpha*A*B**H          +
                                             conjg( alpha )*B*A**H +
                                             beta*C.

  trans = CBLAS_TRANSPOSE::CblasConjTrans,C := alpha*A**H*B          +
                                             conjg( alpha )*B**H*A +  
                                             beta*C.
  \endverbatim

  \param[in] n
  \verbatim
  n is INTEGER
  On entry,  n specifies the order of the matrix C.  n must be
  at least zero.
  \endverbatim

  \param[in] k
  \verbatim
  k is INTEGER
  If trans = CblasNoTrans: k is number of columns of the matrices A & B. 
  Otherwise:               k is number of rows    of the matrices A & B.
  k must be at least  zero.
  \endverbatim

  \param[in] alpha
  \verbatim
  alpha is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, alpha specifies the scalar alpha.
  \endverbatim

  \param[in] A
  \verbatim
  A is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If trans = CblasNoTrans:
  n-by-k , stored in an lda-by-k array [RowMajor: n-by-lda].
  Otherwise:
  k-by-n , stored in an lda-by-n array [RowMajor: k-by-lda].
  \endverbatim

  \param[in] lda
  \verbatim
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  If trans = CblasNoTrans: lda >= max(1, n) [RowMajor: lda >= max(1, k)].
  Otherwise:                lda >= max(1, k) [RowMajor: lda >= max(1, n)].
  \endverbatim

  \param[in] B
  \verbatim
  B is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If trans = CblasNoTrans:
  n-by-k , stored in an ldb-by-k array [RowMajor: n-by-ldb].
  Otherwise:
  k-by-n , stored in an ldb-by-n array [RowMajor: k-by-ldb]
  \endverbatim

  \param[in] ldb
  \verbatim
  ldb is INTEGER
  On entry, ldb specifies the Leading dimension of B
  If trans = CblasNoTrans: ldb >= max(1, n) [RowMajor: ldb >= max(1, k)].
  Otherwise:               ldb >= max(1, k) [RowMajor: ldb >= max(1, n)].
  \endverbatim
  
  \param[in] beta
  \verbatim
  beta is REAL/DOUBLE PRECISION
  On entry, beta specifies the scalar alpha.When  beta  is
  supplied as zero then C need not be set on input.
  \endverbatim

  \param[in,out] C
  \verbatim
  C is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array, dimension :
  The n-by-n Hermitian matrix C,
  stored in an ldc-by-n array [RowMajor: n-by-ldc].
  On exit, the array  C  is overwritten by the  lower/upper 
  triangular part of the updated matrix.
  \endverbatim

  \param[in] ldc
  \verbatim
  ldc is INTEGER
  On entry, ldc specifies the first dimension of C
  ldc >= max(1, n) 
  \endverbatim
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
