#ifndef BLIS_HH
#define BLIS_HH

#include "blis_util.hh"
#include "cblas.hh"
#include <limits>

namespace blis {

// =============================================================================
/// General matrix-matrix multiply,
///     \f[ C = \alpha op(A) \times op(B) + \beta C, \f]
/// where op(X) is one of
///     \f[ op(X) = X,   \f]
///     \f[ op(X) = X^T, \f]
///     \f[ op(X) = X^H, \f]
/// alpha and beta are scalars, and A, B, and C are matrices, with
/// op(A) an m-by-k matrix, op(B) a k-by-n matrix, and C an m-by-n matrix.
///
/// Generic implementation for arbitrary data types.
/// TODO: generic version not yet implemented.
///
/// @param[in] layout
///     Matrix storage, Layout::ColMajor or Layout::RowMajor.
///
/// @param[in] transA
///     The operation op(A) to be used:
///     - Op::NoTrans:   \f$ op(A) = A.   \f$
///     - Op::Trans:     \f$ op(A) = A^T. \f$
///     - Op::ConjTrans: \f$ op(A) = A^H. \f$
///
/// @param[in] transB
///     The operation op(B) to be used:
///     - Op::NoTrans:   \f$ op(B) = B.   \f$
///     - Op::Trans:     \f$ op(B) = B^T. \f$
///     - Op::ConjTrans: \f$ op(B) = B^H. \f$
///
/// @param[in] m
///     Number of rows of the matrix C and op(A). m >= 0.
///
/// @param[in] n
///     Number of columns of the matrix C and op(B). n >= 0.
///
/// @param[in] k
///     Number of columns of op(A) and rows of op(B). k >= 0.
///
/// @param[in] alpha
///     Scalar alpha. If alpha is zero, A and B are not accessed.
///
/// @param[in] A
///     - If transA = NoTrans:
///       the m-by-k matrix A, stored in an lda-by-k array [RowMajor: m-by-lda].
///     - Otherwise:
///       the k-by-m matrix A, stored in an lda-by-m array [RowMajor: k-by-lda].
///
/// @param[in] lda
///     Leading dimension of A.
///     - If transA = NoTrans: lda >= max(1, m) [RowMajor: lda >= max(1, k)].
///     - Otherwise:           lda >= max(1, k) [RowMajor: lda >= max(1, m)].
///
/// @param[in] B
///     - If transB = NoTrans:
///       the k-by-n matrix B, stored in an ldb-by-n array [RowMajor: k-by-ldb].
///     - Otherwise:
///       the n-by-k matrix B, stored in an ldb-by-k array [RowMajor: n-by-ldb].
///
/// @param[in] ldb
///     Leading dimension of B.
///     - If transB = NoTrans: ldb >= max(1, k) [RowMajor: ldb >= max(1, n)].
///     - Otherwise:           ldb >= max(1, n) [RowMajor: ldb >= max(1, k)].
///
/// @param[in] beta
///     Scalar beta. If beta is zero, C need not be set on input.
///
/// @param[in] C
///     The m-by-n matrix C, stored in an ldc-by-n array [RowMajor: m-by-ldc].
///
/// @param[in] ldc
///     Leading dimension of C. ldc >= max(1, m) [RowMajor: ldc >= max(1, n)].
///
/// @ingroup gemm

template< typename TA, typename TB, typename TC >
void gemm(
    CBLAS_LAYOUT layout,
    CBLAS_TRANSPOSE transA,
    CBLAS_TRANSPOSE transB,
    int64_t m, int64_t n, int64_t k,
    scalar_type<TA, TB, TC> alpha,
    TA const *A, int64_t lda,
    TB const *B, int64_t ldb,
    scalar_type<TA, TB, TC> beta,
    TC       *C, int64_t ldc )
{
//    printf("In gemm.cpp\n");
    cblas_gemm(layout, transA, transB, m, n, k, alpha, A,lda, B, ldb, beta, C, ldc);

};

}  // namespace blis

#endif        //  #ifndef BLIS_HH
