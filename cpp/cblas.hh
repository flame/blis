#ifndef CBLAS_HH
#define CBLAS_HH

extern "C" {
#include <cblas.h>
#include <blis.h>
}


#include <complex>

namespace blis{

// =============================================================================
// Level 3 BLAS

// -----------------------------------------------------------------------------
inline void
cblas_gemm(
    CBLAS_ORDER layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
    int m, int n, int k,
    float  alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float  beta,
    float* C, int ldc )
{
//    printf("cblas_sgemm\n");
    cblas_sgemm( layout, transA, transB, m, n, k,
                 alpha, A, lda, B, ldb,
                 beta,  C, ldc );
}

inline void
cblas_gemm(
    CBLAS_ORDER layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
    int m, int n, int k,
    double  alpha,
    double const *A, int lda,
    double const *B, int ldb,
    double  beta,
    double* C, int ldc )
{
//    printf("cblas_dgemm\n");
    cblas_dgemm( layout, transA, transB, m, n, k,
                 alpha, A, lda, B, ldb,
                 beta,  C, ldc );
}

inline void
cblas_gemm(
    CBLAS_ORDER layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
    int m, int n, int k,
    std::complex<float>  alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *B, int ldb,
    std::complex<float>  beta,
    std::complex<float>* C, int ldc )
{
//    printf("cblas_cgemm\n");
    cblas_cgemm( layout, transA, transB, m, n, k,
                 &alpha, A, lda, B, ldb,
                 &beta,  C, ldc );
}

inline void
cblas_gemm(
    CBLAS_ORDER layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
    int m, int n, int k,
    std::complex<double>  alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *B, int ldb,
    std::complex<double>  beta,
    std::complex<double>* C, int ldc )
{
//    printf("cblas_zgemm\n");
    cblas_zgemm( layout, transA, transB, m, n, k,
                 &alpha, A, lda, B, ldb,
                 &beta,  C, ldc );
}
// -----------------------------------------------------------------------------
inline void
cblas_trsm(
    CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
    int m, int n,
    float alpha,
    float const *A, int lda,
    float       *B, int ldb )
{
    cblas_strsm( layout, side, uplo, trans, diag, m, n,  alpha, A, lda, B, ldb);
}

inline void
cblas_trsm(
    CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
    int m, int n,
    double alpha,
    double const *A, int lda,
    double       *B, int ldb )
{
    cblas_dtrsm( layout, side, uplo, trans, diag, m, n,  alpha, A, lda, B, ldb);
}

inline void
cblas_trsm(
    CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
    int m, int n,
    std::complex<float> alpha,
    std::complex<float> const *A, int lda,
    std::complex<float>       *B, int ldb )
{
    cblas_ctrsm( layout, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb );
}

inline void
cblas_trsm(
    CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
    int m, int n,
    std::complex<double> alpha,
    std::complex<double> const *A, int lda,
    std::complex<double>       *B, int ldb )
{
    cblas_ztrsm( layout, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb );
}

// -----------------------------------------------------------------------------
inline void
cblas_hemm(
    CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    float  alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float  beta,
    float* C, int ldc )
{
    cblas_ssymm( layout, side, uplo, m, n,
                 alpha, A, lda, B, ldb,
                 beta,  C, ldc );
}

inline void
cblas_hemm(
    CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    double  alpha,
    double const *A, int lda,
    double const *B, int ldb,
    double  beta,
    double* C, int ldc )
{
    cblas_dsymm( layout, side, uplo, m, n,
                 alpha, A, lda, B, ldb,
                 beta,  C, ldc );
}

inline void
cblas_hemm(
    CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    std::complex<float>  alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *B, int ldb,
    std::complex<float>  beta,
    std::complex<float>* C, int ldc )
{
    cblas_chemm( layout, side, uplo, m, n,
                 &alpha, A, lda, B, ldb,
                 &beta,  C, ldc );
}

inline void
cblas_hemm(
    CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    std::complex<double>  alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *B, int ldb,
    std::complex<double>  beta,
    std::complex<double>* C, int ldc )
{
    cblas_zhemm( layout, side, uplo, m, n,
                 &alpha, A, lda, B, ldb,
                 &beta,  C, ldc );
}

// -----------------------------------------------------------------------------
inline void
cblas_symm(
    CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    float  alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float  beta,
    float* C, int ldc )
{
    cblas_ssymm( layout, side, uplo, m, n,
                 alpha, A, lda, B, ldb,
                 beta,  C, ldc );
}

inline void
cblas_symm(
    CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    double  alpha,
    double const *A, int lda,
    double const *B, int ldb,
    double  beta,
    double* C, int ldc )
{
    cblas_dsymm( layout, side, uplo, m, n,
                 alpha, A, lda, B, ldb,
                 beta,  C, ldc );
}

inline void
cblas_symm(
    CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    std::complex<float>  alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *B, int ldb,
    std::complex<float>  beta,
    std::complex<float>* C, int ldc )
{
    cblas_csymm( layout, side, uplo, m, n,
                 &alpha, A, lda, B, ldb,
                 &beta,  C, ldc );
}

inline void
cblas_symm(
    CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    std::complex<double>  alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *B, int ldb,
    std::complex<double>  beta,
    std::complex<double>* C, int ldc )
{
    cblas_zsymm( layout, side, uplo, m, n,
                 &alpha, A, lda, B, ldb,
                 &beta,  C, ldc );
}


// -----------------------------------------------------------------------------
inline void
cblas_syrk(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    float alpha,
    float const *A, int lda,
    float beta,
    float*       C, int ldc )
{
    cblas_ssyrk( layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc );
}

inline void
cblas_syrk(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    double alpha,
    double const *A, int lda,
    double beta,
    double*       C, int ldc )
{
    cblas_dsyrk( layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc );
}

inline void
cblas_syrk(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    std::complex<float> alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> beta,
    std::complex<float>*       C, int ldc )
{
    cblas_csyrk( layout, uplo, trans, n, k, &alpha, A, lda, &beta, C, ldc );
}

inline void
cblas_syrk(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    std::complex<double> alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> beta,
    std::complex<double>* C, int ldc )
{
    cblas_zsyrk( layout, uplo, trans, n, k, &alpha, A, lda, &beta, C, ldc );
}

// -----------------------------------------------------------------------------
inline void
cblas_herk(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    float alpha,
    float const *A, int lda,
    float beta,
    float*       C, int ldc )
{
    cblas_ssyrk( layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc );
}

inline void
cblas_herk(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    double alpha,
    double const *A, int lda,
    double beta,
    double*       C, int ldc )
{
    cblas_dsyrk( layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc );
}

inline void
cblas_herk(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    float alpha,  // note: real
    std::complex<float> const *A, int lda,
    float beta,   // note: real
    std::complex<float>*       C, int ldc )
{
    cblas_cherk( layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc );
}

inline void
cblas_herk(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    double alpha,  // note: real
    std::complex<double> const *A, int lda,
    double beta,   // note: real
    std::complex<double>* C, int ldc )
{
    cblas_zherk( layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc );
}

// -----------------------------------------------------------------------------
inline void
cblas_syr2k(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta,
    float*       C, int ldc )
{
    cblas_ssyr2k( layout, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
}

inline void
cblas_syr2k(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    double alpha,
    double const *A, int lda,
    double const *B, int ldb,
    double beta,
    double*       C, int ldc )
{
    cblas_dsyr2k( layout, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
}

inline void
cblas_syr2k(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    std::complex<float> alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *B, int ldb,
    std::complex<float> beta,
    std::complex<float>*       C, int ldc )
{
    cblas_csyr2k( layout, uplo, trans, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc );
}

inline void
cblas_syr2k(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    std::complex<double> alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *B, int ldb,
    std::complex<double> beta,
    std::complex<double>* C, int ldc )
{
    cblas_zsyr2k( layout, uplo, trans, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc );
}

// -----------------------------------------------------------------------------
inline void
cblas_her2k(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta,
    float*       C, int ldc )
{
    cblas_ssyr2k( layout, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
}

inline void
cblas_her2k(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    double alpha,
    double const *A, int lda,
    double const *B, int ldb,
    double beta,
    double*       C, int ldc )
{
    cblas_dsyr2k( layout, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
}

inline void
cblas_her2k(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    std::complex<float> alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *B, int ldb,
    float beta,  // note: real
    std::complex<float>*       C, int ldc )
{
    cblas_cher2k( layout, uplo, trans, n, k, &alpha, A, lda, B, ldb, beta, C, ldc );
}

inline void
cblas_her2k(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    std::complex<double> alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *B, int ldb,
    double beta,  // note: real
    std::complex<double>* C, int ldc )
{
    cblas_zher2k( layout, uplo, trans, n, k, &alpha, A, lda, B, ldb, beta, C, ldc );
}
}//namespace blis

#endif        //  #ifndef CBLAS_HH
