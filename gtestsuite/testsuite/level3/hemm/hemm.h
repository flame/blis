#pragma once

#include "blis.h"
#include "common/testing_helpers.h"

/**
 * @brief Performs the operation:
 * For BLIS-typed API:
 *        C := alpha*conj( A )*trans( B ) + beta*C, if side is left
 *     or C := alpha*trans( B )*conj( A ) + beta*C, if side is right
 * For BLAs/CBLAS API:
 *        C := alpha*A*B + beta*C, if side is left
 *     or C := alpha*B*A + beta*C, if side is right
 * @param[in]     storage specifies storage format used for the matrices
 * @param[in]     side   specifies if the hemmetric matrix A appears left or right in
                         the matrix multiplication
 * @param[in]     uplo   specifies if the upper or lower triangular part of A is used
 * @param[in]     conja specifies the form of op( A ) to be used in
                         the matrix multiplication
 * @param[in]     transb specifies the form of op( B ) to be used in
                         the matrix multiplication
 * @param[in]     m      specifies the number of rows and cols of the  matrix
                         op( A ) and rows of the matrix C and B
 * @param[in]     n      specifies the number of columns of the matrix
                         op( B ) and the number of columns of the matrix C
 * @param[in]     alpha  specifies the scalar alpha.
 * @param[in]     ap     specifies pointer which points to the first element of ap
 * @param[in]     rsa    specifies row increment of ap.
 * @param[in]     csa    specifies column increment of ap.
 * @param[in]     bp     specifies pointer which points to the first element of bp
 * @param[in]     rsb    specifies row increment of bp.
 * @param[in]     csb    specifies column increment of bp.
 * @param[in]     beta   specifies the scalar beta.
 * @param[in,out] cp     specifies pointer which points to the first element of cp
 * @param[in]     rsc    specifies row increment of cp.
 * @param[in]     csc    specifies column increment of cp.
 */

template<typename T>
static void hemm_(char side, char uplo, gtint_t m, gtint_t n, T* alpha,
                    T* ap, gtint_t lda,  T* bp, gtint_t ldb, T* beta, T* cp, gtint_t ldc )
{
    if constexpr (std::is_same<T, scomplex>::value)
        chemm_( &side, &uplo, &m, &n, alpha, ap, &lda, bp, &ldb, beta, cp, &ldc );
    else if constexpr (std::is_same<T, dcomplex>::value)
        zhemm_( &side, &uplo, &m, &n, alpha, ap, &lda, bp, &ldb, beta, cp, &ldc );
    else
        throw std::runtime_error("Error in testsuite/level3/hemm.h: Invalid typename in hemm_().");
}

template<typename T>
static void cblas_hemm(char storage, char side, char uplo,
    gtint_t m, gtint_t n, T* alpha, T* ap, gtint_t lda,
    T* bp, gtint_t ldb, T* beta, T* cp, gtint_t ldc)
{
    enum CBLAS_ORDER cblas_order;
    if( storage == 'c' || storage == 'C' )
        cblas_order = CblasColMajor;
    else
        cblas_order = CblasRowMajor;

    enum CBLAS_SIDE cblas_side;
    if( (side == 'l') || (side == 'L') )
        cblas_side = CblasLeft;
    else
        cblas_side = CblasRight;

    enum CBLAS_UPLO cblas_uplo;
    if( (uplo == 'u') || (uplo == 'U') )
        cblas_uplo = CblasUpper;
    else
        cblas_uplo = CblasLower;

    if constexpr (std::is_same<T, scomplex>::value)
        cblas_chemm( cblas_order, cblas_side, cblas_uplo, m, n, alpha, ap, lda, bp, ldb, beta, cp, ldc );
    else if constexpr (std::is_same<T, dcomplex>::value)
        cblas_zhemm( cblas_order, cblas_side, cblas_uplo, m, n, alpha, ap, lda, bp, ldb, beta, cp, ldc );
    else
        throw std::runtime_error("Error in testsuite/level3/hemm.h: Invalid typename in cblas_hemm().");
}

template<typename T>
static void typed_hemm(char storage, char side, char uplo, char conj_a, char trnsb,
    gtint_t m, gtint_t n, T* alpha, T* ap, gtint_t lda,
    T* bp, gtint_t ldb, T* beta, T* cp, gtint_t ldc)
{
    conj_t conja;
    trans_t transb;
    side_t blis_side;
    uplo_t blis_uplo;
    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_conj( conj_a, &conja );
    testinghelpers::char_to_blis_trans( trnsb, &transb );
    testinghelpers::char_to_blis_uplo( uplo, &blis_uplo );
    testinghelpers::char_to_blis_side( side, &blis_side );
    dim_t rsa,csa;
    dim_t rsb,csb;
    dim_t rsc,csc;

    rsa=rsb=rsc=1;
    csa=csb=csc=1;
    /* a = m x k       b = k x n       c = m x n    */
    if( (storage == 'c') || (storage == 'C') ) {
        csa = lda ;
        csb = ldb ;
        csc = ldc ;
    } else {
        rsa = lda ;
        rsb = ldb ;
        rsc = ldc ;
    }

    if constexpr (std::is_same<T, float>::value)
        bli_shemm( blis_side, blis_uplo, conja, transb, m, n, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc );
    else if constexpr (std::is_same<T, double>::value)
        bli_dhemm( blis_side, blis_uplo, conja, transb, m, n, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_chemm( blis_side, blis_uplo, conja, transb, m, n, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zhemm( blis_side, blis_uplo, conja, transb, m, n, alpha, ap, rsa, csa, bp, rsb, csb, beta, cp, rsc, csc );
    else
        throw std::runtime_error("Error in testsuite/level3/hemm.h: Invalid typename in typed_hemm().");
}

template<typename T>
static void hemm( char storage, char side, char uplo, char conja, char transb, gtint_t m, gtint_t n,
    T* alpha, T* ap, gtint_t lda, T* bp, gtint_t ldb, T* beta, T* cp, gtint_t ldc )
{
#ifdef TEST_BLAS
    if( storage == 'c' || storage == 'C' )
        hemm_<T>( side, uplo, m, n, alpha, ap, lda, bp, ldb, beta, cp, ldc );
    else
        throw std::runtime_error("Error in testsuite/level3/hemm.h: BLAS interface cannot be tested for row-major order.");

#elif TEST_CBLAS
    cblas_hemm<T>( storage, side, uplo, m, n, alpha, ap, lda, bp, ldb, beta, cp, ldc );
#elif TEST_BLIS_TYPED
    typed_hemm<T>( storage, side, uplo, conja, transb, m, n, alpha, ap, lda, bp, ldb, beta, cp, ldc );
#else
    throw std::runtime_error("Error in testsuite/level3/hemm.h: No interfaces are set to be tested.");
#endif
}
