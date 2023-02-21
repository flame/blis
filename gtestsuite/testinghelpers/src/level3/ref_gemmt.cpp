#include "blis.h"
#include <dlfcn.h>
#include "level3/ref_gemm.h"
#include "level3/ref_gemmt.h"

/*
 * ==========================================================================
 *  GEMMT performs one of the matrix-matrix operations
 *     C := alpha*op( A )*op( B ) + beta*C,
 *  where  op( X ) is one of
 *     op( X ) = X   or   op( X ) = A**T   or   op( X ) = X**H,
 *  alpha and beta are scalars, and A, B and C are matrices, with op( A )
 *  an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
 *  Only accesses and updates the upper or the lower triangular part.
 * ==========================================================================
**/

namespace testinghelpers {
#if 1
template <typename T>
void ref_gemmt (
    char storage, char uplo, char trnsa, char trnsb,
    gtint_t n, gtint_t k,
    T alpha,
    T* ap, gtint_t lda,
    T* bp, gtint_t ldb,
    T beta,
    T* cp, gtint_t ldc
) {
    gtint_t smc = testinghelpers::matsize( storage, 'n', n, n, ldc );
    std::vector<T> C( smc );
    memcpy(C.data(), cp, (smc*sizeof(T)));
    ref_gemm<T>(storage, trnsa, trnsb, n, n, k, alpha, ap, lda, bp, ldb, beta, C.data(), ldc);
    if( (storage=='c')||(storage=='C') )
    {
        for(gtint_t j=0; j<n; j++)
        {
            for(gtint_t i=0; i<n; i++)
            {
                if( (uplo=='u')||(uplo=='U') )
                {
                    if(i<=j) cp[i+j*ldc] = C[i+j*ldc];
                }
                else if ( (uplo=='l')||(uplo=='L') )
                {
                    if (i>=j) cp[i+j*ldc] = C[i+j*ldc];
                }
                else
                    throw std::runtime_error("Error in level3/ref_gemmt.cpp: side must be 'u' or 'l'.");
            }
        }
    } else
    {
        for(gtint_t i=0; i<n; i++)
        {
            for(gtint_t j=0; j<n; j++)
            {
                if( (uplo=='u')||(uplo=='U') )
                {
                    if(i<=j) cp[j+i*ldc] = C[j+i*ldc];
                }
                else if ( (uplo=='l')||(uplo=='L') )
                {
                    if (i>=j) cp[j+i*ldc] = C[j+i*ldc];
                }
                else
                    throw std::runtime_error("Error in level3/ref_gemmt.cpp: side must be 'u' or 'l'.");
            }
        }
    }
}
#else
template <typename T>
void ref_gemmt (
    char storage, char uplo, char trnsa, char trnsb,
    gtint_t n, gtint_t k,
    T alpha,
    T* ap, gtint_t lda,
    T* bp, gtint_t ldb,
    T beta,
    T* cp, gtint_t ldc
)
{
    gtint_t sma = testinghelpers::matsize( storage, trnsa, n, k, lda );
    gtint_t smb = testinghelpers::matsize( storage, trnsb, k, n, ldb );

    enum CBLAS_ORDER cblas_order;
    if( (storage == 'c') || (storage == 'C') )
        cblas_order = CblasColMajor;
    else
        cblas_order = CblasRowMajor;

    enum CBLAS_TRANSPOSE cblas_transa;
    if( trnsa == 't' )
        cblas_transa = CblasTrans;
    else if( trnsa == 'c' )
        cblas_transa = CblasConjTrans;
    else
        cblas_transa = CblasNoTrans;

    enum CBLAS_TRANSPOSE cblas_transb;
    if( trnsb == 't' )
        cblas_transb = CblasTrans;
    else if( trnsb == 'c' )
        cblas_transb = CblasConjTrans;
    else
        cblas_transb = CblasNoTrans;

    enum CBLAS_UPLO cblas_uplo;
    if( (uplo == 'u') || (uplo == 'U') )
        cblas_uplo = CblasUpper;
    else
        cblas_uplo = CblasLower;

    std::vector<T> A( sma );
    memcpy(A.data(), ap, (sma*sizeof(T)));

    std::vector<T> B( smb );
    memcpy(B.data(), bp, (smb*sizeof(T)));

    if( trnsa == 'h' ) {
        testinghelpers::conj<T>( storage, A.data(), n, k, lda );
    }

    if( trnsb == 'h' ) {
        testinghelpers::conj<T>( storage, B.data(), k, n, ldb );
    }

    using scalar_t = std::conditional_t<testinghelpers::type_info<T>::is_complex, T&, T>;
    typedef void (*Fptr_ref_cblas_gemmt)( const CBLAS_ORDER, const CBLAS_UPLO, const CBLAS_TRANSPOSE, const CBLAS_TRANSPOSE,
                    const f77_int, const f77_int, const scalar_t, const T*, f77_int,
                    const T*, f77_int, const scalar_t, T*, f77_int);
    Fptr_ref_cblas_gemmt ref_cblas_gemmt;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_cblas_gemmt = (Fptr_ref_cblas_gemmt)dlsym(refCBLASModule.get( ), "cblas_sgemmt");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_cblas_gemmt = (Fptr_ref_cblas_gemmt)dlsym(refCBLASModule.get(), "cblas_dgemmt");
    }
    else if (typeid(T) == typeid(scomplex))
    {
        ref_cblas_gemmt = (Fptr_ref_cblas_gemmt)dlsym(refCBLASModule.get(), "cblas_cgemmt");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
        ref_cblas_gemmt = (Fptr_ref_cblas_gemmt)dlsym(refCBLASModule.get(), "cblas_zgemmt");
    }
    else
    {
        throw std::runtime_error("Error in ref_gemmt.cpp: Invalid typename is passed function template.");
    }
    if( !ref_cblas_gemmt ) {
        throw std::runtime_error("Error in ref_gemmt.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_cblas_gemmt( cblas_order, cblas_uplo, cblas_transa, cblas_transb,
                  n, k, alpha, A.data(), lda, B.data(), ldb, beta, cp, ldc );
}
#endif
// Explicit template instantiations
template void ref_gemmt<float>(char, char, char, char, gtint_t, gtint_t, float,
                      float*, gtint_t, float*, gtint_t, float, float*, gtint_t );
template void ref_gemmt<double>(char, char, char, char, gtint_t, gtint_t, double,
                      double*, gtint_t, double*, gtint_t, double, double*, gtint_t );
template void ref_gemmt<scomplex>(char, char, char, char, gtint_t, gtint_t, scomplex,
                      scomplex*, gtint_t, scomplex*, gtint_t, scomplex, scomplex*, gtint_t );
template void ref_gemmt<dcomplex>(char, char, char, char, gtint_t, gtint_t, dcomplex,
                      dcomplex*, gtint_t, dcomplex*, gtint_t, dcomplex, dcomplex*, gtint_t );


} //end of namespace testinghelpers