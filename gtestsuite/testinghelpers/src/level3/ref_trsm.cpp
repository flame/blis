#include "blis.h"
#include <dlfcn.h>
#include "level3/ref_trsm.h"

/*
 * ==========================================================================
 *  TRSM  solves one of the matrix equations
 *     op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
 *  where alpha is a scalar, X and B are m by n matrices, A is a unit, or
 *  non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
 *     op( A ) = A   or   op( A ) = A**T.
 *  The matrix X is overwritten on B.
 * ==========================================================================
 */

namespace testinghelpers {

template <typename T>
void ref_trsm( char storage, char side, char uploa, char transa, char diaga,
    gtint_t m, gtint_t n, T alpha, T *ap, gtint_t lda, T *bp, gtint_t ldb )
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

    enum CBLAS_UPLO cblas_uploa;
    if( (uploa == 'u') || (uploa == 'U') )
        cblas_uploa = CblasUpper;
    else
        cblas_uploa = CblasLower;

    enum CBLAS_TRANSPOSE cblas_transa;
    if( transa == 't' )
        cblas_transa = CblasTrans;
    else if( transa == 'c' )
        cblas_transa = CblasConjTrans;
    else
        cblas_transa = CblasNoTrans;

    enum CBLAS_DIAG cblas_diaga;
    if( (diaga == 'u') || (diaga == 'U') )
        cblas_diaga = CblasUnit;
    else
        cblas_diaga = CblasNonUnit;

    using scalar_t = std::conditional_t<testinghelpers::type_info<T>::is_complex, T&, T>;
    typedef void (*Fptr_ref_cblas_trsm)( const CBLAS_ORDER, const CBLAS_SIDE, const CBLAS_UPLO,
                 const CBLAS_TRANSPOSE, const CBLAS_DIAG, const f77_int, const f77_int,
                 const scalar_t, const T*, f77_int, const T*, f77_int );

    Fptr_ref_cblas_trsm ref_cblas_trsm;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_cblas_trsm = (Fptr_ref_cblas_trsm)dlsym(refCBLASModule.get( ), "cblas_strsm");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_cblas_trsm = (Fptr_ref_cblas_trsm)dlsym(refCBLASModule.get(), "cblas_dtrsm");
    }
    else if (typeid(T) == typeid(scomplex))
    {
        ref_cblas_trsm = (Fptr_ref_cblas_trsm)dlsym(refCBLASModule.get(), "cblas_ctrsm");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
        ref_cblas_trsm = (Fptr_ref_cblas_trsm)dlsym(refCBLASModule.get(), "cblas_ztrsm");
    }
    else
    {
        throw std::runtime_error("Error in ref_trsm.cpp: Invalid typename is passed function template.");
    }
    if( !ref_cblas_trsm ) {
        throw std::runtime_error("Error in ref_trsm.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_cblas_trsm( cblas_order, cblas_side, cblas_uploa, cblas_transa,
                    cblas_diaga, m, n, alpha, ap, lda, bp, ldb );
}

// Explicit template instantiations
template void ref_trsm<float>( char, char, char, char, char,
              gtint_t, gtint_t, float, float *, gtint_t, float *, gtint_t );
template void ref_trsm<double>( char, char, char, char, char,
           gtint_t, gtint_t, double, double *, gtint_t, double *, gtint_t );
template void ref_trsm<scomplex>( char, char, char, char, char,
        gtint_t, gtint_t, scomplex, scomplex *, gtint_t, scomplex *, gtint_t );
template void ref_trsm<dcomplex>( char, char, char, char, char,
        gtint_t, gtint_t, dcomplex, dcomplex *, gtint_t, dcomplex *, gtint_t );

} //end of namespace testinghelpers