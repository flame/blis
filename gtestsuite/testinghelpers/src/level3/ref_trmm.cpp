#include "blis.h"
#include <dlfcn.h>
#include "level3/ref_trmm.h"

/*
 * ==========================================================================
 * TRMM  performs one of the matrix-matrix operations
 *    B := alpha*op( A )*B,   or   B := alpha*B*op( A )
 * where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or
 * non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
 *    op( A ) = A   or   op( A ) = A**T   or   op( A ) = A**H.
 * ==========================================================================
 */

namespace testinghelpers {

template <typename T>
void ref_trmm( char storage, char side, char uploa, char transa, char diaga,
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
    typedef void (*Fptr_ref_cblas_trmm)( const CBLAS_ORDER, const CBLAS_SIDE, const CBLAS_UPLO,
                 const CBLAS_TRANSPOSE, const CBLAS_DIAG, const f77_int, const f77_int,
                 const scalar_t, const T*, f77_int, const T*, f77_int );

    Fptr_ref_cblas_trmm ref_cblas_trmm;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_cblas_trmm = (Fptr_ref_cblas_trmm)dlsym(refCBLASModule.get( ), "cblas_strmm");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_cblas_trmm = (Fptr_ref_cblas_trmm)dlsym(refCBLASModule.get(), "cblas_dtrmm");
    }
    else if (typeid(T) == typeid(scomplex))
    {
        ref_cblas_trmm = (Fptr_ref_cblas_trmm)dlsym(refCBLASModule.get(), "cblas_ctrmm");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
        ref_cblas_trmm = (Fptr_ref_cblas_trmm)dlsym(refCBLASModule.get(), "cblas_ztrmm");
    }
    else
    {
        throw std::runtime_error("Error in ref_trmm.cpp: Invalid typename is passed function template.");
    }
    if( !ref_cblas_trmm ) {
        throw std::runtime_error("Error in ref_trmm.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_cblas_trmm( cblas_order, cblas_side, cblas_uploa, cblas_transa,
                    cblas_diaga, m, n, alpha, ap, lda, bp, ldb );
}

// Explicit template instantiations
template void ref_trmm<float>( char, char, char, char, char,
              gtint_t, gtint_t, float, float *, gtint_t, float *, gtint_t );
template void ref_trmm<double>( char, char, char, char, char,
           gtint_t, gtint_t, double, double *, gtint_t, double *, gtint_t );
template void ref_trmm<scomplex>( char, char, char, char, char,
        gtint_t, gtint_t, scomplex, scomplex *, gtint_t, scomplex *, gtint_t );
template void ref_trmm<dcomplex>( char, char, char, char, char,
        gtint_t, gtint_t, dcomplex, dcomplex *, gtint_t, dcomplex *, gtint_t );

} //end of namespace testinghelpers