#include "blis.h"
#include <dlfcn.h>
#include "level2/ref_trsv.h"

/*
 * ==========================================================================
 * TRSV Solves a triangular system of equations with a single value for the
 *        right side
 *    b := alpha * inv(transa(A)) * x_orig
 * where b and x are n element vectors and A is an n by n unit, or non-unit,
 * upper or lower triangular matrix.
 * ==========================================================================
**/

namespace testinghelpers {

template <typename T>
void ref_trsv( char storage, char uploa, char transa, char diaga,
    gtint_t n, T *alpha, T *ap, gtint_t lda, T *xp, gtint_t incx )
{
    enum CBLAS_ORDER cblas_order;
    if( (storage == 'c') || (storage == 'C') )
        cblas_order = CblasColMajor;
    else
        cblas_order = CblasRowMajor;

    enum CBLAS_UPLO cblas_uploa;
    if( (uploa == 'u') || (uploa == 'U') )
        cblas_uploa = CblasUpper;
    else
        cblas_uploa = CblasLower;

    enum CBLAS_TRANSPOSE cblas_trans;
    if( transa == 't' )
        cblas_trans = CblasTrans;
    else if( transa == 'c' )
        cblas_trans = CblasConjTrans;
    else
        cblas_trans = CblasNoTrans;

    enum CBLAS_DIAG cblas_diaga;
    if( (diaga == 'u') || (diaga == 'U') )
        cblas_diaga = CblasUnit;
    else
        cblas_diaga = CblasNonUnit;

    alphax<T>( n, *alpha, xp, incx );

    typedef void (*Fptr_ref_cblas_trsv)( const CBLAS_ORDER, const CBLAS_UPLO,
                                         const CBLAS_TRANSPOSE, CBLAS_DIAG ,
                                         f77_int, const T*, f77_int, T*, f77_int );
    Fptr_ref_cblas_trsv ref_cblas_trsv;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_cblas_trsv = (Fptr_ref_cblas_trsv)dlsym(refCBLASModule.get(), "cblas_strsv");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_cblas_trsv = (Fptr_ref_cblas_trsv)dlsym(refCBLASModule.get(), "cblas_dtrsv");
    }
    else if (typeid(T) == typeid(scomplex))
    {
        ref_cblas_trsv = (Fptr_ref_cblas_trsv)dlsym(refCBLASModule.get(), "cblas_ctrsv");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
        ref_cblas_trsv = (Fptr_ref_cblas_trsv)dlsym(refCBLASModule.get(), "cblas_ztrsv");
    }
    else
    {
      throw std::runtime_error("Error in ref_trsv.cpp: Invalid typename is passed function template.");
    }
    if (!ref_cblas_trsv) {
        throw std::runtime_error("Error in ref_trsv.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_cblas_trsv( cblas_order, cblas_uploa, cblas_trans, cblas_diaga, n, ap, lda, xp, incx );
}

// Explicit template instantiations
template void ref_trsv<float>( char , char , char , char , gtint_t ,
                              float *, float *, gtint_t , float *, gtint_t );
template void ref_trsv<double>( char , char , char , char , gtint_t ,
                              double *, double *, gtint_t , double *, gtint_t );
template void ref_trsv<scomplex>( char , char , char , char , gtint_t ,
                              scomplex *, scomplex *, gtint_t , scomplex *, gtint_t );
template void ref_trsv<dcomplex>( char , char , char , char , gtint_t ,
                              dcomplex *, dcomplex *, gtint_t , dcomplex *, gtint_t );

} //end of namespace testinghelpers