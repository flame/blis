#include "blis.h"
#include <dlfcn.h>
#include "level2/ref_syr.h"

/*
 * ==========================================================================
 * SYR performs the symmetric rank 1 operation
 *    A := alpha*x*x**T + A,
 *  where alpha is a real scalar, x is an n element vector and A is an
 *  n by n symmetric matrix.
 * ==========================================================================
**/

namespace testinghelpers {

template <typename T>
void ref_syr( char storage, char uploa, char conjx, gtint_t n, T alpha,
                             T *xp, gtint_t incx, T *ap, gtint_t lda )
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

    typedef void (*Fptr_ref_cblas_syr)( const CBLAS_ORDER, const CBLAS_UPLO, const f77_int,
                                        const T, const T*, f77_int, T*, f77_int);

    Fptr_ref_cblas_syr ref_cblas_syr;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_cblas_syr = (Fptr_ref_cblas_syr)dlsym(refCBLASModule.get(), "cblas_ssyr");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_cblas_syr = (Fptr_ref_cblas_syr)dlsym(refCBLASModule.get(), "cblas_dsyr");
    }
    else
    {
      throw std::runtime_error("Error in ref_syr.cpp: Invalid typename is passed function template.");
    }
    if (!ref_cblas_syr) {
        throw std::runtime_error("Error in ref_syr.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_cblas_syr( cblas_order, cblas_uploa, n, alpha, xp, incx, ap, lda );

}

// Explicit template instantiations
template void ref_syr<float>( char , char , char, gtint_t , float ,
                               float *, gtint_t , float *, gtint_t );
template void ref_syr<double>( char , char , char, gtint_t , double ,
                               double *, gtint_t , double *, gtint_t );

} //end of namespace testinghelpers