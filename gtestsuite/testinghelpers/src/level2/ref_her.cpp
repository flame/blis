#include "blis.h"
#include <dlfcn.h>
#include "level2/ref_her.h"

/*
 * ==========================================================================
 * HER performs the hermitian rank 1 operation
 *    A := alpha*x*x**H + A
 *  where alpha is a real scalar, x is an n element vector and A is an
 *  n by n hermitian matrix.
 * ==========================================================================
**/

namespace testinghelpers {

template <typename T, typename Tr>
void ref_her( char storage, char uploa, char conjx, gtint_t n, Tr alpha,
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

    typedef void (*Fptr_ref_cblas_her)( const CBLAS_ORDER, const CBLAS_UPLO, const f77_int,
                                        const Tr, const T*, f77_int, T*, f77_int);
    Fptr_ref_cblas_her ref_cblas_her;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(scomplex))
    {
        ref_cblas_her = (Fptr_ref_cblas_her)dlsym(refCBLASModule.get(), "cblas_cher");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
        ref_cblas_her = (Fptr_ref_cblas_her)dlsym(refCBLASModule.get(), "cblas_zher");
    }
    else
    {
      throw std::runtime_error("Error in ref_her.cpp: Invalid typename is passed function template.");
    }
    if (!ref_cblas_her) {
        throw std::runtime_error("Error in ref_her.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_cblas_her( cblas_order, cblas_uploa, n, alpha, xp, incx, ap, lda );

}

// Explicit template instantiations
template void ref_her<scomplex, float>( char , char , char , gtint_t , float ,
                               scomplex *, gtint_t , scomplex *, gtint_t );
template void ref_her<dcomplex, double>( char , char , char , gtint_t , double ,
                               dcomplex *, gtint_t , dcomplex *, gtint_t );

} //end of namespace testinghelpers