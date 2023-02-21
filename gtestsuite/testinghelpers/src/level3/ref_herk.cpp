#include "blis.h"
#include <dlfcn.h>
#include "level3/ref_herk.h"

namespace testinghelpers {

template <typename T, typename RT>
void ref_herk(
    char storage, char uplo, char trnsa,
    gtint_t m, gtint_t k,
    RT alpha,
    T* ap, gtint_t lda,
    RT beta,
    T* cp, gtint_t ldc
) {
    enum CBLAS_ORDER cblas_order;
    if( (storage == 'c') || (storage == 'C') )
        cblas_order = CblasColMajor;
    else
        cblas_order = CblasRowMajor;

    enum CBLAS_UPLO cblas_uplo;
    if( (uplo == 'u') || (uplo == 'U') )
        cblas_uplo = CblasUpper;
    else
        cblas_uplo = CblasLower;

    enum CBLAS_TRANSPOSE cblas_transa;
    if( trnsa == 't' )
        cblas_transa = CblasTrans;
    else if( trnsa == 'c' )
        cblas_transa = CblasConjTrans;
    else
        cblas_transa = CblasNoTrans;

    typedef void (*Fptr_ref_cblas_herk)( const CBLAS_ORDER, const CBLAS_UPLO, const CBLAS_TRANSPOSE,
                    const f77_int, const f77_int, const RT, const T*, f77_int,
                    const RT, T*, f77_int);
    Fptr_ref_cblas_herk ref_cblas_herk;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(scomplex))
    {
        ref_cblas_herk = (Fptr_ref_cblas_herk)dlsym(refCBLASModule.get(), "cblas_cherk");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
        ref_cblas_herk = (Fptr_ref_cblas_herk)dlsym(refCBLASModule.get(), "cblas_zherk");
    }
    else
    {
        throw std::runtime_error("Error in ref_herk.cpp: Invalid typename is passed function template.");
    }
    if( !ref_cblas_herk ) {
        throw std::runtime_error("Error in ref_herk.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_cblas_herk( cblas_order, cblas_uplo, cblas_transa,
                  m, k, alpha, ap, lda, beta, cp, ldc );
}

// Explicit template instantiations
template void ref_herk<scomplex>(char, char, char, gtint_t, gtint_t, float,
                      scomplex*, gtint_t, float, scomplex*, gtint_t );
template void ref_herk<dcomplex>(char, char, char, gtint_t, gtint_t, double,
                      dcomplex*, gtint_t, double, dcomplex*, gtint_t );

} //end of namespace testinghelpers