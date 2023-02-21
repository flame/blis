#include "blis.h"
#include <dlfcn.h>
#include "level3/ref_her2k.h"

namespace testinghelpers {

template <typename T, typename RT>
void ref_her2k(
    char storage, char uplo, char transa, char transb,
    gtint_t m, gtint_t k,
    T* alpha,
    T* ap, gtint_t lda,
    T* bp, gtint_t ldb,
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
    if( transa == 't' )
        cblas_transa = CblasTrans;
    else if( transa == 'c' )
        cblas_transa = CblasConjTrans;
    else
        cblas_transa = CblasNoTrans;

    typedef void (*Fptr_ref_cblas_her2k)( const CBLAS_ORDER, const CBLAS_UPLO, const CBLAS_TRANSPOSE,
                    const f77_int, const f77_int, const T*, const T*, f77_int,
                    const T*, f77_int, const RT, T*, f77_int);
    Fptr_ref_cblas_her2k ref_cblas_her2k;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(scomplex))
    {
        ref_cblas_her2k = (Fptr_ref_cblas_her2k)dlsym(refCBLASModule.get(), "cblas_cher2k");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
        ref_cblas_her2k = (Fptr_ref_cblas_her2k)dlsym(refCBLASModule.get(), "cblas_zher2k");
    }
    else
    {
        throw std::runtime_error("Error in ref_her2k.cpp: Invalid typename is passed function template.");
    }
    if( !ref_cblas_her2k ) {
        throw std::runtime_error("Error in ref_her2k.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_cblas_her2k( cblas_order, cblas_uplo, cblas_transa,
                  m, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
}

// Explicit template instantiations
template void ref_her2k<scomplex>(char, char, char, char, gtint_t, gtint_t, scomplex*,
                      scomplex*, gtint_t, scomplex*, gtint_t, float, scomplex*, gtint_t );
template void ref_her2k<dcomplex>(char, char, char, char, gtint_t, gtint_t, dcomplex*,
                      dcomplex*, gtint_t, dcomplex*, gtint_t, double, dcomplex*, gtint_t );
} //end of namespace testinghelpers