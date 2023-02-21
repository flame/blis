#include "blis.h"
#include <dlfcn.h>
#include "level3/ref_hemm.h"

namespace testinghelpers {

template<typename T>
void ref_hemm (
    char storage, char side, char uplo, char conja, char transb,
    gtint_t m, gtint_t n,
    T alpha,
    T* ap, gtint_t lda,
    T* bp, gtint_t ldb,
    T beta,
    T* cp, gtint_t ldc
) {
    enum CBLAS_ORDER cblas_order;
    if( (storage == 'c') || (storage == 'C') )
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

    using scalar_t = std::conditional_t<testinghelpers::type_info<T>::is_complex, T&, T>;
    typedef void (*Fptr_ref_cblas_hemm)( const CBLAS_ORDER, const CBLAS_SIDE, const CBLAS_UPLO,
                    const f77_int, const f77_int, const scalar_t, const T*, f77_int,
                    const T*, f77_int, const scalar_t, T*, f77_int);
    Fptr_ref_cblas_hemm ref_cblas_hemm;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(scomplex))
    {
        ref_cblas_hemm = (Fptr_ref_cblas_hemm)dlsym(refCBLASModule.get(), "cblas_chemm");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
        ref_cblas_hemm = (Fptr_ref_cblas_hemm)dlsym(refCBLASModule.get(), "cblas_zhemm");
    }
    else
    {
        throw std::runtime_error("Error in ref_hemm.cpp: Invalid typename is passed function template.");
    }
    if( !ref_cblas_hemm ) {
        throw std::runtime_error("Error in ref_hemm.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_cblas_hemm( cblas_order, cblas_side, cblas_uplo,
                  m, n, alpha, ap, lda, bp, ldb, beta, cp, ldc );
}
// Explicit template instantiations
template void ref_hemm<scomplex>(char, char, char, char, char, gtint_t, gtint_t, scomplex,
                      scomplex*, gtint_t, scomplex*, gtint_t, scomplex, scomplex*, gtint_t );
template void ref_hemm<dcomplex>(char, char, char, char, char, gtint_t, gtint_t, dcomplex,
                      dcomplex*, gtint_t, dcomplex*, gtint_t, dcomplex, dcomplex*, gtint_t );

} //end of namespace testinghelpers