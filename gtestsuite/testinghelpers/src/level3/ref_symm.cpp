#include "blis.h"
#include <dlfcn.h>
#include "level3/ref_symm.h"

namespace testinghelpers {

template<typename T>
void ref_symm (
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
    typedef void (*Fptr_ref_cblas_symm)( const CBLAS_ORDER, const CBLAS_SIDE, const CBLAS_UPLO,
                    const f77_int, const f77_int, const scalar_t, const T*, f77_int,
                    const T*, f77_int, const scalar_t, T*, f77_int);
    Fptr_ref_cblas_symm ref_cblas_symm;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_cblas_symm = (Fptr_ref_cblas_symm)dlsym(refCBLASModule.get( ), "cblas_ssymm");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_cblas_symm = (Fptr_ref_cblas_symm)dlsym(refCBLASModule.get(), "cblas_dsymm");
    }
    else if (typeid(T) == typeid(scomplex))
    {
        ref_cblas_symm = (Fptr_ref_cblas_symm)dlsym(refCBLASModule.get(), "cblas_csymm");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
        ref_cblas_symm = (Fptr_ref_cblas_symm)dlsym(refCBLASModule.get(), "cblas_zsymm");
    }
    else
    {
        throw std::runtime_error("Error in ref_symm.cpp: Invalid typename is passed function template.");
    }
    if( !ref_cblas_symm ) {
        throw std::runtime_error("Error in ref_symm.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_cblas_symm( cblas_order, cblas_side, cblas_uplo,
                  m, n, alpha, ap, lda, bp, ldb, beta, cp, ldc );
}
// Explicit template instantiations
template void ref_symm<float>(char, char, char, char, char, gtint_t, gtint_t, float,
                      float*, gtint_t, float*, gtint_t, float, float*, gtint_t );
template void ref_symm<double>(char, char, char, char, char, gtint_t, gtint_t, double,
                      double*, gtint_t, double*, gtint_t, double, double*, gtint_t );
template void ref_symm<scomplex>(char, char, char, char, char, gtint_t, gtint_t, scomplex,
                      scomplex*, gtint_t, scomplex*, gtint_t, scomplex, scomplex*, gtint_t );
template void ref_symm<dcomplex>(char, char, char, char, char, gtint_t, gtint_t, dcomplex,
                      dcomplex*, gtint_t, dcomplex*, gtint_t, dcomplex, dcomplex*, gtint_t );

} //end of namespace testinghelpers