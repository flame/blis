#include "blis.h"
#include <dlfcn.h>
#include "level3/ref_syrk.h"

namespace testinghelpers {

template <typename T>
void ref_syrk(
    char storage, char uplo, char trnsa,
    gtint_t m, gtint_t k,
    T alpha,
    T* ap, gtint_t lda,
    T beta,
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

    using scalar_t = std::conditional_t<testinghelpers::type_info<T>::is_complex, T&, T>;
    typedef void (*Fptr_ref_cblas_syrk)( const CBLAS_ORDER, const CBLAS_UPLO, const CBLAS_TRANSPOSE,
                    const f77_int, const f77_int, const scalar_t, const T*, f77_int,
                    const scalar_t, T*, f77_int);
    Fptr_ref_cblas_syrk ref_cblas_syrk;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_cblas_syrk = (Fptr_ref_cblas_syrk)dlsym(refCBLASModule.get( ), "cblas_ssyrk");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_cblas_syrk = (Fptr_ref_cblas_syrk)dlsym(refCBLASModule.get(), "cblas_dsyrk");
    }
    else if (typeid(T) == typeid(scomplex))
    {
        ref_cblas_syrk = (Fptr_ref_cblas_syrk)dlsym(refCBLASModule.get(), "cblas_csyrk");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
        ref_cblas_syrk = (Fptr_ref_cblas_syrk)dlsym(refCBLASModule.get(), "cblas_zsyrk");
    }
    else
    {
        throw std::runtime_error("Error in ref_syrk.cpp: Invalid typename is passed function template.");
    }
    if( !ref_cblas_syrk ) {
        throw std::runtime_error("Error in ref_syrk.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_cblas_syrk( cblas_order, cblas_uplo, cblas_transa,
                  m, k, alpha, ap, lda, beta, cp, ldc );
}

// Explicit template instantiations
template void ref_syrk<float>(char, char, char, gtint_t, gtint_t, float,
                      float*, gtint_t, float, float*, gtint_t );
template void ref_syrk<double>(char, char, char, gtint_t, gtint_t, double,
                      double*, gtint_t, double, double*, gtint_t );
template void ref_syrk<scomplex>(char, char, char, gtint_t, gtint_t, scomplex,
                      scomplex*, gtint_t, scomplex, scomplex*, gtint_t );
template void ref_syrk<dcomplex>(char, char, char, gtint_t, gtint_t, dcomplex,
                      dcomplex*, gtint_t, dcomplex, dcomplex*, gtint_t );

} //end of namespace testinghelpers