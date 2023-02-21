#include "blis.h"
#include <dlfcn.h>
#include "level3/ref_syr2k.h"

namespace testinghelpers {

template <typename T>
void ref_syr2k(
    char storage, char uplo, char transa, char transb,
    gtint_t m, gtint_t k,
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

    using scalar_t = std::conditional_t<testinghelpers::type_info<T>::is_complex, T&, T>;
    typedef void (*Fptr_ref_cblas_syr2k)( const CBLAS_ORDER, const CBLAS_UPLO, const CBLAS_TRANSPOSE,
                    const f77_int, const f77_int, const scalar_t, const T*, f77_int,
                    const T*, f77_int, const scalar_t, T*, f77_int);
    Fptr_ref_cblas_syr2k ref_cblas_syr2k;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_cblas_syr2k = (Fptr_ref_cblas_syr2k)dlsym(refCBLASModule.get( ), "cblas_ssyr2k");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_cblas_syr2k = (Fptr_ref_cblas_syr2k)dlsym(refCBLASModule.get(), "cblas_dsyr2k");
    }
    else if (typeid(T) == typeid(scomplex))
    {
        ref_cblas_syr2k = (Fptr_ref_cblas_syr2k)dlsym(refCBLASModule.get(), "cblas_csyr2k");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
        ref_cblas_syr2k = (Fptr_ref_cblas_syr2k)dlsym(refCBLASModule.get(), "cblas_zsyr2k");
    }
    else
    {
        throw std::runtime_error("Error in ref_syr2k.cpp: Invalid typename is passed function template.");
    }
    if( !ref_cblas_syr2k ) {
        throw std::runtime_error("Error in ref_syr2k.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_cblas_syr2k( cblas_order, cblas_uplo, cblas_transa,
                  m, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
}

// Explicit template instantiations
template void ref_syr2k<float>(char, char, char, char, gtint_t, gtint_t, float,
                      float*, gtint_t, float*, gtint_t, float, float*, gtint_t );
template void ref_syr2k<double>(char, char, char, char, gtint_t, gtint_t, double,
                      double*, gtint_t, double*, gtint_t, double, double*, gtint_t );
template void ref_syr2k<scomplex>(char, char, char, char, gtint_t, gtint_t, scomplex,
                      scomplex*, gtint_t, scomplex*, gtint_t, scomplex, scomplex*, gtint_t );
template void ref_syr2k<dcomplex>(char, char, char, char, gtint_t, gtint_t, dcomplex,
                      dcomplex*, gtint_t, dcomplex*, gtint_t, dcomplex, dcomplex*, gtint_t );
} //end of namespace testinghelpers