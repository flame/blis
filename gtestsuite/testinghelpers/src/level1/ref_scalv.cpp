#include "blis.h"
#include <dlfcn.h>
#include "level1/ref_scalv.h"

namespace testinghelpers {

template<typename T>
void ref_scalv(char conjalpha, gtint_t n, T alpha, T* x, gtint_t incx)
{
    using scalar_t = std::conditional_t<testinghelpers::type_info<T>::is_complex, T&, T>;
    typedef void (*Fptr_ref_cblas_scal)( f77_int, scalar_t , T *, f77_int);
    Fptr_ref_cblas_scal ref_cblas_scal;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_cblas_scal = (Fptr_ref_cblas_scal)dlsym(refCBLASModule.get(), "cblas_sscal");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_cblas_scal = (Fptr_ref_cblas_scal)dlsym(refCBLASModule.get(), "cblas_dscal");
    }
    else if (typeid(T) == typeid(scomplex))
    {
        ref_cblas_scal = (Fptr_ref_cblas_scal)dlsym(refCBLASModule.get(), "cblas_cscal");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
        ref_cblas_scal = (Fptr_ref_cblas_scal)dlsym(refCBLASModule.get(), "cblas_zscal");
    }
    else
    {
        throw std::runtime_error("Error in ref_scalv.cpp: Invalid typename is passed function template.");
    }
    if (!ref_cblas_scal) {
        throw std::runtime_error("Error in ref_scalv.cpp: Function pointer == 0 -- symbol not found.");
    }

#ifdef TEST_BLIS_TYPED
    if( chkconj( conjalpha ) )
    {
        T alpha_conj = testinghelpers::conj<T>( alpha );
        ref_cblas_scal( n, alpha_conj, x, incx );
    }
    else
#endif
    {
        ref_cblas_scal( n, alpha, x, incx );
    }

}

// Explicit template instantiations
template void ref_scalv<float>(char, gtint_t, float, float*, gtint_t);
template void ref_scalv<double>(char, gtint_t, double, double*, gtint_t);
template void ref_scalv<scomplex>(char, gtint_t, scomplex, scomplex*, gtint_t);
template void ref_scalv<dcomplex>(char, gtint_t, dcomplex, dcomplex*, gtint_t);

} //end of namespace testinghelpers