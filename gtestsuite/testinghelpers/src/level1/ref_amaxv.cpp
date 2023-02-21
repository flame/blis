#include "blis.h"
#include <dlfcn.h>
#include "level1/ref_amaxv.h"

namespace testinghelpers {

// Since amaxv is not a BLAS/CBLAS interface we use axpy as a reference.
template<typename T>
gtint_t ref_amaxv( gtint_t n, const T* x, gtint_t incx ) {
    gtint_t idx;
    typedef gtint_t (*Fptr_ref_cblas_amaxv)( f77_int, const T *, f77_int );
    Fptr_ref_cblas_amaxv ref_cblas_amaxv;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_cblas_amaxv = (Fptr_ref_cblas_amaxv)dlsym(refCBLASModule.get( ), "cblas_isamax");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_cblas_amaxv = (Fptr_ref_cblas_amaxv)dlsym(refCBLASModule.get(), "cblas_idamax");
    }
    else if (typeid(T) == typeid(scomplex))
    {
        ref_cblas_amaxv = (Fptr_ref_cblas_amaxv)dlsym(refCBLASModule.get(), "cblas_icamax");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
        ref_cblas_amaxv = (Fptr_ref_cblas_amaxv)dlsym(refCBLASModule.get(), "cblas_izamax");
    }
    else
    {
        throw std::runtime_error("Error in ref_amaxv.cpp: Invalid typename is passed function template.");
    }
    if (!ref_cblas_amaxv) {
        throw std::runtime_error("Error in ref_amaxv.cpp: Function pointer == 0 -- symbol not found.");
    }

    idx = ref_cblas_amaxv( n, x, incx );
    return idx;
}


// Explicit template instantiations
template gtint_t ref_amaxv<float>(gtint_t, const float*, gtint_t);
template gtint_t ref_amaxv<double>(gtint_t, const double*, gtint_t);
template gtint_t ref_amaxv<scomplex>(gtint_t, const scomplex*, gtint_t);
template gtint_t ref_amaxv<dcomplex>(gtint_t, const dcomplex*, gtint_t);

} //end of namespace testinghelpers