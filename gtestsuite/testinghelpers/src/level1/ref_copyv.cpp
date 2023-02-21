#include "blis.h"
#include <dlfcn.h>
#include "level1/ref_copyv.h"

namespace testinghelpers {

template<typename T>
void ref_copyv( char conj_x, gtint_t n, const T* xp, gtint_t incx,
                                              T* yp, gtint_t incy ) {

    typedef void (*Fptr_ref_cblas_copyv)(f77_int, const T*, f77_int, T*, f77_int);
    Fptr_ref_cblas_copyv ref_cblas_copyv;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_cblas_copyv = (Fptr_ref_cblas_copyv)dlsym(refCBLASModule.get(), "cblas_scopy");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_cblas_copyv = (Fptr_ref_cblas_copyv)dlsym(refCBLASModule.get(), "cblas_dcopy");
    }
    else if (typeid(T) == typeid(scomplex))
    {
        ref_cblas_copyv = (Fptr_ref_cblas_copyv)dlsym(refCBLASModule.get(), "cblas_ccopy");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
        ref_cblas_copyv = (Fptr_ref_cblas_copyv)dlsym(refCBLASModule.get(), "cblas_zcopy");
    }
    else
    {
        throw std::runtime_error("Error in ref_copyv.cpp: Invalid typename is passed function template.");
    }
    if (!ref_cblas_copyv) {
        throw std::runtime_error("Error in ref_copyv.cpp: Function pointer == 0 -- symbol not found.");
    }

    // Since conjx is not an option in BLAS/CBLAS,
    // we create a temporary xc which holds conj(x).
    if( chkconj( conj_x ) )
    {
        std::vector<T> X( testinghelpers::buff_dim(n, incx) );
        memcpy( X.data(), xp, testinghelpers::buff_dim(n, incx)*sizeof(T) );
        testinghelpers::conj<T>( X.data(), n, incx );
        ref_cblas_copyv( n, X.data(), incx, yp, incy );
    }
    else
    {
        ref_cblas_copyv( n, xp, incx, yp, incy );
    }
}

// Explicit template instantiations
template void ref_copyv<float>(char, gtint_t, const float*, gtint_t, float*, gtint_t);
template void ref_copyv<double>(char, gtint_t, const double*, gtint_t, double*, gtint_t);
template void ref_copyv<scomplex>(char, gtint_t, const scomplex*, gtint_t, scomplex*, gtint_t);
template void ref_copyv<dcomplex>(char, gtint_t, const dcomplex*, gtint_t, dcomplex*, gtint_t);

} //end of namespace testinghelpers