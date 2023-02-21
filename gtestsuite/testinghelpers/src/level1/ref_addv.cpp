#include "blis.h"
#include <dlfcn.h>
#include "level1/ref_addv.h"

namespace testinghelpers {

// Since addv is not a BLAS/CBLAS interface we use axpy as a reference.
template<typename T>
void ref_addv( char conj_x, gtint_t n, const T* x, gtint_t incx,
                                             T* y, gtint_t incy ) {
    using scalar_t = std::conditional_t<testinghelpers::type_info<T>::is_complex, T&, T>;
    typedef void (*Fptr_ref_cblas_axpy)( f77_int, scalar_t , const T *, f77_int , T *, f77_int );
    Fptr_ref_cblas_axpy ref_cblas_axpy;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_cblas_axpy = (Fptr_ref_cblas_axpy)dlsym(refCBLASModule.get( ), "cblas_saxpy");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_cblas_axpy = (Fptr_ref_cblas_axpy)dlsym(refCBLASModule.get(), "cblas_daxpy");
    }
    else if (typeid(T) == typeid(scomplex))
    {
        ref_cblas_axpy = (Fptr_ref_cblas_axpy)dlsym(refCBLASModule.get(), "cblas_caxpy");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
        ref_cblas_axpy = (Fptr_ref_cblas_axpy)dlsym(refCBLASModule.get(), "cblas_zaxpy");
    }
    else
    {
        throw std::runtime_error("Error in ref_addv.cpp: Invalid typename is passed function template.");
    }
    if (!ref_cblas_axpy) {
        throw std::runtime_error("Error in ref_addv.cpp: Function pointer == 0 -- symbol not found.");
    }

    T one;
    testinghelpers::initone(one);
    // Since conjx is not an option in BLAS/CBLAS,
    // we create a temporary xc which holds conj(x).
    if( chkconj( conj_x ) )
    {
        std::vector<T> X( testinghelpers::buff_dim(n, incx) );
        memcpy( X.data(), x, testinghelpers::buff_dim(n, incx)*sizeof(T) );
        testinghelpers::conj<T>( X.data(), n, incx );
        ref_cblas_axpy( n, one, X.data(), incx, y, incy );
    }
    else
    {
        ref_cblas_axpy( n, one, x, incx, y, incy );
    }
}


// Explicit template instantiations
template void ref_addv<float>(char, gtint_t, const float*, gtint_t, float*, gtint_t);
template void ref_addv<double>(char, gtint_t, const double*, gtint_t, double*, gtint_t);
template void ref_addv<scomplex>(char, gtint_t, const scomplex*, gtint_t, scomplex*, gtint_t);
template void ref_addv<dcomplex>(char, gtint_t, const dcomplex*, gtint_t, dcomplex*, gtint_t);

} //end of namespace testinghelpers