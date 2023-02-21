#include "blis.h"
#include <dlfcn.h>
#include "level1/ref_axpyv.h"

namespace testinghelpers {

template<typename T>
void ref_axpyv( char conj_x, gtint_t n, T alpha,
                        const T* x, gtint_t incx, T* y, gtint_t incy ) {

    using scalar_t = std::conditional_t<testinghelpers::type_info<T>::is_complex, T&, T>;
    typedef void (*Fptr_ref_cblas_axpy)( f77_int, scalar_t , const T *, f77_int , T *, f77_int );
    Fptr_ref_cblas_axpy ref_cblas_axpy;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_cblas_axpy = (Fptr_ref_cblas_axpy)dlsym(refCBLASModule.get(), "cblas_saxpy");
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
        throw std::runtime_error("Error in ref_axpy.cpp: Invalid typename is passed function template.");
    }
    if (!ref_cblas_axpy) {
        throw std::runtime_error("Error in ref_axpy.cpp: Function pointer == 0 -- symbol not found.");
    }
#if TEST_BLIS_TYPED
    if( chkconj( conj_x ) )
    {
        std::vector<T> X( testinghelpers::buff_dim(n, incx) );
        memcpy( X.data(), x, testinghelpers::buff_dim(n, incx)*sizeof(T) );
        testinghelpers::conj<T>( X.data(), n, incx );
        ref_cblas_axpy( n, alpha, X.data(), incx, y, incy );
    }
    else
#endif
    {
        ref_cblas_axpy( n, alpha, x, incx, y, incy );
    }
}


// Explicit template instantiations
template void ref_axpyv<float>(char, gtint_t, float, const float*, gtint_t, float*, gtint_t);
template void ref_axpyv<double>(char, gtint_t, double, const double*, gtint_t, double*, gtint_t);
template void ref_axpyv<scomplex>(char, gtint_t, scomplex, const scomplex*, gtint_t, scomplex*, gtint_t);
template void ref_axpyv<dcomplex>(char, gtint_t, dcomplex, const dcomplex*, gtint_t, dcomplex*, gtint_t);

} //end of namespace testinghelpers
