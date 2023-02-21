#include "blis.h"
#include <dlfcn.h>
#include "level2/ref_ger.h"

/*
 * ==========================================================================
 * GER performs the rank 1 operation
 *    A := alpha*x*y**T + A,
 * where alpha is a scalar, x is an m element vector, y is an n element
 * vector and A is an m by n matrix.
 * ==========================================================================
*/

namespace testinghelpers {

template <typename T>
void ref_ger( char storage, char conjx, char conjy, gtint_t m, gtint_t n,
    T alpha, T *xp, gtint_t incx, T *yp, gtint_t incy, T *ap, gtint_t lda )
{
    bool cfy = chkconj( conjy );

    enum CBLAS_ORDER cblas_order;
    if( (storage == 'c') || (storage == 'C') )
        cblas_order = CblasColMajor;
    else
        cblas_order = CblasRowMajor;

    std::vector<T> X( buff_dim(m, incx) );
    memcpy(X.data(), xp, (buff_dim(m, incx)*sizeof(T)));

    using scalar_t = std::conditional_t<testinghelpers::type_info<T>::is_complex, T&, T>;
    typedef void (*Fptr_ref_cblas_ger)( const CBLAS_ORDER, const f77_int, const f77_int,
                     const scalar_t, const T*, f77_int,  const T*, f77_int, T*, f77_int );
    Fptr_ref_cblas_ger ref_cblas_ger;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_cblas_ger = (Fptr_ref_cblas_ger)dlsym(refCBLASModule.get( ), "cblas_sger");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_cblas_ger = (Fptr_ref_cblas_ger)dlsym(refCBLASModule.get(), "cblas_dger");
    }
    else if (typeid(T) == typeid(scomplex))
    {
      if( cfy )
        ref_cblas_ger = (Fptr_ref_cblas_ger)dlsym(refCBLASModule.get(), "cblas_cgerc");
       else
        ref_cblas_ger = (Fptr_ref_cblas_ger)dlsym(refCBLASModule.get(), "cblas_cgeru");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
      if( cfy )
        ref_cblas_ger = (Fptr_ref_cblas_ger)dlsym(refCBLASModule.get(), "cblas_zgerc");
       else
        ref_cblas_ger = (Fptr_ref_cblas_ger)dlsym(refCBLASModule.get(), "cblas_zgeru");
    }
    else
    {
      throw std::runtime_error("Error in ref_ger.cpp: Invalid typename is passed function template.");
    }
    if (!ref_cblas_ger) {
        throw std::runtime_error("Error in ref_ger.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_cblas_ger( cblas_order, m, n, alpha, xp, incx, yp, incy, ap, lda );

}

// Explicit template instantiations
template void ref_ger<float>( char, char, char, gtint_t, gtint_t,
              float, float *, gtint_t, float *, gtint_t, float *, gtint_t );
template void ref_ger<double>( char, char, char, gtint_t, gtint_t,
              double, double *, gtint_t, double *, gtint_t, double *, gtint_t );
template void ref_ger<scomplex>( char, char, char, gtint_t, gtint_t,
              scomplex, scomplex *, gtint_t, scomplex *, gtint_t, scomplex *, gtint_t );
template void ref_ger<dcomplex>( char, char, char, gtint_t, gtint_t,
              dcomplex, dcomplex *, gtint_t, dcomplex *, gtint_t, dcomplex *, gtint_t );

} //end of namespace testinghelpers