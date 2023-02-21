#include "blis.h"
#include <dlfcn.h>
#include "level2/ref_her2.h"

/*
 * ==========================================================================
 * HER2  performs the hermitian rank 2 operation
 *    A := alpha*x*y**H + conjg( alpha )*y*x**H + A,
 * where alpha is a scalar, x and y are n element vectors and A is an n
 * by n hermitian matrix.
 * ==========================================================================
**/

namespace testinghelpers {

template <typename T>
void ref_her2( char storage, char uploa, char conjx, char conjy, gtint_t n,
    T* alpha, T *xp, gtint_t incx, T *yp, gtint_t incy, T *ap, gtint_t lda )
{
    enum CBLAS_ORDER cblas_order;
    if( (storage == 'c') || (storage == 'C') )
        cblas_order = CblasColMajor;
    else
        cblas_order = CblasRowMajor;

    enum CBLAS_UPLO cblas_uploa;
    if( (uploa == 'u') || (uploa == 'U') )
        cblas_uploa = CblasUpper;
    else
        cblas_uploa = CblasLower;

    typedef void (*Fptr_ref_cblas_her2)( const CBLAS_ORDER, const CBLAS_UPLO, const f77_int,
                         const T*, const T*, f77_int, const T*, f77_int, T*, f77_int);

    Fptr_ref_cblas_her2 ref_cblas_her2;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(scomplex))
    {
        ref_cblas_her2 = (Fptr_ref_cblas_her2)dlsym(refCBLASModule.get(), "cblas_cher2");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
        ref_cblas_her2 = (Fptr_ref_cblas_her2)dlsym(refCBLASModule.get(), "cblas_zher2");
    }
    else
    {
      throw std::runtime_error("Error in ref_her2.cpp: Invalid typename is passed function template.");
    }
    if (!ref_cblas_her2) {
        throw std::runtime_error("Error in ref_her2.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_cblas_her2( cblas_order, cblas_uploa, n, alpha, xp, incx, yp, incy, ap, lda );

}

// Explicit template instantiations
template void ref_her2<scomplex>( char, char, char, char, gtint_t, scomplex *,
              scomplex *, gtint_t, scomplex *, gtint_t, scomplex *, gtint_t );
template void ref_her2<dcomplex>( char, char, char, char, gtint_t, dcomplex *,
              dcomplex *, gtint_t, dcomplex *, gtint_t, dcomplex *, gtint_t );

} //end of namespace testinghelpers