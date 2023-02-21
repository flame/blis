#include "blis.h"
#include <dlfcn.h>
#include "level2/ref_hemv.h"

/*
 * ==========================================================================
 * HEMV performs the matrix-vector  operation
 *    y := alpha*A*x + beta*y
 * where alpha and beta are scalars, x and y are n element vectors and
 * A is an n by n hermitian matrix.
 * ==========================================================================
**/

namespace testinghelpers {

template <typename T>
void ref_hemv( char storage, char uploa, char conja, char conjx, gtint_t n,
    T* alpha, T *ap, gtint_t lda, T *xp, gtint_t incx, T* beta,
    T *yp, gtint_t incy )
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

    typedef void (*Fptr_ref_cblas_hemv)( const CBLAS_ORDER, const CBLAS_UPLO, const f77_int,
                         const T*, const T*, f77_int, const T*, f77_int, const T*, T*, f77_int);

    Fptr_ref_cblas_hemv ref_cblas_hemv;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(scomplex))
    {
        ref_cblas_hemv = (Fptr_ref_cblas_hemv)dlsym(refCBLASModule.get(), "cblas_chemv");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
        ref_cblas_hemv = (Fptr_ref_cblas_hemv)dlsym(refCBLASModule.get(), "cblas_zhemv");
    }
    else
    {
      throw std::runtime_error("Error in ref_hemv.cpp: Invalid typename is passed function template.");
    }
    if (!ref_cblas_hemv) {
        throw std::runtime_error("Error in ref_hemv.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_cblas_hemv( cblas_order, cblas_uploa, n, alpha, ap, lda, xp, incx, beta, yp, incy );

}

// Explicit template instantiations
template void ref_hemv<scomplex>( char, char, char, char, gtint_t, scomplex *,
              scomplex *, gtint_t, scomplex *, gtint_t, scomplex *, scomplex *, gtint_t );
template void ref_hemv<dcomplex>( char, char, char, char, gtint_t, dcomplex *,
              dcomplex *, gtint_t, dcomplex *, gtint_t, dcomplex *, dcomplex *, gtint_t );

} //end of namespace testinghelpers