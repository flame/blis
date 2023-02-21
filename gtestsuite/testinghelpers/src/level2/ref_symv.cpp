#include "blis.h"
#include <dlfcn.h>
#include "level2/ref_symv.h"

/*
 * ==========================================================================
 * SYMV performs the matrix-vector  operation
 *    y := alpha*A*x + beta*y
 * where alpha and beta are scalars, x and y are n element vectors and
 * A is an n by n symmetric matrix.
 * ==========================================================================
**/

namespace testinghelpers {

template <typename T>
void ref_symv( char storage, char uploa, char conja, char conjx, gtint_t n,
    T *alpha, T *ap, gtint_t lda, T *xp, gtint_t incx, T *beta,
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

    typedef void (*Fptr_ref_cblas_symv)( const CBLAS_ORDER, const CBLAS_UPLO, const f77_int,
                         const T, const T*, f77_int, const T*, f77_int, const T, T*, f77_int);

    Fptr_ref_cblas_symv ref_cblas_symv;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_cblas_symv = (Fptr_ref_cblas_symv)dlsym(refCBLASModule.get(), "cblas_ssymv");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_cblas_symv = (Fptr_ref_cblas_symv)dlsym(refCBLASModule.get(), "cblas_dsymv");
    }
    else
    {
      throw std::runtime_error("Error in ref_symv.cpp: Invalid typename is passed function template.");
    }
    if (!ref_cblas_symv) {
        throw std::runtime_error("Error in ref_symv.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_cblas_symv( cblas_order, cblas_uploa, n, *alpha, ap, lda, xp, incx, *beta, yp, incy );
}

// Explicit template instantiations
template void ref_symv<float>( char, char, char, char, gtint_t, float *,
              float *, gtint_t, float *, gtint_t, float *, float *, gtint_t );
template void ref_symv<double>( char, char, char, char, gtint_t, double *,
              double *, gtint_t, double *, gtint_t, double *, double *, gtint_t );

} //end of namespace testinghelpers