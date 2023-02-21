#include "blis.h"
#include <dlfcn.h>
#include "level2/ref_syr2.h"

/*
 * ==========================================================================
 * SYR2  performs the symmetric rank 2 operation
 *    A := alpha*x*y**T + alpha*y*x**T + A,
 * where alpha is a scalar, x and y are n element vectors and A is an n
 * by n symmetric matrix.
 * ==========================================================================
**/

namespace testinghelpers {

template <typename T>
void ref_syr2( char storage, char uploa, char conjx, char conjy, gtint_t n,
   T alpha, T *xp, gtint_t incx, T *yp, gtint_t incy, T *ap, gtint_t lda )
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

    typedef void (*Fptr_ref_cblas_syr2)( const CBLAS_ORDER, const CBLAS_UPLO, const f77_int,
                const T, const T*, f77_int, const T*, f77_int, T*, f77_int);

    Fptr_ref_cblas_syr2 ref_cblas_syr2;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_cblas_syr2 = (Fptr_ref_cblas_syr2)dlsym(refCBLASModule.get(), "cblas_ssyr2");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_cblas_syr2 = (Fptr_ref_cblas_syr2)dlsym(refCBLASModule.get(), "cblas_dsyr2");
    }
    else
    {
      throw std::runtime_error("Error in ref_syr2.cpp: Invalid typename is passed function template.");
    }
    if (!ref_cblas_syr2) {
        throw std::runtime_error("Error in ref_syr2.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_cblas_syr2( cblas_order, cblas_uploa, n, alpha, xp, incx, yp, incy, ap, lda );

}

// Explicit template instantiations
template void ref_syr2<float>( char, char, char, char, gtint_t, float,
              float *, gtint_t, float *, gtint_t, float *, gtint_t );
template void ref_syr2<double>( char, char, char, char, gtint_t, double,
              double *, gtint_t, double *, gtint_t, double *, gtint_t );

} //end of namespace testinghelpers