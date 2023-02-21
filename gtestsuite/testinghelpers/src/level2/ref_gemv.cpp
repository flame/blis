#include "blis.h"
#include <dlfcn.h>
#include "level2/ref_gemv.h"

/*
 * ==========================================================================
 * GEMV performs one of the matrix-vector operations
 *    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,   or
 *    y := alpha*A**H*x + beta*y,
 * ==========================================================================
*/

namespace testinghelpers {

template <typename T>
void ref_gemv( char storage, char trans, char conjx, gtint_t m, gtint_t n, T alpha,
    T *ap, gtint_t lda, T *xp, gtint_t incx, T beta, T *yp, gtint_t incy )
{
    gtint_t lenx = chknotrans( trans ) ? n : m ;
    bool cfx = chkconj( conjx );

    enum CBLAS_ORDER cblas_order;
    if( (storage == 'c') || (storage == 'C') )
        cblas_order = CblasColMajor;
    else
        cblas_order = CblasRowMajor;

    enum CBLAS_TRANSPOSE cblas_trans;
    if( trans == 't' )
        cblas_trans = CblasTrans;
    else if( trans == 'c' )
        cblas_trans = CblasConjTrans;
    else
        cblas_trans = CblasNoTrans;

    if( trans == 'h' ) {
        conj<T>(storage, ap, m, n, lda );
    }

    if( cfx ) {
        conj<T>( xp, lenx, incx );
    }

    using scalar_t = std::conditional_t<testinghelpers::type_info<T>::is_complex, T&, T>;
    typedef void (*Fptr_ref_cblas_gemv)( const CBLAS_ORDER, const CBLAS_TRANSPOSE,
                    const f77_int, const f77_int, const scalar_t, const T*, f77_int,
                    const T*, f77_int, const scalar_t, T*, f77_int);
    Fptr_ref_cblas_gemv ref_cblas_gemv;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_cblas_gemv = (Fptr_ref_cblas_gemv)dlsym(refCBLASModule.get( ), "cblas_sgemv");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_cblas_gemv = (Fptr_ref_cblas_gemv)dlsym(refCBLASModule.get(), "cblas_dgemv");
    }
    else if (typeid(T) == typeid(scomplex))
    {
        ref_cblas_gemv = (Fptr_ref_cblas_gemv)dlsym(refCBLASModule.get(), "cblas_cgemv");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
        ref_cblas_gemv = (Fptr_ref_cblas_gemv)dlsym(refCBLASModule.get(), "cblas_zgemv");
    }
    else
    {
      throw std::runtime_error("Error in ref_gemv.cpp: Invalid typename is passed function template.");
    }
    if (!ref_cblas_gemv) {
        throw std::runtime_error("Error in ref_gemv.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_cblas_gemv( cblas_order, cblas_trans, m, n, alpha, ap, lda,
                                                    xp, incx, beta, yp, incy );

}

// Explicit template instantiations
template void ref_gemv<float>( char , char , char , gtint_t , gtint_t , float ,
             float *, gtint_t , float *, gtint_t , float , float *, gtint_t );
template void ref_gemv<double>( char , char , char , gtint_t , gtint_t , double ,
             double *, gtint_t , double *, gtint_t , double , double *, gtint_t );
template void ref_gemv<scomplex>( char , char , char , gtint_t , gtint_t , scomplex ,
             scomplex *, gtint_t , scomplex *, gtint_t , scomplex , scomplex *, gtint_t );
template void ref_gemv<dcomplex>( char , char , char , gtint_t , gtint_t , dcomplex ,
             dcomplex *, gtint_t , dcomplex *, gtint_t , dcomplex , dcomplex *, gtint_t );

} //end of namespace testinghelpers