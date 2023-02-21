#include "blis.h"
#include <dlfcn.h>
#include "level3/ref_gemm.h"

/*
 * ==========================================================================
 * GEMM  performs one of the matrix-matrix operations
 *    C := alpha*op( A )*op( B ) + beta*C,
 * where  op( A ) is one of
 *    op( A ) = A   or   op( A ) = A**T   or   op( A ) = A**H,
 * alpha and beta are scalars, and A, B and C are matrices, with op( A )
 * an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
 * ==========================================================================
 */

namespace testinghelpers {

template <typename T>
void ref_gemm(char storage, char trnsa, char trnsb, gtint_t m, gtint_t n, gtint_t k,
    T alpha, T* ap, gtint_t lda, T* bp, gtint_t ldb, T beta,  T* cp, gtint_t ldc)
{
    enum CBLAS_ORDER cblas_order;
    if( (storage == 'c') || (storage == 'C') )
        cblas_order = CblasColMajor;
    else
        cblas_order = CblasRowMajor;

    enum CBLAS_TRANSPOSE cblas_transa;
    if( trnsa == 't' )
        cblas_transa = CblasTrans;
    else if( trnsa == 'c' )
        cblas_transa = CblasConjTrans;
    else
        cblas_transa = CblasNoTrans;

    enum CBLAS_TRANSPOSE cblas_transb;
    if( trnsb == 't' )
        cblas_transb = CblasTrans;
    else if( trnsb == 'c' )
        cblas_transb = CblasConjTrans;
    else
        cblas_transb = CblasNoTrans;

    if( trnsa == 'h' ) {
        throw std::invalid_argument("Error in file src/level3/ref_gemm.cpp:"
                    "Invalid input. To enable for 'h' update the code and create a temporary matrix A.");
        //testinghelpers::conj<T>( storage, A.data(), m, k, lda );
    }

    if( trnsb == 'h' ) {
        throw std::invalid_argument("Error in file src/level3/ref_gemm.cpp:"
                    "Invalid input. To enable for 'h' update the code and create a temporary matrix B.");
        //testinghelpers::conj<T>( storage, B.data(), k, n, ldb );
    }

    using scalar_t = std::conditional_t<testinghelpers::type_info<T>::is_complex, T&, T>;
    typedef void (*Fptr_ref_cblas_gemm)( const CBLAS_ORDER, const CBLAS_TRANSPOSE, const CBLAS_TRANSPOSE,
                    const f77_int, const f77_int, const f77_int, const scalar_t, const T*, f77_int,
                    const T*, f77_int, const scalar_t, T*, f77_int);
    Fptr_ref_cblas_gemm ref_cblas_gemm;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_cblas_gemm = (Fptr_ref_cblas_gemm)dlsym(refCBLASModule.get( ), "cblas_sgemm");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_cblas_gemm = (Fptr_ref_cblas_gemm)dlsym(refCBLASModule.get(), "cblas_dgemm");
    }
    else if (typeid(T) == typeid(scomplex))
    {
        ref_cblas_gemm = (Fptr_ref_cblas_gemm)dlsym(refCBLASModule.get(), "cblas_cgemm");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
        ref_cblas_gemm = (Fptr_ref_cblas_gemm)dlsym(refCBLASModule.get(), "cblas_zgemm");
    }
    else
    {
        throw std::runtime_error("Error in ref_gemm.cpp: Invalid typename is passed function template.");
    }
    if( !ref_cblas_gemm ) {
        throw std::runtime_error("Error in ref_gemm.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_cblas_gemm( cblas_order, cblas_transa, cblas_transb,
                  m, n, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
}

// Explicit template instantiations
template void ref_gemm<float>(char, char, char, gtint_t, gtint_t, gtint_t, float,
                      float*, gtint_t, float*, gtint_t, float, float*, gtint_t );
template void ref_gemm<double>(char, char, char, gtint_t, gtint_t, gtint_t, double,
                      double*, gtint_t, double*, gtint_t, double, double*, gtint_t );
template void ref_gemm<scomplex>(char, char, char, gtint_t, gtint_t, gtint_t, scomplex,
                      scomplex*, gtint_t, scomplex*, gtint_t, scomplex, scomplex*, gtint_t );
template void ref_gemm<dcomplex>(char, char, char, gtint_t, gtint_t, gtint_t, dcomplex,
                      dcomplex*, gtint_t, dcomplex*, gtint_t, dcomplex, dcomplex*, gtint_t );

} //end of namespace testinghelpers