/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
	- Redistributions of source code must retain the above copyright
	  notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright
	  notice, this list of conditions and the following disclaimer in the
	  documentation and/or other materials provided with the distribution.
	- Neither the name(s) of the copyright holder(s) nor the names of its
	  contributors may be used to endorse or promote products derived
	  from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"
#include <dlfcn.h>
#include "level3/ref_gemm.h"
#include "level3/ref_gemmt.h"

/*
 * ==========================================================================
 *  GEMMT performs one of the matrix-matrix operations
 *     C := alpha*op( A )*op( B ) + beta*C,
 *  where  op( X ) is one of
 *     op( X ) = X   or   op( X ) = A**T   or   op( X ) = X**H,
 *  alpha and beta are scalars, and A, B and C are matrices, with op( A )
 *  an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
 *  Only accesses and updates the upper or the lower triangular part.
 * ==========================================================================
**/

namespace testinghelpers {
#if 1
template <typename T>
void ref_gemmt (
    char storage, char uplo, char trnsa, char trnsb,
    gtint_t n, gtint_t k,
    T alpha,
    T* ap, gtint_t lda,
    T* bp, gtint_t ldb,
    T beta,
    T* cp, gtint_t ldc
) {
    gtint_t smc = testinghelpers::matsize( storage, 'n', n, n, ldc );
    std::vector<T> C( smc );
    memcpy(C.data(), cp, (smc*sizeof(T)));
    ref_gemm<T>(storage, trnsa, trnsb, n, n, k, alpha, ap, lda, bp, ldb, beta, C.data(), ldc);
    if( (storage=='c')||(storage=='C') )
    {
        for(gtint_t j=0; j<n; j++)
        {
            for(gtint_t i=0; i<n; i++)
            {
                if( (uplo=='u')||(uplo=='U') )
                {
                    if(i<=j) cp[i+j*ldc] = C[i+j*ldc];
                }
                else if ( (uplo=='l')||(uplo=='L') )
                {
                    if (i>=j) cp[i+j*ldc] = C[i+j*ldc];
                }
                else
                    throw std::runtime_error("Error in level3/ref_gemmt.cpp: side must be 'u' or 'l'.");
            }
        }
    } else
    {
        for(gtint_t i=0; i<n; i++)
        {
            for(gtint_t j=0; j<n; j++)
            {
                if( (uplo=='u')||(uplo=='U') )
                {
                    if(i<=j) cp[j+i*ldc] = C[j+i*ldc];
                }
                else if ( (uplo=='l')||(uplo=='L') )
                {
                    if (i>=j) cp[j+i*ldc] = C[j+i*ldc];
                }
                else
                    throw std::runtime_error("Error in level3/ref_gemmt.cpp: side must be 'u' or 'l'.");
            }
        }
    }
}
#else
template <typename T>
void ref_gemmt (
    char storage, char uplo, char trnsa, char trnsb,
    gtint_t n, gtint_t k,
    T alpha,
    T* ap, gtint_t lda,
    T* bp, gtint_t ldb,
    T beta,
    T* cp, gtint_t ldc
)
{
    enum CBLAS_ORDER cblas_order;
    enum CBLAS_TRANSPOSE cblas_transa;
    enum CBLAS_TRANSPOSE cblas_transb;
    enum CBLAS_UPLO cblas_uplo;

    char_to_cblas_order( storage, &cblas_order );
    char_to_cblas_trans( trnsa, &cblas_transa );
    char_to_cblas_trans( trnsb, &cblas_transb );
    char_to_cblas_uplo( uplo, &cblas_uplo );

    using scalar_t = std::conditional_t<testinghelpers::type_info<T>::is_complex, T&, T>;
    typedef void (*Fptr_ref_cblas_gemmt)( const CBLAS_ORDER, const CBLAS_UPLO, const CBLAS_TRANSPOSE, const CBLAS_TRANSPOSE,
                    const f77_int, const f77_int, const scalar_t, const T*, f77_int,
                    const T*, f77_int, const scalar_t, T*, f77_int);
    Fptr_ref_cblas_gemmt ref_cblas_gemmt;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_cblas_gemmt = (Fptr_ref_cblas_gemmt)dlsym(refCBLASModule.get( ), "cblas_sgemmt");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_cblas_gemmt = (Fptr_ref_cblas_gemmt)dlsym(refCBLASModule.get(), "cblas_dgemmt");
    }
    else if (typeid(T) == typeid(scomplex))
    {
        ref_cblas_gemmt = (Fptr_ref_cblas_gemmt)dlsym(refCBLASModule.get(), "cblas_cgemmt");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
        ref_cblas_gemmt = (Fptr_ref_cblas_gemmt)dlsym(refCBLASModule.get(), "cblas_zgemmt");
    }
    else
    {
        throw std::runtime_error("Error in ref_gemmt.cpp: Invalid typename is passed function template.");
    }
    if( !ref_cblas_gemmt ) {
        throw std::runtime_error("Error in ref_gemmt.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_cblas_gemmt( cblas_order, cblas_uplo, cblas_transa, cblas_transb,
                              n, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
}
#endif
// Explicit template instantiations
template void ref_gemmt<float>(char, char, char, char, gtint_t, gtint_t, float,
                      float*, gtint_t, float*, gtint_t, float, float*, gtint_t );
template void ref_gemmt<double>(char, char, char, char, gtint_t, gtint_t, double,
                      double*, gtint_t, double*, gtint_t, double, double*, gtint_t );
template void ref_gemmt<scomplex>(char, char, char, char, gtint_t, gtint_t, scomplex,
                      scomplex*, gtint_t, scomplex*, gtint_t, scomplex, scomplex*, gtint_t );
template void ref_gemmt<dcomplex>(char, char, char, char, gtint_t, gtint_t, dcomplex,
                      dcomplex*, gtint_t, dcomplex*, gtint_t, dcomplex, dcomplex*, gtint_t );


} //end of namespace testinghelpers
