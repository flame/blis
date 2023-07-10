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
    enum CBLAS_UPLO cblas_uploa;

    char_to_cblas_order( storage, &cblas_order );
    char_to_cblas_uplo( uploa, &cblas_uploa );

    typedef void (*Fptr_ref_cblas_her2)( const CBLAS_ORDER, const CBLAS_UPLO, const f77_int,
                         const T*, const T*, f77_int, const T*, f77_int, T*, f77_int);

    Fptr_ref_cblas_her2 ref_cblas_her2;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(scomplex))
    {
        ref_cblas_her2 = (Fptr_ref_cblas_her2)refCBLASModule.loadSymbol("cblas_cher2");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
        ref_cblas_her2 = (Fptr_ref_cblas_her2)refCBLASModule.loadSymbol("cblas_zher2");
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
