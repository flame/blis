/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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
    char_to_cblas_order( storage, &cblas_order );

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
        ref_cblas_ger = (Fptr_ref_cblas_ger)refCBLASModule.loadSymbol("cblas_sger");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_cblas_ger = (Fptr_ref_cblas_ger)refCBLASModule.loadSymbol("cblas_dger");
    }
    else if (typeid(T) == typeid(scomplex))
    {
      if( cfy )
        ref_cblas_ger = (Fptr_ref_cblas_ger)refCBLASModule.loadSymbol("cblas_cgerc");
       else
        ref_cblas_ger = (Fptr_ref_cblas_ger)refCBLASModule.loadSymbol("cblas_cgeru");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
      if( cfy )
        ref_cblas_ger = (Fptr_ref_cblas_ger)refCBLASModule.loadSymbol("cblas_zgerc");
       else
        ref_cblas_ger = (Fptr_ref_cblas_ger)refCBLASModule.loadSymbol("cblas_zgeru");
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
