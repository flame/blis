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
#include "level2/ref_syr.h"

/*
 * ==========================================================================
 * SYR performs the symmetric rank 1 operation
 *    A := alpha*x*x**T + A,
 *  where alpha is a real scalar, x is an n element vector and A is an
 *  n by n symmetric matrix.
 * ==========================================================================
**/

namespace testinghelpers {

template <typename T>
void ref_syr( char storage, char uploa, char conjx, gtint_t n, T alpha,
                             T *xp, gtint_t incx, T *ap, gtint_t lda )
{
    enum CBLAS_ORDER cblas_order;
    enum CBLAS_UPLO cblas_uploa;

    char_to_cblas_order( storage, &cblas_order );
    char_to_cblas_uplo( uploa, &cblas_uploa );

    typedef void (*Fptr_ref_cblas_syr)( const CBLAS_ORDER, const CBLAS_UPLO, const f77_int,
                                        const T, const T*, f77_int, T*, f77_int);

    Fptr_ref_cblas_syr ref_cblas_syr;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_cblas_syr = (Fptr_ref_cblas_syr)dlsym(refCBLASModule.get(), "cblas_ssyr");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_cblas_syr = (Fptr_ref_cblas_syr)dlsym(refCBLASModule.get(), "cblas_dsyr");
    }
    else
    {
      throw std::runtime_error("Error in ref_syr.cpp: Invalid typename is passed function template.");
    }
    if (!ref_cblas_syr) {
        throw std::runtime_error("Error in ref_syr.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_cblas_syr( cblas_order, cblas_uploa, n, alpha, xp, incx, ap, lda );

}

// Explicit template instantiations
template void ref_syr<float>( char , char , char, gtint_t , float ,
                               float *, gtint_t , float *, gtint_t );
template void ref_syr<double>( char , char , char, gtint_t , double ,
                               double *, gtint_t , double *, gtint_t );

} //end of namespace testinghelpers
