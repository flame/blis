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
#include "level1/ref_dotv.h"

namespace testinghelpers {

template<typename T>
void ref_dotv(gtint_t len, const T* xp,
              gtint_t incx, const T* yp, gtint_t incy, T* rho) {

    typedef T (*Fptr_ref_cblas_dot)(f77_int, const T*, f77_int, const T*, f77_int );
    Fptr_ref_cblas_dot ref_cblas_dot;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_cblas_dot = (Fptr_ref_cblas_dot)dlsym(refCBLASModule.get(), "cblas_sdot");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_cblas_dot = (Fptr_ref_cblas_dot)dlsym(refCBLASModule.get(), "cblas_ddot");
    }
    else
    {
        throw std::runtime_error("Error in ref_dot.cpp: Invalid typename is passed function template.");
    }
    if ( !ref_cblas_dot ) {
        throw std::runtime_error("Error in ref_dot.cpp: Function pointer == 0 -- symbol not found.");
    }

    *rho = ref_cblas_dot( len, xp, incx, yp, incy );

}

template<typename T>
void ref_dotv( char conj_x, char conj_y, gtint_t len, const T* xp, gtint_t incx,
                                             const T* yp, gtint_t incy, T* rho ) {

    typedef void (*Fptr_ref_cblas_dot)(f77_int, const T*, f77_int, const T*, f77_int, T* );
    Fptr_ref_cblas_dot ref_cblas_dot;

    bool  cfx = chkconj( conj_x );
    bool  cfy = chkconj( conj_y );
    gtint_t svx = buff_dim(len, incx);
    gtint_t svy = buff_dim(len, incy);

    std::vector<T> X( svx );
    memcpy(X.data(), xp, svx*sizeof(T));

    std::vector<T> Y( svy );
    memcpy(Y.data(), yp, svy*sizeof(T));

    if( cfx ) {
        conj<T>( X.data(), len, incx );
    }

    if( cfy ) {
        conj<T>( Y.data(), len, incy );
    }

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(scomplex))
    {
        ref_cblas_dot = (Fptr_ref_cblas_dot)dlsym(refCBLASModule.get(), "cblas_cdotu_sub");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
        ref_cblas_dot = (Fptr_ref_cblas_dot)dlsym(refCBLASModule.get(), "cblas_zdotu_sub");
    }
    else
    {
        throw std::runtime_error("Error in ref_dot.cpp: Invalid typename is passed function template.");
    }
    if (!ref_cblas_dot) {
        throw std::runtime_error("Error in ref_dot.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_cblas_dot( len, X.data(), incx, Y.data(), incy, rho );

}

// Explicit template instantiations
template void ref_dotv<float>( gtint_t, const float*, gtint_t, const float*, gtint_t, float* );
template void ref_dotv<double>( gtint_t, const double*, gtint_t, const double*, gtint_t,double* );
template void ref_dotv<scomplex>(char, char, gtint_t, const scomplex*, gtint_t, const scomplex*, gtint_t, scomplex*);
template void ref_dotv<dcomplex>(char, char, gtint_t, const dcomplex*, gtint_t, const dcomplex*, gtint_t, dcomplex*);

} //end of namespace testinghelpers
