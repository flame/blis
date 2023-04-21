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
#include "level1/ref_dotxv.h"

namespace testinghelpers {

// Since dotxv is not supported by BLAS, we have a local reference implementation.
template<typename T>
void ref_dotxv( char conj_x, char conj_y, gtint_t len, const T Alpha,
    const T* xp, gtint_t incx, const T* yp, gtint_t incy, const T Beta,
    T* rhorig )
{
    gtint_t i, ix, iy;
    T ONE, ZERO;
    initone(ONE);
    initzero(ZERO);
    bool  cfx = chkconj( conj_x );
    bool  cfy = chkconj( conj_y );
    gtint_t svx = buff_dim(len, incx);
    gtint_t svy = buff_dim(len, incy);
    T rho   = *rhorig;

    if (len == 0) {
        *rhorig = rho;
        return;
    }

    rho = rho * Beta;

    std::vector<T> X( svx );
    memcpy(X.data(), xp, svx*sizeof(T));

    std::vector<T> Y( svy );
    memcpy(Y.data(), yp, svy*sizeof(T));

    if( cfx ) {
        conj<T>( X.data(), len, incx );
    }

    if (Alpha != ONE) {
        ix = 0;
        if (Alpha == ZERO) {
            for(i = 0 ; i < len ; i++) {
                X[ix] = ZERO;
                ix = ix + incx;
            }
        }
        else {
            for(i = 0 ; i < len ; i++) {
                X[ix] = Alpha * X[ix];
                ix = ix + incx;
            }
        }
    }

    if( cfy ) {
        conj<T>( Y.data(), len, incy );
    }

    ix = 0;
    iy = 0;
    for(i = 0 ; i < len ; i++) {
        rho = rho + X[ix] * Y[iy];
        ix  = ix + incx;
        iy  = iy + incy;
    }

    *rhorig = rho;
    return;
}

// Explicit template instantiations
template void ref_dotxv<float>(char, char, gtint_t, const float, const float*, gtint_t, const float*, gtint_t, const float, float*);
template void ref_dotxv<double>(char, char, gtint_t, const double, const double*, gtint_t, const double*, gtint_t, const double, double*);
template void ref_dotxv<scomplex>(char, char, gtint_t, const scomplex, const scomplex*, gtint_t, const scomplex*, gtint_t, const scomplex, scomplex*);
template void ref_dotxv<dcomplex>(char, char, gtint_t, const dcomplex, const dcomplex*, gtint_t, const dcomplex*, gtint_t, const dcomplex, dcomplex*);

} //end of namespace testinghelpers
