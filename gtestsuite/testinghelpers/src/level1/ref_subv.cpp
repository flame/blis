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
#include "level1/ref_subv.h"

namespace testinghelpers {

// Since subv is not supported by BLAS/CBLAS, we have a local reference implementation.
template<typename T>
void ref_subv( char conj_x, gtint_t n, const T* xp, gtint_t incx,
                                             T* y, gtint_t incy ) {
    gtint_t i, ix, iy;
    bool cfx    = chkconj( conj_x );
    gtint_t svx = buff_dim(n, incx);

    if (n == 0) {
        return;
    }

    std::vector<T> X( svx );
    memcpy(X.data(), xp, svx*sizeof(T));

    if( cfx ) {
        conj<T>( X.data(), n, incx );
    }

    ix = 0;
    iy = 0;
    for(i = 0 ; i < n ; i++) {
        y[iy] = y[iy] - X[ix];
        ix    = ix + incx;
        iy    = iy + incy;
    }

    return;
}

// Explicit template instantiations
template void ref_subv<float>(char, gtint_t, const float*, gtint_t, float*, gtint_t);
template void ref_subv<double>(char, gtint_t, const double*, gtint_t, double*, gtint_t);
template void ref_subv<scomplex>(char, gtint_t, const scomplex*, gtint_t, scomplex*, gtint_t);
template void ref_subv<dcomplex>(char, gtint_t, const dcomplex*, gtint_t, dcomplex*, gtint_t);

} //end of namespace testinghelpers
