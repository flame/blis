/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

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
#include "level1/ref_swapv.h"

namespace testinghelpers {

template<typename T>
void ref_swapv(gtint_t n, T* x, gtint_t incx, T* y, gtint_t incy)
{
    typedef void (*Fptr_ref_cblas_swapv)( f77_int, T *, f77_int, T *, f77_int);
    Fptr_ref_cblas_swapv ref_cblas_swapv;

    if (typeid(T) == typeid(float))
        ref_cblas_swapv = (Fptr_ref_cblas_swapv)refCBLASModule.loadSymbol("cblas_sswap");
    else if (typeid(T) == typeid(double))
        ref_cblas_swapv = (Fptr_ref_cblas_swapv)refCBLASModule.loadSymbol("cblas_dswap");
    else if (typeid(T) == typeid(scomplex))
        ref_cblas_swapv = (Fptr_ref_cblas_swapv)refCBLASModule.loadSymbol("cblas_cswap");
    else if (typeid(T) == typeid(dcomplex))
        ref_cblas_swapv = (Fptr_ref_cblas_swapv)refCBLASModule.loadSymbol("cblas_zswap");
    else
        throw std::runtime_error("Error in ref_swapv.cpp: Invalid typename is passed function template.");

    if (!ref_cblas_swapv)
        throw std::runtime_error("Error in ref_swapv.cpp: Function pointer == 0 -- symbol not found.");

    ref_cblas_swapv( n, x, incx, y, incy );
}

// Explicit template instantiations
template void ref_swapv<float>(gtint_t, float*, gtint_t, float*, gtint_t);
template void ref_swapv<double>(gtint_t, double*, gtint_t, double*, gtint_t);
template void ref_swapv<scomplex>(gtint_t, scomplex*, gtint_t, scomplex*, gtint_t);
template void ref_swapv<dcomplex>(gtint_t, dcomplex*, gtint_t, dcomplex*, gtint_t);

} //end of namespace testinghelpers
