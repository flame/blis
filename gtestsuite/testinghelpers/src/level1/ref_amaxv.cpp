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
#include "level1/ref_amaxv.h"

namespace testinghelpers {

// Since amaxv is not a BLAS/CBLAS interface we use axpy as a reference.
template<typename T>
gtint_t ref_amaxv( gtint_t n, const T* x, gtint_t incx ) {
    gtint_t idx;
    typedef gtint_t (*Fptr_ref_cblas_amaxv)( f77_int, const T *, f77_int );
    Fptr_ref_cblas_amaxv ref_cblas_amaxv;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_cblas_amaxv = (Fptr_ref_cblas_amaxv)refCBLASModule.loadSymbol("cblas_isamax");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_cblas_amaxv = (Fptr_ref_cblas_amaxv)refCBLASModule.loadSymbol("cblas_idamax");
    }
    else if (typeid(T) == typeid(scomplex))
    {
        ref_cblas_amaxv = (Fptr_ref_cblas_amaxv)refCBLASModule.loadSymbol("cblas_icamax");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
        ref_cblas_amaxv = (Fptr_ref_cblas_amaxv)refCBLASModule.loadSymbol("cblas_izamax");
    }
    else
    {
        throw std::runtime_error("Error in ref_amaxv.cpp: Invalid typename is passed function template.");
    }
    if (!ref_cblas_amaxv) {
        throw std::runtime_error("Error in ref_amaxv.cpp: Function pointer == 0 -- symbol not found.");
    }

    idx = ref_cblas_amaxv( n, x, incx );
    return idx;
}


// Explicit template instantiations
template gtint_t ref_amaxv<float>(gtint_t, const float*, gtint_t);
template gtint_t ref_amaxv<double>(gtint_t, const double*, gtint_t);
template gtint_t ref_amaxv<scomplex>(gtint_t, const scomplex*, gtint_t);
template gtint_t ref_amaxv<dcomplex>(gtint_t, const dcomplex*, gtint_t);

} //end of namespace testinghelpers
