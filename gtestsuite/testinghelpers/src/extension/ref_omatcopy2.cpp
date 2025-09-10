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
#include "extension/ref_omatcopy2.h"

namespace testinghelpers {

#if defined(REF_IS_MKL)
template<typename T>
void ref_omatcopy2( char storage, char trans, gtint_t m, gtint_t n, T alpha, T* A,
                    gtint_t lda, gtint_t stridea, T* B, gtint_t ldb, gtint_t strideb ) {

    // Defining the function pointer type for the native MKL call of omatcopy2
    typedef void (*Fptr_ref_mkl_omatcopy2)(
                                           char, char, size_t, size_t, const T,
                                           const T *, size_t, size_t, T *,
                                           size_t, size_t
                                         );

    // Function pointer to load the MKL symbol
    Fptr_ref_mkl_omatcopy2 ref_mkl_omatcopy2 = nullptr;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_mkl_omatcopy2 = (Fptr_ref_mkl_omatcopy2)refCBLASModule.loadSymbol("MKL_Somatcopy2");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_mkl_omatcopy2 = (Fptr_ref_mkl_omatcopy2)refCBLASModule.loadSymbol("MKL_Domatcopy2");
    }
    else if (typeid(T) == typeid(scomplex))
    {
        ref_mkl_omatcopy2 = (Fptr_ref_mkl_omatcopy2)refCBLASModule.loadSymbol("MKL_Comatcopy2");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
        ref_mkl_omatcopy2 = (Fptr_ref_mkl_omatcopy2)refCBLASModule.loadSymbol("MKL_Zomatcopy2");
    }
    else
    {
        throw std::runtime_error("Error in ref_omatcopy2.cpp: Invalid typename is passed function template.");
    }
    if (!ref_mkl_omatcopy2) {
        throw std::runtime_error("Error in ref_omatcopy2.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_mkl_omatcopy2( storage, trans, m, n, alpha, A, lda, stridea, B, ldb, strideb );
}
#else
template<typename T>
void ref_omatcopy2( char storage, char trans, gtint_t m, gtint_t n, T alpha, T* A,
                    gtint_t lda, gtint_t stridea, T* B, gtint_t ldb, gtint_t strideb ) {
    throw std::runtime_error("Error in ref_omatcopy2.cpp: The provided reference does not support the required operation.");
}
#endif

// Explicit template instantiations
template void ref_omatcopy2<float>( char, char, gtint_t, gtint_t, float, float*, gtint_t, gtint_t, float*, gtint_t, gtint_t );
template void ref_omatcopy2<double>( char, char, gtint_t, gtint_t, double, double*, gtint_t, gtint_t, double*, gtint_t, gtint_t );
template void ref_omatcopy2<scomplex>( char, char, gtint_t, gtint_t, scomplex, scomplex*, gtint_t, gtint_t, scomplex*, gtint_t, gtint_t );
template void ref_omatcopy2<dcomplex>( char, char, gtint_t, gtint_t, dcomplex, dcomplex*, gtint_t, gtint_t, dcomplex*, gtint_t, gtint_t );

} //end of namespace testinghelpers
