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
#include "extension/ref_omatcopy.h"

namespace testinghelpers {

#if defined(REF_IS_OPENBLAS)

// Template function to load and call CBLAS call of OpenBLAS ?omatcopy, only for real datatypes
template<typename T>
void ref_omatcopy_real( char storage, char trans, gtint_t m, gtint_t n, T alpha, T* A,
                        gtint_t lda, T* B, gtint_t ldb ) {

    // Since CBLAS call does not support plain conjugation, we need to conjugate A
    // in case trans == 'r'(only conjugation)
    if( trans == 'r' )
    {
        gtint_t size_a = testinghelpers::matsize(storage, 'n', m, n, lda);
        std::vector<T> A_conj( size_a );
        memcpy( A_conj.data(), A, size_a * sizeof(T) );
        testinghelpers::conj<T>( storage, A_conj.data(), m, n, lda );
        memcpy( A, A_conj.data(), size_a * sizeof(T) );
        trans = 'n';
    }

    enum CBLAS_ORDER cblas_order;
    enum CBLAS_TRANSPOSE cblas_trans;

    char_to_cblas_order( storage, &cblas_order );
    char_to_cblas_trans( trans, &cblas_trans );

    // Defining the function pointer type for CBLAS call of OMATCOPY
    typedef void (*Fptr_ref_cblas_omatcopy)(
                                              const CBLAS_ORDER, const CBLAS_TRANSPOSE,
                                              const f77_int, const f77_int, const T,
                                              const T *, const f77_int, const T *,
                                              const f77_int
                                           );

    // Function pointer to load the CBLAS symbol
    Fptr_ref_cblas_omatcopy ref_cblas_omatcopy = nullptr;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_cblas_omatcopy = (Fptr_ref_cblas_omatcopy)refCBLASModule.loadSymbol("cblas_somatcopy");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_cblas_omatcopy = (Fptr_ref_cblas_omatcopy)refCBLASModule.loadSymbol("cblas_domatcopy");
    }

    if (!ref_cblas_omatcopy) {
        throw std::runtime_error("Error in ref_omatcopy.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_cblas_omatcopy( cblas_order, cblas_trans, m, n, alpha, A, lda, B, ldb );
}

// Template function to load and call CBLAS call of OpenBLAS ?omatcopy, only for complex datatypes
template<typename T>
void ref_omatcopy_complex( char storage, char trans, gtint_t m, gtint_t n, T alpha, T* A,
                        gtint_t lda, T* B, gtint_t ldb ) {

    // Since CBLAS call does not support plain conjugation, we need to conjugate A
    // in case trans == 'r'(only conjugation)
    if( trans == 'r' )
    {
        gtint_t size_a = testinghelpers::matsize(storage, 'n', m, n, lda);
        std::vector<T> A_conj( size_a );
        memcpy( A_conj.data(), A, size_a * sizeof(T) );
        testinghelpers::conj<T>( storage, A_conj.data(), m, n, lda );
        memcpy( A, A_conj.data(), size_a * sizeof(T) );
        trans = 'n';
    }

    // Getting the real-precision of the complex datatype
    using RT = typename testinghelpers::type_info<T>::real_type;

    enum CBLAS_ORDER cblas_order;
    enum CBLAS_TRANSPOSE cblas_trans;

    char_to_cblas_order( storage, &cblas_order );
    char_to_cblas_trans( trans, &cblas_trans );

    // Defining the function pointer type for CBLAS call of OMATCOPY
    typedef void (*Fptr_ref_cblas_omatcopy)(
                                              const CBLAS_ORDER, const CBLAS_TRANSPOSE,
                                              const f77_int, const f77_int, const RT *,
                                              const RT *, const f77_int, const RT *,
                                              const f77_int
                                           );

    // Function pointer to load the CBLAS symbol
    Fptr_ref_cblas_omatcopy ref_cblas_omatcopy = nullptr;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(scomplex))
    {
        ref_cblas_omatcopy = (Fptr_ref_cblas_omatcopy)refCBLASModule.loadSymbol("cblas_comatcopy");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
        ref_cblas_omatcopy = (Fptr_ref_cblas_omatcopy)refCBLASModule.loadSymbol("cblas_zomatcopy");
    }

    if (!ref_cblas_omatcopy) {
        throw std::runtime_error("Error in ref_omatcopy.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_cblas_omatcopy( cblas_order, cblas_trans, m, n, (RT *)(&alpha), (RT *)A, lda, (RT *)B, ldb );
}

template<typename T>
void ref_omatcopy( char storage, char trans, gtint_t m, gtint_t n, T alpha, T* A,
                   gtint_t lda, T* B, gtint_t ldb ) {

    // Due to difference in the CBLAS API signature for OpenBLAS ?omatcopy(among real and complex)
    // types, we have two different template functions(front-ends), that will be called based on the
    // datatype.
    if ((typeid(T) == typeid(float)) || (typeid(T) == typeid(double)))
    {
        ref_omatcopy_real( storage, trans, m, n, alpha, A, lda, B, ldb );
    }
    else if ((typeid(T) == typeid(scomplex)) || (typeid(T) == typeid(dcomplex)))
    {
        ref_omatcopy_complex( storage, trans, m, n, alpha, A, lda, B, ldb );
    }
    else
    {
        throw std::runtime_error("Error in ref_omatcopy.cpp: Invalid typename is passed function template.");
    }
}

#elif defined(REF_IS_MKL)
template<typename T>
void ref_omatcopy( char storage, char trans, gtint_t m, gtint_t n, T alpha, T* A,
                    gtint_t lda, T* B, gtint_t ldb ) {

    // Defining the function pointer type for the native MKL call of OMATCOPY
    typedef void (*Fptr_ref_mkl_omatcopy)(
                                           char, char, size_t, size_t,
                                           const T, const T *, size_t,
                                           T *, size_t
                                         );

    // Function pointer to load the MKL symbol
    Fptr_ref_mkl_omatcopy ref_mkl_omatcopy = nullptr;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_mkl_omatcopy = (Fptr_ref_mkl_omatcopy)refCBLASModule.loadSymbol("MKL_Somatcopy");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_mkl_omatcopy = (Fptr_ref_mkl_omatcopy)refCBLASModule.loadSymbol("MKL_Domatcopy");
    }
    else if (typeid(T) == typeid(scomplex))
    {
        ref_mkl_omatcopy = (Fptr_ref_mkl_omatcopy)refCBLASModule.loadSymbol("MKL_Comatcopy");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
        ref_mkl_omatcopy = (Fptr_ref_mkl_omatcopy)refCBLASModule.loadSymbol("MKL_Zomatcopy");
    }
    else
    {
        throw std::runtime_error("Error in ref_omatcopy.cpp: Invalid typename is passed function template.");
    }
    if (!ref_mkl_omatcopy) {
        throw std::runtime_error("Error in ref_omatcopy.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_mkl_omatcopy( storage, trans, m, n, alpha, A, lda, B, ldb );
}
#else
template<typename T>
void ref_omatcopy( char storage, char trans, gtint_t m, gtint_t n, T alpha, T* A,
                    gtint_t lda, T* B, gtint_t ldb ) {
    throw std::runtime_error("Error in ref_omatcopy.cpp: The provided reference does not support the required operation.");
}
#endif

// Explicit template instantiations
#if defined(REF_IS_OPENBLAS)
template void ref_omatcopy_real<float>( char, char, gtint_t, gtint_t, float, float*, gtint_t, float*, gtint_t );
template void ref_omatcopy_real<double>( char, char, gtint_t, gtint_t, double, double*, gtint_t, double*, gtint_t );
template void ref_omatcopy_complex<scomplex>( char, char, gtint_t, gtint_t, scomplex, scomplex*, gtint_t, scomplex*, gtint_t );
template void ref_omatcopy_complex<dcomplex>( char, char, gtint_t, gtint_t, dcomplex, dcomplex*, gtint_t, dcomplex*, gtint_t );
#endif

template void ref_omatcopy<float>( char, char, gtint_t, gtint_t, float, float*, gtint_t, float*, gtint_t );
template void ref_omatcopy<double>( char, char, gtint_t, gtint_t, double, double*, gtint_t, double*, gtint_t );
template void ref_omatcopy<scomplex>( char, char, gtint_t, gtint_t, scomplex, scomplex*, gtint_t, scomplex*, gtint_t );
template void ref_omatcopy<dcomplex>( char, char, gtint_t, gtint_t, dcomplex, dcomplex*, gtint_t, dcomplex*, gtint_t );

} //end of namespace testinghelpers
