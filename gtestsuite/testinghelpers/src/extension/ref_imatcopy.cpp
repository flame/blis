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
#include "extension/ref_imatcopy.h"

namespace testinghelpers {

#if defined(REF_IS_OPENBLAS)

// Template function to load and call CBLAS call of OpenBLAS ?imatcopy, only for real datatypes
template<typename T>
void ref_imatcopy_real( char storage, char trans, gtint_t m, gtint_t n, T alpha, T* A,
                        gtint_t lda_in, gtint_t lda_out ) {

    // Since CBLAS call does not support plain conjugation, we need to conjugate A
    // in case trans == 'r'(only conjugation)
    if( trans == 'r' )
    {
        gtint_t size_a = testinghelpers::matsize(storage, 'n', m, n, lda_in );
        std::vector<T> A_conj( size_a );
        memcpy( A_conj.data(), A, size_a * sizeof(T) );
        testinghelpers::conj<T>( storage, A_conj.data(), m, n, lda_in );
        memcpy( A, A_conj.data(), size_a * sizeof(T) );
        trans = 'n';
    }

    enum CBLAS_ORDER cblas_order;
    enum CBLAS_TRANSPOSE cblas_trans;

    char_to_cblas_order( storage, &cblas_order );
    char_to_cblas_trans( trans, &cblas_trans );

    // Defining the function pointer type for CBLAS call of imatcopy
    typedef void (*Fptr_ref_cblas_imatcopy)(
                                              const CBLAS_ORDER, const CBLAS_TRANSPOSE,
                                              const f77_int, const f77_int, const T,
                                              const T *, const f77_int, const f77_int
                                           );

    // Function pointer to load the CBLAS symbol
    Fptr_ref_cblas_imatcopy ref_cblas_imatcopy = nullptr;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_cblas_imatcopy = (Fptr_ref_cblas_imatcopy)refCBLASModule.loadSymbol("cblas_simatcopy");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_cblas_imatcopy = (Fptr_ref_cblas_imatcopy)refCBLASModule.loadSymbol("cblas_dimatcopy");
    }

    if (!ref_cblas_imatcopy) {
        throw std::runtime_error("Error in ref_imatcopy.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_cblas_imatcopy( cblas_order, cblas_trans, m, n, alpha, A, lda_in, lda_out );
}

// Template function to load and call CBLAS call of OpenBLAS ?imatcopy, only for complex datatypes
template<typename T>
void ref_imatcopy_complex( char storage, char trans, gtint_t m, gtint_t n, T alpha, T* A,
                        gtint_t lda_in, gtint_t lda_out ) {

    // Since CBLAS call does not support plain conjugation, we need to conjugate A
    // in case trans == 'r'(only conjugation)
    if( trans == 'r' )
    {
        gtint_t size_a = testinghelpers::matsize(storage, 'n', m, n, lda_in );
        std::vector<T> A_conj( size_a );
        memcpy( A_conj.data(), A, size_a * sizeof(T) );
        testinghelpers::conj<T>( storage, A_conj.data(), m, n, lda_in );
        memcpy( A, A_conj.data(), size_a * sizeof(T) );
        trans = 'n';
    }

    // Getting the real-precision of the complex datatype
    using RT = typename testinghelpers::type_info<T>::real_type;

    enum CBLAS_ORDER cblas_order;
    enum CBLAS_TRANSPOSE cblas_trans;

    char_to_cblas_order( storage, &cblas_order );
    char_to_cblas_trans( trans, &cblas_trans );

    // Defining the function pointer type for CBLAS call of imatcopy
    typedef void (*Fptr_ref_cblas_imatcopy)(
                                              const CBLAS_ORDER, const CBLAS_TRANSPOSE,
                                              const f77_int, const f77_int, const RT *,
                                              const RT *, const f77_int, const f77_int
                                           );

    // Function pointer to load the CBLAS symbol
    Fptr_ref_cblas_imatcopy ref_cblas_imatcopy = nullptr;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(scomplex))
    {
        ref_cblas_imatcopy = (Fptr_ref_cblas_imatcopy)refCBLASModule.loadSymbol("cblas_cimatcopy");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
        ref_cblas_imatcopy = (Fptr_ref_cblas_imatcopy)refCBLASModule.loadSymbol("cblas_zimatcopy");
    }

    if (!ref_cblas_imatcopy) {
        throw std::runtime_error("Error in ref_imatcopy.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_cblas_imatcopy( cblas_order, cblas_trans, m, n, (RT *)(&alpha), (RT *)A, lda_in, lda_out );
}

template<typename T>
void ref_imatcopy( char storage, char trans, gtint_t m, gtint_t n, T alpha, T* A,
                   gtint_t lda_in, gtint_t lda_out ) {

    // Due to difference in the CBLAS API signature for OpenBLAS ?imatcopy(among real and complex)
    // types, we have two different template functions(front-ends), that will be called based on the
    // datatype.
    if ((typeid(T) == typeid(float)) || (typeid(T) == typeid(double)))
    {
        ref_imatcopy_real( storage, trans, m, n, alpha, A, lda_in, lda_out );
    }
    else if ((typeid(T) == typeid(scomplex)) || (typeid(T) == typeid(dcomplex)))
    {
        ref_imatcopy_complex( storage, trans, m, n, alpha, A, lda_in, lda_out );
    }
    else
    {
        throw std::runtime_error("Error in ref_imatcopy.cpp: Invalid typename is passed function template.");
    }
}

#elif defined(REF_IS_MKL)
template<typename T>
void ref_imatcopy( char storage, char trans, gtint_t m, gtint_t n, T alpha, T* A,
                    gtint_t lda_in, gtint_t lda_out ) {

    // Defining the function pointer type for the native MKL call of imatcopy
    typedef void (*Fptr_ref_mkl_imatcopy)(
                                           char, char, size_t, size_t,
                                           const T, const T *, size_t,
                                           size_t
                                         );

    // Function pointer to load the MKL symbol
    Fptr_ref_mkl_imatcopy ref_mkl_imatcopy = nullptr;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_mkl_imatcopy = (Fptr_ref_mkl_imatcopy)refCBLASModule.loadSymbol("MKL_Simatcopy");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_mkl_imatcopy = (Fptr_ref_mkl_imatcopy)refCBLASModule.loadSymbol("MKL_Dimatcopy");
    }
    else if (typeid(T) == typeid(scomplex))
    {
        ref_mkl_imatcopy = (Fptr_ref_mkl_imatcopy)refCBLASModule.loadSymbol("MKL_Cimatcopy");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
        ref_mkl_imatcopy = (Fptr_ref_mkl_imatcopy)refCBLASModule.loadSymbol("MKL_Zimatcopy");
    }
    else
    {
        throw std::runtime_error("Error in ref_imatcopy.cpp: Invalid typename is passed function template.");
    }
    if (!ref_mkl_imatcopy) {
        throw std::runtime_error("Error in ref_imatcopy.cpp: Function pointer == 0 -- symbol not found.");
    }

    ref_mkl_imatcopy( storage, trans, m, n, alpha, A, lda_in, lda_out );
}
#else
template<typename T>
void ref_imatcopy( char storage, char trans, gtint_t m, gtint_t n, T alpha, T* A,
                    gtint_t lda_in, gtint_t lda_out ) {
    throw std::runtime_error("Error in ref_imatcopy.cpp: The provided reference does not support the required operation.");
}
#endif

// Explicit template instantiations
#if defined(REF_IS_OPENBLAS)
template void ref_imatcopy_real<float>( char, char, gtint_t, gtint_t, float, float*, gtint_t, gtint_t );
template void ref_imatcopy_real<double>( char, char, gtint_t, gtint_t, double, double*, gtint_t, gtint_t );
template void ref_imatcopy_complex<scomplex>( char, char, gtint_t, gtint_t, scomplex, scomplex*, gtint_t, gtint_t );
template void ref_imatcopy_complex<dcomplex>( char, char, gtint_t, gtint_t, dcomplex, dcomplex*, gtint_t, gtint_t );
#endif

template void ref_imatcopy<float>( char, char, gtint_t, gtint_t, float, float*, gtint_t, gtint_t );
template void ref_imatcopy<double>( char, char, gtint_t, gtint_t, double, double*, gtint_t, gtint_t );
template void ref_imatcopy<scomplex>( char, char, gtint_t, gtint_t, scomplex, scomplex*, gtint_t, gtint_t );
template void ref_imatcopy<dcomplex>( char, char, gtint_t, gtint_t, dcomplex, dcomplex*, gtint_t, gtint_t );

} //end of namespace testinghelpers
