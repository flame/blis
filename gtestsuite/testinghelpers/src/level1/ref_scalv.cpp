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
#include "level1/ref_scalv.h"

namespace testinghelpers {

template<typename T>
void ref_scalv(char conjalpha, gtint_t n, T alpha, T* x, gtint_t incx)
{
    using scalar_t = std::conditional_t<testinghelpers::type_info<T>::is_complex, T&, T>;
    typedef void (*Fptr_ref_cblas_scal)( f77_int, scalar_t , T *, f77_int);
    Fptr_ref_cblas_scal ref_cblas_scal;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_cblas_scal = (Fptr_ref_cblas_scal)refCBLASModule.loadSymbol("cblas_sscal");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_cblas_scal = (Fptr_ref_cblas_scal)refCBLASModule.loadSymbol("cblas_dscal");
    }
    else if (typeid(T) == typeid(scomplex))
    {
        ref_cblas_scal = (Fptr_ref_cblas_scal)refCBLASModule.loadSymbol("cblas_cscal");
    }
    else if (typeid(T) == typeid(dcomplex))
    {
        ref_cblas_scal = (Fptr_ref_cblas_scal)refCBLASModule.loadSymbol("cblas_zscal");
    }
    else
    {
        throw std::runtime_error("Error in ref_scalv.cpp: Invalid typename is passed function template.");
    }
    if (!ref_cblas_scal) {
        throw std::runtime_error("Error in ref_scalv.cpp: Function pointer == 0 -- symbol not found.");
    }

#ifdef TEST_BLIS_TYPED
    if( chkconj( conjalpha ) )
    {
        T alpha_conj = testinghelpers::conj<T>( alpha );
        ref_cblas_scal( n, alpha_conj, x, incx );
    }
    else
#endif
    {
        ref_cblas_scal( n, alpha, x, incx );
    }

}

// Explicit template instantiations
template void ref_scalv<float>(char, gtint_t, float, float*, gtint_t);
template void ref_scalv<double>(char, gtint_t, double, double*, gtint_t);
template void ref_scalv<scomplex>(char, gtint_t, scomplex, scomplex*, gtint_t);
template void ref_scalv<dcomplex>(char, gtint_t, dcomplex, dcomplex*, gtint_t);

} //end of namespace testinghelpers
