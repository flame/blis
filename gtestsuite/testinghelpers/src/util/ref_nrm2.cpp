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
#include "util/ref_nrm2.h"

/*
 * ==========================================================================
 * NORMFV performs vector operations
 *    Compute the euclidean norm of a vector
 *    of the elements in a vector x of length n. The resulting norm is stored to norm
 * ========================================================================
**/

namespace testinghelpers {

template <typename T, typename RT>
RT ref_nrm2(gtint_t n, T* x, gtint_t incx) {

  typedef RT (*Fptr_ref_cblas_nrm2)( f77_int, const T *, f77_int );
  Fptr_ref_cblas_nrm2 ref_cblas_nrm2;

  // Call C function
  /* Check the typename T passed to this function template and call respective function.*/
  if (typeid(T) == typeid(float))
  {
      ref_cblas_nrm2 = (Fptr_ref_cblas_nrm2)refCBLASModule.loadSymbol("cblas_snrm2");
  }
  else if (typeid(T) == typeid(double))
  {
      ref_cblas_nrm2 = (Fptr_ref_cblas_nrm2)refCBLASModule.loadSymbol("cblas_dnrm2");
  }
  else if (typeid(T) == typeid(scomplex))
  {
      ref_cblas_nrm2 = (Fptr_ref_cblas_nrm2)refCBLASModule.loadSymbol("cblas_scnrm2");
  }
  else if (typeid(T) == typeid(dcomplex))
  {
      ref_cblas_nrm2 = (Fptr_ref_cblas_nrm2)refCBLASModule.loadSymbol("cblas_dznrm2");
  }
  else
  {
    throw std::runtime_error("Error in ref_nrm2.cpp: Invalid typename is passed function template.");
  }
  if (!ref_cblas_nrm2) {
    throw std::runtime_error("Error in ref_nrm2.cpp: Function pointer == 0 -- symbol not found.");
  }

  return ref_cblas_nrm2(n, x, incx);
}

// Explicit template instantiations
template float  ref_nrm2<float, float>(gtint_t n, float* x, gtint_t incx);
template double ref_nrm2<double, double>(gtint_t n, double* x, gtint_t incx);
template float  ref_nrm2<scomplex, float>(gtint_t n, scomplex* x, gtint_t incx);
template double ref_nrm2<dcomplex, double>(gtint_t n, dcomplex* x, gtint_t incx);

} //end of namespace testinghelpers
