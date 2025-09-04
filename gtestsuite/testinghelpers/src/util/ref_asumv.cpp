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
#include "util/ref_asumv.h"

/*
 *  ==========================================================================
 *  ASUMV computes the sum of the absolute values of the fundamental elements
 *  of vector x.
 *  ==========================================================================
**/

namespace testinghelpers {

template <typename T, typename RT>
RT ref_asumv(gtint_t n, T* x, gtint_t incx) {

  typedef RT (*Fptr_ref_cblas_asum)( f77_int, const T *, f77_int );
  Fptr_ref_cblas_asum ref_cblas_asum;

  // Call C function
  /* Check the typename T passed to this function template and call respective function.*/
  if (typeid(T) == typeid(float))
  {
      ref_cblas_asum = (Fptr_ref_cblas_asum)refCBLASModule.loadSymbol("cblas_sasum");
  }
  else if (typeid(T) == typeid(double))
  {
      ref_cblas_asum = (Fptr_ref_cblas_asum)refCBLASModule.loadSymbol("cblas_dasum");
  }
  else if (typeid(T) == typeid(scomplex))
  {
      ref_cblas_asum = (Fptr_ref_cblas_asum)refCBLASModule.loadSymbol("cblas_scasum");
  }
  else if (typeid(T) == typeid(dcomplex))
  {
      ref_cblas_asum = (Fptr_ref_cblas_asum)refCBLASModule.loadSymbol("cblas_dzasum");
  }
  else
  {
    throw std::runtime_error("Error in ref_asumv.cpp: Invalid typename is passed function template.");
  }
  if (!ref_cblas_asum) {
    throw std::runtime_error("Error in ref_asumv.cpp: Function pointer == 0 -- symbol not found.");
  }

  return ref_cblas_asum(n, x, incx);
}

// Explicit template instantiations
template float  ref_asumv<   float,  float>(gtint_t n,    float* x, gtint_t incx);
template double ref_asumv<  double, double>(gtint_t n,   double* x, gtint_t incx);
template float  ref_asumv<scomplex,  float>(gtint_t n, scomplex* x, gtint_t incx);
template double ref_asumv<dcomplex, double>(gtint_t n, dcomplex* x, gtint_t incx);

} //end of namespace testinghelpers
