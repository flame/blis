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
#include "level1/ref_dotxv.h"
#include "level1/ref_dotxf.h"

/**
 * dotxf operation is defined as :
 * y := y + alpha * conja(A) * conjx(x)
 * where A is an m x b matrix, and y and x are vectors.
 */
namespace testinghelpers {
template<typename T>
void ref_dotxf( char conj_a,
                char conj_x,
                gtint_t m,
                gtint_t b,
                T *alpha,
                T* A,
                gtint_t inca,
                gtint_t lda,
                T* x,
                gtint_t incx,
                T * beta,
                T* y,
                gtint_t incy
              )
            {
                for ( dim_t i = 0; i < b; ++i )
                {
                  T* a1   = A + (0  )*inca + (i  )*lda;
                  T* x1   = x + (0  )*incx;
                  T* psi1 = y + (i  )*incy;

                  testinghelpers::ref_dotxv<T>
                  (
                    conj_a,
                    conj_x,
                    m,
                    *alpha,
                    a1, inca,
                    x1, incx,
                    *beta,
                    psi1
                  );
                }
            }

template void ref_dotxf<double>(
                char conj_a,
                char conj_x,
                gtint_t m,
                gtint_t b,
                double *alpha,
                double* A,
                gtint_t inca,
                gtint_t lda,
                double* x,
                gtint_t incx,
                double *beta,
                double* y,
                gtint_t incy
              );

template void ref_dotxf<float>(
                char conj_a,
                char conj_x,
                gtint_t m,
                gtint_t b,
                float *alpha,
                float* A,
                gtint_t inca,
                gtint_t lda,
                float* x,
                gtint_t incx,
                float *beta,
                float* y,
                gtint_t incy
              );

template void ref_dotxf<scomplex>(
                char conj_a,
                char conj_x,
                gtint_t m,
                gtint_t b,
                scomplex *alpha,
                scomplex* A,
                gtint_t inca,
                gtint_t lda,
                scomplex* x,
                gtint_t incx,
                scomplex *beta,
                scomplex* y,
                gtint_t incy
              );

template void ref_dotxf<dcomplex>(
                char conj_a,
                char conj_x,
                gtint_t m,
                gtint_t b,
                dcomplex *alpha,
                dcomplex* A,
                gtint_t inca,
                gtint_t lda,
                dcomplex* x,
                gtint_t incx,
                dcomplex *beta,
                dcomplex* y,
                gtint_t incy
              );
}
