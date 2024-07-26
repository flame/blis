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
#include "level1/ref_axpyv.h"
#include "level1/ref_axpyf.h"


namespace testinghelpers {

float bli_cpyscal(conj_t conjx, float *chi1, float *alpha )
{
    float alpha_chi1;
    bli_scopycjs( conjx, *chi1, alpha_chi1 );
    bli_sscals( *alpha, alpha_chi1 );
    return alpha_chi1;
}

double bli_cpyscal(conj_t conjx, double *chi1, double *alpha )
{
    double alpha_chi1;
    bli_dcopycjs( conjx, *chi1, alpha_chi1 );
    bli_dscals( *alpha, alpha_chi1 );
    return alpha_chi1;
}

scomplex bli_cpyscal(conj_t conjx, scomplex *chi1, scomplex *alpha )
{
    scomplex alpha_chi1;
    bli_ccopycjs( conjx, *chi1, alpha_chi1 );
    bli_cscals( *alpha, alpha_chi1 );
    return alpha_chi1;
}

dcomplex bli_cpyscal(conj_t conjx, dcomplex *chi1, dcomplex *alpha )
{
    dcomplex alpha_chi1;
    bli_zcopycjs( conjx, *chi1, alpha_chi1 );
    bli_zscals( *alpha, alpha_chi1 );
    return alpha_chi1;
}

template<typename T>
void ref_axpyf( char conja,
                char conjx,
                gtint_t m,
                gtint_t b,
                T *alpha,
                T* A,
                gtint_t inca,
                gtint_t lda,
                T* x,
                gtint_t incx,
                T* y,
                gtint_t incy
              )
            {
                conj_t blis_conjx;
                testinghelpers::char_to_blis_conj( conjx, &blis_conjx );
                for (gtint_t i = 0; i < b; ++i )
                {
                    T* a1   = A + (0  )*inca + (i  )*lda;
                    T* chi1 = x + (i  )*incx;
                    T* y1   = y + (0  )*incy;

                    T alpha_chi1 = bli_cpyscal( blis_conjx, chi1, alpha );

                    testinghelpers::ref_axpyv<T>( conja, m, alpha_chi1, a1, inca, y1, incy );
                }
            }

template void ref_axpyf<float>(
                char conja,
                char conjx,
                gtint_t m,
                gtint_t b,
                float *alpha,
                float* A,
                gtint_t inca,
                gtint_t lda,
                float* x,
                gtint_t incx,
                float* y,
                gtint_t incy
              );

template void ref_axpyf<double>(
                char conja,
                char conjx,
                gtint_t m,
                gtint_t b,
                double *alpha,
                double* A,
                gtint_t inca,
                gtint_t lda,
                double* x,
                gtint_t incx,
                double* y,
                gtint_t incy
              );

template void ref_axpyf<scomplex>(
                char conja,
                char conjx,
                gtint_t m,
                gtint_t b,
                scomplex *alpha,
                scomplex* A,
                gtint_t inca,
                gtint_t lda,
                scomplex* x,
                gtint_t incx,
                scomplex* y,
                gtint_t incy
              );

template void ref_axpyf<dcomplex>(
                char conja,
                char conjx,
                gtint_t m,
                gtint_t b,
                dcomplex *alpha,
                dcomplex* A,
                gtint_t inca,
                gtint_t lda,
                dcomplex* x,
                gtint_t incx,
                dcomplex* y,
                gtint_t incy
              );
}
