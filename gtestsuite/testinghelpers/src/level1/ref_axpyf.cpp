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
void ref_axpyf( conj_t conja,
                conj_t conjx,
                gint_t m,
                gint_t b,
                T *alpha,
                T* A,
                gint_t inca,
                gint_t lda,
                T* x,
                gint_t incx,
                T* y,
                gint_t incy
              )
            {
                for (gint_t i = 0; i < b; ++i )
                {
                    T* a1   = A + (0  )*inca + (i  )*lda;
                    T* chi1 = x + (i  )*incx;
                    T* y1   = y + (0  )*incy;

                    T alpha_chi1 = bli_cpyscal( conjx, chi1, alpha );

                    testinghelpers::ref_axpyv<T>( conja, m, alpha_chi1, a1, inca, y1, incy );
                }
            }

template void ref_axpyf<float>(
                conj_t conja,
                conj_t conjx,
                gint_t m,
                gint_t b,
                float *alpha,
                float* A,
                gint_t inca,
                gint_t lda,
                float* x,
                gint_t incx,
                float* y,
                gint_t incy
              );

template void ref_axpyf<double>(
                conj_t conja,
                conj_t conjx,
                gint_t m,
                gint_t b,
                double *alpha,
                double* A,
                gint_t inca,
                gint_t lda,
                double* x,
                gint_t incx,
                double* y,
                gint_t incy
              );

template void ref_axpyf<scomplex>(
                conj_t conja,
                conj_t conjx,
                gint_t m,
                gint_t b,
                scomplex *alpha,
                scomplex* A,
                gint_t inca,
                gint_t lda,
                scomplex* x,
                gint_t incx,
                scomplex* y,
                gint_t incy
              );

template void ref_axpyf<dcomplex>(
                conj_t conja,
                conj_t conjx,
                gint_t m,
                gint_t b,
                dcomplex *alpha,
                dcomplex* A,
                gint_t inca,
                gint_t lda,
                dcomplex* x,
                gint_t incx,
                dcomplex* y,
                gint_t incy
              );
}


