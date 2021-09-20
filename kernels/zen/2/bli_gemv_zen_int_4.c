/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021, Advanced Micro Devices, Inc. All rights reserved.

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

#include "immintrin.h"
#include "blis.h"

/*
  This implementation uses 512 bits of cache line efficiently for
  column stored matrix and vectors.
  To achieve this, at each iteration we use 2 ymm registers
  i.e. .512 bits for arithmetic operation. By this we use the
  cache efficiently.
*/
void bli_zgemv_zen_int_4x4
   (
      conj_t              conja,
      conj_t              conjx,
      dim_t               m,
      dim_t               n,
      dcomplex*  restrict alpha,
      dcomplex*  restrict a, inc_t inca, inc_t lda,
      dcomplex*  restrict x, inc_t incx,
      dcomplex*  restrict beta,
      dcomplex*  restrict y, inc_t incy,
      cntx_t*    restrict cntx
   )
{

    const dim_t S_MR = 4; // Kernel size , m = 4
    const dim_t S_NR = 4; // Kernel size , n = 4

    dcomplex            chi0;
    dcomplex            chi1;
    dcomplex            chi2;
    dcomplex            chi3;

    inc_t lda2 = 2*lda;
    inc_t lda3 = 3*lda;

    inc_t incy2 = 2*incy;
    inc_t incx2 = 2*incx;
    inc_t incx3 = 3*incx;
    inc_t inca2 = 2*inca;

    dcomplex* restrict x0 = x;
    dcomplex* restrict y0 = y;
    dcomplex* restrict a0 = a;

    dim_t i,j;

    __m256d ymm0, ymm1, ymm2, ymm3;
    __m256d ymm4, ymm5, ymm6, ymm7;
    __m256d ymm8, ymm9, ymm10, ymm11;
    __m256d ymm12, ymm13, ymm14, ymm15;

    for( i = 0; i+S_NR-1 < n; i+=S_NR )
    {
        a0 = a + (i  )*lda;
        x0 = x + (i  )*incx;
        y0 = y;// For each kernel, y should start form beginning

        chi0 = *( x0);// + 0*incx );
        chi1 = *( x0 + incx );
        chi2 = *( x0 + incx2 );
        chi3 = *( x0 + incx3 );

        // Scale each chi scalar by alpha.
        bli_zscals( *alpha, chi0 );
        bli_zscals( *alpha, chi1 );
        bli_zscals( *alpha, chi2 );
        bli_zscals( *alpha, chi3 );

        // broadcast x0,x1,x2,x3
        // broadcast real & imag parts of 4 elements of x
        ymm0 = _mm256_broadcast_sd(&chi0.real); // real part of x0
        ymm1 = _mm256_broadcast_sd(&chi0.imag); // imag part of x0
        ymm2 = _mm256_broadcast_sd(&chi1.real); // real part of x1
        ymm3 = _mm256_broadcast_sd(&chi1.imag); // imag part of x1
        ymm4 = _mm256_broadcast_sd(&chi2.real); // real part of x2
        ymm5 = _mm256_broadcast_sd(&chi2.imag); // imag part of x2
        ymm6 = _mm256_broadcast_sd(&chi3.real); // real part of x3
        ymm7 = _mm256_broadcast_sd(&chi3.imag); // imag part of x3

        for( j = 0 ; j+S_MR-1 < m ; j+=S_MR )
        {
            //load columns of A
            ymm8  = _mm256_loadu_pd((double const *)(a0));
            ymm9  = _mm256_loadu_pd((double const *)(a0 +  lda));
            ymm10 = _mm256_loadu_pd((double const *)(a0 + lda2));
            ymm11 = _mm256_loadu_pd((double const *)(a0 + lda3));

//--------------------
            //Ar*Xr Ai*Xr Ar*Xr Ai*Xr
            ymm14 = _mm256_mul_pd(ymm8, ymm0);
            //Ar*Xi Ai*Xi Ar*Xi Ai*Xi
            ymm15 = _mm256_mul_pd(ymm8, ymm1);

            /* Next set of A mult by real and imag,
            Add into the previous real and imag results */

            // (Ar*Xr Ai*Xr Ar*Xr Ai*Xr) + (prev iteration real results)
            ymm14 = _mm256_fmadd_pd(ymm9, ymm2, ymm14);
            // (Ar*Xi Ai*Xi Ar*Xi Ai*Xi) + + (prev iteration imag results)
            ymm15 = _mm256_fmadd_pd(ymm9, ymm3, ymm15);

            // (Ar*Xr Ai*Xr Ar*Xr Ai*Xr) + (prev iteration real results)
            ymm14 = _mm256_fmadd_pd(ymm10, ymm4, ymm14);
            // (Ar*Xi Ai*Xi Ar*Xi Ai*Xi) + + (prev iteration imag results)
            ymm15 = _mm256_fmadd_pd(ymm10, ymm5, ymm15);

            // (Ar*Xr Ai*Xr Ar*Xr Ai*Xr) + (prev iteration real results)
            ymm14 = _mm256_fmadd_pd(ymm11, ymm6, ymm14);
            // (Ar*Xi Ai*Xi Ar*Xi Ai*Xi) + + (prev iteration imag results)
            ymm15 = _mm256_fmadd_pd(ymm11, ymm7, ymm15);

            /*Permute the imag acc register to addsub to real accu results */
            // (Ar*Xi Ai*Xi Ar*Xi Ai*Xi) => (Ai*Xi Ar*Xi Ai*Xi Ar*Xi)
            ymm15 = _mm256_permute_pd(ymm15, 5);

            /*AddSub to get the 2 proper complex multipled value*/
            /* Ar*Xi - Ai*Xi, Ai*Xi + Ar*Xi, Ar*Xi - Ai*Xi, Ai*Xi + Ar*Xi*/
            ymm12 = _mm256_addsub_pd(ymm14, ymm15);

            //load Y vector
            ymm14 = _mm256_loadu_pd((double*)y0);
            //Add the results into y
            ymm12 = _mm256_add_pd(ymm14, ymm12);
            // Store the results back
            _mm256_storeu_pd((double*)(y0), ymm12);
//-----------------------

            // Load Next Set of A matrix elements for the same col
            // Ar2 Ai2 Ar3 Ai3
            ymm8  = _mm256_loadu_pd((double const *)(a0 + (inca2)));
            ymm9  = _mm256_loadu_pd((double const *)(a0 + (inca2) + lda));
            ymm10 = _mm256_loadu_pd((double const *)(a0 + (inca2) + lda2));
            ymm11 = _mm256_loadu_pd((double const *)(a0 + (inca2) + lda3));

            //Ar0*Xr Ai0*Xr Ar1*Xr Ai1*Xr
            ymm14 = _mm256_mul_pd(ymm8, ymm0);
            //Ar0*Xi Ai0*Xi Ar1*Xi Ai1*Xi
            ymm15 = _mm256_mul_pd(ymm8, ymm1);

            /* Next set of A mult by real and imag,
            Add into the previous real and imag results */

            // (Ar*Xr Ai*Xr Ar*Xr Ai*Xr) + (prev iteration real results)
            ymm14 = _mm256_fmadd_pd(ymm9, ymm2, ymm14);
            // (Ar*Xi Ai*Xi Ar*Xi Ai*Xi) + + (prev iteration imag results)
            ymm15 = _mm256_fmadd_pd(ymm9, ymm3, ymm15);

            // (Ar*Xr Ai*Xr Ar*Xr Ai*Xr) + (prev iteration real results)
            ymm14 = _mm256_fmadd_pd(ymm10, ymm4, ymm14);
            // (Ar*Xi Ai*Xi Ar*Xi Ai*Xi) + + (prev iteration imag results)
            ymm15 = _mm256_fmadd_pd(ymm10, ymm5, ymm15);

            // (Ar*Xr Ai*Xr Ar*Xr Ai*Xr) + (prev iteration real results)
            ymm14 = _mm256_fmadd_pd(ymm11, ymm6, ymm14);
            // (Ar*Xi Ai*Xi Ar*Xi Ai*Xi) + + (prev iteration imag results)
            ymm15 = _mm256_fmadd_pd(ymm11, ymm7, ymm15);

            /*Permute the imag acc register to addsub to real accu results */
            // (Ar*Xi Ai*Xi Ar*Xi Ai*Xi) => (Ai*Xi Ar*Xi Ai*Xi Ar*Xi)
            ymm15 = _mm256_permute_pd(ymm15, 5);
            /*AddSub to get the 2 proper complex multipled value*/
            /* Ar*Xi - Ai*Xi, Ai*Xi + Ar*Xi, Ar*Xi - Ai*Xi, Ai*Xi + Ar*Xi*/
            ymm13 = _mm256_addsub_pd(ymm14, ymm15);

            // load Y vector
            ymm14 = _mm256_loadu_pd((double *)(y0 + (incy2)));
            // Add the results into y
            ymm13 = _mm256_add_pd(ymm14, ymm13);
            // Store the results back
            _mm256_storeu_pd((double*)(y0 + (incy2)), ymm13);
//-----------------------

            y0 += S_MR*incy ; // Next Set of y0 vector
            a0 += S_MR*inca ; // Next Set of a0 matrix elements in the same col
        }

        // For resisual m
        for( ; j < m ; ++j )
        {
            dcomplex       y0c = *(dcomplex*)y0;

            const dcomplex a0c = *a0;
            const dcomplex a1c = *(a0 + lda);
            const dcomplex a2c = *(a0 + lda2);
            const dcomplex a3c = *(a0 + lda3);

            y0c.real += chi0.real * a0c.real - chi0.imag * a0c.imag;
            y0c.real += chi1.real * a1c.real - chi1.imag * a1c.imag;
            y0c.real += chi2.real * a2c.real - chi2.imag * a2c.imag;
            y0c.real += chi3.real * a3c.real - chi3.imag * a3c.imag;

            y0c.imag += chi0.imag * a0c.real + chi0.real * a0c.imag;
            y0c.imag += chi1.imag * a1c.real + chi1.real * a1c.imag;
            y0c.imag += chi2.imag * a2c.real + chi2.real * a2c.imag;
            y0c.imag += chi3.imag * a3c.real + chi3.real * a3c.imag;

            *(dcomplex*)y0 = y0c;

            a0 += 1;
            y0 += 1;
        }
    }

    // For resisual n, axpyv is used
    for ( ; i < n; ++i )
    {
        dcomplex* a1   = a + (i  )*lda;
        dcomplex* chi1 = x + (i  )*incx;
        dcomplex* y1   = y;
        dcomplex  alpha_chi1;

        bli_zcopycjs( conjx, *chi1, alpha_chi1 );
        bli_zscals( *alpha, alpha_chi1 );

        bli_zaxpyv_zen_int5
        (
            conja,
            m,
            &alpha_chi1,
            a1, inca,
            y1, incy,
            cntx
        );
    }
}

/*
  This implementation uses 512 bits of cache line efficiently for
  column stored matrix and vectors.
  To achieve this, at each iteration we use 2 ymm registers
  i.e. .512 bits for arithmetic operation. By this we use the
  cache efficiently.
*/
void bli_cgemv_zen_int_4x4
(
    conj_t             conja,
    conj_t             conjx,
    dim_t              m,
    dim_t              n,
    scomplex* restrict alpha,
    scomplex* restrict a, inc_t inca, inc_t lda,
    scomplex* restrict x, inc_t incx,
    scomplex* restrict beta,
    scomplex* restrict y, inc_t incy,
    cntx_t*   restrict cntx
)
{

    const dim_t S_MR = 8; // Kernel size , m = 8
    const dim_t S_NR = 4; // Kernel size , n = 4

    scomplex            chi0;
    scomplex            chi1;
    scomplex            chi2;
    scomplex            chi3;

    inc_t lda2 = 2*lda;
    inc_t lda3 = 3*lda;
    inc_t incy4 = 4*incy;
    inc_t incx2 = 2*incx;
    inc_t incx3 = 3*incx;
    inc_t inca2 = 4*inca;

    scomplex* x0 = x;
    scomplex* y0 = y;
    scomplex* a0 = a;

    dim_t i,j;

    __m256 ymm0, ymm1, ymm2, ymm3;
    __m256 ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8, ymm9, ymm10, ymm11;
    __m256 ymm12, ymm13, ymm14, ymm15;

    for( i = 0; i+S_NR-1 < n; i+=S_NR )
    {
        a0 = a + (i  )*lda;
        x0 = x + (i  )*incx;
        y0 = y;// For each kernel, y should start form beginning

        chi0 = *( x0);
        chi1 = *( x0 + incx );
        chi2 = *( x0 + incx2 );
        chi3 = *( x0 + incx3 );

        bli_cscals( *alpha, chi0 );
        bli_cscals( *alpha, chi1 );
        bli_cscals( *alpha, chi2 );
        bli_cscals( *alpha, chi3 );

        ymm0 = _mm256_broadcast_ss(&chi0.real); // real part of x0
        ymm1 = _mm256_broadcast_ss(&chi0.imag); // imag part of x0
        ymm2 = _mm256_broadcast_ss(&chi1.real); // real part of x1
        ymm3 = _mm256_broadcast_ss(&chi1.imag); // imag part of x1
        ymm4 = _mm256_broadcast_ss(&chi2.real); // real part of x2
        ymm5 = _mm256_broadcast_ss(&chi2.imag); // imag part of x2
        ymm6 = _mm256_broadcast_ss(&chi3.real); // real part of x3
        ymm7 = _mm256_broadcast_ss(&chi3.imag); // imag part of x3

        for( j = 0 ; j+S_MR-1 < m ; j+=S_MR )
        {
            //load columns of A, each ymm reg had 4 elements
            ymm8  = _mm256_loadu_ps((float const *)(a0));
            ymm9  = _mm256_loadu_ps((float const *)(a0 +  lda));
            ymm10 = _mm256_loadu_ps((float const *)(a0 + lda2));
            ymm11 = _mm256_loadu_ps((float const *)(a0 + lda3));

            //--------------------
            //Ar*Xr Ai*Xr Ar*Xr Ai*Xr Ar*Xr Ai*Xr Ar*Xr Ai*Xr
            ymm14 = _mm256_mul_ps(ymm8, ymm0);
            //Ar*Xi Ai*Xi Ar*Xi Ai*Xi Ar*Xi Ai*Xi Ar*Xi Ai*Xi
            ymm15 = _mm256_mul_ps(ymm8, ymm1);

            /* Next set of A mult by real and imag,
            Add into the previous real and imag results */
            // (Ar*Xr Ai*Xr Ar*Xr Ai*Xr Ar*Xr Ai*Xr Ar*Xr Ai*Xr)
            // + (prev iteration real results)
            ymm14 = _mm256_fmadd_ps(ymm9, ymm2, ymm14);
            // (Ar*Xi Ai*Xi Ar*Xi Ai*Xi Ar*Xi Ai*Xi Ar*Xi Ai*Xi)
            // + (prev iteration imag results)
            ymm15 = _mm256_fmadd_ps(ymm9, ymm3, ymm15);
            // (Ar*Xr Ai*Xr Ar*Xr Ai*Xr Ar*Xr Ai*Xr Ar*Xr Ai*Xr)
            // + (prev iteration real results)
            ymm14 = _mm256_fmadd_ps(ymm10, ymm4, ymm14);
            // (Ar*Xi Ai*Xi Ar*Xi Ai*Xi Ar*Xi Ai*Xi Ar*Xi Ai*Xi)
            // + (prev iteration imag results)
            ymm15 = _mm256_fmadd_ps(ymm10, ymm5, ymm15);
            // (Ar*Xr Ai*Xr Ar*Xr Ai*Xr Ar*Xr Ai*Xr Ar*Xr Ai*Xr)
            // + (prev iteration real results)
            ymm14 = _mm256_fmadd_ps(ymm11, ymm6, ymm14);
            // (Ar*Xi Ai*Xi Ar*Xi Ai*Xi Ar*Xi Ai*Xi Ar*Xi Ai*Xi)
            // + (prev iteration imag results)
            ymm15 = _mm256_fmadd_ps(ymm11, ymm7, ymm15);
            /*Permute the imag acc register to addsub to real accu results */
            // (Ar*Xi Ai*Xi Ar*Xi Ai*Xi Ar*Xi Ai*Xi Ar*Xi Ai*Xi)
            // => (Ai*Xi Ar*Xi Ai*Xi Ar*Xi Ai*Xi Ar*Xi Ai*Xi Ar*Xi)
            ymm15 = _mm256_permute_ps(ymm15, 0xB1);
            /*AddSub to get the 2 proper complex multipled value*/
            /* Ar*Xi - Ai*Xi, Ai*Xi + Ar*Xi, Ar*Xi - Ai*Xi, Ai*Xi + Ar*Xi,
              Ar*Xi - Ai*Xi, Ai*Xi + Ar*Xi, Ar*Xi - Ai*Xi, Ai*Xi + Ar*Xi*/
            ymm12 = _mm256_addsub_ps(ymm14, ymm15);
            //load Y vector
            ymm14 = _mm256_loadu_ps((float*)y0);
            //Add the results into y
            ymm12 = _mm256_add_ps(ymm14, ymm12);
            // Store the results back
            _mm256_storeu_ps((float*)(y0), ymm12);

//-----------------------

            // Load Next Set of A matrix elements for the same col
            // Ar2 Ai2 Ar3 Ai3
            ymm8  = _mm256_loadu_ps((float const *)(a0 + (inca2)));
            ymm9  = _mm256_loadu_ps((float const *)(a0 + (inca2) + lda));
            ymm10 = _mm256_loadu_ps((float const *)(a0 + (inca2) + lda2));
            ymm11 = _mm256_loadu_ps((float const *)(a0 + (inca2) + lda3));

            //Ar0*Xr Ai0*Xr Ar1*Xr Ai1*Xr
            ymm14 = _mm256_mul_ps(ymm8, ymm0);
            //Ar0*Xi Ai0*Xi Ar1*Xi Ai1*Xi
            ymm15 = _mm256_mul_ps(ymm8, ymm1);

            /* Next set of A mult by real and imag,
            Add into the previous real and imag results */

            // (Ar*Xr Ai*Xr Ar*Xr Ai*Xr) + (prev iteration real results)
            ymm14 = _mm256_fmadd_ps(ymm9, ymm2, ymm14);
            // (Ar*Xi Ai*Xi Ar*Xi Ai*Xi) + + (prev iteration imag results)
            ymm15 = _mm256_fmadd_ps(ymm9, ymm3, ymm15);

            // (Ar*Xr Ai*Xr Ar*Xr Ai*Xr) + (prev iteration real results)
            ymm14 = _mm256_fmadd_ps(ymm10, ymm4, ymm14);
            // (Ar*Xi Ai*Xi Ar*Xi Ai*Xi) + + (prev iteration imag results)
            ymm15 = _mm256_fmadd_ps(ymm10, ymm5, ymm15);

            // (Ar*Xr Ai*Xr Ar*Xr Ai*Xr) + (prev iteration real results)
            ymm14 = _mm256_fmadd_ps(ymm11, ymm6, ymm14);
            // (Ar*Xi Ai*Xi Ar*Xi Ai*Xi) + + (prev iteration imag results)
            ymm15 = _mm256_fmadd_ps(ymm11, ymm7, ymm15);

            /*Permute the imag acc register to addsub to real accu results */
            // (Ar*Xi Ai*Xi Ar*Xi Ai*Xi) => (Ai*Xi Ar*Xi Ai*Xi Ar*Xi)
            ymm15 = _mm256_permute_ps(ymm15, 0xB1);
            /*AddSub to get the 2 proper complex multipled value*/
            /* Ar*Xi - Ai*Xi, Ai*Xi + Ar*Xi, Ar*Xi - Ai*Xi, Ai*Xi + Ar*Xi*/
            ymm13 = _mm256_addsub_ps(ymm14, ymm15);

            // load Y vector
            ymm14 = _mm256_loadu_ps((float *)(y0 + (incy4)));
            // Add the results into y
            ymm13 = _mm256_add_ps(ymm14, ymm13);
            // Store the results back
            _mm256_storeu_ps((float*)(y0 + (incy4)), ymm13);

            y0 += S_MR*incy ; // Next Set of y0 vector
            a0 += S_MR*inca ; // Next Set of a0 matrix elements in the same col
        }

        // For resisual m
        for( ; j < m ; ++j )
        {
            scomplex       y0c = *(scomplex*)y0;
            const scomplex a0c = *a0;
            const scomplex a1c = *(a0 + lda);
            const scomplex a2c = *(a0 + lda2);
            const scomplex a3c = *(a0 + lda3);

            y0c.real += chi0.real * a0c.real - chi0.imag * a0c.imag;
            y0c.real += chi1.real * a1c.real - chi1.imag * a1c.imag;
            y0c.real += chi2.real * a2c.real - chi2.imag * a2c.imag;
            y0c.real += chi3.real * a3c.real - chi3.imag * a3c.imag;

            y0c.imag += chi0.imag * a0c.real + chi0.real * a0c.imag;
            y0c.imag += chi1.imag * a1c.real + chi1.real * a1c.imag;
            y0c.imag += chi2.imag * a2c.real + chi2.real * a2c.imag;
            y0c.imag += chi3.imag * a3c.real + chi3.real * a3c.imag;

            *(scomplex*)y0 = y0c;
            a0 += 1;
            y0 += 1;
        }
    }

    // For resisual n, axpyv is used
    for ( ; i < n; ++i )
    {
        scomplex* a1   = a + (i  )*lda;
        scomplex* chi1 = x + (i  )*incx;
        scomplex* y1   = y;
        scomplex  alpha_chi1;
        bli_ccopycjs( conjx, *chi1, alpha_chi1 );
        bli_cscals( *alpha, alpha_chi1 );
        bli_caxpyv_zen_int5
        (
            conja,
            m,
            &alpha_chi1,
            a1, inca,
            y1, incy,
            cntx
        );
    }
}

