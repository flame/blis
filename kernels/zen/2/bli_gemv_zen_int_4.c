/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.

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

/* Union data structure to access AVX registers
   One 256-bit AVX register holds 8 SP elements. */
typedef union
{
	__m256  v;
	float   f[8] __attribute__((aligned(64)));
} v8sf_t;


/* Union data structure to access AVX registers
*  One 128-bit AVX register holds 4 SP elements. */
typedef union
{
	__m128 v;
	float  f[4] __attribute__((aligned(64)));
} v4sf_t;


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


/*
Function performs multithreaded GEMV for float datatype
All parameters are similar to single thread GEMV except
n_thread which specifies the number of threads to be used
*/
void bli_multi_sgemv_4x2
	(
       conj_t           conjat,
       conj_t           conjx,
       dim_t            m,
       dim_t            b_n,
       float*  restrict alpha,
       float*  restrict a, inc_t inca, inc_t lda,
       float*  restrict x, inc_t incx,
       float*  restrict beta,
       float*  restrict y, inc_t incy,
       cntx_t* restrict cntx,
       dim_t            n_threads
     )
{
    const dim_t b_fuse = 4;
    const dim_t n_elem_per_reg = 8;
    dim_t total_iteration = 0;

    // If the b_n dimension is zero, y is empty and there is no computation.
    if (bli_zero_dim1(b_n))
        return;

    // If the m dimension is zero, or if alpha is zero, the computation
    // simplifies to updating y.
    if (bli_zero_dim1(m) || PASTEMAC(s, eq0)(*alpha))
    {

        bli_sscalv_zen_int10(
            BLIS_NO_CONJUGATE,
            b_n,
            beta,
            y, incy,
            cntx);
        return;
    }

    // If b_n is not equal to the fusing factor, then perform the entire
    // operation as a loop over dotxv.
    if (b_n < b_fuse)
    {
        for (dim_t i = 0; i < b_n; ++i)
        {
            float *a1 = a + (0) * inca + (i)*lda;
            float *x1 = x + (0) * incx;
            float *psi1 = y + (i)*incy;

            bli_sdotxv_zen_int(
                conjat,
                conjx,
                m,
                alpha,
                a1, inca,
                x1, incx,
                beta,
                psi1,
                cntx);
        }
        return;
    }

    // Calculate the total number of multithreaded iteration
    total_iteration = b_n / b_fuse;

#pragma omp parallel for num_threads(n_threads)
    for (dim_t j = 0; j < total_iteration; j++)
    {
        float *A1 = a + (b_fuse * j) * lda;
        float *x1 = x;
        float *y1 = y + (b_fuse * j) * incy;

        // Intermediate variables to hold the completed dot products
        float rho0[4] = {0, 0, 0, 0};

        // If vectorization is possible, perform them with vector
        // instructions.
        if (inca == 1 && incx == 1)
        {
            const dim_t n_iter_unroll = 2;

            // Use the unrolling factor and the number of elements per register
            // to compute the number of vectorized and leftover iterations.
            dim_t l, unroll_inc, m_viter[2], m_left = m;

            unroll_inc = n_elem_per_reg * n_iter_unroll;

            m_viter[0] = m_left / unroll_inc;
            m_left = m_left % unroll_inc;

            m_viter[1] = m_left / n_elem_per_reg ;
            m_left = m_left % n_elem_per_reg;

            // Set up pointers for x and the b_n columns of A (rows of A^T).
            float *restrict x0 = x1;
            float *restrict av[4];

            av[0] = A1 + 0 * lda;
            av[1] = A1 + 1 * lda;
            av[2] = A1 + 2 * lda;
            av[3] = A1 + 3 * lda;

            // Initialize b_n rho vector accumulators to zero.
            v8sf_t rhov[4];

            rhov[0].v = _mm256_setzero_ps();
            rhov[1].v = _mm256_setzero_ps();
            rhov[2].v = _mm256_setzero_ps();
            rhov[3].v = _mm256_setzero_ps();

            v8sf_t xv[2];
            v8sf_t a_vec[8];

            // FMA operation is broken down to mul and add
            // to reduce backend stalls
            for (l = 0; l < m_viter[0]; ++l)
            {
                xv[0].v = _mm256_loadu_ps(x0);
                x0 += n_elem_per_reg;
                xv[1].v = _mm256_loadu_ps(x0);
                x0 += n_elem_per_reg;

                a_vec[0].v = _mm256_loadu_ps(av[0]);
                a_vec[4].v = _mm256_loadu_ps(av[0] + n_elem_per_reg);

                // perform: rho?v += a?v * x0v;
                a_vec[0].v = _mm256_mul_ps(a_vec[0].v, xv[0].v);
                rhov[0].v = _mm256_fmadd_ps(a_vec[4].v, xv[1].v, rhov[0].v);
                rhov[0].v = _mm256_add_ps(a_vec[0].v, rhov[0].v);

                a_vec[1].v = _mm256_loadu_ps(av[1]);
                a_vec[5].v = _mm256_loadu_ps(av[1] + n_elem_per_reg);

                a_vec[1].v = _mm256_mul_ps(a_vec[1].v, xv[0].v);
                rhov[1].v = _mm256_fmadd_ps(a_vec[5].v, xv[1].v, rhov[1].v);
                rhov[1].v = _mm256_add_ps(a_vec[1].v, rhov[1].v);

                a_vec[2].v = _mm256_loadu_ps(av[2]);
                a_vec[6].v = _mm256_loadu_ps(av[2] + n_elem_per_reg);

                a_vec[2].v = _mm256_mul_ps(a_vec[2].v, xv[0].v);
                rhov[2].v = _mm256_fmadd_ps(a_vec[6].v, xv[1].v, rhov[2].v);
                rhov[2].v = _mm256_add_ps(a_vec[2].v, rhov[2].v);

                a_vec[3].v = _mm256_loadu_ps(av[3]);
                a_vec[7].v = _mm256_loadu_ps(av[3] + n_elem_per_reg);

                a_vec[3].v = _mm256_mul_ps(a_vec[3].v, xv[0].v); 
                rhov[3].v = _mm256_fmadd_ps(a_vec[7].v, xv[1].v, rhov[3].v);
                rhov[3].v = _mm256_add_ps(a_vec[3].v, rhov[3].v);

                av[0] += unroll_inc;
                av[1] += unroll_inc;
                av[2] += unroll_inc;
                av[3] += unroll_inc;
            }

            for (l = 0; l < m_viter[1]; ++l)
            {
                // Load the input values.
                xv[0].v = _mm256_loadu_ps(x0);
                x0 += n_elem_per_reg;

                a_vec[0].v = _mm256_loadu_ps(av[0]);
                a_vec[1].v = _mm256_loadu_ps(av[1]);

                rhov[0].v = _mm256_fmadd_ps(a_vec[0].v, xv[0].v, rhov[0].v);
                rhov[1].v = _mm256_fmadd_ps(a_vec[1].v, xv[0].v, rhov[1].v);

                av[0] += n_elem_per_reg;
                av[1] += n_elem_per_reg;

                a_vec[2].v = _mm256_loadu_ps(av[2]);
                a_vec[3].v = _mm256_loadu_ps(av[3]);

                rhov[2].v = _mm256_fmadd_ps(a_vec[2].v, xv[0].v, rhov[2].v);
                rhov[3].v = _mm256_fmadd_ps(a_vec[3].v, xv[0].v, rhov[3].v);

                av[2] += n_elem_per_reg;
                av[3] += n_elem_per_reg;
            }

            // Sum the elements within each vector.
            // Sum the elements of a given rho?v with hadd.
            rhov[0].v = _mm256_hadd_ps(rhov[0].v, rhov[1].v);
            rhov[2].v = _mm256_hadd_ps(rhov[2].v, rhov[3].v);
            rhov[0].v = _mm256_hadd_ps(rhov[0].v, rhov[0].v);
            rhov[2].v = _mm256_hadd_ps(rhov[2].v, rhov[2].v);

            // Manually add the results from above to finish the sum.
            rho0[0] = rhov[0].f[0] + rhov[0].f[4];
            rho0[1] = rhov[0].f[1] + rhov[0].f[5];
            rho0[2] = rhov[2].f[0] + rhov[2].f[4];
            rho0[3] = rhov[2].f[1] + rhov[2].f[5];

            // If leftover elements are more than 4, perform SSE
            if (m_left > 4)
            {
                v4sf_t xv128, a_vec128[4], rhov128[4];

                rhov128[0].v = _mm_set1_ps(0);
                rhov128[1].v = _mm_set1_ps(0);
                rhov128[2].v = _mm_set1_ps(0);
                rhov128[3].v = _mm_set1_ps(0);

                // Load the input values.
                xv128.v = _mm_loadu_ps(x0 + 0 * n_elem_per_reg);
                x0 += 4;
                m_left -= 4;

                a_vec128[0].v = _mm_loadu_ps(av[0]);
                a_vec128[1].v = _mm_loadu_ps(av[1]);

                // perform: rho?v += a?v * x0v;
                rhov128[0].v = _mm_fmadd_ps(a_vec128[0].v, xv128.v, rhov128[0].v);
                rhov128[1].v = _mm_fmadd_ps(a_vec128[1].v, xv128.v, rhov128[1].v);
                rhov128[0].v = _mm_hadd_ps(rhov128[0].v, rhov128[1].v);
                rhov128[0].v = _mm_hadd_ps(rhov128[0].v, rhov128[0].v);

                a_vec128[2].v = _mm_loadu_ps(av[2]);
                a_vec128[3].v = _mm_loadu_ps(av[3]);

                rhov128[2].v = _mm_fmadd_ps(a_vec128[2].v, xv128.v, rhov128[2].v);
                rhov128[3].v = _mm_fmadd_ps(a_vec128[3].v, xv128.v, rhov128[3].v);
                rhov128[2].v = _mm_hadd_ps(rhov128[2].v, rhov128[3].v);
                rhov128[2].v = _mm_hadd_ps(rhov128[2].v, rhov128[2].v);

                rho0[0] += rhov128[0].f[0];
                rho0[1] += rhov128[0].f[1];
                rho0[2] += rhov128[2].f[0];
                rho0[3] += rhov128[2].f[1];

                av[0] += 4;
                av[1] += 4;
                av[2] += 4;
                av[3] += 4;
            }

            // If there are leftover iterations, perform them with scalar code.
            for (l = 0; l < m_left; ++l)
            {
                rho0[0] += *(av[0]) * (*x0);
                rho0[1] += *(av[1]) * (*x0);
                rho0[2] += *(av[2]) * (*x0);
                rho0[3] += *(av[3]) * (*x0);

                x0 += incx;
                av[0] += inca;
                av[1] += inca;
                av[2] += inca;
                av[3] += inca;
            }

        }
        else
        {
            // When vectorization is not possible, perform with scalar code

            // Initialize pointers for x and the b_n columns of A (rows of A^T).
            float *restrict x0 = x1;
            float *restrict a0 = A1 + 0 * lda;
            float *restrict a1 = A1 + 1 * lda;
            float *restrict a2 = A1 + 2 * lda;
            float *restrict a3 = A1 + 3 * lda;

            for (dim_t l = 0; l < m; ++l)
            {
                const float x0c = *x0;

                const float a0c = *a0;
                const float a1c = *a1;
                const float a2c = *a2;
                const float a3c = *a3;

                rho0[0] += a0c * x0c;
                rho0[1] += a1c * x0c;
                rho0[2] += a2c * x0c;
                rho0[3] += a3c * x0c;

                x0 += incx;
                a0 += inca;
                a1 += inca;
                a2 += inca;
                a3 += inca;
            }
        }

        v4sf_t rho0v, y0v;

        rho0v.v = _mm_loadu_ps(rho0);

        // Broadcast the alpha scalar.
        v4sf_t alphav;
        alphav.v = _mm_broadcast_ss(alpha);

        // We know at this point that alpha is nonzero; however, beta may still
        // be zero. If beta is indeed zero, we must overwrite y rather than scale
        // by beta (in case y contains NaN or Inf).
        if (PASTEMAC(s, eq0)(*beta))
        {
            // Apply alpha to the accumulated dot product in rho:
            //   y := alpha * rho
            y0v.v = _mm_mul_ps(alphav.v, rho0v.v);
        }
        else
        {
            // Broadcast the beta scalar.
            v4sf_t betav;
            betav.v = _mm_broadcast_ss(beta);

            if (incy == 0)
            {
                // Load y.
                y0v.v = _mm_loadu_ps(y1 + 0 * n_elem_per_reg);
            }
            else
            {
                // Load y.
                y0v.f[0] = *(y1 + 0 * incy);
                y0v.f[1] = *(y1 + 1 * incy);
                y0v.f[2] = *(y1 + 2 * incy);
                y0v.f[3] = *(y1 + 3 * incy);
            }

            // Apply beta to y and alpha to the accumulated dot product in rho:
            //   y := beta * y + alpha * rho
            y0v.v = _mm_mul_ps(betav.v, y0v.v);
            y0v.v = _mm_fmadd_ps(alphav.v, rho0v.v, y0v.v);
        }

        // Store the output.
        if (incy == 1)
        {
            _mm_storeu_ps((y1 + 0 * n_elem_per_reg), y0v.v);
        }
        else
        {
            // Store the output.
            *(y1 + 0 * incy) = y0v.f[0];
            *(y1 + 1 * incy) = y0v.f[1];
            *(y1 + 2 * incy) = y0v.f[2];
            *(y1 + 3 * incy) = y0v.f[3];
        }
    }

    // Performs the complete computation if OpenMP is not enabled
    dim_t start = total_iteration * b_fuse;
    dim_t new_fuse = 8, f;

    // Left over corner cases completed using fused kernel
    for (dim_t i = start; i < b_n; i += f)
    {
        f = bli_determine_blocksize_dim_f(i, b_n, new_fuse);

        float *A1 = a + (i)*lda + (0) * inca;
        float *x1 = x + (0) * incx;
        float *y1 = y + (i)*incy;

        /* y1 = beta * y1 + alpha * A1 * x; */
        bli_sdotxf_zen_int_8(
            conjat,
            conjx,
            m,
            f,
            alpha,
            A1, inca, lda,
            x1, incx,
            beta,
            y1, incy,
            cntx);
    }
}
