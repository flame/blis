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
*  One 256-bit AVX register holds 4 DP elements. */
typedef union
{
    __m256d v;
    double  d[4] __attribute__((aligned(64)));
} v4df_t;


void bli_caxpyf_zen_int_4
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            b_n,
       scomplex* restrict alpha,
       scomplex* restrict a, inc_t inca, inc_t lda,
       scomplex* restrict x, inc_t incx,
       scomplex* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    inc_t fuse_fac = 4;
    inc_t i;

    __m256 ymm0, ymm1, ymm2, ymm3;
    __m256 ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8, ymm10;
    __m256 ymm12, ymm13;

    float* ap[4];
    float* y0 = (float*)y;

    scomplex            chi0;
    scomplex            chi1;
    scomplex            chi2;
    scomplex            chi3;


    dim_t setPlusOne = 1;

    if ( bli_is_conj(conja) )
    {
        setPlusOne = -1;
    }
    // If either dimension is zero, or if alpha is zero, return early.
    if ( bli_zero_dim2( m, b_n ) || bli_ceq0( *alpha ) ) return;

    // If b_n is not equal to the fusing factor, then perform the entire
    // operation as a loop over axpyv.
    if ( b_n != fuse_fac )
    {
        caxpyv_ker_ft f = bli_cntx_get_l1v_ker_dt( BLIS_SCOMPLEX, BLIS_AXPYV_KER, cntx );

        for ( i = 0; i < b_n; ++i )
        {
            scomplex* a1   = a + (0  )*inca + (i  )*lda;
            scomplex* chi1 = x + (i  )*incx;
            scomplex* y1   = y + (0  )*incy;
            scomplex  alpha_chi1;

            bli_ccopycjs( conjx, *chi1, alpha_chi1 );
            bli_cscals( *alpha, alpha_chi1 );

            f
            (
              conja,
              m,
              &alpha_chi1,
              a1, inca,
              y1, incy,
              cntx
            );
        }

        return;
    }


    // At this point, we know that b_n is exactly equal to the fusing factor.
    if(bli_is_noconj(conjx))
    {
        chi0 = *( x + 0*incx );
        chi1 = *( x + 1*incx );
        chi2 = *( x + 2*incx );
        chi3 = *( x + 3*incx );
    }
    else
    {
        scomplex *pchi0 = x + 0*incx ;
        scomplex *pchi1 = x + 1*incx ;
        scomplex *pchi2 = x + 2*incx ;
        scomplex *pchi3 = x + 3*incx ;

        bli_ccopycjs( conjx, *pchi0, chi0 );
        bli_ccopycjs( conjx, *pchi1, chi1 );
        bli_ccopycjs( conjx, *pchi2, chi2 );
        bli_ccopycjs( conjx, *pchi3, chi3 );
    }

    // Scale each chi scalar by alpha.
    bli_cscals( *alpha, chi0 );
    bli_cscals( *alpha, chi1 );
    bli_cscals( *alpha, chi2 );
    bli_cscals( *alpha, chi3 );

    lda *= 2;
    incx *= 2;
    incy *= 2;
    inca *= 2;

    ap[0] = (float*)a;
    ap[1] = (float*)a + lda;
    ap[2] = ap[1] + lda;
    ap[3] = ap[2] + lda;

    if( inca == 2 && incy == 2 )
    {
        inc_t n1 = m >> 2;// div by 4
        inc_t n2 = m & 3;// mod by 4

        ymm12 = _mm256_setzero_ps();
        ymm13 = _mm256_setzero_ps();

            // broadcast real & imag parts of 4 elements of x
        ymm0 = _mm256_broadcast_ss(&chi0.real); // real part of x0
        ymm1 = _mm256_broadcast_ss(&chi0.imag); // imag part of x0
        ymm2 = _mm256_broadcast_ss(&chi1.real); // real part of x1
        ymm3 = _mm256_broadcast_ss(&chi1.imag); // imag part of x1
        ymm4 = _mm256_broadcast_ss(&chi2.real); // real part of x2
        ymm5 = _mm256_broadcast_ss(&chi2.imag); // imag part of x2
        ymm6 = _mm256_broadcast_ss(&chi3.real); // real part of x3
        ymm7 = _mm256_broadcast_ss(&chi3.imag); // imag part of x3

        for(i = 0; i < n1; i++)
        {
            //load first two columns of A
     	    ymm8  = _mm256_loadu_ps(ap[0] + 0);
            ymm10 = _mm256_loadu_ps(ap[1] + 0);

            ymm12 = _mm256_mul_ps(ymm8, ymm0);
            ymm13 = _mm256_mul_ps(ymm8, ymm1);

            ymm12 = _mm256_fmadd_ps(ymm10, ymm2, ymm12);
            ymm13 = _mm256_fmadd_ps(ymm10, ymm3, ymm13);

	    //load 3rd and 4th columns of A
            ymm8  = _mm256_loadu_ps(ap[2] + 0);
            ymm10 = _mm256_loadu_ps(ap[3] + 0);

            ymm12 = _mm256_fmadd_ps(ymm8, ymm4, ymm12);
            ymm13 = _mm256_fmadd_ps(ymm8, ymm5, ymm13);

            ymm12 = _mm256_fmadd_ps(ymm10, ymm6, ymm12);
            ymm13 = _mm256_fmadd_ps(ymm10, ymm7, ymm13);

	    //load Y vector
            ymm10 = _mm256_loadu_ps(y0 + 0);

            if(bli_is_noconj(conja))
            {
                //printf("Inside no conj if\n");
                ymm13 = _mm256_permute_ps(ymm13, 0xB1);
                ymm8 = _mm256_addsub_ps(ymm12, ymm13);
            }
            else
            {
                ymm12 = _mm256_permute_ps(ymm12, 0xB1);
                ymm8 = _mm256_addsub_ps(ymm13, ymm12);
                ymm8 = _mm256_permute_ps(ymm8, 0xB1);
            }

            ymm12 = _mm256_add_ps(ymm8, ymm10);

            _mm256_storeu_ps((float*)(y0), ymm12);

            y0 += 8;
            ap[0] += 8;
            ap[1] += 8;
            ap[2] += 8;
            ap[3] += 8;
        }

        // If there are leftover iterations, perform them with scalar code.

        for ( i = 0; (i + 0) < n2 ; ++i )
        {

            scomplex       y0c = *(scomplex*)y0;

            const scomplex a0c = *(scomplex*)ap[0];
            const scomplex a1c = *(scomplex*)ap[1];
            const scomplex a2c = *(scomplex*)ap[2];
            const scomplex a3c = *(scomplex*)ap[3];

            y0c.real += chi0.real * a0c.real - chi0.imag * a0c.imag * setPlusOne;
            y0c.real += chi1.real * a1c.real - chi1.imag * a1c.imag * setPlusOne;
            y0c.real += chi2.real * a2c.real - chi2.imag * a2c.imag * setPlusOne;
            y0c.real += chi3.real * a3c.real - chi3.imag * a3c.imag * setPlusOne;

            y0c.imag += chi0.imag * a0c.real + chi0.real * a0c.imag * setPlusOne;
            y0c.imag += chi1.imag * a1c.real + chi1.real * a1c.imag * setPlusOne;
            y0c.imag += chi2.imag * a2c.real + chi2.real * a2c.imag * setPlusOne;
            y0c.imag += chi3.imag * a3c.real + chi3.real * a3c.imag * setPlusOne;

            *(scomplex*)y0 = y0c;

            ap[0] += 2;
            ap[1] += 2;
            ap[2] += 2;
            ap[3] += 2;
            y0 += 2;
        }
    //PASTEMAC(c,fprintm)(stdout, "Y after A*x in axpyf",m, 1, (scomplex*)y, 1, 1, "%4.1f", "");

    }
    else
    {
        for (i = 0 ; (i + 0) < m ; ++i )
        {
            scomplex       y0c = *(scomplex*)y0;
            const scomplex a0c = *(scomplex*)ap[0];
            const scomplex a1c = *(scomplex*)ap[1];
            const scomplex a2c = *(scomplex*)ap[2];
            const scomplex a3c = *(scomplex*)ap[3];

            y0c.real += chi0.real * a0c.real - chi0.imag * a0c.imag * setPlusOne;
            y0c.real += chi1.real * a1c.real - chi1.imag * a1c.imag * setPlusOne;
            y0c.real += chi2.real * a2c.real - chi2.imag * a2c.imag * setPlusOne;
            y0c.real += chi3.real * a3c.real - chi3.imag * a3c.imag * setPlusOne;

            y0c.imag += chi0.imag * a0c.real + chi0.real * a0c.imag * setPlusOne;
            y0c.imag += chi1.imag * a1c.real + chi1.real * a1c.imag * setPlusOne;
            y0c.imag += chi2.imag * a2c.real + chi2.real * a2c.imag * setPlusOne;
            y0c.imag += chi3.imag * a3c.real + chi3.real * a3c.imag * setPlusOne;

            *(scomplex*)y0 = y0c;

            ap[0] += inca;
            ap[1] += inca;
            ap[2] += inca;
            ap[3] += inca;
            y0 += incy;
        }
    }
}


void bli_zaxpyf_zen_int_4
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            b_n,
       dcomplex* restrict alpha,
       dcomplex* restrict a, inc_t inca, inc_t lda,
       dcomplex* restrict x, inc_t incx,
       dcomplex* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    inc_t fuse_fac = 4;
    inc_t i;

    v4df_t ymm0, ymm1, ymm2, ymm3;
    v4df_t ymm4, ymm5, ymm6, ymm7;
    v4df_t ymm8, ymm10;
    v4df_t ymm12, ymm13;

    double* ap[4];
    double* y0 = (double*)y;

    dcomplex            chi0;
    dcomplex            chi1;
    dcomplex            chi2;
    dcomplex            chi3;

    dim_t setPlusOne = 1;

    if ( bli_is_conj(conja) )
    {
        setPlusOne = -1;
    }
    // If either dimension is zero, or if alpha is zero, return early.
    if ( bli_zero_dim2( m, b_n ) || bli_zeq0( *alpha ) ) return;

    // If b_n is not equal to the fusing factor, then perform the entire
    // operation as a loop over axpyv.
    if ( b_n != fuse_fac )
    {
        zaxpyv_ker_ft f = bli_cntx_get_l1v_ker_dt( BLIS_DCOMPLEX, BLIS_AXPYV_KER, cntx );

        for ( i = 0; i < b_n; ++i )
        {
            dcomplex* a1   = a + (0  )*inca + (i  )*lda;
            dcomplex* chi1 = x + (i  )*incx;
            dcomplex* y1   = y + (0  )*incy;
            dcomplex  alpha_chi1;

            bli_zcopycjs( conjx, *chi1, alpha_chi1 );
            bli_zscals( *alpha, alpha_chi1 );

            f
            (
              conja,
              m,
              &alpha_chi1,
              a1, inca,
              y1, incy,
              cntx
            );
        }

        return;
    }


    // At this point, we know that b_n is exactly equal to the fusing factor.
    if(bli_is_noconj(conjx))
    {
        chi0 = *( x + 0*incx );
        chi1 = *( x + 1*incx );
        chi2 = *( x + 2*incx );
        chi3 = *( x + 3*incx );
    }
    else
    {
        dcomplex *pchi0 = x + 0*incx ;
        dcomplex *pchi1 = x + 1*incx ;
        dcomplex *pchi2 = x + 2*incx ;
        dcomplex *pchi3 = x + 3*incx ;

        bli_zcopycjs( conjx, *pchi0, chi0 );
        bli_zcopycjs( conjx, *pchi1, chi1 );
        bli_zcopycjs( conjx, *pchi2, chi2 );
        bli_zcopycjs( conjx, *pchi3, chi3 );
    }

    // Scale each chi scalar by alpha.
    bli_zscals( *alpha, chi0 );
    bli_zscals( *alpha, chi1 );
    bli_zscals( *alpha, chi2 );
    bli_zscals( *alpha, chi3 );

    lda *= 2;
    incx *= 2;
    incy *= 2;
    inca *= 2;

    ap[0] = (double*)a;
    ap[1] = (double*)a + lda;
    ap[2] = ap[1] + lda;
    ap[3] = ap[2] + lda;

    if( inca == 2 && incy == 2 )
    {
        inc_t n1 = m >> 1; // Divide by 2
        inc_t n2 = m & 1; // % 2

        ymm12.v = _mm256_setzero_pd();
        ymm13.v = _mm256_setzero_pd();

        // broadcast real & imag parts of 4 elements of x
        ymm0.v = _mm256_broadcast_sd(&chi0.real); // real part of x0
        ymm1.v = _mm256_broadcast_sd(&chi0.imag); // imag part of x0
        ymm2.v = _mm256_broadcast_sd(&chi1.real); // real part of x1
        ymm3.v = _mm256_broadcast_sd(&chi1.imag); // imag part of x1
        ymm4.v = _mm256_broadcast_sd(&chi2.real); // real part of x2
        ymm5.v = _mm256_broadcast_sd(&chi2.imag); // imag part of x2
        ymm6.v = _mm256_broadcast_sd(&chi3.real); // real part of x3
        ymm7.v = _mm256_broadcast_sd(&chi3.imag); // imag part of x3


        for(i = 0; i < n1; i++)
        {
            //load first two columns of A
            ymm8.v  = _mm256_loadu_pd(ap[0] + 0); // 2 complex values form a0
            ymm10.v = _mm256_loadu_pd(ap[1] + 0); // 2 complex values form a0

            ymm12.v = _mm256_mul_pd(ymm8.v, ymm0.v);
            ymm13.v = _mm256_mul_pd(ymm8.v, ymm1.v);

            ymm12.v = _mm256_fmadd_pd(ymm10.v, ymm2.v, ymm12.v);
            ymm13.v = _mm256_fmadd_pd(ymm10.v, ymm3.v, ymm13.v);

            //load 3rd and 4th columns of A
            ymm8.v = _mm256_loadu_pd(ap[2] + 0);
            ymm10.v = _mm256_loadu_pd(ap[3] + 0);

            ymm12.v = _mm256_fmadd_pd(ymm8.v, ymm4.v, ymm12.v);
            ymm13.v = _mm256_fmadd_pd(ymm8.v, ymm5.v, ymm13.v);

            ymm12.v = _mm256_fmadd_pd(ymm10.v, ymm6.v, ymm12.v);
            ymm13.v = _mm256_fmadd_pd(ymm10.v, ymm7.v, ymm13.v);

            //load Y vector
            ymm10.v = _mm256_loadu_pd(y0 + 0);

            if(bli_is_noconj(conja))
            {
                ymm13.v = _mm256_permute_pd(ymm13.v, 5);
                ymm8.v = _mm256_addsub_pd(ymm12.v, ymm13.v);
            }
            else
            {
                ymm12.v = _mm256_permute_pd(ymm12.v, 5);
                ymm8.v = _mm256_addsub_pd(ymm13.v, ymm12.v);
                ymm8.v = _mm256_permute_pd(ymm8.v, 5);
            }

            ymm12.v = _mm256_add_pd(ymm8.v, ymm10.v);

            _mm256_storeu_pd((double*)(y0), ymm12.v);

            y0 += 4;
            ap[0] += 4;
            ap[1] += 4;
            ap[2] += 4;
            ap[3] += 4;
        }

        // If there are leftover iterations, perform them with scalar code.

        for ( i = 0; (i + 0) < n2 ; ++i )
        {
            dcomplex       y0c = *(dcomplex*)y0;

            const dcomplex a0c = *(dcomplex*)ap[0];
            const dcomplex a1c = *(dcomplex*)ap[1];
            const dcomplex a2c = *(dcomplex*)ap[2];
            const dcomplex a3c = *(dcomplex*)ap[3];

            y0c.real += chi0.real * a0c.real - chi0.imag * a0c.imag * setPlusOne;
            y0c.real += chi1.real * a1c.real - chi1.imag * a1c.imag * setPlusOne;
            y0c.real += chi2.real * a2c.real - chi2.imag * a2c.imag * setPlusOne;
            y0c.real += chi3.real * a3c.real - chi3.imag * a3c.imag * setPlusOne;

            y0c.imag += chi0.imag * a0c.real + chi0.real * a0c.imag * setPlusOne;
            y0c.imag += chi1.imag * a1c.real + chi1.real * a1c.imag * setPlusOne;
            y0c.imag += chi2.imag * a2c.real + chi2.real * a2c.imag * setPlusOne;
            y0c.imag += chi3.imag * a3c.real + chi3.real * a3c.imag * setPlusOne;

            *(dcomplex*)y0 = y0c;

            ap[0] += 2;
            ap[1] += 2;
            ap[2] += 2;
            ap[3] += 2;
            y0 += 2;
        }
    //PASTEMAC(c,fprintm)(stdout, "Y after A*x in axpyf",m, 1, (scomplex*)y, 1, 1, "%4.1f", "");

    }
    else
    {
        for (i = 0 ; (i + 0) < m ; ++i )
        {
            dcomplex       y0c = *(dcomplex*)y0;
            const dcomplex a0c = *(dcomplex*)ap[0];
            const dcomplex a1c = *(dcomplex*)ap[1];
            const dcomplex a2c = *(dcomplex*)ap[2];
            const dcomplex a3c = *(dcomplex*)ap[3];

            y0c.real += chi0.real * a0c.real - chi0.imag * a0c.imag * setPlusOne;
            y0c.real += chi1.real * a1c.real - chi1.imag * a1c.imag * setPlusOne;
            y0c.real += chi2.real * a2c.real - chi2.imag * a2c.imag * setPlusOne;
            y0c.real += chi3.real * a3c.real - chi3.imag * a3c.imag * setPlusOne;

            y0c.imag += chi0.imag * a0c.real + chi0.real * a0c.imag * setPlusOne;
            y0c.imag += chi1.imag * a1c.real + chi1.real * a1c.imag * setPlusOne;
            y0c.imag += chi2.imag * a2c.real + chi2.real * a2c.imag * setPlusOne;
            y0c.imag += chi3.imag * a3c.real + chi3.real * a3c.imag * setPlusOne;

            *(dcomplex*)y0 = y0c;

            ap[0] += inca;
            ap[1] += inca;
            ap[2] += inca;
            ap[3] += inca;
            y0 += incy;
        }
    }
}
