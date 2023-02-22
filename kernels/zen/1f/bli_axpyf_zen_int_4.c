/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021-2022, Advanced Micro Devices, Inc. All rights reserved.

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


void bli_caxpyf_zen_int_4
     (
             conj_t  conja,
             conj_t  conjx,
             dim_t   m,
             dim_t   b_n,
       const void*   alpha0,
       const void*   a0, inc_t inca, inc_t lda,
       const void*   x0, inc_t incx,
             void*   y0, inc_t incy,
       const cntx_t* cntx
     )
{
	const scomplex* restrict alpha = alpha0;
	const scomplex* restrict a     = a0;
	const scomplex* restrict x     = x0;
	      scomplex* restrict y     = y0;

    inc_t fuse_fac = 4;
    inc_t i;

    __m256 ymm0, ymm1, ymm2, ymm3;
    __m256 ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8,       ymm10;
    __m256 ymm12, ymm13;

    float* ap[4];
    float* yp = (float*)y;

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
        if ( cntx == NULL ) cntx = ( cntx_t* )bli_gks_query_cntx();

        axpyv_ker_ft f = bli_cntx_get_ukr_dt( BLIS_SCOMPLEX, BLIS_AXPYV_KER, cntx );

        for ( i = 0; i < b_n; ++i )
        {
            const scomplex* restrict a1   = a + (0  )*inca + (i  )*lda;
            const scomplex* restrict chi1 = x + (i  )*incx;
                  scomplex* restrict y1   = y + (0  )*incy;
                  scomplex           alpha_chi1;

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
        const scomplex* restrict pchi0 = x + 0*incx ;
        const scomplex* restrict pchi1 = x + 1*incx ;
        const scomplex* restrict pchi2 = x + 2*incx ;
        const scomplex* restrict pchi3 = x + 3*incx ;

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
        inc_t n1 = m/4;
        inc_t n2 = m%4;

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
            ymm10 = _mm256_loadu_ps(yp + 0);

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

            _mm256_storeu_ps((float*)(yp), ymm12);

            yp += 8;
            ap[0] += 8;
            ap[1] += 8;
            ap[2] += 8;
            ap[3] += 8;
        }

        // If there are leftover iterations, perform them with scalar code.

        for ( i = 0; (i + 0) < n2 ; ++i )
        {

            scomplex       y0c = *(scomplex*)yp;

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

            *(scomplex*)yp = y0c;

            ap[0] += 2;
            ap[1] += 2;
            ap[2] += 2;
            ap[3] += 2;
            yp += 2;
        }
    //PASTEMAC(c,fprintm)(stdout, "Y after A*x in axpyf",m, 1, (scomplex*)y, 1, 1, "%4.1f", "");

    }
    else
    {
        for (i = 0 ; (i + 0) < m ; ++i )
        {
            scomplex       y0c = *(scomplex*)yp;
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

            *(scomplex*)yp = y0c;

            ap[0] += inca;
            ap[1] += inca;
            ap[2] += inca;
            ap[3] += inca;
            yp += incy;
        }
    }
}
