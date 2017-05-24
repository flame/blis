/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

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

#include "bli_avx_macros.h"

void bli_dpackm_12xk_avx
     (
       conj_t         conja,
       dim_t          n,
       void* restrict kappa_,
       void* restrict a_, inc_t inca, inc_t lda,
       void* restrict p_,             inc_t ldp
     )
{
    (void)conja;

    double* a = (double*)a_;
    double* p = (double*)p_;
    double kappa = *(double*)kappa_;

    if (inca == 1) /* no transpose */
    {
        if (kappa == 1.0)
        {
            /*
             * Unroll by 4
             */
            for (dim_t i = 0;i < n/4;i++)
            {
                copy_avx_d4x4(a+0, lda, p+0, ldp);
                copy_avx_d4x4(a+4, lda, p+4, ldp);
                copy_avx_d4x4(a+8, lda, p+8, ldp);

                a += 4*lda;
                p += 4*ldp;
            }
            /*
             * Remainder loop
             */
            for (dim_t i = 0;i < n%4;i++)
            {
                copy_avx_d4x1(a+0, p+0);
                copy_avx_d4x1(a+4, p+4);
                copy_avx_d4x1(a+8, p+8);

                a += lda;
                p += ldp;
            }
        }
        else
        {
            /*
             * Unroll by 4
             */
            for (dim_t i = 0;i < n/4;i++)
            {
                copy_scale_avx_d4x4(kappa, a+0, lda, p+0, ldp);
                copy_scale_avx_d4x4(kappa, a+4, lda, p+4, ldp);
                copy_scale_avx_d4x4(kappa, a+8, lda, p+8, ldp);

                a += 4*lda;
                p += 4*ldp;
            }
            /*
             * Remainder loop
             */
            for (dim_t i = 0;i < n%4;i++)
            {
                copy_scale_avx_d4x1(kappa, a+0, p+0);
                copy_scale_avx_d4x1(kappa, a+4, p+4);
                copy_scale_avx_d4x1(kappa, a+8, p+8);

                a += lda;
                p += ldp;
            }
        }
    }
    else if (lda == 1) /* transpose */
    {
        if (kappa == 1.0)
        {
            /*
             * Unroll by 4
             */
            for (dim_t i = 0;i < n/4;i++)
            {
                copy_trans_avx_d4x4(a+0*inca, inca, p+0, ldp);
                copy_trans_avx_d4x4(a+4*inca, inca, p+4, ldp);
                copy_trans_avx_d4x4(a+8*inca, inca, p+8, ldp);

                a += 4;
                p += 4*ldp;
            }
            /*
             * Remainder loop
             */
            for (dim_t i = 0;i < n%4;i++)
            {
                p[ 0] = a[ 0*inca];
                p[ 1] = a[ 1*inca];
                p[ 2] = a[ 2*inca];
                p[ 3] = a[ 3*inca];
                p[ 4] = a[ 4*inca];
                p[ 5] = a[ 5*inca];
                p[ 6] = a[ 6*inca];
                p[ 7] = a[ 7*inca];
                p[ 8] = a[ 8*inca];
                p[ 9] = a[ 9*inca];
                p[10] = a[10*inca];
                p[11] = a[11*inca];

                a++;
                p += ldp;
            }
        }
        else
        {
            /*
             * Unroll by 4
             */
            for (dim_t i = 0;i < n/4;i++)
            {
                copy_scale_trans_avx_d4x4(kappa, a+0*inca, inca, p+0, ldp);
                copy_scale_trans_avx_d4x4(kappa, a+4*inca, inca, p+4, ldp);
                copy_scale_trans_avx_d4x4(kappa, a+8*inca, inca, p+8, ldp);

                a += 4;
                p += 4*ldp;
            }
            /*
             * Remainder loop
             */
            for (dim_t i = 0;i < n%4;i++)
            {
                p[ 0] = kappa*a[ 0*inca];
                p[ 1] = kappa*a[ 1*inca];
                p[ 2] = kappa*a[ 2*inca];
                p[ 3] = kappa*a[ 3*inca];
                p[ 4] = kappa*a[ 4*inca];
                p[ 5] = kappa*a[ 5*inca];
                p[ 6] = kappa*a[ 6*inca];
                p[ 7] = kappa*a[ 7*inca];
                p[ 8] = kappa*a[ 8*inca];
                p[ 9] = kappa*a[ 9*inca];
                p[10] = kappa*a[10*inca];
                p[11] = kappa*a[11*inca];

                a++;
                p += ldp;
            }
        }
    }
    else /* general stride */
    {
        while (n --> 0)
        {
            p[ 0] = kappa*a[ 0*inca];
            p[ 1] = kappa*a[ 1*inca];
            p[ 2] = kappa*a[ 2*inca];
            p[ 3] = kappa*a[ 3*inca];
            p[ 4] = kappa*a[ 4*inca];
            p[ 5] = kappa*a[ 5*inca];
            p[ 6] = kappa*a[ 6*inca];
            p[ 7] = kappa*a[ 7*inca];
            p[ 8] = kappa*a[ 8*inca];
            p[ 9] = kappa*a[ 9*inca];
            p[10] = kappa*a[10*inca];
            p[11] = kappa*a[11*inca];

            a += lda;
            p += ldp;
        }
    }
}

void bli_dpackm_8xk_avx
     (
       conj_t         conja,
       dim_t          n,
       void* restrict kappa_,
       void* restrict a_, inc_t inca, inc_t lda,
       void* restrict p_,             inc_t ldp
     )
{
    (void)conja;

    double* a = (double*)a_;
    double* p = (double*)p_;
    double kappa = *(double*)kappa_;

    if (inca == 1) /* no transpose */
    {
        if (kappa == 1.0)
        {
            /*
             * Unroll by 4
             */
            for (dim_t i = 0;i < n/4;i++)
            {
                copy_avx_d4x4(a+0, lda, p+0, ldp);
                copy_avx_d4x4(a+4, lda, p+4, ldp);

                a += 4*lda;
                p += 4*ldp;
            }
            /*
             * Remainder loop
             */
            for (dim_t i = 0;i < n%4;i++)
            {
                copy_avx_d4x1(a+0, p+0);
                copy_avx_d4x1(a+4, p+4);

                a += lda;
                p += ldp;
            }
        }
        else
        {
            /*
             * Unroll by 4
             */
            for (dim_t i = 0;i < n/4;i++)
            {
                copy_scale_avx_d4x4(kappa, a+0, lda, p+0, ldp);
                copy_scale_avx_d4x4(kappa, a+4, lda, p+4, ldp);

                a += 4*lda;
                p += 4*ldp;
            }
            /*
             * Remainder loop
             */
            for (dim_t i = 0;i < n%4;i++)
            {
                copy_scale_avx_d4x1(kappa, a+0, p+0);
                copy_scale_avx_d4x1(kappa, a+4, p+4);

                a += lda;
                p += ldp;
            }
        }
    }
    else if (lda == 1) /* transpose */
    {
        if (kappa == 1.0)
        {
            /*
             * Unroll by 4
             */
            for (dim_t i = 0;i < n/4;i++)
            {
                copy_trans_avx_d4x4(a+0*inca, inca, p+0, ldp);
                copy_trans_avx_d4x4(a+4*inca, inca, p+4, ldp);

                a += 4;
                p += 4*ldp;
            }
            /*
             * Remainder loop
             */
            for (dim_t i = 0;i < n%4;i++)
            {
                p[0] = a[0*inca];
                p[1] = a[1*inca];
                p[2] = a[2*inca];
                p[3] = a[3*inca];
                p[4] = a[4*inca];
                p[5] = a[5*inca];
                p[6] = a[6*inca];
                p[7] = a[7*inca];

                a++;
                p += ldp;
            }
        }
        else
        {
            /*
             * Unroll by 4
             */
            for (dim_t i = 0;i < n/4;i++)
            {
                copy_scale_trans_avx_d4x4(kappa, a+0*inca, inca, p+0, ldp);
                copy_scale_trans_avx_d4x4(kappa, a+4*inca, inca, p+4, ldp);

                a += 4;
                p += 4*ldp;
            }
            /*
             * Remainder loop
             */
            for (dim_t i = 0;i < n%4;i++)
            {
                p[0] = kappa*a[0*inca];
                p[1] = kappa*a[1*inca];
                p[2] = kappa*a[2*inca];
                p[3] = kappa*a[3*inca];
                p[4] = kappa*a[4*inca];
                p[5] = kappa*a[5*inca];
                p[6] = kappa*a[6*inca];
                p[7] = kappa*a[7*inca];

                a++;
                p += ldp;
            }
        }
    }
    else /* general stride */
    {
        while (n --> 0)
        {
            p[0] = kappa*a[0*inca];
            p[1] = kappa*a[1*inca];
            p[2] = kappa*a[2*inca];
            p[3] = kappa*a[3*inca];
            p[4] = kappa*a[4*inca];
            p[5] = kappa*a[5*inca];
            p[6] = kappa*a[6*inca];
            p[7] = kappa*a[7*inca];

            a += lda;
            p += ldp;
        }
    }
}

void bli_dpackm_6xk_avx
     (
       conj_t         conja,
       dim_t          n,
       void* restrict kappa_,
       void* restrict a_, inc_t inca, inc_t lda,
       void* restrict p_,             inc_t ldp
     )
{
    (void)conja;

    double* a = (double*)a_;
    double* p = (double*)p_;
    double kappa = *(double*)kappa_;

    if (inca == 1) /* no transpose */
    {
        if (kappa == 1.0)
        {
            /*
             * Unroll by 4
             */
            for (dim_t i = 0;i < n/4;i++)
            {
                copy_avx_d4x4(a+0, lda, p+0, ldp);
                copy_avx_d2x4(a+4, lda, p+4, ldp);

                a += 4*lda;
                p += 4*ldp;
            }
            /*
             * Remainder loop
             */
            for (dim_t i = 0;i < n%4;i++)
            {
                copy_avx_d4x1(a+0, p+0);
                copy_avx_d2x1(a+4, p+4);

                a += lda;
                p += ldp;
            }
        }
        else
        {
            /*
             * Unroll by 4
             */
            for (dim_t i = 0;i < n/4;i++)
            {
                copy_scale_avx_d4x4(kappa, a+0, lda, p+0, ldp);
                copy_scale_avx_d2x4(kappa, a+4, lda, p+4, ldp);

                a += 4*lda;
                p += 4*ldp;
            }
            /*
             * Remainder loop
             */
            for (dim_t i = 0;i < n%4;i++)
            {
                copy_scale_avx_d4x1(kappa, a+0, p+0);
                copy_scale_avx_d2x1(kappa, a+4, p+4);

                a += lda;
                p += ldp;
            }
        }
    }
    else if (lda == 1) /* transpose */
    {
        if (kappa == 1.0)
        {
            /*
             * Unroll by 4
             */
            for (dim_t i = 0;i < n/4;i++)
            {
                copy_trans_avx_d4x4(a+0*inca, inca, p+0, ldp);
                copy_trans_avx_d4x2(a+4*inca, inca, p+4, ldp);

                a += 4;
                p += 4*ldp;
            }
            /*
             * Remainder loop
             */
            for (dim_t i = 0;i < n%4;i++)
            {
                p[0] = a[0*inca];
                p[1] = a[1*inca];
                p[2] = a[2*inca];
                p[3] = a[3*inca];
                p[4] = a[4*inca];
                p[5] = a[5*inca];

                a++;
                p += ldp;
            }
        }
        else
        {
            /*
             * Unroll by 4
             */
            for (dim_t i = 0;i < n/4;i++)
            {
                copy_scale_trans_avx_d4x4(kappa, a+0*inca, inca, p+0, ldp);
                copy_scale_trans_avx_d4x2(kappa, a+4*inca, inca, p+4, ldp);

                a += 4;
                p += 4*ldp;
            }
            /*
             * Remainder loop
             */
            for (dim_t i = 0;i < n%4;i++)
            {
                p[0] = kappa*a[0*inca];
                p[1] = kappa*a[1*inca];
                p[2] = kappa*a[2*inca];
                p[3] = kappa*a[3*inca];
                p[4] = kappa*a[4*inca];
                p[5] = kappa*a[5*inca];

                a++;
                p += ldp;
            }
        }
    }
    else /* general stride */
    {
        while (n --> 0)
        {
            p[0] = kappa*a[0*inca];
            p[1] = kappa*a[1*inca];
            p[2] = kappa*a[2*inca];
            p[3] = kappa*a[3*inca];
            p[4] = kappa*a[4*inca];
            p[5] = kappa*a[5*inca];

            a += lda;
            p += ldp;
        }
    }
}

void bli_dpackm_4xk_avx
     (
       conj_t         conja,
       dim_t          n,
       void* restrict kappa_,
       void* restrict a_, inc_t inca, inc_t lda,
       void* restrict p_,             inc_t ldp
     )
{
    (void)conja;

    double* a = (double*)a_;
    double* p = (double*)p_;
    double kappa = *(double*)kappa_;

    if (inca == 1) /* no transpose */
    {
        if (kappa == 1.0)
        {
            /*
             * Unroll by 4
             */
            for (dim_t i = 0;i < n/4;i++)
            {
                copy_avx_d4x4(a, lda, p, ldp);

                a += 4*lda;
                p += 4*ldp;
            }
            /*
             * Remainder loop
             */
            for (dim_t i = 0;i < n%4;i++)
            {
                copy_avx_d4x1(a, p);

                a += lda;
                p += ldp;
            }
        }
        else
        {
            /*
             * Unroll by 4
             */
            for (dim_t i = 0;i < n/4;i++)
            {
                copy_scale_avx_d4x4(kappa, a, lda, p, ldp);

                a += 4*lda;
                p += 4*ldp;
            }
            /*
             * Remainder loop
             */
            for (dim_t i = 0;i < n%4;i++)
            {
                copy_scale_avx_d4x1(kappa, a, p);

                a += lda;
                p += ldp;
            }
        }
    }
    else if (lda == 1) /* transpose */
    {
        if (kappa == 1.0)
        {
            /*
             * Unroll by 4
             */
            for (dim_t i = 0;i < n/4;i++)
            {
                copy_trans_avx_d4x4(a, inca, p, ldp);

                a += 4;
                p += 4*ldp;
            }
            /*
             * Remainder loop
             */
            for (dim_t i = 0;i < n%4;i++)
            {
                p[0] = a[0*inca];
                p[1] = a[1*inca];
                p[2] = a[2*inca];
                p[3] = a[3*inca];

                a++;
                p += ldp;
            }
        }
        else
        {
            /*
             * Unroll by 4
             */
            for (dim_t i = 0;i < n/4;i++)
            {
                copy_scale_trans_avx_d4x4(kappa, a, inca, p, ldp);

                a += 4;
                p += 4*ldp;
            }
            /*
             * Remainder loop
             */
            for (dim_t i = 0;i < n%4;i++)
            {
                p[0] = kappa*a[0*inca];
                p[1] = kappa*a[1*inca];
                p[2] = kappa*a[2*inca];
                p[3] = kappa*a[3*inca];

                a++;
                p += ldp;
            }
        }
    }
    else /* general stride */
    {
        while (n --> 0)
        {
            p[0] = kappa*a[0*inca];
            p[1] = kappa*a[1*inca];
            p[2] = kappa*a[2*inca];
            p[3] = kappa*a[3*inca];

            a += lda;
            p += ldp;
        }
    }
}
