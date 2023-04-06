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
#include <immintrin.h>

/*  This kernel performs  y := alpha * conjx(x)

    alpha   = a + i(b)
    X       = x + i(y)

    The computation is performed as follows:

    Step 1:
    -------

    alpha_real_register = broadcast(alpha_real)
    alpha_imag_register = broadcast(alpha_imag)

    x_register = load(x)

    Step 2:
    -------

    temp_storage_1 = x_register * alpha_real_register
    temp_storage_2 = x_register * alpha_imag_register

    temp_storage_1 = ax, ay
    temp_storage_2 = bx, by

    Step 3:
    -------

    In cases when X does NOT have to be conjugated,

        swap_adjacent_elements(temp_storage_2)
        temp_storage_2 = by, bx

    In case X has to be conjugated,

        swap_adjacent_elements(temp_storage_1)
        temp_storage_1 = ay, ax

    Step 4:
    -------

    In case of X conjugate the computation performed
    will be,

        result_compute = ax - by, ay + bx

    In case of where there is no need to conjugate X
    the computation performed will be,

        result_compute =  bx - ay, by + ax

    In case X has to be conjugated,

        swap_adjacent_elements(result_computed)

    Store the result to Y

    Exception
    ----------

    1. When the vector dimension is zero return early


    Compute reduction
    ------------------

    1. When alpha is zero (i.e. both real and imaginary are 0)
       perform the compute as setting Y vector to zero using
       setv
    2. When alpha is one (i.e. real is 1 and imaginary are 0)
       perform the compute as copying the X vector to Y vector
       using copyv.

    Underdefined
    -------------

    1. When incx or incy is passed as zero or less than zero,
       the behaviour is not defined. In this kernel, we return
       without performing any computation.
*/

void bli_zscal2v_zen_int
     (
       conj_t           conjx,
       dim_t            n,
       dcomplex*  restrict alpha,
       dcomplex*  restrict x, inc_t incx,
       dcomplex*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{

    // If the vector dimension is zero, return early.
    // When incx or incy is passed as zero or less than zero,
    // the behaviour is not defined, so return early.
    if (bli_zero_dim1(n)|| incx <= 0 || incy <=0)
        return;

    if (PASTEMAC(z, eq0)(*alpha))
    {
        /* If alpha is zero, use setv. */
        dcomplex *zero = PASTEMAC(z, 0);

        if(cntx == NULL) cntx = bli_gks_query_cntx();

        /* Query the context for the kernel function pointer. */
        const num_t dt = PASTEMAC(z, type);

        PASTECH(z, setv_ker_ft)
        setv_p = bli_cntx_get_l1v_ker_dt(dt, BLIS_SETV_KER, cntx);

        setv_p
        (
            BLIS_NO_CONJUGATE,
            n,
            zero,
            y, incy,
            cntx
        );

        return;
    }
    else if (PASTEMAC(z, eq1)(*alpha))
    {
        /* If alpha is one, use copyv. */
        bli_zcopyv_zen_int
        (
            conjx,
            n,
            x, incx,
            y, incy,
            cntx
        );

        return;
    }

    dim_t i;
    dcomplex *x0 = x;
    dcomplex *y0 = y;

    double real = (*alpha).real;
    double imag = (*alpha).imag;

    if (bli_is_noconj(conjx))
    {
        if (incx == 1 && incy == 1)
        {
            __m256d temp[8], alpha_real, alpha_imag, x_vec[4];

            alpha_real = _mm256_set1_pd(real);
            alpha_imag = _mm256_set1_pd(imag);

            const dim_t n_elem_per_reg = 2;

            for (i = 0; (i + 7) < n; i += 8)
            {
                x_vec[0] = _mm256_loadu_pd((double *)x0);
                x_vec[1] = _mm256_loadu_pd((double *)(x0 + n_elem_per_reg));
                x_vec[2] = _mm256_loadu_pd((double *)(x0 + 2 * n_elem_per_reg));
                x_vec[3] = _mm256_loadu_pd((double *)(x0 + 3 * n_elem_per_reg));

                temp[0] = _mm256_mul_pd(x_vec[0], alpha_real);
                temp[1] = _mm256_mul_pd(x_vec[0], alpha_imag);
                temp[2] = _mm256_mul_pd(x_vec[1], alpha_real);
                temp[3] = _mm256_mul_pd(x_vec[1], alpha_imag);
                temp[4] = _mm256_mul_pd(x_vec[2], alpha_real);
                temp[5] = _mm256_mul_pd(x_vec[2], alpha_imag);
                temp[6] = _mm256_mul_pd(x_vec[3], alpha_real);
                temp[7] = _mm256_mul_pd(x_vec[3], alpha_imag);

                temp[1] = _mm256_permute_pd(temp[1], 0b0101);
                temp[3] = _mm256_permute_pd(temp[3], 0b0101);
                temp[5] = _mm256_permute_pd(temp[5], 0b0101);
                temp[7] = _mm256_permute_pd(temp[7], 0b0101);

                temp[0] = _mm256_addsub_pd(temp[0], temp[1]);
                temp[2] = _mm256_addsub_pd(temp[2], temp[3]);
                temp[4] = _mm256_addsub_pd(temp[4], temp[5]);
                temp[6] = _mm256_addsub_pd(temp[6], temp[7]);

                _mm256_storeu_pd((double *)y0, temp[0]);
                _mm256_storeu_pd((double *)(y0 + n_elem_per_reg), temp[2]);
                _mm256_storeu_pd((double *)(y0 + 2 * n_elem_per_reg), temp[4]);
                _mm256_storeu_pd((double *)(y0 + 3 * n_elem_per_reg), temp[6]);

                x0 += 8;
                y0 += 8;
            }

            for (; (i + 3) < n; i += 4)
            {
                x_vec[0] = _mm256_loadu_pd((double *)x0);
                x_vec[1] = _mm256_loadu_pd((double *)(x0 + n_elem_per_reg));

                temp[0] = _mm256_mul_pd(x_vec[0], alpha_real);
                temp[1] = _mm256_mul_pd(x_vec[0], alpha_imag);
                temp[2] = _mm256_mul_pd(x_vec[1], alpha_real);
                temp[3] = _mm256_mul_pd(x_vec[1], alpha_imag);

                temp[1] = _mm256_permute_pd(temp[1], 0b0101);
                temp[3] = _mm256_permute_pd(temp[3], 0b0101);

                temp[0] = _mm256_addsub_pd(temp[0], temp[1]);
                temp[2] = _mm256_addsub_pd(temp[2], temp[3]);

                _mm256_storeu_pd((double *)y0, temp[0]);
                _mm256_storeu_pd((double *)(y0 + n_elem_per_reg), temp[2]);

                x0 += 4;
                y0 += 4;
            }

            for (; (i + 1) < n; i += 2)
            {
                x_vec[0] = _mm256_loadu_pd((double *)x0);

                temp[0] = _mm256_mul_pd(x_vec[0], alpha_real);
                temp[1] = _mm256_mul_pd(x_vec[0], alpha_imag);

                temp[1] = _mm256_permute_pd(temp[1], 0b0101);

                temp[0] = _mm256_addsub_pd(temp[0], temp[1]);

                _mm256_storeu_pd((double *)y0, temp[0]);

                x0 += 2;
                y0 += 2;
            }
            _mm256_zeroupper();
        }
        /* This else condition handles the computation when
           incx != 1 or incy ! = 1 for no conjugate X cases */
        else
        {
            /* In double complex data type the computation of
              unit stride elements can still be vectorized
              using SSE instructions */
            __m128d temp[8], alpha_real, alpha_imag, x_vec[4];

            alpha_real = _mm_set1_pd(real);
            alpha_imag = _mm_set1_pd(imag);

            for (i = 0; (i + 3) < n; i += 4)
            {
                x_vec[0] = _mm_loadu_pd((double *)x0);
                x_vec[1] = _mm_loadu_pd((double *)(x0 + incx));
                x_vec[2] = _mm_loadu_pd((double *)(x0 + 2 * incx));
                x_vec[3] = _mm_loadu_pd((double *)(x0 + 3 * incx));

                temp[0] = _mm_mul_pd(x_vec[0], alpha_real);
                temp[1] = _mm_mul_pd(x_vec[0], alpha_imag);
                temp[2] = _mm_mul_pd(x_vec[1], alpha_real);
                temp[3] = _mm_mul_pd(x_vec[1], alpha_imag);
                temp[4] = _mm_mul_pd(x_vec[2], alpha_real);
                temp[5] = _mm_mul_pd(x_vec[2], alpha_imag);
                temp[6] = _mm_mul_pd(x_vec[3], alpha_real);
                temp[7] = _mm_mul_pd(x_vec[3], alpha_imag);

                temp[1] = _mm_permute_pd(temp[1], 0b01);
                temp[3] = _mm_permute_pd(temp[3], 0b01);
                temp[5] = _mm_permute_pd(temp[5], 0b01);
                temp[7] = _mm_permute_pd(temp[7], 0b01);

                temp[0] = _mm_addsub_pd(temp[0], temp[1]);
                temp[2] = _mm_addsub_pd(temp[2], temp[3]);
                temp[4] = _mm_addsub_pd(temp[4], temp[5]);
                temp[6] = _mm_addsub_pd(temp[6], temp[7]);

                _mm_storeu_pd((double *)y0, temp[0]);
                _mm_storeu_pd((double *)(y0 + incy), temp[2]);
                _mm_storeu_pd((double *)(y0 + 2 * incy), temp[4]);
                _mm_storeu_pd((double *)(y0 + 3 * incy), temp[6]);

                x0 += 4 * incx;
                y0 += 4 * incy;
            }

            for (; (i + 1) < n; i += 2)
            {
                x_vec[0] = _mm_loadu_pd((double *)x0);
                x_vec[1] = _mm_loadu_pd((double *)(x0 + incx));

                temp[0] = _mm_mul_pd(x_vec[0], alpha_real);
                temp[1] = _mm_mul_pd(x_vec[0], alpha_imag);
                temp[2] = _mm_mul_pd(x_vec[1], alpha_real);
                temp[3] = _mm_mul_pd(x_vec[1], alpha_imag);

                temp[1] = _mm_permute_pd(temp[1], 0b01);
                temp[3] = _mm_permute_pd(temp[3], 0b01);

                temp[0] = _mm_addsub_pd(temp[0], temp[1]);
                temp[2] = _mm_addsub_pd(temp[2], temp[3]);

                _mm_storeu_pd((double *)y0, temp[0]);
                _mm_storeu_pd((double *)(y0 + incy), temp[2]);

                x0 += 2 * incx;
                y0 += 2 * incy;
            }
        }

        /* In double complex data type the computation of
            unit stride elements can still be vectorized
            using SSE instructions */
        __m128d temp[2], alpha_real, alpha_imag, x_vec[1];

        alpha_real = _mm_set1_pd(real);
        alpha_imag = _mm_set1_pd(imag);

        for (; i < n; i++)
        {
            x_vec[0] = _mm_loadu_pd((double *)x0);

            temp[0] = _mm_mul_pd(x_vec[0], alpha_real);
            temp[1] = _mm_mul_pd(x_vec[0], alpha_imag);

            temp[1] = _mm_permute_pd(temp[1], 0b01);

            temp[0] = _mm_addsub_pd(temp[0], temp[1]);

            _mm_storeu_pd((double *)y0, temp[0]);

            x0 += incx;
            y0 += incy;
        }
    }
    /* This else condition handles the computation
        for conjugate X cases */
    else
    {
        if (incx == 1 && incy == 1)
        {
            __m256d temp[8], alpha_real, alpha_imag, x_vec[4];

            alpha_real = _mm256_set1_pd(real);
            alpha_imag = _mm256_set1_pd(imag);

            const dim_t n_elem_per_reg = 2;

            for (i = 0; (i + 7) < n; i += 8)
            {
                x_vec[0] = _mm256_loadu_pd((double *)x0);
                x_vec[1] = _mm256_loadu_pd((double *)(x0 + n_elem_per_reg));
                x_vec[2] = _mm256_loadu_pd((double *)(x0 + 2 * n_elem_per_reg));
                x_vec[3] = _mm256_loadu_pd((double *)(x0 + 3 * n_elem_per_reg));

                temp[0] = _mm256_mul_pd(x_vec[0], alpha_real);
                temp[1] = _mm256_mul_pd(x_vec[0], alpha_imag);
                temp[2] = _mm256_mul_pd(x_vec[1], alpha_real);
                temp[3] = _mm256_mul_pd(x_vec[1], alpha_imag);
                temp[4] = _mm256_mul_pd(x_vec[2], alpha_real);
                temp[5] = _mm256_mul_pd(x_vec[2], alpha_imag);
                temp[6] = _mm256_mul_pd(x_vec[3], alpha_real);
                temp[7] = _mm256_mul_pd(x_vec[3], alpha_imag);

                temp[0] = _mm256_permute_pd(temp[0], 0b0101);
                temp[2] = _mm256_permute_pd(temp[2], 0b0101);
                temp[4] = _mm256_permute_pd(temp[4], 0b0101);
                temp[6] = _mm256_permute_pd(temp[6], 0b0101);

                temp[0] = _mm256_addsub_pd(temp[1], temp[0]);
                temp[2] = _mm256_addsub_pd(temp[3], temp[2]);
                temp[4] = _mm256_addsub_pd(temp[5], temp[4]);
                temp[6] = _mm256_addsub_pd(temp[7], temp[6]);

                temp[0] = _mm256_permute_pd(temp[0], 0b0101);
                temp[2] = _mm256_permute_pd(temp[2], 0b0101);
                temp[4] = _mm256_permute_pd(temp[4], 0b0101);
                temp[6] = _mm256_permute_pd(temp[6], 0b0101);

                _mm256_storeu_pd((double *)y0, temp[0]);
                _mm256_storeu_pd((double *)(y0 + n_elem_per_reg), temp[2]);
                _mm256_storeu_pd((double *)(y0 + 2 * n_elem_per_reg), temp[4]);
                _mm256_storeu_pd((double *)(y0 + 3 * n_elem_per_reg), temp[6]);

                x0 += 8;
                y0 += 8;
            }

            for (; (i + 3) < n; i += 4)
            {
                x_vec[0] = _mm256_loadu_pd((double *)x0);
                x_vec[1] = _mm256_loadu_pd((double *)(x0 + n_elem_per_reg));

                temp[0] = _mm256_mul_pd(x_vec[0], alpha_real);
                temp[1] = _mm256_mul_pd(x_vec[0], alpha_imag);
                temp[2] = _mm256_mul_pd(x_vec[1], alpha_real);
                temp[3] = _mm256_mul_pd(x_vec[1], alpha_imag);

                temp[0] = _mm256_permute_pd(temp[0], 0b0101);
                temp[2] = _mm256_permute_pd(temp[2], 0b0101);

                temp[0] = _mm256_addsub_pd(temp[1], temp[0]);
                temp[2] = _mm256_addsub_pd(temp[3], temp[2]);

                temp[0] = _mm256_permute_pd(temp[0], 0b0101);
                temp[2] = _mm256_permute_pd(temp[2], 0b0101);

                _mm256_storeu_pd((double *)y0, temp[0]);
                _mm256_storeu_pd((double *)(y0 + n_elem_per_reg), temp[2]);

                x0 += 4;
                y0 += 4;
            }

            for (; (i + 1) < n; i += 2)
            {
                x_vec[0] = _mm256_loadu_pd((double *)x0);

                temp[0] = _mm256_mul_pd(x_vec[0], alpha_real);
                temp[1] = _mm256_mul_pd(x_vec[0], alpha_imag);

                temp[0] = _mm256_permute_pd(temp[0], 0b0101);

                temp[0] = _mm256_addsub_pd(temp[1], temp[0]);

                temp[0] = _mm256_permute_pd(temp[0], 0b0101);

                _mm256_storeu_pd((double *)y0, temp[0]);

                x0 += 2;
                y0 += 2;
            }

            _mm256_zeroupper();
        }
        /* This else condition handles the computation when
           incx != 1 or incy != 1 for conjugate X cases */
        else
        {
            /* In double complex data type the computation of
            unit stride elements can still be vectorized
            using SSE instructions */
            __m128d temp[8], alpha_real, alpha_imag, x_vec[4];

            alpha_real = _mm_set1_pd(real);
            alpha_imag = _mm_set1_pd(imag);

            for (i = 0; (i + 3) < n; i += 4)
            {
                x_vec[0] = _mm_loadu_pd((double *)x0);
                x_vec[1] = _mm_loadu_pd((double *)(x0 + incx));
                x_vec[2] = _mm_loadu_pd((double *)(x0 + 2 * incx));
                x_vec[3] = _mm_loadu_pd((double *)(x0 + 3 * incx));

                temp[0] = _mm_mul_pd(x_vec[0], alpha_real);
                temp[1] = _mm_mul_pd(x_vec[0], alpha_imag);
                temp[2] = _mm_mul_pd(x_vec[1], alpha_real);
                temp[3] = _mm_mul_pd(x_vec[1], alpha_imag);
                temp[4] = _mm_mul_pd(x_vec[2], alpha_real);
                temp[5] = _mm_mul_pd(x_vec[2], alpha_imag);
                temp[6] = _mm_mul_pd(x_vec[3], alpha_real);
                temp[7] = _mm_mul_pd(x_vec[3], alpha_imag);

                temp[0] = _mm_permute_pd(temp[0], 0b01);
                temp[2] = _mm_permute_pd(temp[2], 0b01);
                temp[4] = _mm_permute_pd(temp[4], 0b01);
                temp[6] = _mm_permute_pd(temp[6], 0b01);

                temp[0] = _mm_addsub_pd(temp[1], temp[0]);
                temp[2] = _mm_addsub_pd(temp[3], temp[2]);
                temp[4] = _mm_addsub_pd(temp[5], temp[4]);
                temp[6] = _mm_addsub_pd(temp[7], temp[6]);

                temp[0] = _mm_permute_pd(temp[0], 0b01);
                temp[2] = _mm_permute_pd(temp[2], 0b01);
                temp[4] = _mm_permute_pd(temp[4], 0b01);
                temp[6] = _mm_permute_pd(temp[6], 0b01);

                _mm_storeu_pd((double *)y0, temp[0]);
                _mm_storeu_pd((double *)(y0 + incy), temp[2]);
                _mm_storeu_pd((double *)(y0 + 2 * incy), temp[4]);
                _mm_storeu_pd((double *)(y0 + 3 * incy), temp[6]);

                x0 += 4 * incx;
                y0 += 4 * incy;
            }

            for (; (i + 1) < n; i += 2)
            {
                x_vec[0] = _mm_loadu_pd((double *)x0);
                x_vec[1] = _mm_loadu_pd((double *)(x0 + incx));

                temp[0] = _mm_mul_pd(x_vec[0], alpha_real);
                temp[1] = _mm_mul_pd(x_vec[0], alpha_imag);
                temp[2] = _mm_mul_pd(x_vec[1], alpha_real);
                temp[3] = _mm_mul_pd(x_vec[1], alpha_imag);

                temp[0] = _mm_permute_pd(temp[0], 0b01);
                temp[2] = _mm_permute_pd(temp[2], 0b01);

                temp[0] = _mm_addsub_pd(temp[1], temp[0]);
                temp[2] = _mm_addsub_pd(temp[3], temp[2]);

                temp[0] = _mm_permute_pd(temp[0], 0b01);
                temp[2] = _mm_permute_pd(temp[2], 0b01);

                _mm_storeu_pd((double *)y0, temp[0]);
                _mm_storeu_pd((double *)(y0 + incy), temp[2]);

                x0 += 2 * incx;
                y0 += 2 * incy;
            }
        }

        /* In double complex data type the computation of
          unit stride elements can still be vectorized */
        __m128d temp[2], alpha_real, alpha_imag, x_vec[1];

        alpha_real = _mm_set1_pd(real);
        alpha_imag = _mm_set1_pd(imag);

        for (; i < n; ++i)
        {
            x_vec[0] = _mm_loadu_pd((double *)x0);

            temp[0] = _mm_mul_pd(x_vec[0], alpha_real);
            temp[1] = _mm_mul_pd(x_vec[0], alpha_imag);

            temp[0] = _mm_permute_pd(temp[0], 0b01);

            temp[0] = _mm_addsub_pd(temp[1], temp[0]);

            temp[0] = _mm_permute_pd(temp[0], 0b01);

            _mm_storeu_pd((double *)y0, temp[0]);

            x0 += incx;
            y0 += incy;
        }
    }
}
