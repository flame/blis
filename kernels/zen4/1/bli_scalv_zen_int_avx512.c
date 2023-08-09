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

/*
    Functionality
    -------------

    This function scales a single precision floating-point vector by an element of the
    same type.

    x := conjalpha(alpha) * x

    Function Signature
    -------------------

    * 'conjalpha' - Variable specified if alpha needs to be conjugated
    * 'n' - Length of the array passed
    * 'alpha' - Pointer to the element by which the vector is to be scaled
    * 'x' - Float pointer pointing to an array
    * 'incx' - Stride to point to the next element in the array
    * 'cntx' - BLIS context object

    Exception
    ----------

    None

    Deviation from BLAS
    --------------------

    None

    Undefined behaviour
    -------------------

    1. The kernel results in undefined behaviour when n <= 0 and incx <= 1. The expectation
       is that these are standard BLAS exceptions and should be handled in a higher layer.
*/
void bli_sscalv_zen_int_avx512
        (
          conj_t conjalpha,
          dim_t n,
          float *restrict alpha,
          float *restrict x, inc_t incx,
          cntx_t *restrict cntx
        )
{
    dim_t i = 0;
    float *restrict x0 = x;

    if (incx == 1)
    {
        // Number of float in AVX-512
        const dim_t n_elem_per_reg = 16;

        __m512 xv[8], alphav;
        alphav = _mm512_set1_ps(*alpha);

        for (i = 0; (i + 127) < n; i += 128)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_ps(x0 + 0 * n_elem_per_reg);
            xv[1] = _mm512_loadu_ps(x0 + 1 * n_elem_per_reg);
            xv[2] = _mm512_loadu_ps(x0 + 2 * n_elem_per_reg);
            xv[3] = _mm512_loadu_ps(x0 + 3 * n_elem_per_reg);
            xv[4] = _mm512_loadu_ps(x0 + 4 * n_elem_per_reg);
            xv[5] = _mm512_loadu_ps(x0 + 5 * n_elem_per_reg);
            xv[6] = _mm512_loadu_ps(x0 + 6 * n_elem_per_reg);
            xv[7] = _mm512_loadu_ps(x0 + 7 * n_elem_per_reg);

            // perform : x := alpha * x;
            xv[0] = _mm512_mul_ps(alphav, xv[0]);
            xv[1] = _mm512_mul_ps(alphav, xv[1]);
            xv[2] = _mm512_mul_ps(alphav, xv[2]);
            xv[3] = _mm512_mul_ps(alphav, xv[3]);

            _mm512_storeu_ps((x0 + 0 * n_elem_per_reg), xv[0]);
            _mm512_storeu_ps((x0 + 1 * n_elem_per_reg), xv[1]);
            _mm512_storeu_ps((x0 + 2 * n_elem_per_reg), xv[2]);
            _mm512_storeu_ps((x0 + 3 * n_elem_per_reg), xv[3]);

            xv[4] = _mm512_mul_ps(alphav, xv[4]);
            xv[5] = _mm512_mul_ps(alphav, xv[5]);
            xv[6] = _mm512_mul_ps(alphav, xv[6]);
            xv[7] = _mm512_mul_ps(alphav, xv[7]);

            _mm512_storeu_ps((x0 + 4 * n_elem_per_reg), xv[4]);
            _mm512_storeu_ps((x0 + 5 * n_elem_per_reg), xv[5]);
            _mm512_storeu_ps((x0 + 6 * n_elem_per_reg), xv[6]);
            _mm512_storeu_ps((x0 + 7 * n_elem_per_reg), xv[7]);

            x0 += 8 * n_elem_per_reg;
        }

        for (; (i + 63) < n; i += 64)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_ps(x0 + 0 * n_elem_per_reg);
            xv[1] = _mm512_loadu_ps(x0 + 1 * n_elem_per_reg);
            xv[2] = _mm512_loadu_ps(x0 + 2 * n_elem_per_reg);
            xv[3] = _mm512_loadu_ps(x0 + 3 * n_elem_per_reg);

            // perform : x := alpha * x;
            xv[0] = _mm512_mul_ps(alphav, xv[0]);
            xv[1] = _mm512_mul_ps(alphav, xv[1]);
            xv[2] = _mm512_mul_ps(alphav, xv[2]);
            xv[3] = _mm512_mul_ps(alphav, xv[3]);

            _mm512_storeu_ps((x0 + 0 * n_elem_per_reg), xv[0]);
            _mm512_storeu_ps((x0 + 1 * n_elem_per_reg), xv[1]);
            _mm512_storeu_ps((x0 + 2 * n_elem_per_reg), xv[2]);
            _mm512_storeu_ps((x0 + 3 * n_elem_per_reg), xv[3]);

            x0 += 4 * n_elem_per_reg;
        }

        for (; (i + 31) < n; i += 32)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_ps(x0 + 0 * n_elem_per_reg);
            xv[1] = _mm512_loadu_ps(x0 + 1 * n_elem_per_reg);

            // perform : x := alpha * x;
            xv[0] = _mm512_mul_ps(alphav, xv[0]);
            xv[1] = _mm512_mul_ps(alphav, xv[1]);

            _mm512_storeu_ps((x0 + 0 * n_elem_per_reg), xv[0]);
            _mm512_storeu_ps((x0 + 1 * n_elem_per_reg), xv[1]);

            x0 += 2 * n_elem_per_reg;
        }

        for (; (i + 15) < n; i += 16)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_ps(x0 + 0 * n_elem_per_reg);

            // perform : x := alpha * x;
            xv[0] = _mm512_mul_ps(alphav, xv[0]);

            _mm512_storeu_ps((x0 + 0 * n_elem_per_reg), xv[0]);

            x0 += n_elem_per_reg;
        }

        for (; (i + 7) < n; i += 8)
        {
            // Loading the input values
            __m256 x_vec = _mm256_loadu_ps(x0);

            // perform : x := alpha * x;
            x_vec = _mm256_mul_ps(_mm256_set1_ps(*alpha), x_vec);

            // Store the output.
            _mm256_storeu_ps(x0, x_vec);

            x0 += 8;
        }

        /*
            Issue vzeroupper instruction to clear upper lanes of ymm registers.
            This avoids a performance penalty caused by false dependencies when
            transitioning from from AVX to SSE instructions (which may occur
            later, especially if BLIS is compiled with -mfpmath=sse).
        */
        _mm256_zeroupper();

        for (; (i + 3) < n; i += 4)
        {
            // Loading the input values
            __m128 x_vec = _mm_loadu_ps(x0);

            // perform : x := alpha * x;
            x_vec = _mm_mul_ps(_mm_set1_ps(*alpha), x_vec);

            // Store the output.
            _mm_storeu_ps(x0, x_vec);

            x0 += 4;
        }
    }

    const float alphac = *alpha;

    for (; i < n; ++i)
    {
        *x0 *= alphac;

        x0 += incx;
    }
}

// --------------------------------------------------------------------------------------

/*
    Functionality
    -------------

    This function scales a double precision floating-point vector by an element of the
    same type.

    x := conjalpha(alpha) * x

    Function Signature
    -------------------

    * 'conjalpha' - Variable specified if alpha needs to be conjugated
    * 'n' - Length of the array passed
    * 'alpha' - Pointer to the element by which the vector is to be scaled
    * 'x' - Double pointer pointing to an array
    * 'incx' - Stride to point to the next element in the array
    * 'cntx' - BLIS context object

    Exception
    ----------

    None

    Deviation from BLAS
    --------------------

    None

    Undefined behaviour
    -------------------

    1. The kernel results in undefined behaviour when n <= 0 and incx <= 1. The expectation
       is that these are standard BLAS exceptions and should be handled in a higher layer.
*/
void bli_dscalv_zen_int_avx512
        (
          conj_t conjalpha,
          dim_t n,
          double *restrict alpha,
          double *restrict x, inc_t incx,
          cntx_t *restrict cntx
        )
{
    // If the vector dimension is zero, or if alpha is unit, return early.
    if (bli_zero_dim1(n) || PASTEMAC(d, eq1)(*alpha))
        return;

    // If alpha is zero, use setv.
    if (PASTEMAC(d, eq0)(*alpha))
    {
        double *zero = bli_d0;
        if (cntx == NULL) cntx = bli_gks_query_cntx();
        dsetv_ker_ft f = bli_cntx_get_l1v_ker_dt(BLIS_DOUBLE, BLIS_SETV_KER, cntx);

        f
        (
          BLIS_NO_CONJUGATE,
          n,
          zero,
          x, incx,
          cntx
        );

        return;
    }

    dim_t i = 0;
    double *restrict x0;

    // Initialize local pointers.
    x0 = x;

    if (incx == 1)
    {
        // Number of double in AVX-512
        const dim_t n_elem_per_reg = 8;

        __m512d alphav;
        alphav = _mm512_set1_pd(*alpha);
        __m512d xv[8];

        for (i = 0; (i + 63) < n; i += 64)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_pd(x0 + 0 * n_elem_per_reg);
            xv[1] = _mm512_loadu_pd(x0 + 1 * n_elem_per_reg);
            xv[2] = _mm512_loadu_pd(x0 + 2 * n_elem_per_reg);
            xv[3] = _mm512_loadu_pd(x0 + 3 * n_elem_per_reg);
            xv[4] = _mm512_loadu_pd(x0 + 4 * n_elem_per_reg);
            xv[5] = _mm512_loadu_pd(x0 + 5 * n_elem_per_reg);
            xv[6] = _mm512_loadu_pd(x0 + 6 * n_elem_per_reg);
            xv[7] = _mm512_loadu_pd(x0 + 7 * n_elem_per_reg);

            // perform : x := alpha * x;
            xv[0] = _mm512_mul_pd(alphav, xv[0]);
            xv[1] = _mm512_mul_pd(alphav, xv[1]);
            xv[2] = _mm512_mul_pd(alphav, xv[2]);
            xv[3] = _mm512_mul_pd(alphav, xv[3]);

            _mm512_storeu_pd((x0 + 0 * n_elem_per_reg), xv[0]);
            _mm512_storeu_pd((x0 + 1 * n_elem_per_reg), xv[1]);
            _mm512_storeu_pd((x0 + 2 * n_elem_per_reg), xv[2]);
            _mm512_storeu_pd((x0 + 3 * n_elem_per_reg), xv[3]);

            xv[4] = _mm512_mul_pd(alphav, xv[4]);
            xv[5] = _mm512_mul_pd(alphav, xv[5]);
            xv[6] = _mm512_mul_pd(alphav, xv[6]);
            xv[7] = _mm512_mul_pd(alphav, xv[7]);

            _mm512_storeu_pd((x0 + 4 * n_elem_per_reg), xv[4]);
            _mm512_storeu_pd((x0 + 5 * n_elem_per_reg), xv[5]);
            _mm512_storeu_pd((x0 + 6 * n_elem_per_reg), xv[6]);
            _mm512_storeu_pd((x0 + 7 * n_elem_per_reg), xv[7]);

            x0 += 8 * n_elem_per_reg;
        }

        for (; (i + 31) < n; i += 32)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_pd(x0 + 0 * n_elem_per_reg);
            xv[1] = _mm512_loadu_pd(x0 + 1 * n_elem_per_reg);
            xv[2] = _mm512_loadu_pd(x0 + 2 * n_elem_per_reg);
            xv[3] = _mm512_loadu_pd(x0 + 3 * n_elem_per_reg);

            // perform : x := alpha * x;
            xv[0] = _mm512_mul_pd(alphav, xv[0]);
            xv[1] = _mm512_mul_pd(alphav, xv[1]);
            xv[2] = _mm512_mul_pd(alphav, xv[2]);
            xv[3] = _mm512_mul_pd(alphav, xv[3]);

            _mm512_storeu_pd((x0 + 0 * n_elem_per_reg), xv[0]);
            _mm512_storeu_pd((x0 + 1 * n_elem_per_reg), xv[1]);
            _mm512_storeu_pd((x0 + 2 * n_elem_per_reg), xv[2]);
            _mm512_storeu_pd((x0 + 3 * n_elem_per_reg), xv[3]);

            x0 += 4 * n_elem_per_reg;
        }

        for (; (i + 15) < n; i += 16)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_pd(x0 + 0 * n_elem_per_reg);
            xv[1] = _mm512_loadu_pd(x0 + 1 * n_elem_per_reg);

            // perform : x := alpha * x;
            xv[0] = _mm512_mul_pd(alphav, xv[0]);
            xv[1] = _mm512_mul_pd(alphav, xv[1]);

            _mm512_storeu_pd((x0 + 0 * n_elem_per_reg), xv[0]);
            _mm512_storeu_pd((x0 + 1 * n_elem_per_reg), xv[1]);

            x0 += 2 * n_elem_per_reg;
        }

        for (; (i + 7) < n; i += 8)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_pd(x0 + 0 * n_elem_per_reg);

            // perform : x := alpha * x;
            xv[0] = _mm512_mul_pd(alphav, xv[0]);

            _mm512_storeu_pd((x0 + 0 * n_elem_per_reg), xv[0]);

            x0 += n_elem_per_reg;
        }

        for (; (i + 3) < n; i += 4)
        {
            // Loading the input values
            __m256d x_vec = _mm256_loadu_pd(x0);

            // perform : x := alpha * x;
            x_vec = _mm256_mul_pd(_mm256_set1_pd(*alpha), x_vec);

            // Store the output.
            _mm256_storeu_pd(x0, x_vec);

            x0 += 4;
        }

        /*
           Issue vzeroupper instruction to clear upper lanes of ymm registers.
           This avoids a performance penalty caused by false dependencies when
           transitioning from from AVX to SSE instructions (which may occur
           later, especially if BLIS is compiled with -mfpmath=sse).
       */
        _mm256_zeroupper();

        for (; (i + 1) < n; i += 2)
        {
            // Loading the input values
            __m128d x_vec = _mm_loadu_pd(x0);

            // perform : x := alpha * x;
            x_vec = _mm_mul_pd(_mm_set1_pd(*alpha), x_vec);

            // Store the output.
            _mm_storeu_pd(x0, x_vec);

            x0 += 2;
        }
    }

    const double alphac = *alpha;

    for (; i < n; ++i)
    {
        *x0 *= alphac;

        x0 += incx;
    }
}

/*
    Functionality
    -------------

    This function scales a double complex vector by an element of the
    type double.

    x := conjalpha(alpha) * x

    Function Signature
    -------------------

    * 'conjalpha' - Variable specified if alpha needs to be conjugated
    * 'n' - Length of the array passed
    * 'alpha' - Pointer to the element by which the vector is to be scaled
    * 'x' - Double complex pointer pointing to an array
    * 'incx' - Stride to point to the next element in the array
    * 'cntx' - BLIS context object

    Exception
    ----------

    None

    Deviation from BLAS
    --------------------

    None

    Undefined behaviour
    -------------------

    1. The kernel results in undefined behaviour when n <= 0 and incx <= 1. The expectation
       is that these are standard BLAS exceptions and should be handled in a higher layer.
*/
void bli_zdscalv_zen_int_avx512
     (
       conj_t           conjalpha,
       dim_t            n,
       dcomplex* restrict alpha,
       dcomplex* restrict x, inc_t incx,
       cntx_t* restrict cntx
     )
{
    /*
        This kernel only performs the computation
        when alpha is double from the BLAS layer
        alpha is passed as double complex to adhere
        to function pointer definition in BLIS
    */
    const double alphac = (*alpha).real;

    dim_t i = 0;

    double *restrict x0 = (double *)x;

    if (incx == 1)
    {
        __m512d alphav, xv[4];
        const dim_t n_elem_per_reg = 8; // number of elements per register

        alphav = _mm512_set1_pd(alphac);

        for (; (i + 15) < n; i += 16)
        {
            xv[0] = _mm512_loadu_pd(x0);
            xv[1] = _mm512_loadu_pd(x0 + n_elem_per_reg);
            xv[2] = _mm512_loadu_pd(x0 + 2 * n_elem_per_reg);
            xv[3] = _mm512_loadu_pd(x0 + 3 * n_elem_per_reg);

            xv[0] = _mm512_mul_pd(alphav, xv[0]);
            xv[1] = _mm512_mul_pd(alphav, xv[1]);
            xv[2] = _mm512_mul_pd(alphav, xv[2]);
            xv[3] = _mm512_mul_pd(alphav, xv[3]);

            _mm512_storeu_pd(x0, xv[0]);
            _mm512_storeu_pd(x0 + n_elem_per_reg, xv[1]);
            _mm512_storeu_pd(x0 + 2 * n_elem_per_reg, xv[2]);
            _mm512_storeu_pd(x0 + 3 * n_elem_per_reg, xv[3]);

            x0 += 4 * n_elem_per_reg;
        }

        for (; (i + 7) < n; i += 8)
        {
            xv[0] = _mm512_loadu_pd(x0);
            xv[1] = _mm512_loadu_pd(x0 + n_elem_per_reg);

            xv[0] = _mm512_mul_pd(alphav, xv[0]);
            xv[1] = _mm512_mul_pd(alphav, xv[1]);

            _mm512_storeu_pd(x0, xv[0]);
            _mm512_storeu_pd(x0 + n_elem_per_reg, xv[1]);

            x0 += 2 * n_elem_per_reg;
        }

        for (; (i + 3) < n; i += 4)
        {
            xv[0] = _mm512_loadu_pd(x0);

            xv[0] = _mm512_mul_pd(alphav, xv[0]);

            _mm512_storeu_pd(x0, xv[0]);

            x0 += n_elem_per_reg;
        }

        for (; (i + 1) < n; i += 2)
        {
            __m256d xv = _mm256_loadu_pd(x0);

            __m256d alphav = _mm256_set1_pd(alphac);

            xv = _mm256_mul_pd(alphav, xv);

            _mm256_storeu_pd(x0, xv);

            x0 += 4;
        }

        // Issue vzeroupper instruction to clear upper lanes of ymm registers.
        // This avoids a performance penalty caused by false dependencies when
        // transitioning from AVX to SSE instructions (which may occur as soon
        // as the n_left cleanup loop below if BLIS is compiled with
        // -mfpmath=sse).
        _mm256_zeroupper();
    }

    /* In double complex data type the computation of
    unit stride elements can still be vectorized using SSE*/
    __m128d alpha_reg, x_vec;

    alpha_reg = _mm_set1_pd((*alpha).real);

    for (; i < n; ++i)
    {
        x_vec = _mm_loadu_pd(x0);

        x_vec = _mm_mul_pd(x_vec, alpha_reg);

        _mm_storeu_pd(x0, x_vec);

        x0 += 2 * incx;
    }
}
