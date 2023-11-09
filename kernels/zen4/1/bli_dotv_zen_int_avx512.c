/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2016 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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
    Functionality
    -------------

    This function calculates the dot product of two vectors for
    type float.

    rho := conjx(x)^T * conjy(y)

    Function Signature
    -------------------

    * 'conjx' - Variable specified if x needs to be conjugated
    * 'conjy' - Variable specified if x needs to be conjugated
    * 'n' - Length of the array passed
    * 'x' - Float pointer pointing to an array
    * 'y' - Float pointer pointing to an array
    * 'incx' - Stride to point to the next element in x array
    * 'incy' - Stride to point to the next element in y array
    * 'cntx' - BLIS context object

    Exception
    ----------

    None

    Deviation from BLAS
    --------------------

    None

    Undefined behaviour
    -------------------

    1. The kernel results in undefined behaviour when n <= 0, incx <= 1 and incy <= 1.
       The expectation is that these are standard BLAS exceptions and should be handled in
       a higher layer
*/
void bli_sdotv_zen_int_avx512
     (
       conj_t           conjx,
       conj_t           conjy,
       dim_t            n,
       float*  restrict x, inc_t incx,
       float*  restrict y, inc_t incy,
       float*  restrict rho,
       cntx_t* restrict cntx
     )
{
    dim_t i = 0;

    // Initialize local pointers.
    float *restrict x0 = x;
    float *restrict y0 = y;

    float rho0 = 0.0f;

    if (incx == 1 && incy == 1)
    {
        const dim_t n_elem_per_reg = 16;

        __m512 xv[5];
        __m512 yv[5];
        __m512 rhov[5];

        rhov[0] = _mm512_setzero_ps();
        rhov[1] = _mm512_setzero_ps();
        rhov[2] = _mm512_setzero_ps();
        rhov[3] = _mm512_setzero_ps();
        rhov[4] = _mm512_setzero_ps();

        for (i = 0; (i + 79) < n; i += 80)
        {
            xv[0] = _mm512_loadu_ps(x0 + 0 * n_elem_per_reg);
            xv[1] = _mm512_loadu_ps(x0 + 1 * n_elem_per_reg);
            xv[2] = _mm512_loadu_ps(x0 + 2 * n_elem_per_reg);
            xv[3] = _mm512_loadu_ps(x0 + 3 * n_elem_per_reg);
            xv[4] = _mm512_loadu_ps(x0 + 4 * n_elem_per_reg);

            yv[0] = _mm512_loadu_ps(y0 + 0 * n_elem_per_reg);
            yv[1] = _mm512_loadu_ps(y0 + 1 * n_elem_per_reg);
            yv[2] = _mm512_loadu_ps(y0 + 2 * n_elem_per_reg);
            yv[3] = _mm512_loadu_ps(y0 + 3 * n_elem_per_reg);
            yv[4] = _mm512_loadu_ps(y0 + 4 * n_elem_per_reg);

            rhov[0] = _mm512_fmadd_ps(xv[0], yv[0], rhov[0]);
            rhov[1] = _mm512_fmadd_ps(xv[1], yv[1], rhov[1]);
            rhov[2] = _mm512_fmadd_ps(xv[2], yv[2], rhov[2]);
            rhov[3] = _mm512_fmadd_ps(xv[3], yv[3], rhov[3]);
            rhov[4] = _mm512_fmadd_ps(xv[4], yv[4], rhov[4]);

            x0 += 5 * n_elem_per_reg;
            y0 += 5 * n_elem_per_reg;
        }

        for (; (i + 31) < n; i += 32)
        {
            xv[0] = _mm512_loadu_ps(x0 + 0 * n_elem_per_reg);
            xv[1] = _mm512_loadu_ps(x0 + 1 * n_elem_per_reg);

            yv[0] = _mm512_loadu_ps(y0 + 0 * n_elem_per_reg);
            yv[1] = _mm512_loadu_ps(y0 + 1 * n_elem_per_reg);

            rhov[0] = _mm512_fmadd_ps(xv[0], yv[0], rhov[0]);
            rhov[1] = _mm512_fmadd_ps(xv[1], yv[1], rhov[1]);

            x0 += 2 * n_elem_per_reg;
            y0 += 2 * n_elem_per_reg;
        }

        for (; (i + 15) < n; i += 16)
        {
            xv[0] = _mm512_loadu_ps(x0 + 0 * n_elem_per_reg);

            yv[0] = _mm512_loadu_ps(y0 + 0 * n_elem_per_reg);

            rhov[0] = _mm512_fmadd_ps(xv[0], yv[0], rhov[0]);

            x0 += n_elem_per_reg;
            y0 += n_elem_per_reg;
        }

        __m256 temp[2];
        temp[0] = _mm256_setzero_ps();

        for (; (i + 7) < n; i += 8)
        {
            __m256 x_vec = _mm256_loadu_ps(x0 + 0 * n_elem_per_reg);

            __m256 y_vec = _mm256_loadu_ps(y0 + 0 * n_elem_per_reg);

            temp[0] = _mm256_fmadd_ps(x_vec, y_vec, temp[0]);

            x0 += 8;
            y0 += 8;
        }

        __m128 temp_128[2];
        temp_128[0] = _mm_setzero_ps();

        for (; (i + 3) < n; i += 4)
        {
            __m128 x_vec = _mm_loadu_ps(x0 + 0 * n_elem_per_reg);

            __m128 y_vec = _mm_loadu_ps(y0 + 0 * n_elem_per_reg);

            temp_128[0] = _mm_fmadd_ps(x_vec, y_vec, temp_128[0]);

            x0 += 4;
            y0 += 4;
        }

        // Add the results from above to finish the sum.
        rhov[0] = _mm512_add_ps(rhov[0], rhov[2]);
        rhov[1] = _mm512_add_ps(rhov[1], rhov[3]);

        rhov[0] = _mm512_add_ps(rhov[0], rhov[1]);
        rhov[0] = _mm512_add_ps(rhov[0], rhov[4]);

        temp[1] = _mm512_extractf32x8_ps(rhov[0], 0);
        temp[0] = _mm256_add_ps(temp[0], temp[1]);

        temp[1] = _mm512_extractf32x8_ps(rhov[0], 1);
        temp[0] = _mm256_add_ps(temp[0], temp[1]);

        temp_128[1] = _mm256_extractf32x4_ps(temp[0], 0);
        temp_128[0] = _mm_add_ps(temp_128[0], temp_128[1]);
        temp_128[1] = _mm256_extractf32x4_ps(temp[0], 1);
        temp_128[0] = _mm_add_ps(temp_128[0], temp_128[1]);

        rho0 = temp_128[0][0] + temp_128[0][1] + temp_128[0][2] + temp_128[0][3];
    }

    for (; i < n; ++i)
    {
        const float x0c = *x0;
        const float y0c = *y0;

        rho0 += x0c * y0c;

        x0 += incx;
        y0 += incy;
    }

    // Copy the final result into the output variable.
    PASTEMAC(s, copys)(rho0, *rho);
}

// -----------------------------------------------------------------------------

/*
    Functionality
    -------------

    This function calculates the dot product of two vectors for
    type double.

    rho := conjx(x)^T * conjy(y)

    Function Signature
    -------------------

    * 'conjx' - Variable specified if x needs to be conjugated
    * 'conjy' - Variable specified if x needs to be conjugated
    * 'n' - Length of the array passed
    * 'x' - Double pointer pointing to an array
    * 'y' - Double pointer pointing to an array
    * 'incx' - Stride to point to the next element in x array
    * 'incy' - Stride to point to the next element in y array
    * 'cntx' - BLIS context object

    Exception
    ----------

    None

    Deviation from BLAS
    --------------------

    None

    Undefined behaviour
    -------------------

    1. The kernel results in undefined behaviour when n <= 0, incx <= 1 and incy <= 1.
       The expectation is that these are standard BLAS exceptions and should be handled in
       a higher layer
*/
void bli_ddotv_zen_int_avx512
     (
       conj_t           conjx,
       conj_t           conjy,
       dim_t            n,
       double* restrict x, inc_t incx,
       double* restrict y, inc_t incy,
       double* restrict rho,
       cntx_t* restrict cntx
     )
{
    dim_t i = 0;

    // Initialize local pointers.
    double *restrict x0 = x;
    double *restrict y0 = y;

    double rho0 = 0.0;

    if (incx == 1 && incy == 1)
    {
        const dim_t n_elem_per_reg = 8;

        __m512d xv[5];
        __m512d yv[5];
        __m512d rhov[5];

        rhov[0] = _mm512_setzero_pd();
        rhov[1] = _mm512_setzero_pd();
        rhov[2] = _mm512_setzero_pd();
        rhov[3] = _mm512_setzero_pd();
        rhov[4] = _mm512_setzero_pd();

        for (i = 0; (i + 39) < n; i += 40)
        {
            xv[0] = _mm512_loadu_pd(x0 + 0 * n_elem_per_reg);
            xv[1] = _mm512_loadu_pd(x0 + 1 * n_elem_per_reg);
            xv[2] = _mm512_loadu_pd(x0 + 2 * n_elem_per_reg);
            xv[3] = _mm512_loadu_pd(x0 + 3 * n_elem_per_reg);
            xv[4] = _mm512_loadu_pd(x0 + 4 * n_elem_per_reg);

            yv[0] = _mm512_loadu_pd(y0 + 0 * n_elem_per_reg);
            yv[1] = _mm512_loadu_pd(y0 + 1 * n_elem_per_reg);
            yv[2] = _mm512_loadu_pd(y0 + 2 * n_elem_per_reg);
            yv[3] = _mm512_loadu_pd(y0 + 3 * n_elem_per_reg);
            yv[4] = _mm512_loadu_pd(y0 + 4 * n_elem_per_reg);

            rhov[0] = _mm512_fmadd_pd(xv[0], yv[0], rhov[0]);
            rhov[1] = _mm512_fmadd_pd(xv[1], yv[1], rhov[1]);
            rhov[2] = _mm512_fmadd_pd(xv[2], yv[2], rhov[2]);
            rhov[3] = _mm512_fmadd_pd(xv[3], yv[3], rhov[3]);
            rhov[4] = _mm512_fmadd_pd(xv[4], yv[4], rhov[4]);

            x0 += 5 * n_elem_per_reg;
            y0 += 5 * n_elem_per_reg;
        }

        for (; (i + 15) < n; i += 16)
        {
            xv[0] = _mm512_loadu_pd(x0 + 0 * n_elem_per_reg);
            xv[1] = _mm512_loadu_pd(x0 + 1 * n_elem_per_reg);

            yv[0] = _mm512_loadu_pd(y0 + 0 * n_elem_per_reg);
            yv[1] = _mm512_loadu_pd(y0 + 1 * n_elem_per_reg);

            rhov[0] = _mm512_fmadd_pd(xv[0], yv[0], rhov[0]);
            rhov[1] = _mm512_fmadd_pd(xv[1], yv[1], rhov[1]);

            x0 += 2 * n_elem_per_reg;
            y0 += 2 * n_elem_per_reg;
        }
        rhov[0] = _mm512_add_pd(rhov[0], rhov[2]);
        rhov[1] = _mm512_add_pd(rhov[1], rhov[3]);

        rhov[0] = _mm512_add_pd(rhov[0], rhov[4]);
        rhov[0] = _mm512_add_pd(rhov[0], rhov[1]);

        if((i + 7) < n)
        {
            xv[0] = _mm512_loadu_pd(x0);

            yv[0] = _mm512_loadu_pd(y0);

            rhov[0] = _mm512_fmadd_pd(xv[0], yv[0], rhov[0]);

            x0 += n_elem_per_reg;
            y0 += n_elem_per_reg;
            i += 8;
        }
        if(i < n)
        {
            // calculate mask based on remainder elements of vector
            // which are not in multiple of 8.
            // Here bitmask is prepared based on remainder elements
            // to load only required elements from memory into
            // vector register.
            //for example if n-i=3 case bitmask is prepared as following.
            //1 is shifted by n-i(3), mask becomes 0b1000.
            //substracting 1 from it makes mask 0b111 which states that
            //3 elements from memory are to be loaded into vector register.
            __mmask8 mask = (1 << (n-i)) - 1;
            rhov[1] = _mm512_setzero_pd();

            xv[0] = _mm512_mask_loadu_pd(rhov[1], mask, x0);

            yv[0] = _mm512_mask_loadu_pd(rhov[1], mask, y0);

            rhov[0] = _mm512_fmadd_pd(xv[0], yv[0], rhov[0]);

            x0 += (n-i);
            y0 += (n-i);
            i += (n-i);
        }
        rho0 = _mm512_reduce_add_pd(rhov[0]);
    }

    for (; i < n; ++i)
    {
        const double x0c = *x0;
        const double y0c = *y0;

        rho0 += x0c * y0c;

        x0 += incx;
        y0 += incy;
    }

    // Copy the final result into the output variable.
    PASTEMAC(d, copys)(rho0, *rho);
}
