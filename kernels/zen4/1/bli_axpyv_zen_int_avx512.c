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

#include "immintrin.h"
#include "blis.h"

// -----------------------------------------------------------------------------

/*
    Functionality
    -------------

    This function calculates y := y + alpha * x where all three variables are of type
    float.

    Function Signature
    -------------------

    This function takes three float pointer as input, the correspending vector's stride
    and length. It uses the function parameters to return the output.

    * 'conjx' - Info about conjugation of x (This variable is not used in the kernel)
    * 'n' - Length of the array passed
    * 'alpha' - Float pointer to a scalar value
    * 'x' - Float pointer to an array
    * 'incx' - Stride to point to the next element in the array
    * 'y' - Float pointer to an array
    * 'incy' - Stride to point to the next element in the array
    * 'cntx' - BLIS context object

    Exception
    ----------

    None

    Deviation from BLAS
    --------------------

    None

    Undefined behaviour
    -------------------

    1. The kernel results in undefined behaviour when n <= 0, incx <= 0 and incy <= 0.
       The expectation is that these are standard BLAS exceptions and should be handled in
       a higher layer
*/
void bli_saxpyv_zen_int_avx512
     (
       conj_t           conjx,
       dim_t            n,
       float*  restrict alpha,
       float*  restrict x, inc_t incx,
       float*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    const int n_elem_per_reg = 16;

    dim_t i = 0;

    // Initialize local pointers.
    float *restrict x0 = x;
    float *restrict y0 = y;

    if (incx == 1 && incy == 1)
    {
        __m512 xv[8], yv[8], alphav;

        // Broadcast the alpha scalar to all elements of a vector register.
        alphav = _mm512_set1_ps(*alpha);

        for (i = 0; (i + 127) < n; i += 128)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_ps(x0 + 0 * n_elem_per_reg);
            xv[1] = _mm512_loadu_ps(x0 + 1 * n_elem_per_reg);
            xv[2] = _mm512_loadu_ps(x0 + 2 * n_elem_per_reg);
            xv[3] = _mm512_loadu_ps(x0 + 3 * n_elem_per_reg);

            yv[0] = _mm512_loadu_ps(y0 + 0 * n_elem_per_reg);
            yv[1] = _mm512_loadu_ps(y0 + 1 * n_elem_per_reg);
            yv[2] = _mm512_loadu_ps(y0 + 2 * n_elem_per_reg);
            yv[3] = _mm512_loadu_ps(y0 + 3 * n_elem_per_reg);

            // Perform y += alpha * x
            yv[0] = _mm512_fmadd_ps(xv[0], alphav, yv[0]);
            yv[1] = _mm512_fmadd_ps(xv[1], alphav, yv[1]);
            yv[2] = _mm512_fmadd_ps(xv[2], alphav, yv[2]);
            yv[3] = _mm512_fmadd_ps(xv[3], alphav, yv[3]);

            // Store updated y
            _mm512_storeu_ps((y0 + 0 * n_elem_per_reg), yv[0]);
            _mm512_storeu_ps((y0 + 1 * n_elem_per_reg), yv[1]);
            _mm512_storeu_ps((y0 + 2 * n_elem_per_reg), yv[2]);
            _mm512_storeu_ps((y0 + 3 * n_elem_per_reg), yv[3]);

            xv[4] = _mm512_loadu_ps(x0 + 4 * n_elem_per_reg);
            xv[5] = _mm512_loadu_ps(x0 + 5 * n_elem_per_reg);
            xv[6] = _mm512_loadu_ps(x0 + 6 * n_elem_per_reg);
            xv[7] = _mm512_loadu_ps(x0 + 7 * n_elem_per_reg);

            yv[4] = _mm512_loadu_ps(y0 + 4 * n_elem_per_reg);
            yv[5] = _mm512_loadu_ps(y0 + 5 * n_elem_per_reg);
            yv[6] = _mm512_loadu_ps(y0 + 6 * n_elem_per_reg);
            yv[7] = _mm512_loadu_ps(y0 + 7 * n_elem_per_reg);

            yv[4] = _mm512_fmadd_ps(xv[4], alphav, yv[4]);
            yv[5] = _mm512_fmadd_ps(xv[5], alphav, yv[5]);
            yv[6] = _mm512_fmadd_ps(xv[6], alphav, yv[6]);
            yv[7] = _mm512_fmadd_ps(xv[7], alphav, yv[7]);

            _mm512_storeu_ps((y0 + 7 * n_elem_per_reg), yv[7]);
            _mm512_storeu_ps((y0 + 6 * n_elem_per_reg), yv[6]);
            _mm512_storeu_ps((y0 + 5 * n_elem_per_reg), yv[5]);
            _mm512_storeu_ps((y0 + 4 * n_elem_per_reg), yv[4]);

            // Increment the pointer
            x0 += 8 * n_elem_per_reg;
            y0 += 8 * n_elem_per_reg;
        }

        for (; (i + 63) < n; i += 64)
        {
            xv[0] = _mm512_loadu_ps(x0 + 0 * n_elem_per_reg);
            xv[1] = _mm512_loadu_ps(x0 + 1 * n_elem_per_reg);
            xv[2] = _mm512_loadu_ps(x0 + 2 * n_elem_per_reg);
            xv[3] = _mm512_loadu_ps(x0 + 3 * n_elem_per_reg);

            yv[0] = _mm512_loadu_ps(y0 + 0 * n_elem_per_reg);
            yv[1] = _mm512_loadu_ps(y0 + 1 * n_elem_per_reg);
            yv[2] = _mm512_loadu_ps(y0 + 2 * n_elem_per_reg);
            yv[3] = _mm512_loadu_ps(y0 + 3 * n_elem_per_reg);

            yv[0] = _mm512_fmadd_ps(xv[0], alphav, yv[0]);
            yv[1] = _mm512_fmadd_ps(xv[1], alphav, yv[1]);
            yv[2] = _mm512_fmadd_ps(xv[2], alphav, yv[2]);
            yv[3] = _mm512_fmadd_ps(xv[3], alphav, yv[3]);

            _mm512_storeu_ps((y0 + 0 * n_elem_per_reg), yv[0]);
            _mm512_storeu_ps((y0 + 1 * n_elem_per_reg), yv[1]);
            _mm512_storeu_ps((y0 + 2 * n_elem_per_reg), yv[2]);
            _mm512_storeu_ps((y0 + 3 * n_elem_per_reg), yv[3]);

            x0 += 4 * n_elem_per_reg;
            y0 += 4 * n_elem_per_reg;
        }

        for (; (i + 31) < n; i += 32)
        {
            xv[0] = _mm512_loadu_ps(x0 + 0 * n_elem_per_reg);
            xv[1] = _mm512_loadu_ps(x0 + 1 * n_elem_per_reg);

            yv[0] = _mm512_loadu_ps(y0 + 0 * n_elem_per_reg);
            yv[1] = _mm512_loadu_ps(y0 + 1 * n_elem_per_reg);

            yv[0] = _mm512_fmadd_ps(xv[0], alphav, yv[0]);
            yv[1] = _mm512_fmadd_ps(xv[1], alphav, yv[1]);

            _mm512_storeu_ps((y0 + 0 * n_elem_per_reg), yv[0]);
            _mm512_storeu_ps((y0 + 1 * n_elem_per_reg), yv[1]);

            x0 += 2 * n_elem_per_reg;
            y0 += 2 * n_elem_per_reg;
        }

        for (; (i + 15) < n; i += 16)
        {
            xv[0] = _mm512_loadu_ps(x0);

            yv[0] = _mm512_loadu_ps(y0);

            yv[0] = _mm512_fmadd_ps(xv[0], alphav, yv[0]);

            _mm512_storeu_ps(y0, yv[0]);

            x0 += n_elem_per_reg;
            y0 += n_elem_per_reg;
        }

        // This loop uses AVX2 instructions
        for (; (i + 7) < n; i += 8)
        {
            __m256 x_vec = _mm256_loadu_ps(x0);

            __m256 y_vec = _mm256_loadu_ps(y0);

            y_vec = _mm256_fmadd_ps(x_vec, _mm256_set1_ps(*alpha), y_vec);

            _mm256_storeu_ps(y0, y_vec);

            x0 += 8;
            y0 += 8;
        }
    }

    /*
        This loop has two functions:
        1. Handles the remainder of n / 8 when incx and incy are 1.
        2. Performs the complete compute when incx or incy != 1
    */
    for (; i < n; i += 1)
    {
        *y0 += (*alpha) * (*x0);

        x0 += incx;
        y0 += incy;
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
}

// -----------------------------------------------------------------------------

/*
    Functionality
    -------------

    This function calculates y := y + alpha * x where all three variables are of type
    double.

    Function Signature
    -------------------

    This function takes three float pointer as input, the correspending vector's stride
    and length. It uses the function parameters to return the output.

    * 'conjx' - Info about conjugation of x (This variable is not used in the kernel)
    * 'n' - Length of the array passed
    * 'alpha' - Double pointer to a scalar value
    * 'x' - Double pointer to an array
    * 'incx' - Stride to point to the next element in the array
    * 'y' - Double pointer to an array
    * 'incy' - Stride to point to the next element in the array
    * 'cntx' - BLIS context object

    Exception
    ----------

    None

    Deviation from BLAS
    --------------------

    None

    Undefined behaviour
    -------------------

    1. The kernel results in undefined behaviour when n <= 0, incx <= 0 and incy <= 0.
       The expectation is that these are standard BLAS exceptions and should be handled in
       a higher layer
*/
void bli_daxpyv_zen_int_avx512
     (
       conj_t           conjx,
       dim_t            n,
       double*  restrict alpha,
       double*  restrict x, inc_t incx,
       double*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    const int n_elem_per_reg = 8;

    dim_t i = 0;

    // Initialize local pointers.
    double *restrict x0 = x;
    double *restrict y0 = y;

    if (incx == 1 && incy == 1)
    {
        __m512d xv[8], yv[8], alphav;

        // Broadcast the alpha scalar to all elements of a vector register.
        alphav = _mm512_set1_pd(*alpha);

        for (i = 0; (i + 63) < n; i += 64)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_pd(x0 + 0 * n_elem_per_reg);
            xv[1] = _mm512_loadu_pd(x0 + 1 * n_elem_per_reg);
            xv[2] = _mm512_loadu_pd(x0 + 2 * n_elem_per_reg);
            xv[3] = _mm512_loadu_pd(x0 + 3 * n_elem_per_reg);

            yv[0] = _mm512_loadu_pd(y0 + 0 * n_elem_per_reg);
            yv[1] = _mm512_loadu_pd(y0 + 1 * n_elem_per_reg);
            yv[2] = _mm512_loadu_pd(y0 + 2 * n_elem_per_reg);
            yv[3] = _mm512_loadu_pd(y0 + 3 * n_elem_per_reg);

            // Perform y += alpha * x
            yv[0] = _mm512_fmadd_pd(xv[0], alphav, yv[0]);
            yv[1] = _mm512_fmadd_pd(xv[1], alphav, yv[1]);
            yv[2] = _mm512_fmadd_pd(xv[2], alphav, yv[2]);
            yv[3] = _mm512_fmadd_pd(xv[3], alphav, yv[3]);

            // Store updated y
            _mm512_storeu_pd((y0 + 0 * n_elem_per_reg), yv[0]);
            _mm512_storeu_pd((y0 + 1 * n_elem_per_reg), yv[1]);
            _mm512_storeu_pd((y0 + 2 * n_elem_per_reg), yv[2]);
            _mm512_storeu_pd((y0 + 3 * n_elem_per_reg), yv[3]);

            xv[4] = _mm512_loadu_pd(x0 + 4 * n_elem_per_reg);
            xv[5] = _mm512_loadu_pd(x0 + 5 * n_elem_per_reg);
            xv[6] = _mm512_loadu_pd(x0 + 6 * n_elem_per_reg);
            xv[7] = _mm512_loadu_pd(x0 + 7 * n_elem_per_reg);

            yv[4] = _mm512_loadu_pd(y0 + 4 * n_elem_per_reg);
            yv[5] = _mm512_loadu_pd(y0 + 5 * n_elem_per_reg);
            yv[6] = _mm512_loadu_pd(y0 + 6 * n_elem_per_reg);
            yv[7] = _mm512_loadu_pd(y0 + 7 * n_elem_per_reg);

            yv[4] = _mm512_fmadd_pd(xv[4], alphav, yv[4]);
            yv[5] = _mm512_fmadd_pd(xv[5], alphav, yv[5]);
            yv[6] = _mm512_fmadd_pd(xv[6], alphav, yv[6]);
            yv[7] = _mm512_fmadd_pd(xv[7], alphav, yv[7]);

            _mm512_storeu_pd((y0 + 7 * n_elem_per_reg), yv[7]);
            _mm512_storeu_pd((y0 + 6 * n_elem_per_reg), yv[6]);
            _mm512_storeu_pd((y0 + 5 * n_elem_per_reg), yv[5]);
            _mm512_storeu_pd((y0 + 4 * n_elem_per_reg), yv[4]);

            x0 += 8 * n_elem_per_reg;
            y0 += 8 * n_elem_per_reg;
        }

        for (; (i + 31) < n; i += 32)
        {
            xv[0] = _mm512_loadu_pd(x0 + 0 * n_elem_per_reg);
            xv[1] = _mm512_loadu_pd(x0 + 1 * n_elem_per_reg);
            xv[2] = _mm512_loadu_pd(x0 + 2 * n_elem_per_reg);
            xv[3] = _mm512_loadu_pd(x0 + 3 * n_elem_per_reg);

            yv[0] = _mm512_loadu_pd(y0 + 0 * n_elem_per_reg);
            yv[1] = _mm512_loadu_pd(y0 + 1 * n_elem_per_reg);
            yv[2] = _mm512_loadu_pd(y0 + 2 * n_elem_per_reg);
            yv[3] = _mm512_loadu_pd(y0 + 3 * n_elem_per_reg);

            yv[0] = _mm512_fmadd_pd(xv[0], alphav, yv[0]);
            yv[1] = _mm512_fmadd_pd(xv[1], alphav, yv[1]);
            yv[2] = _mm512_fmadd_pd(xv[2], alphav, yv[2]);
            yv[3] = _mm512_fmadd_pd(xv[3], alphav, yv[3]);

            _mm512_storeu_pd((y0 + 0 * n_elem_per_reg), yv[0]);
            _mm512_storeu_pd((y0 + 1 * n_elem_per_reg), yv[1]);
            _mm512_storeu_pd((y0 + 2 * n_elem_per_reg), yv[2]);
            _mm512_storeu_pd((y0 + 3 * n_elem_per_reg), yv[3]);

            x0 += 4 * n_elem_per_reg;
            y0 += 4 * n_elem_per_reg;
        }

        for (; (i + 15) < n; i += 16)
        {
            xv[0] = _mm512_loadu_pd(x0 + 0 * n_elem_per_reg);
            xv[1] = _mm512_loadu_pd(x0 + 1 * n_elem_per_reg);

            yv[0] = _mm512_loadu_pd(y0 + 0 * n_elem_per_reg);
            yv[1] = _mm512_loadu_pd(y0 + 1 * n_elem_per_reg);

            yv[0] = _mm512_fmadd_pd(xv[0], alphav, yv[0]);
            yv[1] = _mm512_fmadd_pd(xv[1], alphav, yv[1]);

            _mm512_storeu_pd((y0 + 0 * n_elem_per_reg), yv[0]);
            _mm512_storeu_pd((y0 + 1 * n_elem_per_reg), yv[1]);

            x0 += 2 * n_elem_per_reg;
            y0 += 2 * n_elem_per_reg;
        }

        for (; (i + 7) < n; i += 8)
        {
            xv[0] = _mm512_loadu_pd(x0);

            yv[0] = _mm512_loadu_pd(y0);

            yv[0] = _mm512_fmadd_pd(xv[0], alphav, yv[0]);

            _mm512_storeu_pd(y0, yv[0]);

            x0 += n_elem_per_reg;
            y0 += n_elem_per_reg;
        }

        // This loop uses AVX2 instructions
        for (; (i + 3) < n; i += 4)
        {
            __m256d x_vec = _mm256_loadu_pd(x0);

            __m256d y_vec = _mm256_loadu_pd(y0);

            y_vec = _mm256_fmadd_pd(x_vec, _mm256_set1_pd(*alpha), y_vec);

            _mm256_storeu_pd(y0, y_vec);

            x0 += 4;
            y0 += 4;
        }
    }

    /*
        This loop has two functions:
        1. Handles the remainder of n / 4 when incx and incy are 1.
        2. Performs the complete compute when incx or incy != 1
    */
    for (; i < n; i += 1)
    {
        *y0 += (*alpha) * (*x0);

        x0 += incx;
        y0 += incy;
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
}
