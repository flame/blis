/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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
BLIS_EXPORT_BLIS void bli_daxpyv_zen_int_avx512
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
void bli_zaxpyv_zen_int_avx512
     (
       conj_t           conjx,
       dim_t            n,
       dcomplex*  restrict alpha,
       dcomplex*  restrict x, inc_t incx,
       dcomplex*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    const int n_elem_per_reg = 8;

    dim_t i = 0;

    // Initialize local pointers.
    double *restrict x0 = (double *)x;
    double *restrict y0 = (double *)y;

    if (incx == 1 && incy == 1)
    {
        __m512d xv[8], yv[8], alphaRv, alphaIv;

        // Broadcast real and imag parts of alpha to separate registers
        alphaRv = _mm512_set1_pd(alpha->real);
        alphaIv = _mm512_set1_pd(alpha->imag);

        xv[0] = _mm512_setzero_pd();

        // Handle X conjugate by negating some elements of alphaRv/alphaIv
        if ( bli_is_noconj( conjx ) )
            alphaIv = _mm512_fmaddsub_pd(xv[0], xv[0], alphaIv);
        else
            alphaRv = _mm512_fmsubadd_pd(xv[0], xv[0], alphaRv);

        // To check if code has to go to masked load/store directly
        if ( n >= 4 )
        {
            for (; (i + 31) < n; i += 32)
            {
                // Loading elements from X
                xv[0] = _mm512_loadu_pd(x0 + 0 * n_elem_per_reg);
                xv[1] = _mm512_loadu_pd(x0 + 1 * n_elem_per_reg);
                xv[2] = _mm512_loadu_pd(x0 + 2 * n_elem_per_reg);
                xv[3] = _mm512_loadu_pd(x0 + 3 * n_elem_per_reg);

                // Loading elements from Y
                yv[0] = _mm512_loadu_pd(y0 + 0 * n_elem_per_reg);
                yv[1] = _mm512_loadu_pd(y0 + 1 * n_elem_per_reg);
                yv[2] = _mm512_loadu_pd(y0 + 2 * n_elem_per_reg);
                yv[3] = _mm512_loadu_pd(y0 + 3 * n_elem_per_reg);

                // Scale X with real-part of alpha and add to Y
                yv[0] = _mm512_fmadd_pd(alphaRv, xv[0], yv[0]);
                yv[1] = _mm512_fmadd_pd(alphaRv, xv[1], yv[1]);
                yv[2] = _mm512_fmadd_pd(alphaRv, xv[2], yv[2]);
                yv[3] = _mm512_fmadd_pd(alphaRv, xv[3], yv[3]);

                // Swapping real and imag parts of every element in X
                xv[0] = _mm512_permute_pd(xv[0], 0x55);
                xv[1] = _mm512_permute_pd(xv[1], 0x55);
                xv[2] = _mm512_permute_pd(xv[2], 0x55);
                xv[3] = _mm512_permute_pd(xv[3], 0x55);

                // Scale X with imag-part of alpha and add to Y
                yv[0] = _mm512_fmadd_pd(alphaIv, xv[0], yv[0]);
                yv[1] = _mm512_fmadd_pd(alphaIv, xv[1], yv[1]);
                yv[2] = _mm512_fmadd_pd(alphaIv, xv[2], yv[2]);
                yv[3] = _mm512_fmadd_pd(alphaIv, xv[3], yv[3]);

                // Store updated Y
                _mm512_storeu_pd((y0 + 0 * n_elem_per_reg), yv[0]);
                _mm512_storeu_pd((y0 + 1 * n_elem_per_reg), yv[1]);
                _mm512_storeu_pd((y0 + 2 * n_elem_per_reg), yv[2]);
                _mm512_storeu_pd((y0 + 3 * n_elem_per_reg), yv[3]);

                // Loading elements from X
                xv[4] = _mm512_loadu_pd(x0 + 4 * n_elem_per_reg);
                xv[5] = _mm512_loadu_pd(x0 + 5 * n_elem_per_reg);
                xv[6] = _mm512_loadu_pd(x0 + 6 * n_elem_per_reg);
                xv[7] = _mm512_loadu_pd(x0 + 7 * n_elem_per_reg);

                // Loading elements from Y
                yv[4] = _mm512_loadu_pd(y0 + 4 * n_elem_per_reg);
                yv[5] = _mm512_loadu_pd(y0 + 5 * n_elem_per_reg);
                yv[6] = _mm512_loadu_pd(y0 + 6 * n_elem_per_reg);
                yv[7] = _mm512_loadu_pd(y0 + 7 * n_elem_per_reg);

                // Scale X with real-part of alpha and add to Y
                yv[4] = _mm512_fmadd_pd(alphaRv, xv[4], yv[4]);
                yv[5] = _mm512_fmadd_pd(alphaRv, xv[5], yv[5]);
                yv[6] = _mm512_fmadd_pd(alphaRv, xv[6], yv[6]);
                yv[7] = _mm512_fmadd_pd(alphaRv, xv[7], yv[7]);

                // Swapping real and imag parts of every element in X
                xv[4] = _mm512_permute_pd(xv[4], 0x55);
                xv[5] = _mm512_permute_pd(xv[5], 0x55);
                xv[6] = _mm512_permute_pd(xv[6], 0x55);
                xv[7] = _mm512_permute_pd(xv[7], 0x55);

                // Scale X with imag-part of alpha and add to Y
                yv[4] = _mm512_fmadd_pd(alphaIv, xv[4], yv[4]);
                yv[5] = _mm512_fmadd_pd(alphaIv, xv[5], yv[5]);
                yv[6] = _mm512_fmadd_pd(alphaIv, xv[6], yv[6]);
                yv[7] = _mm512_fmadd_pd(alphaIv, xv[7], yv[7]);

                // Store updated Y
                _mm512_storeu_pd((y0 + 4 * n_elem_per_reg), yv[4]);
                _mm512_storeu_pd((y0 + 5 * n_elem_per_reg), yv[5]);
                _mm512_storeu_pd((y0 + 6 * n_elem_per_reg), yv[6]);
                _mm512_storeu_pd((y0 + 7 * n_elem_per_reg), yv[7]);

                x0 += 8 * n_elem_per_reg;
                y0 += 8 * n_elem_per_reg;
            }

            for (; (i + 15) < n; i += 16)
            {
                // Loading elements from X
                xv[0] = _mm512_loadu_pd(x0 + 0 * n_elem_per_reg);
                xv[1] = _mm512_loadu_pd(x0 + 1 * n_elem_per_reg);
                xv[2] = _mm512_loadu_pd(x0 + 2 * n_elem_per_reg);
                xv[3] = _mm512_loadu_pd(x0 + 3 * n_elem_per_reg);

                // Loading elements from Y
                yv[0] = _mm512_loadu_pd(y0 + 0 * n_elem_per_reg);
                yv[1] = _mm512_loadu_pd(y0 + 1 * n_elem_per_reg);
                yv[2] = _mm512_loadu_pd(y0 + 2 * n_elem_per_reg);
                yv[3] = _mm512_loadu_pd(y0 + 3 * n_elem_per_reg);

                // Scale X with real-part of alpha and add to Y
                yv[0] = _mm512_fmadd_pd(alphaRv, xv[0], yv[0]);
                yv[1] = _mm512_fmadd_pd(alphaRv, xv[1], yv[1]);
                yv[2] = _mm512_fmadd_pd(alphaRv, xv[2], yv[2]);
                yv[3] = _mm512_fmadd_pd(alphaRv, xv[3], yv[3]);

                // Swapping real and imag parts of every element in X
                xv[0] = _mm512_permute_pd(xv[0], 0x55);
                xv[1] = _mm512_permute_pd(xv[1], 0x55);
                xv[2] = _mm512_permute_pd(xv[2], 0x55);
                xv[3] = _mm512_permute_pd(xv[3], 0x55);

                // Scale X with imag-part of alpha and add to Y
                yv[0] = _mm512_fmadd_pd(alphaIv, xv[0], yv[0]);
                yv[1] = _mm512_fmadd_pd(alphaIv, xv[1], yv[1]);
                yv[2] = _mm512_fmadd_pd(alphaIv, xv[2], yv[2]);
                yv[3] = _mm512_fmadd_pd(alphaIv, xv[3], yv[3]);

                // Store updated Y
                _mm512_storeu_pd((y0 + 0 * n_elem_per_reg), yv[0]);
                _mm512_storeu_pd((y0 + 1 * n_elem_per_reg), yv[1]);
                _mm512_storeu_pd((y0 + 2 * n_elem_per_reg), yv[2]);
                _mm512_storeu_pd((y0 + 3 * n_elem_per_reg), yv[3]);

                x0 += 4 * n_elem_per_reg;
                y0 += 4 * n_elem_per_reg;
            }

            for (; (i + 7) < n; i += 8)
            {
                // Loading elements from X
                xv[0] = _mm512_loadu_pd(x0 + 0 * n_elem_per_reg);
                xv[1] = _mm512_loadu_pd(x0 + 1 * n_elem_per_reg);

                // Loading elements from Y
                yv[0] = _mm512_loadu_pd(y0 + 0 * n_elem_per_reg);
                yv[1] = _mm512_loadu_pd(y0 + 1 * n_elem_per_reg);

                // Scale X with real-part of alpha and add to Y
                yv[0] = _mm512_fmadd_pd(alphaRv, xv[0], yv[0]);
                yv[1] = _mm512_fmadd_pd(alphaRv, xv[1], yv[1]);

                // Swapping real and imag parts of every element in X
                xv[0] = _mm512_permute_pd(xv[0], 0x55);
                xv[1] = _mm512_permute_pd(xv[1], 0x55);

                // Scale X with imag-part of alpha and add to Y
                yv[0] = _mm512_fmadd_pd(alphaIv, xv[0], yv[0]);
                yv[1] = _mm512_fmadd_pd(alphaIv, xv[1], yv[1]);

                // Store updated Y
                _mm512_storeu_pd((y0 + 0 * n_elem_per_reg), yv[0]);
                _mm512_storeu_pd((y0 + 1 * n_elem_per_reg), yv[1]);

                x0 += 2 * n_elem_per_reg;
                y0 += 2 * n_elem_per_reg;
            }

            for (; (i + 3) < n; i += 4)
            {
                // Loading elements from X
                xv[0] = _mm512_loadu_pd(x0 + 0 * n_elem_per_reg);

                // Loading elements from Y
                yv[0] = _mm512_loadu_pd(y0 + 0 * n_elem_per_reg);

                // Scale X with real-part of alpha and add to Y
                yv[0] = _mm512_fmadd_pd(alphaRv, xv[0], yv[0]);

                                // Swapping real and imag parts of every element in X
                xv[0] = _mm512_permute_pd(xv[0], 0x55);

                // Scale X with imag-part of alpha and add to Y
                yv[0] = _mm512_fmadd_pd(alphaIv, xv[0], yv[0]);

                // Store updated Y
                _mm512_storeu_pd((y0 + 0 * n_elem_per_reg), yv[0]);

                x0 += n_elem_per_reg;
                y0 += n_elem_per_reg;

            }
        }

        if ( i < n )
        {
            // Setting the mask bit based on remaining elements
            // Since each dcomplex elements corresponds to 2 doubles
            // we need to load and store 2*(n-i) elements.
            __mmask8 n_mask = (1 << 2*(n - i)) - 1;

            // Loading elements from X
            xv[0] = _mm512_maskz_loadu_pd(n_mask, x0);

            // Loading elements from Y
            yv[0] = _mm512_maskz_loadu_pd(n_mask, y0);

            // Scale X with real-part of alpha and add to Y
            yv[0] = _mm512_fmadd_pd(alphaRv, xv[0], yv[0]);

            // Swapping real and imag parts of every element in X
            xv[0] = _mm512_permute_pd(xv[0], 0x55);

            // Scale X with imag-part of alpha and add to Y
            yv[0] = _mm512_fmadd_pd(alphaIv, xv[0], yv[0]);

            // Store updated Y
            _mm512_mask_storeu_pd(y0, n_mask, yv[0]);
        }
    }
    else
    {
        __m128d xv, yv, temp, alphaRv, alphaIv;

        alphaRv = _mm_loaddup_pd((double *)alpha);
        alphaIv = _mm_loaddup_pd((double *)alpha + 1);

        xv = _mm_setzero_pd();

        if (bli_is_noconj(conjx))
            alphaIv = _mm_addsub_pd(xv, alphaIv);
        else
        {
            alphaRv = _mm_addsub_pd(xv, alphaRv);
            alphaRv = _mm_shuffle_pd(alphaRv, alphaRv, 0x01);
        }

        for (; i < n; i += 1)
        {
            xv = _mm_loadu_pd(x0);
            yv = _mm_loadu_pd(y0);

            temp = _mm_shuffle_pd(xv, xv, 0x01);

            temp = _mm_mul_pd(alphaIv, temp);
            xv = _mm_mul_pd(alphaRv, xv);

            xv = _mm_add_pd(xv, temp);
            yv = _mm_add_pd(yv, xv);

            _mm_storeu_pd(y0, yv);

            x0 += 2 * incx;
            y0 += 2 * incy;
        }
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
}
