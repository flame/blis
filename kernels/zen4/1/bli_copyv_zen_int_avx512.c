/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

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

// --------------------------------------------------------------------------------------

/*
    Functionality
    -------------

    This function copies a vector x to a vector y for
    type float.

    y := conj?(x)

    Function Signature
    -------------------

    * 'conjx' - Variable specified if x needs to be conjugated
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

    1. The kernel results in undefined behaviour when n < 0, incx < 1 and incy < 1.
       The expectation is that these are standard BLAS exceptions and should be handled in
       a higher layer
*/

void bli_scopyv_zen_int_avx512
(
    conj_t           conjx,
    dim_t            n,
    float*  restrict x, inc_t incx,
    float*  restrict y, inc_t incy,
    cntx_t* restrict cntx
)
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2)
    dim_t i = 0;

    // Initialize local pointers.
    float *restrict x0 = x;
    float *restrict y0 = y;

    // If the vector dimension is zero return early.
    if (bli_zero_dim1(n))
    {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2)
        return;
    }

    if (incx == 1 && incy == 1)
    {
        const dim_t num_elem_per_reg = 16;
        __m512  xv[32];

        // n & (~0x1FF) = n & 0xFFFFFE00 -> this masks the numbers less than 512,
        // if value of n < 512, then (n & (~0xFF)) = 0
        // the copy operation will be done for the multiples of 512
        for (i = 0; i < (n & (~0x1FF)); i += 512)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_ps(x0 + num_elem_per_reg * 0);
            xv[1] = _mm512_loadu_ps(x0 + num_elem_per_reg * 1);
            xv[2] = _mm512_loadu_ps(x0 + num_elem_per_reg * 2);
            xv[3] = _mm512_loadu_ps(x0 + num_elem_per_reg * 3);

            xv[4] = _mm512_loadu_ps(x0 + num_elem_per_reg * 4);
            xv[5] = _mm512_loadu_ps(x0 + num_elem_per_reg * 5);
            xv[6] = _mm512_loadu_ps(x0 + num_elem_per_reg * 6);
            xv[7] = _mm512_loadu_ps(x0 + num_elem_per_reg * 7);

            xv[8] = _mm512_loadu_ps(x0 + num_elem_per_reg * 8);
            xv[9] = _mm512_loadu_ps(x0 + num_elem_per_reg * 9);
            xv[10] = _mm512_loadu_ps(x0 + num_elem_per_reg * 10);
            xv[11] = _mm512_loadu_ps(x0 + num_elem_per_reg * 11);

            xv[12] = _mm512_loadu_ps(x0 + num_elem_per_reg * 12);
            xv[13] = _mm512_loadu_ps(x0 + num_elem_per_reg * 13);
            xv[14] = _mm512_loadu_ps(x0 + num_elem_per_reg * 14);
            xv[15] = _mm512_loadu_ps(x0 + num_elem_per_reg * 15);

            xv[16] = _mm512_loadu_ps(x0 + num_elem_per_reg * 16);
            xv[17] = _mm512_loadu_ps(x0 + num_elem_per_reg * 17);
            xv[18] = _mm512_loadu_ps(x0 + num_elem_per_reg * 18);
            xv[19] = _mm512_loadu_ps(x0 + num_elem_per_reg * 19);

            xv[20] = _mm512_loadu_ps(x0 + num_elem_per_reg * 20);
            xv[21] = _mm512_loadu_ps(x0 + num_elem_per_reg * 21);
            xv[22] = _mm512_loadu_ps(x0 + num_elem_per_reg * 22);
            xv[23] = _mm512_loadu_ps(x0 + num_elem_per_reg * 23);

            xv[24] = _mm512_loadu_ps(x0 + num_elem_per_reg * 24);
            xv[25] = _mm512_loadu_ps(x0 + num_elem_per_reg * 25);
            xv[26] = _mm512_loadu_ps(x0 + num_elem_per_reg * 26);
            xv[27] = _mm512_loadu_ps(x0 + num_elem_per_reg * 27);

            xv[28] = _mm512_loadu_ps(x0 + num_elem_per_reg * 28);
            xv[29] = _mm512_loadu_ps(x0 + num_elem_per_reg * 29);
            xv[30] = _mm512_loadu_ps(x0 + num_elem_per_reg * 30);
            xv[31] = _mm512_loadu_ps(x0 + num_elem_per_reg * 31);

            // Storing the values to destination
            _mm512_storeu_ps(y0 + num_elem_per_reg * 0, xv[0]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 1, xv[1]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 2, xv[2]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 3, xv[3]);

            _mm512_storeu_ps(y0 + num_elem_per_reg * 4, xv[4]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 5, xv[5]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 6, xv[6]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 7, xv[7]);

            _mm512_storeu_ps(y0 + num_elem_per_reg * 8, xv[8]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 9 , xv[9]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 10, xv[10]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 11, xv[11]);

            _mm512_storeu_ps(y0 + num_elem_per_reg * 12, xv[12]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 13, xv[13]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 14, xv[14]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 15, xv[15]);

            _mm512_storeu_ps(y0 + num_elem_per_reg * 16, xv[16]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 17, xv[17]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 18, xv[18]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 19, xv[19]);

            _mm512_storeu_ps(y0 + num_elem_per_reg * 20, xv[20]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 21, xv[21]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 22, xv[22]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 23, xv[23]);

            _mm512_storeu_ps(y0 + num_elem_per_reg * 24, xv[24]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 25, xv[25]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 26, xv[26]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 27, xv[27]);

            _mm512_storeu_ps(y0 + num_elem_per_reg * 28, xv[28]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 29, xv[29]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 30, xv[30]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 31, xv[31]);

            // Increment the pointer
            x0 += 32 * num_elem_per_reg;
            y0 += 32 * num_elem_per_reg;
        }

        for (; i < (n & (~0xFF)); i += 256)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_ps(x0 + num_elem_per_reg * 0);
            xv[1] = _mm512_loadu_ps(x0 + num_elem_per_reg * 1);
            xv[2] = _mm512_loadu_ps(x0 + num_elem_per_reg * 2);
            xv[3] = _mm512_loadu_ps(x0 + num_elem_per_reg * 3);

            xv[4] = _mm512_loadu_ps(x0 + num_elem_per_reg * 4);
            xv[5] = _mm512_loadu_ps(x0 + num_elem_per_reg * 5);
            xv[6] = _mm512_loadu_ps(x0 + num_elem_per_reg * 6);
            xv[7] = _mm512_loadu_ps(x0 + num_elem_per_reg * 7);

            xv[8] = _mm512_loadu_ps(x0 + num_elem_per_reg * 8);
            xv[9] = _mm512_loadu_ps(x0 + num_elem_per_reg * 9);
            xv[10] = _mm512_loadu_ps(x0 + num_elem_per_reg * 10);
            xv[11] = _mm512_loadu_ps(x0 + num_elem_per_reg * 11);

            xv[12] = _mm512_loadu_ps(x0 + num_elem_per_reg * 12);
            xv[13] = _mm512_loadu_ps(x0 + num_elem_per_reg * 13);
            xv[14] = _mm512_loadu_ps(x0 + num_elem_per_reg * 14);
            xv[15] = _mm512_loadu_ps(x0 + num_elem_per_reg * 15);

            // Storing the values to destination
            _mm512_storeu_ps(y0 + num_elem_per_reg * 0, xv[0]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 1, xv[1]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 2, xv[2]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 3, xv[3]);

            _mm512_storeu_ps(y0 + num_elem_per_reg * 4, xv[4]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 5, xv[5]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 6, xv[6]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 7, xv[7]);

            _mm512_storeu_ps(y0 + num_elem_per_reg * 8, xv[8]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 9 , xv[9]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 10, xv[10]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 11, xv[11]);

            _mm512_storeu_ps(y0 + num_elem_per_reg * 12, xv[12]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 13, xv[13]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 14, xv[14]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 15, xv[15]);

            // Increment the pointer
            x0 += 16 * num_elem_per_reg;
            y0 += 16 * num_elem_per_reg;
        }

        for (; i < (n & (~0x7F)); i += 128)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_ps(x0 + num_elem_per_reg * 0);
            xv[1] = _mm512_loadu_ps(x0 + num_elem_per_reg * 1);
            xv[2] = _mm512_loadu_ps(x0 + num_elem_per_reg * 2);
            xv[3] = _mm512_loadu_ps(x0 + num_elem_per_reg * 3);

            xv[4] = _mm512_loadu_ps(x0 + num_elem_per_reg * 4);
            xv[5] = _mm512_loadu_ps(x0 + num_elem_per_reg * 5);
            xv[6] = _mm512_loadu_ps(x0 + num_elem_per_reg * 6);
            xv[7] = _mm512_loadu_ps(x0 + num_elem_per_reg * 7);

            // Storing the values to destination
            _mm512_storeu_ps(y0 + num_elem_per_reg * 0, xv[0]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 1, xv[1]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 2, xv[2]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 3, xv[3]);

            _mm512_storeu_ps(y0 + num_elem_per_reg * 4, xv[4]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 5, xv[5]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 6, xv[6]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 7, xv[7]);

            // Increment the pointer
            x0 += 8 * num_elem_per_reg;
            y0 += 8 * num_elem_per_reg;
        }

        for (; i < (n & (~0x3F)); i += 64)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_ps(x0 + num_elem_per_reg * 0);
            xv[1] = _mm512_loadu_ps(x0 + num_elem_per_reg * 1);
            xv[2] = _mm512_loadu_ps(x0 + num_elem_per_reg * 2);
            xv[3] = _mm512_loadu_ps(x0 + num_elem_per_reg * 3);

            // Storing the values to destination
            _mm512_storeu_ps(y0 + num_elem_per_reg * 0, xv[0]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 1, xv[1]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 2, xv[2]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 3, xv[3]);

            // Increment the pointer
            x0 += 4 * num_elem_per_reg;
            y0 += 4 * num_elem_per_reg;
        }

        for (; i < (n & (~0x1F)); i += 32)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_ps(x0 + num_elem_per_reg * 0);
            xv[1] = _mm512_loadu_ps(x0 + num_elem_per_reg * 1);

            // Storing the values to destination
            _mm512_storeu_ps(y0 + num_elem_per_reg * 0, xv[0]);
            _mm512_storeu_ps(y0 + num_elem_per_reg * 1, xv[1]);

            // Increment the pointer
            x0 += 2 * num_elem_per_reg;
            y0 += 2 * num_elem_per_reg;
        }

        for (; i < (n & (~0x0F)); i += 16)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_ps(x0 + num_elem_per_reg * 0);

            // Storing the values to destination
            _mm512_storeu_ps(y0 + num_elem_per_reg * 0, xv[0]);

            // Increment the pointer
            x0 += num_elem_per_reg;
            y0 += num_elem_per_reg;
        }

        if ( i < n )
        {
            xv[1] = _mm512_setzero_ps();

            // Creating the mask
            __mmask16 mask = (1 << (n-i)) - 1;

            // Loading the input values
            xv[0] = _mm512_mask_loadu_ps(xv[1], mask, x0 + num_elem_per_reg * 0);

            // Storing the values to destination
            _mm512_mask_storeu_ps(y0 + num_elem_per_reg * 0, mask, xv[0]);

        }
    }
    else
    {
        for ( i = 0; i < n; ++i)
        {
            *y0 = *x0;

            x0 += incx;
            y0 += incy;
        }
    }
}


// --------------------------------------------------------------------------------------

/*
    Functionality
    -------------

    This function copies a vector x to a vector y for
    type double.

    y := conj?(x)

    Function Signature
    -------------------

    * 'conjx' - Variable specified if x needs to be conjugated
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

    1. The kernel results in undefined behaviour when n < 0, incx < 1 and incy < 1.
       The expectation is that these are standard BLAS exceptions and should be handled in
       a higher layer
*/

void bli_dcopyv_zen_int_avx512
(
    conj_t           conjx,
    dim_t            n,
    double*  restrict x, inc_t incx,
    double*  restrict y, inc_t incy,
    cntx_t* restrict cntx
    )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2)
    dim_t i = 0;

    // Initialize local pointers.
    double *restrict x0 = x;
    double *restrict y0 = y;

    // If the vector dimension is zero return early.
    if (bli_zero_dim1(n))
    {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2)
        return;
    }

    if (incx == 1 && incy == 1)
    {
        const dim_t num_elem_per_reg = 8;
        __m512d  xv[32];

        // n & (~0x7F) = n & 0xFFFFF00 -> this masks the numbers less than 256,
        // if value of n < 256, then (n & (~0xFF)) = 0
        // the copy operation will be done for the multiples of 256
        for (i = 0; i < (n & (~0xFF)); i += 256)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_pd(x0 + num_elem_per_reg * 0);
            xv[1] = _mm512_loadu_pd(x0 + num_elem_per_reg * 1);
            xv[2] = _mm512_loadu_pd(x0 + num_elem_per_reg * 2);
            xv[3] = _mm512_loadu_pd(x0 + num_elem_per_reg * 3);

            xv[4] = _mm512_loadu_pd(x0 + num_elem_per_reg * 4);
            xv[5] = _mm512_loadu_pd(x0 + num_elem_per_reg * 5);
            xv[6] = _mm512_loadu_pd(x0 + num_elem_per_reg * 6);
            xv[7] = _mm512_loadu_pd(x0 + num_elem_per_reg * 7);

            xv[8] = _mm512_loadu_pd(x0 + num_elem_per_reg * 8);
            xv[9] = _mm512_loadu_pd(x0 + num_elem_per_reg * 9);
            xv[10] = _mm512_loadu_pd(x0 + num_elem_per_reg * 10);
            xv[11] = _mm512_loadu_pd(x0 + num_elem_per_reg * 11);

            xv[12] = _mm512_loadu_pd(x0 + num_elem_per_reg * 12);
            xv[13] = _mm512_loadu_pd(x0 + num_elem_per_reg * 13);
            xv[14] = _mm512_loadu_pd(x0 + num_elem_per_reg * 14);
            xv[15] = _mm512_loadu_pd(x0 + num_elem_per_reg * 15);

            xv[16] = _mm512_loadu_pd(x0 + num_elem_per_reg * 16);
            xv[17] = _mm512_loadu_pd(x0 + num_elem_per_reg * 17);
            xv[18] = _mm512_loadu_pd(x0 + num_elem_per_reg * 18);
            xv[19] = _mm512_loadu_pd(x0 + num_elem_per_reg * 19);

            xv[20] = _mm512_loadu_pd(x0 + num_elem_per_reg * 20);
            xv[21] = _mm512_loadu_pd(x0 + num_elem_per_reg * 21);
            xv[22] = _mm512_loadu_pd(x0 + num_elem_per_reg * 22);
            xv[23] = _mm512_loadu_pd(x0 + num_elem_per_reg * 23);

            xv[24] = _mm512_loadu_pd(x0 + num_elem_per_reg * 24);
            xv[25] = _mm512_loadu_pd(x0 + num_elem_per_reg * 25);
            xv[26] = _mm512_loadu_pd(x0 + num_elem_per_reg * 26);
            xv[27] = _mm512_loadu_pd(x0 + num_elem_per_reg * 27);

            xv[28] = _mm512_loadu_pd(x0 + num_elem_per_reg * 28);
            xv[29] = _mm512_loadu_pd(x0 + num_elem_per_reg * 29);
            xv[30] = _mm512_loadu_pd(x0 + num_elem_per_reg * 30);
            xv[31] = _mm512_loadu_pd(x0 + num_elem_per_reg * 31);

            // Storing the values to destination
            _mm512_storeu_pd(y0 + num_elem_per_reg * 0, xv[0]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 1, xv[1]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 2, xv[2]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 3, xv[3]);

            _mm512_storeu_pd(y0 + num_elem_per_reg * 4, xv[4]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 5, xv[5]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 6, xv[6]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 7, xv[7]);

            _mm512_storeu_pd(y0 + num_elem_per_reg * 8, xv[8]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 9 , xv[9]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 10, xv[10]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 11, xv[11]);

            _mm512_storeu_pd(y0 + num_elem_per_reg * 12, xv[12]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 13, xv[13]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 14, xv[14]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 15, xv[15]);

            _mm512_storeu_pd(y0 + num_elem_per_reg * 16, xv[16]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 17, xv[17]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 18, xv[18]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 19, xv[19]);

            _mm512_storeu_pd(y0 + num_elem_per_reg * 20, xv[20]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 21, xv[21]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 22, xv[22]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 23, xv[23]);

            _mm512_storeu_pd(y0 + num_elem_per_reg * 24, xv[24]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 25, xv[25]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 26, xv[26]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 27, xv[27]);

            _mm512_storeu_pd(y0 + num_elem_per_reg * 28, xv[28]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 29, xv[29]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 30, xv[30]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 31, xv[31]);

            // Increment the pointer
            x0 += 32 * num_elem_per_reg;
            y0 += 32 * num_elem_per_reg;
        }

        for (; i < (n & (~0x7F)); i += 128)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_pd(x0 + num_elem_per_reg * 0);
            xv[1] = _mm512_loadu_pd(x0 + num_elem_per_reg * 1);
            xv[2] = _mm512_loadu_pd(x0 + num_elem_per_reg * 2);
            xv[3] = _mm512_loadu_pd(x0 + num_elem_per_reg * 3);

            xv[4] = _mm512_loadu_pd(x0 + num_elem_per_reg * 4);
            xv[5] = _mm512_loadu_pd(x0 + num_elem_per_reg * 5);
            xv[6] = _mm512_loadu_pd(x0 + num_elem_per_reg * 6);
            xv[7] = _mm512_loadu_pd(x0 + num_elem_per_reg * 7);

            xv[8] = _mm512_loadu_pd(x0 + num_elem_per_reg * 8);
            xv[9] = _mm512_loadu_pd(x0 + num_elem_per_reg * 9);
            xv[10] = _mm512_loadu_pd(x0 + num_elem_per_reg * 10);
            xv[11] = _mm512_loadu_pd(x0 + num_elem_per_reg * 11);

            xv[12] = _mm512_loadu_pd(x0 + num_elem_per_reg * 12);
            xv[13] = _mm512_loadu_pd(x0 + num_elem_per_reg * 13);
            xv[14] = _mm512_loadu_pd(x0 + num_elem_per_reg * 14);
            xv[15] = _mm512_loadu_pd(x0 + num_elem_per_reg * 15);

            // Storing the values to destination
            _mm512_storeu_pd(y0 + num_elem_per_reg * 0, xv[0]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 1, xv[1]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 2, xv[2]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 3, xv[3]);

            _mm512_storeu_pd(y0 + num_elem_per_reg * 4, xv[4]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 5, xv[5]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 6, xv[6]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 7, xv[7]);

            _mm512_storeu_pd(y0 + num_elem_per_reg * 8, xv[8]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 9 , xv[9]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 10, xv[10]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 11, xv[11]);

            _mm512_storeu_pd(y0 + num_elem_per_reg * 12, xv[12]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 13, xv[13]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 14, xv[14]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 15, xv[15]);

            // Increment the pointer
            x0 += 16 * num_elem_per_reg;
            y0 += 16 * num_elem_per_reg;
        }

        for (; i < (n & (~0x3F)); i += 64)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_pd(x0 + num_elem_per_reg * 0);
            xv[1] = _mm512_loadu_pd(x0 + num_elem_per_reg * 1);
            xv[2] = _mm512_loadu_pd(x0 + num_elem_per_reg * 2);
            xv[3] = _mm512_loadu_pd(x0 + num_elem_per_reg * 3);

            xv[4] = _mm512_loadu_pd(x0 + num_elem_per_reg * 4);
            xv[5] = _mm512_loadu_pd(x0 + num_elem_per_reg * 5);
            xv[6] = _mm512_loadu_pd(x0 + num_elem_per_reg * 6);
            xv[7] = _mm512_loadu_pd(x0 + num_elem_per_reg * 7);

            // Storing the values to destination
            _mm512_storeu_pd(y0 + num_elem_per_reg * 0, xv[0]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 1, xv[1]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 2, xv[2]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 3, xv[3]);

            _mm512_storeu_pd(y0 + num_elem_per_reg * 4, xv[4]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 5, xv[5]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 6, xv[6]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 7, xv[7]);

            // Increment the pointer
            x0 += 8 * num_elem_per_reg;
            y0 += 8 * num_elem_per_reg;
        }

        for (; i < (n & (~0x1F)); i += 32)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_pd(x0 + num_elem_per_reg * 0);
            xv[1] = _mm512_loadu_pd(x0 + num_elem_per_reg * 1);
            xv[2] = _mm512_loadu_pd(x0 + num_elem_per_reg * 2);
            xv[3] = _mm512_loadu_pd(x0 + num_elem_per_reg * 3);

            // Storing the values to destination
            _mm512_storeu_pd(y0 + num_elem_per_reg * 0, xv[0]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 1, xv[1]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 2, xv[2]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 3, xv[3]);

            // Increment the pointer
            x0 += 4 * num_elem_per_reg;
            y0 += 4 * num_elem_per_reg;
        }

        for (; i < (n & (~0x0F)); i += 16)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_pd(x0 + num_elem_per_reg * 0);
            xv[1] = _mm512_loadu_pd(x0 + num_elem_per_reg * 1);

            // Storing the values to destination
            _mm512_storeu_pd(y0 + num_elem_per_reg * 0, xv[0]);
            _mm512_storeu_pd(y0 + num_elem_per_reg * 1, xv[1]);

            // Increment the pointer
            x0 += 2 * num_elem_per_reg;
            y0 += 2 * num_elem_per_reg;
        }

        for (; i < (n & (~0x07)); i += 8)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_pd(x0 + num_elem_per_reg * 0);

            // Storing the values to destination
            _mm512_storeu_pd(y0 + num_elem_per_reg * 0, xv[0]);

            // Increment the pointer
            x0 += num_elem_per_reg;
            y0 += num_elem_per_reg;
        }

        if ( i < n )
        {
            xv[1] = _mm512_setzero_pd();

            // Creating the mask
            __mmask8 mask = (1 << (n-i)) - 1;

            // Loading the input values
            xv[0] = _mm512_mask_loadu_pd(xv[1], mask, x0 + num_elem_per_reg * 0);

            // Storing the values to destination
            _mm512_mask_storeu_pd(y0 + num_elem_per_reg * 0, mask, xv[0]);

        }
    }
    else
    {
        for ( i = 0; i < n; ++i)
        {
            *y0 = *x0;

            x0 += incx;
            y0 += incy;
        }
    }
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2)
}

// -----------------------------------------------------------------------------

/*
    Functionality
    -------------

    This function copies a double complex vector x to a double complex vector y.

    y := conj?(x)

    Function Signature
    -------------------

    * 'conjx' - Variable specified if x needs to be conjugated
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

    1. The kernel results in undefined behaviour when n < 0, incx < 1 and incy < 1.
       The expectation is that these are standard BLAS exceptions and should be handled in
       a higher layer
*/

void bli_zcopyv_zen_int_avx512
(
    conj_t           conjx,
    dim_t            n,
    dcomplex*  restrict x, inc_t incx,
    dcomplex*  restrict y, inc_t incy,
    cntx_t* restrict cntx
)
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2)
    dim_t i = 0;

    // Initialize local pointers.
    dcomplex *x0 = x;
    dcomplex *y0 = y;

    // If the vector dimension is zero return early.
    if (bli_zero_dim1(n))
    {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2)
        return;
    }

    // Check if conjugation is required and select the required code path
    if (bli_is_conj(conjx))
    {

        if (incx == 1 && incy == 1)
        {
            const dim_t num_elem_per_reg = 8;
            __m512d  xv[16];
            __m512d zero_reg = _mm512_setzero_pd();

            // n & (~0x3F) = n & 0xFFFFFFC0 -> this masks the numbers less than 64,
            // if value of n < 64, then (n & (~0x3F)) = 0
            // the copy operation will be done for the multiples of 64
            for (i = 0; i < (n & (~0x3F)); i += 64)
            {
                // Loading the input values
                xv[0] = _mm512_loadu_pd((double *)(x0 + num_elem_per_reg * 0));
                xv[1] = _mm512_loadu_pd((double *)(x0 + num_elem_per_reg * 1));
                xv[2] = _mm512_loadu_pd((double *)(x0 + num_elem_per_reg * 2));
                xv[3] = _mm512_loadu_pd((double *)(x0 + num_elem_per_reg * 3));

                xv[4] = _mm512_loadu_pd((double *)(x0 + num_elem_per_reg * 4));
                xv[5] = _mm512_loadu_pd((double *)(x0 + num_elem_per_reg * 5));
                xv[6] = _mm512_loadu_pd((double *)(x0 + num_elem_per_reg * 6));
                xv[7] = _mm512_loadu_pd((double *)(x0 + num_elem_per_reg * 7));

                xv[8] = _mm512_loadu_pd((double *)(x0 + num_elem_per_reg * 8));
                xv[9] = _mm512_loadu_pd((double *)(x0 + num_elem_per_reg * 9));
                xv[10] = _mm512_loadu_pd((double *)(x0 + num_elem_per_reg * 10));
                xv[11] = _mm512_loadu_pd((double *)(x0 + num_elem_per_reg * 11));

                xv[12] = _mm512_loadu_pd((double *)(x0 + num_elem_per_reg * 12));
                xv[13] = _mm512_loadu_pd((double *)(x0 + num_elem_per_reg * 13));
                xv[14] = _mm512_loadu_pd((double *)(x0 + num_elem_per_reg * 14));
                xv[15] = _mm512_loadu_pd((double *)(x0 + num_elem_per_reg * 15));

                // Perform conjugation by multiplying the imaginary part with -1 and real part with 1
                xv[0] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[0]);
                xv[1] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[1]);
                xv[2] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[2]);
                xv[3] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[3]);

                xv[4] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[4]);
                xv[5] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[5]);
                xv[6] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[6]);
                xv[7] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[7]);

                xv[8] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[8]);
                xv[9] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[9]);
                xv[10] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[10]);
                xv[11] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[11]);

                xv[12] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[12]);
                xv[13] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[13]);
                xv[14] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[14]);
                xv[15] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[15]);

                // Storing the values to destination
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 0), xv[0]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 1), xv[1]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 2), xv[2]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 3), xv[3]);

                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 4), xv[4]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 5), xv[5]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 6), xv[6]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 7), xv[7]);

                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 8), xv[8]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 9), xv[9]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 10), xv[10]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 11), xv[11]);

                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 12), xv[12]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 13), xv[13]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 14), xv[14]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 15), xv[15]);

                // Increment the pointer
                x0 += 16 * num_elem_per_reg;
                y0 += 16 * num_elem_per_reg;
            }

            for (; i < (n & (~0x1F)); i += 32)
            {
                // Loading the input values
                xv[0] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 0));
                xv[1] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 1));
                xv[2] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 2));
                xv[3] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 3));

                xv[4] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 4));
                xv[5] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 5));
                xv[6] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 6));
                xv[7] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 7));

                // Perform conjugation by multiplying the imaginary part with -1 and real part with 1
                xv[0] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[0]);
                xv[1] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[1]);
                xv[2] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[2]);
                xv[3] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[3]);

                xv[4] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[4]);
                xv[5] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[5]);
                xv[6] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[6]);
                xv[7] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[7]);

                // Storing the values to destination
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 0), xv[0]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 1), xv[1]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 2), xv[2]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 3), xv[3]);

                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 4), xv[4]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 5), xv[5]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 6), xv[6]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 7), xv[7]);

                // Increment the pointer
                x0 += 8 * num_elem_per_reg;
                y0 += 8 * num_elem_per_reg;
            }

            for (; i < (n & (~0x0F)); i += 16)
            {
                // Loading the input values
                xv[0] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 0));
                xv[1] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 1));
                xv[2] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 2));
                xv[3] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 3));

                // Perform conjugation by multiplying the imaginary part with -1 and real part with 1
                xv[0] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[0]);
                xv[1] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[1]);
                xv[2] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[2]);
                xv[3] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[3]);

                // Storing the values to destination
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 0), xv[0]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 1), xv[1]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 2), xv[2]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 3), xv[3]);

                // Increment the pointer
                x0 += 4 * num_elem_per_reg;
                y0 += 4 * num_elem_per_reg;
            }

            for (; i < (n & (~0x07)); i += 8)
            {
                // Loading the input values
                xv[0] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 0));
                xv[1] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 1));

                // Perform conjugation by multiplying the imaginary part with -1 and real part with 1
                xv[0] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[0]);
                xv[1] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[1]);

                // Storing the values to destination
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 0), xv[0]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 1), xv[1]);

                // Increment the pointer
                x0 += 2 * num_elem_per_reg;
                y0 += 2 * num_elem_per_reg;
            }

            for (; i < (n & (~0x03)); i += 4)
            {
                // Loading the input values
                xv[0] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 0));

                // Perform conjugation by multiplying the imaginary part with -1 and real part with 1
                xv[0] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[0]);

                // Storing the values to destination
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 0), xv[0]);

                // Increment the pointer
                x0 += num_elem_per_reg;
                y0 += num_elem_per_reg;
            }

            if ( i < n )
            {
                xv[1] = _mm512_setzero_pd();

                // Creating the mask
                __mmask8 mask = (1 << 2*(n-i)) - 1;

                // Loading the input values
                xv[0] = _mm512_mask_loadu_pd( zero_reg, mask,(double *)( x0 + num_elem_per_reg * 0));

                // Perform conjugation by multiplying the imaginary part with -1 and real part with 1
                xv[0] = _mm512_fmsubadd_pd( zero_reg, zero_reg, xv[0]);

                // Storing the values to destination
                _mm512_mask_storeu_pd((double *)(y0 + num_elem_per_reg * 0), mask, xv[0]);

            }
        }
        else
        {
            // Since double complex elements are of size 128 bits,
            // vectorization can be done using XMM registers when incx and incy are not 1.
            // This is done in the else condition.
            __m128d  xv[16];
            __m128d conj_reg = _mm_setr_pd(1, -1);

            // n & (~0x0F) = n & 0xFFFFFFF0 -> this masks the numbers less than 16,
            // if value of n < 16, then (n & (~0x0F)) = 0
            // the copy operation will be done for the multiples of 16
            for ( i = 0; i < (n & (~0x0F)); i += 16)
            {
                // Loading the input values
                xv[0] = _mm_loadu_pd((double *)(x0 + 0 * incx));
                xv[1] = _mm_loadu_pd((double *)(x0 + 1 * incx));
                xv[2] = _mm_loadu_pd((double *)(x0 + 2 * incx));
                xv[3] = _mm_loadu_pd((double *)(x0 + 3 * incx));

                xv[4] = _mm_loadu_pd((double *)(x0 + 4 * incx));
                xv[5] = _mm_loadu_pd((double *)(x0 + 5 * incx));
                xv[6] = _mm_loadu_pd((double *)(x0 + 6 * incx));
                xv[7] = _mm_loadu_pd((double *)(x0 + 7 * incx));

                xv[8] = _mm_loadu_pd((double *)(x0 + 8 * incx));
                xv[9] = _mm_loadu_pd((double *)(x0 + 9 * incx));
                xv[10] = _mm_loadu_pd((double *)(x0 + 10 * incx));
                xv[11] = _mm_loadu_pd((double *)(x0 + 11 * incx));

                xv[12] = _mm_loadu_pd((double *)(x0 + 12 * incx));
                xv[13] = _mm_loadu_pd((double *)(x0 + 13 * incx));
                xv[14] = _mm_loadu_pd((double *)(x0 + 14 * incx));
                xv[15] = _mm_loadu_pd((double *)(x0 + 15 * incx));

                // Perform conjugation by multiplying the imaginary part with -1 and real part with 1
                xv[0] = _mm_mul_pd(xv[0], conj_reg);
                xv[1] = _mm_mul_pd(xv[1], conj_reg);
                xv[2] = _mm_mul_pd(xv[2], conj_reg);
                xv[3] = _mm_mul_pd(xv[3], conj_reg);

                xv[4] = _mm_mul_pd(xv[4], conj_reg);
                xv[5] = _mm_mul_pd(xv[5], conj_reg);
                xv[6] = _mm_mul_pd(xv[6], conj_reg);
                xv[7] = _mm_mul_pd(xv[7], conj_reg);

                xv[8] = _mm_mul_pd(xv[8], conj_reg);
                xv[9] = _mm_mul_pd(xv[9], conj_reg);
                xv[10] = _mm_mul_pd(xv[10], conj_reg);
                xv[11] = _mm_mul_pd(xv[11], conj_reg);

                xv[12] = _mm_mul_pd(xv[12], conj_reg);
                xv[13] = _mm_mul_pd(xv[13], conj_reg);
                xv[14] = _mm_mul_pd(xv[14], conj_reg);
                xv[15] = _mm_mul_pd(xv[15], conj_reg);

                // Storing the values to destination

                _mm_storeu_pd((double *)(y0 + incy * 0), xv[0]);
                _mm_storeu_pd((double *)(y0 + incy * 1), xv[1]);
                _mm_storeu_pd((double *)(y0 + incy * 2), xv[2]);
                _mm_storeu_pd((double *)(y0 + incy * 3), xv[3]);

                _mm_storeu_pd((double *)(y0 + incy * 4), xv[4]);
                _mm_storeu_pd((double *)(y0 + incy * 5), xv[5]);
                _mm_storeu_pd((double *)(y0 + incy * 6), xv[6]);
                _mm_storeu_pd((double *)(y0 + incy * 7), xv[7]);

                _mm_storeu_pd((double *)(y0 + incy * 8), xv[8]);
                _mm_storeu_pd((double *)(y0 + incy * 9 ), xv[9]);
                _mm_storeu_pd((double *)(y0 + incy * 10), xv[10]);
                _mm_storeu_pd((double *)(y0 + incy * 11), xv[11]);

                _mm_storeu_pd((double *)(y0 + incy * 12), xv[12]);
                _mm_storeu_pd((double *)(y0 + incy * 13), xv[13]);
                _mm_storeu_pd((double *)(y0 + incy * 14), xv[14]);
                _mm_storeu_pd((double *)(y0 + incy * 15), xv[15]);

                // Increment the pointer
                x0 += 16 * incx;
                y0 += 16 * incy;
            }

            for ( ; i < (n & (~0x07)); i += 8)
            {
                // Loading the input values
                xv[0] = _mm_loadu_pd((double *)(x0 + 0 * incx));
                xv[1] = _mm_loadu_pd((double *)(x0 + 1 * incx));
                xv[2] = _mm_loadu_pd((double *)(x0 + 2 * incx));
                xv[3] = _mm_loadu_pd((double *)(x0 + 3 * incx));

                xv[4] = _mm_loadu_pd((double *)(x0 + 4 * incx));
                xv[5] = _mm_loadu_pd((double *)(x0 + 5 * incx));
                xv[6] = _mm_loadu_pd((double *)(x0 + 6 * incx));
                xv[7] = _mm_loadu_pd((double *)(x0 + 7 * incx));

                // Perform conjugation by multiplying the imaginary part with -1 and real part with 1
                xv[0] = _mm_mul_pd(xv[0], conj_reg);
                xv[1] = _mm_mul_pd(xv[1], conj_reg);
                xv[2] = _mm_mul_pd(xv[2], conj_reg);
                xv[3] = _mm_mul_pd(xv[3], conj_reg);

                xv[4] = _mm_mul_pd(xv[4], conj_reg);
                xv[5] = _mm_mul_pd(xv[5], conj_reg);
                xv[6] = _mm_mul_pd(xv[6], conj_reg);
                xv[7] = _mm_mul_pd(xv[7], conj_reg);

                // Storing the values to destination

                _mm_storeu_pd((double *)(y0 + incy * 0), xv[0]);
                _mm_storeu_pd((double *)(y0 + incy * 1), xv[1]);
                _mm_storeu_pd((double *)(y0 + incy * 2), xv[2]);
                _mm_storeu_pd((double *)(y0 + incy * 3), xv[3]);

                _mm_storeu_pd((double *)(y0 + incy * 4), xv[4]);
                _mm_storeu_pd((double *)(y0 + incy * 5), xv[5]);
                _mm_storeu_pd((double *)(y0 + incy * 6), xv[6]);
                _mm_storeu_pd((double *)(y0 + incy * 7), xv[7]);

                // Increment the pointer
                x0 += 8 * incx;
                y0 += 8 * incy;
            }

            for ( ; i < (n & (~0x03)); i += 4)
            {
                // Loading the input values
                xv[0] = _mm_loadu_pd((double *)(x0 + 0 * incx));
                xv[1] = _mm_loadu_pd((double *)(x0 + 1 * incx));
                xv[2] = _mm_loadu_pd((double *)(x0 + 2 * incx));
                xv[3] = _mm_loadu_pd((double *)(x0 + 3 * incx));

                // Perform conjugation by multiplying the imaginary part with -1 and real part with 1
                xv[0] = _mm_mul_pd(xv[0], conj_reg);
                xv[1] = _mm_mul_pd(xv[1], conj_reg);
                xv[2] = _mm_mul_pd(xv[2], conj_reg);
                xv[3] = _mm_mul_pd(xv[3], conj_reg);

                // Storing the values to destination

                _mm_storeu_pd((double *)(y0 + incy * 0), xv[0]);
                _mm_storeu_pd((double *)(y0 + incy * 1), xv[1]);
                _mm_storeu_pd((double *)(y0 + incy * 2), xv[2]);
                _mm_storeu_pd((double *)(y0 + incy * 3), xv[3]);

                // Increment the pointer
                x0 += 4 * incx;
                y0 += 4 * incy;
            }

            for ( ; i < (n & (~0x01)); i += 2)
            {
                // Loading the input values
                xv[0] = _mm_loadu_pd((double *)(x0 + 0 * incx));
                xv[1] = _mm_loadu_pd((double *)(x0 + 1 * incx));

                // Perform conjugation by multiplying the imaginary part with -1 and real part with 1
                xv[0] = _mm_mul_pd(xv[0], conj_reg);
                xv[1] = _mm_mul_pd(xv[1], conj_reg);

                // Storing the values to destination

                _mm_storeu_pd((double *)(y0 + incy * 0), xv[0]);
                _mm_storeu_pd((double *)(y0 + incy * 1), xv[1]);

                // Increment the pointer
                x0 += 2 * incx;
                y0 += 2 * incy;
            }

            for ( ; i < n; i += 1)
            {
                // Loading the input values
                xv[0] = _mm_loadu_pd((double *)(x0 + 0 * incx));

                // Perform conjugation by multiplying the imaginary part with -1 and real part with 1
                xv[0] = _mm_mul_pd(xv[0], conj_reg);

                // Storing the values to destination
                _mm_storeu_pd((double *)(y0 + incy * 0), xv[0]);

                // Increment the pointer
                x0 += 1 * incx;
                y0 += 1 * incy;
            }
        }
    }
    else
    {
        if (incx == 1 && incy == 1)
        {
            const dim_t num_elem_per_reg = 8;
            __m512d  xv[32];

            // n & (~0xFF) = n & 0xFFFFFF00 -> this masks the numbers less than 256,
            // if value of n < 256, then (n & (~0xFF)) = 0
            // the copy operation will be done for the multiples of 256
             for (i = 0; i < (n & (~0xFF)); i += 256)
            {
                // Loading the input values
                xv[0] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 0));
                xv[1] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 1));
                xv[2] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 2));
                xv[3] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 3));

                xv[4] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 4));
                xv[5] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 5));
                xv[6] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 6));
                xv[7] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 7));

                xv[8] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 8));
                xv[9] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 9));
                xv[10] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 10));
                xv[11] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 11));

                xv[12] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 12));
                xv[13] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 13));
                xv[14] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 14));
                xv[15] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 15));

                xv[16] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 16));
                xv[17] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 17));
                xv[18] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 18));
                xv[19] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 19));

                xv[20] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 20));
                xv[21] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 21));
                xv[22] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 22));
                xv[23] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 23));

                xv[24] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 24));
                xv[25] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 25));
                xv[26] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 26));
                xv[27] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 27));

                xv[28] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 28));
                xv[29] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 29));
                xv[30] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 30));
                xv[31] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 31));

                // Storing the values to destination
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 0), xv[0]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 1), xv[1]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 2), xv[2]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 3), xv[3]);

                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 4), xv[4]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 5), xv[5]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 6), xv[6]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 7), xv[7]);

                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 8), xv[8]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 9), xv[9]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 10), xv[10]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 11), xv[11]);

                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 12), xv[12]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 13), xv[13]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 14), xv[14]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 15), xv[15]);

                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 16), xv[16]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 17), xv[17]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 18), xv[18]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 19), xv[19]);

                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 20), xv[20]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 21), xv[21]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 22), xv[22]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 23), xv[23]);

                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 24), xv[24]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 25), xv[25]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 26), xv[26]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 27), xv[27]);

                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 28), xv[28]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 29), xv[29]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 30), xv[30]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 31), xv[31]);

                // Increment the pointer
                x0 += 32 * num_elem_per_reg;
                y0 += 32 * num_elem_per_reg;
            }

            for (; i < (n & (~0x7F)); i += 128)
            {
                // Loading the input values
                xv[0] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 0));
                xv[1] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 1));
                xv[2] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 2));
                xv[3] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 3));

                xv[4] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 4));
                xv[5] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 5));
                xv[6] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 6));
                xv[7] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 7));

                xv[8] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 8));
                xv[9] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 9));
                xv[10] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 10));
                xv[11] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 11));

                xv[12] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 12));
                xv[13] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 13));
                xv[14] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 14));
                xv[15] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 15));

                // Storing the values to destination
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 0), xv[0]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 1), xv[1]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 2), xv[2]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 3), xv[3]);

                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 4), xv[4]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 5), xv[5]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 6), xv[6]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 7), xv[7]);

                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 8), xv[8]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 9), xv[9]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 10), xv[10]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 11), xv[11]);

                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 12), xv[12]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 13), xv[13]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 14), xv[14]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 15), xv[15]);

                // Increment the pointer
                x0 += 16 * num_elem_per_reg;
                y0 += 16 * num_elem_per_reg;
            }

            for (; i < (n & (~0x3F)); i += 64)
            {
                // Loading the input values
                xv[0] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 0));
                xv[1] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 1));
                xv[2] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 2));
                xv[3] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 3));

                xv[4] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 4));
                xv[5] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 5));
                xv[6] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 6));
                xv[7] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 7));

                // Storing the values to destination
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 0), xv[0]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 1), xv[1]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 2), xv[2]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 3), xv[3]);

                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 4), xv[4]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 5), xv[5]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 6), xv[6]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 7), xv[7]);

                // Increment the pointer
                x0 += 8 * num_elem_per_reg;
                y0 += 8 * num_elem_per_reg;
            }

            for (; i < (n & (~0x1F)); i += 32)
            {
                // Loading the input values
                xv[0] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 0));
                xv[1] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 1));
                xv[2] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 2));
                xv[3] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 3));

                // Storing the values to destination
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 0), xv[0]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 1), xv[1]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 2), xv[2]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 3), xv[3]);

                // Increment the pointer
                x0 += 4 * num_elem_per_reg;
                y0 += 4 * num_elem_per_reg;
            }

            for (; i < (n & (~0x0F)); i += 16)
            {
                // Loading the input values
                xv[0] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 0));
                xv[1] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 1));

                // Storing the values to destination
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 0), xv[0]);
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 1), xv[1]);

                // Increment the pointer
                x0 += 2 * num_elem_per_reg;
                y0 += 2 * num_elem_per_reg;
            }

            for (; i < (n & (~0x07)); i += 8)
            {
                // Loading the input values
                xv[0] = _mm512_loadu_pd((double *)(x0+ num_elem_per_reg * 0));

                // Storing the values to destination
                _mm512_storeu_pd((double *)(y0 + num_elem_per_reg * 0), xv[0]);

                // Increment the pointer
                x0 += num_elem_per_reg;
                y0 += num_elem_per_reg;
            }

            if ( i < n )
            {
                xv[1] = _mm512_setzero_pd();

                // Creating the mask
                __mmask8 mask = (1 << 2*(n-i)) - 1;

                // Loading the input values
                xv[0] = _mm512_mask_loadu_pd(xv[1], mask, (double *)(x0 + num_elem_per_reg * 0));

                // Storing the values to destination
                _mm512_mask_storeu_pd((double *)(y0 + num_elem_per_reg * 0), mask, xv[0]);

            }
        }
        else
        {
            // Since double complex elements are of size 128 bits,
            // vectorization can be done using XMM registers when incx and incy are not 1.
            // This is done in the else condition.
            __m128d  xv[32];

            // n & (~0x1F) = n & 0xFFFFFFE0 -> this masks the numbers less than 32,
            // if value of n < 32, then (n & (~0x1F)) = 0
            // the copy operation will be done for the multiples of 32
            for ( i = 0; i < (n & (~0x1F)); i += 32)
            {
                // Loading the input values
                xv[0] = _mm_loadu_pd((double *)(x0 + 0 * incx));
                xv[1] = _mm_loadu_pd((double *)(x0 + 1 * incx));
                xv[2] = _mm_loadu_pd((double *)(x0 + 2 * incx));
                xv[3] = _mm_loadu_pd((double *)(x0 + 3 * incx));

                xv[4] = _mm_loadu_pd((double *)(x0 + 4 * incx));
                xv[5] = _mm_loadu_pd((double *)(x0 + 5 * incx));
                xv[6] = _mm_loadu_pd((double *)(x0 + 6 * incx));
                xv[7] = _mm_loadu_pd((double *)(x0 + 7 * incx));

                xv[8] = _mm_loadu_pd((double *)(x0 + 8 * incx));
                xv[9] = _mm_loadu_pd((double *)(x0 + 9 * incx));
                xv[10] = _mm_loadu_pd((double *)(x0 + 10 * incx));
                xv[11] = _mm_loadu_pd((double *)(x0 + 11 * incx));

                xv[12] = _mm_loadu_pd((double *)(x0 + 12 * incx));
                xv[13] = _mm_loadu_pd((double *)(x0 + 13 * incx));
                xv[14] = _mm_loadu_pd((double *)(x0 + 14 * incx));
                xv[15] = _mm_loadu_pd((double *)(x0 + 15 * incx));

                xv[16] = _mm_loadu_pd((double *)(x0 + 16 * incx));
                xv[17] = _mm_loadu_pd((double *)(x0 + 17 * incx));
                xv[18] = _mm_loadu_pd((double *)(x0 + 18 * incx));
                xv[19] = _mm_loadu_pd((double *)(x0 + 19 * incx));

                xv[20] = _mm_loadu_pd((double *)(x0 + 20 * incx));
                xv[21] = _mm_loadu_pd((double *)(x0 + 21 * incx));
                xv[22] = _mm_loadu_pd((double *)(x0 + 22 * incx));
                xv[23] = _mm_loadu_pd((double *)(x0 + 23 * incx));

                xv[24] = _mm_loadu_pd((double *)(x0 + 24 * incx));
                xv[25] = _mm_loadu_pd((double *)(x0 + 25 * incx));
                xv[26] = _mm_loadu_pd((double *)(x0 + 26 * incx));
                xv[27] = _mm_loadu_pd((double *)(x0 + 27 * incx));

                xv[28] = _mm_loadu_pd((double *)(x0 + 28 * incx));
                xv[29] = _mm_loadu_pd((double *)(x0 + 29 * incx));
                xv[30] = _mm_loadu_pd((double *)(x0 + 30 * incx));
                xv[31] = _mm_loadu_pd((double *)(x0 + 31 * incx));

                // Storing the values to destination
                _mm_storeu_pd((double *)(y0 + incy * 0), xv[0]);
                _mm_storeu_pd((double *)(y0 + incy * 1), xv[1]);
                _mm_storeu_pd((double *)(y0 + incy * 2), xv[2]);
                _mm_storeu_pd((double *)(y0 + incy * 3), xv[3]);

                _mm_storeu_pd((double *)(y0 + incy * 4), xv[4]);
                _mm_storeu_pd((double *)(y0 + incy * 5), xv[5]);
                _mm_storeu_pd((double *)(y0 + incy * 6), xv[6]);
                _mm_storeu_pd((double *)(y0 + incy * 7), xv[7]);

                _mm_storeu_pd((double *)(y0 + incy * 8), xv[8]);
                _mm_storeu_pd((double *)(y0 + incy * 9 ), xv[9]);
                _mm_storeu_pd((double *)(y0 + incy * 10), xv[10]);
                _mm_storeu_pd((double *)(y0 + incy * 11), xv[11]);

                _mm_storeu_pd((double *)(y0 + incy * 12), xv[12]);
                _mm_storeu_pd((double *)(y0 + incy * 13), xv[13]);
                _mm_storeu_pd((double *)(y0 + incy * 14), xv[14]);
                _mm_storeu_pd((double *)(y0 + incy * 15), xv[15]);

                _mm_storeu_pd((double *)(y0 + incy * 16), xv[16]);
                _mm_storeu_pd((double *)(y0 + incy * 17), xv[17]);
                _mm_storeu_pd((double *)(y0 + incy * 18), xv[18]);
                _mm_storeu_pd((double *)(y0 + incy * 19), xv[19]);

                _mm_storeu_pd((double *)(y0 + incy * 20), xv[20]);
                _mm_storeu_pd((double *)(y0 + incy * 21), xv[21]);
                _mm_storeu_pd((double *)(y0 + incy * 22), xv[22]);
                _mm_storeu_pd((double *)(y0 + incy * 23), xv[23]);

                _mm_storeu_pd((double *)(y0 + incy * 24), xv[24]);
                _mm_storeu_pd((double *)(y0 + incy * 25), xv[25]);
                _mm_storeu_pd((double *)(y0 + incy * 26), xv[26]);
                _mm_storeu_pd((double *)(y0 + incy * 27), xv[27]);

                _mm_storeu_pd((double *)(y0 + incy * 28), xv[28]);
                _mm_storeu_pd((double *)(y0 + incy * 29), xv[29]);
                _mm_storeu_pd((double *)(y0 + incy * 30), xv[30]);
                _mm_storeu_pd((double *)(y0 + incy * 31), xv[31]);

                // Increment the pointer
                x0 += 32 * incx;
                y0 += 32 * incy;
            }

            for ( ; i < (n & (~0x0F)); i += 16)
            {
                // Loading the input values
                xv[0] = _mm_loadu_pd((double *)(x0 + 0 * incx));
                xv[1] = _mm_loadu_pd((double *)(x0 + 1 * incx));
                xv[2] = _mm_loadu_pd((double *)(x0 + 2 * incx));
                xv[3] = _mm_loadu_pd((double *)(x0 + 3 * incx));

                xv[4] = _mm_loadu_pd((double *)(x0 + 4 * incx));
                xv[5] = _mm_loadu_pd((double *)(x0 + 5 * incx));
                xv[6] = _mm_loadu_pd((double *)(x0 + 6 * incx));
                xv[7] = _mm_loadu_pd((double *)(x0 + 7 * incx));

                xv[8] = _mm_loadu_pd((double *)(x0 + 8 * incx));
                xv[9] = _mm_loadu_pd((double *)(x0 + 9 * incx));
                xv[10] = _mm_loadu_pd((double *)(x0 + 10 * incx));
                xv[11] = _mm_loadu_pd((double *)(x0 + 11 * incx));

                xv[12] = _mm_loadu_pd((double *)(x0 + 12 * incx));
                xv[13] = _mm_loadu_pd((double *)(x0 + 13 * incx));
                xv[14] = _mm_loadu_pd((double *)(x0 + 14 * incx));
                xv[15] = _mm_loadu_pd((double *)(x0 + 15 * incx));

                // Storing the values to destination
                _mm_storeu_pd((double *)(y0 + incy * 0), xv[0]);
                _mm_storeu_pd((double *)(y0 + incy * 1), xv[1]);
                _mm_storeu_pd((double *)(y0 + incy * 2), xv[2]);
                _mm_storeu_pd((double *)(y0 + incy * 3), xv[3]);

                _mm_storeu_pd((double *)(y0 + incy * 4), xv[4]);
                _mm_storeu_pd((double *)(y0 + incy * 5), xv[5]);
                _mm_storeu_pd((double *)(y0 + incy * 6), xv[6]);
                _mm_storeu_pd((double *)(y0 + incy * 7), xv[7]);

                _mm_storeu_pd((double *)(y0 + incy * 8), xv[8]);
                _mm_storeu_pd((double *)(y0 + incy * 9), xv[9]);
                _mm_storeu_pd((double *)(y0 + incy * 10), xv[10]);
                _mm_storeu_pd((double *)(y0 + incy * 11), xv[11]);

                _mm_storeu_pd((double *)(y0 + incy * 12), xv[12]);
                _mm_storeu_pd((double *)(y0 + incy * 13), xv[13]);
                _mm_storeu_pd((double *)(y0 + incy * 14), xv[14]);
                _mm_storeu_pd((double *)(y0 + incy * 15), xv[15]);

                // Increment the pointer
                x0 += 16 * incx;
                y0 += 16 * incy;
            }

            for ( ; i < (n & (~0x07)); i += 8)
            {
                // Loading the input values
                xv[0] = _mm_loadu_pd((double *)(x0 + 0 * incx));
                xv[1] = _mm_loadu_pd((double *)(x0 + 1 * incx));
                xv[2] = _mm_loadu_pd((double *)(x0 + 2 * incx));
                xv[3] = _mm_loadu_pd((double *)(x0 + 3 * incx));

                xv[4] = _mm_loadu_pd((double *)(x0 + 4 * incx));
                xv[5] = _mm_loadu_pd((double *)(x0 + 5 * incx));
                xv[6] = _mm_loadu_pd((double *)(x0 + 6 * incx));
                xv[7] = _mm_loadu_pd((double *)(x0 + 7 * incx));

                // Storing the values to destination
                _mm_storeu_pd((double *)(y0 + incy * 0), xv[0]);
                _mm_storeu_pd((double *)(y0 + incy * 1), xv[1]);
                _mm_storeu_pd((double *)(y0 + incy * 2), xv[2]);
                _mm_storeu_pd((double *)(y0 + incy * 3), xv[3]);

                _mm_storeu_pd((double *)(y0 + incy * 4), xv[4]);
                _mm_storeu_pd((double *)(y0 + incy * 5), xv[5]);
                _mm_storeu_pd((double *)(y0 + incy * 6), xv[6]);
                _mm_storeu_pd((double *)(y0 + incy * 7), xv[7]);

                // Increment the pointer
                x0 += 8 * incx;
                y0 += 8 * incy;
            }

            for ( ; i < (n & (~0x03)); i += 4)
            {
                // Loading the input values
                xv[0] = _mm_loadu_pd((double *)(x0 + 0 * incx));
                xv[1] = _mm_loadu_pd((double *)(x0 + 1 * incx));
                xv[2] = _mm_loadu_pd((double *)(x0 + 2 * incx));
                xv[3] = _mm_loadu_pd((double *)(x0 + 3 * incx));

                // Storing the values to destination
                _mm_storeu_pd((double *)(y0 + incy * 0), xv[0]);
                _mm_storeu_pd((double *)(y0 + incy * 1), xv[1]);
                _mm_storeu_pd((double *)(y0 + incy * 2), xv[2]);
                _mm_storeu_pd((double *)(y0 + incy * 3), xv[3]);

                // Increment the pointer
                x0 += 4 * incx;
                y0 += 4 * incy;
            }

            for ( ; i < (n & (~0x01)); i += 2)
            {
                // Loading the input values
                xv[0] = _mm_loadu_pd((double *)(x0 + 0 * incx));
                xv[1] = _mm_loadu_pd((double *)(x0 + 1 * incx));

                // Storing the values to desti-nation
                _mm_storeu_pd((double *)(y0 + incy * 0), xv[0]);
                _mm_storeu_pd((double *)(y0 + incy * 1), xv[1]);

                // Increment the pointer
                x0 += 2 * incx;
                y0 += 2 * incy;
            }

            for ( ; i < n; i += 1)
            {
                // Loading the input values
                xv[0] = _mm_loadu_pd((double *)(x0 + 0 * incx));

                // Storing the values to destination
                _mm_storeu_pd((double *)(y0 + incy * 0), xv[0]);

                // Increment the pointer
                x0 += 1 * incx;
                y0 += 1 * incy;
            }
        }
    }
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2)
}
