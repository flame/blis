/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2016 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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

#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"

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

/*
    Functionality
    -------------

    This function calculates the dot product of two vectors for
    type double complex.

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
void bli_zdotv_zen_int_avx512
     (
       conj_t             conjx,
       conj_t             conjy,
       dim_t              n,
       dcomplex* restrict x, inc_t incx,
       dcomplex* restrict y, inc_t incy,
       dcomplex* restrict rho,
       cntx_t*   restrict cntx
     )
{
    // Initialize local pointers.
    double* restrict x0 = (double*)x;
    double* restrict y0 = (double*)y;

    dcomplex rho0 = *bli_z0;

    conj_t conjx_use = conjx;
    if ( bli_is_conj( conjy ) )
        bli_toggle_conj( &conjx_use );

    dim_t i = 0;
    if ( incx == 1 && incy == 1 )
    {
        const dim_t n_elem_per_reg = 8;

        __m512d xv[8];
        __m512d yv[8];
        __m512d rhov[16];

        // Initialize rho accumulation vectors to 0.
        // rhov[0] - rhov[7] store the real part of intermediate result.
        // rhov[8] - rhov[15] store the imaginary part of intermediate result.
        rhov[0] = _mm512_setzero_pd();
        rhov[1] = _mm512_setzero_pd();
        rhov[2] = _mm512_setzero_pd();
        rhov[3] = _mm512_setzero_pd();
        rhov[4] = _mm512_setzero_pd();
        rhov[5] = _mm512_setzero_pd();
        rhov[6] = _mm512_setzero_pd();
        rhov[7] = _mm512_setzero_pd();
        rhov[8] = _mm512_setzero_pd();
        rhov[9] = _mm512_setzero_pd();
        rhov[10] = _mm512_setzero_pd();
        rhov[11] = _mm512_setzero_pd();
        rhov[12] = _mm512_setzero_pd();
        rhov[13] = _mm512_setzero_pd();
        rhov[14] = _mm512_setzero_pd();
        rhov[15] = _mm512_setzero_pd();

        /**
         * General Algorithm:
         *
         * xv[0]   = x0R x0I x1R x1I ...
         * yv[0]   = y0R y0I y1R y1I ...
         * rhov[0] = xv[0] * yv[0] + rhov[0]
         *         = x0R*y0R x0I*y0I x1R*y1R x1I*y0I ...
         * yv[0]   = permute(0x55)
         *         = y0I y0R y1I y1R ...
         * rhov[8] = xv[0] * yv[0] + rhov[8]
         *         = x0R*y0I x0I*y0R x1R*y1I x1I*y1R ...
        */

        // Processing 32 dcomplex elements per iteration.
        for ( ; (i + 31) < n; i += 32 )
        {
            // Load elements from x vector.
            xv[0] = _mm512_loadu_pd( x0 + 0*n_elem_per_reg );
            xv[1] = _mm512_loadu_pd( x0 + 1*n_elem_per_reg );
            xv[2] = _mm512_loadu_pd( x0 + 2*n_elem_per_reg );
            xv[3] = _mm512_loadu_pd( x0 + 3*n_elem_per_reg );
            xv[4] = _mm512_loadu_pd( x0 + 4*n_elem_per_reg );
            xv[5] = _mm512_loadu_pd( x0 + 5*n_elem_per_reg );
            xv[6] = _mm512_loadu_pd( x0 + 6*n_elem_per_reg );
            xv[7] = _mm512_loadu_pd( x0 + 7*n_elem_per_reg );

            // Load elements from y vector.
            yv[0] = _mm512_loadu_pd( y0 + 0*n_elem_per_reg );
            yv[1] = _mm512_loadu_pd( y0 + 1*n_elem_per_reg );
            yv[2] = _mm512_loadu_pd( y0 + 2*n_elem_per_reg );
            yv[3] = _mm512_loadu_pd( y0 + 3*n_elem_per_reg );
            yv[4] = _mm512_loadu_pd( y0 + 4*n_elem_per_reg );
            yv[5] = _mm512_loadu_pd( y0 + 5*n_elem_per_reg );
            yv[6] = _mm512_loadu_pd( y0 + 6*n_elem_per_reg );
            yv[7] = _mm512_loadu_pd( y0 + 7*n_elem_per_reg );

            // Operation: rhov = xv * yv + rhov
            rhov[0] = _mm512_fmadd_pd( xv[0], yv[0], rhov[0] );
            rhov[1] = _mm512_fmadd_pd( xv[1], yv[1], rhov[1] );
            rhov[2] = _mm512_fmadd_pd( xv[2], yv[2], rhov[2] );
            rhov[3] = _mm512_fmadd_pd( xv[3], yv[3], rhov[3] );
            rhov[4] = _mm512_fmadd_pd( xv[4], yv[4], rhov[4] );
            rhov[5] = _mm512_fmadd_pd( xv[5], yv[5], rhov[5] );
            rhov[6] = _mm512_fmadd_pd( xv[6], yv[6], rhov[6] );
            rhov[7] = _mm512_fmadd_pd( xv[7], yv[7], rhov[7] );

            // Operation: yv -> yv'
            // yv  = y0R y0I y1R y1I ...
            // yv' = y0I y0R y1I y1R ...
            yv[0] = _mm512_permute_pd( yv[0], 0x55 );
            yv[1] = _mm512_permute_pd( yv[1], 0x55 );
            yv[2] = _mm512_permute_pd( yv[2], 0x55 );
            yv[3] = _mm512_permute_pd( yv[3], 0x55 );
            yv[4] = _mm512_permute_pd( yv[4], 0x55 );
            yv[5] = _mm512_permute_pd( yv[5], 0x55 );
            yv[6] = _mm512_permute_pd( yv[6], 0x55 );
            yv[7] = _mm512_permute_pd( yv[7], 0x55 );

            // Operation: rhov = xv * yv' + rhov
            rhov[8] = _mm512_fmadd_pd( xv[0], yv[0], rhov[8] );
            rhov[9] = _mm512_fmadd_pd( xv[1], yv[1], rhov[9] );
            rhov[10] = _mm512_fmadd_pd( xv[2], yv[2], rhov[10] );
            rhov[11] = _mm512_fmadd_pd( xv[3], yv[3], rhov[11] );
            rhov[12] = _mm512_fmadd_pd( xv[4], yv[4], rhov[12] );
            rhov[13] = _mm512_fmadd_pd( xv[5], yv[5], rhov[13] );
            rhov[14] = _mm512_fmadd_pd( xv[6], yv[6], rhov[14] );
            rhov[15] = _mm512_fmadd_pd( xv[7], yv[7], rhov[15] );

            // Increment x0 and y0 vector pointers.
            x0 += 8 * n_elem_per_reg;
            y0 += 8 * n_elem_per_reg;
        }

        // Accumulating intermediate results to rhov[0] and rhov[8].
        rhov[0] = _mm512_add_pd( rhov[0], rhov[4] );
        rhov[0] = _mm512_add_pd( rhov[0], rhov[5] );
        rhov[0] = _mm512_add_pd( rhov[0], rhov[6] );
        rhov[0] = _mm512_add_pd( rhov[0], rhov[7] );

        rhov[8] = _mm512_add_pd( rhov[8], rhov[12] );
        rhov[8] = _mm512_add_pd( rhov[8], rhov[13] );
        rhov[8] = _mm512_add_pd( rhov[8], rhov[14] );
        rhov[8] = _mm512_add_pd( rhov[8], rhov[15] );

        // Processing 16 dcomplex elements per iteration.
        for ( ; (i + 15) < n; i += 16 )
        {
            xv[0] = _mm512_loadu_pd( x0 + 0*n_elem_per_reg );
            xv[1] = _mm512_loadu_pd( x0 + 1*n_elem_per_reg );
            xv[2] = _mm512_loadu_pd( x0 + 2*n_elem_per_reg );
            xv[3] = _mm512_loadu_pd( x0 + 3*n_elem_per_reg );

            yv[0] = _mm512_loadu_pd( y0 + 0*n_elem_per_reg );
            yv[1] = _mm512_loadu_pd( y0 + 1*n_elem_per_reg );
            yv[2] = _mm512_loadu_pd( y0 + 2*n_elem_per_reg );
            yv[3] = _mm512_loadu_pd( y0 + 3*n_elem_per_reg );

            rhov[0] = _mm512_fmadd_pd( xv[0], yv[0], rhov[0] );
            rhov[1] = _mm512_fmadd_pd( xv[1], yv[1], rhov[1] );
            rhov[2] = _mm512_fmadd_pd( xv[2], yv[2], rhov[2] );
            rhov[3] = _mm512_fmadd_pd( xv[3], yv[3], rhov[3] );

            yv[0] = _mm512_permute_pd( yv[0], 0x55 );
            yv[1] = _mm512_permute_pd( yv[1], 0x55 );
            yv[2] = _mm512_permute_pd( yv[2], 0x55 );
            yv[3] = _mm512_permute_pd( yv[3], 0x55 );

            rhov[8] = _mm512_fmadd_pd( xv[0], yv[0], rhov[8] );
            rhov[9] = _mm512_fmadd_pd( xv[1], yv[1], rhov[9] );
            rhov[10] = _mm512_fmadd_pd( xv[2], yv[2], rhov[10] );
            rhov[11] = _mm512_fmadd_pd( xv[3], yv[3], rhov[11] );

            x0 += 4 * n_elem_per_reg;
            y0 += 4 * n_elem_per_reg;
        }

        rhov[0] = _mm512_add_pd( rhov[0], rhov[3] );
        rhov[8] = _mm512_add_pd( rhov[8], rhov[11] );

        // Processing 12 dcomplex elements per iteration.
        for ( ; (i + 11) < n; i += 12 )
        {
            xv[0] = _mm512_loadu_pd( x0 + 0*n_elem_per_reg );
            xv[1] = _mm512_loadu_pd( x0 + 1*n_elem_per_reg );
            xv[2] = _mm512_loadu_pd( x0 + 2*n_elem_per_reg );

            yv[0] = _mm512_loadu_pd( y0 + 0*n_elem_per_reg );
            yv[1] = _mm512_loadu_pd( y0 + 1*n_elem_per_reg );
            yv[2] = _mm512_loadu_pd( y0 + 2*n_elem_per_reg );

            rhov[0] = _mm512_fmadd_pd( xv[0], yv[0], rhov[0] );
            rhov[1] = _mm512_fmadd_pd( xv[1], yv[1], rhov[1] );
            rhov[2] = _mm512_fmadd_pd( xv[2], yv[2], rhov[2] );

            yv[0] = _mm512_permute_pd( yv[0], 0x55 );
            yv[1] = _mm512_permute_pd( yv[1], 0x55 );
            yv[2] = _mm512_permute_pd( yv[2], 0x55 );

            rhov[8] = _mm512_fmadd_pd( xv[0], yv[0], rhov[8] );
            rhov[9] = _mm512_fmadd_pd( xv[1], yv[1], rhov[9] );
            rhov[10] = _mm512_fmadd_pd( xv[2], yv[2], rhov[10] );

            x0 += 3 * n_elem_per_reg;
            y0 += 3 * n_elem_per_reg;
        }

        rhov[0] = _mm512_add_pd( rhov[0], rhov[2] );
        rhov[8] = _mm512_add_pd( rhov[8], rhov[10] );

        // Processing 8 dcomplex elements per iteration.
        for ( ; (i + 7) < n; i += 8 )
        {
            xv[0] = _mm512_loadu_pd( x0 + 0*n_elem_per_reg );
            xv[1] = _mm512_loadu_pd( x0 + 1*n_elem_per_reg );

            yv[0] = _mm512_loadu_pd( y0 + 0*n_elem_per_reg );
            yv[1] = _mm512_loadu_pd( y0 + 1*n_elem_per_reg );

            rhov[0] = _mm512_fmadd_pd( xv[0], yv[0], rhov[0] );
            rhov[1] = _mm512_fmadd_pd( xv[1], yv[1], rhov[1] );

            yv[0] = _mm512_permute_pd( yv[0], 0x55 );
            yv[1] = _mm512_permute_pd( yv[1], 0x55 );

            rhov[8] = _mm512_fmadd_pd( xv[0], yv[0], rhov[8] );
            rhov[9] = _mm512_fmadd_pd( xv[1], yv[1], rhov[9] );

            x0 += 2 * n_elem_per_reg;
            y0 += 2 * n_elem_per_reg;
        }

        rhov[0] = _mm512_add_pd( rhov[0], rhov[1] );
        rhov[8] = _mm512_add_pd( rhov[8], rhov[9] );

        // Processing 4 dcomplex elements per iteration.
        for ( ; (i + 3) < n; i += 4 )
        {
            xv[0] = _mm512_loadu_pd( x0 + 0*n_elem_per_reg );

            yv[0] = _mm512_loadu_pd( y0 + 0*n_elem_per_reg );

            rhov[0] = _mm512_fmadd_pd( xv[0], yv[0], rhov[0] );

            yv[0] = _mm512_permute_pd( yv[0], 0x55 );

            rhov[8] = _mm512_fmadd_pd( xv[0], yv[0], rhov[8] );

            x0 += 1 * n_elem_per_reg;
            y0 += 1 * n_elem_per_reg;
        }

        // Processing the remainder elements.
        if( i < n )
        {
            // Setting the mask bit based on remaining elements
            // Since each dcomplex elements corresponds to 2 doubles
            // we need to load and store 2*(m-i) elements.
            __mmask8 mask = (1 << (2 * (n-i)) ) - 1;

            // Clearing the rhov[1] register for mask-load.
            rhov[1] = _mm512_setzero_pd();

            xv[0] = _mm512_mask_loadu_pd( rhov[1], mask, x0 );

            yv[0] = _mm512_mask_loadu_pd( rhov[1], mask, y0 );

            rhov[0] = _mm512_fmadd_pd( xv[0], yv[0], rhov[0] );

            yv[0] = _mm512_permute_pd( yv[0], 0x55 );

            rhov[8] = _mm512_fmadd_pd( xv[0], yv[0], rhov[8] );
        }

        // Initialize mask for reduce-add based on conjugate.
        __m512d mask = _mm512_set_pd(-1, 1, -1, 1, -1, 1, -1, 1);
        if ( bli_is_conj( conjx_use ) )
        {
            rho0.real = _mm512_reduce_add_pd( rhov[0] );
            rhov[8] = _mm512_mul_pd( rhov[8], mask );
            rho0.imag = _mm512_reduce_add_pd( rhov[8] );
        }
        else
        {
            rhov[0] = _mm512_mul_pd( rhov[0], mask );
            rho0.real = _mm512_reduce_add_pd( rhov[0] );
            rho0.imag = _mm512_reduce_add_pd( rhov[8] );
        }
    }
    else    // Non-Unit Increments
    {
        if ( !bli_is_conj( conjx_use ) )
        {
            for ( i = 0; i < n; ++i )
            {
                const double x0c = *x0;
                const double y0c = *y0;

                const double x1c = *( x0 + 1 );
                const double y1c = *( y0 + 1 );

                rho0.real += x0c * y0c - x1c * y1c;
                rho0.imag += x0c * y1c + x1c * y0c;

                x0 += incx * 2;
                y0 += incy * 2;
            }
        }
        else
        {
            for ( i = 0; i < n; ++i )
            {
                const double x0c = *x0;
                const double y0c = *y0;

                const double x1c = *( x0 + 1 );
                const double y1c = *( y0 + 1 );

                rho0.real += x0c * y0c + x1c * y1c;
                rho0.imag += x0c * y1c - x1c * y0c;

                x0 += incx * 2;
                y0 += incy * 2;
            }
        }
    }

    // Negate the sign of imaginary value when conjy is enabled.
    if ( bli_is_conj( conjy ) )
        rho0.imag = -rho0.imag;

    // Copy the result to rho.
    PASTEMAC(z,copys)( rho0, *rho );
}

/*
    Functionality
    -------------

    This function calculates the dot product of two vectors for
    type double complex.

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
void bli_zdotv_zen4_asm_avx512
     (
       conj_t             conjx,
       conj_t             conjy,
       dim_t              n,
       dcomplex* restrict x, inc_t incx,
       dcomplex* restrict y, inc_t incy,
       dcomplex* restrict rho,
       cntx_t*   restrict cntx
     )
{
    // Initialize local pointers.
    double* restrict x0 = (double*)x;
    double* restrict y0 = (double*)y;

    dcomplex rho0 = *bli_z0;
    double* restrict rho0R = &rho0.real;
    double* restrict rho0I = &rho0.imag;

    // Using a local unit value for setting a unit register.
    double one_l = 1.0;
    double* restrict one = &one_l;

    conj_t conjx_use = conjx;
    if ( bli_is_conj( conjy ) )
        bli_toggle_conj( &conjx_use );

    // Copying conjx_use to  a local conj variable for simple condition check
    // within inline assembly.
    dim_t conj = 0;
    if ( bli_is_conj( conjx_use ) ) conj = 1;

    if ( incx == 1 && incy == 1 )   // Inline ASM used to handle unit-increment.
    {
        begin_asm()

        mov( var( n ), rsi )        // load n to rsi.
        mov( var( x0 ), rax )       // load location of x vec to rax.
        mov( var( y0 ), rbx )       // load location of y vec to rbx.

        // Initialize 16 registers (zmm0 - zmm15) to zero.
        // These will be used for accumulation of rho.
        // zmm0 - zmm7: real intermediate values of rho.
        // zmm8 - zmm15: imaginary intermediate values of rho.
        vxorpd( zmm0, zmm0, zmm0 )
        vxorpd( zmm1, zmm1, zmm1 )
        vxorpd( zmm2, zmm2, zmm2 )
        vxorpd( zmm3, zmm3, zmm3 )
        vxorpd( zmm4, zmm4, zmm4 )
        vxorpd( zmm5, zmm5, zmm5 )
        vxorpd( zmm6, zmm6, zmm6 )
        vxorpd( zmm7, zmm7, zmm7 )
        vxorpd( zmm8, zmm8, zmm8 )
        vxorpd( zmm9, zmm9, zmm9 )
        vxorpd( zmm10, zmm10, zmm10 )
        vxorpd( zmm11, zmm11, zmm11 )
        vxorpd( zmm12, zmm12, zmm12 )
        vxorpd( zmm13, zmm13, zmm13 )
        vxorpd( zmm14, zmm14, zmm14 )
        vxorpd( zmm15, zmm15, zmm15 )


        /**
         * General Algorithm:
         *
         * zmm16 = x0R x0I x1R x1I ...
         * zmm24 = y0R y0I y1R y1I ...
         * zmm0  = zmm16 * zmm24 + zmm0
         *       = x0R*y0R x0I*y0I x1R*y1R x1I*y0I ...
         * zmm24 = permute(0x55)
         *       = y0I y0R y1I y1R ...
         * zmm8  = zmm16 * zmm24 + zmm8
         *       = x0R*y0I x0I*y0R x1R*y1I x1I*y1R ...
        */


        // Each iteration of L32 handles 32 elements.
        // Each zmm register can handle 8 doubles, i.e., 4 dcomplex elements.
        // Thus, using 8 registers each for x and y vectors we handle 32
        // elements in every iteration of the loop.
        label( .L32 )
        cmp( imm(32), rsi )
        jl( .ACCUM32 )

        // Alternate loads from x & y.
        vmovupd(      ( rax ), zmm16 )      // load from x
        vmovupd(      ( rbx ), zmm24 )      // load from y
        vmovupd(  0x40( rax ), zmm17 )
        vmovupd(  0x40( rbx ), zmm25 )
        vmovupd(  0x80( rax ), zmm18 )
        vmovupd(  0x80( rbx ), zmm26 )
        vmovupd(  0xC0( rax ), zmm19 )
        vmovupd(  0xC0( rbx ), zmm27 )
        vmovupd( 0x100( rax ), zmm20 )
        vmovupd( 0x100( rbx ), zmm28 )
        vmovupd( 0x140( rax ), zmm21 )
        vmovupd( 0x140( rbx ), zmm29 )
        vmovupd( 0x180( rax ), zmm22 )
        vmovupd( 0x180( rbx ), zmm30 )
        vmovupd( 0x1C0( rax ), zmm23 )
        vmovupd( 0x1C0( rbx ), zmm31 )

        // Increment x0 and y0 vector pointers.
        add( imm(512), rax )
        add( imm(512), rbx )

        // Operation: rhov = xv * yv + rhov
        vfmadd231pd( zmm16, zmm24, zmm0 )
        vfmadd231pd( zmm17, zmm25, zmm1 )
        vfmadd231pd( zmm18, zmm26, zmm2 )
        vfmadd231pd( zmm19, zmm27, zmm3 )
        vfmadd231pd( zmm20, zmm28, zmm4 )
        vfmadd231pd( zmm21, zmm29, zmm5 )
        vfmadd231pd( zmm22, zmm30, zmm6 )
        vfmadd231pd( zmm23, zmm31, zmm7 )

        // Operation: yv -> yv'
        // yv  = y0R y0I y1R y1I ...
        // yv' = y0I y0R y1I y1R ...
        vpermilpd( imm(0x55), zmm24, zmm24 )
        vpermilpd( imm(0x55), zmm25, zmm25 )
        vpermilpd( imm(0x55), zmm26, zmm26 )
        vpermilpd( imm(0x55), zmm27, zmm27 )
        vpermilpd( imm(0x55), zmm28, zmm28 )
        vpermilpd( imm(0x55), zmm29, zmm29 )
        vpermilpd( imm(0x55), zmm30, zmm30 )
        vpermilpd( imm(0x55), zmm31, zmm31 )

        // Operation: rhov = xv * yv' + rhov
        vfmadd231pd( zmm16, zmm24, zmm8 )
        vfmadd231pd( zmm17, zmm25, zmm9 )
        vfmadd231pd( zmm18, zmm26, zmm10 )
        vfmadd231pd( zmm19, zmm27, zmm11 )
        vfmadd231pd( zmm20, zmm28, zmm12 )
        vfmadd231pd( zmm21, zmm29, zmm13 )
        vfmadd231pd( zmm22, zmm30, zmm14 )
        vfmadd231pd( zmm23, zmm31, zmm15 )

        // Loop decrement.
        sub( imm(32), rsi )
        jmp( .L32 )

        
        // Accumulating intermediate results to zmm0 and zmm8.
        label( .ACCUM32 )
        vaddpd( zmm4, zmm0, zmm0 )
        vaddpd( zmm5, zmm0, zmm0 )
        vaddpd( zmm6, zmm0, zmm0 )
        vaddpd( zmm7, zmm0, zmm0 )

        vaddpd( zmm12, zmm8, zmm8 )
        vaddpd( zmm13, zmm8, zmm8 )
        vaddpd( zmm14, zmm8, zmm8 )
        vaddpd( zmm15, zmm8, zmm8 )

        // Each iteration of L16 handles 16 elements.
        label( .L16 )
        cmp( imm(16), rsi )
        jl( .ACCUM16 )

        // Alternate loads from x & y.
        vmovupd(      ( rax ), zmm16 )      // load from x
        vmovupd(      ( rbx ), zmm24 )      // load from y
        vmovupd(  0x40( rax ), zmm17 )
        vmovupd(  0x40( rbx ), zmm25 )
        vmovupd(  0x80( rax ), zmm18 )
        vmovupd(  0x80( rbx ), zmm26 )
        vmovupd(  0xC0( rax ), zmm19 )
        vmovupd(  0xC0( rbx ), zmm27 )

        // Increment x0 and y0 vector pointers.
        add( imm(256), rax )
        add( imm(256), rbx )

        // Operation: rhov = xv * yv + rhov
        vfmadd231pd( zmm16, zmm24, zmm0 )
        vfmadd231pd( zmm17, zmm25, zmm1 )
        vfmadd231pd( zmm18, zmm26, zmm2 )
        vfmadd231pd( zmm19, zmm27, zmm3 )

        // Operation: yv -> yv'
        // yv  = y0R y0I y1R y1I ...
        // yv' = y0I y0R y1I y1R ...
        vpermilpd( imm(0x55), zmm24, zmm24 )
        vpermilpd( imm(0x55), zmm25, zmm25 )
        vpermilpd( imm(0x55), zmm26, zmm26 )
        vpermilpd( imm(0x55), zmm27, zmm27 )

        // Operation: rhov = xv * yv' + rhov
        vfmadd231pd( zmm16, zmm24, zmm8 )
        vfmadd231pd( zmm17, zmm25, zmm9 )
        vfmadd231pd( zmm18, zmm26, zmm10 )
        vfmadd231pd( zmm19, zmm27, zmm11 )

        // Loop decrement.
        sub( imm(16), rsi )
        jmp( .L16 )


        // Accumulating intermediate results to zmm0 and zmm8.
        label( .ACCUM16 )
        vaddpd( zmm2, zmm0, zmm0 )
        vaddpd( zmm3, zmm0, zmm0 )

        vaddpd( zmm10, zmm8, zmm8 )
        vaddpd( zmm11, zmm8, zmm8 )

        // Each iteration of L8 handles 8 elements.
        label( .L8 )
        cmp( imm(8), rsi )
        jl( .ACCUM8 )

        // Alternate loads from x & y.
        vmovupd(      ( rax ), zmm16 )      // load from x
        vmovupd(      ( rbx ), zmm24 )      // load from y
        vmovupd( 0x40 ( rax ), zmm17 )
        vmovupd( 0x40 ( rbx ), zmm25 )

        // Increment x0 and y0 vector pointers.
        add( imm(128), rax )
        add( imm(128), rbx )

        // Operation: rhov = xv * yv + rhov
        vfmadd231pd( zmm16, zmm24, zmm0 )
        vfmadd231pd( zmm17, zmm25, zmm1 )

        // Operation: yv -> yv'
        // yv  = y0R y0I y1R y1I ...
        // yv' = y0I y0R y1I y1R ...
        vpermilpd( imm(0x55), zmm24, zmm24 )
        vpermilpd( imm(0x55), zmm25, zmm25 )

        // Operation: rhov = xv * yv' + rhov
        vfmadd231pd( zmm16, zmm24, zmm8 )
        vfmadd231pd( zmm17, zmm25, zmm9 )

        // Loop decrement.
        sub( imm(8), rsi )
        jmp( .L8 )


        // Accumulating intermediate results to zmm0 and zmm8.
        label( .ACCUM8 )
        vaddpd( zmm1, zmm0, zmm0 )
        vaddpd( zmm9, zmm8, zmm8 )


        // Each iteration of L4 handles 4 elements.
        label( .L4 )
        cmp( imm(4), rsi )
        jl( .FRINGE )

        // Alternate loads from x & y.
        vmovupd(      ( rax ), zmm16 )      // load from x
        vmovupd(      ( rbx ), zmm24 )      // load from y

        // Increment x0 and y0 vector pointers.
        add( imm(64), rax )
        add( imm(64), rbx )

        // Operation: rhov = xv * yv + rhov
        vfmadd231pd( zmm16, zmm24, zmm0 )

        // Operation: yv -> yv'
        // yv  = y0R y0I y1R y1I ...
        // yv' = y0I y0R y1I y1R ...
        vpermilpd( imm(0x55), zmm24, zmm24 )

        // Operation: rhov = xv * yv' + rhov
        vfmadd231pd( zmm16, zmm24, zmm8 )

        // Loop decrement.
        sub( imm(4), rsi )
        jmp( .L4 )


        // Fringe case to process the remainder elements.
        LABEL( .FRINGE )
        cmp( imm(0x0), rsi )
        je( .CONJ )

        vxorpd( zmm16, zmm16, zmm16 )
        vxorpd( zmm24, zmm24, zmm24 )
        mov( imm(255), ecx )
        shlx( esi, ecx, ecx )
        shlx( esi, ecx, ecx )
        xor( imm(255), ecx )
        kmovw( ecx, K(1) )

        vmovupd( mem(rax), zmm16 MASK_(K(1)) )

        vmovupd( mem(rbx), zmm24 MASK_(K(1)) )

        vfmadd231pd( zmm16, zmm24, zmm0 )

        vpermilpd( imm(0x55), zmm24, zmm24 )

        vfmadd231pd( zmm16, zmm24, zmm8 )


        // Handling conjugates.
        LABEL( .CONJ )
        // set zmm1 to all zeros
        vxorpd( xmm1, xmm1, xmm1 )
        // broadcast one (1) to zmm2
        mov( var(one), rax )
        vbroadcastsd( (rax), zmm2 )
        vfmsubadd231pd( zmm1, zmm2, zmm2 )

        // load rho0R and rho0I into memory.
        mov( var(rho0R), rax )
        mov( var(rho0I), rbx )

        mov( var(conj), rcx)
        cmp( imm(0x0), rcx )
        je( .NOCONJX)

        // if conjx_use
        label( .CONJX )
        vextractf64x4( imm(0x1), zmm0, ymm2 )
        vaddpd( ymm0, ymm2, ymm0 )
        vextractf128( imm(0x1), ymm0, xmm2 )
        vaddpd( xmm2, xmm0, xmm0 )
        vshufpd( imm(0x1), xmm0, xmm0, xmm2 )
        vaddpd( xmm2, xmm0, xmm0 )
        vmovupd( xmm0, (rax) )      // store result to rho0R

        vmulpd( zmm1, zmm8, zmm8 )
        vextractf64x4( imm(0x1), zmm8, ymm2 )
        vaddpd( ymm8, ymm2, ymm8 )
        vextractf128( imm(0x1), ymm8, xmm2 )
        vaddpd( xmm2, xmm8, xmm8 )
        vshufpd( imm(0x1), xmm8, xmm8, xmm2 )
        vaddpd( xmm2, xmm8, xmm8 )
        vmovupd( xmm8, (rbx) )      // store result to rho0I
        jmp( .END )

        // if !conjx_use
        label( .NOCONJX )
        vmulpd( zmm2, zmm0, zmm0 )
        vextractf64x4( imm(0x1), zmm0, ymm2 )
        vaddpd( ymm0, ymm2, ymm0 )
        vextractf128( imm(0x1), ymm0, xmm2 )
        vaddpd( xmm2, xmm0, xmm0 )
        vshufpd( imm(0x1), xmm0, xmm0, xmm2 )
        vaddpd( xmm2, xmm0, xmm0 )
        vmovupd( xmm0, (rax) )      // store result to rho0R

        vextractf64x4( imm(0x1), zmm8, ymm2 )
        vaddpd( ymm8, ymm2, ymm8 )
        vextractf128( imm(0x1), ymm8, xmm2 )
        vaddpd( xmm2, xmm8, xmm8 )
        vshufpd( imm(0x1), xmm8, xmm8, xmm2 )
        vaddpd( xmm2, xmm8, xmm8 )
        vmovupd( xmm8, (rbx) )      // store result to rho0I

        label( .END )

        end_asm(
            : // output operands (none)
            : // input operands
            [n] "m" (n),
            [x0] "m" (x0),
            [y0] "m" (y0),
            [rho0R] "m" (rho0R),
            [rho0I] "m" (rho0I),
            [one]   "m" (one),
            [conj] "m" (conj)
            : // register clobber list
            "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
            "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
            "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm12",
            "zmm0", "zmm1", "zmm2", "zmm3",
            "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
            "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
            "zmm16", "zmm17", "zmm18", "zmm19",
            "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
            "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
            "k1", "xmm8", "ymm0", "ymm2", "ymm8", "memory"
        )

        rho0.real = *rho0R;
        rho0.imag = *rho0I;
    }
    else    // Non-Unit Increments
    {
        dim_t i = 0;
        if ( !bli_is_conj( conjx_use ) )
        {
            for ( i = 0; i < n; ++i )
            {
                const double x0c = *x0;
                const double y0c = *y0;

                const double x1c = *( x0 + 1 );
                const double y1c = *( y0 + 1 );

                rho0.real += x0c * y0c - x1c * y1c;
                rho0.imag += x0c * y1c + x1c * y0c;

                x0 += incx * 2;
                y0 += incy * 2;
            }
        }
        else
        {
            for ( i = 0; i < n; ++i )
            {
                const double x0c = *x0;
                const double y0c = *y0;

                const double x1c = *( x0 + 1 );
                const double y1c = *( y0 + 1 );

                rho0.real += x0c * y0c + x1c * y1c;
                rho0.imag += x0c * y1c - x1c * y0c;

                x0 += incx * 2;
                y0 += incy * 2;
            }
        }
    }

    // Negate the sign of imaginary value when conjy is enabled.
    if ( bli_is_conj( conjy ) )
        rho0.imag = -rho0.imag;

    // Copy the result to rho.
    PASTEMAC(z,copys)( rho0, *rho );
}
