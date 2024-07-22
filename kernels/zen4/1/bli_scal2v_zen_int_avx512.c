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

#include "blis.h"
#include <immintrin.h>

// This kernel performs  y := alpha * conjx(x)
void bli_dscal2v_zen_int_avx512
     (
       conj_t           conjx,
       dim_t            n,
       double*  restrict alpha,
       double*  restrict x, inc_t incx,
       double*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    // If the vector dimension is zero, return early.
    if ( bli_zero_dim1( n ) )
        return;

    // Redirecting to DSETV, if alpha is 0
    if ( PASTEMAC( d, eq0 )( *alpha ) )
    {
        double *zero = PASTEMAC( d, 0 );

        bli_dsetv_zen_int_avx512
        (
            BLIS_NO_CONJUGATE,
            n,
            zero,
            y, incy,
            cntx
        );

        return;
    }
    // Redirecting to DCOPYV, if alpha is 1
    else if ( PASTEMAC( d, eq1 )( *alpha ) )
    {
        bli_dcopyv_zen4_asm_avx512
        (
            conjx,
            n,
            x, incx,
            y, incy,
            cntx
        );

        return;
    }

    // Initializing the pointer aliases and iterator
    dim_t i = 0;
    double *x0 = x;
    double *y0 = y;

    // Handling unit-strided inputs
    if ( incx == 1 && incy == 1 )
    {
        // Vectors to be used in the scal2v computation
        __m512d x_vec[8], alphav;

        // Broadcasting alpha to a 512-bit register
        alphav = _mm512_set1_pd( *alpha );

        const dim_t n_elem_per_reg = 8;

        // Iterating in blocks of 64 elements
        for ( ; ( i + 63 ) < n; i += 64 )
        {
            // Loading X vector
            x_vec[0] = _mm512_loadu_pd( x0 );
            x_vec[1] = _mm512_loadu_pd( x0 + 1 * n_elem_per_reg );
            x_vec[2] = _mm512_loadu_pd( x0 + 2 * n_elem_per_reg );
            x_vec[3] = _mm512_loadu_pd( x0 + 3 * n_elem_per_reg );

            // Scaling X vector with alpha
            x_vec[0] = _mm512_mul_pd( x_vec[0], alphav );
            x_vec[1] = _mm512_mul_pd( x_vec[1], alphav );
            x_vec[2] = _mm512_mul_pd( x_vec[2], alphav );
            x_vec[3] = _mm512_mul_pd( x_vec[3], alphav );

            // Storing onto Y
            _mm512_storeu_pd( y0, x_vec[0] );
            _mm512_storeu_pd( y0 + 1 * n_elem_per_reg, x_vec[1] );
            _mm512_storeu_pd( y0 + 2 * n_elem_per_reg, x_vec[2] );
            _mm512_storeu_pd( y0 + 3 * n_elem_per_reg, x_vec[3] );

            // Loading X vector
            x_vec[4] = _mm512_loadu_pd( x0 + 4 * n_elem_per_reg );
            x_vec[5] = _mm512_loadu_pd( x0 + 5 * n_elem_per_reg );
            x_vec[6] = _mm512_loadu_pd( x0 + 6 * n_elem_per_reg );
            x_vec[7] = _mm512_loadu_pd( x0 + 7 * n_elem_per_reg );

            // Scaling X vector with alpha
            x_vec[4] = _mm512_mul_pd( x_vec[4], alphav );
            x_vec[5] = _mm512_mul_pd( x_vec[5], alphav );
            x_vec[6] = _mm512_mul_pd( x_vec[6], alphav );
            x_vec[7] = _mm512_mul_pd( x_vec[7], alphav );

            // Storing onto Y
            _mm512_storeu_pd( y0 + 4 * n_elem_per_reg, x_vec[4] );
            _mm512_storeu_pd( y0 + 5 * n_elem_per_reg, x_vec[5] );
            _mm512_storeu_pd( y0 + 6 * n_elem_per_reg, x_vec[6] );
            _mm512_storeu_pd( y0 + 7 * n_elem_per_reg, x_vec[7] );

            // Adjusting the pointers for the next iteration
            x0 += 8 * n_elem_per_reg;
            y0 += 8 * n_elem_per_reg;
        }

        // Iterating in blocks of 32 elements
        for ( ; ( i + 31 ) < n; i += 32 )
        {
            // Loading X vector
            x_vec[0] = _mm512_loadu_pd( x0 );
            x_vec[1] = _mm512_loadu_pd( x0 + 1 * n_elem_per_reg );
            x_vec[2] = _mm512_loadu_pd( x0 + 2 * n_elem_per_reg );
            x_vec[3] = _mm512_loadu_pd( x0 + 3 * n_elem_per_reg );

            // Scaling X vector with alpha
            x_vec[0] = _mm512_mul_pd( x_vec[0], alphav );
            x_vec[1] = _mm512_mul_pd( x_vec[1], alphav );
            x_vec[2] = _mm512_mul_pd( x_vec[2], alphav );
            x_vec[3] = _mm512_mul_pd( x_vec[3], alphav );

            // Storing onto Y
            _mm512_storeu_pd( y0, x_vec[0] );
            _mm512_storeu_pd( y0 + 1 * n_elem_per_reg, x_vec[1] );
            _mm512_storeu_pd( y0 + 2 * n_elem_per_reg, x_vec[2] );
            _mm512_storeu_pd( y0 + 3 * n_elem_per_reg, x_vec[3] );

            // Adjusting the pointers for the next iteration
            x0 += 4 * n_elem_per_reg;
            y0 += 4 * n_elem_per_reg;
        }

        // Iterating in blocks of 16 elements
        for ( ; ( i + 15 ) < n; i += 16 )
        {
            // Loading X vector
            x_vec[0] = _mm512_loadu_pd( x0 );
            x_vec[1] = _mm512_loadu_pd( x0 + 1 * n_elem_per_reg );

            // Scaling X vector with alpha
            x_vec[0] = _mm512_mul_pd( x_vec[0], alphav );
            x_vec[1] = _mm512_mul_pd( x_vec[1], alphav );

            // Storing onto Y
            _mm512_storeu_pd( y0, x_vec[0] );
            _mm512_storeu_pd( y0 + 1 * n_elem_per_reg, x_vec[1] );

            // Adjusting the pointers for the next iteration
            x0 += 2 * n_elem_per_reg;
            y0 += 2 * n_elem_per_reg;
        }

        // Iterating in blocks of 8 elements
        for ( ; ( i + 7 ) < n; i += 8 )
        {
            // Loading X vector
            x_vec[0] = _mm512_loadu_pd( x0 );

            // Scaling X vector with alpha
            x_vec[0] = _mm512_mul_pd( x_vec[0], alphav );

            // Storing onto Y
            _mm512_storeu_pd( y0, x_vec[0] );

            // Adjusting the pointers for the next iteration
            x0 += 1 * n_elem_per_reg;
            y0 += 1 * n_elem_per_reg;
        }

        // Handling the fringe case
        if ( i < n )
        {
            // Setting the mask for loading and storing the vectors
            __mmask8 n_mask = (1 << ( n - i )) - 1;

            // Loading X vector
            x_vec[0] = _mm512_maskz_loadu_pd( n_mask, x0 );

            // Scaling X vector with alpha
            x_vec[0] = _mm512_mul_pd( x_vec[0], alphav );

            // Storing onto Y
            _mm512_mask_storeu_pd( y0, n_mask, x_vec[0] );
        }
    }

    else
    {
        // Handling fringe case or non-unit strides
        for ( ; i < n; i += 1 )
        {
            *y0 = (*alpha) * (*x0);
            x0 += incx;
            y0 += incy;
        }
    }
}
