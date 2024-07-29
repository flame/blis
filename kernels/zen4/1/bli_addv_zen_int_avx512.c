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

void bli_daddv_zen_int_avx512
     (
       conj_t           conjx,
       dim_t            n,
       double*  restrict x, inc_t incx,
       double*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
  const dim_t num_elem_per_reg = 8;
  dim_t       i = 0;
  __m512d      yv[8];

  // If the vector dimension is zero return early.
  if ( bli_zero_dim1( n ) ) return;

  double *x0 = x;
  double *y0 = y;

  if ( incx == 1 && incy ==1  )
  {
    // n & (~0x3F) = n & 0xFFFFFFC0 -> this masks the numbers less than 64,
    // the copy operation will be done for the multiples of 64
    for ( ; i < (n & (~0x3F)); i += 64 )
    {
      // Loading input values
      yv[0] =  _mm512_loadu_pd( y0 );
      yv[1] =  _mm512_loadu_pd( y0 + 1*num_elem_per_reg );
      yv[2] =  _mm512_loadu_pd( y0 + 2*num_elem_per_reg );
      yv[3] =  _mm512_loadu_pd( y0 + 3*num_elem_per_reg );
      yv[4] =  _mm512_loadu_pd( y0 + 4*num_elem_per_reg );
      yv[5] =  _mm512_loadu_pd( y0 + 5*num_elem_per_reg );
      yv[6] =  _mm512_loadu_pd( y0 + 6*num_elem_per_reg );
      yv[7] =  _mm512_loadu_pd( y0 + 7*num_elem_per_reg );

      // y := y + x
      yv[0] = _mm512_add_pd
                (
                  _mm512_loadu_pd( x0 ),
                  yv[0]
                );
      yv[1] = _mm512_add_pd
                (
                  _mm512_loadu_pd( x0 + 1*num_elem_per_reg ),
                  yv[1]
                );
      yv[2] = _mm512_add_pd
                (
                  _mm512_loadu_pd( x0 + 2*num_elem_per_reg ),
                  yv[2]
                );
      yv[3] = _mm512_add_pd
                (
                  _mm512_loadu_pd( x0 + 3*num_elem_per_reg ),
                  yv[3]
                );
      yv[4] = _mm512_add_pd
                (
                  _mm512_loadu_pd( x0 + 4*num_elem_per_reg ),
                  yv[4]
                );
      yv[5] = _mm512_add_pd
                (
                  _mm512_loadu_pd( x0 + 5*num_elem_per_reg ),
                  yv[5]
                );
      yv[6] = _mm512_add_pd
                (
                  _mm512_loadu_pd( x0 + 6*num_elem_per_reg ),
                  yv[6]
                );
      yv[7] = _mm512_add_pd
                (
                  _mm512_loadu_pd( x0 + 7*num_elem_per_reg ),
                  yv[7]
                );

      _mm512_storeu_pd( y0, yv[0] );
      _mm512_storeu_pd( ( y0 + 1*num_elem_per_reg ), yv[1] );
      _mm512_storeu_pd( ( y0 + 2*num_elem_per_reg ), yv[2] );
      _mm512_storeu_pd( ( y0 + 3*num_elem_per_reg ), yv[3] );
      _mm512_storeu_pd( ( y0 + 4*num_elem_per_reg ), yv[4] );
      _mm512_storeu_pd( ( y0 + 5*num_elem_per_reg ), yv[5] );
      _mm512_storeu_pd( ( y0 + 6*num_elem_per_reg ), yv[6] );
      _mm512_storeu_pd( ( y0 + 7*num_elem_per_reg ), yv[7] );

      x0 += 8 * num_elem_per_reg;
      y0 += 8 * num_elem_per_reg;
    }

    for ( ; i < (n & (~0x1F)); i += 32 )
    {
      // Loading input values
      yv[0] =  _mm512_loadu_pd( y0 );
      yv[1] =  _mm512_loadu_pd( y0 + 1*num_elem_per_reg );
      yv[2] =  _mm512_loadu_pd( y0 + 2*num_elem_per_reg );
      yv[3] =  _mm512_loadu_pd( y0 + 3*num_elem_per_reg );

      // y := y + x
      yv[0] = _mm512_add_pd
                (
                  _mm512_loadu_pd( x0 ),
                  yv[0]
                );
      yv[1] = _mm512_add_pd
                (
                  _mm512_loadu_pd( x0 + 1*num_elem_per_reg ),
                  yv[1]
                );
      yv[2] = _mm512_add_pd
                (
                  _mm512_loadu_pd( x0 + 2*num_elem_per_reg ),
                  yv[2]
                );
      yv[3] = _mm512_add_pd
                (
                  _mm512_loadu_pd( x0 + 3*num_elem_per_reg ),
                  yv[3]
                );

      _mm512_storeu_pd( y0, yv[0] );
      _mm512_storeu_pd( ( y0 + 1*num_elem_per_reg ), yv[1] );
      _mm512_storeu_pd( ( y0 + 2*num_elem_per_reg ), yv[2] );
      _mm512_storeu_pd( ( y0 + 3*num_elem_per_reg ), yv[3] );

      x0 += 4 * num_elem_per_reg;
      y0 += 4 * num_elem_per_reg;
    }

    for ( ; i < (n & (~0x0F)); i += 16 )
    {
      // Loading input values
      yv[0] =  _mm512_loadu_pd( y0 );
      yv[1] =  _mm512_loadu_pd( y0 + 1*num_elem_per_reg );

      // y := y + x
      yv[0] = _mm512_add_pd
                (
                  _mm512_loadu_pd( x0 ),
                  yv[0]
                );
      yv[1] = _mm512_add_pd
                (
                  _mm512_loadu_pd( x0 + 1*num_elem_per_reg ),
                  yv[1]
                );

      _mm512_storeu_pd( y0, yv[0] );
      _mm512_storeu_pd( ( y0 + 1*num_elem_per_reg ), yv[1] );

      x0 += 2 * num_elem_per_reg;
      y0 += 2 * num_elem_per_reg;
    }

    for ( ; i < (n & (~0x07)); i += 8 )
    {
      // Loading input values
      yv[0] =  _mm512_loadu_pd( y0 );

      // y := y + x
      yv[0] = _mm512_add_pd
                (
                  _mm512_loadu_pd( x0 ),
                  yv[0]
                );

      _mm512_storeu_pd( y0, yv[0] );

      x0 += 1 * num_elem_per_reg;
      y0 += 1 * num_elem_per_reg;
    }

    // Handling the frine case
    if ( i < n )
    {
      // Setting the mask for loading and storing the vectors
      __mmask8 n_mask = (1 << ( n - i )) - 1;

      // Loading input values
      yv[0] = _mm512_maskz_loadu_pd( n_mask, y0 );

      // y := y + x
      yv[0] = _mm512_add_pd
                (
                  _mm512_maskz_loadu_pd( n_mask, x0 ),
                  yv[0]
                );

      _mm512_mask_storeu_pd( y0, n_mask, yv[0] );
    }
  }

  else
  {
    // Handling fringe cases or non-unit strided vectors
    for ( ; i < n; i += 1 )
    {
        *y0 += *x0;

        x0 += incx;
        y0 += incy;
    }
  }
}
