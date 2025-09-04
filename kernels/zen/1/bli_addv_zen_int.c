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

void bli_saddv_zen_int
     (
       conj_t           conjx,
       dim_t            n,
       float*  restrict x, inc_t incx,
       float*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
  const dim_t num_elem_per_reg = 8;
  dim_t       i = 0;
  __m256      yv[16];

  // If the vector dimension is zero return early.
  if ( bli_zero_dim1( n ) ) return;

  float *x0 = x;
  float *y0 = y;

  if ( incx == 1 && incy ==1  )
  {
    // For loop with n & ~0x7F => n & 0xFFFFFF80 masks the lower bits and results in multiples of 128
    // for example if n = 255
    // n & ~0x7F results in 128: copy from 0 to 128 happens in first loop
    // n & ~0x3F results in 192: copy from 128 to 192 happens in second loop
    // n & ~0x1F results in 224: copy from 128 to 192 happens in third loop and so on.
    for ( ; i < (n & (~0x7F)); i += 128 )
    {
      // Loading input values
      yv[0] =  _mm256_loadu_ps( y0 + 0*num_elem_per_reg );
      yv[1] =  _mm256_loadu_ps( y0 + 1*num_elem_per_reg );
      yv[2] =  _mm256_loadu_ps( y0 + 2*num_elem_per_reg );
      yv[3] =  _mm256_loadu_ps( y0 + 3*num_elem_per_reg );
      yv[4] =  _mm256_loadu_ps( y0 + 4*num_elem_per_reg );
      yv[5] =  _mm256_loadu_ps( y0 + 5*num_elem_per_reg );
      yv[6] =  _mm256_loadu_ps( y0 + 6*num_elem_per_reg );
      yv[7] =  _mm256_loadu_ps( y0 + 7*num_elem_per_reg );

      // y := y + x
      yv[0] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 0*num_elem_per_reg ),
                  yv[0]
                );
      yv[1] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 1*num_elem_per_reg ),
                  yv[1]
                );
      yv[2] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 2*num_elem_per_reg ),
                  yv[2]
                );
      yv[3] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 3*num_elem_per_reg ),
                  yv[3]
                );
      yv[4] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 4*num_elem_per_reg ),
                  yv[4]
                );
      yv[5] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 5*num_elem_per_reg ),
                  yv[5]
                );
      yv[6] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 6*num_elem_per_reg ),
                  yv[6]
                );
      yv[7] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 7*num_elem_per_reg ),
                  yv[7]
                );

      _mm256_storeu_ps( ( y0 + 0*num_elem_per_reg ), yv[0] );
      _mm256_storeu_ps( ( y0 + 1*num_elem_per_reg ), yv[1] );
      _mm256_storeu_ps( ( y0 + 2*num_elem_per_reg ), yv[2] );
      _mm256_storeu_ps( ( y0 + 3*num_elem_per_reg ), yv[3] );
      _mm256_storeu_ps( ( y0 + 4*num_elem_per_reg ), yv[4] );
      _mm256_storeu_ps( ( y0 + 5*num_elem_per_reg ), yv[5] );
      _mm256_storeu_ps( ( y0 + 6*num_elem_per_reg ), yv[6] );
      _mm256_storeu_ps( ( y0 + 7*num_elem_per_reg ), yv[7] );

      yv[8] =  _mm256_loadu_ps( y0 + 8*num_elem_per_reg );
      yv[9] =  _mm256_loadu_ps( y0 + 9*num_elem_per_reg );
      yv[10] =  _mm256_loadu_ps( y0 + 10*num_elem_per_reg );
      yv[11] =  _mm256_loadu_ps( y0 + 11*num_elem_per_reg );
      yv[12] =  _mm256_loadu_ps( y0 + 12*num_elem_per_reg );
      yv[13] =  _mm256_loadu_ps( y0 + 13*num_elem_per_reg );
      yv[14] =  _mm256_loadu_ps( y0 + 14*num_elem_per_reg );
      yv[15] =  _mm256_loadu_ps( y0 + 15*num_elem_per_reg );

      yv[8] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 8*num_elem_per_reg ),
                  yv[8]
                );
      yv[9] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 9*num_elem_per_reg ),
                  yv[9]
                );
      yv[10] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 10*num_elem_per_reg ),
                  yv[10]
                );
      yv[11] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 11*num_elem_per_reg ),
                  yv[11]
                );
      yv[12] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 12*num_elem_per_reg ),
                  yv[12]
                );
      yv[13] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 13*num_elem_per_reg ),
                  yv[13]
                );
      yv[14] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 14*num_elem_per_reg ),
                  yv[14]
                );
      yv[15] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 15*num_elem_per_reg ),
                  yv[15]
                );

      _mm256_storeu_ps( ( y0 + 8*num_elem_per_reg  ), yv[8] );
      _mm256_storeu_ps( ( y0 + 9*num_elem_per_reg  ), yv[9] );
      _mm256_storeu_ps( ( y0 + 10*num_elem_per_reg ), yv[10] );
      _mm256_storeu_ps( ( y0 + 11*num_elem_per_reg ), yv[11] );
      _mm256_storeu_ps( ( y0 + 12*num_elem_per_reg ), yv[12] );
      _mm256_storeu_ps( ( y0 + 13*num_elem_per_reg ), yv[13] );
      _mm256_storeu_ps( ( y0 + 14*num_elem_per_reg ), yv[14] );
      _mm256_storeu_ps( ( y0 + 15*num_elem_per_reg ), yv[15] );

      x0 += 16 * num_elem_per_reg;
      y0 += 16 * num_elem_per_reg;
    }

    for ( ; i < (n & (~0x3F)); i += 64 )
    {
      // Loading input values
      yv[0] =  _mm256_loadu_ps( y0 + 0*num_elem_per_reg );
      yv[1] =  _mm256_loadu_ps( y0 + 1*num_elem_per_reg );
      yv[2] =  _mm256_loadu_ps( y0 + 2*num_elem_per_reg );
      yv[3] =  _mm256_loadu_ps( y0 + 3*num_elem_per_reg );
      yv[4] =  _mm256_loadu_ps( y0 + 4*num_elem_per_reg );
      yv[5] =  _mm256_loadu_ps( y0 + 5*num_elem_per_reg );
      yv[6] =  _mm256_loadu_ps( y0 + 6*num_elem_per_reg );
      yv[7] =  _mm256_loadu_ps( y0 + 7*num_elem_per_reg );

      // y := y + x
      yv[0] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 0*num_elem_per_reg ),
                  yv[0]
                );
      yv[1] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 1*num_elem_per_reg ),
                  yv[1]
                );
      yv[2] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 2*num_elem_per_reg ),
                  yv[2]
                );
      yv[3] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 3*num_elem_per_reg ),
                  yv[3]
                );
      yv[4] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 4*num_elem_per_reg ),
                  yv[4]
                );
      yv[5] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 5*num_elem_per_reg ),
                  yv[5]
                );
      yv[6] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 6*num_elem_per_reg ),
                  yv[6]
                );
      yv[7] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 7*num_elem_per_reg ),
                  yv[7]
                );

      _mm256_storeu_ps( ( y0 + 0*num_elem_per_reg ), yv[0] );
      _mm256_storeu_ps( ( y0 + 1*num_elem_per_reg ), yv[1] );
      _mm256_storeu_ps( ( y0 + 2*num_elem_per_reg ), yv[2] );
      _mm256_storeu_ps( ( y0 + 3*num_elem_per_reg ), yv[3] );
      _mm256_storeu_ps( ( y0 + 4*num_elem_per_reg ), yv[4] );
      _mm256_storeu_ps( ( y0 + 5*num_elem_per_reg ), yv[5] );
      _mm256_storeu_ps( ( y0 + 6*num_elem_per_reg ), yv[6] );
      _mm256_storeu_ps( ( y0 + 7*num_elem_per_reg ), yv[7] );

      x0 += 8 * num_elem_per_reg;
      y0 += 8 * num_elem_per_reg;
    }

    for ( ; i < (n & (~0x1F)); i += 32 )
    {
      // Loading input values
      yv[0] =  _mm256_loadu_ps( y0 + 0*num_elem_per_reg );
      yv[1] =  _mm256_loadu_ps( y0 + 1*num_elem_per_reg );
      yv[2] =  _mm256_loadu_ps( y0 + 2*num_elem_per_reg );
      yv[3] =  _mm256_loadu_ps( y0 + 3*num_elem_per_reg );

      // y := y + x
      yv[0] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 0*num_elem_per_reg ),
                  yv[0]
                );
      yv[1] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 1*num_elem_per_reg ),
                  yv[1]
                );
      yv[2] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 2*num_elem_per_reg ),
                  yv[2]
                );
      yv[3] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 3*num_elem_per_reg ),
                  yv[3]
                );

      _mm256_storeu_ps( ( y0 + 0*num_elem_per_reg ), yv[0] );
      _mm256_storeu_ps( ( y0 + 1*num_elem_per_reg ), yv[1] );
      _mm256_storeu_ps( ( y0 + 2*num_elem_per_reg ), yv[2] );
      _mm256_storeu_ps( ( y0 + 3*num_elem_per_reg ), yv[3] );

      x0 += 4 * num_elem_per_reg;
      y0 += 4 * num_elem_per_reg;
    }

    for ( ; i < (n & (~0x0F)); i += 16 )
    {
      // Loading input values
      yv[0] =  _mm256_loadu_ps( y0 + 0*num_elem_per_reg );
      yv[1] =  _mm256_loadu_ps( y0 + 1*num_elem_per_reg );

      // y := y + x
      yv[0] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 0*num_elem_per_reg ),
                  yv[0]
                );
      yv[1] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 1*num_elem_per_reg ),
                  yv[1]
                );

      _mm256_storeu_ps( ( y0 + 0*num_elem_per_reg ), yv[0] );
      _mm256_storeu_ps( ( y0 + 1*num_elem_per_reg ), yv[1] );

      x0 += 2 * num_elem_per_reg;
      y0 += 2 * num_elem_per_reg;
    }

    for ( ; i < (n & (~0x07)); i += 8 )
    {
      // Loading input values
      yv[0] =  _mm256_loadu_ps( y0 + 0*num_elem_per_reg );

      // y := y + x
      yv[0] = _mm256_add_ps
                (
                  _mm256_loadu_ps( x0 + 0*num_elem_per_reg ),
                  yv[0]
                );

      _mm256_storeu_ps( ( y0 + 0*num_elem_per_reg ), yv[0] );

      x0 += num_elem_per_reg;
      y0 += num_elem_per_reg;
    }
  }

  // Handling fringe cases or non-unit strided vectors
  for ( ; i < n; i += 1 )
  {
    *y0 += *x0;

    x0 += incx;
    y0 += incy;
  }
}

void bli_daddv_zen_int
     (
       conj_t           conjx,
       dim_t            n,
       double*  restrict x, inc_t incx,
       double*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
  const dim_t num_elem_per_reg = 4;
  dim_t       i = 0;
  __m256d      yv[16];

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
      yv[0] =  _mm256_loadu_pd( y0 + 0*num_elem_per_reg );
      yv[1] =  _mm256_loadu_pd( y0 + 1*num_elem_per_reg );
      yv[2] =  _mm256_loadu_pd( y0 + 2*num_elem_per_reg );
      yv[3] =  _mm256_loadu_pd( y0 + 3*num_elem_per_reg );
      yv[4] =  _mm256_loadu_pd( y0 + 4*num_elem_per_reg );
      yv[5] =  _mm256_loadu_pd( y0 + 5*num_elem_per_reg );
      yv[6] =  _mm256_loadu_pd( y0 + 6*num_elem_per_reg );
      yv[7] =  _mm256_loadu_pd( y0 + 7*num_elem_per_reg );

      // y := y + x
      yv[0] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 0*num_elem_per_reg ),
                  yv[0]
                );
      yv[1] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 1*num_elem_per_reg ),
                  yv[1]
                );
      yv[2] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 2*num_elem_per_reg ),
                  yv[2]
                );
      yv[3] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 3*num_elem_per_reg ),
                  yv[3]
                );
      yv[4] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 4*num_elem_per_reg ),
                  yv[4]
                );
      yv[5] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 5*num_elem_per_reg ),
                  yv[5]
                );
      yv[6] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 6*num_elem_per_reg ),
                  yv[6]
                );
      yv[7] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 7*num_elem_per_reg ),
                  yv[7]
                );

      _mm256_storeu_pd( ( y0 + 0*num_elem_per_reg ), yv[0] );
      _mm256_storeu_pd( ( y0 + 1*num_elem_per_reg ), yv[1] );
      _mm256_storeu_pd( ( y0 + 2*num_elem_per_reg ), yv[2] );
      _mm256_storeu_pd( ( y0 + 3*num_elem_per_reg ), yv[3] );
      _mm256_storeu_pd( ( y0 + 4*num_elem_per_reg ), yv[4] );
      _mm256_storeu_pd( ( y0 + 5*num_elem_per_reg ), yv[5] );
      _mm256_storeu_pd( ( y0 + 6*num_elem_per_reg ), yv[6] );
      _mm256_storeu_pd( ( y0 + 7*num_elem_per_reg ), yv[7] );

      yv[8] =  _mm256_loadu_pd( y0 + 8*num_elem_per_reg );
      yv[9] =  _mm256_loadu_pd( y0 + 9*num_elem_per_reg );
      yv[10] =  _mm256_loadu_pd( y0 + 10*num_elem_per_reg );
      yv[11] =  _mm256_loadu_pd( y0 + 11*num_elem_per_reg );
      yv[12] =  _mm256_loadu_pd( y0 + 12*num_elem_per_reg );
      yv[13] =  _mm256_loadu_pd( y0 + 13*num_elem_per_reg );
      yv[14] =  _mm256_loadu_pd( y0 + 14*num_elem_per_reg );
      yv[15] =  _mm256_loadu_pd( y0 + 15*num_elem_per_reg );

      yv[8] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 8*num_elem_per_reg ),
                  yv[8]
                );
      yv[9] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 9*num_elem_per_reg ),
                  yv[9]
                );
      yv[10] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 10*num_elem_per_reg ),
                  yv[10]
                );
      yv[11] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 11*num_elem_per_reg ),
                  yv[11]
                );
      yv[12] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 12*num_elem_per_reg ),
                  yv[12]
                );
      yv[13] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 13*num_elem_per_reg ),
                  yv[13]
                );
      yv[14] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 14*num_elem_per_reg ),
                  yv[14]
                );
      yv[15] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 15*num_elem_per_reg ),
                  yv[15]
                );

      _mm256_storeu_pd( ( y0 + 8*num_elem_per_reg  ), yv[8] );
      _mm256_storeu_pd( ( y0 + 9*num_elem_per_reg  ), yv[9] );
      _mm256_storeu_pd( ( y0 + 10*num_elem_per_reg ), yv[10] );
      _mm256_storeu_pd( ( y0 + 11*num_elem_per_reg ), yv[11] );
      _mm256_storeu_pd( ( y0 + 12*num_elem_per_reg ), yv[12] );
      _mm256_storeu_pd( ( y0 + 13*num_elem_per_reg ), yv[13] );
      _mm256_storeu_pd( ( y0 + 14*num_elem_per_reg ), yv[14] );
      _mm256_storeu_pd( ( y0 + 15*num_elem_per_reg ), yv[15] );

      x0 += 16 * num_elem_per_reg;
      y0 += 16 * num_elem_per_reg;
    }

    for ( ; i < (n & (~0x1F)); i += 32 )
    {
      // Loading input values
      yv[0] =  _mm256_loadu_pd( y0 + 0*num_elem_per_reg );
      yv[1] =  _mm256_loadu_pd( y0 + 1*num_elem_per_reg );
      yv[2] =  _mm256_loadu_pd( y0 + 2*num_elem_per_reg );
      yv[3] =  _mm256_loadu_pd( y0 + 3*num_elem_per_reg );
      yv[4] =  _mm256_loadu_pd( y0 + 4*num_elem_per_reg );
      yv[5] =  _mm256_loadu_pd( y0 + 5*num_elem_per_reg );
      yv[6] =  _mm256_loadu_pd( y0 + 6*num_elem_per_reg );
      yv[7] =  _mm256_loadu_pd( y0 + 7*num_elem_per_reg );

      // y := y + x
      yv[0] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 0*num_elem_per_reg ),
                  yv[0]
                );
      yv[1] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 1*num_elem_per_reg ),
                  yv[1]
                );
      yv[2] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 2*num_elem_per_reg ),
                  yv[2]
                );
      yv[3] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 3*num_elem_per_reg ),
                  yv[3]
                );
      yv[4] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 4*num_elem_per_reg ),
                  yv[4]
                );
      yv[5] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 5*num_elem_per_reg ),
                  yv[5]
                );
      yv[6] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 6*num_elem_per_reg ),
                  yv[6]
                );
      yv[7] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 7*num_elem_per_reg ),
                  yv[7]
                );

      _mm256_storeu_pd( ( y0 + 0*num_elem_per_reg ), yv[0] );
      _mm256_storeu_pd( ( y0 + 1*num_elem_per_reg ), yv[1] );
      _mm256_storeu_pd( ( y0 + 2*num_elem_per_reg ), yv[2] );
      _mm256_storeu_pd( ( y0 + 3*num_elem_per_reg ), yv[3] );
      _mm256_storeu_pd( ( y0 + 4*num_elem_per_reg ), yv[4] );
      _mm256_storeu_pd( ( y0 + 5*num_elem_per_reg ), yv[5] );
      _mm256_storeu_pd( ( y0 + 6*num_elem_per_reg ), yv[6] );
      _mm256_storeu_pd( ( y0 + 7*num_elem_per_reg ), yv[7] );

      x0 += 8 * num_elem_per_reg;
      y0 += 8 * num_elem_per_reg;
    }

    for ( ; i < (n & (~0x0F)); i += 16 )
    {
      // Loading input values
      yv[0] =  _mm256_loadu_pd( y0 + 0*num_elem_per_reg );
      yv[1] =  _mm256_loadu_pd( y0 + 1*num_elem_per_reg );
      yv[2] =  _mm256_loadu_pd( y0 + 2*num_elem_per_reg );
      yv[3] =  _mm256_loadu_pd( y0 + 3*num_elem_per_reg );

      // y := y + x
      yv[0] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 0*num_elem_per_reg ),
                  yv[0]
                );
      yv[1] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 1*num_elem_per_reg ),
                  yv[1]
                );
      yv[2] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 2*num_elem_per_reg ),
                  yv[2]
                );
      yv[3] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 3*num_elem_per_reg ),
                  yv[3]
                );

      _mm256_storeu_pd( ( y0 + 0*num_elem_per_reg ), yv[0] );
      _mm256_storeu_pd( ( y0 + 1*num_elem_per_reg ), yv[1] );
      _mm256_storeu_pd( ( y0 + 2*num_elem_per_reg ), yv[2] );
      _mm256_storeu_pd( ( y0 + 3*num_elem_per_reg ), yv[3] );

      x0 += 4 * num_elem_per_reg;
      y0 += 4 * num_elem_per_reg;
    }

    for ( ; i < (n & (~0x07)); i += 8 )
    {
      // Loading input values
      yv[0] =  _mm256_loadu_pd( y0 + 0*num_elem_per_reg );
      yv[1] =  _mm256_loadu_pd( y0 + 1*num_elem_per_reg );

      // y := y + x
      yv[0] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 0*num_elem_per_reg ),
                  yv[0]
                );
      yv[1] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 1*num_elem_per_reg ),
                  yv[1]
                );

      _mm256_storeu_pd( ( y0 + 0*num_elem_per_reg ), yv[0] );
      _mm256_storeu_pd( ( y0 + 1*num_elem_per_reg ), yv[1] );

      x0 += 2 * num_elem_per_reg;
      y0 += 2 * num_elem_per_reg;
    }

    for ( ; i < (n & (~0x03)); i += 4 )
    {
      // Loading input values
      yv[0] =  _mm256_loadu_pd( y0 + 0*num_elem_per_reg );

      // y := y + x
      yv[0] = _mm256_add_pd
                (
                  _mm256_loadu_pd( x0 + 0*num_elem_per_reg ),
                  yv[0]
                );

      _mm256_storeu_pd( ( y0 + 0*num_elem_per_reg ), yv[0] );

      x0 += num_elem_per_reg;
      y0 += num_elem_per_reg;
    }
  }

  // Handling fringe cases or non-unit strided vectors
  for ( ; i < n; i += 1 )
  {
    *y0 += *x0;

    x0 += incx;
    y0 += incy;
  }
}

void bli_caddv_zen_int
     (
       conj_t           conjx,
       dim_t            n,
       scomplex*  restrict x, inc_t incx,
       scomplex*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
  const dim_t num_elem_per_reg = 8;
  dim_t       i = 0;
  __m256      yv[12];

  // If the vector dimension is zero return early.
  if ( bli_zero_dim1( n ) ) return;

  float *x0 = (float *)x;
  float *y0 = (float *)y;

  if( bli_is_conj( conjx ) )
  {
    __m256 conjv = _mm256_set1_ps(1.0f);
    if ( incx == 1 && incy ==1  )
    {
      for ( ; (i + 47) < n; i += 48 )
      {
        // Loading input values
        yv[0] =  _mm256_loadu_ps( y0 + 0*num_elem_per_reg );
        yv[1] =  _mm256_loadu_ps( y0 + 1*num_elem_per_reg );
        yv[2] =  _mm256_loadu_ps( y0 + 2*num_elem_per_reg );
        yv[3] =  _mm256_loadu_ps( y0 + 3*num_elem_per_reg );
        yv[4] =  _mm256_loadu_ps( y0 + 4*num_elem_per_reg );
        yv[5] =  _mm256_loadu_ps( y0 + 5*num_elem_per_reg );
        yv[6] =  _mm256_loadu_ps( y0 + 6*num_elem_per_reg );
        yv[7] =  _mm256_loadu_ps( y0 + 7*num_elem_per_reg );

        // y := y + x
        yv[0] = _mm256_fmsubadd_ps
                  (
                    conjv,
                    yv[0],
                    _mm256_loadu_ps( x0 + 0*num_elem_per_reg )
                  );
        yv[1] = _mm256_fmsubadd_ps
                  (
                    conjv,
                    yv[1],
                    _mm256_loadu_ps( x0 + 1*num_elem_per_reg )
                  );
        yv[2] = _mm256_fmsubadd_ps
                  (
                    conjv,
                    yv[2],
                    _mm256_loadu_ps( x0 + 2*num_elem_per_reg )
                  );
        yv[3] = _mm256_fmsubadd_ps
                  (
                    conjv,
                    yv[3],
                    _mm256_loadu_ps( x0 + 3*num_elem_per_reg )
                  );
        yv[4] = _mm256_fmsubadd_ps
                  (
                    conjv,
                    yv[4],
                    _mm256_loadu_ps( x0 + 4*num_elem_per_reg )
                  );
        yv[5] = _mm256_fmsubadd_ps
                  (
                    conjv,
                    yv[5],
                    _mm256_loadu_ps( x0 + 5*num_elem_per_reg )
                  );
        yv[6] = _mm256_fmsubadd_ps
                  (
                    conjv,
                    yv[6],
                    _mm256_loadu_ps( x0 + 6*num_elem_per_reg )
                  );
        yv[7] = _mm256_fmsubadd_ps
                  (
                    conjv,
                    yv[7],
                    _mm256_loadu_ps( x0 + 7*num_elem_per_reg )
                  );

        _mm256_storeu_ps( y0 + 0*num_elem_per_reg, yv[0] );
        _mm256_storeu_ps( y0 + 1*num_elem_per_reg, yv[1] );
        _mm256_storeu_ps( y0 + 2*num_elem_per_reg, yv[2] );
        _mm256_storeu_ps( y0 + 3*num_elem_per_reg, yv[3] );
        _mm256_storeu_ps( y0 + 4*num_elem_per_reg, yv[4] );
        _mm256_storeu_ps( y0 + 5*num_elem_per_reg, yv[5] );
        _mm256_storeu_ps( y0 + 6*num_elem_per_reg, yv[6] );
        _mm256_storeu_ps( y0 + 7*num_elem_per_reg, yv[7] );

        yv[8] =  _mm256_loadu_ps( y0 + 8*num_elem_per_reg );
        yv[9] =  _mm256_loadu_ps( y0 + 9*num_elem_per_reg );
        yv[10] =  _mm256_loadu_ps( y0 + 10*num_elem_per_reg );
        yv[11] =  _mm256_loadu_ps( y0 + 11*num_elem_per_reg );

        yv[8] = _mm256_fmsubadd_ps
                  (
                    conjv,
                    yv[8],
                    _mm256_loadu_ps( x0 + 8*num_elem_per_reg )
                  );
        yv[9] = _mm256_fmsubadd_ps
                  (
                    conjv,
                    yv[9],
                    _mm256_loadu_ps( x0 + 9*num_elem_per_reg )
                  );
        yv[10] = _mm256_fmsubadd_ps
                  (
                    conjv,
                    yv[10],
                    _mm256_loadu_ps( x0 + 10*num_elem_per_reg )
                  );
        yv[11] = _mm256_fmsubadd_ps
                  (
                    conjv,
                    yv[11],
                    _mm256_loadu_ps( x0 + 11*num_elem_per_reg )
                  );

        _mm256_storeu_ps( y0 + 8*num_elem_per_reg, yv[8] );
        _mm256_storeu_ps( y0 + 9*num_elem_per_reg, yv[9] );
        _mm256_storeu_ps( y0 + 10*num_elem_per_reg, yv[10] );
        _mm256_storeu_ps( y0 + 11*num_elem_per_reg, yv[11] );

        x0 += 12 * num_elem_per_reg;
        y0 += 12 * num_elem_per_reg;
      }

      for ( ; (i + 31) < n; i += 32 )
      {
        // Loading input values
        yv[0] =  _mm256_loadu_ps( y0 + 0*num_elem_per_reg );
        yv[1] =  _mm256_loadu_ps( y0 + 1*num_elem_per_reg );
        yv[2] =  _mm256_loadu_ps( y0 + 2*num_elem_per_reg );
        yv[3] =  _mm256_loadu_ps( y0 + 3*num_elem_per_reg );
        yv[4] =  _mm256_loadu_ps( y0 + 4*num_elem_per_reg );
        yv[5] =  _mm256_loadu_ps( y0 + 5*num_elem_per_reg );
        yv[6] =  _mm256_loadu_ps( y0 + 6*num_elem_per_reg );
        yv[7] =  _mm256_loadu_ps( y0 + 7*num_elem_per_reg );

        // y := y + x
        yv[0] = _mm256_fmsubadd_ps
                  (
                    conjv,
                    yv[0],
                    _mm256_loadu_ps( x0 + 0*num_elem_per_reg )
                  );
        yv[1] = _mm256_fmsubadd_ps
                  (
                    conjv,
                    yv[1],
                    _mm256_loadu_ps( x0 + 1*num_elem_per_reg )
                  );
        yv[2] = _mm256_fmsubadd_ps
                  (
                    conjv,
                    yv[2],
                    _mm256_loadu_ps( x0 + 2*num_elem_per_reg )
                  );
        yv[3] = _mm256_fmsubadd_ps
                  (
                    conjv,
                    yv[3],
                    _mm256_loadu_ps( x0 + 3*num_elem_per_reg )
                  );
        yv[4] = _mm256_fmsubadd_ps
                  (
                    conjv,
                    yv[4],
                    _mm256_loadu_ps( x0 + 4*num_elem_per_reg )
                  );
        yv[5] = _mm256_fmsubadd_ps
                  (
                    conjv,
                    yv[5],
                    _mm256_loadu_ps( x0 + 5*num_elem_per_reg )
                  );
        yv[6] = _mm256_fmsubadd_ps
                  (
                    conjv,
                    yv[6],
                    _mm256_loadu_ps( x0 + 6*num_elem_per_reg )
                  );
        yv[7] = _mm256_fmsubadd_ps
                  (
                    conjv,
                    yv[7],
                    _mm256_loadu_ps( x0 + 7*num_elem_per_reg )
                  );

        _mm256_storeu_ps( y0 + 0*num_elem_per_reg, yv[0] );
        _mm256_storeu_ps( y0 + 1*num_elem_per_reg, yv[1] );
        _mm256_storeu_ps( y0 + 2*num_elem_per_reg, yv[2] );
        _mm256_storeu_ps( y0 + 3*num_elem_per_reg, yv[3] );
        _mm256_storeu_ps( y0 + 4*num_elem_per_reg, yv[4] );
        _mm256_storeu_ps( y0 + 5*num_elem_per_reg, yv[5] );
        _mm256_storeu_ps( y0 + 6*num_elem_per_reg, yv[6] );
        _mm256_storeu_ps( y0 + 7*num_elem_per_reg, yv[7] );

        x0 += 8 * num_elem_per_reg;
        y0 += 8 * num_elem_per_reg;
      }

      for ( ; (i + 15) < n; i += 16 )
      {
        // Loading input values
        yv[0] =  _mm256_loadu_ps( y0 + 0*num_elem_per_reg );
        yv[1] =  _mm256_loadu_ps( y0 + 1*num_elem_per_reg );
        yv[2] =  _mm256_loadu_ps( y0 + 2*num_elem_per_reg );
        yv[3] =  _mm256_loadu_ps( y0 + 3*num_elem_per_reg );

        // y := y + x
        yv[0] = _mm256_fmsubadd_ps
                  (
                    conjv,
                    yv[0],
                    _mm256_loadu_ps( x0 + 0*num_elem_per_reg )
                  );
        yv[1] = _mm256_fmsubadd_ps
                  (
                    conjv,
                    yv[1],
                    _mm256_loadu_ps( x0 + 1*num_elem_per_reg )
                  );
        yv[2] = _mm256_fmsubadd_ps
                  (
                    conjv,
                    yv[2],
                    _mm256_loadu_ps( x0 + 2*num_elem_per_reg )
                  );
        yv[3] = _mm256_fmsubadd_ps
                  (
                    conjv,
                    yv[3],
                    _mm256_loadu_ps( x0 + 3*num_elem_per_reg )
                  );

        _mm256_storeu_ps( y0 + 0*num_elem_per_reg, yv[0] );
        _mm256_storeu_ps( y0 + 1*num_elem_per_reg, yv[1] );
        _mm256_storeu_ps( y0 + 2*num_elem_per_reg, yv[2] );
        _mm256_storeu_ps( y0 + 3*num_elem_per_reg, yv[3] );

        x0 += 4 * num_elem_per_reg;
        y0 += 4 * num_elem_per_reg;
      }

      for ( ; (i + 7) < n; i += 8 )
      {
        // Loading input values
        yv[0] =  _mm256_loadu_ps( y0 + 0*num_elem_per_reg );
        yv[1] =  _mm256_loadu_ps( y0 + 1*num_elem_per_reg );

        // y := y + x
        yv[0] = _mm256_fmsubadd_ps
                  (
                    conjv,
                    yv[0],
                    _mm256_loadu_ps( x0 + 0*num_elem_per_reg )
                  );
        yv[1] = _mm256_fmsubadd_ps
                  (
                    conjv,
                    yv[1],
                    _mm256_loadu_ps( x0 + 1*num_elem_per_reg )
                  );

        _mm256_storeu_ps( y0 + 0*num_elem_per_reg, yv[0] );
        _mm256_storeu_ps( y0 + 1*num_elem_per_reg, yv[1] );

        x0 += 2 * num_elem_per_reg;
        y0 += 2 * num_elem_per_reg;
      }

      for ( ; (i + 3) < n; i += 4 )
      {
        // Loading input values
        yv[0] =  _mm256_loadu_ps( y0 + 0*num_elem_per_reg );

        // y := y + x
        yv[0] = _mm256_fmsubadd_ps
                  (
                    conjv,
                    yv[0],
                    _mm256_loadu_ps( x0 + 0*num_elem_per_reg )
                  );

        _mm256_storeu_ps( y0 + 0*num_elem_per_reg, yv[0] );

        x0 += num_elem_per_reg;
        y0 += num_elem_per_reg;
      }
    }

    // Handling fringe cases or non-unit strided vectors
    for ( ; i < n; i += 1 )
    {
      *y0 += *x0;
      *(y0 + 1) -= *(x0 + 1);

      x0 += 2 * incx;
      y0 += 2 * incy;
    }
  }
  else
  {
    if ( incx == 1 && incy ==1  )
    {
      for ( ; (i + 47) < n; i += 48 )
      {
        // Loading input values
        yv[0] =  _mm256_loadu_ps( y0 + 0*num_elem_per_reg );
        yv[1] =  _mm256_loadu_ps( y0 + 1*num_elem_per_reg );
        yv[2] =  _mm256_loadu_ps( y0 + 2*num_elem_per_reg );
        yv[3] =  _mm256_loadu_ps( y0 + 3*num_elem_per_reg );
        yv[4] =  _mm256_loadu_ps( y0 + 4*num_elem_per_reg );
        yv[5] =  _mm256_loadu_ps( y0 + 5*num_elem_per_reg );
        yv[6] =  _mm256_loadu_ps( y0 + 6*num_elem_per_reg );
        yv[7] =  _mm256_loadu_ps( y0 + 7*num_elem_per_reg );

        // y := y + x
        yv[0] = _mm256_add_ps
                  (
                    _mm256_loadu_ps( x0 + 0*num_elem_per_reg ),
                    yv[0]
                  );
        yv[1] = _mm256_add_ps
                  (
                    _mm256_loadu_ps( x0 + 1*num_elem_per_reg ),
                    yv[1]
                  );
        yv[2] = _mm256_add_ps
                  (
                    _mm256_loadu_ps( x0 + 2*num_elem_per_reg ),
                    yv[2]
                  );
        yv[3] = _mm256_add_ps
                  (
                    _mm256_loadu_ps( x0 + 3*num_elem_per_reg ),
                    yv[3]
                  );
        yv[4] = _mm256_add_ps
                  (
                    _mm256_loadu_ps( x0 + 4*num_elem_per_reg ),
                    yv[4]
                  );
        yv[5] = _mm256_add_ps
                  (
                    _mm256_loadu_ps( x0 + 5*num_elem_per_reg ),
                    yv[5]
                  );
        yv[6] = _mm256_add_ps
                  (
                    _mm256_loadu_ps( x0 + 6*num_elem_per_reg ),
                    yv[6]
                  );
        yv[7] = _mm256_add_ps
                  (
                    _mm256_loadu_ps( x0 + 7*num_elem_per_reg ),
                    yv[7]
                  );

        _mm256_storeu_ps( y0 + 0*num_elem_per_reg, yv[0] );
        _mm256_storeu_ps( y0 + 1*num_elem_per_reg, yv[1] );
        _mm256_storeu_ps( y0 + 2*num_elem_per_reg, yv[2] );
        _mm256_storeu_ps( y0 + 3*num_elem_per_reg, yv[3] );
        _mm256_storeu_ps( y0 + 4*num_elem_per_reg, yv[4] );
        _mm256_storeu_ps( y0 + 5*num_elem_per_reg, yv[5] );
        _mm256_storeu_ps( y0 + 6*num_elem_per_reg, yv[6] );
        _mm256_storeu_ps( y0 + 7*num_elem_per_reg, yv[7] );

        yv[8] =  _mm256_loadu_ps( y0 + 8*num_elem_per_reg );
        yv[9] =  _mm256_loadu_ps( y0 + 9*num_elem_per_reg );
        yv[10] =  _mm256_loadu_ps( y0 + 10*num_elem_per_reg );
        yv[11] =  _mm256_loadu_ps( y0 + 11*num_elem_per_reg );

        yv[8] = _mm256_add_ps
                  (
                    _mm256_loadu_ps( x0 + 8*num_elem_per_reg ),
                    yv[8]
                  );
        yv[9] = _mm256_add_ps
                  (
                    _mm256_loadu_ps( x0 + 9*num_elem_per_reg ),
                    yv[9]
                  );
        yv[10] = _mm256_add_ps
                  (
                    _mm256_loadu_ps( x0 + 10*num_elem_per_reg ),
                    yv[10]
                  );
        yv[11] = _mm256_add_ps
                  (
                    _mm256_loadu_ps( x0 + 11*num_elem_per_reg ),
                    yv[11]
                  );

        _mm256_storeu_ps( y0 + 8*num_elem_per_reg, yv[8] );
        _mm256_storeu_ps( y0 + 9*num_elem_per_reg, yv[9] );
        _mm256_storeu_ps( y0 + 10*num_elem_per_reg, yv[10] );
        _mm256_storeu_ps( y0 + 11*num_elem_per_reg, yv[11] );

        x0 += 12 * num_elem_per_reg;
        y0 += 12 * num_elem_per_reg;
      }

      for ( ; (i + 31) < n; i += 32 )
      {
        // Loading input values
        yv[0] =  _mm256_loadu_ps( y0 + 0*num_elem_per_reg );
        yv[1] =  _mm256_loadu_ps( y0 + 1*num_elem_per_reg );
        yv[2] =  _mm256_loadu_ps( y0 + 2*num_elem_per_reg );
        yv[3] =  _mm256_loadu_ps( y0 + 3*num_elem_per_reg );
        yv[4] =  _mm256_loadu_ps( y0 + 4*num_elem_per_reg );
        yv[5] =  _mm256_loadu_ps( y0 + 5*num_elem_per_reg );
        yv[6] =  _mm256_loadu_ps( y0 + 6*num_elem_per_reg );
        yv[7] =  _mm256_loadu_ps( y0 + 7*num_elem_per_reg );

        // y := y + x
        yv[0] = _mm256_add_ps
                  (
                    _mm256_loadu_ps( x0 + 0*num_elem_per_reg ),
                    yv[0]
                  );
        yv[1] = _mm256_add_ps
                  (
                    _mm256_loadu_ps( x0 + 1*num_elem_per_reg ),
                    yv[1]
                  );
        yv[2] = _mm256_add_ps
                  (
                    _mm256_loadu_ps( x0 + 2*num_elem_per_reg ),
                    yv[2]
                  );
        yv[3] = _mm256_add_ps
                  (
                    _mm256_loadu_ps( x0 + 3*num_elem_per_reg ),
                    yv[3]
                  );
        yv[4] = _mm256_add_ps
                  (
                    _mm256_loadu_ps( x0 + 4*num_elem_per_reg ),
                    yv[4]
                  );
        yv[5] = _mm256_add_ps
                  (
                    _mm256_loadu_ps( x0 + 5*num_elem_per_reg ),
                    yv[5]
                  );
        yv[6] = _mm256_add_ps
                  (
                    _mm256_loadu_ps( x0 + 6*num_elem_per_reg ),
                    yv[6]
                  );
        yv[7] = _mm256_add_ps
                  (
                    _mm256_loadu_ps( x0 + 7*num_elem_per_reg ),
                    yv[7]
                  );

        _mm256_storeu_ps( y0 + 0*num_elem_per_reg, yv[0] );
        _mm256_storeu_ps( y0 + 1*num_elem_per_reg, yv[1] );
        _mm256_storeu_ps( y0 + 2*num_elem_per_reg, yv[2] );
        _mm256_storeu_ps( y0 + 3*num_elem_per_reg, yv[3] );
        _mm256_storeu_ps( y0 + 4*num_elem_per_reg, yv[4] );
        _mm256_storeu_ps( y0 + 5*num_elem_per_reg, yv[5] );
        _mm256_storeu_ps( y0 + 6*num_elem_per_reg, yv[6] );
        _mm256_storeu_ps( y0 + 7*num_elem_per_reg, yv[7] );

        x0 += 8 * num_elem_per_reg;
        y0 += 8 * num_elem_per_reg;
      }

      for ( ; (i + 15) < n; i += 16 )
      {
        // Loading input values
        yv[0] =  _mm256_loadu_ps( y0 + 0*num_elem_per_reg );
        yv[1] =  _mm256_loadu_ps( y0 + 1*num_elem_per_reg );
        yv[2] =  _mm256_loadu_ps( y0 + 2*num_elem_per_reg );
        yv[3] =  _mm256_loadu_ps( y0 + 3*num_elem_per_reg );

        // y := y + x
        yv[0] = _mm256_add_ps
                  (
                    _mm256_loadu_ps( x0 + 0*num_elem_per_reg ),
                    yv[0]
                  );
        yv[1] = _mm256_add_ps
                  (
                    _mm256_loadu_ps( x0 + 1*num_elem_per_reg ),
                    yv[1]
                  );
        yv[2] = _mm256_add_ps
                  (
                    _mm256_loadu_ps( x0 + 2*num_elem_per_reg ),
                    yv[2]
                  );
        yv[3] = _mm256_add_ps
                  (
                    _mm256_loadu_ps( x0 + 3*num_elem_per_reg ),
                    yv[3]
                  );

        _mm256_storeu_ps( y0 + 0*num_elem_per_reg, yv[0] );
        _mm256_storeu_ps( y0 + 1*num_elem_per_reg, yv[1] );
        _mm256_storeu_ps( y0 + 2*num_elem_per_reg, yv[2] );
        _mm256_storeu_ps( y0 + 3*num_elem_per_reg, yv[3] );

        x0 += 4 * num_elem_per_reg;
        y0 += 4 * num_elem_per_reg;
      }

      for ( ; (i + 7) < n; i += 8 )
      {
        // Loading input values
        yv[0] =  _mm256_loadu_ps( y0 + 0*num_elem_per_reg );
        yv[1] =  _mm256_loadu_ps( y0 + 1*num_elem_per_reg );

        // y := y + x
        yv[0] = _mm256_add_ps
                  (
                    _mm256_loadu_ps( x0 + 0*num_elem_per_reg ),
                    yv[0]
                  );
        yv[1] = _mm256_add_ps
                  (
                    _mm256_loadu_ps( x0 + 1*num_elem_per_reg ),
                    yv[1]
                  );

        _mm256_storeu_ps( y0 + 0*num_elem_per_reg, yv[0] );
        _mm256_storeu_ps( y0 + 1*num_elem_per_reg, yv[1] );

        x0 += 2 * num_elem_per_reg;
        y0 += 2 * num_elem_per_reg;
      }

      for ( ; (i + 3) < n; i += 4 )
      {
        // Loading input values
        yv[0] =  _mm256_loadu_ps( y0 + 0*num_elem_per_reg );

        // y := y + x
        yv[0] = _mm256_add_ps
                  (
                    _mm256_loadu_ps( x0 + 0*num_elem_per_reg ),
                    yv[0]
                  );

        _mm256_storeu_ps( y0 + 0*num_elem_per_reg, yv[0] );

        x0 += num_elem_per_reg;
        y0 += num_elem_per_reg;
      }
    }

    // Handling fringe cases or non-unit strided vectors
    for ( ; i < n; i += 1 )
    {
      *y0 += *x0;
      *(y0 + 1) += *(x0 + 1);

      x0 += 2 * incx;
      y0 += 2 * incy;
    }
  }
}

void bli_zaddv_zen_int
     (
       conj_t           conjx,
       dim_t            n,
       dcomplex*  restrict x, inc_t incx,
       dcomplex*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
  const dim_t num_elem_per_reg = 4;
  dim_t       i = 0;

  // If the vector dimension is zero return early.
  if ( bli_zero_dim1( n ) ) return;

  double *x0 = (double *)x;
  double *y0 = (double *)y;

  if( bli_is_conj( conjx ) )
  {
    __m256d      yv[12];
    __m256d conjv = _mm256_set1_pd(1.0);
    if ( incx == 1 && incy ==1  )
    {
      for ( ; (i + 23) < n; i += 24 )
      {
        // Loading input values
        yv[0] =  _mm256_loadu_pd( y0 + 0*num_elem_per_reg );
        yv[1] =  _mm256_loadu_pd( y0 + 1*num_elem_per_reg );
        yv[2] =  _mm256_loadu_pd( y0 + 2*num_elem_per_reg );
        yv[3] =  _mm256_loadu_pd( y0 + 3*num_elem_per_reg );
        yv[4] =  _mm256_loadu_pd( y0 + 4*num_elem_per_reg );
        yv[5] =  _mm256_loadu_pd( y0 + 5*num_elem_per_reg );
        yv[6] =  _mm256_loadu_pd( y0 + 6*num_elem_per_reg );
        yv[7] =  _mm256_loadu_pd( y0 + 7*num_elem_per_reg );

        // y := y + x
        yv[0] = _mm256_fmsubadd_pd
                  (
                    conjv,
                    yv[0],
                    _mm256_loadu_pd( x0 + 0*num_elem_per_reg )
                  );
        yv[1] = _mm256_fmsubadd_pd
                  (
                    conjv,
                    yv[1],
                    _mm256_loadu_pd( x0 + 1*num_elem_per_reg )
                  );
        yv[2] = _mm256_fmsubadd_pd
                  (
                    conjv,
                    yv[2],
                    _mm256_loadu_pd( x0 + 2*num_elem_per_reg )
                  );
        yv[3] = _mm256_fmsubadd_pd
                  (
                    conjv,
                    yv[3],
                    _mm256_loadu_pd( x0 + 3*num_elem_per_reg )
                  );
        yv[4] = _mm256_fmsubadd_pd
                  (
                    conjv,
                    yv[4],
                    _mm256_loadu_pd( x0 + 4*num_elem_per_reg )
                  );
        yv[5] = _mm256_fmsubadd_pd
                  (
                    conjv,
                    yv[5],
                    _mm256_loadu_pd( x0 + 5*num_elem_per_reg )
                  );
        yv[6] = _mm256_fmsubadd_pd
                  (
                    conjv,
                    yv[6],
                    _mm256_loadu_pd( x0 + 6*num_elem_per_reg )
                  );
        yv[7] = _mm256_fmsubadd_pd
                  (
                    conjv,
                    yv[7],
                    _mm256_loadu_pd( x0 + 7*num_elem_per_reg )
                  );

        _mm256_storeu_pd( y0 + 0*num_elem_per_reg, yv[0] );
        _mm256_storeu_pd( y0 + 1*num_elem_per_reg, yv[1] );
        _mm256_storeu_pd( y0 + 2*num_elem_per_reg, yv[2] );
        _mm256_storeu_pd( y0 + 3*num_elem_per_reg, yv[3] );
        _mm256_storeu_pd( y0 + 4*num_elem_per_reg, yv[4] );
        _mm256_storeu_pd( y0 + 5*num_elem_per_reg, yv[5] );
        _mm256_storeu_pd( y0 + 6*num_elem_per_reg, yv[6] );
        _mm256_storeu_pd( y0 + 7*num_elem_per_reg, yv[7] );

        yv[8] =  _mm256_loadu_pd( y0 + 8*num_elem_per_reg );
        yv[9] =  _mm256_loadu_pd( y0 + 9*num_elem_per_reg );
        yv[10] =  _mm256_loadu_pd( y0 + 10*num_elem_per_reg );
        yv[11] =  _mm256_loadu_pd( y0 + 11*num_elem_per_reg );

        yv[8] = _mm256_fmsubadd_pd
                  (
                    conjv,
                    yv[8],
                    _mm256_loadu_pd( x0 + 8*num_elem_per_reg )
                  );
        yv[9] = _mm256_fmsubadd_pd
                  (
                    conjv,
                    yv[9],
                    _mm256_loadu_pd( x0 + 9*num_elem_per_reg )
                  );
        yv[10] = _mm256_fmsubadd_pd
                  (
                    conjv,
                    yv[10],
                    _mm256_loadu_pd( x0 + 10*num_elem_per_reg )
                  );
        yv[11] = _mm256_fmsubadd_pd
                  (
                    conjv,
                    yv[11],
                    _mm256_loadu_pd( x0 + 11*num_elem_per_reg )
                  );

        _mm256_storeu_pd( y0 + 8*num_elem_per_reg, yv[8] );
        _mm256_storeu_pd( y0 + 9*num_elem_per_reg, yv[9] );
        _mm256_storeu_pd( y0 + 10*num_elem_per_reg, yv[10] );
        _mm256_storeu_pd( y0 + 11*num_elem_per_reg, yv[11] );

        x0 += 12 * num_elem_per_reg;
        y0 += 12 * num_elem_per_reg;
      }

      for ( ; (i + 15) < n; i += 16 )
      {
        // Loading input values
        yv[0] =  _mm256_loadu_pd( y0 + 0*num_elem_per_reg );
        yv[1] =  _mm256_loadu_pd( y0 + 1*num_elem_per_reg );
        yv[2] =  _mm256_loadu_pd( y0 + 2*num_elem_per_reg );
        yv[3] =  _mm256_loadu_pd( y0 + 3*num_elem_per_reg );
        yv[4] =  _mm256_loadu_pd( y0 + 4*num_elem_per_reg );
        yv[5] =  _mm256_loadu_pd( y0 + 5*num_elem_per_reg );
        yv[6] =  _mm256_loadu_pd( y0 + 6*num_elem_per_reg );
        yv[7] =  _mm256_loadu_pd( y0 + 7*num_elem_per_reg );

        // y := y + x
        yv[0] = _mm256_fmsubadd_pd
                  (
                    conjv,
                    yv[0],
                    _mm256_loadu_pd( x0 + 0*num_elem_per_reg )
                  );
        yv[1] = _mm256_fmsubadd_pd
                  (
                    conjv,
                    yv[1],
                    _mm256_loadu_pd( x0 + 1*num_elem_per_reg )
                  );
        yv[2] = _mm256_fmsubadd_pd
                  (
                    conjv,
                    yv[2],
                    _mm256_loadu_pd( x0 + 2*num_elem_per_reg )
                  );
        yv[3] = _mm256_fmsubadd_pd
                  (
                    conjv,
                    yv[3],
                    _mm256_loadu_pd( x0 + 3*num_elem_per_reg )
                  );
        yv[4] = _mm256_fmsubadd_pd
                  (
                    conjv,
                    yv[4],
                    _mm256_loadu_pd( x0 + 4*num_elem_per_reg )
                  );
        yv[5] = _mm256_fmsubadd_pd
                  (
                    conjv,
                    yv[5],
                    _mm256_loadu_pd( x0 + 5*num_elem_per_reg )
                  );
        yv[6] = _mm256_fmsubadd_pd
                  (
                    conjv,
                    yv[6],
                    _mm256_loadu_pd( x0 + 6*num_elem_per_reg )
                  );
        yv[7] = _mm256_fmsubadd_pd
                  (
                    conjv,
                    yv[7],
                    _mm256_loadu_pd( x0 + 7*num_elem_per_reg )
                  );

        _mm256_storeu_pd( y0 + 0*num_elem_per_reg, yv[0] );
        _mm256_storeu_pd( y0 + 1*num_elem_per_reg, yv[1] );
        _mm256_storeu_pd( y0 + 2*num_elem_per_reg, yv[2] );
        _mm256_storeu_pd( y0 + 3*num_elem_per_reg, yv[3] );
        _mm256_storeu_pd( y0 + 4*num_elem_per_reg, yv[4] );
        _mm256_storeu_pd( y0 + 5*num_elem_per_reg, yv[5] );
        _mm256_storeu_pd( y0 + 6*num_elem_per_reg, yv[6] );
        _mm256_storeu_pd( y0 + 7*num_elem_per_reg, yv[7] );

        x0 += 8 * num_elem_per_reg;
        y0 += 8 * num_elem_per_reg;
      }

      for ( ; (i + 7) < n; i += 8 )
      {
        // Loading input values
        yv[0] =  _mm256_loadu_pd( y0 + 0*num_elem_per_reg );
        yv[1] =  _mm256_loadu_pd( y0 + 1*num_elem_per_reg );
        yv[2] =  _mm256_loadu_pd( y0 + 2*num_elem_per_reg );
        yv[3] =  _mm256_loadu_pd( y0 + 3*num_elem_per_reg );

        // y := y + x
        yv[0] = _mm256_fmsubadd_pd
                  (
                    conjv,
                    yv[0],
                    _mm256_loadu_pd( x0 + 0*num_elem_per_reg )
                  );
        yv[1] = _mm256_fmsubadd_pd
                  (
                    conjv,
                    yv[1],
                    _mm256_loadu_pd( x0 + 1*num_elem_per_reg )
                  );
        yv[2] = _mm256_fmsubadd_pd
                  (
                    conjv,
                    yv[2],
                    _mm256_loadu_pd( x0 + 2*num_elem_per_reg )
                  );
        yv[3] = _mm256_fmsubadd_pd
                  (
                    conjv,
                    yv[3],
                    _mm256_loadu_pd( x0 + 3*num_elem_per_reg )
                  );

        _mm256_storeu_pd( y0 + 0*num_elem_per_reg, yv[0] );
        _mm256_storeu_pd( y0 + 1*num_elem_per_reg, yv[1] );
        _mm256_storeu_pd( y0 + 2*num_elem_per_reg, yv[2] );
        _mm256_storeu_pd( y0 + 3*num_elem_per_reg, yv[3] );

        x0 += 4 * num_elem_per_reg;
        y0 += 4 * num_elem_per_reg;
      }

      for ( ; (i + 3) < n; i += 4 )
      {
        // Loading input values
        yv[0] =  _mm256_loadu_pd( y0 + 0*num_elem_per_reg );
        yv[1] =  _mm256_loadu_pd( y0 + 1*num_elem_per_reg );

        // y := y + x
        yv[0] = _mm256_fmsubadd_pd
                  (
                    conjv,
                    yv[0],
                    _mm256_loadu_pd( x0 + 0*num_elem_per_reg )
                  );
        yv[1] = _mm256_fmsubadd_pd
                  (
                    conjv,
                    yv[1],
                    _mm256_loadu_pd( x0 + 1*num_elem_per_reg )
                  );

        _mm256_storeu_pd( y0 + 0*num_elem_per_reg, yv[0] );
        _mm256_storeu_pd( y0 + 1*num_elem_per_reg, yv[1] );

        x0 += 2 * num_elem_per_reg;
        y0 += 2 * num_elem_per_reg;
      }

      for ( ; (i + 1) < n; i += 2 )
      {
        // Loading input values
        yv[0] =  _mm256_loadu_pd( y0 + 0*num_elem_per_reg );

        // y := y + x
        yv[0] = _mm256_fmsubadd_pd
                  (
                    conjv,
                    yv[0],
                    _mm256_loadu_pd( x0 + 0*num_elem_per_reg )
                  );

        _mm256_storeu_pd( y0 + 0*num_elem_per_reg, yv[0] );

        x0 += num_elem_per_reg;
        y0 += num_elem_per_reg;
      }

      _mm256_zeroupper();
    }

    __m128d x_vec, y_vec;
    x_vec = _mm_setzero_pd();
    y_vec = _mm_setzero_pd();

    for( ; i < n; i += 1 )
    {
      x_vec = _mm_loadu_pd( x0 );
      y_vec = _mm_loadu_pd( y0 );

      x_vec = _mm_shuffle_pd(x_vec, x_vec, 0x1);
      y_vec = _mm_shuffle_pd(y_vec, y_vec, 0x1);

      y_vec =_mm_addsub_pd(y_vec, x_vec);

      y_vec = _mm_shuffle_pd(y_vec, y_vec, 0x1);

      _mm_storeu_pd(y0, y_vec);

      x0 += 2 * incx;
      y0 += 2 * incy;
    }
  }
  else
  {
    __m256d      yv[12];
    if ( incx == 1 && incy ==1  )
    {
      for ( ; (i + 23) < n; i += 24 )
      {
        // Loading input values
        yv[0] =  _mm256_loadu_pd( y0 + 0*num_elem_per_reg );
        yv[1] =  _mm256_loadu_pd( y0 + 1*num_elem_per_reg );
        yv[2] =  _mm256_loadu_pd( y0 + 2*num_elem_per_reg );
        yv[3] =  _mm256_loadu_pd( y0 + 3*num_elem_per_reg );
        yv[4] =  _mm256_loadu_pd( y0 + 4*num_elem_per_reg );
        yv[5] =  _mm256_loadu_pd( y0 + 5*num_elem_per_reg );
        yv[6] =  _mm256_loadu_pd( y0 + 6*num_elem_per_reg );
        yv[7] =  _mm256_loadu_pd( y0 + 7*num_elem_per_reg );

        // y := y + x
        yv[0] = _mm256_add_pd
                  (
                    _mm256_loadu_pd( x0 + 0*num_elem_per_reg ),
                    yv[0]
                  );
        yv[1] = _mm256_add_pd
                  (
                    _mm256_loadu_pd( x0 + 1*num_elem_per_reg ),
                    yv[1]
                  );
        yv[2] = _mm256_add_pd
                  (
                    _mm256_loadu_pd( x0 + 2*num_elem_per_reg ),
                    yv[2]
                  );
        yv[3] = _mm256_add_pd
                  (
                    _mm256_loadu_pd( x0 + 3*num_elem_per_reg ),
                    yv[3]
                  );
        yv[4] = _mm256_add_pd
                  (
                    _mm256_loadu_pd( x0 + 4*num_elem_per_reg ),
                    yv[4]
                  );
        yv[5] = _mm256_add_pd
                  (
                    _mm256_loadu_pd( x0 + 5*num_elem_per_reg ),
                    yv[5]
                  );
        yv[6] = _mm256_add_pd
                  (
                    _mm256_loadu_pd( x0 + 6*num_elem_per_reg ),
                    yv[6]
                  );
        yv[7] = _mm256_add_pd
                  (
                    _mm256_loadu_pd( x0 + 7*num_elem_per_reg ),
                    yv[7]
                  );

        _mm256_storeu_pd( y0 + 0*num_elem_per_reg, yv[0] );
        _mm256_storeu_pd( y0 + 1*num_elem_per_reg, yv[1] );
        _mm256_storeu_pd( y0 + 2*num_elem_per_reg, yv[2] );
        _mm256_storeu_pd( y0 + 3*num_elem_per_reg, yv[3] );
        _mm256_storeu_pd( y0 + 4*num_elem_per_reg, yv[4] );
        _mm256_storeu_pd( y0 + 5*num_elem_per_reg, yv[5] );
        _mm256_storeu_pd( y0 + 6*num_elem_per_reg, yv[6] );
        _mm256_storeu_pd( y0 + 7*num_elem_per_reg, yv[7] );

        yv[8] =  _mm256_loadu_pd( y0 + 8*num_elem_per_reg );
        yv[9] =  _mm256_loadu_pd( y0 + 9*num_elem_per_reg );
        yv[10] =  _mm256_loadu_pd( y0 + 10*num_elem_per_reg );
        yv[11] =  _mm256_loadu_pd( y0 + 11*num_elem_per_reg );

        yv[8] = _mm256_add_pd
                  (
                    _mm256_loadu_pd( x0 + 8*num_elem_per_reg ),
                    yv[8]
                  );
        yv[9] = _mm256_add_pd
                  (
                    _mm256_loadu_pd( x0 + 9*num_elem_per_reg ),
                    yv[9]
                  );
        yv[10] = _mm256_add_pd
                  (
                    _mm256_loadu_pd( x0 + 10*num_elem_per_reg ),
                    yv[10]
                  );
        yv[11] = _mm256_add_pd
                  (
                    _mm256_loadu_pd( x0 + 11*num_elem_per_reg ),
                    yv[11]
                  );

        _mm256_storeu_pd( y0 + 8*num_elem_per_reg, yv[8] );
        _mm256_storeu_pd( y0 + 9*num_elem_per_reg, yv[9] );
        _mm256_storeu_pd( y0 + 10*num_elem_per_reg, yv[10] );
        _mm256_storeu_pd( y0 + 11*num_elem_per_reg, yv[11] );

        x0 += 12 * num_elem_per_reg;
        y0 += 12 * num_elem_per_reg;
      }

      for ( ; (i + 15) < n; i += 16 )
      {
        // Loading input values
        yv[0] =  _mm256_loadu_pd( y0 + 0*num_elem_per_reg );
        yv[1] =  _mm256_loadu_pd( y0 + 1*num_elem_per_reg );
        yv[2] =  _mm256_loadu_pd( y0 + 2*num_elem_per_reg );
        yv[3] =  _mm256_loadu_pd( y0 + 3*num_elem_per_reg );
        yv[4] =  _mm256_loadu_pd( y0 + 4*num_elem_per_reg );
        yv[5] =  _mm256_loadu_pd( y0 + 5*num_elem_per_reg );
        yv[6] =  _mm256_loadu_pd( y0 + 6*num_elem_per_reg );
        yv[7] =  _mm256_loadu_pd( y0 + 7*num_elem_per_reg );

        // y := y + x
        yv[0] = _mm256_add_pd
                  (
                    _mm256_loadu_pd( x0 + 0*num_elem_per_reg ),
                    yv[0]
                  );
        yv[1] = _mm256_add_pd
                  (
                    _mm256_loadu_pd( x0 + 1*num_elem_per_reg ),
                    yv[1]
                  );
        yv[2] = _mm256_add_pd
                  (
                    _mm256_loadu_pd( x0 + 2*num_elem_per_reg ),
                    yv[2]
                  );
        yv[3] = _mm256_add_pd
                  (
                    _mm256_loadu_pd( x0 + 3*num_elem_per_reg ),
                    yv[3]
                  );
        yv[4] = _mm256_add_pd
                  (
                    _mm256_loadu_pd( x0 + 4*num_elem_per_reg ),
                    yv[4]
                  );
        yv[5] = _mm256_add_pd
                  (
                    _mm256_loadu_pd( x0 + 5*num_elem_per_reg ),
                    yv[5]
                  );
        yv[6] = _mm256_add_pd
                  (
                    _mm256_loadu_pd( x0 + 6*num_elem_per_reg ),
                    yv[6]
                  );
        yv[7] = _mm256_add_pd
                  (
                    _mm256_loadu_pd( x0 + 7*num_elem_per_reg ),
                    yv[7]
                  );

        _mm256_storeu_pd( y0 + 0*num_elem_per_reg, yv[0] );
        _mm256_storeu_pd( y0 + 1*num_elem_per_reg, yv[1] );
        _mm256_storeu_pd( y0 + 2*num_elem_per_reg, yv[2] );
        _mm256_storeu_pd( y0 + 3*num_elem_per_reg, yv[3] );
        _mm256_storeu_pd( y0 + 4*num_elem_per_reg, yv[4] );
        _mm256_storeu_pd( y0 + 5*num_elem_per_reg, yv[5] );
        _mm256_storeu_pd( y0 + 6*num_elem_per_reg, yv[6] );
        _mm256_storeu_pd( y0 + 7*num_elem_per_reg, yv[7] );

        x0 += 8 * num_elem_per_reg;
        y0 += 8 * num_elem_per_reg;
      }

      for ( ; (i + 7) < n; i += 8 )
      {
        // Loading input values
        yv[0] =  _mm256_loadu_pd( y0 + 0*num_elem_per_reg );
        yv[1] =  _mm256_loadu_pd( y0 + 1*num_elem_per_reg );
        yv[2] =  _mm256_loadu_pd( y0 + 2*num_elem_per_reg );
        yv[3] =  _mm256_loadu_pd( y0 + 3*num_elem_per_reg );

        // y := y + x
        yv[0] = _mm256_add_pd
                  (
                    _mm256_loadu_pd( x0 + 0*num_elem_per_reg ),
                    yv[0]
                  );
        yv[1] = _mm256_add_pd
                  (
                    _mm256_loadu_pd( x0 + 1*num_elem_per_reg ),
                    yv[1]
                  );
        yv[2] = _mm256_add_pd
                  (
                    _mm256_loadu_pd( x0 + 2*num_elem_per_reg ),
                    yv[2]
                  );
        yv[3] = _mm256_add_pd
                  (
                    _mm256_loadu_pd( x0 + 3*num_elem_per_reg ),
                    yv[3]
                  );

        _mm256_storeu_pd( y0 + 0*num_elem_per_reg, yv[0] );
        _mm256_storeu_pd( y0 + 1*num_elem_per_reg, yv[1] );
        _mm256_storeu_pd( y0 + 2*num_elem_per_reg, yv[2] );
        _mm256_storeu_pd( y0 + 3*num_elem_per_reg, yv[3] );

        x0 += 4 * num_elem_per_reg;
        y0 += 4 * num_elem_per_reg;
      }

      for ( ; (i + 3) < n; i += 4 )
      {
        // Loading input values
        yv[0] =  _mm256_loadu_pd( y0 + 0*num_elem_per_reg );
        yv[1] =  _mm256_loadu_pd( y0 + 1*num_elem_per_reg );

        // y := y + x
        yv[0] = _mm256_add_pd
                  (
                    _mm256_loadu_pd( x0 + 0*num_elem_per_reg ),
                    yv[0]
                  );
        yv[1] = _mm256_add_pd
                  (
                    _mm256_loadu_pd( x0 + 1*num_elem_per_reg ),
                    yv[1]
                  );

        _mm256_storeu_pd( y0 + 0*num_elem_per_reg, yv[0] );
        _mm256_storeu_pd( y0 + 1*num_elem_per_reg, yv[1] );

        x0 += 2 * num_elem_per_reg;
        y0 += 2 * num_elem_per_reg;
      }

      for ( ; (i + 1) < n; i += 2 )
      {
        // Loading input values
        yv[0] =  _mm256_loadu_pd( y0 + 0*num_elem_per_reg );

        // y := y + x
        yv[0] = _mm256_add_pd
                  (
                    _mm256_loadu_pd( x0 + 0*num_elem_per_reg ),
                    yv[0]
                  );

        _mm256_storeu_pd( y0 + 0*num_elem_per_reg, yv[0] );

        x0 += num_elem_per_reg;
        y0 += num_elem_per_reg;
      }
    }

    __m128d x_vec, y_vec;
    x_vec = _mm_setzero_pd();
    y_vec = _mm_setzero_pd();

    for( ; i < n; i += 1 )
    {
      x_vec = _mm_loadu_pd( x0 );
      y_vec = _mm_loadu_pd( y0 );

      y_vec =_mm_add_pd(y_vec, x_vec);

      _mm_storeu_pd(y0, y_vec);

      x0 += 2 * incx;
      y0 += 2 * incy;
    }
  }
}
