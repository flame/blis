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

// -----------------------------------------------------------------------------

void bli_ssetv_zen_int_avx512
     (
       conj_t           conjalpha,
       dim_t            n,
       float*  restrict alpha,
       float*  restrict x, inc_t incx,
       cntx_t* restrict cntx
     )
{
  // Declaring and initializing local variables and pointers
  const dim_t num_elem_per_reg = 16;
  dim_t       i = 0;
  float     *x0 = x;

  // If the vector dimension is zero return early.
  if ( bli_zero_dim1( n ) ) return;

  // Handling unit strides
  if ( incx == 1 )
  {
    __m512 alphav;

    // Broadcast alpha to the register
    alphav = _mm512_set1_ps( *alpha );

    // The condition n & ~0x1FF => n & 0xFFFFFE00
    // This sets the lower 9 bits to 0 and results in multiples of 512
    // Thus, we iterate in blocks of 512 elements
    // Fringe loops have similar conditions to set their masks(256, 128, ...)
    for ( i = 0; i < (n & (~0x1FF)); i += 512 )
    {
      _mm512_storeu_ps(x0 + num_elem_per_reg * 0, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 1, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 2, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 3, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 4, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 5, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 6, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 7, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 8, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 9, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 10, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 11, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 12, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 13, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 14, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 15, alphav);

      _mm512_storeu_ps(x0 + num_elem_per_reg * 16, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 17, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 18, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 19, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 20, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 21, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 22, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 23, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 24, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 25, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 26, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 27, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 28, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 29, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 30, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 31, alphav);

      x0 += 512;
    }
    for ( ; i < (n & (~0xFF)); i += 256 )
    {
      _mm512_storeu_ps(x0 + num_elem_per_reg * 0, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 1, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 2, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 3, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 4, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 5, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 6, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 7, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 8, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 9, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 10, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 11, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 12, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 13, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 14, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 15, alphav);

      x0 += 256;
    }
    for ( ; i < (n & (~0x7F)); i += 128 )
    {
      _mm512_storeu_ps(x0 + num_elem_per_reg * 0, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 1, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 2, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 3, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 4, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 5, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 6, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 7, alphav);

      x0 += 128;
    }
    for ( ; i < (n & (~0x3F)); i += 64 )
    {
      _mm512_storeu_ps(x0 + num_elem_per_reg * 0, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 1, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 2, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 3, alphav);

      x0 += 64;
    }
    for ( ; i < (n & (~0x1F)); i += 32 )
    {
      _mm512_storeu_ps(x0 + num_elem_per_reg * 0, alphav);
      _mm512_storeu_ps(x0 + num_elem_per_reg * 1, alphav);

      x0 += 32;
    }
    for ( ; i < (n & (~0x0F)); i += 16 )
    {
      _mm512_storeu_ps(x0 + num_elem_per_reg * 0, alphav);
      x0 += 16;
    }
    if (i < n)
    {
      // Setting the mask register to store the remaining elements
      __mmask16 m_mask = ( 1 << (n - i)) - 1;
      _mm512_mask_storeu_ps(x0 + num_elem_per_reg * 0, m_mask, alphav);
    }
  }
  else
  {
    // Scalar loop to handle non-unit strides
    for ( dim_t i = 0; i < n; ++i )
    {
      *x0 = *alpha;
      x0 += incx;
    }
  }
}

void  bli_dsetv_zen_int_avx512
     (
       conj_t           conjalpha,
       dim_t            n,
       double* restrict alpha,
       double* restrict x, inc_t incx,
       cntx_t* restrict cntx
     )
{
  // Declaring and initializing local variables and pointers
  const dim_t num_elem_per_reg = 8;
  dim_t       i = 0;
  double *x0 = x;

  // If the vector dimension is zero return early.
  if ( bli_zero_dim1( n ) ) return;

  if ( incx == 1 )
  {
    __m512d alphav;

    // Broadcast alpha to the register
    alphav = _mm512_set1_pd( *alpha );

    // The condition n & ~0xFF => n & 0xFFFFFF00
    // This sets the lower 8 bits to 0 and results in multiples of 256
    // Thus, we iterate in blocks of 256 elements
    // Fringe loops have similar conditions to set their masks(128, 64, ...)
    for ( i = 0; i < (n & (~0xFF)); i += 256 )
    {
      _mm512_storeu_pd(x0 + num_elem_per_reg * 0, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 1, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 2, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 3, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 4, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 5, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 6, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 7, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 8, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 9, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 10, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 11, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 12, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 13, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 14, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 15, alphav);

      _mm512_storeu_pd(x0 + num_elem_per_reg * 16, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 17, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 18, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 19, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 20, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 21, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 22, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 23, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 24, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 25, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 26, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 27, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 28, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 29, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 30, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 31, alphav);

      x0 += 256;
    }
    for ( ; i < (n & (~0x7F)); i += 128 )
    {
      _mm512_storeu_pd(x0 + num_elem_per_reg * 0, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 1, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 2, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 3, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 4, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 5, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 6, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 7, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 8, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 9, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 10, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 11, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 12, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 13, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 14, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 15, alphav);

      x0 += 128;
    }
    for ( ; i < (n & (~0x3F)); i += 64 )
    {
      _mm512_storeu_pd(x0 + num_elem_per_reg * 0, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 1, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 2, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 3, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 4, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 5, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 6, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 7, alphav);

      x0 += 64;
    }
    for ( ; i < (n & (~0x1F)); i += 32 )
    {
      _mm512_storeu_pd(x0 + num_elem_per_reg * 0, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 1, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 2, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 3, alphav);

      x0 += 32;
    }
    for ( ; i < (n & (~0x0F)); i += 16 )
    {
      _mm512_storeu_pd(x0 + num_elem_per_reg * 0, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 1, alphav);

      x0 += 16;
    }
    for ( ; i < (n & (~0x07)); i += 8 )
    {
      _mm512_storeu_pd(x0 + num_elem_per_reg * 0, alphav);
      x0 += 8;
    }
    if (i < n)
    {
      __mmask8 m_mask = ( 1 << (n - i)) - 1;
      _mm512_mask_storeu_pd(x0 + num_elem_per_reg * 0, m_mask, alphav);
    }
  }
  else
  {
    // Scalar loop to handle non-unit-strides
    for ( i = 0; i < n; ++i )
    {
      *x0 = *alpha;
      x0 += incx;
    }
  }
}

void  bli_zsetv_zen_int_avx512
     (
       conj_t           conjalpha,
       dim_t            n,
       dcomplex* restrict alpha,
       dcomplex* restrict x, inc_t incx,
       cntx_t* restrict cntx
     )
{
  // Declaring and initializing local variables and pointers
  const dim_t num_elem_per_reg = 8;
  dim_t       i = 0;
  double *x0 = (double *)x;

  // If the vector dimension is zero return early.
  if ( bli_zero_dim1( n ) ) return;

  // Handle conjugation of alpha
  if ( bli_is_conj( conjalpha ) ) alpha->imag = -alpha->imag;

  if ( incx == 1 )
  {
    __m512d alphaRv, alphaIv;
    __m512d alphav;

    // Broadcast alpha(real and imag) to the separate registers
    alphaRv = _mm512_set1_pd((double)(alpha->real));
    alphaIv = _mm512_set1_pd((double)(alpha->imag));

    // Unpack and store it in interleaved format
    alphav = _mm512_unpacklo_pd(alphaRv, alphaIv);

    // The condition n & ~0x7F => n & 0xFFFFFE80
    // This sets the lower 7 bits to 0 and results in multiples of 128
    // Thus, we iterate in blocks of 128 elements
    // Fringe loops have similar conditions to set their masks(64, 32, ...)
    for ( ; i < (n & (~0x7F)); i += 128 )
    {
      _mm512_storeu_pd(x0 + num_elem_per_reg * 0, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 1, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 2, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 3, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 4, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 5, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 6, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 7, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 8, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 9, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 10, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 11, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 12, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 13, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 14, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 15, alphav);

      _mm512_storeu_pd(x0 + num_elem_per_reg * 16, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 17, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 18, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 19, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 20, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 21, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 22, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 23, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 24, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 25, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 26, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 27, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 28, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 29, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 30, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 31, alphav);

      x0 += 256;
    }
    for ( ; i < (n & (~0x3F)); i += 64 )
    {
      _mm512_storeu_pd(x0 + num_elem_per_reg * 0, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 1, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 2, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 3, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 4, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 5, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 6, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 7, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 8, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 9, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 10, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 11, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 12, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 13, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 14, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 15, alphav);

      x0 += 128;
    }
    for ( ; i < (n & (~0x1F)); i += 32 )
    {
      _mm512_storeu_pd(x0 + num_elem_per_reg * 0, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 1, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 2, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 3, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 4, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 5, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 6, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 7, alphav);

      x0 += 64;
    }
    for ( ; i < (n & (~0x0F)); i += 16 )
    {
      _mm512_storeu_pd(x0 + num_elem_per_reg * 0, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 1, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 2, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 3, alphav);

      x0 += 32;
    }
    for ( ; i < (n & (~0x07)); i += 8 )
    {
      _mm512_storeu_pd(x0 + num_elem_per_reg * 0, alphav);
      _mm512_storeu_pd(x0 + num_elem_per_reg * 1, alphav);

      x0 += 16;
    }
    for ( ; i < (n & (~0x03)); i += 4 )
    {
      _mm512_storeu_pd(x0 + num_elem_per_reg * 0, alphav);
      x0 += 8;
    }
    if (i < n)
    {
      // Set the mask to load the remaining elements
      // One double complex elements corresponds to two doubles in memory
      __mmask8 m_mask = ( 1 << 2*(n - i)) - 1;
      _mm512_mask_storeu_pd(x0 + num_elem_per_reg * 0, m_mask, alphav);
    }
  }
  else
  {
    __m128d alphav;
    alphav = _mm_loadu_pd((const double*)alpha);

    for( ; i < n; i += 1 )
    {
      _mm_storeu_pd(x0, alphav);
      x0 += 2 * incx;
    }
  }
}
