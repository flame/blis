/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2020 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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

void bli_ssetv_zen_int
     (
       conj_t           conjalpha,
       dim_t            n,
       float*  restrict alpha,
       float*  restrict x, inc_t incx,
       cntx_t* restrict cntx
     )
{
  const dim_t num_elem_per_reg = 8;
  dim_t       i = 0;
  __m256      alphav;

  float *x0 = x;

  // If the vector dimension is zero return early.
  if ( bli_zero_dim1( n ) ) return;

  if ( incx == 1 )
  {
    alphav = _mm256_broadcast_ss( alpha );

    // For loop with n & ~0x7F => n & 0xFFFFFF80 masks the lower bits and results in multiples of 128
    // for example if n = 255
    // n & ~0x7F results in 128: copy from 0 to 128 happens in first loop
    // n & ~0x3F results in 192: copy from 128 to 192 happens in second loop
    // n & ~0x1F results in 224: copy from 128 to 192 happens in third loop and so on.
    for ( i = 0; i < (n & (~0x7F)); i += 128 )
    {
      _mm256_storeu_ps(x0 + num_elem_per_reg * 0, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 1, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 2, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 3, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 4, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 5, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 6, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 7, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 8, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 9, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 10, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 11, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 12, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 13, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 14, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 15, alphav);

      x0 += 128;
    }
    for ( ; i < (n & (~0x3F)); i += 64 )
    {
      _mm256_storeu_ps(x0 + num_elem_per_reg * 0, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 1, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 2, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 3, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 4, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 5, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 6, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 7, alphav);

      x0 += 64;
    }
    for ( ; i < (n & (~0x1F)); i += 32 )
    {
      _mm256_storeu_ps(x0 + num_elem_per_reg * 0, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 1, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 2, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 3, alphav);

      x0 += 32;
    }
    for ( ; i < (n & (~0x0F)); i += 16 )
    {
      _mm256_storeu_ps(x0 + num_elem_per_reg * 0, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 1, alphav);

      x0 += 16;
    }
    for ( ; i < (n & (~0x07)); i += 8 )
    {
      _mm256_storeu_ps(x0 + num_elem_per_reg * 0, alphav);
      x0 += 8;
    }
    for ( ; i < n; ++i )
    {
      *x0++ = *alpha;
    }
  }
  else
  {
    for ( dim_t i = 0; i < n; ++i )
    {
      *x0 = *alpha;
      x0 += incx;
    }
  }
}

void  bli_dsetv_zen_int
     (
       conj_t           conjalpha,
       dim_t            n,
       double* restrict alpha,
       double* restrict x, inc_t incx,
       cntx_t* restrict cntx
     )
{
  const dim_t num_elem_per_reg = 4;
  dim_t       i = 0;
  __m256d     alphav;

  double *x0 = x;

  // If the vector dimension is zero return early.
  if ( bli_zero_dim1( n ) ) return;

  if ( incx == 1 )
  {
    // Broadcast the alpha scalar to all elements of a vector register.
    alphav = _mm256_broadcast_sd( alpha );

    // n & (~0x3F) = n & 0xFFFFFFC0 -> this masks the numbers less than 64,
    // the copy operation will be done for the multiples of 64
    for ( i = 0; i < (n & (~0x3F)); i += 64 )
    {
      _mm256_storeu_pd(x0 + num_elem_per_reg * 0, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 1, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 2, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 3, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 4, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 5, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 6, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 7, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 8, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 9, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 10, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 11, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 12, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 13, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 14, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 15, alphav);

      x0 += num_elem_per_reg * 16;
    }
    for ( ; i < (n & (~0x1F)); i += 32 )
    {
      _mm256_storeu_pd(x0 + num_elem_per_reg * 0, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 1, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 2, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 3, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 4, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 5, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 6, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 7, alphav);

      x0 += num_elem_per_reg * 8;
    }
    for ( ; i < (n & (~0xF)); i += 16 )
    {
      _mm256_storeu_pd(x0 + num_elem_per_reg * 0, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 1, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 2, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 3, alphav);

      x0 += num_elem_per_reg * 4;
    }
    for ( ; i < (n & (~0x07)); i += 8 )
    {
      _mm256_storeu_pd(x0 + num_elem_per_reg * 0, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 1, alphav);

      x0 += num_elem_per_reg * 2;
    }
    for ( ; i < (n & (~0x03)); i += 4 )
    {
      _mm256_storeu_pd(x0 + num_elem_per_reg * 0, alphav);
      x0 += num_elem_per_reg;
    }
    for ( ; i < n; ++i )
    {
      *x0++ = *alpha;
    }
  }
  else
  {
    for ( i = 0; i < n; ++i )
    {
      *x0 = *alpha;

      x0 += incx;
    }
  }
}

void  bli_csetv_zen_int
     (
       conj_t           conjalpha,
       dim_t            n,
       scomplex* restrict alpha,
       scomplex* restrict x, inc_t incx,
       cntx_t* restrict cntx
     )
{
  // Declaring and initializing local variables and pointers
  const dim_t num_elem_per_reg = 8;
  dim_t       i = 0;
  float *x0 = (float *)x;

  // If the vector dimension is zero return early.
  if ( bli_zero_dim1( n ) ) return;
  scomplex alpha_conj =  *alpha;

  // Handle conjugation of alpha
  if( bli_is_conj( conjalpha ) ) alpha_conj.imag = -alpha_conj.imag;

  if ( incx == 1 )
  {
    __m256 alphaRv, alphaIv, alphav;

    // Broadcast the scomplex alpha value
    alphaRv = _mm256_broadcast_ss( &(alpha_conj.real) );
    alphaIv = _mm256_broadcast_ss( &(alpha_conj.imag) );
    alphav = _mm256_unpacklo_ps( alphaRv, alphaIv );

    // The condition n & ~0x3F => n & 0xFFFFFFC0
    // This sets the lower 6 bits to 0 and results in multiples of 64
    // Thus, we iterate in blocks of 64 scomplex elements
    // Fringe loops have similar conditions to set their masks(32, 16, ...)
    for ( i = 0; i < (n & (~0x3F)); i += 64 )
    {
      _mm256_storeu_ps(x0 + num_elem_per_reg * 0, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 1, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 2, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 3, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 4, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 5, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 6, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 7, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 8, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 9, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 10, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 11, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 12, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 13, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 14, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 15, alphav);

      x0 += num_elem_per_reg * 16;
    }
    for ( ; i < (n & (~0x1F)); i += 32 )
    {
      _mm256_storeu_ps(x0 + num_elem_per_reg * 0, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 1, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 2, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 3, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 4, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 5, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 6, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 7, alphav);

      x0 += num_elem_per_reg * 8;
    }
    for ( ; i < (n & (~0x0F)); i += 16 )
    {
      _mm256_storeu_ps(x0 + num_elem_per_reg * 0, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 1, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 2, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 3, alphav);

      x0 += num_elem_per_reg * 4;
    }
    for ( ; i < (n & (~0x07)); i += 8 )
    {
      _mm256_storeu_ps(x0 + num_elem_per_reg * 0, alphav);
      _mm256_storeu_ps(x0 + num_elem_per_reg * 1, alphav);

      x0 += num_elem_per_reg * 2;
    }
    for ( ; i < (n & (~0x03)); i += 4 )
    {
      _mm256_storeu_ps(x0 + num_elem_per_reg * 0, alphav);
      x0 += num_elem_per_reg;
    }
  }

  // Code-section for non-unit stride
  for( ; i < n; i += 1 )
  {
    *x0       = alpha_conj.real;
    *(x0 + 1) = alpha_conj.imag;

    x0 += 2 * incx;
  }

}

void  bli_zsetv_zen_int
     (
       conj_t           conjalpha,
       dim_t            n,
       dcomplex* restrict alpha,
       dcomplex* restrict x, inc_t incx,
       cntx_t* restrict cntx
     )
{
  // Declaring and initializing local variables and pointers
  const dim_t num_elem_per_reg = 4;
  dim_t       i = 0;
  double *x0 = (double *)x;

  // If the vector dimension is zero return early.
  if ( bli_zero_dim1( n ) ) return;

  // Handle conjugation of alpha
  if( bli_is_conj( conjalpha ) ) alpha->imag = -alpha->imag;

  if ( incx == 1 )
  {
    __m256d alphav;

    // Broadcast the dcomplex alpha value
    alphav = _mm256_broadcast_pd( (const __m128d *)alpha );

    // The condition n & ~0x1F => n & 0xFFFFFFE0
    // This sets the lower 5 bits to 0 and results in multiples of 32
    // Thus, we iterate in blocks of 32 elements
    // Fringe loops have similar conditions to set their masks(16, 8, ...)
    for ( i = 0; i < (n & (~0x1F)); i += 32 )
    {
      _mm256_storeu_pd(x0 + num_elem_per_reg * 0, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 1, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 2, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 3, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 4, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 5, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 6, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 7, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 8, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 9, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 10, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 11, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 12, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 13, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 14, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 15, alphav);

      x0 += num_elem_per_reg * 16;
    }
    for ( ; i < (n & (~0x0F)); i += 16 )
    {
      _mm256_storeu_pd(x0 + num_elem_per_reg * 0, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 1, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 2, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 3, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 4, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 5, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 6, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 7, alphav);

      x0 += num_elem_per_reg * 8;
    }
    for ( ; i < (n & (~0x07)); i += 8 )
    {
      _mm256_storeu_pd(x0 + num_elem_per_reg * 0, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 1, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 2, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 3, alphav);

      x0 += num_elem_per_reg * 4;
    }
    for ( ; i < (n & (~0x03)); i += 4 )
    {
      _mm256_storeu_pd(x0 + num_elem_per_reg * 0, alphav);
      _mm256_storeu_pd(x0 + num_elem_per_reg * 1, alphav);

      x0 += num_elem_per_reg * 2;
    }
    for ( ; i < (n & (~0x01)); i += 2 )
    {
      _mm256_storeu_pd(x0 + num_elem_per_reg * 0, alphav);
      x0 += num_elem_per_reg;
    }

    // Issue vzeroupper instruction to clear upper lanes of ymm registers.
    // This avoids a performance penalty caused by false dependencies when
    // transitioning from AVX to SSE instructions (which may occur later,
    // especially if BLIS is compiled with -mfpmath=sse).
    _mm256_zeroupper();
  }

  if ( i < n )
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

