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
    - Redistributions in binarsy form must reproduce the above copyright
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

#include <immintrin.h>
#include <string.h>
#include "blis.h"

#ifdef BLIS_ADDON_LPGEMM

#define UNPACKLO_PS16 \
  b_reg[0] = _mm512_unpacklo_ps(a_reg[0], a_reg[1]); \
  b_reg[1] = _mm512_unpacklo_ps(a_reg[2], a_reg[3]); \
  b_reg[2] = _mm512_unpacklo_ps(a_reg[4], a_reg[5]); \
  b_reg[3] = _mm512_unpacklo_ps(a_reg[6], a_reg[7]); \
  b_reg[4] = _mm512_unpacklo_ps(a_reg[8], a_reg[9]); \
  b_reg[5] = _mm512_unpacklo_ps(a_reg[10], a_reg[11]); \
  b_reg[6] = _mm512_unpacklo_ps(a_reg[12], a_reg[13]); \
  b_reg[7] = _mm512_unpacklo_ps(a_reg[14], a_reg[15]);

#define UNPACKHI_PS16 \
  b_reg[8] = _mm512_unpackhi_ps(a_reg[0], a_reg[1]); \
  b_reg[9] = _mm512_unpackhi_ps(a_reg[2], a_reg[3]); \
  b_reg[10] = _mm512_unpackhi_ps(a_reg[4], a_reg[5]); \
  b_reg[11] = _mm512_unpackhi_ps(a_reg[6], a_reg[7]); \
  b_reg[12] = _mm512_unpackhi_ps(a_reg[8], a_reg[9]); \
  b_reg[13] = _mm512_unpackhi_ps(a_reg[10], a_reg[11]); \
  b_reg[14] = _mm512_unpackhi_ps(a_reg[12], a_reg[13]); \
  b_reg[15] = _mm512_unpackhi_ps(a_reg[14], a_reg[15]);

#define SHUFFLE_64x2 \
  a_reg[0] = _mm512_shuffle_ps(b_reg[0], b_reg[1], 0x44); \
  a_reg[1] = _mm512_shuffle_ps(b_reg[0], b_reg[1], 0xEE); \
  a_reg[2] = _mm512_shuffle_ps(b_reg[2], b_reg[3], 0x44); \
  a_reg[3] = _mm512_shuffle_ps(b_reg[2], b_reg[3], 0xEE); \
\
  a_reg[4] = _mm512_shuffle_ps(b_reg[4], b_reg[5], 0x44); \
  a_reg[5] = _mm512_shuffle_ps(b_reg[4], b_reg[5], 0xEE); \
  a_reg[6] = _mm512_shuffle_ps(b_reg[6], b_reg[7], 0x44); \
  a_reg[7] = _mm512_shuffle_ps(b_reg[6], b_reg[7], 0xEE); \
\
  a_reg[8] = _mm512_shuffle_ps(b_reg[8], b_reg[9], 0x44); \
  a_reg[9] = _mm512_shuffle_ps(b_reg[8], b_reg[9], 0xEE); \
  a_reg[10] = _mm512_shuffle_ps(b_reg[10], b_reg[11], 0x44); \
  a_reg[11] = _mm512_shuffle_ps(b_reg[10], b_reg[11], 0xEE); \
\
  a_reg[12] = _mm512_shuffle_ps(b_reg[12], b_reg[13], 0x44); \
  a_reg[13] = _mm512_shuffle_ps(b_reg[12], b_reg[13], 0xEE); \
  a_reg[14] = _mm512_shuffle_ps(b_reg[14], b_reg[15], 0x44); \
  a_reg[15] = _mm512_shuffle_ps(b_reg[14], b_reg[15], 0xEE);

#define MASKED_STORE_PS(mask) \
  _mm512_mask_storeu_ps((pack_a_buffer + (ic+0) * KC + kr ), mask, a_reg[0]); \
  _mm512_mask_storeu_ps((pack_a_buffer + (ic+1) * KC + kr ), mask, a_reg[1]); \
  _mm512_mask_storeu_ps((pack_a_buffer + (ic+2) * KC + kr ), mask, a_reg[2]); \
  _mm512_mask_storeu_ps((pack_a_buffer + (ic+3) * KC + kr ), mask, a_reg[3]); \
  _mm512_mask_storeu_ps((pack_a_buffer + (ic+4) * KC + kr ), mask, a_reg[4]); \
  _mm512_mask_storeu_ps((pack_a_buffer + (ic+5) * KC + kr ), mask, a_reg[5]); \
  _mm512_mask_storeu_ps((pack_a_buffer + (ic+6) * KC + kr ), mask, a_reg[6]); \
  _mm512_mask_storeu_ps((pack_a_buffer + (ic+7) * KC + kr ), mask, a_reg[7]); \
  _mm512_mask_storeu_ps((pack_a_buffer + (ic+8) * KC + kr ), mask, a_reg[8]); \
  _mm512_mask_storeu_ps((pack_a_buffer + (ic+9) * KC + kr ), mask, a_reg[9]); \
  _mm512_mask_storeu_ps((pack_a_buffer + (ic+10) * KC + kr ), mask, a_reg[10]); \
  _mm512_mask_storeu_ps((pack_a_buffer + (ic+11) * KC + kr ), mask, a_reg[11]); \
  _mm512_mask_storeu_ps((pack_a_buffer + (ic+12) * KC + kr ), mask, a_reg[12]); \
  _mm512_mask_storeu_ps((pack_a_buffer + (ic+13) * KC + kr ), mask, a_reg[13]); \
  _mm512_mask_storeu_ps((pack_a_buffer + (ic+14) * KC + kr ), mask, a_reg[14]); \
  _mm512_mask_storeu_ps((pack_a_buffer + (ic+15) * KC + kr ), mask, a_reg[15]);

#define PERMUTE4x4( mask1, mask2 )   \
  b_reg[0] = _mm512_permutex2var_ps( a_reg[0], mask1, a_reg[2] );   \
  b_reg[1] = _mm512_permutex2var_ps( a_reg[1], mask1, a_reg[3] );   \
  b_reg[2] = _mm512_permutex2var_ps( a_reg[8], mask1, a_reg[10] );  \
  b_reg[3] = _mm512_permutex2var_ps( a_reg[9], mask1, a_reg[11] );  \
\
  b_reg[4] = _mm512_permutex2var_ps( a_reg[4], mask1, a_reg[6]);    \
  b_reg[5] = _mm512_permutex2var_ps( a_reg[5], mask1, a_reg[7]);    \
  b_reg[6] = _mm512_permutex2var_ps( a_reg[12], mask1, a_reg[14]);  \
  b_reg[7] = _mm512_permutex2var_ps( a_reg[13], mask1, a_reg[15]);  \
\
  b_reg[8] = _mm512_permutex2var_ps( a_reg[0], mask2, a_reg[2]);    \
  b_reg[9] = _mm512_permutex2var_ps( a_reg[1], mask2, a_reg[3]);    \
  b_reg[10] = _mm512_permutex2var_ps( a_reg[8], mask2, a_reg[10]);  \
  b_reg[11] = _mm512_permutex2var_ps( a_reg[9], mask2, a_reg[11]);  \
\
  b_reg[12] = _mm512_permutex2var_ps( a_reg[4], mask2, a_reg[6]);   \
  b_reg[13] = _mm512_permutex2var_ps( a_reg[5], mask2, a_reg[7]);   \
  b_reg[14] = _mm512_permutex2var_ps( a_reg[12], mask2, a_reg[14]); \
  b_reg[15] = _mm512_permutex2var_ps( a_reg[13], mask2, a_reg[15]);

#define PERMUTE8x8( mask3, mask4 )  \
  a_reg[0] = _mm512_permutex2var_ps( b_reg[0], mask3, b_reg[4]);   \
  a_reg[1] = _mm512_permutex2var_ps( b_reg[1], mask3, b_reg[5]);   \
  a_reg[2] = _mm512_permutex2var_ps( b_reg[2], mask3, b_reg[6]);   \
  a_reg[3] = _mm512_permutex2var_ps( b_reg[3], mask3, b_reg[7]);   \
\
  a_reg[4] = _mm512_permutex2var_ps( b_reg[0], mask4, b_reg[4]);   \
  a_reg[5] = _mm512_permutex2var_ps( b_reg[1], mask4, b_reg[5]);   \
  a_reg[6] = _mm512_permutex2var_ps( b_reg[2], mask4, b_reg[6]);   \
  a_reg[7] = _mm512_permutex2var_ps( b_reg[3], mask4, b_reg[7]);   \
\
  a_reg[8] = _mm512_permutex2var_ps( b_reg[8], mask3, b_reg[12]);   \
  a_reg[9] = _mm512_permutex2var_ps( b_reg[9], mask3, b_reg[13]);   \
  a_reg[10] = _mm512_permutex2var_ps( b_reg[10], mask3, b_reg[14]); \
  a_reg[11] = _mm512_permutex2var_ps( b_reg[11], mask3, b_reg[15]); \
\
  a_reg[12] = _mm512_permutex2var_ps( b_reg[8], mask4, b_reg[12]);  \
  a_reg[13] = _mm512_permutex2var_ps( b_reg[9], mask4, b_reg[13]);  \
  a_reg[14] = _mm512_permutex2var_ps( b_reg[10], mask4, b_reg[14]); \
  a_reg[15] = _mm512_permutex2var_ps( b_reg[11], mask4, b_reg[15]);

void packa_mr16_f32f32f32of32_col_major
(
  float*	      pack_a_buffer,
  const float* a,
  const dim_t     rs_a,
  const dim_t     cs_a,
  const dim_t     MC,
  const dim_t     KC,
  dim_t*          rs_p,
  dim_t*          cs_p
)
{
  dim_t MR = 16;
  dim_t ic, kr;
  dim_t m_left = MC % 4;

  __m512 a_reg[16], b_reg[16];

  __m512i mask1 = _mm512_set_epi32( 0x17, 0x16, 0x15, 0x14,
                                    0x07, 0x06, 0x05, 0x04,
                                    0x13, 0x12, 0x11, 0x10,
                                    0x03, 0x02, 0x01, 0x00 );

  __m512i mask2 = _mm512_set_epi32( 0x1F, 0x1E, 0x1D, 0x1C,
                                    0x0F, 0x0E, 0x0D, 0x0C,
                                    0x1B, 0x1A, 0x19, 0x18,
                                    0x0B, 0x0A, 0x9, 0x08 );

  __m512i mask3 = _mm512_set_epi32( 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x10,
                                      0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00 );
  __m512i mask4 = _mm512_set_epi32( 0x1F, 0x1E, 0x1D, 0x1C, 0x1B, 0x1A, 0x19, 0x18,
                                      0x0F, 0x0E, 0x0D, 0x0C, 0x0B, 0x0A, 0x09, 0x08 );

  // These registers are set with zeroes to avoid compiler warnings
  // To-DO: TO be removed when pack code is optimized for fringe cases.
  a_reg[0] = _mm512_setzero_ps();
  a_reg[1] = _mm512_setzero_ps();
  a_reg[2] = _mm512_setzero_ps();
  a_reg[3] = _mm512_setzero_ps();
  a_reg[4] = _mm512_setzero_ps();
  a_reg[5] = _mm512_setzero_ps();
  a_reg[6] = _mm512_setzero_ps();
  a_reg[7] = _mm512_setzero_ps();
  a_reg[8] = _mm512_setzero_ps();
  a_reg[9] = _mm512_setzero_ps();
  a_reg[10] = _mm512_setzero_ps();
  a_reg[11] = _mm512_setzero_ps();
  a_reg[12] = _mm512_setzero_ps();
  a_reg[13] = _mm512_setzero_ps();
  a_reg[14] = _mm512_setzero_ps();
  a_reg[15] = _mm512_setzero_ps();

  for( ic = 0; ( ic + MR - 1 ) < MC; ic += MR)
  {
    for( kr = 0; ( kr + 15 ) < KC; kr += 16)
    {
      a_reg[0] = _mm512_loadu_ps( (__m512 const *) ( a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) ) );
      a_reg[1] = _mm512_loadu_ps( (__m512 const *) ( a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) ) );
      a_reg[2] = _mm512_loadu_ps( (__m512 const *) ( a + ( ic * rs_a ) + ( ( kr + 2 ) * cs_a ) ) );
      a_reg[3] = _mm512_loadu_ps( (__m512 const *) ( a + ( ic * rs_a ) + ( ( kr + 3 ) * cs_a ) ) );
      a_reg[4] = _mm512_loadu_ps( (__m512 const *) ( a + ( ic * rs_a ) + ( ( kr + 4 ) * cs_a ) ) );
      a_reg[5] = _mm512_loadu_ps( (__m512 const *) ( a + ( ic * rs_a ) + ( ( kr + 5 ) * cs_a ) ) );
      a_reg[6] = _mm512_loadu_ps( (__m512 const *) ( a + ( ic * rs_a ) + ( ( kr + 6 ) * cs_a ) ) );
      a_reg[7] = _mm512_loadu_ps( (__m512 const *) ( a + ( ic * rs_a ) + ( ( kr + 7 ) * cs_a ) ) );
      a_reg[8] = _mm512_loadu_ps( (__m512 const *) ( a + ( ic * rs_a ) + ( ( kr + 8 ) * cs_a ) ) );
      a_reg[9] = _mm512_loadu_ps( (__m512 const *) ( a + ( ic * rs_a ) + ( ( kr + 9 ) * cs_a ) ) );
      a_reg[10] = _mm512_loadu_ps( (__m512 const *) ( a + ( ic * rs_a ) +
                                  ( ( kr + 10 ) * cs_a ) ) );
      a_reg[11] = _mm512_loadu_ps( (__m512 const *) ( a + ( ic * rs_a ) +
                                  ( ( kr + 11 ) * cs_a ) ) );
      a_reg[12] = _mm512_loadu_ps( (__m512 const *) ( a + ( ic * rs_a ) +
                                  ( ( kr + 12 ) * cs_a ) ) );
      a_reg[13] = _mm512_loadu_ps( (__m512 const *) ( a + ( ic * rs_a ) +
                                  ( ( kr + 13 ) * cs_a ) ) );
      a_reg[14] = _mm512_loadu_ps( (__m512 const *) ( a + ( ic * rs_a ) +
                                  ( ( kr + 14 ) * cs_a ) ) );
      a_reg[15] = _mm512_loadu_ps( (__m512 const *) ( a + ( ic * rs_a ) +
                                  ( ( kr + 15 ) * cs_a ) ) );

      UNPACKLO_PS16
      UNPACKHI_PS16
      SHUFFLE_64x2
      PERMUTE4x4( mask1, mask2 );
      PERMUTE8x8( mask3, mask4 )

      _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), a_reg[0] );
      _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 1 ) * KC + kr ), a_reg[1] );
      _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 2 ) * KC + kr ), a_reg[2] );
      _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 3 ) * KC + kr ), a_reg[3] );
      _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 4 ) * KC + kr ), a_reg[4] );
      _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 5 ) * KC + kr ), a_reg[5] );
      _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 6 ) * KC + kr ), a_reg[6] );
      _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 7 ) * KC + kr ), a_reg[7] );
      _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 8 ) * KC + kr ), a_reg[8] );
      _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 9 ) * KC + kr ), a_reg[9] );
      _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 10 ) * KC + kr ), a_reg[10] );
      _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 11 ) * KC + kr ), a_reg[11] );
      _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 12 ) * KC + kr ), a_reg[12] );
      _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 13 ) * KC + kr ), a_reg[13] );
      _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 14 ) * KC + kr ), a_reg[14] );
      _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 15 ) * KC + kr ), a_reg[15] );
    }
    for ( ; ( kr + 7 ) < KC; kr += 8 )
    {
      a_reg[0] = _mm512_loadu_ps( (__m512 const *)( a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) ) );
      a_reg[1] = _mm512_loadu_ps( (__m512 const *)( a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) ) );
      a_reg[2] = _mm512_loadu_ps( (__m512 const *)( a + ( ic * rs_a ) + ( ( kr + 2 ) * cs_a ) ) );
      a_reg[3] = _mm512_loadu_ps( (__m512 const *)( a + ( ic * rs_a ) + ( ( kr + 3 ) * cs_a ) ) );
      a_reg[4] = _mm512_loadu_ps( (__m512 const *)( a + ( ic * rs_a ) + ( ( kr + 4 ) * cs_a ) ) );
      a_reg[5] = _mm512_loadu_ps( (__m512 const *)( a + ( ic * rs_a ) + ( ( kr + 5 ) * cs_a ) ) );
      a_reg[6] = _mm512_loadu_ps( (__m512 const *)( a + ( ic * rs_a ) + ( ( kr + 6 ) * cs_a ) ) );
      a_reg[7] = _mm512_loadu_ps( (__m512 const *)( a + ( ic * rs_a ) + ( ( kr + 7 ) * cs_a ) ) );

      UNPACKLO_PS16
      UNPACKHI_PS16
      SHUFFLE_64x2
      PERMUTE4x4( mask1, mask2 )
      PERMUTE8x8(mask3, mask4)
      MASKED_STORE_PS(0xFF);
    }
    for( ; ( kr + 3 ) < KC; kr += 4)
    {
      a_reg[0] = _mm512_loadu_ps( (__m512 const *)( a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) ) );
      a_reg[1] = _mm512_loadu_ps( (__m512 const *)( a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) ) );
      a_reg[2] = _mm512_loadu_ps( (__m512 const *)( a + ( ic * rs_a ) + ( ( kr + 2 ) * cs_a ) ) );
      a_reg[3] = _mm512_loadu_ps( (__m512 const *)( a + ( ic * rs_a ) + ( ( kr + 3 ) * cs_a ) ) );

      UNPACKLO_PS16
      UNPACKHI_PS16
      SHUFFLE_64x2
      PERMUTE4x4( mask1, mask2 )
      PERMUTE8x8( mask3, mask4 )
      MASKED_STORE_PS(0x0F);
    }
    for( ; ( kr + 1 ) < KC; kr += 2)
    {
      a_reg[0] = _mm512_loadu_ps( (__m512 const *)( a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) ) );
      a_reg[1] = _mm512_loadu_ps( (__m512 const *)( a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) ) );

      UNPACKLO_PS16
      UNPACKHI_PS16
      SHUFFLE_64x2
      PERMUTE4x4( mask1, mask2 )
      PERMUTE8x8(mask3, mask4)
      MASKED_STORE_PS(0x03);
    }
    for( ; ( kr ) < KC; kr += 1)
    {
      a_reg[0] = _mm512_loadu_ps( (__m512 const *)(a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) ) );

      UNPACKLO_PS16
      UNPACKHI_PS16
      SHUFFLE_64x2
      PERMUTE4x4( mask1, mask2 )
      PERMUTE8x8(mask3, mask4)
      MASKED_STORE_PS(0x01);
    }
  }
  for( ; (ic + 8 - 1) < MC; ic += 8)
  {
    for( kr = 0; ( kr + 15 ) < KC; kr += 16)
    {
      a_reg[0] = _mm512_maskz_loadu_ps( 0xFF, (__m512 const *) ( a + ( ic * rs_a ) +
                                      ( ( kr + 0 ) * cs_a ) ) );
      a_reg[1] = _mm512_maskz_loadu_ps( 0xFF, (__m512 const *) ( a + ( ic * rs_a ) +
                                      ( ( kr + 1 ) * cs_a ) ) );
      a_reg[2] = _mm512_maskz_loadu_ps( 0xFF, (__m512 const *) ( a + ( ic * rs_a ) +
                                      ( ( kr + 2 ) * cs_a ) ) );
      a_reg[3] = _mm512_maskz_loadu_ps( 0xFF, (__m512 const *) ( a + ( ic * rs_a ) +
                                      ( ( kr + 3 ) * cs_a ) ) );
      a_reg[4] = _mm512_maskz_loadu_ps( 0xFF, (__m512 const *) ( a + ( ic * rs_a ) +
                                      ( ( kr + 4 ) * cs_a ) ) );
      a_reg[5] = _mm512_maskz_loadu_ps( 0xFF, (__m512 const *) ( a + ( ic * rs_a ) +
                                      ( ( kr + 5 ) * cs_a ) ) );
      a_reg[6] = _mm512_maskz_loadu_ps( 0xFF, (__m512 const *) ( a + ( ic * rs_a ) +
                                      ( ( kr + 6 ) * cs_a ) ) );
      a_reg[7] = _mm512_maskz_loadu_ps( 0xFF, (__m512 const *) ( a + ( ic * rs_a ) +
                                      ( ( kr + 7 ) * cs_a ) ) );
      a_reg[8] = _mm512_maskz_loadu_ps( 0xFF, (__m512 const *) ( a + ( ic * rs_a ) +
                                      ( ( kr + 8 ) * cs_a ) ) );
      a_reg[9] = _mm512_maskz_loadu_ps( 0xFF, (__m512 const *) ( a + ( ic * rs_a ) +
                                      ( ( kr + 9 ) * cs_a ) ) );
      a_reg[10] = _mm512_maskz_loadu_ps( 0xFF, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 10 ) * cs_a ) ) );
      a_reg[11] = _mm512_maskz_loadu_ps( 0xFF, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 11 ) * cs_a ) ) );
      a_reg[12] = _mm512_maskz_loadu_ps( 0xFF, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 12 ) * cs_a ) ) );
      a_reg[13] = _mm512_maskz_loadu_ps( 0xFF, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 13 ) * cs_a ) ) );
      a_reg[14] = _mm512_maskz_loadu_ps( 0xFF, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 14 ) * cs_a ) ) );
      a_reg[15] = _mm512_maskz_loadu_ps( 0xFF, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 15 ) * cs_a ) ) );

      UNPACKLO_PS16
      UNPACKHI_PS16
      SHUFFLE_64x2
      PERMUTE4x4( mask1, mask2 )
      PERMUTE8x8( mask3, mask4 )
      _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), a_reg[0] );
      _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 1 ) * KC + kr ), a_reg[1] );
      _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 2 ) * KC + kr ), a_reg[2] );
      _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 3 ) * KC + kr ), a_reg[3] );
      _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 4 ) * KC + kr ), a_reg[4] );
      _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 5 ) * KC + kr ), a_reg[5] );
      _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 6 ) * KC + kr ), a_reg[6] );
      _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 7 ) * KC + kr ), a_reg[7] );
    }
    for( ; ( kr + 7 ) < KC; kr += 8)
    {
      a_reg[0] = (__m512)_mm512_maskz_loadu_ps( 0xFF, a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) );
      a_reg[1] = (__m512)_mm512_maskz_loadu_ps( 0xFF, a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) );
      a_reg[2] = (__m512)_mm512_maskz_loadu_ps( 0xFF, a + ( ic * rs_a ) + ( ( kr + 2 ) * cs_a ) );
      a_reg[3] = (__m512)_mm512_maskz_loadu_ps( 0xFF, a + ( ic * rs_a ) + ( ( kr + 3 ) * cs_a ) );
      a_reg[4] = (__m512)_mm512_maskz_loadu_ps( 0xFF, a + ( ic * rs_a ) + ( ( kr + 4 ) * cs_a ) );
      a_reg[5] = (__m512)_mm512_maskz_loadu_ps( 0xFF, a + ( ic * rs_a ) + ( ( kr + 5 ) * cs_a ) );
      a_reg[6] = (__m512)_mm512_maskz_loadu_ps( 0xFF, a + ( ic * rs_a ) + ( ( kr + 6 ) * cs_a ) );
      a_reg[7] = (__m512)_mm512_maskz_loadu_ps( 0xFF, a + ( ic * rs_a ) + ( ( kr + 7 ) * cs_a ) );

      UNPACKLO_PS16
      UNPACKHI_PS16
      SHUFFLE_64x2
      PERMUTE4x4(mask1, mask2)
      PERMUTE8x8(mask3, mask4)

      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0xFF, a_reg[0] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0xFF, a_reg[1] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 2 ) * KC + kr ), 0xFF, a_reg[2] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 3 ) * KC + kr ), 0xFF, a_reg[3] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 4 ) * KC + kr ), 0xFF, a_reg[4] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 5 ) * KC + kr ), 0xFF, a_reg[5] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 6 ) * KC + kr ), 0xFF, a_reg[6] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 7 ) * KC + kr ), 0xFF, a_reg[7] );
    }
    for( ; ( kr + 3 ) < KC; kr += 4)
    {
      a_reg[0] = (__m512)_mm512_maskz_loadu_ps( 0xFF, a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) );
      a_reg[1] = (__m512)_mm512_maskz_loadu_ps( 0xFF, a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) );
      a_reg[2] = (__m512)_mm512_maskz_loadu_ps( 0xFF, a + ( ic * rs_a ) + ( ( kr + 2 ) * cs_a ) );
      a_reg[3] = (__m512)_mm512_maskz_loadu_ps( 0xFF, a + ( ic * rs_a ) + ( ( kr + 3 ) * cs_a ) );

      UNPACKLO_PS16
      UNPACKHI_PS16
      SHUFFLE_64x2
      PERMUTE4x4( mask1, mask2 )
      PERMUTE8x8( mask3, mask4 )
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x0F, a_reg[0] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0x0F, a_reg[1] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 2 ) * KC + kr ), 0x0F, a_reg[2] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 3 ) * KC + kr ), 0x0F, a_reg[3] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 4 ) * KC + kr ), 0x0F, a_reg[4] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 5 ) * KC + kr ), 0x0F, a_reg[5] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 6 ) * KC + kr ), 0x0F, a_reg[6] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 7 ) * KC + kr ), 0x0F, a_reg[7] );
    }
    for( ; ( kr + 1 ) < KC; kr += 2)
    {
      a_reg[0] = (__m512)_mm512_maskz_loadu_ps( 0xFF, a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) );
      a_reg[1] = (__m512)_mm512_maskz_loadu_ps( 0xFF, a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) );

      UNPACKLO_PS16
      UNPACKHI_PS16
      SHUFFLE_64x2
      PERMUTE4x4( mask1, mask2 )
      PERMUTE8x8( mask3, mask4 )

      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x03, a_reg[0] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0x03, a_reg[1] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 2 ) * KC + kr ), 0x03, a_reg[2] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 3 ) * KC + kr ), 0x03, a_reg[3] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 4 ) * KC + kr ), 0x03, a_reg[4] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 5 ) * KC + kr ), 0x03, a_reg[5] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 6 ) * KC + kr ), 0x03, a_reg[6] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 7 ) * KC + kr ), 0x03, a_reg[7] );

    }
    for( ; ( kr ) < KC; kr += 1)
    {
      a_reg[0] = (__m512)_mm512_maskz_loadu_ps( 0xFF, a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) );

      UNPACKLO_PS16
      UNPACKHI_PS16
      SHUFFLE_64x2
      PERMUTE4x4( mask1, mask2 )
      PERMUTE8x8( mask3, mask4 )

      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x01, a_reg[0] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0x01, a_reg[1] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 2 ) * KC + kr ), 0x01, a_reg[2] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 3 ) * KC + kr ), 0x01, a_reg[3] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 4 ) * KC + kr ), 0x01, a_reg[4] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 5 ) * KC + kr ), 0x01, a_reg[5] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 6 ) * KC + kr ), 0x01, a_reg[6] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 7 ) * KC + kr ), 0x01, a_reg[7] );
    }
  }
  for( ; ( ic + 4 - 1 ) < MC; ic += 4)
  {
    for( kr = 0; ( kr + 15 ) < KC; kr += 16)
    {
      a_reg[0] = _mm512_maskz_loadu_ps ( 0x0F, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 0 ) * cs_a ) ) );
      a_reg[1] = _mm512_maskz_loadu_ps ( 0x0F, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 1 ) * cs_a ) ) );
      a_reg[2] = _mm512_maskz_loadu_ps ( 0x0F, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 2 ) * cs_a ) ) );
      a_reg[3] = _mm512_maskz_loadu_ps ( 0x0F, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 3 ) * cs_a ) ) );
      a_reg[4] = _mm512_maskz_loadu_ps ( 0x0F, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 4 ) * cs_a ) ) );
      a_reg[5] = _mm512_maskz_loadu_ps ( 0x0F, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 5 ) * cs_a ) ) );
      a_reg[6] = _mm512_maskz_loadu_ps ( 0x0F, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 6 ) * cs_a ) ) );
      a_reg[7] = _mm512_maskz_loadu_ps ( 0x0F, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 7 ) * cs_a ) ) );
      a_reg[8] = _mm512_maskz_loadu_ps ( 0x0F, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 8 ) * cs_a ) ) );
      a_reg[9] = _mm512_maskz_loadu_ps ( 0x0F, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 9 ) * cs_a ) ) );
      a_reg[10] = _mm512_maskz_loadu_ps ( 0x0F, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 10 ) * cs_a ) ) );
      a_reg[11] = _mm512_maskz_loadu_ps ( 0x0F, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 11 ) * cs_a ) ) );
      a_reg[12] = _mm512_maskz_loadu_ps ( 0x0F, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 12 ) * cs_a ) ) );
      a_reg[13] = _mm512_maskz_loadu_ps ( 0x0F, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 13 ) * cs_a ) ) );
      a_reg[14] = _mm512_maskz_loadu_ps ( 0x0F, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 14 ) * cs_a ) ) );
      a_reg[15] = _mm512_maskz_loadu_ps ( 0x0F, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 15 ) * cs_a ) ) );

      UNPACKLO_PS16
      UNPACKHI_PS16
      SHUFFLE_64x2
      PERMUTE4x4( mask1, mask2 )
      PERMUTE8x8( mask3, mask4 )

      _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), a_reg[0] );
      _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 1 ) * KC + kr ), a_reg[1] );
      _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 2 ) * KC + kr ), a_reg[2] );
      _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 3 ) * KC + kr ), a_reg[3] );
    }
    for( ; ( kr + 7 ) < KC; kr += 8)
    {
      a_reg[0] = (__m512)_mm512_maskz_loadu_ps( 0xFF, a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) );
      a_reg[1] = (__m512)_mm512_maskz_loadu_ps( 0xFF, a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) );
      a_reg[2] = (__m512)_mm512_maskz_loadu_ps( 0xFF, a + ( ic * rs_a ) + ( ( kr + 2 ) * cs_a ) );
      a_reg[3] = (__m512)_mm512_maskz_loadu_ps( 0xFF, a + ( ic * rs_a ) + ( ( kr + 3 ) * cs_a ) );
      a_reg[4] = (__m512)_mm512_maskz_loadu_ps( 0xFF, a + ( ic * rs_a ) + ( ( kr + 4 ) * cs_a ) );
      a_reg[5] = (__m512)_mm512_maskz_loadu_ps( 0xFF, a + ( ic * rs_a ) + ( ( kr + 5 ) * cs_a ) );
      a_reg[6] = (__m512)_mm512_maskz_loadu_ps( 0xFF, a + ( ic * rs_a ) + ( ( kr + 6 ) * cs_a ) );
      a_reg[7] = (__m512)_mm512_maskz_loadu_ps( 0xFF, a + ( ic * rs_a ) + ( ( kr + 7 ) * cs_a ) );

      UNPACKLO_PS16
      UNPACKHI_PS16
      SHUFFLE_64x2
      PERMUTE4x4(mask1, mask2)
      PERMUTE8x8(mask3, mask4)

      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0xFF, a_reg[0] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0xFF, a_reg[1] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 2 ) * KC + kr ), 0xFF, a_reg[2] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 3 ) * KC + kr ), 0xFF, a_reg[3] );
    }
    for( ; ( kr + 3 ) < KC; kr += 4)
    {
      a_reg[0] = (__m512)_mm512_maskz_loadu_ps( 0xFF, a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) );
      a_reg[1] = (__m512)_mm512_maskz_loadu_ps( 0xFF, a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) );
      a_reg[2] = (__m512)_mm512_maskz_loadu_ps( 0xFF, a + ( ic * rs_a ) + ( ( kr + 2 ) * cs_a ) );
      a_reg[3] = (__m512)_mm512_maskz_loadu_ps( 0xFF, a + ( ic * rs_a ) + ( ( kr + 3 ) * cs_a ) );

      UNPACKLO_PS16
      UNPACKHI_PS16
      SHUFFLE_64x2
      PERMUTE4x4( mask1, mask2 )
      PERMUTE8x8( mask3, mask4 )

      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x0F, a_reg[0] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0x0F, a_reg[1] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 2 ) * KC + kr ), 0x0F, a_reg[2] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 3 ) * KC + kr ), 0x0F, a_reg[3] );
    }
    for( ; ( kr + 1 ) < KC; kr += 2)
    {
      a_reg[0] = (__m512)_mm512_maskz_loadu_ps( 0xFF, a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) );
      a_reg[1] = (__m512)_mm512_maskz_loadu_ps( 0xFF, a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) );

      UNPACKLO_PS16
      UNPACKHI_PS16
      SHUFFLE_64x2
      PERMUTE4x4( mask1, mask2 )
      PERMUTE8x8( mask3, mask4 )

      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x03, a_reg[0] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0x03, a_reg[1] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 2 ) * KC + kr ), 0x03, a_reg[2] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 3 ) * KC + kr ), 0x03, a_reg[3] );
    }
    for( ; ( kr ) < KC; kr += 1)
    {
      a_reg[0] = (__m512)_mm512_maskz_loadu_ps( 0xFF, a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) );

      UNPACKLO_PS16
      UNPACKHI_PS16
      SHUFFLE_64x2
      PERMUTE4x4( mask1, mask2 )
      PERMUTE8x8( mask3, mask4 )

      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x01, a_reg[0] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0x01, a_reg[1] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 2 ) * KC + kr ), 0x01, a_reg[2] );
      _mm512_mask_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 3 ) * KC + kr ), 0x01, a_reg[3] );
    }
  }
  if( m_left ) {
    __mmask16 mask = 0xFFFF >> ( 16 - m_left );
    for( kr = 0; ( kr + 15 ) < KC; kr += 16)
    {
      a_reg[0] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 0 ) * cs_a ) ) );
      a_reg[1] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 1 ) * cs_a ) ) );
      a_reg[2] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 2 ) * cs_a ) ) );
      a_reg[3] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 3 ) * cs_a ) ) );
      a_reg[4] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 4 ) * cs_a ) ) );
      a_reg[5] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 5 ) * cs_a ) ) );
      a_reg[6] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 6 ) * cs_a ) ) );
      a_reg[7] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 7 ) * cs_a ) ) );
      a_reg[8] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 8 ) * cs_a ) ) );
      a_reg[9] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 9 ) * cs_a ) ) );
      a_reg[10] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 10 ) * cs_a ) ) );
      a_reg[11] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 11 ) * cs_a ) ) );
      a_reg[12] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 12 ) * cs_a ) ) );
      a_reg[13] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 13 ) * cs_a ) ) );
      a_reg[14] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 14 ) * cs_a ) ) );
      a_reg[15] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 15 ) * cs_a ) ) );

      UNPACKLO_PS16
      UNPACKHI_PS16
      SHUFFLE_64x2
      PERMUTE4x4( mask1, mask2 )
      PERMUTE8x8( mask3, mask4 )

      switch( m_left )
      {
        case 3:
          _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), a_reg[0] );
          _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 1 ) * KC + kr ), a_reg[1] );
          _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 2 ) * KC + kr ), a_reg[2] );
          break;

        case 2:
          _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), a_reg[0] );
          _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 1 ) * KC + kr ), a_reg[1] );
          break;

        case 1:
          _mm512_storeu_ps( (__m512 *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), a_reg[0] );
          break;
      }
    }
    for( ; ( kr + 7 ) < KC; kr += 8)
    {
      a_reg[0] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 0 ) * cs_a ) ) );
      a_reg[1] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 1 ) * cs_a ) ) );
      a_reg[2] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 2 ) * cs_a ) ) );
      a_reg[3] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 3 ) * cs_a ) ) );
      a_reg[4] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 4 ) * cs_a ) ) );
      a_reg[5] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 5 ) * cs_a ) ) );
      a_reg[6] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 6 ) * cs_a ) ) );
      a_reg[7] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 7 ) * cs_a ) ) );

      UNPACKLO_PS16
      UNPACKHI_PS16
      SHUFFLE_64x2
      PERMUTE4x4( mask1, mask2 )
      PERMUTE8x8( mask3, mask4 )

      switch( m_left )
      {
        case 3:
          _mm512_mask_storeu_ps((__m512 *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0xFF, a_reg[0]);
          _mm512_mask_storeu_ps((__m512 *)( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0xFF, a_reg[1]);
          _mm512_mask_storeu_ps((__m512 *)( pack_a_buffer + ( ic + 2 ) * KC + kr ), 0xFF, a_reg[2]);
          break;

        case 2:
          _mm512_mask_storeu_ps((__m512 *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0xFF, a_reg[0]);
          _mm512_mask_storeu_ps((__m512 *)( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0xFF, a_reg[1]);
          break;

        case 1:
          _mm512_mask_storeu_ps((__m512 *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0xFF, a_reg[0]);
          break;
      }
    }
    for( ; ( kr + 3 ) < KC; kr += 4)
    {
      a_reg[0] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 0 ) * cs_a ) ) );
      a_reg[1] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 1 ) * cs_a ) ) );
      a_reg[2] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 2 ) * cs_a ) ) );
      a_reg[3] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 3 ) * cs_a ) ) );

      UNPACKLO_PS16
      UNPACKHI_PS16
      SHUFFLE_64x2
      PERMUTE4x4( mask1, mask2 )
      PERMUTE8x8( mask3, mask4 )

      switch( m_left )
      {
        case 3:
          _mm512_mask_storeu_ps((__m512 *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x0F, a_reg[0]);
          _mm512_mask_storeu_ps((__m512 *)( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0x0F, a_reg[1]);
          _mm512_mask_storeu_ps((__m512 *)( pack_a_buffer + ( ic + 2 ) * KC + kr ), 0x0F, a_reg[2]);
          break;

        case 2:
          _mm512_mask_storeu_ps((__m512 *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x0F, a_reg[0]);
          _mm512_mask_storeu_ps((__m512 *)( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0x0F, a_reg[1]);
          break;

        case 1:
          _mm512_mask_storeu_ps((__m512 *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x0F, a_reg[0]);
          break;
      }
    }
    for( ; ( kr + 1 ) < KC; kr += 2)
    {
      a_reg[0] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 0 ) * cs_a ) ) );
      a_reg[1] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 1 ) * cs_a ) ) );

      UNPACKLO_PS16
      UNPACKHI_PS16
      SHUFFLE_64x2
      PERMUTE4x4( mask1, mask2 )
      PERMUTE8x8( mask3, mask4 )

      switch( m_left )
      {
        case 3:
          _mm512_mask_storeu_ps((__m512 *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x03, a_reg[0]);
          _mm512_mask_storeu_ps((__m512 *)( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0x03, a_reg[1]);
          _mm512_mask_storeu_ps((__m512 *)( pack_a_buffer + ( ic + 2 ) * KC + kr ), 0x03, a_reg[2]);
          break;

        case 2:
          _mm512_mask_storeu_ps((__m512 *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x03, a_reg[0]);
          _mm512_mask_storeu_ps((__m512 *)( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0x03, a_reg[1]);
          break;

        case 1:
          _mm512_mask_storeu_ps((__m512 *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x03, a_reg[0]);
          break;
      }
    }
    for( ; ( kr ) < KC; kr += 1)
    {
      a_reg[0] = _mm512_maskz_loadu_ps ( mask, (__m512 const *) ( a + ( ic * rs_a ) +
                                        ( ( kr + 0 ) * cs_a ) ) );

      UNPACKLO_PS16
      UNPACKHI_PS16
      SHUFFLE_64x2
      PERMUTE4x4( mask1, mask2 )
      PERMUTE8x8( mask3, mask4 )

      switch( m_left )
      {
        case 3:
          _mm512_mask_storeu_ps((__m512 *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x01, a_reg[0]);
          _mm512_mask_storeu_ps((__m512 *)( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0x01, a_reg[1]);
          _mm512_mask_storeu_ps((__m512 *)( pack_a_buffer + ( ic + 2 ) * KC + kr ), 0x01, a_reg[2]);
          break;

        case 2:
          _mm512_mask_storeu_ps((__m512 *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x01, a_reg[0]);
          _mm512_mask_storeu_ps((__m512 *)( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0x01, a_reg[1]);
          break;

        case 1:
          _mm512_mask_storeu_ps((__m512 *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x01, a_reg[0]);
          break;
      }
    }
  }
  *rs_p = KC;
  *cs_p = 1;
}
#endif
