/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#define F32_ROW_MAJOR_K_PACK_LOOP(pack_a_buf, KC, kr) \
	a01 = _mm512_unpacklo_ps( a0, b0 ); \
	a0 = _mm512_unpackhi_ps( a0, b0 ); \
 \
	c01 = _mm512_unpacklo_ps( c0, d0 ); \
	c0 = _mm512_unpackhi_ps( c0, d0 ); \
 \
	e01 = _mm512_unpacklo_ps( e0, f0 ); /* Elem 4 */ \
	e0 = _mm512_unpackhi_ps( e0, f0 ); /* Elem 5 */ \
 \
	b0 = _mm512_castpd_ps( _mm512_unpacklo_pd( _mm512_castps_pd( a01 ), \
			_mm512_castps_pd( c01 ) ) ); \
	a01 = _mm512_castpd_ps( _mm512_unpackhi_pd( _mm512_castps_pd( a01 ), \
			_mm512_castps_pd( c01 ) ) ); \
 \
	d0 = _mm512_castpd_ps( _mm512_unpacklo_pd( _mm512_castps_pd( a0 ), \
			_mm512_castps_pd( c0 ) ) ); \
	c01 = _mm512_castpd_ps( _mm512_unpackhi_pd( _mm512_castps_pd( a0 ), \
			_mm512_castps_pd( c0 ) ) ); \
 \
	a0 = _mm512_castpd_ps( _mm512_permutex2var_pd( _mm512_castps_pd( b0 ), \
			selector1, _mm512_castps_pd( a01 ) ) ); \
	c0 = _mm512_castpd_ps( _mm512_permutex2var_pd( _mm512_castps_pd( d0 ), \
			selector1, _mm512_castps_pd( c01 ) ) ); \
	b0 = _mm512_castpd_ps( _mm512_permutex2var_pd( _mm512_castps_pd( b0 ), \
			selector1_1, _mm512_castps_pd( a01 ) ) ); \
	d0 = _mm512_castpd_ps( _mm512_permutex2var_pd( _mm512_castps_pd( d0 ), \
			selector1_1, _mm512_castps_pd( c01 ) ) ); \
 \
	a01 = _mm512_castpd_ps( _mm512_permutex2var_pd( _mm512_castps_pd( a0 ), \
			selector2, _mm512_castps_pd( c0 ) ) ); /* a[0] */ \
	c01 = _mm512_castpd_ps( _mm512_permutex2var_pd( _mm512_castps_pd( b0 ), \
			selector2, _mm512_castps_pd( d0 ) ) ); /* a[2] */ \
	a0 = _mm512_castpd_ps( _mm512_permutex2var_pd( _mm512_castps_pd( a0 ), \
			selector2_1, _mm512_castps_pd( c0 ) ) ); /* a[1] */ \
	c0 = _mm512_castpd_ps( _mm512_permutex2var_pd( _mm512_castps_pd( b0 ), \
			selector2_1, _mm512_castps_pd( d0 ) ) ); /* a[3] */ \
 \
	/* First half */ \
	b0 = _mm512_castpd_ps( _mm512_permutex2var_pd( _mm512_castps_pd( a01 ), \
			selector3, _mm512_castps_pd( e01 ) ) ); /* 1st 16 */ \
	a01 = _mm512_castpd_ps( _mm512_permutex2var_pd( _mm512_castps_pd( a01 ), \
			selector4, _mm512_castps_pd( e0 ) ) ); /* 1st 8 */ \
	d0 = _mm512_castpd_ps( _mm512_permutex2var_pd( _mm512_castps_pd( a0 ), \
			selector5, _mm512_castps_pd( e01 ) ) ); /* 2nd 16 */ \
	a0 = _mm512_castpd_ps( _mm512_permutex2var_pd( _mm512_castps_pd( a0 ), \
			selector6, _mm512_castps_pd( e0 ) ) ); /* 2nd 4 */ \
 \
	_mm512_storeu_ps( pack_a_buf + ( ( ic * KC ) + ( ( kr * MR ) + ( 0 ) ) ), b0 ); \
	_mm512_storeu_ps( pack_a_buf + ( ( ic * KC ) + ( ( kr * MR ) + ( 16 ) ) ) , a01 ); \
	_mm512_storeu_ps( pack_a_buf + ( ( ic * KC ) + ( ( kr * MR ) + ( 24 ) ) ), d0 ); \
	/* Last piece */ \
	last_piece = _mm512_castps512_ps256( a0 ); \
	_mm256_mask_storeu_ps \
	( \
	  pack_a_buf + ( ( ic * KC ) + ( ( kr * MR ) + ( 40 ) ) ), \
	  _cvtu32_mask16( 0xFFFF), \
	  last_piece \
	); \
 \
	/* Second half */ \
	b0 = _mm512_castpd_ps( _mm512_permutex2var_pd( _mm512_castps_pd( c01 ), \
			selector7, _mm512_castps_pd( e01 ) ) ); /* 3rd 16 */ \
	c01 = _mm512_castpd_ps( _mm512_permutex2var_pd( _mm512_castps_pd( c01 ), \
			selector8, _mm512_castps_pd( e0 ) ) ); /* 3rd 8 */ \
	d0 = _mm512_castpd_ps( _mm512_permutex2var_pd( _mm512_castps_pd( c0 ), \
			selector9, _mm512_castps_pd( e01 ) ) ); /* 4th 16 */ \
	c0 = _mm512_castpd_ps( _mm512_permutex2var_pd( _mm512_castps_pd( c0 ), \
			selector10, _mm512_castps_pd( e0 ) ) ); /* 4th 8 */ \
 \
	_mm512_storeu_ps( pack_a_buf + ( ( ic * KC ) + ( ( kr * MR ) + ( 48 ) ) ), b0 ); \
	_mm512_storeu_ps( pack_a_buf + ( ( ic * KC ) + ( ( kr * MR ) + ( 64 ) ) ) , c01 ); \
	_mm512_storeu_ps( pack_a_buf + ( ( ic * KC ) + ( ( kr * MR ) + ( 72 ) ) ), d0 ); \
	/* Last piece */ \
	last_piece = _mm512_castps512_ps256( c0 ); \
	_mm256_mask_storeu_ps \
	( \
	  pack_a_buf + ( ( ic * KC ) + ( ( kr * MR ) + ( 88 ) ) ), \
	  _cvtu32_mask16( 0xFFFF), \
	  last_piece \
	); \

// Row Major Packing in blocks of MRxKC
void packa_f32f32f32of32_row_major_avx512
     (
       float*       pack_a_buf,
       const float* a,
       const dim_t  lda,
       const dim_t  MC,
       const dim_t  KC,
       dim_t*       rs_a,
       dim_t*       cs_a
     )
{
	const dim_t MR = 6;
	const dim_t KR_NDIM = 16;

	// Used for permuting the mm512i elements for use in vpdpbusd instruction.
	// These are indexes of the format a0-a1-b0-b1-a2-a3-b2-b3 and a0-a1-a2-a3-b0-b1-b2-b3.
	// Adding 4 int32 wise gives format a4-a5-b4-b5-a6-a7-b6-b7 and a4-a5-a6-a7-b4-b5-b6-b7.
	__m512i selector1 = _mm512_setr_epi64( 0x0, 0x1, 0x8, 0x9, 0x2, 0x3, 0xA, 0xB );
	__m512i selector1_1 = _mm512_setr_epi64( 0x4, 0x5, 0xC, 0xD, 0x6, 0x7, 0xE, 0xF );
	__m512i selector2 = _mm512_setr_epi64( 0x0, 0x1, 0x2, 0x3, 0x8, 0x9, 0xA, 0xB );
	__m512i selector2_1 = _mm512_setr_epi64( 0x4, 0x5, 0x6, 0x7, 0xC, 0xD, 0xE, 0xF );

	// First half.
	__m512i selector3 = _mm512_setr_epi64( 0x0, 0x1, 0x8, 0x2, 0x3, 0x9, 0x4, 0x5 ); // 64 elems
	__m512i selector4 = _mm512_setr_epi64( 0x8, 0x6, 0x7, 0x9, 0x0, 0x0, 0x0, 0x0 ); // 32 elems
	__m512i selector5 = _mm512_setr_epi64( 0x0, 0x1, 0xA, 0x2, 0x3, 0xB, 0x4, 0x5 ); // 64 elems
	__m512i selector6 = _mm512_setr_epi64( 0xA, 0x6, 0x7, 0xB, 0x0, 0x0, 0x0, 0x0 ); // 32 elems

	// Second half.
	__m512i selector7 = _mm512_setr_epi64( 0x0, 0x1, 0xC, 0x2, 0x3, 0xD, 0x4, 0x5 ); // 64 elems
	__m512i selector8 = _mm512_setr_epi64( 0xC, 0x6, 0x7, 0xD, 0x0, 0x0, 0x0, 0x0 ); // 32 elems
	__m512i selector9 = _mm512_setr_epi64( 0x0, 0x1, 0xE, 0x2, 0x3, 0xF, 0x4, 0x5 ); // 64 elems
	__m512i selector10 = _mm512_setr_epi64( 0xE, 0x6, 0x7, 0xF, 0x0, 0x0, 0x0, 0x0 ); // 32 elems

	dim_t m_full_pieces = MC / MR;
	dim_t m_full_pieces_loop_limit = m_full_pieces * MR;
	dim_t m_partial_pieces = MC % MR;

	dim_t kr_full_pieces = KC / KR_NDIM;
	dim_t kr_full_pieces_loop_limit = kr_full_pieces * KR_NDIM;
	dim_t kr_partial_pieces = KC % KR_NDIM;

	__m512 a0;
	__m512 b0;
	__m512 c0;
	__m512 d0;
	__m512 e0;
	__m512 f0;
	__m512 a01;
	__m512 c01;
	__m512 e01;
	__m256 last_piece;

	__mmask16 mmask[6];
	for ( dim_t ic = 0; ic < MC; ic += MR )
	{
		if ( ic == m_full_pieces_loop_limit )
		{
			for ( int ii = 0; ii < m_partial_pieces; ++ii )
			{
				mmask[ii] = _cvtu32_mask16( 0xFFFF );
			}
			for ( int ii = m_partial_pieces; ii < MR; ++ii )
			{
				mmask[ii] = _cvtu32_mask16( 0x0 );
			}
		}
		else
		{
			for ( int ii = 0; ii < MR; ++ii )
			{
				mmask[ii] = _cvtu32_mask16( 0xFFFF );
			}
		}
		for ( dim_t kr = 0; kr < kr_full_pieces_loop_limit; kr += KR_NDIM )
		{
			a0 = _mm512_maskz_loadu_ps( mmask[0], a + ( lda * ( ic + 0 ) ) + kr );
			b0 = _mm512_maskz_loadu_ps( mmask[1], a + ( lda * ( ic + 1 ) ) + kr );
			c0 = _mm512_maskz_loadu_ps( mmask[2], a + ( lda * ( ic + 2 ) ) + kr );
			d0 = _mm512_maskz_loadu_ps( mmask[3], a + ( lda * ( ic + 3 ) ) + kr );
			e0 = _mm512_maskz_loadu_ps( mmask[4], a + ( lda * ( ic + 4 ) ) + kr );
			f0 = _mm512_maskz_loadu_ps( mmask[5], a + ( lda * ( ic + 5 ) ) + kr );

			F32_ROW_MAJOR_K_PACK_LOOP(pack_a_buf, KC, kr);
		}
		if ( kr_partial_pieces > 0 )
		{
			err_t r_val;
			size_t temp_size = MR * KR_NDIM * sizeof( float );
			float* temp_pack_a_buf = bli_malloc_user( temp_size, &r_val );

			__mmask16 lmask = _cvtu32_mask16( 0xFFFF >> ( 16 - kr_partial_pieces ) );
			for ( int ii = 0; ii < MR; ++ii )
			{
				mmask[ii] = _mm512_kand( mmask[ii], lmask );
			}

			a0 = _mm512_maskz_loadu_ps( mmask[0], a + ( lda * ( ic + 0 ) ) +
							kr_full_pieces_loop_limit );
			b0 = _mm512_maskz_loadu_ps( mmask[1], a + ( lda * ( ic + 1 ) ) +
							kr_full_pieces_loop_limit );
			c0 = _mm512_maskz_loadu_ps( mmask[2], a + ( lda * ( ic + 2 ) ) +
							kr_full_pieces_loop_limit );
			d0 = _mm512_maskz_loadu_ps( mmask[3], a + ( lda * ( ic + 3 ) ) +
							kr_full_pieces_loop_limit );
			e0 = _mm512_maskz_loadu_ps( mmask[4], a + ( lda * ( ic + 4 ) ) +
							kr_full_pieces_loop_limit );
			f0 = _mm512_maskz_loadu_ps( mmask[5], a + ( lda * ( ic + 5 ) ) +
							kr_full_pieces_loop_limit );

			F32_ROW_MAJOR_K_PACK_LOOP(temp_pack_a_buf, 0, 0);

			memcpy
			(
			  pack_a_buf + ( ic * KC ) + ( kr_full_pieces_loop_limit * MR ),
			  temp_pack_a_buf,
			  kr_partial_pieces * MR * sizeof( float )
			);

			bli_free_user( temp_pack_a_buf );
		}
	}

	*rs_a = 1;
	*cs_a = 6;
}

void packa_f32f32f32of32_col_major_avx512
     (
       float*       pack_a_buf,
       const float* a,
       const dim_t  lda,
       const dim_t  MC,
       const dim_t  KC,
       dim_t*       rs_a,
       dim_t*       cs_a
     )
{
	const dim_t MR = 6;
	const dim_t KR_NDIM = 16;

	dim_t m_full_pieces = MC / MR;
	dim_t m_full_pieces_loop_limit = m_full_pieces * MR;
	dim_t m_partial_pieces = MC % MR;

	dim_t kr_full_pieces = KC / KR_NDIM;
	dim_t kr_full_pieces_loop_limit = kr_full_pieces * KR_NDIM;
	dim_t kr_partial_pieces = KC % KR_NDIM;

	__m256 a0;
	__m256 b0;
	__m256 c0;
	__m256 d0;
	__m256 e0;
	__m256 f0;
	__m256 g0;
	__m256 h0;
	__m256 i0;
	__m256 j0;
	__m256 k0;
	__m256 l0;
	__m256 m0;
	__m256 n0;
	__m256 o0;
	__m256 p0;

	__mmask16 mmask[16];

	for ( dim_t ic = 0; ic < MC; ic += MR )
	{
		if ( ic == m_full_pieces_loop_limit )
		{
			for ( int ii = 0; ii < 16; ++ii )
			{
				mmask[ii] = _cvtu32_mask16( 0x3F >> ( MR - m_partial_pieces ) );
			}
		}
		/* Inside the kr loop, the mmask is modified. Need to reset it
		 * at beginning of each ic loop iteration. */
		else
		{
			for ( int ii = 0; ii < 16; ++ii )
			{
				mmask[ii] = _cvtu32_mask16( 0x3F );
			}
		}
		for ( dim_t kr = 0; kr < KC; kr += KR_NDIM )
		{
			if ( kr == kr_full_pieces_loop_limit )
			{
				for ( int ii = kr_partial_pieces; ii < 16; ++ii )
				{
					mmask[ii] = _cvtu32_mask16( 0x0 );
				}
			}
			a0 = _mm256_maskz_loadu_ps( mmask[0], a + ic + ( lda * ( kr + 0 ) ) );
			b0 = _mm256_maskz_loadu_ps( mmask[1], a + ic + ( lda * ( kr + 1 ) ) );
			c0 = _mm256_maskz_loadu_ps( mmask[2], a + ic + ( lda * ( kr + 2 ) ) );
			d0 = _mm256_maskz_loadu_ps( mmask[3], a + ic + ( lda * ( kr + 3 ) ) );
			e0 = _mm256_maskz_loadu_ps( mmask[4], a + ic + ( lda * ( kr + 4 ) ) );
			f0 = _mm256_maskz_loadu_ps( mmask[5], a + ic + ( lda * ( kr + 5 ) ) );
			g0 = _mm256_maskz_loadu_ps( mmask[6], a + ic + ( lda * ( kr + 6 ) ) );
			h0 = _mm256_maskz_loadu_ps( mmask[7], a + ic + ( lda * ( kr + 7 ) ) );
			i0 = _mm256_maskz_loadu_ps( mmask[8], a + ic + ( lda * ( kr + 8 ) ) );
			j0 = _mm256_maskz_loadu_ps( mmask[9], a + ic + ( lda * ( kr + 9 ) ) );
			k0 = _mm256_maskz_loadu_ps( mmask[10], a + ic + ( lda * ( kr + 10 ) ) );
			l0 = _mm256_maskz_loadu_ps( mmask[11], a + ic + ( lda * ( kr + 11 ) ) );
			m0 = _mm256_maskz_loadu_ps( mmask[12], a + ic + ( lda * ( kr + 12 ) ) );
			n0 = _mm256_maskz_loadu_ps( mmask[13], a + ic + ( lda * ( kr + 13 ) ) );
			o0 = _mm256_maskz_loadu_ps( mmask[14], a + ic + ( lda * ( kr + 14 ) ) );
			p0 = _mm256_maskz_loadu_ps( mmask[15], a + ic + ( lda * ( kr + 15 ) ) );

			_mm256_mask_storeu_ps
			(
			  pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 0 ) ),
			  mmask[0], a0
			);
			_mm256_mask_storeu_ps
			(
			  pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 1 ) ),
			  mmask[1], b0
			);
			_mm256_mask_storeu_ps
			(
			  pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 2 ) ),
			  mmask[2], c0
			);
			_mm256_mask_storeu_ps
			(
			  pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 3 ) ),
			  mmask[3], d0
			);
			_mm256_mask_storeu_ps
			(
			  pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 4 ) ),
			  mmask[4], e0
			);
			_mm256_mask_storeu_ps
			(
			  pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 5 ) ),
			  mmask[5], f0
			);
			_mm256_mask_storeu_ps
			(
			  pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 6 ) ),
			  mmask[6], g0
			);
			_mm256_mask_storeu_ps
			(
			  pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 7 ) ),
			  mmask[7], h0
			);
			_mm256_mask_storeu_ps
			(
			  pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 8 ) ),
			  mmask[8], i0
			);
			_mm256_mask_storeu_ps
			(
			  pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 9 ) ),
			  mmask[9], j0
			);
			_mm256_mask_storeu_ps
			(
			  pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 10 ) ),
			  mmask[10], k0
			);
			_mm256_mask_storeu_ps
			(
			  pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 11 ) ),
			  mmask[11], l0
			);
			_mm256_mask_storeu_ps
			(
			  pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 12 ) ),
			  mmask[12], m0
			);
			_mm256_mask_storeu_ps
			(
			  pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 13 ) ),
			  mmask[13], n0
			);
			_mm256_mask_storeu_ps
			(
			  pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 14 ) ),
			  mmask[14], o0
			);
			_mm256_mask_storeu_ps
			(
			  pack_a_buf + ( ic * KC ) + ( ( kr * MR ) + ( MR * 15 ) ),
			  mmask[15], p0
			);
		}
	}
}

void packa_mr6_f32f32f32of32_avx512
     (
       float*       pack_a_buf,
       const float* a,
       const dim_t  rs,
       const dim_t  cs,
       const dim_t  MC,
       const dim_t  KC,
       dim_t*       rs_a,
       dim_t*       cs_a
     )
{
	if( cs == 1 )
	{
		packa_f32f32f32of32_row_major_avx512
		( pack_a_buf, a, rs, MC, KC, rs_a, cs_a );
	}
	else
	{
		packa_f32f32f32of32_col_major_avx512
		( pack_a_buf, a, cs, MC, KC, rs_a, cs_a );
	}
}

#endif
