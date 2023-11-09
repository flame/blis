/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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


#define UNPACKLO_EPI16 \
	b_reg[0] = _mm256_unpacklo_epi16(a_reg[0], a_reg[1]); \
	b_reg[1] = _mm256_unpacklo_epi16(a_reg[2], a_reg[3]); \
	b_reg[2] = _mm256_unpacklo_epi16(a_reg[4], a_reg[5]); \
	b_reg[3] = _mm256_unpacklo_epi16(a_reg[6], a_reg[7]); \
	b_reg[4] = _mm256_unpacklo_epi16(a_reg[8], a_reg[9]); \
	b_reg[5] = _mm256_unpacklo_epi16(a_reg[10], a_reg[11]); \
	b_reg[6] = _mm256_unpacklo_epi16(a_reg[12], a_reg[13]); \
	b_reg[7] = _mm256_unpacklo_epi16(a_reg[14], a_reg[15]);

#define UNPACKHI_EPI16 \
	b_reg[8] = _mm256_unpackhi_epi16(a_reg[0], a_reg[1]); \
	b_reg[9] = _mm256_unpackhi_epi16(a_reg[2], a_reg[3]); \
	b_reg[10] = _mm256_unpackhi_epi16(a_reg[4], a_reg[5]); \
	b_reg[11] = _mm256_unpackhi_epi16(a_reg[6], a_reg[7]); \
	b_reg[12] = _mm256_unpackhi_epi16(a_reg[8], a_reg[9]); \
	b_reg[13] = _mm256_unpackhi_epi16(a_reg[10], a_reg[11]); \
	b_reg[14] = _mm256_unpackhi_epi16(a_reg[12], a_reg[13]); \
	b_reg[15] = _mm256_unpackhi_epi16(a_reg[14], a_reg[15]);

#define UNPACKLO_EPI32 \
	a_reg[0] = _mm256_unpacklo_epi32(b_reg[0], b_reg[1]); \
	a_reg[1] = _mm256_unpacklo_epi32(b_reg[2], b_reg[3]); \
	a_reg[2] = _mm256_unpacklo_epi32(b_reg[4], b_reg[5]); \
	a_reg[3] = _mm256_unpacklo_epi32(b_reg[6], b_reg[7]); \
\
	a_reg[8] = _mm256_unpacklo_epi32(b_reg[8], b_reg[9]); \
	a_reg[9] = _mm256_unpacklo_epi32(b_reg[10], b_reg[11]); \
	a_reg[10] = _mm256_unpacklo_epi32(b_reg[12], b_reg[13]); \
	a_reg[11] = _mm256_unpacklo_epi32(b_reg[14], b_reg[15]);

#define UNPACKHI_EPI32 \
	a_reg[4] = _mm256_unpackhi_epi32(b_reg[0], b_reg[1]); \
	a_reg[5] = _mm256_unpackhi_epi32(b_reg[2], b_reg[3]); \
	a_reg[6] = _mm256_unpackhi_epi32(b_reg[4], b_reg[5]); \
	a_reg[7] = _mm256_unpackhi_epi32(b_reg[6], b_reg[7]); \
\
	a_reg[12] = _mm256_unpackhi_epi32(b_reg[8], b_reg[9]); \
	a_reg[13] = _mm256_unpackhi_epi32(b_reg[10], b_reg[11]); \
	a_reg[14] = _mm256_unpackhi_epi32(b_reg[12], b_reg[13]); \
	a_reg[15] = _mm256_unpackhi_epi32(b_reg[14], b_reg[15]);

#define UNPACKLO_EPI64 \
	b_reg[0] = _mm256_unpacklo_epi64(a_reg[0], a_reg[1]); \
	b_reg[1] = _mm256_unpacklo_epi64(a_reg[2], a_reg[3]); \
	b_reg[2] = _mm256_unpacklo_epi64(a_reg[4], a_reg[5]); \
	b_reg[3] = _mm256_unpacklo_epi64(a_reg[6], a_reg[7]); \
\
	b_reg[8] = _mm256_unpacklo_epi64(a_reg[8], a_reg[9]); \
	b_reg[9] = _mm256_unpacklo_epi64(a_reg[10], a_reg[11]); \
	b_reg[10] = _mm256_unpacklo_epi64(a_reg[12], a_reg[13]); \
	b_reg[11] = _mm256_unpacklo_epi64(a_reg[14], a_reg[15]);

#define UNPACKHI_EPI64 \
	b_reg[4] = _mm256_unpackhi_epi64(a_reg[0], a_reg[1]); \
	b_reg[5] = _mm256_unpackhi_epi64(a_reg[2], a_reg[3]); \
	b_reg[6] = _mm256_unpackhi_epi64(a_reg[4], a_reg[5]); \
	b_reg[7] = _mm256_unpackhi_epi64(a_reg[6], a_reg[7]); \
\
	b_reg[12] = _mm256_unpackhi_epi64(a_reg[8], a_reg[9]); \
	b_reg[13] = _mm256_unpackhi_epi64(a_reg[10], a_reg[11]); \
	b_reg[14] = _mm256_unpackhi_epi64(a_reg[12], a_reg[13]); \
	b_reg[15] = _mm256_unpackhi_epi64(a_reg[14], a_reg[15]);

#define SHUFFLE_64x2 \
	a_reg[0] = _mm256_shuffle_i64x2(b_reg[0], b_reg[1], 0x0); \
	a_reg[1] = _mm256_shuffle_i64x2(b_reg[0], b_reg[1], 0x3); \
	a_reg[2] = _mm256_shuffle_i64x2(b_reg[2], b_reg[3], 0x0); \
	a_reg[3] = _mm256_shuffle_i64x2(b_reg[2], b_reg[3], 0x3); \
\
	a_reg[4] = _mm256_shuffle_i64x2(b_reg[4], b_reg[5], 0x0); \
	a_reg[5] = _mm256_shuffle_i64x2(b_reg[4], b_reg[5], 0x3); \
	a_reg[6] = _mm256_shuffle_i64x2(b_reg[6], b_reg[7], 0x0); \
	a_reg[7] = _mm256_shuffle_i64x2(b_reg[6], b_reg[7], 0x3); \
\
	a_reg[8] = _mm256_shuffle_i64x2(b_reg[8], b_reg[9], 0x0); \
	a_reg[9] = _mm256_shuffle_i64x2(b_reg[8], b_reg[9], 0x3); \
	a_reg[10] = _mm256_shuffle_i64x2(b_reg[10], b_reg[11], 0x0); \
	a_reg[11] = _mm256_shuffle_i64x2(b_reg[10], b_reg[11], 0x3); \
\
	a_reg[12] = _mm256_shuffle_i64x2(b_reg[12], b_reg[13], 0x0); \
	a_reg[13] = _mm256_shuffle_i64x2(b_reg[12], b_reg[13], 0x3); \
	a_reg[14] = _mm256_shuffle_i64x2(b_reg[14], b_reg[15], 0x0); \
	a_reg[15] = _mm256_shuffle_i64x2(b_reg[14], b_reg[15], 0x3);

#define MASKED_STORE_EPI64(mask) \
	_mm256_mask_storeu_epi64((pack_a_buffer + (ic+0) * KC + kr ), mask, a_reg[0]); \
	_mm256_mask_storeu_epi64((pack_a_buffer + (ic+1) * KC + kr ), mask, a_reg[4]); \
	_mm256_mask_storeu_epi64((pack_a_buffer + (ic+2) * KC + kr ), mask, a_reg[2]); \
	_mm256_mask_storeu_epi64((pack_a_buffer + (ic+3) * KC + kr ), mask, a_reg[6]); \
	_mm256_mask_storeu_epi64((pack_a_buffer + (ic+4) * KC + kr ), mask, a_reg[8]); \
	_mm256_mask_storeu_epi64((pack_a_buffer + (ic+5) * KC + kr ), mask, a_reg[12]); \
	_mm256_mask_storeu_epi64((pack_a_buffer + (ic+6) * KC + kr ), mask, a_reg[10]); \
	_mm256_mask_storeu_epi64((pack_a_buffer + (ic+7) * KC + kr ), mask, a_reg[14]); \
	_mm256_mask_storeu_epi64((pack_a_buffer + (ic+8) * KC + kr ), mask, a_reg[1]); \
	_mm256_mask_storeu_epi64((pack_a_buffer + (ic+9) * KC + kr ), mask, a_reg[5]); \
	_mm256_mask_storeu_epi64((pack_a_buffer + (ic+10) * KC + kr ), mask, a_reg[3]); \
	_mm256_mask_storeu_epi64((pack_a_buffer + (ic+11) * KC + kr ), mask, a_reg[7]); \
	_mm256_mask_storeu_epi64((pack_a_buffer + (ic+12) * KC + kr ), mask, a_reg[9]); \
	_mm256_mask_storeu_epi64((pack_a_buffer + (ic+13) * KC + kr ), mask, a_reg[13]); \
	_mm256_mask_storeu_epi64((pack_a_buffer + (ic+14) * KC + kr ), mask, a_reg[11]); \
	_mm256_mask_storeu_epi64((pack_a_buffer + (ic+15) * KC + kr ), mask, a_reg[15]);

#define MASKED_STORE_EPI32(mask) \
	_mm256_mask_storeu_epi32((pack_a_buffer + (ic+0) * KC + kr ), mask, a_reg[0]); \
	_mm256_mask_storeu_epi32((pack_a_buffer + (ic+1) * KC + kr ), mask, a_reg[4]); \
	_mm256_mask_storeu_epi32((pack_a_buffer + (ic+2) * KC + kr ), mask, a_reg[2]); \
	_mm256_mask_storeu_epi32((pack_a_buffer + (ic+3) * KC + kr ), mask, a_reg[6]); \
	_mm256_mask_storeu_epi32((pack_a_buffer + (ic+4) * KC + kr ), mask, a_reg[8]); \
	_mm256_mask_storeu_epi32((pack_a_buffer + (ic+5) * KC + kr ), mask, a_reg[12]); \
	_mm256_mask_storeu_epi32((pack_a_buffer + (ic+6) * KC + kr ), mask, a_reg[10]); \
	_mm256_mask_storeu_epi32((pack_a_buffer + (ic+7) * KC + kr ), mask, a_reg[14]); \
	_mm256_mask_storeu_epi32((pack_a_buffer + (ic+8) * KC + kr ), mask, a_reg[1]); \
	_mm256_mask_storeu_epi32((pack_a_buffer + (ic+9) * KC + kr ), mask, a_reg[5]); \
	_mm256_mask_storeu_epi32((pack_a_buffer + (ic+10) * KC + kr ), mask, a_reg[3]); \
	_mm256_mask_storeu_epi32((pack_a_buffer + (ic+11) * KC + kr ), mask, a_reg[7]); \
	_mm256_mask_storeu_epi32((pack_a_buffer + (ic+12) * KC + kr ), mask, a_reg[9]); \
	_mm256_mask_storeu_epi32((pack_a_buffer + (ic+13) * KC + kr ), mask, a_reg[13]); \
	_mm256_mask_storeu_epi32((pack_a_buffer + (ic+14) * KC + kr ), mask, a_reg[11]); \
	_mm256_mask_storeu_epi32((pack_a_buffer + (ic+15) * KC + kr ), mask, a_reg[15]);

#define MASKED_STORE_EPI16(mask) \
	_mm256_mask_storeu_epi16((pack_a_buffer + (ic+0) * KC + kr ), mask, a_reg[0]); \
	_mm256_mask_storeu_epi16((pack_a_buffer + (ic+1) * KC + kr ), mask, a_reg[4]); \
	_mm256_mask_storeu_epi16((pack_a_buffer + (ic+2) * KC + kr ), mask, a_reg[2]); \
	_mm256_mask_storeu_epi16((pack_a_buffer + (ic+3) * KC + kr ), mask, a_reg[6]); \
	_mm256_mask_storeu_epi16((pack_a_buffer + (ic+4) * KC + kr ), mask, a_reg[8]); \
	_mm256_mask_storeu_epi16((pack_a_buffer + (ic+5) * KC + kr ), mask, a_reg[12]); \
	_mm256_mask_storeu_epi16((pack_a_buffer + (ic+6) * KC + kr ), mask, a_reg[10]); \
	_mm256_mask_storeu_epi16((pack_a_buffer + (ic+7) * KC + kr ), mask, a_reg[14]); \
	_mm256_mask_storeu_epi16((pack_a_buffer + (ic+8) * KC + kr ), mask, a_reg[1]); \
	_mm256_mask_storeu_epi16((pack_a_buffer + (ic+9) * KC + kr ), mask, a_reg[5]); \
	_mm256_mask_storeu_epi16((pack_a_buffer + (ic+10) * KC + kr ), mask, a_reg[3]); \
	_mm256_mask_storeu_epi16((pack_a_buffer + (ic+11) * KC + kr ), mask, a_reg[7]); \
	_mm256_mask_storeu_epi16((pack_a_buffer + (ic+12) * KC + kr ), mask, a_reg[9]); \
	_mm256_mask_storeu_epi16((pack_a_buffer + (ic+13) * KC + kr ), mask, a_reg[13]); \
	_mm256_mask_storeu_epi16((pack_a_buffer + (ic+14) * KC + kr ), mask, a_reg[11]); \
	_mm256_mask_storeu_epi16((pack_a_buffer + (ic+15) * KC + kr ), mask, a_reg[15]);

#define MASKED_LOAD_32_ROWS_AVX512( mask ) \
	a_reg[0] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 0 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[1] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 1 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[2] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 2 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[3] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 3 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[4] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 4 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[5] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 5 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[6] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 6 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[7] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 7 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[8] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 8 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[9] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 9 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[10] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 10 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[11] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 11 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[12] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 12 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[13] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 13 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[14] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 14 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[15] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 15 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[16] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 16 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[17] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 17 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[18] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 18 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[19] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 19 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[20] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 20 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[21] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 21 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[22] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 22 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[23] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 23 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[24] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 24 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[25] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 25 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[26] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 26 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[27] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 27 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[28] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 28 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[29] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 29 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[30] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 30 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[31] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 31 ) * rs_a ) + ( kr * cs_a ));

#define MASKED_STORE_32_ROWS_AVX512( mask ) \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr, mask, a_reg[0] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 1 ) * KC ) + kr, mask, a_reg[1] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 2 ) * KC ) + kr, mask, a_reg[2] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 3 ) * KC ) + kr, mask, a_reg[3] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 4 ) * KC ) + kr, mask, a_reg[4] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 5 ) * KC ) + kr, mask, a_reg[5] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 6 ) * KC ) + kr, mask, a_reg[6] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 7 ) * KC ) + kr, mask, a_reg[7] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 8 ) * KC ) + kr, mask, a_reg[8] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 9 ) * KC ) + kr, mask, a_reg[9] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 10 ) * KC ) + kr, mask, a_reg[10] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 11 ) * KC ) + kr, mask, a_reg[11] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 12 ) * KC ) + kr, mask, a_reg[12] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 13 ) * KC ) + kr, mask, a_reg[13] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 14 ) * KC ) + kr, mask, a_reg[14] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 15 ) * KC ) + kr, mask, a_reg[15] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 16 ) * KC ) + kr, mask, a_reg[16] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 17 ) * KC ) + kr, mask, a_reg[17] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 18 ) * KC ) + kr, mask, a_reg[18] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 19 ) * KC ) + kr, mask, a_reg[19] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 20 ) * KC ) + kr, mask, a_reg[20] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 21 ) * KC ) + kr, mask, a_reg[21] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 22 ) * KC ) + kr, mask, a_reg[22] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 23 ) * KC ) + kr, mask, a_reg[23] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 24 ) * KC ) + kr, mask, a_reg[24] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 25 ) * KC ) + kr, mask, a_reg[25] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 26 ) * KC ) + kr, mask, a_reg[26] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 27 ) * KC ) + kr, mask, a_reg[27] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 28 ) * KC ) + kr, mask, a_reg[28] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 29 ) * KC ) + kr, mask, a_reg[29] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 30 ) * KC ) + kr, mask, a_reg[30] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 31 ) * KC ) + kr, mask, a_reg[31] );


#define MASKED_LOAD_16_ROWS_AVX512( mask ) \
	a_reg[0] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 0 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[1] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 1 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[2] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 2 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[3] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 3 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[4] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 4 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[5] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 5 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[6] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 6 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[7] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 7 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[8] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 8 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[9] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 9 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[10] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 10 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[11] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 11 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[12] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 12 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[13] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 13 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[14] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 14 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[15] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 15 ) * rs_a ) + ( kr * cs_a ));

#define MASKED_STORE_16_ROWS_AVX512( mask ) \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr, mask, a_reg[0] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 1 ) * KC ) + kr, mask, a_reg[1] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 2 ) * KC ) + kr, mask, a_reg[2] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 3 ) * KC ) + kr, mask, a_reg[3] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 4 ) * KC ) + kr, mask, a_reg[4] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 5 ) * KC ) + kr, mask, a_reg[5] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 6 ) * KC ) + kr, mask, a_reg[6] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 7 ) * KC ) + kr, mask, a_reg[7] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 8 ) * KC ) + kr, mask, a_reg[8] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 9 ) * KC ) + kr, mask, a_reg[9] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 10 ) * KC ) + kr, mask, a_reg[10] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 11 ) * KC ) + kr, mask, a_reg[11] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 12 ) * KC ) + kr, mask, a_reg[12] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 13 ) * KC ) + kr, mask, a_reg[13] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 14 ) * KC ) + kr, mask, a_reg[14] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 15 ) * KC ) + kr, mask, a_reg[15] );


#define MASKED_LOAD_8_ROWS_AVX512( mask ) \
	a_reg[0] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 0 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[1] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 1 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[2] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 2 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[3] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 3 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[4] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 4 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[5] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 5 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[6] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 6 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[7] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 7 ) * rs_a ) + ( kr * cs_a ));

#define MASKED_STORE_8_ROWS_AVX512( mask ) \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr, mask, a_reg[0] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 1 ) * KC ) + kr, mask, a_reg[1] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 2 ) * KC ) + kr, mask, a_reg[2] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 3 ) * KC ) + kr, mask, a_reg[3] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 4 ) * KC ) + kr, mask, a_reg[4] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 5 ) * KC ) + kr, mask, a_reg[5] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 6 ) * KC ) + kr, mask, a_reg[6] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 7 ) * KC ) + kr, mask, a_reg[7] );


#define MASKED_LOAD_4_ROWS_AVX512( mask ) \
	a_reg[0] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 0 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[1] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 1 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[2] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 2 ) * rs_a ) + ( kr * cs_a )); \
	a_reg[3] = _mm512_maskz_loadu_epi16( mask, a + ( ( ic + 3 ) * rs_a ) + ( kr * cs_a ));

#define MASKED_STORE_4_ROWS_AVX512( mask ) \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr, mask, a_reg[0] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 1 ) * KC ) + kr, mask, a_reg[1] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 2 ) * KC ) + kr, mask, a_reg[2] ); \
	_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 3 ) * KC ) + kr, mask, a_reg[3] );

void packa_mr16_bf16bf16f32of32_row_major
    (
      bfloat16*	      pack_a_buffer,
      const bfloat16* a,
      const dim_t     rs_a,
      const dim_t     cs_a,
      const dim_t      MC,
      const dim_t     KC,
      dim_t*          rs_p,
      dim_t*          cs_p
    );

void packa_mr16_bf16bf16f32of32_col_major
    (
      bfloat16*	      pack_a_buffer,
      const bfloat16* a,
      const dim_t     rs_a,
      const dim_t     cs_a,
      const dim_t     MC,
      const dim_t     KC,
      dim_t*          rs_p,
      dim_t*          cs_p
    );

void packa_mr16_bf16bf16f32of32
    (
      bfloat16*	      pack_a_buffer,
      const bfloat16* a,
      const dim_t     rs_a,
      const dim_t     cs_a,
      const dim_t     MC,
      const dim_t     KC,
      dim_t*          rs_p,
      dim_t*          cs_p
    )
{
	if( cs_a == 1 )
	{
		packa_mr16_bf16bf16f32of32_row_major
		( pack_a_buffer, a, rs_a, cs_a, MC, KC, rs_p, cs_p);
	}
	else
	{
		packa_mr16_bf16bf16f32of32_col_major
		( pack_a_buffer, a, rs_a, cs_a, MC, KC, rs_p, cs_p);
	}
}

void packa_mr16_bf16bf16f32of32_row_major
    (
      bfloat16*	      pack_a_buffer,
      const bfloat16* a,
      const dim_t     rs_a,
      const dim_t     cs_a,
      const dim_t     MC,
      const dim_t     KC,
      dim_t*          rs_p,
      dim_t*          cs_p
    )
{
	dim_t MR = 32;

	__m512i a_reg[32];

	dim_t ic = 0, kr = 0;

	for( ic = 0; ( ic + MR - 1 ) < MC; ic += MR )
	{
		for( kr = 0; ( kr + 32 - 1) < KC; kr += 32 )
		{
			a_reg[0] = _mm512_loadu_si512( a + ( ( ic + 0 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[1] = _mm512_loadu_si512( a + ( ( ic + 1 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[2] = _mm512_loadu_si512( a + ( ( ic + 2 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[3] = _mm512_loadu_si512( a + ( ( ic + 3 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[4] = _mm512_loadu_si512( a + ( ( ic + 4 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[5] = _mm512_loadu_si512( a + ( ( ic + 5 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[6] = _mm512_loadu_si512( a + ( ( ic + 6 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[7] = _mm512_loadu_si512( a + ( ( ic + 7 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[8] = _mm512_loadu_si512( a + ( ( ic + 8 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[9] = _mm512_loadu_si512( a + ( ( ic + 9 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[10] = _mm512_loadu_si512( a + ( ( ic + 10 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[11] = _mm512_loadu_si512( a + ( ( ic + 11 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[12] = _mm512_loadu_si512( a + ( ( ic + 12 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[13] = _mm512_loadu_si512( a + ( ( ic + 13 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[14] = _mm512_loadu_si512( a + ( ( ic + 14 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[15] = _mm512_loadu_si512( a + ( ( ic + 15 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[16] = _mm512_loadu_si512( a + ( ( ic + 16 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[17] = _mm512_loadu_si512( a + ( ( ic + 17 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[18] = _mm512_loadu_si512( a + ( ( ic + 18 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[19] = _mm512_loadu_si512( a + ( ( ic + 19 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[20] = _mm512_loadu_si512( a + ( ( ic + 20 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[21] = _mm512_loadu_si512( a + ( ( ic + 21 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[22] = _mm512_loadu_si512( a + ( ( ic + 22 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[23] = _mm512_loadu_si512( a + ( ( ic + 23 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[24] = _mm512_loadu_si512( a + ( ( ic + 24 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[25] = _mm512_loadu_si512( a + ( ( ic + 25 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[26] = _mm512_loadu_si512( a + ( ( ic + 26 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[27] = _mm512_loadu_si512( a + ( ( ic + 27 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[28] = _mm512_loadu_si512( a + ( ( ic + 28 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[29] = _mm512_loadu_si512( a + ( ( ic + 29 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[30] = _mm512_loadu_si512( a + ( ( ic + 30 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[31] = _mm512_loadu_si512( a + ( ( ic + 31 ) * rs_a ) + ( kr * cs_a ) );


			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr , a_reg[0] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 1 ) * KC ) + kr , a_reg[1] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 2 ) * KC ) + kr , a_reg[2] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 3 ) * KC ) + kr , a_reg[3] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 4 ) * KC ) + kr , a_reg[4] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 5 ) * KC ) + kr , a_reg[5] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 6 ) * KC ) + kr , a_reg[6] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 7 ) * KC ) + kr , a_reg[7] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 8 ) * KC ) + kr , a_reg[8] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 9 ) * KC ) + kr , a_reg[9] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 10 ) * KC ) + kr , a_reg[10] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 11 ) * KC ) + kr , a_reg[11] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 12 ) * KC ) + kr , a_reg[12] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 13 ) * KC ) + kr , a_reg[13] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 14 ) * KC ) + kr , a_reg[14] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 15 ) * KC ) + kr , a_reg[15] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 16 ) * KC ) + kr , a_reg[16] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 17 ) * KC ) + kr , a_reg[17] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 18 ) * KC ) + kr , a_reg[18] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 19 ) * KC ) + kr , a_reg[19] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 20 ) * KC ) + kr , a_reg[20] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 21 ) * KC ) + kr , a_reg[21] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 22 ) * KC ) + kr , a_reg[22] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 23 ) * KC ) + kr , a_reg[23] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 24 ) * KC ) + kr , a_reg[24] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 25 ) * KC ) + kr , a_reg[25] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 26 ) * KC ) + kr , a_reg[26] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 27 ) * KC ) + kr , a_reg[27] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 28 ) * KC ) + kr , a_reg[28] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 29 ) * KC ) + kr , a_reg[29] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 30 ) * KC ) + kr , a_reg[30] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 31 ) * KC ) + kr , a_reg[31] );
		}
		for( ; ( kr + 15 ) < KC; kr += 16 )
		{
			MASKED_LOAD_32_ROWS_AVX512( 0xFFFF )

			MASKED_STORE_32_ROWS_AVX512( 0xFFFF )
		}
		for( ; ( kr + 7 ) < KC; kr += 8 )
		{
			MASKED_LOAD_32_ROWS_AVX512( 0xFF )

			MASKED_STORE_32_ROWS_AVX512( 0xFF )
		}
		for( ; ( kr + 3 ) < KC; kr += 4 )
		{
			MASKED_LOAD_32_ROWS_AVX512( 0xF )

			MASKED_STORE_32_ROWS_AVX512( 0xF )
		}
		for( ; ( kr + 1 ) < KC; kr += 2 )
		{
			MASKED_LOAD_32_ROWS_AVX512( 0x3 )

			MASKED_STORE_32_ROWS_AVX512( 0x3 )
		}
		for( ; ( kr ) < KC; kr += 1 )
		{
			MASKED_LOAD_32_ROWS_AVX512( 0x1 )

			MASKED_STORE_32_ROWS_AVX512( 0x1 )
		}
	}
	for( ; ( ic + 16 - 1 ) < MC; ic += 16 )
	{
		for( kr = 0; ( kr + 32 - 1 ) < KC; kr += 32 )
		{
			a_reg[0] = _mm512_loadu_si512( a + ( ( ic + 0 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[1] = _mm512_loadu_si512( a + ( ( ic + 1 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[2] = _mm512_loadu_si512( a + ( ( ic + 2 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[3] = _mm512_loadu_si512( a + ( ( ic + 3 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[4] = _mm512_loadu_si512( a + ( ( ic + 4 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[5] = _mm512_loadu_si512( a + ( ( ic + 5 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[6] = _mm512_loadu_si512( a + ( ( ic + 6 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[7] = _mm512_loadu_si512( a + ( ( ic + 7 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[8] = _mm512_loadu_si512( a + ( ( ic + 8 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[9] = _mm512_loadu_si512( a + ( ( ic + 9 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[10] = _mm512_loadu_si512( a + ( ( ic + 10 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[11] = _mm512_loadu_si512( a + ( ( ic + 11 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[12] = _mm512_loadu_si512( a + ( ( ic + 12 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[13] = _mm512_loadu_si512( a + ( ( ic + 13 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[14] = _mm512_loadu_si512( a + ( ( ic + 14 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[15] = _mm512_loadu_si512( a + ( ( ic + 15 ) * rs_a ) + ( kr * cs_a ) );

			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr , a_reg[0] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 1 ) * KC ) + kr , a_reg[1] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 2 ) * KC ) + kr , a_reg[2] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 3 ) * KC ) + kr , a_reg[3] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 4 ) * KC ) + kr , a_reg[4] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 5 ) * KC ) + kr , a_reg[5] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 6 ) * KC ) + kr , a_reg[6] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 7 ) * KC ) + kr , a_reg[7] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 8 ) * KC ) + kr , a_reg[8] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 9 ) * KC ) + kr , a_reg[9] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 10 ) * KC ) + kr , a_reg[10] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 11 ) * KC ) + kr , a_reg[11] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 12 ) * KC ) + kr , a_reg[12] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 13 ) * KC ) + kr , a_reg[13] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 14 ) * KC ) + kr , a_reg[14] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 15 ) * KC ) + kr , a_reg[15] );
		}
		for( ; ( kr + 16 - 1 ) < KC; kr += 16 )
		{
			MASKED_LOAD_16_ROWS_AVX512( 0xFFFF )
			MASKED_STORE_16_ROWS_AVX512( 0xFFFF )
		}
		for( ; ( kr + 7 ) < KC; kr += 8 )
		{
			MASKED_LOAD_16_ROWS_AVX512( 0xFF )

			MASKED_STORE_16_ROWS_AVX512( 0xFF )
		}
		for( ; ( kr + 3 ) < KC; kr += 4 )
		{
			MASKED_LOAD_16_ROWS_AVX512( 0xF )

			MASKED_STORE_16_ROWS_AVX512( 0xF )
		}
		for( ; ( kr + 1 ) < KC; kr += 2 )
		{
			MASKED_LOAD_16_ROWS_AVX512( 0x3 )

			MASKED_STORE_16_ROWS_AVX512( 0x3 )
		}
		for( ; ( kr ) < KC; kr += 1 )
		{
			MASKED_LOAD_16_ROWS_AVX512( 0x1 )

			MASKED_STORE_16_ROWS_AVX512( 0x1 )
		}
	}
	for( ; ( ic + 7 - 1 ) < MC; ic += 8 )
	{
		for( kr = 0; ( kr + 32 - 1 ) < KC; kr += 32 )
		{
			a_reg[0] = _mm512_loadu_si512( a + ( ( ic + 0 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[1] = _mm512_loadu_si512( a + ( ( ic + 1 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[2] = _mm512_loadu_si512( a + ( ( ic + 2 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[3] = _mm512_loadu_si512( a + ( ( ic + 3 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[4] = _mm512_loadu_si512( a + ( ( ic + 4 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[5] = _mm512_loadu_si512( a + ( ( ic + 5 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[6] = _mm512_loadu_si512( a + ( ( ic + 6 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[7] = _mm512_loadu_si512( a + ( ( ic + 7 ) * rs_a ) + ( kr * cs_a ) );

			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr , a_reg[0] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 1 ) * KC ) + kr , a_reg[1] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 2 ) * KC ) + kr , a_reg[2] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 3 ) * KC ) + kr , a_reg[3] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 4 ) * KC ) + kr , a_reg[4] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 5 ) * KC ) + kr , a_reg[5] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 6 ) * KC ) + kr , a_reg[6] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 7 ) * KC ) + kr , a_reg[7] );
		}
		for( ; ( kr + 16 - 1 ) < KC; kr += 16 )
		{
			MASKED_LOAD_8_ROWS_AVX512( 0xFFFF )
			MASKED_STORE_8_ROWS_AVX512( 0xFFFF )
		}
		for( ; ( kr + 7 ) < KC; kr += 8 )
		{
			MASKED_LOAD_8_ROWS_AVX512( 0xFF )

			MASKED_STORE_8_ROWS_AVX512( 0xFF )
		}
		for( ; ( kr + 3 ) < KC; kr += 4 )
		{
			MASKED_LOAD_8_ROWS_AVX512( 0xF )

			MASKED_STORE_8_ROWS_AVX512( 0xF )
		}
		for( ; ( kr + 1 ) < KC; kr += 2 )
		{
			MASKED_LOAD_8_ROWS_AVX512( 0x3 )

			MASKED_STORE_8_ROWS_AVX512( 0x3 )
		}
		for( ; ( kr ) < KC; kr += 1 )
		{
			MASKED_LOAD_8_ROWS_AVX512( 0x1 )

			MASKED_STORE_8_ROWS_AVX512( 0x1 )
		}
	}
	for( ; ( ic + 4 - 1 ) < MC; ic += 4 )
	{
		for( kr = 0; ( kr + 32 - 1 ) < KC; kr += 32 )
		{
			a_reg[0] = _mm512_loadu_si512( a + ( ( ic + 0 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[1] = _mm512_loadu_si512( a + ( ( ic + 1 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[2] = _mm512_loadu_si512( a + ( ( ic + 2 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[3] = _mm512_loadu_si512( a + ( ( ic + 3 ) * rs_a ) + ( kr * cs_a ) );

			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr , a_reg[0] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 1 ) * KC ) + kr , a_reg[1] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 2 ) * KC ) + kr , a_reg[2] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 3 ) * KC ) + kr , a_reg[3] );
		}
		for( ; ( kr + 16 - 1 ) < KC; kr += 16 )
		{
			MASKED_LOAD_4_ROWS_AVX512( 0xFFFF )
			MASKED_STORE_4_ROWS_AVX512( 0xFFFF )
		}
		for( ; ( kr + 7 ) < KC; kr += 8 )
		{
			MASKED_LOAD_4_ROWS_AVX512( 0xFF )

			MASKED_STORE_4_ROWS_AVX512( 0xFF )
		}
		for( ; ( kr + 3 ) < KC; kr += 4 )
		{
			MASKED_LOAD_4_ROWS_AVX512( 0xF )

			MASKED_STORE_4_ROWS_AVX512( 0xF )
		}
		for( ; ( kr + 1 ) < KC; kr += 2 )
		{
			MASKED_LOAD_4_ROWS_AVX512( 0x3 )

			MASKED_STORE_4_ROWS_AVX512( 0x3 )
		}
		for( ; ( kr ) < KC; kr += 1 )
		{
			MASKED_LOAD_4_ROWS_AVX512( 0x1 )

			MASKED_STORE_4_ROWS_AVX512( 0x1 )
		}
	}

	for( ; ( ic + 2 - 1 ) < MC; ic += 2 )
	{
		for( kr = 0; ( kr + 32 - 1 ) < KC; kr += 32 )
		{
			a_reg[0] = _mm512_loadu_si512( a + ( ( ic + 0 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[1] = _mm512_loadu_si512( a + ( ( ic + 1 ) * rs_a ) + ( kr * cs_a ) );

			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr , a_reg[0] );
			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 1 ) * KC ) + kr , a_reg[1] );
		}
		for( ; ( kr + 16 - 1 ) < KC; kr += 16 )
		{
			a_reg[0] = _mm512_maskz_loadu_epi16( 0xFFFF, a + ( ( ic + 0 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[1] = _mm512_maskz_loadu_epi16( 0xFFFF, a + ( ( ic + 1 ) * rs_a ) + ( kr * cs_a ) );

			_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr, 0xFFFF, a_reg[0] );
			_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 1 ) * KC ) + kr, 0xFFFF, a_reg[1] );
		}
		for( ; ( kr + 7 ) < KC; kr += 8 )
		{
			a_reg[0] = _mm512_maskz_loadu_epi16( 0xFF, a + ( ( ic + 0 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[1] = _mm512_maskz_loadu_epi16( 0xFF, a + ( ( ic + 1 ) * rs_a ) + ( kr * cs_a ) );

			_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr, 0xFF, a_reg[0] );
			_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 1 ) * KC ) + kr, 0xFF, a_reg[1] );
		}
		for( ; ( kr + 3 ) < KC; kr += 4 )
		{
			a_reg[0] = _mm512_maskz_loadu_epi16( 0xF, a + ( ( ic + 0 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[1] = _mm512_maskz_loadu_epi16( 0xF, a + ( ( ic + 1 ) * rs_a ) + ( kr * cs_a ) );

			_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr, 0xF, a_reg[0] );
			_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 1 ) * KC ) + kr, 0xF, a_reg[1] );
		}
		for( ; ( kr + 1 ) < KC; kr += 2 )
		{
			a_reg[0] = _mm512_maskz_loadu_epi16( 0x3, a + ( ( ic + 0 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[1] = _mm512_maskz_loadu_epi16( 0x3, a + ( ( ic + 1 ) * rs_a ) + ( kr * cs_a ) );

			_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr, 0x3, a_reg[0] );
			_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 1 ) * KC ) + kr, 0x3, a_reg[1] );
		}
		for( ; ( kr ) < KC; kr += 1 )
		{
			a_reg[0] = _mm512_maskz_loadu_epi16( 0x1, a + ( ( ic + 0 ) * rs_a ) + ( kr * cs_a ) );
			a_reg[1] = _mm512_maskz_loadu_epi16( 0x1, a + ( ( ic + 1 ) * rs_a ) + ( kr * cs_a ) );

			_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr, 0x1, a_reg[0] );
			_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 1 ) * KC ) + kr, 0x1, a_reg[1] );
		}
	}
	for( ; ( ic ) < MC; ic += 1 )
	{
		for( kr = 0; ( kr + 32 - 1 ) < KC; kr += 32 )
		{
			a_reg[0] = _mm512_loadu_si512( a + ( ( ic + 0 ) * rs_a ) + ( kr * cs_a ) );

			_mm512_storeu_si512( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr , a_reg[0]);
		}
		for( ; ( kr + 16 - 1 ) < KC; kr += 16 )
		{
			a_reg[0] = _mm512_maskz_loadu_epi16( 0xFFFF, a + ( ( ic + 0 ) * rs_a ) + ( kr * cs_a ) );

			_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr, 0xFFFF, a_reg[0] );
		}
		for( ; ( kr + 7 ) < KC; kr += 8 )
		{
			a_reg[0] = _mm512_maskz_loadu_epi16( 0xFF, a + ( ( ic + 0 ) * rs_a ) + ( kr * cs_a ) );

			_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr, 0xFF, a_reg[0] );
		}
		for( ; ( kr + 3 ) < KC; kr += 4 )
		{
			a_reg[0] = _mm512_maskz_loadu_epi16( 0xF, a + ( ( ic + 0 ) * rs_a ) + ( kr * cs_a ) );

			_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr, 0xF, a_reg[0] );
		}
		for( ; ( kr + 1 ) < KC; kr += 2 )
		{
			a_reg[0] = _mm512_maskz_loadu_epi16( 0x3, a + ( ( ic + 0 ) * rs_a ) + ( kr * cs_a ) );

			_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr, 0x3, a_reg[0] );
		}
		for( ; ( kr ) < KC; kr += 1 )
		{
			a_reg[0] = _mm512_maskz_loadu_epi16( 0x1, a + ( ( ic + 0 ) * rs_a ) + ( kr * cs_a ) );

			_mm512_mask_storeu_epi16( pack_a_buffer + ( ( ic + 0 ) * KC ) + kr, 0x1, a_reg[0] );
		}
	}
	*rs_p = KC;
	*cs_p = 2;

}
void packa_mr16_bf16bf16f32of32_col_major
    (
      bfloat16*	      pack_a_buffer,
      const bfloat16* a,
      const dim_t     rs_a,
      const dim_t     cs_a,
      const dim_t     MC,
      const dim_t     KC,
      dim_t*          rs_p,
      dim_t*          cs_p
    )
{
	dim_t MR = 16;

	dim_t m_left = MC % 4;

	__m256i a_reg[16], b_reg[16];

	dim_t ic, kr;

	for( ic = 0; ( ic + MR - 1 ) < MC; ic += MR)
	{
		for( kr = 0; ( kr + 15 ) < KC; kr += 16)
		{
			a_reg[0] = _mm256_loadu_si256( (__m256i const *) ( a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) ) );
			a_reg[1] = _mm256_loadu_si256( (__m256i const *) ( a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) ) );
			a_reg[2] = _mm256_loadu_si256( (__m256i const *) ( a + ( ic * rs_a ) + ( ( kr + 2 ) * cs_a ) ) );
			a_reg[3] = _mm256_loadu_si256( (__m256i const *) ( a + ( ic * rs_a ) + ( ( kr + 3 ) * cs_a ) ) );
			a_reg[4] = _mm256_loadu_si256( (__m256i const *) ( a + ( ic * rs_a ) + ( ( kr + 4 ) * cs_a ) ) );
			a_reg[5] = _mm256_loadu_si256( (__m256i const *) ( a + ( ic * rs_a ) + ( ( kr + 5 ) * cs_a ) ) );
			a_reg[6] = _mm256_loadu_si256( (__m256i const *) ( a + ( ic * rs_a ) + ( ( kr + 6 ) * cs_a ) ) );
			a_reg[7] = _mm256_loadu_si256( (__m256i const *) ( a + ( ic * rs_a ) + ( ( kr + 7 ) * cs_a ) ) );
			a_reg[8] = _mm256_loadu_si256( (__m256i const *) ( a + ( ic * rs_a ) + ( ( kr + 8 ) * cs_a ) ) );
			a_reg[9] = _mm256_loadu_si256( (__m256i const *) ( a + ( ic * rs_a ) + ( ( kr + 9 ) * cs_a ) ) );
			a_reg[10] = _mm256_loadu_si256( (__m256i const *) ( a + ( ic * rs_a ) + ( ( kr + 10 ) * cs_a ) ) );
			a_reg[11] = _mm256_loadu_si256( (__m256i const *) ( a + ( ic * rs_a ) + ( ( kr + 11 ) * cs_a ) ) );
			a_reg[12] = _mm256_loadu_si256( (__m256i const *) ( a + ( ic * rs_a ) + ( ( kr + 12 ) * cs_a ) ) );
			a_reg[13] = _mm256_loadu_si256( (__m256i const *) ( a + ( ic * rs_a ) + ( ( kr + 13 ) * cs_a ) ) );
			a_reg[14] = _mm256_loadu_si256( (__m256i const *) ( a + ( ic * rs_a ) + ( ( kr + 14 ) * cs_a ) ) );
			a_reg[15] = _mm256_loadu_si256( (__m256i const *) ( a + ( ic * rs_a ) + ( ( kr + 15 ) * cs_a ) ) );

			UNPACKLO_EPI16
			UNPACKHI_EPI16
			UNPACKLO_EPI32
			UNPACKHI_EPI32
			UNPACKLO_EPI64
			UNPACKHI_EPI64
			SHUFFLE_64x2

			_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), a_reg[0] );
			_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 1 ) * KC + kr ), a_reg[4] );
			_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 2 ) * KC + kr ), a_reg[2] );
			_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 3 ) * KC + kr ), a_reg[6] );
			_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 4 ) * KC + kr ), a_reg[8] );
			_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 5 ) * KC + kr ), a_reg[12] );
			_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 6 ) * KC + kr ), a_reg[10] );
			_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 7 ) * KC + kr ), a_reg[14] );
			_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 8 ) * KC + kr ), a_reg[1] );
			_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 9 ) * KC + kr ), a_reg[5] );
			_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 10 ) * KC + kr ), a_reg[3] );
			_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 11 ) * KC + kr ), a_reg[7] );
			_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 12 ) * KC + kr ), a_reg[9] );
			_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 13 ) * KC + kr ), a_reg[13] );
			_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 14 ) * KC + kr ), a_reg[11] );
			_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 15 ) * KC + kr ), a_reg[15] );
		}

		for( ; ( kr + 7 ) < KC; kr += 8)
		{
			a_reg[0] = _mm256_loadu_si256( (__m256i const *)( a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) ) );
			a_reg[1] = _mm256_loadu_si256( (__m256i const *)( a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) ) );
			a_reg[2] = _mm256_loadu_si256( (__m256i const *)( a + ( ic * rs_a ) + ( ( kr + 2 ) * cs_a ) ) );
			a_reg[3] = _mm256_loadu_si256( (__m256i const *)( a + ( ic * rs_a ) + ( ( kr + 3 ) * cs_a ) ) );
			a_reg[4] = _mm256_loadu_si256( (__m256i const *)( a + ( ic * rs_a ) + ( ( kr + 4 ) * cs_a ) ) );
			a_reg[5] = _mm256_loadu_si256( (__m256i const *)( a + ( ic * rs_a ) + ( ( kr + 5 ) * cs_a ) ) );
			a_reg[6] = _mm256_loadu_si256( (__m256i const *)( a + ( ic * rs_a ) + ( ( kr + 6 ) * cs_a ) ) );
			a_reg[7] = _mm256_loadu_si256( (__m256i const *)( a + ( ic * rs_a ) + ( ( kr + 7 ) * cs_a ) ) );
			a_reg[8] = _mm256_setzero_si256();
			a_reg[9] = _mm256_setzero_si256();
			a_reg[10] = _mm256_setzero_si256();
			a_reg[11] = _mm256_setzero_si256();
			a_reg[12] = _mm256_setzero_si256();
			a_reg[13] = _mm256_setzero_si256();
			a_reg[14] = _mm256_setzero_si256();
			a_reg[15] = _mm256_setzero_si256();

			UNPACKLO_EPI16
			UNPACKHI_EPI16
			UNPACKLO_EPI32
			UNPACKHI_EPI32
			UNPACKLO_EPI64
			UNPACKHI_EPI64
			SHUFFLE_64x2
			MASKED_STORE_EPI64(0x03)

		}
		for( ; ( kr + 3 ) < KC; kr += 4)
		{
			a_reg[0] = _mm256_loadu_si256( (__m256i const *)( a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) ) );
			a_reg[1] = _mm256_loadu_si256( (__m256i const *)( a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) ) );
			a_reg[2] = _mm256_loadu_si256( (__m256i const *)( a + ( ic * rs_a ) + ( ( kr + 2 ) * cs_a ) ) );
			a_reg[3] = _mm256_loadu_si256( (__m256i const *)( a + ( ic * rs_a ) + ( ( kr + 3 ) * cs_a ) ) );
			a_reg[4] = _mm256_setzero_si256();
			a_reg[5] = _mm256_setzero_si256();
			a_reg[6] = _mm256_setzero_si256();
			a_reg[7] = _mm256_setzero_si256();
			a_reg[8] = _mm256_setzero_si256();
			a_reg[9] = _mm256_setzero_si256();
			a_reg[10] = _mm256_setzero_si256();
			a_reg[11] = _mm256_setzero_si256();
			a_reg[12] = _mm256_setzero_si256();
			a_reg[13] = _mm256_setzero_si256();
			a_reg[14] = _mm256_setzero_si256();
			a_reg[15] = _mm256_setzero_si256();

			UNPACKLO_EPI16
			UNPACKHI_EPI16
			UNPACKLO_EPI32
			UNPACKHI_EPI32
			UNPACKLO_EPI64
			UNPACKHI_EPI64
			SHUFFLE_64x2
			MASKED_STORE_EPI64(0x01)
		}
		for( ; ( kr + 1 ) < KC; kr += 2)
		{
			a_reg[0] = _mm256_loadu_si256( (__m256i const *)( a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) ) );
			a_reg[1] = _mm256_loadu_si256( (__m256i const *)( a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) ) );
			a_reg[2] = _mm256_setzero_si256();
			a_reg[3] = _mm256_setzero_si256();
			a_reg[4] = _mm256_setzero_si256();
			a_reg[5] = _mm256_setzero_si256();
			a_reg[6] = _mm256_setzero_si256();
			a_reg[7] = _mm256_setzero_si256();
			a_reg[8] = _mm256_setzero_si256();
			a_reg[9] = _mm256_setzero_si256();
			a_reg[10] = _mm256_setzero_si256();
			a_reg[11] = _mm256_setzero_si256();
			a_reg[12] = _mm256_setzero_si256();
			a_reg[13] = _mm256_setzero_si256();
			a_reg[14] = _mm256_setzero_si256();
			a_reg[15] = _mm256_setzero_si256();

			UNPACKLO_EPI16
			UNPACKHI_EPI16
			UNPACKLO_EPI32
			UNPACKHI_EPI32
			UNPACKLO_EPI64
			UNPACKHI_EPI64
			SHUFFLE_64x2
			MASKED_STORE_EPI32(0x01)
		}
		for( ; ( kr ) < KC; kr += 1)
		{
			a_reg[0] = _mm256_loadu_si256( (__m256i const *)(a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) ) );
			a_reg[1] = _mm256_setzero_si256();
			a_reg[2] = _mm256_setzero_si256();
			a_reg[3] = _mm256_setzero_si256();
			a_reg[4] = _mm256_setzero_si256();
			a_reg[5] = _mm256_setzero_si256();
			a_reg[6] = _mm256_setzero_si256();
			a_reg[7] = _mm256_setzero_si256();
			a_reg[8] = _mm256_setzero_si256();
			a_reg[9] = _mm256_setzero_si256();
			a_reg[10] = _mm256_setzero_si256();
			a_reg[11] = _mm256_setzero_si256();
			a_reg[12] = _mm256_setzero_si256();
			a_reg[13] = _mm256_setzero_si256();
			a_reg[14] = _mm256_setzero_si256();
			a_reg[15] = _mm256_setzero_si256();

			UNPACKLO_EPI16
			UNPACKHI_EPI16
			UNPACKLO_EPI32
			UNPACKHI_EPI32
			UNPACKLO_EPI64
			UNPACKHI_EPI64
			SHUFFLE_64x2
			MASKED_STORE_EPI16(0x01)
		}
	}

	for( ; ( ic + 8 - 1) < MC; ic += 8)
	{
		for( kr = 0; ( kr + 15 ) < KC; kr += 16)
		{
			a_reg[0] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) );
			a_reg[1] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) );
			a_reg[2] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 2 ) * cs_a ) );
			a_reg[3] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 3 ) * cs_a ) );
			a_reg[4] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 4 ) * cs_a ) );
			a_reg[5] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 5 ) * cs_a ) );
			a_reg[6] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 6 ) * cs_a ) );
			a_reg[7] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 7 ) * cs_a ) );
			a_reg[8] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 8 ) * cs_a ) );
			a_reg[9] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 9 ) * cs_a ) );
			a_reg[10] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 10 ) * cs_a ) );
			a_reg[11] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 11 ) * cs_a ) );
			a_reg[12] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 12 ) * cs_a ) );
			a_reg[13] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 13 ) * cs_a ) );
			a_reg[14] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 14 ) * cs_a ) );
			a_reg[15] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 15 ) * cs_a ) );

			UNPACKLO_EPI16
			UNPACKHI_EPI16
			UNPACKLO_EPI32
			UNPACKHI_EPI32
			UNPACKLO_EPI64
			UNPACKHI_EPI64
			SHUFFLE_64x2

			_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), a_reg[0] );
			_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 1 ) * KC + kr ), a_reg[4] );
			_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 2 ) * KC + kr ), a_reg[2] );
			_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 3 ) * KC + kr ), a_reg[6] );
			_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 4 ) * KC + kr ), a_reg[8] );
			_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 5 ) * KC + kr ), a_reg[12] );
			_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 6 ) * KC + kr ), a_reg[10] );
			_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 7 ) * KC + kr ), a_reg[14] );
			_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 8 ) * KC + kr ), a_reg[1] );
		}

		for( ; ( kr + 7 ) < KC; kr += 8)
		{
			a_reg[0] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) );
			a_reg[1] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) );
			a_reg[2] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 2 ) * cs_a ) );
			a_reg[3] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 3 ) * cs_a ) );
			a_reg[4] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 4 ) * cs_a ) );
			a_reg[5] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 5 ) * cs_a ) );
			a_reg[6] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 6 ) * cs_a ) );
			a_reg[7] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 7 ) * cs_a ) );
			a_reg[8] = _mm256_setzero_si256();
			a_reg[9] = _mm256_setzero_si256();
			a_reg[10] = _mm256_setzero_si256();
			a_reg[11] = _mm256_setzero_si256();
			a_reg[12] = _mm256_setzero_si256();
			a_reg[13] = _mm256_setzero_si256();
			a_reg[14] = _mm256_setzero_si256();
			a_reg[15] = _mm256_setzero_si256();
			UNPACKLO_EPI16
			UNPACKHI_EPI16
			UNPACKLO_EPI32
			UNPACKHI_EPI32
			UNPACKLO_EPI64
			UNPACKHI_EPI64
			SHUFFLE_64x2

			_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x03, a_reg[0] );
			_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0x03, a_reg[4] );
			_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 2 ) * KC + kr ), 0x03, a_reg[2] );
			_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 3 ) * KC + kr ), 0x03, a_reg[6] );
			_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 4 ) * KC + kr ), 0x03, a_reg[8] );
			_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 5 ) * KC + kr ), 0x03, a_reg[12] );
			_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 6 ) * KC + kr ), 0x03, a_reg[10] );
			_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 7 ) * KC + kr ), 0x03, a_reg[14] );
		}
		for( ; ( kr + 3 ) < KC; kr += 4)
		{
			a_reg[0] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) );
			a_reg[1] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) );
			a_reg[2] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 2 ) * cs_a ) );
			a_reg[3] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 3 ) * cs_a ) );
			a_reg[4] = _mm256_setzero_si256();
			a_reg[5] = _mm256_setzero_si256();
			a_reg[6] = _mm256_setzero_si256();
			a_reg[7] = _mm256_setzero_si256();
			a_reg[8] = _mm256_setzero_si256();
			a_reg[9] = _mm256_setzero_si256();
			a_reg[10] = _mm256_setzero_si256();
			a_reg[11] = _mm256_setzero_si256();
			a_reg[12] = _mm256_setzero_si256();
			a_reg[13] = _mm256_setzero_si256();
			a_reg[14] = _mm256_setzero_si256();
			a_reg[15] = _mm256_setzero_si256();

			UNPACKLO_EPI16
			UNPACKHI_EPI16
			UNPACKLO_EPI32
			UNPACKHI_EPI32
			UNPACKLO_EPI64
			UNPACKHI_EPI64
			SHUFFLE_64x2
			_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x01, a_reg[0] );
			_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0x01, a_reg[4] );
			_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 2 ) * KC + kr ), 0x01, a_reg[2] );
			_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 3 ) * KC + kr ), 0x01, a_reg[6] );
			_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 4 ) * KC + kr ), 0x01, a_reg[8] );
			_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 5 ) * KC + kr ), 0x01, a_reg[12] );
			_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 6 ) * KC + kr ), 0x01, a_reg[10] );
			_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 7 ) * KC + kr ), 0x01, a_reg[14] );
		}
		for( ; ( kr + 1 ) < KC; kr += 2)
		{
			a_reg[0] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) );
			a_reg[1] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) );
			a_reg[2] = _mm256_setzero_si256();
			a_reg[3] = _mm256_setzero_si256();
			a_reg[4] = _mm256_setzero_si256();
			a_reg[5] = _mm256_setzero_si256();
			a_reg[6] = _mm256_setzero_si256();
			a_reg[7] = _mm256_setzero_si256();
			a_reg[8] = _mm256_setzero_si256();
			a_reg[9] = _mm256_setzero_si256();
			a_reg[10] = _mm256_setzero_si256();
			a_reg[11] = _mm256_setzero_si256();
			a_reg[12] = _mm256_setzero_si256();
			a_reg[13] = _mm256_setzero_si256();
			a_reg[14] = _mm256_setzero_si256();
			a_reg[15] = _mm256_setzero_si256();

			UNPACKLO_EPI16
			UNPACKHI_EPI16
			UNPACKLO_EPI32
			UNPACKHI_EPI32
			UNPACKLO_EPI64
			UNPACKHI_EPI64
			SHUFFLE_64x2
			_mm256_mask_storeu_epi32( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x01, a_reg[0] );
			_mm256_mask_storeu_epi32( ( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0x01, a_reg[4] );
			_mm256_mask_storeu_epi32( ( pack_a_buffer + ( ic + 2 ) * KC + kr ), 0x01, a_reg[2] );
			_mm256_mask_storeu_epi32( ( pack_a_buffer + ( ic + 3 ) * KC + kr ), 0x01, a_reg[6] );
			_mm256_mask_storeu_epi32( ( pack_a_buffer + ( ic + 4 ) * KC + kr ), 0x01, a_reg[8] );
			_mm256_mask_storeu_epi32( ( pack_a_buffer + ( ic + 5 ) * KC + kr ), 0x01, a_reg[12] );
			_mm256_mask_storeu_epi32( ( pack_a_buffer + ( ic + 6 ) * KC + kr ), 0x01, a_reg[10] );
			_mm256_mask_storeu_epi32( ( pack_a_buffer + ( ic + 7 ) * KC + kr ), 0x01, a_reg[14] );
		}
		for( ; ( kr ) < KC; kr += 1)
		{
			a_reg[0] = _mm256_maskz_loadu_epi16( 0xFF, a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) );
			a_reg[1] = _mm256_setzero_si256();
			a_reg[2] = _mm256_setzero_si256();
			a_reg[3] = _mm256_setzero_si256();
			a_reg[4] = _mm256_setzero_si256();
			a_reg[5] = _mm256_setzero_si256();
			a_reg[6] = _mm256_setzero_si256();
			a_reg[7] = _mm256_setzero_si256();
			a_reg[8] = _mm256_setzero_si256();
			a_reg[9] = _mm256_setzero_si256();
			a_reg[10] = _mm256_setzero_si256();
			a_reg[11] = _mm256_setzero_si256();
			a_reg[12] = _mm256_setzero_si256();
			a_reg[13] = _mm256_setzero_si256();
			a_reg[14] = _mm256_setzero_si256();
			a_reg[15] = _mm256_setzero_si256();

			UNPACKLO_EPI16
			UNPACKHI_EPI16
			UNPACKLO_EPI32
			UNPACKHI_EPI32
			UNPACKLO_EPI64
			UNPACKHI_EPI64
			SHUFFLE_64x2
			_mm256_mask_storeu_epi16( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x01, a_reg[0] );
			_mm256_mask_storeu_epi16( ( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0x01, a_reg[4] );
			_mm256_mask_storeu_epi16( ( pack_a_buffer + ( ic + 2 ) * KC + kr ), 0x01, a_reg[2] );
			_mm256_mask_storeu_epi16( ( pack_a_buffer + ( ic + 3 ) * KC + kr ), 0x01, a_reg[6] );
			_mm256_mask_storeu_epi16( ( pack_a_buffer + ( ic + 4 ) * KC + kr ), 0x01, a_reg[8] );
			_mm256_mask_storeu_epi16( ( pack_a_buffer + ( ic + 5 ) * KC + kr ), 0x01, a_reg[12] );
			_mm256_mask_storeu_epi16( ( pack_a_buffer + ( ic + 6 ) * KC + kr ), 0x01, a_reg[10] );
			_mm256_mask_storeu_epi16( ( pack_a_buffer + ( ic + 7 ) * KC + kr ), 0x01, a_reg[14] );
		}
	}

	for( ; ( ic + 4 - 1 ) < MC; ic += 4)
	{
		for( kr = 0; ( kr + 15 ) < KC; kr += 16)
		{
			a_reg[0] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) );
			a_reg[1] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) );
			a_reg[2] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 2 ) * cs_a ) );
			a_reg[3] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 3 ) * cs_a ) );
			a_reg[4] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 4 ) * cs_a ) );
			a_reg[5] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 5 ) * cs_a ) );
			a_reg[6] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 6 ) * cs_a ) );
			a_reg[7] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 7 ) * cs_a ) );
			a_reg[8] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 8 ) * cs_a ) );
			a_reg[9] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 9 ) * cs_a ) );
			a_reg[10] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 10 ) * cs_a ) );
			a_reg[11] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 11 ) * cs_a ) );
			a_reg[12] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 12 ) * cs_a ) );
			a_reg[13] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 13 ) * cs_a ) );
			a_reg[14] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 14 ) * cs_a ) );
			a_reg[15] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 15 ) * cs_a ) );

			UNPACKLO_EPI16
			UNPACKHI_EPI16
			UNPACKLO_EPI32
			UNPACKHI_EPI32
			UNPACKLO_EPI64
			UNPACKHI_EPI64
			SHUFFLE_64x2

			_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), a_reg[0] );
			_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 1 ) * KC + kr ), a_reg[4] );
			_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 2 ) * KC + kr ), a_reg[2] );
			_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 3 ) * KC + kr ), a_reg[6] );
		}

		for( ; ( kr + 7 ) < KC; kr += 8)
		{
			a_reg[0] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) );
			a_reg[1] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) );
			a_reg[2] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 2 ) * cs_a ) );
			a_reg[3] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 3 ) * cs_a ) );
			a_reg[4] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 4 ) * cs_a ) );
			a_reg[5] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 5 ) * cs_a ) );
			a_reg[6] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 6 ) * cs_a ) );
			a_reg[7] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 7 ) * cs_a ) );
			a_reg[8] = _mm256_setzero_si256();
			a_reg[9] = _mm256_setzero_si256();
			a_reg[10] = _mm256_setzero_si256();
			a_reg[11] = _mm256_setzero_si256();
			a_reg[12] = _mm256_setzero_si256();
			a_reg[13] = _mm256_setzero_si256();
			a_reg[14] = _mm256_setzero_si256();
			a_reg[15] = _mm256_setzero_si256();
			UNPACKLO_EPI16
			UNPACKHI_EPI16
			UNPACKLO_EPI32
			UNPACKHI_EPI32
			UNPACKLO_EPI64
			UNPACKHI_EPI64
			SHUFFLE_64x2

			_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x03, a_reg[0] );
			_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0x03, a_reg[4] );
			_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 2 ) * KC + kr ), 0x03, a_reg[2] );
			_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 3 ) * KC + kr ), 0x03, a_reg[6] );
		}
		for( ; ( kr + 3 ) < KC; kr += 4)
		{
			a_reg[0] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) );
			a_reg[1] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) );
			a_reg[2] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 2 ) * cs_a ) );
			a_reg[3] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 3 ) * cs_a ) );
			a_reg[4] = _mm256_setzero_si256();
			a_reg[5] = _mm256_setzero_si256();
			a_reg[6] = _mm256_setzero_si256();
			a_reg[7] = _mm256_setzero_si256();
			a_reg[8] = _mm256_setzero_si256();
			a_reg[9] = _mm256_setzero_si256();
			a_reg[10] = _mm256_setzero_si256();
			a_reg[11] = _mm256_setzero_si256();
			a_reg[12] = _mm256_setzero_si256();
			a_reg[13] = _mm256_setzero_si256();
			a_reg[14] = _mm256_setzero_si256();
			a_reg[15] = _mm256_setzero_si256();

			UNPACKLO_EPI16
			UNPACKHI_EPI16
			UNPACKLO_EPI32
			UNPACKHI_EPI32
			UNPACKLO_EPI64
			UNPACKHI_EPI64
			SHUFFLE_64x2
			_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x01, a_reg[0] );
			_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0x01, a_reg[4] );
			_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 2 ) * KC + kr ), 0x01, a_reg[2] );
			_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 3 ) * KC + kr ), 0x01, a_reg[6] );
		}
		for( ; ( kr + 1 ) < KC; kr += 2)
		{
			a_reg[0] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) );
			a_reg[1] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) );
			a_reg[2] = _mm256_setzero_si256();
			a_reg[3] = _mm256_setzero_si256();
			a_reg[4] = _mm256_setzero_si256();
			a_reg[5] = _mm256_setzero_si256();
			a_reg[6] = _mm256_setzero_si256();
			a_reg[7] = _mm256_setzero_si256();
			a_reg[8] = _mm256_setzero_si256();
			a_reg[9] = _mm256_setzero_si256();
			a_reg[10] = _mm256_setzero_si256();
			a_reg[11] = _mm256_setzero_si256();
			a_reg[12] = _mm256_setzero_si256();
			a_reg[13] = _mm256_setzero_si256();
			a_reg[14] = _mm256_setzero_si256();
			a_reg[15] = _mm256_setzero_si256();

			UNPACKLO_EPI16
			UNPACKHI_EPI16
			UNPACKLO_EPI32
			UNPACKHI_EPI32
			UNPACKLO_EPI64
			UNPACKHI_EPI64
			SHUFFLE_64x2
			_mm256_mask_storeu_epi32( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x01, a_reg[0] );
			_mm256_mask_storeu_epi32( ( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0x01, a_reg[4] );
			_mm256_mask_storeu_epi32( ( pack_a_buffer + ( ic + 2 ) * KC + kr ), 0x01, a_reg[2] );
			_mm256_mask_storeu_epi32( ( pack_a_buffer + ( ic + 3 ) * KC + kr ), 0x01, a_reg[6] );
		}
		for( ; ( kr ) < KC; kr += 1)
		{
			a_reg[0] = _mm256_maskz_loadu_epi16( 0x0F, a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) );
			a_reg[1] = _mm256_setzero_si256();
			a_reg[2] = _mm256_setzero_si256();
			a_reg[3] = _mm256_setzero_si256();
			a_reg[4] = _mm256_setzero_si256();
			a_reg[5] = _mm256_setzero_si256();
			a_reg[6] = _mm256_setzero_si256();
			a_reg[7] = _mm256_setzero_si256();
			a_reg[8] = _mm256_setzero_si256();
			a_reg[9] = _mm256_setzero_si256();
			a_reg[10] = _mm256_setzero_si256();
			a_reg[11] = _mm256_setzero_si256();
			a_reg[12] = _mm256_setzero_si256();
			a_reg[13] = _mm256_setzero_si256();
			a_reg[14] = _mm256_setzero_si256();
			a_reg[15] = _mm256_setzero_si256();

			UNPACKLO_EPI16
			UNPACKHI_EPI16
			UNPACKLO_EPI32
			UNPACKHI_EPI32
			UNPACKLO_EPI64
			UNPACKHI_EPI64
			SHUFFLE_64x2
			_mm256_mask_storeu_epi16( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x01, a_reg[0] );
			_mm256_mask_storeu_epi16( ( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0x01, a_reg[4] );
			_mm256_mask_storeu_epi16( ( pack_a_buffer + ( ic + 2 ) * KC + kr ), 0x01, a_reg[2] );
			_mm256_mask_storeu_epi16( ( pack_a_buffer + ( ic + 3 ) * KC + kr ), 0x01, a_reg[6] );
		}
	}

	if( m_left )
	{
		__mmask16 mask = 0xFFFF >> ( 16 - m_left );
		for( kr = 0; ( kr + 15 ) < KC; kr += 16)
		{
			a_reg[0] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) );
			a_reg[1] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) );
			a_reg[2] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 2 ) * cs_a ) );
			a_reg[3] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 3 ) * cs_a ) );
			a_reg[4] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 4 ) * cs_a ) );
			a_reg[5] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 5 ) * cs_a ) );
			a_reg[6] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 6 ) * cs_a ) );
			a_reg[7] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 7 ) * cs_a ) );
			a_reg[8] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 8 ) * cs_a ) );
			a_reg[9] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 9 ) * cs_a ) );
			a_reg[10] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 10 ) * cs_a ) );
			a_reg[11] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 11 ) * cs_a ) );
			a_reg[12] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 12 ) * cs_a ) );
			a_reg[13] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 13 ) * cs_a ) );
			a_reg[14] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 14 ) * cs_a ) );
			a_reg[15] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 15 ) * cs_a ) );

			UNPACKLO_EPI16
			UNPACKHI_EPI16
			UNPACKLO_EPI32
			UNPACKHI_EPI32
			UNPACKLO_EPI64
			UNPACKHI_EPI64
			SHUFFLE_64x2

			switch( m_left )
			{
				case 3:
					_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), a_reg[0] );
					_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 1 ) * KC + kr ), a_reg[4] );
					_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 2 ) * KC + kr ), a_reg[2] );
					break;
				case 2:
					_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), a_reg[0] );
					_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 1 ) * KC + kr ), a_reg[4] );
					break;
				case 1:
					_mm256_storeu_si256( (__m256i *)( pack_a_buffer + ( ic + 0 ) * KC + kr ), a_reg[0] );
					break;
			}
		}

		for( ; ( kr + 7 ) < KC; kr += 8)
		{
			a_reg[0] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) );
			a_reg[1] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) );
			a_reg[2] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 2 ) * cs_a ) );
			a_reg[3] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 3 ) * cs_a ) );
			a_reg[4] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 4 ) * cs_a ) );
			a_reg[5] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 5 ) * cs_a ) );
			a_reg[6] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 6 ) * cs_a ) );
			a_reg[7] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 7 ) * cs_a ) );
			a_reg[8] = _mm256_setzero_si256();
			a_reg[9] = _mm256_setzero_si256();
			a_reg[10] = _mm256_setzero_si256();
			a_reg[11] = _mm256_setzero_si256();
			a_reg[12] = _mm256_setzero_si256();
			a_reg[13] = _mm256_setzero_si256();
			a_reg[14] = _mm256_setzero_si256();
			a_reg[15] = _mm256_setzero_si256();
			UNPACKLO_EPI16
			UNPACKHI_EPI16
			UNPACKLO_EPI32
			UNPACKHI_EPI32
			UNPACKLO_EPI64
			UNPACKHI_EPI64
			SHUFFLE_64x2

			switch( m_left )
			{
				case 3:
					_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x03, a_reg[0] );
					_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0x03, a_reg[4] );
					_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 2 ) * KC + kr ), 0x03, a_reg[2] );
					break;
				case 2:
					_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x03, a_reg[0] );
					_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0x03, a_reg[4] );
					break;
				case 1:
					_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x03, a_reg[0] );
					break;
			}
		}
		for( ; ( kr + 3 ) < KC; kr += 4)
		{
			a_reg[0] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) );
			a_reg[1] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) );
			a_reg[2] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 2 ) * cs_a ) );
			a_reg[3] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 3 ) * cs_a ) );
			a_reg[4] = _mm256_setzero_si256();
			a_reg[5] = _mm256_setzero_si256();
			a_reg[6] = _mm256_setzero_si256();
			a_reg[7] = _mm256_setzero_si256();
			a_reg[8] = _mm256_setzero_si256();
			a_reg[9] = _mm256_setzero_si256();
			a_reg[10] = _mm256_setzero_si256();
			a_reg[11] = _mm256_setzero_si256();
			a_reg[12] = _mm256_setzero_si256();
			a_reg[13] = _mm256_setzero_si256();
			a_reg[14] = _mm256_setzero_si256();
			a_reg[15] = _mm256_setzero_si256();

			UNPACKLO_EPI16
			UNPACKHI_EPI16
			UNPACKLO_EPI32
			UNPACKHI_EPI32
			UNPACKLO_EPI64
			UNPACKHI_EPI64
			SHUFFLE_64x2

			switch( m_left )
			{
				case 3:
					_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x01, a_reg[0] );
					_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0x01, a_reg[4] );
					_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 2 ) * KC + kr ), 0x01, a_reg[2] );
					break;
				case 2:
					_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x01, a_reg[0]);
					_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0x01, a_reg[4]);
					break;
				case 1:
					_mm256_mask_storeu_epi64( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x01, a_reg[0]);
					break;
			}
		}
		for( ; ( kr + 1 ) < KC; kr += 2)
		{
			a_reg[0] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) );
			a_reg[1] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 1 ) * cs_a ) );
			a_reg[2] = _mm256_setzero_si256();
			a_reg[3] = _mm256_setzero_si256();
			a_reg[4] = _mm256_setzero_si256();
			a_reg[5] = _mm256_setzero_si256();
			a_reg[6] = _mm256_setzero_si256();
			a_reg[7] = _mm256_setzero_si256();
			a_reg[8] = _mm256_setzero_si256();
			a_reg[9] = _mm256_setzero_si256();
			a_reg[10] = _mm256_setzero_si256();
			a_reg[11] = _mm256_setzero_si256();
			a_reg[12] = _mm256_setzero_si256();
			a_reg[13] = _mm256_setzero_si256();
			a_reg[14] = _mm256_setzero_si256();
			a_reg[15] = _mm256_setzero_si256();

			UNPACKLO_EPI16
			UNPACKHI_EPI16
			UNPACKLO_EPI32
			UNPACKHI_EPI32
			UNPACKLO_EPI64
			UNPACKHI_EPI64
			SHUFFLE_64x2
			switch( m_left )
			{
				case 3:
					_mm256_mask_storeu_epi32( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x01, a_reg[0] );
					_mm256_mask_storeu_epi32( ( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0x01, a_reg[4] );
					_mm256_mask_storeu_epi32( ( pack_a_buffer + ( ic + 2 ) * KC + kr ), 0x01, a_reg[2] );
					break;
				case 2:
					_mm256_mask_storeu_epi32( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x01, a_reg[0] );
					_mm256_mask_storeu_epi32( ( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0x01, a_reg[4] );
					break;
				case 1:
					_mm256_mask_storeu_epi32(  (pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x01, a_reg[0] );
					break;
			}
		}
		for( ; ( kr ) < KC; kr += 1)
		{
			a_reg[0] = _mm256_maskz_loadu_epi16( mask, a + ( ic * rs_a ) + ( ( kr + 0 ) * cs_a ) );
			a_reg[1] = _mm256_setzero_si256();
			a_reg[2] = _mm256_setzero_si256();
			a_reg[3] = _mm256_setzero_si256();
			a_reg[4] = _mm256_setzero_si256();
			a_reg[5] = _mm256_setzero_si256();
			a_reg[6] = _mm256_setzero_si256();
			a_reg[7] = _mm256_setzero_si256();
			a_reg[8] = _mm256_setzero_si256();
			a_reg[9] = _mm256_setzero_si256();
			a_reg[10] = _mm256_setzero_si256();
			a_reg[11] = _mm256_setzero_si256();
			a_reg[12] = _mm256_setzero_si256();
			a_reg[13] = _mm256_setzero_si256();
			a_reg[14] = _mm256_setzero_si256();
			a_reg[15] = _mm256_setzero_si256();

			UNPACKLO_EPI16
			UNPACKHI_EPI16
			UNPACKLO_EPI32
			UNPACKHI_EPI32
			UNPACKLO_EPI64
			UNPACKHI_EPI64
			SHUFFLE_64x2
			switch( m_left )
			{
				case 3:
					_mm256_mask_storeu_epi16( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x01, a_reg[0] );
					_mm256_mask_storeu_epi16( ( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0x01, a_reg[4] );
					_mm256_mask_storeu_epi16( ( pack_a_buffer + ( ic + 2 ) * KC + kr ), 0x01, a_reg[2] );
					break;
				case 2:
					_mm256_mask_storeu_epi16( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x01, a_reg[0] );
					_mm256_mask_storeu_epi16( ( pack_a_buffer + ( ic + 1 ) * KC + kr ), 0x01, a_reg[4] );
					break;
				case 1:
					_mm256_mask_storeu_epi16( ( pack_a_buffer + ( ic + 0 ) * KC + kr ), 0x01, a_reg[0] );
					break;
			}
		}
	}

	*rs_p = KC;
	*cs_p = 2;
}
#endif
