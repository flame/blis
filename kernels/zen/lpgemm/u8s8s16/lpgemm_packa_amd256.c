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

#include <immintrin.h>
#include "blis.h"

#ifdef BLIS_ADDON_LPGEMM

void packa_mr16_u8s8s16o16_col_major
    (
      uint8_t*	      pack_a_buffer_u8s8s16o16,
      const uint8_t*  a,
      const dim_t     rs,
      const dim_t     cs,
      const dim_t     MC,
      const dim_t     KC,
      dim_t*          rs_a,
      dim_t*          cs_a
    );

void packa_u8s8s16os16
     (
       uint8_t*       pack_a_buffer_u8s8s16o16,
       const uint8_t* a,
       const dim_t    rs,
       const dim_t    cs,
       const dim_t    MC,
       const dim_t    KC,
       dim_t*         rs_a,
       dim_t*         cs_a
     )
{
	if( ( cs == 1 ) && ( MC != 1 ) )
	{
		// Not yet supported
	}
	else
	{
		packa_mr16_u8s8s16o16_col_major
		( pack_a_buffer_u8s8s16o16, a, rs, cs, MC, KC, rs_a, cs_a );
	}
}

#define SET_REGISTERS_ZERO \
	a_reg[0] = _mm_setzero_si128(); \
	a_reg[1] = _mm_setzero_si128(); \
	a_reg[2] = _mm_setzero_si128(); \
	a_reg[3] = _mm_setzero_si128(); \
	a_reg[4] = _mm_setzero_si128(); \
	a_reg[5] = _mm_setzero_si128(); \
	a_reg[6] = _mm_setzero_si128(); \
	a_reg[7] = _mm_setzero_si128(); \
	a_reg[8] = _mm_setzero_si128(); \
	a_reg[9] = _mm_setzero_si128(); \
	a_reg[10] = _mm_setzero_si128(); \
	a_reg[11] = _mm_setzero_si128(); \
	a_reg[12] = _mm_setzero_si128(); \
	a_reg[13] = _mm_setzero_si128(); \
	a_reg[14] = _mm_setzero_si128(); \
	a_reg[15] = _mm_setzero_si128();

#define UNPACKLOW_EPI8 \
	b_reg[0] = _mm_unpacklo_epi8( a_reg[0], a_reg[1] ); \
	b_reg[1] = _mm_unpacklo_epi8( a_reg[2], a_reg[3] ); \
	b_reg[2] = _mm_unpacklo_epi8( a_reg[4], a_reg[5] ); \
	b_reg[3] = _mm_unpacklo_epi8( a_reg[6], a_reg[7] ); \
	b_reg[4] = _mm_unpacklo_epi8( a_reg[8], a_reg[9] ); \
	b_reg[5] = _mm_unpacklo_epi8( a_reg[10], a_reg[11] ); \
	b_reg[6] = _mm_unpacklo_epi8( a_reg[12], a_reg[13] ); \
	b_reg[7] = _mm_unpacklo_epi8( a_reg[14], a_reg[15] );

#define UNPACKHI_EPI8 \
	b_reg[8] = _mm_unpackhi_epi8( a_reg[0], a_reg[1] ); \
	b_reg[9] = _mm_unpackhi_epi8( a_reg[2], a_reg[3] ); \
	b_reg[10] = _mm_unpackhi_epi8( a_reg[4], a_reg[5] ); \
	b_reg[11] = _mm_unpackhi_epi8( a_reg[6], a_reg[7] ); \
	b_reg[12] = _mm_unpackhi_epi8( a_reg[8], a_reg[9] ); \
	b_reg[13] = _mm_unpackhi_epi8( a_reg[10], a_reg[11] ); \
	b_reg[14] = _mm_unpackhi_epi8( a_reg[12], a_reg[13] ); \
	b_reg[15] = _mm_unpackhi_epi8( a_reg[14], a_reg[15] );

#define UNPACKLOW_EPI16 \
	a_reg[0] = _mm_unpacklo_epi16( b_reg[0], b_reg[1] ); \
	a_reg[1] = _mm_unpacklo_epi16( b_reg[2], b_reg[3] ); \
	a_reg[2] = _mm_unpacklo_epi16( b_reg[4], b_reg[5] ); \
	a_reg[3] = _mm_unpacklo_epi16( b_reg[6], b_reg[7] ); \
\
	a_reg[8] = _mm_unpacklo_epi16( b_reg[8], b_reg[9] ); \
	a_reg[9] = _mm_unpacklo_epi16( b_reg[10], b_reg[11] ); \
	a_reg[10] = _mm_unpacklo_epi16( b_reg[12], b_reg[13] ); \
	a_reg[11] = _mm_unpacklo_epi16( b_reg[14], b_reg[15] );

#define UNPACKHI_EPI16 \
	a_reg[4] = _mm_unpackhi_epi16( b_reg[0], b_reg[1] ); \
	a_reg[5] = _mm_unpackhi_epi16( b_reg[2], b_reg[3] ); \
	a_reg[6] = _mm_unpackhi_epi16( b_reg[4], b_reg[5] ); \
	a_reg[7] = _mm_unpackhi_epi16( b_reg[6], b_reg[7] ); \
\
	a_reg[12] = _mm_unpackhi_epi16( b_reg[8], b_reg[9] ); \
	a_reg[13] = _mm_unpackhi_epi16( b_reg[10], b_reg[11] ); \
	a_reg[14] = _mm_unpackhi_epi16( b_reg[12], b_reg[13] ); \
	a_reg[15] = _mm_unpackhi_epi16( b_reg[14], b_reg[15] );

#define UNPACKLOW_EPI32 \
	b_reg[0] = _mm_unpacklo_epi32( a_reg[0], a_reg[1] ); \
	b_reg[1] = _mm_unpacklo_epi32( a_reg[2], a_reg[3] ); \
	b_reg[2] = _mm_unpacklo_epi32( a_reg[4], a_reg[5] ); \
	b_reg[3] = _mm_unpacklo_epi32( a_reg[6], a_reg[7] ); \
\
	b_reg[8] = _mm_unpacklo_epi32( a_reg[8], a_reg[9] ); \
	b_reg[9] = _mm_unpacklo_epi32( a_reg[10], a_reg[11] ); \
	b_reg[10] = _mm_unpacklo_epi32( a_reg[12], a_reg[13] ); \
	b_reg[11] = _mm_unpacklo_epi32( a_reg[14], a_reg[15] );

#define UNPACKHI_EPI32 \
	b_reg[4] = _mm_unpackhi_epi32( a_reg[0], a_reg[1] ); \
	b_reg[5] = _mm_unpackhi_epi32( a_reg[2], a_reg[3] ); \
	b_reg[6] = _mm_unpackhi_epi32( a_reg[4], a_reg[5] ); \
	b_reg[7] = _mm_unpackhi_epi32( a_reg[6], a_reg[7] ); \
\
	b_reg[12] = _mm_unpackhi_epi32( a_reg[8], a_reg[9] ); \
	b_reg[13] = _mm_unpackhi_epi32( a_reg[10], a_reg[11] ); \
	b_reg[14] = _mm_unpackhi_epi32( a_reg[12], a_reg[13] ); \
	b_reg[15] = _mm_unpackhi_epi32( a_reg[14], a_reg[15] );

#define UNPACKLOW_EPI64 \
	a_reg[0] = _mm_unpacklo_epi64( b_reg[0], b_reg[1] ); \
	a_reg[2] = _mm_unpacklo_epi64( b_reg[2], b_reg[3] ); \
	a_reg[4] = _mm_unpacklo_epi64( b_reg[4], b_reg[5] ); \
	a_reg[6] = _mm_unpacklo_epi64( b_reg[6], b_reg[7]) ; \
\
	a_reg[8] = _mm_unpacklo_epi64( b_reg[8], b_reg[9] ); \
	a_reg[10] = _mm_unpacklo_epi64( b_reg[10], b_reg[11] ); \
	a_reg[12] = _mm_unpacklo_epi64( b_reg[12], b_reg[13] ); \
	a_reg[14] = _mm_unpacklo_epi64( b_reg[14], b_reg[15] );

#define UNPACKHI_EPI64 \
	a_reg[1] = _mm_unpackhi_epi64( b_reg[0], b_reg[1] ); \
	a_reg[3] = _mm_unpackhi_epi64( b_reg[2], b_reg[3] ); \
	a_reg[5] = _mm_unpackhi_epi64( b_reg[4], b_reg[5] ); \
	a_reg[7] = _mm_unpackhi_epi64( b_reg[6], b_reg[7] ); \
\
	a_reg[9] = _mm_unpackhi_epi64( b_reg[8], b_reg[9] ); \
	a_reg[11] = _mm_unpackhi_epi64( b_reg[10], b_reg[11] ); \
	a_reg[13] = _mm_unpackhi_epi64( b_reg[12], b_reg[13] ); \
	a_reg[15] = _mm_unpackhi_epi64( b_reg[14], b_reg[15] );

#define UNPACKLOW_EPI16_MR8 \
	a_reg[0] = _mm_unpacklo_epi16( b_reg[0], b_reg[1] ); \
	a_reg[1] = _mm_unpacklo_epi16( b_reg[2], b_reg[3] ); \
	a_reg[2] = _mm_unpacklo_epi16( b_reg[4], b_reg[5] ); \
	a_reg[3] = _mm_unpacklo_epi16( b_reg[6], b_reg[7] );

#define UNPACKHI_EPI16_MR8 \
	a_reg[4] = _mm_unpackhi_epi16( b_reg[0], b_reg[1] ); \
	a_reg[5] = _mm_unpackhi_epi16( b_reg[2], b_reg[3] ); \
	a_reg[6] = _mm_unpackhi_epi16( b_reg[4], b_reg[5] ); \
	a_reg[7] = _mm_unpackhi_epi16( b_reg[6], b_reg[7] );

#define UNPACKLOW_EPI32_MR8 \
	b_reg[0] = _mm_unpacklo_epi32( a_reg[0], a_reg[1] ); \
	b_reg[1] = _mm_unpacklo_epi32( a_reg[2], a_reg[3] ); \
	b_reg[2] = _mm_unpacklo_epi32( a_reg[4], a_reg[5] ); \
	b_reg[3] = _mm_unpacklo_epi32( a_reg[6], a_reg[7] );

#define UNPACKHI_EPI32_MR8 \
	b_reg[4] = _mm_unpackhi_epi32( a_reg[0], a_reg[1] ); \
	b_reg[5] = _mm_unpackhi_epi32( a_reg[2], a_reg[3] ); \
	b_reg[6] = _mm_unpackhi_epi32( a_reg[4], a_reg[5] ); \
	b_reg[7] = _mm_unpackhi_epi32( a_reg[6], a_reg[7] );

#define UNPACKLOW_EPI64_MR8 \
	a_reg[0] = _mm_unpacklo_epi64( b_reg[0], b_reg[1] ); \
	a_reg[2] = _mm_unpacklo_epi64( b_reg[2], b_reg[3] ); \
	a_reg[4] = _mm_unpacklo_epi64( b_reg[4], b_reg[5] ); \
	a_reg[6] = _mm_unpacklo_epi64( b_reg[6], b_reg[7] );

#define UNPACKHI_EPI64_MR8 \
	a_reg[1] = _mm_unpackhi_epi64( b_reg[0], b_reg[1] ); \
	a_reg[3] = _mm_unpackhi_epi64( b_reg[2], b_reg[3] ); \
	a_reg[5] = _mm_unpackhi_epi64( b_reg[4], b_reg[5] ); \
	a_reg[7] = _mm_unpackhi_epi64( b_reg[6], b_reg[7] );

#define UNPACKLOW_EPI32_MR4 \
	b_reg[0] = _mm_unpacklo_epi32( a_reg[0], a_reg[1] ); \
	b_reg[1] = _mm_unpacklo_epi32( a_reg[2], a_reg[3] );

#define UNPACKHI_EPI32_MR4 \
	b_reg[4] = _mm_unpackhi_epi32( a_reg[0], a_reg[1] ); \
	b_reg[5] = _mm_unpackhi_epi32( a_reg[2], a_reg[3] );

#define UNPACKLOW_EPI64_MR4 \
	a_reg[0] = _mm_unpacklo_epi64( b_reg[0], b_reg[1] ); \
	a_reg[4] = _mm_unpacklo_epi64( b_reg[4], b_reg[5] );

#define UNPACKHI_EPI64_MR4 \
	a_reg[1] = _mm_unpackhi_epi64( b_reg[0], b_reg[1] ); \
	a_reg[5] = _mm_unpackhi_epi64( b_reg[4], b_reg[5] );

#define MASKED_STORE_EPI32(mask) \
	_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( ic + 0 ) * KC + kr ), mask, a_reg[0] ); \
	_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( ic + 1 ) * KC + kr ), mask, a_reg[1] ); \
	_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( ic + 2 ) * KC + kr ), mask, a_reg[4] ); \
	_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( ic + 3 ) * KC + kr ), mask, a_reg[5] ); \
	_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( ic + 4 ) * KC + kr ), mask, a_reg[2] ); \
	_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( ic + 5 ) * KC + kr ), mask, a_reg[3] ); \
	_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( ic + 6 ) * KC + kr ), mask, a_reg[6] ); \
	_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( ic + 7 ) * KC + kr ), mask, a_reg[7] ); \
	_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( ic + 8 ) * KC + kr ), mask, a_reg[8] ); \
	_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( ic + 9 ) * KC + kr ), mask, a_reg[9] ); \
	_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( ic + 10 ) * KC + kr ), mask, a_reg[12] ); \
	_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( ic + 11 ) * KC + kr ), mask, a_reg[13] ); \
	_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( ic + 12 ) * KC + kr ), mask, a_reg[10] ); \
	_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( ic + 13 ) * KC + kr ), mask, a_reg[11] ); \
	_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( ic + 14 ) * KC + kr ), mask, a_reg[14] ); \
	_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( ic + 15 ) * KC + kr ), mask, a_reg[15] );

// Column-major transformation to row-major in blocks of MCxKC

void packa_mr8_u8s8s16o16_col_major
    (
      uint8_t*        pack_a_buffer_u8s8s16o16,
      const uint8_t*  a,
      const dim_t     cs,
      const dim_t     KC
    );

void packa_mr4_u8s8s16o16_col_major
    (
      uint8_t*        pack_a_buffer_u8s8s16o16,
      const uint8_t*  a,
      const dim_t     cs,
      const dim_t     KC
    );

void packa_mrlt4_u8s8s16o16_col_major
    (
      uint8_t*        pack_a_buffer_u8s8s16o16,
      const uint8_t*  a,
      const dim_t     cs,
      const dim_t     KC,
      const dim_t     m_left
    );

void packa_mr16_u8s8s16o16_col_major
    (
      uint8_t*        pack_a_buffer_u8s8s16o16,
      const uint8_t*  a,
      const dim_t     rs,
      const dim_t     cs,
      const dim_t     MC,
      const dim_t     KC,
      dim_t*          rs_a,
      dim_t*          cs_a
    )
{
	dim_t mr = 16;
	__m128i a_reg[16], b_reg[16];

	dim_t m_partial_pieces = MC % mr;
	dim_t k_partial_pieces = KC % 16;
	dim_t m_left = MC % 4;
	__m128i mask;

	SET_REGISTERS_ZERO

	dim_t ic, kr;

	for ( ic =0; ( ic + mr - 1 ) < MC; ic += mr )
	{
		for ( kr = 0; ( kr + 15 ) < KC; kr += 16 )
		{
			a_reg[0] = _mm_loadu_si128 ( (__m128i const *) ( a + ( ic * rs ) + ( ( kr + 0 ) * cs ) ) );
			a_reg[1] = _mm_loadu_si128 ( (__m128i const *) ( a + ( ic * rs ) + ( ( kr + 1 ) * cs ) ) );
			a_reg[2] = _mm_loadu_si128 ( (__m128i const *) ( a + ( ic * rs ) + ( ( kr + 2 ) * cs ) ) );
			a_reg[3] = _mm_loadu_si128 ( (__m128i const *) ( a + ( ic * rs ) + ( ( kr + 3 ) * cs ) ) );
			a_reg[4] = _mm_loadu_si128 ( (__m128i const *) ( a + ( ic * rs ) + ( ( kr + 4 ) * cs ) ) );
			a_reg[5] = _mm_loadu_si128 ( (__m128i const *) ( a + ( ic * rs ) + ( ( kr + 5 ) * cs ) ) );
			a_reg[6] = _mm_loadu_si128 ( (__m128i const *) ( a + ( ic * rs ) + ( ( kr + 6 ) * cs ) ) );
			a_reg[7] = _mm_loadu_si128 ( (__m128i const *) ( a + ( ic * rs ) + ( ( kr + 7 ) * cs ) ) );
			a_reg[8] = _mm_loadu_si128 ( (__m128i const *) ( a + ( ic * rs ) + ( ( kr + 8 ) * cs ) ) );
			a_reg[9] = _mm_loadu_si128 ( (__m128i const *) ( a + ( ic * rs ) + ( ( kr + 9 ) * cs ) ) );
			a_reg[10] = _mm_loadu_si128 ( (__m128i const *) ( a + ( ic * rs ) + ( ( kr + 10 ) * cs ) ) );
			a_reg[11] = _mm_loadu_si128 ( (__m128i const *) ( a + ( ic * rs ) + ( ( kr + 11 ) * cs ) ) );
			a_reg[12] = _mm_loadu_si128 ( (__m128i const *) ( a + ( ic * rs ) + ( ( kr + 12 ) * cs ) ) );
			a_reg[13] = _mm_loadu_si128 ( (__m128i const *) ( a + ( ic * rs ) + ( ( kr + 13 ) * cs ) ) );
			a_reg[14] = _mm_loadu_si128 ( (__m128i const *) ( a + ( ic * rs ) + ( ( kr + 14 ) * cs ) ) );
			a_reg[15] = _mm_loadu_si128 ( (__m128i const *) ( a + ( ic * rs ) + ( ( kr + 15 ) * cs ) ) );

			// Transpose operations
			UNPACKLOW_EPI8
			UNPACKHI_EPI8

			UNPACKLOW_EPI16
			UNPACKHI_EPI16

			UNPACKLOW_EPI32
			UNPACKHI_EPI32

			UNPACKLOW_EPI64
			UNPACKHI_EPI64

			_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( ic + 0 ) * KC + kr ), a_reg[0] );
			_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( ic + 1 ) * KC + kr ), a_reg[1] );
			_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( ic + 2 ) * KC + kr ), a_reg[4] );
			_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( ic + 3 ) * KC + kr ), a_reg[5] );
			_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( ic + 4 ) * KC + kr ), a_reg[2] );
			_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( ic + 5 ) * KC + kr ), a_reg[3] );
			_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( ic + 6 ) * KC + kr ), a_reg[6] );
			_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( ic + 7 ) * KC + kr ), a_reg[7] );
			_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( ic + 8 ) * KC + kr ), a_reg[8] );
			_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( ic + 9 ) * KC + kr ), a_reg[9] );
			_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( ic + 10 ) * KC + kr ), a_reg[12] );
			_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( ic + 11 ) * KC + kr ), a_reg[13] );
			_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( ic + 12 ) * KC + kr ), a_reg[10] );
			_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( ic + 13 ) * KC + kr ), a_reg[11] );
			_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( ic + 14 ) * KC + kr ), a_reg[14] );
			_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( ic + 15 ) * KC + kr ), a_reg[15] );

		}

		// Handle k remainder.
		if ( k_partial_pieces > 0 )
		{
			// k fringe 8
			if (( kr + 7 ) < KC )
			{
				a_reg[0] = _mm_loadu_si128 ( (__m128i const *) ( a + ( ic * rs ) + ( ( kr + 0 ) * cs ) ) );
				a_reg[1] = _mm_loadu_si128 ( (__m128i const *) ( a + ( ic * rs ) + ( ( kr + 1 ) * cs ) ) );
				a_reg[2] = _mm_loadu_si128 ( (__m128i const *) ( a + ( ic * rs ) + ( ( kr + 2 ) * cs ) ) );
				a_reg[3] = _mm_loadu_si128 ( (__m128i const *) ( a + ( ic * rs ) + ( ( kr + 3 ) * cs ) ) );
				a_reg[4] = _mm_loadu_si128 ( (__m128i const *) ( a + ( ic * rs ) + ( ( kr + 4 ) * cs ) ) );
				a_reg[5] = _mm_loadu_si128 ( (__m128i const *) ( a + ( ic * rs ) + ( ( kr + 5 ) * cs ) ) );
				a_reg[6] = _mm_loadu_si128 ( (__m128i const *) ( a + ( ic * rs ) + ( ( kr + 6 ) * cs ) ) );
				a_reg[7] = _mm_loadu_si128 ( (__m128i const *) ( a + ( ic * rs ) + ( ( kr + 7 ) * cs ) ) );

				// Transpose operations
				UNPACKLOW_EPI8
				UNPACKHI_EPI8

				UNPACKLOW_EPI16
				UNPACKHI_EPI16

				UNPACKLOW_EPI32
				UNPACKHI_EPI32

				UNPACKLOW_EPI64
				UNPACKHI_EPI64

				mask = _mm_set_epi32 (0, 0, -1, -1);

				MASKED_STORE_EPI32(mask);

				kr += 8;
			}

			// k fringe 4
			if ( ( kr + 3 ) < KC )
			{
				a_reg[0] = _mm_loadu_si128( (__m128i const *)( a + ( ic * rs ) + ( ( kr + 0 ) * cs ) ) );
				a_reg[1] = _mm_loadu_si128( (__m128i const *)( a + ( ic * rs ) + ( ( kr + 1 ) * cs ) ) );
				a_reg[2] = _mm_loadu_si128( (__m128i const *)( a + ( ic * rs ) + ( ( kr + 2 ) * cs ) ) );
				a_reg[3] = _mm_loadu_si128( (__m128i const *)( a + ( ic * rs ) + ( ( kr + 3 ) * cs ) ) );

				// Transpose operations
				UNPACKLOW_EPI8
				UNPACKHI_EPI8

				UNPACKLOW_EPI16
				UNPACKHI_EPI16

				UNPACKLOW_EPI32
				UNPACKHI_EPI32

				UNPACKLOW_EPI64
				UNPACKHI_EPI64

				mask = _mm_set_epi32 (0, 0, 0, -1);

				MASKED_STORE_EPI32(mask);

				kr += 4;
			}

			// k fringe 2
			if ( ( kr + 1 ) < KC )
			{
				a_reg[0] = _mm_loadu_si128( (__m128i const *)( a + ( ic * rs ) + ( ( kr + 0 ) * cs ) ) );
				a_reg[1] = _mm_loadu_si128( (__m128i const *)( a + ( ic * rs ) + ( ( kr + 1 ) * cs ) ) );

				// Transpose operations
				UNPACKLOW_EPI8
				UNPACKHI_EPI8

				UNPACKLOW_EPI16
				UNPACKHI_EPI16

				UNPACKLOW_EPI32
				UNPACKHI_EPI32

				UNPACKLOW_EPI64
				UNPACKHI_EPI64

				uint8_t buf[16];
				dim_t n0_rem_bytes = 2 * sizeof( uint8_t );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[0] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+0) * KC + kr ), buf, n0_rem_bytes );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[1] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+1) * KC + kr ), buf, n0_rem_bytes );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[4] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+2) * KC + kr ), buf, n0_rem_bytes );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[5] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+3) * KC + kr ), buf, n0_rem_bytes );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[2] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+4) * KC + kr ), buf, n0_rem_bytes );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[3] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+5) * KC + kr ), buf, n0_rem_bytes );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[6] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+6) * KC + kr ), buf, n0_rem_bytes );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[7] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+7) * KC + kr ), buf, n0_rem_bytes );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[8] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+8) * KC + kr ), buf, n0_rem_bytes );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[9] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+9) * KC + kr ), buf, n0_rem_bytes );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[12] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+10) * KC + kr ), buf, n0_rem_bytes );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[13] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+11) * KC + kr ), buf, n0_rem_bytes );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[10] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+12) * KC + kr ), buf, n0_rem_bytes );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[11] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+13) * KC + kr ), buf, n0_rem_bytes );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[14] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+14) * KC + kr ), buf, n0_rem_bytes );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[15] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+15) * KC + kr ), buf, n0_rem_bytes );

				kr += 2;
			}

			// k fringe 1
			if ( ( kr ) < KC )
			{
				a_reg[0] = _mm_loadu_si128( (__m128i const *)( a + ( ic * rs ) + ( ( kr + 0 ) * cs ) ) );

				// Transpose operations
				UNPACKLOW_EPI8
				UNPACKHI_EPI8

				UNPACKLOW_EPI16
				UNPACKHI_EPI16

				UNPACKLOW_EPI32
				UNPACKHI_EPI32

				UNPACKLOW_EPI64
				UNPACKHI_EPI64

				uint8_t buf[16];
				dim_t n0_rem_bytes = 1 * sizeof( uint8_t );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[0] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+0) * KC + kr ), buf, n0_rem_bytes );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[1] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+1) * KC + kr ), buf, n0_rem_bytes );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[4] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+2) * KC + kr ), buf, n0_rem_bytes );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[5] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+3) * KC + kr ), buf, n0_rem_bytes );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[2] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+4) * KC + kr ), buf, n0_rem_bytes );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[3] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+5) * KC + kr ), buf, n0_rem_bytes );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[6] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+6) * KC + kr ), buf, n0_rem_bytes );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[7] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+7) * KC + kr ), buf, n0_rem_bytes );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[8] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+8) * KC + kr ), buf, n0_rem_bytes );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[9] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+9) * KC + kr ), buf, n0_rem_bytes );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[12] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+10) * KC + kr ), buf, n0_rem_bytes );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[13] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+11) * KC + kr ), buf, n0_rem_bytes );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[10] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+12) * KC + kr ), buf, n0_rem_bytes );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[11] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+13) * KC + kr ), buf, n0_rem_bytes );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[14] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+14) * KC + kr ), buf, n0_rem_bytes );

				_mm_storeu_si128( ( __m128i* )buf, a_reg[15] );
				memcpy( ( pack_a_buffer_u8s8s16o16 + (ic+15) * KC + kr ), buf, n0_rem_bytes );

				kr += 1;
			}
		}
	}

	if( m_partial_pieces > 0 )
	{
		if ( ( ic + 8 - 1 ) < MC )
		{
			packa_mr8_u8s8s16o16_col_major
				(
					( pack_a_buffer_u8s8s16o16 + ( ic * KC ) ),
					( a + ic * rs ), cs, KC
				);

			ic += 8;
		}

		if ( ( ic + 4 - 1 ) < MC )
		{
			packa_mr4_u8s8s16o16_col_major
				(
					( pack_a_buffer_u8s8s16o16 + ( ic * KC ) ),
					( a + ic * rs ), cs, KC
				);

			ic += 4;
		}

		if ( m_left )
		{
			packa_mrlt4_u8s8s16o16_col_major
				(
					( pack_a_buffer_u8s8s16o16 + ( ic * KC ) ),
					( a + ic * rs ), cs, KC, m_left
				);
		}
	}

	*rs_a = KC;
	*cs_a = 1;
}

void packa_mr8_u8s8s16o16_col_major
    (
      uint8_t*        pack_a_buffer_u8s8s16o16,
      const uint8_t*  a,
      const dim_t     cs,
      const dim_t     KC
    )
{
	dim_t kr = 0;
	__m128i a_reg[16], b_reg[16];

	dim_t k_partial_pieces = KC % 16;
	__m128i mask;

	SET_REGISTERS_ZERO

	for( kr = 0; ( kr + 15 ) < KC; kr += 16 )
	{
		mask = _mm_set_epi32 (0, 0, -1, -1);

		a_reg[0] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 0 ) * cs ) ), mask );
		a_reg[1] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 1 ) * cs ) ), mask );
		a_reg[2] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 2 ) * cs ) ), mask );
		a_reg[3] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 3 ) * cs ) ), mask );
		a_reg[4] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 4 ) * cs ) ), mask );
		a_reg[5] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 5 ) * cs ) ), mask );
		a_reg[6] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 6 ) * cs ) ), mask );
		a_reg[7] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 7 ) * cs ) ), mask );
		a_reg[8] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 8 ) * cs ) ), mask );
		a_reg[9] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 9 ) * cs ) ), mask );
		a_reg[10] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 10 ) * cs ) ), mask );
		a_reg[11] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 11 ) * cs ) ), mask );
		a_reg[12] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 12 ) * cs ) ), mask );
		a_reg[13] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 13 ) * cs ) ), mask );
		a_reg[14] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 14 ) * cs ) ), mask );
		a_reg[15] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 15 ) * cs ) ), mask );

		// Transpose operations
		UNPACKLOW_EPI8

		UNPACKLOW_EPI16_MR8
		UNPACKHI_EPI16_MR8

		UNPACKLOW_EPI32_MR8
		UNPACKHI_EPI32_MR8

		UNPACKLOW_EPI64_MR8
		UNPACKHI_EPI64_MR8

		_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( 0 ) * KC + kr ), a_reg[0] );
		_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( 1 ) * KC + kr ), a_reg[1] );
		_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( 2 ) * KC + kr ), a_reg[4] );
		_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( 3 ) * KC + kr ), a_reg[5] );
		_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( 4 ) * KC + kr ), a_reg[2] );
		_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( 5 ) * KC + kr ), a_reg[3] );
		_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( 6 ) * KC + kr ), a_reg[6] );
		_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( 7 ) * KC + kr ), a_reg[7] );
	}

	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		// k fringe 8
		if ( ( kr + 7 ) < KC )
		{
			mask = _mm_set_epi32 (0, 0, -1, -1);

			a_reg[0] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 0 ) * cs ) ), mask );
			a_reg[1] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 1 ) * cs ) ), mask );
			a_reg[2] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 2 ) * cs ) ), mask );
			a_reg[3] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 3 ) * cs ) ), mask );
			a_reg[4] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 4 ) * cs ) ), mask );
			a_reg[5] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 5 ) * cs ) ), mask );
			a_reg[6] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 6 ) * cs ) ), mask );
			a_reg[7] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 7 ) * cs ) ), mask );

			// Transpose operations
			UNPACKLOW_EPI8

			UNPACKLOW_EPI16_MR8
			UNPACKHI_EPI16_MR8

			UNPACKLOW_EPI32_MR8
			UNPACKHI_EPI32_MR8

			UNPACKLOW_EPI64_MR8
			UNPACKHI_EPI64_MR8

			_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( 0 ) * KC + kr ), mask, a_reg[0] );
			_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( 1 ) * KC + kr ), mask, a_reg[1] );
			_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( 2 ) * KC + kr ), mask, a_reg[4] );
			_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( 3 ) * KC + kr ), mask, a_reg[5] );
			_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( 4 ) * KC + kr ), mask, a_reg[2] );
			_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( 5 ) * KC + kr ), mask, a_reg[3] );
			_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( 6 ) * KC + kr ), mask, a_reg[6] );
			_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( 7 ) * KC + kr ), mask, a_reg[7] );

			kr += 8;
		}

		// k fringe 4
		if ( ( kr + 3 ) < KC )
		{
			mask = _mm_set_epi32 (0, 0, -1, -1);

			a_reg[0] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 0 ) * cs ) ), mask );
			a_reg[1] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 1 ) * cs ) ), mask );
			a_reg[2] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 2 ) * cs ) ), mask );
			a_reg[3] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 3 ) * cs ) ), mask );

			// Transpose operations
			UNPACKLOW_EPI8

			UNPACKLOW_EPI16_MR8
			UNPACKHI_EPI16_MR8

			UNPACKLOW_EPI32_MR8
			UNPACKHI_EPI32_MR8

			UNPACKLOW_EPI64_MR8
			UNPACKHI_EPI64_MR8

			mask = _mm_set_epi32 (0, 0, 0, -1);

			_mm_maskstore_epi32( ( int* )( pack_a_buffer_u8s8s16o16 + ( 0 ) * KC + kr ), mask, a_reg[0] );
			_mm_maskstore_epi32( ( int* )( pack_a_buffer_u8s8s16o16 + ( 1 ) * KC + kr ), mask, a_reg[1] );
			_mm_maskstore_epi32( ( int* )( pack_a_buffer_u8s8s16o16 + ( 2 ) * KC + kr ), mask, a_reg[4] );
			_mm_maskstore_epi32( ( int* )( pack_a_buffer_u8s8s16o16 + ( 3 ) * KC + kr ), mask, a_reg[5] );
			_mm_maskstore_epi32( ( int* )( pack_a_buffer_u8s8s16o16 + ( 4 ) * KC + kr ), mask, a_reg[2] );
			_mm_maskstore_epi32( ( int* )( pack_a_buffer_u8s8s16o16 + ( 5 ) * KC + kr ), mask, a_reg[3] );
			_mm_maskstore_epi32( ( int* )( pack_a_buffer_u8s8s16o16 + ( 6 ) * KC + kr ), mask, a_reg[6] );
			_mm_maskstore_epi32( ( int* )( pack_a_buffer_u8s8s16o16 + ( 7 ) * KC + kr ), mask, a_reg[7] );

			kr += 4;
		}

		// k fringe 2
		if ( ( kr + 1 ) < KC )
		{
			mask = _mm_set_epi32 (0, 0, -1, -1);

			a_reg[0] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 0 ) * cs ) ), mask );
			a_reg[1] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 1 ) * cs ) ), mask );

			// Transpose operations
			UNPACKLOW_EPI8

			UNPACKLOW_EPI16_MR8
			UNPACKHI_EPI16_MR8

			UNPACKLOW_EPI32_MR8
			UNPACKHI_EPI32_MR8

			UNPACKLOW_EPI64_MR8
			UNPACKHI_EPI64_MR8

			uint8_t buf[16];
			dim_t n0_rem_bytes = 2 * sizeof( uint8_t );

			_mm_storeu_si128( ( __m128i* )buf, a_reg[0] );
			memcpy( ( pack_a_buffer_u8s8s16o16 + ( 0 ) * KC + kr ), buf, n0_rem_bytes );

			_mm_storeu_si128( ( __m128i* )buf, a_reg[1] );
			memcpy( ( pack_a_buffer_u8s8s16o16 + ( 1 ) * KC + kr ), buf, n0_rem_bytes );

			_mm_storeu_si128( ( __m128i* )buf, a_reg[4] );
			memcpy( ( pack_a_buffer_u8s8s16o16 + ( 2 ) * KC + kr ), buf, n0_rem_bytes );

			_mm_storeu_si128( ( __m128i* )buf, a_reg[5] );
			memcpy( ( pack_a_buffer_u8s8s16o16 + ( 3 ) * KC + kr ), buf, n0_rem_bytes );

			_mm_storeu_si128( ( __m128i* )buf, a_reg[2] );
			memcpy( ( pack_a_buffer_u8s8s16o16 + ( 4 ) * KC + kr ), buf, n0_rem_bytes );

			_mm_storeu_si128( ( __m128i* )buf, a_reg[3] );
			memcpy( ( pack_a_buffer_u8s8s16o16 + ( 5 ) * KC + kr ), buf, n0_rem_bytes );

			_mm_storeu_si128( ( __m128i* )buf, a_reg[6] );
			memcpy( ( pack_a_buffer_u8s8s16o16 + ( 6 ) * KC + kr ), buf, n0_rem_bytes );

			_mm_storeu_si128( ( __m128i* )buf, a_reg[7] );
			memcpy( ( pack_a_buffer_u8s8s16o16 + ( 7 ) * KC + kr ), buf, n0_rem_bytes );

			kr += 2;

		}

		// k fringe 1
		if ( ( kr ) < KC )
		{
			mask = _mm_set_epi32 (0, 0, -1, -1);

			a_reg[0] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 0 ) * cs ) ), mask );

			// Transpose operations
			UNPACKLOW_EPI8

			UNPACKLOW_EPI16_MR8
			UNPACKHI_EPI16_MR8

			UNPACKLOW_EPI32_MR8
			UNPACKHI_EPI32_MR8

			UNPACKLOW_EPI64_MR8
			UNPACKHI_EPI64_MR8

			uint8_t buf[16];
			dim_t n0_rem_bytes = 1 * sizeof( uint8_t );

			_mm_storeu_si128( ( __m128i* )buf, a_reg[0] );
			memcpy( ( pack_a_buffer_u8s8s16o16 + ( 0 ) * KC + kr ), buf, n0_rem_bytes );

			_mm_storeu_si128( ( __m128i* )buf, a_reg[1] );
			memcpy( ( pack_a_buffer_u8s8s16o16 + ( 1 ) * KC + kr ), buf, n0_rem_bytes );

			_mm_storeu_si128( ( __m128i* )buf, a_reg[4] );
			memcpy( ( pack_a_buffer_u8s8s16o16 + ( 2 ) * KC + kr ), buf, n0_rem_bytes );

			_mm_storeu_si128( ( __m128i* )buf, a_reg[5] );
			memcpy( ( pack_a_buffer_u8s8s16o16 + ( 3 ) * KC + kr ), buf, n0_rem_bytes );

			_mm_storeu_si128( ( __m128i* )buf, a_reg[2] );
			memcpy( ( pack_a_buffer_u8s8s16o16 + ( 4 ) * KC + kr ), buf, n0_rem_bytes );

			_mm_storeu_si128( ( __m128i* )buf, a_reg[3] );
			memcpy( ( pack_a_buffer_u8s8s16o16 + ( 5 ) * KC + kr ), buf, n0_rem_bytes );

			_mm_storeu_si128( ( __m128i* )buf, a_reg[6] );
			memcpy( ( pack_a_buffer_u8s8s16o16 + ( 6 ) * KC + kr ), buf, n0_rem_bytes );

			_mm_storeu_si128( ( __m128i* )buf, a_reg[7] );
			memcpy( ( pack_a_buffer_u8s8s16o16 + ( 7 ) * KC + kr ), buf, n0_rem_bytes );

			kr += 1;
		}
	}
}


void packa_mr4_u8s8s16o16_col_major
    (
      uint8_t*        pack_a_buffer_u8s8s16o16,
      const uint8_t*  a,
      const dim_t     cs,
      const dim_t     KC
    )
{
	dim_t kr = 0;
	__m128i a_reg[16], b_reg[16];
	__m128i mask;

	SET_REGISTERS_ZERO

	dim_t k_partial_pieces = KC % 16;

	for( kr = 0; ( kr + 15 ) < KC; kr += 16 )
	{
		mask = _mm_set_epi32 (0, -1, -1, -1);

		a_reg[0] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 0 ) * cs ) ), mask );
		a_reg[1] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 1 ) * cs ) ), mask );
		a_reg[2] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 2 ) * cs ) ), mask );
		a_reg[3] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 3 ) * cs ) ), mask );
		a_reg[4] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 4 ) * cs ) ), mask );
		a_reg[5] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 5 ) * cs ) ), mask );
		a_reg[6] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 6 ) * cs ) ), mask );
		a_reg[7] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 7 ) * cs ) ), mask );
		a_reg[8] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 8 ) * cs ) ), mask );
		a_reg[9] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 9 ) * cs ) ), mask );
		a_reg[10] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 10 ) * cs ) ), mask );
		a_reg[11] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 11 ) * cs ) ), mask );
		a_reg[12] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 12 ) * cs ) ), mask );
		a_reg[13] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 13 ) * cs ) ), mask );
		a_reg[14] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 14 ) * cs ) ), mask );
		a_reg[15] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 15 ) * cs ) ), mask );

		// Transpose operations
		UNPACKLOW_EPI8

		UNPACKLOW_EPI16_MR8

		UNPACKLOW_EPI32_MR4
		UNPACKHI_EPI32_MR4

		UNPACKLOW_EPI64_MR4
		UNPACKHI_EPI64_MR4

		_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( 0 ) * KC + kr ), a_reg[0] );
		_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( 1 ) * KC + kr ), a_reg[1] );
		_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( 2 ) * KC + kr ), a_reg[4] );
		_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( 3 ) * KC + kr ), a_reg[5] );
	}

	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		// k fringe 8
		if ( ( kr + 7 ) < KC )
		{
			mask = _mm_set_epi32 (0, -1, -1, -1);

			a_reg[0] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 0 ) * cs ) ), mask );
			a_reg[1] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 1 ) * cs ) ), mask );
			a_reg[2] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 2 ) * cs ) ), mask );
			a_reg[3] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 3 ) * cs ) ), mask );
			a_reg[4] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 4 ) * cs ) ), mask );
			a_reg[5] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 5 ) * cs ) ), mask );
			a_reg[6] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 6 ) * cs ) ), mask );
			a_reg[7] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 7 ) * cs ) ), mask );

			// Transpose operations
			UNPACKLOW_EPI8

			UNPACKLOW_EPI16_MR8

			UNPACKLOW_EPI32_MR4
			UNPACKHI_EPI32_MR4

			UNPACKLOW_EPI64_MR4
			UNPACKHI_EPI64_MR4

			mask = _mm_set_epi32 (0, 0, -1, -1);

			_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( 0 ) * KC + kr ), mask, a_reg[0] );
			_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( 1 ) * KC + kr ), mask, a_reg[1] );
			_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( 2 ) * KC + kr ), mask, a_reg[4] );
			_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( 3 ) * KC + kr ), mask, a_reg[5] );

			kr += 8;
		}

		// k fringe 4
		if ( ( kr + 3 ) < KC )
		{
			mask = _mm_set_epi32 (0, -1, -1, -1);

			a_reg[0] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 0 ) * cs ) ), mask );
			a_reg[1] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 1 ) * cs ) ), mask );
			a_reg[2] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 2 ) * cs ) ), mask );
			a_reg[3] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 3 ) * cs ) ), mask );

			// Transpose operations
			UNPACKLOW_EPI8

			UNPACKLOW_EPI16_MR8

			UNPACKLOW_EPI32_MR4
			UNPACKHI_EPI32_MR4

			UNPACKLOW_EPI64_MR4
			UNPACKHI_EPI64_MR4

			mask = _mm_set_epi32 (0, 0, 0, -1);

			_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( 0 ) * KC + kr ), mask, a_reg[0] );
			_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( 1 ) * KC + kr ), mask, a_reg[1] );
			_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( 2 ) * KC + kr ), mask, a_reg[4] );
			_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( 3 ) * KC + kr ), mask, a_reg[5] );

			kr += 4;
		}

		// k fringe 2
		if ( ( kr + 1 ) < KC )
		{
			mask = _mm_set_epi32 (0, -1, -1, -1);

			a_reg[0] = _mm_maskload_epi32 ( (int const* ) ( a + ( ( kr + 0 ) * cs ) ), mask );
			a_reg[1] = _mm_maskload_epi32 ( (int const* ) ( a + ( ( kr + 1 ) * cs ) ), mask );

			// Transpose operations
			UNPACKLOW_EPI8

			UNPACKLOW_EPI16_MR8

			UNPACKLOW_EPI32_MR4
			UNPACKHI_EPI32_MR4

			UNPACKLOW_EPI64_MR4
			UNPACKHI_EPI64_MR4

			uint8_t buf[16];
			dim_t n0_rem_bytes = 2 * sizeof( uint8_t );

			_mm_storeu_si128( ( __m128i* )buf, a_reg[0] );
			memcpy( ( pack_a_buffer_u8s8s16o16 + ( 0 ) * KC + kr ), buf, n0_rem_bytes );

			_mm_storeu_si128( ( __m128i* )buf, a_reg[1] );
			memcpy( ( pack_a_buffer_u8s8s16o16 + ( 1 ) * KC + kr ), buf, n0_rem_bytes );

			_mm_storeu_si128( ( __m128i* )buf, a_reg[4] );
			memcpy( ( pack_a_buffer_u8s8s16o16 + ( 2 ) * KC + kr ), buf, n0_rem_bytes );

			_mm_storeu_si128( ( __m128i* )buf, a_reg[5] );
			memcpy( ( pack_a_buffer_u8s8s16o16 + ( 3 ) * KC + kr ), buf, n0_rem_bytes );


			kr += 2;
		}

		// k fringe 1
		if ( ( kr ) < KC )
		{
			mask = _mm_set_epi32 (0, -1, -1, -1);

			a_reg[0] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 0 ) * cs ) ), mask );

			// Transpose operations
			UNPACKLOW_EPI8

			UNPACKLOW_EPI16_MR8

			UNPACKLOW_EPI32_MR4
			UNPACKHI_EPI32_MR4

			UNPACKLOW_EPI64_MR4
			UNPACKHI_EPI64_MR4

			uint8_t buf[16];
			dim_t n0_rem_bytes = 1 * sizeof( uint8_t );

			_mm_storeu_si128( ( __m128i* )buf, a_reg[0] );
			memcpy( ( pack_a_buffer_u8s8s16o16 + ( 0 ) * KC + kr ), buf, n0_rem_bytes );

			_mm_storeu_si128( ( __m128i* )buf, a_reg[1] );
			memcpy( ( pack_a_buffer_u8s8s16o16 + ( 1 ) * KC + kr ), buf, n0_rem_bytes );

			_mm_storeu_si128( ( __m128i* )buf, a_reg[4] );
			memcpy( ( pack_a_buffer_u8s8s16o16 + ( 2 ) * KC + kr ), buf, n0_rem_bytes );

			_mm_storeu_si128( ( __m128i* )buf, a_reg[5] );
			memcpy( ( pack_a_buffer_u8s8s16o16 + ( 3 ) * KC + kr ), buf, n0_rem_bytes );

			kr += 1;
		}
	}
}

void packa_mrlt4_u8s8s16o16_col_major
    (
      uint8_t*        pack_a_buffer_u8s8s16o16,
      const uint8_t*  a,
      const dim_t     cs,
      const dim_t     KC,
      const dim_t     m_left
    )
{
	dim_t kr = 0;
	__m128i a_reg[16], b_reg[16];
	__m128i mask;

	SET_REGISTERS_ZERO

	dim_t k_partial_pieces = KC % 16;

	for( kr = 0; ( kr + 15 ) < KC; kr += 16 )
	{
		mask = _mm_set_epi32 (0, -1, -1, -1);

		a_reg[0] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 0 ) * cs ) ), mask );
		a_reg[1] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 1 ) * cs ) ), mask );
		a_reg[2] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 2 ) * cs ) ), mask );
		a_reg[3] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 3 ) * cs ) ), mask );
		a_reg[4] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 4 ) * cs ) ), mask );
		a_reg[5] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 5 ) * cs ) ), mask );
		a_reg[6] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 6 ) * cs ) ), mask );
		a_reg[7] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 7 ) * cs ) ), mask );
		a_reg[8] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 8 ) * cs ) ), mask );
		a_reg[9] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 9 ) * cs ) ), mask );
		a_reg[10] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 10 ) * cs ) ), mask );
		a_reg[11] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 11 ) * cs ) ), mask );
		a_reg[12] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 12 ) * cs ) ), mask );
		a_reg[13] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 13 ) * cs ) ), mask );
		a_reg[14] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 14 ) * cs ) ), mask );
		a_reg[15] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 15 ) * cs ) ), mask );

		// Transpose operations
		UNPACKLOW_EPI8

		UNPACKLOW_EPI16_MR8

		UNPACKLOW_EPI32_MR4
		UNPACKHI_EPI32_MR4

		UNPACKLOW_EPI64_MR4
		UNPACKHI_EPI64_MR4

		switch( m_left )
		{
			case 3:
				_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( 0 ) * KC + kr ), a_reg[0] );
				_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( 1 ) * KC + kr ), a_reg[1] );
				_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( 2 ) * KC + kr ), a_reg[4] );
				break;

			case 2:
				_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( 0 ) * KC + kr ), a_reg[0] );
				_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( 1 ) * KC + kr ), a_reg[1] );
				break;

			case 1:
				_mm_storeu_si128( (__m128i *)( pack_a_buffer_u8s8s16o16 + ( 0 ) * KC + kr ), a_reg[0] );
				break;
		}
	}

	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		// k fringe 8
		if ( ( kr + 7 ) < KC )
		{
			mask = _mm_set_epi32 (0, -1, -1, -1);

			a_reg[0] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 0 ) * cs ) ), mask );
			a_reg[1] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 1 ) * cs ) ), mask );
			a_reg[2] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 2 ) * cs ) ), mask );
			a_reg[3] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 3 ) * cs ) ), mask );
			a_reg[4] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 4 ) * cs ) ), mask );
			a_reg[5] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 5 ) * cs ) ), mask );
			a_reg[6] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 6 ) * cs ) ), mask );
			a_reg[7] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 7 ) * cs ) ), mask );

			// Transpose operations
			UNPACKLOW_EPI8

			UNPACKLOW_EPI16_MR8

			UNPACKLOW_EPI32_MR4
			UNPACKHI_EPI32_MR4

			UNPACKLOW_EPI64_MR4
			UNPACKHI_EPI64_MR4

			mask = _mm_set_epi32 (0, 0, -1, -1);

			switch( m_left )
			{
				case 3:
					_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + (0) * KC + kr ), mask, a_reg[0] );
					_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + (1) * KC + kr ), mask, a_reg[1] );
					_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + (2) * KC + kr ), mask, a_reg[4] );
					break;

				case 2:
					_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + (0) * KC + kr ), mask, a_reg[0] );
					_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + (1) * KC + kr ), mask, a_reg[1] );
					break;

				case 1:
					_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + (0) * KC + kr ), mask, a_reg[0] );
					break;
			}

			kr += 8;
		}

		// k fringe 4
		if ( ( kr + 3 ) < KC )
		{
			mask = _mm_set_epi32 (0, -1, -1, -1);

			a_reg[0] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 0 ) * cs ) ), mask );
			a_reg[1] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 1 ) * cs ) ), mask );
			a_reg[2] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 2 ) * cs ) ), mask );
			a_reg[3] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 3 ) * cs ) ), mask );

			// Transpose operations
			UNPACKLOW_EPI8

			UNPACKLOW_EPI16_MR8

			UNPACKLOW_EPI32_MR4
			UNPACKHI_EPI32_MR4

			UNPACKLOW_EPI64_MR4
			UNPACKHI_EPI64_MR4

			mask = _mm_set_epi32 (0, 0, 0, -1);

			switch( m_left )
			{
				case 3:
					_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( 0 ) * KC + kr ), mask, a_reg[0] );
					_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( 1 ) * KC + kr ), mask, a_reg[1] );
					_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( 2 ) * KC + kr ), mask, a_reg[4] );
					break;

				case 2:
					_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( 0 ) * KC + kr ), mask, a_reg[0] );
					_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( 1 ) * KC + kr ), mask, a_reg[1] );
					break;

				case 1:
					_mm_maskstore_epi32( ( int* ) ( pack_a_buffer_u8s8s16o16 + ( 0 ) * KC + kr ), mask, a_reg[0] );
					break;
			}

			kr += 4;
		}

		// k fringe 2
		if ( ( kr + 1 ) < KC )
		{
			mask = _mm_set_epi32 (0, -1, -1, -1);

			a_reg[0] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 0 ) * cs ) ), mask );
			a_reg[1] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 1 ) * cs ) ), mask );

			// Transpose operations
			UNPACKLOW_EPI8

			UNPACKLOW_EPI16_MR8

			UNPACKLOW_EPI32_MR4
			UNPACKHI_EPI32_MR4

			UNPACKLOW_EPI64_MR4
			UNPACKHI_EPI64_MR4

			uint8_t buf[16];
			dim_t n0_rem_bytes = 2 * sizeof( uint8_t );

			switch( m_left )
			{
				case 3:
					_mm_storeu_si128( ( __m128i* )buf, a_reg[0] );
					memcpy( ( pack_a_buffer_u8s8s16o16 + ( 0 ) * KC + kr ), buf, n0_rem_bytes );

					_mm_storeu_si128( ( __m128i* )buf, a_reg[1] );
					memcpy( ( pack_a_buffer_u8s8s16o16 + ( 1 ) * KC + kr ), buf, n0_rem_bytes );

					_mm_storeu_si128( ( __m128i* )buf, a_reg[4] );
					memcpy( ( pack_a_buffer_u8s8s16o16 + ( 2 ) * KC + kr ), buf, n0_rem_bytes );

					break;

				case 2:
					_mm_storeu_si128( ( __m128i* )buf, a_reg[0] );
					memcpy( ( pack_a_buffer_u8s8s16o16 + ( 0 ) * KC + kr ), buf, n0_rem_bytes );

					_mm_storeu_si128( ( __m128i* )buf, a_reg[1] );
					memcpy( ( pack_a_buffer_u8s8s16o16 + ( 1 ) * KC + kr ), buf, n0_rem_bytes );

					break;

				case 1:
					_mm_storeu_si128( ( __m128i* )buf, a_reg[0] );
					memcpy( ( pack_a_buffer_u8s8s16o16 + ( 0 ) * KC + kr ), buf, n0_rem_bytes );

					break;
			}

			kr += 2;
		}

		// k fringe 1
		if ( ( kr ) < KC )
		{
			mask = _mm_set_epi32 (0, -1, -1, -1);

			a_reg[0] = _mm_maskload_epi32 ( ( int const* ) ( a + ( ( kr + 0 ) * cs )), mask );

			// Transpose operations
			UNPACKLOW_EPI8

			UNPACKLOW_EPI16_MR8

			UNPACKLOW_EPI32_MR4
			UNPACKHI_EPI32_MR4

			UNPACKLOW_EPI64_MR4
			UNPACKHI_EPI64_MR4

			uint8_t buf[16];
			dim_t n0_rem_bytes = 1 * sizeof( uint8_t );

			switch( m_left )
			{
				case 3:
					_mm_storeu_si128( ( __m128i* )buf, a_reg[0] );
					memcpy( ( pack_a_buffer_u8s8s16o16 + ( 0 ) * KC + kr ), buf, n0_rem_bytes );

					_mm_storeu_si128( ( __m128i* )buf, a_reg[1] );
					memcpy( ( pack_a_buffer_u8s8s16o16 + ( 1 ) * KC + kr ), buf, n0_rem_bytes );

					_mm_storeu_si128( ( __m128i* )buf, a_reg[4] );
					memcpy( ( pack_a_buffer_u8s8s16o16 + ( 2 ) * KC + kr ), buf, n0_rem_bytes );

					break;

				case 2:
					_mm_storeu_si128( ( __m128i* )buf, a_reg[0] );
					memcpy( ( pack_a_buffer_u8s8s16o16 + ( 0 ) * KC + kr ), buf, n0_rem_bytes );

					_mm_storeu_si128( ( __m128i* )buf, a_reg[1] );
					memcpy( ( pack_a_buffer_u8s8s16o16 + ( 1 ) * KC + kr ), buf, n0_rem_bytes );

					break;

				case 1:
					_mm_storeu_si128( ( __m128i* )buf, a_reg[0] );
					memcpy( ( pack_a_buffer_u8s8s16o16 + ( 0 ) * KC + kr ), buf, n0_rem_bytes );

					break;
			}

			kr += 1;
		}
	}
}


#endif
