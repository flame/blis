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

void unpackb_nr48_bf16bf16f32of32_row_major
	(
	  const bfloat16* b,
	  bfloat16*       unpack_b_buffer,
	  const dim_t     KC,
	  dim_t           ldb
	)
{
	dim_t NR1 = 32;

	// Used for permuting the mm512i elements.
	__m512i selector_even = _mm512_set_epi16( 0x1E,  0x1C,  0x1A,  0x18,  0x16,  0x14,  0x12,  0x10,  0xE, 0xC, 0xA, 0x8, 0x6, 0x4, 0x2, 0x0,
	                                          0x3E, 0x3C, 0x3A, 0x38, 0x36, 0x34, 0x32, 0x30, 0x2E, 0x2C, 0x2A, 0x28, 0x26, 0x24, 0x22, 0x20);

	__m512i selector_odd = _mm512_set_epi16( 0x1F,  0x1D,  0x1B,  0x19,  0x17,  0x15,  0x13,  0x11,  0xF, 0xD, 0xB, 0x9, 0x7, 0x5, 0x3, 0x1,
	                                          0x3F, 0x3D, 0x3B, 0x39, 0x37, 0x35, 0x33, 0x31, 0x2F, 0x2D, 0x2B, 0x29, 0x27, 0x25, 0x23, 0x21);

	__m512i a0, a01, b0;
	__m512i c0, d0, c01;

	dim_t k_full_pieces_blks = KC / 2;
	dim_t k_full_pieces = k_full_pieces_blks * 2;
	dim_t k_partial_pieces = KC % 2;

	dim_t kr_new = 0;

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 2 )
	{
		// First 2x32 elements
		a0 = _mm512_loadu_si512( b + ( ( kr_new + 0 ) * NR1 ) );
		b0 = _mm512_loadu_si512( b + ( ( kr_new + 1 ) * NR1 ) );

		a01 = _mm512_permutex2var_epi16( b0, selector_even, a0 );
		b0  = _mm512_permutex2var_epi16( b0, selector_odd, a0 );

		_mm512_storeu_si512( unpack_b_buffer + ( ldb * ( kr + 0 ) ), a01 );
		_mm512_storeu_si512( unpack_b_buffer + ( ldb * ( kr + 1 ) ), b0 );

		c0 = _mm512_loadu_si512( b + ( ( kr_new + 2 ) * NR1 ) );
		d0 = _mm512_setzero_si512();

		c01 = _mm512_permutex2var_epi16( d0, selector_even, c0 );
		d0  = _mm512_permutex2var_epi16( d0, selector_odd, c0 );

		_mm512_mask_storeu_epi16( unpack_b_buffer + ( ldb * ( kr + 0 ) ) + NR1, 0xFFFF, c01 );
		_mm512_mask_storeu_epi16( unpack_b_buffer + ( ldb * ( kr + 1 ) ) + NR1, 0xFFFF, d0 );

		kr_new += 3;
	}

	if( k_partial_pieces > 0 )
	{
		// First 2x32 elements
		a0 = _mm512_loadu_si512( b + ( ( kr_new + 0 ) * NR1 ) );
		b0 = _mm512_loadu_si512( b + ( ( kr_new + 1 ) * NR1 ) );

		a01 = _mm512_permutex2var_epi16( b0, selector_even, a0 );

		_mm512_storeu_si512( unpack_b_buffer + ( ldb * ( k_full_pieces + 0 ) ), a01 );

		c0 = _mm512_loadu_si512( b + ( ( kr_new + 2 ) * NR1 ) );
		c01 = _mm512_permutex2var_epi16( c0, selector_even, c0 );

		_mm512_mask_storeu_epi16( unpack_b_buffer + ( ldb * ( k_full_pieces + 0 ) ) + NR1, 0xFFFF, c01 );
	}
}
void unpackb_nr32_bf16bf16f32of32_row_major
	(
	  const bfloat16* b,
	  bfloat16*       unpack_b_buffer,
	  const dim_t     KC,
	  dim_t           ldb
	)
{
	dim_t NR = 32;

	// Used for permuting the mm512i elements.
	__m512i selector_even = _mm512_set_epi16( 0x1E,  0x1C,  0x1A,  0x18,  0x16,  0x14,  0x12,  0x10,  0xE, 0xC, 0xA, 0x8, 0x6, 0x4, 0x2, 0x0,
	                                          0x3E, 0x3C, 0x3A, 0x38, 0x36, 0x34, 0x32, 0x30, 0x2E, 0x2C, 0x2A, 0x28, 0x26, 0x24, 0x22, 0x20);

	__m512i selector_odd = _mm512_set_epi16( 0x1F,  0x1D,  0x1B,  0x19,  0x17,  0x15,  0x13,  0x11,  0xF, 0xD, 0xB, 0x9, 0x7, 0x5, 0x3, 0x1,
	                                          0x3F, 0x3D, 0x3B, 0x39, 0x37, 0x35, 0x33, 0x31, 0x2F, 0x2D, 0x2B, 0x29, 0x27, 0x25, 0x23, 0x21);

	dim_t k_full_pieces_blks = KC / 2;
	dim_t k_full_pieces = k_full_pieces_blks * 2;
	dim_t k_partial_pieces = KC % 2;

	__m512i a0, c0;
	__m512i a01;

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 2 )
	{
		// Rearrange for dpbf16_ps, read 2 rows from B with 64 elements in each row.
		a0 = _mm512_loadu_si512( b + ( ( kr + 0 ) * NR ) );
		c0 = _mm512_loadu_si512( b + ( ( kr + 1 ) * NR ) );

		a01 = _mm512_permutex2var_epi16( c0, selector_even, a0 );
		c0  = _mm512_permutex2var_epi16( c0, selector_odd, a0 );

		// Store to unpack buffer
		_mm512_storeu_si512( unpack_b_buffer + ( ldb * ( kr + 0 ) ), a01 );
		_mm512_storeu_si512( unpack_b_buffer + ( ldb * ( kr + 1 ) ), c0 );

	}
	if( k_partial_pieces > 0 )
	{
		a0 = _mm512_loadu_si512( b + ( ( k_full_pieces + 0 ) * NR ) );
		c0 = _mm512_loadu_si512( b + ( ( k_full_pieces + 1 ) * NR ) );

		a0 = _mm512_permutex2var_epi16( c0, selector_even, a0 );

		// Store to unpack buffer
		_mm512_storeu_si512( unpack_b_buffer + ( ldb * ( k_full_pieces + 0 ) ), a0 );
	}
}
void unpackb_nr16_bf16bf16f32of32_row_major
	(
	  const bfloat16* b,
	  bfloat16*       unpack_b_buffer,
	  const dim_t     KC,
	  dim_t           ldb
	)
{
	dim_t NR = 16;

	// Used for permuting the mm512i elements.
	__m512i selector_even = _mm512_set_epi16( 0x1E,  0x1C,  0x1A,  0x18,  0x16,  0x14,  0x12,  0x10,  0xE, 0xC, 0xA, 0x8, 0x6, 0x4, 0x2, 0x0,
	                                          0x3E, 0x3C, 0x3A, 0x38, 0x36, 0x34, 0x32, 0x30, 0x2E, 0x2C, 0x2A, 0x28, 0x26, 0x24, 0x22, 0x20);

	__m512i selector_odd = _mm512_set_epi16( 0x1F,  0x1D,  0x1B,  0x19,  0x17,  0x15,  0x13,  0x11,  0xF, 0xD, 0xB, 0x9, 0x7, 0x5, 0x3, 0x1,
	                                          0x3F, 0x3D, 0x3B, 0x39, 0x37, 0x35, 0x33, 0x31, 0x2F, 0x2D, 0x2B, 0x29, 0x27, 0x25, 0x23, 0x21);

	__m512i a0;
	__m512i c0;
	__m512i a01;

	dim_t k_full_pieces_blks = KC / 2;
	dim_t k_full_pieces = k_full_pieces_blks * 2;
	dim_t k_partial_pieces = KC % 2;

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 2 )
	{
		// Rearrange for dpbf16_ps, read 2 rows from B with 16 elements in each row.
		a0 = _mm512_loadu_si512( b + ( ( kr + 0 ) * NR ) );

		a01 = _mm512_permutex2var_epi16( a0, selector_even, a0 );
		c0  = _mm512_permutex2var_epi16( a0, selector_odd, a0 );

		// Store to unpack buffer
		_mm512_mask_storeu_epi16( unpack_b_buffer + ( ldb * ( kr + 0 ) ), 0xFFFF, a01 );
		_mm512_mask_storeu_epi16( unpack_b_buffer + ( ldb * ( kr + 1 ) ), 0xFFFF, c0 );
	}
	if( k_partial_pieces > 0 )
	{
		a0 = _mm512_loadu_si512( b + ( ( k_full_pieces + 0 ) * NR ) );

		a0 = _mm512_permutex2var_epi16( a0, selector_even, a0 );

		// Store to unpack buffer
		_mm512_mask_storeu_epi16( unpack_b_buffer + ( ldb * ( k_full_pieces + 0 ) ), 0xFFFF, a0 );
	}
}
void unpackb_nrlt16_bf16bf16f32of32_row_major
	(
	  const bfloat16* b,
	  bfloat16*       unpack_b_buffer,
	  const dim_t     KC,
	  dim_t           ldb,
	  dim_t           n0_partial_rem
	)
{
	dim_t NR = 16;

	// Used for permuting the mm512i elements.
	__m512i selector_even = _mm512_set_epi16( 0x1E,  0x1C,  0x1A,  0x18,  0x16,  0x14,  0x12,  0x10,  0xE, 0xC, 0xA, 0x8, 0x6, 0x4, 0x2, 0x0,
	                                          0x3E, 0x3C, 0x3A, 0x38, 0x36, 0x34, 0x32, 0x30, 0x2E, 0x2C, 0x2A, 0x28, 0x26, 0x24, 0x22, 0x20);

	__m512i selector_odd = _mm512_set_epi16( 0x1F,  0x1D,  0x1B,  0x19,  0x17,  0x15,  0x13,  0x11,  0xF, 0xD, 0xB, 0x9, 0x7, 0x5, 0x3, 0x1,
	                                          0x3F, 0x3D, 0x3B, 0x39, 0x37, 0x35, 0x33, 0x31, 0x2F, 0x2D, 0x2B, 0x29, 0x27, 0x25, 0x23, 0x21);

	__m512i a0;
	__m512i c0;
	__m512i a01;

	__mmask32 store_mask = _cvtu32_mask32( 0xFFFF >> ( 16 - n0_partial_rem ) );

	dim_t k_full_pieces_blks = KC / 2;
	dim_t k_full_pieces = k_full_pieces_blks * 2;
	dim_t k_partial_pieces = KC % 2;

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 2 )
	{
		// Rearrange for dpbf16_ps, read 2 rows from B with 16 elements in each row.
		a0 = _mm512_loadu_si512( b + ( ( kr + 0 ) * NR ) );

		a01 = _mm512_permutex2var_epi16( a0, selector_even, a0 );
		c0  = _mm512_permutex2var_epi16( a0, selector_odd, a0 );

		// Store to unpack buffer
		_mm512_mask_storeu_epi16( unpack_b_buffer + ( ldb * ( kr + 0 ) ), store_mask, a01 );
		_mm512_mask_storeu_epi16( unpack_b_buffer + ( ldb * ( kr + 1 ) ), store_mask, c0 );
	}
	if( k_partial_pieces > 0 )
	{
		a0 = _mm512_loadu_si512( b + ( ( k_full_pieces + 0 ) * NR ) );

		a0 = _mm512_permutex2var_epi16( a0, selector_even, a0 );

		// Store to unpack buffer
		_mm512_mask_storeu_epi16( unpack_b_buffer + ( ldb * ( k_full_pieces + 0 ) ), store_mask, a0 );
	}
}

void unpackb_nr64_bf16bf16f32of32_row_major
	(
	  const bfloat16* b,
	  bfloat16*       unpack_b_buffer,
	  const dim_t     NC,
	  const dim_t     KC,
	  dim_t           ldb
	)
{
	dim_t NR = 64;

	// Used for permuting the mm512i elements.
	__m512i selector_even = _mm512_set_epi16( 0x1E,  0x1C,  0x1A,  0x18,  0x16,  0x14,  0x12,  0x10,  0xE, 0xC, 0xA, 0x8, 0x6, 0x4, 0x2, 0x0,
	                                          0x3E, 0x3C, 0x3A, 0x38, 0x36, 0x34, 0x32, 0x30, 0x2E, 0x2C, 0x2A, 0x28, 0x26, 0x24, 0x22, 0x20);

	__m512i selector_odd = _mm512_set_epi16( 0x1F,  0x1D,  0x1B,  0x19,  0x17,  0x15,  0x13,  0x11,  0xF, 0xD, 0xB, 0x9, 0x7, 0x5, 0x3, 0x1,
	                                          0x3F, 0x3D, 0x3B, 0x39, 0x37, 0x35, 0x33, 0x31, 0x2F, 0x2D, 0x2B, 0x29, 0x27, 0x25, 0x23, 0x21);

	dim_t n_full_pieces = NC / NR;
	dim_t n_full_pieces_loop_limit = n_full_pieces * NR;
	dim_t n_partial_pieces = NC % NR;

	dim_t k_full_pieces_blks = KC / 2;
	dim_t k_full_pieces = k_full_pieces_blks * 2;
	dim_t k_partial_pieces = KC % 2;

	// KC when not multiple of 2 will have padding to make it multiple of 2 in packed buffer.
	dim_t KC_updated = KC;
	if ( k_partial_pieces > 0 )
	{
		KC_updated += ( 2 - k_partial_pieces );
	}

	__m512i a0, b0, c0, d0;
	__m512i a01, c01;

	for ( dim_t jc = 0; jc < n_full_pieces_loop_limit; jc += NR )
	{
		for ( dim_t kr = 0; kr < k_full_pieces; kr += 2 )
		{
			// Rearrange for dpbf16_ps, read 2 rows from B with 64 elements in each row.
			a0 = _mm512_loadu_si512( b + ( jc * KC_updated ) + ( ( kr + 0 ) * NR ) );
			b0 = _mm512_loadu_si512( b + ( jc * KC_updated ) + ( ( kr + 0 ) * NR ) + 32 );
			c0 = _mm512_loadu_si512( b + ( jc * KC_updated ) + ( ( kr + 1 ) * NR ) );
			d0 = _mm512_loadu_si512( b + ( jc * KC_updated ) + ( ( kr + 1 ) * NR ) + 32 );

			a01 = _mm512_permutex2var_epi16( b0, selector_even, a0 );
			b0  = _mm512_permutex2var_epi16( b0, selector_odd, a0 );

			c01 = _mm512_permutex2var_epi16( d0, selector_even, c0 );
			d0  = _mm512_permutex2var_epi16( d0, selector_odd, c0 );

			// Store to unpack buffer
			_mm512_storeu_si512( unpack_b_buffer + ( ldb * ( kr + 0 ) ) + jc, a01 );
			_mm512_storeu_si512( unpack_b_buffer + ( ldb * ( kr + 0 ) ) + jc + 32, c01 );
			_mm512_storeu_si512( unpack_b_buffer + ( ldb * ( kr + 1 ) ) + jc, b0 );
			_mm512_storeu_si512( unpack_b_buffer + ( ldb * ( kr + 1 ) ) + jc + 32, d0 );

		}
		if( k_partial_pieces > 0 )
		{
			a0 = _mm512_loadu_si512( b + ( jc * KC_updated ) + ( ( k_full_pieces + 0 ) * NR ) );
			b0 = _mm512_loadu_si512( b + ( jc * KC_updated ) + ( ( k_full_pieces + 0 ) * NR ) + 32 );
			c0 = _mm512_loadu_si512( b + ( jc * KC_updated ) + ( ( k_full_pieces + 1 ) * NR ) );
			d0 = _mm512_loadu_si512( b + ( jc * KC_updated ) + ( ( k_full_pieces + 1 ) * NR ) + 32 );

			a01 = _mm512_permutex2var_epi16( b0, selector_even, a0 );

			c01 = _mm512_permutex2var_epi16( d0, selector_even, c0 );

			// Store to unpack buffer
			_mm512_storeu_si512( unpack_b_buffer + ( ldb * ( k_full_pieces + 0 ) ) + jc, a01 );
			_mm512_storeu_si512( unpack_b_buffer + ( ldb * ( k_full_pieces + 0 ) ) + jc + 32, c01 );
		}
	}

	if( n_partial_pieces > 0 )
	{
		dim_t n0_partial_rem = n_partial_pieces % 16;
		dim_t n0_partial_unpack = 0;

		// Split into multiple smaller fringe kernels, so as to maximize
		// vectorization after packing. Any n0 < NR(64) can be expressed
		// as n0 = 48 + n` / n0 = 32 + n` / n0 = 16 + n`, where n` < 16.
		dim_t n0_48 = n_partial_pieces / 48;
		dim_t n0_32 = n_partial_pieces / 32;
		dim_t n0_16 = n_partial_pieces / 16;

		if ( n0_48 == 1 )
		{
			unpackb_nr48_bf16bf16f32of32_row_major
			    (
			     ( b + ( n_full_pieces_loop_limit * KC_updated ) ),
			     ( unpack_b_buffer + n_full_pieces_loop_limit ), KC, ldb
			    );

			n0_partial_unpack = 48;
		}
		else if ( n0_32 == 1 )
		{
			unpackb_nr32_bf16bf16f32of32_row_major
			    (
			     ( b + ( n_full_pieces_loop_limit * KC_updated ) ),
			     ( unpack_b_buffer + n_full_pieces_loop_limit ), KC, ldb
			    );

			n0_partial_unpack = 32;
		}
		else if ( n0_16 == 1 )
		{
			unpackb_nr16_bf16bf16f32of32_row_major
			    (
			     ( b + ( n_full_pieces_loop_limit * KC_updated ) ),
			     ( unpack_b_buffer + n_full_pieces_loop_limit ), KC, ldb
			    );

			n0_partial_unpack = 16;
		}

		if ( n0_partial_rem > 0 )
		{
			unpackb_nrlt16_bf16bf16f32of32_row_major
			    (
			     ( b + ( n_full_pieces_loop_limit * KC_updated ) +
				   ( n0_partial_unpack * KC_updated ) ),
				 ( unpack_b_buffer + n_full_pieces_loop_limit + n0_partial_unpack ), KC, ldb,
				 n0_partial_rem
			    );
		}
	}
}

#define STORE_16_COLS_AVX512 \
	_mm512_storeu_si512( unpack_b_buffer + ( ldb * ( jr + 0 ) ) + kr, a_reg[0] ); \
	_mm512_storeu_si512( unpack_b_buffer + ( ldb * ( jr + 1 ) ) + kr, a_reg[1] ); \
	_mm512_storeu_si512( unpack_b_buffer + ( ldb * ( jr + 2 ) ) + kr, a_reg[2] ); \
	_mm512_storeu_si512( unpack_b_buffer + ( ldb * ( jr + 3 ) ) + kr, a_reg[3] ); \
	_mm512_storeu_si512( unpack_b_buffer + ( ldb * ( jr + 4 ) ) + kr, a_reg[4] ); \
	_mm512_storeu_si512( unpack_b_buffer + ( ldb * ( jr + 5 ) ) + kr, a_reg[5] ); \
	_mm512_storeu_si512( unpack_b_buffer + ( ldb * ( jr + 6 ) ) + kr, a_reg[6] ); \
	_mm512_storeu_si512( unpack_b_buffer + ( ldb * ( jr + 7 ) ) + kr, a_reg[7] ); \
	_mm512_storeu_si512( unpack_b_buffer + ( ldb * ( jr + 8 ) ) + kr, a_reg[8] ); \
	_mm512_storeu_si512( unpack_b_buffer + ( ldb * ( jr + 9 ) ) + kr, a_reg[9] ); \
	_mm512_storeu_si512( unpack_b_buffer + ( ldb * ( jr + 10 ) ) + kr, a_reg[10] ); \
	_mm512_storeu_si512( unpack_b_buffer + ( ldb * ( jr + 11 ) ) + kr, a_reg[11] ); \
	_mm512_storeu_si512( unpack_b_buffer + ( ldb * ( jr + 12 ) ) + kr, a_reg[12] ); \
	_mm512_storeu_si512( unpack_b_buffer + ( ldb * ( jr + 13 ) ) + kr, a_reg[13] ); \
	_mm512_storeu_si512( unpack_b_buffer + ( ldb * ( jr + 14 ) ) + kr, a_reg[14] ); \
	_mm512_storeu_si512( unpack_b_buffer + ( ldb * ( jr + 15 ) ) + kr, a_reg[15] );

#define MASK_STORE_16_COLS_AVX512( mask ) \
	_mm512_mask_storeu_epi16( unpack_b_buffer + ( ldb * ( jr + 0 ) ) + kr, mask, a_reg[0] ); \
	_mm512_mask_storeu_epi16( unpack_b_buffer + ( ldb * ( jr + 1 ) ) + kr, mask, a_reg[1] ); \
	_mm512_mask_storeu_epi16( unpack_b_buffer + ( ldb * ( jr + 2 ) ) + kr, mask, a_reg[2] ); \
	_mm512_mask_storeu_epi16( unpack_b_buffer + ( ldb * ( jr + 3 ) ) + kr, mask, a_reg[3] ); \
	_mm512_mask_storeu_epi16( unpack_b_buffer + ( ldb * ( jr + 4 ) ) + kr, mask, a_reg[4] ); \
	_mm512_mask_storeu_epi16( unpack_b_buffer + ( ldb * ( jr + 5 ) ) + kr, mask, a_reg[5] ); \
	_mm512_mask_storeu_epi16( unpack_b_buffer + ( ldb * ( jr + 6 ) ) + kr, mask, a_reg[6] ); \
	_mm512_mask_storeu_epi16( unpack_b_buffer + ( ldb * ( jr + 7 ) ) + kr, mask, a_reg[7] ); \
	_mm512_mask_storeu_epi16( unpack_b_buffer + ( ldb * ( jr + 8 ) ) + kr, mask, a_reg[8] ); \
	_mm512_mask_storeu_epi16( unpack_b_buffer + ( ldb * ( jr + 9 ) ) + kr, mask, a_reg[9] ); \
	_mm512_mask_storeu_epi16( unpack_b_buffer + ( ldb * ( jr + 10 ) ) + kr, mask, a_reg[10] ); \
	_mm512_mask_storeu_epi16( unpack_b_buffer + ( ldb * ( jr + 11 ) ) + kr, mask, a_reg[11] ); \
	_mm512_mask_storeu_epi16( unpack_b_buffer + ( ldb * ( jr + 12 ) ) + kr, mask, a_reg[12] ); \
	_mm512_mask_storeu_epi16( unpack_b_buffer + ( ldb * ( jr + 13 ) ) + kr, mask, a_reg[13] ); \
	_mm512_mask_storeu_epi16( unpack_b_buffer + ( ldb * ( jr + 14 ) ) + kr, mask, a_reg[14] ); \
	_mm512_mask_storeu_epi16( unpack_b_buffer + ( ldb * ( jr + 15 ) ) + kr, mask, a_reg[15] ); \



#define UNPACKHILO32_AVX512 \
	b_reg[0] = _mm512_unpacklo_epi32(a_reg[0], a_reg[1]); \
	b_reg[2] = _mm512_unpacklo_epi32(a_reg[2], a_reg[3]); \
	b_reg[4] = _mm512_unpacklo_epi32(a_reg[4], a_reg[5]); \
	b_reg[6] = _mm512_unpacklo_epi32(a_reg[6], a_reg[7]); \
	b_reg[8] = _mm512_unpacklo_epi32(a_reg[8], a_reg[9]); \
	b_reg[10] = _mm512_unpacklo_epi32(a_reg[10], a_reg[11]); \
	b_reg[12] = _mm512_unpacklo_epi32(a_reg[12], a_reg[13]); \
	b_reg[14] = _mm512_unpacklo_epi32(a_reg[14], a_reg[15]); \
\
	b_reg[1] = _mm512_unpackhi_epi32(a_reg[0], a_reg[1]); \
	b_reg[3] = _mm512_unpackhi_epi32(a_reg[2], a_reg[3]); \
	b_reg[5] = _mm512_unpackhi_epi32(a_reg[4], a_reg[5]); \
	b_reg[7] = _mm512_unpackhi_epi32(a_reg[6], a_reg[7]); \
	b_reg[9] = _mm512_unpackhi_epi32(a_reg[8], a_reg[9]); \
	b_reg[11] = _mm512_unpackhi_epi32(a_reg[10], a_reg[11]); \
	b_reg[13] = _mm512_unpackhi_epi32(a_reg[12], a_reg[13]); \
	b_reg[15] = _mm512_unpackhi_epi32(a_reg[14], a_reg[15]);

#define UNPACKHILO64_AVX512 \
	a_reg[0] = _mm512_unpacklo_epi64(b_reg[0], b_reg[2]); \
	a_reg[1] = _mm512_unpacklo_epi64(b_reg[4], b_reg[6]); \
	a_reg[2] = _mm512_unpacklo_epi64(b_reg[8], b_reg[10]); \
	a_reg[3] = _mm512_unpacklo_epi64(b_reg[12], b_reg[14]); \
	a_reg[4] = _mm512_unpacklo_epi64(b_reg[1], b_reg[3]); \
	a_reg[5] = _mm512_unpacklo_epi64(b_reg[5], b_reg[7]); \
	a_reg[6] = _mm512_unpacklo_epi64(b_reg[9], b_reg[11]); \
	a_reg[7] = _mm512_unpacklo_epi64(b_reg[13], b_reg[15]); \
\
	a_reg[8] = _mm512_unpackhi_epi64(b_reg[0], b_reg[2]); \
	a_reg[9] = _mm512_unpackhi_epi64(b_reg[4], b_reg[6]); \
	a_reg[10] = _mm512_unpackhi_epi64(b_reg[8], b_reg[10]); \
	a_reg[11] = _mm512_unpackhi_epi64(b_reg[12], b_reg[14]); \
	a_reg[12] = _mm512_unpackhi_epi64(b_reg[1], b_reg[3]); \
	a_reg[13] = _mm512_unpackhi_epi64(b_reg[5], b_reg[7]); \
	a_reg[14] = _mm512_unpackhi_epi64(b_reg[9], b_reg[11]); \
	a_reg[15] = _mm512_unpackhi_epi64(b_reg[13], b_reg[15]);

#define PERMUTEX2_VAR64_AVX512 \
	b_reg[0] = _mm512_permutex2var_epi64(a_reg[0], selector1, a_reg[1]); \
	b_reg[1] = _mm512_permutex2var_epi64(a_reg[2], selector1, a_reg[3]); \
	b_reg[2] = _mm512_permutex2var_epi64(a_reg[8], selector1, a_reg[9]); \
	b_reg[3] = _mm512_permutex2var_epi64(a_reg[10], selector1, a_reg[11]); \
	b_reg[4] = _mm512_permutex2var_epi64(a_reg[4], selector1, a_reg[5]); \
	b_reg[5] = _mm512_permutex2var_epi64(a_reg[6], selector1, a_reg[7]); \
	b_reg[6] = _mm512_permutex2var_epi64(a_reg[12], selector1, a_reg[13]); \
	b_reg[7] = _mm512_permutex2var_epi64(a_reg[14], selector1, a_reg[15]); \
	b_reg[8] = _mm512_permutex2var_epi64(a_reg[0], selector2, a_reg[1]); \
	b_reg[9] = _mm512_permutex2var_epi64(a_reg[2], selector2, a_reg[3]); \
	b_reg[10] = _mm512_permutex2var_epi64(a_reg[8], selector2, a_reg[9]); \
	b_reg[11] = _mm512_permutex2var_epi64(a_reg[10], selector2, a_reg[11]); \
	b_reg[12] = _mm512_permutex2var_epi64(a_reg[4], selector2, a_reg[5]); \
	b_reg[13] = _mm512_permutex2var_epi64(a_reg[6], selector2, a_reg[7]); \
	b_reg[14] = _mm512_permutex2var_epi64(a_reg[12], selector2, a_reg[13]); \
	b_reg[15] = _mm512_permutex2var_epi64(a_reg[14], selector2, a_reg[15]);

#define SHUFFLE64x2_AVX512 \
	a_reg[0] = _mm512_shuffle_i64x2(b_reg[0], b_reg[1], 0x44); \
	a_reg[1] = _mm512_shuffle_i64x2(b_reg[2], b_reg[3], 0x44); \
	a_reg[2] = _mm512_shuffle_i64x2(b_reg[4], b_reg[5], 0x44); \
	a_reg[3] = _mm512_shuffle_i64x2(b_reg[6], b_reg[7], 0x44); \
	a_reg[4] = _mm512_shuffle_i64x2(b_reg[8], b_reg[9], 0x44); \
	a_reg[5] = _mm512_shuffle_i64x2(b_reg[10], b_reg[11], 0x44); \
	a_reg[6] = _mm512_shuffle_i64x2(b_reg[12], b_reg[13], 0x44); \
	a_reg[7] = _mm512_shuffle_i64x2(b_reg[14], b_reg[15], 0x44); \
	a_reg[8] = _mm512_shuffle_i64x2(b_reg[0], b_reg[1], 0xEE); \
	a_reg[9] = _mm512_shuffle_i64x2(b_reg[2], b_reg[3], 0xEE); \
	a_reg[10] = _mm512_shuffle_i64x2(b_reg[4], b_reg[5], 0xEE); \
	a_reg[11] = _mm512_shuffle_i64x2(b_reg[6], b_reg[7], 0xEE); \
	a_reg[12] = _mm512_shuffle_i64x2(b_reg[8], b_reg[9], 0xEE); \
	a_reg[13] = _mm512_shuffle_i64x2(b_reg[10], b_reg[11], 0xEE); \
	a_reg[14] = _mm512_shuffle_i64x2(b_reg[12], b_reg[13], 0xEE); \
	a_reg[15] = _mm512_shuffle_i64x2(b_reg[14], b_reg[15], 0xEE);


void unpackb_nrlt16_bf16bf16f32of32_col_major
	(
	  const bfloat16* b,
	  bfloat16*       unpack_b_buffer,
	  const dim_t     KC,
	  dim_t           ldb,
	  dim_t           n0_partial_rem
	)
{
		dim_t NR = 16;

	// Used for permuting the mm512i elements for use in dpbf16_ps instruction.
	__m512i selector1 = _mm512_setr_epi64( 0x0, 0x1, 0x8, 0x9, 0x4, 0x5, 0xC, 0xD );
	__m512i selector2 = _mm512_setr_epi64( 0x2, 0x3, 0xA, 0xB, 0x6, 0x7, 0xE, 0xF );

	__m512i a_reg[16];
	__m512i b_reg[16];

	// These registers are set with zeroes to avoid compiler warnings
	// To-DO: TO be removed when pack code is optimized for fringe cases.
	a_reg[0] = _mm512_setzero_si512();
	a_reg[1] = _mm512_setzero_si512();
	a_reg[2] = _mm512_setzero_si512();
	a_reg[3] = _mm512_setzero_si512();
	a_reg[4] = _mm512_setzero_si512();
	a_reg[5] = _mm512_setzero_si512();
	a_reg[6] = _mm512_setzero_si512();
	a_reg[7] = _mm512_setzero_si512();
	a_reg[8] = _mm512_setzero_si512();
	a_reg[9] = _mm512_setzero_si512();
	a_reg[10] = _mm512_setzero_si512();
	a_reg[11] = _mm512_setzero_si512();
	a_reg[12] = _mm512_setzero_si512();
	a_reg[13] = _mm512_setzero_si512();
	a_reg[14] = _mm512_setzero_si512();
	a_reg[15] = _mm512_setzero_si512();

	dim_t kr = 0, jr = 0;
	for ( kr = 0; ( kr + 31 ) < KC; kr += 32 )
	{
		a_reg[0] = _mm512_loadu_si512( b + ( ( kr + 0 ) * NR ) );
		a_reg[1] = _mm512_loadu_si512( b + ( ( kr + 2 ) * NR ) );
		a_reg[2] = _mm512_loadu_si512( b + ( ( kr + 4 ) * NR ) );
		a_reg[3] = _mm512_loadu_si512( b + ( ( kr + 6 ) * NR ) );
		a_reg[4] = _mm512_loadu_si512( b + ( ( kr + 8 ) * NR ) );
		a_reg[5] = _mm512_loadu_si512( b + ( ( kr + 10 ) * NR ) );
		a_reg[6] = _mm512_loadu_si512( b + ( ( kr + 12 ) * NR ) );
		a_reg[7] = _mm512_loadu_si512( b + ( ( kr + 14 ) * NR ) );
		a_reg[8] = _mm512_loadu_si512( b + ( ( kr + 16 ) * NR ) );
		a_reg[9] = _mm512_loadu_si512( b + ( ( kr + 18 ) * NR ) );
		a_reg[10] = _mm512_loadu_si512( b + ( ( kr + 20 ) * NR ) );
		a_reg[11] = _mm512_loadu_si512( b + ( ( kr + 22 ) * NR ) );
		a_reg[12] = _mm512_loadu_si512( b + ( ( kr + 24 ) * NR ) );
		a_reg[13] = _mm512_loadu_si512( b + ( ( kr + 26 ) * NR ) );
		a_reg[14] = _mm512_loadu_si512( b + ( ( kr + 28 ) * NR ) );
		a_reg[15] = _mm512_loadu_si512( b + ( ( kr + 30 ) * NR ) );

		UNPACKHILO32_AVX512
		UNPACKHILO64_AVX512
		PERMUTEX2_VAR64_AVX512
		SHUFFLE64x2_AVX512

		for( jr = 0; jr < n0_partial_rem; jr += 1 )
		{
			_mm512_storeu_si512( unpack_b_buffer + ( ldb * ( jr + 0 ) ) + kr, a_reg[jr] );
		}
	}

	for ( ; ( kr + 15 ) < KC; kr += 16 )
	{
		a_reg[0] = _mm512_loadu_si512( b + ( ( kr + 0 ) * NR ) );
		a_reg[1] = _mm512_loadu_si512( b + ( ( kr + 2 ) * NR ) );
		a_reg[2] = _mm512_loadu_si512( b + ( ( kr + 4 ) * NR ) );
		a_reg[3] = _mm512_loadu_si512( b + ( ( kr + 6 ) * NR ) );
		a_reg[4] = _mm512_loadu_si512( b + ( ( kr + 8 ) * NR ) );
		a_reg[5] = _mm512_loadu_si512( b + ( ( kr + 10 ) * NR ) );
		a_reg[6] = _mm512_loadu_si512( b + ( ( kr + 12 ) * NR ) );
		a_reg[7] = _mm512_loadu_si512( b + ( ( kr + 14 ) * NR ) );

		UNPACKHILO32_AVX512
		UNPACKHILO64_AVX512
		PERMUTEX2_VAR64_AVX512
		SHUFFLE64x2_AVX512

		for( jr = 0; jr  < n0_partial_rem; jr += 1 )
		{
			_mm512_mask_storeu_epi16( unpack_b_buffer + ( ldb * ( jr + 0 ) ) + kr,  0xFFFF, a_reg[jr]  );
		}
	}

	for ( ; ( kr + 7 ) < KC; kr += 8 )
	{
		a_reg[0] = _mm512_loadu_si512( b + ( ( kr + 0 ) * NR ) );
		a_reg[1] = _mm512_loadu_si512( b + ( ( kr + 2 ) * NR ) );
		a_reg[2] = _mm512_loadu_si512( b + ( ( kr + 4 ) * NR ) );
		a_reg[3] = _mm512_loadu_si512( b + ( ( kr + 6 ) * NR ) );

		UNPACKHILO32_AVX512
		UNPACKHILO64_AVX512
		PERMUTEX2_VAR64_AVX512
		SHUFFLE64x2_AVX512

		for( jr = 0; jr  < n0_partial_rem; jr += 1 )
		{
			_mm512_mask_storeu_epi16( unpack_b_buffer + ( ldb * ( jr + 0 ) ) + kr,  0xFF, a_reg[jr]  );
		}
	}

	for ( ; (kr+3) < KC; kr += 4 )
	{
		a_reg[0] = _mm512_loadu_si512( b + ( ( kr + 0 ) * NR ) );
		a_reg[1] = _mm512_loadu_si512( b + ( ( kr + 2 ) * NR ) );

		UNPACKHILO32_AVX512
		UNPACKHILO64_AVX512
		PERMUTEX2_VAR64_AVX512
		SHUFFLE64x2_AVX512

		for( jr = 0; jr  < n0_partial_rem; jr += 1 )
		{
			_mm512_mask_storeu_epi16( unpack_b_buffer + ( ldb * ( jr + 0 ) ) + kr,  0xF, a_reg[jr]  );
		}
	}

	for ( ; ( kr + 1 ) < KC; kr += 2 )
	{
		a_reg[0] = _mm512_loadu_si512( b + ( ( kr + 0 ) * NR ) );

		UNPACKHILO32_AVX512
		UNPACKHILO64_AVX512
		PERMUTEX2_VAR64_AVX512
		SHUFFLE64x2_AVX512

		for( jr = 0; jr  < n0_partial_rem; jr += 1 )
		{
			_mm512_mask_storeu_epi16( unpack_b_buffer + ( ldb * ( jr + 0 ) ) + kr,  0x03, a_reg[jr]  );
		}
	}

	for ( ; kr < KC; kr += 1 )
	{
		a_reg[0] = _mm512_loadu_si512( b + ( ( kr + 0 ) * NR ) );

		UNPACKHILO32_AVX512
		UNPACKHILO64_AVX512
		PERMUTEX2_VAR64_AVX512
		SHUFFLE64x2_AVX512

		for( jr = 0; jr  < n0_partial_rem; jr += 1 )
		{
			_mm512_mask_storeu_epi16( unpack_b_buffer + ( ldb * ( jr + 0 ) ) + kr,  0x01, a_reg[jr]  );
		}
	}
}

void unpackb_nr_mult_16_bf16bf16f32of32_col_major
	(
	  const bfloat16* b,
	  bfloat16*       unpack_b_buffer,
	  const dim_t     NR,
	  const dim_t     KC,
	  dim_t           ldb
	)
{
	// Used for permuting the mm512i elements for use in dpbf16_ps instruction.
	__m512i selector1 = _mm512_setr_epi64( 0x0, 0x1, 0x8, 0x9, 0x4, 0x5, 0xC, 0xD );
	__m512i selector2 = _mm512_setr_epi64( 0x2, 0x3, 0xA, 0xB, 0x6, 0x7, 0xE, 0xF );

	__m512i a_reg[16];
	__m512i b_reg[16];

	// These registers are set with zeroes to avoid compiler warnings
	// To-DO: TO be removed when pack code is optimized for fringe cases.
	a_reg[0] = _mm512_setzero_si512();
	a_reg[1] = _mm512_setzero_si512();
	a_reg[2] = _mm512_setzero_si512();
	a_reg[3] = _mm512_setzero_si512();
	a_reg[4] = _mm512_setzero_si512();
	a_reg[5] = _mm512_setzero_si512();
	a_reg[6] = _mm512_setzero_si512();
	a_reg[7] = _mm512_setzero_si512();
	a_reg[8] = _mm512_setzero_si512();
	a_reg[9] = _mm512_setzero_si512();
	a_reg[10] = _mm512_setzero_si512();
	a_reg[11] = _mm512_setzero_si512();
	a_reg[12] = _mm512_setzero_si512();
	a_reg[13] = _mm512_setzero_si512();
	a_reg[14] = _mm512_setzero_si512();
	a_reg[15] = _mm512_setzero_si512();

	dim_t kr = 0;
	for ( kr = 0; ( kr + 31 ) < KC; kr += 32 )
	{
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			a_reg[0] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 0 ) * NR ) );
			a_reg[1] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 2 ) * NR ) );
			a_reg[2] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 4 ) * NR ) );
			a_reg[3] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 6 ) * NR ) );
			a_reg[4] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 8 ) * NR ) );
			a_reg[5] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 10 ) * NR ) );
			a_reg[6] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 12 ) * NR ) );
			a_reg[7] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 14 ) * NR ) );
			a_reg[8] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 16 ) * NR ) );
			a_reg[9] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 18 ) * NR ) );
			a_reg[10] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 20 ) * NR ) );
			a_reg[11] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 22 ) * NR ) );
			a_reg[12] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 24 ) * NR ) );
			a_reg[13] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 26 ) * NR ) );
			a_reg[14] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 28 ) * NR ) );
			a_reg[15] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 30 ) * NR ) );

			UNPACKHILO32_AVX512
			UNPACKHILO64_AVX512
			PERMUTEX2_VAR64_AVX512
			SHUFFLE64x2_AVX512
			STORE_16_COLS_AVX512
		}
	}

	for ( ; ( kr + 15 ) < KC; kr += 16 )
	{
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			// Rearrange for dpbf16_ps, read 2 rows from B with 64 elements in each row.

			a_reg[0] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 0 ) * NR ) );
			a_reg[1] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 2 ) * NR ) );
			a_reg[2] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 4 ) * NR ) );
			a_reg[3] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 6 ) * NR ) );
			a_reg[4] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 8 ) * NR ) );
			a_reg[5] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 10 ) * NR ) );
			a_reg[6] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 12 ) * NR ) );
			a_reg[7] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 14 ) * NR ) );

			UNPACKHILO32_AVX512
			UNPACKHILO64_AVX512
			PERMUTEX2_VAR64_AVX512
			SHUFFLE64x2_AVX512
			MASK_STORE_16_COLS_AVX512( 0xFFFF )
		}
	}

	for( ; ( kr +7 ) < KC; kr += 8 )
	{
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			// Rearrange for dpbf16_ps, read 2 rows from B with 64 elements in each row.

			a_reg[0] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 0 ) * NR ) );
			a_reg[1] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 2 ) * NR ) );
			a_reg[2] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 4 ) * NR ) );
			a_reg[3] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 6 ) * NR ) );

			UNPACKHILO32_AVX512
			UNPACKHILO64_AVX512
			PERMUTEX2_VAR64_AVX512
			SHUFFLE64x2_AVX512
			MASK_STORE_16_COLS_AVX512( 0xFF )
		}
	}

	for( ; ( kr +3 ) < KC; kr += 4 )
	{
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			// Rearrange for dpbf16_ps, read 2 rows from B with 64 elements in each row.
			a_reg[0] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 0 ) * NR ) );
			a_reg[1] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 2 ) * NR ) );

			UNPACKHILO32_AVX512
			UNPACKHILO64_AVX512
			PERMUTEX2_VAR64_AVX512
			SHUFFLE64x2_AVX512
			MASK_STORE_16_COLS_AVX512( 0xF )
		}
	}

	for( ; ( kr +1 ) < KC; kr += 2 )
	{
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			// Rearrange for dpbf16_ps, read 2 rows from B with 64 elements in each row.
			a_reg[0] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 0 ) * NR ) );

			UNPACKHILO32_AVX512
			UNPACKHILO64_AVX512
			PERMUTEX2_VAR64_AVX512
			SHUFFLE64x2_AVX512
			MASK_STORE_16_COLS_AVX512( 0x3 )
		}
	}

	for( ; kr < KC; kr += 1 )
	{
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			// Rearrange for dpbf16_ps, read 2 rows from B with 64 elements in each row.
			a_reg[0] = _mm512_loadu_si512( b + ( jr * 2 ) + ( ( kr + 0 ) * NR ) );

			UNPACKHILO32_AVX512
			UNPACKHILO64_AVX512
			PERMUTEX2_VAR64_AVX512
			SHUFFLE64x2_AVX512
			MASK_STORE_16_COLS_AVX512( 0x1 )
		}
	}
};

void unpackb_nr64_bf16bf16f32of32_col_major
	(
	  const bfloat16* b,
	  bfloat16*       unpack_b_buffer,
	  const dim_t     NC,
	  const dim_t     KC,
	  dim_t           ldb
	)
{
	dim_t NR = 64;

	dim_t n_full_pieces = NC / NR;
	dim_t n_full_pieces_loop_limit = n_full_pieces * NR;
	dim_t n_partial_pieces = NC % NR;

	dim_t k_partial_pieces = KC % 2;

	dim_t KC_updated = KC;
	if ( k_partial_pieces > 0 )
	{
		KC_updated += ( 2 - k_partial_pieces );
	}

	for ( dim_t jc = 0; jc < n_full_pieces_loop_limit; jc += NR )
	{
		unpackb_nr_mult_16_bf16bf16f32of32_col_major
			( b + (jc * KC_updated),
			  unpack_b_buffer + (jc * ldb), 64, KC, ldb
			);
	}

	if(n_partial_pieces > 0)
	{
		dim_t n0_partial_rem = n_partial_pieces % 16;
		dim_t n0_partial_pack = 0;

		// Split into multiple smaller fringe kernels, so as to maximize
		// vectorization after packing. Any n0 < NR(64) can be expressed
		// as n0 = 48 + n` / n0 = 32 + n` / n0 = 16 + n`, where n` < 16.
		dim_t n0_48 = n_partial_pieces / 48;
		dim_t n0_32 = n_partial_pieces / 32;
		dim_t n0_16 = n_partial_pieces / 16;

		if ( n0_48 == 1 )
		{
			unpackb_nr_mult_16_bf16bf16f32of32_col_major
				(
				 ( b + ( n_full_pieces_loop_limit * KC_updated ) ),
				 ( unpack_b_buffer + n_full_pieces_loop_limit * ldb ), 48, KC, ldb
				);

			n0_partial_pack = 48;
		}
		else if ( n0_32 == 1 )
		{
			unpackb_nr_mult_16_bf16bf16f32of32_col_major
				(
				 ( b + ( n_full_pieces_loop_limit * KC_updated ) ),
				 ( unpack_b_buffer + n_full_pieces_loop_limit * ldb ), 32, KC, ldb
				);

			n0_partial_pack = 32;
		}
		else if ( n0_16 == 1 )
		{
			unpackb_nr_mult_16_bf16bf16f32of32_col_major
				(
				 ( b + ( n_full_pieces_loop_limit * KC_updated ) ),
				 ( unpack_b_buffer + n_full_pieces_loop_limit * ldb ), 16, KC, ldb
				);

			n0_partial_pack = 16;
		}

		if ( n0_partial_rem > 0 )
		{
			unpackb_nrlt16_bf16bf16f32of32_col_major
				(
				 ( b + ( n_full_pieces_loop_limit * KC_updated ) +
				   ( n0_partial_pack * KC_updated ) ),
				 ( unpack_b_buffer + ( n_full_pieces_loop_limit + n0_partial_pack ) * ldb ), KC, ldb,
				 n0_partial_rem
				);
		}
	}
};

void unpackb_nr64_bf16bf16f32of32
	(
	  const bfloat16* b,
	  bfloat16*       unpack_b_buffer,
	  const dim_t	  NC,
	  const dim_t     KC,
	  dim_t           rs_b,
	  dim_t           cs_b
	)
{
	if( cs_b == 1 )
	{
		unpackb_nr64_bf16bf16f32of32_row_major( b, unpack_b_buffer, NC, KC, rs_b );
	}
	else
	{
		unpackb_nr64_bf16bf16f32of32_col_major( b, unpack_b_buffer, NC, KC, cs_b );
	}
}
#endif // BLIS_ADDON_LPGEMM
