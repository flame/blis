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


void packb_nr64_bf16bf16f32of32_row_major
     (
       bfloat16*       pack_b_buffer_bf16bf16f32of32,
       const bfloat16* b,
       const dim_t     ldb,
       const dim_t     NC,
       const dim_t     KC,
       dim_t*          rs_b,
       dim_t*          cs_b
     );

void packb_nr64_bf16bf16f32of32_col_major
     (
       bfloat16*       pack_b_buffer_bf16bf16f32of32,
       const bfloat16* b,
       const dim_t     ldb,
       const dim_t     NC,
       const dim_t     KC,
       dim_t*          rs_b,
       dim_t*          cs_b
     );

void packb_nrlt16_bf16bf16f32of32_row_major
    (
      bfloat16*       pack_b_buffer_bf16bf16f32of32,
      const bfloat16* b,
      const dim_t     ldb,
      const dim_t     KC,
      const dim_t     n0_partial_rem
    );

void packb_nr16_bf16bf16f32of32_row_major
    (
      bfloat16*       pack_b_buffer_bf16bf16f32of32,
      const bfloat16* b,
      const dim_t     ldb,
      const dim_t     KC
    );

void packb_nr32_bf16bf16f32of32_row_major
    (
      bfloat16*       pack_b_buffer_bf16bf16f32of32,
      const bfloat16* b,
      const dim_t     ldb,
      const dim_t     KC
    );

void packb_nr48_bf16bf16f32of32_row_major
    (
      bfloat16*       pack_b_buffer_bf16bf16f32of32,
      const bfloat16* b,
      const dim_t     ldb,
      const dim_t     KC
    );


void packb_nrlt16_bf16bf16f32of32_col_major
    (
      bfloat16*       pack_b_buffer_bf16bf16f32of32,
      const bfloat16* b,
      const dim_t     ldb,
      const dim_t     KC,
      const dim_t     n0_partial_rem
    );

void packb_nr_mult_16_bf16bf16f32of32_col_major
    (
      bfloat16*       pack_b_buffer,
      const bfloat16* b,
	  const dim_t     NR,
      const dim_t     ldb,
      const dim_t     KC
    );


void packb_nr64_bf16bf16f32of32
    (
      bfloat16*       pack_b_buffer_bf16bf16f32of32,
      const bfloat16* b,
      const dim_t     rs_b,
	  const dim_t     cs_b,
      const dim_t     NC,
      const dim_t     KC,
      dim_t*          rs_p,
      dim_t*          cs_p
    )
{
	if( cs_b == 1 )
	{
		packb_nr64_bf16bf16f32of32_row_major( pack_b_buffer_bf16bf16f32of32,
		                                     b, rs_b, NC, KC, rs_p, cs_p );
	}
	else
	{
		packb_nr64_bf16bf16f32of32_col_major( pack_b_buffer_bf16bf16f32of32,
		                                     b, cs_b, NC, KC, rs_p, cs_p );
	}
}
void packb_nr64_bf16bf16f32of32_row_major
    (
      bfloat16*       pack_b_buffer_bf16bf16f32of32,
      const bfloat16* b,
      const dim_t     ldb,
      const dim_t     NC,
      const dim_t     KC,
      dim_t*          rs_b,
      dim_t*          cs_b
    )
{
    dim_t NR = 64;

	// Used for permuting the mm512i elements for use in dpbf16_ps instruction.
	__m512i selector1 = _mm512_setr_epi64(0x0, 0x1, 0x8, 0x9, 0x2, 0x3, 0xA, 0xB);
	__m512i selector1_1 = _mm512_setr_epi64( 0x4, 0x5, 0xC, 0xD, 0x6, 0x7, 0xE, 0xF );

	__m512i a0;
	__m512i b0;
	__m512i c0;
	__m512i d0;
	__m512i a01;
	__m512i c01;

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

	for ( dim_t jc = 0; jc < n_full_pieces_loop_limit; jc += NR )
	{
		for ( dim_t kr = 0; kr < k_full_pieces; kr += 2 )
		{
			// Rearrange for dpbf16_ps, read 2 rows from B with 64 elements in each row.
			a0 = _mm512_loadu_si512( b + ( ldb * ( kr + 0 ) ) + jc  );
			b0 = _mm512_loadu_si512( b + ( ldb * ( kr + 0 ) ) + jc + 32 );
			c0 = _mm512_loadu_si512( b + ( ldb * ( kr + 1 ) ) + jc );
			d0 = _mm512_loadu_si512( b + ( ldb * ( kr + 1 ) ) + jc + 32 );

			a01 = _mm512_unpacklo_epi16( a0, c0 );
			a0 = _mm512_unpackhi_epi16( a0, c0 );

			c01 = _mm512_unpacklo_epi16( b0, d0 );
			c0 = _mm512_unpackhi_epi16( b0, d0 );

			b0 = _mm512_permutex2var_epi64( a01, selector1, a0 );
			d0 = _mm512_permutex2var_epi64( c01, selector1, c0 );
			a0 = _mm512_permutex2var_epi64( a01, selector1_1, a0 );
			c0 = _mm512_permutex2var_epi64( c01, selector1_1, c0 );

			//store to pack_b buffer
			_mm512_storeu_si512( pack_b_buffer_bf16bf16f32of32 + ( jc * KC_updated ) + ( ( kr + 0 ) * NR ), b0 );
			_mm512_storeu_si512( pack_b_buffer_bf16bf16f32of32 + ( jc * KC_updated ) + ( ( kr + 0 ) * NR ) + 32, a0 );
			_mm512_storeu_si512( pack_b_buffer_bf16bf16f32of32 + ( jc * KC_updated ) + ( ( kr + 1 ) * NR ), d0 );
			_mm512_storeu_si512( pack_b_buffer_bf16bf16f32of32 + ( jc * KC_updated ) + ( ( kr + 1 ) * NR ) + 32, c0 );
		}
		// Handle k remainder.
		if( k_partial_pieces > 0)
		{
			a0 = _mm512_loadu_si512( b + ( ldb * ( k_full_pieces + 0 ) ) + jc  );
			b0 = _mm512_loadu_si512( b + ( ldb * ( k_full_pieces + 0 ) ) + jc + 32 );
			c0 = _mm512_setzero_si512();
			d0 = _mm512_setzero_si512();

			a01 = _mm512_unpacklo_epi16( a0, c0 );
			a0 = _mm512_unpackhi_epi16( a0, c0 );

			c01 = _mm512_unpacklo_epi16( b0, d0 );
			c0 = _mm512_unpackhi_epi16( b0, d0 );

			b0 = _mm512_permutex2var_epi64( a01, selector1, a0 );
			d0 = _mm512_permutex2var_epi64( c01, selector1, c0 );
			a0 = _mm512_permutex2var_epi64( a01, selector1_1, a0 );
			c0 = _mm512_permutex2var_epi64( c01, selector1_1, c0 );

			//store to pack_b buffer
			_mm512_storeu_si512( pack_b_buffer_bf16bf16f32of32 + ( jc * KC_updated ) + ( ( k_full_pieces + 0 ) * NR ), b0 );
			_mm512_storeu_si512( pack_b_buffer_bf16bf16f32of32 + ( jc * KC_updated ) + ( ( k_full_pieces + 0 ) * NR ) + 32, a0 );
			_mm512_storeu_si512( pack_b_buffer_bf16bf16f32of32 + ( jc * KC_updated ) + ( ( k_full_pieces + 1 ) * NR ), d0 );
			_mm512_storeu_si512( pack_b_buffer_bf16bf16f32of32 + ( jc * KC_updated ) + ( ( k_full_pieces + 1 ) * NR ) + 32, c0 );
		}
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
			packb_nr48_bf16bf16f32of32_row_major
				(
				 ( pack_b_buffer_bf16bf16f32of32 + ( n_full_pieces_loop_limit * KC_updated ) ),
				 ( b + n_full_pieces_loop_limit ), ldb, KC
				);

			n0_partial_pack = 48;
		}
		else if ( n0_32 == 1 )
		{
			packb_nr32_bf16bf16f32of32_row_major
				(
				 ( pack_b_buffer_bf16bf16f32of32 + ( n_full_pieces_loop_limit * KC_updated ) ),
				 ( b + n_full_pieces_loop_limit ), ldb, KC
				);

			n0_partial_pack = 32;
		}
		else if ( n0_16 == 1 )
		{
			packb_nr16_bf16bf16f32of32_row_major
				(
				 ( pack_b_buffer_bf16bf16f32of32 + ( n_full_pieces_loop_limit * KC_updated ) ),
				 ( b + n_full_pieces_loop_limit ), ldb, KC
				);

			n0_partial_pack = 16;
		}

		if ( n0_partial_rem > 0 )
		{
			packb_nrlt16_bf16bf16f32of32_row_major
				(
				 ( pack_b_buffer_bf16bf16f32of32 + ( n_full_pieces_loop_limit * KC_updated ) +
				   ( n0_partial_pack * KC_updated ) ),
				 ( b + n_full_pieces_loop_limit + n0_partial_pack ), ldb, KC,
				 n0_partial_rem
				);
		}
	}
	*rs_b = NR * 2;
	*cs_b = NR / 2;
}

void packb_nr48_bf16bf16f32of32_row_major
(
 bfloat16*       pack_b_buffer_bf16bf16f32of32,
 const bfloat16* b,
 const dim_t     ldb,
 const dim_t     KC
 )
{
	dim_t NR1 = 32;
	dim_t NR2 = 16;

	// Used for permuting the mm512i elements for use in dpbf16_ps instruction.
	__m512i selector1 = _mm512_setr_epi64(0x0, 0x1, 0x8, 0x9, 0x2, 0x3, 0xA, 0xB);
	__m512i selector1_1 = _mm512_setr_epi64( 0x4, 0x5, 0xC, 0xD, 0x6, 0x7, 0xE, 0xF );

	__m512i a0x;
	__m512i b0x;
	__m512i c0x;
	__m512i a01x;

	__m256i a0;
	__m256i b0;
	__m256i c0;
	__m256i a01;

	dim_t k_full_pieces_blks = KC / 2;
	dim_t k_full_pieces = k_full_pieces_blks * 2;
	dim_t k_partial_pieces = KC % 2;

	dim_t kr_new = 0;

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 2 )
	{
		// Rearrange for dpbf16_ps, read 2 rows from B with 32 elements in each row.
		a0x = _mm512_loadu_si512( b + ( ldb * ( kr + 0 ) )  );
		c0x = _mm512_loadu_si512( b + ( ldb * ( kr + 1 ) )  );

		a01x = _mm512_unpacklo_epi16( a0x, c0x );
		a0x = _mm512_unpackhi_epi16( a0x, c0x );

		b0x = _mm512_permutex2var_epi64( a01x, selector1, a0x );
		a0x = _mm512_permutex2var_epi64( a01x, selector1_1, a0x );

		//First 2x32 elements
		_mm512_storeu_si512( pack_b_buffer_bf16bf16f32of32 + ( ( kr_new + 0 ) * NR1 ), b0x );
		_mm512_storeu_si512( pack_b_buffer_bf16bf16f32of32 + ( ( kr_new + 1 ) * NR1 ), a0x );

		// Rearrange for dpbf16_ps, read 2 rows from B with next 16 elements in each row.
		a0 = _mm256_maskz_loadu_epi16( 0xFFFF, b + ( ldb * ( kr + 0 ) ) + NR1 );
		c0 = _mm256_maskz_loadu_epi16( 0xFFFF, b + ( ldb * ( kr + 1 ) ) + NR1 );

		a01 = _mm256_unpacklo_epi16( a0, c0 );
		a0 = _mm256_unpackhi_epi16( a0, c0 );

		b0 = _mm256_permute2f128_si256(a01, a0, 0x20);
		a0 = _mm256_permute2f128_si256(a01, a0, 0x31);

		//Last 2x16 elements
		_mm256_mask_storeu_epi64
		(
		  pack_b_buffer_bf16bf16f32of32 + ( ( kr_new + 2 ) * NR1 ),
		  0xFF, b0
		);
		_mm256_mask_storeu_epi64
		(
		  pack_b_buffer_bf16bf16f32of32 + ( ( kr_new + 2 ) * NR1 ) + NR2,
		  0xFF, a0
		);

		kr_new += 3;
	}

	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		a0x = _mm512_loadu_si512( b + ( ldb * ( k_full_pieces + 0 ) )  );
		c0x = _mm512_setzero_si512();

		a01x = _mm512_unpacklo_epi16( a0x, c0x );
		a0x = _mm512_unpackhi_epi16( a0x, c0x );

		b0x = _mm512_permutex2var_epi64( a01x, selector1, a0x );
		a0x = _mm512_permutex2var_epi64( a01x, selector1_1, a0x );

		//First 2x32 elements
		_mm512_storeu_si512( pack_b_buffer_bf16bf16f32of32 + ( ( kr_new + 0 ) * NR1 ), b0x );
		_mm512_storeu_si512( pack_b_buffer_bf16bf16f32of32 + ( ( kr_new + 1 ) * NR1 ), a0x );

		a0 = _mm256_maskz_loadu_epi16( 0xFFFF, b + ( ldb * ( k_full_pieces + 0 ) ) + NR1 );
		c0 = _mm256_setzero_si256();

		a01 = _mm256_unpacklo_epi16( a0, c0 );
		a0 = _mm256_unpackhi_epi16( a0, c0 );

		b0 = _mm256_permute2f128_si256(a01, a0, 0x20);
		a0 = _mm256_permute2f128_si256(a01, a0, 0x31);

		//Last 2x16 elements
		_mm256_mask_storeu_epi64
		(
		  pack_b_buffer_bf16bf16f32of32 + ( ( kr_new + 2 ) * NR1 ),
		  0xFF, b0
		);
		_mm256_mask_storeu_epi64
		(
		  pack_b_buffer_bf16bf16f32of32 + ( ( kr_new + 2 ) * NR1 ) + NR2,
		  0xFF, a0
		);
	}
}

void packb_nr32_bf16bf16f32of32_row_major
(
 bfloat16*       pack_b_buffer_bf16bf16f32of32,
 const bfloat16* b,
 const dim_t     ldb,
 const dim_t     KC
 )
{
	dim_t NR = 32;

	// Used for permuting the mm512i elements for use in dpbf16_ps instruction.
	__m512i selector1 = _mm512_setr_epi64(0x0, 0x1, 0x8, 0x9, 0x2, 0x3, 0xA, 0xB);
	__m512i selector1_1 = _mm512_setr_epi64( 0x4, 0x5, 0xC, 0xD, 0x6, 0x7, 0xE, 0xF );

	__m512i a0;
	__m512i b0;
	__m512i c0;
	__m512i a01;

	dim_t k_full_pieces_blks = KC / 2;
	dim_t k_full_pieces = k_full_pieces_blks * 2;
	dim_t k_partial_pieces = KC % 2;

	dim_t kr_new = 0;

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 2 )
	{
		// Rearrange for dpbf16_ps, read 2 rows from B with 32 elements in each row.
		a0 = _mm512_loadu_si512( b + ( ldb * ( kr + 0 ) )  );
		c0 = _mm512_loadu_si512( b + ( ldb * ( kr + 1 ) ) );

		a01 = _mm512_unpacklo_epi16( a0, c0 );
		a0 = _mm512_unpackhi_epi16( a0, c0 );

		b0 = _mm512_permutex2var_epi64( a01, selector1, a0 );
		a0 = _mm512_permutex2var_epi64( a01, selector1_1, a0 );

		_mm512_storeu_si512( pack_b_buffer_bf16bf16f32of32 + ( ( kr_new ) * NR ), b0 );
		_mm512_storeu_si512( pack_b_buffer_bf16bf16f32of32 + ( ( kr_new + 1 ) * NR ), a0 );

		kr_new += 2;
	}

	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		a0 = _mm512_loadu_si512( b + ( ldb * ( k_full_pieces + 0 ) )  );
		c0 = _mm512_setzero_si512();

		a01 = _mm512_unpacklo_epi16( a0, c0 );
		a0 = _mm512_unpackhi_epi16( a0, c0 );

		b0 = _mm512_permutex2var_epi64( a01, selector1, a0 );
		a0 = _mm512_permutex2var_epi64( a01, selector1_1, a0 );

		_mm512_storeu_si512( pack_b_buffer_bf16bf16f32of32 + ( ( kr_new ) * NR ), b0 );
		_mm512_storeu_si512( pack_b_buffer_bf16bf16f32of32 + ( ( kr_new + 1 ) * NR ), a0 );
	}
}

void packb_nr16_bf16bf16f32of32_row_major
(
 bfloat16*       pack_b_buffer_bf16bf16f32of32,
 const bfloat16* b,
 const dim_t     ldb,
 const dim_t     KC
 )
{
	dim_t NR = 16;

	__m256i a0;
	__m256i b0;
	__m256i c0;
	__m256i a01;

	dim_t k_full_pieces_blks = KC / 2;
	dim_t k_full_pieces = k_full_pieces_blks * 2;
	dim_t k_partial_pieces = KC % 2;

	dim_t kr_new = 0;

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 2 )
	{
		// Rearrange for dpbf16_ps, read 2 rows from B with 16 elements in each row.
		a0 = _mm256_maskz_loadu_epi16( 0xFFFF, b + ( ldb * ( kr + 0 ) )  );
		c0 = _mm256_maskz_loadu_epi16( 0xFFFF, b + ( ldb * ( kr + 1 ) )  );

		a01 = _mm256_unpacklo_epi16( a0, c0 );
		a0 = _mm256_unpackhi_epi16( a0, c0 );

		b0 = _mm256_permute2f128_si256(a01, a0, 0x20);
		a0 = _mm256_permute2f128_si256(a01, a0, 0x31);

		_mm256_mask_storeu_epi64
		(
		  pack_b_buffer_bf16bf16f32of32 + ( ( kr_new + 0 ) * NR ),
		  0xFF, b0
		);
		_mm256_mask_storeu_epi64
		(
		  pack_b_buffer_bf16bf16f32of32 + ( ( kr_new + 1 ) * NR ),
		  0xFF, a0
		);

		kr_new += 2;
	}

	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		a0 = _mm256_maskz_loadu_epi16( 0xFFFF, b + ( ldb * ( k_full_pieces + 0 ) )  );
		c0 = _mm256_setzero_si256();

		a01 = _mm256_unpacklo_epi16( a0, c0 );
		a0 = _mm256_unpackhi_epi16( a0, c0 );

		b0 = _mm256_permute2f128_si256(a01, a0, 0x20);
		a0 = _mm256_permute2f128_si256(a01, a0, 0x31);

		_mm256_mask_storeu_epi64
		(
		  pack_b_buffer_bf16bf16f32of32 + ( ( kr_new + 0 ) * NR ),
		  0xFF, b0
		);
		_mm256_mask_storeu_epi64
		(
		  pack_b_buffer_bf16bf16f32of32 + ( ( kr_new + 1 ) * NR ),
		  0xFF, a0
		);
	}
}

void packb_nrlt16_bf16bf16f32of32_row_major
(
 bfloat16*       pack_b_buffer_bf16bf16f32of32,
 const bfloat16* b,
 const dim_t     ldb,
 const dim_t     KC,
 const dim_t     n0_partial_rem
 )
{
	dim_t NR = 16;

	__m256i a0;
	__m256i b0;
	__m256i c0;
	__m256i a01;

	dim_t k_full_pieces_blks = KC / 2;
	dim_t k_full_pieces = k_full_pieces_blks * 2;
	dim_t k_partial_pieces = KC % 2;

	dim_t kr_new = 0;

	__mmask16 load_mask = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_partial_rem ) );

	for ( int kr = 0; kr < k_full_pieces; kr += 2 )
	{
		// Rearrange for dpbf16_ps, read 2 rows from B with next 16 elements in each row.
		a0 = _mm256_maskz_loadu_epi16( load_mask, b + ( ldb * ( kr + 0 ) ) );
		c0 = _mm256_maskz_loadu_epi16( load_mask, b + ( ldb * ( kr + 1 ) ) );

		a01 = _mm256_unpacklo_epi16( a0, c0 );
		a0 = _mm256_unpackhi_epi16( a0, c0 );

		b0 = _mm256_permute2f128_si256(a01, a0, 0x20);
		a0 = _mm256_permute2f128_si256(a01, a0, 0x31);

		_mm256_mask_storeu_epi64
		(
		  pack_b_buffer_bf16bf16f32of32 + ( ( kr_new + 0 ) * NR ),
		  0xFF, b0
		);
		_mm256_mask_storeu_epi64
		(
		  pack_b_buffer_bf16bf16f32of32 + ( ( kr_new + 1 ) * NR ),
		  0xFF, a0
		);

		kr_new += 2;
	}

	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		a0 = _mm256_maskz_loadu_epi16( load_mask, b + ( ldb * ( k_full_pieces + 0 ) ) );
		c0 = _mm256_setzero_si256();

		a01 = _mm256_unpacklo_epi16( a0, c0 );
		a0 = _mm256_unpackhi_epi16( a0, c0 );

		b0 = _mm256_permute2f128_si256(a01, a0, 0x20);
		a0 = _mm256_permute2f128_si256(a01, a0, 0x31);

		_mm256_mask_storeu_epi64
		(
		  pack_b_buffer_bf16bf16f32of32 + ( ( kr_new + 0 ) * NR ),
		  0xFF, b0
		);
		_mm256_mask_storeu_epi64
		(
		  pack_b_buffer_bf16bf16f32of32 + ( ( kr_new + 1 ) * NR ),
		  0xFF, a0
		);
	}
}

#define LOAD_16_COLS_AVX512 \
	a_reg[0] = _mm512_loadu_si512(b + ( ldb * ( jr + 0 ) ) + kr); \
	a_reg[1] = _mm512_loadu_si512(b + ( ldb * ( jr + 1 ) ) + kr); \
	a_reg[2] = _mm512_loadu_si512(b + ( ldb * ( jr + 2 ) ) + kr); \
	a_reg[3] = _mm512_loadu_si512(b + ( ldb * ( jr + 3 ) ) + kr); \
	a_reg[4] = _mm512_loadu_si512(b + ( ldb * ( jr + 4 ) ) + kr); \
	a_reg[5] = _mm512_loadu_si512(b + ( ldb * ( jr + 5 ) ) + kr); \
	a_reg[6] = _mm512_loadu_si512(b + ( ldb * ( jr + 6 ) ) + kr); \
	a_reg[7] = _mm512_loadu_si512(b + ( ldb * ( jr + 7 ) ) + kr); \
	a_reg[8] = _mm512_loadu_si512(b + ( ldb * ( jr + 8 ) ) + kr); \
	a_reg[9] = _mm512_loadu_si512(b + ( ldb * ( jr + 9 ) ) + kr); \
	a_reg[10] = _mm512_loadu_si512(b + ( ldb * ( jr + 10 ) ) + kr); \
	a_reg[11] = _mm512_loadu_si512(b + ( ldb * ( jr + 11 ) ) + kr); \
	a_reg[12] = _mm512_loadu_si512(b + ( ldb * ( jr + 12 ) ) + kr); \
	a_reg[13] = _mm512_loadu_si512(b + ( ldb * ( jr + 13 ) ) + kr); \
	a_reg[14] = _mm512_loadu_si512(b + ( ldb * ( jr + 14 ) ) + kr); \
	a_reg[15] = _mm512_loadu_si512(b + ( ldb * ( jr + 15 ) ) + kr);

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

#define MASK_LOAD_16_COLS_AVX512(mask) \
	a_reg[0] = _mm512_maskz_loadu_epi16( mask, b + ( ldb * ( jr + 0 ) ) + kr); \
	a_reg[1] = _mm512_maskz_loadu_epi16( mask, b + ( ldb * ( jr + 1 ) ) + kr); \
	a_reg[2] = _mm512_maskz_loadu_epi16( mask, b + ( ldb * ( jr + 2 ) ) + kr); \
	a_reg[3] = _mm512_maskz_loadu_epi16( mask, b + ( ldb * ( jr + 3 ) ) + kr); \
	a_reg[4] = _mm512_maskz_loadu_epi16( mask, b + ( ldb * ( jr + 4 ) ) + kr); \
	a_reg[5] = _mm512_maskz_loadu_epi16( mask, b + ( ldb * ( jr + 5 ) ) + kr); \
	a_reg[6] = _mm512_maskz_loadu_epi16( mask, b + ( ldb * ( jr + 6 ) ) + kr); \
	a_reg[7] = _mm512_maskz_loadu_epi16( mask, b + ( ldb * ( jr + 7 ) ) + kr); \
	a_reg[8] = _mm512_maskz_loadu_epi16( mask, b + ( ldb * ( jr + 8 ) ) + kr); \
	a_reg[9] = _mm512_maskz_loadu_epi16( mask, b + ( ldb * ( jr + 9 ) ) + kr); \
	a_reg[10] = _mm512_maskz_loadu_epi16( mask, b + ( ldb * ( jr + 10 ) ) + kr); \
	a_reg[11] = _mm512_maskz_loadu_epi16( mask, b + ( ldb * ( jr + 11 ) ) + kr); \
	a_reg[12] = _mm512_maskz_loadu_epi16( mask, b + ( ldb * ( jr + 12 ) ) + kr); \
	a_reg[13] = _mm512_maskz_loadu_epi16( mask, b + ( ldb * ( jr + 13 ) ) + kr); \
	a_reg[14] = _mm512_maskz_loadu_epi16( mask, b + ( ldb * ( jr + 14 ) ) + kr); \
	a_reg[15] = _mm512_maskz_loadu_epi16( mask, b + ( ldb * ( jr + 15 ) ) + kr);

void packb_nr64_bf16bf16f32of32_col_major
    (
      bfloat16*       pack_b_buffer,
      const bfloat16* b,
      const dim_t     ldb,
      const dim_t     NC,
      const dim_t     KC,
      dim_t*          rs_b,
      dim_t*          cs_b
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
		packb_nr_mult_16_bf16bf16f32of32_col_major
			( pack_b_buffer + (jc * KC_updated),
			  b + (jc * ldb), 64, ldb, KC
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
			packb_nr_mult_16_bf16bf16f32of32_col_major
				(
				 ( pack_b_buffer + ( n_full_pieces_loop_limit * KC_updated ) ),
				 ( b + n_full_pieces_loop_limit * ldb ), 48, ldb, KC
				);

			n0_partial_pack = 48;
		}
		else if ( n0_32 == 1 )
		{
			packb_nr_mult_16_bf16bf16f32of32_col_major
				(
				 ( pack_b_buffer + ( n_full_pieces_loop_limit * KC_updated ) ),
				 ( b + n_full_pieces_loop_limit * ldb ), 32, ldb, KC
				);

			n0_partial_pack = 32;
		}
		else if ( n0_16 == 1 )
		{
			packb_nr_mult_16_bf16bf16f32of32_col_major
				(
				 ( pack_b_buffer + ( n_full_pieces_loop_limit * KC_updated ) ),
				 ( b + n_full_pieces_loop_limit * ldb ), 16, ldb, KC
				);

			n0_partial_pack = 16;
		}

		if ( n0_partial_rem > 0 )
		{
			packb_nrlt16_bf16bf16f32of32_col_major
				(
				 ( pack_b_buffer + ( n_full_pieces_loop_limit * KC_updated ) +
				   ( n0_partial_pack * KC_updated ) ),
				 ( b + ( n_full_pieces_loop_limit + n0_partial_pack ) * ldb ), ldb, KC,
				 n0_partial_rem
				);
		}
	}
	*rs_b = NR * 2;
	*cs_b = NR / 2;
}

void packb_nr_mult_16_bf16bf16f32of32_col_major
    (
      bfloat16*       pack_b_buffer,
      const bfloat16* b,
      const dim_t     NR,
      const dim_t     ldb,
      const dim_t     KC
    )
{
	// Used for permuting the mm512i elements for use in dpbf16_ps instruction.
	__m512i selector1 = _mm512_setr_epi64( 0x0, 0x1, 0x8, 0x9, 0x4, 0x5, 0xC, 0xD );
	__m512i selector2 = _mm512_setr_epi64( 0x2, 0x3, 0xA, 0xB, 0x6, 0x7, 0xE, 0xF );

	__m512i a_reg[16];
	__m512i b_reg[16];

	dim_t kr = 0;
	for ( kr = 0; ( kr + 31 ) < KC; kr += 32 )
	{
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			// Rearrange for dpbf16_ps, read 2 rows from B with 64 elements in each row.
			LOAD_16_COLS_AVX512
			UNPACKHILO32_AVX512
			UNPACKHILO64_AVX512
			PERMUTEX2_VAR64_AVX512
			SHUFFLE64x2_AVX512

			// store to pack_b buffer
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr + 0  ) * NR ), a_reg[0]  );
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr + 2  ) * NR ), a_reg[1]  );
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr + 4  ) * NR ), a_reg[2]  );
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr + 6  ) * NR ), a_reg[3]  );
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr + 8  ) * NR ), a_reg[4]  );
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr + 10 ) * NR ), a_reg[5]  );
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr + 12 ) * NR ), a_reg[6]  );
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr + 14 ) * NR ), a_reg[7]  );
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr + 16 ) * NR ), a_reg[8]  );
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr + 18 ) * NR ), a_reg[9]  );
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr + 20 ) * NR ), a_reg[10] );
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr + 22 ) * NR ), a_reg[11] );
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr + 24 ) * NR ), a_reg[12] );
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr + 26 ) * NR ), a_reg[13] );
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr + 28 ) * NR ), a_reg[14] );
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr + 30 ) * NR ), a_reg[15] );
		}
	}

	for ( ; ( kr + 15 ) < KC; kr += 16 )
	{
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			// Rearrange for dpbf16_ps, read 2 rows from B with 64 elements in each row.

			MASK_LOAD_16_COLS_AVX512( 0xFFFF )
			UNPACKHILO32_AVX512
			UNPACKHILO64_AVX512
			PERMUTEX2_VAR64_AVX512
			SHUFFLE64x2_AVX512

			// store to pack_b buffer
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr + 0  ) * NR ), a_reg[0] );
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr + 2  ) * NR ), a_reg[1] );
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr + 4  ) * NR ), a_reg[2] );
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr + 6  ) * NR ), a_reg[3] );
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr + 8  ) * NR ), a_reg[4] );
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr + 10 ) * NR ), a_reg[5] );
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr + 12 ) * NR ), a_reg[6] );
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr + 14 ) * NR ), a_reg[7] );
		}
	}

	for( ; ( kr +7 ) < KC; kr += 8 )
	{
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			// Rearrange for dpbf16_ps, read 2 rows from B with 64 elements in each row.

			MASK_LOAD_16_COLS_AVX512( 0xFF )
			UNPACKHILO32_AVX512
			UNPACKHILO64_AVX512
			PERMUTEX2_VAR64_AVX512
			SHUFFLE64x2_AVX512

			// store to pack_b buffer
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr + 0 ) * NR ), a_reg[0] );
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr + 2 ) * NR ), a_reg[1] );
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr + 4 ) * NR ), a_reg[2] );
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr + 6 ) * NR ), a_reg[3] );
		}
	}

	for( ; ( kr +3 ) < KC; kr += 4 )
	{
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			// Rearrange for dpbf16_ps, read 2 rows from B with 64 elements in each row.
			MASK_LOAD_16_COLS_AVX512( 0x0F )
			UNPACKHILO32_AVX512
			UNPACKHILO64_AVX512
			PERMUTEX2_VAR64_AVX512
			SHUFFLE64x2_AVX512

			// store to pack_b buffer
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr + 0 ) * NR ), a_reg[0] );
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr + 2 ) * NR ), a_reg[1] );
		}
	}

	for( ; ( kr +1 ) < KC; kr += 2 )
	{
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			// Rearrange for dpbf16_ps, read 2 rows from B with 64 elements in each row.
			MASK_LOAD_16_COLS_AVX512( 0x03 )
			UNPACKHILO32_AVX512
			UNPACKHILO64_AVX512
			PERMUTEX2_VAR64_AVX512
			SHUFFLE64x2_AVX512

			// store to pack_b buffer
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( ( kr ) * NR ), a_reg[0] );
		}
	}

	for( ; kr < KC; kr += 1 )
	{
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			// Rearrange for dpbf16_ps, read 2 rows from B with 64 elements in each row.
			MASK_LOAD_16_COLS_AVX512( 0x01 )
			UNPACKHILO32_AVX512
			UNPACKHILO64_AVX512
			PERMUTEX2_VAR64_AVX512
			SHUFFLE64x2_AVX512

			// store to pack_b buffer
			_mm512_storeu_si512( pack_b_buffer + ( jr * 2 ) + ( kr * NR ), a_reg[0] );
		}
	}
}


void packb_nrlt16_bf16bf16f32of32_col_major
    (
      bfloat16*       pack_b_buffer,
      const bfloat16* b,
      const dim_t     ldb,
      const dim_t     KC,
      const dim_t     n0_partial_rem
    )
{
	dim_t NR = 16;

	// Used for permuting the mm512i elements for use in dpbf16_ps instruction.
	__m512i selector1 = _mm512_setr_epi64( 0x0, 0x1, 0x8, 0x9, 0x4, 0x5, 0xC, 0xD );
	__m512i selector2 = _mm512_setr_epi64( 0x2, 0x3, 0xA, 0xB, 0x6, 0x7, 0xE, 0xF );

	__m512i a_reg[16];
	__m512i b_reg[16];

	dim_t kr = 0, jr = 0;
	for ( kr = 0; ( kr + 31 ) < KC; kr += 32 )
	{
		for( jr = 0; jr < n0_partial_rem; jr += 1 )
		{
			// Rearrange for dpbf16_ps, read 2 rows from B with 64 elements in each row.
			a_reg[jr] = _mm512_loadu_si512( b + ( ldb * ( jr + 0 ) ) + kr );
		}
		for(; jr < NR; jr++)
		{
			a_reg[jr] = _mm512_setzero_si512();
		}

		UNPACKHILO32_AVX512
		UNPACKHILO64_AVX512
		PERMUTEX2_VAR64_AVX512
		SHUFFLE64x2_AVX512

		// store to pack_b buffer
		_mm512_storeu_si512( pack_b_buffer + ( ( kr + 0  ) * NR ), a_reg[0]  );
		_mm512_storeu_si512( pack_b_buffer + ( ( kr + 2  ) * NR ), a_reg[1]  );
		_mm512_storeu_si512( pack_b_buffer + ( ( kr + 4  ) * NR ), a_reg[2]  );
		_mm512_storeu_si512( pack_b_buffer + ( ( kr + 6  ) * NR ), a_reg[3]  );
		_mm512_storeu_si512( pack_b_buffer + ( ( kr + 8  ) * NR ), a_reg[4]  );
		_mm512_storeu_si512( pack_b_buffer + ( ( kr + 10 ) * NR ), a_reg[5]  );
		_mm512_storeu_si512( pack_b_buffer + ( ( kr + 12 ) * NR ), a_reg[6]  );
		_mm512_storeu_si512( pack_b_buffer + ( ( kr + 14 ) * NR ), a_reg[7]  );
		_mm512_storeu_si512( pack_b_buffer + ( ( kr + 16 ) * NR ), a_reg[8]  );
		_mm512_storeu_si512( pack_b_buffer + ( ( kr + 18 ) * NR ), a_reg[9]  );
		_mm512_storeu_si512( pack_b_buffer + ( ( kr + 20 ) * NR ), a_reg[10] );
		_mm512_storeu_si512( pack_b_buffer + ( ( kr + 22 ) * NR ), a_reg[11] );
		_mm512_storeu_si512( pack_b_buffer + ( ( kr + 24 ) * NR ), a_reg[12] );
		_mm512_storeu_si512( pack_b_buffer + ( ( kr + 26 ) * NR ), a_reg[13] );
		_mm512_storeu_si512( pack_b_buffer + ( ( kr + 28 ) * NR ), a_reg[14] );
		_mm512_storeu_si512( pack_b_buffer + ( ( kr + 30 ) * NR ), a_reg[15] );

	}

	for ( ; ( kr + 15 ) < KC; kr += 16 )
	{
		for( jr = 0; jr  < n0_partial_rem; jr += 1 )
		{
			// Rearrange for dpbf16_ps, read 2 rows from B with 64 elements in each row.
			a_reg[jr] = _mm512_maskz_loadu_epi16( 0xFFFF, b + ( ldb * ( jr + 0 ) ) + kr );
		}
		for( ; jr < NR; jr++ )
		{
			a_reg[jr] = _mm512_setzero_si512();
		}

		UNPACKHILO32_AVX512
		UNPACKHILO64_AVX512
		PERMUTEX2_VAR64_AVX512
		SHUFFLE64x2_AVX512

		// store to pack_b buffer
		_mm512_storeu_si512( pack_b_buffer + ( ( kr + 0  ) * NR ), a_reg[0] );
		_mm512_storeu_si512( pack_b_buffer + ( ( kr + 2  ) * NR ), a_reg[1] );
		_mm512_storeu_si512( pack_b_buffer + ( ( kr + 4  ) * NR ), a_reg[2] );
		_mm512_storeu_si512( pack_b_buffer + ( ( kr + 6  ) * NR ), a_reg[3] );
		_mm512_storeu_si512( pack_b_buffer + ( ( kr + 8  ) * NR ), a_reg[4] );
		_mm512_storeu_si512( pack_b_buffer + ( ( kr + 10 ) * NR ), a_reg[5] );
		_mm512_storeu_si512( pack_b_buffer + ( ( kr + 12 ) * NR ), a_reg[6] );
		_mm512_storeu_si512( pack_b_buffer + ( ( kr + 14 ) * NR ), a_reg[7] );
	}

	for ( ; ( kr + 7 ) < KC; kr += 8 )
	{
		for( jr = 0; jr  < n0_partial_rem; jr += 1 )
		{
			// Rearrange for dpbf16_ps, read 2 rows from B with 64 elements in each row.
			a_reg[jr] = _mm512_maskz_loadu_epi16( 0xFF, b + ( ldb * ( jr + 0 ) ) + kr );
		}
		for( ; jr < NR; jr++ )
		{
			a_reg[jr] = _mm512_setzero_si512();
		}

		UNPACKHILO32_AVX512
		UNPACKHILO64_AVX512
		PERMUTEX2_VAR64_AVX512
		SHUFFLE64x2_AVX512

		// store to pack_b buffer
		_mm512_storeu_si512( pack_b_buffer + ( ( kr + 0 ) * NR ), a_reg[0] );
		_mm512_storeu_si512( pack_b_buffer + ( ( kr + 2 ) * NR ), a_reg[1] );
		_mm512_storeu_si512( pack_b_buffer + ( ( kr + 4 ) * NR ), a_reg[2] );
		_mm512_storeu_si512( pack_b_buffer + ( ( kr + 6 ) * NR ), a_reg[3] );
	}

	for ( ; (kr+3) < KC; kr += 4 )
	{
		for( jr = 0; jr  < n0_partial_rem; jr += 1 )
		{
			// Rearrange for dpbf16_ps, read 2 rows from B with 64 elements in each row.
			a_reg[jr] = _mm512_maskz_loadu_epi16( 0x0F, b + ( ldb * ( jr + 0 ) ) + kr );
		}
		for( ; jr < NR; jr++ )
		{
			a_reg[jr] = _mm512_setzero_si512();
		}

		UNPACKHILO32_AVX512
		UNPACKHILO64_AVX512
		PERMUTEX2_VAR64_AVX512
		SHUFFLE64x2_AVX512

		// store to pack_b buffer
		_mm512_storeu_si512( pack_b_buffer + ( ( kr + 0 ) * NR ), a_reg[0] );
		_mm512_storeu_si512( pack_b_buffer + ( ( kr + 2 ) * NR ), a_reg[1] );
	}

	for ( ; ( kr + 1 ) < KC; kr += 2 )
	{
		for( jr = 0; jr  < n0_partial_rem; jr += 1 )
		{
			// Rearrange for dpbf16_ps, read 2 rows from B with 64 elements in each row.
			a_reg[jr] = _mm512_maskz_loadu_epi16( 0x03, b + ( ldb * ( jr + 0 ) ) + kr );
		}
		for( ; jr < NR; jr++ )
		{
			a_reg[jr] = _mm512_setzero_si512();
		}
		UNPACKHILO32_AVX512
		UNPACKHILO64_AVX512
		PERMUTEX2_VAR64_AVX512
		SHUFFLE64x2_AVX512

		// store to pack_b buffer
		_mm512_storeu_si512( pack_b_buffer + ( kr * NR ), a_reg[0] );
	}

	for ( ; kr < KC; kr += 1 )
	{
		for( jr = 0; jr  < n0_partial_rem; jr += 1 )
		{
			// Rearrange for dpbf16_ps, read 2 rows from B with 64 elements in each row.
			a_reg[jr] = _mm512_maskz_loadu_epi16( 0x01, b + ( ldb * ( jr + 0 ) ) + kr );
		}
		for( ; jr < NR; jr++ )
		{
			a_reg[jr] = _mm512_setzero_si512();
		}

		UNPACKHILO32_AVX512
		UNPACKHILO64_AVX512
		PERMUTEX2_VAR64_AVX512
		SHUFFLE64x2_AVX512

		// store to pack_b buffer
		_mm512_storeu_si512( pack_b_buffer + ( kr * NR ), a_reg[0] );
	}
}
#endif
