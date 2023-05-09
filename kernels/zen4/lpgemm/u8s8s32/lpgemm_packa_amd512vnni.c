/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022-2023, Advanced Micro Devices, Inc. All rights reserved.

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

#define MR 6
#define NR 64

void packa_m5_k64_u8s8s32o32
     (
       uint8_t*       pack_a_buffer_u8s8s32o32,
       const uint8_t* a,
       const dim_t    lda,
       const dim_t    KC
     );

void packa_m4_k64_u8s8s32o32
     (
       uint8_t*       pack_a_buffer_u8s8s32o32,
       const uint8_t* a,
       const dim_t    lda,
       const dim_t    KC
     );

void packa_m3_k64_u8s8s32o32
     (
       uint8_t*       pack_a_buffer_u8s8s32o32,
       const uint8_t* a,
       const dim_t    lda,
       const dim_t    KC
     );

void packa_m2_k64_u8s8s32o32
     (
       uint8_t*       pack_a_buffer_u8s8s32o32,
       const uint8_t* a,
       const dim_t    lda,
       const dim_t    KC
     );

void packa_m1_k64_u8s8s32o32
     (
       uint8_t*       pack_a_buffer_u8s8s32o32,
       const uint8_t* a,
       const dim_t    lda,
       const dim_t    KC
     );

// TODO: k fringe till k=4, k%4=0 and padding to make k'%4 = 0 if k%4 != 0 originally.
void packa_k64_u8s8s32o32
     (
       uint8_t*       pack_a_buffer_u8s8s32o32,
       const uint8_t* a,
       const dim_t    lda,
       const dim_t    MC,
       const dim_t    KC,
       dim_t*         rs_a,
       dim_t*         cs_a
     )
{
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

	__m512i a0;
	__m512i b0;
	__m512i c0;
	__m512i d0;
	__m512i e0;
	__m512i f0;
	__m512i a01;
	__m512i c01;
	__m512i e01;
	__m256i last_piece;

	for ( dim_t ic = 0; ic < m_full_pieces_loop_limit; ic += MR )
	{
		for ( dim_t kr = 0; kr < KC; kr += NR )
		{
			// Rearrange for vpdpbusd, read 6 rows from A with 64 elements in each row.
			a0 = _mm512_loadu_si512( a + ( lda * ( ic + 0 ) ) + kr );
			b0 = _mm512_loadu_si512( a + ( lda * ( ic + 1 ) ) + kr );
			c0 = _mm512_loadu_si512( a + ( lda * ( ic + 2 ) ) + kr );
			d0 = _mm512_loadu_si512( a + ( lda * ( ic + 3 ) ) + kr );
			e0 = _mm512_loadu_si512( a + ( lda * ( ic + 4 ) ) + kr );
			f0 = _mm512_loadu_si512( a + ( lda * ( ic + 5 ) ) + kr );

			a01 = _mm512_unpacklo_epi32( a0, b0 );
			a0 = _mm512_unpackhi_epi32( a0, b0 );

			c01 = _mm512_unpacklo_epi32( c0, d0 );
			c0 = _mm512_unpackhi_epi32( c0, d0 );

			e01 = _mm512_unpacklo_epi32( e0, f0 ); // Elem 4
			e0 = _mm512_unpackhi_epi32( e0, f0 ); // Elem 5

			b0 = _mm512_unpacklo_epi64( a01, c01 );
			a01 = _mm512_unpackhi_epi64( a01, c01 );

			d0 = _mm512_unpacklo_epi64( a0, c0 );
			c01 = _mm512_unpackhi_epi64( a0, c0 );

			a0 = _mm512_permutex2var_epi64( b0, selector1, a01 );
			c0 = _mm512_permutex2var_epi64( d0, selector1, c01 );
			b0 = _mm512_permutex2var_epi64( b0, selector1_1, a01 );
			d0 = _mm512_permutex2var_epi64( d0, selector1_1, c01 );

			a01 = _mm512_permutex2var_epi64( a0, selector2, c0 ); // a[0]
			c01 = _mm512_permutex2var_epi64( b0, selector2, d0 ); // a[2]
			a0 = _mm512_permutex2var_epi64( a0, selector2_1, c0 ); // a[1]
			c0 = _mm512_permutex2var_epi64( b0, selector2_1, d0 ); // a[3]

			// First half
			b0 = _mm512_permutex2var_epi64( a01, selector3, e01 ); // 1st 64
			a01 = _mm512_permutex2var_epi64( a01, selector4, e0 ); // 1st 32
			d0 = _mm512_permutex2var_epi64( a0, selector5, e01 ); // 2nd 64
			a0 = _mm512_permutex2var_epi64( a0, selector6, e0 ); // 2nd 32

			_mm512_storeu_si512( pack_a_buffer_u8s8s32o32 + ( ( ic * KC ) + ( ( kr * MR ) + ( 0 ) ) ), b0 );
			_mm512_storeu_si512( pack_a_buffer_u8s8s32o32 + ( ( ic * KC ) + ( ( kr * MR ) + ( 64 ) ) ) , a01 );
			_mm512_storeu_si512( pack_a_buffer_u8s8s32o32 + ( ( ic * KC ) + ( ( kr * MR ) + ( 96 ) ) ), d0 );
			// Last piece
			last_piece = _mm512_castsi512_si256( a0 );
			_mm256_mask_storeu_epi64
			(
			  pack_a_buffer_u8s8s32o32 + ( ( ic * KC ) + ( ( kr * MR ) + ( 160 ) ) ),
			  0xFF,
			  last_piece
			);

			// Second half
			b0 = _mm512_permutex2var_epi64( c01, selector7, e01 ); // 3rd 64
			c01 = _mm512_permutex2var_epi64( c01, selector8, e0 ); // 3rd 32
			d0 = _mm512_permutex2var_epi64( c0, selector9, e01 ); // 4th 64
			c0 = _mm512_permutex2var_epi64( c0, selector10, e0 ); // 4th 32

			_mm512_storeu_si512( pack_a_buffer_u8s8s32o32 + ( ( ic * KC ) + ( ( kr * MR ) + ( 192 ) ) ), b0 );
			_mm512_storeu_si512( pack_a_buffer_u8s8s32o32 + ( ( ic * KC ) + ( ( kr * MR ) + ( 256 ) ) ) , c01 );
			_mm512_storeu_si512( pack_a_buffer_u8s8s32o32 + ( ( ic * KC ) + ( ( kr * MR ) + ( 288 ) ) ), d0 );
			// Last piece
			last_piece = _mm512_castsi512_si256( c0 );
			_mm256_mask_storeu_epi64
			(
			  pack_a_buffer_u8s8s32o32 + ( ( ic * KC ) + ( ( kr * MR ) + ( 352 ) ) ),
			  0xFF,
			  last_piece
			);
		}
		//TODO: Handle kc < 64 case, 48,32,16
	}

	if ( m_partial_pieces > 0 )
	{
		if ( m_partial_pieces == 5 )
		{
			packa_m5_k64_u8s8s32o32
			(
			  pack_a_buffer_u8s8s32o32 +  ( m_full_pieces_loop_limit * KC ),
			  a + ( lda * m_full_pieces_loop_limit ), lda, KC
			);
		}
		else if ( m_partial_pieces == 4 )
		{
			packa_m4_k64_u8s8s32o32
			(
			  pack_a_buffer_u8s8s32o32 + ( m_full_pieces_loop_limit * KC ),
			  a + ( lda * m_full_pieces_loop_limit ), lda, KC
			);
		}
		else if ( m_partial_pieces == 3 )
		{
			packa_m3_k64_u8s8s32o32
			(
			  pack_a_buffer_u8s8s32o32 + ( m_full_pieces_loop_limit * KC ),
			  a + ( lda * m_full_pieces_loop_limit ), lda, KC
			);
		}
		else if ( m_partial_pieces == 2 )
		{
			packa_m2_k64_u8s8s32o32
			(
			  pack_a_buffer_u8s8s32o32 + ( m_full_pieces_loop_limit * KC ),
			  a + ( lda * m_full_pieces_loop_limit ), lda, KC
			);
		}
		else if ( m_partial_pieces == 1 )
		{
			packa_m1_k64_u8s8s32o32
			(
			  pack_a_buffer_u8s8s32o32 + ( m_full_pieces_loop_limit * KC ),
			  a + ( lda * m_full_pieces_loop_limit ), lda, KC
			);
		}
	}
	*rs_a = 4;
	*cs_a = 24;
}

void packa_m5_k64_u8s8s32o32
     (
       uint8_t*       pack_a_buffer_u8s8s32o32,
       const uint8_t* a,
       const dim_t    lda,
       const dim_t    KC
     )
{
	// Used for permuting the mm512i elements for use in vpdpbusd instruction.
	// These are indexes of the format a0-a1-b0-b1-a2-a3-b2-b3 and a0-a1-a2-a3-b0-b1-b2-b3.
	// Adding 4 int32 wise gives format a4-a5-b4-b5-a6-a7-b6-b7 and a4-a5-a6-a7-b4-b5-b6-b7.
	__m512i selector1 = _mm512_setr_epi64( 0x0, 0x1, 0x8, 0x9, 0x2, 0x3, 0xA, 0xB );
	__m512i selector1_1 = _mm512_setr_epi64( 0x4, 0x5, 0xC, 0xD, 0x6, 0x7, 0xE, 0xF );
	__m512i selector2 = _mm512_setr_epi64( 0x0, 0x1, 0x2, 0x3, 0x8, 0x9, 0xA, 0xB );
	__m512i selector2_1 = _mm512_setr_epi64( 0x4, 0x5, 0x6, 0x7, 0xC, 0xD, 0xE, 0xF );

	// First half.
	__m512i selector3 = _mm512_setr_epi32( 0x0, 0x1, 0x2, 0x3, 0x10, 0x4, 0x5, 0x6, 0x7, 0x11, 0x8, 0x9, 0xA, 0xB, 0x12, 0xC);
	__m512i selector4 = _mm512_setr_epi32( 0xD, 0xE, 0xF, 0x13, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0);
	__m512i selector5 = _mm512_setr_epi32( 0x0, 0x1, 0x2, 0x3, 0x14, 0x4, 0x5, 0x6, 0x7, 0x15, 0x8, 0x9, 0xA, 0xB, 0x16, 0xC);
	__m512i selector6 = _mm512_setr_epi32( 0xD, 0xE, 0xF, 0x17, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0);

	// Second half.
	__m512i selector7 = _mm512_setr_epi32( 0x0, 0x1, 0x2, 0x3, 0x18, 0x4, 0x5, 0x6, 0x7, 0x19, 0x8, 0x9, 0xA, 0xB, 0x1A, 0xC);
	__m512i selector8 = _mm512_setr_epi32( 0xD, 0xE, 0xF, 0x1B, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0);
	__m512i selector9 = _mm512_setr_epi32( 0x0, 0x1, 0x2, 0x3, 0x1C, 0x4, 0x5, 0x6, 0x7, 0x1D, 0x8, 0x9, 0xA, 0xB, 0x1E, 0xC);
	__m512i selector10 = _mm512_setr_epi32( 0xD, 0xE, 0xF, 0x1F, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0);
	
	__m512i a0;
	__m512i b0;
	__m512i c0;
	__m512i d0;
	__m512i e0;
	__m512i a01;
	__m512i c01;
	__m128i last_piece;

	for ( dim_t kr = 0; kr < KC; kr += NR )
	{
		// Rearrange for vpdpbusd, read 5 rows from A with 64 elements in each row.
		a0 = _mm512_loadu_si512( a + ( lda * 0 ) + kr );
		b0 = _mm512_loadu_si512( a + ( lda * 1 ) + kr );
		c0 = _mm512_loadu_si512( a + ( lda * 2 ) + kr );
		d0 = _mm512_loadu_si512( a + ( lda * 3 ) + kr );
		e0 = _mm512_loadu_si512( a + ( lda * 4 ) + kr );

		a01 = _mm512_unpacklo_epi32( a0, b0 );
		a0 = _mm512_unpackhi_epi32( a0, b0 );

		c01 = _mm512_unpacklo_epi32( c0, d0 );
		c0 = _mm512_unpackhi_epi32( c0, d0 );

		b0 = _mm512_unpacklo_epi64( a01, c01 );
		a01 = _mm512_unpackhi_epi64( a01, c01 );

		d0 = _mm512_unpacklo_epi64( a0, c0 );
		c01 = _mm512_unpackhi_epi64( a0, c0 );

		a0 = _mm512_permutex2var_epi64( b0, selector1, a01 );
		c0 = _mm512_permutex2var_epi64( d0, selector1, c01 );
		b0 = _mm512_permutex2var_epi64( b0, selector1_1, a01 );
		d0 = _mm512_permutex2var_epi64( d0, selector1_1, c01 );

		a01 = _mm512_permutex2var_epi64( a0, selector2, c0 ); // a[0]
		c01 = _mm512_permutex2var_epi64( b0, selector2, d0 ); // a[2]
		a0 = _mm512_permutex2var_epi64( a0, selector2_1, c0 ); // a[1]
		c0 = _mm512_permutex2var_epi64( b0, selector2_1, d0 ); // a[3]

		// First half
		b0 = _mm512_permutex2var_epi32( a01, selector3, e0 );
		a01 = _mm512_permutex2var_epi32( a01, selector4, e0 );
		d0 = _mm512_permutex2var_epi32( a0, selector5, e0 );
		a0 = _mm512_permutex2var_epi32( a0, selector6, e0 );

		_mm512_storeu_si512( pack_a_buffer_u8s8s32o32 + ( ( kr * 5 ) + ( 0 ) ), b0 );
		_mm512_storeu_si512( pack_a_buffer_u8s8s32o32 + ( ( kr * 5 ) + ( 64 ) ) , a01 );
		_mm512_storeu_si512( pack_a_buffer_u8s8s32o32 + ( ( kr * 5 ) + ( 80 ) ), d0 );
		// Last piece
		last_piece = _mm512_castsi512_si128( a0 );
		_mm_mask_storeu_epi64
		(
		  pack_a_buffer_u8s8s32o32 + ( ( kr * 5 ) + ( 144 ) ),
		  0xFF,
		  last_piece
		);

		// Second half
		b0 = _mm512_permutex2var_epi32( c01, selector7, e0 );
		c01 = _mm512_permutex2var_epi32( c01, selector8, e0 );
		d0 = _mm512_permutex2var_epi32( c0, selector9, e0 );
		c0 = _mm512_permutex2var_epi32( c0, selector10, e0 );

		_mm512_storeu_si512( pack_a_buffer_u8s8s32o32 + ( ( kr * 5 ) + ( 160 ) ), b0 );
		_mm512_storeu_si512( pack_a_buffer_u8s8s32o32 + ( ( kr * 5 ) + ( 224 ) ) , c01 );
		_mm512_storeu_si512( pack_a_buffer_u8s8s32o32 + ( ( kr * 5 ) + ( 240 ) ), d0 );
		// Last piece
		last_piece = _mm512_castsi512_si128( c0 );
		_mm_mask_storeu_epi64
		(
		  pack_a_buffer_u8s8s32o32 + ( ( kr * 5 ) + ( 304 ) ),
		  0xFF,
		  last_piece
		);
	}
}

void packa_m4_k64_u8s8s32o32
     (
       uint8_t*       pack_a_buffer_u8s8s32o32,
       const uint8_t* a,
       const dim_t    lda,
       const dim_t    KC
     )
{
	// Used for permuting the mm512i elements for use in vpdpbusd instruction.
	// These are indexes of the format a0-a1-b0-b1-a2-a3-b2-b3 and a0-a1-a2-a3-b0-b1-b2-b3.
	// Adding 4 int32 wise gives format a4-a5-b4-b5-a6-a7-b6-b7 and a4-a5-a6-a7-b4-b5-b6-b7.
	__m512i selector1 = _mm512_setr_epi64( 0x0, 0x1, 0x8, 0x9, 0x2, 0x3, 0xA, 0xB );
	__m512i selector1_1 = _mm512_setr_epi64( 0x4, 0x5, 0xC, 0xD, 0x6, 0x7, 0xE, 0xF );
	__m512i selector2 = _mm512_setr_epi64( 0x0, 0x1, 0x2, 0x3, 0x8, 0x9, 0xA, 0xB );
	__m512i selector2_1 = _mm512_setr_epi64( 0x4, 0x5, 0x6, 0x7, 0xC, 0xD, 0xE, 0xF );
	
	__m512i a0;
	__m512i b0;
	__m512i c0;
	__m512i d0;
	__m512i a01;
	__m512i c01;

	for ( dim_t kr = 0; kr < KC; kr += NR )
	{
		// Rearrange for vpdpbusd, read 4 rows from A with 64 elements in each row.
		a0 = _mm512_loadu_si512( a + ( lda * 0 ) + kr );
		b0 = _mm512_loadu_si512( a + ( lda * 1 ) + kr );
		c0 = _mm512_loadu_si512( a + ( lda * 2 ) + kr );
		d0 = _mm512_loadu_si512( a + ( lda * 3 ) + kr );

		a01 = _mm512_unpacklo_epi32( a0, b0 );
		a0 = _mm512_unpackhi_epi32( a0, b0 );

		c01 = _mm512_unpacklo_epi32( c0, d0 );
		c0 = _mm512_unpackhi_epi32( c0, d0 );

		b0 = _mm512_unpacklo_epi64( a01, c01 );
		a01 = _mm512_unpackhi_epi64( a01, c01 );

		d0 = _mm512_unpacklo_epi64( a0, c0 );
		c01 = _mm512_unpackhi_epi64( a0, c0 );

		a0 = _mm512_permutex2var_epi64( b0, selector1, a01 );
		c0 = _mm512_permutex2var_epi64( d0, selector1, c01 );
		b0 = _mm512_permutex2var_epi64( b0, selector1_1, a01 );
		d0 = _mm512_permutex2var_epi64( d0, selector1_1, c01 );

		a01 = _mm512_permutex2var_epi64( a0, selector2, c0 ); // a[0]
		c01 = _mm512_permutex2var_epi64( b0, selector2, d0 ); // a[2]
		a0 = _mm512_permutex2var_epi64( a0, selector2_1, c0 ); // a[1]
		c0 = _mm512_permutex2var_epi64( b0, selector2_1, d0 ); // a[3]

		_mm512_storeu_si512( pack_a_buffer_u8s8s32o32 + ( ( kr * 4 ) + ( 0 ) ), a01 );
		_mm512_storeu_si512( pack_a_buffer_u8s8s32o32 + ( ( kr * 4 ) + ( 64 ) ) , a0 );
		_mm512_storeu_si512( pack_a_buffer_u8s8s32o32 + ( ( kr * 4 ) + ( 128 ) ), c01 );
		_mm512_storeu_si512( pack_a_buffer_u8s8s32o32 + ( ( kr * 4 ) + ( 192 ) ), c0 );
	}
}

void packa_m3_k64_u8s8s32o32
     (
       uint8_t*       pack_a_buffer_u8s8s32o32,
       const uint8_t* a,
       const dim_t    lda,
       const dim_t    KC
     )
{
	// Used for permuting the mm512i elements for use in vpdpbusd instruction.
	// These are indexes of the format a0-a1-b0-b1-a2-a3-b2-b3 and a0-a1-a2-a3-b0-b1-b2-b3.
	// Adding 4 int32 wise gives format a4-a5-b4-b5-a6-a7-b6-b7 and a4-a5-a6-a7-b4-b5-b6-b7.
	__m512i selector1 = _mm512_setr_epi64( 0x0, 0x1, 0x8, 0x9, 0x2, 0x3, 0xA, 0xB );
	__m512i selector1_1 = _mm512_setr_epi64( 0x4, 0x5, 0xC, 0xD, 0x6, 0x7, 0xE, 0xF );

	// First half
	__m512i selector3 = _mm512_setr_epi32( 0x0, 0x1, 0x10, 0x2, 0x3, 0x11, 0x4, 0x5, 0x12, 0x6, 0x7, 0x13, 0x8, 0x9, 0x14, 0xA );
	__m512i selector4 = _mm512_setr_epi32( 0xB, 0x15, 0xC, 0xD, 0x16, 0xE, 0xF, 0x17, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 );

	// Second half
	__m512i selector5 = _mm512_setr_epi32( 0x0, 0x1, 0x18, 0x2, 0x3, 0x19, 0x4, 0x5, 0x1A, 0x6, 0x7, 0x1B, 0x8, 0x9, 0x1C, 0xA );
	__m512i selector6 = _mm512_setr_epi32( 0xB, 0x1D, 0xC, 0xD, 0x1E, 0xE, 0xF, 0x1F, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 );
	
	__m512i a0;
	__m512i b0;
	__m512i c0;
	__m512i a01;
	__m256i last_piece;

	for ( dim_t kr = 0; kr < KC; kr += NR )
	{
		// Rearrange for vpdpbusd, read 3 rows from A with 64 elements in each row.
		a0 = _mm512_loadu_si512( a + ( lda * 0 ) + kr );
		b0 = _mm512_loadu_si512( a + ( lda * 1 ) + kr );
		c0 = _mm512_loadu_si512( a + ( lda * 2 ) + kr );

		a01 = _mm512_unpacklo_epi32( a0, b0 );
		a0 = _mm512_unpackhi_epi32( a0, b0 );

		b0 = _mm512_permutex2var_epi64( a01, selector1, a0 ); // a[0]
		a01 = _mm512_permutex2var_epi64( a01, selector1_1, a0 ); // a[1]

		a0 = _mm512_permutex2var_epi32( b0, selector3, c0 );
		b0 = _mm512_permutex2var_epi32( b0, selector4, c0 );

		_mm512_storeu_si512( pack_a_buffer_u8s8s32o32 + ( ( kr * 3 ) + ( 0 ) ), a0 );
		_mm512_storeu_si512( pack_a_buffer_u8s8s32o32 + ( ( kr * 3 ) + ( 64 ) ) , b0 );

		a0 = _mm512_permutex2var_epi32( a01, selector5, c0 );
		b0 = _mm512_permutex2var_epi32( a01, selector6, c0 );

		_mm512_storeu_si512( pack_a_buffer_u8s8s32o32 + ( ( kr * 3 ) + ( 96 ) ), a0 );
		// Last piece
		last_piece = _mm512_castsi512_si256( b0 );
		_mm256_mask_storeu_epi64
		(
		  pack_a_buffer_u8s8s32o32 + ( ( kr * 3 ) + ( 160 ) ),
		  0xFF,
		  last_piece
		);
	}
}

void packa_m2_k64_u8s8s32o32
     (
       uint8_t*       pack_a_buffer_u8s8s32o32,
       const uint8_t* a,
       const dim_t    lda,
       const dim_t    KC
     )
{
	// Used for permuting the mm512i elements for use in vpdpbusd instruction.
	// These are indexes of the format a0-a1-b0-b1-a2-a3-b2-b3 and a0-a1-a2-a3-b0-b1-b2-b3.
	// Adding 4 int32 wise gives format a4-a5-b4-b5-a6-a7-b6-b7 and a4-a5-a6-a7-b4-b5-b6-b7.
	__m512i selector1 = _mm512_setr_epi64( 0x0, 0x1, 0x8, 0x9, 0x2, 0x3, 0xA, 0xB );
	__m512i selector1_1 = _mm512_setr_epi64( 0x4, 0x5, 0xC, 0xD, 0x6, 0x7, 0xE, 0xF );
	
	__m512i a0;
	__m512i b0;
	__m512i a01;

	for ( dim_t kr = 0; kr < KC; kr += NR )
	{
		// Rearrange for vpdpbusd, read 2 rows from A with 64 elements in each row.
		a0 = _mm512_loadu_si512( a + ( lda * 0 ) + kr );
		b0 = _mm512_loadu_si512( a + ( lda * 1 ) + kr );

		a01 = _mm512_unpacklo_epi32( a0, b0 );
		a0 = _mm512_unpackhi_epi32( a0, b0 );

		b0 = _mm512_permutex2var_epi64( a01, selector1, a0 ); // a[0]
		a01 = _mm512_permutex2var_epi64( a01, selector1_1, a0 ); // a[1]

		_mm512_storeu_si512( pack_a_buffer_u8s8s32o32 + ( ( kr * 2 ) + ( 0 ) ), b0 );
		_mm512_storeu_si512( pack_a_buffer_u8s8s32o32 + ( ( kr * 2 ) + ( 64 ) ) , a01 );
	}
}

void packa_m1_k64_u8s8s32o32
     (
       uint8_t*       pack_a_buffer_u8s8s32o32,
       const uint8_t* a,
       const dim_t    lda,
       const dim_t    KC
     )
{
	__m512i a0;

	for ( dim_t kr = 0; kr < KC; kr += NR )
	{
		// Rearrange for vpdpbusd, read 1 row from A with 64 elements in each row.
		a0 = _mm512_loadu_si512( a + ( lda * 0 ) + kr );

		_mm512_storeu_si512( pack_a_buffer_u8s8s32o32 + ( ( kr * 1 ) + ( 0 ) ), a0 );
	}
}
#endif
