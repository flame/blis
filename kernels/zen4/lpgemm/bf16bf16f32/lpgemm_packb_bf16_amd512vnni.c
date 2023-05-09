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
#include <string.h>
#include "blis.h"

#ifdef BLIS_ADDON_LPGEMM

void packb_nrlt16_bf16bf16f32of32
    (
      bfloat16*       pack_b_buffer_bf16bf16f32of32,
      const bfloat16* b,
      const dim_t     ldb,
      const dim_t     KC, 
      const dim_t     n0_partial_rem
    );

void packb_nr16_bf16bf16f32of32
    (
      bfloat16*       pack_b_buffer_bf16bf16f32of32,
      const bfloat16* b,
      const dim_t     ldb,
      const dim_t     KC 
    );

void packb_nr32_bf16bf16f32of32
    (
      bfloat16*       pack_b_buffer_bf16bf16f32of32,
      const bfloat16* b,
      const dim_t     ldb,
      const dim_t     KC 
    );

void packb_nr48_bf16bf16f32of32
    (
      bfloat16*       pack_b_buffer_bf16bf16f32of32,
      const bfloat16* b,
      const dim_t     ldb,
      const dim_t     KC  
    );

void packb_nr64_bf16bf16f32of32
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
			packb_nr48_bf16bf16f32of32
				(
				 ( pack_b_buffer_bf16bf16f32of32 + ( n_full_pieces_loop_limit * KC_updated ) ),
				 ( b + n_full_pieces_loop_limit ), ldb, KC
				);

			n0_partial_pack = 48;
		}		
		else if ( n0_32 == 1 )
		{
			packb_nr32_bf16bf16f32of32
				( 
				 ( pack_b_buffer_bf16bf16f32of32 + ( n_full_pieces_loop_limit * KC_updated ) ),
				 ( b + n_full_pieces_loop_limit ), ldb, KC
				);

			n0_partial_pack = 32;
		}		
		else if ( n0_16 == 1 )
		{
			packb_nr16_bf16bf16f32of32
				(
				 ( pack_b_buffer_bf16bf16f32of32 + ( n_full_pieces_loop_limit * KC_updated ) ),
				 ( b + n_full_pieces_loop_limit ), ldb, KC
				);

			n0_partial_pack = 16;
		}	

		if ( n0_partial_rem > 0 )
		{
			packb_nrlt16_bf16bf16f32of32
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

void packb_nr48_bf16bf16f32of32
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

void packb_nr32_bf16bf16f32of32
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

void packb_nr16_bf16bf16f32of32
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

void packb_nrlt16_bf16bf16f32of32
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

	bfloat16 buf0[16];
	bfloat16 buf1[16];

	for ( int kr = 0; kr < k_full_pieces; kr += 2 )
	{            
		memcpy( buf0, ( b + ( ldb * ( kr + 0 ) ) ), ( n0_partial_rem * sizeof( bfloat16 ) ) );
		memcpy( buf1, ( b + ( ldb * ( kr + 1 ) ) ), ( n0_partial_rem * sizeof( bfloat16 ) ) );
		// Rearrange for dpbf16_ps, read 2 rows from B with next 16 elements in each row.
		a0 = _mm256_maskz_loadu_epi16( 0xFFFF, buf0 );
		c0 = _mm256_maskz_loadu_epi16( 0xFFFF, buf1 );

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
		memcpy( buf0, ( b + ( ldb * ( k_full_pieces + 0 ) ) ), ( n0_partial_rem * sizeof( bfloat16 ) ) );
		a0 = _mm256_maskz_loadu_epi16( 0xFFFF, buf0 );
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
#endif
