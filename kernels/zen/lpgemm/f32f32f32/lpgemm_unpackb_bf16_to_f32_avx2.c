/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

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
// BF16 -> F32 convert helpers. reg: __m256 - CVT_BF16_F32_SHIFT_AVX2
#include "lpgemm_kernel_macros_f32_avx2.h"

#define LOAD_AND_CONVERT_FIRST32_NR64_BF16_NR32_AVX2(k)  \
	a_reg[0] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( ( jc * KC_updated ) + ( k + 0 ) * NR ) + 0 ) ) );	\
	a_reg[1] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( ( jc * KC_updated ) +  ( k + 0 ) * NR ) + 8 ) ) );	\
	a_reg[2] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( ( jc * KC_updated ) +  ( k + 0 ) * NR ) + 16 ) ) );	\
	a_reg[3] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( ( jc * KC_updated ) +  ( k + 0 ) * NR ) + 24 ) ) );	\
\
	a_reg[4] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( ( jc * KC_updated ) + ( ( k + 1 ) * NR) ) + 0 ) ) );	\
	a_reg[5] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( ( jc * KC_updated ) +  (( k + 1 ) * NR )) + 8 ) ) );	\
	a_reg[6] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( ( jc * KC_updated ) +  (( k + 1 ) * NR) ) + 16 ) ) ); 	\
	a_reg[7] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( ( jc * KC_updated ) + ( ( k + 1 ) * NR) ) + 24 ) ) );

#define LOAD_AND_CONVERT_LAST32_NR64_BF16_AVX2(k)  \
	a_reg[0] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( jc * KC_updated ) + ( ( k + 0 ) * NR ) + 32 ) ) );	\
	a_reg[1] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( jc * KC_updated ) + ( ( k + 0 ) * NR ) + 40 ) ) );	\
	a_reg[2] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( jc * KC_updated ) + ( ( k + 0 ) * NR ) + 48 ) ) );	\
	a_reg[3] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( jc * KC_updated ) + ( ( k + 0 ) * NR ) + 56 ) ) );	\
\
	a_reg[4] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( jc * KC_updated ) + ( ( k + 1 ) * NR ) + 32 ) ) );	\
	a_reg[5] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( jc * KC_updated ) + ( ( k + 1 ) * NR ) + 40 ) ) );	\
	a_reg[6] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( jc * KC_updated ) + ( ( k + 1 ) * NR ) + 48 ) ) );	\
	a_reg[7] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( jc * KC_updated ) + ( ( k + 1 ) * NR ) + 56 ) ) );

#define LOAD_AND_CONVERT_NR32_BF16_NR32_AVX2(k)  \
	a_reg[0] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( ( k + 0 ) * NR ) + 0 ) ) );	\
	a_reg[1] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( ( k + 0 ) * NR ) + 8 ) ) );	\
	a_reg[2] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( ( k + 0 ) * NR ) + 16 ) ) );	\
	a_reg[3] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( ( k + 0 ) * NR ) + 24 ) ) );	\
\
	a_reg[4] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( ( k + 1 ) * NR ) + 0 ) ) );	\
	a_reg[5] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( ( k + 1 ) * NR ) + 8 ) ) );	\
	a_reg[6] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( ( k + 1 ) * NR ) + 16 ) ) ); 	\
	a_reg[7] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( ( k + 1 ) * NR ) + 24 ) ) );


#define LOAD_AND_CONVERT_FIRST32_BF16_NR32_AVX2(k)  \
	a_reg[0] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( ( k + 0 ) * NR ) + 0 ) ) );	\
	a_reg[1] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( ( k + 0 ) * NR ) + 8 ) ) );	\
	a_reg[2] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( ( k + 0 ) * NR ) + 16 ) ) );	\
	a_reg[3] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( ( k + 0 ) * NR ) + 24 ) ) );	\
\
	a_reg[4] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( ( k + 1 ) * NR ) + 0 ) ) );	\
	a_reg[5] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( ( k + 1 ) * NR ) + 8 ) ) );	\
	a_reg[6] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( ( k + 1 ) * NR ) + 16 ) ) ); 	\
	a_reg[7] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( ( k + 1 ) * NR ) + 24 ) ) );

#define SHUFFLE_F32_ELEMENTS_NR32_AVX2    \
	b_reg[0] = _mm256_shuffle_ps( a_reg[0], a_reg[1] , 0x88 );   \
	b_reg[1] = _mm256_shuffle_ps( a_reg[2], a_reg[3] , 0x88 );   \
	b_reg[2] = _mm256_shuffle_ps( a_reg[0], a_reg[1] , 0xDD );   \
	b_reg[3] = _mm256_shuffle_ps( a_reg[2], a_reg[3] , 0xDD );   \
	b_reg[4] = _mm256_shuffle_ps( a_reg[4], a_reg[5] , 0x88 );   \
	b_reg[5] = _mm256_shuffle_ps( a_reg[6], a_reg[7] , 0x88 );   \
	b_reg[6] = _mm256_shuffle_ps( a_reg[4], a_reg[5] , 0xDD );   \
	b_reg[7] = _mm256_shuffle_ps( a_reg[6], a_reg[7] , 0xDD );

#define PERMUTE_NR32_ELEMENTS_AVX2   \
	a_reg[0] = (__m256)_mm256_permute4x64_epi64( (__m256i)b_reg[0], 0xD8 );   \
	a_reg[1] = (__m256)_mm256_permute4x64_epi64( (__m256i)b_reg[1], 0xD8 );   \
	a_reg[2] = (__m256)_mm256_permute4x64_epi64( (__m256i)b_reg[2], 0xD8 );   \
	a_reg[3] = (__m256)_mm256_permute4x64_epi64( (__m256i)b_reg[3], 0xD8 );   \
	a_reg[4] = (__m256)_mm256_permute4x64_epi64( (__m256i)b_reg[4], 0xD8 );   \
	a_reg[5] = (__m256)_mm256_permute4x64_epi64( (__m256i)b_reg[5], 0xD8 );   \
	a_reg[6] = (__m256)_mm256_permute4x64_epi64( (__m256i)b_reg[6], 0xD8 );   \
	a_reg[7] = (__m256)_mm256_permute4x64_epi64( (__m256i)b_reg[7], 0xD8 );

#define LOAD_AND_CONVERT_BF16_NR16_AVX2(k)  \
	a_reg[0] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( ( k + 0 ) * NR ) + 0 ) ) );	\
	a_reg[1] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( ( k + 0 ) * NR ) + 8 ) ) );	\
	a_reg[2] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( ( k + 1 ) * NR ) + 0 ) ) );	\
	a_reg[3] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128( \
		(__m128i const*)(b + ( ( k + 1 ) * NR ) + 8 ) ) );

#define SHUFFLE_F32_ELEMENTS_NR16_AVX2    \
	b_reg[0] = _mm256_shuffle_ps( a_reg[0], a_reg[1] , 0x88 );   \
	b_reg[1] = _mm256_shuffle_ps( a_reg[2], a_reg[3] , 0x88 );   \
	b_reg[2] = _mm256_shuffle_ps( a_reg[0], a_reg[1] , 0xDD );   \
	b_reg[3] = _mm256_shuffle_ps( a_reg[2], a_reg[3] , 0xDD );

#define PERMUTE_NR16_ELEMENTS_AVX2   \
	a_reg[0] = (__m256)_mm256_permute4x64_epi64( (__m256i)b_reg[0], 0xD8 );   \
	a_reg[1] = (__m256)_mm256_permute4x64_epi64( (__m256i)b_reg[1], 0xD8 );   \
	a_reg[2] = (__m256)_mm256_permute4x64_epi64( (__m256i)b_reg[2], 0xD8 );   \
	a_reg[3] = (__m256)_mm256_permute4x64_epi64( (__m256i)b_reg[3], 0xD8 );   \

void unpackb_nr48_bf16_f32_row_major
	(
	  const bfloat16* b,
	  float*       unpack_b_buffer,
	  const dim_t     KC,
	  dim_t           ldb_unpack
	)
{
	dim_t NR = 32;

	__m256 a_reg[8], b_reg[8];

	dim_t k_full_pieces_blks = KC / 2;
	dim_t k_full_pieces = k_full_pieces_blks * 2;
	dim_t k_partial_pieces = KC % 2;

	dim_t kr_new = 0;

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 2 )
	{
		/*Read first rows from B with 32 elements in each row.*/
		LOAD_AND_CONVERT_NR32_BF16_NR32_AVX2(kr_new)
		SHUFFLE_F32_ELEMENTS_NR32_AVX2
		PERMUTE_NR32_ELEMENTS_AVX2

		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 0 ) ) + 0, a_reg[0] );
		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 0 ) ) + 8, a_reg[1] );
		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 0 ) ) + 16, a_reg[4] );
		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 0 ) ) + 24, a_reg[5] );
		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 1 ) ) + 0, a_reg[2] );
		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 1 ) ) + 8, a_reg[3] );
		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 1 ) ) + 16, a_reg[6] );
		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 1 ) ) + 24, a_reg[7] );

		/* Read the remaining 16 elements from the first two rows*/
		a_reg[0] = CVT_BF16_F32_SHIFT_AVX2(	(__m128i)_mm_loadu_si128(
				(__m128i const*)(b + ( ( kr_new + 2 ) * NR ) + 0 ) ) );
		a_reg[1] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128(
				(__m128i const*)(b + ( ( kr_new + 2 ) * NR ) + 8 ) ) );
		a_reg[2] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128(
				(__m128i const*)(b + ( ( kr_new + 2 ) * NR ) + 16 ) ) );
		a_reg[3] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128(
				(__m128i const*)(b + ( ( kr_new + 2 ) * NR ) + 24 ) ) );

		SHUFFLE_F32_ELEMENTS_NR16_AVX2
		PERMUTE_NR16_ELEMENTS_AVX2

		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 0 ) ) + NR + 0,
							a_reg[0] );
		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 0 ) ) + NR + 8,
							a_reg[1] );
		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 1 ) ) + NR + 0,
							a_reg[2] );
		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 1 ) ) + NR + 8,
							a_reg[3] );

		kr_new += 3;
	}
	if( k_partial_pieces > 0 )
	{
		LOAD_AND_CONVERT_NR32_BF16_NR32_AVX2(kr_new)
		SHUFFLE_F32_ELEMENTS_NR32_AVX2
		PERMUTE_NR32_ELEMENTS_AVX2

		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( k_full_pieces + 0 )) + 0,
							a_reg[0] );
		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( k_full_pieces + 0 )) + 8,
							a_reg[1] );
		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( k_full_pieces + 0 )) + 16,
							a_reg[4] );
		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( k_full_pieces + 0 )) + 24,
							a_reg[5] );

		/* Read the remaining 16 elements from the first two rows*/
		a_reg[0] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128(
				(__m128i const*)(b + ( ( kr_new + 2 ) * NR ) + 0 ) ) );
		a_reg[1] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128(
				(__m128i const*)(b + ( ( kr_new + 2 ) * NR ) + 8 ) ) );
		a_reg[2] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128(
				(__m128i const*)(b + ( ( kr_new + 2 ) * NR ) + 16 ) ) );
		a_reg[3] = CVT_BF16_F32_SHIFT_AVX2( (__m128i)_mm_loadu_si128(
				(__m128i const*)(b + ( ( kr_new + 2 ) * NR ) + 24 ) ) );

		SHUFFLE_F32_ELEMENTS_NR16_AVX2
		PERMUTE_NR16_ELEMENTS_AVX2

		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( k_full_pieces + 0 ) )
							+ NR + 0, a_reg[0] );
		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( k_full_pieces + 0 ) )
							+ NR + 8, a_reg[1] );
	}
}

void unpackb_nr32_bf16_f32_row_major
	(
	  const bfloat16* b,
	  float*       unpack_b_buffer,
	  const dim_t     KC,
	  dim_t           ldb_unpack
	)
{
	dim_t NR = 32;

	dim_t k_full_pieces_blks = KC / 2;
	dim_t k_full_pieces = k_full_pieces_blks * 2;
	dim_t k_partial_pieces = KC % 2;

	__m256 a_reg[8], b_reg[8];

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 2 )
	{
		/*Read two rows from B with 32 elements each.*/
		LOAD_AND_CONVERT_NR32_BF16_NR32_AVX2(kr)
		SHUFFLE_F32_ELEMENTS_NR32_AVX2
		PERMUTE_NR32_ELEMENTS_AVX2

		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 0 ) ) + 0, a_reg[0] );
		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 0 ) ) + 8, a_reg[1] );
		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 0 ) ) + 16, a_reg[4] );
		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 0 ) ) + 24, a_reg[5] );
		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 1 ) ) + 0, a_reg[2] );
		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 1 ) ) + 8, a_reg[3] );
		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 1 ) ) + 16, a_reg[6] );
		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 1 ) ) + 24, a_reg[7] );
	}
	if( k_partial_pieces > 0 )
	{
		LOAD_AND_CONVERT_NR32_BF16_NR32_AVX2(k_full_pieces)
		SHUFFLE_F32_ELEMENTS_NR32_AVX2
		PERMUTE_NR32_ELEMENTS_AVX2

		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( k_full_pieces + 0 ) ) + 0,
								a_reg[0]);
		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( k_full_pieces + 0 ) ) + 8,
								a_reg[1]);
		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( k_full_pieces + 0 ) ) + 16,
								a_reg[4]);
		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( k_full_pieces + 0 ) ) + 24,
								a_reg[5]);
	}
}

void unpackb_nr16_bf16_f32_row_major
	(
	  const bfloat16* b,
	  float*       unpack_b_buffer,
	  const dim_t     KC,
	  dim_t           ldb_unpack
	)
{
	dim_t NR = 16;

	__m256 a_reg[4], b_reg[4];

	dim_t k_full_pieces_blks = KC / 2;
	dim_t k_full_pieces = k_full_pieces_blks * 2;
	dim_t k_partial_pieces = KC % 2;

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 2 )
	{
		//Read 2-rows with 16 elements in each row
		LOAD_AND_CONVERT_BF16_NR16_AVX2(kr)
		SHUFFLE_F32_ELEMENTS_NR16_AVX2
		PERMUTE_NR16_ELEMENTS_AVX2

		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 0 ) ), a_reg[0] );
		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 0 ) + 8 ), a_reg[1] );
		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 1 ) ), a_reg[2] );
		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 1 ) + 8 ), a_reg[3] );
	}
	if( k_partial_pieces > 0 )
	{
		LOAD_AND_CONVERT_BF16_NR16_AVX2(k_full_pieces)
		SHUFFLE_F32_ELEMENTS_NR16_AVX2
		PERMUTE_NR16_ELEMENTS_AVX2

		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( k_full_pieces + 0 ) ),
								a_reg[0] );
		_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( k_full_pieces + 0 ) + 8 ),
								a_reg[1] );
	}
}

void unpackb_nrlt16_bf16_f32_row_major
	(
	  const bfloat16* b,
	  float*       unpack_b_buffer,
	  const dim_t     KC,
	  dim_t           ldb_unpack,
	  dim_t           n0_partial_rem
	)
{
	dim_t NR = 16;
	__m256 a_reg[4], b_reg[4];


	/*
	In case of BF16 re-ordered buffer padding would enable availability of 16 elements
	even though NR < 16. Hence, masks isn't neeeded for loading. But after the conversion
	to F32 only max of 8 elements could be stored at a time. For ex., if n0_partial_rem = 11
	after the conversion and permute there would be 8 elements in a_reg[i] and 3 elements in
	a_reg[i+1]. Hence, there is a need for 2 store masks calculated based on the n0_partial_rem.
	*/

	dim_t mask1, mask2;
	__m256i store_mask1, store_mask2;

	if( n0_partial_rem > 7 )
	{
		mask1 = 8;
		mask2 = ( n0_partial_rem - 8 );
	}
	else
	{
		mask1 = n0_partial_rem;
		mask2 = 0;
	}

	GET_STORE_MASK(mask1,store_mask1);
	GET_STORE_MASK(mask2,store_mask2);

	dim_t k_full_pieces_blks = KC / 2;
	dim_t k_full_pieces = k_full_pieces_blks * 2;
	dim_t k_partial_pieces = KC % 2;

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 2 )
	{
		LOAD_AND_CONVERT_BF16_NR16_AVX2(kr)
		SHUFFLE_F32_ELEMENTS_NR16_AVX2
		PERMUTE_NR16_ELEMENTS_AVX2

		/*store using storemask*/
		_mm256_maskstore_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 0 ) ),
								store_mask1, a_reg[0] );
		_mm256_maskstore_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 0 ) + 8 ),
								store_mask2, a_reg[1]);
		_mm256_maskstore_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 1 ) ),
								store_mask1, a_reg[2] );
		_mm256_maskstore_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 1 ) + 8 ),
								store_mask2, a_reg[3]);
	}
	if( k_partial_pieces > 0 )
	{
		LOAD_AND_CONVERT_BF16_NR16_AVX2(k_full_pieces)
		SHUFFLE_F32_ELEMENTS_NR16_AVX2
		PERMUTE_NR16_ELEMENTS_AVX2

		/*store using storemask*/
		_mm256_maskstore_ps( unpack_b_buffer + ( ldb_unpack * ( k_full_pieces + 0 ) ),
								store_mask1, a_reg[0] );
		_mm256_maskstore_ps( unpack_b_buffer + ( ldb_unpack * ( k_full_pieces + 0 ) + 8 ),
								store_mask2, a_reg[1] );
		_mm256_maskstore_ps( unpack_b_buffer + ( ldb_unpack * ( k_full_pieces + 1 ) ),
								store_mask1, a_reg[2] );
		_mm256_maskstore_ps( unpack_b_buffer + ( ldb_unpack * ( k_full_pieces + 1 ) + 8 ),
								store_mask2, a_reg[3] );
	}
}

void unpackb_nr64_bf16_f32_row_major
	(
	  const bfloat16* b,
	  float*       unpack_b_buffer,
	  const dim_t     NC,
	  const dim_t     KC,
	  dim_t           ldb_unpack
	)
{
	dim_t NR = 64;

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

	/*Since there are only 16 registers, we do the NR=64 in blocks of 32*/
	__m256 a_reg[8], b_reg[8];

	for( dim_t jc = 0; jc < n_full_pieces_loop_limit; jc += NR )
	{
		for( dim_t kr = 0; kr < k_full_pieces; kr += 2 )
		{
			//Load 2-rows with 64 elements each.
			LOAD_AND_CONVERT_FIRST32_NR64_BF16_NR32_AVX2(kr)
			SHUFFLE_F32_ELEMENTS_NR32_AVX2
			PERMUTE_NR32_ELEMENTS_AVX2

			_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 0 ) ) + jc + 0,
								a_reg[0] );
			_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 0 ) ) + jc + 8,
								a_reg[1] );
			_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 0 ) ) + jc + 32,
								a_reg[4] );
			_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 0 ) ) + jc + 40,
								a_reg[5] );

			_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 1 ) ) + jc + 0,
								a_reg[2] );
			_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 1 ) ) + jc + 8,
								a_reg[3] );
			_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 1 ) ) + jc + 32,
								a_reg[6] );
			_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 1 ) ) + jc + 40,
								a_reg[7] );

			LOAD_AND_CONVERT_LAST32_NR64_BF16_AVX2(kr)
			SHUFFLE_F32_ELEMENTS_NR32_AVX2
			PERMUTE_NR32_ELEMENTS_AVX2

			_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 0 ) ) + jc + 16,
								a_reg[0] );
			_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 0 ) ) + jc + 24,
								a_reg[1] );
			_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 0 ) ) + jc + 48,
								a_reg[4] );
			_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 0 ) ) + jc + 56,
								a_reg[5] );

			_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 1 ) ) + jc + 16,
								a_reg[2] );
			_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 1 ) ) + jc + 24,
								a_reg[3] );
			_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 1 ) ) + jc + 48,
								a_reg[6] );
			_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( kr + 1 ) ) + jc + 56,
								a_reg[7] );
		}
		if( k_partial_pieces > 0 )
		{
			//Load 2-rows with 64 elements each.
			LOAD_AND_CONVERT_FIRST32_NR64_BF16_NR32_AVX2(k_full_pieces)
			SHUFFLE_F32_ELEMENTS_NR32_AVX2
			PERMUTE_NR32_ELEMENTS_AVX2

			_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( k_full_pieces + 0 ) )
								+ jc + 0, a_reg[0] );
			_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( k_full_pieces + 0 ) )
								+ jc + 8, a_reg[1] );
			_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( k_full_pieces + 0 ) )
								+ jc + 32, a_reg[4] );
			_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( k_full_pieces + 0 ) )
								+ jc + 40, a_reg[5] );

			LOAD_AND_CONVERT_LAST32_NR64_BF16_AVX2(k_full_pieces)
			SHUFFLE_F32_ELEMENTS_NR32_AVX2
			PERMUTE_NR32_ELEMENTS_AVX2

			_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( k_full_pieces + 0 ) )
								+ jc + 16, a_reg[0] );
			_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( k_full_pieces + 0 ) )
								+ jc + 24, a_reg[1] );
			_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( k_full_pieces + 0 ) )
								+ jc + 48, a_reg[4] );
			_mm256_storeu_ps( unpack_b_buffer + ( ldb_unpack * ( k_full_pieces + 0 ) )
								+ jc + 56, a_reg[5] );
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
			unpackb_nr48_bf16_f32_row_major
			    (
			     ( b + ( n_full_pieces_loop_limit * KC_updated ) ),
			     ( unpack_b_buffer + n_full_pieces_loop_limit ), KC, ldb_unpack
			    );

			n0_partial_unpack = 48;
		}
		else if ( n0_32 == 1 )
		{
			unpackb_nr32_bf16_f32_row_major
			    (
			     ( b + ( n_full_pieces_loop_limit * KC_updated ) ),
			     ( unpack_b_buffer + n_full_pieces_loop_limit ), KC, ldb_unpack
			    );

			n0_partial_unpack = 32;
		}
		else if ( n0_16 == 1 )
		{
			unpackb_nr16_bf16_f32_row_major
			    (
			     ( b + ( n_full_pieces_loop_limit * KC_updated ) ),
			     ( unpack_b_buffer + n_full_pieces_loop_limit ), KC, ldb_unpack
			    );

			n0_partial_unpack = 16;
		}

		if ( n0_partial_rem > 0 )
		{
			unpackb_nrlt16_bf16_f32_row_major
			    (
			     ( b + ( n_full_pieces_loop_limit * KC_updated ) +
				 ( n0_partial_unpack * KC_updated ) ),
				 ( unpack_b_buffer + n_full_pieces_loop_limit + n0_partial_unpack ),
				 KC, ldb_unpack, n0_partial_rem
			    );
		}
	}

}

void unpackb_nr64_bf16_f32
	(
	  const bfloat16* b,
	  float*       unpack_b_buffer,
	  const dim_t	  KC,
	  const dim_t     NC,
	  dim_t           rs_b,
	  dim_t           cs_b
	)
{
	if( cs_b == 1 )
	{
		unpackb_nr64_bf16_f32_row_major( b, unpack_b_buffer, NC, KC, rs_b );
	}
}
#endif