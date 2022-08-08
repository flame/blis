/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.

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
#include "lpgemm_packb_s16.h"
#include "lpgemm_config.h"

void get_packb_nr32_u8s8s16o16_strides
	(
		dim_t *rs_b,
		dim_t *cs_b
	)
{
	*rs_b = lpgemm_get_block_size_NR_global_cntx( U8S8S16OS16 ) * 2;
	*cs_b = lpgemm_get_block_size_NR_global_cntx( U8S8S16OS16 );
}

void packb_nrlt16_u8s8s16o16
	(
		int8_t *pack_b_buffer_u8s8s16o16,
		const int8_t *b,
		const dim_t ldb,
		const dim_t rows,
		dim_t n0_partial_rem
	)
{
	dim_t k_full_pieces_blks = rows / 2;
	dim_t k_full_pieces = k_full_pieces_blks * 2;
	dim_t k_partial_pieces = rows % 2;
	dim_t NR = 16;
	dim_t kr_new = 0;

	int8_t buf0[16], buf1[16];

	__m128i b_vec[2], inter_vec[2];

	for (dim_t kr = 0; kr < k_full_pieces; kr += 2)
	{
		memcpy(buf0, (b + (ldb * (kr + 0))), (n0_partial_rem * sizeof(int8_t)));
		memcpy(buf1, (b + (ldb * (kr + 1))), (n0_partial_rem * sizeof(int8_t)));

		// Read b[0,0], b[0,1], b[0,2]......., b[0,15]
		b_vec[0] = _mm_loadu_si128((__m128i *)buf0);
		// Read b[1,0], b[1,1], b[1,2]......., b[1,15]
		b_vec[1] = _mm_loadu_si128((__m128i *)buf1);

		// Reorder B matrix inputs to suit vpmaddubsw instructions
		inter_vec[0] = _mm_unpacklo_epi8(b_vec[0], b_vec[1]);
		inter_vec[1] = _mm_unpackhi_epi8(b_vec[0], b_vec[1]);

		// Store b[0,0], b[1,0], b[0,1]......., b[0,7], b[1,7]
		_mm_storeu_si128((__m128i *)(pack_b_buffer_u8s8s16o16 + (kr_new * NR)), inter_vec[0]);
		// Store b[0,8], b[1,8], b[0,9]......., b[0,15], b[1,15]
		_mm_storeu_si128((__m128i *)(pack_b_buffer_u8s8s16o16 + ((kr_new + 1) * NR)), inter_vec[1]);

		// Increment to ignore the padded bits
		kr_new += 2;
	}

	// Handle k partial cases
	if (k_partial_pieces > 0)
	{
		memcpy(buf0, (b + (ldb * (k_full_pieces + 0))), (n0_partial_rem * sizeof(int8_t)));

		// Read b[0,0], b[0,1], b[0,2]......., b[0,15]
		b_vec[0] = _mm_loadu_si128((__m128i *)buf0);
		b_vec[1] = _mm_setzero_si128(); // Initialize with zero for padding

		// Reorder B matrix inputs to suit vpmaddubsw instructions
		inter_vec[0] = _mm_unpacklo_epi8(b_vec[0], b_vec[1]);
		inter_vec[1] = _mm_unpackhi_epi8(b_vec[0], b_vec[1]);

		// Store b[0,0], 0, b[0,1]......., b[0,7], 0
		_mm_storeu_si128((__m128i *)(pack_b_buffer_u8s8s16o16 + ((kr_new + 0) * NR)), inter_vec[0]);

		// Store b[0,8], 0, b[0,9]......., b[0,15], 0
		_mm_storeu_si128((__m128i *)(pack_b_buffer_u8s8s16o16 + ((kr_new + 1) * NR)), inter_vec[1]);
	}
}

void packb_nr16_u8s8s16o16(
	int8_t *pack_b_buffer_u8s8s16o16,
	const int8_t *b,
	const dim_t ldb,
	const dim_t rows)
{
	dim_t k_full_pieces_blks = rows / 2;
	dim_t k_full_pieces = k_full_pieces_blks * 2;
	dim_t k_partial_pieces = rows % 2;
	dim_t NR = 16;
	dim_t kr_new = 0;

	__m128i b_vec[2], inter_vec[2];

	for (dim_t kr = 0; kr < k_full_pieces; kr += 2)
	{
		// Read b[0,0], b[0,1], b[0,2]......., b[0,15]
		b_vec[0] = _mm_loadu_si128((__m128i const *)(b + (ldb * (kr + 0))));

		// Read b[1,0], b[1,1], b[1,2]......., b[1,15]
		b_vec[1] = _mm_loadu_si128((__m128i const *)(b + (ldb * (kr + 1))));

		// Reorder B matrix inputs to suit vpmaddubsw instructions
		inter_vec[0] = _mm_unpacklo_epi8(b_vec[0], b_vec[1]);
		inter_vec[1] = _mm_unpackhi_epi8(b_vec[0], b_vec[1]);

		// Store b[0,0], b[1,0], b[0,1]......., b[0,7], b[1,7]
		_mm_storeu_si128((__m128i *)(pack_b_buffer_u8s8s16o16 + ((kr_new + 0) * NR)), inter_vec[0]);

		// Store b[0,8], b[1,8], b[0,9]......., b[0,15], b[1,15]
		_mm_storeu_si128((__m128i *)(pack_b_buffer_u8s8s16o16 + ((kr_new + 1) * NR)), inter_vec[1]);

		// Increment to ignore the padded bits
		kr_new += 2;
	}

	if (k_partial_pieces > 0)
	{
		// Read b[0,0], b[0,1], b[0,2]......., b[0,15]
		b_vec[0] = _mm_loadu_si128((__m128i const *)(b + (ldb * (k_full_pieces + 0))));
		b_vec[1] = _mm_setzero_si128(); // Initialize with zero for padding

		// Reorder B matrix inputs to suit vpmaddubsw instructions
		inter_vec[0] = _mm_unpacklo_epi8(b_vec[0], b_vec[1]);
		inter_vec[1] = _mm_unpackhi_epi8(b_vec[0], b_vec[1]);

		// Store b[0,0], 0, b[0,1]......., b[0,7], 0
		_mm_storeu_si128((__m128i *)(pack_b_buffer_u8s8s16o16 + ((kr_new + 0) * NR)), inter_vec[0]);
		// Store b[0,8], 0, b[0,9]......., b[0,15], 0
		_mm_storeu_si128((__m128i *)(pack_b_buffer_u8s8s16o16 + ((kr_new + 1) * NR)), inter_vec[1]);
	}
}

void packb_nr32_u8s8s16o16(
	int8_t *pack_b_buffer_u8s8s16o16,
	const int8_t *b,
	const dim_t ldb,
	const dim_t cols,
	const dim_t rows,
	dim_t *rs_b,
	dim_t *cs_b)
{
	dim_t NR = 32;

	dim_t n_full_pieces = cols / NR;
	dim_t n_full_pieces_loop_limit = n_full_pieces * NR;
	dim_t n_partial_pieces = cols % NR;
	dim_t k_full_pieces_blks = rows / 2;
	dim_t k_full_pieces = k_full_pieces_blks * 2;
	dim_t k_partial_pieces = rows % 2;

	dim_t KC_updated = rows;

	// Making multiple of 2 to suit k in vpmaddubsw
	KC_updated += (KC_updated & 0x1);

	__m256i b_vec[2], inter_vec[2];

	for (dim_t jc = 0; jc < n_full_pieces_loop_limit; jc += NR)
	{
		for (dim_t kr = 0; kr < k_full_pieces; kr += 2)
		{
			// Read b[0,0], b[0,1], b[0,2]......., b[0,31]
			b_vec[0] = _mm256_loadu_si256((__m256i const *)(b + (ldb * (kr + 0)) + jc));

			//  Read b[1,0], b[1,1], b[1,2]......., b[1,31]
			b_vec[1] = _mm256_loadu_si256((__m256i const *)(b + (ldb * (kr + 1)) + jc));

			//  Reorder B matrix inputs to suit vpmaddubsw instructions
			inter_vec[0] = _mm256_unpacklo_epi8(b_vec[0], b_vec[1]);
			inter_vec[1] = _mm256_unpackhi_epi8(b_vec[0], b_vec[1]);

			b_vec[0] = _mm256_permute2f128_si256(inter_vec[0], inter_vec[1], 0x20);
			b_vec[1] = _mm256_permute2f128_si256(inter_vec[0], inter_vec[1], 0x31);

			// Store B[0,0], B[1,0], B[0,1], B[1,1], ......, B[0,15], B[1,15]
			_mm256_storeu_si256((__m256i *)(pack_b_buffer_u8s8s16o16 + ((jc * KC_updated) + (kr * NR))), b_vec[0]);
			// Store B[0,16], B[1,16], B[0,17], B[1,17], ......, B[0,31], B[1,31]
			_mm256_storeu_si256((__m256i *)(pack_b_buffer_u8s8s16o16 + ((jc * KC_updated) + ((kr + 1) * NR))), b_vec[1]);
		}

		if (k_partial_pieces > 0)
		{
			// Read b[0,0], b[0,1], b[0,2]......., b[0,31]
			b_vec[0] = _mm256_loadu_si256((__m256i const *)(b + (ldb * (k_full_pieces + 0)) + jc));
			b_vec[1] = _mm256_setzero_si256(); // Initialize with zero for padding

			// Reorder B matrix inputs to suit vpmaddubsw instructions
			inter_vec[0] = _mm256_unpacklo_epi8(b_vec[0], b_vec[1]);
			inter_vec[1] = _mm256_unpackhi_epi8(b_vec[0], b_vec[1]);

			b_vec[0] = _mm256_permute2f128_si256(inter_vec[0], inter_vec[1], 0x20);
			b_vec[1] = _mm256_permute2f128_si256(inter_vec[0], inter_vec[1], 0x31);

			// Store B[0,0], B[1,0], B[0,1], B[1,1], ......, B[0,15], B[1,15]
			_mm256_storeu_si256((__m256i *)(pack_b_buffer_u8s8s16o16 + ((jc * KC_updated) + (k_full_pieces * NR))), b_vec[0]);
			// Store B[0,16], B[1,16], B[0,17], B[1,17], ......, B[0,31], B[1,31]
			_mm256_storeu_si256((__m256i *)(pack_b_buffer_u8s8s16o16 + ((jc * KC_updated) + ((k_full_pieces + 1) * NR))), b_vec[1]);
		}
	}

	// B matrix packing when n < NR
	if (n_partial_pieces > 0)
	{
		// Split into multiple smaller fringe kernels, so as to maximize
		// vectorization after packing. Any n0 < NR(32) can be expressed
		// as n0 = 16 + n`.
		dim_t n0_16 = n_partial_pieces / 16;
		dim_t n0_partial_rem = n_partial_pieces % 16;

		dim_t n0_partial_pack = 0;

		if (n0_16 == 1)
		{
			packb_nr16_u8s8s16o16(
				(pack_b_buffer_u8s8s16o16 +
				 (n_full_pieces_loop_limit * KC_updated)),
				(b + n_full_pieces_loop_limit), ldb, rows);

			n0_partial_pack = 16;
		}

		if (n0_partial_rem > 0)
		{
			packb_nrlt16_u8s8s16o16(
				(pack_b_buffer_u8s8s16o16 + (n_full_pieces_loop_limit * KC_updated) +
				 (n0_partial_pack * KC_updated)),
				(b + n_full_pieces_loop_limit + n0_partial_pack),
				ldb, rows, n0_partial_rem);
		}
	}

	*rs_b = NR * 2;
	*cs_b = NR;
}