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

/*
	Below are the reference packb functions which are 
    varied based on block size NR (64, 48, 32, 16, lt) and 
    order (row / column (transpose)).
*/

static void  packb_nr48_bf16bf16f32of32_row_major_ref
(
	bfloat16*       pack_b,
	bfloat16*       b,
	const dim_t     ldb,
	const dim_t     KC
)
{
	dim_t NR1 = 32;
	dim_t NR2 = 16;

	dim_t k_full_pieces_blks = KC / 2;
	dim_t k_full_pieces = k_full_pieces_blks * 2;
	dim_t k_partial_pieces = KC % 2;

	dim_t kr_new = 0;

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 2 )
	{
		bfloat16* inp0 = ( b + ( ldb * ( kr + 0 ) ));
		bfloat16* inp1 = ( b + ( ldb * ( kr + 1 ) ));
		bfloat16* inp2 = ( b + ( ldb * ( kr + 0 ) ) + NR2);
		bfloat16* inp3 = ( b + ( ldb * ( kr + 1 ) ) + NR2);

		bfloat16* outp0 = ( pack_b + ( ( kr_new + 0 ) * NR1 ));
		bfloat16* outp1 = ( pack_b + ( ( kr_new + 1 ) * NR1 ));

		for(dim_t i = 0; i < 16; i++)
		{
			*outp0++ = *inp0++;
			*outp0++ = *inp1++;
			*outp1++ = *inp2++;
			*outp1++ = *inp3++;
		}

		inp0 = ( b + ( ldb * ( kr + 0 ) ) + NR1);
		inp1 = ( b + ( ldb * ( kr + 1 ) ) + NR1);
		inp2 = ( b + ( ldb * ( kr + 0 ) ) + NR1 + 8);
		inp3 = ( b + ( ldb * ( kr + 1 ) ) + NR1 + 8);

		outp0 = ( pack_b + ( ( kr_new + 2 ) * NR1 ));
		outp1 = ( pack_b + ( ( kr_new + 2 ) * NR1 + NR2));

		for(dim_t i = 0; i < 8; i++)
		{
			*outp0++ = *inp0++;
			*outp0++ = *inp1++;
			*outp1++ = *inp2++;
			*outp1++ = *inp3++;
		}
		kr_new += 3;
	}

	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		bfloat16* inp0 = ( b + ( ldb * ( k_full_pieces + 0 ) ));
		bfloat16* inp2 = ( b + ( ldb * ( k_full_pieces + 0 ) ) + NR2);
		bfloat16* outp0 = ( pack_b + ( ( kr_new + 0 ) * NR1 ));
		bfloat16* outp1 = ( pack_b + ( ( kr_new + 1 ) * NR1 ));

		for(dim_t i = 0; i < 16; i++)
		{
			*outp0++ = *inp0++;
			*outp0++ = 0;
			*outp1++ = *inp2++;
			*outp1++ = 0;
		}

		inp0 = ( b + ( ldb * ( k_full_pieces + 0 ) ) + NR1);
		inp2 = ( b + ( ldb * ( k_full_pieces + 0 ) ) + NR1 + 8);

		outp0 = ( pack_b + ( ( kr_new + 2 ) * NR1 ));
		outp1 = ( pack_b + ( ( kr_new + 2 ) * NR1 + NR2));

		for(dim_t i = 0; i < 8; i++)
		{
			*outp0++ = *inp0++;
			*outp0++ = 0;
			*outp1++ = *inp2++;
			*outp1++ = 0;
		}
	}
}

static void  packb_nr32_bf16bf16f32of32_row_major_ref
(
    bfloat16*       pack_b,
    bfloat16*       b,
    const dim_t     ldb,
    const dim_t     KC
 )
{
	dim_t NR = 32;
	dim_t NR2 = 16;

	dim_t k_full_pieces_blks = KC / 2;
	dim_t k_full_pieces = k_full_pieces_blks * 2;
	dim_t k_partial_pieces = KC % 2;

	dim_t kr_new = 0;

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 2 )
	{
		bfloat16* inp0 = ( b + ( ldb * ( kr + 0 ) ));
		bfloat16* inp1 = ( b + ( ldb * ( kr + 1 ) ));
		bfloat16* inp2 = ( b + ( ldb * ( kr + 0 ) ) + NR2);
		bfloat16* inp3 = ( b + ( ldb * ( kr + 1 ) ) + NR2);

		bfloat16* outp0 = ( pack_b + ( ( kr_new + 0 ) * NR ));
		bfloat16* outp1 = ( pack_b + ( ( kr_new + 1 ) * NR ));

		for(dim_t i = 0; i < 16; i++)
		{
			*outp0++ = *inp0++;
			*outp0++ = *inp1++;
			*outp1++ = *inp2++;
			*outp1++ = *inp3++;
		}
		kr_new += 2;
	}

	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		bfloat16* inp0 = ( b + ( ldb * ( k_full_pieces + 0 ) ));
		bfloat16* inp2 = ( b + ( ldb * ( k_full_pieces + 0 ) ) + NR2);
		bfloat16* outp0 = ( pack_b + ( ( kr_new + 0 ) * NR ));
		bfloat16* outp1 = ( pack_b + ( ( kr_new + 1 ) * NR ));

		for(dim_t i = 0; i < 16; i++)
		{
			*outp0++ = *inp0++;
			*outp0++ = 0;
			*outp1++ = *inp2++;
			*outp1++ = 0;
		}
	}
}

static void  packb_nr16_bf16bf16f32of32_row_major_ref
(
    bfloat16*       pack_b,
    bfloat16*       b,
    const dim_t     ldb,
    const dim_t     KC
 )
{
	dim_t NR = 16;
	dim_t NRBY2 = 8;
	dim_t k_full_pieces_blks = KC / 2;
	dim_t k_full_pieces = k_full_pieces_blks * 2;
	dim_t k_partial_pieces = KC % 2;

	dim_t kr_new = 0;

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 2 )
	{
		bfloat16* inp0 = ( b + ( ldb * ( kr + 0 ) ));
		bfloat16* inp1 = ( b + ( ldb * ( kr + 1 ) ));
		bfloat16* inp2 = ( b + ( ldb * ( kr + 0 ) ) + NRBY2);
		bfloat16* inp3 = ( b + ( ldb * ( kr + 1 ) ) + NRBY2);

		bfloat16* outp0 = ( pack_b + ( ( kr_new + 0 ) * NR ));
		bfloat16* outp1 = ( pack_b + ( ( kr_new + 1 ) * NR ));

		for(dim_t i = 0; i < NRBY2; i++)
		{
			*outp0++ = *inp0++;
			*outp0++ = *inp1++;
			*outp1++ = *inp2++;
			*outp1++ = *inp3++;
		}
		kr_new += 2;
	}

	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		bfloat16* inp0 = ( b + ( ldb * ( k_full_pieces + 0 ) ));
		bfloat16* inp2 = ( b + ( ldb * ( k_full_pieces + 0 ) ) + NRBY2);
		bfloat16* outp0 = ( pack_b + ( ( kr_new + 0 ) * NR ));
		bfloat16* outp1 = ( pack_b + ( ( kr_new + 1 ) * NR ));

		for(dim_t i = 0; i < NRBY2; i++)
		{
			*outp0++ = *inp0++;
			*outp0++ = 0;
			*outp1++ = *inp2++;
			*outp1++ = 0;
		}
	}
}

static void  packb_nrlt16_bf16bf16f32of32_row_major_ref
(
    bfloat16*       pack_b,
    bfloat16*       b,
    const dim_t     ldb,
    const dim_t     KC,
    const dim_t     n0_partial_rem
 )
{
	dim_t NR = 16;
	dim_t NRBY2 = 8;
	dim_t k_full_pieces_blks = KC / 2;
	dim_t k_full_pieces = k_full_pieces_blks * 2;
	dim_t k_partial_pieces = KC % 2;

	dim_t kr_new = 0;
	bfloat16 buf0[NR];
	bfloat16 buf1[NR];

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 2 )
	{
		memcpy( buf0, ( b + ( ldb * ( kr + 0 ) ) ), ( n0_partial_rem * sizeof( bfloat16 ) ) );
		memcpy( buf1, ( b + ( ldb * ( kr + 1 ) ) ), ( n0_partial_rem * sizeof( bfloat16 ) ) );
		for ( dim_t i = n0_partial_rem; i < 16; i++ )
		{
			buf0[i] = 0;
			buf1[i] = 0;
		}

		bfloat16* inp0 = buf0;
		bfloat16* inp1 = buf1;
		bfloat16* inp2 = buf0 + NRBY2;
		bfloat16* inp3 = buf1 + NRBY2;

		bfloat16* outp0 = ( pack_b + ( ( kr_new + 0 ) * NR ));
		bfloat16* outp1 = ( pack_b + ( ( kr_new + 1 ) * NR ));

		for(dim_t i = 0; i < NRBY2; i++)
		{
			*outp0++ = *inp0++;
			*outp0++ = *inp1++;
			*outp1++ = *inp2++;
			*outp1++ = *inp3++;
		}
		kr_new += 2;
	}

	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		memcpy( buf0, ( b + ( ldb * ( k_full_pieces + 0 ) ) ), ( n0_partial_rem * sizeof( bfloat16 ) ) );
		for ( dim_t i = n0_partial_rem; i < 16; i++ )
		{
			buf0[i] = 0;
		}

		bfloat16* inp0 = buf0;
		bfloat16* inp2 = buf0 + NRBY2;

		bfloat16* outp0 = ( pack_b + ( ( kr_new + 0 ) * NR ));
		bfloat16* outp1 = ( pack_b + ( ( kr_new + 1 ) * NR ));

		for(dim_t i = 0; i < NRBY2; i++)
		{
			*outp0++ = *inp0++;
			*outp0++ = 0;
			*outp1++ = *inp2++;
			*outp1++ = 0;
		}
	}
}

static void   packb_nr64_bf16bf16f32of32_row_major_ref
(
	bfloat16*       pack_b,
	bfloat16*       b,
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
			bfloat16* inp0 = ( b + ( ldb * ( kr + 0 ) ) + jc  );
			bfloat16* inp1 = ( b + ( ldb * ( kr + 0 ) ) + jc + 32 );
			bfloat16* inp2 = ( b + ( ldb * ( kr + 1 ) ) + jc );
			bfloat16* inp3 = ( b + ( ldb * ( kr + 1 ) ) + jc + 32 );

			//store to pack_b buffer
			bfloat16* outp0 = ( pack_b + ( jc * KC_updated ) + ( ( kr + 0 ) * NR ));
			bfloat16* outp1 = ( pack_b + ( jc * KC_updated ) + ( ( kr + 1 ) * NR ));

			for(dim_t i = 0; i < 32; i++)
			{
				*outp0++ = *inp0++;
				*outp0++ = *inp2++;
				*outp1++ = *inp1++;
				*outp1++ = *inp3++;
			}
		}

		// Handle k remainder.
		if( k_partial_pieces > 0)
		{
			bfloat16* inp0 = ( b + ( ldb * ( k_full_pieces + 0 ) ) + jc  );
			bfloat16* inp1 = ( b + ( ldb * ( k_full_pieces + 0 ) ) + jc + 32 );

			//store to pack_b buffer
			bfloat16* outp0 = ( pack_b + ( jc * KC_updated ) + ( ( k_full_pieces + 0 ) * NR ) );
			bfloat16* outp1 = ( pack_b + ( jc * KC_updated ) + ( ( k_full_pieces + 1 ) * NR ) );
			for(dim_t i = 0; i < 32; i++)
			{
				*outp0++ = *inp0++;
				*outp0++ = 0;
				*outp1++ = *inp1++;
				*outp1++ = 0;
			}
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
			packb_nr48_bf16bf16f32of32_row_major_ref
			(
				( pack_b + ( n_full_pieces_loop_limit * KC_updated ) ),
				( b + n_full_pieces_loop_limit ), ldb, KC
			);

			n0_partial_pack = 48;
		}
		else if ( n0_32 == 1 )
		{
			packb_nr32_bf16bf16f32of32_row_major_ref
			(
				( pack_b + ( n_full_pieces_loop_limit * KC_updated ) ),
				( b + n_full_pieces_loop_limit ), ldb, KC
			);

			n0_partial_pack = 32;
		}
		else if ( n0_16 == 1 )
		{
			packb_nr16_bf16bf16f32of32_row_major_ref
			(
				 ( pack_b + ( n_full_pieces_loop_limit * KC_updated ) ),
				 ( b + n_full_pieces_loop_limit ), ldb, KC
			);

			n0_partial_pack = 16;
		}

		if ( n0_partial_rem > 0 )
		{
			packb_nrlt16_bf16bf16f32of32_row_major_ref
			(
				( pack_b + ( n_full_pieces_loop_limit * KC_updated ) +
				   ( n0_partial_pack * KC_updated ) ),
				 ( b + n_full_pieces_loop_limit + n0_partial_pack ), ldb, KC,
			 	n0_partial_rem
			);
		}
	}
	*rs_b = NR * 2;
	*cs_b = NR / 2;
}

static void  packb_nr_mult_16_bf16bf16f32of32_col_major_ref
(
    bfloat16*       pack_b_buffer,
    bfloat16*       b,
    const dim_t     NR,
    const dim_t     ldb,
    const dim_t     KC
)
{
	// Used for permuting the mm512i elements for use in dpbf16_ps instruction.
	dim_t kr = 0;
	for ( kr = 0; ( kr + 31 ) < KC; kr += 32 )
	{
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			bfloat16 *inp, *outp;
			for( dim_t i = 0; i < 16; i++ )
			{
				inp  = (b + ( ldb * ( jr + i ) ) + kr);
				outp = pack_b_buffer + ( jr * 2 ) + (kr * NR) + i * 2;
				for( dim_t j = 0; j < 16; j++ )
				{
					*(outp + ( j * 2 * NR)) = *inp++;
					*(outp + (( j * 2 * NR) + 1)) = *inp++;
				}
			}
		}
	}

	for ( ; ( kr + 15 ) < KC; kr += 16 )
	{
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			bfloat16 *inp, *outp;
			for( dim_t i = 0; i < 16; i++ )
			{
				inp  = (b + ( ldb * ( jr + i ) ) + kr);
				outp = pack_b_buffer + ( jr * 2 ) + (kr * NR) + i * 2;
				for( dim_t j = 0; j < 8; j++ )
				{
					*(outp + ( j * 2 * NR)) = *inp++;
					*(outp + (( j * 2 * NR) + 1)) = *inp++;
				}
			}
		}
	}

	for( ; ( kr +7 ) < KC; kr += 8 )
	{
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			bfloat16 *inp, *outp;
			for( dim_t i = 0; i < 16; i++ )
			{
				inp  = (b + ( ldb * ( jr + i ) ) + kr);
				outp = pack_b_buffer + ( jr * 2 ) + (kr * NR) + i * 2;
				for( dim_t j = 0; j < 4; j++ )
				{
					*(outp + ( j * 2 * NR)) = *inp++;
					*(outp + (( j * 2 * NR) + 1)) = *inp++;
				}
			}
		}
	}

	for( ; ( kr +3 ) < KC; kr += 4 )
	{
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			bfloat16 *inp, *outp;
			for( dim_t i = 0; i < 16; i++ )
			{
				inp  = (b + ( ldb * ( jr + i ) ) + kr);
				outp = pack_b_buffer + ( jr * 2 ) + (kr * NR) + i * 2;
				for( dim_t j = 0; j < 2; j++ )
				{
					*(outp + ( j * 2 * NR)) = *inp++;
					*(outp + (( j * 2 * NR) + 1)) = *inp++;
				}
			}
		}
	}

	for( ; ( kr +1 ) < KC; kr += 2 )
	{
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			bfloat16 *inp, *outp;
			for( dim_t i = 0; i < 16; i++ )
			{
				inp  = (b + ( ldb * ( jr + i ) ) + kr);
				outp = pack_b_buffer + ( jr * 2 ) + (kr * NR) + i * 2;
				for( dim_t j = 0; j < 1; j++ )
				{
					*(outp + ( j * 2 * NR)) = *inp++;
					*(outp + (( j * 2 * NR) + 1)) = *inp++;
				}
			}
		}
	}

	for( ; kr < KC; kr += 1 )
	{
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			bfloat16 *inp, *outp;
			for( dim_t i = 0; i < 16; i++ )
			{
				inp  = (b + ( ldb * ( jr + i ) ) + kr);
				outp = pack_b_buffer + ( jr * 2 ) + (kr * NR) + i * 2;
				for( dim_t j = 0; j < 1; j++ )
				{
					*(outp + ( j * 2 * NR)) = *inp++;
					*(outp + (( j * 2 * NR) + 1)) = 0;
				}
			}
		}
	}
}

static void  packb_nrlt16_bf16bf16f32of32_col_major_ref
(
    bfloat16*       pack_b_buffer,
    bfloat16*       b,
    const dim_t     ldb,
    const dim_t     KC,
    const dim_t     n0_partial_rem
)
{
	dim_t NR = 16;

	dim_t kr = 0;
	for ( kr = 0; ( kr + 31 ) < KC; kr += 32 )
	{
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			bfloat16 *inp, *outp;
			for( dim_t i = 0; i < n0_partial_rem; i++ )
			{
				inp  = (b + ( ldb * ( jr + i ) ) + kr);
				outp = pack_b_buffer + ( jr * 2 ) + (kr * NR) + i * 2;
				for( dim_t j = 0; j < 16; j++ )
				{
					*(outp + ( j * 2 * NR)) = *inp++;
					*(outp + (( j * 2 * NR) + 1)) = *inp++;
				}
			}
		}
	}

	for ( ; ( kr + 15 ) < KC; kr += 16 )
	{
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			bfloat16 *inp, *outp;
			for( dim_t i = 0; i < n0_partial_rem; i++ )
			{
				inp  = (b + ( ldb * ( jr + i ) ) + kr);
				outp = pack_b_buffer + ( jr * 2 ) + (kr * NR) + i * 2;
				for( dim_t j = 0; j < 8; j++ )
				{
					*(outp + ( j * 2 * NR)) = *inp++;
					*(outp + (( j * 2 * NR) + 1)) = *inp++;
				}
			}
		}
	}

	for( ; ( kr +7 ) < KC; kr += 8 )
	{
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			bfloat16 *inp, *outp;
			for( dim_t i = 0; i < n0_partial_rem; i++ )
			{
				inp  = (b + ( ldb * ( jr + i ) ) + kr);
				outp = pack_b_buffer + ( jr * 2 ) + (kr * NR) + i * 2;
				for( dim_t j = 0; j < 4; j++ )
				{
					*(outp + ( j * 2 * NR)) = *inp++;
					*(outp + (( j * 2 * NR) + 1)) = *inp++;
				}
			}
		}
	}

	for( ; ( kr +3 ) < KC; kr += 4 )
	{
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			bfloat16 *inp, *outp;
			for( dim_t i = 0; i < n0_partial_rem; i++ )
			{
				inp  = (b + ( ldb * ( jr + i ) ) + kr);
				outp = pack_b_buffer + ( jr * 2 ) + (kr * NR) + i * 2;
				for( dim_t j = 0; j < 2; j++ )
				{
					*(outp + ( j * 2 * NR)) = *inp++;
					*(outp + (( j * 2 * NR) + 1)) = *inp++;
				}
			}
		}
	}

	for( ; ( kr +1 ) < KC; kr += 2 )
	{
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			bfloat16 *inp, *outp;
			for( dim_t i = 0; i < n0_partial_rem; i++ )
			{
				inp  = (b + ( ldb * ( jr + i ) ) + kr);
				outp = pack_b_buffer + ( jr * 2 ) + (kr * NR) + i * 2;
				for( dim_t j = 0; j < 1; j++ )
				{
					*(outp + ( j * 2 * NR)) = *inp++;
					*(outp + (( j * 2 * NR) + 1)) = *inp++;
				}
			}
		}
	}

	for( ; kr < KC; kr += 1 )
	{
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			bfloat16 *inp, *outp;
			for( dim_t i = 0; i < n0_partial_rem; i++ )
			{
				inp  = (b + ( ldb * ( jr + i ) ) + kr);
				outp = pack_b_buffer + ( jr * 2 ) + (kr * NR) + i * 2;
				for( dim_t j = 0; j < 1; j++ )
				{
					*(outp + ( j * 2 * NR)) = *inp++;
					*(outp + (( j * 2 * NR) + 1)) = 0;
				}
			}
		}
	}
}

static void  packb_nr64_bf16bf16f32of32_col_major_ref
(
    bfloat16*       pack_b_buffer,
    bfloat16*       b,
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
		packb_nr_mult_16_bf16bf16f32of32_col_major_ref
		( 
			pack_b_buffer + (jc * KC_updated),
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
			packb_nr_mult_16_bf16bf16f32of32_col_major_ref
			(
				( pack_b_buffer + ( n_full_pieces_loop_limit * KC_updated ) ),
				( b + n_full_pieces_loop_limit * ldb ), 48, ldb, KC
			);

			n0_partial_pack = 48;
		}
		else if ( n0_32 == 1 )
		{
			packb_nr_mult_16_bf16bf16f32of32_col_major_ref
			(
				( pack_b_buffer + ( n_full_pieces_loop_limit * KC_updated ) ),
				( b + n_full_pieces_loop_limit * ldb ), 32, ldb, KC
			);

			n0_partial_pack = 32;
		}
		else if ( n0_16 == 1 )
		{
			packb_nr_mult_16_bf16bf16f32of32_col_major_ref
			(
				( pack_b_buffer + ( n_full_pieces_loop_limit * KC_updated ) ),
				( b + n_full_pieces_loop_limit * ldb ), 16, ldb, KC
			);

			n0_partial_pack = 16;
		}

		if ( n0_partial_rem > 0 )
		{
			packb_nrlt16_bf16bf16f32of32_col_major_ref
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

void  packb_nr64_bf16bf16f32of32_reference
(
    bfloat16*       pack_b,
    bfloat16*       b,
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
		packb_nr64_bf16bf16f32of32_row_major_ref( pack_b,
		                                          b, rs_b, NC, KC, rs_p, cs_p );
	}
	else
	{
		packb_nr64_bf16bf16f32of32_col_major_ref( pack_b,
		                                          b, cs_b, NC, KC, rs_p, cs_p );
	}
}

#endif
