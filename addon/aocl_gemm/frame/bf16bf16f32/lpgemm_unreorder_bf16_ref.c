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

/*
	Below are the reference unpackb functions which are 
    varied based on block size NR (64, 48, 32, 16, lt) and 
    order (row / column (transpose)).
*/

void unpackb_nr48_bf16bf16f32of32_row_major_ref
	(
	  bfloat16*    b,
	  bfloat16*    unpack_b,
	  const dim_t  KC,
	  dim_t        ldb
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
		bfloat16* outp0 = ( unpack_b + ( ldb * ( kr + 0 ) ));
		bfloat16* outp1 = ( unpack_b + ( ldb * ( kr + 1 ) ));
		bfloat16* outp2 = ( unpack_b + ( ldb * ( kr + 0 ) ) + NR2);
		bfloat16* outp3 = ( unpack_b + ( ldb * ( kr + 1 ) ) + NR2);

		bfloat16* inp0 = ( b + ( ( kr_new + 0 ) * NR1 ));
		bfloat16* inp1 = ( b + ( ( kr_new + 1 ) * NR1 ));

		for(dim_t i = 0; i < 16; i++)
		{
			*outp0++ = *inp0++;
			*outp1++ = *inp0++;
			*outp2++ = *inp1++;
			*outp3++ = *inp1++;
		}

		outp0 = ( unpack_b + ( ldb * ( kr + 0 ) ) + NR1);
		outp1 = ( unpack_b + ( ldb * ( kr + 1 ) ) + NR1);
		outp2 = ( unpack_b + ( ldb * ( kr + 0 ) ) + NR1 + 8);
		outp3 = ( unpack_b + ( ldb * ( kr + 1 ) ) + NR1 + 8);

		inp0 = ( b + ( ( kr_new + 2 ) * NR1 ));
		inp1 = ( b + ( ( kr_new + 2 ) * NR1 + NR2));

		for(dim_t i = 0; i < 8; i++)
		{
			*outp0++ = *inp0++;
			*outp1++ = *inp0++;
			*outp2++ = *inp1++;
			*outp3++ = *inp1++;
		}
		kr_new += 3;
	}

	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		bfloat16* inp0 = ( b + ( ldb * ( k_full_pieces + 0 ) ));
		bfloat16* inp2 = ( b + ( ldb * ( k_full_pieces + 0 ) ) + NR2);

		bfloat16* outp0 = ( unpack_b + ( ( kr_new + 0 ) * NR1 ));
		bfloat16* outp1 = ( unpack_b + ( ( kr_new + 1 ) * NR1 ));

		for(dim_t i = 0; i < 16; i++)
		{
			*outp0++ = *inp0++;
			*outp0++ = 0;
			*outp1++ = *inp2++;
			*outp1++ = 0;
		}

		inp0 = ( b + ( ldb * ( k_full_pieces + 0 ) ) + NR1);
		inp2 = ( b + ( ldb * ( k_full_pieces + 0 ) ) + NR1 + 8);

		outp0 = ( unpack_b + ( ( kr_new + 2 ) * NR1 ));
		outp1 = ( unpack_b + ( ( kr_new + 2 ) * NR1 + NR2));

		for(dim_t i = 0; i < 8; i++)
		{
			*outp0++ = *inp0++;
			*outp0++ = 0;
			*outp1++ = *inp2++;
			*outp1++ = 0;
		}
	}
}

void unpackb_nr32_bf16bf16f32of32_row_major_ref
	(
	  bfloat16*    b,
	  bfloat16*    unpack_b,
	  const dim_t  KC,
	  dim_t        ldb
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
		bfloat16* outp0 = ( unpack_b + ( ldb * ( kr + 0 ) ));
		bfloat16* outp1 = ( unpack_b + ( ldb * ( kr + 1 ) ));
		bfloat16* outp2 = ( unpack_b + ( ldb * ( kr + 0 ) ) + NR2);
		bfloat16* outp3 = ( unpack_b + ( ldb * ( kr + 1 ) ) + NR2);

		bfloat16* inp0 = ( b + ( ( kr_new + 0 ) * NR ));
		bfloat16* inp1 = ( b + ( ( kr_new + 1 ) * NR ));

		for(dim_t i = 0; i < 16; i++)
		{
			*outp0++ = *inp0++;
			*outp1++ = *inp0++;
			*outp2++ = *inp1++;
			*outp3++ = *inp1++;
		}

		kr_new += 2;
	}

	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		bfloat16* outp0 = ( unpack_b + ( ldb * ( k_full_pieces + 0 ) ));
		bfloat16* outp2 = ( unpack_b + ( ldb * ( k_full_pieces + 0 ) ) + NR2);

		bfloat16* inp0 = ( b + ( ( kr_new + 0 ) * NR ));
		bfloat16* inp1 = ( b + ( ( kr_new + 1 ) * NR ));

		for(dim_t i = 0; i < 16; i++)
		{
			*outp0++ = *inp0++;
			*outp2++ = *inp1++;
		}
	}
}

void unpackb_nr16_bf16bf16f32of32_row_major_ref
	(
	  bfloat16*    b,
	  bfloat16*    unpack_b,
	  const dim_t  KC,
	  dim_t        ldb
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
		bfloat16* outp0 = ( unpack_b + ( ldb * ( kr + 0 ) ));
		bfloat16* outp1 = ( unpack_b + ( ldb * ( kr + 1 ) ));
		bfloat16* outp2 = ( unpack_b + ( ldb * ( kr + 0 ) ) + NRBY2);
		bfloat16* outp3 = ( unpack_b + ( ldb * ( kr + 1 ) ) + NRBY2);

		bfloat16* inp0 = ( b + ( ( kr_new + 0 ) * NR ));
		bfloat16* inp1 = ( b + ( ( kr_new + 1 ) * NR ));

		for(dim_t i = 0; i < NRBY2; i++)
		{
			*outp0++ = *inp0++;
			*outp1++ = *inp0++;
			*outp2++ = *inp1++;
			*outp3++ = *inp1++;
		}
		kr_new += 2;
	}

	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		bfloat16* outp0 = ( unpack_b + ( ldb * ( k_full_pieces + 0 ) ));
		bfloat16* outp2 = ( unpack_b + ( ldb * ( k_full_pieces + 0 ) ) + NRBY2);

		bfloat16* inp0 = ( b + ( ( kr_new + 0 ) * NR ));
		bfloat16* inp1 = ( b + ( ( kr_new + 1 ) * NR ));

		for(dim_t i = 0; i < NRBY2; i++)
		{
			*outp0++ = *inp0++;
			*outp2++ = *inp1++;
		}
	}
}

void unpackb_nrlt16_bf16bf16f32of32_row_major_ref
    (
      bfloat16*    b,
      bfloat16*    unpack_b,
      const dim_t  KC,
      dim_t        ldb,
      dim_t        n0_partial_rem
    )
{
	dim_t NR = 16;

	dim_t k_full_pieces_blks = KC / 2;
	dim_t k_full_pieces = k_full_pieces_blks * 2;
	dim_t k_partial_pieces = KC % 2;

	dim_t kr_new = 0;

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 2 )
	{
		bfloat16* outp0 = ( unpack_b + ( ldb * ( kr + 0 ) ));
		bfloat16* outp1 = ( unpack_b + ( ldb * ( kr + 1 ) ));

		bfloat16* inp0 = ( b + ( ( kr_new + 0 ) * NR ));

		for(dim_t i = 0; i < n0_partial_rem; i++)
		{
			*outp0++ = *inp0++;
			*outp1++ = *inp0++;
		}
		kr_new += 2;
	}

	// Handle k remainder.
	if ( k_partial_pieces > 0 )
	{
		bfloat16* outp0 = ( unpack_b + ( ldb * ( k_full_pieces + 0 ) ));

		bfloat16* inp0 = ( b + ( ( kr_new + 0 ) * NR ));

		for(dim_t i = 0; i < n0_partial_rem; i++)
		{
			*outp0++ = *inp0++;
		}
	}
}

void unpackb_nr64_bf16bf16f32of32_row_major_ref
	(
	  bfloat16*     b,
	  bfloat16*     unpack_b,
	  const dim_t   NC,
	  const dim_t   KC,
	  dim_t         ldb
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
			bfloat16* outp0 = ( unpack_b + ( ldb * ( kr + 0 ) ) + jc  );
			bfloat16* outp1 = ( unpack_b + ( ldb * ( kr + 0 ) ) + jc + 32 );
			bfloat16* outp2 = ( unpack_b + ( ldb * ( kr + 1 ) ) + jc );
			bfloat16* outp3 = ( unpack_b + ( ldb * ( kr + 1 ) ) + jc + 32 );

			//load from b reordered buffer
			bfloat16* inp0 = ( b + ( jc * KC_updated ) + ( ( kr + 0 ) * NR ));
			bfloat16* inp1 = ( b + ( jc * KC_updated ) + ( ( kr + 1 ) * NR ));

			for(dim_t i = 0; i < 32; i++)
			{
				*outp0++ = *inp0++;
				*outp2++ = *inp0++;
				*outp1++ = *inp1++;
				*outp3++ = *inp1++;
			}
		}

		if( k_partial_pieces > 0 )
		{
			bfloat16* outp0 = ( unpack_b + ( ldb * ( k_full_pieces + 0 ) ) + jc  );
			bfloat16* outp1 = ( unpack_b + ( ldb * ( k_full_pieces + 0 ) ) + jc + 32 );

			//load from b reordered buffer
			bfloat16* inp0 = ( b + ( jc * KC_updated ) + ( ( k_full_pieces + 0 ) * NR ) );
			bfloat16* inp1 = ( b + ( jc * KC_updated ) + ( ( k_full_pieces + 1 ) * NR ) );
			for(dim_t i = 0; i < 32; i++)
			{
				*outp0++ = *inp0++;
				*outp0++ = 0;
				*outp1++ = *inp1++;
				*outp1++ = 0;
			}
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
			unpackb_nr48_bf16bf16f32of32_row_major_ref
			(
				( b + ( n_full_pieces_loop_limit * KC_updated ) ),
				( unpack_b + n_full_pieces_loop_limit ), KC, ldb
			);

			n0_partial_unpack = 48;
		}
		else if ( n0_32 == 1 )
		{
			unpackb_nr32_bf16bf16f32of32_row_major_ref
			(
				( b + ( n_full_pieces_loop_limit * KC_updated ) ),
				( unpack_b + n_full_pieces_loop_limit ), KC, ldb
			);

			n0_partial_unpack = 32;
		}
		else if ( n0_16 == 1 )
		{
			unpackb_nr16_bf16bf16f32of32_row_major_ref
			(
				( b + ( n_full_pieces_loop_limit * KC_updated ) ),
				( unpack_b + n_full_pieces_loop_limit ), KC, ldb
			);

			n0_partial_unpack = 16;
		}

		if ( n0_partial_rem > 0 )
		{
			unpackb_nrlt16_bf16bf16f32of32_row_major_ref
			(
				( b + ( n_full_pieces_loop_limit * KC_updated ) +
				( n0_partial_unpack * KC_updated ) ),
				( unpack_b + n_full_pieces_loop_limit + n0_partial_unpack ), KC, ldb,
				n0_partial_rem
			);
		}
	}

}

void unpackb_nrlt16_bf16bf16f32of32_col_major_ref
	(
	  bfloat16* 	b,
	  bfloat16*     unpack_b,
	  const dim_t   KC,
	  dim_t         ldb,
	  dim_t         n0_partial_rem
	)
{
	dim_t kr = 0;
	dim_t NR = 16;

	for ( kr = 0; ( kr + 31 ) < KC; kr += 32 )
	{
		bfloat16 *inp, *outp;
		{
			for( dim_t i = 0; i < 16; i++ )
			{
				inp = ( b +  ( ( kr +  i * 2 ) * NR ) );
				outp = ( unpack_b + kr + i * 2 );
				for( dim_t j = 0; j < n0_partial_rem; j++ )
				{
					*(outp +  j * ldb)  = *inp++;
					*(outp +  j * ldb + 1) = *inp++;
				}
			}
		}
	}

	for ( ; ( kr + 15 ) < KC; kr += 16 )
	{
		bfloat16 *inp, *outp;
		{
			for( dim_t i = 0; i < 8; i++)
			{
				inp = ( b +  ( ( kr + i * 2 ) * NR ));
				outp = ( unpack_b + kr + i * 2);
				for( dim_t j = 0; j < n0_partial_rem; j++ )
				{
					*(outp +  j*ldb)  = *inp++;
					*(outp +  j*ldb + 1) = *inp++;
				}
			}
		}
	}

	for ( ; ( kr + 7 ) < KC; kr += 8 )
	{
		bfloat16 *inp, *outp;
		{
			for( dim_t i = 0; i < 4; i++)
			{
				inp = ( b +  ( ( kr + i * 2 ) * NR ));
				outp = ( unpack_b + kr + i * 2);
				for( dim_t j = 0; j < n0_partial_rem; j++ )
				{
					*(outp +  j*ldb)  = *inp++;
					*(outp +  j*ldb + 1) = *inp++;
				}
			}
		}
	}

	for ( ; ( kr + 3 ) < KC; kr += 4 )
	{
		bfloat16 *inp, *outp;
		{
			for( dim_t i = 0; i < 2; i++)
			{
				inp = ( b +  ( ( kr + i * 2 ) * NR ));
				outp = ( unpack_b + kr + i * 2);
				for( dim_t j = 0; j < n0_partial_rem; j++ )
				{
					*(outp +  j*ldb)  = *inp++;
					*(outp +  j*ldb + 1) = *inp++;
				}
			}
		}
	}

	for ( ; ( kr + 1 ) < KC; kr += 2 )
	{
		bfloat16 *inp, *outp;
		{
			for( dim_t i = 0; i < 1; i++)
			{
				inp = ( b +  ( ( kr + i * 2 ) * NR ));
				outp = ( unpack_b + kr + i * 2);
				for( dim_t j = 0; j < n0_partial_rem; j++ )
				{
					*(outp +  j*ldb)  = *inp++;
					*(outp +  j*ldb + 1) = *inp++;
				}
			}
		}
	}

	for ( ;  kr < KC; kr += 1 )
	{
		bfloat16 *inp, *outp;
		{
			for( dim_t i = 0; i < 1; i++)
			{
				inp = ( b +  ( ( kr + i * 2 ) * NR ));
				outp = ( unpack_b + kr + i * 2);
				for( dim_t j = 0; j < n0_partial_rem; j++ )
				{
					*(outp +  j*ldb)  = *inp++;
					inp++;
				}
			}
		}
	}
}


void unpackb_nr_mult_16_bf16bf16f32of32_col_major_ref
	(
	  bfloat16*    b,
	  bfloat16*    unpack_b,
	  const dim_t  NR,
	  const dim_t  KC,
	  dim_t        ldb
	)
{
	dim_t kr = 0;
	for ( kr = 0; ( kr + 31 ) < KC; kr += 32 )
	{
		bfloat16 *inp, *outp;
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			for( dim_t i = 0; i < 16; i++)
			{
				inp = ( b + ( jr * 2 ) + ( ( kr + i * 2 ) * NR ));
				outp = ( unpack_b + i * 2 + ( ldb * jr ) + kr);
				for( dim_t j = 0; j < 16; j++ )
				{
					*(outp +  j*ldb)  = *inp++;
					*(outp +  j*ldb + 1) = *inp++;
				}
			}
		}
	}

	for ( ; ( kr + 15 ) < KC; kr += 16 )
	{
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			bfloat16 *inp, *outp;
			for( dim_t i = 0; i < 8; i++)
			{
				inp = ( b + ( jr * 2 ) + ( ( kr + i * 2 ) * NR ));
				outp = ( unpack_b + i * 2 + ( ldb * jr ) + kr);
				for( dim_t j = 0; j < 16; j++ )
				{
					*(outp +  j*ldb)  = *inp++;
					*(outp +  j*ldb + 1) = *inp++;
				}
			}
		}
	}

	for( ; ( kr +7 ) < KC; kr += 8 )
	{
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			bfloat16 *inp, *outp;
			for( dim_t i = 0; i < 4; i++)
			{
				inp = ( b + ( jr * 2 ) + ( ( kr + i * 2 ) * NR ));
				outp = ( unpack_b + i * 2 + ( ldb * jr ) + kr);
				for( dim_t j = 0; j < 16; j++ )
				{
					*(outp +  j*ldb)  = *inp++;
					*(outp +  j*ldb + 1) = *inp++;
				}
			}
		}
	}

	for( ; ( kr +3 ) < KC; kr += 4 )
	{
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			bfloat16 *inp, *outp;
			for( dim_t i = 0; i < 2; i++)
			{
				inp = ( b + ( jr * 2 ) + ( ( kr + i * 2 ) * NR ));
				outp = ( unpack_b + i * 2 + ( ldb * jr ) + kr);
				for( dim_t j = 0; j < 16; j++ )
				{
					*(outp +  j*ldb)  = *inp++;
					*(outp +  j*ldb + 1) = *inp++;
				}
			}
		}
	}

	for( ; ( kr +1 ) < KC; kr += 2 )
	{
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			bfloat16 *inp, *outp;
			for( dim_t i = 0; i < 1; i++)
			{
				inp = (b + ( jr * 2 ) + ( ( kr + i * 2 ) * NR ));
				outp = (unpack_b + i * 2 + ( ldb * jr ) + kr);
				for( dim_t j = 0; j < 16; j++ )
				{
					*(outp +  j*ldb)  = *inp++;
					*(outp +  j*ldb + 1) = *inp++;
				}		
			}
		}
	}

	for( ; kr < KC; kr += 1 )
	{
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			bfloat16 *inp, *outp;
			for( dim_t i = 0; i < 1; i++)
			{
				inp = ( b + ( jr * 2 ) + ( ( kr + i * 2 ) * NR ));
				outp = ( unpack_b + i * 2 + ( ldb * jr ) + kr);
				for( dim_t j = 0; j < 16; j++ )
				{
					*(outp +  j * ldb)  = *inp++;
					inp++;
				}
			}
		}
	}
}

void unpackb_nr64_bf16bf16f32of32_col_major_ref
	(
	  bfloat16*    b,
	  bfloat16*    unpack_b,
	  const dim_t  NC,
	  const dim_t  KC,
	  dim_t        ldb
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
		unpackb_nr_mult_16_bf16bf16f32of32_col_major_ref
		( 
			b + (jc * KC_updated),
			unpack_b + (jc * ldb), 64, KC, ldb
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
			unpackb_nr_mult_16_bf16bf16f32of32_col_major_ref
			(
				( b + ( n_full_pieces_loop_limit * KC_updated ) ),
				( unpack_b + n_full_pieces_loop_limit * ldb ), 48, KC, ldb
			);

			n0_partial_pack = 48;
		}
		else if ( n0_32 == 1 )
		{
			unpackb_nr_mult_16_bf16bf16f32of32_col_major_ref
			(
				( b + ( n_full_pieces_loop_limit * KC_updated ) ),
				( unpack_b + n_full_pieces_loop_limit * ldb ), 32, KC, ldb
			);

			n0_partial_pack = 32;
		}
		else if ( n0_16 == 1 )
		{
			unpackb_nr_mult_16_bf16bf16f32of32_col_major_ref
			(
				( b + ( n_full_pieces_loop_limit * KC_updated ) ),
				( unpack_b + n_full_pieces_loop_limit * ldb ), 16, KC, ldb
			);

			n0_partial_pack = 16;
		}

		if ( n0_partial_rem > 0 )
		{
			unpackb_nrlt16_bf16bf16f32of32_col_major_ref
			(
				( b + ( n_full_pieces_loop_limit * KC_updated ) + 
				  ( n0_partial_pack * KC_updated ) ),
				( unpack_b + ( n_full_pieces_loop_limit + n0_partial_pack ) * ldb ), 
				KC, ldb, n0_partial_rem
			);
		}
	}
};

void unpackb_nr64_bf16bf16f32of32_reference
	(
	  bfloat16*    b,
	  bfloat16*    unpack_b,
	  const dim_t  NC,
	  const dim_t  KC,
	  dim_t        rs_b,
	  dim_t        cs_b
	)
{
	if( cs_b == 1 )
	{
		unpackb_nr64_bf16bf16f32of32_row_major_ref( b, unpack_b, NC, KC, rs_b );
	}
	else
	{
		unpackb_nr64_bf16bf16f32of32_col_major_ref( b, unpack_b, NC, KC, cs_b );
	}
}

#endif
