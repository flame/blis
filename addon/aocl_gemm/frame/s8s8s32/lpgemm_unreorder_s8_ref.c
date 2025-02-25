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

void unpackb_nr48_s8_row_major_ref
	(
		int8_t*    b,
		int8_t*    unpack_b,
		const dim_t  KC,
		dim_t        ldb
	)
{
	dim_t NR = 64;

	dim_t k_full_pieces_blks = KC / 4;
	dim_t k_full_pieces = k_full_pieces_blks * 4;
	dim_t k_partial_pieces = KC % 4;

	dim_t kr_new = 0;

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 4 )
	{
		int8_t* outp0 = ( unpack_b + ( ldb * ( kr + 0 ) ));
		int8_t* outp1 = ( unpack_b + ( ldb * ( kr + 1 ) ));
		int8_t* outp2 = ( unpack_b + ( ldb * ( kr + 2 ) ));
		int8_t* outp3 = ( unpack_b + ( ldb * ( kr + 3 ) ));

		//load from b reordered buffer
		int8_t* inp0 = ( b + ( kr_new + 0 ) * NR );
		int8_t* inp1 = ( b + ( kr_new + 1 ) * NR );
		int8_t* inp2 = ( b + ( kr_new + 2 ) * NR );

		for(dim_t i = 0; i < 16; i++)
		{
			*outp0 = *inp0++;
			*outp1 = *inp0++;
			*outp2 = *inp0++;
			*outp3 = *inp0++;

			*(outp0 + 16) = *inp1++;
			*(outp1 + 16) = *inp1++;
			*(outp2 + 16) = *inp1++;
			*(outp3 + 16) = *inp1++;

			*(outp0 + 32) = *inp2++;
			*(outp1 + 32) = *inp2++;
			*(outp2 + 32) = *inp2++;
			*(outp3 + 32) = *inp2++;

			outp0++;
			outp1++;
			outp2++;
			outp3++;
		}

		kr_new += 3;
	}

	if( k_partial_pieces > 0 )
	{
		int8_t* outp0 = ( unpack_b + ( ldb * ( k_full_pieces + 0 ) ) );
		int8_t* outp1 = ( unpack_b + ( ldb * ( k_full_pieces + 1 ) ) );
		int8_t* outp2 = ( unpack_b + ( ldb * ( k_full_pieces + 2 ) ) );

		//load from b reordered buffer
		int8_t* inp0 = ( b + ( ( kr_new + 0 ) * NR ) );
		int8_t* inp1 = ( b + ( ( kr_new + 1 ) * NR ) );
		int8_t* inp2 = ( b + ( ( kr_new + 2 ) * NR ) );

		for(dim_t i = 0; i < 16; i++)
		{
			*outp0 = *inp0++;
			*(outp0 + 16) = *inp1++;
			*(outp0 + 32) = *inp2++;

			if(k_partial_pieces > 1)
			{
				*outp1 = *inp0;
				*(outp1 + 16) = *inp1;
				*(outp1 + 32) = *inp2;
			}
			inp0++; inp1++; inp2++;

			if(k_partial_pieces > 2)
			{
				*outp2 = *inp0;
				*(outp2 + 16) = *inp1;
				*(outp2 + 32) = *inp2;
			}

			inp0++; inp1++; inp2++;
			inp0++; inp1++; inp2++;

			outp0++;
			outp1++;
			outp2++;
		}
	}
}

void unpackb_nr32_s8_row_major_ref
	(
		int8_t*      b,
		int8_t*      unpack_b,
		const dim_t  KC,
		dim_t        ldb
	)
{
	dim_t NR = 64;

	dim_t k_full_pieces_blks = KC / 4;
	dim_t k_full_pieces = k_full_pieces_blks * 4;
	dim_t k_partial_pieces = KC % 4;

	dim_t kr_new = 0;

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 4 )
	{
		int8_t* outp0 = ( unpack_b + ( ldb * ( kr + 0 ) ));
		int8_t* outp1 = ( unpack_b + ( ldb * ( kr + 1 ) ));
		int8_t* outp2 = ( unpack_b + ( ldb * ( kr + 2 ) ));
		int8_t* outp3 = ( unpack_b + ( ldb * ( kr + 3 ) ));

		//load from b reordered buffer
		int8_t* inp0 = ( b + ( kr_new + 0 ) * NR );
		int8_t* inp1 = ( b + ( kr_new + 1 ) * NR );

		for(dim_t i = 0; i < 16; i++)
		{
			*outp0 = *inp0++;
			*outp1 = *inp0++;
			*outp2 = *inp0++;
			*outp3 = *inp0++;

			*(outp0 + 16) = *inp1++;
			*(outp1 + 16) = *inp1++;
			*(outp2 + 16) = *inp1++;
			*(outp3 + 16) = *inp1++;

			outp0++;
			outp1++;
			outp2++;
			outp3++;
		}

		kr_new += 2;
	}

	if( k_partial_pieces > 0 )
	{
		int8_t* outp0 = ( unpack_b + ( ldb * ( k_full_pieces + 0 ) ) );
		int8_t* outp1 = ( unpack_b + ( ldb * ( k_full_pieces + 1 ) ) );
		int8_t* outp2 = ( unpack_b + ( ldb * ( k_full_pieces + 2 ) ) );

		//load from b reordered buffer
		int8_t* inp0 = ( b + ( ( kr_new + 0 ) * NR ) );
		int8_t* inp1 = ( b + ( ( kr_new + 1 ) * NR ) );

		for(dim_t i = 0; i < 16; i++)
		{
			*outp0 = *inp0++;
			*(outp0 + 16) = *inp1++;

			if(k_partial_pieces > 1)
			{
				*outp1 = *inp0;
				*(outp1 + 16) = *inp1;
			}
			inp0++; inp1++;

			if(k_partial_pieces > 2)
			{
				*outp2 = *inp0;
				*(outp2 + 16) = *inp1;
			}

			inp0++; inp1++;
			inp0++; inp1++;

			outp0++;
			outp1++;
			outp2++;
		}
	}
}

void unpackb_nr16_s8_row_major_ref
	(
		int8_t*      b,
		int8_t*      unpack_b,
		const dim_t  KC,
		dim_t        ldb
	)
{
	dim_t NR = 64;

	dim_t k_full_pieces_blks = KC / 4;
	dim_t k_full_pieces = k_full_pieces_blks * 4;
	dim_t k_partial_pieces = KC % 4;

	dim_t kr_new = 0;

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 4 )
	{
		int8_t* outp0 = ( unpack_b + ( ldb * ( kr + 0 ) ));
		int8_t* outp1 = ( unpack_b + ( ldb * ( kr + 1 ) ));
		int8_t* outp2 = ( unpack_b + ( ldb * ( kr + 2 ) ));
		int8_t* outp3 = ( unpack_b + ( ldb * ( kr + 3 ) ));

		//load from b reordered buffer
		int8_t* inp0 = ( b + ( kr_new + 0 ) * NR );

		for(dim_t i = 0; i < 16; i++)
		{
			*outp0 = *inp0++;
			*outp1 = *inp0++;
			*outp2 = *inp0++;
			*outp3 = *inp0++;

			outp0++;
			outp1++;
			outp2++;
			outp3++;
		}

		kr_new += 1;
	}

	if( k_partial_pieces > 0 )
	{
		int8_t* outp0 = ( unpack_b + ( ldb * ( k_full_pieces + 0 ) ) );
		int8_t* outp1 = ( unpack_b + ( ldb * ( k_full_pieces + 1 ) ) );
		int8_t* outp2 = ( unpack_b + ( ldb * ( k_full_pieces + 2 ) ) );

		//load from b reordered buffer
		int8_t* inp0 = ( b + ( ( kr_new + 0 ) * NR ) );

		for(dim_t i = 0; i < 16; i++)
		{
			*outp0 = *inp0++;

			if(k_partial_pieces > 1)
			{
				*outp1 = *inp0;
			}

			inp0++;

			if(k_partial_pieces > 2)
			{
				*outp2 = *inp0;
			}

			inp0++;
			inp0++;

			outp0++;
			outp1++;
			outp2++;
		}
	}
}

void unpackb_nrlt16_s8_row_major_ref
    (
		int8_t*    b,
		int8_t*    unpack_b,
		const dim_t  KC,
		dim_t        ldb,
		dim_t        n0_partial_rem
    )
{
	dim_t NR = 64;

	dim_t k_full_pieces_blks = KC / 4;
	dim_t k_full_pieces = k_full_pieces_blks * 4;
	dim_t k_partial_pieces = KC % 4;

	dim_t kr_new = 0;

	for ( dim_t kr = 0; kr < k_full_pieces; kr += 4 )
	{
		int8_t* outp0 = ( unpack_b + ( ldb * ( kr + 0 ) ));
		int8_t* outp1 = ( unpack_b + ( ldb * ( kr + 1 ) ));
		int8_t* outp2 = ( unpack_b + ( ldb * ( kr + 2 ) ));
		int8_t* outp3 = ( unpack_b + ( ldb * ( kr + 3 ) ));

		//load from b reordered buffer
		int8_t* inp0 = ( b + ( kr_new + 0 ) * NR );

		for(dim_t i = 0; i < n0_partial_rem; i++)
		{
			*outp0 = *inp0++;
			*outp1 = *inp0++;
			*outp2 = *inp0++;
			*outp3 = *inp0++;

			outp0++;
			outp1++;
			outp2++;
			outp3++;
		}

		kr_new += 1;
	}

	if( k_partial_pieces > 0 )
	{
		int8_t* outp0 = ( unpack_b + ( ldb * ( k_full_pieces + 0 ) ) );
		int8_t* outp1 = ( unpack_b + ( ldb * ( k_full_pieces + 1 ) ) );
		int8_t* outp2 = ( unpack_b + ( ldb * ( k_full_pieces + 2 ) ) );

		//load from b reordered buffer
		int8_t* inp0 = ( b + ( ( kr_new + 0 ) * NR ) );

		for(dim_t i = 0; i < n0_partial_rem; i++)
		{
			*outp0 = *inp0++;

			if(k_partial_pieces > 1)
			{
				*outp1 = *inp0;
			}
			inp0++;

			if(k_partial_pieces > 2)
			{
				*outp2 = *inp0;
			}

			inp0++;
			inp0++;

			outp0++;
			outp1++;
			outp2++;
		}
	}
}

void unpackb_nr64_s8_row_major_ref
	(
		int8_t*     b,
		int8_t*     unpack_b,
		const dim_t   NC,
		const dim_t   KC,
		dim_t         ldb
	)
{
	dim_t NR = 64;

	dim_t n_full_pieces = NC / NR;
	dim_t n_full_pieces_loop_limit = n_full_pieces * NR;
	dim_t n_partial_pieces = NC % NR;

	dim_t k_full_pieces_blks = KC / 4;
	dim_t k_full_pieces = k_full_pieces_blks * 4;
	dim_t k_partial_pieces = KC % 4;

	// KC when not multiple of 2 will have padding to make
	// it multiple of 2 in packed buffer.

	dim_t KC_updated = KC;
	if ( k_partial_pieces > 0 )
	{
		KC_updated += ( 4 - k_partial_pieces );
	}

	for ( dim_t jc = 0; jc < n_full_pieces_loop_limit; jc += NR )
	{
		for ( dim_t kr = 0; kr < k_full_pieces; kr += 4 )
		{
			int8_t* outp0 = ( unpack_b + ( ldb * ( kr + 0 ) ) + jc  );
			int8_t* outp1 = ( unpack_b + ( ldb * ( kr + 1 ) ) + jc  );
			int8_t* outp2 = ( unpack_b + ( ldb * ( kr + 2 ) ) + jc  );
			int8_t* outp3 = ( unpack_b + ( ldb * ( kr + 3 ) ) + jc  );

			//load from b reordered buffer
			int8_t* inp0 = ( b + ( jc * KC_updated ) + ( ( kr + 0 ) * NR ));
			int8_t* inp1 = ( b + ( jc * KC_updated ) + ( ( kr + 1 ) * NR ));
			int8_t* inp2 = ( b + ( jc * KC_updated ) + ( ( kr + 2 ) * NR ));
			int8_t* inp3 = ( b + ( jc * KC_updated ) + ( ( kr + 3 ) * NR ));

			for(dim_t i = 0; i < 16; i++)
			{
				*outp0 = *inp0++;
				*outp1 = *inp0++;
				*outp2 = *inp0++;
				*outp3 = *inp0++;

				*(outp0 + 16) = *inp1++;
				*(outp1 + 16) = *inp1++;
				*(outp2 + 16) = *inp1++;
				*(outp3 + 16) = *inp1++;

				*(outp0 + 32) = *inp2++;
				*(outp1 + 32) = *inp2++;
				*(outp2 + 32) = *inp2++;
				*(outp3 + 32) = *inp2++;

				*(outp0 + 48) = *inp3++;
				*(outp1 + 48) = *inp3++;
				*(outp2 + 48) = *inp3++;
				*(outp3 + 48) = *inp3++;

				outp0++;
				outp1++;
				outp2++;
				outp3++;
			}
		}

		if( k_partial_pieces > 0 )
		{
			int8_t* outp0 = ( unpack_b + ( ldb * ( k_full_pieces + 0 ) ) + jc  );
			int8_t* outp1 = ( unpack_b + ( ldb * ( k_full_pieces + 1 ) ) + jc  );
			int8_t* outp2 = ( unpack_b + ( ldb * ( k_full_pieces + 2 ) ) + jc  );

			//load from b reordered buffer
			int8_t* inp0 = ( b + ( jc * KC_updated ) + ( ( k_full_pieces + 0 ) * NR ));
			int8_t* inp1 = ( b + ( jc * KC_updated ) + ( ( k_full_pieces + 1 ) * NR ));
			int8_t* inp2 = ( b + ( jc * KC_updated ) + ( ( k_full_pieces + 2 ) * NR ));
			int8_t* inp3 = ( b + ( jc * KC_updated ) + ( ( k_full_pieces + 3 ) * NR ));

			for(dim_t i = 0; i < 16; i++)
			{
				*outp0 = *inp0++;
				*(outp0 + 16) = *inp1++;
				*(outp0 + 32) = *inp2++;
				*(outp0 + 48) = *inp3++;

				if(k_partial_pieces > 1)
				{
					*outp1 = *inp0;
					*(outp1 + 16) = *inp1;
					*(outp1 + 32) = *inp2;
					*(outp1 + 48) = *inp3;
				}
				inp0++; inp1++; inp2++; inp3++;

				if(k_partial_pieces > 2)
				{
					*outp2 = *inp0;
					*(outp2 + 16) = *inp1;
					*(outp2 + 32) = *inp2;
					*(outp2 + 48) = *inp3;
				}

				inp0++; inp1++; inp2++; inp3++;
				inp0++; inp1++; inp2++; inp3++;

				outp0++;
				outp1++;
				outp2++;
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
			unpackb_nr48_s8_row_major_ref
			(
				( b + ( n_full_pieces_loop_limit * KC_updated ) ),
				( unpack_b + n_full_pieces_loop_limit ), KC, ldb
			);

			n0_partial_unpack = 48;
		}
		else if ( n0_32 == 1 )
		{
			unpackb_nr32_s8_row_major_ref
			(
				( b + ( n_full_pieces_loop_limit * KC_updated ) ),
				( unpack_b + n_full_pieces_loop_limit ), KC, ldb
			);

			n0_partial_unpack = 32;
		}
		else if ( n0_16 == 1 )
		{
			unpackb_nr16_s8_row_major_ref
			(
				( b + ( n_full_pieces_loop_limit * KC_updated ) ),
				( unpack_b + n_full_pieces_loop_limit ), KC, ldb
			);

			n0_partial_unpack = 16;
		}

		if ( n0_partial_rem > 0 )
		{
			unpackb_nrlt16_s8_row_major_ref
			(
				( b + ( n_full_pieces_loop_limit * KC_updated ) +
				( n0_partial_unpack * KC_updated ) ),
				( unpack_b + n_full_pieces_loop_limit + n0_partial_unpack ), KC, ldb,
				n0_partial_rem
			);
		}
	}

}

void unpackb_nrlt16_s8_col_major_ref
	(
		int8_t* 	b,
		int8_t*     unpack_b,
		const dim_t   KC,
		dim_t         ldb,
		dim_t         n0_partial_rem
	)
{
	dim_t NR = 16;

	dim_t kr = 0;
	for ( kr = 0; ( kr + 63 ) < KC; kr += 64 )
	{
		int8_t *inp, *outp;
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			for( dim_t i = 0; i < 16; i++)
			{
				inp = ( b + ( jr * 4 ) + ( ( kr + i * 4 ) * NR ));
				outp = ( unpack_b + i * 4 + ( ldb * jr ) + kr);
				for( dim_t j = 0; j < n0_partial_rem; j++ )
				{
					*(outp +  j*ldb)  = *inp++;
					*(outp +  j*ldb + 1) = *inp++;
					*(outp +  j*ldb + 2) = *inp++;
					*(outp +  j*ldb + 3) = *inp++;
				}
			}
		}
	}

	for ( ; ( kr + 31 ) < KC; kr += 32 )
	{
		int8_t *inp, *outp;
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			for( dim_t i = 0; i < 8; i++)
			{
				inp = ( b + ( jr * 4 ) + ( ( kr + i * 4 ) * NR ));
				outp = ( unpack_b + i * 4 + ( ldb * jr ) + kr);
				for( dim_t j = 0; j < n0_partial_rem; j++ )
				{
					*(outp +  j*ldb)  = *inp++;
					*(outp +  j*ldb + 1) = *inp++;
					*(outp +  j*ldb + 2) = *inp++;
					*(outp +  j*ldb + 3) = *inp++;
				}
			}
		}
	}

	for ( ; ( kr + 15 ) < KC; kr += 16 )
	{
		int8_t *inp, *outp;
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			for( dim_t i = 0; i < 4; i++)
			{
				inp = ( b + ( jr * 4 ) + ( ( kr + i * 4 ) * NR ));
				outp = ( unpack_b + i * 4 + ( ldb * jr ) + kr);
				for( dim_t j = 0; j < n0_partial_rem; j++ )
				{
					*(outp +  j*ldb)  = *inp++;
					*(outp +  j*ldb + 1) = *inp++;
					*(outp +  j*ldb + 2) = *inp++;
					*(outp +  j*ldb + 3) = *inp++;
				}
			}
		}
	}

	for ( ; ( kr + 7 ) < KC; kr += 8 )
	{
		int8_t *inp, *outp;
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			for( dim_t i = 0; i < 2; i++)
			{
				inp = ( b + ( jr * 4 ) + ( ( kr + i * 4 ) * NR ));
				outp = ( unpack_b + i * 4 + ( ldb * jr ) + kr);
				for( dim_t j = 0; j < n0_partial_rem; j++ )
				{
					*(outp +  j*ldb)  = *inp++;
					*(outp +  j*ldb + 1) = *inp++;
					*(outp +  j*ldb + 2) = *inp++;
					*(outp +  j*ldb + 3) = *inp++;
				}
			}
		}
	}

	for ( ; ( kr + 3 ) < KC; kr += 4 )
	{
		int8_t *inp, *outp;
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			for( dim_t i = 0; i < 1; i++)
			{
				inp = ( b + ( jr * 4 ) + ( ( kr + i * 4 ) * NR ));
				outp = ( unpack_b + i * 4 + ( ldb * jr ) + kr);
				for( dim_t j = 0; j < n0_partial_rem; j++ )
				{
					*(outp +  j*ldb)  = *inp++;
					*(outp +  j*ldb + 1) = *inp++;
					*(outp +  j*ldb + 2) = *inp++;
					*(outp +  j*ldb + 3) = *inp++;
				}
			}
		}
	}

	for ( ;  kr < KC; kr+=3 )
	{
		int8_t *inp, *outp;
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			for( dim_t i = 0; i < 1; i++)
			{
				inp = ( b + ( jr * 4 ) + ( ( kr + i * 4 ) * NR ));
				outp = ( unpack_b + i * (KC & 0x3) + ( ldb * jr ) + kr);
				for( dim_t j = 0; j < n0_partial_rem; j++ )
				{
					*(outp +  j*ldb)  = *inp++;

					if(KC & 0x2) *(outp +  j*ldb + 1) = *inp;
					inp++;

					if(KC & 0x3) *(outp +  j*ldb + 2) = *inp;
					inp++;

					inp++;
				}
			}
		}
	}
}


void unpackb_nr_mult_16_s8_col_major_ref
	(
		int8_t*      b,
		int8_t*      unpack_b,
		const dim_t  NR,
		const dim_t  KC,
		dim_t        ldb
	)
{
	dim_t kr = 0;
	for ( kr = 0; ( kr + 63 ) < KC; kr += 64 )
	{
		int8_t *inp, *outp;
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			for( dim_t i = 0; i < 16; i++)
			{
				inp = ( b + ( jr * 4 ) + ( ( kr + i * 4 ) * NR ));
				outp = ( unpack_b + i * 4 + ( ldb * jr ) + kr);
				for( dim_t j = 0; j < 16; j++ )
				{
					*(outp +  j*ldb)  = *inp++;
					*(outp +  j*ldb + 1) = *inp++;
					*(outp +  j*ldb + 2) = *inp++;
					*(outp +  j*ldb + 3) = *inp++;
				}
			}
		}
	}

	for ( ; ( kr + 31 ) < KC; kr += 32 )
	{
		int8_t *inp, *outp;
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			for( dim_t i = 0; i < 8; i++)
			{
				inp = ( b + ( jr * 4 ) + ( ( kr + i * 4 ) * NR ));
				outp = ( unpack_b + i * 4 + ( ldb * jr ) + kr);
				for( dim_t j = 0; j < 16; j++ )
				{
					*(outp +  j*ldb)  = *inp++;
					*(outp +  j*ldb + 1) = *inp++;
					*(outp +  j*ldb + 2) = *inp++;
					*(outp +  j*ldb + 3) = *inp++;
				}
			}
		}
	}

	for ( ; ( kr + 15 ) < KC; kr += 16 )
	{
		int8_t *inp, *outp;
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			for( dim_t i = 0; i < 4; i++)
			{
				inp = ( b + ( jr * 4 ) + ( ( kr + i * 4 ) * NR ));
				outp = ( unpack_b + i * 4 + ( ldb * jr ) + kr);
				for( dim_t j = 0; j < 16; j++ )
				{
					*(outp +  j*ldb)  = *inp++;
					*(outp +  j*ldb + 1) = *inp++;
					*(outp +  j*ldb + 2) = *inp++;
					*(outp +  j*ldb + 3) = *inp++;
				}
			}
		}
	}

	for ( ; ( kr + 7 ) < KC; kr += 8 )
	{
		int8_t *inp, *outp;
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			for( dim_t i = 0; i < 2; i++)
			{
				inp = ( b + ( jr * 4 ) + ( ( kr + i * 4 ) * NR ));
				outp = ( unpack_b + i * 4 + ( ldb * jr ) + kr);
				for( dim_t j = 0; j < 16; j++ )
				{
					*(outp +  j*ldb)  = *inp++;
					*(outp +  j*ldb + 1) = *inp++;
					*(outp +  j*ldb + 2) = *inp++;
					*(outp +  j*ldb + 3) = *inp++;
				}
			}
		}
	}

	for ( ; ( kr + 3 ) < KC; kr += 4 )
	{
		int8_t *inp, *outp;
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			for( dim_t i = 0; i < 1; i++)
			{
				inp = ( b + ( jr * 4 ) + ( ( kr + i * 4 ) * NR ));
				outp = ( unpack_b + i * 4 + ( ldb * jr ) + kr);
				for( dim_t j = 0; j < 16; j++ )
				{
					*(outp +  j*ldb)  = *inp++;
					*(outp +  j*ldb + 1) = *inp++;
					*(outp +  j*ldb + 2) = *inp++;
					*(outp +  j*ldb + 3) = *inp++;
				}
			}
		}
	}

	for ( ;  kr < KC; kr+=3 )
	{
		int8_t *inp, *outp;
		for( dim_t jr = 0; jr < NR; jr += 16 )
		{
			for( dim_t i = 0; i < 1; i++)
			{
				inp = ( b + ( jr * 4 ) + ( ( kr + i * 4 ) * NR ));
				outp = ( unpack_b + i * (KC & 0x3) + ( ldb * jr ) + kr);
				for( dim_t j = 0; j < 16; j++ )
				{
					*(outp +  j*ldb)  = *inp++;

					if(KC & 0x2) *(outp +  j*ldb + 1) = *inp;
					inp++;

					if(KC & 0x3) *(outp +  j*ldb + 2) = *inp;
					inp++;

					inp++;
				}
			}
		}
	}
}

void unpackb_nr64_s8_col_major_ref
	(
		int8_t*      b,
		int8_t*      unpack_b,
		const dim_t  NC,
		const dim_t  KC,
		dim_t        ldb
	)
{
	dim_t NR = 64;

	dim_t n_full_pieces = NC / NR;
	dim_t n_full_pieces_loop_limit = n_full_pieces * NR;
	dim_t n_partial_pieces = NC % NR;

	dim_t k_partial_pieces = KC % 4;

	dim_t KC_updated = KC;
	if ( k_partial_pieces > 0 )
	{
		KC_updated += ( 4 - k_partial_pieces );
	}

	for ( dim_t jc = 0; jc < n_full_pieces_loop_limit; jc += NR )
	{
		unpackb_nr_mult_16_s8_col_major_ref
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
			unpackb_nr_mult_16_s8_col_major_ref
			(
				( b + ( n_full_pieces_loop_limit * KC_updated ) ),
				( unpack_b + n_full_pieces_loop_limit * ldb ), 48, KC, ldb
			);

			n0_partial_pack = 48;
		}
		else if ( n0_32 == 1 )
		{
			unpackb_nr_mult_16_s8_col_major_ref
			(
				( b + ( n_full_pieces_loop_limit * KC_updated ) ),
				( unpack_b + n_full_pieces_loop_limit * ldb ), 32, KC, ldb
			);

			n0_partial_pack = 32;
		}
		else if ( n0_16 == 1 )
		{
			unpackb_nr_mult_16_s8_col_major_ref
			(
				( b + ( n_full_pieces_loop_limit * KC_updated ) ),
				( unpack_b + n_full_pieces_loop_limit * ldb ), 16, KC, ldb
			);

			n0_partial_pack = 16;
		}

		if ( n0_partial_rem > 0 )
		{
			unpackb_nrlt16_s8_col_major_ref
			(
				( b + ( n_full_pieces_loop_limit * KC_updated ) + 
				  ( n0_partial_pack * KC_updated ) ),
				( unpack_b + ( n_full_pieces_loop_limit + n0_partial_pack ) * ldb ), 
				KC, ldb, n0_partial_rem
			);
		}
	}
};

void unpackb_nr64_s8_reference
	(
		int8_t*      b,
		int8_t*      unpack_b,
		const dim_t  NC,
		const dim_t  KC,
		dim_t        rs_b,
		dim_t        cs_b
	)
{
	if( cs_b == 1 )
	{
		unpackb_nr64_s8_row_major_ref( b, unpack_b, NC, KC, rs_b );
	}
	else
	{
		unpackb_nr64_s8_col_major_ref( b, unpack_b, NC, KC, cs_b );
	}
}

#endif
