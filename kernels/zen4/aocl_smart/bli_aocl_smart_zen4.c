/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#include "blis.h"

/* This function determines if we need to take SUP or native path
   for given matrix sizes for zen4 configuration.
   * Returns TRUE if the dimensions fall under SUP range
   * Returns FALSE if the dimensions fall under Native range
*/
bool bli_cntx_gemmsup_thresh_is_met_zen4( obj_t* a, obj_t* b, obj_t* c, cntx_t* cntx )
{
	num_t       dt          =   bli_obj_dt( c );

	if( dt == BLIS_DOUBLE )
	{
		dim_t k           =   bli_obj_width_after_trans( a );
		dim_t m, n;

		const stor3_t stor_id = bli_obj_stor3_from_strides( c, a, b );

		if ( bli_cntx_l3_sup_ker_dislikes_storage_of( c, stor_id, cntx ) )
		{
			m = bli_obj_width(c);
			n = bli_obj_length(c);
		}
		else
		{
			m = bli_obj_length( c );
			n = bli_obj_width( c );
		}
		// For skinny sizes where one/two dimensions are small
		if((m < 1000) || (n < 1000) || (k < 116)) return TRUE;
		// For all combinations in small sizes
		if((m < 2200) && (n < 2200) && (k < 2200)) return TRUE;
		return FALSE;
	}
	else if( dt == BLIS_DCOMPLEX )
	{
		dim_t k           =   bli_obj_width_after_trans( a );
		dim_t m, n;

		const stor3_t stor_id = bli_obj_stor3_from_strides( c, a, b );

		if ( bli_cntx_l3_sup_ker_dislikes_storage_of( c, stor_id, cntx ) )
		{
			m = bli_obj_width(c);
			n = bli_obj_length(c);
		}
		else
		{
			m = bli_obj_length( c );
			n = bli_obj_width( c );
		}
		// For skinny sizes where m and/or n is small
        // The threshold for m is a single value, but for n, it is
        // also based on the packing size of A, since the kernels are
        // column preferential
        if( ( ( m <= 1380 ) || ( n <= 1520 ) || ( k <= 128 ) ) ) return TRUE;

        // For all combinations in small sizes
        if( ( m <= 512 ) && ( n <= 512 ) && ( k <= 512 ) ) return TRUE;
        return FALSE;
	}
	else if( dt == BLIS_SCOMPLEX )
	{
		dim_t k           =   bli_obj_width_after_trans( a );
		dim_t m, n;

		const stor3_t stor_id = bli_obj_stor3_from_strides( c, a, b );

		if ( bli_cntx_l3_sup_ker_dislikes_storage_of( c, stor_id, cntx ) )
		{
			m = bli_obj_width(c);
			n = bli_obj_length(c);
		}
		else
		{
			m = bli_obj_length( c );
			n = bli_obj_width( c );
		}

		// The threshold conditionals are as follows:
		if( n <= 540 )
		{
			if( n <= 420 ) return TRUE;
			else if( m <= 1260 ) return TRUE;
		}
		else
		{
			if( m <= 420 )
			{
				if( m <= 180 ) return TRUE;
				else if( n <= 2100 ) return TRUE;
			}
			else
			{
				if( k <= 540 )
				{
					if( n <= 1260 ) return TRUE;
					else if( m <= 900 ) return TRUE;
				}
			}
		}
		return FALSE;
	}
	else
		return bli_cntx_l3_sup_thresh_is_met( a, b, c, cntx );
}

/* This function determines the ideal blocksizes for given datatype
   and num_threads.
*/
void bli_dynamic_blkszs_zen4( dim_t n_threads, cntx_t* cntx, num_t dt )
{
	// dynamic blocksizes enabled only for double datatype.
	if (dt != BLIS_DOUBLE) return;

	blksz_t blkszs[ BLIS_NUM_BLKSZS ];
	dim_t mc, kc, nc;
	model_t model = bli_init_model_query_id();

	// determine ideal blocksize
	if ( model == BLIS_MODEL_GENOA)
	{
		if (n_threads < 4 )
		{
			mc = 80, kc = 512, nc = 4032;
		}
		else
		{
			mc = 144, kc = 512, nc = 2016;
		}
	}
	else // BLIS_MODEL_BERGAMO
	{
		if (n_threads < 4 )
		{
			mc = 80, kc = 512, nc = 4032;
		}
		else
		{
			mc = 144, kc = 512, nc = 2016;
		}
	}
	
	// set blocksizes
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],   512,  mc,   144,    60 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],   480,  kc,   256,   512 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],  6144,  nc,  4080,  2004 );

	bli_cntx_set_blkszs
	(
		BLIS_NAT, 3,
		BLIS_NC, &blkszs[ BLIS_NC ], BLIS_NR,
		BLIS_KC, &blkszs[ BLIS_KC ], BLIS_KR,
		BLIS_MC, &blkszs[ BLIS_MC ], BLIS_MR,
		cntx
	);
}
