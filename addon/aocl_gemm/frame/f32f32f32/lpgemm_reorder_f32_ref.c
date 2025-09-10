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
	Below are the reference packb functions which are
    varied based on block size NR (64, 48, 32, 16, lt) and
    order (row / column (transpose)).
*/

static void   packb_f32f32f32of32_row_major_ref
(
	float*       	pack_b,
	float*       	b,
	const dim_t     ldb,
	const dim_t     NC,
	const dim_t     KC,
	const dim_t     NR,
	dim_t*          rs_b,
	dim_t*          cs_b
)
{
	dim_t n_full_pieces = NC / NR;
	dim_t n_full_pieces_loop_limit = n_full_pieces * NR;
	dim_t n_partial_pieces = NC % NR;
	for ( dim_t jc = 0; jc < n_full_pieces_loop_limit; jc += NR )
	{
		for ( dim_t kr = 0; kr < KC; kr ++ )
		{
			float* inp0 = ( b + ( ldb * kr) + jc  );
			float* outp0 = ( pack_b + ( jc * KC ) + ( kr * NR ));
			for(dim_t i = 0; i < NR; i++) *outp0++ = *inp0++;
		}
	}

	if(n_partial_pieces > 0)
	{
		float* pack_b_rem  = ( pack_b + ( n_full_pieces_loop_limit * KC ) );
		float* b_rem = ( b + n_full_pieces_loop_limit );
		for ( dim_t kr = 0; kr < KC; kr ++ )
		{
			float* inp0 = ( b_rem + ( ldb * kr ) );
			float* outp0 = ( pack_b_rem + ( kr * NR ) );
			for(dim_t i = 0; i < n_partial_pieces; i++)	*outp0++ = *inp0++;
		}
	}

	*rs_b = NR;
	*cs_b = 1;
}

static void  packb_nr_f32f32f32of32_col_major_ref
(
    float*       	pack_b_buffer,
    float*       	b,
	const dim_t     NR,
    const dim_t     ldb,
    const dim_t     KC,
    const dim_t     n0_partial_rem
)
{
	for( dim_t i = 0; i < n0_partial_rem; i++ )
	{
		float* inp  = (b + ( ldb * i ));
		float* outp = pack_b_buffer + i;
		for( dim_t j = 0; j < KC; j++ )
		{
			*(outp + ( j * NR)) = *inp++;
		}
	}
	for( dim_t i = n0_partial_rem; i < NR; i++ )
	{
		float* outp = pack_b_buffer + i;
		for( dim_t j = 0; j < KC; j++ )
		{
			*(outp + ( j * NR)) = 0;
		}
	}
}

static void  packb_f32f32f32of32_col_major_ref
(
    float*      	pack_b_buffer,
    float*    b,
    const dim_t     ldb,
    const dim_t     NC,
    const dim_t     KC,
    const dim_t     NR,
    dim_t*          rs_b,
    dim_t*          cs_b
)
{
	dim_t n_full_pieces = NC / NR;
	dim_t n_full_pieces_loop_limit = n_full_pieces * NR;
	dim_t n_partial_pieces = NC % NR;

	for ( dim_t jc = 0; jc < n_full_pieces_loop_limit; jc += NR )
	{
		packb_nr_f32f32f32of32_col_major_ref
		(
			pack_b_buffer + (jc * KC),
			b + (jc * ldb), NR, ldb, KC, NR
		);
	}

	if(n_partial_pieces > 0)
	{
		packb_nr_f32f32f32of32_col_major_ref
		(
			( pack_b_buffer + ( n_full_pieces_loop_limit * KC ) ),
			( b + n_full_pieces_loop_limit * ldb ), NR, ldb, KC, n_partial_pieces
		);
	}

	*rs_b = NR;
	*cs_b = 1;
}

void  packb_f32f32f32of32_reference
(
    float*       	pack_b,
    float*       	b,
    const dim_t  	rs_b,
    const dim_t  	cs_b,
    const dim_t  	NC,
    const dim_t  	KC,
	const dim_t  	NR,
    dim_t*       	rs_p,
    dim_t*       	cs_p
)
{
	if( cs_b == 1 ) {
		packb_f32f32f32of32_row_major_ref( pack_b, b, rs_b, NC, KC, NR, rs_p, cs_p );
	}else{
		packb_f32f32f32of32_col_major_ref( pack_b, b, cs_b, NC, KC, NR, rs_p, cs_p );
	}
}

#endif
