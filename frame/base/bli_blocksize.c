/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

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


blksz_t* bli_blksz_obj_create( dim_t b_s, dim_t be_s,
                               dim_t b_d, dim_t be_d,
                               dim_t b_c, dim_t be_c,
                               dim_t b_z, dim_t be_z )
{
	blksz_t* b;

	b = ( blksz_t* ) bli_malloc( sizeof(blksz_t) );	

	bli_blksz_obj_init( b,
	                    b_s, be_s,
	                    b_d, be_d,
	                    b_c, be_c,
	                    b_z, be_z );

	return b;
}

void bli_blksz_obj_init( blksz_t* b,
                         dim_t    b_s, dim_t be_s,
                         dim_t    b_d, dim_t be_d,
                         dim_t    b_c, dim_t be_c,
                         dim_t    b_z, dim_t be_z )
{
	b->v[BLIS_FLOAT]    = b_s;
	b->v[BLIS_DOUBLE]   = b_d;
	b->v[BLIS_SCOMPLEX] = b_c;
	b->v[BLIS_DCOMPLEX] = b_z;
	b->e[BLIS_FLOAT]    = be_s;
	b->e[BLIS_DOUBLE]   = be_d;
	b->e[BLIS_SCOMPLEX] = be_c;
	b->e[BLIS_DCOMPLEX] = be_z;

	// By default, set the blocksize multiple, mr, and nr fields
	// to NULL.
	b->mult = NULL;
	b->mr   = NULL;
	b->nr   = NULL;
}

void bli_blksz_obj_attach_mult_to( blksz_t* br,
                                   blksz_t* bc )
{
	bc->mult = br;
}

void bli_blksz_obj_attach_mr_nr_to( blksz_t* bmr,
                                    blksz_t* bnr,
                                    blksz_t* bc )
{
	bc->mr = bmr;
	bc->nr = bnr;
}

void bli_blksz_obj_free( blksz_t* b )
{
	bli_free( b );
}

// -----------------------------------------------------------------------------

void bli_blksz_set_def( dim_t    val,
                        num_t    dt,
                        blksz_t* b )
{
	b->v[ dt ] = val;
}

void bli_blksz_set_max( dim_t    val,
                        num_t    dt,
                        blksz_t* b )
{
	b->e[ dt ] = val;
}

void bli_blksz_set_def_max( dim_t    def,
                            dim_t    max,
                            num_t    dt,
                            blksz_t* b )
{
	bli_blksz_set_def( def, dt, b );
	bli_blksz_set_max( max, dt, b );
}

// -----------------------------------------------------------------------------

void bli_blksz_reduce_to_mult( blksz_t* b )
{
	num_t dt;

	// If there is no blocksize multiple currently attached, we
	// do nothing.
	if ( bli_blksz_mult( b ) == NULL ) return;

	for ( dt = BLIS_DT_LO; dt <= BLIS_DT_HI; ++dt )
	{
		dim_t b_def  = bli_blksz_get_def( dt, b );
		dim_t b_max  = bli_blksz_get_max( dt, b );
		dim_t b_mult = bli_blksz_get_mult( dt, b );

		// If the blocksize multiple is zero, we skip this datatype.
		if ( b_mult == 0 ) continue;

		// Round default and maximum blocksize values down to nearest
		// multiple of b_mult.
		b_def = ( b_def / b_mult ) * b_mult;
		b_max = ( b_max / b_mult ) * b_mult;

		// Make sure the blocksizes are at least b_mult.
		if ( b_def == 0 ) b_def = b_mult;
		if ( b_max == 0 ) b_max = b_mult;

		// Store the new blocksizes back to the object.
		bli_blksz_set_def_max( b_def, b_max, dt, b );
	}
}

// -----------------------------------------------------------------------------

dim_t bli_blksz_get_def( num_t dt, blksz_t* b )
{
	return b->v[ dt ];
}

dim_t bli_blksz_get_max( num_t dt, blksz_t* b )
{
	return b->e[ dt ];
}

dim_t bli_blksz_get_def_for_obj( obj_t* obj, blksz_t* b )
{
	return bli_blksz_get_def( bli_obj_datatype( *obj ), b );
}

dim_t bli_blksz_get_max_for_obj( obj_t* obj, blksz_t* b )
{
	return bli_blksz_get_max( bli_obj_datatype( *obj ), b );
}

// -----------------------------------------------------------------------------

blksz_t* bli_blksz_mult( blksz_t* b )
{
	return b->mult;
}

dim_t bli_blksz_get_mult( num_t dt, blksz_t* b )
{
	return bli_blksz_get_def( dt, bli_blksz_mult( b ) );
}

dim_t bli_blksz_get_mult_for_obj( obj_t* obj, blksz_t* b )
{
	return bli_blksz_get_mult( bli_obj_datatype( *obj ), b );
}

// -----------------------------------------------------------------------------

blksz_t* bli_blksz_mr( blksz_t* b )
{
	return b->mr;
}

blksz_t* bli_blksz_nr( blksz_t* b )
{
	return b->nr;
}

dim_t bli_blksz_get_mr( num_t dt, blksz_t* b )
{
	return bli_blksz_get_def( dt, bli_blksz_mr( b ) );
}

dim_t bli_blksz_get_nr( num_t dt, blksz_t* b )
{
	return bli_blksz_get_def( dt, bli_blksz_nr( b ) );
}

// -----------------------------------------------------------------------------

dim_t bli_determine_blocksize_f( dim_t    i,
                                 dim_t    dim,
                                 obj_t*   obj,
                                 blksz_t* bsize )
{
	num_t dt;
	dim_t b_alg, b_max;
	dim_t b_use;

	// Extract the execution datatype and use it to query the corresponding
	// blocksize and blocksize maximum values from the blksz_t object.
	dt    = bli_obj_execution_datatype( *obj );
	b_alg = bli_blksz_get_def( dt, bsize );
	b_max = bli_blksz_get_max( dt, bsize );

	b_use = bli_determine_blocksize_f_sub( i, dim, b_alg, b_max );

	return b_use;
}

dim_t bli_determine_blocksize_f_sub( dim_t  i,
                                     dim_t  dim,
                                     dim_t  b_alg,
                                     dim_t  b_max )
{
	dim_t b_now;
	dim_t dim_left_now;

	// We assume that this function is being called from an algorithm that
	// is moving "forward" (ie: top to bottom, left to right, top-left
	// to bottom-right).

	// Compute how much of the matrix dimension is left, including the
	// chunk that will correspond to the blocksize we are computing now.
	dim_left_now = dim - i;

	// If the dimension currently remaining is less than the maximum
	// blocksize, use it instead of the default blocksize b_alg.
	// Otherwise, use b_alg.
	if ( dim_left_now <= b_max )
	{
		b_now = dim_left_now;
	}
	else
	{
		b_now = b_alg;
	}

	return b_now;
}

dim_t bli_determine_blocksize_b( dim_t    i,
                                 dim_t    dim,
                                 obj_t*   obj,
                                 blksz_t* bsize )
{
	num_t dt;
	dim_t b_alg, b_max;
	dim_t b_use;

	// Extract the execution datatype and use it to query the corresponding
	// blocksize and blocksize maximum values from the blksz_t object.
	dt    = bli_obj_execution_datatype( *obj );
	b_alg = bli_blksz_get_def( dt, bsize );
	b_max = bli_blksz_get_max( dt, bsize );

	b_use = bli_determine_blocksize_b_sub( i, dim, b_alg, b_max );

	return b_use;
}

dim_t bli_determine_blocksize_b_sub( dim_t  i,
                                     dim_t  dim,
                                     dim_t  b_alg,
                                     dim_t  b_max )
{
	dim_t b_now;
	dim_t dim_left_now;
	dim_t dim_at_edge;

	// We assume that this function is being called from an algorithm that
	// is moving "backward" (ie: bottom to top, right to left, bottom-right
	// to top-left).

	// Compute how much of the matrix dimension is left, including the
	// chunk that will correspond to the blocksize we are computing now.
	dim_left_now = dim - i;

	dim_at_edge = dim_left_now % b_alg;

	// If dim_left_now is a multiple of b_alg, we can safely return b_alg
	// without going any further.
	if ( dim_at_edge == 0 )
		return b_alg;

	// If the dimension currently remaining is less than the maximum
	// blocksize, use it as the chosen blocksize. If this is not the case,
	// then we know dim_left_now is greater than the maximum blocksize.
	// To determine how much of it we should use for the current blocksize,
	// we inspect dim_at_edge; if it is smaller than (or equal to) b_max -
	// b_alg, then we use b_alg + dim_at_edge. Otherwise, dim_at_edge is
	// greater than b_max - b_alg, in which case we use dim_at_edge.
	if ( dim_left_now <= b_max )
	{
		b_now = dim_left_now;
	}
	else // if ( dim_left_now > b_max )
	{
		if ( dim_at_edge <= b_max - b_alg )
		{
			b_now = b_alg + dim_at_edge;
		}
		else // if ( dim_at_edge > b_max - b_alg )
		{
			b_now = dim_at_edge;
		}
	}

	return b_now;
}

