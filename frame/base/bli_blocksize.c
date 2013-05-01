/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2013, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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
	b->v[BLIS_BITVAL_FLOAT_TYPE]    = b_s;
	b->v[BLIS_BITVAL_DOUBLE_TYPE]   = b_d;
	b->v[BLIS_BITVAL_SCOMPLEX_TYPE] = b_c;
	b->v[BLIS_BITVAL_DCOMPLEX_TYPE] = b_z;
	b->e[BLIS_BITVAL_FLOAT_TYPE]    = be_s;
	b->e[BLIS_BITVAL_DOUBLE_TYPE]   = be_d;
	b->e[BLIS_BITVAL_SCOMPLEX_TYPE] = be_c;
	b->e[BLIS_BITVAL_DCOMPLEX_TYPE] = be_z;
}


void bli_blksz_obj_free( blksz_t* b )
{
	bli_free( b );
}


dim_t bli_blksz_for_type( num_t    dt,
                          blksz_t* b )
{
	return b->v[ dt ];
}


dim_t bli_blksz_ext_for_type( num_t    dt,
                              blksz_t* b )
{
	return b->e[ dt ];
}


dim_t bli_blksz_for_obj( obj_t*   obj,
                         blksz_t* b )
{
	return bli_blksz_for_type( bli_obj_datatype( *obj ), b );
}


dim_t bli_blksz_ext_for_obj( obj_t*   obj,
                             blksz_t* b )
{
	return bli_blksz_ext_for_type( bli_obj_datatype( *obj ), b );
}


dim_t bli_determine_blocksize_f( dim_t    i,
                                 dim_t    dim,
                                 obj_t*   obj,
                                 blksz_t* b )
{
	num_t dt;
	dim_t b_alg, b_ext, b_now;
	dim_t dim_left_now;

	// We assume that this function is being called from an algorithm that
	// is moving "forward" (ie: top to bottom, left to right, top-left
	// to bottom-right).

	// Extract the execution datatype and use it to query the corresponding
	// blocksize and blocksize extension values rom the blksz_t object.
	dt    = bli_obj_execution_datatype( *obj );
	b_alg = bli_blksz_for_type( dt, b );
	b_ext = bli_blksz_ext_for_type( dt, b );
	
	// Compute how much of the matrix dimension is left, including the
	// chunk that will correspond to the blocksize we are computing now.
	dim_left_now = dim - i;

	// If the dimension currently remaining is less than the blocksize
	// plus its allowed extension, use it instead of the default
	// blocksize b_alg. Otherwise, use b_alg.
	if ( dim_left_now <= b_alg + b_ext )
	{
		b_now = dim_left_now;
	}
	else
	{
		b_now = b_alg;
	}

	return b_now;

#if 0
	num_t dt;
	dim_t b_alg;

	// We assume that this function is being called from an algorithm that
	// is moving "forward" (ie: top to bottom, left to right, top-left
	// to bottom-right).

	// Extract the execution datatype and use it to query the corresponding
	// blocksize value from the blksz_t object.
	dt    = bli_obj_execution_datatype( *obj );
	b_alg = bli_blksz_for_type( dt, b );

	// If we are moving "forward" (ie: top to bottom, left to right, or
	// top-left to bottom-right), then return b_alg, unless dim - 1 is
	// smaller, in which case we return that remaining value.
	b_alg = bli_min( b_alg, dim - i );


	return b_alg;
#endif
}


dim_t bli_determine_blocksize_b( dim_t    i,
                                 dim_t    dim,
                                 obj_t*   obj,
                                 blksz_t* b )
{
	num_t dt;
	dim_t b_alg, b_ext, b_now;
	dim_t dim_at_edge;
	dim_t dim_left_now;

	// We assume that this function is being called from an algorithm that
	// is moving "backward" (ie: bottom to top, right to left, bottom-right
	// to top-left).

	// Extract the execution datatype and use it to query the corresponding
	// blocksize and blocksize extension values rom the blksz_t object.
	dt    = bli_obj_execution_datatype( *obj );
	b_alg = bli_blksz_for_type( dt, b );
	b_ext = bli_blksz_ext_for_type( dt, b );
	
	dim_at_edge = dim % b_alg;

	// If dim is a multiple of b_alg, we can safely return b_alg without
	// going any further.
	if ( dim_at_edge == 0 )
		return b_alg;

	// Compute how much of the matrix dimension is left, including the
	// chunk that will correspond to the blocksize we are computing now.
	dim_left_now = dim - i;

	// If the dimension currently remaining is less than the blocksize
	// plus its allowed extension, use it instead of the default
	// blocksize b_alg. Otherwise, use either the edge case dimension
	// (if this is the first iteration) or use the default blocksize.
	if ( dim_left_now <= b_alg + b_ext )
	{
		b_now = dim_left_now;
	}
	else
	{
		if ( i == 0 ) b_now = dim_at_edge;
		else          b_now = b_alg;
	}

	return b_now;

#if 0
	num_t dt;
	dim_t b_alg;
	dim_t dim_at_edge;

	// We assume that this function is being called from an algorithm that
	// is moving "backward" (ie: bottom to top, right to left, bottom-right
	// to top-left).

	// Extract the execution datatype and use it to query the corresponding
	// blocksize value from the blksz_t object.
	dt    = bli_obj_execution_datatype( *obj );
	b_alg = bli_blksz_for_type( dt, b );

	// If it is the first iteration, AND dim is NOT a multiple of b_alg, then
	// we want to return the edge-case blocksize first to allow the first
	// (and subsequent) iterations' subpartitions to be aligned (provided
	// that the blocksize induces subpartitions and movement along aligned
	// boundaries, which should always be the case).
	// If it is the first iteration but dim IS a multiple of b_alg, then we
	// want to simply return b_alg as it is stored in the blocksize object.

	dim_at_edge = dim % b_alg;

	if ( i == 0 && dim_at_edge != 0 )
		b_alg = dim_at_edge;

	return b_alg;
#endif
}

