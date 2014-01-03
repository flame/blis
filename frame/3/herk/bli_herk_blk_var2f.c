/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas

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

void bli_herk_blk_var2f( obj_t*  a,
                         obj_t*  ah,
                         obj_t*  c,
                         herk_t* cntl )
{
	obj_t a_pack, aS_pack;
	obj_t ah1, ah1_pack;
	obj_t c1;
	obj_t c1S, c1S_pack;

	dim_t i;
	dim_t b_alg;
	dim_t n_trans;
	subpart_t stored_part;

	// Initialize all pack objects that are passed into packm_init().
	bli_obj_init_pack( &a_pack );
	bli_obj_init_pack( &ah1_pack );
	bli_obj_init_pack( &c1S_pack );

	// The upper and lower variants are identical, except for which
	// merged subpartition is acquired in the loop body.
	if ( bli_obj_is_lower( *c ) ) stored_part = BLIS_SUBPART1B;
	else                          stored_part = BLIS_SUBPART1T;

	// Query dimension in partitioning direction.
	n_trans = bli_obj_width_after_trans( *c );

	// Scale C by beta (if instructed).
	bli_scalm_int( &BLIS_ONE,
	               c,
	               cntl_sub_scalm( cntl ) );

	// Initialize object for packing A.
	bli_packm_init( a, &a_pack,
	                cntl_sub_packm_a( cntl ) );

	// Pack A (if instructed).
	bli_packm_int( a, &a_pack,
	               cntl_sub_packm_a( cntl ) );

	// Partition along the n dimension.
	for ( i = 0; i < n_trans; i += b_alg )
	{
		// Determine the current algorithmic blocksize.
		b_alg = bli_determine_blocksize_f( i, n_trans, a,
		                                   cntl_blocksize( cntl ) );

		// Acquire partitions for A1' and C1.
		bli_acquire_mpart_l2r( BLIS_SUBPART1,
		                       i, b_alg, ah, &ah1 );
		bli_acquire_mpart_l2r( BLIS_SUBPART1,
		                       i, b_alg, c, &c1 );

		// Partition off the stored region of C1 and the corresponding region
		// of A_pack.
		bli_acquire_mpart_t2b( stored_part,
		                       i, b_alg, &c1, &c1S );
		bli_acquire_mpart_t2b( stored_part,
		                       i, b_alg, &a_pack, &aS_pack );

		// Initialize objects for packing A1' and C1.
		bli_packm_init( &ah1, &ah1_pack,
		                cntl_sub_packm_b( cntl ) );
		bli_packm_init( &c1S, &c1S_pack,
		                cntl_sub_packm_c( cntl ) );

		// Pack A1' (if instructed).
		bli_packm_int( &ah1, &ah1_pack,
		               cntl_sub_packm_b( cntl ) );

		// Pack C1 (if instructed).
		bli_packm_int( &c1S, &c1S_pack,
		               cntl_sub_packm_c( cntl ) );

		// Perform herk subproblem.
		bli_herk_int( &BLIS_ONE,
		              &aS_pack,
		              &ah1_pack,
		              &BLIS_ONE,
		              &c1S_pack,
		              cntl_sub_herk( cntl ) );

		// Unpack C1 (if C1 was packed).
		bli_unpackm_int( &c1S_pack, &c1S,
		                 cntl_sub_unpackm_c( cntl ) );
	}

	// If any packing buffers were acquired within packm, release them back
	// to the memory manager.
	bli_obj_release_pack( &a_pack );
	bli_obj_release_pack( &ah1_pack );
	bli_obj_release_pack( &c1S_pack );
}

