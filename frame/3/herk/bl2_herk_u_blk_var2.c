/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2012, The University of Texas

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

#include "blis2.h"

void bl2_herk_u_blk_var2( obj_t*  alpha,
                          obj_t*  a,
                          obj_t*  ah,
                          obj_t*  beta,
                          obj_t*  c,
                          herk_t* cntl )
{
	obj_t a_pack, aT_pack;
	obj_t ah1, ah1_pack;
	obj_t c1;
	obj_t c1T, c1T_pack;

	dim_t i;
	dim_t b_alg;
	dim_t n_trans;
	dim_t offT, mT;

	// Initialize all pack objects that are passed into packm_init().
	bl2_obj_init_pack( &a_pack );
	bl2_obj_init_pack( &ah1_pack );
	bl2_obj_init_pack( &c1T_pack );

	// Query dimension in partitioning direction.
	n_trans = bl2_obj_width_after_trans( *c );

	// Scale C by beta (if instructed).
	bl2_scalm_int( beta,
	               c,
	               cntl_sub_scalm( cntl ) );

	// Initialize object for packing A.
	bl2_packm_init( a, &a_pack,
	                cntl_sub_packm_a( cntl ) );

	// Pack A and scale by alpha (if instructed).
	bl2_packm_int( alpha,
	               a, &a_pack,
	               cntl_sub_packm_a( cntl ) );

	// Partition along the n dimension.
	for ( i = 0; i < n_trans; i += b_alg )
	{
		// Determine the current algorithmic blocksize.
		b_alg = bl2_determine_blocksize_b( i, n_trans, a,
		                                   cntl_blocksize( cntl ) );

		// Acquire partitions for A1' and C1.
		bl2_acquire_mpart_r2l( BLIS_SUBPART1,
		                       i, b_alg, ah, &ah1 );
		bl2_acquire_mpart_r2l( BLIS_SUBPART1,
		                       i, b_alg, c, &c1 );

		// Partition off the stored region of C1 and the corresponding region
		// of A_pack. We compute the length of the subpartition taking the
		// location of the diagonal into account.
		offT = 0;
		mT   = bl2_min( bl2_obj_length_after_trans( c1 ),
		               -bl2_obj_diag_offset_after_trans( c1 ) + b_alg );
		bl2_acquire_mpart_t2b( BLIS_SUBPART1,
		                       offT, mT, &c1, &c1T );
		bl2_acquire_mpart_t2b( BLIS_SUBPART1,
		                       offT, mT, &a_pack, &aT_pack );

		// Initialize objects for packing A1' and C1.
		bl2_packm_init( &ah1, &ah1_pack,
		                cntl_sub_packm_b( cntl ) );
		bl2_packm_init( &c1T, &c1T_pack,
		                cntl_sub_packm_c( cntl ) );

		// Pack A1' and scale by alpha (if instructed).
		bl2_packm_int( alpha,
		               &ah1, &ah1_pack,
		               cntl_sub_packm_b( cntl ) );

		// Pack C1 and scale by beta (if instructed).
		bl2_packm_int( beta,
		               &c1T, &c1T_pack,
		               cntl_sub_packm_c( cntl ) );

		// Perform herk subproblem.
		bl2_herk_int( alpha,
		              &aT_pack,
		              &ah1_pack,
		              beta,
		              &c1T_pack,
		              cntl_sub_herk( cntl ) );

		// Unpack C1 (if C1 was packed).
		bl2_unpackm_int( &c1T_pack, &c1T,
		                 cntl_sub_unpackm_c( cntl ) );
	}

	// If any packing buffers were acquired within packm, release them back
	// to the memory manager.
	bl2_obj_release_pack( &a_pack );
	bl2_obj_release_pack( &ah1_pack );
	bl2_obj_release_pack( &c1T_pack );
}

