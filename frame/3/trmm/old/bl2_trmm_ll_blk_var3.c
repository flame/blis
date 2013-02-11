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

#include "blis2.h"

void bl2_trmm_ll_blk_var3( obj_t*  alpha,
                           obj_t*  a,
                           obj_t*  b,
                           trmm_t* cntl )
{
	obj_t a_pack;
	obj_t b1, b1_pack;

	dim_t j;
	dim_t b_alg;
	dim_t n_trans;

	// Initialize objects for packing.
	bl2_obj_init_pack( &a_pack );
	bl2_obj_init_pack( &b1_pack );

	// Query dimension in partitioning direction.
	n_trans = bl2_obj_width_after_trans( *b );

	// Scale B by alpha (if instructed).
	bl2_scalm_int( alpha,
	               b,
	               cntl_sub_scalm( cntl ) );

	// Partition along the n dimension.
	for ( j = 0; j < n_trans; j += b_alg )
	{
		// Determine the current algorithmic blocksize.
		// NOTE: Use of a (for execution datatype) is intentional!
		// This causes the right blocksize to be used if c and a are
		// complex and b is real.
		b_alg = bl2_determine_blocksize_f( j, n_trans,
		                                   a,
		                                   cntl_blocksize( cntl ) );

		// Acquire partitions for B1.
		bl2_acquire_mpart_l2r( BLIS_SUBPART1,
		                       j, b_alg, b, &b1 );

		// Copy/pack A (if instructed) and scale by alpha (if instructed).
		bl2_packm_int( alpha,
		               a,
		               &a_pack,
		               cntl_sub_packm_a( cntl ) );

		// Copy/pack B1 (if instructed) and scale by alpha (if instructed).
		bl2_packm_int( alpha,
		               &b1,
		               &b1_pack,
		               cntl_sub_packm_b( cntl ) );

		// B1 = tril( A ) * B1;
		bl2_trmm_int( BLIS_LEFT,
		              alpha,
		              &a_pack,
		              &b1_pack,
		              cntl_sub_trmm( cntl ) );

		// Copy/unpack B1 (if B1 was packed).
		bl2_unpackm_int( &b1_pack,
		                 &b1,
		                 cntl_sub_unpackm_b( cntl ) );

	}

	// If any packing buffers were acquired within packm, release them back
	// to the memory manager.
	bl2_obj_release_pack( &a_pack );
	bl2_obj_release_pack( &b1_pack );
}

