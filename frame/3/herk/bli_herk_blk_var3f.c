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

void bli_herk_blk_var3f( obj_t*  a,
                         obj_t*  ah,
                         obj_t*  c,
                         gemm_t* cntl,
                         herk_thrinfo_t* thread )
{
    obj_t  c_pack_s;
    obj_t  a1_pack_s, ah1_pack_s;

	obj_t  a1, ah1;
    obj_t* a1_pack = NULL;
    obj_t* ah1_pack = NULL;
	obj_t* c_pack = NULL;

	dim_t  i;
	dim_t  b_alg;
	dim_t  k_trans;

	// Prune any zero region that exists along the partitioning dimension.
	bli_herk_prune_unref_mparts_k( a, ah, c );

    if( thread_am_ochief( thread ) ) {
        // Initialize object for packing C.
	    bli_obj_init_pack( &c_pack_s );
        bli_packm_init( c, &c_pack_s,
                        cntl_sub_packm_c( cntl ) );
        
        // Scale C by beta (if instructed).
        bli_scalm_int( &BLIS_ONE,
                       c,
                       cntl_sub_scalm( cntl ) );
    }
    c_pack = thread_obroadcast( thread, &c_pack_s );

	// Initialize all pack objects that are passed into packm_init().
    if( thread_am_ichief( thread ) ) {
        bli_obj_init_pack( &a1_pack_s );
        bli_obj_init_pack( &ah1_pack_s );
    }
    a1_pack = thread_ibroadcast( thread, &a1_pack_s );
    ah1_pack = thread_ibroadcast( thread, &ah1_pack_s );

	// Pack C (if instructed).
	bli_packm_int( c, c_pack,
	               cntl_sub_packm_c( cntl ),
                   herk_thread_sub_opackm( thread ) );

	// Query dimension in partitioning direction.
	k_trans = bli_obj_width_after_trans( *a );

	// Partition along the k dimension.
	for ( i = 0; i < k_trans; i += b_alg )
	{
		// Determine the current algorithmic blocksize.
		b_alg = bli_determine_blocksize_f( i, k_trans, a,
		                                   cntl_blocksize( cntl ) );

		// Acquire partitions for A1 and A1'.
		bli_acquire_mpart_l2r( BLIS_SUBPART1,
		                       i, b_alg, a, &a1 );
		bli_acquire_mpart_t2b( BLIS_SUBPART1,
		                       i, b_alg, ah, &ah1 );

		// Initialize objects for packing A1 and A1'.
        if( thread_am_ichief( thread ) ) {
            bli_packm_init( &a1, a1_pack,
                            cntl_sub_packm_a( cntl ) );
            bli_packm_init( &ah1, ah1_pack,
                            cntl_sub_packm_b( cntl ) );
        }
        thread_ibarrier( thread );

		// Pack A1 (if instructed).
		bli_packm_int( &a1, a1_pack,
		               cntl_sub_packm_a( cntl ),
                       herk_thread_sub_ipackm( thread ) );

		// Pack B1 (if instructed).
		bli_packm_int( &ah1, ah1_pack,
		               cntl_sub_packm_b( cntl ),
                       herk_thread_sub_ipackm( thread ) );

		// Perform herk subproblem.
		bli_herk_int( &BLIS_ONE,
		              a1_pack,
		              ah1_pack,
		              &BLIS_ONE,
		              c_pack,
		              cntl_sub_gemm( cntl ),
                      herk_thread_sub_herk( thread ) );

        // This variant executes multiple rank-k updates. Therefore, if the
        // internal beta scalar on matrix C is non-zero, we must use it
        // only for the first iteration (and then BLIS_ONE for all others).
        // And since c_pack is a local obj_t, we can simply overwrite the
        // internal beta scalar with BLIS_ONE once it has been used in the
        // first iteration.
        thread_ibarrier( thread );
        if ( i == 0 && thread_am_ichief( thread ) ) bli_obj_scalar_reset( c_pack );

	}

    thread_obarrier( thread );
    
	// Unpack C (if C was packed).
    bli_unpackm_int( c_pack, c,
                     cntl_sub_unpackm_c( cntl ),
                     herk_thread_sub_opackm( thread ) );

	// If any packing buffers were acquired within packm, release them back
	// to the memory manager.
    if( thread_am_ochief( thread ) ) {
	    bli_packm_release( c_pack, cntl_sub_packm_c( cntl ) );
    }
    if( thread_am_ichief( thread ) ) {
        bli_packm_release( a1_pack, cntl_sub_packm_a( cntl ) );
        bli_packm_release( ah1_pack, cntl_sub_packm_b( cntl ) );
    }
}

