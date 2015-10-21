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

void bli_herk_blk_var2f( obj_t*  a,
                         obj_t*  ah,
                         obj_t*  c,
                         gemm_t* cntl,
                         herk_thrinfo_t* thread )
{
    obj_t a_pack_s;
    obj_t ah1_pack_s, c1_pack_s;

    obj_t ah1, c1;
    obj_t* a_pack;
    obj_t* ah1_pack;
    obj_t* c1_pack;

	dim_t i;
	dim_t b_alg;

	// Prune any zero region that exists along the partitioning dimension.
	bli_herk_prune_unref_mparts_n( a, ah, c );

    if( thread_am_ochief( thread ) ) {
        // Initialize object for packing A
	    bli_obj_init_pack( &a_pack_s );
        bli_packm_init( a, &a_pack_s,
                        cntl_sub_packm_a( cntl ) );

        // Scale C by beta (if instructed).
        bli_scalm_int( &BLIS_ONE,
                       c,
                       cntl_sub_scalm( cntl ) );
    }
    a_pack = thread_obroadcast( thread, &a_pack_s );

	// Initialize pack objects for C and A' that are passed into packm_init().
    if( thread_am_ichief( thread ) ) {
        bli_obj_init_pack( &ah1_pack_s );
        bli_obj_init_pack( &c1_pack_s );
    }
    ah1_pack = thread_ibroadcast( thread, &ah1_pack_s );
    c1_pack  = thread_ibroadcast( thread, &c1_pack_s );

	// Pack A (if instructed).
	bli_packm_int( a, a_pack,
	               cntl_sub_packm_a( cntl ),
                   herk_thread_sub_opackm( thread ) );

    dim_t my_start, my_end;
    bli_get_range_weighted_l2r( thread, c,
                                bli_blksz_get_mult_for_obj( a, cntl_blocksize( cntl ) ),
                                &my_start, &my_end );

	// Partition along the n dimension.
	for ( i = my_start; i < my_end; i += b_alg )
	{
		// Determine the current algorithmic blocksize.
		b_alg = bli_determine_blocksize_f( i, my_end, a,
		                                   cntl_blocksize( cntl ) );

		// Acquire partitions for A1' and C1.
		bli_acquire_mpart_l2r( BLIS_SUBPART1,
		                       i, b_alg, ah, &ah1 );
		bli_acquire_mpart_l2r( BLIS_SUBPART1,
		                       i, b_alg, c, &c1 );

		// Initialize objects for packing A1' and C1.
        if( thread_am_ichief( thread ) ) {
            bli_packm_init( &ah1, ah1_pack,
                            cntl_sub_packm_b( cntl ) );
            bli_packm_init( &c1, c1_pack,
                            cntl_sub_packm_c( cntl ) );
        }
        thread_ibarrier( thread ) ;

		// Pack A1' (if instructed).
		bli_packm_int( &ah1, ah1_pack,
		               cntl_sub_packm_b( cntl ),
                       herk_thread_sub_ipackm( thread ) );

		// Pack C1 (if instructed).
		bli_packm_int( &c1, c1_pack,
		               cntl_sub_packm_c( cntl ),
                       herk_thread_sub_ipackm( thread ) ) ;

		// Perform herk subproblem.
		bli_herk_int( &BLIS_ONE,
		              a_pack,
		              ah1_pack,
		              &BLIS_ONE,
		              c1_pack,
		              cntl_sub_gemm( cntl ),
                      herk_thread_sub_herk( thread ) );

        thread_ibarrier( thread );

		// Unpack C1 (if C1 was packed).
        bli_unpackm_int( c1_pack, &c1,
                         cntl_sub_unpackm_c( cntl ),
                         herk_thread_sub_ipackm( thread ) );
	}

	// If any packing buffers were acquired within packm, release them back
	// to the memory manager.
    thread_obarrier( thread );
    if( thread_am_ochief( thread ) )
        bli_packm_release( a_pack, cntl_sub_packm_a( cntl ) );
    if( thread_am_ichief( thread ) ) {
        bli_packm_release( ah1_pack, cntl_sub_packm_b( cntl ) );
        bli_packm_release( c1_pack, cntl_sub_packm_c( cntl ) );
    }
}

