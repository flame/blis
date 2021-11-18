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

#ifdef BLIS_ENABLE_DMA

void bli_gemm_blk_var3_dma
     (
       obj_t*  a,
       obj_t*  b,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm,
       cntl_t* cntl,
       thrinfo_t* thread
     )
{
	obj_t a1, b1;

	// Determine the direction in which to partition (forwards or backwards).
	dir_t direct = bli_l3_direct( a, b, c, cntl );

	// Prune any zero region that exists along the partitioning dimension.
	bli_l3_prune_unref_mparts_k( a, b, c, cntl );

	// Query dimension in partitioning direction.
	dim_t k_trans = bli_obj_width_after_trans( a );

	// ========================================================================
	// DMA settings
	// ========================================================================
	obj_t b1_dma;
	dim_t b_alg, b_alg_next;

	// Event DMA
	dma_event_t event_b1_dma;

	// Initialize mem_t for b1_dma
	mem_t mem_b1_dma = BLIS_MEM_INITIALIZER;
	bli_mem_set_buf_type( BLIS_BUFFER_FOR_B_PANEL, &mem_b1_dma );

	// Setup next DMA prefetch to the packm subnode
	bli_cntl_packm_params_set_a_dma(     &b1,           bli_cntl_sub_node( cntl ) );
	bli_cntl_packm_params_set_p_dma(     &b1_dma,       bli_cntl_sub_node( cntl ) );
	bli_cntl_packm_params_set_mem_p_dma( &mem_b1_dma,   bli_cntl_sub_node( cntl ) );
	bli_cntl_packm_params_set_event_dma( &event_b1_dma, bli_cntl_sub_node( cntl ) );

	// ========================================================================
	// PROLOG DMA: Get the first panel B
	// ========================================================================
	// Acquire next partitions for B1.
	b_alg = 0;
	b_alg_next = bli_l3_determine_kc( direct, 0 /*i*/, k_trans, a, b,
	                             bli_cntl_bszid( cntl ), cntx, cntl );

	#ifdef BLIS_DMA_DEBUG
	fprintf( stdout, "\n" );
	fprintf( stdout, "%s(): k_trans %d b_alg %d b_alg_next %d\n",
	         __FUNCTION__, k_trans, b_alg, b_alg_next );
	#endif // BLIS_DMA_DEBUG

	bli_acquire_mpart_mdim( direct, BLIS_SUBPART1,
	                        0 /*i*/, b_alg_next, b, &b1 );

	// Get first B panel
	bli_dma_get( &b1, &b1_dma, &mem_b1_dma, &event_b1_dma,
	             rntm, bli_thrinfo_sub_node( thread ) );

	// ========================================================================
	// Loop: Partition along the k dimension.
	// ========================================================================
	for ( dim_t i = 0; i < k_trans; i += b_alg )
	{
		// Update current b_alg with b_alg_next of the previous iteration
		b_alg = b_alg_next;

		// Determine the next algorithmic blocksize.
		b_alg_next = bli_l3_determine_kc( direct, i+b_alg, k_trans, a, b,
		                                  bli_cntl_bszid( cntl ), cntx, cntl );

		#ifdef BLIS_DMA_DEBUG
		fprintf( stdout, "%s(): k_trans %d b_alg %d b_alg_next %d\n",
		         __FUNCTION__, k_trans, b_alg, b_alg_next );
		#endif // BLIS_DMA_DEBUG

		// DMA: prepare to get next block (if any)
		if ( b_alg_next > 0 )
		{
			// Acquire next partitions for B1.
			bli_acquire_mpart_mdim( direct, BLIS_SUBPART1,
			                        i+b_alg, b_alg_next, b, &b1 );
		}
		else
		{
			// If no more block, stop DMA-prefetching after packm by setting
			// obj_t* and mem_t* to NULL
			bli_cntl_packm_params_set_a_dma( NULL, bli_cntl_sub_node( cntl ) );
			bli_cntl_packm_params_set_p_dma( NULL, bli_cntl_sub_node( cntl ) );
			bli_cntl_packm_params_set_mem_p_dma( NULL, bli_cntl_sub_node( cntl ) );
		}

		// Acquire current partitions for A1.
		bli_acquire_mpart_ndim( direct, BLIS_SUBPART1,
		                        i, b_alg, a, &a1 );

		// DMA: wait for arrival of current partitions B1
		bli_dma_wait( &event_b1_dma, bli_thrinfo_sub_node( thread ) );

		// Perform gemm subproblem.
		bli_gemm_int
		(
		  &BLIS_ONE,
		  &a1,
		  &b1_dma,
		  &BLIS_ONE,
		  c,
		  cntx,
		  rntm,
		  bli_cntl_sub_node( cntl ),
		  bli_thrinfo_sub_node( thread )
		);

		bli_thread_barrier( bli_thrinfo_sub_node( thread ) );

		// This variant executes multiple rank-k updates. Therefore, if the
		// internal beta scalar on matrix C is non-zero, we must use it
		// only for the first iteration (and then BLIS_ONE for all others).
		// And since c is a locally aliased obj_t (see _int() function), we
		// can simply overwrite the internal beta scalar with BLIS_ONE once
		// it has been used in the first iteration. However...

		// Unlike variant 3 of gemm and herk, which reset the internal scalar
		// on C at the end of the first iteration so that subsequent iterations
		// do not erroneously apply beta more than once, it is important that
		// this behavior not be applied to trmm. That is because the order of
		// computation is always such that the beta that is passed into the
		// macro-kernel must be zero, since the macro-kernel only applies that
		// beta to (and thus overwrites) the row-panel of C that corresponds to
		// the current block intersecting the diagonal. It turns out that this
		// same pattern holds for trmm3 as well--except there, the beta scalar
		// is potentially non-zero, but is still applied only to the current
		// row-panel of C, and thus beta is applied to all of C exactly once.
		// Thus, for neither trmm nor trmm3 should we reset the scalar on C
		// after the first iteration.
		if ( bli_cntl_family( cntl ) != BLIS_TRMM )
		if ( i == 0 ) bli_obj_scalar_reset( c );
	}

	// ========================================================================
	// Release DMA buffer of B1 at the end
	// ========================================================================
	if ( bli_thread_am_ochief( bli_thrinfo_sub_node( thread ) ) )
	{
		if ( bli_mem_is_alloc( &mem_b1_dma ) ) {
			bli_pba_release( rntm, &mem_b1_dma );
		}
	}
}

#endif // BLIS_ENABLE_DMA
