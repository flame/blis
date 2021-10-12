/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.

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

//#define PRINT

void bli_trsm_blk_var1_dma
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
	dim_t i;
	dim_t my_start, my_end;
	dim_t b_alg;
	const dim_t kc = bli_obj_width( a );

	// Determine the direction in which to partition (forwards or backwards).
	dir_t direct = bli_l3_direct( a, b, c, cntl );

	// Prune any zero region that exists along the partitioning dimension.
	bli_l3_prune_unref_mparts_m( a, b, c, cntl );

	// ========================================================================
	// DMA settings
	// ========================================================================
	dim_t b_alg_next = 0;

	// A-DMA
	obj_t       a1_dma;
	dma_event_t event_a1_dma;
	mem_t       mem_a1_dma = BLIS_MEM_INITIALIZER;

	// Triple-buffering on C-DMA
	// - one for computing
	// - one for putting
	// - one for getting
	obj_t       c1_dma      [3];
	dma_event_t event_c1_dma[3];
	mem_t       mem_c1_dma  [3] = { BLIS_MEM_INITIALIZER };

	// Initialize mem_t for A-DMA and C-DMA
	bli_mem_set_buf_type( BLIS_BUFFER_FOR_A_BLOCK, &mem_a1_dma );

	bli_mem_set_buf_type( BLIS_BUFFER_FOR_C_PANEL, &mem_c1_dma[0] );
	bli_mem_set_buf_type( BLIS_BUFFER_FOR_C_PANEL, &mem_c1_dma[1] );
	bli_mem_set_buf_type( BLIS_BUFFER_FOR_C_PANEL, &mem_c1_dma[2] );


	// ========================================================================
	// Step 1:
	// Isolate the diagonal block A11 and its corresponding row panel C1.
	// ========================================================================
	{
		obj_t a11,   c1;
		obj_t a11_1, c1_1;
		obj_t        c1_1_next;

		bli_acquire_mpart_mdim( direct, BLIS_SUBPART1,
		                        0, kc, a, &a11 );
		bli_acquire_mpart_mdim( direct, BLIS_SUBPART1,
		                        0, kc, c, &c1 );

		// All threads iterate over the entire diagonal block A11.
		my_start = 0; my_end = kc;

		#ifdef PRINT
		printf( "bli_trsm_blk_var1(): a11 is %d x %d at offsets (%3d, %3d)\n",
		        (int)bli_obj_length( &a11 ), (int)bli_obj_width( &a11 ),
		        (int)bli_obj_row_off( &a11 ), (int)bli_obj_col_off( &a11 ) );
		printf( "bli_trsm_blk_var1(): entering trsm subproblem loop.\n" );
		#endif

		// Track if a put is outstanding on any slot, to avoid calling wait twice
		// on any slot
		bool        putting_c1_dma[3] = { FALSE };
		dim_t       c1_counter;

		// Setup next A-DMA prefetch to the packm subnode
		bli_cntl_packm_params_set_a_dma(     &a11_1,        bli_cntl_sub_prenode( cntl ) );
		bli_cntl_packm_params_set_p_dma(     &a1_dma,       bli_cntl_sub_prenode( cntl ) );
		bli_cntl_packm_params_set_mem_p_dma( &mem_a1_dma,   bli_cntl_sub_prenode( cntl ) );
		bli_cntl_packm_params_set_event_dma( &event_a1_dma, bli_cntl_sub_prenode( cntl ) );

		// PROLOG DMA: Get the first panel A and block C
		i  = my_start;
		b_alg = 0;
		b_alg_next = bli_determine_blocksize( direct, i, my_end, a,
		                                      bli_cntl_bszid( cntl ), cntx );

		#ifdef BLIS_DMA_DEBUG
		fprintf( stdout, "\n" );
		fprintf( stdout, "    %s(): b_alg %d b_alg_next %d\n", __FUNCTION__, b_alg, b_alg_next );
		#endif // BLIS_DMA_DEBUG

		// Acquire partitions for A1 and C1.
		bli_acquire_mpart_mdim( direct, BLIS_SUBPART1, i, b_alg_next, &a11, &a11_1 );
		bli_acquire_mpart_mdim( direct, BLIS_SUBPART1, i, b_alg_next, &c1, &c1_1_next );

		bli_dma_get( &a11_1,   &a1_dma,     &mem_a1_dma,     &event_a1_dma,
		             rntm, bli_thrinfo_sub_prenode( thread ) );
		bli_dma_get( &c1_1_next, &c1_dma[0], &mem_c1_dma[0], &event_c1_dma[0],
		             rntm, bli_thrinfo_sub_prenode( thread ) );

		// Partition along the m dimension for the trsm subproblem.
		c1_counter = 0;
		for ( i = my_start; i < my_end; i += b_alg )
		{
			#ifdef PRINT
			printf( "bli_trsm_blk_var1():   a11_1 is %d x %d at offsets (%3d, %3d)\n",
			        (int)bli_obj_length( &a11_1 ), (int)bli_obj_width( &a11_1 ),
			        (int)bli_obj_row_off( &a11_1 ), (int)bli_obj_col_off( &a11_1 ) );
			#endif

			// Update current b_alg with b_alg_next of the previous iteration
			b_alg = b_alg_next;

			// Update c1 with the c1_1_next of the previous iteration
			c1_1 = c1_1_next;

			// Determine current ic slot
			dim_t ic = c1_counter % 3;

			// Increment counter of number of C blocks
			++c1_counter;

			// Determine next ic slot
			dim_t ic_next = c1_counter % 3;

			// Determine the next algorithmic blocksize.
			b_alg_next = bli_determine_blocksize( direct, i+b_alg, my_end, &a11,
			                                 bli_cntl_bszid( cntl ), cntx );

			#ifdef BLIS_DMA_DEBUG
			fprintf( stdout, "    %s(): b_alg %d b_alg_next %d\n", __FUNCTION__, b_alg, b_alg_next );
			#endif // BLIS_DMA_DEBUG

			// DMA: get next block (if any)
			if ( b_alg_next > 0 )
			{
				// Sanity: Before triggering get on slot c1[ic_next], we must
				// wait for its previous put (if any) to finish. This prevents the
				// DMA-get from overriding the "being put" data on the same slot.
				// This wait is needed from the 3rd iteration (i.e c1_counter >= 3,
				// or putting_c1_dma[ic_next] is TRUE).
				if ( putting_c1_dma[ic_next] )
				{
					bli_dma_wait( &event_c1_dma[ic_next], bli_thrinfo_sub_prenode( thread ) );
					putting_c1_dma[ic_next] = FALSE;
				}

				// Acquire next partitions for A1 and C1.
				bli_acquire_mpart_mdim( direct, BLIS_SUBPART1,
				                        i+b_alg, b_alg_next, &a11, &a11_1 );
				bli_acquire_mpart_mdim( direct, BLIS_SUBPART1,
				                        i+b_alg, b_alg_next, &c1, &c1_1_next );

				// get next C
				bli_dma_get( &c1_1_next, &c1_dma[ic_next], &mem_c1_dma[ic_next],
				             &event_c1_dma[ic_next], rntm, bli_thrinfo_sub_prenode( thread ) );
			}
			else
			{
				// If no more block, stop DMA-prefetching of A after packm by setting
				// obj_t* and mem_t* to NULL
				bli_cntl_packm_params_set_a_dma( NULL, bli_cntl_sub_prenode( cntl ) );
				bli_cntl_packm_params_set_p_dma( NULL, bli_cntl_sub_prenode( cntl ) );
				bli_cntl_packm_params_set_mem_p_dma( NULL, bli_cntl_sub_prenode( cntl ) );
			}

			// DMA: wait for arrival of current partitions A1 and C1
			bli_dma_wait( &event_a1_dma    , bli_thrinfo_sub_prenode( thread ) );
			bli_dma_wait( &event_c1_dma[ic], bli_thrinfo_sub_prenode( thread ) );

			// Perform trsm subproblem.
			bli_trsm_int
			(
			  &BLIS_ONE,
			  &a1_dma,
			  b,
			  &BLIS_ONE,
			  &c1_dma[ic],
			  cntx,
			  rntm,
			  bli_cntl_sub_prenode( cntl ),
			  bli_thrinfo_sub_prenode( thread )
			);

			// DMA: put C to global memory
			bli_dma_put( &c1_1, &c1_dma[ic], &event_c1_dma[ic], bli_thrinfo_sub_prenode( thread ) );
			putting_c1_dma[ic] = TRUE;
		}

		// EPILOG DMA: Wait for put C
		for( dim_t ic = 0; ic < 3; ++ic )
		{
			if ( putting_c1_dma[ic] )
			{
				bli_dma_wait( &event_c1_dma[ic], bli_thrinfo_sub_prenode( thread ) );
				putting_c1_dma[ic] = FALSE;
			}
		}

		#ifdef PRINT
		printf( "bli_trsm_blk_var1(): finishing trsm subproblem loop.\n" );
		#endif
	} // Step 1: Isolate the diagonal block A11 and its corresponding row panel C1.


	// ========================================================================
	// We must execute a barrier here because the upcoming rank-k update
	// requires the packed matrix B to be fully updated by the trsm
	// subproblem.
	// ========================================================================
	bli_thread_barrier( thread );


	// ========================================================================
	// Step 2:
	// Isolate the remaining part of the column panel matrix A, which we do by
	// acquiring the subpartition ahead of A11 (that is, A21 or A01, depending
	// on whether we are moving forwards or backwards, respectively).
	// ========================================================================
	{
		obj_t ax1, cx1;
		obj_t a11, c1;
		obj_t      c1_next;
		bli_acquire_mpart_mdim( direct, BLIS_SUBPART1A,
		                        0, kc, a, &ax1 );
		bli_acquire_mpart_mdim( direct, BLIS_SUBPART1A,
		                        0, kc, c, &cx1 );

		#ifdef PRINT
		printf( "bli_trsm_blk_var1(): ax1 is %d x %d at offsets (%3d, %3d)\n",
		        (int)bli_obj_length( &ax1 ), (int)bli_obj_width( &ax1 ),
		        (int)bli_obj_row_off( &ax1 ), (int)bli_obj_col_off( &ax1 ) );
		#endif

		// Determine the current thread's subpartition range for the gemm
		// subproblem over Ax1.
		bli_thread_range_mdim
		(
		  direct, thread, &ax1, b, &cx1, cntl, cntx,
		  &my_start, &my_end
		);

		#ifdef PRINT
		printf( "bli_trsm_blk_var1(): entering gemm subproblem loop (%d->%d).\n", (int)my_start, (int)my_end );
		#endif

		// Track if a put is outstanding on any slot, to avoid calling wait twice
		// on any slot
		bool        putting_c1_dma[3] = { FALSE };
		dim_t       c1_counter;

		// Setup next A-DMA prefetch to the packm subnode
		bli_cntl_packm_params_set_a_dma(     &a11,          bli_cntl_sub_node( cntl ) );
		bli_cntl_packm_params_set_p_dma(     &a1_dma,       bli_cntl_sub_node( cntl ) );
		bli_cntl_packm_params_set_mem_p_dma( &mem_a1_dma,   bli_cntl_sub_node( cntl ) );
		bli_cntl_packm_params_set_event_dma( &event_a1_dma, bli_cntl_sub_node( cntl ) );

		// PROLOG DMA: Get the first panel A and block C
		i  = my_start;
		b_alg = 0;
		b_alg_next = bli_determine_blocksize( direct, i, my_end, a,
		                                      bli_cntl_bszid( cntl ), cntx );

		#ifdef BLIS_DMA_DEBUG
		fprintf( stdout, "\n" );
		fprintf( stdout, "    %s(): b_alg %d b_alg_next %d\n", __FUNCTION__, b_alg, b_alg_next );
		#endif // BLIS_DMA_DEBUG

		// Acquire partitions for A1 and C1.
		bli_acquire_mpart_mdim( direct, BLIS_SUBPART1, i, b_alg_next, &ax1, &a11 );
		bli_acquire_mpart_mdim( direct, BLIS_SUBPART1, i, b_alg_next, &cx1, &c1_next );

		bli_dma_get( &a11,   &a1_dma,     &mem_a1_dma,     &event_a1_dma,
		             rntm, bli_thrinfo_sub_node( thread ) );
		bli_dma_get( &c1_next, &c1_dma[0], &mem_c1_dma[0], &event_c1_dma[0],
		             rntm, bli_thrinfo_sub_node( thread ) );

		// Partition along the m dimension for the gemm subproblem.
		c1_counter = 0;
		for ( i = my_start; i < my_end; i += b_alg )
		{
			// Update current b_alg with b_alg_next of the previous iteration
			b_alg = b_alg_next;

			// Update c1 with the c1_1_next of the previous iteration
			c1 = c1_next;

			// Determine current ic slot
			dim_t ic = c1_counter % 3;

			// Increment counter of number of C blocks
			++c1_counter;

			// Determine next ic slot
			dim_t ic_next = c1_counter % 3;

			// Determine the next algorithmic blocksize.
			b_alg_next = bli_determine_blocksize( direct, i+b_alg, my_end, &ax1,
			                                 bli_cntl_bszid( cntl ), cntx );

			// DMA: get next block (if any)
			if ( b_alg_next > 0 )
			{
				// Sanity: Before triggering get on slot c1[ic_next], we must
				// wait for its previous put (if any) to finish. This prevents the
				// DMA-get from overriding the "being put" data on the same slot.
				// This wait is needed from the 3rd iteration (i.e c1_counter >= 3,
				// or putting_c1_dma[ic_next] is TRUE).
				if ( putting_c1_dma[ic_next] )
				{
					bli_dma_wait( &event_c1_dma[ic_next], bli_thrinfo_sub_node( thread ) );
					putting_c1_dma[ic_next] = FALSE;
				}

				// Acquire next partitions for A1 and C1.
				bli_acquire_mpart_mdim( direct, BLIS_SUBPART1,
				                        i+b_alg, b_alg_next, &ax1, &a11 );
				bli_acquire_mpart_mdim( direct, BLIS_SUBPART1,
				                        i+b_alg, b_alg_next, &cx1, &c1_next );

				// get next C
				bli_dma_get( &c1_next, &c1_dma[ic_next], &mem_c1_dma[ic_next],
				             &event_c1_dma[ic_next], rntm, bli_thrinfo_sub_node( thread ) );
			}
			else
			{
				// If no more block, stop DMA-prefetching of A after packm by setting
				// obj_t* and mem_t* to NULL
				bli_cntl_packm_params_set_a_dma( NULL, bli_cntl_sub_node( cntl ) );
				bli_cntl_packm_params_set_p_dma( NULL, bli_cntl_sub_node( cntl ) );
				bli_cntl_packm_params_set_mem_p_dma( NULL, bli_cntl_sub_node( cntl ) );
			}

			// DMA: wait for arrival of current partitions A1 and C1
			bli_dma_wait( &event_a1_dma    , bli_thrinfo_sub_node( thread ) );
			bli_dma_wait( &event_c1_dma[ic], bli_thrinfo_sub_node( thread ) );

			// Perform trsm subproblem.
			bli_trsm_int
			(
			  &BLIS_ONE,
			  &a1_dma,
			  b,
			  &BLIS_ONE,
			  &c1_dma[ic],
			  cntx,
			  rntm,
			  bli_cntl_sub_node( cntl ),
			  bli_thrinfo_sub_node( thread )
			);

			// DMA: put C to global memory
			bli_dma_put( &c1, &c1_dma[ic], &event_c1_dma[ic], bli_thrinfo_sub_node( thread ) );
			putting_c1_dma[ic] = TRUE;
		}

		// EPILOG DMA: Wait for put C
		for( dim_t ic = 0; ic < 3; ++ic )
		{
			if ( putting_c1_dma[ic] )
			{
				bli_dma_wait( &event_c1_dma[ic], bli_thrinfo_sub_node( thread ) );
				putting_c1_dma[ic] = FALSE;
			}
		}

		#ifdef PRINT
		printf( "bli_trsm_blk_var1(): finishing gemm subproblem loop.\n" );
		#endif
	}  // Step 2: Isolate the remaining part of the column panel matrix A


	// ========================================================================
	// Release DMA buffer of A and C at the end
	// ========================================================================
	if ( bli_thread_am_ochief( bli_thrinfo_sub_node( thread ) ) )
	{
		// release A-DMA
		if ( bli_mem_is_alloc( &mem_a1_dma ) )
		{
			bli_pba_release( rntm, &mem_a1_dma );
		}

		// release C-DMA
		for( dim_t ic = 0; ic < 3; ++ic )
		{
			if ( bli_mem_is_alloc( &mem_c1_dma[ic] ) )
			{
				bli_pba_release( rntm, &mem_c1_dma[ic] );
			}
		}
	}
}

#endif // BLIS_ENABLE_DMA
