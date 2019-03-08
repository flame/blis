/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018, Advanced Micro Devices, Inc.

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
#include "assert.h"

void bli_l3_thrinfo_init_single
     (
       thrinfo_t* thread
     )
{
	bli_thrinfo_init_single( thread );
}

void bli_l3_thrinfo_free
     (
       rntm_t*    rntm,
       thrinfo_t* thread
     )
{
	bli_thrinfo_free( rntm, thread );
}

// -----------------------------------------------------------------------------

void bli_l3_thrinfo_create_root
     (
       dim_t       id,
       thrcomm_t*  gl_comm,
       rntm_t*     rntm,
       cntl_t*     cntl,
       thrinfo_t** thread
     )
{
	// Query the global communicator for the total number of threads to use.
	dim_t   n_threads  = bli_thrcomm_num_threads( gl_comm );

	// Use the thread id passed in as the global communicator id.
	dim_t   gl_comm_id = id;

	// Use the blocksize id of the current (root) control tree node to
	// query the top-most ways of parallelism to obtain.
	bszid_t bszid      = bli_cntl_bszid( cntl );
	dim_t   xx_way     = bli_rntm_ways_for( bszid, rntm );

	// Determine the work id for this thrinfo_t node.
	dim_t   work_id    = gl_comm_id / ( n_threads / xx_way );

	// Create the root thrinfo_t node.
	*thread = bli_thrinfo_create
	(
	  rntm,
	  gl_comm,
	  gl_comm_id,
	  xx_way,
	  work_id,
	  TRUE,
	  bszid,
	  NULL
	);
}

// -----------------------------------------------------------------------------

void bli_l3_thrinfo_print_gemm_paths
     (
       thrinfo_t** threads
     )
{
	dim_t n_threads = bli_thread_num_threads( threads[0] );
	dim_t gl_id;

	thrinfo_t* jc_info  = threads[0];
	thrinfo_t* pc_info  = bli_thrinfo_sub_node( jc_info );
	thrinfo_t* pb_info  = bli_thrinfo_sub_node( pc_info );
	thrinfo_t* ic_info  = bli_thrinfo_sub_node( pb_info );
	thrinfo_t* pa_info  = bli_thrinfo_sub_node( ic_info );
	thrinfo_t* jr_info  = bli_thrinfo_sub_node( pa_info );
	thrinfo_t* ir_info  = bli_thrinfo_sub_node( jr_info );

	dim_t jc_way = bli_thread_n_way( jc_info );
	dim_t pc_way = bli_thread_n_way( pc_info );
	dim_t pb_way = bli_thread_n_way( pb_info );
	dim_t ic_way = bli_thread_n_way( ic_info );
	dim_t pa_way = bli_thread_n_way( pa_info );
	dim_t jr_way = bli_thread_n_way( jr_info );
	dim_t ir_way = bli_thread_n_way( ir_info );

	dim_t jc_nt = bli_thread_num_threads( jc_info );
	dim_t pc_nt = bli_thread_num_threads( pc_info );
	dim_t pb_nt = bli_thread_num_threads( pb_info );
	dim_t ic_nt = bli_thread_num_threads( ic_info );
	dim_t pa_nt = bli_thread_num_threads( pa_info );
	dim_t jr_nt = bli_thread_num_threads( jr_info );
	dim_t ir_nt = bli_thread_num_threads( ir_info );

	printf( "            jc   kc   pb   ic   pa   jr   ir\n" );
	printf( "xx_nt:    %4lu %4lu %4lu %4lu %4lu %4lu %4lu\n",
	( unsigned long )jc_nt,
	( unsigned long )pc_nt,
	( unsigned long )pb_nt,
	( unsigned long )ic_nt,
	( unsigned long )pa_nt,
	( unsigned long )jr_nt,
	( unsigned long )ir_nt );
	printf( "xx_way:   %4lu %4lu %4lu %4lu %4lu %4lu %4lu\n",
    ( unsigned long )jc_way,
	( unsigned long )pc_way,
	( unsigned long )pb_way,
	( unsigned long )ic_way,
	( unsigned long )pa_way,
	( unsigned long )jr_way,
	( unsigned long )ir_way );
	printf( "============================================\n" );

	dim_t jc_comm_id;
	dim_t pc_comm_id;
	dim_t pb_comm_id;
	dim_t ic_comm_id;
	dim_t pa_comm_id;
	dim_t jr_comm_id;
	dim_t ir_comm_id;

	dim_t jc_work_id;
	dim_t pc_work_id;
	dim_t pb_work_id;
	dim_t ic_work_id;
	dim_t pa_work_id;
	dim_t jr_work_id;
	dim_t ir_work_id;

	for ( gl_id = 0; gl_id < n_threads; ++gl_id )
	{
		jc_info = threads[gl_id];

		// NOTE: We must check each thrinfo_t pointer for NULLness. Certain threads
		// may not fully build their thrinfo_t structures--specifically when the
		// dimension being parallelized is not large enough for each thread to have
		// even one unit of work (where as unit is usually a single micropanel's
		// width, MR or NR).
		if ( !jc_info )
		{
			jc_comm_id = pc_comm_id = pb_comm_id = ic_comm_id = pa_comm_id = jr_comm_id = ir_comm_id = -1;
			jc_work_id = pc_work_id = pb_work_id = ic_work_id = pa_work_id = jr_work_id = ir_work_id = -1;
		}
		else
		{
			jc_comm_id = bli_thread_ocomm_id( jc_info );
			jc_work_id = bli_thread_work_id( jc_info );
			pc_info = bli_thrinfo_sub_node( jc_info );

			if ( !pc_info )
			{
				pc_comm_id = pb_comm_id = ic_comm_id = pa_comm_id = jr_comm_id = ir_comm_id = -1;
				pc_work_id = pb_work_id = ic_work_id = pa_work_id = jr_work_id = ir_work_id = -1;
			}
			else
			{
				pc_comm_id = bli_thread_ocomm_id( pc_info );
				pc_work_id = bli_thread_work_id( pc_info );
				pb_info = bli_thrinfo_sub_node( pc_info );

				if ( !pb_info )
				{
					pb_comm_id = ic_comm_id = pa_comm_id = jr_comm_id = ir_comm_id = -1;
					pb_work_id = ic_work_id = pa_work_id = jr_work_id = ir_work_id = -1;
				}
				else
				{
					pb_comm_id = bli_thread_ocomm_id( pb_info );
					pb_work_id = bli_thread_work_id( pb_info );
					ic_info = bli_thrinfo_sub_node( pb_info );

					if ( !ic_info )
					{
						ic_comm_id = pa_comm_id = jr_comm_id = ir_comm_id = -1;
						ic_work_id = pa_work_id = jr_work_id = ir_work_id = -1;
					}
					else
					{
						ic_comm_id = bli_thread_ocomm_id( ic_info );
						ic_work_id = bli_thread_work_id( ic_info );
						pa_info = bli_thrinfo_sub_node( ic_info );

						if ( !pa_info )
						{
							pa_comm_id = jr_comm_id = ir_comm_id = -1;
							pa_work_id = jr_work_id = ir_work_id = -1;
						}
						else
						{
							pa_comm_id = bli_thread_ocomm_id( pa_info );
							pa_work_id = bli_thread_work_id( pa_info );
							jr_info = bli_thrinfo_sub_node( pa_info );

							if ( !jr_info )
							{
								jr_comm_id = ir_comm_id = -1;
								jr_work_id = ir_work_id = -1;
							}
							else
							{
								jr_comm_id = bli_thread_ocomm_id( jr_info );
								jr_work_id = bli_thread_work_id( jr_info );
								ir_info = bli_thrinfo_sub_node( jr_info );

								if ( !ir_info )
								{
									ir_comm_id = -1;
									ir_work_id = -1;
								}
								else
								{
									ir_comm_id = bli_thread_ocomm_id( ir_info );
									ir_work_id = bli_thread_work_id( ir_info );
								}
							}
						}
					}
				}
			}
		}

		//printf( "            gl   jc   pb   kc   pa   ic   jr  \n" );
		//printf( "            gl   jc   kc   pb   ic   pa   jr  \n" );
		printf( "comm ids: %4ld %4ld %4ld %4ld %4ld %4ld %4ld\n",
		( long )jc_comm_id,
		( long )pc_comm_id,
		( long )pb_comm_id,
		( long )ic_comm_id,
		( long )pa_comm_id,
		( long )jr_comm_id,
		( long )ir_comm_id );
		printf( "work ids: %4ld %4ld %4ld %4ld %4ld %4ld %4ld\n",
		( long )jc_work_id,
		( long )pc_work_id,
		( long )pb_work_id,
		( long )ic_work_id,
		( long )pa_work_id,
		( long )jr_work_id,
		( long )ir_work_id );
		printf( "--------------------------------------------\n" );
	}

}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

void bli_l3_thrinfo_print_trsm_paths
     (
       thrinfo_t** threads
     )
{
	dim_t n_threads = bli_thread_num_threads( threads[0] );
	dim_t gl_id;

	thrinfo_t* jc_info  = threads[0];
	thrinfo_t* pc_info  = bli_thrinfo_sub_node( jc_info );
	thrinfo_t* pb_info  = bli_thrinfo_sub_node( pc_info );
	thrinfo_t* ic_info  = bli_thrinfo_sub_node( pb_info );

	thrinfo_t* pa_info  = bli_thrinfo_sub_node( ic_info );
	thrinfo_t* jr_info  = bli_thrinfo_sub_node( pa_info );
	thrinfo_t* ir_info  = bli_thrinfo_sub_node( jr_info );
	thrinfo_t* pa_info0 = bli_thrinfo_sub_prenode( ic_info );
	thrinfo_t* jr_info0 = ( pa_info0 ? bli_thrinfo_sub_node( pa_info0 ) : NULL );
	thrinfo_t* ir_info0 = ( jr_info0 ? bli_thrinfo_sub_node( jr_info0 ) : NULL );

	dim_t jc_way  = bli_thread_n_way( jc_info );
	dim_t pc_way  = bli_thread_n_way( pc_info );
	dim_t pb_way  = bli_thread_n_way( pb_info );
	dim_t ic_way  = bli_thread_n_way( ic_info );

	dim_t pa_way  = bli_thread_n_way( pa_info );
	dim_t jr_way  = bli_thread_n_way( jr_info );
	dim_t ir_way  = bli_thread_n_way( ir_info );
	dim_t pa_way0 = ( pa_info0 ? bli_thread_n_way( pa_info0 ) : -1 );
	dim_t jr_way0 = ( jr_info0 ? bli_thread_n_way( jr_info0 ) : -1 );
	dim_t ir_way0 = ( ir_info0 ? bli_thread_n_way( ir_info0 ) : -1 );

	dim_t jc_nt  = bli_thread_num_threads( jc_info );
	dim_t pc_nt  = bli_thread_num_threads( pc_info );
	dim_t pb_nt  = bli_thread_num_threads( pb_info );
	dim_t ic_nt  = bli_thread_num_threads( ic_info );

	dim_t pa_nt  = bli_thread_num_threads( pa_info );
	dim_t jr_nt  = bli_thread_num_threads( jr_info );
	dim_t ir_nt  = bli_thread_num_threads( ir_info );
	dim_t pa_nt0 = ( pa_info0 ? bli_thread_num_threads( pa_info0 ) : -1 );
	dim_t jr_nt0 = ( jr_info0 ? bli_thread_num_threads( jr_info0 ) : -1 );
	dim_t ir_nt0 = ( ir_info0 ? bli_thread_num_threads( ir_info0 ) : -1 );

	printf( "            jc   kc   pb   ic     pa     jr     ir\n" );
	printf( "xx_nt:    %4ld %4ld %4ld %4ld  %2ld|%2ld  %2ld|%2ld  %2ld|%2ld\n",
	( long )jc_nt,
	( long )pc_nt,
	( long )pb_nt,
	( long )ic_nt,
	( long )pa_nt0, ( long )pa_nt,
	( long )jr_nt0, ( long )jr_nt,
	( long )ir_nt0, ( long )ir_nt );
	printf( "xx_way:   %4ld %4ld %4ld %4ld  %2ld|%2ld  %2ld|%2ld  %2ld|%2ld\n",
    ( long )jc_way,
	( long )pc_way,
	( long )pb_way,
	( long )ic_way,
	( long )pa_way0, ( long )pa_way,
	( long )jr_way0, ( long )jr_way,
	( long )ir_way0, ( long )ir_way );
	printf( "==================================================\n" );

	dim_t jc_comm_id;
	dim_t pc_comm_id;
	dim_t pb_comm_id;
	dim_t ic_comm_id;
	dim_t pa_comm_id0, pa_comm_id;
	dim_t jr_comm_id0, jr_comm_id;
	dim_t ir_comm_id0, ir_comm_id;

	dim_t jc_work_id;
	dim_t pc_work_id;
	dim_t pb_work_id;
	dim_t ic_work_id;
	dim_t pa_work_id0, pa_work_id;
	dim_t jr_work_id0, jr_work_id;
	dim_t ir_work_id0, ir_work_id;

	for ( gl_id = 0; gl_id < n_threads; ++gl_id )
	{
		jc_info = threads[gl_id];

		// NOTE: We must check each thrinfo_t pointer for NULLness. Certain threads
		// may not fully build their thrinfo_t structures--specifically when the
		// dimension being parallelized is not large enough for each thread to have
		// even one unit of work (where as unit is usually a single micropanel's
		// width, MR or NR).
		if ( !jc_info )
		{
			jc_comm_id = pc_comm_id = pb_comm_id = ic_comm_id = pa_comm_id = jr_comm_id = ir_comm_id = -1;
			jc_work_id = pc_work_id = pb_work_id = ic_work_id = pa_work_id = jr_work_id = ir_work_id = -1;
		}
		else
		{
			jc_comm_id = bli_thread_ocomm_id( jc_info );
			jc_work_id = bli_thread_work_id( jc_info );
			pc_info = bli_thrinfo_sub_node( jc_info );

			if ( !pc_info )
			{
				pc_comm_id = pb_comm_id = ic_comm_id = pa_comm_id = jr_comm_id = ir_comm_id = -1;
				pc_work_id = pb_work_id = ic_work_id = pa_work_id = jr_work_id = ir_work_id = -1;
			}
			else
			{
				pc_comm_id = bli_thread_ocomm_id( pc_info );
				pc_work_id = bli_thread_work_id( pc_info );
				pb_info = bli_thrinfo_sub_node( pc_info );

				if ( !pb_info )
				{
					pb_comm_id = ic_comm_id = pa_comm_id = jr_comm_id = ir_comm_id = -1;
					pb_work_id = ic_work_id = pa_work_id = jr_work_id = ir_work_id = -1;
				}
				else
				{
					pb_comm_id = bli_thread_ocomm_id( pb_info );
					pb_work_id = bli_thread_work_id( pb_info );
					ic_info = bli_thrinfo_sub_node( pb_info );

					if ( !ic_info )
					{
						ic_comm_id = pa_comm_id = jr_comm_id = ir_comm_id = -1;
						ic_work_id = pa_work_id = jr_work_id = ir_work_id = -1;
					}
					else
					{
						ic_comm_id = bli_thread_ocomm_id( ic_info );
						ic_work_id = bli_thread_work_id( ic_info );
						pa_info0 = bli_thrinfo_sub_prenode( ic_info );
						pa_info = bli_thrinfo_sub_node( ic_info );

						// Prenode
						if ( !pa_info0 )
						{
							pa_comm_id0 = jr_comm_id0 = ir_comm_id0 = -1;
							pa_work_id0 = jr_work_id0 = ir_work_id0 = -1;
						}
						else
						{
							pa_comm_id0 = bli_thread_ocomm_id( pa_info0 );
							pa_work_id0 = bli_thread_work_id( pa_info0 );
							jr_info0 = bli_thrinfo_sub_node( pa_info0 );

							if ( !jr_info0 )
							{
								jr_comm_id0 = ir_comm_id0 = -1;
								jr_work_id0 = ir_work_id0 = -1;
							}
							else
							{
								jr_comm_id0 = bli_thread_ocomm_id( jr_info0 );
								jr_work_id0 = bli_thread_work_id( jr_info0 );
								ir_info0 = bli_thrinfo_sub_node( jr_info0 );

								if ( !ir_info0 )
								{
									ir_comm_id0 = -1;
									ir_work_id0 = -1;
								}
								else
								{
									ir_comm_id0 = bli_thread_ocomm_id( ir_info0 );
									ir_work_id0 = bli_thread_work_id( ir_info0 );
								}
							}
						}

						// Main node
						if ( !pa_info )
						{
							pa_comm_id = jr_comm_id = ir_comm_id = -1;
							pa_work_id = jr_work_id = ir_work_id = -1;
						}
						else
						{
							pa_comm_id = bli_thread_ocomm_id( pa_info );
							pa_work_id = bli_thread_work_id( pa_info );
							jr_info = bli_thrinfo_sub_node( pa_info );

							if ( !jr_info )
							{
								jr_comm_id = ir_comm_id = -1;
								jr_work_id = ir_work_id = -1;
							}
							else
							{
								jr_comm_id = bli_thread_ocomm_id( jr_info );
								jr_work_id = bli_thread_work_id( jr_info );
								ir_info = bli_thrinfo_sub_node( jr_info );

								if ( !ir_info )
								{
									ir_comm_id = -1;
									ir_work_id = -1;
								}
								else
								{
									ir_comm_id = bli_thread_ocomm_id( ir_info );
									ir_work_id = bli_thread_work_id( ir_info );
								}
							}
						}
					}
				}
			}
		}

		printf( "comm ids: %4ld %4ld %4ld %4ld  %2ld|%2ld  %2ld|%2ld  %2ld|%2ld\n",
		( long )jc_comm_id,
		( long )pc_comm_id,
		( long )pb_comm_id,
		( long )ic_comm_id,
		( long )pa_comm_id0, ( long )pa_comm_id,
		( long )jr_comm_id0, ( long )jr_comm_id,
		( long )ir_comm_id0, ( long )ir_comm_id );
		printf( "work ids: %4ld %4ld %4ld %4ld  %2ld|%2ld  %2ld|%2ld  %2ld|%2ld\n",
		( long )jc_work_id,
		( long )pc_work_id,
		( long )pb_work_id,
		( long )ic_work_id,
		( long )pa_work_id0, ( long )pa_work_id,
		( long )jr_work_id0, ( long )jr_work_id,
		( long )ir_work_id0, ( long )ir_work_id );
		printf( "--------------------------------------------------\n" );
	}

}

// -----------------------------------------------------------------------------

void bli_l3_thrinfo_free_paths
     (
       rntm_t*     rntm,
       thrinfo_t** threads
     )
{
	dim_t n_threads = bli_thread_num_threads( threads[0] );
	dim_t i;

	for ( i = 0; i < n_threads; ++i )
		bli_l3_thrinfo_free( rntm, threads[i] );

	bli_free_intl( threads );
}

