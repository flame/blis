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
#include "assert.h"

#if 0
thrinfo_t* bli_l3_thrinfo_create
     (
       thrcomm_t* ocomm,
       dim_t      ocomm_id,
       dim_t      n_way,
       dim_t      work_id,
       thrinfo_t* sub_node
     )
{
	return bli_thrinfo_create
	(
	  ocomm, ocomm_id,
	  n_way,
	  work_id,
	  TRUE,
	  sub_node
	);
}
#endif

void bli_l3_thrinfo_init
     (
       thrinfo_t* thread,
       thrcomm_t* ocomm,
       dim_t      ocomm_id,
       dim_t      n_way,
       dim_t      work_id,
       thrinfo_t* sub_node
     )
{
	bli_thrinfo_init
	(
	  thread,
	  ocomm, ocomm_id,
	  n_way,
	  work_id,
	  TRUE,
	  sub_node
	);
}

void bli_l3_thrinfo_init_single
     (
       thrinfo_t* thread
     )
{
	bli_thrinfo_init_single( thread );
}

void bli_l3_thrinfo_free
     (
       thrinfo_t* thread
     )
{
	if ( thread == NULL ||
	     thread == &BLIS_PACKM_SINGLE_THREADED ||
	     thread == &BLIS_GEMM_SINGLE_THREADED
	   ) return;

	thrinfo_t* thrinfo_sub_node = bli_thrinfo_sub_node( thread );

	// Free the communicators, but only if the current thrinfo_t struct
	// is marked as needing them to be freed. The most common example of
	// thrinfo_t nodes NOT marked as needing their comms freed are those
	// associated with packm thrinfo_t nodes.
	if ( bli_thrinfo_needs_free_comm( thread ) )
	{
		// The ochief always frees his communicator, and the ichief free its
		// communicator if we are at the leaf node.
		if ( bli_thread_am_ochief( thread ) )
			bli_thrcomm_free( bli_thrinfo_ocomm( thread ) );
	}

	// Free all children of the current thrinfo_t.
	bli_l3_thrinfo_free( thrinfo_sub_node );

	// Free the thrinfo_t struct.
	bli_free_intl( thread );
}

// -----------------------------------------------------------------------------

void bli_l3_thrinfo_create_root
     (
       dim_t       id,
       thrcomm_t*  gl_comm,
       cntx_t*     cntx,
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
	dim_t   xx_way     = bli_cntx_way_for_bszid( bszid, cntx );

	// Determine the work id for this thrinfo_t node.
	dim_t   work_id    = gl_comm_id / ( n_threads / xx_way );

	// Create the root thrinfo_t node.
	*thread = bli_thrinfo_create
	(
	  gl_comm,
	  gl_comm_id,
	  xx_way,
	  work_id,
	  TRUE,
	  NULL
	);
}

// -----------------------------------------------------------------------------

void bli_l3_thrinfo_print_paths
     (
       thrinfo_t** threads
     )
{
	dim_t n_threads = bli_thread_num_threads( threads[0] );
	dim_t gl_comm_id;

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

	dim_t gl_nt = bli_thread_num_threads( jc_info );
	dim_t jc_nt = bli_thread_num_threads( pc_info );
	dim_t pc_nt = bli_thread_num_threads( pb_info );
	dim_t pb_nt = bli_thread_num_threads( ic_info );
	dim_t ic_nt = bli_thread_num_threads( pa_info );
	dim_t pa_nt = bli_thread_num_threads( jr_info );
	dim_t jr_nt = bli_thread_num_threads( ir_info );

	printf( "            gl   jc   kc   pb   ic   pa   jr   ir\n" );
	printf( "xx_nt:    %4lu %4lu %4lu %4lu %4lu %4lu %4lu %4lu\n",
	gl_nt, jc_nt, pc_nt, pb_nt, ic_nt, pa_nt, jr_nt, (dim_t)1 );
	printf( "\n" );
	printf( "            jc   kc   pb   ic   pa   jr   ir\n" );
	printf( "xx_way:   %4lu %4lu %4lu %4lu %4lu %4lu %4lu\n",
    jc_way, pc_way, pb_way, ic_way, pa_way, jr_way, ir_way );
	printf( "=================================================\n" );

	for ( gl_comm_id = 0; gl_comm_id < n_threads; ++gl_comm_id )
	{
		jc_info = threads[gl_comm_id];
		pc_info = bli_thrinfo_sub_node( jc_info );
		pb_info = bli_thrinfo_sub_node( pc_info );
		ic_info = bli_thrinfo_sub_node( pb_info );
		pa_info = bli_thrinfo_sub_node( ic_info );
		jr_info = bli_thrinfo_sub_node( pa_info );
		ir_info = bli_thrinfo_sub_node( jr_info );

		dim_t gl_comm_id = bli_thread_ocomm_id( jc_info );
		dim_t jc_comm_id = bli_thread_ocomm_id( pc_info );
		dim_t pc_comm_id = bli_thread_ocomm_id( pb_info );
		dim_t pb_comm_id = bli_thread_ocomm_id( ic_info );
		dim_t ic_comm_id = bli_thread_ocomm_id( pa_info );
		dim_t pa_comm_id = bli_thread_ocomm_id( jr_info );
		dim_t jr_comm_id = bli_thread_ocomm_id( ir_info );

		dim_t jc_work_id = bli_thread_work_id( jc_info );
		dim_t pc_work_id = bli_thread_work_id( pc_info );
		dim_t pb_work_id = bli_thread_work_id( pb_info );
		dim_t ic_work_id = bli_thread_work_id( ic_info );
		dim_t pa_work_id = bli_thread_work_id( pa_info );
		dim_t jr_work_id = bli_thread_work_id( jr_info );
		dim_t ir_work_id = bli_thread_work_id( ir_info );

printf( "            gl   jc   pb   kc   pa   ic   jr  \n" );
printf( "comm ids: %4lu %4lu %4lu %4lu %4lu %4lu %4lu\n",
gl_comm_id, jc_comm_id, pc_comm_id, pb_comm_id, ic_comm_id, pa_comm_id, jr_comm_id );
printf( "work ids: %4ld %4ld %4lu %4lu %4ld %4ld %4ld\n",
jc_work_id, pc_work_id, pb_work_id, ic_work_id, pa_work_id, jr_work_id, ir_work_id );
printf( "---------------------------------------\n" );
	}

}

// -----------------------------------------------------------------------------

#if 0
thrinfo_t** bli_l3_thrinfo_create_roots
     (
       cntx_t* cntx,
       cntl_t* cntl
     )
{
	// Query the context for the total number of threads to use.
	dim_t       n_threads = bli_cntx_get_num_threads( cntx );

	// Create a global thread communicator for all the threads.
	thrcomm_t*  gl_comm   = bli_thrcomm_create( n_threads );

	// Allocate an array of thrinfo_t pointers, one for each thread.
	thrinfo_t** paths     = bli_malloc_intl( n_threads * sizeof( thrinfo_t* ) );

	// Use the blocksize id of the current (root) control tree node to
	// query the top-most ways of parallelism to obtain.
	bszid_t     bszid     = bli_cntl_bszid( cntl );
	dim_t       xx_way    = bli_cntx_way_for_bszid( bszid, cntx );

	dim_t       gl_comm_id;

	// Create one thrinfo_t node for each thread in the (global) communicator.
	for ( gl_comm_id = 0; gl_comm_id < n_threads; ++gl_comm_id )
	{
		dim_t work_id = gl_comm_id / ( n_threads / xx_way );

		paths[ gl_comm_id ] = bli_thrinfo_create
		(
		  gl_comm,
		  gl_comm_id,
		  xx_way,
		  work_id,
		  TRUE,
		  NULL
		);
	}

	return paths;
}

//#define PRINT_THRINFO

thrinfo_t** bli_l3_thrinfo_create_full_paths
     (
       cntx_t* cntx
     )
{
	dim_t jc_way = bli_cntx_jc_way( cntx );
	dim_t pc_way = bli_cntx_pc_way( cntx );
	dim_t ic_way = bli_cntx_ic_way( cntx );
	dim_t jr_way = bli_cntx_jr_way( cntx );
	dim_t ir_way = bli_cntx_ir_way( cntx );

	dim_t gl_nt  = jc_way * pc_way * ic_way * jr_way * ir_way;
	dim_t jc_nt  = pc_way * ic_way * jr_way * ir_way;
	dim_t pc_nt  = ic_way * jr_way * ir_way;
	dim_t ic_nt  = jr_way * ir_way;
	dim_t jr_nt  = ir_way;
	dim_t ir_nt  = 1;

	assert( gl_nt != 0 );

#ifdef PRINT_THRINFO
printf( "            gl   jc   kc   pb   ic   pa   jr   ir\n" );
printf( "xx_nt:    %4lu %4lu %4lu %4lu %4lu %4lu %4lu %4lu\n",
gl_nt, jc_nt, pc_nt, pc_nt, ic_nt, ic_nt, jr_nt, ir_nt );
printf( "\n" );
printf( "            jc   kc   pb   ic   pa   jr   ir\n" );
printf( "xx_way:   %4lu %4lu %4lu %4lu %4lu %4lu %4lu\n",
jc_way, pc_way, (dim_t)0, ic_way, (dim_t)0, jr_way, ir_way );
printf( "=================================================\n" );
#endif

	thrinfo_t** paths = bli_malloc_intl( gl_nt * sizeof( thrinfo_t* ) );

	thrcomm_t* gl_comm = bli_thrcomm_create( gl_nt );

	for( int a = 0; a < jc_way; a++ )
	{
		thrcomm_t* jc_comm = bli_thrcomm_create( jc_nt );

		for( int b = 0; b < pc_way; b++ )
		{
			thrcomm_t* pc_comm = bli_thrcomm_create( pc_nt );

			for( int c = 0; c < ic_way; c++ )
			{
				thrcomm_t* ic_comm = bli_thrcomm_create( ic_nt );

				for( int d = 0; d < jr_way; d++ )
				{
					thrcomm_t* jr_comm = bli_thrcomm_create( jr_nt );

					for( int e = 0; e < ir_way; e++ )
					{
						//thrcomm_t* ir_comm = bli_thrcomm_create( ir_nt );
						dim_t      ir_comm_id = 0;
						dim_t      jr_comm_id = e*ir_nt + ir_comm_id;
						dim_t      ic_comm_id = d*jr_nt + jr_comm_id;
						dim_t      pc_comm_id = c*ic_nt + ic_comm_id;
						dim_t      jc_comm_id = b*pc_nt + pc_comm_id;
						dim_t      gl_comm_id = a*jc_nt + jc_comm_id;

						// macro-kernel loops
						thrinfo_t* ir_info
						=
						bli_l3_thrinfo_create( jr_comm, jr_comm_id,
						                       ir_way, e,
						                       NULL );
						thrinfo_t* jr_info
						=
						bli_l3_thrinfo_create( ic_comm, ic_comm_id,
						                       jr_way, d,
						                       ir_info );
						// packa
						thrinfo_t* pa_info
						=
						bli_packm_thrinfo_create( ic_comm, ic_comm_id,
						                          ic_nt, ic_comm_id,
						                          jr_info );
						// blk_var1
						thrinfo_t* ic_info
						=
						bli_l3_thrinfo_create( pc_comm, pc_comm_id,
						                       ic_way, c,
						                       pa_info );
						// packb
						thrinfo_t* pb_info
						=
						bli_packm_thrinfo_create( pc_comm, pc_comm_id,
						                          pc_nt, pc_comm_id,
						                          ic_info );
						// blk_var3
						thrinfo_t* pc_info
						=
						bli_l3_thrinfo_create( jc_comm, jc_comm_id,
						                       pc_way, b,
						                       pb_info );
						// blk_var2
						thrinfo_t* jc_info
						=
						bli_l3_thrinfo_create( gl_comm, gl_comm_id,
						                       jc_way, a,
						                       pc_info );

						paths[gl_comm_id] = jc_info;

#ifdef PRINT_THRINFO
{
dim_t gl_comm_id = bli_thread_ocomm_id( jc_info );
dim_t jc_comm_id = bli_thread_ocomm_id( pc_info );
dim_t pc_comm_id = bli_thread_ocomm_id( pb_info );
dim_t pb_comm_id = bli_thread_ocomm_id( ic_info );
dim_t ic_comm_id = bli_thread_ocomm_id( pa_info );
dim_t pa_comm_id = bli_thread_ocomm_id( jr_info );
dim_t jr_comm_id = bli_thread_ocomm_id( ir_info );

dim_t jc_work_id = bli_thread_work_id( jc_info );
dim_t pc_work_id = bli_thread_work_id( pc_info );
dim_t pb_work_id = bli_thread_work_id( pb_info );
dim_t ic_work_id = bli_thread_work_id( ic_info );
dim_t pa_work_id = bli_thread_work_id( pa_info );
dim_t jr_work_id = bli_thread_work_id( jr_info );
dim_t ir_work_id = bli_thread_work_id( ir_info );

printf( "            gl   jc   pb   kc   pa   ic   jr  \n" );
printf( "comm ids: %4lu %4lu %4lu %4lu %4lu %4lu %4lu\n",
gl_comm_id, jc_comm_id, pc_comm_id, pb_comm_id, ic_comm_id, pa_comm_id, jr_comm_id );
printf( "work ids: %4ld %4ld %4lu %4lu %4ld %4ld %4ld\n",
jc_work_id, pc_work_id, pb_work_id, ic_work_id, pa_work_id, jr_work_id, ir_work_id );
printf( "-------------------------------------------------\n" );
}
#endif

					}
				}
			}
		}
	}
#ifdef PRINT_THRINFO
exit(1);
#endif


	return paths;
}
#endif

void bli_l3_thrinfo_free_paths
     (
       thrinfo_t** threads
     )
{
	dim_t n_threads = bli_thread_num_threads( threads[0] );
	dim_t i;

	for ( i = 0; i < n_threads; ++i )
		bli_l3_thrinfo_free( threads[i] );

	bli_free_intl( threads );
}

