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
	( unsigned long )gl_nt,
	( unsigned long )jc_nt,
	( unsigned long )pc_nt,
	( unsigned long )pb_nt,
	( unsigned long )ic_nt,
	( unsigned long )pa_nt,
	( unsigned long )jr_nt,
	( unsigned long )1 );
	printf( "\n" );
	printf( "            jc   kc   pb   ic   pa   jr   ir\n" );
	printf( "xx_way:   %4lu %4lu %4lu %4lu %4lu %4lu %4lu\n",
    ( unsigned long )jc_way,
	( unsigned long )pc_way,
	( unsigned long )pb_way,
	( unsigned long )ic_way,
	( unsigned long )pa_way,
	( unsigned long )jr_way,
	( unsigned long )ir_way );
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
		( unsigned long )gl_comm_id,
		( unsigned long )jc_comm_id,
		( unsigned long )pc_comm_id,
		( unsigned long )pb_comm_id,
		( unsigned long )ic_comm_id,
		( unsigned long )pa_comm_id,
		( unsigned long )jr_comm_id );
		printf( "work ids: %4ld %4ld %4lu %4lu %4ld %4ld %4ld\n",
		( unsigned long )jc_work_id,
		( unsigned long )pc_work_id,
		( unsigned long )pb_work_id,
		( unsigned long )ic_work_id,
		( unsigned long )pa_work_id,
		( unsigned long )jr_work_id,
		( unsigned long )ir_work_id );
		printf( "---------------------------------------\n" );
	}

}

// -----------------------------------------------------------------------------

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

