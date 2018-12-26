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

thrinfo_t* bli_thrinfo_create
     (
       rntm_t*    rntm,
       thrcomm_t* ocomm,
       dim_t      ocomm_id,
       dim_t      n_way,
       dim_t      work_id, 
       bool_t     free_comm,
       thrinfo_t* sub_node
     )
{
	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_thrinfo_create(): " );
	#endif

    thrinfo_t* thread = bli_sba_acquire( rntm, sizeof( thrinfo_t ) );

    bli_thrinfo_init
	(
	  thread,
	  ocomm, ocomm_id,
	  n_way, work_id, 
	  free_comm,
	  sub_node
	);

    return thread;
}

void bli_thrinfo_init
     (
       thrinfo_t* thread,
       thrcomm_t* ocomm,
       dim_t      ocomm_id,
       dim_t      n_way,
       dim_t      work_id, 
       bool_t     free_comm,
       thrinfo_t* sub_node
     )
{
	thread->ocomm     = ocomm;
	thread->ocomm_id  = ocomm_id;
	thread->n_way     = n_way;
	thread->work_id   = work_id;
	thread->free_comm = free_comm;

	thread->sub_node  = sub_node;
}

void bli_thrinfo_init_single
     (
       thrinfo_t* thread
     )
{
	bli_thrinfo_init
	(
	  thread,
	  &BLIS_SINGLE_COMM, 0,
	  1,
	  0,
	  FALSE,
	  thread
	);
}

void bli_thrinfo_free
     (
       rntm_t*    rntm,
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
			bli_thrcomm_free( rntm, bli_thrinfo_ocomm( thread ) );
	}

	// Recursively free all children of the current thrinfo_t.
	bli_thrinfo_free( rntm, thrinfo_sub_node );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_thrinfo_free(): " );
	#endif

	// Free the thrinfo_t struct.
	bli_sba_release( rntm, thread );
}

// -----------------------------------------------------------------------------

#include "assert.h"

#define BLIS_NUM_STATIC_COMMS 80

thrinfo_t* bli_thrinfo_create_for_cntl
     (
       rntm_t*    rntm,
       cntl_t*    cntl_par,
       cntl_t*    cntl_chl,
       thrinfo_t* thread_par
     )
{
	thrcomm_t*  static_comms[ BLIS_NUM_STATIC_COMMS ];
	thrcomm_t** new_comms = NULL;

	thrinfo_t* thread_chl;

	const bszid_t bszid_chl = bli_cntl_bszid( cntl_chl );

	const dim_t parent_nt_in   = bli_thread_num_threads( thread_par );
	const dim_t parent_n_way   = bli_thread_n_way( thread_par );
	const dim_t parent_comm_id = bli_thread_ocomm_id( thread_par );
	const dim_t parent_work_id = bli_thread_work_id( thread_par );

	dim_t child_nt_in;
	dim_t child_comm_id;
	dim_t child_n_way;
	dim_t child_work_id;

	// Sanity check: make sure the number of threads in the parent's
	// communicator is divisible by the number of new sub-groups.
	assert( parent_nt_in % parent_n_way == 0 );

	// Compute:
	// - the number of threads inside the new child comm,
	// - the current thread's id within the new communicator,
	// - the current thread's work id, given the ways of parallelism
	//   to be obtained within the next loop.
	child_nt_in   = bli_cntl_calc_num_threads_in( rntm, cntl_chl );
	child_n_way   = bli_rntm_ways_for( bszid_chl, rntm );
	child_comm_id = parent_comm_id % child_nt_in;
	child_work_id = child_comm_id / ( child_nt_in / child_n_way );

	// The parent's chief thread creates a temporary array of thrcomm_t
	// pointers.
	if ( bli_thread_am_ochief( thread_par ) )
	{
		if ( parent_n_way > BLIS_NUM_STATIC_COMMS )
			new_comms = bli_malloc_intl( parent_n_way * sizeof( thrcomm_t* ) );
		else
			new_comms = static_comms;
	}

	// Broadcast the temporary array to all threads in the parent's
	// communicator.
	new_comms = bli_thread_obroadcast( thread_par, new_comms );

	// Chiefs in the child communicator allocate the communicator
	// object and store it in the array element corresponding to the
	// parent's work id.
	if ( child_comm_id == 0 )
		new_comms[ parent_work_id ] = bli_thrcomm_create( rntm, child_nt_in );

	bli_thread_obarrier( thread_par );

	// All threads create a new thrinfo_t node using the communicator
	// that was created by their chief, as identified by parent_work_id.
	thread_chl = bli_thrinfo_create
	(
	  rntm,
	  new_comms[ parent_work_id ],
	  child_comm_id,
	  child_n_way,
	  child_work_id,
	  TRUE,
	  NULL
	);

	bli_thread_obarrier( thread_par );

	// The parent's chief thread frees the temporary array of thrcomm_t
	// pointers.
	if ( bli_thread_am_ochief( thread_par ) )
	{
		if ( parent_n_way > BLIS_NUM_STATIC_COMMS )
			bli_free_intl( new_comms );
	}

	return thread_chl;
}

void bli_thrinfo_grow
     (
       rntm_t*    rntm,
       cntl_t*    cntl,
       thrinfo_t* thread
     )
{
	// If the sub-node of the thrinfo_t object is non-NULL, we don't
	// need to create it, and will just use the existing sub-node as-is.
	if ( bli_thrinfo_sub_node( thread ) != NULL ) return;

	// Create a new node (or, if needed, multiple nodes) and return the
	// pointer to the (eldest) child.
	thrinfo_t* thread_child = bli_thrinfo_rgrow
	(
	  rntm,
	  cntl,
	  bli_cntl_sub_node( cntl ),
	  thread
	);

	// Attach the child thrinfo_t node to its parent structure.
	bli_thrinfo_set_sub_node( thread_child, thread );
}

thrinfo_t* bli_thrinfo_rgrow
     (
       rntm_t*    rntm,
       cntl_t*    cntl_par,
       cntl_t*    cntl_cur,
       thrinfo_t* thread_par
     )
{
	thrinfo_t* thread_cur;

	// We must handle two cases: those where the next node in the
	// control tree is a partitioning node, and those where it is
	// a non-partitioning (ie: packing) node.
	if ( bli_cntl_bszid( cntl_cur ) != BLIS_NO_PART )
	{
		// Create the child thrinfo_t node corresponding to cntl_cur,
		// with cntl_par being the parent.
		thread_cur = bli_thrinfo_create_for_cntl
		(
		  rntm,
		  cntl_par,
		  cntl_cur,
		  thread_par
		);
	}
	else // if ( bli_cntl_bszid( cntl_cur ) == BLIS_NO_PART )
	{
		// Recursively grow the thread structure and return the top-most
		// thrinfo_t node of that segment.
		thrinfo_t* thread_seg = bli_thrinfo_rgrow
		(
		  rntm,
		  cntl_par,
		  bli_cntl_sub_node( cntl_cur ),
		  thread_par
		);

		// Create a thrinfo_t node corresponding to cntl_cur. Notice that
		// the free_comm field is set to FALSE, since cntl_cur is a
		// non-partitioning node. The communicator used here will be
		// freed when thread_seg, or one of its descendents, is freed.
		thread_cur = bli_thrinfo_create
		(
		  rntm,
		  bli_thrinfo_ocomm( thread_seg ),
		  bli_thread_ocomm_id( thread_seg ),
		  bli_cntl_calc_num_threads_in( rntm, cntl_cur ),
		  bli_thread_ocomm_id( thread_seg ),
		  FALSE,
		  thread_seg
		);

		// Attach the child thrinfo_t node to its parent structure.
		bli_thrinfo_set_sub_node( thread_cur, thread_par );
	}

	return thread_cur;
}

