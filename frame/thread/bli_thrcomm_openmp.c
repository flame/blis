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

#ifdef BLIS_ENABLE_OPENMP

thrcomm_t* bli_thrcomm_create( dim_t n_threads )
{
	thrcomm_t* comm = bli_malloc_intl( sizeof(thrcomm_t) );
	bli_thrcomm_init( comm, n_threads );

	return comm;
}

void bli_thrcomm_free( thrcomm_t* comm )
{
	if ( comm == NULL ) return;
	bli_thrcomm_cleanup( comm );
	bli_free_intl( comm );
}

#ifndef BLIS_TREE_BARRIER

void bli_thrcomm_init( thrcomm_t* comm, dim_t n_threads)
{
	if ( comm == NULL ) return;
	comm->sent_object = NULL;
	comm->n_threads = n_threads;
	comm->barrier_sense = 0;
	comm->barrier_threads_arrived = 0;
}


void bli_thrcomm_cleanup( thrcomm_t* comm )
{
	if ( comm == NULL ) return;
}

//'Normal' barrier for openmp
//barrier routine taken from art of multicore programming
void bli_thrcomm_barrier( thrcomm_t* comm, dim_t t_id )
{
#if 0
	if ( comm == NULL || comm->n_threads == 1 )
		return;
	bool_t my_sense = comm->barrier_sense;
	dim_t my_threads_arrived;

	_Pragma( "omp atomic capture" )
		my_threads_arrived = ++(comm->barrier_threads_arrived);

	if ( my_threads_arrived == comm->n_threads )
	{
		comm->barrier_threads_arrived = 0;
		comm->barrier_sense = !comm->barrier_sense;
	}
	else
	{
		volatile bool_t* listener = &comm->barrier_sense;
		while ( *listener == my_sense ) {}
	}
#endif
	bli_thrcomm_barrier_atomic( comm, t_id );
}

#else

void bli_thrcomm_init( thrcomm_t* comm, dim_t n_threads)
{
	if ( comm == NULL ) return;
	comm->sent_object = NULL;
	comm->n_threads = n_threads;
	comm->barriers = bli_malloc_intl( sizeof( barrier_t* ) * n_threads );
	bli_thrcomm_tree_barrier_create( n_threads, BLIS_TREE_BARRIER_ARITY, comm->barriers, 0 );
}

//Tree barrier used for Intel Xeon Phi
barrier_t* bli_thrcomm_tree_barrier_create( int num_threads, int arity, barrier_t** leaves, int leaf_index )
{
	barrier_t* me = bli_malloc_intl( sizeof(barrier_t) );

	me->dad = NULL;
	me->signal = 0;

	// Base Case
	if ( num_threads <= arity )
	{
		//Now must be registered as a leaf
		for ( int i = 0; i < num_threads; i++ )
		{
			leaves[ leaf_index + i ] = me;
		}
		me->count = num_threads;
		me->arity = num_threads;
	}
	else
	{
		// Otherwise this node has children
		int threads_per_kid = num_threads / arity;
		int defecit = num_threads - threads_per_kid * arity;

		for ( int i = 0; i < arity; i++ )
		{
			int threads_this_kid = threads_per_kid;
			if ( i < defecit ) threads_this_kid++;

			barrier_t* kid = bli_thrcomm_tree_barrier_create( threads_this_kid, arity, leaves, leaf_index );
			kid->dad = me;

			leaf_index += threads_this_kid;
		}  
		me->count = arity;
		me->arity = arity;
	}  

	return me;
}

void bli_thrcomm_cleanup( thrcomm_t* comm )
{
	if ( comm == NULL ) return;
	for ( dim_t i = 0; i < comm->n_threads; i++ )
	{
	   bli_thrcomm_tree_barrier_free( comm->barriers[i] );
	}
	bli_free_intl( comm->barriers );
}

void bli_thrcomm_tree_barrier_free( barrier_t* barrier )
{
	if ( barrier == NULL )
		return;
	barrier->count--;
	if ( barrier->count == 0 )
	{
		bli_thrcomm_tree_barrier_free( barrier->dad );
		bli_free_intl( barrier );
	}
	return;
}

void bli_thrcomm_barrier( thrcomm_t* comm, dim_t t_id )
{
	bli_thrcomm_tree_barrier( comm->barriers[t_id] );
}

void bli_thrcomm_tree_barrier( barrier_t* barack )
{
	int my_signal = barack->signal;
	int my_count;

	_Pragma( "omp atomic capture" )
		my_count = barack->count--;

	if ( my_count == 1 )
	{
		if ( barack->dad != NULL )
		{
			bli_thrcomm_tree_barrier( barack->dad );
		}
		barack->count = barack->arity;
		barack->signal = !barack->signal;
	}
	else
	{
		volatile int* listener = &barack->signal;
		while ( *listener == my_signal ) {}
	}
}

#endif

// Define a dummy function bli_l3_thread_entry(), which is needed in the
// pthreads version, so that when building Windows DLLs (with OpenMP enabled
// or no multithreading) we don't risk having an unresolved symbol.
void* bli_l3_thread_entry( void* data_void ) { return NULL; }

//#define PRINT_THRINFO

void bli_l3_thread_decorator
     (
       l3int_t     func,
       opid_t      family,
       obj_t*      alpha,
       obj_t*      a,
       obj_t*      b,
       obj_t*      beta,
       obj_t*      c,
       cntx_t*     cntx,
       rntm_t*     rntm,
       cntl_t*     cntl
     )
{
	// Query the total number of threads from the context.
	dim_t       n_threads = bli_rntm_num_threads( rntm );

	// Allcoate a global communicator for the root thrinfo_t structures.
	thrcomm_t*  gl_comm   = bli_thrcomm_create( n_threads );

#ifdef PRINT_THRINFO
	thrinfo_t** threads   = bli_malloc_intl( n_threads * sizeof( thrinfo_t* ) );
#endif

	_Pragma( "omp parallel num_threads(n_threads)" )
	{
		dim_t      n_threads_real = omp_get_num_threads();
		dim_t      id = omp_get_thread_num();

		// Check if the number of OpenMP threads created within this parallel
		// region is different from the number of threads that were requested
		// of BLIS. This inequality may trigger when, for example, the
		// following conditions are satisfied:
		// - an application is executing an OpenMP parallel region in which
		//   BLIS is invokved,
		// - BLIS is configured for multithreading via OpenMP,
		// - OMP_NUM_THREADS = t > 1,
		// - the number of threads requested of BLIS (regardless of method)
		//   is p <= t,
		// - OpenMP nesting is disabled.
		// In this situation, the application spawns t threads. Each application
		// thread calls gemm (for example). Each gemm will attempt to spawn p
		// threads via OpenMP. However, since nesting is disabled, the OpenMP
		// implementation finds that t >= p threads are already spawned, and
		// thus it doesn't spawn *any* additional threads for each gemm.
		if ( n_threads_real != n_threads )
		{
 			// If the number of threads active in the current region is not
			// equal to the number requested of BLIS, we then only continue
			// if the number of threads in the current region is 1. If, for
			// example, BLIS requested 4 threads but only got 3, then we
			// abort().
			if ( id == 0 )
			{
				if ( n_threads_real != 1 )
				{
					bli_print_msg( "A different number of threads was "
					               "created than was requested.",
					               __FILE__, __LINE__ );
					bli_abort();
				}

				n_threads = 1;
				bli_thrcomm_init( gl_comm, 1 );
				bli_rntm_set_num_threads_only( 1, rntm );
				bli_rntm_set_ways_only( 1, 1, 1, 1, 1, rntm );
			}

			// Synchronize all threads and continue.
			_Pragma( "omp barrier" )
		}

		obj_t      a_t, b_t, c_t;
		cntl_t*    cntl_use;
		thrinfo_t* thread;

		// Alias thread-local copies of A, B, and C. These will be the objects
		// we pass down the algorithmic function stack. Making thread-local
		// alaises IS ABSOLUTELY IMPORTANT and MUST BE DONE because each thread
		// will read the schemas from A and B and then reset the schemas to
		// their expected unpacked state (in bli_l3_cntl_create_if()).
		bli_obj_alias_to( a, &a_t );
		bli_obj_alias_to( b, &b_t );
		bli_obj_alias_to( c, &c_t );

		// Create a default control tree for the operation, if needed.
		bli_l3_cntl_create_if( family, &a_t, &b_t, &c_t, cntl, &cntl_use );

		// Create the root node of the current thread's thrinfo_t structure.
		bli_l3_thrinfo_create_root( id, gl_comm, rntm, cntl_use, &thread );

		func
		(
		  alpha,
		  &a_t,
		  &b_t,
		  beta,
		  &c_t,
		  cntx,
		  rntm,
		  cntl_use,
		  thread
		);

		// Free the control tree, if one was created locally.
		bli_l3_cntl_free_if( &a_t, &b_t, &c_t, cntl, cntl_use, thread );

#ifdef PRINT_THRINFO
		threads[id] = thread;
#else
		// Free the current thread's thrinfo_t structure.
		bli_l3_thrinfo_free( thread );
#endif
	}

	// We shouldn't free the global communicator since it was already freed
	// by the global communicator's chief thread in bli_l3_thrinfo_free()
	// (called above).


#ifdef PRINT_THRINFO
	bli_l3_thrinfo_print_paths( threads );
	bli_l3_thrinfo_free_paths( threads );
	exit(1);
#endif
}

#endif

