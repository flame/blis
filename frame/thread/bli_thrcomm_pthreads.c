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

#ifdef BLIS_ENABLE_PTHREADS

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

#ifdef BLIS_USE_PTHREAD_BARRIER

void bli_thrcomm_init( thrcomm_t* comm, dim_t n_threads)
{
	if ( comm == NULL ) return;
	comm->sent_object = NULL;
	comm->n_threads = n_threads;
	bli_pthread_barrier_init( &comm->barrier, NULL, n_threads );
}

void bli_thrcomm_cleanup( thrcomm_t* comm )
{
	if ( comm == NULL ) return;
	bli_pthread_barrier_destroy( &comm->barrier );
}

void bli_thrcomm_barrier( thrcomm_t* comm, dim_t t_id )
{
	bli_pthread_barrier_wait( &comm->barrier );
}

#else

void bli_thrcomm_init( thrcomm_t* comm, dim_t n_threads)
{
	if ( comm == NULL ) return;
	comm->sent_object = NULL;
	comm->n_threads = n_threads;
	comm->barrier_sense = 0;
	comm->barrier_threads_arrived = 0;

//#ifdef BLIS_USE_PTHREAD_MUTEX
//	bli_pthread_mutex_init( &comm->mutex, NULL );
//#endif
}

void bli_thrcomm_cleanup( thrcomm_t* comm )
{
//#ifdef BLIS_USE_PTHREAD_MUTEX
//	if ( comm == NULL ) return;
//	bli_pthread_mutex_destroy( &comm->mutex );
//#endif
}

void bli_thrcomm_barrier( thrcomm_t* comm, dim_t t_id )
{
#if 0
	if ( comm == NULL || comm->n_threads == 1 ) return;
	bool_t my_sense = comm->sense;
	dim_t my_threads_arrived;

#ifdef BLIS_USE_PTHREAD_MUTEX
	bli_pthread_mutex_lock( &comm->mutex );
	my_threads_arrived = ++(comm->threads_arrived);
	bli_pthread_mutex_unlock( &comm->mutex );
#else
	my_threads_arrived = __sync_add_and_fetch(&(comm->threads_arrived), 1);
#endif

	if ( my_threads_arrived == comm->n_threads )
	{
		comm->threads_arrived = 0;
		comm->sense = !comm->sense;
	}
	else
	{
		volatile bool_t* listener = &comm->sense;
		while( *listener == my_sense ) {}
	}
#endif
	bli_thrcomm_barrier_atomic( comm, t_id );
}

#endif


// A data structure to assist in passing operands to additional threads.
typedef struct thread_data
{
	l3int_t    func;
	opid_t     family;
	obj_t*     alpha;
	obj_t*     a;
	obj_t*     b;
	obj_t*     beta;
	obj_t*     c;
	cntx_t*    cntx;
	rntm_t*    rntm;
	cntl_t*    cntl;
	dim_t      id;
	thrcomm_t* gl_comm;
} thread_data_t;

// Entry point for additional threads
void* bli_l3_thread_entry( void* data_void )
{
	thread_data_t* data     = data_void;

	l3int_t        func     = data->func;
	opid_t         family   = data->family;
	obj_t*         alpha    = data->alpha;
	obj_t*         a        = data->a;
	obj_t*         b        = data->b;
	obj_t*         beta     = data->beta;
	obj_t*         c        = data->c;
	cntx_t*        cntx     = data->cntx;
	rntm_t*        rntm     = data->rntm;
	cntl_t*        cntl     = data->cntl;
	dim_t          id       = data->id;
	thrcomm_t*     gl_comm  = data->gl_comm;

	obj_t          a_t, b_t, c_t;
	cntl_t*        cntl_use;
	thrinfo_t*     thread;

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

	// Free the current thread's thrinfo_t structure.
	bli_l3_thrinfo_free( thread );

	return NULL;
}

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
	dim_t          n_threads = bli_rntm_num_threads( rntm );

	// Allocate an array of pthread objects and auxiliary data structs to pass
	// to the thread entry functions.
	bli_pthread_t* pthreads  = bli_malloc_intl( sizeof( bli_pthread_t ) * n_threads );
	thread_data_t* datas     = bli_malloc_intl( sizeof( thread_data_t ) * n_threads );

	// Allocate a global communicator for the root thrinfo_t structures.
	thrcomm_t*     gl_comm   = bli_thrcomm_create( n_threads );

	// NOTE: We must iterate backwards so that the chief thread (thread id 0)
	// can spawn all other threads before proceeding with its own computation.
	for ( dim_t id = n_threads - 1; 0 <= id; id-- )
	{
		// Set up thread data for additional threads (beyond thread 0).
		datas[id].func    = func;
		datas[id].family  = family;
		datas[id].alpha   = alpha;
		datas[id].a       = a;
		datas[id].b       = b;
		datas[id].beta    = beta;
		datas[id].c       = c;
		datas[id].cntx    = cntx;
		datas[id].rntm    = rntm;
		datas[id].cntl    = cntl;
		datas[id].id      = id;
		datas[id].gl_comm = gl_comm;

		// Spawn additional threads for ids greater than 1.
		if ( id != 0 )
			bli_pthread_create( &pthreads[id], NULL, &bli_l3_thread_entry, &datas[id] );
		else
			bli_l3_thread_entry( ( void* )(&datas[0]) );
	}

	// We shouldn't free the global communicator since it was already freed
	// by the global communicator's chief thread in bli_l3_thrinfo_free()
	// (called from the thread entry function).

	// Thread 0 waits for additional threads to finish.
	for ( dim_t id = 1; id < n_threads; id++ )
	{
		bli_pthread_join( pthreads[id], NULL );
	}

	bli_free_intl( pthreads );
	bli_free_intl( datas );
}


#endif

