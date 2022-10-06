/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021, The University of Texas at Austin

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

#ifdef BLIS_ENABLE_PTHREADS

// A data structure to assist in passing operands to additional threads.
typedef struct thread_data
{
	l3sbxint_ft func;
	opid_t      family;
	obj_t*      alpha;
	obj_t*      a;
	obj_t*      b;
	obj_t*      beta;
	obj_t*      c;
	cntx_t*     cntx;
	rntm_t*     rntm;
	dim_t       tid;
	thrcomm_t*  gl_comm;
	array_t*    array;
} thread_data_t;

// Entry point function for additional threads.
void* bls_l3_thread_entry( void* data_void )
{
	thread_data_t* data     = data_void;

	l3sbxint_ft    func     = data->func;
	opid_t         family   = data->family;
	obj_t*         alpha    = data->alpha;
	obj_t*         a        = data->a;
	obj_t*         b        = data->b;
	obj_t*         beta     = data->beta;
	obj_t*         c        = data->c;
	cntx_t*        cntx     = data->cntx;
	rntm_t*        rntm     = data->rntm;
	dim_t          tid      = data->tid;
	array_t*       array    = data->array;
	thrcomm_t*     gl_comm  = data->gl_comm;

	( void )family;

	// Create a thread-local copy of the master thread's rntm_t. This is
	// necessary since we want each thread to be able to track its own
	// small block pool_t as it executes down the function stack.
	rntm_t rntm_l = *rntm;

	// Create the root node of the thread's thrinfo_t structure.
    pool_t*    pool   = bli_apool_array_elem( tid, array );
	thrinfo_t* thread = bli_l3_sup_thrinfo_create( tid, gl_comm, pool, &rntm_l );

	func
	(
	  alpha,
	  a,
	  b,
	  beta,
	  c,
	  cntx,
	  &rntm_l,
	  bli_thrinfo_sub_node( thread )
	);

	// Free the current thread's thrinfo_t structure.
	bli_thrinfo_free( thread );

	return NULL;
}

void bls_l3_thread_decorator_pthreads
     (
       l3sbxint_ft func,
       opid_t      family,
       obj_t*      alpha,
       obj_t*      a,
       obj_t*      b,
       obj_t*      beta,
       obj_t*      c,
       cntx_t*     cntx,
       rntm_t*     rntm
     )
{
	err_t r_val;

	// Query the total number of threads from the context.
	const dim_t n_threads = bli_rntm_num_threads( rntm );

	// NOTE: The sba was initialized in bli_init().

	// Check out an array_t from the small block allocator. This is done
	// with an internal lock to ensure only one application thread accesses
	// the sba at a time. bli_sba_checkout_array() will also automatically
	// resize the array_t, if necessary.
	array_t* array = bli_sba_checkout_array( n_threads );

	// Allocate a global communicator for the root thrinfo_t structures.
	thrcomm_t* gl_comm = bli_thrcomm_create( NULL, BLIS_POSIX, n_threads );

	// Allocate an array of pthread objects and auxiliary data structs to pass
	// to the thread entry functions.

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_l3_thread_decorator().pth: " );
	#endif
	bli_pthread_t* pthreads = bli_malloc_intl( sizeof( bli_pthread_t ) * n_threads, &r_val );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_l3_thread_decorator().pth: " );
	#endif
	thread_data_t* datas    = bli_malloc_intl( sizeof( thread_data_t ) * n_threads, &r_val );

	// NOTE: We must iterate backwards so that the chief thread (thread id 0)
	// can spawn all other threads before proceeding with its own computation.
	for ( dim_t tid = n_threads - 1; 0 <= tid; tid-- )
	{
		// Set up thread data for additional threads (beyond thread 0).
		datas[tid].func     = func;
		datas[tid].family   = family;
		datas[tid].alpha    = alpha;
		datas[tid].a        = a;
		datas[tid].b        = b;
		datas[tid].beta     = beta;
		datas[tid].c        = c;
		datas[tid].cntx     = cntx;
		datas[tid].rntm     = rntm;
		datas[tid].tid      = tid;
		datas[tid].gl_comm  = gl_comm;
		datas[tid].array    = array;

		// Spawn additional threads for ids greater than 1.
		if ( tid != 0 )
			bli_pthread_create( &pthreads[tid], NULL, &bls_l3_thread_entry, &datas[tid] );
		else
			bls_l3_thread_entry( ( void* )(&datas[0]) );
	}

	// We shouldn't free the global communicator since it was already freed
	// by the global communicator's chief thread in bli_l3_thrinfo_free()
	// (called from the thread entry function).

	// Thread 0 waits for additional threads to finish.
	for ( dim_t tid = 1; tid < n_threads; tid++ )
	{
		bli_pthread_join( pthreads[tid], NULL );
	}

	// Check the array_t back into the small block allocator. Similar to the
	// check-out, this is done using a lock embedded within the sba to ensure
	// mutual exclusion.
	bli_sba_checkin_array( array );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_l3_thread_decorator().pth: " );
	#endif
	bli_free_intl( pthreads );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_l3_thread_decorator().pth: " );
	#endif
	bli_free_intl( datas );
}

#else

// Define a dummy function bli_l3_thread_entry(), which is needed for
// consistent dynamic linking behavior when building shared objects in Linux
// or OSX, or Windows DLLs; otherwise, we risk having an unresolved symbol.
void* bli_l3_thread_entry( void* data_void ) { return NULL; }

#endif

