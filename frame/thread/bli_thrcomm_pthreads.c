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

void bli_thrcomm_free( thrcomm_t* communicator )
{
	if ( communicator == NULL ) return;
	bli_thrcomm_cleanup( communicator );
	bli_free_intl( communicator );
}

#ifdef BLIS_USE_PTHREAD_BARRIER

void bli_thrcomm_init( thrcomm_t* communicator, dim_t n_threads)
{
	if ( communicator == NULL ) return;
	communicator->sent_object = NULL;
	communicator->n_threads = n_threads;
	pthread_barrier_init( &communicator->barrier, NULL, n_threads );
}

void bli_thrcomm_cleanup( thrcomm_t* communicator )
{
	if ( communicator == NULL ) return;
	pthread_barrier_destroy( &communicator->barrier );
}

void bli_thrcomm_barrier( thrcomm_t* communicator, dim_t t_id )
{
	pthread_barrier_wait( &communicator->barrier );
}

#else

void bli_thrcomm_init( thrcomm_t* communicator, dim_t n_threads)
{
	if ( communicator == NULL ) return;
	communicator->sent_object = NULL;
	communicator->n_threads = n_threads;
	communicator->sense = 0;
	communicator->threads_arrived = 0;
 
#ifdef BLIS_USE_PTHREAD_MUTEX
	pthread_mutex_init( &communicator->mutex, NULL );
#endif
}

void bli_thrcomm_cleanup( thrcomm_t* communicator )
{
#ifdef BLIS_USE_PTHREAD_MUTEX
	if ( communicator == NULL ) return;
	pthread_mutex_destroy( &communicator->mutex );
#endif
}

void bli_thrcomm_barrier( thrcomm_t* communicator, dim_t t_id )
{
	if ( communicator == NULL || communicator->n_threads == 1 ) return;
	bool_t my_sense = communicator->sense;
	dim_t my_threads_arrived;

#ifdef BLIS_USE_PTHREAD_MUTEX
	pthread_mutex_lock( &communicator->mutex );
	my_threads_arrived = ++(communicator->threads_arrived);
	pthread_mutex_unlock( &communicator->mutex );
#else
	my_threads_arrived = __sync_add_and_fetch(&(communicator->threads_arrived), 1);
#endif

	if ( my_threads_arrived == communicator->n_threads )
	{
		communicator->threads_arrived = 0;
		communicator->sense = !communicator->sense;
	}
	else
	{
		volatile bool_t* listener = &communicator->sense;
		while( *listener == my_sense ) {}
	}
}

#endif


void* thread_decorator_helper( void* data_void );

typedef struct thread_data
{
	l3_int_t func;
	obj_t* alpha; 
	obj_t* a; 
	obj_t* b; 
	obj_t* beta;
	obj_t* c;
	void* cntx;
	void* cntl;
	void* thread;
} thread_data_t;

void* thread_decorator_helper( void* data_void )
{
	thread_data_t* data = data_void;

	data->func
	(
	  data->alpha,
	  data->a,
	  data->b,
	  data->beta,
	  data->c,
	  data->cntx,
	  data->cntl,
	  data->thread
	);

	return NULL;
}

void bli_l3_thread_decorator
     (
       dim_t    n_threads,
       l3_int_t func,
       obj_t*   alpha,
       obj_t*   a,
       obj_t*   b,
       obj_t*   beta,
       obj_t*   c,
       void*    cntx,
       void*    cntl,
       void**   thread
     )
{
	pthread_t*     pthreads = bli_malloc_intl( sizeof( pthread_t ) * n_threads );
	thread_data_t* datas    = bli_malloc_intl( sizeof( thread_data_t ) * n_threads );

	for ( int i = 1; i < n_threads; i++ )
	{
		//Setup the thread data
		datas[i].func = func;
		datas[i].alpha = alpha;
		datas[i].a = a;
		datas[i].b = b;
		datas[i].beta = beta;
		datas[i].c = c;
		datas[i].cntx = cntx;
		datas[i].cntl = cntl;
		datas[i].thread = thread[i];

		pthread_create( &pthreads[i], NULL, &thread_decorator_helper, &datas[i] );
	}

	func( alpha, a, b, beta, c, cntx, cntl, thread[0] );

	for ( int i = 1; i < n_threads; i++)
	{
		pthread_join( pthreads[i], NULL );
	}

	bli_free_intl( pthreads );
	bli_free_intl( datas );
}


#endif

