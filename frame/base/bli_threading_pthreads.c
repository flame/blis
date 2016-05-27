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

#ifdef __APPLE__

int pthread_barrier_init(pthread_barrier_t *barrier, const pthread_barrierattr_t *attr, unsigned int count)
{
    if( barrier == NULL ) return 0;
    barrier->n_threads = count;
    barrier->sense = 0;
    barrier->threads_arrived = 0;
    pthread_mutex_init( &barrier->mutex, NULL );
    return 0;
}

int pthread_barrier_destroy(pthread_barrier_t *barrier)
{
    if( barrier == NULL ) return 0;
    pthread_mutex_destroy( &barrier->mutex );
    return 0;
}

int pthread_barrier_wait(pthread_barrier_t *barrier)
{
    if(barrier == NULL || barrier->n_threads == 1) return 0;
    bool_t my_sense = barrier->sense;
    dim_t my_threads_arrived;

    pthread_mutex_lock( &barrier->mutex );
    my_threads_arrived = ++(barrier->threads_arrived);
    pthread_mutex_unlock( &barrier->mutex );

    if( my_threads_arrived == barrier->n_threads ) {
        barrier->threads_arrived = 0;
        barrier->sense = !barrier->sense;
    }
    else {
        volatile bool_t* listener = &barrier->sense;
        while( *listener == my_sense ) { BLIS_YIELD(); }
    }
    return 0;
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

void bli_level3_thread_decorator
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
    pthread_t* pthreads = (pthread_t*) bli_malloc_intl(sizeof(pthread_t) * n_threads);
    thread_data_t* datas = (thread_data_t*) bli_malloc_intl(sizeof(thread_data_t) * n_threads);

    for( int i = 1; i < n_threads; i++ )
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

    for( int i = 1; i < n_threads; i++)
    {
        pthread_join( pthreads[i], NULL );
    }

    bli_free_intl( pthreads );
    bli_free_intl( datas );
}

//barrier routine taken from art of multicore programming
void bli_barrier( thread_comm_t* communicator, dim_t t_id )
{
    pthread_barrier_wait( &communicator->barrier );
}

//Constructors and destructors for constructors
thread_comm_t* bli_create_communicator( dim_t n_threads )
{
    thread_comm_t* comm = (thread_comm_t*) bli_malloc_intl( sizeof(thread_comm_t) );
    bli_setup_communicator( comm, n_threads );
    return comm;
}

void bli_setup_communicator( thread_comm_t* communicator, dim_t n_threads)
{
    if( communicator == NULL ) return;
    communicator->sent_object = NULL;
    communicator->n_threads = n_threads;
    pthread_barrier_init( &communicator->barrier, NULL, n_threads );
}

void bli_free_communicator( thread_comm_t* communicator )
{
    if( communicator == NULL ) return;
    bli_cleanup_communicator( communicator );
    bli_free_intl( communicator );
}

void bli_cleanup_communicator( thread_comm_t* communicator )
{
    if( communicator == NULL ) return;
    pthread_barrier_destroy( &communicator->barrier );
}
#endif
